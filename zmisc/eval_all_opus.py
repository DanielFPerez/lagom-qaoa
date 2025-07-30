# eval_all_batched.py
import ray
import json
import os
from pathlib import Path
import torch
import random
import networkx as nx
import src.utils.graphs as graph_utils
import src.utils.config as config_utils
import scripts.run_rl_trainer_khairy as khairy_tools
import scripts.run_rl_trainer_gnn_withinit as gnnrl_tools
import time
import numpy as np
from datetime import datetime

PROJECT_ROOT = config_utils.get_project_root()

KH1 = PROJECT_ROOT + "/outputs/07-30_01-23_rl_model_khairy_p{p}"
GAT = PROJECT_ROOT + "/outputs/07-30_01-20_gnn_rl_model_GAT_p{p}_withinit"
TCN = PROJECT_ROOT + "/outputs/07-30_01-21_gnn_rl_model_TCN_p{p}_withinit"
GIN = PROJECT_ROOT + "/outputs/07-30_01-19_gnn_rl_model_GIN_p{p}_withinit"

MODELS = []
hidden_dim = 512

# ── Khairy families ───────────────────────────────────────────
for family, label in [(KH1, "khairy_v1")]:
    for p in (1, 2, 3):
        MODELS.append(
            dict(
                key=f"{label}_p{p}",
                p=p,
                path=family.format(p=p) + "/best_model.pth",
                cls=khairy_tools.QAOAOptimizer,
                kwargs=dict(hidden_dim=hidden_dim),
            )
        )

# ── GNN families ──────────────────────────────────────────────
for gnn_type, templ in [("GIN", GIN), ("GAT", GAT), ("TCN", TCN)]:
    for p in (1, 2, 3):
        path = templ.format(p=p) + f"/best_model_{gnn_type}_withinit.pth"
        MODELS.append(
            dict(
                key=f"{gnn_type.upper()}_p{p}",
                p=p,
                path=path,
                cls=gnnrl_tools.GNNQAOAOptimizer,
                kwargs=dict(
                    gnn_type=gnn_type,
                    gnn_hidden_dim=256,
                    hidden_dim=hidden_dim,
                ),
            )
        )

print(f"Total models to evaluate: {len(MODELS)}")

# --- GPU Worker that processes multiple models sequentially ---
@ray.remote(num_gpus=1)
def gpu_worker(model_configs: list, graphs: list, gpu_id: int, n_runs: int = 5):
    """
    Worker that processes multiple models on a single GPU.
    This avoids resource contention by running models sequentially on each GPU.
    """
    # Set GPU visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"  # Always 0 since we only see one GPU
    
    print(f"[GPU {gpu_id}] Worker started with {len(model_configs)} models")
    
    all_patches = {}
    
    for cfg in model_configs:
        try:
            print(f"[GPU {gpu_id}] Loading model {cfg['key']} from {cfg['path']}")
            
            # Load optimizer
            opt = cfg["cls"](
                str(cfg["path"]),
                p=cfg["p"],
                device=device,
                **cfg["kwargs"],
            )
            
            print(f"[GPU {gpu_id}] Evaluating {cfg['key']} on {len(graphs)} graphs")
            
            # Evaluate all graphs
            patch = {}
            for i_graph, g in enumerate(graphs):
                if i_graph % 20 == 0:
                    print(f"[GPU {gpu_id}] {cfg['key']}: {i_graph}/{len(graphs)} graphs")
                
                try:
                    G = graph_utils.read_graph_from_dict(g["graph_dict"])
                    vals, params, runtimes = [], [], []
                    
                    for run_i in range(n_runs):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        
                        if hasattr(opt, "optimize_rl_only"):
                            prms, v = opt.optimize_rl_only(G)
                        else:
                            prms, v = opt.optimize_rl_with_gnn_init(G)
                        
                        torch.cuda.synchronize()
                        runtimes.append(time.perf_counter() - t0)
                        
                        vals.append(v)
                        params.append(prms)
                    
                    best_idx = int(np.argmax(vals))
                    gamma = params[best_idx][:cfg["p"]]
                    beta = params[best_idx][cfg["p"]:]
                    
                    # Convert to list if numpy array
                    if hasattr(gamma, 'tolist'):
                        gamma = gamma.tolist()
                    if hasattr(beta, 'tolist'):
                        beta = beta.tolist()
                    
                    patch.setdefault(g["id"], {})[cfg["key"]] = dict(
                        p=cfg["p"],
                        gamma=gamma,
                        beta=beta,
                        max_cut_from_runs=float(vals[best_idx]),
                        evaluation_runs=[float(v) for v in vals],
                        optimization_time=float(np.mean(runtimes)),
                    )
                    
                except Exception as e:
                    print(f"[GPU {gpu_id}] Error on graph {g['id']}: {e}")
                    patch.setdefault(g["id"], {})[cfg["key"]] = dict(
                        error=str(e),
                        p=cfg["p"]
                    )
            
            all_patches.update(patch)
            print(f"[GPU {gpu_id}] Completed {cfg['key']}")
            
            # Clean up GPU memory
            del opt
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"[GPU {gpu_id}] Failed to process model {cfg['key']}: {e}")
    
    print(f"[GPU {gpu_id}] Worker completed all models")
    return all_patches

def distribute_models_to_gpus(models: list, num_gpus: int = 3):
    """Distribute models evenly across GPUs"""
    gpu_assignments = [[] for _ in range(num_gpus)]
    
    for i, model in enumerate(models):
        gpu_idx = i % num_gpus
        gpu_assignments[gpu_idx].append(model)
    
    return gpu_assignments

def main(graphs_path="data/debug_graphs.json",
         out_path=None,
         num_gpus=3):
    
    # Setup output path
    if out_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"data/test_results_with_rl_{timestamp}.json"
    
    # Convert to absolute paths
    graphs_path = Path(graphs_path)
    if not graphs_path.is_absolute():
        graphs_path = PROJECT_ROOT / graphs_path
    
    out_path = Path(out_path)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    
    # Load graphs
    print(f"Loading graphs from {graphs_path}")
    with open(graphs_path) as f:
        graphs = json.load(f)
    print(f"Loaded {len(graphs)} graphs")
    
    # Initialize Ray with minimal resources per task
    ray.init(
        num_gpus=num_gpus,
        ignore_reinit_error=True,
        # Don't set object store memory - let Ray handle it
    )
    
    try:
        # Distribute models across GPUs
        gpu_assignments = distribute_models_to_gpus(MODELS, num_gpus)
        
        print(f"\nGPU assignments:")
        for gpu_id, models in enumerate(gpu_assignments):
            print(f"  GPU {gpu_id}: {len(models)} models - {[m['key'] for m in models]}")
        
        # Submit tasks - one per GPU
        print("\nSubmitting GPU worker tasks...")
        start_time = time.time()
        
        futures = []
        for gpu_id, model_list in enumerate(gpu_assignments):
            if model_list:  # Only submit if there are models for this GPU
                future = gpu_worker.remote(model_list, graphs, gpu_id, n_runs=5)
                futures.append(future)
        
        # Wait for results
        print(f"\nWaiting for {len(futures)} GPU workers to complete...")
        patches_list = ray.get(futures)
        
        # Merge all results
        print("\nMerging results...")
        patches_by_id = {}
        for patches in patches_list:
            for g_id, newdata in patches.items():
                patches_by_id.setdefault(g_id, {}).update(newdata)
        
        # Update graphs
        for g in graphs:
            g.update(patches_by_id.get(g["id"], {}))
        
        # Save results
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(graphs, f, indent=2)
        
        total_time = time.time() - start_time
        print(f"\n✓ Evaluation complete!")
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(f"  Results saved to: {out_path}")
        
        # Print summary statistics
        print("\nModel Performance Summary:")
        print("-" * 60)
        for model in MODELS:
            model_results = []
            for g in graphs:
                if model['key'] in g and 'max_cut_from_runs' in g[model['key']]:
                    model_results.append(g[model['key']]['max_cut_from_runs'])
            if model_results:
                print(f"{model['key']:20s}: avg = {np.mean(model_results):7.2f}, "
                      f"std = {np.std(model_results):6.2f}")
        
    finally:
        ray.shutdown()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=3, help='Number of GPUs to use')
    args = parser.parse_args()
    
    main(num_gpus=args.gpus)