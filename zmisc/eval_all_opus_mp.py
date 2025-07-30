# eval_all_simple.py
import json
import os
from pathlib import Path
import torch
import multiprocessing as mp
from functools import partial
import time
import numpy as np
from datetime import datetime

# Your imports
import src.utils.graphs as graph_utils
import src.utils.config as config_utils
import scripts.run_rl_trainer_khairy as khairy_tools
import scripts.run_rl_trainer_gnn_withinit as gnnrl_tools

def evaluate_models_on_gpu(gpu_id, model_configs, graphs, n_runs=5):
    """Function that runs in a separate process for each GPU"""
    # Set this process to use only the assigned GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"  # Will map to the assigned GPU
    
    print(f"[GPU {gpu_id}] Process started with {len(model_configs)} models")
    
    all_results = {}
    
    for cfg in model_configs:
        print(f"[GPU {gpu_id}] Processing {cfg['key']}")
        
        try:
            # Load model
            opt = cfg["cls"](
                str(cfg["path"]),
                p=cfg["p"],
                device=device,
                **cfg["kwargs"],
            )
            
            # Evaluate all graphs
            for i, g in enumerate(graphs):
                if i % 50 == 0:
                    print(f"[GPU {gpu_id}] {cfg['key']}: {i}/{len(graphs)} graphs")
                
                G = graph_utils.read_graph_from_dict(g["graph_dict"])
                vals, params = [], []
                
                for _ in range(n_runs):
                    if hasattr(opt, "optimize_rl_only"):
                        prms, v = opt.optimize_rl_only(G)
                    else:
                        prms, v = opt.optimize_rl_with_gnn_init(G)
                    
                    vals.append(v)
                    params.append(prms)
                
                best_idx = int(np.argmax(vals))
                gamma = params[best_idx][:cfg["p"]]
                beta = params[best_idx][cfg["p"]:]
                
                # Convert to list
                if hasattr(gamma, 'tolist'):
                    gamma = gamma.tolist()
                if hasattr(beta, 'tolist'):
                    beta = beta.tolist()
                
                if g["id"] not in all_results:
                    all_results[g["id"]] = {}
                
                all_results[g["id"]][cfg["key"]] = {
                    "p": cfg["p"],
                    "gamma": gamma,
                    "beta": beta,
                    "max_cut_from_runs": float(vals[best_idx]),
                    "evaluation_runs": [float(v) for v in vals]
                }
            
            # Clean up
            del opt
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"[GPU {gpu_id}] Error with {cfg['key']}: {e}")
    
    return all_results

def main():
    PROJECT_ROOT = config_utils.get_project_root()
    
    # Define models (same as before)
    KH1 = PROJECT_ROOT + "/outputs/07-30_01-23_rl_model_khairy_p{p}"
    GCN = PROJECT_ROOT + "/outputs/07-30_01-19_gnn_rl_model_GCN_p{p}_withinit"
    TCN = PROJECT_ROOT + "/outputs/07-30_01-19_gnn_rl_model_TCN_p{p}_withinit"
    GIN = PROJECT_ROOT + "/outputs/07-30_01-19_gnn_rl_model_GIN_p{p}_withinit"
    
    MODELS = []
    hidden_dim = 512
    
    # Build model list (same as before)
    for family, label in [(KH1, "khairy_v1")]:
        for p in (1, 2, 3):
            MODELS.append(dict(
                key=f"{label}_p{p}",
                p=p,
                path=family.format(p=p) + "/best_model.pth",
                cls=khairy_tools.QAOAOptimizer,
                kwargs=dict(hidden_dim=hidden_dim),
            ))
    
    for gnn_type, templ in [("GIN", GIN), ("GCN", GCN), ("TCN", TCN)]:
        for p in (1, 2, 3):
            path = templ.format(p=p) + f"/best_model_{gnn_type}_with_init.pth"
            MODELS.append(dict(
                key=f"{gnn_type.upper()}_p{p}",
                p=p,
                path=path,
                cls=gnnrl_tools.GNNQAOAOptimizer,
                kwargs=dict(
                    gnn_type=gnn_type,
                    gnn_hidden_dim=256,
                    hidden_dim=hidden_dim,
                ),
            ))
    
    # Load graphs
    graphs_path = PROJECT_ROOT + "/data/debug_graphs.json"
    with open(graphs_path) as f:
        graphs = json.load(f)
    
    print(f"Loaded {len(graphs)} graphs")
    print(f"Total models: {len(MODELS)}")
    
    # Distribute models to GPUs
    num_gpus = 3
    gpu_assignments = [[] for _ in range(num_gpus)]
    for i, model in enumerate(MODELS):
        gpu_assignments[i % num_gpus].append(model)
    
    # Print assignment
    for gpu_id, models in enumerate(gpu_assignments):
        print(f"GPU {gpu_id}: {[m['key'] for m in models]}")
    
    # Run in parallel
    start_time = time.time()
    
    with mp.Pool(processes=num_gpus) as pool:
        # Create tasks
        tasks = []
        for gpu_id, model_list in enumerate(gpu_assignments):
            if model_list:
                task = pool.apply_async(
                    evaluate_models_on_gpu,
                    args=(gpu_id, model_list, graphs)
                )
                tasks.append(task)
        
        # Wait for completion
        results = []
        for task in tasks:
            results.append(task.get())
    
    # Merge results
    final_results = {}
    for gpu_results in results:
        for graph_id, model_results in gpu_results.items():
            if graph_id not in final_results:
                final_results[graph_id] = {}
            final_results[graph_id].update(model_results)
    
    # Update original graphs
    for g in graphs:
        if g["id"] in final_results:
            g.update(final_results[g["id"]])
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = PROJECT_ROOT + f"/data/test_results_with_rl_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(graphs, f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed/60:.1f} minutes")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()