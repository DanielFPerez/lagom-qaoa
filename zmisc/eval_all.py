# eval_all.py
import ray, json, os, itertools
from pathlib import Path

import scripts.run_rl_trainer_khairy as khairy_tools
import scripts.run_rl_trainer_gnn_withinit as gnnrl_tools

import torch, random, os, networkx as nx
import src.utils.graphs as graph_utils
import src.utils.config as config_utils

import time, numpy as np


PROJECT_ROOT = config_utils.get_project_root()

KH1 = PROJECT_ROOT + "/outputs/07-30_01-23_rl_model_khairy_p{p}"      # first Khairy family
# KH2 = PROJECT_ROOT / "outputs/07-31_09-02_rl_model_khairy_big"  # second Khairy family
GAT = PROJECT_ROOT + "/outputs/07-30_01-20_gnn_rl_model_GAT_p{p}_withinit"
TCN = PROJECT_ROOT + "/outputs/07-30_01-21_gnn_rl_model_TCN_p{p}_withinit"
GIN = PROJECT_ROOT + "/outputs/07-30_01-19_gnn_rl_model_GIN_p{p}_withinit"

MODELS = []
hidden_dim = 512            # <-- tweak globally once

# ── 2 Khairy families ───────────────────────────────────────────
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

# ── 3 GNN families ──────────────────────────────────────────────
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


# --- Evaluate a single model on a set of graphs ---
def evaluate_model(cfg: dict, graphs_path: str, device: str = "cuda",
                   n_runs: int = 5, seed0: int = 1):
    """
    Returns a patch of the JSON:
        {graph_id: {cfg["key"]: {...}}, ...}
    """
    # Inside a Ray worker exactly ONE GPU is visible → always index 0
    if torch.cuda.is_available():
        print(f"Using GPU for model {cfg['key']}")
        device = "cuda:0"

    rng = random.Random(seed0 + hash(cfg["key"]) % 10_000)
    torch.manual_seed(rng.randint(0, 2**31 - 1))

    with open(graphs_path) as f:
        graphs = json.load(f)

    # ── lazy-load optimiser once per process ──
    opt = cfg["cls"](
        str(cfg["path"]),
        p=cfg["p"],
        device=device,
        **cfg["kwargs"],
    )

    patch = {}

    for i_graph, g in enumerate(graphs):
        if (i_graph % 50) == 0:
            print(f"[{cfg['key']} | {device}] done {i_graph}/{len(graphs)} graphs")

        G = graph_utils.read_graph_from_dict(g["graph_dict"])
        vals, params, runtimes = [], [], []

        for _ in range(n_runs):
            torch.cuda.synchronize()               # ensure clean timing
            t0 = time.perf_counter()

            prms, v = (
                opt.optimize_rl_only(G)
                if hasattr(opt, "optimize_rl_only")
                else opt.optimize_rl_with_gnn_init(G)
            )

            torch.cuda.synchronize()               # wait for kernel to finish
            runtimes.append(time.perf_counter() - t0)

            vals.append(v)
            params.append(prms)

        best_idx = int(np.argmax(vals))
        gamma, beta = params[best_idx][: cfg["p"]], params[best_idx][cfg["p"]:]

        patch.setdefault(g["id"], {})[cfg["key"]] = dict(
            p=cfg["p"],
            gamma=gamma,
            beta=beta,
            max_cut_from_runs=vals[best_idx],
            evaluation_runs=vals,
            optimization_time=float(np.mean(runtimes)),   # <── NEW
        )

    return patch


# ── Ray setup ───────────────────────────────────────────────────
ray.init(num_gpus=3, ignore_reinit_error=True)

@ray.remote(num_gpus=1)
def _worker(cfg, graphs_path):
    return evaluate_model(cfg, graphs_path, device="cuda")


def main(graphs_path="data/debug_graphs.json",
         out_path="data/test_results_with_rl.json"):
    
    futures = [_worker.remote(m, graphs_path) for m in MODELS]
    
    patches = ray.get(futures)            # waits for all 15 jobs

    # ── merge ───────────────────────────────────
    with open(graphs_path) as f:
        graphs = json.load(f)

    patches_by_id = {}
    for p in patches:
        for g_id, newdata in p.items():
            patches_by_id.setdefault(g_id, {}).update(newdata)

    for g in graphs:
        g.update(patches_by_id.get(g["id"], {}))

    # merge project root with output path
    out_path = Path(out_path)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    Path(out_path).write_text(json.dumps(graphs, indent=2))
    print(f"Saved combined results → {out_path}")


if __name__ == "__main__":
    main()