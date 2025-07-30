#!/usr/bin/env python3
"""
eval_single_model.py - Evaluate a single model on a specific GPU

Usage:
    python eval_single_model.py --model-type khairy --model-path /path/to/model --p 2 --gpu 0
    python eval_single_model.py --model-type gnn --model-path /path/to/model --p 2 --gpu 1 --gnn-type GIN
"""

import argparse
import json
import os
import time
from pathlib import Path
import torch
import logging
import numpy as np
from datetime import datetime

import src.utils.graphs as graph_utils
import src.utils.config as config_utils
import scripts.run_rl_trainer_khairy as khairy_tools
import scripts.run_rl_trainer_gnn_withinit as gnnrl_tools
from src.utils.logger import setup_logger


def evaluate_single_model(args):
    """Evaluate a single model on all test graphs"""
    loggger = logging.getLogger(__name__)

    # Set GPU
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Running on GPU {args.gpu} (device: {device})")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"p value: {args.p}")
    
    # Load the model
    logger.info("Loading model...")
    if args.model_type == "khairy":
        optimizer = khairy_tools.QAOAOptimizer(
            args.model_path,
            p=args.p,
            device=device,
            hidden_dim=args.hidden_dim
        )
    elif args.model_type == "gnn":
        if not args.gnn_type or args.gnn_type not in ["GIN", "GCN", "TCN"] or not args.gnn_hidden_dim:
            raise ValueError("--gnn-type and --gnn-hidden-dim required for GNN models")
        optimizer = gnnrl_tools.GNNQAOAOptimizer(
            args.model_path,
            p=args.p,
            device=device,
            gnn_type=args.gnn_type,
            gnn_hidden_dim=args.gnn_hidden_dim,
            hidden_dim=args.hidden_dim
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    logger.info("Model loaded successfully!")
    
    # Load test graphs
    PROJECT_ROOT = config_utils.get_project_root()
    graphs_path = Path(args.graphs_path)
    if not graphs_path.is_absolute():
        graphs_path = PROJECT_ROOT / graphs_path
    
    logger.info(f"Loading graphs from {graphs_path}")
    with open(graphs_path, 'r') as f:
        graphs = json.load(f)
    logger.info(f"Loaded {len(graphs)} graphs")
    
    # Evaluate all graphs
    results = {}
    start_time = time.time()

    solver_name = f"{args.model_type}{ "_" + str(args.gnn_type) if args.model_type == 'gnn' else ''}_{args.hidden_dim}_p{args.p}"
    logger.info(f"Evaluating model {solver_name} on {len(graphs)} graphs...")
    comp_classic_name = f"nelder-mead_p{args.p}"
    graphs = graphs[0:3]
    for i, g in enumerate(graphs):
        logger.debug(f"\nEvaluating graph {g['id']} (index {i+1}/{len(graphs)})")
        if i % 10 == 0:
            elapsed = time.time() - start_time
            if i > 0:
                avg_time = elapsed / i
                remaining = (len(graphs) - i) * avg_time
                logger.info(f"Progress: {i+1}/{len(graphs)} graphs. "
                      f"Elapsed: {elapsed/60:.1f} min, "
                      f"Remaining: {remaining/60:.1f} min")
            else:
                logger.info(f"Progress: {i}/{len(graphs)} graphs")
        
        try:
            # Load graph
            G = graph_utils.read_graph_from_dict(g["graph_dict"])
            
            # Run multiple evaluations
            vals = []
            params = []
            runtimes = []
            
            for run in range(args.n_runs):
                logger.debug(f"Evaluating graph {g['id']} (run {run+1}/{args.n_runs})")
                t0 = time.perf_counter()
                # Optimize
                if hasattr(optimizer, "optimize_rl_only"):
                    prms, v = optimizer.optimize_rl_only(G)
                else:
                    prms, v = optimizer.optimize_rl_with_gnn_init(G)
                
                runtime = time.perf_counter() - t0
                
                vals.append(v)
                params.append(prms)
                runtimes.append(runtime)
            
            # Find best run
            best_idx = int(np.argmax(vals))

            logger.debug(f"OR-Tools optimal value: {g['ortools']['cut_val']}")
            logger.debug(f"Nelder-mead optimal value for p={args.p}: {g[comp_classic_name]['max_cut_from_runs']}")
            logger.debug(f"Best run value for {solver_name}: {vals[best_idx]} (index {best_idx})")

            # Extract parameters
            if isinstance(params[best_idx], dict):
                gamma = params[best_idx].get('gamma', params[best_idx].get('gammas', []))
                beta = params[best_idx].get('beta', params[best_idx].get('betas', []))
            else:
                # Assume flat array
                gamma = params[best_idx][:args.p]
                beta = params[best_idx][args.p:]
            
            # Convert to list if numpy
            if hasattr(gamma, 'tolist'):
                gamma = gamma.tolist()
            if hasattr(beta, 'tolist'):
                beta = beta.tolist()
            
            # Store results
            results[g["id"]] = {
                "p": args.p,
                "gamma": gamma,
                "beta": beta,
                "max_cut_from_runs": float(vals[best_idx]),
                "evaluation_runs": [float(v) for v in vals],
                "optimization_time": float(np.mean(runtimes)),
                "optimization_time_std": float(np.std(runtimes))
            }
            
        except Exception as e:
            logger.info(f"Error on graph {g['id']}: {e}")
            results[g["id"]] = {
                "p": args.p,
                "error": str(e)
            }
        except KeyboardInterrupt:
            logger.info("Evaluation interrupted by user.")
            logger.info("Saving partial results...")
            break
    
    # Save results
    output_name = solver_name
    
    output_path = os.path.join(os.path.join(config_utils.get_project_root(), args.output_dir), f"{output_name}_results.json")
    logger.info(f"Saving results to {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    total_time = time.time() - start_time
    logger.info(f"\n✓ Evaluation complete!")
    logger.info(f"  Total time: {total_time/60:.1f} minutes")
    logger.info(f"  Results saved to: {output_path}")
    


def get_parser():
    parser = argparse.ArgumentParser(description='Evaluate a single model on test graphs')
    
    # Required arguments
    parser.add_argument('--model-type', type=str, required=True, choices=['khairy', 'gnn'],
                        help='Type of model (khairy or gnn)')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model file (e.g., best_model.pth)')
    parser.add_argument('--p', type=int, required=True, choices=[1, 2, 3],
                        help='QAOA depth p')
    parser.add_argument('--gpu', type=int, required=True,
                        help='GPU ID to use (0, 1, or 2)')
    
    # Optional arguments
    parser.add_argument('--gnn-type', type=str, choices=['GIN', 'GCN', 'TCN'],
                        help='GNN type (required for gnn models)')
    parser.add_argument('--graphs-path', type=str, 
                        default='data/optimized_graphs_classic/test_results.json',
                        help='Path to input graphs JSON')
    parser.add_argument('--output-dir', type=str, default='data/indiv_rlmodel_results',
                        help='Directory to save results')
    parser.add_argument('--n-runs', type=int, default=5,
                        help='Number of evaluation runs per graph')
    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='Hidden dimension for models')
    parser.add_argument('--gnn-hidden-dim', type=int, default=256,
                        help='GNN hidden dimension')
    
    # Logging parameters
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--log-format", type=str, default="", 
                        help="Format for log messages")
    
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # Create log filename based on tiemstamp and model type
    timestamp = datetime.now().strftime("%m%d_%H%M")
    log_filename = f"{args.model_type}{ "_" + str(args.gnn_type) if args.model_type == "gnn" else ''}_p{args.p}_{timestamp}.log"  

    log_path = os.path.join(config_utils.get_project_root(), f"outputs/logs/model_eval/{log_filename}")

    # Log level from arguments
    log_level = getattr(logging, args.log_level, logging.INFO)

    # Setup logging with arguments
    setup_logger(log_file_path=log_path, log_format=args.log_format,
                 console_level=log_level, 
                 file_level=log_level,
                 console_logging=False)
    
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    if args.model_type == 'gnn' and not args.gnn_type:
        parser.error("--gnn-type is required for GNN models")
    
    evaluate_single_model(args)