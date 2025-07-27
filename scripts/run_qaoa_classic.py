import os
import json
import time
import networkx as nx
from typing import List, Dict
import argparse
import logging
import numpy as np
from copy import deepcopy

import src.utils.graphs as graphutils
import src.utils.config as utils_config
from src.utils.logger import setup_logger

# Import your QAOA functions
from src.solvers.classic_nlopt import optimize_qaoa_nlopt
from src.qaoa.qaoa_model import eval_circuit


def evaluate_qaoa_multiple_runs(graph: nx.Graph, p: int, gamma: np.ndarray, beta: np.ndarray, 
                               n_runs: int = 10, reps_per_run: int = 1000) -> Dict:
    """
    Evaluate QAOA circuit multiple times to get statistics
    
    Args:
        graph: NetworkX graph
        p: QAOA depth
        gamma: Optimized gamma parameters
        beta: Optimized beta parameters
        n_runs: Number of evaluation runs
        reps_per_run: Number of circuit repetitions per run
        
    Returns:
        Dictionary with evaluation results
    """
    cut_values = []
    
    for i in range(n_runs):
        cut_val = eval_circuit(graph, p, gamma, beta, reps=reps_per_run)
        cut_values.append(cut_val)
    
    return {
        "cut_values": cut_values,
        "max_cut": max(cut_values),
        "mean_cut": np.mean(cut_values),
        "std_cut": np.std(cut_values)
    }


def process_qaoa_graph_instance(graphinst_dict: Dict, p: int, method: str, 
                               xtol_rel: float, ftol_abs: float,
                               n_eval_runs: int = 10, reps_per_run: int = 1000,
                               logger: logging.Logger = None) -> Dict:
    """
    Process a single graph instance with QAOA
    
    Args:
        graphinst_dict: Dictionary containing graph data
        p: QAOA depth
        method: Optimization method ('COBYLA', 'NELDER-MEAD', 'BFGS')
        xtol_rel: Relative tolerance for x
        ftol_abs: Absolute tolerance for function value
        n_eval_runs: Number of evaluation runs after optimization
        reps_per_run: Number of circuit repetitions per evaluation run
        logger: Logger instance
        
    Returns:
        Updated dictionary with QAOA results
    """
    ret_dict = deepcopy(graphinst_dict)
    nxgraph = graphutils.read_graph_from_dict(graphinst_dict['graph_dict'])
    
    # Run QAOA optimization
    start_time = time.time()
    try:
        optimized_params, f_opt, status_code = optimize_qaoa_nlopt(
            nxgraph, p, method=method, 
            in_xtol_rel=xtol_rel, 
            in_ftol_abs=ftol_abs
        )
        optimization_time = time.time() - start_time
        
        # Extract gamma and beta from optimized parameters
        gamma = optimized_params[:p]
        beta = optimized_params[p:]
        
        # Evaluate circuit multiple times
        eval_results = evaluate_qaoa_multiple_runs(
            nxgraph, p, gamma, beta, 
            n_runs=n_eval_runs, 
            reps_per_run=reps_per_run
        )
        
        # Store results with method name as key
        method_key = f"{method.lower()}_p{p}"
        ret_dict[method_key] = {
            "p": p,
            "gamma": gamma.tolist(),
            "beta": beta.tolist(),
            "optimization_time": optimization_time,
            "optimization_status": int(status_code),
            "optimized_objective": -f_opt,  # Convert back from negative
            "evaluation_runs": eval_results["cut_values"],
            "max_cut_from_runs": eval_results["max_cut"],
            "mean_cut_from_runs": eval_results["mean_cut"],
            "std_cut_from_runs": eval_results["std_cut"],
            "n_eval_runs": n_eval_runs,
            "reps_per_run": reps_per_run
        }
        
        if logger:
            logger.info(f"QAOA optimization completed. Max cut: {eval_results['max_cut']:.3f}, Ortools MaxCut: {ret_dict.get('ortools', {}).get('cut_val', 'N/A')}, " +
                       f"Mean: {eval_results['mean_cut']:.3f}, Time: {optimization_time:.2f}s")
        
    except Exception as e:
        if logger:
            logger.error(f"Error during QAOA optimization: {str(e)}")
        method_key = f"{method.lower()}_p{p}"
        ret_dict[method_key] = {
            "error": str(e),
            "status": "failed"
        }
    
    return ret_dict


def solve_unsolved_graphs(graphs_list: List[Dict], solved_ids: set, 
                         p: int, method: str, xtol_rel: float, ftol_abs: float,
                         n_eval_runs: int, reps_per_run: int,
                         logger: logging.Logger):
    """
    Process graphs that haven't been solved yet with the specified method
    """
    new_instances = []
    method_key = f"{method.lower()}_p{p}"
    
    for graph in graphs_list:
        gid = graph["id"]
        
        # Check if already solved with this specific method and p value
        if gid in solved_ids and method_key in graph:
            logger.info(f"Graph {gid} already solved with {method_key}, skipping.")
            continue
            
        logger.info(f"Processing graph {gid} with {method_key}...")
        try:
            result = process_qaoa_graph_instance(
                graph, p, method, xtol_rel, ftol_abs,
                n_eval_runs, reps_per_run, logger
            )
            new_instances.append(result)
        except Exception as e:
            logger.exception(f"Exception while processing graph {gid}: {str(e)}")
    
    return new_instances


def merge_results(existing_instances: List[Dict], new_instances: List[Dict]) -> List[Dict]:
    """
    Merge new results with existing ones, updating graphs that already exist
    """
    # Create a mapping of id to instance for existing data
    existing_map = {inst["id"]: inst for inst in existing_instances}
    
    # Update with new results
    for new_inst in new_instances:
        gid = new_inst["id"]
        if gid in existing_map:
            # Update existing instance with new QAOA results
            existing_map[gid].update(new_inst)
        else:
            # Add new instance
            existing_map[gid] = new_inst
    
    return list(existing_map.values())


def main(src_dir: str, splits: List[str], dst_dir: str, 
         p: int, method: str, xtol_rel: float, ftol_abs: float,
         n_eval_runs: int, reps_per_run: int):
    logger = logging.getLogger(__name__)

    src_read = os.path.join(utils_config.get_project_root(), src_dir)
    dst_write = os.path.join(utils_config.get_project_root(), dst_dir)
    if not os.path.exists(dst_write):
        logger.info(f"Creating saving directory: {dst_write}")
        os.makedirs(dst_write, exist_ok=True)

    for split in splits:
        solved_instances = []
        solved_ids = set()

        dst_split_path = os.path.join(dst_write, f"{split}_results.json")
        logger.info(f"File to save results: {dst_split_path}")
        
        if os.path.exists(dst_split_path):
            logger.warning(f"File already exists!")
            logger.info(f"Reading {split} split from {dst_split_path}")
            with open(dst_split_path, "r") as f:
                solved_instances = json.load(f)
            solved_ids = {elem["id"] for elem in solved_instances}

        src_split_path = os.path.join(src_read, f"{split}.json")
        logger.info(f"File to read {split} graph json from: {src_split_path}")
        if not os.path.exists(src_split_path):
            # Try reading from results file if source doesn't exist
            src_split_path = os.path.join(src_read, f"{split}_results.json")
            if not os.path.exists(src_split_path):
                logger.error(f"Source file {src_split_path} does not exist!")
                raise IOError(f"PATH DOES NOT EXISTS!")
        
        if src_split_path.endswith("_results.json"):
            # Reading from results file
            with open(src_split_path, "r") as f:
                graphs_list = json.load(f)
        else:
            # Reading from raw graph file
            graphs_list = graphutils.open_merged_graph_json(src_split_path)
        
        new_instances = solve_unsolved_graphs(
            graphs_list, solved_ids, p, method, 
            xtol_rel, ftol_abs, n_eval_runs, reps_per_run, logger
        )

        logger.info(f"Number of new solved instances: {len(new_instances)}")
        if len(new_instances) > 0:
            # Merge results if we have existing instances
            if solved_instances:
                merged_instances = merge_results(solved_instances, new_instances)
            else:
                merged_instances = new_instances
                
            with open(dst_split_path, "w") as f:
                json.dump(merged_instances, f, indent=2)
            logger.info(f"Saved {len(merged_instances)} instances to {dst_split_path}\n")
        else:
            logger.info(f"No new instances to solve for split {split}.\n")


def get_parser():
    parser = argparse.ArgumentParser(description="QAOA solver for Max-Cut experiments.")
    parser.add_argument('--src_dir', type=str, required=True, 
                        help='Parent directory to read graph files or results files.')
    parser.add_argument('--splits', nargs='+', default=["train", "test"])
    parser.add_argument('--dst_dir', type=str, required=True, 
                        help='Directory to store the results (RELATIVE to project root)')
    
    # QAOA specific parameters
    parser.add_argument('--p', type=int, default=1, 
                        help='QAOA depth parameter')
    parser.add_argument('--method', type=str, default='NELDER-MEAD',
                        choices=['COBYLA', 'NELDER-MEAD', 'BFGS'],
                        help='Optimization method')
    parser.add_argument('--xtol_rel', type=float, default=1e-4,
                        help='Relative tolerance for x')
    parser.add_argument('--ftol_abs', type=float, default=1e-3,
                        help='Absolute tolerance for function value')
    
    # Evaluation parameters
    parser.add_argument('--n_eval_runs', type=int, default=10,
                        help='Number of evaluation runs after optimization')
    parser.add_argument('--reps_per_run', type=int, default=1000,
                        help='Number of circuit repetitions per evaluation run')
    
    # Logging parameters
    parser.add_argument("--log_filename", type=str, default="qaoa_solver.log",
                        help="Name of the log file")
    parser.add_argument("--log_level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--log_format", type=str, default="", 
                        help="Format for log messages.")
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    log_path = os.path.join(utils_config.get_project_root(), f"outputs/logs/{args.log_filename}")

    # log level from arguments
    log_level = getattr(logging, args.log_level, logging.INFO)

    # Setup logging with arguments
    setup_logger(log_file_path=log_path, log_format=args.log_format,
                 console_level=log_level, 
                 file_level=log_level)
    
    main(args.src_dir, args.splits, args.dst_dir, 
         args.p, args.method, args.xtol_rel, args.ftol_abs,
         args.n_eval_runs, args.reps_per_run)