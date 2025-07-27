import os
import json
import time
import networkx as nx
from typing import List, Dict
import argparse
import logging

from copy import deepcopy

import src.solvers.opt_ortools as ortools_solver
import src.utils.graphs as graphutils
import src.utils.config as utils_config
from src.utils.logger import setup_logger



def process_ortools_graph_instance(graphinst_dict: Dict, nworkers: int, timeout: int):
    ret_dict = deepcopy(graphinst_dict)
    nxgraph = graphutils.read_graph_from_dict(graphinst_dict['graph_dict'])
    solution = ortools_solver.ortools_solve_maxcut(nxgraph, nworkers=nworkers, timeout=timeout)

    ret_dict["ortools"] = {"cut_val": solution.best_obj,
                            "status": ortools_solver.ORTOOLS_STATUS_TO_STR[solution.status],
                            "partition": solution.best_partition,
                            "runtime": solution.runtime}
    return ret_dict


def solve_unsolved_graphs(graphs_list: List[Dict], solved_ids: set, nworkers: int, timeout: int, logger: logging.Logger):
    new_instances = []
    for graph in graphs_list:
        gid = graph["id"]
        if gid in solved_ids:
            logger.info(f"Graph {gid} already solved, skipping.")
            continue
        logger.info(f"Processing graph {gid}...")
        try:
            result = process_ortools_graph_instance(graph, nworkers, timeout)
            new_instances.append(result)
        except Exception as e:
            logger.exception(f"Exception while processing graph {gid}: {str(e)}")
    return new_instances



def main(src_dir: str, splits: List[str], dst_dir: str, timeout: int, nworkers: int):
    logger = logging.getLogger(__name__)

    src_read = os.path.join(utils_config.get_project_root(), src_dir)
    dst_write = os.path.join(utils_config.get_project_root(), dst_dir)
    if not os.path.exists(dst_write):
        logger.info(f"Creating saving directory: {dst_write}")
        os.makedirs(dst_write, exist_ok=True)

    for split in splits:
        solved_instances = list()
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
            logger.error(f"Source file {src_split_path} does not exist!")
            raise IOError(f"PATH DOES NOT EXISTS!")
        
        graphs_list = graphutils.open_merged_graph_json(src_split_path)
        new_instances = solve_unsolved_graphs(graphs_list, solved_ids, nworkers, timeout, logger)

        logger.info(f"Number of new solved instances: {len(new_instances)}")
        if len(new_instances) > 0:
            solved_instances.extend(new_instances)
            with open(dst_split_path, "w") as f:
                json.dump(solved_instances, f, indent=2)
            logger.info(f"Saved {len(solved_instances)} solved instances to {dst_split_path}\n")
        else:
            logger.info(f"No new instances to solve for split {split}.\n")



def get_parser():
    parser = argparse.ArgumentParser(description="Graph generator for Max-Cut experiments.")
    parser.add_argument('--src_dir', type=str, required=True, help='Parent directory to read subfiles "train.json" and "test.json".')
    parser.add_argument('--splits', nargs='+', default=["train", "test"])
    parser.add_argument('--dst_dir', type=str, required=True, help='Directory to store the results (RELATIVE to project root)')
    parser.add_argument('--timeout', type=int, default=3600, help='Timeout for the solver in seconds for each graph instance')
    parser.add_argument('--nworkers', type=int, default=16, help='Number of workers for parallel solving')
    parser.add_argument("--log_filename", type=str, default="graphgen.log",
                        help="Name of the log file")
    parser.add_argument("--log_level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--log_format", type=str, default="", help="Format for log messages.")
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
    
    main(args.src_dir, args.splits, args.dst_dir, args.timeout, args.nworkers)