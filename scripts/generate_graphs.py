#!/usr/bin/env python3

import argparse
import json
import os
import sys
import random
import scipy
import networkx as nx
import logging
import numpy as np

sys.path.append("../")
from src.utils.logger import setup_logger
import src.utils.config as utils_config 
from src.utils.graphs import save_list_graphs_to_json, GRAPH_BUILDERS, REQUIRED_PARAMS



def build_graphs(gtype: str, params, num: int):
    """Create *num* NetworkX graphs of *gtype* with *params*."""
    graphs = []
    for i in range(num):
        params_with_seed = dict(params)  # shallow copy
        params_with_seed.setdefault("seed", random.randrange(2**32))
        g = GRAPH_BUILDERS[gtype](params_with_seed)
        graphs.append(g)
    return graphs



def parse_kv_pairs(kv_list):
    """Convert key=value pairs into correctly‑typed dict."""
    params = {}
    logging.debug(f"Parsing parameters: {kv_list}")
    for item in kv_list:
        if '=' not in item:
            raise ValueError(f"Malformed --params entry '{item}'. Use 'key=value' format.")
        key, value = item.split('=', 1)
        # Heuristically cast to int or float when possible
        try:
            value_cast = int(value)
        except ValueError:
            try:
                value_cast = float(value)
            except ValueError:
                value_cast = value
        params[key] = value_cast
    return params



def get_parser():
    parser = argparse.ArgumentParser(description="Graph generator for Max-Cut experiments.")
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to store generated graphs (RELATIVE to project root)')
    parser.add_argument('--graph_type', type=str, required=True,
                        choices=GRAPH_BUILDERS.keys(),
                        help='Type of graph to generate')
    parser.add_argument('--params', nargs='+', required=True,
                        help='Graph generation parameters in the form key=value')
        # "erdos_renyi": {n: int, p: float}
        # "barabasi_albert": {n: int, m: int (<n)}
        # "realistic_geometric": {n: int}
    parser.add_argument('--n_graphs', type=int, required=True,
                        help='Number of graphs to generate')
    parser.add_argument("--log_filename", type=str, default="graphgen.log",
                        help="Name of the log file")
    parser.add_argument("--log_level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--log_format", type=str, default="", help="Format for log messages.")
    return parser



def main(args):
    try:
        param_dict = parse_kv_pairs(args.params)
    except ValueError as exc:
        logging.error(exc, file=sys.stderr)
        sys.exit(1)

    required_keys = REQUIRED_PARAMS[args.graph_type]

    missing = required_keys - param_dict.keys()
    if missing:
        logging.error(f"Missing required parameter(s) for {args.graph_type}: {', '.join(sorted(missing))}")
        raise ValueError(f"Missing required parameter(s) for {args.graph_type}: {', '.join(sorted(missing))}")

    # Build graphs
    graphs = build_graphs(args.graph_type, param_dict, args.n_graphs)

    # Make sure directory exists
    out_dir = os.path.join(utils_config.get_project_root(), args.save_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Compose file name
    param_str = "_".join([str(key) + "-" + str(val) for key,val in param_dict.items()])
    file_name = f"{args.graph_type}_{param_str}_ngraphs-{args.n_graphs}.json"
    logging.info(f"Name of the file to save the graphs: {file_name}")

    file_path = os.path.join(args.save_dir, file_name)
    
    save_list_graphs_to_json(graphs, file_path)

    logging.info(f"Saved {len(graphs)} graph(s) to {file_path}")
    
    

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
    
    # Process the dataset files
    logging.info(f"Selected graph type: {args.graph_type}")
    logging.info(f"Starting graph generation with parameters: {args.params}")
    main(args)
    logging.info(f"Graph generation completed successfully. Graphs saved to {args.save_dir}")


