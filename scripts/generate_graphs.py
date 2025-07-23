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
from src.utils.graphs import save_list_graphs_to_json


GRAPH_BUILDERS = {
    "erdos_renyi": lambda p: erdos_renyi_with_retry(n=p["n"], p=p["p"]),
    "barabasi_albert": lambda p: barabasi_albert(n=p["n"], m=p["m"]),
    "realistic_geometric": lambda p: realistic_geometric_with_retry(n=p["n"]),
    #"caveman": lambda p: nx.caveman_graph(l=p["nc"], k=p["nk"]),
    #"ladder": lambda p: nx.ladder_graph(n=p["n"])
}

REQUIRED_PARAMS = {
    "erdos_renyi": {"n", "p"}, # n: number of nodes, p: probability of edge creation
    "barabasi_albert": {"n", "m"}, # n: number of nodes, m: edges to attach from a new node to existing nodes
    "realistic_geometric": {"n"},
    #"caveman": {"nc", "nk"},
    #"ladder": {"n"}
}


def with_spring_layout(g: nx.Graph, seed=None, scale=1.0):
    """Assign spring-layout positions as node features."""
    positions = nx.spring_layout(g, seed=seed, scale=scale)
    for node, pos in positions.items():
        g.nodes[node]["pos"] = tuple(pos)
    return g


def barabasi_albert(n: int, m: int, scale=1.0):
    """Generate a Barabasi-Albert graph with spring layout."""
    if m < 1 or m >= n:
        raise ValueError("Parameter 'm' must be in the range [1, n-1].")
    g = nx.barabasi_albert_graph(n, m)
    g.graph["n"] = n
    g.graph["m"] = m
    g.graph["topology"] = "barabasi_albert"
    return with_spring_layout(g, scale=scale)


def erdos_renyi_with_retry(n: int, p: float, max_retries=1000):

    """Generate an Erdos-Renyi graph with retries on failure."""
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        g = nx.erdos_renyi_graph(n, p)
        if nx.is_connected(g):
            g.graph["n"] = n
            g.graph["p"] = p
            g.graph["topology"] = "erdos_renyi"
            return with_spring_layout(g)
    
    raise Exception(f"Failed to generate a connected Erdos-Renyi graph "
                    f"in {max_retries} tries.")


def realistic_geometric_with_retry(n: int, side: int = 100, max_retries: int = 1000):
    """Generate a realistic geometric graph with retries on failure."""

    _CC2538_TX_POWER = 7
    _ANT_GAIN = 3
    _FREQUENCY = 2.45e9
    _WAVELENGTH = scipy.constants.c / _FREQUENCY # Wavelength in m, for frequency G=2.45GHz #
    _THR_DISTANCE = 50
    scale_ratio = n/12

    def generate_positions(n_nodes, in_side, in_scale_ratio):
        scale = in_side * np.sqrt(in_scale_ratio)
        return {
            i: (np.random.uniform(high=scale), np.random.uniform(high=scale))
            for i in range(n_nodes)
        }

    attempt = 0
    while attempt < max_retries:
        attempt += 1
        positions = generate_positions(n, side, scale_ratio)
        g = nx.random_geometric_graph(n, _THR_DISTANCE, pos=positions)
        if nx.is_connected(g):
            g.graph["n"] = n
            g.graph["side"] = side
            g.graph["scale_ratio"] = scale_ratio
            g.graph["topology"] = "realistic_geometric"
            for node_id, pos in positions.items():
                g.nodes[node_id]['pos'] = pos
            return g
        
    raise Exception(f"Failed to generate a connected graph "
                    f"in {max_retries} tries.")


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


