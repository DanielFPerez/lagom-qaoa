import argparse
import json
import os
import sys
import random
import scipy
import networkx as nx
import numpy as np


GRAPH_BUILDERS = {
    "erdos_renyi": lambda p: nx.erdos_renyi_graph(n=p["n"], p=p["p"], seed=p.get("seed")),
    "barabasi_albert": lambda p: nx.barabasi_albert_graph(n=p["n"], m=p["m"], seed=p.get("seed")),
    "realistic_geometric": lambda p: realistic_geometric_with_retry(n=p["n"]),
    "caveman": lambda p: nx.caveman_graph(l=p["nc"], k=p["nk"]),
    "ladder": lambda p: nx.ladder_graph(n=p["n"])
}

REQUIRED_PARAMS = {
    "erdos_renyi": {"n", "p"}, # n: number of nodes, p: probability of edge creation
    "barabasi_albert": {"n", "m"}, # n: number of nodes, m: edges to attach from a new node to existing nodes
    "realistic_geometric": {"n"},
    "caveman": {"nc", "nk"},
    "ladder": {"n"}
}


def realistic_geometric_with_retry(n, side=100, max_retries=1000):
    """Generate a realistic geometric graph with retries on failure."""

    _CC2538_TX_POWER = 7
    _ANT_GAIN = 3
    _WAVELENGTH = scipy.constants.c / _FREQUENCY # Wavelength in m, for frequency G=2.45GHz #
    _FREQUENCY = 2.45e9
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
            for node_id, pos in positions.items():
                g.nodes[node_id]['pos'] = pos
            return g
        
    raise Exception(f"Failed to generate a connected graph "
                    f"in {max_retries} tries.")


def build_graphs(gtype, params, num):
    """Create *num* NetworkX graphs of *gtype* with *params*."""
    graphs = []
    for i in range(num):
        params_with_seed = dict(params)  # shallow copy
        params_with_seed.setdefault("seed", random.randrange(2**32))
        g = GRAPH_BUILDERS[gtype](params_with_seed)
        g.graph["topology"] = gtype
        for key, val in params:
            g.graph[key] = val
        graphs.append(g)
    return graphs



def parse_kv_pairs(kv_list):
    """Convert key=value pairs into correctly‑typed dict."""
    params = {}
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
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to store generated graphs')
    parser.add_argument('--graph_type', type=str, required=True,
                        choices=GRAPH_BUILDERS.keys(),
                        help='Type of graph to generate')
    parser.add_argument('--params', nargs='+', required=True,
                        help='Graph generation parameters in the form key=value')
    parser.add_argument('--n_graphs', type=int, required=True,
                        help='Number of graphs to generate')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    try:
        param_dict = parse_kv_pairs(args.params)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    required_keys = REQUIRED_PARAMS[args.graph_type]

    missing = required_keys - param_dict.keys()
    if missing:
        print(f"Missing required parameter(s) for {args.graph_type}: {', '.join(sorted(missing))}", file=sys.stderr)
        sys.exit(1)

    # Build graphs
    graphs = build_graphs(args.graph_type, param_dict, args.num_graphs)
    # Prepare JSON‑serialisable list
    graphs_json = [nx.readwrite.json_graph.node_link_data(g, edges="edges") for g in graphs]

    # Make sure directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Compose file name
    param_str = "_".join([str(key) + "-" + str(val) for key,val in param_dict.items()])
    file_name = f"{args.graph_type}_{param_str}_ngraphs-{args.num_graphs}.json"
    file_path = os.path.join(args.output_dir, file_name)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(graphs_json, f, indent=2)

    print(f"Saved {len(graphs)} graph(s) to {file_path}")
    

if __name__ == '__main__':
    main()
