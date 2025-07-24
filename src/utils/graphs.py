import os 
import networkx as nx
from typing import List


GRAPH_BUILDERS = {
    "erdos_renyi": lambda p: erdos_renyi_with_retry(n=p["n"], p=p["p"]),
    "barabasi_albert": lambda p: barabasi_albert(n=p["n"], m=p["m"]),
    "realistic_geometric": lambda p: realistic_geometric_with_retry(n=p["n"]),
    "cycle": lambda p: cycle_with_spring_layout(n=p["n"]),
    #"caveman": lambda p: nx.caveman_graph(l=p["nc"], k=p["nk"]),
    #"ladder": lambda p: nx.ladder_graph(n=p["n"])
}

REQUIRED_PARAMS = {
    "erdos_renyi": {"n", "p"}, # n: number of nodes, p: probability of edge creation
    "barabasi_albert": {"n", "m"}, # n: number of nodes, m: edges to attach from a new node to existing nodes
    "realistic_geometric": {"n"},
    "cycle": {"n"}
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


def cycle_with_spring_layout(n: int, scale=1.0):
    """Generate a cycle graph with spring layout."""
    g = nx.cycle_graph(n)
    g.graph["n"] = n
    g.graph["topology"] = "cycle"
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


def show_graph(G: nx.Graph, font: str = 'white', node_size: int = 600):
    """
    Show a graph
    """
    return nx.draw(G, pos=G.nodes(data='pos'), with_labels=True, font_color=font, node_size=node_size)


def save_list_graphs_to_json(graphs: List[nx.Graph], file_path: str):
    """
    Save a list of graphs to a JSON file.
    
    Parameters:
    - graphs (list): List of NetworkX graphs.
    - file_path (str): Path to the output JSON file.
    """
    import json
    import logging
    
    logger = logging.getLogger(__name__)

    if os.path.exists(file_path):
        logger.warning(f"####  NOTE File {file_path} already exists!")
        logger.warning("Loading the existing file and appending new graphs.")
        tmp_graphs = load_list_graphs_from_json(file_path)
        graphs = tmp_graphs + graphs
    
    with open(file_path, 'w') as f:
        json.dump([nx.node_link_data(g, edges="edges") for g in graphs], f, indent=2)
    
    logger.info(f"Saved {len(graphs)} graph(s) to {file_path}")


def load_list_graphs_from_json(file_path: str) -> List[nx.Graph]:
    """
    Load a list of graphs from a JSON file.
    
    Parameters:
    - file_path (str): Path to the input JSON file.
    
    Returns:
    - List[nx.Graph]: List of NetworkX graphs.
    """
    import json
    import logging
    
    logger = logging.getLogger(__name__)
    
    with open(file_path, 'r') as f:
        graphs_data = json.load(f)
    
    graphs = [nx.node_link_graph(g, edges="edges") for g in graphs_data]
    
    logger.info(f"Loaded {len(graphs)} graph(s) from {file_path}")
    
    return graphs