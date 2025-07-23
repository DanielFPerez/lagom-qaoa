import os 
import networkx as nx
from typing import List

def show_graph(G: nx.Graph, font: str = 'white', node_size: int = 600):
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
        logger.warning(f"#### NOTE File {file_path} already exists!")
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