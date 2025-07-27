import os
import json
import argparse
import logging
from typing import List, Dict
import src.utils.graphs as graphutils
import src.utils.config as utils_config
from src.utils.logger import setup_logger


def get_indiv_graphs_paths(src_graph_dir):
    return [os.path.join(src_graph_dir, elem) for elem in os.listdir(src_graph_dir) if elem.endswith(".json")]


def process_graph_instance(graph_dict: Dict):
    """" creates a dict with hash ID of the graph, the graph, topology and parameters for having created the graph"""
    graph_id = utils_config.hash_graph_dict(graph_dict)
    topology = graph_dict['graph']['topology']
    metadata = {elem: graph_dict['graph'][elem] for elem in list(graphutils.REQUIRED_PARAMS[topology])}
    return {"id": graph_id, 
            "graph_dict": graph_dict, 
            "n_nodes": graph_dict['graph']['n'],
            "topology": topology, 
            "topology_metadata": metadata}

def merge_graphs(src_dir: str = "data/graphs_khairy/", splits: List[str]=["train", "test"], dst_dir: str = "data/merged"):
    """
    Merges individual graph json files from the source directory into a single json file for each split
    Args:
        src_dir (str): Source directory containing individual graph json files.
        splits (List[str]): List of splits to process (e.g., ["train", "test"]).
        dst_dir (str): Destination directory to save the merged json files.
    """
    logger = logging.getLogger(__name__)

    src_read = os.path.join(utils_config.get_project_root(), src_dir)
    dst_write = os.path.join(utils_config.get_project_root(), dst_dir)

    if not os.path.exists(dst_write):
        logger.info(f"Creating saving directory: {dst_write}")
        os.makedirs(dst_write, exist_ok=True)

    for split in splits: 
        merged_split = list()
        existing_ids = set()
        
        src_split_dir = os.path.join(src_read, split)
        logger.info(f"Folder to read individual graph jsons from: {src_split_dir}")
        if not os.path.exists(src_split_dir):
            logger.error(f"Source directory {src_split_dir} does not exist!")
            raise IOError(f"PATH DOES NOT EXISTS!")
        
        dst_split_dir = os.path.join(dst_write, f"{split}.json")
        logger.info(f"File to save results: {dst_split_dir}")
        if os.path.exists(dst_split_dir):
            logger.warning(f"File already existst!")
            logger.info(f"Reading {split} split from {src_split_dir}")
            with open(dst_split_dir, "r") as f:
                merged_split = json.load(f) 
            existing_ids = {elem["id"] for elem in merged_split}
            if len(existing_ids) != len(merged_split):
                logger.error(f"The # of ids {len(existing_ids)} does not match with the number of elements {len(merged_split)}")
                raise AssertionError(f"The # of ids {len(existing_ids)} does not match with the number of elements {len(merged_split)}")
            logger.info(f"Loaded graphs list. Extending the list...")
            
        graphdirs = get_indiv_graphs_paths(src_split_dir)
        logger.info(f"Found {len(graphdirs)} graph files.")
        
        logger.info(f"Processing {len(graphdirs)} files...")
        for i, tmpdir in enumerate(graphdirs):
            if i % 2 == 0:
                logger.debug(f"processing file {i}")
            with open(tmpdir, "r") as f:
                tmpgraphs = json.load(f)
            processed = [process_graph_instance(elem) for elem in tmpgraphs]
            if len(existing_ids) > 0:
                cached_len = len(processed)
                processed = [elem for elem in processed if elem["id"] not in existing_ids]
                if cached_len != len(processed): 
                    logger.info(f"Found {cached_len-len(processed)} elements that had already been added.")
            
            merged_split += processed
        logger.info(f"Finished processing {split} split.")
        logger.info(f"Saving data to {dst_split_dir}")
        with open(dst_split_dir, "w") as f:
            json.dump(merged_split, f, indent=2)
    logger.info(f"Finished merging the graphs the graphs")        



def get_parser():
    parser = argparse.ArgumentParser(description="Graph generator for Max-Cut experiments.")
    parser.add_argument('--src_dir', type=str, required=True, help='Parent directory to subfolders "train" and "test" where individual json graph files are located.')
    parser.add_argument('--splits', nargs='+', default=["train", "test"])
    parser.add_argument('--dst_dir', type=str, required=True, help='Directory to store merged graphs (RELATIVE to project root)')
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
    
    # Process the dataset files
    merge_graphs(args.src_dir, args.splits, args.dst_dir)
