#!/usr/bin/env python3
"""
merge_rl_results.py - Merge individual RL model results into the main test_results.json file

This script takes all individual result JSON files (e.g., gnn_GIN_512_p3_results.json)
and merges them into the original test_results.json file, adding each solver's results
as a new key in each graph's dictionary.
"""

import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
import shutil

import src.utils.config as config_utils
from src.utils.logger import setup_logger


def merge_results(original_json_path, results_dir, output_path):
    """
    Merge all individual RL model result files into the main graphs JSON.
    Only includes graphs that appear in the result files.
    """
    PROJECT_ROOT = config_utils.get_project_root()
    logger = logging.getLogger(__name__)
    
    # Convert to absolute paths
    original_json_path = Path(original_json_path)
    logger.info(f"Original JSON path: {original_json_path}")
    if not original_json_path.is_absolute():
        original_json_path = PROJECT_ROOT / original_json_path
    
    results_dir = Path(results_dir)
    logger.info(f"Results directory: {results_dir}")
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir
    
    output_path = Path(output_path)
    logger.info(f"Output path: {output_path}")
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    
    # Load original graphs
    logger.info(f"Loading original graphs from: {original_json_path}")
    with open(original_json_path, 'r') as f:
        graphs = json.load(f)
    logger.info(f"Loaded {len(graphs)} graphs")

    # Create mapping for faster lookup
    graph_id_to_idx = {g["id"]: i for i, g in enumerate(graphs)}
    
    # Find all result files
    result_files = list(results_dir.glob("*_results.json"))
    logger.info(f"Found {len(result_files)} result files")

    if not result_files:
        logger.warning("No result files found!")
        return
    
    # Track which graphs have results
    graph_ids_with_results = set()
    
    # Process each result file
    for result_file in sorted(result_files):
        solver_name = result_file.stem.replace("_results", "")
        logger.info(f"Processing: {solver_name}")
        
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        # Track graph IDs and merge results
        for graph_id, result_data in results.items():
            graph_ids_with_results.add(graph_id)
            if graph_id in graph_id_to_idx:
                idx = graph_id_to_idx[graph_id]
                graphs[idx][solver_name] = result_data
    
    # Filter to only include graphs with results
    filtered_graphs = [g for g in graphs if g['id'] in graph_ids_with_results]

    logger.info(f"Filtered from {len(graphs)} to {len(filtered_graphs)} graphs")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(filtered_graphs, f, indent=2)
    
    logger.info(f"Saved to: {output_path}")
    logger.info("Done!")


def get_parser():
    parser = argparse.ArgumentParser(
        description='Merge individual RL model results into the main test_results.json file'
    )
    
    parser.add_argument('--original-json', type=str, default='data/optimized_graphs_classic/test_results.json',
                        help='Path to the original test_results.json file')
    
    parser.add_argument('--results-dir', type=str, default='data/indiv_rlmodel_results', 
                        help='Directory containing individual result files (*_results.json)')
    
    parser.add_argument('--output', type=str, default='data/test_results_with_rl_merged.json',
                        help='Output path for the merged JSON file')
    
    # Logging parameters
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    parser.add_argument("--log-format", type=str, default="", help="Format for log messages")
    
    return parser


if __name__ == "__main__":
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # Setup logging
    timestamp = datetime.now().strftime("%m%d_%H%M")
    log_filename = f"merge_results_{timestamp}.log"
    log_path = Path(config_utils.get_project_root()) / f"outputs/logs/merge/{log_filename}"
    
    # Create log directory if it doesn't exist
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Log level from arguments
    log_level = getattr(logging, args.log_level, logging.INFO)
    
    # Setup logging with arguments
    setup_logger(
        log_file_path=str(log_path), 
        log_format=args.log_format,
        console_level=log_level, 
        file_level=log_level,
        console_logging=True  # Enable console output for merge script
    )
    
    logger = logging.getLogger(__name__)
    
    logger.info("Starting merge process...")
    logger.info(f"Original JSON: {args.original_json}")
    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Output path: {args.output}")
    
    try:
        merge_results(
            original_json_path=args.original_json,
            results_dir=args.results_dir,
            output_path=args.output
        )
    except Exception as e:
        logger.error(f"Error during merge: {e}", exc_info=True)
        raise