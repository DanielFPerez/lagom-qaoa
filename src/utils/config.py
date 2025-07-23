import os

def check_save_folder(save_folder: str) -> bool:
    """
    Check if the save folder exists.
    
    Parameters:
    - save_folder (str): Path to the save folder.
    
    Returns:
    - bool: True if the folder exists, False otherwise.
    """
    import logging
    logger = logging.getLogger(__name__)
    if not os.path.exists(save_folder):
        logger.error(f"Save folder does not exist: {save_folder}")
        raise FileNotFoundError(f"Save folder does not exist: {save_folder}")
    else:
        return True
    

def get_project_root(target_name="lagom-qaoa") -> str:
    """
    Get the project root directory.
    
    Parameters:
    - target_name (str): Name of the target project directory.
    
    Returns:
    - str: Path to the project root directory.
    """
    ctr = 10
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(current_dir) != target_name and ctr > 0:
        current_dir = os.path.dirname(current_dir)
        ctr -= 1
    if ctr <= 0:
        raise FileNotFoundError(f"Could not find project root directory with name: {target_name}")
    return current_dir