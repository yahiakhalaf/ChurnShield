import logging
import os
import glob
from datetime import datetime

def setup_logger(module_name):
    """Set up a new global logger with a timestamped log file named ChurnShield_<timestamp>.log.
    
    Args:
        module_name (str): The name of the module using the logger (e.g., 'ModelTraining', 'ChurnPredictor').
    
    Returns:
        logging.Logger: The configured logger instance.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'ChurnShield_{timestamp}.log')
    
    logger = logging.getLogger(f'ChurnShield_{timestamp}')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(f'%(asctime)s - %(levelname)s - {module_name}: %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Store timestamp in logger for use by load_logger
    logger.timestamp = timestamp
    
    return logger

def load_logger(module_name):
    """Load the global logger associated with the most recent ChurnShield_<timestamp>.log file.
    
    Args:
        module_name (str): The name of the module using the logger (e.g., 'ModelTraining', 'ChurnPredictor').
    
    Returns:
        logging.Logger: The configured logger instance for the most recent log file.
    """
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Find the most recent log file
    log_files = glob.glob(os.path.join(log_dir, 'ChurnShield_*.log'))
    if not log_files:
        # If no log files exist, create a new logger
        return setup_logger(module_name)
    
    # Extract timestamps from filenames and find the latest
    latest_log_file = max(log_files, key=lambda f: os.path.basename(f)[11:-4])
    timestamp = os.path.basename(latest_log_file)[11:-4]
    
    logger = logging.getLogger(f'ChurnShield_{timestamp}')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(latest_log_file)
        formatter = logging.Formatter(f'%(asctime)s - %(levelname)s - {module_name}: %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Store timestamp in logger
    logger.timestamp = timestamp
    
    return logger