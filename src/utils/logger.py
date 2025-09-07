"""
Logging utilities for the Swiggy delivery prediction project.
"""
import logging
import os
from pathlib import Path
from datetime import datetime

def setup_logger(name: str, log_file: str = None, level: str = "INFO"):
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name (str): Logger name
        log_file (str): Path to log file
        level (str): Logging level
    
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid adding multiple handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_model_performance(logger, model_name: str, metrics: dict):
    """
    Log model performance metrics.
    
    Args:
        logger: Logger instance
        model_name (str): Name of the model
        metrics (dict): Performance metrics
    """
    logger.info(f"Model Performance - {model_name}")
    logger.info("-" * 50)
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    logger.info("-" * 50)

def log_prediction_request(logger, features: dict, prediction: float):
    """
    Log prediction request and result.
    
    Args:
        logger: Logger instance
        features (dict): Input features
        prediction (float): Predicted delivery time
    """
    logger.info(f"Prediction Request - Time: {datetime.now()}")
    logger.info(f"Features: {features}")
    logger.info(f"Predicted Delivery Time: {prediction:.2f} minutes")

def log_data_info(logger, data_shape: tuple, missing_values: int = 0):
    """
    Log data information.
    
    Args:
        logger: Logger instance
        data_shape (tuple): Shape of the dataset
        missing_values (int): Number of missing values
    """
    logger.info(f"Dataset Info:")
    logger.info(f"Shape: {data_shape}")
    logger.info(f"Missing values: {missing_values}")