"""
Logging configuration for the recommendation system.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

def setup_logging(
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Parameters:
    -----------
    log_file : str, optional
        Path to the log file. If None, logs to console only.
    log_level : int
        Logging level (default: INFO)
    max_bytes : int
        Maximum size of log file before rotation (default: 10MB)
    backup_count : int
        Number of backup files to keep (default: 5)
        
    Returns:
    --------
    logging.Logger
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('recommendation_system')
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create default logger instance
logger = setup_logging(
    log_file=os.path.join('logs', 'recommendation_system.log'),
    log_level=logging.INFO
) 