\
import logging
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from logging.handlers import RotatingFileHandler
from src.utils.config import load_config

def setup_logging():
    """
    Set up logging configuration based on config.yaml.
    Logs are written to both file and console with rotation.
    """
    # Load configuration
    config = load_config()
    log_config = config['logging']
    log_level = getattr(logging, log_config['level'].upper(), logging.INFO)
    log_file = log_config['log_file']

    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure logger
    logger = logging.getLogger('TractionMotorDiagnosis')
    logger.setLevel(log_level)

    # Avoid duplicate handlers
    if not logger.handlers:
        # File handler with rotation (max 5MB, keep 5 backups)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5*1024*1024, backupCount=5
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger

if __name__ == "__main__":
    # Test logging setup
    logger = setup_logging()
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")