import yaml
import os

def load_config(config_path="configs/config.yaml"):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
    
    Returns:
        dict: Configuration parameters.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")

def get_data_paths(config):
    """Extract data paths from config."""
    return config['data']['raw'], config['data']['processed']

def get_model_params(config, algorithm):
    """Extract model parameters for the specified algorithm."""
    return config['model'][algorithm]

def get_optimization_params(config):
    """Extract optimization parameters."""
    return config['optimization']

def get_logging_config(config):
    """Extract logging configuration."""
    return config['logging']