import logging
import os
import yaml

def setup_logging(log_dir="logs", log_name="run.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config