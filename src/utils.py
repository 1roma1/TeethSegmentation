from typing import Dict

import yaml
import mlflow
import numpy as np


def load_configuration(config_file: str) -> Dict:
    """Load configuration from yaml file"""

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
