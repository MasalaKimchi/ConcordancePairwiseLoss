import json
import os
from dataclasses import dataclass
from typing import Dict


@dataclass
class DatasetConfig:
    name: str
    auc_time: float
    auc_time_unit: str = "days"


def load_dataset_configs(config_path: str = None) -> Dict[str, DatasetConfig]:
    """Load dataset configurations from a JSON file."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "dataset_configs.json")
    with open(config_path, "r") as f:
        raw_configs = json.load(f)
    return {key: DatasetConfig(**cfg) for key, cfg in raw_configs.items()}
