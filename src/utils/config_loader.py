# src/utils/config_loader.py
"""
Handles loading and validation of the project's configuration file.
"""
import yaml
import sys
from pathlib import Path
from typing import Dict, Any


def _ensure_directories(config: Dict[str, Any]):
    """
    Ensure all required directories from the config exist.
    """
    paths_to_check = [
        config['governance']['provenance']['log_path'],
        config['output']['results_dir'],
        config['output'].get('plots_dir'),
        config['logging']['log_file'],
        config['knowledge']['process_kg_path'],
        config['knowledge']['rule_pool_path'],
    ]

    for path_str in paths_to_check:
        if not path_str:
            continue

        path = Path(path_str).parent

        # If it's a file, remove it to create a directory
        if path.exists() and path.is_file():
            print(f"⚠️  Removing file '{path}' to create directory.")
            path.unlink()

        path.mkdir(parents=True, exist_ok=True)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Loads configuration from a YAML file and ensures necessary directories exist.

    Args:
        config_path: Path to the configuration YAML file.

    Returns:
        The loaded configuration as a dictionary.
    """
    config_file = Path(config_path)

    if not config_file.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error parsing config file: {e}")
        sys.exit(1)

    # Ensure critical directories exist before proceeding
    _ensure_directories(config)

    return config