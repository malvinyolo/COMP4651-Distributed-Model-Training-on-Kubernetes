"""Artifact saving: checkpoints, metrics, configs, normalization stats."""
import os
import json
import yaml
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy arrays and other numeric types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def make_run_dir(save_root: str, run_name: Optional[str] = None) -> str:
    """
    Create a timestamped run directory.
    
    Args:
        save_root: Root directory for outputs (e.g., 'outputs')
        run_name: Optional custom run name (otherwise uses timestamp)
    
    Returns:
        Path to created run directory
    """
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
    
    run_dir = os.path.join(save_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_state_dict(model: nn.Module, path: str) -> None:
    """
    Save model state dict.
    
    Args:
        model: PyTorch model
        path: Path to save checkpoint
    """
    torch.save(model.state_dict(), path)


def save_json(obj: Any, path: str) -> None:
    """
    Save object as JSON with support for numpy arrays.
    
    Args:
        obj: Object to serialize (supports numpy arrays)
        path: Path to save JSON file
    """
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, cls=NumpyEncoder)


def save_yaml(obj: Any, path: str) -> None:
    """
    Save object as YAML.
    
    Args:
        obj: Object to serialize
        path: Path to save YAML file
    """
    with open(path, 'w') as f:
        yaml.dump(obj, f, default_flow_style=False)


def save_all_artifacts(
    run_dir: str,
    config: Dict,
    norm_stats: Dict,
    val_metrics: Dict,
    test_metrics: Dict,
    timing_info: Dict
) -> None:
    """
    Save all artifacts for a training run.
    
    Args:
        run_dir: Directory to save artifacts
        config: Training configuration
        norm_stats: Normalization statistics
        val_metrics: Validation metrics
        test_metrics: Test metrics
        timing_info: Timing information
    """
    # Save config
    config_path = os.path.join(run_dir, 'config.yaml')
    save_yaml(config, config_path)
    
    # Save normalization stats
    norm_path = os.path.join(run_dir, 'norm_stats.json')
    save_json(norm_stats, norm_path)
    
    # Save validation metrics
    val_metrics_path = os.path.join(run_dir, 'metrics_valid.json')
    save_json(val_metrics, val_metrics_path)
    
    # Save test metrics
    test_metrics_path = os.path.join(run_dir, 'metrics_test.json')
    save_json(test_metrics, test_metrics_path)
    
    # Save timing info
    timing_path = os.path.join(run_dir, 'timing.json')
    save_json(timing_info, timing_path)
