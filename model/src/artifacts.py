"""
Artifacts: save/load checkpoints, metrics, configs, confusion matrices
"""
import os
import json
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

# Use non-interactive backend for saving plots
matplotlib.use('Agg')


def make_run_dir(save_root: str, run_name: str | None) -> str:
    """
    Create run directory with timestamp if run_name is None.
    
    Args:
        save_root: Root directory for outputs
        run_name: Optional custom run name
    
    Returns:
        Path to created run directory
    """
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
    
    run_dir = os.path.join(save_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_state_dict(model: torch.nn.Module, path: str) -> None:
    """
    Save model state_dict.
    
    Args:
        model: PyTorch model
        path: Save path
    """
    torch.save(model.state_dict(), path)


def save_json(obj: dict, path: str) -> None:
    """
    Save dictionary as JSON.
    
    Args:
        obj: Dictionary to save
        path: Save path
    """
    # Convert numpy types to native Python types
    def convert(o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [convert(item) for item in o]
        return o
    
    obj_converted = convert(obj)
    with open(path, 'w') as f:
        json.dump(obj_converted, f, indent=2)


def save_yaml(obj: dict, path: str) -> None:
    """
    Save dictionary as YAML.
    
    Args:
        obj: Dictionary to save
        path: Save path
    """
    with open(path, 'w') as f:
        yaml.dump(obj, f, default_flow_style=False, sort_keys=False)


def save_confusion(cm: np.ndarray, path: str, labels=("0", "1")) -> None:
    """
    Save confusion matrix as PNG heatmap.
    
    Args:
        cm: 2x2 confusion matrix
        path: Save path
        labels: Class labels
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels,
           yticklabels=labels,
           ylabel='True label',
           xlabel='Predicted label',
           title='Confusion Matrix')
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
