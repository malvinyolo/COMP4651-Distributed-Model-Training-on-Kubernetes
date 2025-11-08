"""Utilities: seeding, device selection, timing, logging."""
import random
import numpy as np
import torch
import time
from datetime import datetime
from typing import Optional


def seed_everything(seed: int) -> None:
    """Set seed for reproducibility across all random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(pref: str = "auto") -> str:
    """
    Get device string based on preference.
    
    Args:
        pref: 'auto', 'cpu', 'cuda', or 'mps'
    
    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if pref == "cpu":
        return "cpu"
    elif pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif pref == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    else:  # auto
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.elapsed = 0.0
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start


def log(msg: str) -> None:
    """Print timestamped message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")
