"""
Utils: seeding, device selection, timing, logging
"""
import random
import numpy as np
import torch
from datetime import datetime
import time


def seed_everything(seed: int) -> None:
    """Set seeds for python, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(pref: str) -> torch.device:
    """
    Get torch device based on preference.
    
    Args:
        pref: "cpu", "cuda", or "auto" (uses cuda if available)
    
    Returns:
        torch.device
    """
    if pref == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


class Timer:
    """Simple context manager for timing blocks."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.name:
            log(f"{self.name} took {self.elapsed:.2f}s")
    
    def __str__(self):
        return f"{self.elapsed:.2f}s" if self.elapsed else "N/A"


def log(msg: str) -> None:
    """Simple print with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")
