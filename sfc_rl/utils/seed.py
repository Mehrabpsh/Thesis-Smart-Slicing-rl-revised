"""Seed management for reproducibility."""

import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Get a numpy random number generator.
    
    Args:
        seed: Optional seed for the generator
        
    Returns:
        NumPy random generator
    """
    return np.random.default_rng(seed)

