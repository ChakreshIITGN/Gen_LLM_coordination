"""Utilities for random seed management."""

import numpy as np
import random
from typing import Optional


def set_seed(seed: Optional[int] = None) -> np.random.Generator:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed (None for random)
        
    Returns:
        NumPy random generator
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    return np.random.default_rng(seed)
