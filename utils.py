"""
common utility functions
"""

import torch
import numpy as np

def set_seed(seed: int) -> None:
    """set random seed"""
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_device() -> str:
    """get device to train"""
    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else 'cpu'
    print(f"{'#'*10} Using {device} {'#'*10}")
    return device
