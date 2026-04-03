"""good_op_golden.py — 纯 PyTorch golden fixture（无 import pypto）"""
import torch
import numpy as np


def good_op_golden(x):
    """Golden reference implementation using pure PyTorch"""
    return torch.sin(x)
