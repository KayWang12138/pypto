"""partial_bad_op_golden.py — 纯 PyTorch golden"""
import torch


def partial_bad_op_golden(x):
    return torch.relu(x)
