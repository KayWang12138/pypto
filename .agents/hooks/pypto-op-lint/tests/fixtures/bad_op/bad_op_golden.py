"""bad_op_golden.py — 违反 OL15 的 golden fixture（导入了 pypto）"""
import pypto  # OL15 FAIL
import torch


def bad_op_golden(x):
    return torch.sin(x)
