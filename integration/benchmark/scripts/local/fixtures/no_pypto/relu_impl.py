"""no_pypto fixture: 不 import pypto, 纯 torch 实现.

cheat_detector 应判 cheat (import_pypto + has_jit 双 fail).
"""
import torch
import torch.nn.functional as F


def relu_wrapper(x):
    return F.relu(x)
