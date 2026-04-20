"""no_jit fixture: import pypto 但没有 @pypto.jit, cheat_detector 应判 cheat (has_jit)."""
import torch
import pypto


def relu_wrapper(x):
    return torch.relu(x)
