"""KernelBench-style task desc for ReLUCheat (复用 19_ReLU 的输入 shape)."""
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x):
        return torch.relu(x)


def get_inputs():
    return [torch.randn(16, 16384)]


def get_init_inputs():
    return []
