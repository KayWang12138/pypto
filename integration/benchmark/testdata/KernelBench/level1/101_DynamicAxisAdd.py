#!/usr/bin/env python3
# coding: utf-8
"""KernelBench-style level1 case with SPEC metadata globals."""

import torch
import torch.nn as nn


FORMULA = "out[b, s, d] = x[b, s, d] + bias[d]"
DYNAMIC_AXIS = ["B", "S"]


class Model(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return x + bias


def get_inputs():
    batch = 2
    seq_len = 4
    hidden_size = 8
    x = torch.randn(batch, seq_len, hidden_size, dtype=torch.float32)
    bias = torch.randn(hidden_size, dtype=torch.float32)
    return [x, bias]


def get_init_inputs():
    return [8]
