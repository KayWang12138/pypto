import torch
import torch.nn as nn

from ReLUCheat_impl import ReLUCheat_wrapper


class ModelNew(nn.Module):
    def __init__(self, *init_args, **init_kwargs):
        super().__init__()

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return ReLUCheat_wrapper(*inputs)
