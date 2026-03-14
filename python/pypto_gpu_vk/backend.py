from __future__ import annotations

from dataclasses import replace

import torch

import pypto  # noqa: F401
from pypto import pypto_impl

from .config import CompileConfig


def _normalize_inputs(inputs):
    normalized = []
    for tensor in inputs:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, but got {type(tensor).__name__}.")
        normalized.append(tensor.contiguous() if not tensor.is_contiguous() else tensor)
    return normalized


class GpuVkBackend:
    def __init__(self, config: CompileConfig | None = None):
        self.config = replace(config) if config is not None else CompileConfig()

    def build_tensor_graph(self, op_name: str, *inputs: torch.Tensor):
        tensors = _normalize_inputs(inputs)
        shapes = [list(tensor.shape) for tensor in tensors]
        return pypto_impl.GpuVkBuildTensorGraph(op_name, shapes, 0)

    def lower_to_dispatch(self, graph):
        return pypto_impl.GpuVkLowerToDispatch(graph)

    def lower_to_shader(self, graph):
        return pypto_impl.GpuVkLowerToShader(graph)

    def compile_artifact(self, graph):
        return pypto_impl.GpuVkCompileArtifact(graph, self.config.optimize, self.config.debug_info)
