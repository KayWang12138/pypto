#!/usr/bin/env python3

import torch

from pypto_gpu_vk.compile import build_tensor_graph


def test_build_tensor_graph_add():
    graph = build_tensor_graph("add", torch.randn(8), torch.randn(8))
    assert graph.node_count() == 1
    assert graph.value_count() == 3
    assert graph.input_names() == ["input0", "input1"]
    assert graph.output_names() == ["output0"]
