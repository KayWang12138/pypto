#!/usr/bin/env python3

import torch

from pypto_gpu_vk.compile import build_tensor_graph, lower_to_dispatch


def test_lower_to_dispatch_uses_default_launch():
    dispatch_graph = lower_to_dispatch(build_tensor_graph("relu", torch.randn(128)))
    dispatch = dispatch_graph.dispatch()
    assert dispatch.local_x == 64
    assert dispatch.group_x >= 1
    assert dispatch_graph.kernel_name() == "relu_graph"
