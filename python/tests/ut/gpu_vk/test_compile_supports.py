#!/usr/bin/env python3

import torch

from pypto_gpu_vk.compile import compile_op


def test_compile_op_marks_only_real_elementwise_ops_as_supported():
    add_compiled = compile_op("add", torch.randn(8), torch.randn(8))
    matmul_compiled = compile_op("matmul", torch.randn(8, 8), torch.randn(8, 8))
    assert add_compiled.supports_real_vulkan is True
    assert matmul_compiled.supports_real_vulkan is False
