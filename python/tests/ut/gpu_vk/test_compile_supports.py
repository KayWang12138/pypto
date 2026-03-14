#!/usr/bin/env python3

import pytest
import torch

from pypto_gpu_vk.compile import compile_op


def test_compile_op_marks_only_real_elementwise_ops_as_supported():
    add_compiled = compile_op("add", torch.randn(8), torch.randn(8))
    vector_dot_compiled = compile_op("matmul", torch.randn(8), torch.randn(8))
    matrix_vector_compiled = compile_op("matmul", torch.randn(5, 8), torch.randn(8))
    vector_matrix_compiled = compile_op("matmul", torch.randn(8), torch.randn(8, 5))
    matmul_compiled = compile_op("matmul", torch.randn(8, 8), torch.randn(8, 8))
    batched_matmul_compiled = compile_op("matmul", torch.randn(3, 8, 8), torch.randn(3, 8, 8))
    broadcast_lhs_matmul_compiled = compile_op("matmul", torch.randn(8, 8), torch.randn(3, 8, 8))
    broadcast_rhs_matmul_compiled = compile_op("matmul", torch.randn(3, 8, 8), torch.randn(8, 8))
    batched_vector_matrix_compiled = compile_op("matmul", torch.randn(8), torch.randn(3, 8, 5))
    batched_matrix_vector_compiled = compile_op("matmul", torch.randn(3, 5, 8), torch.randn(8))
    assert add_compiled.supports_real_vulkan is True
    assert vector_dot_compiled.supports_real_vulkan is True
    assert matrix_vector_compiled.supports_real_vulkan is True
    assert vector_matrix_compiled.supports_real_vulkan is True
    assert matmul_compiled.supports_real_vulkan is True
    assert batched_matmul_compiled.supports_real_vulkan is True
    assert broadcast_lhs_matmul_compiled.supports_real_vulkan is True
    assert broadcast_rhs_matmul_compiled.supports_real_vulkan is True
    assert batched_vector_matrix_compiled.supports_real_vulkan is True
    assert batched_matrix_vector_compiled.supports_real_vulkan is True


def test_compile_op_rejects_invalid_matmul_shapes():
    with pytest.raises(ValueError, match="rank-1, rank-2 or rank-3"):
        compile_op("matmul", torch.randn(2, 3, 4, 5), torch.randn(5))

    with pytest.raises(ValueError, match="lhs.shape\\[1\\] == rhs.shape\\[0\\]"):
        compile_op("matmul", torch.randn(4, 8), torch.randn(7, 5))

    with pytest.raises(ValueError, match="contracted dimensions"):
        compile_op("matmul", torch.randn(8), torch.randn(7))

    with pytest.raises(ValueError, match="compatible batch dimensions"):
        compile_op("matmul", torch.randn(2, 4, 8), torch.randn(3, 8, 5))
