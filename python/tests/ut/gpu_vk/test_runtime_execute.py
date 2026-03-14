#!/usr/bin/env python3

import pytest
import torch

from pypto import pypto_impl
from pypto_gpu_vk.compile import compile_op
from pypto_gpu_vk.runtime import VkRuntime


def _require_runtime():
    runtime = VkRuntime(enable_cache=True)
    if not runtime.available():
        pytest.skip(f"Real Vulkan runtime unavailable: {runtime.status}")
    return runtime


def test_runtime_execute_add_matches_torch():
    runtime = _require_runtime()
    input0 = torch.randn(257)
    input1 = torch.randn(257)
    compiled = compile_op("add", input0, input1)
    output = runtime.execute(compiled, input0, input1)
    assert runtime.last_execute_status == pypto_impl.GpuVkStatus.SUCCESS
    assert torch.allclose(output, input0 + input1)


def test_runtime_execute_mul_matches_torch():
    runtime = _require_runtime()
    input0 = torch.randn(129)
    input1 = torch.randn(129)
    compiled = compile_op("mul", input0, input1)
    output = runtime.execute(compiled, input0, input1)
    assert runtime.last_execute_status == pypto_impl.GpuVkStatus.SUCCESS
    assert torch.allclose(output, input0 * input1)


def test_runtime_execute_relu_matches_torch():
    runtime = _require_runtime()
    input0 = torch.randn(145)
    compiled = compile_op("relu", input0)
    output = runtime.execute(compiled, input0)
    assert runtime.last_execute_status == pypto_impl.GpuVkStatus.SUCCESS
    assert torch.allclose(output, torch.relu(input0))


def test_runtime_execute_where_matches_torch():
    runtime = _require_runtime()
    cond = torch.randn(193)
    input0 = torch.randn(193)
    input1 = torch.randn(193)
    compiled = compile_op("where", cond, input0, input1)
    output = runtime.execute(compiled, cond, input0, input1)
    assert runtime.last_execute_status == pypto_impl.GpuVkStatus.SUCCESS
    assert torch.allclose(output, torch.where(cond.bool(), input0, input1))


def test_runtime_execute_matmul_matches_torch():
    runtime = _require_runtime()
    lhs = torch.randn(17, 19)
    rhs = torch.randn(19, 11)
    compiled = compile_op("matmul", lhs, rhs)
    output = runtime.execute(compiled, lhs, rhs)
    assert runtime.last_execute_status == pypto_impl.GpuVkStatus.SUCCESS
    assert output.shape == (17, 11)
    assert torch.allclose(output, torch.matmul(lhs, rhs), atol=1e-5, rtol=1e-5)


def test_runtime_execute_vector_dot_matches_torch():
    runtime = _require_runtime()
    lhs = torch.randn(19)
    rhs = torch.randn(19)
    compiled = compile_op("matmul", lhs, rhs)
    output = runtime.execute(compiled, lhs, rhs)
    assert runtime.last_execute_status == pypto_impl.GpuVkStatus.SUCCESS
    assert output.shape == ()
    assert torch.allclose(output, torch.matmul(lhs, rhs), atol=1e-5, rtol=1e-5)


def test_runtime_execute_matrix_vector_matches_torch():
    runtime = _require_runtime()
    lhs = torch.randn(17, 19)
    rhs = torch.randn(19)
    compiled = compile_op("matmul", lhs, rhs)
    output = runtime.execute(compiled, lhs, rhs)
    assert runtime.last_execute_status == pypto_impl.GpuVkStatus.SUCCESS
    assert output.shape == (17,)
    assert torch.allclose(output, torch.matmul(lhs, rhs), atol=1e-5, rtol=1e-5)


def test_runtime_execute_vector_matrix_matches_torch():
    runtime = _require_runtime()
    lhs = torch.randn(19)
    rhs = torch.randn(19, 11)
    compiled = compile_op("matmul", lhs, rhs)
    output = runtime.execute(compiled, lhs, rhs)
    assert runtime.last_execute_status == pypto_impl.GpuVkStatus.SUCCESS
    assert output.shape == (11,)
    assert torch.allclose(output, torch.matmul(lhs, rhs), atol=1e-5, rtol=1e-5)


def test_runtime_execute_batched_matmul_matches_torch():
    runtime = _require_runtime()
    lhs = torch.randn(3, 17, 19)
    rhs = torch.randn(3, 19, 11)
    compiled = compile_op("matmul", lhs, rhs)
    output = runtime.execute(compiled, lhs, rhs)
    assert runtime.last_execute_status == pypto_impl.GpuVkStatus.SUCCESS
    assert output.shape == (3, 17, 11)
    assert torch.allclose(output, torch.matmul(lhs, rhs), atol=1e-5, rtol=1e-5)


def test_runtime_execute_batched_vector_matrix_matches_torch():
    runtime = _require_runtime()
    lhs = torch.randn(19)
    rhs = torch.randn(3, 19, 11)
    compiled = compile_op("matmul", lhs, rhs)
    output = runtime.execute(compiled, lhs, rhs)
    assert runtime.last_execute_status == pypto_impl.GpuVkStatus.SUCCESS
    assert output.shape == (3, 11)
    assert torch.allclose(output, torch.matmul(lhs, rhs), atol=1e-5, rtol=1e-5)


def test_runtime_execute_batched_matrix_vector_matches_torch():
    runtime = _require_runtime()
    lhs = torch.randn(3, 17, 19)
    rhs = torch.randn(19)
    compiled = compile_op("matmul", lhs, rhs)
    output = runtime.execute(compiled, lhs, rhs)
    assert runtime.last_execute_status == pypto_impl.GpuVkStatus.SUCCESS
    assert output.shape == (3, 17)
    assert torch.allclose(output, torch.matmul(lhs, rhs), atol=1e-5, rtol=1e-5)


def test_runtime_execute_broadcast_lhs_matmul_matches_torch():
    runtime = _require_runtime()
    lhs = torch.randn(17, 19)
    rhs = torch.randn(3, 19, 11)
    compiled = compile_op("matmul", lhs, rhs)
    output = runtime.execute(compiled, lhs, rhs)
    assert runtime.last_execute_status == pypto_impl.GpuVkStatus.SUCCESS
    assert output.shape == (3, 17, 11)
    assert torch.allclose(output, torch.matmul(lhs, rhs), atol=1e-5, rtol=1e-5)


def test_runtime_execute_broadcast_rhs_matmul_matches_torch():
    runtime = _require_runtime()
    lhs = torch.randn(3, 17, 19)
    rhs = torch.randn(19, 11)
    compiled = compile_op("matmul", lhs, rhs)
    output = runtime.execute(compiled, lhs, rhs)
    assert runtime.last_execute_status == pypto_impl.GpuVkStatus.SUCCESS
    assert output.shape == (3, 17, 11)
    assert torch.allclose(output, torch.matmul(lhs, rhs), atol=1e-5, rtol=1e-5)
