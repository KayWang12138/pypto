#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Test and Example Usage of FFN Module

This script demonstrates how to use the FFN module with different configurations
and validates the implementation against PyTorch reference.
"""

import os
import sys
import argparse
import math
from dataclasses import dataclass
from typing import Literal
import pypto
import pytest
import torch
import numpy as np
from numpy.testing import assert_allclose




# Constants
F_1 = 1.0
F_NEGA_1 = -1.0
GELU_COEFF = 1.702


@dataclass
class FFNConfig:
    """Configuration for FFN module"""
    batch_size: int
    hidden_size: int
    intermediate_size: int
    activation: Literal["gelu", "swiglu", "relu"] = "swiglu"
    dtype: pypto.DataType = pypto.DT_FP16
    use_dynamic_shape: bool = False
    vec_tile_shape: tuple = (4, 128)
    cube_tile_shape: tuple = (4, 128, 128)
    basic_batch: int = 32  # For dynamic batching
    run_mode: pypto.RunMode = pypto.RunMode.NPU


def get_device_id():
    """
    Get and validate TILE_FWK_DEVICE_ID from environment variable.

    Returns:
        int: The device ID if valid, None otherwise.
    """
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("If no NPU environment is available, set --run_mode sim to run in simulation mode;")
        print("otherwise, set the environment variable TILE_FWK_DEVICE_ID.")
        print("Please set it before running this example:")
        print("  export TILE_FWK_DEVICE_ID=0")
        return None

    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


def gelu_torch(x):
    """PyTorch reference for GELU"""
    return x * torch.sigmoid(1.702 * x)


def swiglu_torch(gate, up):
    """PyTorch reference for SwiGLU."""
    swish = gate * torch.sigmoid(gate)
    return swish * up   


def ceil_div(a, b):
    """Calculate ceiling division: (a + b - 1) // b"""
    return (a + b - 1) // b


def relu_activation_core(x: pypto.tensor) -> pypto.tensor:
    """
    ReLU activation function: max(0, x)

    Parameters
    ----------
    x : pypto.tensor
        Input tensor

    Returns
    -------
    pypto.tensor
        ReLU activated tensor
    """
    pypto.set_vec_tile_shapes(*x.shape[:2] if len(x.shape) >= 2 else (32, 128))
    zero = pypto.full(x.shape, 0, x.dtype, valid_shape=x.shape)
    return pypto.maximum(x, zero)


def swiglu_activation_core(gate: pypto.tensor, up: pypto.tensor) -> pypto.tensor:
    """
    SwiGLU activation function: Swish(gate) * up
    where Swish(x) = x * sigmoid(x)

    Parameters
    ----------
    gate : pypto.tensor
        Gate tensor
    up : pypto.tensor
        Up projection tensor

    Returns
    -------
    pypto.tensor
        SwiGLU activated tensor
    """
    pypto.set_vec_tile_shapes(*gate.shape[:2] if len(gate.shape) >= 2 else (32, 128))

    gate_neg = pypto.mul(gate, F_NEGA_1)
    exp_neg = pypto.exp(gate_neg)
    # Create FP32 constant tensor and cast to target dtype to avoid pypto.full() bfloat16 limitation
    # sigmoid = 1 / (1 + exp(-gate))
    ones_fp32 = pypto.full(exp_neg.shape, F_1, pypto.DT_FP32, valid_shape=exp_neg.shape)
    ones = pypto.cast(ones_fp32, exp_neg.dtype)
    sigmoid = pypto.div(ones, pypto.add(exp_neg, F_1))
    swish = pypto.mul(gate, sigmoid)

    # Multiply with up projection
    return pypto.mul(swish, up)

def ffn(config: FFNConfig) -> torch.Tensor:

    batch_size, hidden_size, intermediate_size = config.batch_size, config.hidden_size, config.intermediate_size
    
    @pypto.frontend.jit(runtime_options={"run_mode": config.run_mode})
    def ffn_activation_kernel(
        hidden_states: pypto.tensor((batch_size, hidden_size), config.dtype),
        gate_proj_weight: pypto.tensor((hidden_size, intermediate_size), config.dtype),
        up_proj_weight: pypto.tensor((hidden_size, intermediate_size), config.dtype),
        down_proj_weight: pypto.tensor((intermediate_size, hidden_size), config.dtype),
    ) -> pypto.tensor((batch_size, hidden_size), config.dtype):
        # Configure tiling for matrix operations
        pypto.set_cube_tile_shapes(
            [config.cube_tile_shape[0], config.cube_tile_shape[0]],
            [config.cube_tile_shape[1], config.cube_tile_shape[1]],
            [config.cube_tile_shape[2], config.cube_tile_shape[2]]
        )
        pypto.set_vec_tile_shapes(*config.vec_tile_shape)

        # Gate projection: [batch_size, hidden_size] @ [hidden_size, intermediate_size]
        gate = pypto.matmul(hidden_states, gate_proj_weight, config.dtype)
        
        
        # SwiGLU activation
        up_proj_weight = pypto.matmul(hidden_states, up_proj_weight, config.dtype)
        activated = swiglu_activation_core(gate, up_proj_weight)

        output = pypto.matmul(activated, down_proj_weight, config.dtype, b_trans=False)
        return output
    
    return ffn_activation_kernel


def test_ffn_static_swiglu(device_id=0, run_mode: str = "npu"):
    """Test static FFN with SwiGLU activation."""
    print("=" * 60)
    print("Testing Static FFN with SwiGLU Activation")
    print("=" * 60)

    batch_size = 16
    hidden_size = 126
    intermediate_size = 256
    dtype = torch.bfloat16
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    config = FFNConfig(
        batch_size=batch_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="swiglu",
        dtype=pypto.DT_BF16,
        use_dynamic_shape=False,
        vec_tile_shape=(16, 32),
        cube_tile_shape=(16, 32, 32),
        run_mode=pypto.RunMode.NPU if run_mode == "npu" else pypto.RunMode.SIM
    )
    # Create PyTorch tensors
    hidden_states_torch = torch.randn(batch_size, hidden_size, 
                                        dtype=dtype, device=device) / math.sqrt(batch_size)
    gate_proj_weight_torch = torch.randn(hidden_size, intermediate_size, 
                                        dtype=dtype, device=device) / math.sqrt(batch_size)
    up_proj_weight_torch = torch.randn(hidden_size, intermediate_size, 
                                        dtype=dtype, device=device) / math.sqrt(batch_size)
    down_proj_weight_torch = torch.randn(intermediate_size, hidden_size, 
                                        dtype=dtype, device=device) / math.sqrt(batch_size)
    
    # PyTorch reference computation
    gate_torch = torch.matmul(hidden_states_torch, gate_proj_weight_torch)
    up_torch = torch.matmul(hidden_states_torch, up_proj_weight_torch)
    activated_torch = swiglu_torch(gate_torch.float(), up_torch.float()).to(dtype)
    output_torch_ref = torch.matmul(activated_torch, down_proj_weight_torch)

    output = ffn(config)(hidden_states_torch, gate_proj_weight_torch, up_proj_weight_torch, down_proj_weight_torch)
    print(f"Input shape: {hidden_states_torch.shape}")
    print(f"Gate weight shape: {gate_proj_weight_torch.shape}")
    print(f"Up weight shape: {up_proj_weight_torch.shape}")
    print(f"Down weight shape: {down_proj_weight_torch.shape}")
    print(f"Output shape: {output_torch_ref.shape}")
    print(f"Output range: [{output_torch_ref.min().item():.4f}, {output_torch_ref.max().item():.4f}]")

    if run_mode == "npu":
        assert_allclose(output.cpu().to(torch.float32), output_torch_ref.cpu().to(torch.float32), rtol=3e-3, atol=3e-3)
    print("✅ Static FFN with SwiGLU test completed")
    # print()



def main():
    test_ffn_static_swiglu(0)

if __name__ == "__main__":
    main()

































# def dynamic_gelu_activation_core(output: pypto.tensor, hidden_states: pypto.tensor, 
#     gate_proj_weight: pypto.tensor, down_proj_weight: pypto.tensor, config: FFNConfig) -> None:
#     hidden_size, intermediate_size = config.hidden_size, config.intermediate_size
#     basic_batch = config.basic_batch
#     if basic_batch == 0:
#         raise ValueError("basic_batch must be greater than 0")
#     # Calculate number of iterations needed
#     batch_size = hidden_states.shape[0]
#     num_iterations = ceil_div(batch_size, basic_batch)
#     # Process in chunks
#     for idx in pypto.loop(0, num_iterations, 1, name="LOOP_FFN_BATCH", idx_name="idx"):
#         batch_offset = idx * basic_batch
#         # View current batch chunk
#         hidden_chunk = pypto.view(
#             hidden_states,
#             [basic_batch, hidden_size],
#             [batch_offset, 0],
#             valid_shape=[(batch_size - batch_offset).min(basic_batch), hidden_size]
#         )
#         # Configure tiling for matrix operations
#         pypto.set_matrix_size([basic_batch, hidden_size, intermediate_size])
#         # Gate projection
#         gate = pypto.matmul(hidden_chunk, gate_proj_weight, config.dtype)
#         pypto.set_vec_tile_shapes(*config.vec_tile_shape)
#         activated = gelu_activation_core(gate)
#         # Down projection
#         pypto.set_cube_tile_shapes(
#             [config.cube_tile_shape[0], config.cube_tile_shape[0]],
#             [config.cube_tile_shape[1], config.cube_tile_shape[1]],
#             [config.cube_tile_shape[2], config.cube_tile_shape[2]]
#         )
#         pypto.set_matrix_size([basic_batch, intermediate_size, hidden_size])
#         output_chunk = pypto.matmul(activated, down_proj_weight, config.dtype, b_trans=False)
#         # Assemble result back to output
#         pypto.assemble(output_chunk, [batch_offset, 0], output)
#     return



# def dynamic_gelu_activation_core(output: pypto.tensor, hidden_states: pypto.tensor, 
#     gate_proj_weight: pypto.tensor, down_proj_weight: pypto.tensor, config: FFNConfig) -> None:
#     hidden_size, intermediate_size = config.hidden_size, config.intermediate_size
#     basic_batch = config.basic_batch
#     if basic_batch == 0:
#         raise ValueError("basic_batch must be greater than 0")
#     # Calculate number of iterations needed
#     batch_size = hidden_states.shape[0]
#     num_iterations = ceil_div(batch_size, basic_batch)
#     # Process in chunks
#     for idx in pypto.loop(0, num_iterations, 1, name="LOOP_FFN_BATCH", idx_name="idx"):
#         batch_offset = idx * basic_batch
#         # View current batch chunk
#         hidden_chunk = pypto.view(
#             hidden_states,
#             [basic_batch, hidden_size],
#             [batch_offset, 0],
#             valid_shape=[(batch_size - batch_offset).min(basic_batch), hidden_size]
#         )
#         # Configure tiling for matrix operations
#         pypto.set_matrix_size([basic_batch, hidden_size, intermediate_size])
#         # Gate projection
#         gate = pypto.matmul(hidden_chunk, gate_proj_weight, config.dtype)
#         pypto.set_vec_tile_shapes(*config.vec_tile_shape)
#         activated = gelu_activation_core(gate)
#         # Down projection
#         pypto.set_cube_tile_shapes(
#             [config.cube_tile_shape[0], config.cube_tile_shape[0]],
#             [config.cube_tile_shape[1], config.cube_tile_shape[1]],
#             [config.cube_tile_shape[2], config.cube_tile_shape[2]]
#         )
#         pypto.set_matrix_size([basic_batch, intermediate_size, hidden_size])
#         output_chunk = pypto.matmul(activated, down_proj_weight, config.dtype, b_trans=False)
#         # Assemble result back to output
#         pypto.assemble(output_chunk, [batch_offset, 0], output)
#     return