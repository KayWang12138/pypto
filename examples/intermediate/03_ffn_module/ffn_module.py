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
Feed-Forward Network (FFN) Module using PyPTO

This module implements a flexible FFN with support for:
- Multiple activation functions (GELU, SwiGLU, ReLU)
- Static and dynamic batch sizes
- Configurable tiling strategies
- Optional quantization support
"""

import pypto
from typing import Optional, Literal
from dataclasses import dataclass


# Runtime configuration keys
KEY_ONLY_CODEGEN = "only_codegen"

# Constants
F_1 = 1.0
F_NEGA_1 = -1.0
GELU_COEFF = 0.044715


@dataclass
class FFNConfig:
    """Configuration for FFN module"""
    hidden_size: int
    intermediate_size: int
    activation: Literal["gelu", "swiglu", "relu"] = "gelu"
    dtype: pypto.DataType = pypto.DT_FP16
    use_dynamic_shape: bool = False
    vec_tile_shape: tuple = (64, 128)
    cube_tile_shape: tuple = (64, 128, 128)
    basic_batch: int = 32  # For dynamic batching


def ceil_div(a, b):
    """Calculate ceiling division: (a + b - 1) // b"""
    return (a + b - 1) // b


def gelu_activation(x: pypto.tensor) -> pypto.tensor:
    """
    GELU activation function: x * 0.5 * (1 + erf(x / sqrt(2)))
    
    Approximated as: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    
    Parameters
    ----------
    x : pypto.tensor
        Input tensor
        
    Returns
    -------
    pypto.tensor
        GELU activated tensor
    """
    pypto.set_vec_tile_shapes(*x.shape[:2] if len(x.shape) >= 2 else (32, 128))
    
    # GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    # Using simpler approximation: x * sigmoid(1.702 * x)
    coeff = pypto.element(x.dtype, 1.702)
    x_scaled = pypto.mul_s(x, coeff)
    
    # sigmoid(x) = 1 / (1 + exp(-x))
    x_neg = pypto.mul_s(x_scaled, pypto.element(x.dtype, F_NEGA_1))
    exp_neg = pypto.exp(x_neg)
    one = pypto.element(x.dtype, F_1)
    sigmoid = pypto.div(one, pypto.add_s(exp_neg, one))
    
    return pypto.mul(x, sigmoid)


def swiglu_activation(gate: pypto.tensor, up: pypto.tensor) -> pypto.tensor:
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
    
    # Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
    gate_neg = pypto.mul_s(gate, pypto.element(gate.dtype, F_NEGA_1))
    exp_neg = pypto.exp(gate_neg)
    one = pypto.element(gate.dtype, F_1)
    sigmoid = pypto.div(one, pypto.add_s(exp_neg, one))
    swish = pypto.mul(gate, sigmoid)
    
    # Multiply with up projection
    return pypto.mul(swish, up)


def relu_activation(x: pypto.tensor) -> pypto.tensor:
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
    zero = pypto.element(x.dtype, 0.0)
    return pypto.maximum(x, zero)


@pypto.jit
def ffn_static(
    hidden_states: pypto.tensor,
    gate_proj_weight: pypto.tensor,
    up_proj_weight: pypto.tensor,
    down_proj_weight: pypto.tensor,
    output: pypto.tensor,
    config: FFNConfig
):
    """
    Static FFN implementation with fixed batch size.
    
    Architecture:
        hidden_states -> [gate_proj] -> [activation] -> [down_proj] -> output
                        -> [up_proj]  (for SwiGLU)
    
    Parameters
    ----------
    hidden_states : pypto.tensor
        Input tensor of shape [batch_size, hidden_size]
    gate_proj_weight : pypto.tensor
        Gate projection weight of shape [hidden_size, intermediate_size]
    up_proj_weight : pypto.tensor
        Up projection weight of shape [hidden_size, intermediate_size] (for SwiGLU)
    down_proj_weight : pypto.tensor
        Down projection weight of shape [intermediate_size, hidden_size]
    output : pypto.tensor
        Output tensor of shape [batch_size, hidden_size]
    config : FFNConfig
        FFN configuration
    """
    pypto.set_host_option(KEY_ONLY_CODEGEN, True)
    
    input_tensors = [hidden_states, gate_proj_weight, up_proj_weight, down_proj_weight]
    output_tensors = [output]
    
    def inside_ffn():
        batch_size = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        intermediate_size = config.intermediate_size
        
        # Configure tiling for matrix operations
        pypto.set_cube_tile_shapes(
            [config.cube_tile_shape[0], config.cube_tile_shape[0]],
            [config.cube_tile_shape[1], config.cube_tile_shape[1]],
            [config.cube_tile_shape[2], config.cube_tile_shape[2]]
        )
        pypto.set_matrix_size({batch_size, hidden_size, intermediate_size})
        
        # Gate projection: [batch_size, hidden_size] @ [hidden_size, intermediate_size]
        gate = pypto.matmul(hidden_states, gate_proj_weight, config.dtype)
        
        if config.activation == "swiglu":
            # Up projection: [batch_size, hidden_size] @ [hidden_size, intermediate_size]
            up = pypto.matmul(hidden_states, up_proj_weight, config.dtype)
            
            # SwiGLU activation
            pypto.set_vec_tile_shapes(*config.vec_tile_shape)
            activated = swiglu_activation(gate, up)
        elif config.activation == "gelu":
            # GELU activation
            pypto.set_vec_tile_shapes(*config.vec_tile_shape)
            activated = gelu_activation(gate)
        elif config.activation == "relu":
            # ReLU activation
            pypto.set_vec_tile_shapes(*config.vec_tile_shape)
            activated = relu_activation(gate)
        else:
            raise ValueError(f"Unsupported activation: {config.activation}")
        
        # Down projection: [batch_size, intermediate_size] @ [intermediate_size, hidden_size]
        pypto.set_cube_tile_shapes(
            [config.cube_tile_shape[0], config.cube_tile_shape[0]],
            [config.cube_tile_shape[1], config.cube_tile_shape[1]],
            [config.cube_tile_shape[2], config.cube_tile_shape[2]]
        )
        pypto.set_matrix_size({batch_size, intermediate_size, hidden_size})
        output[:] = pypto.matmul(activated, down_proj_weight, config.dtype, b_trans=True)
    
    inside_ffn()


@pypto.jit
def ffn_dynamic(
    hidden_states: pypto.tensor,
    gate_proj_weight: pypto.tensor,
    up_proj_weight: pypto.tensor,
    down_proj_weight: pypto.tensor,
    output: pypto.tensor,
    config: FFNConfig
):
    """
    Dynamic FFN implementation with variable batch size.
    
    Processes input in chunks of `basic_batch` size to handle dynamic batch dimensions.
    
    Parameters
    ----------
    hidden_states : pypto.tensor
        Input tensor with dynamic first dimension [batch_size, hidden_size]
    gate_proj_weight : pypto.tensor
        Gate projection weight of shape [hidden_size, intermediate_size]
    up_proj_weight : pypto.tensor
        Up projection weight of shape [hidden_size, intermediate_size] (for SwiGLU)
    down_proj_weight : pypto.tensor
        Down projection weight of shape [intermediate_size, hidden_size]
    output : pypto.tensor
        Output tensor with dynamic first dimension [batch_size, hidden_size]
    config : FFNConfig
        FFN configuration
    """
    pypto.set_host_option(KEY_ONLY_CODEGEN, True)
    pypto.set_codegen_option("support_dynamic_unaligned", True)
    
    input_tensors = [hidden_states, gate_proj_weight, up_proj_weight, down_proj_weight]
    output_tensors = [output]
    
    def inside_ffn():
        hidden_size = hidden_states.shape[1]
        intermediate_size = config.intermediate_size
        basic_batch = config.basic_batch
        
        if basic_batch == 0:
            raise ValueError("basic_batch must be greater than 0")
        
        # Calculate number of iterations needed
        batch_size = pypto.get_input_shape(hidden_states, 0)
        num_iterations = ceil_div(batch_size, basic_batch)
        
        # Process in chunks
        for idx in pypto.loop(0, num_iterations, 1, name="LOOP_FFN_BATCH", idx_name="idx"):
            def process_batch_chunk(idx):
                batch_offset = idx * basic_batch
                
                # View current batch chunk
                hidden_chunk = pypto.view(
                    hidden_states,
                    [basic_batch, hidden_size],
                    [batch_offset, 0],
                    valid_shape=[(batch_size - batch_offset).min(basic_batch), hidden_size]
                )
                
                # Configure tiling for matrix operations
                pypto.set_cube_tile_shapes(
                    [config.cube_tile_shape[0], config.cube_tile_shape[0]],
                    [config.cube_tile_shape[1], config.cube_tile_shape[1]],
                    [config.cube_tile_shape[2], config.cube_tile_shape[2]]
                )
                pypto.set_matrix_size({basic_batch, hidden_size, intermediate_size})
                
                # Gate projection
                gate = pypto.matmul(hidden_chunk, gate_proj_weight, config.dtype)
                
                if config.activation == "swiglu":
                    # Up projection
                    up = pypto.matmul(hidden_chunk, up_proj_weight, config.dtype)
                    
                    # SwiGLU activation
                    pypto.set_vec_tile_shapes(*config.vec_tile_shape)
                    activated = swiglu_activation(gate, up)
                elif config.activation == "gelu":
                    # GELU activation
                    pypto.set_vec_tile_shapes(*config.vec_tile_shape)
                    activated = gelu_activation(gate)
                elif config.activation == "relu":
                    # ReLU activation
                    pypto.set_vec_tile_shapes(*config.vec_tile_shape)
                    activated = relu_activation(gate)
                else:
                    raise ValueError(f"Unsupported activation: {config.activation}")
                
                # Down projection
                pypto.set_cube_tile_shapes(
                    [config.cube_tile_shape[0], config.cube_tile_shape[0]],
                    [config.cube_tile_shape[1], config.cube_tile_shape[1]],
                    [config.cube_tile_shape[2], config.cube_tile_shape[2]]
                )
                pypto.set_matrix_size({basic_batch, intermediate_size, hidden_size})
                output_chunk = pypto.matmul(activated, down_proj_weight, config.dtype, b_trans=True)
                
                # Assemble result back to output
                pypto.assemble(output_chunk, [batch_offset, 0], output)
            
            process_batch_chunk(idx)
    
    inside_ffn()


def create_ffn_module(config: FFNConfig, use_dynamic: Optional[bool] = None):
    """
    Create an FFN module based on configuration.
    
    Parameters
    ----------
    config : FFNConfig
        FFN configuration
    use_dynamic : bool, optional
        Override dynamic shape setting from config
        
    Returns
    -------
    callable
        FFN function (static or dynamic)
    """
    use_dynamic = use_dynamic if use_dynamic is not None else config.use_dynamic_shape
    
    if use_dynamic:
        return lambda inputs, outputs: ffn_dynamic(
            inputs[0], inputs[1], inputs[2], inputs[3], outputs[0], config
        )
    else:
        return lambda inputs, outputs: ffn_static(
            inputs[0], inputs[1], inputs[2], inputs[3], outputs[0], config
        )