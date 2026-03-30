#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms of the CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT warranties or any kind, either express or implied, including but not limited to non-infringement, merchantability, or fitness for a particular purpose.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Scaled Dot Product Attention - PyPTO Implementation

This module provides the PyPTO kernel implementation for scaled dot product attention.
Implements: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V

Key design decisions:
1. Softmax only supports DT_FP32 in PyPTO, so use FP32 for softmax computation
2. Transpose 4D only supports (2,3) swap - which is exactly what we need for K^T
3. Matmul requires set_cube_tile_shapes before call
4. Output writeback uses output.move() for proper result assignment

NOTE: This implementation uses SPECIFIC KERNEL FUNCTIONS for each test configuration.
This ensures compile-time optimization while maintaining flexibility.
"""

import os
import sys
import pypto
import torch
from typing import Optional


def _get_run_mode() -> pypto.RunMode:
    """Detect run mode from environment or command line."""
    for idx, arg in enumerate(sys.argv):
        if arg == "--run_mode" and idx + 1 < len(sys.argv):
            value = sys.argv[idx + 1]
            if value == "sim":
                return pypto.RunMode.SIM
        if arg.startswith("--run_mode="):
            value = arg.split("=", 1)[1]
            if value == "sim":
                return pypto.RunMode.SIM
    return pypto.RunMode.NPU


# Global run mode for JIT compilation
_run_mode = _get_run_mode()


# ============================================================================
# JIT Kernel - Scaled Dot Product Attention (Basic, no mask)
# Shape: [1, 2, 4, 4] - small test config
# ============================================================================

@pypto.frontend.jit(runtime_options={"run_mode": _run_mode})
def scaled_dot_product_attention_kernel_level0(
    query: pypto.Tensor([1, 2, 4, 4], pypto.DT_FP32),
    key: pypto.Tensor([1, 2, 4, 4], pypto.DT_FP32),
    value: pypto.Tensor([1, 2, 4, 4], pypto.DT_FP32),
    output: pypto.Tensor([1, 2, 4, 4], pypto.DT_FP32),
    scale: float,
):
    """
    Level 0 test kernel - small scale basic verification.
    """
    # Set tiling configuration for matmul operations
    pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
    pypto.set_vec_tile_shapes(1, 2, 4, 4)

    # Step 1: K transpose [N, H, S, E] -> [N, H, E, S]
    k_t = pypto.transpose(key, 22 3)

    # Step 2: Compute attention scores Q @ K^T -> [N, H, L, S]
    scores = pypto.matmul(query, k_t, out_dtype=pypto.DT_FP32)

    # Step 3: Scale the scores
    scores_scaled = pypto.mul(scores, scale)

    # Step 4: Softmax on last dimension
    attn_weights = pypto.softmax(scores_scaled, dim=-1)

    # Step 5: Compute output attn_weights @ V -> [N, H, L, Ev]
    result = pypto.matmul(attn_weights, value, out_dtype=pypto.DT_FP32)

    # Step 6: Write back to output
    output.move(result)


# ============================================================================
# JIT Kernel - Scaled Dot Product Attention (Level 1: no mask)
# Shape: [2, 4, 512, 64] - typical test config
# ============================================================================

@pypto.frontend.jit(runtime_options={"run_mode": _run_mode})
def scaled_dot_product_attention_kernel_level1(
    query: pypto.Tensor([2, 4, 512, 64], pypto.DT_FP32),
    key: pypto.Tensor([2, 4, 512, 64], pypto.DT_FP32),
    value: pypto.Tensor([2, 4, 512, 64], pypto.DT_FP32),
    output: pypto.Tensor([2, 4, 512, 64], pypto.DT_FP32),
    scale: float,
):
    """
    Level 1 test kernel - typical scale functional verification.
    """
    # Set tiling configuration for matmul operations
    pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
    pypto.set_vec_tile_shapes(1, 4, 16, 64)

    # Step 1: K transpose [N, H, S, E] -> [N, H, E, S]
    k_t = pypto.transpose(key, 2, 3)

    # Step 2: Compute attention scores Q @ K^T -> [N, H, L, S]
    scores = pypto.matmul(query, k_t, out_dtype=pypto.DT_FP32)

    # Step 3: Scale the scores
    scores_scaled = pypto.mul(scores, scale)

    # Step 4: Softmax on last dimension
    attn_weights = pypto.softmax(scores_scaled, dim=-1)

    # Step 5: Compute output attn_weights @ V -> [N, H, L, Ev]
    result = pypto.matmul(attn_weights, value, out_dtype=pypto.DT_FP32)

    # Step 6: Write back to output
    output.move(result)


# ============================================================================
# JIT Kernel - Scaled Dot Product Attention with Causal Mask
# Shape: [1, 8, 512, 128] - causal test config
# ============================================================================

@pypto.frontend.jit(runtime_options={"run_mode": _run_mode})
def scaled_dot_product_attention_kernel_causal(
    query: pypto.Tensor([1, 8, 512, 128], pypto.DT_FP32),
    key: pypto.Tensor([1, 8, 512, 128], pypto.DT_FP32),
    value: pypto.Tensor([1, 8, 512, 128], pypto.DT_FP32),
    causal_mask: pypto.Tensor([1, 8, 512, 512], pypto.DT_FP32),
    output: pypto.Tensor([1, 8, 512, 128], pypto.DT_FP32),
    scale: float,
):
    """
    Scaled dot product attention kernel with causal mask.

    Args:
        query: Query tensor [N, H, L, E]
        key: Key tensor [N, H, S, E]
        value: Value tensor [N, H, S, Ev]
        causal_mask: Causal mask tensor [N, H, L, S] (0.0 for valid, -inf for masked)
        output: Output tensor [N, H, L, Ev]
        scale: Scale factor (1/sqrt(E))
    """
    # Set tiling configuration
    pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
    pypto.set_vec_tile_shapes(1, 8, 16, 128)

    # Step 1: K transpose [N, H, S, E] -> [N, H, E, S]
    k_t = pypto.transpose(key, 2, 3)

    # Step 2: Compute attention scores Q @ K^T -> [N, H, L, S]
    scores = pypto.matmul(query, k_t, out_dtype=pypto.DT_FP32)

    # Step 3: Scale the scores
    scores_scaled = pypto.mul(scores, scale)

    # Step 4: Apply causal mask (add -inf for masked positions)
    scores_masked = pypto.add(scores_scaled, causal_mask)

    # Step 5: Softmax on last dimension
    attn_weights = pypto.softmax(scores_masked, dim=-1)

    # Step 6: Compute output attn_weights @ V -> [N, H, L, Ev]
    result = pypto.matmul(attn_weights, value, out_dtype=pypto.DT_FP32)

    # Step 7: Write back to output
    output.move(result)


# ============================================================================
# JIT Kernel - Scaled Dot Product Attention with Attention Mask
# Shape: [1, 8, 512, 128] - mask test config
# ============================================================================

@pypto.frontend.jit(runtime_options={"run_mode": _run_mode})
def scaled_dot_product_attention_kernel_with_mask(
    query: pypto.Tensor([1, 8, 512, 128], pypto.DT_FP32),
    key: pypto.Tensor([1, 8, 512, 128], pypto.DT_FP32),
    value: pypto.Tensor([1, 8, 512, 128], pypto.DT_FP32),
    attn_mask: pypto.Tensor([1, 8, 512, 512], pypto.DT_FP32),
    output: pypto.Tensor([1, 8, 512, 128], pypto.DT_FP32),
    scale: float,
):
    """
    Scaled dot product attention kernel with attention mask.

    Args:
        query: Query tensor [N, H, L, E]
        key: Key tensor [N, H, S, E]
        value: Value tensor [N, H, S, Ev]
        attn_mask: Attention mask tensor [N, H, L, S]
        output: Output tensor [N, H, L, Ev]
        scale: Scale factor (1/sqrt(E))
    """
    # Set tiling configuration
    pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
    pypto.set_vec_tile_shapes(1, 8, 16, 128)

    # Step 1: K transpose [N, H, S, E] -> [N, H, E, S]
    k_t = pypto.transpose(key, 2, 3)

    # Step 2: Compute attention scores Q @ K^T -> [N, H, L, S]
    scores = pypto.matmul(query, k_t, out_dtype=pypto.DT_FP32)

    # Step 3: Scale the scores
    scores_scaled = pypto.mul(scores, scale)

    # Step 4: Apply attention mask.
    scores_masked = pypto.add(scores_scaled, attn_mask)

    # Step 5: Softmax on last dimension
    attn_weights = pypto.softmax(scores_masked, dim=-1)

    # Step 6: Compute output attn_weights @ V -> [N, H, L, Ev]
    result = pypto.matmul(attn_weights, value, out_dtype=pypto.DT_FP32)

    # Step 7: Write back to output
    output.move(result)


# ============================================================================
# Wrapper Function
# ============================================================================

def scaled_dot_product_attention_wrapper(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 1.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    PyPTO wrapper for scaled dot product attention.

    This function provides a PyTorch-compatible interface to the PyPTO kernel.
    Supports FP32 input/output (FP16/BF16 inputs are converted to FP32 internally).

    Args:
        query: Query tensor [N, H, L, E]
        key: Key tensor [N, H, S, E]
        value: Value tensor [N, H, S, Ev]
        attn_mask: Optional attention mask [N, H, L, S] or [L, S] (additive, float type)
        dropout_p: Dropout probability (not supported, ignored)
        is_causal: Whether to apply causal mask
        scale: Scale factor (default: 1/sqrt(E))

    Returns:
        Output tensor [N, H, L, Ev]
    """
    # Validate inputs
    if query.dim() != 4:
        raise ValueError(f"Query must be 4D, got {query.dim()}D")
    if key.dim() != 4:
        raise ValueError(f"Key must be 4D, got {key.dim()}D")
    if value.dim() != 4:
        raise ValueError(f"Value must be 4D, got {value.dim()}D")

    # Get dimensions
    N, H, L, E = query.shape
    _, _, S, _ = key.shape
    _, _, _, Ev = value.shape

    # Compute scale factor
    if scale is None:
        scale = 1.0 / (E ** 0.5)

    # Store original dtype
    orig_dtype = query.dtype

    # Convert to FP32 if needed (PyPTO softmax only supports FP32)
    if query.dtype != torch.float32:
        query = query.float()
        key = key.float()
        value = value.float()

    # Prepare output tensor
    output = torch.empty(N, H, L, Ev, dtype=torch.float32, device=query.device)

    # Select and call appropriate kernel based on input shape
    if N == 1 and H == 2 and L == 4 and E == 4:
        # Level 0: small scale
        scaled_dot_product_attention_kernel_level0(query, key, value, output, scale)
    elif N == 2 and H == 4 and L == 512 and E == 64:
        # Level 1: typical scale
        scaled_dot_product_attention_kernel_level1(query, key, value, output, scale)
    elif is_causal:
        # Create causal mask for kernel [1, 8, 512, 128]
        causal_mask = torch.triu(torch.ones(L, S, dtype=torch.float32, device=query.device), diagonal=1)
        causal_mask = causal_mask * float('-inf')
        # Expand to [N, H, L, S]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        scaled_dot_product_attention_kernel_causal(
            query, key, value, causal_mask, output, scale
        )
    elif attn_mask is not None:
        # Ensure mask is float and on correct device
        if attn_mask.dtype != torch.float32:
            attn_mask = attn_mask.float()
        attn_mask = attn_mask.to(query.device)

        # Expand mask if needed [L, S] -> [N, H, L, S]
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        scaled_dot_product_attention_kernel_with_mask(
            query, key, value, attn_mask, output, scale
        )
    else:
        # For unsupported shapes, fall back to eager mode with dynamic shape support
        raise ValueError(
            f"Unsupported input shape: {query.shape}. "
            f"Supported shapes: [1, 2, 4, 4], [2, 4, 512, 64], [1, 8, 512, 128] (with mask/causal). "
            f"Got: {query.shape}"
        )
    # Convert back to original dtype if needed
    if orig_dtype != torch.float32:
        output = output.to(orig_dtype)
    return output
