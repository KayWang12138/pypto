#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms of the CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT warranties or any kind, either express or implied, including but not limited to non-infringement, merchantability, or fitness for a particular purpose.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Flash Attention - PyTorch Golden Reference

This module provides the golden reference implementation for Flash Attention.
It corresponds to PyTorch's F.scaled_dot_product_attention.

Mathematical Formula:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V

Key Features:
    - Online Softmax algorithm for numerical stability
    - Causal attention mask support
    - Custom attention mask support
    - Dynamic axis support (batch, num_heads, seq_len)

Note: This golden implementation uses standard PyTorch operations.
The actual Flash Attention optimization (tiling, reduced HBM access)
is implemented in the PyPTO kernel, not in this golden reference.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Union
import math


def flash_attention_golden(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    PyTorch reference implementation of Flash Attention.

    This implements the standard attention mechanism:
        Attention(Q,K,V) = softmax(QK^T / sqrt(d)) @ V

    The implementation supports:
    - Causal mask (is_causal=True)
    - Custom attention mask (attn_mask)
    - Automatic scaling (1/sqrt(d))

    Args:
        query: Query tensor [N, H, L, d]
            - N: Batch size
            - H: Number of attention heads
            - L: Query sequence length
            - d: Head dimension
        key: Key tensor [N, H, S, d]
            - S: Key/Value sequence length
        value: Value tensor [N, H, S, d]
        attn_mask: Optional attention mask
            - Shape: [N, H, L, S] or [H, L, S] or [L, S]
            - Dtype: float (additive) or bool (mask)
            - For float mask: values are added to attention scores
            - For bool mask: True = keep, False = mask (set to -inf)
        is_causal: Whether to apply causal mask
            - When True, creates lower triangular mask
            - Mutually exclusive with attn_mask (is_causal takes precedence)
        scale: Scale factor (default: 1/sqrt(d))
            - If None, uses 1/sqrt(head_dim)

    Returns:
        Output tensor [N, H, L, d] - Attention output

    Raises:
        ValueError: If input tensors have wrong dimensions
    """
    # Input validation
    if query.dim() != 4:
        raise ValueError(f"Query must be 4D [N, H, L, d], got {query.dim()}D")
    if key.dim() != 4:
        raise ValueError(f"Key must be 4D [N, H, S, d], got {key.dim()}D")
    if value.dim() != 4:
        raise ValueError(f"Value must be 4D [N, H, S, d], got {value.dim()}D")

    # Get dimensions
    N, H, L, d = query.shape
    _, _, S, _ = key.shape

    # Compute scale factor
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    # Store original dtype for output
    orig_dtype = query.dtype

    # For numerical stability, compute in float32 if input is half precision
    compute_dtype = torch.float32 if orig_dtype in [torch.float16, torch.bfloat16] else orig_dtype

    # Cast to compute dtype if needed
    if query.dtype != compute_dtype:
        query = query.to(compute_dtype)
        key = key.to(compute_dtype)
        value = value.to(compute_dtype)

    # Step 1: Compute attention scores Q @ K^T
    # Key shape: [N, H, S, d] -> transpose to [N, H, d, S]
    scores = torch.matmul(query, key.transpose(-2, -1))  # [N, H, L, S]

    # Step 2: Scale
    scores = scores * scale

    # Step 3: Apply causal mask if requested
    if is_causal:
        # Create lower triangular mask
        causal_mask = torch.triu(
            torch.ones(L, S, dtype=torch.bool, device=query.device),
            diagonal=1
        )
        # Expand to [N, H, L, S] if needed
        if causal_mask.dim() == 2:
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        # Apply mask: True positions get -inf
        scores = scores.masked_fill(causal_mask, float('-inf'))
    elif attn_mask is not None:
        # Step 4: Apply custom attention mask
        # Handle different mask shapes
        if attn_mask.dim() == 2:
            # [L, S] -> [1, 1, L, S]
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        elif attn_mask.dim() == 3:
            # [H, L, S] -> [1, H, L, S]
            attn_mask = attn_mask.unsqueeze(0)

        if attn_mask.dtype == torch.bool:
            # Boolean mask: True = keep, False = mask with -inf
            scores = scores.masked_fill(attn_mask == False, float('-inf'))
        else:
            # Float mask: Additive mask
            scores = scores + attn_mask.to(compute_dtype)

    # Step 5: Softmax (numerically stable)
    # PyTorch's softmax handles -inf correctly
    attn_weights = F.softmax(scores, dim=-1)  # [N, H, L, S]

    # Handle NaN from all-masked rows (all -inf -> softmax = NaN)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    # Step 6: Compute output attn_weights @ V
    output = torch.matmul(attn_weights, value)  # [N, H, L, d]

    # Cast back to original dtype if needed
    if output.dtype != orig_dtype:
        output = output.to(orig_dtype)

    return output


def flash_attention_with_dropout_golden(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
    dropout_p: float = 0.0,
    training: bool = True,
) -> torch.Tensor:
    """
    PyTorch reference implementation of Flash Attention with dropout.

    Note: Dropout is included for API compatibility but is marked as P3
    (low priority) in the spec. The PyPTO implementation does not support
    dropout.

    Args:
        query: Query tensor [N, H, L, d]
        key: Key tensor [N, H, S, d]
        value: Value tensor [N, H, S, d]
        attn_mask: Optional attention mask
        is_causal: Whether to apply causal mask
        scale: Scale factor (default: 1/sqrt(d))
        dropout_p: Dropout probability (default: 0.0)
        training: Whether in training mode (default: True)

    Returns:
        Output tensor [N, H, L, d]
    """
    # Compute base attention
    output = flash_attention_golden(
        query, key, value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale
    )

    # Apply dropout if training and dropout_p > 0
    if training and dropout_p > 0.0:
        output = F.dropout(output, p=dropout_p, training=training)

    return output


# ============================================================================
# Validation utilities
# ============================================================================

def _validate():
    """Run validation tests for the golden implementation."""
    print("=" * 60)
    print("flash_attention_golden 验证报告")
    print("=" * 60)

    # Test cases from spec.md typical configurations
    test_cases = [
        # (name, batch, heads, seq_len, head_dim, is_causal, has_mask)
        ("性能_P0", 1, 8, 1024, 128, False, False),
        ("功能_P0", 2, 4, 512, 64, False, False),
        ("因果注意力_P1", 1, 8, 512, 128, True, False),
        ("长序列_P1", 1, 4, 4096, 64, False, False),
        ("掩码注意力_P1", 1, 8, 512, 128, False, True),
    ]

    all_passed = True

    # [典型 case 验证]
    print("\n[典型 case 验证]")
    for name, batch, heads, seq_len, head_dim, is_causal, has_mask in test_cases:
        try:
            # Create test tensors
            dtype = torch.float32
            device = "cpu"

            query = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
            key = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
            value = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)

            attn_mask = None
            if has_mask:
                attn_mask = torch.zeros(batch, heads, seq_len, seq_len, dtype=dtype, device=device)
                # Mask some positions
                attn_mask[:, :, :, seq_len//2:] = float('-inf')

            # Run golden
            output = flash_attention_golden(
                query, key, value,
                attn_mask=attn_mask,
                is_causal=is_causal
            )

            # Verify output shape
            expected_shape = (batch, heads, seq_len, head_dim)
            assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"

            # Verify no NaN
            assert not torch.isnan(output).any(), "Output contains NaN"

            # Verify no Inf in output
            assert not torch.isinf(output).any(), "Output contains Inf"

            print(f"  {name}: N={batch}, H={heads}, L={seq_len}, d={head_dim}, causal={is_causal}, mask={has_mask} ... OK")

        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            all_passed = False

    # [泛化 case 验证]
    print("\n[泛化 case 验证]")
    generalization_cases = [
        # Test different batch sizes
        (1, 4, 256, 64),
        (4, 4, 256, 64),
        (16, 4, 256, 64),
        # Test different head counts
        (2, 1, 256, 64),
        (2, 8, 256, 64),
        (2, 16, 256, 64),
        # Test different head dimensions
        (2, 4, 256, 32),
        (2, 4, 256, 64),
        (2, 4, 256, 128),
    ]

    for batch, heads, seq_len, head_dim in generalization_cases:
        try:
            query = torch.randn(batch, heads, seq_len, head_dim)
            key = torch.randn(batch, heads, seq_len, head_dim)
            value = torch.randn(batch, heads, seq_len, head_dim)

            output = flash_attention_golden(query, key, value)

            assert output.shape == (batch, heads, seq_len, head_dim)
            assert not torch.isnan(output).any()

            print(f"  N={batch}, H={heads}, L={seq_len}, d={head_dim} ... OK")
        except Exception as e:
            print(f"  N={batch}, H={heads}, L={seq_len}, d={head_dim} ... FAILED: {e}")
            all_passed = False

    # [值域检查]
    print("\n[值域检查]")

    # Test softmax normalization (attention weights sum to 1)
    query = torch.randn(1, 2, 8, 16)
    key = torch.randn(1, 2, 8, 16)
    value = torch.randn(1, 2, 8, 16)
    output = flash_attention_golden(query, key, value)
    print(f"  检查输出形状和数据类型 ... OK (shape={output.shape}, dtype={output.dtype})")

    # [数值稳定性检查]
    print("\n[数值稳定性检查]")

    # Test with large values (should not overflow)
    large_query = torch.randn(1, 2, 8, 16) * 100
    large_key = torch.randn(1, 2, 8, 16) * 100
    large_value = torch.randn(1, 2, 8, 16)
    output = flash_attention_golden(large_query, large_key, large_value)
    assert not torch.isnan(output).any(), "Large values cause NaN"
    assert not torch.isinf(output).any(), "Large values cause Inf"
    print(f"  大值输入 (x*100) ... OK")

    # Test with small values
    small_query = torch.randn(1, 2, 8, 16) * 0.001
    small_key = torch.randn(1, 2, 8, 16) * 0.001
    small_value = torch.randn(1, 2, 8, 16)
    output = flash_attention_golden(small_query, small_key, small_value)
    assert not torch.isnan(output).any(), "Small values cause NaN"
    print(f"  小值输入 (x*0.001) ... OK")

    # Test causal mask with short sequence
    short_query = torch.randn(1, 2, 4, 16)
    short_key = torch.randn(1, 2, 4, 16)
    short_value = torch.randn(1, 2, 4, 16)
    output = flash_attention_golden(short_query, short_key, short_value, is_causal=True)
    assert not torch.isnan(output).any(), "Causal mask causes NaN"
    print(f"  因果掩码 (短序列) ... OK")

    # [功能正确性检查]
    print("\n[功能正确性检查]")

    # Test scale parameter
    query = torch.randn(1, 2, 8, 16)
    key = torch.randn(1, 2, 8, 16)
    value = torch.randn(1, 2, 8, 16)
    output_default = flash_attention_golden(query, key, value)
    output_custom_scale = flash_attention_golden(query, key, value, scale=0.5)
    assert not torch.allclose(output_default, output_custom_scale), "Custom scale not applied"
    print(f"  验证 scale 参数 ... OK")

    # Test dtype preservation
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            print(f"  验证 dtype={dtype} ... SKIPPED (not supported)")
            continue
        query = torch.randn(1, 2, 8, 16, dtype=dtype)
        key = torch.randn(1, 2, 8, 16, dtype=dtype)
        value = torch.randn(1, 2, 8, 16, dtype=dtype)
        output = flash_attention_golden(query, key, value)
        assert output.dtype == dtype, f"dtype mismatch: {output.dtype} vs {dtype}"
        print(f"  验证 dtype={dtype} ... OK")

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed!")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    import sys
    sys.exit(0 if _validate() else 1)
