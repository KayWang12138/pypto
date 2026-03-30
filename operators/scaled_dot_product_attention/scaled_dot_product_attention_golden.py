#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms of the CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT warranties or any kind, either express or implied, including but not limited to non-infringement, merchantability, or fitness for a particular purpose.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Scaled Dot Product Attention - PyTorch Golden Reference

This module provides the golden reference implementation for scaled dot product attention.
It corresponds to PyTorch's F.scaled_dot_product_attention.

"""

import torch
from typing import Optional


def scaled_dot_product_attention_golden(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    PyTorch reference implementation of scaled dot product attention.


    This implements the standard attention mechanism:
        Attention(Q,K,V) = softmax(QK^T / sqrt(d)) @ V


    Args:
        query: Query tensor [N, H, L, E]
        key: Key tensor [N, H, S, E]
        value: Value tensor [N, H, S, Ev]
        attn_mask: Attention mask [N, H, L, S] or [L, S]
        dropout_p: Dropout probability
        is_causal: Whether to apply causal mask
        scale: Scale factor (default: 1/sqrt(E))

    Returns:
        Output tensor [N, H, L, Ev]
    """
    # Validate inputs
    if query.dim() != 4:
        raise ValueError(f"Query must be 4D, got {query.dim()}D")
    if key.dim() != 4:
        raise ValueError(f"Key should be 4D, got {key.dim()}D")
    if value.dim() != 4:
        raise ValueError(f"Value should be 4D, got {value.dim()}D")

    # Get dimensions
    N, H, L, E = query.shape
    _, _, S, _ = key.shape
    _, _, _, Ev = value.shape

    # Compute scale factor
    if scale is None:
        scale = 1.0 / (E ** 0.5)

    # Compute attention scores: Q @ K^T
    # Key shape: [N, H, S, E] -> transpose to [N, H, E, S]
    scores = torch.matmul(query, key.transpose(-2, -1))  # [N, H, L, S]

    # Scale
    scores = scores * scale

    # Apply causal mask if requested
    if is_causal:
        # Create lower triangular mask
        causal_mask = torch.tril(torch.ones(L, S), diagonal=1)
        # Expand to [N, H, L, S] if needed
        if causal_mask.dim() == 2:
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
        scores = scores.masked_fill(scores == float('-inf'))

    # Apply attention mask if provided
    if attn_mask is not None:
        # Handle different mask types
        if attn_mask.dtype == torch.bool:
            # Boolean mask: True positions are kept, False are masked with -inf
            scores = scores.masked_fill(attn_mask == False, float('-inf'))
        else:
            # Float mask: Add directly
            scores = scores + attn_mask

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)  # [N, H, L, S]

    # Apply dropout (PyTorch reference doesn't apply dropout in golden)
    # Note: In actual PyTorch, dropout is applied during training only
    # We keep this for API compatibility but don't apply it in golden

    # Compute output: attn_weights @ V
    output = torch.matmul(attn_weights, value)  # [N, H, L, Ev]

    return output


