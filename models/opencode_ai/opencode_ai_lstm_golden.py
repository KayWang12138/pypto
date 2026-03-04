#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
OpenCode AI LSTM Golden Reference Module

This module provides golden reference implementations for LSTM computation.
Used for validating the correctness of PyPTO kernel implementations.

Main Functions:
    - rms_norm_golden: Golden reference for RMSNorm
    - gelu_approx_sigmoid_golden: Golden reference for GELU activation
    - opencode_ai_lstm_compute: Golden reference for LSTM computation
"""
import torch
import numpy as np


def rms_norm_gold(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Golden reference for RMSNorm."""
    x = x.to(torch.float32)
    mean_square = x.pow(2).mean(-1, keepdim=True)
    inv_rms = torch.rsqrt(mean_square + eps)
    return x * inv_rms


def gelu_approx_sigmoid_golden(x: torch.Tensor) -> torch.Tensor:
    """
    GELU approximation using Sigmoid: x * sigmoid(1.702 * x).
    Matches the NPU implementation for alignment.
    """
    return x * torch.sigmoid(1.702 * x)


def opencode_ai_lstm_compute(
    states_4d: torch.Tensor,
    z4_4d: torch.Tensor,
    prev_cell: torch.Tensor,
    alpha: float,
    eps_cell: float,
    eps_state: float,
    w_cell: torch.Tensor = None,
    b_cell: torch.Tensor = None,
    w_state: torch.Tensor = None,
    b_state: torch.Tensor = None,
) -> tuple:
    """Golden reference for OpenCode AI LSTM kernel."""

    # 1. Input Fusion
    fused = states_4d + alpha * z4_4d

    # 2. Chunking: [Forget, Input, Output, Cell_Candidate]
    chunk_size = fused.shape[-1] // 4
    pre_f, pre_i, pre_o, pre_c = torch.split(fused, chunk_size, dim=-1)

    # 3. Gates
    f_gate = torch.sigmoid(pre_f)
    i_gate = torch.sigmoid(pre_i)

    # 4. Pre-Cell Path
    c_cand_norm = rms_norm_gold(pre_c, eps_cell)

    if w_cell is not None:
        c_cand_norm = c_cand_norm * w_cell
    if b_cell is not None:
        c_cand_norm = c_cand_norm + b_cell

    c_act = gelu_approx_sigmoid_golden(c_cand_norm)

    # 5. Cell Update
    c_new = prev_cell * f_gate + c_act * i_gate

    # 6. Post-Cell Path
    h_temp = rms_norm_gold(c_new, eps_state)

    if w_state is not None:
        h_temp = h_temp * w_state
    if b_state is not None:
        h_temp = h_temp + b_state

    h_act = gelu_approx_sigmoid_golden(h_temp)

    # 7. Output Gate & Final Output
    o_gate = torch.sigmoid(pre_o)
    h_new = h_act * o_gate

    return h_new, c_new
