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
FlashAttentionScoreGrad Golden 参考实现

公式:
  前向: Y = Softmax(Q @ K^T / sqrt(D)) @ V
  反向:
    p  = exp(Q @ K^T * scale - softmax_max) / softmax_sum   (online softmax 重算)
    d  = sum(dY * attention_out, dim=-1, keepdim=True)
    dp = dY @ V^T
    ds = p * (dp - d)
    dv = p^T @ dY
    dq = ds @ K * scale
    dk = ds^T @ Q * scale

置信度: ⭐⭐⭐⭐ (标准 Flash Attention backward 公式)
"""

import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class AttentionGradInputs:
    """Container for attention gradient inputs."""

    def __init__(self, query, key, value, dy,
                 softmax_max, softmax_sum, attention_out, scale_value):
        self.query = query
        self.key = key
        self.value = value
        self.dy = dy
        self.softmax_max = softmax_max
        self.softmax_sum = softmax_sum
        self.attention_out = attention_out
        self.scale_value = scale_value


def flash_attention_score_grad_golden(
        inputs: AttentionGradInputs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    FlashAttentionScoreGrad 参考实现 (纯 PyTorch)

    Args:
        inputs: AttentionGradInputs container

    Returns:
        (dQ, dK, dV): 各与对应输入同 shape 和 dtype
    """
    orig_dtype = inputs.query.dtype

    # 全部提升到 FP32
    q = inputs.query.float()
    k = inputs.key.float()
    v = inputs.value.float()
    dy_f = inputs.dy.float()
    attn_out_f = inputs.attention_out.float()

    # 提取 softmax_max/sum 的有效值 (只用第 0 列)
    s_max = inputs.softmax_max[:, :, :, 0:1]
    s_sum = inputs.softmax_sum[:, :, :, 0:1]

    # Recompute p online
    scores = torch.matmul(q, k.transpose(-2, -1)) * inputs.scale_value
    p_mat = torch.exp(scores - s_max) / s_sum

    # Compute row sum of dY * attention_out
    d_var = (dy_f * attn_out_f).sum(dim=-1, keepdim=True)

    # Compute dY @ V^T
    dp_var = torch.matmul(dy_f, v.transpose(-2, -1))

    # Compute p * (dp - d)
    ds_var = p_mat * (dp_var - d_var)

    # Compute output gradients
    dq_out = torch.matmul(ds_var, k) * inputs.scale_value
    dk_out = torch.matmul(ds_var.transpose(-2, -1), q) * inputs.scale_value
    dv_out = torch.matmul(p_mat.transpose(-2, -1), dy_f)

    return dq_out.to(orig_dtype), dk_out.to(orig_dtype), dv_out.to(orig_dtype)


def generate_forward_data(
        batch_size, num_heads, seq_len, head_dim,
        dtype=torch.bfloat16, device='cpu'):
    """生成前向数据和中间结果，供反向测试使用。"""
    torch.manual_seed(42)
    scale = 1.0 / (head_dim ** 0.5)

    q = torch.randn(
        batch_size, num_heads, seq_len, head_dim,
        dtype=dtype, device=device)
    k = torch.randn(
        batch_size, num_heads, seq_len, head_dim,
        dtype=dtype, device=device)
    v = torch.randn(
        batch_size, num_heads, seq_len, head_dim,
        dtype=dtype, device=device)
    dy = torch.randn(
        batch_size, num_heads, seq_len, head_dim,
        dtype=dtype, device=device)

    # 前向计算得到 softmax 统计量
    scores = torch.matmul(
        q.float(), k.float().transpose(-2, -1)) * scale
    row_max = scores.amax(dim=-1, keepdim=True)
    exp_scores = torch.exp(scores - row_max)
    row_sum = exp_scores.sum(dim=-1, keepdim=True)
    p_mat = exp_scores / row_sum
    attention_out = torch.matmul(p_mat, v.float()).to(dtype)

    # 填充到 [B, N, S, 8] 格式
    softmax_max = torch.zeros(
        batch_size, num_heads, seq_len, 8,
        dtype=torch.float32, device=device)
    softmax_max[:, :, :, 0:1] = row_max
    softmax_sum = torch.zeros(
        batch_size, num_heads, seq_len, 8,
        dtype=torch.float32, device=device)
    softmax_sum[:, :, :, 0:1] = row_sum

    return q, k, v, dy, softmax_max, softmax_sum, attention_out, scale


def _validate():
    """自动生成的验证函数"""
    import numpy as np

    logger.info("=" * 60)
    logger.info("flash_attention_score_grad_golden 验证报告")
    logger.info("=" * 60)

    test_cases = [
        {"name": "Level 0: 最小", "B": 1, "N": 1, "S": 16, "D": 64},
        {"name": "Level 1: 典型", "B": 2, "N": 8, "S": 64, "D": 64},
        {"name": "Level 2: 中等", "B": 2, "N": 8, "S": 128, "D": 128},
    ]

    for tc in test_cases:
        logger.info(
            "\n--- %s (B=%d, N=%d, S=%d, D=%d) ---",
            tc['name'], tc['B'], tc['N'], tc['S'], tc['D'])
        q, k, v, dy, sm, ss, ao, scale = generate_forward_data(
            tc['B'], tc['N'], tc['S'], tc['D']
        )
        inputs = AttentionGradInputs(q, k, v, dy, sm, ss, ao, scale)
        dQ, dK, dV = flash_attention_score_grad_golden(inputs)

        logger.info("  dQ shape: %s, dtype: %s", dQ.shape, dQ.dtype)
        logger.info("  dK shape: %s, dtype: %s", dK.shape, dK.dtype)
        logger.info("  dV shape: %s, dtype: %s", dV.shape, dV.dtype)

        # 基本检查
        assert dQ.shape == q.shape
        assert dK.shape == k.shape
        assert dV.shape == v.shape
        assert not torch.isnan(dQ).any()
        assert not torch.isnan(dK).any()
        assert not torch.isnan(dV).any()
        assert not torch.isinf(dQ).any()
        assert not torch.isinf(dK).any()
        assert not torch.isinf(dV).any()

        # 值域检查
        for name, t in [("dQ", dQ), ("dK", dK), ("dV", dV)]:
            t_f = t.float()
            logger.info(
                "  %s range: [%.4f, %.4f]", name, t_f.min().item(),
                t_f.max().item())

        logger.info("  ✓ Passed")

    # 交叉验证
    logger.info("\n--- 交叉验证: PyTorch autograd ---")
    batch_size, num_heads, seq_len, head_dim = 1, 2, 16, 32
    scale = 1.0 / (head_dim ** 0.5)
    q = torch.randn(
        batch_size, num_heads, seq_len, head_dim,
        dtype=torch.float64, requires_grad=True)
    k = torch.randn(
        batch_size, num_heads, seq_len, head_dim,
        dtype=torch.float64, requires_grad=True)
    v = torch.randn(
        batch_size, num_heads, seq_len, head_dim,
        dtype=torch.float64, requires_grad=True)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    p_mat = torch.softmax(scores, dim=-1)
    y_out = torch.matmul(p_mat, v)

    dy = torch.randn_like(y_out)
    y_out.backward(dy)

    with torch.no_grad():
        scores_det = torch.matmul(q, k.transpose(-2, -1)) * scale
        row_max = scores_det.amax(dim=-1, keepdim=True)
        row_sum = torch.exp(scores_det - row_max).sum(dim=-1, keepdim=True)

    sm = torch.zeros(
        batch_size, num_heads, seq_len, 8, dtype=torch.float64)
    sm[:, :, :, 0:1] = row_max
    ss = torch.zeros(
        batch_size, num_heads, seq_len, 8, dtype=torch.float64)
    ss[:, :, :, 0:1] = row_sum

    inputs = AttentionGradInputs(
        q.detach(), k.detach(), v.detach(), dy.detach(),
        sm.float(), ss.float(), y_out.detach(), scale)
    dQ_g, dK_g, dV_g = flash_attention_score_grad_golden(inputs)

    for name, grad_auto, grad_golden in [
            ("dQ", q.grad, dQ_g), ("dK", k.grad, dK_g),
            ("dV", v.grad, dV_g)]:
        diff = (
            grad_auto.float() - grad_golden.float()
        ).abs().max().item()
        logger.info("  %s max diff vs autograd: %.6e", name, diff)
        assert diff < 1e-4

    logger.info("  ✓ Autograd cross-validation passed")

    logger.info("\n" + "=" * 60)
    logger.info("验证完成 - 所有测试通过")
    logger.info("=" * 60)


if __name__ == "__main__":
    _validate()
