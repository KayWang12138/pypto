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
    P  = exp(Q @ K^T * scale - softmax_max) / softmax_sum   (online softmax 重算)
    D  = sum(dY * attention_out, dim=-1, keepdim=True)
    dP = dY @ V^T
    dS = P * (dP - D)
    dV = P^T @ dY
    dQ = dS @ K * scale
    dK = dS^T @ Q * scale

置信度: ⭐⭐⭐⭐ (标准 Flash Attention backward 公式)
"""

import logging
from typing import Tuple

import torch

# Configure logger for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
formatter = logging.Formatter(
    fmt='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    datefmt='[%Y-%m-%d %H:%M:%S]'
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(handler)


def flash_attention_score_grad_golden(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dy: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    attention_out: torch.Tensor,
    scale_value: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    FlashAttentionScoreGrad 参考实现 (纯 PyTorch)

    Args:
        query:         [B, N, S, D], FP16/BF16
        key:           [B, N, S, D], FP16/BF16
        value:         [B, N, S, D], FP16/BF16
        dy:            [B, N, S, D], FP16/BF16, 输出梯度
        softmax_max:   [B, N, S, 8], FP32, 前向 softmax 的 max 值
        softmax_sum:   [B, N, S, 8], FP32, 前向 softmax 的 sum 值
        attention_out: [B, N, S, D], FP16/BF16, 前向输出
        scale_value:   float, 通常为 1.0 / sqrt(D)

    Returns:
        (dQ, dK, dV): 各与对应输入同 shape 和 dtype
    """
    orig_dtype = query.dtype

    # 全部提升到 FP32
    q = query.float()
    k = key.float()
    v = value.float()
    dy_f = dy.float()
    attn_out_f = attention_out.float()

    # 提取 softmax_max/sum 的有效值 (只用第 0 列)
    s_max = softmax_max[:, :, :, 0:1]  # [B, N, S, 1]
    s_sum = softmax_sum[:, :, :, 0:1]  # [B, N, S, 1]

    # Step 1-2: 在线重算 P
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale_value  # [B, N, S, S]
    P = torch.exp(scores - s_max) / s_sum  # [B, N, S, S]

    # Step 3: D = rowsum(dY * attention_out)
    D = (dy_f * attn_out_f).sum(dim=-1, keepdim=True)  # [B, N, S, 1]

    # Step 4: dP = dY @ V^T
    dP = torch.matmul(dy_f, v.transpose(-2, -1))  # [B, N, S, S]

    # Step 5: dS = P * (dP - D)
    dS = P * (dP - D)  # [B, N, S, S]

    # Step 6-8: 输出梯度
    dV = torch.matmul(P.transpose(-2, -1), dy_f)  # [B, N, S, D]
    dQ = torch.matmul(dS, k) * scale_value  # [B, N, S, D]
    dK = torch.matmul(dS.transpose(-2, -1), q) * scale_value  # [B, N, S, D]

    return dQ.to(orig_dtype), dK.to(orig_dtype), dV.to(orig_dtype)


def generate_forward_data(B, N, S, D, dtype=torch.bfloat16, device='cpu'):
    """生成前向数据和中间结果，供反向测试使用。"""
    torch.manual_seed(42)
    scale = 1.0 / (D ** 0.5)

    q = torch.randn(B, N, S, D, dtype=dtype, device=device)
    k = torch.randn(B, N, S, D, dtype=dtype, device=device)
    v = torch.randn(B, N, S, D, dtype=dtype, device=device)
    dy = torch.randn(B, N, S, D, dtype=dtype, device=device)

    # 前向计算得到 softmax 统计量
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    row_max = scores.amax(dim=-1, keepdim=True)  # [B, N, S, 1]
    exp_scores = torch.exp(scores - row_max)
    row_sum = exp_scores.sum(dim=-1, keepdim=True)  # [B, N, S, 1]
    P = exp_scores / row_sum
    attention_out = torch.matmul(P, v.float()).to(dtype)

    # 填充到 [B, N, S, 8] 格式
    softmax_max = torch.zeros(B, N, S, 8, dtype=torch.float32, device=device)
    softmax_max[:, :, :, 0:1] = row_max
    softmax_sum = torch.zeros(B, N, S, 8, dtype=torch.float32, device=device)
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
        logger.info(f"\n--- {tc['name']} (B={tc['B']}, N={tc['N']}, S={tc['S']}, D={tc['D']}) ---")
        q, k, v, dy, sm, ss, ao, scale = generate_forward_data(
            tc['B'], tc['N'], tc['S'], tc['D']
        )
        dQ, dK, dV = flash_attention_score_grad_golden(q, k, v, dy, sm, ss, ao, scale)

        logger.info(f"  dQ shape: {dQ.shape}, dtype: {dQ.dtype}")
        logger.info(f"  dK shape: {dK.shape}, dtype: {dK.dtype}")
        logger.info(f"  dV shape: {dV.shape}, dtype: {dV.dtype}")

        # 基本检查: shape 正确、无 NaN/Inf
        assert dQ.shape == q.shape, f"dQ shape mismatch: {dQ.shape} vs {q.shape}"
        assert dK.shape == k.shape, f"dK shape mismatch: {dK.shape} vs {k.shape}"
        assert dV.shape == v.shape, f"dV shape mismatch: {dV.shape} vs {v.shape}"
        assert not torch.isnan(dQ).any(), "dQ contains NaN"
        assert not torch.isnan(dK).any(), "dK contains NaN"
        assert not torch.isnan(dV).any(), "dV contains NaN"
        assert not torch.isinf(dQ).any(), "dQ contains Inf"
        assert not torch.isinf(dK).any(), "dK contains Inf"
        assert not torch.isinf(dV).any(), "dV contains Inf"

        # 值域检查
        for name, t in [("dQ", dQ), ("dK", dK), ("dV", dV)]:
            t_f = t.float()
            logger.info(f"  {name} range: [{t_f.min().item():.4f}, {t_f.max().item():.4f}]")

        logger.info("  ✓ Passed")

    # 交叉验证: 用 PyTorch autograd 验证
    logger.info("\n--- 交叉验证: PyTorch autograd ---")
    B, N, S, D = 1, 2, 16, 32
    scale = 1.0 / (D ** 0.5)
    q = torch.randn(B, N, S, D, dtype=torch.float64, requires_grad=True)
    k = torch.randn(B, N, S, D, dtype=torch.float64, requires_grad=True)
    v = torch.randn(B, N, S, D, dtype=torch.float64, requires_grad=True)

    # 标准前向 (用 torch.softmax 保证梯度正确)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    P = torch.softmax(scores, dim=-1)
    Y = torch.matmul(P, v)

    dy = torch.randn_like(Y)
    Y.backward(dy)

    # 计算 online softmax 所需的 max 和 sum
    with torch.no_grad():
        scores_det = torch.matmul(q, k.transpose(-2, -1)) * scale
        row_max = scores_det.amax(dim=-1, keepdim=True)
        row_sum = torch.exp(scores_det - row_max).sum(dim=-1, keepdim=True)

    sm = torch.zeros(B, N, S, 8, dtype=torch.float64)
    sm[:, :, :, 0:1] = row_max
    ss = torch.zeros(B, N, S, 8, dtype=torch.float64)
    ss[:, :, :, 0:1] = row_sum

    dQ_g, dK_g, dV_g = flash_attention_score_grad_golden(
        q.detach(), k.detach(), v.detach(), dy.detach(),
        sm.float(), ss.float(), Y.detach(), scale
    )

    for name, grad_auto, grad_golden in [("dQ", q.grad, dQ_g), ("dK", k.grad, dK_g), ("dV", v.grad, dV_g)]:
        diff = (grad_auto.float() - grad_golden.float()).abs().max().item()
        logger.info(f"  {name} max diff vs autograd: {diff:.6e}")
        assert diff < 1e-4, f"{name} diff too large: {diff}"

    logger.info("  ✓ Autograd cross-validation passed")

    logger.info("\n" + "=" * 60)
    logger.info("验证完成 - 所有测试通过")
    logger.info("=" * 60)


if __name__ == "__main__":
    _validate()
