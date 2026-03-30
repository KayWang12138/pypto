#!/usr/bin/env python3
# coding: utf-8

"""PyPTO rms_norm golden reference implementation.

模板说明:
  - 本文件是 rms_norm_golden.py 的固定模板，由 pypto-golden-generator 在 Stage 2a 生成。
  - 所有 rms_norm 占位符需替换为实际算子名称。
  - golden 必须是纯 PyTorch 实现，禁止引入 pypto。
  - 导出函数 rms_norm_golden() 供 test_rms_norm.py 调用。
  - 参考 examples/ 中的 golden 函数风格（如 silu_golden, layernorm_golden)。
"""

import torch


# ----------------------------------------------------------------
# Golden 参考实现（纯 torch)
# ----------------------------------------------------------------

def rms_norm_golden(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """PyTorch 参考实现.

    根据 spec.md 中的数学公式实现:
    y = x * gamma / sqrt(mean(x^2) + eps)

    仅使用 torch 标准操作，不依赖 pypto.

    Args:
        x: 输入 tensor, shape [b, s, d]
        weight: 缩放参数 gamma, shape [d]
        eps: 防止除零的小常数, 默认 1e-6

    Returns:
        计算结果 tensor, shape [b, s, d]
    """
    # RMS Norm: y = x * gamma / sqrt(mean(x^2) + eps)
    # 1. 计算 x^2
    x_squared = x * x

    # 2. 在最后一个维度上计算均值
    mean_sq = x_squared.mean(dim=-1, keepdim=True)

    # 3. 计算 RMS = sqrt(mean(x^2) + eps)
    rms = torch.sqrt(mean_sq + eps)

    # 4. 归一化: x / rms
    normalized = x / rms

    # 5. 应用缩放参数: normalized * weight
    # weight 的 shape 是 [d]，需要广播到 [b, s, d]
    y = normalized * weight

    return y
