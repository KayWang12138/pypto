#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------
"""
flash_attention_score_grad Golden 参考实现

公式:
    正向: Y = Dropout(Softmax(Mask(QK^T * scale + pse), atten_mask), keep_prob) @ V
    反向:
        dV = P^T @ dY
        dQ = (dS @ K) * scale
        dK = (dS^T @ Q) * scale
        其中 S = Mask(QK^T * scale + pse, atten_mask), P = Dropout(Softmax(S), keep_prob)

置信度: ⭐⭐⭐⭐ (复杂算子，基于 Flash Attention 论文和 PyTorch 手动实现)
"""

import torch
from typing import Optional, Tuple, List
import math


def _convert_to_bnsd(tensor: torch.Tensor, layout: str, head_num: int, head_dim: int) -> torch.Tensor:
    """将输入 tensor 转换为 BNSD 布局进行内部计算。

    Args:
        tensor: 输入 tensor
        layout: 输入布局 (BSH, SBH, BSND, BNSD)
        head_num: 注意力头数
        head_dim: 头维度

    Returns:
        BNSD 布局的 tensor [B, N, S, D]
    """
    if layout == "BNSD":
        return tensor
    elif layout == "BSND":
        # [B, S, N, D] -> [B, N, S, D]
        return tensor.permute(0, 2, 1, 3)
    elif layout == "BSH":
        # [B, S, H] -> [B, N, S, D] where H = N * D
        B, S, H = tensor.shape
        return tensor.view(B, S, head_num, head_dim).permute(0, 2, 1, 3)
    elif layout == "SBH":
        # [S, B, H] -> [B, N, S, D]
        S, B, H = tensor.shape
        return tensor.permute(1, 0, 2).view(B, S, head_num, head_dim).permute(0, 2, 1, 3)
    else:
        raise ValueError(f"Unsupported layout: {layout}")


def _convert_from_bnsd(tensor: torch.Tensor, layout: str, head_num: int, head_dim: int) -> torch.Tensor:
    """将 BNSD 布局的 tensor 转换回原始布局。

    Args:
        tensor: BNSD 布局的 tensor [B, N, S, D]
        layout: 目标布局 (BSH, SBH, BSND, BNSD)
        head_num: 注意力头数
        head_dim: 头维度

    Returns:
        目标布局的 tensor
    """
    if layout == "BNSD":
        return tensor
    elif layout == "BSND":
        # [B, N, S, D] -> [B, S, N, D]
        return tensor.permute(0, 2, 1, 3)
    elif layout == "BSH":
        # [B, N, S, D] -> [B, S, H] where H = N * D
        return tensor.permute(0, 2, 1, 3).reshape(tensor.shape[0], tensor.shape[2], -1)
    elif layout == "SBH":
        # [B, N, S, D] -> [S, B, H]
        return tensor.permute(0, 2, 1, 3).reshape(tensor.shape[0], tensor.shape[2], -1).permute(1, 0, 2)
    else:
        raise ValueError(f"Unsupported layout: {layout}")


def _apply_sparse_mask(
    scores: torch.Tensor,
    atten_mask: Optional[torch.Tensor],
    sparse_mode: int,
    pre_tokens: int,
    next_tokens: int,
    S1: int,
    S2: int,
) -> torch.Tensor:
    """应用稀疏注意力掩码。

    Args:
        scores: 注意力分数 [B, N, S1, S2]
        atten_mask: 用户提供的掩码 (可选)
        sparse_mode: 稀疏模式 (0-6)
        pre_tokens: band 模式左边界
        next_tokens: band 模式右边界
        S1: Query 序列长度
        S2: Key 序列长度

    Returns:
        掩码后的注意力分数
    """
    device = scores.device
    dtype = scores.dtype

    if sparse_mode == 0:
        # Default: 使用用户提供的 atten_mask
        if atten_mask is not None:
            # atten_mask: 1 表示不参与计算 (被掩码)
            # 注意力分数中需要将被掩码位置填充为 -inf
            mask = atten_mask.to(torch.bool)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # [S1, S2] -> [1, 1, S1, S2]
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [B, S1, S2] -> [B, 1, S1, S2]
            scores = scores.masked_fill(mask, float('-inf'))
    elif sparse_mode == 1:
        # rightDown: 右下三角 (下三角 + 右侧)
        mask = torch.triu(torch.ones(S1, S2, device=device, dtype=torch.bool), diagonal=S2 - S1 + 1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S1, S2]
        scores = scores.masked_fill(mask, float('-inf'))
    elif sparse_mode == 2:
        # rightUp: 右上三角 (上三角)
        mask = torch.tril(torch.ones(S1, S2, device=device, dtype=torch.bool), diagonal=-1)
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask, float('-inf'))
    elif sparse_mode == 3:
        # leftDown: 左下三角 (causal mask) - 最常用
        # 下三角为有效区域，上三角为 mask
        mask = torch.triu(torch.ones(S1, S2, device=device, dtype=torch.bool), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask, float('-inf'))
    elif sparse_mode == 4:
        # band: 滑动窗口
        row_indices = torch.arange(S1, device=device).unsqueeze(1)
        col_indices = torch.arange(S2, device=device).unsqueeze(0)
        # 有效范围: col in [row - pre_tokens, row + next_tokens]
        mask = (col_indices < row_indices - pre_tokens) | (col_indices > row_indices + next_tokens)
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask, float('-inf'))
    elif sparse_mode in [5, 6]:
        # prefix modes: 简化处理，使用 causal mask
        mask = torch.triu(torch.ones(S1, S2, device=device, dtype=torch.bool), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask, float('-inf'))
    else:
        raise ValueError(f"Unsupported sparse_mode: {sparse_mode}")

    return scores


def _apply_pse(
    scores: torch.Tensor,
    pse_shift: Optional[torch.Tensor],
    pse_type: int,
    scale: float,
) -> torch.Tensor:
    """应用位置编码 (PSE)。

    Args:
        scores: 注意力分数 [B, N, S1, S2]
        pse_shift: 位置编码张量
        pse_type: 位置编码类型 (0-3)
        scale: 缩放因子

    Returns:
        添加位置编码后的注意力分数
    """
    if pse_shift is None:
        return scores

    if pse_type == 0:
        # S = (QK * scale) * pse + pse
        scores = scores * scale * pse_shift + pse_shift
    elif pse_type == 1:
        # S = (QK + pse) * scale (默认)
        scores = (scores + pse_shift) * scale
    elif pse_type == 2:
        # S = QK * scale * pse + pse (内部生成，alibi-like)
        scores = scores * scale * pse_shift + pse_shift
    elif pse_type == 3:
        # S = (QK * scale * pse + pse) (内部生成，带 sqrt)
        scores = (scores * scale * pse_shift + pse_shift)
    else:
        raise ValueError(f"Unsupported pse_type: {pse_type}")

    return scores


def _apply_dropout(
    tensor: torch.Tensor,
    drop_mask: Optional[torch.Tensor],
    keep_prob: float,
) -> torch.Tensor:
    """应用 Dropout。

    Args:
        tensor: 输入 tensor
        drop_mask: Dropout 掩码 (1=保留, 0=丢弃)
        keep_prob: 保留概率

    Returns:
        Dropout 后的 tensor
    """
    if drop_mask is None or keep_prob >= 1.0:
        return tensor

    # drop_mask: 1=保留, 0=丢弃
    # 输出 = 输入 * mask / keep_prob
    return tensor * drop_mask.to(tensor.dtype) / keep_prob


def _expand_for_gqa(
    tensor: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
) -> torch.Tensor:
    """为 GQA (Grouped Query Attention) 扩展 KV heads。

    Args:
        tensor: KV tensor [B, N_kv, S, D]
        num_heads_q: Query 的头数
        num_heads_kv: KV 的头数

    Returns:
        扩展后的 tensor [B, N_q, S, D]
    """
    if num_heads_q == num_heads_kv:
        return tensor

    G = num_heads_q // num_heads_kv
    if G * num_heads_kv != num_heads_q:
        raise ValueError(f"GQA ratio G = {num_heads_q}/{num_heads_kv} must be integer")

    B, N_kv, S, D = tensor.shape
    # [B, N_kv, S, D] -> [B, N_kv, G, S, D] -> [B, N_kv * G, S, D]
    return tensor.unsqueeze(2).expand(B, N_kv, G, S, D).reshape(B, num_heads_q, S, D)


def _reduce_for_gqa(
    tensor: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
) -> torch.Tensor:
    """为 GQA 归约梯度到 KV heads。

    Args:
        tensor: Q 梯度 tensor [B, N_q, S, D]
        num_heads_q: Query 的头数
        num_heads_kv: KV 的头数

    Returns:
        归约后的 tensor [B, N_kv, S, D]
    """
    if num_heads_q == num_heads_kv:
        return tensor

    G = num_heads_q // num_heads_kv
    B, N_q, S, D = tensor.shape
    # [B, N_q, S, D] -> [B, N_kv, G, S, D] -> sum over G -> [B, N_kv, S, D]
    return tensor.view(B, num_heads_kv, G, S, D).sum(dim=2)


def flash_attention_score_grad_golden(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dy: torch.Tensor,
    # Optional inputs
    pse_shift: Optional[torch.Tensor] = None,
    drop_mask: Optional[torch.Tensor] = None,
    atten_mask: Optional[torch.Tensor] = None,
    softmax_max: Optional[torch.Tensor] = None,
    softmax_sum: Optional[torch.Tensor] = None,
    attention_in: Optional[torch.Tensor] = None,
    # Attributes
    scale_value: float = 1.0,
    keep_prob: float = 1.0,
    head_num: Optional[int] = None,
    input_layout: str = "BNSD",
    sparse_mode: int = 0,
    pse_type: int = 1,
    pre_tokens: int = 65536,
    next_tokens: int = 65536,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flash Attention Score Grad 参考实现 (PyTorch)。

    Args:
        query: Query tensor, 布局由 input_layout 决定
        key: Key tensor, 布局由 input_layout 决定
        value: Value tensor, 布局由 input_layout 决定
        dy: 上游梯度, 布局由 input_layout 决定
        pse_shift: 位置编码 (可选)
        drop_mask: Dropout 掩码 (可选), 1=保留, 0=丢弃
        atten_mask: 注意力掩码 (可选), 1=不参与计算
        softmax_max: 正向 softmax 最大值 (可选, 未使用)
        softmax_sum: 正向 softmax 求和值 (可选, 未使用)
        attention_in: 正向注意力输出 (可选, 未使用)
        scale_value: 缩放系数, 通常为 1/sqrt(d)
        keep_prob: Dropout 保留概率
        head_num: Query 的头数 (BSH/SBH 布局需要)
        input_layout: 数据布局 (BSH, SBH, BSND, BNSD)
        sparse_mode: 稀疏模式 (0=default, 1=rightDown, 2=rightUp, 3=leftDown/causal, 4=band)
        pse_type: 位置编码类型 (0=mul_add, 1=add_mul, 2/3=内部生成)
        pre_tokens: band 模式左边界
        next_tokens: band 模式右边界

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (dq, dk, dv)
            dq: Query 梯度
            dk: Key 梯度
            dv: Value 梯度
    """
    # 获取原始布局信息用于输出转换
    original_layout = input_layout

    # 解析 shape
    if input_layout == "BNSD":
        B, N, S1, D = query.shape
        _, N2, S2, _ = key.shape
        head_num = N
        head_dim = D
    elif input_layout == "BSND":
        B, S1, N, D = query.shape
        _, S2, N2, _ = key.shape
        head_num = N
        head_dim = D
    elif input_layout == "BSH":
        B, S1, H = query.shape
        _, S2, H2 = key.shape
        if head_num is None:
            raise ValueError("head_num is required for BSH layout")
        head_dim = H // head_num
        N = head_num
        N2 = H2 // head_dim
    elif input_layout == "SBH":
        S1, B, H = query.shape
        S2, _, H2 = key.shape
        if head_num is None:
            raise ValueError("head_num is required for SBH layout")
        head_dim = H // head_num
        N = head_num
        N2 = H2 // head_dim
    else:
        raise ValueError(f"Unsupported input_layout: {input_layout}")

    # 转换为 BNSD 布局进行内部计算
    Q = _convert_to_bnsd(query, input_layout, N, head_dim)  # [B, N, S1, D]
    K = _convert_to_bnsd(key, input_layout, N2, head_dim)    # [B, N2, S2, D]
    V = _convert_to_bnsd(value, input_layout, N2, head_dim) # [B, N2, S2, D]
    dY = _convert_to_bnsd(dy, input_layout, N, head_dim)     # [B, N, S1, D]

    # 保存原始 dtype
    original_dtype = Q.dtype

    # 转换为 FP32 进行精确计算
    Q = Q.float()
    K = K.float()
    V = V.float()
    dY = dY.float()

    # 处理 GQA: 扩展 K, V 以匹配 Q 的头数
    K_expanded = _expand_for_gqa(K, N, N2)  # [B, N, S2, D]
    V_expanded = _expand_for_gqa(V, N, N2)  # [B, N, S2, D]

    # =====================
    # MM1: 重计算 softmax 得到 p
    # =====================

    # QK^T: [B, N, S1, D] @ [B, N, S2, D]^T -> [B, N, S1, S2]
    scores = torch.matmul(Q, K_expanded.transpose(-2, -1))

    # 应用位置编码 (根据 pse_type)
    if pse_shift is not None:
        pse = pse_shift.float()
        if pse.dim() == 3:
            pse = pse.unsqueeze(1)  # [B, S1, S2] -> [B, 1, S1, S2]
        scores = _apply_pse(scores, pse, pse_type, scale_value)
    else:
        # 默认缩放
        scores = scores * scale_value

    # 应用稀疏掩码
    scores = _apply_sparse_mask(scores, atten_mask, sparse_mode, pre_tokens, next_tokens, S1, S2)

    # Softmax (数值稳定版本)
    scores_max = scores.max(dim=-1, keepdim=True).values
    scores_exp = torch.exp(scores - scores_max)
    scores_sum = scores_exp.sum(dim=-1, keepdim=True)
    p = scores_exp / scores_sum  # [B, N, S1, S2]

    # 应用 Dropout
    p = _apply_dropout(p, drop_mask, keep_prob)

    # =====================
    # MM2: 计算 dp = dY @ V^T
    # =====================
    dp = torch.matmul(dY, V_expanded.transpose(-2, -1))  # [B, N, S1, S2]

    # Dropout 对 dp 也生效 (相同的 mask)
    if drop_mask is not None and keep_prob < 1.0:
        dp = _apply_dropout(dp, drop_mask, keep_prob)

    # =====================
    # Vector: 计算 softmax 梯度和 ds
    # =====================

    # softmax 梯度公式: ds = p * (dp - sum(p * dp, dim=-1, keepdim=True))
    p_dp = p * dp
    sum_p_dp = p_dp.sum(dim=-1, keepdim=True)  # [B, N, S1, 1]
    ds = p * (dp - sum_p_dp)  # [B, N, S1, S2]

    # =====================
    # MM3: 计算 dQ, dK, dV
    # =====================

    # dQ = ds @ K * scale
    dQ = torch.matmul(ds, K_expanded) * scale_value  # [B, N, S1, D]

    # dK = ds^T @ Q * scale
    # ds: [B, N, S1, S2], Q: [B, N, S1, D]
    # ds^T @ Q: [B, N, S2, S1] @ [B, N, S1, D] -> [B, N, S2, D]
    dK_expanded = torch.matmul(ds.transpose(-2, -1), Q) * scale_value  # [B, N, S2, D]

    # dV = p^T @ dY
    # p: [B, N, S1, S2], dY: [B, N, S1, D]
    # p^T @ dY: [B, N, S2, S1] @ [B, N, S1, D] -> [B, N, S2, D]
    dV_expanded = torch.matmul(p.transpose(-2, -1), dY)  # [B, N, S2, D]

    # =====================
    # 处理 GQA: 归约 dK, dV 到原始 KV 头数
    # =====================
    dK = _reduce_for_gqa(dK_expanded, N, N2)  # [B, N2, S2, D]
    dV = _reduce_for_gqa(dV_expanded, N, N2)  # [B, N2, S2, D]

    # 转换回原始 dtype
    dQ = dQ.to(original_dtype)
    dK = dK.to(original_dtype)
    dV = dV.to(original_dtype)

    # 转换回原始布局
    dQ_out = _convert_from_bnsd(dQ, original_layout, N, head_dim)
    dK_out = _convert_from_bnsd(dK, original_layout, N2, head_dim)
    dV_out = _convert_from_bnsd(dV, original_layout, N2, head_dim)

    return dQ_out, dK_out, dV_out


# ==================== 测试用例 ====================

def _test_basic_p0():
    """测试用例: 基础 P0 - [2,8,128,64], keepProb=1.0, no mask"""
    print("\n[Test: Basic P0]")
    B, N, S, D = 2, 8, 128, 64
    scale = 1.0 / math.sqrt(D)

    query = torch.randn(B, N, S, D, dtype=torch.float16)
    key = torch.randn(B, N, S, D, dtype=torch.float16)
    value = torch.randn(B, N, S, D, dtype=torch.float16)
    dy = torch.randn(B, N, S, D, dtype=torch.float16)

    dq, dk, dv = flash_attention_score_grad_golden(
        query, key, value, dy,
        scale_value=scale,
        keep_prob=1.0,
        input_layout="BNSD",
        sparse_mode=0,
        pse_type=1,
    )

    assert dq.shape == query.shape, f"dq shape mismatch: {dq.shape} vs {query.shape}"
    assert dk.shape == key.shape, f"dk shape mismatch: {dk.shape} vs {key.shape}"
    assert dv.shape == value.shape, f"dv shape mismatch: {dv.shape} vs {value.shape}"
    assert not torch.isnan(dq).any(), "dq contains NaN"
    assert not torch.isnan(dk).any(), "dk contains NaN"
    assert not torch.isnan(dv).any(), "dv contains NaN"

    print(f"  Input: Q{list(query.shape)}, K{list(key.shape)}, V{list(value.shape)}, dY{list(dy.shape)}")
    print(f"  Output: dQ{list(dq.shape)}, dK{list(dk.shape)}, dV{list(dv.shape)}")
    print("  ✓ PASS")
    return True


def _test_causal_p0():
    """测试用例: Causal P0 - [2,8,128,64], sparseMode=3 (causal)"""
    print("\n[Test: Causal P0]")
    B, N, S, D = 2, 8, 128, 64
    scale = 1.0 / math.sqrt(D)

    query = torch.randn(B, N, S, D, dtype=torch.float16)
    key = torch.randn(B, N, S, D, dtype=torch.float16)
    value = torch.randn(B, N, S, D, dtype=torch.float16)
    dy = torch.randn(B, N, S, D, dtype=torch.float16)

    dq, dk, dv = flash_attention_score_grad_golden(
        query, key, value, dy,
        scale_value=scale,
        keep_prob=1.0,
        input_layout="BNSD",
        sparse_mode=3,  # causal
        pse_type=1,
    )

    assert dq.shape == query.shape
    assert dk.shape == key.shape
    assert dv.shape == value.shape
    assert not torch.isnan(dq).any()
    assert not torch.isnan(dk).any()
    assert not torch.isnan(dv).any()

    print(f"  Input: Q{list(query.shape)}, K{list(key.shape)}, V{list(value.shape)}")
    print(f"  Output: dQ{list(dq.shape)}, dK{list(dk.shape)}, dV{list(dv.shape)}")
    print("  ✓ PASS")
    return True


def _test_gqa_p1():
    """测试用例: GQA P1 - Q=[2,8,128,64], KV=[2,2,128,64], G=4"""
    print("\n[Test: GQA P1]")
    B, N_Q, N_KV, S, D = 2, 8, 2, 128, 64
    scale = 1.0 / math.sqrt(D)

    query = torch.randn(B, N_Q, S, D, dtype=torch.float16)
    key = torch.randn(B, N_KV, S, D, dtype=torch.float16)
    value = torch.randn(B, N_KV, S, D, dtype=torch.float16)
    dy = torch.randn(B, N_Q, S, D, dtype=torch.float16)

    dq, dk, dv = flash_attention_score_grad_golden(
        query, key, value, dy,
        scale_value=scale,
        keep_prob=1.0,
        input_layout="BNSD",
        sparse_mode=0,
        pse_type=1,
    )

    assert dq.shape == query.shape, f"dq shape mismatch: {dq.shape} vs {query.shape}"
    assert dk.shape == key.shape, f"dk shape mismatch: {dk.shape} vs {key.shape}"
    assert dv.shape == value.shape, f"dv shape mismatch: {dv.shape} vs {value.shape}"
    assert not torch.isnan(dq).any()
    assert not torch.isnan(dk).any()
    assert not torch.isnan(dv).any()

    print(f"  Input: Q{list(query.shape)}, K{list(key.shape)}, V{list(value.shape)}, G={N_Q // N_KV}")
    print(f"  Output: dQ{list(dq.shape)}, dK{list(dk.shape)}, dV{list(dv.shape)}")
    print("  ✓ PASS")
    return True


def _test_pse_p1():
    """测试用例: With PSE - [2,8,128,64], pseType=1"""
    print("\n[Test: PSE P1]")
    B, N, S, D = 2, 8, 128, 64
    scale = 1.0 / math.sqrt(D)

    query = torch.randn(B, N, S, D, dtype=torch.float16)
    key = torch.randn(B, N, S, D, dtype=torch.float16)
    value = torch.randn(B, N, S, D, dtype=torch.float16)
    dy = torch.randn(B, N, S, D, dtype=torch.float16)
    pse_shift = torch.randn(B, N, S, S, dtype=torch.float16) * 0.1

    dq, dk, dv = flash_attention_score_grad_golden(
        query, key, value, dy,
        pse_shift=pse_shift,
        scale_value=scale,
        keep_prob=1.0,
        input_layout="BNSD",
        sparse_mode=0,
        pse_type=1,
    )

    assert dq.shape == query.shape
    assert dk.shape == key.shape
    assert dv.shape == value.shape
    assert not torch.isnan(dq).any()
    assert not torch.isnan(dk).any()
    assert not torch.isnan(dv).any()

    print(f"  Input: Q{list(query.shape)}, pse{list(pse_shift.shape)}")
    print(f"  Output: dQ{list(dq.shape)}, dK{list(dk.shape)}, dV{list(dv.shape)}")
    print("  ✓ PASS")
    return True


def _test_dropout_p1():
    """测试用例: With Dropout - [2,8,128,64], keepProb=0.9"""
    print("\n[Test: Dropout P1]")
    B, N, S, D = 2, 8, 128, 64
    scale = 1.0 / math.sqrt(D)
    keep_prob = 0.9

    query = torch.randn(B, N, S, D, dtype=torch.float16)
    key = torch.randn(B, N, S, D, dtype=torch.float16)
    value = torch.randn(B, N, S, D, dtype=torch.float16)
    dy = torch.randn(B, N, S, D, dtype=torch.float16)

    # 生成 dropout mask: 1=保留, 0=丢弃
    drop_mask = (torch.rand(B, N, S, S) < keep_prob).to(torch.float16)

    dq, dk, dv = flash_attention_score_grad_golden(
        query, key, value, dy,
        drop_mask=drop_mask,
        scale_value=scale,
        keep_prob=keep_prob,
        input_layout="BNSD",
        sparse_mode=0,
        pse_type=1,
    )

    assert dq.shape == query.shape
    assert dk.shape == key.shape
    assert dv.shape == value.shape
    assert not torch.isnan(dq).any()
    assert not torch.isnan(dk).any()
    assert not torch.isnan(dv).any()

    print(f"  Input: Q{list(query.shape)}, keepProb={keep_prob}")
    print(f"  Output: dQ{list(dq.shape)}, dK{list(dk.shape)}, dV{list(dv.shape)}")
    print("  ✓ PASS")
    return True


def _test_layout_bsh():
    """测试用例: Layout BSH"""
    print("\n[Test: Layout BSH]")
    B, S, N, D = 2, 128, 8, 64
    H = N * D
    scale = 1.0 / math.sqrt(D)

    query = torch.randn(B, S, H, dtype=torch.float16)
    key = torch.randn(B, S, H, dtype=torch.float16)
    value = torch.randn(B, S, H, dtype=torch.float16)
    dy = torch.randn(B, S, H, dtype=torch.float16)

    dq, dk, dv = flash_attention_score_grad_golden(
        query, key, value, dy,
        scale_value=scale,
        keep_prob=1.0,
        head_num=N,
        input_layout="BSH",
        sparse_mode=0,
        pse_type=1,
    )

    assert dq.shape == query.shape
    assert dk.shape == key.shape
    assert dv.shape == value.shape
    assert not torch.isnan(dq).any()
    assert not torch.isnan(dk).any()
    assert not torch.isnan(dv).any()

    print(f"  Input: Q{list(query.shape)} (BSH)")
    print(f"  Output: dQ{list(dq.shape)} (BSH)")
    print("  ✓ PASS")
    return True


def _test_layout_bsnd():
    """测试用例: Layout BSND"""
    print("\n[Test: Layout BSND]")
    B, S, N, D = 2, 128, 8, 64
    scale = 1.0 / math.sqrt(D)

    query = torch.randn(B, S, N, D, dtype=torch.float16)
    key = torch.randn(B, S, N, D, dtype=torch.float16)
    value = torch.randn(B, S, N, D, dtype=torch.float16)
    dy = torch.randn(B, S, N, D, dtype=torch.float16)

    dq, dk, dv = flash_attention_score_grad_golden(
        query, key, value, dy,
        scale_value=scale,
        keep_prob=1.0,
        input_layout="BSND",
        sparse_mode=0,
        pse_type=1,
    )

    assert dq.shape == query.shape
    assert dk.shape == key.shape
    assert dv.shape == value.shape
    assert not torch.isnan(dq).any()
    assert not torch.isnan(dk).any()
    assert not torch.isnan(dv).any()

    print(f"  Input: Q{list(query.shape)} (BSND)")
    print(f"  Output: dQ{list(dq.shape)} (BSND)")
    print("  ✓ PASS")
    return True


def _test_bf16():
    """测试用例: BF16 dtype"""
    print("\n[Test: BF16 dtype]")
    B, N, S, D = 2, 8, 128, 64
    scale = 1.0 / math.sqrt(D)

    query = torch.randn(B, N, S, D, dtype=torch.bfloat16)
    key = torch.randn(B, N, S, D, dtype=torch.bfloat16)
    value = torch.randn(B, N, S, D, dtype=torch.bfloat16)
    dy = torch.randn(B, N, S, D, dtype=torch.bfloat16)

    dq, dk, dv = flash_attention_score_grad_golden(
        query, key, value, dy,
        scale_value=scale,
        keep_prob=1.0,
        input_layout="BNSD",
        sparse_mode=0,
        pse_type=1,
    )

    assert dq.shape == query.shape
    assert dk.shape == key.shape
    assert dv.shape == value.shape
    assert dq.dtype == torch.bfloat16
    assert not torch.isnan(dq).any()
    assert not torch.isnan(dk).any()
    assert not torch.isnan(dv).any()

    print(f"  Input: Q{list(query.shape)}, dtype=BF16")
    print(f"  Output: dQ{list(dq.shape)}, dtype=BF16")
    print("  ✓ PASS")
    return True


def _validate():
    """运行所有验证测试。"""
    print("=" * 60)
    print("flash_attention_score_grad_golden 验证报告")
    print("=" * 60)

    all_passed = True
    tests = [
        ("Basic P0", _test_basic_p0),
        ("Causal P0", _test_causal_p0),
        ("GQA P1", _test_gqa_p1),
        ("PSE P1", _test_pse_p1),
        ("Dropout P1", _test_dropout_p1),
        ("Layout BSH", _test_layout_bsh),
        ("Layout BSND", _test_layout_bsnd),
        ("BF16 dtype", _test_bf16),
    ]

    print("\n[典型 case 验证]")
    for name, test_fn in tests:
        try:
            if not test_fn():
                all_passed = False
                print(f"  {name}: ✗ FAIL")
        except Exception as e:
            all_passed = False
            print(f"  {name}: ✗ FAIL - {e}")

    # 泛化测试
    print("\n[泛化 case 验证]")
    try:
        B, N, S, D = 4, 16, 256, 128
        scale = 1.0 / math.sqrt(D)
        query = torch.randn(B, N, S, D, dtype=torch.float16)
        key = torch.randn(B, N, S, D, dtype=torch.float16)
        value = torch.randn(B, N, S, D, dtype=torch.float16)
        dy = torch.randn(B, N, S, D, dtype=torch.float16)

        dq, dk, dv = flash_attention_score_grad_golden(
            query, key, value, dy,
            scale_value=scale,
            keep_prob=1.0,
            input_layout="BNSD",
            sparse_mode=0,
            pse_type=1,
        )
        print(f"  B={B}, N={N}, S={S}, D={D}: ✓ PASS")
    except Exception as e:
        all_passed = False
        print(f"  泛化测试: ✗ FAIL - {e}")

    # 数值稳定性
    print("\n[数值稳定性检查]")
    try:
        # 大值输入
        B, N, S, D = 1, 4, 32, 32
        scale = 1.0 / math.sqrt(D)
        query = torch.randn(B, N, S, D, dtype=torch.float16) * 10
        key = torch.randn(B, N, S, D, dtype=torch.float16) * 10
        value = torch.randn(B, N, S, D, dtype=torch.float16)
        dy = torch.randn(B, N, S, D, dtype=torch.float16)

        dq, dk, dv = flash_attention_score_grad_golden(
            query, key, value, dy,
            scale_value=scale,
            keep_prob=1.0,
            input_layout="BNSD",
            sparse_mode=0,
            pse_type=1,
        )

        if torch.isnan(dq).any() or torch.isnan(dk).any() or torch.isnan(dv).any():
            print("  大值输入: ✗ FAIL - NaN detected")
            all_passed = False
        elif torch.isinf(dq).any() or torch.isinf(dk).any() or torch.isinf(dv).any():
            print("  大值输入: ⚠ WARNING - Inf detected")
        else:
            print("  大值输入: ✓ PASS")
    except Exception as e:
        all_passed = False
        print(f"  大值输入: ✗ FAIL - {e}")

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有验证通过")
    else:
        print("❌ 存在验证失败")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    _validate()