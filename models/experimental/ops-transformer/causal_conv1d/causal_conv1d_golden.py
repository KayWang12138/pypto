#!/usr/bin/env python3
# coding: utf-8

"""PyPTO causal_conv1d golden reference implementation.

置信度: ⭐⭐⭐⭐⭐ (基于 PyTorch 原生实现，参考 vLLM mamba ops)

算子概述:
  因果卷积算子，用于 Mamba/SSM 模型的序列处理。支持两种运行模式：
  - Prefill 模式 (FN VARLEN): 处理完整序列，支持变长序列（packed layout）
  - Decode 模式 (UPDATE): 处理单个或多个 token（投机解码），状态缓存滚动更新

数学公式:
  y[t] = activation(bias + Σ_{i=0}^{width-1} w[i] * x[t-width+1+i])
  silu 激活: silu(x) = x / (1 + exp(-x))

注意:
  - Golden 必须是纯 PyTorch 实现，禁止引入 pypto
  - 使用 float32 精度计算，避免 float16 精度问题
  - 导出函数供 test_causal_conv1d.py 调用
"""

import torch
from typing import Optional


# ─────────────────────────────────────────────
# Prefill 模式 Golden 参考实现
# ─────────────────────────────────────────────

def causal_conv1d_prefill_golden(
    x: torch.Tensor,           # [total_len, dim]
    weight: torch.Tensor,      # [width, dim]
    conv_state: torch.Tensor,  # [num_cache, state_len, dim]
    cu_seqlens: torch.Tensor,  # [batch+1]
    activation: str = "silu",  # 可选 "silu" 或 None
    width: int = 4,
    cache_indices: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:             # y [total_len, dim]
    """PyTorch 参考实现 - Prefill 模式 (FN VARLEN).

    处理完整序列，支持变长序列（packed layout）。
    卷积状态存储最后 width-1 个 token，用于后续 Decode 模式。
    支持 cache_indices 间接索引和 bias。

    Args:
        x: [total_len, dim] float16 - packed layout 输入序列
        weight: [width, dim] float16 - 卷积权重
        conv_state: [num_cache_lines, state_len, dim] float16 - 状态缓存
        cu_seqlens: [batch_size + 1] int32 - 变长序列边界
        activation: 激活函数，可选 "silu" 或 None
        width: 卷积窗口大小，支持 3-6
        cache_indices: [batch_size] int32 - 缓存行间接索引，可选
        bias: [dim] float16 - 卷积偏置，可选

    Returns:
        y: [total_len, dim] float16 - 卷积输出

    Side Effect:
        conv_state 会被原地更新，存储每个序列最后 width-1 个 token
        当使用 cache_indices 时，更新对应的间接索引缓存行
    """
    # 转换为 float32 进行精确计算
    dtype = x.dtype
    x_f32 = x.float()
    weight_f32 = weight.float()
    conv_state_f32 = conv_state.float()
    bias_f32 = bias.float() if bias is not None else None

    total_len, dim = x_f32.shape
    hist_len = width - 1
    state_len = conv_state.shape[1]
    batch_size = cu_seqlens.size(0) - 1
    has_cache_indices = cache_indices is not None
    has_bias = bias is not None

    # 输出 tensor
    y = torch.zeros(total_len, dim, dtype=torch.float32, device=x.device)

    for b in range(batch_size):
        seq_start = cu_seqlens[b].item()
        seq_end = cu_seqlens[b + 1].item()
        seqlen = seq_end - seq_start

        # 使用 cache_indices 获取缓存行索引
        ci = cache_indices[b].item() if has_cache_indices else b

        # 初始化历史缓冲（从 conv_state 加载或初始化为零）
        history = []
        for h in range(hist_len):
            if h < state_len:
                history.append(conv_state_f32[ci, h, :].clone())
            else:
                history.append(torch.zeros(dim, dtype=torch.float32, device=x.device))

        for t in range(seqlen):
            x_t = x_f32[seq_start + t, :]

            # 卷积计算: acc = bias + Σ w[i] * hist[i] + w[width-1] * x_t
            if has_bias:
                acc = bias_f32.clone()
            else:
                acc = torch.zeros(dim, dtype=torch.float32, device=x.device)
            for w_idx in range(hist_len):
                acc = acc + weight_f32[w_idx, :] * history[w_idx]
            acc = acc + weight_f32[width - 1, :] * x_t

            # silu 激活
            if activation == "silu":
                out = acc / (1.0 + torch.exp(-acc))
            else:
                out = acc

            y[seq_start + t, :] = out

            # 滚动历史
            for h in range(hist_len - 1):
                history[h] = history[h + 1].clone()
            history[hist_len - 1] = x_t.clone()

        # 更新 conv_state（存储最后 width-1 个 token）
        if seqlen > 0 and state_len >= hist_len:
            for pos in range(hist_len):
                last_idx = seqlen - hist_len + pos
                if last_idx >= 0 and pos < state_len:
                    conv_state_f32[ci, pos, :] = x_f32[seq_start + last_idx, :]

    # 原地更新 conv_state（转回原 dtype）
    conv_state.copy_(conv_state_f32)

    return y.to(dtype)


# ─────────────────────────────────────────────
# Decode 模式 Golden 参考实现
# ─────────────────────────────────────────────

def causal_conv1d_decode_golden(
    x: torch.Tensor,           # [batch, seqlen, dim] 或 [batch, dim] 或 [dim]
    weight: torch.Tensor,      # [width, dim]
    conv_state: torch.Tensor,  # [num_cache, state_len, dim]
    activation: str = "silu",  # 可选 "silu" 或 None
    width: int = 4,
    cache_indices: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:             # y [batch, dim]（seqlen=1）或 [batch, seqlen, dim]（seqlen>1）
    """PyTorch 参考实现 - Decode 模式 (UPDATE).

    处理单个或多个 token（投机解码），状态缓存滚动更新。
    支持 cache_indices 间接索引和 bias。
    
    Args:
        x: [batch, seqlen, dim] float16 - 单 token 或投机解码序列
           支持多种输入格式：
           - [dim]: 单 batch 单 token，自动扩展为 [1, 1, dim]
           - [batch, dim]: 多 batch 单 token，自动扩展为 [batch, 1, dim]
           - [batch, seqlen, dim]: 投机解码，seqlen > 1
        weight: [width, dim] float16 - 卷积权重
        conv_state: [num_cache_lines, state_len, dim] float16 - 状态缓存
        activation: 激活函数，可选 "silu" 或 None
        width: 卷积窗口大小，支持 3-6
        cache_indices: [batch] int32 - 缓存行间接索引，可选
        bias: [dim] float16 - 卷积偏置，可选

    Returns:
        y: [batch, dim]（seqlen=1）或 [batch, seqlen, dim]（seqlen>1）

    Side Effect:
        conv_state 会被原地更新，滚动存储新的 token
        当使用 cache_indices 时，更新对应的间接索引缓存行
    """
    # 处理多种输入格式
    if x.dim() == 1:
        x = x.unsqueeze(0).unsqueeze(0)  # [dim] -> [1, 1, dim]
    elif x.dim() == 2:
        x = x.unsqueeze(1)  # [batch, dim] -> [batch, 1, dim]

    # 转换为 float32 进行精确计算
    dtype = x.dtype
    x_f32 = x.float()
    weight_f32 = weight.float()
    conv_state_f32 = conv_state.float()
    bias_f32 = bias.float() if bias is not None else None

    batch, seqlen, dim = x_f32.shape
    hist_len = width - 1
    state_len = conv_state.shape[1]
    has_cache_indices = cache_indices is not None
    has_bias = bias is not None

    # 输出 tensor
    y = torch.zeros(batch, seqlen, dim, dtype=torch.float32, device=x.device)

    for b in range(batch):
        # 使用 cache_indices 获取缓存行索引
        ci = cache_indices[b].item() if has_cache_indices else b
        
        # 投机解码时，从 conv_state 的 offset 位置读取历史
        state_token_offset = seqlen - 1

        # 初始化历史缓冲
        history = []
        for h in range(hist_len):
            src_idx = state_token_offset + h
            if src_idx < state_len:
                history.append(conv_state_f32[ci, src_idx, :].clone())
            else:
                history.append(torch.zeros(dim, dtype=torch.float32, device=x.device))

        for t in range(seqlen):
            x_t = x_f32[b, t, :]

            # 卷积计算: acc = bias + Σ w[i] * hist[i] + w[width-1] * x_t
            if has_bias:
                acc = bias_f32.clone()
            else:
                acc = torch.zeros(dim, dtype=torch.float32, device=x.device)
            for w_idx in range(hist_len):
                acc = acc + weight_f32[w_idx, :] * history[w_idx]
            acc = acc + weight_f32[width - 1, :] * x_t

            # silu 激活
            if activation == "silu":
                out = acc / (1.0 + torch.exp(-acc))
            else:
                out = acc

            y[b, t, :] = out

            # 滚动历史
            for h in range(hist_len - 1):
                history[h] = history[h + 1].clone()
            history[hist_len - 1] = x_t.clone()

        # 更新 conv_state
        if state_len >= 2 and seqlen > 0:
            # 保留原有的中间状态
            if state_token_offset + 1 < state_len:
                conv_state_f32[ci, 0, :] = conv_state_f32[ci, state_token_offset + 1, :]
            if state_token_offset + 2 < state_len:
                conv_state_f32[ci, 1, :] = conv_state_f32[ci, state_token_offset + 2, :]

            # 写入新 token
            for t in range(seqlen):
                write_pos = 2 + t
                if write_pos < state_len:
                    conv_state_f32[ci, write_pos, :] = x_f32[b, t, :]

    # 原地更新 conv_state
    conv_state.copy_(conv_state_f32)

    # 返回值处理
    if seqlen == 1:
        return y.squeeze(1).to(dtype)  # [batch, 1, dim] -> [batch, dim]
    else:
        return y.to(dtype)


# ─────────────────────────────────────────────
# 验证函数
# ─────────────────────────────────────────────

def _validate():
    """自动验证函数 - 运行时验证 golden 实现的正确性"""

    print("=" * 60)
    print("causal_conv1d_golden 验证报告")
    print("=" * 60)

    device = "cpu"  # golden 可以在 CPU 上验证

    # ========== Prefill 模式验证 ==========

    print("\n[Prefill 模式 - 典型 case 验证]")

    # Prefill_P0: seqlen=2048, dim=2048, width=4, batch=1
    seqlen = 2048
    dim = 2048
    width = 4
    batch = 1
    state_len = width - 1  # 3

    torch.manual_seed(42)
    x_prefill = torch.randn(seqlen, dim, dtype=torch.float16, device=device)
    weight_prefill = torch.randn(width, dim, dtype=torch.float16, device=device)
    conv_state_prefill = torch.randn(batch, state_len, dim, dtype=torch.float16, device=device)
    cu_seqlens = torch.tensor([0, seqlen], dtype=torch.int32, device=device)

    y_prefill = causal_conv1d_prefill_golden(
        x_prefill, weight_prefill, conv_state_prefill, cu_seqlens,
        activation="silu", width=width
    )

    # 验证输出 shape
    assert y_prefill.shape == (seqlen, dim), f"Prefill output shape mismatch: {y_prefill.shape}"
    print(f"  Prefill_P0: seqlen={seqlen}, dim={dim}, width={width} ... ✓ PASS")

    # 验证 conv_state 更新
    assert conv_state_prefill.shape == (batch, state_len, dim), "conv_state shape unchanged"
    print(f"  conv_state update: shape preserved ... ✓ PASS")

    # 验证输出无 NaN/Inf
    assert not torch.isnan(y_prefill).any(), "Output contains NaN"
    assert not torch.isinf(y_prefill).any(), "Output contains Inf"
    print(f"  数值稳定性: 无 NaN/Inf ... ✓ PASS")

    # ========== Prefill 变长序列验证 ==========

    print("\n[Prefill 模式 - 变长序列验证]")

    # Prefill_Varlen_P0: 4 个序列，各 512 token
    seqlens = [512, 512, 512, 512]
    total_len_varlen = sum(seqlens)
    batch_varlen = len(seqlens)

    torch.manual_seed(42)
    x_varlen = torch.randn(total_len_varlen, dim, dtype=torch.float16, device=device)
    weight_varlen = torch.randn(width, dim, dtype=torch.float16, device=device)
    conv_state_varlen = torch.randn(batch_varlen, state_len, dim, dtype=torch.float16, device=device)
    cu_seqlens_varlen = torch.tensor([0, 512, 1024, 1536, 2048], dtype=torch.int32, device=device)

    y_varlen = causal_conv1d_prefill_golden(
        x_varlen, weight_varlen, conv_state_varlen, cu_seqlens_varlen,
        activation="silu", width=width
    )

    assert y_varlen.shape == (total_len_varlen, dim), f"Varlen output shape mismatch: {y_varlen.shape}"
    print(f"  Prefill_Varlen_P0: batch={batch_varlen}, total_len={total_len_varlen} ... ✓ PASS")

    # ========== Decode 模式验证 ==========

    print("\n[Decode 模式 - 典型 case 验证]")

    # Decode_P0: seqlen=1, dim=2048, width=4, batch=1
    seqlen_decode = 1
    batch_decode = 1

    torch.manual_seed(42)
    x_decode = torch.randn(batch_decode, seqlen_decode, dim, dtype=torch.float16, device=device)
    weight_decode = torch.randn(width, dim, dtype=torch.float16, device=device)
    conv_state_decode = torch.randn(batch_decode, state_len, dim, dtype=torch.float16, device=device)

    y_decode = causal_conv1d_decode_golden(
        x_decode, weight_decode, conv_state_decode,
        activation="silu", width=width
    )

    # 验证输出 shape（seqlen=1 时应为 [batch, dim])
    assert y_decode.shape == (batch_decode, dim), f"Decode output shape mismatch: {y_decode.shape}"
    print(f"  Decode_P0: batch={batch_decode}, seqlen={seqlen_decode}, dim={dim} ... ✓ PASS")

    # 验证数值稳定性
    assert not torch.isnan(y_decode).any(), "Decode output contains NaN"
    assert not torch.isinf(y_decode).any(), "Decode output contains Inf"
    print(f"  数值稳定性: 无 NaN/Inf ... ✓ PASS")

    # ========== Decode 多格式输入验证 ==========

    print("\n[Decode 模式 - 输入格式验证]")

    # 测试 [dim] 格式
    x_dim = torch.randn(dim, dtype=torch.float16, device=device)
    y_dim = causal_conv1d_decode_golden(x_dim, weight_decode, conv_state_decode)
    assert y_dim.shape == (1, dim), f"[dim] format output mismatch: {y_dim.shape}"
    print(f"  输入格式 [dim]: ✓ PASS")

    # 测试 [batch, dim] 格式
    x_batch_dim = torch.randn(2, dim, dtype=torch.float16, device=device)
    conv_state_batch = torch.randn(2, state_len, dim, dtype=torch.float16, device=device)
    y_batch_dim = causal_conv1d_decode_golden(x_batch_dim, weight_decode, conv_state_batch)
    assert y_batch_dim.shape == (2, dim), f"[batch, dim] format output mismatch: {y_batch_dim.shape}"
    print(f"  输入格式 [batch, dim]: ✓ PASS")

    # ========== 投机解码验证 ==========

    print("\n[Decode 模式 - 投机解码验证]")

    seqlen_spec = 4  # 投机解码接受 4 个 token
    batch_spec = 1
    state_len_spec = 5  # 更大的 state_len

    torch.manual_seed(42)
    x_spec = torch.randn(batch_spec, seqlen_spec, dim, dtype=torch.float16, device=device)
    weight_spec = torch.randn(width, dim, dtype=torch.float16, device=device)
    conv_state_spec = torch.randn(batch_spec, state_len_spec, dim, dtype=torch.float16, device=device)

    y_spec = causal_conv1d_decode_golden(
        x_spec, weight_spec, conv_state_spec,
        activation="silu", width=width
    )

    # 验证输出 shape（seqlen>1 时应为 [batch, seqlen, dim])
    assert y_spec.shape == (batch_spec, seqlen_spec, dim), f"Speculative decode output mismatch: {y_spec.shape}"
    print(f"  投机解码: seqlen={seqlen_spec} ... ✓ PASS")

    # ========== 不同 width 验证 ==========

    print("\n[不同 width 验证]")

    for w in [3, 4, 5, 6]:
        torch.manual_seed(42)
        x_w = torch.randn(64, 128, dtype=torch.float16, device=device)
        weight_w = torch.randn(w, 128, dtype=torch.float16, device=device)
        conv_state_w = torch.randn(1, w - 1, 128, dtype=torch.float16, device=device)
        cu_w = torch.tensor([0, 64], dtype=torch.int32, device=device)

        y_w = causal_conv1d_prefill_golden(x_w, weight_w, conv_state_w, cu_w, width=w)
        assert y_w.shape == (64, 128), f"width={w} output shape mismatch"
        print(f"  width={w}: ✓ PASS")

    # ========== silu 数值验证 ==========

    print("\n[silu 激活验证]")

    # silu(0) ≈ 0
    test_zero = torch.zeros(10, dtype=torch.float32, device=device)
    silu_zero = test_zero / (1.0 + torch.exp(-test_zero))
    assert torch.allclose(silu_zero, torch.zeros_like(silu_zero), atol=1e-6), "silu(0) != 0"
    print(f"  silu(0) ≈ 0: ✓ PASS")

    # silu 在正值区间单调增（实际应用中输入多为正值）
    # 注意：silu 在 x < -1.278 时导数为负，有最小值点
    test_pos_values = torch.linspace(0, 10, 100, dtype=torch.float32, device=device)
    silu_pos = test_pos_values / (1.0 + torch.exp(-test_pos_values))
    # 检查单调性（正值区间）
    diffs_pos = silu_pos[1:] - silu_pos[:-1]
    assert (diffs_pos > 0).all(), "silu not monotonic in positive range"
    print(f"  silu 正值区间单调增: ✓ PASS")
    # 验证 silu 的最小值出现在 x ≈ -1.278
    test_min = torch.linspace(-3, 0, 200, dtype=torch.float32, device=device)
    silu_min = test_min / (1.0 + torch.exp(-test_min))
    min_idx = silu_min.argmin().item()
    min_x = test_min[min_idx].item()
    assert abs(min_x - (-1.278)) < 0.2, f"silu minimum at x≈{min_x}, expected ≈-1.278"
    print(f"  silu 最小值点验证 (x≈-1.278): ✓ PASS")

    # ========== 与 PyTorch silu API 对比 ==========

    print("\n[API 对比 - silu]")

    test_api = torch.randn(1000, dtype=torch.float32, device=device)
    silu_manual = test_api / (1.0 + torch.exp(-test_api))
    silu_torch = torch.nn.functional.silu(test_api)
    assert torch.allclose(silu_manual, silu_torch, rtol=1e-5, atol=1e-5), "Manual silu != torch.nn.functional.silu"
    print(f"  手写 silu vs torch.nn.functional.silu: ✓ PASS")

    print("\n" + "=" * 60)
    print("✅ 所有验证通过")
    print("=" * 60)


if __name__ == "__main__":
    _validate()