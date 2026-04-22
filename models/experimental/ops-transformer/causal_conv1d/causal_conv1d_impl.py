#!/usr/bin/env python3
# coding: utf-8

"""PyPTO causal_conv1d kernel implementation - v9.0.

核心特性：
1. 支持动态 width (3, 4, 5, 6) - 为每个 width 创建专门的 kernel
2. cache_indices 在 wrapper 层面处理（PyPTO kernel 内不支持动态间接索引）
3. 使用切片赋值替代 assemble 输出
4. 分离 offset 计算到变量，避免表达式作为 view 的 offset
5. 使用 valid_shape 处理动态边界

遗留问题说明：
- 动态 width: 已解决，通过为不同 width 创建专门的 kernel 函数
- cache_indices: PyPTO 不支持在 kernel 内从 int32 tensor 读取值作为 FP16 tensor 的动态索引，
  因此在 wrapper 层面处理 cache_indices（与 TileLang 原始实现的 kernel 内支持不同）

参考：
- custom/chunk_gated_delta_rule/chunk_gated_delta_rule_impl.py
- docs/api/operation/pypto-view.md
- causal_conv1d/causal_conv1d.py (TileLang 原始实现)
"""

import pypto
import torch

F_NEGA_1 = -1.0
F_1 = 1.0

BLOCK_M = 64  # 每个 block 处理 64 个 token
BLOCK_D = 512  # 每个 block 处理 512 维


def silu_activation(acc: pypto.tensor) -> pypto.tensor:
    """手动实现 silu: x / (1 + exp(-x))
    
    使用更稳定的计算方式。
    """
    neg_acc = pypto.mul(acc, F_NEGA_1)
    exp_neg = pypto.exp(neg_acc)
    denom = pypto.add(exp_neg, F_1)
    return pypto.div(acc, denom)


# ============================================================================
# Prefill Kernels - 每个宽度单独的 kernel
# ============================================================================

def _make_prefill_kernel(width: int):
    """生成指定 width 的 Prefill kernel 函数。
    
    Args:
        width: 卷积窗口大小 (3, 4, 5, 6)
    
    Returns:
        编译后的 kernel 函数
    """
    hist_len = width - 1
    
    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU, "device_sched_mode": 1})
    def causal_conv1d_prefill_kernel_w(
        x_padded: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),
        weight: pypto.Tensor([width, pypto.DYNAMIC], pypto.DT_FP32),
        bias: pypto.Tensor([1, pypto.DYNAMIC], pypto.DT_FP32),
        cu_seqlens: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
        y: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),
        activation: str = "silu",
    ):
        """Prefill kernel for causal_conv1d with specific width.
        
        Uses pypto.loop for all dynamic loops and slice assignment for output.
        """
        dim = x_padded.shape[1]
        batch_size = cu_seqlens.shape[0] - 1
        
        dim_num = (dim + BLOCK_D - 1) // BLOCK_D
        
        pypto.set_vec_tile_shapes(1, BLOCK_D)
        
        # Batch Loop
        for b_idx in pypto.loop(batch_size, name="batch", idx_name="b_idx", unroll_list=[1]):
            seq_start = cu_seqlens[b_idx]
            seq_end = cu_seqlens[b_idx + 1]
            seqlen = seq_end - seq_start
            
            block_num = (seqlen + BLOCK_M - 1) // BLOCK_M
            
            # Dim Block Loop
            for d_idx in pypto.loop(dim_num, name="dim_block", idx_name="d_idx", unroll_list=[1]):
                d_offset = d_idx * BLOCK_D
                
                actual_block_d = (dim - d_offset).min(BLOCK_D)
                
                # 加载所有权重
                weights = []
                for w_idx in range(width):
                    w = pypto.view(weight, [1, BLOCK_D], [w_idx, d_offset], valid_shape=[1, actual_block_d])
                    weights.append(w)
                
                # 加载 bias
                bias_ub = pypto.view(bias, [1, BLOCK_D], [0, d_offset], valid_shape=[1, actual_block_d])
                
                # Seq Block Loop
                for block_idx in pypto.loop(block_num, name="seq_block", idx_name="block_idx", unroll_list=[1]):
                    block_start = block_idx * BLOCK_M
                    actual_block_size = (seqlen - block_start).min(BLOCK_M)
                    
                    # Token in Block Loop
                    for t_in_block in pypto.loop(actual_block_size, name="token_in_block", idx_name="t_in_block", unroll_list=[8, 4, 2, 1]):
                        t_idx = block_start + t_in_block
                        
                        t_global_padded = seq_start + hist_len + t_idx
                        
                        # 加载当前 token
                        x_cur_fp16 = pypto.view(x_padded, [1, BLOCK_D], [t_global_padded, d_offset], valid_shape=[1, actual_block_d])
                        x_cur_fp32 = pypto.cast(x_cur_fp16, pypto.DT_FP32)
                        
                        # 加载历史并计算卷积
                        acc = bias_ub
                        for h_idx in range(hist_len):
                            hist_t_offset = t_global_padded - hist_len + h_idx
                            hist_fp16 = pypto.view(x_padded, [1, BLOCK_D], [hist_t_offset, d_offset], valid_shape=[1, actual_block_d])
                            hist_fp32 = pypto.cast(hist_fp16, pypto.DT_FP32)
                            prod = pypto.mul(weights[h_idx], hist_fp32)
                            acc = pypto.add(acc, prod)
                        
                        # 当前 token 权重
                        prod_cur = pypto.mul(weights[width - 1], x_cur_fp32)
                        acc = pypto.add(acc, prod_cur)
                        
                        # silu 激活
                        if activation == "silu":
                            out_fp32 = silu_activation(acc)
                        else:
                            out_fp32 = acc
                        
                        out_fp16 = pypto.cast(out_fp32, pypto.DT_FP16)
                        
                        # 切片赋值输出
                        y_t_offset = seq_start + t_idx
                        y[y_t_offset : y_t_offset + 1, d_offset : d_offset + BLOCK_D] = out_fp16
    
    return causal_conv1d_prefill_kernel_w


# 创建各 width 的 Prefill kernel
_prefill_kernel_w3 = _make_prefill_kernel(3)
_prefill_kernel_w4 = _make_prefill_kernel(4)
_prefill_kernel_w5 = _make_prefill_kernel(5)
_prefill_kernel_w6 = _make_prefill_kernel(6)

_PREFILL_KERNELS = {
    3: _prefill_kernel_w3,
    4: _prefill_kernel_w4,
    5: _prefill_kernel_w5,
    6: _prefill_kernel_w6,
}


def causal_conv1d_prefill_wrapper(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    activation: str = "silu",
    width: int = 4,
    cache_indices: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Prefill wrapper
    
    预先扩展 x tensor，填充 conv_state 的历史数据
    支持 cache_indices 间接索引和 bias
    
    注意：cache_indices 在 wrapper 层面处理，不在 kernel 内处理
    """
    if width not in _PREFILL_KERNELS:
        raise ValueError(f"Unsupported width: {width}. Supported values: 3, 4, 5, 6")
    
    hist_len = width - 1
    batch_size = cu_seqlens.shape[0] - 1
    total_len, dim = x.shape
    has_cache_indices = cache_indices is not None
    has_bias = bias is not None
    
    # 为每个 batch 在其数据之前预留 hist_len 空间
    cu_seqlens_padded = []
    offset = 0
    for b in range(batch_size):
        cu_seqlens_padded.append(offset)
        seq_start = cu_seqlens[b].item()
        seq_end = cu_seqlens[b + 1].item()
        seqlen = seq_end - seq_start
        offset += hist_len + seqlen
    cu_seqlens_padded.append(offset)
    
    # 创建 x_padded
    x_padded = torch.zeros(offset, dim, dtype=x.dtype, device=x.device)
    
    # 填充数据（使用 cache_indices 间接索引）
    for b in range(batch_size):
        seq_start = cu_seqlens[b].item()
        seq_end = cu_seqlens[b + 1].item()
        seqlen = seq_end - seq_start
        padded_start = cu_seqlens_padded[b]
        
        # 填充历史
        if has_cache_indices:
            ci = cache_indices[b].item()
            # 检查 conv_state 的第二维是否足够存储历史
            if hist_len <= conv_state.shape[1]:
                x_padded[padded_start:padded_start + hist_len] = conv_state[ci, :hist_len, :]
            else:
                # 如果 conv_state 不够大，只取可用的部分
                available = min(hist_len, conv_state.shape[1])
                x_padded[padded_start:padded_start + available] = conv_state[ci, :available, :]
        else:
            if hist_len <= conv_state.shape[1]:
                x_padded[padded_start:padded_start + hist_len] = conv_state[b, :hist_len, :]
            else:
                available = min(hist_len, conv_state.shape[1])
                x_padded[padded_start:padded_start + available] = conv_state[b, :available, :]
        
        # 填充数据
        x_padded[padded_start + hist_len:padded_start + hist_len + seqlen] = x[seq_start:seq_end]
    
    # 创建 cu_seqlens_padded tensor
    cu_seqlens_padded_tensor = torch.tensor(cu_seqlens_padded, dtype=torch.int32, device=x.device)
    
    # 创建 bias tensor
    if not has_bias:
        bias_tensor = torch.zeros(dim, dtype=torch.float32, device=x.device)
    else:
        bias_tensor = bias.float()
    bias_tensor_2d = bias_tensor.unsqueeze(0)
    
    y = torch.zeros_like(x)
    weight_fp32 = weight.float()
    
    # 选择对应 width 的 kernel
    kernel = _PREFILL_KERNELS[width]
    kernel(x_padded, weight_fp32, bias_tensor_2d, cu_seqlens_padded_tensor, y, activation)
    
    # 更新 conv_state（存储最后 width-1 个 token，供后续 Decode 使用）
    state_len = conv_state.shape[1]
    for b in range(batch_size):
        seq_start = cu_seqlens[b].item()
        seq_end = cu_seqlens[b + 1].item()
        seqlen = seq_end - seq_start
        ci = cache_indices[b].item() if has_cache_indices else b
        
        if seqlen > 0:
            for pos in range(hist_len):
                last_idx = seqlen - hist_len + pos
                if last_idx >= 0 and pos < state_len:
                    conv_state[ci, pos, :] = x[seq_start + last_idx, :]
    
    return y


# ============================================================================
# Decode Kernels - 每个宽度单独的 kernel
# ============================================================================

def _make_decode_kernel(width: int):
    """生成指定 width 的 Decode kernel 函数。
    
    Args:
        width: 卷积窗口大小 (3, 4, 5, 6)
    
    Returns:
        编译后的 kernel 函数
    """
    hist_len = width - 1
    
    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU, "device_sched_mode": 1})
    def causal_conv1d_decode_kernel_w(
        x_padded: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),
        weight: pypto.Tensor([width, pypto.DYNAMIC], pypto.DT_FP32),
        bias: pypto.Tensor([1, pypto.DYNAMIC], pypto.DT_FP32),
        y: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),
        activation: str = "silu",
    ):
        """Decode kernel for causal_conv1d with specific width.
        
        Uses pypto.loop for all dynamic loops and slice assignment for output.
        """
        dim = x_padded.shape[1]
        seqlen = y.shape[0]
        
        dim_num = (dim + BLOCK_D - 1) // BLOCK_D
        
        pypto.set_vec_tile_shapes(1, BLOCK_D)
        
        # Dim Block Loop
        for d_idx in pypto.loop(dim_num, name="dim_block", idx_name="d_idx", unroll_list=[1]):
            d_offset = d_idx * BLOCK_D
            
            actual_block_d = (dim - d_offset).min(BLOCK_D)
            
            # 加载所有权重
            weights = []
            for w_idx in range(width):
                w = pypto.view(weight, [1, BLOCK_D], [w_idx, d_offset], valid_shape=[1, actual_block_d])
                weights.append(w)
            
            # 加载 bias
            bias_ub = pypto.view(bias, [1, BLOCK_D], [0, d_offset], valid_shape=[1, actual_block_d])
            
            # Seqlen Loop
            for t_idx in pypto.loop(seqlen, name="seqlen", idx_name="t_idx", unroll_list=[1]):
                t_global_padded = hist_len + t_idx
                
                # 加载当前 token
                x_cur_fp16 = pypto.view(x_padded, [1, BLOCK_D], [t_global_padded, d_offset], valid_shape=[1, actual_block_d])
                x_cur_fp32 = pypto.cast(x_cur_fp16, pypto.DT_FP32)
                
                # 加载历史并计算卷积
                acc = bias_ub
                for h_idx in range(hist_len):
                    hist_t_offset = t_global_padded - hist_len + h_idx
                    hist_fp16 = pypto.view(x_padded, [1, BLOCK_D], [hist_t_offset, d_offset], valid_shape=[1, actual_block_d])
                    hist_fp32 = pypto.cast(hist_fp16, pypto.DT_FP32)
                    prod = pypto.mul(weights[h_idx], hist_fp32)
                    acc = pypto.add(acc, prod)
                
                # 当前 token 权重
                prod_cur = pypto.mul(weights[width - 1], x_cur_fp32)
                acc = pypto.add(acc, prod_cur)
                
                # silu 激活
                if activation == "silu":
                    out_fp32 = silu_activation(acc)
                else:
                    out_fp32 = acc
                
                out_fp16 = pypto.cast(out_fp32, pypto.DT_FP16)
                
                # 切片赋值输出
                y[t_idx : t_idx + 1, d_offset : d_offset + BLOCK_D] = out_fp16
    
    return causal_conv1d_decode_kernel_w


# 创建各 width 的 Decode kernel
_decode_kernel_w3 = _make_decode_kernel(3)
_decode_kernel_w4 = _make_decode_kernel(4)
_decode_kernel_w5 = _make_decode_kernel(5)
_decode_kernel_w6 = _make_decode_kernel(6)

_DECODE_KERNELS = {
    3: _decode_kernel_w3,
    4: _decode_kernel_w4,
    5: _decode_kernel_w5,
    6: _decode_kernel_w6,
}


def causal_conv1d_decode_wrapper(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    activation: str = "silu",
    width: int = 4,
    cache_indices: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode wrapper
    
    预先扩展 x tensor，填充 conv_state 的历史数据
    支持 cache_indices 间接索引和 bias
    
    注意：cache_indices 在 wrapper 层面处理，不在 kernel 内处理
    """
    if width not in _DECODE_KERNELS:
        raise ValueError(f"Unsupported width: {width}. Supported values: 3, 4, 5, 6")
    
    hist_len = width - 1
    has_cache_indices = cache_indices is not None
    has_bias = bias is not None
    
    # 处理多种输入格式
    if x.dim() == 1:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 2:
        x = x.unsqueeze(1)
    
    batch, seqlen, dim = x.shape
    
    # 创建 bias tensor
    if not has_bias:
        bias_tensor = torch.zeros(dim, dtype=torch.float32, device=x.device)
    else:
        bias_tensor = bias.float()
    bias_tensor_2d = bias_tensor.unsqueeze(0)
    
    y_list = []
    for b in range(batch):
        # 为每个 batch 创建扩展后的 x
        x_b = x[b]  # [seqlen, dim]
        x_padded_b = torch.zeros(seqlen + hist_len, dim, dtype=x.dtype, device=x.device)
        
        # 使用 cache_indices 间接索引获取缓存行
        if has_cache_indices:
            ci = cache_indices[b].item()
        else:
            ci = b
        
        # 填充历史（投机解码时从 offset 位置读取）
        state_token_offset = seqlen - 1
        state_len = conv_state.shape[1]
        
        # 计算可用的历史长度
        available_hist_from_state = min(hist_len, state_len - state_token_offset)
        if available_hist_from_state > 0:
            start_pos = state_token_offset
            end_pos = state_token_offset + available_hist_from_state
            x_padded_b[:available_hist_from_state] = conv_state[ci, start_pos:end_pos, :]
        
        # 填充实际数据
        x_padded_b[hist_len:hist_len + seqlen] = x_b
        
        # 创建输出
        y_b = torch.zeros(seqlen, dim, dtype=x.dtype, device=x.device)
        weight_fp32 = weight.float()
        
        # 选择对应 width 的 kernel
        kernel = _DECODE_KERNELS[width]
        kernel(x_padded_b, weight_fp32, bias_tensor_2d, y_b, activation)
        
        y_list.append(y_b)
        
        # 更新 conv_state（在 wrapper 层面处理）
        if state_len >= 2 and seqlen > 0:
            # 保留原有的中间状态
            if state_token_offset + 1 < state_len:
                conv_state[ci, 0, :] = conv_state[ci, state_token_offset + 1, :]
            if state_token_offset + 2 < state_len:
                conv_state[ci, 1, :] = conv_state[ci, state_token_offset + 2, :]
            
            # 写入新 token
            for t in range(seqlen):
                write_pos = 2 + t
                if write_pos < state_len:
                    conv_state[ci, write_pos, :] = x_b[t, :]
    
    # 合并结果
    y = torch.stack(y_list, dim=0)  # [batch, seqlen, dim]
    
    if seqlen == 1:
        return y.squeeze(1)  # [batch, dim]
    return y