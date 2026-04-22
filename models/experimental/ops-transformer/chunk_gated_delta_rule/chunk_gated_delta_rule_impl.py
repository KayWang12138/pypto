#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.

"""PyPTO chunk_gated_delta_rule kernel implementation.

基于 chunk 的门控 Delta Rule 前向传播算子，用于线性注意力机制（Linear Attention）
中的隐藏状态递推计算。支持定长序列和变长序列两种模式。

特性：
    - 使用 pypto.DYNAMIC 标注 H/Hg 维度，单一 kernel 支持任意配置
    - H 作为参数传入，使用 Python range 静态遍历（避免编译复杂度过高）
    - 支持定长序列和变长序列两种模式
    - 状态传递使用切片赋值
    - FP16 输入，FP32 计算精度控制
    - GQA head mapping: k_head = h_idx // (H // Hg)

参考：models/qwen3_next/gated_delta_rule_impl.py, examples/
"""

import pypto
import torch


K_DIM = 128
V_DIM = 128
BT = 64


@pypto.frontend.jit
def chunk_gated_delta_rule_fixed_kernel(
    k: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, K_DIM], pypto.DT_FP16),
    w: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, K_DIM], pypto.DT_FP16),
    v: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, V_DIM], pypto.DT_FP16),
    g: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    h0: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, K_DIM, V_DIM], pypto.DT_FP16),
    h_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, K_DIM, V_DIM], pypto.DT_FP16),
    v_new_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, V_DIM], pypto.DT_FP16),
    ht_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, K_DIM, V_DIM], pypto.DT_FP16),
    H: int = 8,
    Hg: int = 4,
):
    GROUP = H // Hg
    
    B = k.shape[0]
    T = k.shape[1]
    NT = (T + BT - 1) // BT
    
    for b_idx in pypto.loop(B, name="LOOP_B", idx_name="b_idx", unroll_list=[4, 2, 1]):
        for h_idx in range(H):
            k_head_idx = h_idx // GROUP
            
            pypto.set_vec_tile_shapes(K_DIM, V_DIM)
            h_state_fp16 = h0[b_idx, h_idx]
            h_state = pypto.cast(h_state_fp16, pypto.DT_FP32)
            
            for chunk_idx in pypto.loop(NT, name="LOOP_CHUNK", idx_name="chunk_idx", unroll_list=[16, 4, 1]):
                t_start = chunk_idx * BT
                actual_bt = (T - t_start).min(BT)
                
                pypto.set_cube_tile_shapes([64, 64], [128, 128], [64, 128])
                w_chunk = pypto.view(w, [BT, K_DIM], [b_idx, t_start, h_idx, 0], valid_shape=[actual_bt, K_DIM])
                w_chunk_fp32 = pypto.cast(w_chunk, pypto.DT_FP32)
                wh = pypto.matmul(w_chunk_fp32, h_state, pypto.DT_FP32, b_trans=True)
                
                pypto.set_vec_tile_shapes(BT, V_DIM)
                v_chunk_fp16 = pypto.view(v, [BT, V_DIM], [b_idx, t_start, h_idx, 0], valid_shape=[actual_bt, V_DIM])
                v_chunk = pypto.cast(v_chunk_fp16, pypto.DT_FP32)
                v_new = pypto.sub(v_chunk, wh)
                
                pypto.set_vec_tile_shapes(BT)
                g_chunk = pypto.view(g, [BT, 1], [b_idx, t_start, h_idx], valid_shape=[actual_bt, 1])
                g_last = g_chunk[actual_bt - 1:actual_bt]
                
                g_diff = pypto.sub(g_last, g_chunk)
                g_exp = pypto.exp(g_diff)
                g_exp_broadcast = pypto.expand_clone(g_exp, (BT, V_DIM))
                
                pypto.set_vec_tile_shapes(BT, V_DIM)
                v_new = pypto.mul(v_new, g_exp_broadcast)
                
                g_last_exp = pypto.exp(g_last)
                g_last_exp_k = pypto.expand_clone(g_last_exp, (K_DIM, 1))
                
                pypto.set_vec_tile_shapes(K_DIM, V_DIM)
                h_state = pypto.mul(h_state, g_last_exp_k)
                
                pypto.set_cube_tile_shapes([64, 64], [64, 128], [64, 128])
                k_chunk = pypto.view(k, [BT, K_DIM], [b_idx, t_start, k_head_idx, 0], valid_shape=[actual_bt, K_DIM])
                k_chunk_fp32 = pypto.cast(k_chunk, pypto.DT_FP32)
                h_upd = pypto.matmul(k_chunk_fp32, v_new, pypto.DT_FP32, a_trans=True)
                
                pypto.set_vec_tile_shapes(K_DIM, V_DIM)
                h_state = pypto.add(h_state, h_upd)
                
                h_state_fp16 = pypto.cast(h_state, pypto.DT_FP16)
                v_new_fp16 = pypto.cast(v_new, pypto.DT_FP16)
                
                h_out[b_idx, chunk_idx, h_idx] = h_state_fp16
                v_new_out[b_idx, t_start : t_start + BT, h_idx] = v_new_fp16
            
            h_state_fp16 = pypto.cast(h_state, pypto.DT_FP16)
            ht_out[b_idx, h_idx] = h_state_fp16


@pypto.frontend.jit
def chunk_gated_delta_rule_varlen_kernel(
    k: pypto.Tensor([1, pypto.DYNAMIC, pypto.DYNAMIC, K_DIM], pypto.DT_FP16),
    w: pypto.Tensor([1, pypto.DYNAMIC, pypto.DYNAMIC, K_DIM], pypto.DT_FP16),
    v: pypto.Tensor([1, pypto.DYNAMIC, pypto.DYNAMIC, V_DIM], pypto.DT_FP16),
    g: pypto.Tensor([1, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    h0: pypto.Tensor([1, pypto.DYNAMIC, pypto.DYNAMIC, K_DIM, V_DIM], pypto.DT_FP16),
    cu_seqlens: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    h_out: pypto.Tensor([1, pypto.DYNAMIC, pypto.DYNAMIC, K_DIM, V_DIM], pypto.DT_FP16),
    v_new_out: pypto.Tensor([1, pypto.DYNAMIC, pypto.DYNAMIC, V_DIM], pypto.DT_FP16),
    ht_out: pypto.Tensor([1, pypto.DYNAMIC, pypto.DYNAMIC, K_DIM, V_DIM], pypto.DT_FP16),
    H: int = 8,
    Hg: int = 4,
):
    GROUP = H // Hg
    
    T_total = k.shape[1]
    N = cu_seqlens.shape[0] - 1
    
    chunk_offset = pypto.SymbolicScalar(0)
    
    for seq_idx in pypto.loop(N, name="LOOP_SEQ", idx_name="seq_idx", unroll_list=[4, 2, 1]):
        bos = cu_seqlens[seq_idx]
        eos = cu_seqlens[seq_idx + 1]
        s_len = eos - bos
        
        nt_seq = (s_len + BT - 1) // BT
        
        for h_idx in range(H):
            k_head_idx = h_idx // GROUP
            
            pypto.set_vec_tile_shapes(K_DIM, V_DIM)
            h_state_fp16 = h0[0, seq_idx, h_idx]
            h_state = pypto.cast(h_state_fp16, pypto.DT_FP32)
            
            for chunk_idx in pypto.loop(nt_seq, name="LOOP_CHUNK", idx_name="chunk_idx"):
                t_start = chunk_idx * BT
                actual_bt = (s_len - t_start).min(BT)
                
                data_offset = bos + t_start
                
                pypto.set_cube_tile_shapes([64, 64], [128, 128], [64, 128])
                w_chunk = pypto.view(w, [BT, K_DIM], [0, data_offset, h_idx, 0], valid_shape=[actual_bt, K_DIM])
                w_chunk_fp32 = pypto.cast(w_chunk, pypto.DT_FP32)
                wh = pypto.matmul(w_chunk_fp32, h_state, pypto.DT_FP32, b_trans=True)
                
                pypto.set_vec_tile_shapes(BT, V_DIM)
                v_chunk_fp16 = pypto.view(v, [BT, V_DIM], [0, data_offset, h_idx, 0], valid_shape=[actual_bt, V_DIM])
                v_chunk = pypto.cast(v_chunk_fp16, pypto.DT_FP32)
                v_new = pypto.sub(v_chunk, wh)
                
                pypto.set_vec_tile_shapes(BT)
                g_chunk = pypto.view(g, [BT, 1], [0, data_offset, h_idx], valid_shape=[actual_bt, 1])
                g_last = g_chunk[actual_bt - 1:actual_bt]
                
                g_diff = pypto.sub(g_last, g_chunk)
                g_exp = pypto.exp(g_diff)
                g_exp_broadcast = pypto.expand_clone(g_exp, (BT, V_DIM))
                
                pypto.set_vec_tile_shapes(BT, V_DIM)
                v_new = pypto.mul(v_new, g_exp_broadcast)
                
                g_last_exp = pypto.exp(g_last)
                g_last_exp_k = pypto.expand_clone(g_last_exp, (K_DIM, 1))
                
                pypto.set_vec_tile_shapes(K_DIM, V_DIM)
                h_state = pypto.mul(h_state, g_last_exp_k)
                
                pypto.set_cube_tile_shapes([64, 64], [64, 128], [64, 128])
                k_chunk = pypto.view(k, [BT, K_DIM], [0, data_offset, k_head_idx, 0], valid_shape=[actual_bt, K_DIM])
                k_chunk_fp32 = pypto.cast(k_chunk, pypto.DT_FP32)
                h_upd = pypto.matmul(k_chunk_fp32, v_new, pypto.DT_FP32, a_trans=True)
                
                pypto.set_vec_tile_shapes(K_DIM, V_DIM)
                h_state = pypto.add(h_state, h_upd)
                
                h_state_fp16 = pypto.cast(h_state, pypto.DT_FP16)
                v_new_fp16 = pypto.cast(v_new, pypto.DT_FP16)
                
                h_out[0, chunk_offset + chunk_idx, h_idx] = h_state_fp16
                v_new_out[0, data_offset : data_offset + BT, h_idx] = v_new_fp16
            
            h_state_fp16 = pypto.cast(h_state, pypto.DT_FP16)
            ht_out[0, seq_idx, h_idx] = h_state_fp16
        
        chunk_offset = chunk_offset + nt_seq


def chunk_gated_delta_rule_wrapper(
    k: torch.Tensor,
    w: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor = None,
    h0: torch.Tensor = None,
    output_final_state: bool = True,
    chunk_size: int = 64,
    cu_seqlens: torch.Tensor = None,
) -> tuple:
    """算子 wrapper，使用动态轴支持任意 H/Hg 配置。
    
    Args:
        k: Key 向量，[B, T, Hg, K] 或 [1, T_total, Hg, K], float16
        w: 门控权重，[B, T, H, K] 或 [1, T_total, H, K], float16
        v: Value 向量，[B, T, H, V] 或 [1, T_total, H, V], float16
        g: 门控向量，[B, T, H] 或 [1, T_total, H], float32 (可选)
        h0: 初始状态，[B, H, K, V] 或 [1, N, H, K, V], float16 (可选)
        output_final_state: 是否输出最终状态 ht
        chunk_size: chunk 分块大小，默认 64
        cu_seqlens: 变长序列边界 [N+1], int32 (可选)
    
    Returns:
        h: 每个 chunk 的隐藏状态，[B, NT, H, K, V] 或 [1, NT_total, H, K, V], float16
        v_new: 更新后的 value，[B, T, H, V] 或 [1, T_total, H, V], float16
        ht: 最终隐藏状态，[B, H, K, V] 或 [1, N, H, K, V], float16
    """
    use_g = g is not None
    use_initial_state = h0 is not None
    is_varlen = cu_seqlens is not None
    
    BT = chunk_size
    K_DIM_LOCAL = k.shape[-1]
    V_DIM_LOCAL = v.shape[-1]
    
    if not is_varlen:
        B, T_len, Hg, K = k.shape
        _, _, H, V = v.shape
        NT = (T_len + BT - 1) // BT
        
        if H % Hg != 0:
            raise ValueError(f"H must be divisible by Hg: H={H}, Hg={Hg}")
        
        g_placeholder = g if use_g else torch.zeros(B, T_len, H, dtype=torch.float32, device=k.device)
        h0_placeholder = h0 if use_initial_state else torch.zeros(B, H, K_DIM_LOCAL, V_DIM_LOCAL, dtype=torch.float16, device=k.device)
        
        h_out = torch.zeros(B, NT, H, K_DIM_LOCAL, V_DIM_LOCAL, dtype=torch.float16, device=k.device)
        v_new_out = torch.zeros(B, T_len, H, V_DIM_LOCAL, dtype=torch.float16, device=k.device)
        ht_out = torch.zeros(B, H, K_DIM_LOCAL, V_DIM_LOCAL, dtype=torch.float16, device=k.device)
        
        chunk_gated_delta_rule_fixed_kernel(
            k, w, v, g_placeholder, h0_placeholder,
            h_out, v_new_out, ht_out, H=H, Hg=Hg
        )
        
        return h_out, v_new_out, ht_out
    
    else:
        _, T_total, Hg, K = k.shape
        _, _, H, V = v.shape
        N = len(cu_seqlens) - 1
        
        if H % Hg != 0:
            raise ValueError(f"H must be divisible by Hg: H={H}, Hg={Hg}")
        
        NT_total = sum([
            (int(cu_seqlens[i + 1]) - int(cu_seqlens[i]) + BT - 1) // BT
            for i in range(N)
        ])
        
        g_placeholder = g if use_g else torch.zeros(1, T_total, H, dtype=torch.float32, device=k.device)
        h0_placeholder = h0 if use_initial_state else torch.zeros(1, N, H, K_DIM_LOCAL, V_DIM_LOCAL, dtype=torch.float16, device=k.device)
        
        h_out = torch.zeros(1, NT_total, H, K_DIM_LOCAL, V_DIM_LOCAL, dtype=torch.float16, device=k.device)
        v_new_out = torch.zeros(1, T_total, H, V_DIM_LOCAL, dtype=torch.float16, device=k.device)
        ht_out = torch.zeros(1, N, H, K_DIM_LOCAL, V_DIM_LOCAL, dtype=torch.float16, device=k.device)
        
        chunk_gated_delta_rule_varlen_kernel(
            k, w, v, g_placeholder, h0_placeholder, cu_seqlens,
            h_out, v_new_out, ht_out, H=H, Hg=Hg
        )
        
        return h_out, v_new_out, ht_out