#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""
"""
import os
import torch
import pypto
import pytest
import torch
import numpy as np
from numpy.testing import assert_allclose
import torch.nn.functional as F
import time


def l2norm(
    query: pypto.Tensor,
    key: pypto.Tensor,
    eps: float = 1e-6)-> tuple[pypto.Tensor, pypto.Tensor]:
    """
    L2 normalization.

    Parameters
    ---------
    query: [L, D]
    key:   [L, D]
    eps=1e-6

    Return
     ---------
    query_after_l2norm: [L, D]
    key_after_l2norm: [L, D]
    """

    pypto.set_vec_tile_shapes(128, 128)
    # L2
    query_after_l2norm = query / pypto.sqrt((query * query).sum(-1, keepdim=True) + eps)
    key_after_l2norm = key / pypto.sqrt((key * key).sum(-1, keepdim=True) + eps)

    return query_after_l2norm, key_after_l2norm


def pre_attn(
    gate_view: pypto.Tensor,
    key_view_2d: pypto.Tensor,
    beta_view: pypto.Tensor,
    tril: pypto.Tensor,
    mask: pypto.Tensor
    )-> tuple[pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor]:
    """
    Calculate gate_cumsum、decay_mask、beta_k、kkt

    Parameters
    ---------
    gate: [L, 1]
    key: [L, D]
    beta: [L, 1]
    tril: [L, L]
    mask: [L, L]

    Return
    ---------
    gate_cum: [L, 1]
    decay_mask: [L, L]
    A: [L, L]
    key_beta: [L, D]
    """

    pypto.set_vec_tile_shapes(128, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    # cal_cumsum
    gate_cum = pypto.matmul(tril, gate_view, pypto.DT_FP32) #[L,1]
    # cal_decay_mask
    decay_mask = ((gate_cum - gate_cum.transpose(0,1)) * tril).exp() #[L,L]
    # beta_k
    key_beta = key_view_2d * beta_view #[L,D]
    # kkt计算
    kkt = pypto.matmul(key_beta, key_view_2d, pypto.DT_FP32, b_trans=True) #[L,L]
    A = kkt * decay_mask * mask #[L,L]

    return gate_cum, decay_mask, A, key_beta


def pre_attn_unaligned(
    gate_view: pypto.Tensor,
    key_view_2d: pypto.Tensor,
    beta_view: pypto.Tensor,
    tril: pypto.Tensor,
    mask: pypto.Tensor
    )-> tuple[pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor]:
    """
    Calculate gate_cumsum、decay_mask、beta_k、kkt

    Parameters
    ---------
    gate: [L, 1]
    key: [L, D]
    beta: [L, 1]
    tril: [L, L]
    mask: [L, L]

    Return
    ---------
    gate_cum: [L, 1]
    decay_mask: [L, L]
    A: [L, L]
    key_beta: [L, D]
    """
    L = gate_view.shape[0]
    actual_L = gate_view.valid_shape[0]
    pypto.set_vec_tile_shapes(128, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    # cal_cumsum
    gate_cum = pypto.matmul(tril, gate_view, pypto.DT_FP32)  # [L,1]
    # cal_decay_mask
    gate_T = gate_cum.reshape([1, L], valid_shape=[1, actual_L])
    g_sub = gate_cum - gate_T
    decay_mask = (g_sub * tril).exp() * tril  # [L,L]
    # beta_k
    key_beta = key_view_2d * beta_view  # [L,D]
    # kkt
    kkt = pypto.matmul(key_beta, key_view_2d, pypto.DT_FP32, b_trans=True)  # [L,L]
    A = kkt * decay_mask * mask  # [L,L]

    return gate_cum, decay_mask, A, key_beta


def inverse_pto_min_length(
    attn: pypto.Tensor,
    eye: pypto.Tensor,
    min_length: int) -> None:

    attn_inv_list = {}
    attn_inv_list[1] = attn[:2, :]
    attn_initial = pypto.tensor(attn.shape, dtype=attn.dtype)
    pypto.assemble(attn, [0, 0], attn_initial)

    pypto.set_vec_tile_shapes(128, 128)
    attn_transpose = attn.transpose(dim0=0, dim1=1)

    pypto.set_pass_options(sg_set_scope=1)
    for i in range(2, min_length, 1):
        row = attn_initial.view([1, min_length], [i, 0])
        row_expand = attn_transpose.view([i, 1], [0, i])
        # row_expand = row[:, :i].reshape([i, 1])
        prod = (row_expand * attn_inv_list[i - 1]).sum(0, keepdim=True)
        attn_update = row + prod

        attn_inv_list[i] = pypto.concat([attn_inv_list[i - 1], attn_update], dim=0)

    res = attn_inv_list[min_length - 1] + eye
    pypto.set_pass_options(sg_set_scope=-1)

    return res


def inverse_pto(
    attn: pypto.Tensor,
    eye: pypto.Tensor,
    size: int) -> pypto.Tensor:
    half_size = size // 2
    min_length = 16
    attn_1_1 = attn.view([half_size, half_size], [0, 0])
    attn_2_2 = attn.view([half_size, half_size], [half_size, half_size])

    if half_size == min_length:
        attn_1_1_inv = inverse_pto_min_length(attn_1_1, eye, min_length)
        attn_2_2_inv = inverse_pto_min_length(attn_2_2, eye, min_length)
    else:
        attn_1_1_inv = inverse_pto(attn_1_1, eye, half_size)
        attn_2_2_inv = inverse_pto(attn_2_2, eye, half_size)

    pypto.set_vec_tile_shapes(64, 64)
    pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])

    attn_2_1 = attn.view([half_size, half_size], [half_size, 0])

    attn_2_1_inv = ( attn_2_2_inv @ attn_2_1 ) @ attn_1_1_inv

    attn_inv = pypto.tensor(attn.shape, dtype=attn.dtype)
    attn_inv[0:half_size, 0:half_size] = attn_1_1_inv
    attn_inv[0:half_size, half_size:size] = pypto.full(size=[half_size, half_size], fill_value=0.0, dtype=attn.dtype)
    attn_inv[half_size:size, 0:half_size] = attn_2_1_inv
    attn_inv[half_size:size, half_size:size] = attn_2_2_inv

    return attn_inv


def cal_value_and_key_cumdecay(
    attn: pypto.Tensor,
    value_view: pypto.Tensor,
    beta_view: pypto.Tensor,
    key_beta: pypto.Tensor,
    gate_cum: pypto.Tensor)-> tuple[pypto.Tensor, pypto.Tensor]:
    """
    Calculate value and k cumdecay

    Parameters:
    -------------
    attn: [L, L]
    value_view: [L, D]
    beta_view: [L, D]
    key_beta: [L, D]
    gate_cum: [L, 1]

    Return:
    -------------
    value_out: [L, D]
    key_cum_out: [L, D]
    """

    pypto.set_vec_tile_shapes(128, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    # value_out
    value_beta_view = value_view * beta_view # [L, D]
    value_out = pypto.matmul(attn, value_beta_view, pypto.DT_FP32) # [L, D]
    # k_cumdecay_out
    g_exp = pypto.exp(gate_cum) # [L, 1]
    weighted_k_beta_view = key_beta * g_exp # [L, D]
    key_cum_out = pypto.matmul(attn, weighted_k_beta_view, pypto.DT_FP32) # [L, D]

    return value_out, key_cum_out


def recurrent_state_attn_all(
    query: pypto.Tensor,
    key: pypto.Tensor,
    value: pypto.Tensor,
    k_cumdecay: pypto.Tensor,
    gate: pypto.Tensor,
    state: pypto.Tensor,
    decay_mask: pypto.Tensor,
    tril: pypto.Tensor,
) -> tuple[pypto.Tensor, pypto.Tensor]:
    """
    Calculate attention.

    Parameters
    ---------
    query: [L, D]
    key: [L, D]
    value:[L, Dv]
    k_cumdecay:[L, Dk]
    gate: [L, 1]
    state: [D, D]
    decay_mask: [L, L]
    tril: [L, L]

    Return
    ---------
    chunk_attn_out: [L, D]
    state_new:[Dv, Dk]
    """
    Dv = value.shape[-1]
    L = gate.valid_shape[0]
    gate_exp = gate.exp()
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    pypto.set_vec_tile_shapes(64, 128)
    _last_gate_1 = gate[L - 1 : L, :]
    kgexp = key * (_last_gate_1 - gate).exp()  # [L, Dk]
    qgexp = query * gate_exp
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [64, 64])
    v_prime = pypto.matmul(k_cumdecay, state, pypto.DT_FP32, b_trans=True)  # [L, Dk] @ [Dk, Dv] = [L, Dv]
    attn_inter = pypto.matmul(qgexp, state, pypto.DT_FP32, b_trans=True)  # [L, Dk] @ [Dk, Dv] = [L, Dv]
    pypto.set_cube_tile_shapes([64, 64], [128, 128], [128, 128])
    temp_matmul_vprime = pypto.matmul(v_prime, kgexp, pypto.DT_FP32, a_trans=True)  # [Dv, L] @ [L, Dk] = [Dv, Dk]
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    temp_matmul_value = pypto.matmul(value, kgexp, pypto.DT_FP32, a_trans=True)  # [Dv, L] @ [L, Dk] = [L, Dk]
    attn = pypto.matmul(query, key, pypto.DT_FP32, b_trans=True)  # [L, Dk] @ [Dk, L] = [L, L]
    _last_gate_2 = pypto.expand_clone(gate_exp[L-1:L, :], (Dv, 1))  # [Dv, 1]
    final_state_1 = state * _last_gate_2 
    state_new = final_state_1 + temp_matmul_value - temp_matmul_vprime
    attn_tmp = attn * decay_mask * tril  # [L, L]
    chunk_attn_value = pypto.matmul(attn_tmp, value, pypto.DT_FP32)  # [L, L] @ [L, Dv] = [L, Dv]
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [64, 64])
    chunk_attn_vprime = pypto.matmul(attn_tmp, v_prime, pypto.DT_FP32)  # [L, L] @ [L, Dv] = [L, Dv]
    chunk_attn_out = attn_inter + chunk_attn_value - chunk_attn_vprime
    return chunk_attn_out, state_new



@pypto.jit(
    runtime_options={
        "stitch_function_inner_memory": 128 * 16,
        "stitch_function_num_initial": 128,
        "stitch_function_outcast_memory": 128 * 16
        },
    # debug_options={"runtime_debug_mode": 1},
)
def chunk_gated_delta_rule(query, key, value, beta, gate, states, mask,
    tril_mask, eye, act_seq_len, core_attn_out, last_state_data):

    _, Nqk, D = query.shape
    _, Nv, D = value.shape
    B = states.shape[0]
    L, L = mask.shape
    group = Nv // Nqk
    last_state = pypto.tensor([D, D], pypto.DT_FP32)
    for b_idx in pypto.loop(B, name="LOOP_B_TND", idx_name="b_idx"):
        S = act_seq_len[b_idx + 1] - act_seq_len[b_idx]
        b_ofs = act_seq_len[b_idx]
        for nv_idx in pypto.loop(Nv, name="LOOP_Nv_TND", idx_name="nv_idx"):
            nqk_idx = nv_idx // group
            pypto.set_vec_tile_shapes(16, 16, 128, 128)
            last_state = states[b_idx, nv_idx]
            for s_idx in pypto.loop(0, S, L, name="LOOP_S_TND", idx_name="s_idx", unroll_list=[16,1]):
                bs_ofs = b_ofs + s_idx
                actual_L = (S - s_idx).min(L)
                ## view
                query_view = pypto.view(query, [L, 1, D], [bs_ofs, nqk_idx, 0], valid_shape =[actual_L, 1, D])
                key_view = pypto.view(key, [L, 1, D], [bs_ofs, nqk_idx, 0], valid_shape =[actual_L, 1, D])
                value_view = pypto.view(value, [L, 1, D], [bs_ofs, nv_idx, 0], valid_shape =[actual_L, 1, D])
                beta_view = pypto.view(beta, [L, 1], [bs_ofs, nv_idx], valid_shape =[actual_L, 1])
                gate_view = pypto.view(gate, [L, 1], [bs_ofs, nv_idx], valid_shape =[actual_L, 1])

                pypto.set_vec_tile_shapes(128, 128, 128)
                query_view_2d = pypto.reshape(query_view, [L, D], valid_shape=[actual_L, D])
                key_view_2d = pypto.reshape(key_view, [L, D], valid_shape=[actual_L, D])
                value_view_2d = pypto.reshape(value_view, [L, D], valid_shape=[actual_L, D])

                query_view_2d32 = pypto.cast(query_view_2d, pypto.DT_FP32)
                key_view_2d32 = pypto.cast(key_view_2d, pypto.DT_FP32)
                value_view_2d32 = pypto.cast(value_view_2d, pypto.DT_FP32)
                beta_view32 = pypto.cast(beta_view, pypto.DT_FP32)

                # compute
                # qk_l2norm
                query_norm, key_norm = l2norm(query_view_2d32, key_view_2d32)
                scale = 1 / D ** 0.5
                query_scale = query_norm * scale

                if pypto.cond(pypto.is_loop_end(s_idx)):
                    # kv_beta & g_cumsum & decay_mask & pre_attn
                    mask_view = pypto.view(mask, [L, L], [0, 0], valid_shape=[actual_L, actual_L])
                    tril_mask_view = pypto.view(tril_mask, [L, L], [0, 0], valid_shape=[actual_L, actual_L])
                    gate_cum, decay_mask, A_block, key_beta = pre_attn_unaligned(gate_view, key_norm, beta_view32, tril_mask_view, mask_view)
                    # inverse
                    A_block_view = pypto.view(A_block, [L, L], [0, 0], valid_shape=[actual_L, actual_L])
                    A_block_inverse_aligned = inverse_pto(A_block_view, eye, 128)
                    pypto.set_vec_tile_shapes(128, 128)
                    A_block_inverse = pypto.view(A_block_inverse_aligned, [L, L], [0, 0], valid_shape=[actual_L, actual_L])
                    # cal_value_and_keycumdecay
                    value_out, key_cum_out= cal_value_and_key_cumdecay(A_block_inverse, value_view_2d32, beta_view32, key_beta, gate_cum)
                    chunk_attn_out, cur_state = recurrent_state_attn_all(query_scale, key_norm, value_out, key_cum_out, gate_cum, last_state, decay_mask, tril_mask_view)
                    chunk_attn_out16 = pypto.cast(chunk_attn_out, pypto.DT_BF16)
                    # assemble
                    last_state[:] = cur_state
                    pypto.set_vec_tile_shapes(128, 16, 128)
                    chunk_attn_out_16_reshaped = chunk_attn_out16.reshape([L, 1, D], valid_shape=[actual_L, 1, D])
                    pypto.assemble(chunk_attn_out_16_reshaped, [bs_ofs, nv_idx, 0], core_attn_out)
                    pypto.set_vec_tile_shapes(16, 16, 128, 128)
                    last_state_data[b_idx, nv_idx] = last_state
                else:
                # if True:
                    gate_cum, decay_mask, A_block, key_beta = pre_attn(gate_view, key_norm, beta_view32, tril_mask, mask)
                    # inverse
                    A_block_inverse = inverse_pto(A_block, eye, 128)

                    # cal_value_and_keycumdecay
                    value_out, key_cum_out = cal_value_and_key_cumdecay(A_block_inverse, value_view_2d32, beta_view32, key_beta, gate_cum)
                    chunk_attn_out, cur_state = recurrent_state_attn_all(query_scale, key_norm, value_out, key_cum_out, gate_cum, last_state, decay_mask, tril_mask)
                    chunk_attn_out16 = pypto.cast(chunk_attn_out, pypto.DT_BF16)

                    # assemble
                    pypto.set_vec_tile_shapes(16, 16, 128, 128)
                    last_state[:] = cur_state
                    core_attn_out[bs_ofs:bs_ofs + L, nv_idx] = chunk_attn_out16
                    last_state_data[b_idx, nv_idx] = last_state


def pypto_chunk_gated_delta_rule(
    query_data,
    key_data,
    value_data,
    beta_data,
    gate_data,
    state_data,
    act_seq_len):
    """
    PyPTO calculate chunk Gated Delta Rule.

    Parameters
    ---------
    query_data: [T, Nqk, D]
    key_data: [T, Nqk, D]
    value_data: [T, Nv, D]
    beta_data: [T, Nv]
    gate_data: [T, Nv]
    state_data: [B, Nv, D, D]
    act_seq_len: [B,]

    Return
     ---------
    core_attn_out: [T, Nv, D]
    state_data: [B, Nv, D, D]
    """
    T, Nv, D = value_data.shape
    L = 128
    B = state_data.shape[0]
    
    if not query_data.is_contiguous():
        query_data = query_data.contiguous()
    if not key_data.is_contiguous():
        key_data = key_data.contiguous()
    if not value_data.is_contiguous():
        value_data = value_data.contiguous()
    if not beta_data.is_contiguous():
        beta_data = beta_data.contiguous()
    if not gate_data.is_contiguous():
        gate_data = gate_data.contiguous()
    if not state_data.is_contiguous():
        state_data = state_data.contiguous()

    # output
    core_attn_out = torch.ones([T, Nv, D], dtype=torch.bfloat16, device=query_data.device)
    last_state_data = torch.zeros([B, Nv, D, D], dtype=torch.float32, device=query_data.device)
    # helper data
    mask_data = torch.tril(-torch.ones([L, L], dtype=torch.float32, device=query_data.device), diagonal=-1)
    tril_mask_data = torch.ones([L, L], device=query_data.device).float().tril() # lower triangular
    eye_data = torch.eye(16, device=query_data.device).float()

    inputs = {
        query_data: [0],
        key_data: [0],
        value_data: [0],
        beta_data: [0],
        gate_data: [0],
        state_data: [0],
        mask_data: [],
        tril_mask_data: [],
        eye_data: [],
        act_seq_len: [0],
    }
    outputs = {
        core_attn_out: [0],
        last_state_data: [0]
    }

    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
    torch.npu.empty_cache()
    chunk_gated_delta_rule(*pto_inputs, *pto_outputs)
    # torch.npu.synchronize()
    return core_attn_out, last_state_data
