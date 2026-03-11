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
""" """

import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from numpy.testing import assert_allclose

import pypto


def l2norm(
    query_z: pypto.Tensor, key: pypto.Tensor, eps: float = 1e-6
) -> tuple[pypto.Tensor, pypto.Tensor]:
    """
    L2 normalization.

    Parameters
    ---------
    query_z: [L, D]
    key: [L, D]
    eps=1e-6

    Return
    ---------
    query_after_l2norm: [L, D]
    key_after_l2norm: [L, D]
    """

    pypto.set_vec_tile_shapes(128, 128)
    # L2
    query_after_l2norm = query_z / pypto.sqrt((query_z * query_z).sum(-1, keepdim=True) + eps)
    key_after_l2norm = key / pypto.sqrt((key * key).sum(-1, keepdim=True) + eps)

    return query_after_l2norm, key_after_l2norm


def pre_attn(
    gate_view: pypto.Tensor,
    key_view_2d: pypto.Tensor,
    beta_view: pypto.Tensor,
    tril: pypto.Tensor,
    mask: pypto.Tensor,
) -> tuple[pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor]:
    """
    Calculate gate_cumsum, decay_mask, beta_k and kkt.

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
    gate_cum = pypto.matmul(tril, gate_view, pypto.DT_FP32)  # [L,1]
    # cal_decay_mask
    decay_mask = ((gate_cum - gate_cum.transpose(0, 1)) * tril).exp()  # [L,L]
    # beta_k
    key_beta = key_view_2d * beta_view  # [L,D]
    # kkt
    kkt = pypto.matmul(key_beta, key_view_2d, pypto.DT_FP32, b_trans=True)  # [L,L]
    A = kkt * decay_mask * mask  # [L,L]

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


def inverse_pto(attn: pypto.Tensor, eye: pypto.Tensor, size: int) -> pypto.Tensor:
    """
    Calculate inverse of big matrix.

    Parameters
    ---------
    attn: [L, L]
    eye: [L, L]
    size: matrix size

    Return
    ---------
    attn_inv: [L, L]
    """
    min_length = size // 8
    pypto.set_vec_tile_shapes(128, 128)

    attn_8_8_list = []
    for i in range(8):
        attn_8_8_list.append(attn.view([min_length, min_length], [min_length*i, min_length*i]) + 0.0)
    attn_tmp_dim0 = pypto.concat(attn_8_8_list, dim=0)
    attn_tmp_dim1 = pypto.concat(attn_8_8_list, dim=1)

    attn_tmp_dim1_inv = inverse_pto_min_length(attn_tmp_dim0, attn_tmp_dim1, eye, min_length, min_length*8)

    attn_8_8_inv_list = []
    for i in range(8):
        attn_8_8_inv_list.append(attn_tmp_dim1_inv[:, min_length*i:min_length*(i+1)] + 0.0)

    attn_4_inv_list = []
    for i in range(4):
        attn_4_inv_list.append(inverse_matmul(attn, attn_8_8_inv_list[i*2], attn_8_8_inv_list[i*2+1], min_length*i*2, min_length*i*2, min_length))

    attn_2_inv_list = []
    for i in range(2):
        attn_2_inv_list.append(inverse_matmul(attn, attn_4_inv_list[i*2], attn_4_inv_list[i*2+1], min_length*i*4, min_length*i*4, min_length*2))

    attn_inv = inverse_matmul(attn, attn_2_inv_list[0], attn_2_inv_list[1], 0, 0, min_length*4)
    return attn_inv


def inverse_pto_min_length(
    attn_dim0: pypto.Tensor,
    attn_dim1: pypto.Tensor,
    eye: pypto.Tensor,
    row_num: int,
    col_num: int,
) -> pypto.Tensor:
    """
    Calculate inverse of matrix with tail concat optimization.

    Parameters
    ---------
    attn_dim0: [L, L // 8]
    attn_dim1: [L // 8, L]
    eye: [L, L]
    row_num: L // 8
    col_num: L

    Return
    ---------
    res: [L, L]
    """
    size = col_num // row_num

    attn_inv_list = {}
    attn_inv_list[1] = attn_dim1[:2, :]
    pypto.set_vec_tile_shapes(128, 128)

    attn_dim0_trans = attn_dim0.transpose(0, 1).reshape([col_num, row_num])

    for i in range(2, row_num, 1):
        # Add 0.0 to enable attn_inv_cur to enter the UB in advance
        attn_inv_cur = attn_inv_list[i - 1] + 0.0
        row = attn_dim1.view([1, col_num], [i, 0])

        # used when combine_axis is enbaled
        # row_expand = row.reshape([size, row_num]).view([size, i], [0, 0]).transpose(1, 0).reshape([size*i, 1])

        row_expand = attn_dim0_trans.view([size * i, 1], [0, i])
        attn_inv_cur_reshape = attn_inv_cur.reshape([size * i, row_num])
        prod_mul = (row_expand * attn_inv_cur_reshape).reshape([i, col_num])

        prod = prod_mul.sum(0, keepdim=True)
        attn_update = row + prod

        attn_inv_list[i] = pypto.concat([attn_inv_cur, attn_update], dim=0)

    res = attn_inv_list[row_num - 1] + eye

    return res


def inverse_matmul(
    attn: pypto.Tensor,
    attn_1_1_inv: pypto.Tensor,
    attn_2_2_inv: pypto.Tensor,
    x_ofs: int,
    y_ofs: int,
    len: int,
) -> pypto.Tensor:
    """
    Calculate inverse of samll matrix.

    Parameters
    ---------
    attn: [L, L]
    attn_1_1_inv: attn upper left matrix
    attn_2_2_inv: attn bottom right matrix
    x_ofs: row offset
    y_ofs: column offset
    len: matrix length

    Return
    ---------
    attn_inv: [len * 2, len * 2]
    """
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

    attn_2_1 = attn.view([len, len], [x_ofs + len, y_ofs])

    attn_2_1_inv = (attn_2_2_inv @ attn_2_1) @ attn_1_1_inv

    attn_inv = pypto.tensor([len * 2, len * 2], dtype=attn_1_1_inv.dtype)
    attn_inv[0:len, 0:len] = attn_1_1_inv
    attn_inv[len : len * 2, 0:len] = attn_2_1_inv
    attn_inv[len : len * 2, len : len * 2] = attn_2_2_inv

    return attn_inv


def cal_value_and_key_cumdecay(
    attn: pypto.Tensor,
    value_view: pypto.Tensor,
    beta_view: pypto.Tensor,
    key_beta: pypto.Tensor,
    gate_cum: pypto.Tensor,
) -> tuple[pypto.Tensor, pypto.Tensor]:
    """
    Calculate value and k cumdecay.

    Parameters:
    ---------
    attn: [L, L]
    value_view: [L, D]
    beta_view: [L, D]
    key_beta: [L, D]
    gate_cum: [L, 1]

    Return:
    ---------
    value_out: [L, D]
    key_cum_out: [L, D]
    """

    pypto.set_vec_tile_shapes(128, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    # value_out
    value_beta_view = value_view * beta_view  # [L, D]
    value_out = pypto.matmul(attn, value_beta_view, pypto.DT_FP32)  # [L, D]
    # k_cumdecay_out
    g_exp = pypto.exp(gate_cum)  # [L, 1]
    weighted_k_beta_view = key_beta * g_exp  # [L, D]
    key_cum_out = pypto.matmul(attn, weighted_k_beta_view, pypto.DT_FP32)  # [L, D]

    return value_out, key_cum_out


def recurrent_state_attn_all(
    query_z: pypto.Tensor,
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
    query_z: [L, D]
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
    L = gate.shape[0]
    gate_exp = gate.exp()
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    pypto.set_vec_tile_shapes(64, 128)
    _last_gate_1 = gate[L - 1 : L, :]
    kgexp = key * (_last_gate_1 - gate).exp()  # [L, Dk]
    qgexp = query_z * gate_exp
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [64, 64])
    v_prime_t = pypto.matmul(k_cumdecay, state, pypto.DT_FP32, b_trans=True)  # [L, Dk] @ [Dk, Dv] = [L, Dv]
    attn_inter = pypto.matmul(qgexp, state, pypto.DT_FP32, b_trans=True)  # [L, Dk] @ [Dk, Dv] = [L, Dv]
    pypto.set_cube_tile_shapes([64, 64], [128, 128], [128, 128])
    temp_matmul_vprime = pypto.matmul(v_prime_t, kgexp, pypto.DT_FP32, a_trans=True)  # [Dv, L] @ [L, Dk] = [Dv, Dk]
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    temp_matmul_value = pypto.matmul(value, kgexp, pypto.DT_FP32, a_trans=True)  # [Dv, L] @ [L, Dk] = [L, Dk]
    attn = pypto.matmul(query_z, key, pypto.DT_FP32, b_trans=True)  # [L, Dk] @ [Dk, L] = [L, L]
    _last_gate_2 = pypto.expand_clone(gate_exp[L-1:L, :], (Dv, 1))  # [Dv, 1]
    final_state_1 = state * _last_gate_2
    state_new = final_state_1 + temp_matmul_value - temp_matmul_vprime
    attn_tmp = attn * decay_mask * tril  # [L, L]
    chunk_attn_value = pypto.matmul(attn_tmp, value, pypto.DT_FP32)  # [L, L] @ [L, Dv] = [L, Dv]
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [64, 64])
    chunk_attn_vprime = pypto.matmul(attn_tmp, v_prime_t, pypto.DT_FP32)  # [L, L] @ [L, Dv] = [L, Dv]
    chunk_attn_out = attn_inter + chunk_attn_value - chunk_attn_vprime
    return chunk_attn_out, state_new


def recurrent_state_attn_all_unaligend(
    query_z: pypto.Tensor,
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
    query_z: [L, D]
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
    actual_L = gate.valid_shape[0]
    gate_exp = gate.exp()
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    pypto.set_vec_tile_shapes(64, 128)
    _last_gate_1 = gate[actual_L - 1 : actual_L, :]
    kgexp = key * (_last_gate_1 - gate).exp()  # [L, Dk]
    qgexp = query_z * gate_exp
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [64, 64])
    v_prime_t = pypto.matmul(k_cumdecay, state, pypto.DT_FP32, b_trans=True)  # [L, Dk] @ [Dk, Dv] = [L, Dv]
    attn_inter = pypto.matmul(qgexp, state, pypto.DT_FP32, b_trans=True)  # [L, Dk] @ [Dk, Dv] = [L, Dv]
    pypto.set_cube_tile_shapes([64, 64], [128, 128], [128, 128])
    temp_matmul_vprime = pypto.matmul(v_prime_t, kgexp, pypto.DT_FP32, a_trans=True)  # [Dv, L] @ [L, Dk] = [Dv, Dk]
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    temp_matmul_value = pypto.matmul(value, kgexp, pypto.DT_FP32, a_trans=True)  # [Dv, L] @ [L, Dk] = [L, Dk]
    attn = pypto.matmul(query_z, key, pypto.DT_FP32, b_trans=True)  # [L, Dk] @ [Dk, L] = [L, L]
    _last_gate_2 = pypto.expand_clone(gate_exp[actual_L-1:actual_L, :], (Dv, 1))  # [Dv, 1]
    final_state_1 = state * _last_gate_2
    state_new = final_state_1 + temp_matmul_value - temp_matmul_vprime
    tril_view = tril.view(tril.shape, [0, 0], valid_shape=[actual_L, actual_L])
    attn_tmp = attn * decay_mask * tril_view  # [L, L]
    chunk_attn_value = pypto.matmul(attn_tmp, value, pypto.DT_FP32)  # [L, L] @ [L, Dv] = [L, Dv]
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [64, 64])
    chunk_attn_vprime = pypto.matmul(attn_tmp, v_prime_t, pypto.DT_FP32)  # [L, L] @ [L, Dv] = [L, Dv]
    chunk_attn_out = attn_inter + chunk_attn_value - chunk_attn_vprime
    return chunk_attn_out, state_new


verify_options = {
    "enable_pass_verify": True,
    "pass_verify_save_tensor": True,
}


@pypto.jit(
    host_options={
        "compile_monitor_enable": True,
        "compile_timeout": 120,
        "compile_timeout_stage": 20,
        "compile_monitor_print_interval": 5
    },
    runtime_options={
        "stitch_function_inner_memory": 128 * 32,
        # "stitch_function_num_initial": 512,
        "stitch_function_num_initial": 128,
        "stitch_function_outcast_memory": 128 * 32,
    },
    debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0},
    # debug_options={"runtime_debug_mode": 1},
    # verify_options=verify_options
)
def chunk_gated_delta_rule(
    query_z,
    key,
    value,
    beta,
    gate,
    states,
    mask,
    tril_mask,
    eye,
    act_seq_len,
    core_attn_out_t,
    last_state_data,
):

    _, Nqk, D = query_z.shape
    _, Nv, D = value.shape
    B = states.shape[0]
    L, L = mask.shape
    group = Nv // Nqk
    for b_idx in pypto.loop(B, name="LOOP_B_TND", idx_name="b_idx"):
        S = act_seq_len[b_idx + 1] - act_seq_len[b_idx]
        b_ofs = act_seq_len[b_idx]
        for nv_idx in pypto.loop(Nv, name="LOOP_Nv_TND", idx_name="nv_idx"):
            nqk_idx = nv_idx // group
            pypto.set_vec_tile_shapes(16, 16, 128, 128)
            last_state = states[b_idx, nv_idx]
            for s_idx in pypto.loop(0, S, L, name="LOOP_S_TND", idx_name="s_idx", unroll_list=[6]):
                bs_ofs = b_ofs + s_idx
                actual_L = (S - s_idx).min(L)
                ## view
                query_view = pypto.view(query_z, [L, 1, D], [bs_ofs, nqk_idx, 0], valid_shape =[actual_L, 1, D])
                key_view = pypto.view(key, [L, 1, D], [bs_ofs, nqk_idx, 0], valid_shape =[actual_L, 1, D])
                value_view = pypto.view(value, [L, 1, D], [bs_ofs, nv_idx, 0], valid_shape =[actual_L, 1, D])
                beta_view = pypto.view(beta, [L, 1], [bs_ofs, nv_idx], valid_shape =[actual_L, 1])
                gate_view = pypto.view(gate, [L, 1], [bs_ofs, nv_idx], valid_shape =[actual_L, 1])

                pypto.set_vec_tile_shapes(128, 128, 128)
                query_view_2d = pypto.reshape(query_view, [L, D], valid_shape=[actual_L, D])
                key_view_2d = pypto.reshape(key_view, [L, D], valid_shape=[actual_L, D])
                value_view_2d = pypto.reshape(value_view, [L, D], valid_shape=[actual_L, D])

                # compute
                # qk_l2norm
                query_norm, key_norm = l2norm(query_view_2d, key_view_2d)
                scale = 1 / D**0.5
                query_scale = query_norm * scale

                # kv_beta & g_cumsum & decay_mask & pre_attn
                # if pypto.cond(pypto.is_loop_end(s_idx)): # ! 构图不会卡死
                if pypto.cond(actual_L < L): # ! 构图卡死
                    # inverse
                    mask_view = pypto.view(mask, [L, L], [0, 0], valid_shape=[actual_L, actual_L])
                    tril_mask_view = pypto.view(tril_mask, [L, L], [0, 0], valid_shape=[actual_L, actual_L])
                    gate_cum, decay_mask, A_block, key_beta = pre_attn_unaligned(gate_view, key_norm, beta_view, tril_mask_view, mask_view)

                    Tmp = pypto.full([L, L], 0.0, pypto.DT_FP32)
                    Eyechunk = pypto.tensor([L, L], pypto.DT_FP32)
                    Eyechunk[0:,0:] = Tmp
                    pypto.assemble(A_block, [0, 0], Eyechunk)
                    A_block_inverse_aligned = inverse_pto(Eyechunk, eye, 128)

                    pypto.set_vec_tile_shapes(128, 128)
                    A_block_inverse = pypto.view(A_block_inverse_aligned, [L, L], [0, 0], valid_shape=[actual_L, actual_L])

                    # cal_value_and_keycumdecay
                    value_out, key_cum_out= cal_value_and_key_cumdecay(A_block_inverse, value_view_2d, beta_view, key_beta, gate_cum)
                    chunk_attn_out, cur_state = recurrent_state_attn_all_unaligend(query_scale, key_norm, value_out, key_cum_out, gate_cum, last_state, decay_mask, tril_mask)
                    # chunk_attn_out16 = pypto.cast(chunk_attn_out, pypto.DT_BF16)
                    # assemble
                    last_state[:] = cur_state

                    chunk_attn_out_16_reshaped = chunk_attn_out.reshape([L, 1, D], valid_shape=[actual_L, 1, D])
                    pypto.set_vec_tile_shapes(128,16,128)
                    pypto.assemble(chunk_attn_out_16_reshaped, [bs_ofs, nv_idx, 0], core_attn_out_t)
                    last_state_data[b_idx, nv_idx] = last_state
                else:
                    gate_cum, decay_mask, A_block, key_beta = pre_attn(gate_view, key_norm, beta_view, tril_mask, mask)
                    # inverse
                    A_block_inverse = inverse_pto(A_block, eye, 128)

                    # cal_value_and_keycumdecay
                    value_out, key_cum_out = cal_value_and_key_cumdecay(A_block_inverse, value_view_2d, beta_view, key_beta, gate_cum)

                    chunk_attn_out, cur_state = recurrent_state_attn_all(query_scale, key_norm, value_out, key_cum_out, gate_cum, last_state, decay_mask, tril_mask)

                    # assemble
                    last_state[:] = cur_state
                    core_attn_out_t[bs_ofs : bs_ofs + L, nv_idx] = chunk_attn_out
                    last_state_data[b_idx, nv_idx] = last_state


def pypto_chunk_gated_delta_rule(
    query_data: pypto.Tensor,
    key_data: pypto.Tensor,
    value_data: pypto.Tensor,
    beta_data: pypto.Tensor,
    gate_data: pypto.Tensor,
    state_data: pypto.Tensor,
    act_seq_len: pypto.Tensor,
) -> pypto.Tensor:
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
    core_attn_out_t: [T, Nv, D]
    state_data: [B, Nv, D, D]
    """

    T, Nv, D = value_data.shape
    L = 128
    B = state_data.shape[0]

    # output
    core_attn_out_t = torch.ones([T, Nv, D], dtype=torch.float32, device=query_data.device)
    last_state_data = torch.zeros([B, Nv, D, D], dtype=torch.float32, device=query_data.device)

    # helper data
    mask_data = torch.tril(-torch.ones([L, L], dtype=torch.float32, device=query_data.device), diagonal=-1)
    # lower triangular
    tril_mask_data = torch.ones([L, L], device=query_data.device).float().tril()
    eye_data = torch.eye(16, device=query_data.device).float()
    eye_data = eye_data.repeat(1, 8)

    inputs = {
        query_data: [0],
        key_data: [0],
        value_data: [0],
        beta_data: [0],
        gate_data: [0],
        state_data: [],
        mask_data: [],
        tril_mask_data: [],
        eye_data: [],
        act_seq_len: [0],
    }
    outputs = {core_attn_out_t: [0], last_state_data: []}

    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
    chunk_gated_delta_rule(*pto_inputs, *pto_outputs)

    return core_attn_out_t, last_state_data


@pytest.mark.skip(reason="large test case")
def test_chunk_gated_delta_rule():
    # pypto.set_compiler_monitor_options(
    #     enable=True,
    #     interval_sec=5,
    #     timeout_sec=20,
    #     total_timeout_sec=120,
    # )

    T = 128
    Nqk = 1
    Nv = 1
    D = 128
    act_seq_len = [0, 128]


    B = len(act_seq_len) - 1

    device_id = 0
    torch.npu.set_device(device_id)
    print("device id:", device_id)

    # prepare input data
    torch.manual_seed(0)
    query_data = torch.rand([T, Nqk, D], dtype=torch.float32, device=f'npu:{device_id}') * (1.3655 + 0.2785) - (1.3655 + 0.2785)
    key_data = torch.rand([T, Nqk, D], dtype=torch.float32, device=f'npu:{device_id}') * (1.4664 + 0.2785) - (1.4664 + 0.2785)
    value_data = torch.rand([T, Nv, D], dtype=torch.float32, device=f'npu:{device_id}') * (1.6488 + 0.2785) - (1.6488 + 0.2785)
    beta_data = torch.rand([T, Nv], dtype=torch.float32, device=f'npu:{device_id}') * (0.8927 - 0.0889) - (0.8927 - 0.0889)
    gate_data = torch.rand([T, Nv], dtype=torch.float32, device=f'npu:{device_id}') * (-0.1343 + 37.5452) - (-0.1343 + 37.5452)
    states_data = torch.zeros([B, Nv, D, D], dtype=torch.float32, device=f'npu:{device_id}')
    act_seq_len = torch.tensor(act_seq_len, dtype=torch.int32, device=f'npu:{device_id}')

    # calculate torch result
    core_attn_out_torch, final_state_torch = segs_chunk_gated_delta_rule(
        query_data.clone(),
        key_data.clone(),
        value_data.clone(),
        gate_data.clone(),
        beta_data.clone(),
        initial_state=states_data.clone(),
        act_seq_len=act_seq_len.clone(),
    )
    print("finish torch")
    # calculate pypto result
    inputs = [
        query_data,
        key_data,
        value_data,
        beta_data,
        gate_data,
        states_data,
        act_seq_len,
    ]
    core_attn_out_pypto, final_state_pypto = pypto_chunk_gated_delta_rule(*inputs)

    # compare results
    detailed_tensor_compare(core_attn_out_torch, core_attn_out_pypto)
    detailed_tensor_compare(final_state_torch, final_state_pypto)


def segs_chunk_gated_delta_rule(
    query_z,
    key,
    value,
    g,
    beta,
    act_seq_len,
    chunk_size_t=128,
    initial_state=None,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
):
    t, n1, d = query_z.shape
    t, n, d = value.shape
    batch = act_seq_len.shape[0] - 1

    query_z = query_z.repeat_interleave(n // n1, dim=1)
    key = key.repeat_interleave(n // n1, dim=1)

    final_state = torch.zeros([batch, n, d, d], dtype=torch.float32, device=query_z.device)

    query_z, key, value, beta, g = [
        x.transpose(0, 1).contiguous().to(torch.float32) for x in (query_z, key, value, beta, g)
    ]
    final_attn = torch.zeros([t, n, d], dtype=torch.float32, device=query_z.device)

    for b_idx in range(batch):
        s = act_seq_len[b_idx + 1] - act_seq_len[b_idx]
        b_ofs = act_seq_len[b_idx]
        l = 128
        c = max(1, s // l)
        seg_s = 128
        # assert s % seg_s == 0
        pad_size_zyt = (chunk_size_t - s % chunk_size_t) % chunk_size_t
        pad_seq_length = s + pad_size_zyt
        batch_query = F.pad(query_z[:, b_ofs : b_ofs + s], (0, 0, 0, pad_size_zyt))
        batch_key = F.pad(key[:, b_ofs : b_ofs + s], (0, 0, 0, pad_size_zyt))
        batch_value = F.pad(value[:, b_ofs : b_ofs + s], (0, 0, 0, pad_size_zyt))
        batch_beta = F.pad(beta[:, b_ofs : b_ofs + s], (0, pad_size_zyt))
        batch_g = F.pad(g[:, b_ofs : b_ofs + s], (0, pad_size_zyt))
        result_list = []
        recurrent_state = initial_state[b_idx : b_idx + 1, ...]
        for s_idx in range(0, pad_seq_length, seg_s):
            chunk_query = batch_query[:, s_idx : s_idx + seg_s, :].reshape(1, n, seg_s, d)
            chunk_key = batch_key[:, s_idx : s_idx + seg_s, :].reshape(1, n, seg_s, d)
            chunk_value = batch_value[:, s_idx : s_idx + seg_s, :].reshape(1, n, seg_s, d)
            chunk_gate = batch_g[:, s_idx : s_idx + seg_s].reshape(1, n, seg_s)
            chunk_beta = batch_beta[:, s_idx : s_idx + seg_s].reshape(1, n, seg_s)
            cur_attn, cur_state = torch_chunk_gated_delta_rule(
                chunk_query,
                chunk_key,
                chunk_value,
                chunk_gate,
                chunk_beta,
                chunk_size_t,
                recurrent_state,
                output_final_state,
                use_qk_l2norm_in_kernel,
            )
            result_list.append(cur_attn.squeeze(0))
            recurrent_state = cur_state
        batch_attn = torch.cat(result_list, dim=0)[:s]
        final_attn[b_ofs : b_ofs + s] = batch_attn
        final_state[b_idx : b_idx + 1, ...] = recurrent_state
    return final_attn, final_state


def torch_chunk_gated_delta_rule(
    query_z,
    key,
    value,
    g,
    beta,
    chunk_size_t=128,
    initial_state=None,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
):
    b, n, s, d = value.shape
    l = 128
    c = max(1, s // l)

    initial_state = initial_state.transpose(3, 2)
    initial_dtype = query_z.dtype
    if use_qk_l2norm_in_kernel:
        query_z = query_z * torch.rsqrt((query_z * query_z).sum(dim=-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size_zyt = (chunk_size_t - sequence_length % chunk_size_t) % chunk_size_t
    query_z = F.pad(query_z, (0, 0, 0, pad_size_zyt))
    key = F.pad(key, (0, 0, 0, pad_size_zyt))
    value = F.pad(value, (0, 0, 0, pad_size_zyt))
    beta = F.pad(beta, (0, pad_size_zyt))
    g = F.pad(g, (0, pad_size_zyt))

    total_sequence_length = sequence_length + pad_size_zyt
    scale = 1 / (query_z.shape[-1] ** 0.5)
    query_z = query_z * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query_z, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size_t, x.shape[-1])
        for x in (query_z, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size_t)
    mask = torch.triu(torch.ones(chunk_size_t, chunk_size_t, dtype=torch.bool, device=query_z.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)  # cal_cumsum

    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril() # cal_decay_mask

    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(
        mask, 0
    )  # cal_pre_attn

    for i in range(1, chunk_size_t):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(
        chunk_size_t, dtype=attn.dtype, device=attn.device
    )  # cal_inverse

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))  # cal_value_and_kcumdecay

    last_recurrent_state_t = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=query_z.device).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    core_attn_out_t = torch.zeros_like(value).to(query_z.device)
    mask = torch.triu(torch.ones(chunk_size_t, chunk_size_t, dtype=torch.bool, device=query_z.device), diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size_t):
        q_i, k_i, v_i = query_z[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime_t = (k_cumdecay[:, :, i]) @ last_recurrent_state_t
        v_new = v_i - v_prime_t
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state_t
        core_attn_out_t[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state_t = (
            last_recurrent_state_t * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state_t = None
    core_attn_out_t = core_attn_out_t.reshape(core_attn_out_t.shape[0], core_attn_out_t.shape[1], -1, core_attn_out_t.shape[-1])
    core_attn_out_t = core_attn_out_t[:, :, :sequence_length]
    core_attn_out_t = core_attn_out_t.transpose(1, 2).contiguous()
    last_recurrent_state_t = last_recurrent_state_t.transpose(3, 2)

    return core_attn_out_t, last_recurrent_state_t


def detailed_tensor_compare(
    tensor1, tensor2, rtol=1e-3, atol=1e-3, verbose=True, max_outliers_display=20
):
    """
    Detailed tensor comparison.

    Parameters
    ---------
    tensor1: first tensor
    tensor2: second tensor
    rtol: relative tolerance
    atol: absolute tolerance
    verbose: print details
    max_outliers_display: Maximum number of elements exceeding tolerance displayed

    Return
    ---------
    dict: dictionary containing the comparison results
    """
    t1, t2 = tensor1.cpu().float(), tensor2.cpu().float()

    # calculate differences
    diff = torch.abs(t1 - t2)
    relative_diff = diff / (torch.abs(t2) + 1e-8)  # avoid division by zero

    # tolerance check
    tolerance_mask = diff <= atol + rtol * torch.abs(t2)
    out_of_tolerance_mask = ~tolerance_mask

    # collect information statistics
    total_elements = t1.numel()
    out_of_tolerance_count = out_of_tolerance_mask.sum().item()
    out_of_tolerance_ratio = out_of_tolerance_count / total_elements

    # count differences
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    std_diff = torch.std(diff).item()

    # count differences out of tolerance
    if out_of_tolerance_count > 0:
        out_of_tolerance_diff = diff[out_of_tolerance_mask]
        max_out_diff = torch.max(out_of_tolerance_diff).item()
        mean_out_diff = torch.mean(out_of_tolerance_diff).item()

        # obtain the index and value out of tolerance
        outlier_indices = torch.nonzero(out_of_tolerance_mask, as_tuple=True)
        outlier_values1 = t1[out_of_tolerance_mask]
        outlier_values2 = t2[out_of_tolerance_mask]
        outlier_diffs = diff[out_of_tolerance_mask]
        outlier_relative_diffs = relative_diff[out_of_tolerance_mask]

        # sort by the difference (from largest to smallest)
        sorted_indices = torch.argsort(outlier_diffs, descending=True)
        sorted_outlier_indices = tuple(ind[sorted_indices] for ind in outlier_indices)
        sorted_outlier_values1 = outlier_values1[sorted_indices]
        sorted_outlier_values2 = outlier_values2[sorted_indices]
        sorted_outlier_diffs = outlier_diffs[sorted_indices]
        sorted_outlier_relative_diffs = outlier_relative_diffs[sorted_indices]

    else:
        max_out_diff = 0.0
        mean_out_diff = 0.0
        sorted_outlier_indices = None
        sorted_outlier_values1 = None
        sorted_outlier_values2 = None
        sorted_outlier_diffs = None
        sorted_outlier_relative_diffs = None

    result = {
        "total_elements": total_elements,
        "out_of_tolerance_count": out_of_tolerance_count,
        "out_of_tolerance_ratio": out_of_tolerance_ratio,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "max_out_of_tolerance_diff": max_out_diff,
        "mean_out_of_tolerance_diff": mean_out_diff,
        "all_close": out_of_tolerance_count == 0,
        "tolerance_mask": tolerance_mask,
        "diff_tensor": diff,
        "outlier_indices": sorted_outlier_indices,
        "outlier_values1": sorted_outlier_values1,
        "outlier_values2": sorted_outlier_values2,
        "outlier_diffs": sorted_outlier_diffs,
        "outlier_relative_diffs": sorted_outlier_relative_diffs,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("📊 Detailed tensor comparison report")
        print("=" * 60)
        print(f"Total elements: {total_elements:,}")
        print(f"Out of tolerance count: {out_of_tolerance_count:,}")
        print(
            f"Out of tolerance ratio: {out_of_tolerance_ratio:.6f} ({out_of_tolerance_ratio*100:.4f}%)"
        )
        print(f"Maximum difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        print(f"Standard difference: {std_diff:.6f}")
        print(f"Relative tolerance: rtol={rtol}, atol={atol}")

        if out_of_tolerance_count > 0:
            print(f"Maximum difference exceeding the tolerance: {max_out_diff:.6f}")
            print(f"Mean difference exceeding the tolerance: {mean_out_diff:.6f}")

            # display out of tolerance details
            print(
                f"\nDetails of elements out of tolerance (displaying the first {min(max_outliers_display, out_of_tolerance_count)} elements):"
            )
            print("-" * 80)
            print(f"{'Index':<20} {'Tensor1':<15} {'Tensor2':<15} {'Absolute tolerance':<12} {'Relative tolerance':<12}")
            print("-" * 80)

            for i in range(min(max_outliers_display, out_of_tolerance_count)):
                idx_str = str(tuple(sorted_outlier_indices[j][i].item() for j in range(len(sorted_outlier_indices))))
                print(f"{idx_str:<20} {sorted_outlier_values1[i].item():<15.6f} {sorted_outlier_values2[i].item():<15.6f} "
                      f"{sorted_outlier_diffs[i].item():<12.6f} {sorted_outlier_relative_diffs[i].item():<12.6f}")

            if out_of_tolerance_count > max_outliers_display:
                print(f"... There are {out_of_tolerance_count - max_outliers_display} out of tolerance elements not shown.")

        print(f"\n✅ Tensor Matching: {result['all_close']}")
        print("=" * 60)

    return result


if __name__ == "__main__":
    test_chunk_gated_delta_rule()
