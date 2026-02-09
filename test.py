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

def detailed_tensor_compare(tensor1, tensor2, rtol=1e-3, atol=1e-3, verbose=True, max_outliers_display=20):
    """
    详细的张量比较，分析不在容差范围内的元素比例，并显示超出容差的具体信息

    Args:
        tensor1: 第一个张量
        tensor2: 第二个张量
        rtol: 相对容差
        atol: 绝对容差
        verbose: 是否打印详细信息
        max_outliers_display: 最大显示的超出容差的元素数量

    Returns:
        dict: 包含比较结果的字典
    """
    # 确保张量可以比较
    t1, t2 = tensor1.cpu().float(), tensor2.cpu().float()

    # 计算差异
    diff = torch.abs(t1 - t2)
    relative_diff = diff / (torch.abs(t2) + 1e-8)  # 避免除零

    # 容差检查
    tolerance_mask = diff <= atol + rtol * torch.abs(t2)
    out_of_tolerance_mask = ~tolerance_mask

    # 统计信息
    total_elements = t1.numel()
    out_of_tolerance_count = out_of_tolerance_mask.sum().item()
    out_of_tolerance_ratio = out_of_tolerance_count / total_elements

    # 差异统计
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    std_diff = torch.std(diff).item()

    # 超出容差的差异统计
    if out_of_tolerance_count > 0:
        out_of_tolerance_diff = diff[out_of_tolerance_mask]
        max_out_diff = torch.max(out_of_tolerance_diff).item()
        mean_out_diff = torch.mean(out_of_tolerance_diff).item()

        # 获取超出容差的索引和值
        outlier_indices = torch.nonzero(out_of_tolerance_mask, as_tuple=True)
        outlier_values1 = t1[out_of_tolerance_mask]
        outlier_values2 = t2[out_of_tolerance_mask]
        outlier_diffs = diff[out_of_tolerance_mask]
        outlier_relative_diffs = relative_diff[out_of_tolerance_mask]

        # 按差异大小排序（从大到小）
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
        'total_elements': total_elements,
        'out_of_tolerance_count': out_of_tolerance_count,
        'out_of_tolerance_ratio': out_of_tolerance_ratio,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'max_out_of_tolerance_diff': max_out_diff,
        'mean_out_of_tolerance_diff': mean_out_diff,
        'all_close': out_of_tolerance_count == 0,
        'tolerance_mask': tolerance_mask,
        'diff_tensor': diff,
        'outlier_indices': sorted_outlier_indices,
        'outlier_values1': sorted_outlier_values1,
        'outlier_values2': sorted_outlier_values2,
        'outlier_diffs': sorted_outlier_diffs,
        'outlier_relative_diffs': sorted_outlier_relative_diffs
    }

    if verbose:
        print("\n" + "="*60)
        print("📊 张量详细比较报告")
        print("="*60)
        print(f"总元素数量: {total_elements:,}")
        print(f"超出容差元素数量: {out_of_tolerance_count:,}")
        print(f"超出容差比例: {out_of_tolerance_ratio:.6f} ({out_of_tolerance_ratio*100:.4f}%)")
        print(f"最大差异: {max_diff:.6f}")
        print(f"平均差异: {mean_diff:.6f}")
        print(f"差异标准差: {std_diff:.6f}")
        print(f"容差设置: rtol={rtol}, atol={atol}")

        if out_of_tolerance_count > 0:
            print(f"超出容差的最大差异: {max_out_diff:.6f}")
            print(f"超出容差的平均差异: {mean_out_diff:.6f}")

            # 显示超出容差的详细信息
            print(f"\n🔍 超出容差的元素详情 (显示前{min(max_outliers_display, out_of_tolerance_count)}个):")
            print("-" * 80)
            print(f"{'索引':<20} {'Tensor1值':<15} {'Tensor2值':<15} {'绝对差异':<12} {'相对差异':<12}")
            print("-" * 80)

            for i in range(min(max_outliers_display, out_of_tolerance_count)):
                idx_str = str(tuple(sorted_outlier_indices[j][i].item() for j in range(len(sorted_outlier_indices))))
                print(f"{idx_str:<20} {sorted_outlier_values1[i].item():<15.6f} {sorted_outlier_values2[i].item():<15.6f} "
                      f"{sorted_outlier_diffs[i].item():<12.6f} {sorted_outlier_relative_diffs[i].item():<12.6f}")

            if out_of_tolerance_count > max_outliers_display:
                print(f"... 还有 {out_of_tolerance_count - max_outliers_display} 个超出容差的元素未显示")

        print(f"\n✅ 张量匹配: {result['all_close']}")
        print("="*60)

    return result

def print_aligned_matrix(tensor, precision=4):
    """打印对齐的矩阵"""
    if tensor.dim() != 2:
        print("仅支持2D张量")
        return

    # 转换为字符串并找到最大长度
    str_matrix = []
    max_len = 0
    for i in range(tensor.shape[0]):
        row_strs = []
        for j in range(tensor.shape[1]):
            s = f"{tensor[i, j].item():.{precision}f}"
            row_strs.append(s)
            max_len = max(max_len, len(s))
        str_matrix.append(row_strs)

    # 打印对齐的矩阵
    for i in range(tensor.shape[0]):
        print("[", end="")
        for j in range(tensor.shape[1]):
            print(f"{str_matrix[i][j]:>{max_len}}", end="")
            if j < tensor.shape[1] - 1:
                print(", ", end="")
        print("]")


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
    L = gate_view.shape[0]
    actual_L = gate_view.valid_shape[0]
    pypto.set_vec_tile_shapes(128, 128)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    # cal_cumsum
    gate_cum = pypto.matmul(tril, gate_view, pypto.DT_FP32) #[L,1],精度通过，应该是不会报错
    # cal_decay_mask
    # ! old
    # decay_mask = ((gate_cum - gate_cum.transpose(0, 1)) * tril).exp() * tril #[L,L]

    gate_T = gate_cum.transpose(0, 1)
    gate_T_view = gate_T.view([1, L], [0, 0], valid_shape=[1, actual_L])
    g_sub = gate_cum - gate_T_view

    # g_sub_view = pypto.view(g_sub, [L, L], [0, 0], valid_shape=[actual_L, actual_L])
    decay_mask = (g_sub * tril).exp() * tril #[L,L]


    # ! new
    # g_sub = gate_cum - gate_cum.transpose(0,1)

    # L = gate_view.shape[0]
    # actual_L = gate_view.valid_shape[0]
    # g_sub_view = pypto.view(g_sub, [L, L], [0, 0], valid_shape=[actual_L, actual_L])
    # #pypto.pass_verify_print("=====g_sub_view\n", g_sub_view)
    # #pypto.pass_verify_print("=====tril\n", tril)
    # g_tril_exp = (g_sub_view * tril).exp()
    # # pypto.pass_verify_print("=====g_tril_exp\n", g_tril_exp)
    # decay_mask = (g_tril_exp * tril).exp() * tril #[L,L]

    # beta_k
    key_beta = key_view_2d * beta_view #[L,D]
    # kkt计算
    kkt = pypto.matmul(key_beta, key_view_2d, pypto.DT_FP32, b_trans=True) #[L,L]
    A = kkt * decay_mask * mask #[L,L]

    return gate_cum, decay_mask, A, key_beta


def inverse_pto_min_length(
    attn: pypto.Tensor,
    eye: pypto.Tensor,
    min_length: int) -> None:
    # len = 30 = 16 + 14
    actual_L = attn.valid_shape[0]
    attn_inv_list = {}
    # attn_inv_list[1] = attn[:2, :] # [2, 14]
    attn_inv_list[1] = attn.view([2, min_length], [0, 0], valid_shape=[2, actual_L])
    attn_initial = pypto.tensor(attn.shape, dtype=attn.dtype)

    pypto.assemble(attn, [0, 0], attn_initial)

    pypto.set_vec_tile_shapes(128, 128)
    attn_transpose = attn.transpose(dim0=0, dim1=1)

    pypto.set_pass_options(sg_set_scope=1)
    for i in range(2, min_length, 1):
        row = attn_initial.view([1, min_length], [i, 0])
        row_expand = attn_transpose.view([i, 1], [0, i])
        prod = (row_expand * attn_inv_list[i - 1]).sum(0, keepdim=True)
        # attn_update = row + prod

        row_view = row.view(row.shape, [0, 0], valid_shape=[row.shape[0], actual_L])
        prod_view = prod.view(prod.shape, [0, 0], valid_shape=[prod.shape[0], actual_L])
        attn_update = row_view + prod_view

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

    attn_2_1_inv = (attn_2_2_inv @ attn_2_1 ) @ attn_1_1_inv

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


def recurrent_attn(
    query: pypto.Tensor,
    key: pypto.Tensor,
    decay_mask: pypto.Tensor,
    tril: pypto.Tensor,
    gate: pypto.Tensor,
    state: pypto.Tensor,
    value_new: pypto.Tensor) -> pypto.Tensor:
    """
    Calculate attention.

    Parameters
    ---------
    query: [L, D]
    key: [L, D]
    decay_mask: [L, L]
    tril: [L, L]
    gate: [L, 1]
    state: [D, D]
    value_new: [L, D]

    Return
     ---------
    chunk_attn_out: [L, D]
    """

    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    pypto.set_vec_tile_shapes(128, 128)

    attn = pypto.matmul(query, key, pypto.DT_FP32, b_trans=True) * decay_mask * tril # [L, L]
    query_mul_gate_exp = query * gate.exp()
    attn_inter = pypto.matmul(query_mul_gate_exp, state, pypto.DT_FP32, b_trans=True) # [L, D]
    chunk_attn_out = attn_inter + pypto.matmul(attn, value_new, pypto.DT_FP32) # [L, D]

    return chunk_attn_out


def recurrent_state(
    key:pypto.Tensor,
    value:pypto.Tensor,
    k_cumdecay:pypto.Tensor,
    gate:pypto.Tensor,
    state:pypto.Tensor)-> tuple[pypto.Tensor, pypto.Tensor]:
    """
    Calculate state

    Parameters
    ——————————
    key:[L, Dk]
    value:[L, Dv]
    k_cumdecay:[L, Dk]
    gate:[L, 1]
    state:[Dv, Dk]

    Return
    ——————————
    v_new:[L, Dv]
    state_new:[Dv, Dk]
    """

    Dv = value.shape[-1]
    L = gate.shape[0]

    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    pypto.set_vec_tile_shapes(128, 128)

    v_prime = pypto.matmul(k_cumdecay, state, pypto.DT_FP32, b_trans=True) # [L, Dv]
    v_new = value - v_prime # [L, Dv]
    _last_gate_2 = pypto.expand_clone(gate[L-1:L, :], [Dv, 1]).exp() # [Dv, 1]
    final_state_1 = state * _last_gate_2 # [Dv, Dk]
    _last_gate_1 = pypto.expand_clone(gate[L-1:L, :], [L, 1]) # [L, 1]
    temp_matmul_result_right = key * (_last_gate_1 - gate).exp() # [L, Dk]
    temp_matmul_result = pypto.matmul(v_new, temp_matmul_result_right, pypto.DT_FP32, a_trans=True) # [Dv, Dk]
    state_new = final_state_1 + temp_matmul_result # [Dv, Dk]

    return v_new, state_new


def recurrent_pre(
    query: pypto.Tensor,
    key: pypto.Tensor,
    value: pypto.Tensor,
    gate: pypto.Tensor,
    decay_mask: pypto.Tensor,
    tril: pypto.Tensor) -> tuple[pypto.Tensor, pypto.Tensor]:
    """
    Calculate attention.

    Parameters
    ---------
    query: [L, D]
    key: [L, D]
    value:[L, Dv]
    gate: [L, 1]
    decay_mask: [L, L]
    tril: [L, L]

    Return
    ----------
    gate_2: [Dv, 1]
    temp_matmul_result_right: [L, Dk]
    attn: [L, L]
    query_mul_gate_exp: [L, D]
    chunk_attn_out: [L, D]
    """

    Dv = value.shape[-1]
    L = gate.shape[0]
    gate_exp = gate.exp()
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    pypto.set_vec_tile_shapes(64, 128)
    gate_2 = pypto.expand_clone(gate_exp[L-1:L, :], [Dv, 1]) # [Dv, 1]
    gate_1 = pypto.expand_clone(gate[L-1:L, :], [L, 1]) # [L, 1]
    temp_matmul_result_right = key * (gate_1 - gate).exp() # [L, Dk]
    attn = pypto.matmul(query, key, pypto.DT_FP32, b_trans=True) * decay_mask * tril # [L, L]
    query_mul_gate_exp = query * gate_exp
    return gate_2, temp_matmul_result_right, attn, query_mul_gate_exp


def recurrent_state_attn(
    k_cumdecay,
    gate_2,
    value,
    matmul_result_right,
    attn,
    query_mul_gate_exp,
    state
):
    """
    Calculate attention.

    Parameters
    ---------
    k_cumdecay: [L, Dk]
    gate_2: [Dv, 1]
    value:[L, Dv]
    matmul_result_right: [L, Dk]
    attn: [L, L]
    query_mul_gate_exp: [L, D]
    state: [Dv, Dk]

    Return
    ----------
    state_new: [Dv, Dk]
    """

    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    pypto.set_vec_tile_shapes(32, 128)
    pypto.set_pass_options(sg_set_scope=1)
    v_prime = pypto.matmul(k_cumdecay, state, pypto.DT_FP32, b_trans=True) # [L, Dv]
    temp_matmul_vprime = pypto.matmul(v_prime, matmul_result_right, pypto.DT_FP32, a_trans=True)
    temp_matmul_value = pypto.matmul(value, matmul_result_right, pypto.DT_FP32, a_trans=True)
    temp_chunk_vprime = pypto.matmul(attn, v_prime, pypto.DT_FP32)
    temp_chunk_value = pypto.matmul(attn, value, pypto.DT_FP32)
    # v_new = value - v_prime # [L, Dv]
    final_state_1 = state * gate_2 # [Dv, Dk]
    temp_matmul_result = temp_matmul_value - temp_matmul_vprime # [Dv, Dk]
    state_new = final_state_1 + temp_matmul_result # [Dv, Dk]
    attn_inter = pypto.matmul(query_mul_gate_exp, state, pypto.DT_FP32, b_trans=True) # [L, D]
    chunk_attn_out = attn_inter + temp_chunk_value - temp_chunk_vprime # [L, D]
    pypto.set_pass_options(sg_set_scope=-1)
    return state_new, chunk_attn_out


def recurrent_state_attn_all(
    query: pypto.Tensor,
    key: pypto.Tensor,
    value: pypto.Tensor,
    k_cumdecay: pypto.Tensor,
    gate: pypto.Tensor,
    state: pypto.Tensor,
    decay_mask: pypto.Tensor,
    tril: pypto.Tensor) -> tuple[pypto.Tensor, pypto.Tensor]:
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
    ----------
    chunk_attn_out: [L, D]
    state_new:[Dv, Dk]
    """
    Dv = value.shape[-1]
    L = gate.shape[0]
    actual_L = gate.valid_shape[0]
    gate_exp = gate.exp()
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    pypto.set_vec_tile_shapes(128, 128)
    # _last_gate_1 = gate[L-1:L, :]
    _last_gate_1 = gate[actual_L-1:actual_L, :] # [1, x] -》 [0, x]
    temp_matmul_result_right = key * (_last_gate_1 - gate).exp() # [L, Dk]
    v_prime = pypto.matmul(k_cumdecay, state, pypto.DT_FP32, b_trans=True) # [L, Dv]
    temp_matmul_vprime = pypto.matmul(v_prime, temp_matmul_result_right, pypto.DT_FP32, a_trans=True)
    temp_matmul_value = pypto.matmul(value, temp_matmul_result_right, pypto.DT_FP32, a_trans=True)
    # v_new = value - v_prime # [L, Dv]
    query_mul_gate_exp = query * gate_exp
    pypto.set_vec_tile_shapes(32, 128)
    temp_matmul_result = temp_matmul_value - temp_matmul_vprime # [Dv, Dk]
    _last_gate_2 = pypto.expand_clone(gate_exp[actual_L-1:actual_L, :], (Dv, 1)) # [Dv, 1]
    final_state_1 = state * _last_gate_2 # [L, L]-》[x, x]  [L, L]-> [L, L]
    state_new = final_state_1 + temp_matmul_result
    pypto.set_vec_tile_shapes(128, 128)
    attn_inter = pypto.matmul(query_mul_gate_exp, state, pypto.DT_FP32, b_trans=True) # [L, D]
    attn = pypto.matmul(query, key, pypto.DT_FP32, b_trans=True)
    # print(f"===== {decay_mask.valid_shape=}")
    decay_mask_view = decay_mask.view(decay_mask.shape, [0, 0], valid_shape=[actual_L, actual_L])
    tril_view = tril.view(tril.shape, [0, 0], valid_shape=[actual_L, actual_L])
    attn_tmp = attn * decay_mask_view * tril_view # [L, L]
    chunk_attn_value = pypto.matmul(attn_tmp, value, pypto.DT_FP32)
    chunk_attn_vprime = pypto.matmul(attn_tmp, v_prime, pypto.DT_FP32)
    chunk_attn_out = attn_inter + chunk_attn_value - chunk_attn_vprime
    return chunk_attn_out, state_new


verify_options = {
    "enable_pass_verify": True,
    "pass_verify_save_tensor": True,
}
@pypto.jit(
    runtime_options={
        "stitch_function_inner_memory": 128*16,
        # "stitch_function_num_initial": 512,
        "stitch_function_num_initial": 128,
        "stitch_function_outcast_memory": 128*16
        },
    debug_options={"runtime_debug_mode": 1},
    verify_options=verify_options
)
def chunk_gated_delta_rule(query, key, value, beta, gate, states, mask,
    tril_mask, eye, act_seq_len,eye_chunk,dump_k_beta,dump_gate,dump_decay):

    _, Nqk, D = query.shape
    _, Nv, D = value.shape
    B = states.shape[0]
    L, L = mask.shape
    group = Nv // Nqk
    for b_idx in pypto.loop(B, name="LOOP_B_TND", idx_name="b_idx"):
        if pypto.cond(pypto.is_loop_begin(b_idx)):
            S = act_seq_len[b_idx]
            b_ofs = 0
        else:
            S = act_seq_len[b_idx] - act_seq_len[b_idx-1]
            b_ofs = act_seq_len[b_idx-1]
        for nv_idx in pypto.loop(Nv, name="LOOP_Nv_TND", idx_name="nv_idx"):
            nqk_idx = nv_idx // group
            last_state = pypto.tensor([D, D], dtype=states.dtype)
            for s_idx in pypto.loop(0, S, L, name="LOOP_S_TND", idx_name="s_idx"):
                bs_ofs = b_ofs + s_idx
                actual_L = (S - s_idx).min(L)
                ## view
                if pypto.cond(pypto.is_loop_begin(s_idx)):
                    pypto.set_vec_tile_shapes(16, 16, 128, 128)
                    last_state[:] = states[b_idx, nv_idx]
                query_view = pypto.view(query, [L, 1, D], [bs_ofs, nqk_idx, 0], valid_shape =[actual_L, 1, D])
                key_view = pypto.view(key, [L, 1, D], [bs_ofs, nqk_idx, 0], valid_shape =[actual_L, 1, D])
                value_view = pypto.view(value, [L, 1, D], [bs_ofs, nv_idx, 0], valid_shape =[actual_L, 1, D])
                beta_view = pypto.view(beta, [L, 1], [bs_ofs, nv_idx], valid_shape =[actual_L, 1])
                gate_view = pypto.view(gate, [L, 1], [bs_ofs, nv_idx], valid_shape =[actual_L, 1])

                pypto.set_vec_tile_shapes(128, 128, 128)
                query_view_2d = pypto.reshape(query_view, [L, D], valid_shape=[actual_L, D])
                key_view_2d = pypto.reshape(key_view, [L, D], valid_shape=[actual_L, D])
                value_view_2d = pypto.reshape(value_view, [L, D], valid_shape=[actual_L, D])

                mask_view = pypto.view(mask, [L, L], [0, 0], valid_shape=[actual_L, actual_L])
                tril_mask_view = pypto.view(tril_mask, [L, L], [0, 0], valid_shape=[actual_L, actual_L])

                # compute
                # qk_l2norm
                query_norm, key_norm = l2norm(query_view_2d, key_view_2d)
                # query_norm = l2norm_v2(query_view_2d)
                # key_norm = l2norm_v2(key_view_2d)
                scale = 1 / D ** 0.5
                query_scale = query_norm * scale

                # kv_beta & g_cumsum & decay_mask & pre_attn
               # gate_cum, decay_mask, A_block, key_beta = pre_attn(gate_view, key_norm, beta_view, tril_mask_view, mask_view)
                pypto.set_vec_tile_shapes(128, 128, 128)
                A_block = pypto.view(tril_mask, [L, L], [0, 0], valid_shape=[actual_L, actual_L])
                key_beta =  pypto.reshape(value_view, [L, D], valid_shape=[actual_L, D])
                decay_mask =  pypto.view(tril_mask, [L, L], [0, 0], valid_shape=[actual_L, actual_L])
  
                A_block_view = A_block.reshape([1, L, L], valid_shape=[1, actual_L, actual_L]) #
                key_beta_view = key_beta.reshape([1, L, D], valid_shape=[1, actual_L, D]) #报错
                # gate_cum_view = gate_cum.reshape([1, L], valid_shape=[1, actual_L])
                decay_mask_view = decay_mask.reshape([1, L, L], valid_shape=[1, actual_L, actual_L])


                # TODO reshape然后assemble
                # eye_chunk[nv_idx] = A_block_view
                pypto.assemble(A_block_view, [nv_idx, 0, 0], eye_chunk)
                pypto.assemble(key_beta_view, [nv_idx, 0, 0], dump_k_beta)
                # pypto.assemble(gate_cum_view, [nv_idx, 0], dump_gate) #true
                pypto.assemble(decay_mask_view, [nv_idx, 0, 0], dump_decay)

           


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
    core_attn_out: [T, Nv, D]
    state_data: [B, Nv, D, D]
    """

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

    T, Nv, D = value_data.shape
    L = 128
    B = state_data.shape[0]

    # output
    core_attn_out = torch.ones([T, Nv, D], dtype=torch.bfloat16, device=query_data.device)
    last_state_data = torch.zeros([B, Nv, D, D], dtype=torch.float32, device=query_data.device)

    # helper data
    mask_data = torch.tril(-torch.ones([L, L], dtype=torch.float32, device=query_data.device), diagonal=-1)
    tril_mask_data = torch.ones([L, L], device=query_data.device).float().tril() # lower triangular
    eye_data = torch.eye(16, device=query_data.device).float()
   # eye_chunk = torch.eye(L, device=query_data.device).float()

    eye_chunk = torch.zeros([Nv, L,L], device=query_data.device).float()
    dump_decay =torch.zeros([Nv, L,L], device=query_data.device).float()
    dump_gate = torch.zeros([Nv, L], device=query_data.device).float()  #数据需要改
    dump_k_beta = torch.zeros([Nv, L, D], device=query_data.device).float()


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
    outputs = {

        eye_chunk:[],
        dump_decay:[],
        dump_gate:[],
        dump_k_beta:[]

    }

    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
    chunk_gated_delta_rule(*pto_inputs, *pto_outputs)

    return  eye_chunk,dump_decay,dump_gate,dump_k_beta


def test_chunk_gated_delta_rule(T_len):

    T = 30
    B = 1
    Nqk = 1
    Nv = 1
    D = 128
    act_seq_len = [0, T]
    # B = len(act_seq_len)
    print(f"run T = {T}")
    device_id = 5
    torch.npu.set_device(device_id)
    print("device id:", device_id)

    # ! prepare inputs data
    torch.manual_seed(0)
    query_data = torch.rand([T, Nqk, D], dtype=torch.float32, device=f'npu:{device_id}') * (1.3655 + 0.2785) - (1.3655 + 0.2785)
    key_data = torch.rand([T, Nqk, D], dtype=torch.float32, device=f'npu:{device_id}') * (1.4664 + 0.2785) - (1.4664 + 0.2785)
    value_data = torch.rand([T, Nv, D], dtype=torch.float32, device=f'npu:{device_id}') * (1.6488 + 0.2785) - (1.6488 + 0.2785)
    beta_data = torch.rand([T, Nv], dtype=torch.float32, device=f'npu:{device_id}') * (0.8927 - 0.0889) - (0.8927 - 0.0889)
    gate_data = torch.rand([T, Nv], dtype=torch.float32, device=f'npu:{device_id}') * (-0.1343 + 37.5452) - (-0.1343 + 37.5452)
    states_data = torch.zeros([B, Nv, D, D], dtype=torch.float32, device=f'npu:{device_id}')
    last_state_data = torch.zeros([B, Nv, D, D], dtype=torch.float32, device=f'npu:{device_id}')
    act_seq_len = torch.tensor(act_seq_len, dtype=torch.int32, device=f'npu:{device_id}')

    print("=========1===============")
    print("query_data.shape", query_data.shape)
    print("key_data.shape", key_data.shape)

    print("value_data.shape", value_data.shape)
    print("beta_data.shape", beta_data.shape)
    print("gate_data.shape", gate_data.shape)
    print("states_data.shape", states_data.shape)
    print("act_seq_len.shape", act_seq_len.shape)
    print("act_seq_len", act_seq_len)

    # ! calculate torch result


    query_data = query_data.view(-1, Nqk, D)
    key_data = key_data.view(-1, Nqk, D)
    value_data = value_data.view(-1, Nv, D)
    beta_data = beta_data.view(-1, Nv)
    gate_data = gate_data.view(-1, Nv)
    act_seq_len = act_seq_len[1:]

    # ! calculate pypto result
    inputs = [query_data, key_data, value_data, beta_data, gate_data, states_data, act_seq_len]
    # pypto.set_verify_golden_data(goldens=[None, None, None, None, None, None, None, None, None, None, golden1_cpu, golden2_cpu])
    eye_chunk ,dump_decay,dump_gate,dump_k_beta = pypto_chunk_gated_delta_rule(*inputs)


    print("============golden vs pto=========")

    print("\ncompare\n")





if __name__ == "__main__":
    test_chunk_gated_delta_rule(30)