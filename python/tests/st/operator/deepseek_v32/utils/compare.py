
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
"""
import os
import math
import logging
import torch
import torch_npu
import numpy as np
from numpy.testing import assert_allclose
import pypto


def compare(t: torch.Tensor, t_ref: torch.Tensor, name, atol, rtol, max_error_ratio=0.005, max_error_count=10):
    """
    比较两个张量的差异，超过阈值时打印错误点并抛出断言错误
    Args:
        t: 待比较张量
        t_ref: 参考张量
        name: 张量名称（用于日志）
        atol: 绝对容差
        rtol: 相对容差
        max_error_ratio: 误差点占总元素数的最大比例
        max_error_count: 显示的最大误差点数量（同时也是误差点阈值的上限）
    """
    def check_is_nan_inf():
        # ========== 核心新增：检测t中的NaN和Inf并直接报错 ==========
        # 1. 检测NaN
        nan_mask = torch.isnan(t)
        nan_count = nan_mask.sum().item()

        # 2. 检测Inf（包含+Inf和-Inf）
        inf_mask = torch.isinf(t)
        inf_count = inf_mask.sum().item()

        # 若存在NaN或Inf，拼接错误信息并报错
        if nan_count > 0 or inf_count > 0:
            error_msg = f"\n========== 张量 {name} 检测到非法值（禁止存在NaN/Inf）=========="

            # 打印NaN信息
            if nan_count > 0:
                nan_positions = torch.nonzero(nan_mask, as_tuple=False)
                show_nan_count = min(nan_count, max_error_count)
                error_msg += f"\n- NaN数量：{nan_count}，前 {show_nan_count} 个位置："
                for i in range(show_nan_count):
                    pos_tuple = tuple(p.item() for p in nan_positions[i])
                    error_msg += f"\n  位置 {pos_tuple}"

            # 打印Inf信息（区分+Inf/-Inf）
            if inf_count > 0:
                inf_positions = torch.nonzero(inf_mask, as_tuple=False)
                show_inf_count = min(inf_count, max_error_count)
                error_msg += f"\n- Inf数量：{inf_count}，前 {show_inf_count} 个位置（值类型）："
                for i in range(show_inf_count):
                    pos = inf_positions[i]
                    pos_tuple = tuple(p.item() for p in pos)
                    inf_val = t[pos_tuple].item()
                    inf_type = "+Inf" if inf_val == float('inf') else "-Inf"
                    error_msg += f"\n  位置 {pos_tuple}：{inf_type}"
            error_msg += "\n" + "=" * 80 + "\n"

            # 抛出断言错误，终止函数执行
            assert False, error_msg

    # check 是否是nan 或 inf
    check_is_nan_inf()

    # 先验证张量的基本属性一致
    assert t.shape == t_ref.shape, f"张量形状不一致：t.shape={t.shape}, t_ref.shape={t_ref.shape}"
    assert t.dtype == t_ref.dtype, f"张量数据类型不一致：t.dtype={t.dtype}, t_ref.dtype={t_ref.dtype}"
    assert t.device == t_ref.device, f"张量设备不一致：t.device={t.device}, t_ref.device={t_ref.device}"

    # 计算误差点数量的阈值（取比例计算值和最大数量的较小值）
    error_count_threshold = round(max_error_ratio * t_ref.numel())

    # 计算误差掩码（超过阈值的位置为True）
    diff_abs = (t - t_ref).abs()
    tolerance = atol + rtol * t_ref.abs()
    diff_mask = diff_abs > tolerance
    error_count = diff_mask.sum().item()

    # 计算最大误差和其位置
    max_diff, flat_max_pos = torch.max(diff_abs.flatten(), dim=0)
    max_pos = torch.unravel_index(flat_max_pos, t.shape)
    max_pos = tuple(idx.item() for idx in max_pos)

    # 打印错误点的逻辑（如果有误差点）
    if error_count > 0:
        print(f"\n========== 张量 {name} 存在 {error_count} 个误差点（阈值：{error_count_threshold}）==========")

        # 获取所有误差点的位置
        error_positions = torch.nonzero(diff_mask, as_tuple=False)  # shape: [error_count, dims]

        # 限制显示的误差点数量（避免数据量过大）
        show_count = min(error_count, max_error_count)
        print(f"显示前 {show_count} 个误差点（位置 | 待比较值 | 参考值 | 绝对误差 | 允许阈值）：")

        # 遍历前N个误差点打印详细信息
        for i in range(show_count):
            pos = error_positions[i]

            # 转换为元组格式的位置（如 (0, 2, 3)）
            pos_tuple = tuple(p.item() for p in pos)

            # 获取对应位置的数值
            t_val = t[pos_tuple].item()
            t_ref_val = t_ref[pos_tuple].item()
            diff_val = diff_abs[pos_tuple].item()
            tol_val = tolerance[pos_tuple].item()

            # 格式化输出，保留足够小数位
            print(f"  位置 {pos_tuple}: {t_val:.8f} vs {t_ref_val:.8f} | 误差={diff_val:.8f} | 阈值={tol_val:.8f}")

        # 打印最大误差点
        print(f"\n最大误差点：位置 {max_pos} | 误差={max_diff.item():.8f} | 阈值={tolerance[max_pos].item():.8f}")
        print("=" * 80 + "\n")

    # 断言误差点数量不超过阈值
    assert error_count <= error_count_threshold, \
        (f"compare fail: {name}, max diff: {max_diff.item():.8f} at {max_pos}, "
         f"error_count: {error_count}, error_count_threshold: {error_count_threshold}")


small_value_thres_dict = {
        torch.float16: 2**-11,
        torch.bfloat16: 2**-8,
        torch.float32: 2**-14,
        torch.uint8: 2**-4, torch.float8_e4m3fn: 2**-4
    }


small_value_error_thres_dict = {
        torch.float16: 2**-16,
        torch.bfloat16: 2**-16,
        torch.float32: 2**-30,
        torch.uint8: 2**-6, torch.float8_e4m3fn: 2**-6
    }


def get_split_index(golden_data, dtype):
    thres = small_value_thres_dict[dtype]
    large_mask = torch.abs(golden_data) >= thres
    small_mask = torch.abs(golden_data) < thres
    return large_mask, small_mask, thres


def compute_matrix_small_value(input_data, golden_data, dtype, small_mask):
    if not torch.any(small_mask):
        return 0
    thres = small_value_error_thres_dict[dtype]
    
    # 直接使用布尔掩码索引
    error_count = torch.sum(torch.abs(input_data[small_mask] - golden_data[small_mask]) > thres).item()
    return error_count


def compute_matrix_large_value(input_data, golden_data, large_mask):
    if not torch.any(large_mask):
        return 0, 0, 0, 0, 0
    
    input_large = input_data[large_mask]
    golden_large = golden_data[large_mask]
    
    abs_diff = torch.abs(input_large - golden_large)
    relative_error = abs_diff / (torch.abs(golden_large) + 1e-7)

    mare = torch.max(relative_error).item()
    mere = torch.mean(relative_error).item()
    rmse = torch.sqrt(torch.mean((input_large - golden_large) ** 2)).item()
    
    return mare, mere, rmse, relative_error, abs_diff


def compute_re_matrix(input_value, bm_value, small_value_thres):
    if math.isinf(bm_value) or math.isnan(bm_value):
        return 1
    if math.isinf(input_value) or math.isnan(input_value):
        return 1000
    return input_value / max(bm_value, small_value_thres)


def compute_re_triplet_matrix(npu_matrix, golden_matrix, small_value_thres):
    mare_npu, mere_npu, rmse_npu = npu_matrix
    mare_bm, mere_bm, rmse_bm = golden_matrix
    mare_matrix = compute_re_matrix(mare_npu, mare_bm, small_value_thres)
    mere_matrix = compute_re_matrix(mere_npu, mere_bm, small_value_thres)
    rmse_matrix = compute_re_matrix(rmse_npu, rmse_bm, small_value_thres)
    return mare_matrix, mere_matrix, rmse_matrix


def precision_compare_triple(npu_data, bm_data, golden_data, thres=(2, 1.2, 1.2)):
    dtype = npu_data.dtype
    if dtype in ["int8", "int32"]:
        raise NotImplementedError("precision compare triplet only support float")

    if dtype == torch.uint8:
        npu_data = torch_npu.npu_dtype_cast(npu_data, torch.float32, input_dtype=torch_npu.hifloat8)
        bm_data = torch_npu.npu_dtype_cast(bm_data, torch.float32, input_dtype=torch_npu.hifloat8)
        golden_data = torch_npu.npu_dtype_cast(golden_data, torch.float32, input_dtype=torch_npu.hifloat8)
    else:
        npu_data = npu_data.to(torch.float32)
        bm_data = bm_data.to(torch.float32)
        golden_data = golden_data.to(torch.float32)

    npu_data = npu_data.cpu()
    bm_data = bm_data.cpu()
    golden_data = golden_data.cpu()

    large_value_idx, small_value_idx, small_value_thres = get_split_index(golden_data, dtype)

    # 小值域场景
    npu_error_count = compute_matrix_small_value(npu_data, golden_data, dtype, small_value_idx)
    bm_error_count = compute_matrix_small_value(bm_data, golden_data, dtype, small_value_idx)
    small_value_matrix = npu_error_count / max(bm_error_count, 1)

    # 大值域场景
    mare_npu, mere_npu, rmse_npu, npu_relative_error, npu_absolute_error = compute_matrix_large_value(npu_data, golden_data, large_value_idx)
    mare_bm, mere_bm, rmse_bm, bm_relative_error, bm_absolute_error = compute_matrix_large_value(bm_data, golden_data, large_value_idx)
    mare_matrix, mere_matrix, rmse_matrix = compute_re_triplet_matrix([mare_npu, mere_npu, rmse_npu], [mare_bm, mere_bm, rmse_bm], small_value_thres)

    if small_value_matrix <= 2 and mare_matrix <= thres[0] and mere_matrix <= thres[1] and rmse_matrix <= thres[2]:
        result = "PASS"
    else:
        result = "FAILED"

    return result, mare_matrix, mere_matrix, rmse_matrix, small_value_matrix