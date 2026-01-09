#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
GLM-4.5 FFN Router Expert Quantization Module

This module implements the quantized FFN computation for router experts in MoE architecture.
Router experts are dynamically selected based on input token features, allowing the model
to use only a subset of experts while maintaining a large parameter count.

Main Functions:
    - ffn_router_expert_quant: Main function for router expert FFN quantization
    - moe_router_expert_main: JIT compiled kernel for router expert computation
    - expert_infer_base: Base inference function for a single expert
"""
import os
import torch
import torch_npu
import pypto
import numpy as np
from numpy.testing import assert_allclose
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
from utils.get_format import get_format


def main():
    test_ffn_router()


def ffn_router_torch_npu(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    group_list: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor
) -> torch.Tensor:
    group_list = group_list.to(torch.int64)
    group_list_cumsum = group_list.cumsum(dim=0)
    output_dtype = w2_scale.dtype
    w13_int8_nz = torch_npu.npu_format_cast(w13, 29)
    hidden_states, swiglu_out_scale, _ = torch_npu.npu_grouped_matmul_swiglu_quant(
                x=hidden_states,
                weight=w13_int8_nz,
                bias=None,
                group_list=group_list_cumsum,
                weight_scale=w13_scale,
                x_scale=hidden_states_scale)
    hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[w2],
            scale=[w2_scale],
            bias=None,
            per_token_scale=[swiglu_out_scale],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=group_list,
            output_dtype=output_dtype)[0]
    return hidden_states


def get_token_acc_table(group_list):
    assert len(group_list.shape) == 1
    return (torch.cumsum(group_list, dim=0) - group_list).to(group_list.dtype)


def ffn_golden_quan_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor per token (per row).

    Args:
        x: Input tensor to quantize

    Returns:
        Tuple of (quantized_int8_tensor, dequantization_scale)
    """
    x_fp32 = x.to(torch.float32)
    max_value = x_fp32.abs().max(dim=1, keepdim=True)[0]
    scale_quant = 127.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_rint = torch.round(y_fp32).to(torch.int32)
    y_round = torch.round(y_rint).to(torch.float16)
    y_int8 = torch.trunc(y_round).to(torch.int8)
    scale_dequant = (1 / scale_quant)
    return y_int8, scale_dequant


def ffn_golden_quan_per_channel_3d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor per channel (per column) for 3D tensors.

    Note: This function currently uses per-token quantization (dim=1)
    but is kept for backward compatibility.

    Args:
        x: Input tensor to quantize

    Returns:
        Tuple of (quantized_int8_tensor, dequantization_scale)
    """
    x_fp32 = x.to(torch.float32)
    max_value = x_fp32.abs().max(dim=1, keepdim=True)[0]
    scale_quant = 127.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_rint = torch.round(y_fp32).to(torch.int32)
    y_round = torch.round(y_rint).to(torch.float16)
    y_int8 = torch.trunc(y_round).to(torch.int8)
    scale_dequant = (1 / scale_quant)
    return y_int8, scale_dequant


def gen_input(
    b: int,
    s: int,
    topk: int,
    per_expert_num: int,
    hidden_size: int,
    intermediate_size: int,
    dtypes: torch.dtype,
    device_id: int
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(42)
    hidden_states = torch.randn((b * s * topk, hidden_size), dtype = dtypes, device = f'npu:{device_id}') * 0.01 * 2 - 0.01
    hidden_states, hidden_states_scale = ffn_golden_quan_per_token(hidden_states)
    hidden_states_scale = hidden_states_scale.reshape(-1).to(torch.float32)

    group_list = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype = torch.int32, device = f'npu:{device_id}')
    group_list_cumsum = get_token_acc_table(group_list).to(torch.int32)
    w13 = torch.randn((per_expert_num, hidden_size, intermediate_size * 2), dtype = dtypes, device = f'npu:{device_id}')  * 0.01 * 2 - 0.01
    w13, w13_scale = ffn_golden_quan_per_channel_3d(w13)
    w13_scale = w13_scale.squeeze(1).to(torch.float32)

    w2 = torch.randn((per_expert_num, intermediate_size, hidden_size), dtype = dtypes, device = f'npu:{device_id}')  * 0.01 * 2 - 0.01
    w2, w2_scale = ffn_golden_quan_per_channel_3d(w2)
    w2_scale = w2_scale.squeeze(1).to(dtypes)

    ffn_res = torch.empty(hidden_states.shape, dtype =w2_scale.dtype, device = f'npu:{device_id}')
    return hidden_states, hidden_states_scale, group_list, group_list_cumsum, w13, w13_scale, w2, w2_scale, ffn_res


def symmetric_quantization_per_token(input_tensor) -> Tuple:
    """
    Perform symmetric quantization per token (per row).

    Args:
        input_tensor: Input tensor to quantize

    Returns:
        Tuple of (quantized_int8_tensor, dequantization_scale)
    """
    x_fp32 = pypto.block.cast(input_tensor, pypto.block.DT_FP32)
    x_abs = pypto.block.abs(x_fp32)
    x_max = pypto.block.amax(x_abs, -1, True)
    shape_0, shape_1 = x_max.shape[:2]
    x_scale = pypto.block.div(pypto.block.full([shape_0, shape_1], 127.0, pypto.block.DT_FP32), x_max)
    x_mul = pypto.block.mul(x_fp32, x_scale)
    x_int32 = pypto.block.cast(x_mul, pypto.block.DT_INT32, pypto.block.CastMode.CAST_RINT)
    x_fp16 = pypto.block.cast(x_int32, pypto.block.DT_FP16, pypto.block.CastMode.CAST_ROUND)
    x_int8 = pypto.block.cast(x_fp16, pypto.block.DT_INT8, pypto.block.CastMode.CAST_TRUNC)
    x_scale_quant = pypto.block.div(pypto.block.full([shape_0, shape_1], 1.0, pypto.block.DT_FP32), x_scale)
    return x_int8, x_scale_quant


def dequant_dynamic(in_tensor, scale_1, scale_2):
    """
    Perform dynamic dequantization using two scale factors.

    Args:
        in_tensor: Quantized input tensor
        scale_1: First scale factor
        scale_2: Second scale factor

    Returns:
        Dequantized tensor
    """
    in_tensor_fp32 = pypto.block.cast(in_tensor, pypto.block.DT_FP32, pypto.block.CastMode.CAST_NONE)
    scale_1_fp32 = pypto.block.cast(scale_1, pypto.block.DT_FP32, pypto.block.CastMode.CAST_NONE)
    scale_2_fp32 = pypto.block.cast(scale_2, pypto.block.DT_FP32, pypto.block.CastMode.CAST_NONE)
    out_scale_2 =pypto.block.mul(in_tensor_fp32, scale_2_fp32)
    out =pypto.block.mul(out_scale_2, scale_1_fp32)
    return out


def swiglu(up_proj_left, up_proj_right):
    """
    Apply SwiGLU activation function: x * sigmoid(x) * right_half.

    Args:
        up_proj: Input tensor with shape [batch, intermediate_size * 2]

    Returns:
        SwiGLU activated tensor with shape [batch, intermediate_size]
    """
    swiglu_mul = pypto.block.mul(up_proj_left, -1.0)
    swiglu_exp = pypto.block.exp(swiglu_mul)
    swiglu_add = pypto.block.add(swiglu_exp, 1.0)
    swiglu_div = pypto.block.div(up_proj_left, swiglu_add)
    swiglu_out = pypto.block.mul(swiglu_div, up_proj_right)
    return swiglu_out


@pypto.block
def ffn_router_expert_block_mm1(hidden_states, group_list, group_list_cumsum, w13, up_proj, block_cube_size_config):
    # tiling config
    if ASCEND_IS_AIC:
        # 设置tile参数
        # 核内tile切分
        tile_m, tile_k, tile_n = 128, 256, 256
        # 核间切分tile
        # block_cube_m, block_cube_k, block_cube_n = 128, 5120, 256
        block_cube_m, block_cube_k, block_cube_n = block_cube_size_config
        expert_num = group_list.shape
        block_idx = pypto.block.get_block_idx()
        aic_core_num = pypto.block.get_aic_core_num()
        hidden_size = hidden_states.shape[1]
        intermediate_size = w13.shape[1] // 2
        # 负载均衡
        count = 0
        for exp_idx in expert_num:
            # 获取激活专家的实际token数
            token_num = group_list[exp_idx, ]
            if (token_num == 0 or hidden_size <= 0 or intermediate_size <= 0):
                # 没有激活的专家，直接跳过
                continue
            # 计算当前激活专家的 总block块数 用于分核
            exp_idx_offset_start = group_list_cumsum[exp_idx, ]
            block_count_m = (token_num + block_cube_m - 1) / block_cube_m
            block_count_n = (intermediate_size + block_cube_n - 1) / block_cube_n
            curCount = count + block_count_m * block_count_n
            # 负载均衡 确保任务不会集中在core_0
            curBlock = block_idx if block_idx >= count else block_idx + aic_core_num
            #当前核循环计算block：block_i = block_i + core_total_num
            while (curBlock < curCount):
                # 计算核内的block块的 m_idx / n_idx
                block_m_idx = (curBlock + block_cube_m - 1) / block_cube_m
                block_n_idx = (curBlock + block_cube_n - 1) / block_cube_n
                tile_count = (hidden_size + tile_k - 1) / tile_k
                # k轴切分
                for tile_k_idx in tile_count:
                    # 获取该激活专家 token
                    hidden_states_offset = [exp_idx_offset_start + block_m_idx * block_cube_m, tile_k_idx * tile_k]
                    m_valid_size = pypto.block.min(token_num - block_m_idx * block_cube_m, tile_m)
                    k_valid_size = tile_k if tile_k_idx ==  tile_count - 1 else pypto.block.min(hidden_size - tile_k_idx * tile_k, tile_k)
                    # L0A
                    # 待补充接口
                    x = pypto.block.load_l1_to_l0a(hidden_states, [m_valid_size, k_valid_size], hidden_states_offset)

                    # 获取当前专家的 weight
                    weight_13_offset = [exp_idx * hidden_size + tile_k_idx * tile_k, block_n_idx * block_cube_n]
                    k_valid_size = tile_k if tile_k_idx ==  tile_count - 1 else pypto.block.min(hidden_size - tile_k_idx * tile_k, tile_k)
                    n_valid_size = pypto.block.min(intermediate_size * 2 - block_n_idx * block_cube_n, tile_n)
                    # L0B
                    # 待补充接口
                    w13_weight_2d = pypto.block.load_l1_to_l0b(w13, [k_valid_size, n_valid_size], weight_13_offset)

                    # up_proj的matmul计算
                    # L0C
                    if tile_k_idx == 0:
                        up_proj_result = pypto.block.matmul(x, w13_weight_2d, pypto.block.DT_INT32)
                    else:
                        # 待补充接口
                        up_proj_result = pypto.block.acc_matmul(x, w13_weight_2d, up_proj_result, pypto.block.DT_INT32)
                # 待补充接口
                pypto.block.store_l0c_to_gm(up_proj_result, [block_m_idx * block_cube_m, block_n_idx * block_cube_n], up_proj)
                curBlock = curBlock + aic_core_num
            count = curCount % aic_core_num


@pypto.block
def ffn_router_expert_block_vec1(hidden_states_scale, group_list, group_list_cumsum, w13_scale, up_proj, down_proj_quant, down_proj_scale, block_vec_size_config):
    if ASCEND_IS_AIV:
        # tiling config
        block_vec_m, block_vec_n = block_vec_size_config
        tile_m, tile_n1, tile_n2 = 128, 3072, 1536
        expert_num = group_list.shape
        block_idx = pypto.block.get_block_idx()
        aiv_core_num = pypto.block.get_aiv_core_num()
        intermediate_size = w13_scale.shape[1] // 2
        # 负载均衡
        count = 0
        for exp_idx in expert_num:
            # 获取激活专家的实际token数
            token_num = group_list[exp_idx, ]
            if (token_num == 0 or intermediate_size <= 0):
                # 没有激活的专家，直接跳过
                continue
            # 计算当前激活专家scale的 总block块数 用于分核
            exp_idx_offset_start = group_list_cumsum[exp_idx, ]
            block_count_m = (token_num + block_vec_m - 1) / block_vec_m
            # block_count_n = (intermediate_size + block_vec_n - 1) / block_vec_n
            curCount = count + block_count_m
            # 负载均衡 确保任务不会集中在core_0
            curBlock = block_idx if block_idx >= count else block_idx + aiv_core_num
            #当前核循环计算block：block_i = block_i + core_total_num
            while (curBlock < curCount):
                # 获取当前专家的实际token数和scale
                x_scale_offset = [exp_idx_offset_start + tile_m * curBlock, 0]
                cur_x_scale_size = pypto.block.min(token_num - curBlock * block_vec_m, tile_m)
                x_scale = pypto.block.load_gm_to_ub(hidden_states_scale, [cur_x_scale_size, 1], x_scale_offset)

                # 获取当前专家的weght_13和scale
                w13_scale_offset = [exp_idx, 0]
                w13_scale_valid = pypto.block.load_gm_to_ub(w13_scale, [1, tile_n1], w13_scale_offset)

                # up_proj
                # 获取up_proj的tile
                up_proj_offset = [exp_idx_offset_start + tile_m * curBlock, 0]
                cur_up_proj_size = pypto.block.min(token_num - curBlock * block_vec_m, tile_m)
                up_proj_tile = pypto.block.load_gm_to_ub(up_proj, [cur_up_proj_size, tile_n1], up_proj_offset)
                # dequant
                # [hidden_size , intermediate_size * 2]
                up_proj_out = dequant_dynamic(up_proj_tile, w13_scale_valid, x_scale)
                # swiglu 需要支持核内slice操作
                #  left : [hidden_size , intermediate_size]
                # right : [hidden_size , intermediate_size]
                up_proj_left = pypto.block.view(up_proj_out, [tile_m, tile_n2], [0, 0])
                up_proj_right = pypto.block.view(up_proj_out, [tile_m, tile_n2], [0, tile_n2])
                swiglu_out = swiglu(up_proj_left, up_proj_right)

                # down_proj
                # quant
                down_proj_quant_tile, down_proj_scale_tile = symmetric_quantization_per_token(swiglu_out)
                pypto.block.store_ub_to_gm(down_proj_quant_tile, [exp_idx_offset_start + tile_m * curBlock, 0], down_proj_quant)
                pypto.block.store_ub_to_gm(down_proj_scale_tile, [exp_idx_offset_start + tile_m * curBlock, 0], down_proj_scale)

@pypto.block
def ffn_router_expert_block_mm2(hidden_states, group_list, group_list_cumsum, w2, down_proj_quant, down_proj, block_cube_size_config):
    if ASCEND_IS_AIC:
        # 设置tile参数
        # 核内tile切分
        tile_m, tile_k, tile_n = 128, 256, 256
        # 核间切分tile
        # block_cube_m, block_cube_k, block_cube_n = 128, 5120, 256
        block_cube_m, block_cube_k, block_cube_n = block_cube_size_config
        expert_num = group_list.shape
        block_idx = pypto.block.get_block_idx()
        aic_core_num = pypto.block.get_aic_core_num()
        hidden_size = hidden_states.shape[1]
        intermediate_size = w2.shape[1]
        # 负载均衡
        count = 0
        for exp_idx in pypto.block.loop(expert_num):
            # 获取激活专家的实际token数
            token_num = group_list[exp_idx, ]
            if (token_num == 0 or hidden_size <= 0 or intermediate_size <= 0):
                # 没有激活的专家，直接跳过
                continue
            # 计算当前激活专家的 总block块数 用于分核
            exp_idx_offset_start = group_list_cumsum[exp_idx, ]
            block_count_m = (token_num + block_cube_m - 1) / block_cube_m
            block_count_n = (hidden_size + block_cube_n - 1) / block_cube_n
            curCount = count + block_count_m * block_count_n
            # 负载均衡 确保任务不会集中在core_0
            curBlock = block_idx if block_idx >= count else block_idx + aic_core_num
            #当前核循环计算block：block_i = block_i + core_total_num
            while (curBlock < curCount):
                # 计算核内的block块的 m_idx / n_idx
                block_m_idx = (curBlock + block_cube_m - 1) / block_cube_m
                block_n_idx = (curBlock + block_cube_n - 1) / block_cube_n
                tile_count = (intermediate_size + tile_k - 1) / tile_k
                # k轴切分
                for tile_k_idx in tile_count:
                    # 第一块只进行覆盖，后续进行acc
                    acc_flag = False if tile_k_idx == 0 else True
                    # 获取该激活专家 token
                    down_proj_offset = [exp_idx_offset_start + block_m_idx * block_cube_m, tile_k_idx * tile_k]
                    m_valid_size = pypto.block.min(token_num - block_m_idx * block_cube_m, tile_m)
                    k_valid_size = tile_k if tile_k_idx ==  tile_count - 1 else pypto.block.min(intermediate_size - tile_k_idx * tile_k, tile_k)
                    # L0A
                    # 待补充接口
                    x = pypto.block.load_l1_to_l0a(down_proj_quant, [m_valid_size, k_valid_size], down_proj_offset)

                    # 获取当前专家的 weight
                    weight_2_offset = [exp_idx * intermediate_size + tile_k_idx * tile_k, block_n_idx * block_cube_n]
                    k_valid_size = tile_k if tile_k_idx ==  tile_count - 1 else pypto.block.min(intermediate_size - tile_k_idx * tile_k, tile_k)
                    n_valid_size = pypto.block.min(intermediate_size - block_n_idx * block_cube_n, tile_n)
                    # L0B
                    # 待补充接口
                    w2_weight = pypto.block.load_l1_to_l0b(w2, [k_valid_size, n_valid_size], weight_2_offset)

                    # up_proj的matmul计算
                    # L0C
                    if acc_flag:
                        up_proj_result = pypto.block.matmul(x, w2_weight, pypto.block.DT_INT32)
                    else:
                        # 待补充接口
                        up_proj_result = pypto.block.acc_matmul(x, w2_weight, up_proj_result, pypto.block.DT_INT32)
                # 待补充接口
                pypto.block.store_l0c_to_gm(up_proj_result, [block_m_idx * block_cube_m, block_n_idx * block_cube_n], down_proj)
                curBlock = curBlock + aic_core_num
            count = curCount % aic_core_num


@pypto.block
def ffn_router_expert_block_vec2(hidden_states, group_list, group_list_cumsum, w2_scale, down_proj, down_proj_scale, ffn_res, block_vec_size_config):
    if ASCEND_IS_AIV:
        # tiling config
        block_vec_m, block_vec_n = block_vec_size_config
        tile_m, tile_n = 128, 1536
        expert_num = group_list.shape
        block_idx = pypto.block.get_block_idx()
        aiv_core_num = pypto.block.get_aiv_core_num()
        intermediate_size = w2_scale.shape[1]
        x_dtype = w2_scale.dtype
        # 负载均衡
        count = 0
        for exp_idx in expert_num:
            # 获取激活专家的实际token数
            token_num = group_list[exp_idx, ]
            if (token_num == 0 or intermediate_size <= 0):
                # 没有激活的专家，直接跳过
                continue
            # 计算当前激活专家scale的 总block块数 用于分核
            exp_idx_offset_start = group_list_cumsum[exp_idx, ]
            block_count_m = (token_num + block_vec_m - 1) / block_vec_m
            # block_count_n = (intermediate_size + block_vec_n - 1) / block_vec_n
            curCount = count + block_count_m
            # 负载均衡 确保任务不会集中在core_0
            curBlock = block_idx if block_idx >= count else block_idx + aiv_core_num
            #当前核循环计算block：block_i = block_i + core_total_num
            while (curBlock < curCount):
                # 获取当前专家的实际token数和scale
                down_proj_scale_offset = [exp_idx_offset_start + tile_m * curBlock, 0]
                down_proj_scale_size = pypto.block.min(token_num - curBlock * block_vec_m, tile_m)
                x_scale = pypto.block.load_gm_to_ub(down_proj_scale, [down_proj_scale_size, 1], down_proj_scale_offset)

                # 获取当前专家的weght_13和scale
                w2_scale_offset = [exp_idx, 0]
                w2_scale_valid = pypto.block.load_gm_to_ub(w2_scale, [1, tile_n], w2_scale_offset)

                # up_proj
                # 获取up_proj的tile
                up_proj_offset = [exp_idx_offset_start + tile_m * curBlock, 0]
                cur_up_proj_size = pypto.block.min(token_num - curBlock * block_vec_m, tile_m)
                down_proj_tile = pypto.block.load_gm_to_ub(down_proj, [cur_up_proj_size, tile_n], up_proj_offset)
                # dequant
                # [hidden_size , intermediate_size * 2]
                out_tile = dequant_dynamic(down_proj_tile, w2_scale_valid, x_scale)
                out = pypto.block.cast(out_tile, x_dtype)

                pypto.block.store_ub_to_gm(out, down_proj_scale_offset, down_proj_scale)


def up_align(n, align):
    return (n + align - 1) / align


ASCEND_IS_AIC = True
ASCEND_IS_AIV = True


@pypto.jit(
    host_options={"only_codegen": True},
    codegen_options={"codegen_expression_fusion": True},
    runtime_options={"device_sched_mode": 1},
    pass_options={"cube_l1_reuse_mode": 2}
)
def moe_router_expert_main(hidden_states, hidden_states_scale,
                           group_list, group_list_cumsum, w13,
                           w13_scale, w2, w2_scale, ffn_res):
    """
    JIT compiled kernel for router expert FFN quantization.

    This kernel processes multiple experts in a loop, where each expert processes
    a subset of tokens based on the group_list. The computation is done in tiles
    to support efficient execution on NPU.

    Args:
        hidden_states: Quantized input hidden states (int8) [num_tokens * topk, hidden_size]
        hidden_states_scale: Per-token quantization scale [num_tokens * topk]
        group_list: Group list containing token counts per expert [per_device_expert_num]
        group_list_cumsum: Cumulative sum of group list [per_device_expert_num]
        w13: Gate and up projection weights (int8) [per_device_expert_num, hidden_size, intermediate_size * 2]
        w13_scale: w13 weight scales [per_device_expert_num, intermediate_size * 2]
        w2: Down projection weights (int8) [per_device_expert_num, intermediate_size, hidden_size]
        w2_scale: w2 weight scales [per_device_expert_num, hidden_size]
        ffn_res: Output tensor [num_tokens * topk, hidden_size]

    Note:
        This function uses cube L1 reuse mode 2 for better memory efficiency.
        Each expert processes tokens in tiles of size 8.
    """

    # 输入Tensor shape转换为2维
    w13_2d_shape = (w13.shape[0] * w13.shape[1], w13.shape[2])
    w2_2d_shape = (w2.shape[0] * w2.shape[1], w2.shape[2])
    hidden_states_scale_shape = (hidden_states_scale.shape[0], 1)

    w13_2d = pypto.reshape(w13, w13_2d_shape, inplace=True)
    w2_2d = pypto.reshape(w2, w2_2d_shape, inplace=True)
    hidden_states_scale_2d = pypto.reshape(hidden_states_scale, hidden_states_scale_shape, inplace=True)

    # cube
    bs_topk, hidden_size = hidden_states.shape
    intermediate_size = w2_2d.shape[1]
    up_proj = pypto.tensor([bs_topk, hidden_size], pypto.DT_INT32, "up_proj")
    block_cube_m, block_cube_k, block_cube_n = 128, 5120, 256
    block_cube_size_config = [block_cube_m, block_cube_k, block_cube_n]
    ffn_router_expert_block_mm1(up_align(hidden_states.shape[0], block_cube_m) * up_align(w13_2d.shape[1], block_cube_n),
                     hidden_states, group_list, group_list_cumsum, w13_2d, up_proj, block_cube_size_config)
    # vector
    x0, x1 = hidden_states.shape
    block_vec_m, block_vec_n = 128, 5120
    block_vec_size_config = [block_vec_m, block_vec_n]
    down_proj_quant = pypto.tensor([bs_topk, intermediate_size], pypto.DT_INT8, "down_proj_quant")
    down_proj_scale = pypto.tensor([bs_topk, 1], pypto.DT_FP32, "down_proj_scale")
    ffn_router_expert_block_vec1(up_align(up_proj.shape[0], block_vec_m) * up_align(up_proj.shape[1], block_vec_n), hidden_states_scale_2d,
                     group_list, group_list_cumsum, w13_scale, up_proj, down_proj_quant, down_proj_scale, block_vec_size_config)
    # cube
    down_proj = pypto.tensor([bs_topk, hidden_size], pypto.DT_INT32, "down_proj")
    ffn_router_expert_block_mm2(up_align(down_proj_quant.shape[0], block_cube_m) * up_align(w2_2d.shape[1], block_cube_n),
                     hidden_states, group_list, group_list_cumsum, w2_2d, down_proj_quant, down_proj, block_cube_size_config)
    # vector
    ffn_router_expert_block_vec2(up_align(down_proj.shape[0], block_vec_m) * up_align(down_proj.shape[1], block_vec_n),
                     hidden_states, group_list, group_list_cumsum, w2_scale, down_proj, down_proj_scale, ffn_res, block_vec_size_config)


def test_ffn_router() -> None:
    dtype = torch.bfloat16
    # parameter config
    s = 1
    intermediate_size = 1536
    hidden_size = 5120
    per_expert_num = 20
    topk = 8
    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    # Test with different batch sizes
    for b in [1, 2]:
        # hidden_states, hidden_states_scale, group_list, group_list_cumsum, w13, w13_scale, w2, w2_scale, ffn_res
        hidden_states, hidden_states_scale, group_list, group_list_cumsum, w13, w13_scale, w2, w2_scale, ffn_res = \
            gen_input(b, s, topk, per_expert_num, hidden_size, intermediate_size, dtype, device_id)

        inputs = {
            hidden_states: [0],
            hidden_states_scale: [0],
            group_list: [],
            group_list_cumsum: [],
            w13: [],
            w13_scale: [],
            w2: [],
            w2_scale: []
        }
        outputs = {
            ffn_res: [0]
        }
        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
        pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
        moe_router_expert_main(*pto_inputs, *pto_outputs)
        pypto.runtime._device_synchronize()#内部接口，不推荐使用

        # golden
        golden = ffn_router_torch_npu(hidden_states, hidden_states_scale, group_list, w13, w13_scale, w2, w2_scale)

        # calc valid token num for compare
        vaild_token_cumsum = group_list.cumsum(dim=0)
        valid_size = vaild_token_cumsum[vaild_token_cumsum.shape[0] - 1] * hidden_size
        assert_allclose(np.array(ffn_res.cpu().flatten().tolist()[0 : valid_size]),\
                        np.array(golden.cpu().flatten().tolist()[0 : valid_size]), rtol=0.0078125, atol=0.0001)


if __name__ == "__main__":
    main()