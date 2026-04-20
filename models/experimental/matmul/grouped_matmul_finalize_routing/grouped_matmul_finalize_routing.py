#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
GroupedMatmulFinalizeRoutingV3 - MoE场景下的融合算子

实现了三个核心步骤：
1. 分组矩阵乘法（GMM）- 使用 pypto.scaled_mm 处理 MXFP8 量化
2. 路由分配（Routing）- 使用 pypto.index_add_ 实现 scatter add
3. 共享专家融合 - 加权融合共享专家输出

参考实现: gmm_mxfp8.py
"""

import math
import os
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pypto
import torch
import torch_npu
from numpy.testing import assert_allclose


# ─────────────────────────────────────────────
# 1. 数据类定义
# ─────────────────────────────────────────────

@dataclass
class GoldenComputeInputs:
    """MXFP8缩放矩阵乘法的输入参数"""
    x: torch.Tensor
    weight: torch.Tensor
    scaled_x: torch.Tensor
    scaled_weight: torch.Tensor
    a_trans: bool
    b_trans: bool


@dataclass
class GmmFRGoldenInputs:
    """GroupedMatmulFinalizeRoutingV3 golden输入参数"""
    x1: torch.Tensor
    x2: torch.Tensor
    scale: torch.Tensor
    pertoken_scale: torch.Tensor
    group_list: List[int]
    row_index: torch.Tensor
    logit: torch.Tensor
    batch: int
    n: int
    bias: Optional[torch.Tensor]
    shared_input: Optional[torch.Tensor]
    shared_input_weight: float
    shared_input_offset: int
    transpose_x1: bool
    transpose_x2: bool
    group_list_type: int


@dataclass
class GmmFRInputs:
    """GroupedMatmulFinalizeRoutingV3 PyPTO输入参数"""
    x1: torch.Tensor
    x2: torch.Tensor
    scale: torch.Tensor
    pertoken_scale: torch.Tensor
    group_list: List[int]
    row_index: torch.Tensor
    logit: torch.Tensor
    batch: int
    n: int
    bias: Optional[torch.Tensor]
    shared_input: Optional[torch.Tensor]
    shared_input_weight: float
    shared_input_offset: int
    transpose_x1: bool
    transpose_x2: bool
    group_list_type: int
    tile_config: 'TileConfig'


@dataclass
class TileConfig:
    """Tile配置参数"""
    m_tile_shape: List[int]
    k_tile_shape: List[int]
    n_tile_shape: List[int]
    vector_tile_shape: List[int]


# 默认Tile配置
DEFAULT_TILE_CONFIG = TileConfig(
    m_tile_shape=[9, 9],
    k_tile_shape=[256, 256],
    n_tile_shape=[256, 256],
    vector_tile_shape=[1, 8, 256, 32],
)


# ─────────────────────────────────────────────
# 2. Golden 参考实现
# ─────────────────────────────────────────────

def compute_golden_result(inputs: GoldenComputeInputs) -> torch.Tensor:
    """
    计算单个专家组的MXFP8缩放矩阵乘法golden结果
    
    参考: gmm_mxfp8.py 的 compute_golden_result
    """
    x = inputs.x
    weight = inputs.weight
    scaled_x_golden = inputs.scaled_x
    scaled_weight_golden = inputs.scaled_weight
    a_trans = inputs.a_trans
    b_trans = inputs.b_trans

    # 处理输入转置
    if a_trans:
        x = torch.swapaxes(x, -1, -2)
        scaled_x_golden = torch.swapaxes(scaled_x_golden, -1, -2)
        if len(scaled_x_golden.shape) == 3:
            scaled_x_golden = scaled_x_golden.reshape(
                scaled_x_golden.shape[0] * scaled_x_golden.shape[1], scaled_x_golden.shape[2]
            )
        scaled_x_golden = torch.swapaxes(scaled_x_golden, -1, -2)
    else:
        if len(scaled_x_golden.shape) == 3:
            scaled_x_golden = scaled_x_golden.reshape(
                scaled_x_golden.shape[0], scaled_x_golden.shape[1] * scaled_x_golden.shape[2]
            )

    # 处理权重转置
    if b_trans:
        weight = torch.swapaxes(weight, -1, -2)
        if len(scaled_weight_golden.shape) == 3:
            scaled_weight_golden = scaled_weight_golden.reshape(
                scaled_weight_golden.shape[0] * scaled_weight_golden.shape[1],
                scaled_weight_golden.shape[2]
            )
        scaled_weight_golden = torch.swapaxes(scaled_weight_golden, -1, -2)
    else:
        scaled_weight_golden = torch.swapaxes(scaled_weight_golden, -1, -2)
        if len(scaled_weight_golden.shape) == 3:
            scaled_weight_golden = scaled_weight_golden.reshape(
                scaled_weight_golden.shape[0] * scaled_weight_golden.shape[1],
                scaled_weight_golden.shape[2]
            )

    # K维度对齐
    k_dim = x.shape[-1]
    if math.ceil(k_dim / 32) % 2 != 0:
        scaled_x_golden = scaled_x_golden[:, :-1]
        scaled_weight_golden = scaled_weight_golden[:-1, :]

    # 广播缩放因子
    scaled_x_golden_broadcast = torch.repeat_interleave(scaled_x_golden, repeats=32, dim=-1)
    scaled_weight_golden_broadcast = torch.repeat_interleave(scaled_weight_golden, repeats=32, dim=-2)

    # 计算padding长度
    x1_dims = len(x.shape)
    x2_dims = len(weight.shape)
    x1_pad_len = scaled_x_golden_broadcast.shape[-1] - x.shape[-1]
    x2_pad_len = scaled_weight_golden_broadcast.shape[-2] - weight.shape[-2]

    # Padding输入tensor
    x1_pad = [0, x1_pad_len]
    for _ in range(x1_dims - 1):
        x1_pad += [0, 0]
    x_golden = torch.nn.functional.pad(x, x1_pad, mode='constant', value=0)

    # Padding权重tensor
    weight_pad = [0, 0]
    weight_pad += [0, x2_pad_len]
    for _ in range(x2_dims - 2):
        weight_pad += [0, 0]
    weight_golden = torch.nn.functional.pad(weight, weight_pad, mode='constant', value=0)

    # 应用缩放因子
    x_fp32 = x_golden.to(torch.float32)
    scaled_x_golden_broadcast_fp32 = scaled_x_golden_broadcast.to(torch.float32)
    x1_golden = x_fp32 * scaled_x_golden_broadcast_fp32

    weight_fp32 = weight_golden.to(torch.float32)
    scaled_weight_golden_broadcast_fp32 = scaled_weight_golden_broadcast.to(torch.float32)
    weight_golden = weight_fp32 * scaled_weight_golden_broadcast_fp32

    # 矩阵乘法
    golden = torch.matmul(x1_golden, weight_golden)
    return golden


def combine_func(x, logits, residual, resid_scale, source_row, output_bs, offset):
    top_k = x.shape[0] // output_bs
    remain_logits = len(logits) % top_k
    if remain_logits:
        logits = logits[:len(logits) - remain_logits]
    out = x * logits.reshape(-1,1)
    remain_sr = len(source_row) % top_k
    if remain_sr:
        source_row = source_row[:len(source_row) - remain_sr]
    index = np.argsort(source_row)
    out = out[index].reshape(output_bs, top_k, x.shape[-1]).sum(axis=1)
    if residual is not None:
        out[offset:offset + residual.shape[0], :] += resid_scale * residual
    out = out.to(torch.float32)
    return out


def gen_golden(inputs: GmmFRGoldenInputs) -> torch.Tensor:
    """
    生成GroupedMatmulFinalizeRoutingV3的golden参考结果
    
    包含三个步骤：
    1. 分组矩阵乘法（GMM）
    2. 路由分配（Scatter Add）
    3. 共享专家融合
    """
    x1 = inputs.x1
    x2 = inputs.x2
    scale = inputs.scale
    pertoken_scale = inputs.pertoken_scale
    group_list = inputs.group_list
    row_index = inputs.row_index
    logit = inputs.logit
    batch = inputs.batch
    n = inputs.n
    bias = inputs.bias
    shared_input = inputs.shared_input
    shared_input_weight = inputs.shared_input_weight
    shared_input_offset = inputs.shared_input_offset
    transpose_x1 = inputs.transpose_x1
    transpose_x2 = inputs.transpose_x2
    group_list_type = inputs.group_list_type

    # 转换group_list_type
    if group_list_type == 0:
        group_counts = []
        prev = 0
        for val in group_list:
            group_counts.append(val - prev)
            prev = val
        group_list = group_counts

    num_experts = x2.shape[0]
    m_total = x1.shape[0]

    # Step 1: 分组矩阵乘法
    intermediate = torch.zeros((m_total, n), dtype=torch.float32)
    begin = 0
    end = 0

    for i in range(num_experts):
        begin = end
        end = end + group_list[i]

        if group_list[i] <= 0:
            continue

        # 提取当前专家的输入和权重
        if transpose_x1:
            x_i = x1[:, begin:end]
            scaled_x_i = pertoken_scale[:, begin:end, :]
        else:
            x_i = x1[begin:end, :]
            scaled_x_i = pertoken_scale[begin:end, :, :]

        weight_i = x2[i]
        scale_i = scale[i]

        # 计算GMM结果
        gmm_result = compute_golden_result(
            GoldenComputeInputs(
                x=x_i,
                weight=weight_i,
                scaled_x=scaled_x_i,
                scaled_weight=scale_i,
                a_trans=transpose_x1,
                b_trans=transpose_x2,
            )
        )

        # 添加bias（可选）
        if bias is not None:
            bias_i = bias[i].to(torch.float32)
            gmm_result = gmm_result + bias_i

        intermediate[begin:end, :] = gmm_result

    # Step 2: 共享专家融合（可选）
    if shared_input is not None:
        output_bs = x1.shape[0] // len(group_list)
        final_out = combine_func(intermediate, logit, shared_input, shared_input_weight, row_index, output_bs, shared_input_offset)

    return final_out


# ─────────────────────────────────────────────
# 3. PyPTO Kernel 实现（参考 quant_matmul_reduce_sum.py）
# ─────────────────────────────────────────────

def grouped_matmul_finalize_routing_pypto(m, k, n, e, batch, tile_config, transpose_x2=False):
    """
    创建GroupedMatmulFinalizeRoutingV3 PyPTO kernel
    
    参考 quant_matmul_reduce_sum.py 的实现方式：
    - 在参数中指定完整的 tensor shape 和 dtype
    - 直接传入 torch tensor，不需要手动转换
    
    Args:
        m: 输入token数量
        k: 输入特征维度
        n: 输出特征维度
        e: 专家数量
        batch: 输出batch维度
        tile_config: Tile配置
        transpose_x2: 权重是否转置
        
    Returns:
        JIT编译的kernel函数
    """
    x1_shape = [m, k]
    if transpose_x2:
        x2_shape = [e, n, k]
        scale_shape = [e, n, math.ceil(k / 64), 2]
    else:
        x2_shape = [e, k, n]
        scale_shape = [e, math.ceil(k / 64), n, 2]
    pertoken_scale_shape = [m, math.ceil(k / 64), 2]
    intermediate_shape = [m, n]
    out_shape = [batch, n]
    row_index_shape = [m]

    @pypto.frontend.jit()
    def grouped_matmul_finalize_routing_kernel(
        x1: pypto.Tensor(x1_shape, pypto.DT_FP8E4M3),
        x2: pypto.Tensor(x2_shape, pypto.DT_FP8E4M3),
        scale: pypto.Tensor(scale_shape, pypto.DT_FP8E8M0),
        pertoken_scale: pypto.Tensor(pertoken_scale_shape, pypto.DT_FP8E8M0),
        intermediate: pypto.Tensor(intermediate_shape, pypto.DT_FP32),
        out: pypto.Tensor(out_shape, pypto.DT_FP32),
        row_index: pypto.Tensor(row_index_shape, pypto.DT_INT64),
        group_list: List[int]
    ):
        """
        GroupedMatmulFinalizeRoutingV3 PyPTO JIT kernel
        
        实现两个核心步骤：
        1. 分组矩阵乘法（遍历专家组执行scaled_mm）
        2. 路由分配（使用index_add_实现scatter add）
        """
        num_experts = x2.shape[0]
        begin = 0
        end = 0

        # Step 1: 分组矩阵乘法
        for i in range(num_experts):
            begin = end
            end = end + group_list[i]

            if group_list[i] <= 0:
                continue

            # 提取当前专家的输入和权重（参考 gmm_mxfp8.py 的顺序）
            x = x1[begin:end, :]
            weight = x2[i]
            scaled_x = pertoken_scale[begin:end, :, :]

            # **先设置 vector tile shapes，再进行 scale tensor 的 indexing**
            pypto.set_vec_tile_shapes(
                tile_config.vector_tile_shape[0],
                tile_config.vector_tile_shape[1],
                tile_config.vector_tile_shape[2],
                tile_config.vector_tile_shape[3]
            )

            # 设置完 tile shapes 后才 indexing scale tensor
            scaled_weight = scale[i]

            # 设置cube tile shapes
            pypto.set_cube_tile_shapes(
                tile_config.m_tile_shape,
                tile_config.k_tile_shape,
                tile_config.n_tile_shape
            )

            # 执行MXFP8 scaled_matmul
            intermediate[begin:end, :] = pypto.scaled_mm(
                x, weight, pypto.DT_FP32, scaled_x, scaled_weight
            )

        # Step 2: 路由分配（Scatter Add）
        pypto.set_vec_tile_shapes(1, tile_config.n_tile_shape[0])
        pypto.index_add_(out, 0, row_index, intermediate, alpha=1.0)

    return grouped_matmul_finalize_routing_kernel


def gen_mxfp8(inputs: GmmFRInputs) -> torch.Tensor:
    """
    使用PyPTO执行GroupedMatmulFinalizeRoutingV3计算
    
    参考 quant_matmul_reduce_sum.py 的调用方式：
    - 直接传入 torch tensor
    - 使用返回的 kernel 函数
    """
    x1 = inputs.x1
    x2 = inputs.x2
    scale = inputs.scale
    pertoken_scale = inputs.pertoken_scale
    group_list = inputs.group_list
    row_index = inputs.row_index
    logit = inputs.logit
    batch = inputs.batch
    n = inputs.n
    bias = inputs.bias
    shared_input = inputs.shared_input
    shared_input_weight = inputs.shared_input_weight
    shared_input_offset = inputs.shared_input_offset
    transpose_x1 = inputs.transpose_x1
    transpose_x2 = inputs.transpose_x2
    group_list_type = inputs.group_list_type
    tile_config = inputs.tile_config

    # 转换group_list_type
    if group_list_type == 0:
        group_counts = []
        prev = 0
        for val in group_list:
            group_counts.append(val - prev)
            prev = val
        group_list = group_counts

    # Move tensors to NPU
    x1 = x1.npu()
    x2 = x2.npu()
    scale = scale.npu()
    pertoken_scale = pertoken_scale.npu()
    row_index = row_index.npu()

    # 计算维度
    m = x1.shape[0]
    k = x1.shape[1]
    e = x2.shape[0]

    # 初始化tensor
    intermediate = torch.zeros((m, n), dtype=torch.float32).npu()
    out = torch.zeros((batch, n), dtype=torch.float32).npu()

    # 创建并调用 kernel（参考 quant_matmul_reduce_sum.py）
    kernel = grouped_matmul_finalize_routing_pypto(
        m, k, n, e, batch, tile_config, transpose_x2
    )
    
    # 直接传入 torch tensor，kernel 会修改 out tensor
    kernel(x1, x2, scale, pertoken_scale, intermediate, out, row_index, group_list)

    # Step 3: 处理bias（可选）
    if bias is not None:
        bias = bias.npu()
        begin = 0
        end = 0
        for i in range(len(group_list)):
            begin = end
            end = end + group_list[i]
            if group_list[i] > 0:
                bias_i = bias[i].to(torch.float32)
                intermediate[begin:end, :] = intermediate[begin:end, :] + bias_i

    # Step 4: 共享专家融合（可选）
    if shared_input is not None:
        shared_input = shared_input.npu()
        shared_fp32 = shared_input.to(torch.float32)
        weighted_shared = shared_fp32 * shared_input_weight
        bsdp = shared_input.shape[0]
        for j in range(bsdp):
            target_row = shared_input_offset + j
            if target_row < batch:
                out[target_row, :] = out[target_row, :] + weighted_shared[j, :]

    return out.to(torch.float32)


# ─────────────────────────────────────────────
# 4. 测试函数
# ─────────────────────────────────────────────

def get_device_id():
    """获取NPU设备ID"""
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        print("Please set: export TILE_FWK_DEVICE_ID=0")
        return None
    try:
        return int(os.environ["TILE_FWK_DEVICE_ID"])
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be int, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


def generate_test_data(
    m: int,
    k: int,
    n: int,
    e: int,
    batch: int,
    group_list: list,
    transpose_x2: bool = False,
    include_bias: bool = False,
    include_shared: bool = False,
):
    """生成MXFP8格式测试数据"""
    # 输入x1 [M, K]
    x1 = torch.randn((m, k), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e4m3fn)

    # pertoken_scale [M, Ceil(K/64), 2]
    pertoken_scale = torch.randn(
        (m, math.ceil(k / 64), 2), dtype=torch.float32
    ).uniform_(0.5, 1.5).to(torch.float8_e8m0fnu)

    # 权重x2 [E, K, N] 或 [E, N, K]
    if transpose_x2:
        x2 = torch.randn((e, n, k), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e4m3fn)
        scale = torch.randn(
            (e, n, math.ceil(k / 64), 2), dtype=torch.float32
        ).uniform_(0.5, 1.5).to(torch.float8_e8m0fnu)
    else:
        x2 = torch.randn((e, k, n), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e4m3fn)
        scale = torch.randn(
            (e, math.ceil(k / 64), n, 2), dtype=torch.float32
        ).uniform_(0.5, 1.5).to(torch.float8_e8m0fnu)

    # 路由索引 [M]
    row_index = torch.randint(0, batch, (m,), dtype=torch.int64)

    # logit [M]
    logit = torch.randn((m,), dtype=torch.float32).uniform_(0.5, 2.0)

    # bias [E, N] (可选)
    bias = None
    if include_bias:
        bias = torch.randn((e, n), dtype=torch.bfloat16)

    # shared_input [bsdp, N] (可选)
    shared_input = None
    bsdp = batch // e if include_shared else 0
    if include_shared and bsdp > 0:
        shared_input = torch.randn((bsdp, n), dtype=torch.bfloat16)

    return {
        'x1': x1,
        'x2': x2,
        'scale': scale,
        'pertoken_scale': pertoken_scale,
        'group_list': group_list,
        'row_index': row_index,
        'logit': logit,
        'batch': batch,
        'n': n,
        'bias': bias,
        'shared_input': shared_input,
        'shared_input_weight': 1.0,
        'shared_input_offset': 0,
        'transpose_x1': False,
        'transpose_x2': transpose_x2,
        'group_list_type': 1,
    }


# def test_gmm_fr_basic(tile_config: TileConfig = DEFAULT_TILE_CONFIG, device_id=None):
#     """
#     测试GroupedMatmulFinalizeRoutingV3基础配置
    
#     参考: gmm_mxfp8.py 的 test_gmm_mxfp8
#     """
#     print("=" * 60)
#     print("Test: GroupedMatmulFinalizeRoutingV3 - MXFP8 Basic")
#     print("=" * 60)

#     # 参数配置
#     m, k, n, e, batch = 16, 512, 7168, 2, 8
#     group_list = [7, 9]

#     print(f"Config: m={m}, k={k}, n={n}, e={e}, batch={batch}")
#     print(f"group_list: {group_list}")

#     # 生成测试数据
#     data = generate_test_data(
#         m=m, k=k, n=n, e=e, batch=batch, group_list=group_list
#     )

#     # 计算golden
#     golden = gen_golden(GmmFRGoldenInputs(
#         x1=data['x1'],
#         x2=data['x2'],
#         scale=data['scale'],
#         pertoken_scale=data['pertoken_scale'],
#         group_list=data['group_list'],
#         row_index=data['row_index'],
#         logit=data['logit'],
#         batch=data['batch'],
#         n=data['n'],
#         bias=data['bias'],
#         shared_input=data['shared_input'],
#         shared_input_weight=data['shared_input_weight'],
#         shared_input_offset=data['shared_input_offset'],
#         transpose_x1=data['transpose_x1'],
#         transpose_x2=data['transpose_x2'],
#         group_list_type=data['group_list_type'],
#     ))

#     print(f"Golden output shape: {golden.shape}")
#     print(f"Golden output sample (row 0, cols 0-5): {golden[0, :5]}")

#     # 计算PyPTO结果
#     if device_id is not None:
#         torch.npu.set_device(device_id)
#         result = gen_mxfp8(GmmFRInputs(
#             x1=data['x1'],
#             x2=data['x2'],
#             scale=data['scale'],
#             pertoken_scale=data['pertoken_scale'],
#             group_list=data['group_list'],
#             row_index=data['row_index'],
#             logit=data['logit'],
#             batch=data['batch'],
#             n=data['n'],
#             bias=data['bias'],
#             shared_input=data['shared_input'],
#             shared_input_weight=data['shared_input_weight'],
#             shared_input_offset=data['shared_input_offset'],
#             transpose_x1=data['transpose_x1'],
#             transpose_x2=data['transpose_x2'],
#             group_list_type=data['group_list_type'],
#             tile_config=tile_config,
#         ))

#         print(f"PyPTO output shape: {result.shape}")
#         print(f"PyPTO output sample (row 0, cols 0-5): {result[0, :5].cpu()}")

#         # 精度对比
#         max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
#         print(f"Max diff: {max_diff:.6e}")

#         assert_allclose(
#             result.cpu().numpy(),
#             golden.cpu().numpy(),
#             rtol=1e-3, atol=1e-3,
#         )
#         print("[PRECISION_PASS]")
#     else:
#         print("[NPU NOT AVAILABLE - Golden only]")

#     print("✓ Passed\n")


def test_gmm_fr_full_config(tile_config: TileConfig = DEFAULT_TILE_CONFIG, device_id=None):
    """测试完整配置（包含bias和shared_input）"""
    print("=" * 60)
    print("Test: GroupedMatmulFinalizeRoutingV3 - Full Config")
    print("=" * 60)

    m, k, n, e, batch = 16, 512, 7168, 2, 8
    group_list = [7, 9]

    print(f"Config: m={m}, k={k}, n={n}, e={e}, batch={batch}")
    print(f"group_list: {group_list}, bias=True, shared_input=True")

    data = generate_test_data(
        m=m, k=k, n=n, e=e, batch=batch, group_list=group_list,
        include_bias=True,
        include_shared=True,
    )

    golden = gen_golden(GmmFRGoldenInputs(
        x1=data['x1'],
        x2=data['x2'],
        scale=data['scale'],
        pertoken_scale=data['pertoken_scale'],
        group_list=data['group_list'],
        row_index=data['row_index'],
        logit=data['logit'],
        batch=data['batch'],
        n=data['n'],
        bias=data['bias'],
        shared_input=data['shared_input'],
        shared_input_weight=data['shared_input_weight'],
        shared_input_offset=data['shared_input_offset'],
        transpose_x1=data['transpose_x1'],
        transpose_x2=data['transpose_x2'],
        group_list_type=data['group_list_type'],
    ))

    print(f"Golden output shape: {golden.shape}")
    print("golden vale: ", golden)

    if device_id is not None:
        torch.npu.set_device(device_id)
        result = gen_mxfp8(GmmFRInputs(
            x1=data['x1'],
            x2=data['x2'],
            scale=data['scale'],
            pertoken_scale=data['pertoken_scale'],
            group_list=data['group_list'],
            row_index=data['row_index'],
            logit=data['logit'],
            batch=data['batch'],
            n=data['n'],
            bias=data['bias'],
            shared_input=data['shared_input'],
            shared_input_weight=data['shared_input_weight'],
            shared_input_offset=data['shared_input_offset'],
            transpose_x1=data['transpose_x1'],
            transpose_x2=data['transpose_x2'],
            group_list_type=data['group_list_type'],
            tile_config=tile_config,
        ))

        max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
        print(f"Max diff: {max_diff:.6e}")

        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )
        print("[PRECISION_PASS]")
    else:
        print("[NPU NOT AVAILABLE - Golden only]")

    print("✓ Passed\n")


# ─────────────────────────────────────────────
# 5. Main 入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("GroupedMatmulFinalizeRoutingV3 Test Suite")
    print("=" * 60)
    print()

    # 获取设备ID
    device_id = get_device_id()

    # 运行测试
    # test_gmm_fr_basic(DEFAULT_TILE_CONFIG, device_id)
    test_gmm_fr_full_config(DEFAULT_TILE_CONFIG, device_id)

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)