#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------
"""
linear 算子 PyPTO 实现

公式: output = input @ weight^T + bias

功能: 线性层（全连接层）算子，对输入张量进行线性变换。
      支持任意维度的输入张量（2D、3D、4D），对最后一维进行线性变换，保持其他维度不变。

实现策略:
  - 所有场景统一使用 2D kernel
  - 3D/4D 输入在 wrapper 中 reshape 为 2D，计算后 reshape 回原形状
  - 2D 场景使用 matmul 的 extend_params={'bias_tensor': bias} 融合 bias（性能更优）
"""

import pypto
import torch
from typing import Optional


# ─────────────────────────────────────────────
# 2D 场景 Kernel（bias 融合方案）
# ─────────────────────────────────────────────

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def linear_kernel_2d_with_bias(
    input_tensor: pypto.Tensor([], pypto.DT_FP32),
    weight_tensor: pypto.Tensor([], pypto.DT_FP32),
    bias_tensor: pypto.Tensor([], pypto.DT_FP32),
    output_tensor: pypto.Tensor([], pypto.DT_FP32),
):
    """2D linear kernel with bias fusion.

    使用 matmul 的 extend_params 融合 bias，性能更优。
    公式: output = input @ weight^T + bias
    """
    # 设置 cube tiling（根据矩阵大小动态选择）
    m = input_tensor.shape[0]  # batch or flattened batch
    k = input_tensor.shape[1]  # in_features
    n = weight_tensor.shape[0]  # out_features

    if m <= 64 and k <= 64 and n <= 64:
        pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    elif m <= 2048 and k <= 2048 and n <= 2048:
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    else:
        pypto.set_cube_tile_shapes([256, 256], [256, 256], [256, 256])

    # 使用 b_trans=True 实现 weight 转置，extend_params 融合 bias
    extend_params = {"bias_tensor": bias_tensor}
    output_tensor[:] = pypto.matmul(
        input_tensor, weight_tensor, pypto.DT_FP32,
        b_trans=True, extend_params=extend_params
    )


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def linear_kernel_2d_no_bias(
    input_tensor: pypto.Tensor([], pypto.DT_FP32),
    weight_tensor: pypto.Tensor([], pypto.DT_FP32),
    output_tensor: pypto.Tensor([], pypto.DT_FP32),
):
    """2D linear kernel without bias.

    公式: output = input @ weight^T
    """
    # 设置 cube tiling
    m = input_tensor.shape[0]
    k = input_tensor.shape[1]
    n = weight_tensor.shape[0]

    if m <= 64 and k <= 64 and n <= 64:
        pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    elif m <= 2048 and k <= 2048 and n <= 2048:
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    else:
        pypto.set_cube_tile_shapes([256, 256], [256, 256], [256, 256])

    output_tensor[:] = pypto.matmul(
        input_tensor, weight_tensor, pypto.DT_FP32, b_trans=True
    )


# ─────────────────────────────────────────────
# Wrapper 函数（导出接口）
# ─────────────────────────────────────────────

def linear_wrapper(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """linear 算子 wrapper 函数

    对输入张量进行线性变换: output = input @ weight^T + bias

    Args:
        input_tensor: 输入张量, shape [..., in_features], dtype: float32
                      支持 2D、3D、4D 输入
        weight: 权重矩阵, shape [out_features, in_features], dtype: float32
        bias: 偏置向量, shape [out_features], dtype: float32（可选）

    Returns:
        输出张量, shape [..., out_features], dtype: float32
    """
    # 验证输入
    assert input_tensor.is_contiguous(), "Input must be contiguous"
    assert weight.is_contiguous(), "Weight must be contiguous"
    assert input_tensor.shape[-1] == weight.shape[1], \
        f"in_features mismatch: input has {input_tensor.shape[-1]}, weight expects {weight.shape[1]}"

    if bias is not None:
        assert bias.is_contiguous(), "Bias must be contiguous"
        assert bias.shape[0] == weight.shape[0], \
            f"out_features mismatch: bias has {bias.shape[0]}, weight has {weight.shape[0]}"

    # 获取维度信息
    ndim = input_tensor.ndim
    original_shape = input_tensor.shape
    out_features = weight.shape[0]

    if ndim < 2 or ndim > 4:
        raise ValueError(f"Unsupported input dimensions: {ndim}, expected 2-4")

    # 对于 3D/4D 输入，reshape 为 2D
    if ndim > 2:
        # 保存原始形状以便最后 reshape 回去
        batch_dims = original_shape[:-1]  # [..., ]
        flattened_batch = 1
        for d in batch_dims:
            flattened_batch *= d
        # Reshape input to 2D: [flattened_batch, in_features]
        input_2d = input_tensor.view(flattened_batch, -1)
    else:
        input_2d = input_tensor
        flattened_batch = original_shape[0]

    # 创建 2D 输出 tensor
    output_2d = torch.empty(flattened_batch, out_features, dtype=input_tensor.dtype, device=input_tensor.device)

    # 调用 2D kernel
    if bias is not None:
        # bias 需要reshape为 [1, out_features] 以便融合
        bias_2d = bias.unsqueeze(0)  # [1, out_features]
        # 扩展 bias_2d 以匹配 flattened_batch（因为 bias_tensor 要求 shape 匹配）
        # 但实际上 extend_params 的 bias_tensor 会被广播，所以用 [1, out_features] 即可
        linear_kernel_2d_with_bias(input_2d, weight, bias_2d, output_2d)
    else:
        linear_kernel_2d_no_bias(input_2d, weight, output_2d)

    # 对于 3D/4D 输入，reshape 回原始形状
    if ndim > 2:
        output_shape = batch_dims + (out_features,)
        output = output_2d.view(*output_shape)
    else:
        output = output_2d

    return output
