#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------
"""
matmul 算子 PyPTO 实现

功能: 矩阵乘法算子，公式: C = A @ B
"""

import pypto
import torch

def matmul_core(a: pypto.Tensor, b: pypto.Tensor, out: pypto.Tensor):
    """核心 matmul 计算"""
    # 根据矩阵大小选择 tiling
    m, k, n = a.shape[-2], a.shape[-1], b.shape[-1]
    
    if m <= 64 and k <= 64 and n <= 64:
        pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    elif m <= 2048 and k <= 2048 and n <= 2048:
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    else:
        pypto.set_cube_tile_shapes([256, 256], [256, 256], [256, 256])
    
    # 执行 matmul
    out[:] = pypto.matmul(a, b, a.dtype)


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def matmul_kernel_2d(
    a: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    b: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32)
):
    """2D matmul kernel"""
    matmul_core(a, b, out)

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def matmul_kernel_3d(
    a: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    b: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32)
):
    """3D batch matmul kernel"""
    # 设置 vector tiling
    pypto.set_vec_tile_shapes(128, 128)
    matmul_core(a, b, out)

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def matmul_kernel_4d(
    a: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    b: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32)
):
    """4D batch matmul kernel"""
    # 设置 vector tiling
    pypto.set_vec_tile_shapes(128, 128, 128, 128)
    matmul_core(a, b, out)

def matmul_wrapper(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """算子 wrapper 函数
    
    Args:
        a: 左操作数
        b: 右操作数
    
    Returns:
        输出矩阵
    """
    # 验证输入
    assert a.is_contiguous(), "Input A must be contiguous"
    assert b.is_contiguous(), "Input B must be contiguous"
    assert a.shape[-1] == b.shape[-2], f"K dimension mismatch: {a.shape[-1]} vs {b.shape[-2]}"
    
    # 获取输出 shape
    output_shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2]) + (a.shape[-2], b.shape[-1])
    output = torch.empty(output_shape, dtype=a.dtype, device=a.device)
    
    # 转换为 PyPTO tensor
    a_pto = pypto.from_torch(a)
    b_pto = pypto.from_torch(b)
    out_pto = pypto.from_torch(output)
    
    # 根据 dim 选择 kernel
    ndim = a.ndim
    if ndim == 2:
        matmul_kernel_2d(a_pto, b_pto, out_pto)
    elif ndim == 3:
        matmul_kernel_3d(a_pto, b_pto, out_pto)
    elif ndim == 4:
        matmul_kernel_4d(a_pto, b_pto, out_pto)
    else:
        raise ValueError(f"Unsupported dimensions: {ndim}")
    
    return output
