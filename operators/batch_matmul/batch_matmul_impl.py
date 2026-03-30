#!/usr/bin/env python3
# coding: utf-8
"""
batch_matmul PyPTO kernel implementation.

批量矩阵乘法算子实现。
- 支持批量矩阵乘法: [batch, m, k] @ [batch, k, n]
- 支持可选转置配置: transpose_x1, transpose_x2
"""

import pypto
import torch


@pypto.frontend.jit
def batch_matmul_kernel(
    x1: pypto.Tensor([], pypto.DT_FP32),
    x2: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
    transpose_x1: bool = False,
    transpose_x2: bool = False,
):
    """PyPTO JIT kernel for batch matrix multiplication.

    根据输入 shape 动态计算 y = x1 @ x2， 输出写回到 out.

    Args:
        x1: 左矩阵 [batch, m, k]
        x2: 右矩阵 [batch, k, n]
        out: 输出矩阵 [batch, m, n]
        transpose_x1: 是否对左矩阵转置
        transpose_x2: 是否对右矩阵转置
    """
    # 设置 Cube Tiling（matmul 必须配置）
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    # 3D matmul 需要设置 vec_tile_shapes
    pypto.set_vec_tile_shapes(128, 128, 128)

    # 根据 transpose 参数选择计算方式
    if transpose_x1 and transpose_x2:
        result = pypto.matmul(x1, x2, pypto.DT_FP32, a_trans=True, b_trans=True)
    elif transpose_x1:
        result = pypto.matmul(x1, x2, pypto.DT_FP32, a_trans=True)
    elif transpose_x2:
        result = pypto.matmul(x1, x2, pypto.DT_FP32, b_trans=True)
    else:
        result = pypto.matmul(x1, x2, pypto.DT_FP32)

    # 输出写回
    out[:] = result


def batch_matmul_wrapper(
    x1: torch.Tensor,
    x2: torch.Tensor,
    transpose_x1: bool = False,
    transpose_x2: bool = False
) -> torch.Tensor:
    """算子 wrapper，供 test_batch_matmul.py 调用。

    负责:
    1. 构造输出 torch.Tensor
    2. 调用 JIT kernel
    3. 返回结果 torch.Tensor

    Args:
        x1: 左矩阵 [batch, m, k]
        x2: 右矩阵 [batch, k, n]
        transpose_x1: 是否对左矩阵转置
        transpose_x2: 是否对右矩阵转置

    Returns:
        输出矩阵 [batch, m, n]
    """
    # 输出 shape: [batch, m, n]
    batch1, m, k1 = x1.shape
    batch2, k2, n = x2.shape
    batch = max(batch1, batch2)  # 支持广播场景
    k = k1  # k 维度必须一致

    output = torch.empty((batch, m, n), dtype=x1.dtype, device=x1.device)

    batch_matmul_kernel(x1, x2, output, transpose_x1, transpose_x2)

    return output


