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
Simple Configurable Matrix Multiplication (matmul) Operator

This is a simplified version where:
- Matrix shapes (M, K, N) and tile shapes are hardcoded in __main__
- Uses simple torch.allclose for verification
"""

import os
import torch
import pypto
import numpy as np


def golden_matmul(a_np, b_np, dtype_str='float32'):
    """Compute matrix multiplication using numpy or torch (golden reference)."""
    if dtype_str == 'bfloat16':
        a_torch = torch.from_numpy(a_np.astype(np.float32)).to(dtype=torch.bfloat16)
        b_torch = torch.from_numpy(b_np.astype(np.float32)).to(dtype=torch.bfloat16)
        result_torch = torch.matmul(a_torch, b_torch)
        return result_torch.float().numpy()
    else:
        return np.matmul(a_np, b_np)


def create_matmul_kernel(dtype_pypto, tile_m, tile_k, tile_n):
    """Create a matmul kernel function with specified dtype and tile shapes."""
    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 1}
    )
    def matmul_kernel(
        a: pypto.Tensor([], dtype_pypto),
        b: pypto.Tensor([], dtype_pypto),
        out: pypto.Tensor([], dtype_pypto),
    ):
        pypto.set_cube_tile_shapes(tile_m, tile_k, tile_n)
        out[:] = pypto.matmul(a, b, dtype_pypto)
    return matmul_kernel


if __name__ == "__main__":
    # 硬编码参数 - 在这里修改来配置算子
    M, K, N = 64, 128, 256          # 矩阵维度: A[M,K] @ B[K,N] = C[M,N]
    tile_m = [32, 32]                # M维度的tile大小 [mL0, mL1]
    tile_k = [32, 32]                # K维度的tile大小 [kL0, kL1]
    tile_n = [32, 32]                # N维度的tile大小 [nL0, nL1]
    dtype_str = 'float16'            # 数据类型: float32/float16/bfloat16
    seed = 42                        # 随机数种子
    
    # 检查环境变量
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("ERROR: Please set TILE_FWK_DEVICE_ID environment variable")
        print("Example: export TILE_FWK_DEVICE_ID=0")
        exit(1)
    
    device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
    device = f'npu:{device_id}'
    
    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 数据类型映射
    if dtype_str == 'float32':
        torch_dtype, np_dtype, pypto_dtype = torch.float32, np.float32, pypto.DT_FP32
    elif dtype_str == 'float16':
        torch_dtype, np_dtype, pypto_dtype = torch.float16, np.float16, pypto.DT_FP16
    elif dtype_str == 'bfloat16':
        torch_dtype, np_dtype, pypto_dtype = torch.bfloat16, np.float32, pypto.DT_BF16
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    
    # 生成测试数据
    a_np = np.random.randn(M, K).astype(np_dtype)
    b_np = np.random.randn(K, N).astype(np_dtype)
    expected_np = golden_matmul(a_np, b_np, dtype_str)
    
    a_torch = torch.from_numpy(a_np).to(dtype=torch_dtype, device=device)
    b_torch = torch.from_numpy(b_np).to(dtype=torch_dtype, device=device)
    out_torch = torch.empty((M, N), dtype=torch_dtype, device=device)
    
    # 创建并运行内核
    matmul_kernel = create_matmul_kernel(pypto_dtype, tile_m, tile_k, tile_n)
    matmul_kernel(a_torch, b_torch, out_torch)
    
    # 简单验证
    if dtype_str == 'bfloat16':
        out_np = out_torch.float().cpu().numpy()
        expected_np = expected_np.astype(np.float32)
    else:
        out_np = out_torch.cpu().numpy()
    
    # 使用torch的allclose进行简单校验
    if torch.allclose(torch.from_numpy(out_np), torch.from_numpy(expected_np), rtol=1e-3, atol=1e-3):
        print("✓ Test passed!")
        print(f"Shapes: A{M,K} @ B{K,N} = C{M,N}")
        print(f"Dtype: {dtype_str}")
        print(f"Tile: M{tile_m}, K{tile_k}, N{tile_n}")
    else:
        print("✗ Test failed!")
        print(f"Max diff: {np.max(np.abs(out_np - expected_np))}")
        exit(1)