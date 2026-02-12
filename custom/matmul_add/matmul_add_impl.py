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
matmul_add 算子实现

y = a @ b^T + c
"""

import pypto


def matmul_add_op(shape: tuple, run_mode: str = "npu"):
    """
    matmul_add operator implementation.

    Args:
        shape: Matrix shapes (m, k, n).
        run_mode: Execution mode ("npu" or "sim").

    Returns:
        Compiled kernel function.
    """
    m, k, n = shape

    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")

    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def matmul_add_kernel(
        a: pypto.Tensor((m, k), pypto.DT_BF16),
        b: pypto.Tensor((n, k), pypto.DT_BF16),
        c: pypto.Tensor((m, n), pypto.DT_BF16),
    ) -> pypto.Tensor((m, n), pypto.DT_BF16):
        """
        matmul_add function: a @ b^T + c

        Formula: y = matmul(a, transpose(b)) + c

        Args:
            a: Input tensor [m, k].
            b: Input tensor [n, k].
            c: Input tensor [m, n].

        Returns:
            Output tensor [m, n] with matmul_add(a, b, c) values.
        """
        # Set vector tile shapes for transpose operation
        pypto.set_vec_tile_shapes(32, 32)

        # Transpose b: [n, k] -> [k, n]
        b_t = pypto.transpose(b, 0, 1)

        # Set cube tile shapes for matmul optimization
        pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])

        # Compute matmul: a @ b^T = [m, k] @ [k, n] = [m, n]
        matmul_result = pypto.matmul(a, b_t, pypto.DT_BF16)

        # Add bias: matmul_result + c = [m, n] + [m, n] = [m, n]
        out = pypto.add(matmul_result, c)

        return out

    return matmul_add_kernel
