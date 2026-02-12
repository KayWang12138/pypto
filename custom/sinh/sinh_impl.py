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
sinh 算子实现

sinh(x) = (e^x - e^(-x)) / 2
"""

import pypto


def sinh_op(shape: tuple, run_mode: str = "npu"):
    """
    sinh operator implementation.

    Args:
        shape: Input tensor shape [b, s, n, d].
        run_mode: Execution mode ("npu" or "sim").

    Returns:
        Compiled kernel function.
    """
    b, s, n, d = shape

    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")

    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def sinh_kernel(
        x: pypto.Tensor((b, s, n, d), pypto.DT_FP32),
    ) -> pypto.Tensor((b, s, n, d), pypto.DT_FP32):
        """
        sinh function: (e^x - e^(-x)) / 2

        Formula: sinh(x) = (exp(x) - exp(-x)) / 2

        Args:
            x: Input tensor [b, s, n, d].

        Returns:
            Output tensor [b, s, n, d] with sinh(x) values.
        """
        # Set vector tile shapes for element-wise operations
        # 4维张量需要4个参数
        pypto.set_vec_tile_shapes(32, 32, 32, 32)

        out = pypto.tensor((b, s, n, d), pypto.DT_FP32)

        # sinh(x) = (e^x - e^(-x)) / 2
        exp_x = pypto.exp(x)
        neg_x = pypto.neg(x)  # 使用 neg 函数而不是 -x
        exp_neg_x = pypto.exp(neg_x)
        numerator = pypto.sub(exp_x, exp_neg_x)
        out[:] = pypto.div(numerator, 2.0)

        return out

    return sinh_kernel
