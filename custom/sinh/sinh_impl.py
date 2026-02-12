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
Sinh Activation Function Implementation for PyPTO

Formula: sinh(x) = (e^x - e^{-x}) / 2
"""

import pypto
import torch


def configure_tiling(x):
    """Configure vector tile shapes based on input tensor shape."""
    if len(x.shape) >= 2:
        tile_list = [32 for _ in range(len(x.shape))]
        pypto.set_vec_tile_shapes(*tile_list)
    else:
        pypto.set_vec_tile_shapes(32, 128)


def sinh_op(x: torch.Tensor, run_mode: str = "npu") -> torch.Tensor:
    """
    Compute sinh(x) = (e^x - e^{-x}) / 2

    Args:
        x: Input tensor
        run_mode: Run mode ('npu' or 'sim')

    Returns:
        Output tensor with sinh(x) computed
    """
    x_shape = x.shape

    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")

    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def sinh_kernel(
        x: pypto.Tensor(x_shape, pypto.DT_FP32),
    ) -> pypto.Tensor(x_shape, pypto.DT_FP32):
        """
        Sinh (hyperbolic sine) activation function.

        Formula: sinh(x) = (e^x - e^{-x}) / 2

        This implementation uses the following steps:
        1. Compute e^x
        2. Compute e^{-x} by first negating x, then computing exp
        3. Subtract e^{-x} from e^x
        4. Divide the result by 2
        """
        configure_tiling(x)

        exp_x = pypto.exp(x)
        neg_x = pypto.neg(x)
        exp_neg_x = pypto.exp(neg_x)
        diff = pypto.sub(exp_x, exp_neg_x)
        result = pypto.div(diff, 2.0)

        return result

    out = sinh_kernel(x)
    return out
