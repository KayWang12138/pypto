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
"""
import pypto
import pytest


def test_quant_mx_basic():
    x = pypto.tensor((8, 128), pypto.DT_FP32, "x")

    with pypto.function("QUANT_MX", x):
        pypto.set_vec_tile_shapes(8, 128)
        quantized, scale = pypto.quant_mx(x)

    assert isinstance(quantized, pypto.tensor)
    assert isinstance(scale, pypto.tensor)
    assert quantized.shape == [8, 128]
    assert scale.shape == [8, 2, 2]
    assert quantized.dtype == pypto.DT_FP8E4M3
    assert scale.dtype == pypto.DT_FP8E8M0


def test_quant_mx_tensor_method():
    x = pypto.tensor((1, 64), pypto.DT_FP32, "x")

    with pypto.function("QUANT_MX_METHOD", x):
        pypto.set_vec_tile_shapes(1, 64)
        quantized, scale = x.quant_mx()

    assert quantized.shape == [1, 64]
    assert scale.shape == [1, 1, 2]


def test_quant_mx_scaled_mm_integration():
    a = pypto.tensor((16, 128), pypto.DT_FP32, "a")
    b = pypto.tensor((64, 128), pypto.DT_FP32, "b")

    with pypto.function("QUANT_MX_MM", a, b):
        pypto.set_vec_tile_shapes(16, 128)
        qa, sa = pypto.quant_mx(a)
        qb, sb = pypto.quant_mx(b)
        pypto.set_cube_tile_shapes([64, 64], [128, 128], [64, 64])
        out = pypto.scaled_mm(qa, qb, pypto.DT_FP16, sa, sb, b_trans=True, scale_b_trans=True)

    assert isinstance(out, pypto.tensor)
    assert out.shape == [16, 64]
    assert out.dtype == pypto.DT_FP16


def test_quant_mx_rejects_non_fp32():
    x = pypto.tensor((8, 128), pypto.DT_FP16, "x")

    with pypto.function("QUANT_MX_BAD_DTYPE", x):
        pypto.set_vec_tile_shapes(8, 128)
        with pytest.raises(RuntimeError):
            pypto.quant_mx(x)


def test_quant_mx_rejects_non_nd():
    x = pypto.tensor((8, 128), pypto.DT_FP32, "x", pypto.TileOpFormat.TILEOP_NZ)

    with pypto.function("QUANT_MX_BAD_FORMAT", x):
        pypto.set_vec_tile_shapes(8, 128)
        with pytest.raises(RuntimeError):
            pypto.quant_mx(x)


def test_quant_mx_rejects_non_2d():
    x = pypto.tensor((2, 8, 128), pypto.DT_FP32, "x")

    with pypto.function("QUANT_MX_BAD_RANK", x):
        pypto.set_vec_tile_shapes(8, 128)
        with pytest.raises(RuntimeError):
            pypto.quant_mx(x)


def test_quant_mx_rejects_unaligned_k():
    x = pypto.tensor((8, 96), pypto.DT_FP32, "x")

    with pypto.function("QUANT_MX_BAD_ALIGN", x):
        pypto.set_vec_tile_shapes(8, 96)
        with pytest.raises(RuntimeError):
            pypto.quant_mx(x)
