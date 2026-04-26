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
        quantized, scale = pypto.quant_mx(x, mode=pypto.ROUND_DOWN)

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
        quantized, scale = x.quant_mx(mode=pypto.ROUND_DOWN)

    assert quantized.shape == [1, 64]
    assert scale.shape == [1, 1, 2]


def test_quant_mx_accepts_positive_last_axis():
    x = pypto.tensor((8, 128), pypto.DT_FP32, "x")

    with pypto.function("QUANT_MX_AXIS_LAST", x):
        pypto.set_vec_tile_shapes(8, 128)
        quantized, scale = pypto.quant_mx(x, axis=1)

    assert quantized.shape == [8, 128]
    assert scale.shape == [8, 2, 2]


def test_quant_mx_supports_3d():
    x = pypto.tensor((2, 8, 128), pypto.DT_FP32, "x")

    with pypto.function("QUANT_MX_3D", x):
        pypto.set_vec_tile_shapes(1, 8, 128)
        quantized, scale = pypto.quant_mx(x, mode=pypto.ROUND_DOWN)

    assert quantized.shape == [2, 8, 128]
    assert scale.shape == [2, 8, 2, 2]


def test_quant_mx_performance_mode_keeps_public_scale_shape():
    x = pypto.tensor((2, 8, 128), pypto.DT_FP32, "x")

    with pypto.function("QUANT_MX_PERFORMANCE_MODE", x):
        pypto.set_vec_tile_shapes(1, 8, 128)
        quantized, scale = pypto.quant_mx(x, mode=pypto.ROUND_DOWN, performance_mode=True)
        vec_tile = pypto.get_vec_tile_shapes()

    assert quantized.shape == [2, 8, 128]
    assert scale.shape == [2, 8, 2, 2]
    assert vec_tile == [1, 8, 128]


def test_quant_mx_performance_mode_rejects_split_last_axis():
    x = pypto.tensor((2, 8, 128), pypto.DT_FP32, "x")

    with pypto.function("QUANT_MX_PERFORMANCE_MODE_BAD_LAST_AXIS", x):
        pypto.set_vec_tile_shapes(1, 8, 64)
        with pytest.raises(RuntimeError):
            pypto.quant_mx(x, mode=pypto.ROUND_DOWN, performance_mode=True)


def test_quant_mx_supports_4d():
    x = pypto.tensor((2, 3, 8, 128), pypto.DT_FP32, "x")

    with pypto.function("QUANT_MX_4D", x):
        pypto.set_vec_tile_shapes(1, 1, 8, 128)
        quantized, scale = pypto.quant_mx(x, mode=pypto.ROUND_DOWN)

    assert quantized.shape == [2, 3, 8, 128]
    assert scale.shape == [2, 3, 8, 2, 2]


def test_quant_mx_rejects_non_float_input():
    x = pypto.tensor((8, 128), pypto.DT_INT32, "x")

    with pypto.function("QUANT_MX_BAD_DTYPE", x):
        pypto.set_vec_tile_shapes(8, 128)
        with pytest.raises(RuntimeError):
            pypto.quant_mx(x, mode=pypto.ROUND_DOWN)


def test_quant_mx_supports_fp16_input():
    x = pypto.tensor((8, 128), pypto.DT_FP16, "x")

    with pypto.function("QUANT_MX_FP16", x):
        pypto.set_vec_tile_shapes(8, 128)
        quantized, scale = pypto.quant_mx(x, mode=pypto.ROUND_DOWN)

    assert quantized.shape == [8, 128]
    assert scale.shape == [8, 2, 2]
    assert quantized.dtype == pypto.DT_FP8E4M3
    assert scale.dtype == pypto.DT_FP8E8M0


def test_quant_mx_supports_explicit_fp8_output():
    x = pypto.tensor((8, 128), pypto.DT_FP32, "x")

    with pypto.function("QUANT_MX_EXPLICIT_FP8", x):
        pypto.set_vec_tile_shapes(8, 128)
        quantized, scale = pypto.quant_mx(x, pypto.DT_FP8E4M3, pypto.ROUND_DOWN)

    assert quantized.shape == [8, 128]
    assert scale.shape == [8, 2, 2]
    assert quantized.dtype == pypto.DT_FP8E4M3
    assert scale.dtype == pypto.DT_FP8E8M0


def test_quant_mx_rejects_round_up_mode():
    x = pypto.tensor((8, 128), pypto.DT_FP32, "x")

    with pypto.function("QUANT_MX_BAD_ROUND_UP", x):
        pypto.set_vec_tile_shapes(8, 128)
        with pytest.raises(RuntimeError):
            pypto.quant_mx(x, pypto.DT_FP8E4M3, pypto.ROUND_UP)


def test_quant_mx_supports_default_round_down_mode():
    x = pypto.tensor((8, 128), pypto.DT_FP32, "x")

    with pypto.function("QUANT_MX_DEFAULT_MODE", x):
        pypto.set_vec_tile_shapes(8, 128)
        quantized, scale = pypto.quant_mx(x)

    assert quantized.shape == [8, 128]
    assert scale.shape == [8, 2, 2]
    assert quantized.dtype == pypto.DT_FP8E4M3
    assert scale.dtype == pypto.DT_FP8E8M0


@pytest.mark.parametrize("quant_dtype", [pypto.DT_FP4_E2M1X2, pypto.DT_FP4_E1M2X2])
def test_quant_mx_rejects_fp4_output(quant_dtype):
    x = pypto.tensor((8, 128), pypto.DT_FP32, "x")

    with pypto.function("QUANT_MX_BAD_FP4_OUTPUT", x):
        pypto.set_vec_tile_shapes(8, 128)
        with pytest.raises(RuntimeError):
            pypto.quant_mx(x, quant_dtype, pypto.ROUND_DOWN)


def test_quant_mx_supports_bf16_input():
    x = pypto.tensor((8, 128), pypto.DT_BF16, "x")

    with pypto.function("QUANT_MX_BF16", x):
        pypto.set_vec_tile_shapes(8, 128)
        quantized, scale = pypto.quant_mx(x, mode=pypto.ROUND_DOWN)

    assert quantized.shape == [8, 128]
    assert scale.shape == [8, 2, 2]
    assert quantized.dtype == pypto.DT_FP8E4M3
    assert scale.dtype == pypto.DT_FP8E8M0


def test_quant_mx_rejects_non_last_axis():
    x = pypto.tensor((8, 128), pypto.DT_FP32, "x")

    with pypto.function("QUANT_MX_BAD_AXIS", x):
        pypto.set_vec_tile_shapes(8, 128)
        with pytest.raises(RuntimeError):
            pypto.quant_mx(x, axis=0)


def test_quant_mx_rejects_fp16_tile_not_256b_aligned():
    x = pypto.tensor((8, 128), pypto.DT_FP16, "x")

    with pypto.function("QUANT_MX_FP16_BAD_TILE_ALIGN", x):
        pypto.set_vec_tile_shapes(8, 64)
        with pytest.raises(RuntimeError):
            pypto.quant_mx(x, mode=pypto.ROUND_DOWN)


def test_quant_mx_rejects_non_nd():
    x = pypto.tensor((8, 128), pypto.DT_FP32, "x", pypto.TileOpFormat.TILEOP_NZ)

    with pypto.function("QUANT_MX_BAD_FORMAT", x):
        pypto.set_vec_tile_shapes(8, 128)
        with pytest.raises(RuntimeError):
            pypto.quant_mx(x, mode=pypto.ROUND_DOWN)


def test_quant_mx_rejects_rank_below_2():
    x = pypto.tensor((128,), pypto.DT_FP32, "x")

    with pypto.function("QUANT_MX_BAD_RANK_LOW", x):
        pypto.set_vec_tile_shapes(128)
        with pytest.raises(RuntimeError):
            pypto.quant_mx(x, mode=pypto.ROUND_DOWN)


def test_quant_mx_rejects_rank_above_4():
    x = pypto.tensor((2, 3, 4, 5, 128), pypto.DT_FP32, "x")

    with pypto.function("QUANT_MX_BAD_RANK_HIGH", x):
        pypto.set_vec_tile_shapes(1, 1, 1, 5, 128)
        with pytest.raises(RuntimeError):
            pypto.quant_mx(x, mode=pypto.ROUND_DOWN)


def test_quant_mx_rejects_unaligned_k():
    x = pypto.tensor((8, 96), pypto.DT_FP32, "x")

    with pypto.function("QUANT_MX_BAD_ALIGN", x):
        pypto.set_vec_tile_shapes(8, 96)
        with pytest.raises(RuntimeError):
            pypto.quant_mx(x, mode=pypto.ROUND_DOWN)
