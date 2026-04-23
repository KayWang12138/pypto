#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Test reshape codegen
"""

import pypto
import torch
import numpy as np
import unittest


def compare_cos(a, b):
    a = a.flatten()
    b = b.flatten()
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_fp16(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 16)
    out_tensor[:] = pypto.reshape(input_tensor, [1, 2, 1, 2])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_fp16_002(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(2, 16)
    out_tensor[:] = pypto.reshape(input_tensor, [16])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_fp16_003(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 1, 16)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 12])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_fp16_004(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 1, 1, 16)
    out_tensor[:] = pypto.reshape(input_tensor, [4, 4])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_fp16_005(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(16)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 4])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_fp16_006(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 16)
    out_tensor[:] = pypto.reshape(input_tensor, [3, 3, 3])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_fp16_007(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 1, 16)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 2, 2, 3])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_fp16_008(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(2, 2, 2, 16)
    out_tensor[:] = pypto.reshape(input_tensor, [256])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_fp16_009(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(16)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 2, 3])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_fp16_010(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 16)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 2, 3, 1])


class TestLiteNPUReshapeFP16(unittest.TestCase):
    def test_reshape_fp16_001(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (2, 2)
        shape_out = (1, 2, 1, 2)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp16(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_fp16_002(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (4, 4)
        shape_out = (16,)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp16_002(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_fp16_003(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (2, 3, 4)
        shape_out = (2, 12)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp16_003(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_fp16_004(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (2, 2, 2, 2)
        shape_out = (4, 4)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp16_004(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_fp16_005(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (8,)
        shape_out = (2, 4)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp16_005(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_fp16_006(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (3, 9)
        shape_out = (3, 3, 3)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp16_006(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_fp16_007(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (2, 2, 6)
        shape_out = (2, 2, 2, 3)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp16_007(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_fp16_008(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (4, 4, 4, 4)
        shape_out = (256,)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp16_008(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_fp16_009(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (12,)
        shape_out = (2, 2, 3)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp16_009(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_fp16_010(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (2, 6)
        shape_out = (2, 2, 3, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp16_010(input_tensor, out_tensor)

        self.assertEqual(1, 1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_fp32(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 8)
    out_tensor[:] = pypto.reshape(input_tensor, [1, 2, 1, 2])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_fp32_002(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(2, 8)
    out_tensor[:] = pypto.reshape(input_tensor, [16])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_fp32_003(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 1, 8)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 12])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_fp32_004(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 1, 1, 8)
    out_tensor[:] = pypto.reshape(input_tensor, [4, 4])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_fp32_005(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(8)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 4])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_fp32_006(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 8)
    out_tensor[:] = pypto.reshape(input_tensor, [3, 3, 3])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_fp32_007(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 1, 8)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 2, 2, 3])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_fp32_008(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(2, 2, 2, 8)
    out_tensor[:] = pypto.reshape(input_tensor, [256])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_fp32_009(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(8)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 2, 3])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_fp32_010(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 8)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 2, 3, 1])


class TestLiteNPUReshapeFP32(unittest.TestCase):
    def test_reshape_fp32_001(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 2)
        shape_out = (1, 2, 1, 2)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp32(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_fp32_002(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (4, 4)
        shape_out = (16,)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp32_002(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_fp32_003(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 3, 4)
        shape_out = (2, 12)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp32_003(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_fp32_004(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 2, 2, 2)
        shape_out = (4, 4)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp32_004(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_fp32_005(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (8,)
        shape_out = (2, 4)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp32_005(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_fp32_006(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (3, 9)
        shape_out = (3, 3, 3)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp32_006(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_fp32_007(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 2, 6)
        shape_out = (2, 2, 2, 3)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp32_007(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_fp32_008(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (4, 4, 4, 4)
        shape_out = (256,)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp32_008(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_fp32_009(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (12,)
        shape_out = (2, 2, 3)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp32_009(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_fp32_010(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 6)
        shape_out = (2, 2, 3, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp32_010(input_tensor, out_tensor)

        self.assertEqual(1, 1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_inplace_fp32(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 8)
    out_tensor[:] = pypto.reshape(input_tensor, [1, 2, 1, 2], True)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_inplace_fp32_002(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(2, 8)
    out_tensor[:] = pypto.reshape(input_tensor, [16], True)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_inplace_fp32_003(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 1, 8)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 12], True)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_inplace_fp32_004(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 1, 1, 8)
    out_tensor[:] = pypto.reshape(input_tensor, [4, 4], True)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_inplace_fp32_005(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(8)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 4], True)


class TestLiteNPUReshapeInplaceFP32(unittest.TestCase):
    def test_reshape_inplace_fp32_001(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 2)
        shape_out = (1, 2, 1, 2)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_inplace_fp32(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_inplace_fp32_002(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (4, 4)
        shape_out = (16,)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_inplace_fp32_002(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_inplace_fp32_003(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 3, 4)
        shape_out = (2, 12)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_inplace_fp32_003(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_inplace_fp32_004(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 2, 2, 2)
        shape_out = (4, 4)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_inplace_fp32_004(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_inplace_fp32_005(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (8,)
        shape_out = (2, 4)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_inplace_fp32_005(input_tensor, out_tensor)

        self.assertEqual(1, 1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_int8(
    input_tensor: pypto.Tensor([...], pypto.DT_INT8),
    out_tensor: pypto.Tensor([...], pypto.DT_INT8),
):
    pypto.set_vec_tile_shapes(1, 32)
    out_tensor[:] = pypto.reshape(input_tensor, [1, 2, 1, 2])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_int8_002(
    input_tensor: pypto.Tensor([...], pypto.DT_INT8),
    out_tensor: pypto.Tensor([...], pypto.DT_INT8),
):
    pypto.set_vec_tile_shapes(2, 32)
    out_tensor[:] = pypto.reshape(input_tensor, [16])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_int8_003(
    input_tensor: pypto.Tensor([...], pypto.DT_INT8),
    out_tensor: pypto.Tensor([...], pypto.DT_INT8),
):
    pypto.set_vec_tile_shapes(32)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 4])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_int8_004(
    input_tensor: pypto.Tensor([...], pypto.DT_INT8),
    out_tensor: pypto.Tensor([...], pypto.DT_INT8),
):
    pypto.set_vec_tile_shapes(1, 1, 32)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 12])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_int8_005(
    input_tensor: pypto.Tensor([...], pypto.DT_INT8),
    out_tensor: pypto.Tensor([...], pypto.DT_INT8),
):
    pypto.set_vec_tile_shapes(32)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 2, 3])


class TestLiteNPUReshapeInt8(unittest.TestCase):
    def test_reshape_int8_001(self):
        device = "cpu"
        dtype = torch.int8
        shape_input = (2, 2)
        shape_out = (1, 2, 1, 2)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        reshape_kernel_int8(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_int8_002(self):
        device = "cpu"
        dtype = torch.int8
        shape_input = (4, 4)
        shape_out = (16,)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        reshape_kernel_int8_002(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_int8_003(self):
        device = "cpu"
        dtype = torch.int8
        shape_input = (8,)
        shape_out = (2, 4)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        reshape_kernel_int8_003(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_int8_004(self):
        device = "cpu"
        dtype = torch.int8
        shape_input = (2, 3, 4)
        shape_out = (2, 12)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        reshape_kernel_int8_004(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_int8_005(self):
        device = "cpu"
        dtype = torch.int8
        shape_input = (12,)
        shape_out = (2, 2, 3)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        reshape_kernel_int8_005(input_tensor, out_tensor)

        self.assertEqual(1, 1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_int16(
    input_tensor: pypto.Tensor([...], pypto.DT_INT16),
    out_tensor: pypto.Tensor([...], pypto.DT_INT16),
):
    pypto.set_vec_tile_shapes(1, 16)
    out_tensor[:] = pypto.reshape(input_tensor, [1, 2, 1, 2])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_int16_002(
    input_tensor: pypto.Tensor([...], pypto.DT_INT16),
    out_tensor: pypto.Tensor([...], pypto.DT_INT16),
):
    pypto.set_vec_tile_shapes(2, 16)
    out_tensor[:] = pypto.reshape(input_tensor, [16])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_int16_003(
    input_tensor: pypto.Tensor([...], pypto.DT_INT16),
    out_tensor: pypto.Tensor([...], pypto.DT_INT16),
):
    pypto.set_vec_tile_shapes(16)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 4])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_int16_004(
    input_tensor: pypto.Tensor([...], pypto.DT_INT16),
    out_tensor: pypto.Tensor([...], pypto.DT_INT16),
):
    pypto.set_vec_tile_shapes(1, 1, 16)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 12])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_int16_005(
    input_tensor: pypto.Tensor([...], pypto.DT_INT16),
    out_tensor: pypto.Tensor([...], pypto.DT_INT16),
):
    pypto.set_vec_tile_shapes(16)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 2, 3])


class TestLiteNPUReshapeInt16(unittest.TestCase):
    def test_reshape_int16_001(self):
        device = "cpu"
        dtype = torch.int16
        shape_input = (2, 2)
        shape_out = (1, 2, 1, 2)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        reshape_kernel_int16(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_int16_002(self):
        device = "cpu"
        dtype = torch.int16
        shape_input = (4, 4)
        shape_out = (16,)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        reshape_kernel_int16_002(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_int16_003(self):
        device = "cpu"
        dtype = torch.int16
        shape_input = (8,)
        shape_out = (2, 4)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        reshape_kernel_int16_003(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_int16_004(self):
        device = "cpu"
        dtype = torch.int16
        shape_input = (2, 3, 4)
        shape_out = (2, 12)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        reshape_kernel_int16_004(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_int16_005(self):
        device = "cpu"
        dtype = torch.int16
        shape_input = (12,)
        shape_out = (2, 2, 3)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        reshape_kernel_int16_005(input_tensor, out_tensor)

        self.assertEqual(1, 1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_int32(
    input_tensor: pypto.Tensor([...], pypto.DT_INT32),
    out_tensor: pypto.Tensor([...], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(1, 8)
    out_tensor[:] = pypto.reshape(input_tensor, [1, 2, 1, 2])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_int32_002(
    input_tensor: pypto.Tensor([...], pypto.DT_INT32),
    out_tensor: pypto.Tensor([...], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(2, 8)
    out_tensor[:] = pypto.reshape(input_tensor, [16])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_int32_003(
    input_tensor: pypto.Tensor([...], pypto.DT_INT32),
    out_tensor: pypto.Tensor([...], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(8)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 4])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_int32_004(
    input_tensor: pypto.Tensor([...], pypto.DT_INT32),
    out_tensor: pypto.Tensor([...], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(1, 1, 8)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 12])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.SIM})
def reshape_kernel_int32_005(
    input_tensor: pypto.Tensor([...], pypto.DT_INT32),
    out_tensor: pypto.Tensor([...], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(8)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 2, 3])


class TestLiteNPUReshapeInt32(unittest.TestCase):
    def test_reshape_int32_001(self):
        device = "cpu"
        dtype = torch.int32
        shape_input = (2, 2)
        shape_out = (1, 2, 1, 2)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        reshape_kernel_int32(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_int32_002(self):
        device = "cpu"
        dtype = torch.int32
        shape_input = (4, 4)
        shape_out = (16,)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        reshape_kernel_int32_002(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_int32_003(self):
        device = "cpu"
        dtype = torch.int32
        shape_input = (8,)
        shape_out = (2, 4)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        reshape_kernel_int32_003(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_int32_004(self):
        device = "cpu"
        dtype = torch.int32
        shape_input = (2, 3, 4)
        shape_out = (2, 12)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        reshape_kernel_int32_004(input_tensor, out_tensor)

        self.assertEqual(1, 1)

    def test_reshape_int32_005(self):
        device = "cpu"
        dtype = torch.int32
        shape_input = (12,)
        shape_out = (2, 2, 3)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        reshape_kernel_int32_005(input_tensor, out_tensor)

        self.assertEqual(1, 1)


if __name__ == '__main__':
    unittest.main()
