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
Test transpose codegen
"""

import pypto
import torch
import numpy as np
import unittest


def compare_cos(davinci1_input, davinci2_input):
    davinci1_input = davinci1_input.reshape(-1).astype(np.float64)
    davinci2_input = davinci2_input.reshape(-1).astype(np.float64)
    print(davinci1_input.shape)
    print(davinci2_input.shape)
    print("NPU_out:",davinci1_input)
    print("CPU_out:",davinci2_input)
    print("max diff: ", np.max(np.abs(davinci1_input-davinci2_input)))
    index = np.argmax(np.abs(davinci1_input-davinci2_input))
    print("max diff index = ", index, " dav1 value: ", davinci1_input[index], "dav2 value: ", davinci2_input[index])
    print("average diff: ", np.mean(np.abs(davinci1_input - davinci2_input)))
    ab = np.sum(np.multiply(davinci1_input, davinci2_input))
    aa = np.sqrt(np.sum(np.multiply(davinci1_input, davinci1_input)))
    bb = np.sqrt(np.sum(np.multiply(davinci2_input, davinci2_input)))
    if aa*bb == 0 and ab == 0:
        cos = 1.0
    else:
        cos = ab / (aa*bb)
    print(cos)
    print()
    return cos


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_fp16(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 16)
    out_tensor[:] = pypto.transpose(input_tensor, 0, 1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_fp16_003(
    input_tensor: pypto.Tensor([3, 4, 5], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 2, 16)
    out_tensor[:] = pypto.transpose(input_tensor, 1, 2)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_fp16_004(
    input_tensor: pypto.Tensor([2, 3, 4], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 3, 16)
    out_tensor[:] = pypto.transpose(input_tensor, 0, 2)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_fp16_005(
    input_tensor: pypto.Tensor([2, 3, 4, 5], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 1, 2, 16)
    out_tensor[:] = pypto.transpose(input_tensor, 2, 3)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_fp16_006(
    input_tensor: pypto.Tensor([2, 3, 4, 5], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 3, 1, 16)
    out_tensor[:] = pypto.transpose(input_tensor, 1, 2)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_fp16_007(
    input_tensor: pypto.Tensor([2, 3, 4, 5, 6], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 1, 1, 2, 16)
    out_tensor[:] = pypto.transpose(input_tensor, 3, 4)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_fp16_008(
    input_tensor: pypto.Tensor([4, 5], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(2, 16)
    out_tensor[:] = pypto.transpose(input_tensor, 0, 1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_fp16_009(
    input_tensor: pypto.Tensor([2, 5, 6], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 5, 16)
    out_tensor[:] = pypto.transpose(input_tensor, 1, 2)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_fp16_010(
    input_tensor: pypto.Tensor([3, 2, 4, 5], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 2, 2, 16)
    out_tensor[:] = pypto.transpose(input_tensor, 2, 3)


class TestLiteNPUTransposeFP16(unittest.TestCase):
    def test_transpose_fp16_001(self):
        device = "cpu"
        dtype = torch.float16
        shape = (2, 3)

        input_tensor = torch.rand(shape, dtype=dtype, device=device)
        out_tensor = torch.rand(shape[::-1], dtype=dtype, device=device)

        transpose_kernel_fp16(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 0, 1)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_fp16_002(self):
        device = "cpu"
        dtype = torch.float16
        shape = (3, 4)

        input_tensor = torch.rand(shape, dtype=dtype, device=device)
        out_tensor = torch.rand(shape[::-1], dtype=dtype, device=device)

        transpose_kernel_fp16(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 0, 1)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_fp16_003(self):
        device = "cpu"
        dtype = torch.float16
        shape = (3, 4, 5)

        input_tensor = torch.rand(shape, dtype=dtype, device=device)
        out_tensor = torch.rand(shape[0], shape[2], shape[1], dtype=dtype, device=device)

        transpose_kernel_fp16_003(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 1, 2)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_fp16_004(self):
        device = "cpu"
        dtype = torch.float16
        shape = (2, 3, 4)

        input_tensor = torch.rand(shape, dtype=dtype, device=device)
        out_tensor = torch.rand(shape[2], shape[1], shape[0], dtype=dtype, device=device)

        transpose_kernel_fp16_004(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 0, 2)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_fp16_005(self):
        device = "cpu"
        dtype = torch.float16
        shape = (2, 3, 4, 5)

        input_tensor = torch.rand(shape, dtype=dtype, device=device)
        out_tensor = torch.rand(shape[0], shape[1], shape[3], shape[2], dtype=dtype, device=device)

        transpose_kernel_fp16_005(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 2, 3)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_fp16_006(self):
        device = "cpu"
        dtype = torch.float16
        shape = (2, 3, 4, 5)

        input_tensor = torch.rand(shape, dtype=dtype, device=device)
        out_tensor = torch.rand(shape[0], shape[2], shape[1], shape[3], dtype=dtype, device=device)

        transpose_kernel_fp16_006(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 1, 2)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_fp16_007(self):
        device = "cpu"
        dtype = torch.float16
        shape = (2, 3, 4, 5, 6)

        input_tensor = torch.rand(shape, dtype=dtype, device=device)
        out_tensor = torch.rand(shape[0], shape[1], shape[2], shape[4], shape[3], dtype=dtype, device=device)

        transpose_kernel_fp16_007(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 3, 4)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_fp16_008(self):
        device = "cpu"
        dtype = torch.float16
        shape = (4, 5)

        input_tensor = torch.rand(shape, dtype=dtype, device=device)
        out_tensor = torch.rand(shape[::-1], dtype=dtype, device=device)

        transpose_kernel_fp16_008(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 0, 1)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_fp16_009(self):
        device = "cpu"
        dtype = torch.float16
        shape = (2, 5, 6)

        input_tensor = torch.rand(shape, dtype=dtype, device=device)
        out_tensor = torch.rand(shape[0], shape[2], shape[1], dtype=dtype, device=device)

        transpose_kernel_fp16_009(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 1, 2)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_fp16_010(self):
        device = "cpu"
        dtype = torch.float16
        shape = (3, 2, 4, 5)

        input_tensor = torch.rand(shape, dtype=dtype, device=device)
        out_tensor = torch.rand(shape[0], shape[1], shape[3], shape[2], dtype=dtype, device=device)

        transpose_kernel_fp16_010(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 2, 3)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_fp32(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 8)
    out_tensor[:] = pypto.transpose(input_tensor, 0, 1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_fp32_003(
    input_tensor: pypto.Tensor([3, 4, 5], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 2, 8)
    out_tensor[:] = pypto.transpose(input_tensor, 1, 2)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_fp32_004(
    input_tensor: pypto.Tensor([2, 3, 4], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 3, 8)
    out_tensor[:] = pypto.transpose(input_tensor, 0, 2)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_fp32_005(
    input_tensor: pypto.Tensor([2, 3, 4, 5], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 1, 2, 8)
    out_tensor[:] = pypto.transpose(input_tensor, 2, 3)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_fp32_006(
    input_tensor: pypto.Tensor([2, 3, 4, 5], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 3, 1, 8)
    out_tensor[:] = pypto.transpose(input_tensor, 1, 2)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_fp32_007(
    input_tensor: pypto.Tensor([2, 3, 4, 5, 6], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 1, 1, 2, 8)
    out_tensor[:] = pypto.transpose(input_tensor, 3, 4)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_fp32_008(
    input_tensor: pypto.Tensor([4, 5], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(2, 8)
    out_tensor[:] = pypto.transpose(input_tensor, 0, 1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_fp32_009(
    input_tensor: pypto.Tensor([2, 5, 6], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 5, 8)
    out_tensor[:] = pypto.transpose(input_tensor, 1, 2)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_fp32_010(
    input_tensor: pypto.Tensor([3, 2, 4, 5], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 2, 2, 8)
    out_tensor[:] = pypto.transpose(input_tensor, 2, 3)


class TestLiteNPUTransposeFP32(unittest.TestCase):
    def test_transpose_fp32_001(self):
        device = "cpu"
        dtype = torch.float32
        shape = (2, 3)

        input_tensor = torch.rand(shape, dtype=dtype, device=device)
        out_tensor = torch.rand(shape[::-1], dtype=dtype, device=device)

        transpose_kernel_fp32(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 0, 1)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_fp32_002(self):
        device = "cpu"
        dtype = torch.float32
        shape = (3, 4)

        input_tensor = torch.rand(shape, dtype=dtype, device=device)
        out_tensor = torch.rand(shape[::-1], dtype=dtype, device=device)

        transpose_kernel_fp32(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 0, 1)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_fp32_003(self):
        device = "cpu"
        dtype = torch.float32
        shape = (3, 4, 5)

        input_tensor = torch.rand(shape, dtype=dtype, device=device)
        out_tensor = torch.rand(shape[0], shape[2], shape[1], dtype=dtype, device=device)

        transpose_kernel_fp32_003(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 1, 2)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_fp32_004(self):
        device = "cpu"
        dtype = torch.float32
        shape = (2, 3, 4)

        input_tensor = torch.rand(shape, dtype=dtype, device=device)
        out_tensor = torch.rand(shape[2], shape[1], shape[0], dtype=dtype, device=device)

        transpose_kernel_fp32_004(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 0, 2)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_fp32_005(self):
        device = "cpu"
        dtype = torch.float32
        shape = (2, 3, 4, 5)

        input_tensor = torch.rand(shape, dtype=dtype, device=device)
        out_tensor = torch.rand(shape[0], shape[1], shape[3], shape[2], dtype=dtype, device=device)

        transpose_kernel_fp32_005(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 2, 3)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_fp32_006(self):
        device = "cpu"
        dtype = torch.float32
        shape = (2, 3, 4, 5)

        input_tensor = torch.rand(shape, dtype=dtype, device=device)
        out_tensor = torch.rand(shape[0], shape[2], shape[1], shape[3], dtype=dtype, device=device)

        transpose_kernel_fp32_006(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 1, 2)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_fp32_007(self):
        device = "cpu"
        dtype = torch.float32
        shape = (2, 3, 4, 5, 6)

        input_tensor = torch.rand(shape, dtype=dtype, device=device)
        out_tensor = torch.rand(shape[0], shape[1], shape[2], shape[4], shape[3], dtype=dtype, device=device)

        transpose_kernel_fp32_007(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 3, 4)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_fp32_008(self):
        device = "cpu"
        dtype = torch.float32
        shape = (4, 5)

        input_tensor = torch.rand(shape, dtype=dtype, device=device)
        out_tensor = torch.rand(shape[::-1], dtype=dtype, device=device)

        transpose_kernel_fp32_008(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 0, 1)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_fp32_009(self):
        device = "cpu"
        dtype = torch.float32
        shape = (2, 5, 6)

        input_tensor = torch.rand(shape, dtype=dtype, device=device)
        out_tensor = torch.rand(shape[0], shape[2], shape[1], dtype=dtype, device=device)

        transpose_kernel_fp32_009(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 1, 2)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_fp32_010(self):
        device = "cpu"
        dtype = torch.float32
        shape = (3, 2, 4, 5)

        input_tensor = torch.rand(shape, dtype=dtype, device=device)
        out_tensor = torch.rand(shape[0], shape[1], shape[3], shape[2], dtype=dtype, device=device)

        transpose_kernel_fp32_010(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 2, 3)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_int32(
    input_tensor: pypto.Tensor([...], pypto.DT_INT32),
    out_tensor: pypto.Tensor([...], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(1, 8)
    out_tensor[:] = pypto.transpose(input_tensor, 0, 1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_int32_003(
    input_tensor: pypto.Tensor([3, 4, 5], pypto.DT_INT32),
    out_tensor: pypto.Tensor([...], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(1, 2, 8)
    out_tensor[:] = pypto.transpose(input_tensor, 1, 2)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_int32_004(
    input_tensor: pypto.Tensor([2, 3, 4, 5], pypto.DT_INT32),
    out_tensor: pypto.Tensor([...], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(1, 1, 2, 8)
    out_tensor[:] = pypto.transpose(input_tensor, 2, 3)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_int32_005(
    input_tensor: pypto.Tensor([4, 5], pypto.DT_INT32),
    out_tensor: pypto.Tensor([...], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(2, 8)
    out_tensor[:] = pypto.transpose(input_tensor, 0, 1)


class TestLiteNPUTransposeInt32(unittest.TestCase):
    def test_transpose_int32_001(self):
        device = "cpu"
        dtype = torch.int32
        shape = (2, 3)

        input_tensor = torch.randint(0, 100, shape, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape[::-1], dtype=dtype, device=device)

        transpose_kernel_int32(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 0, 1)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_int32_002(self):
        device = "cpu"
        dtype = torch.int32
        shape = (3, 4)

        input_tensor = torch.randint(0, 100, shape, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape[::-1], dtype=dtype, device=device)

        transpose_kernel_int32(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 0, 1)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_int32_003(self):
        device = "cpu"
        dtype = torch.int32
        shape = (3, 4, 5)

        input_tensor = torch.randint(0, 100, shape, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, (shape[0], shape[2], shape[1]), dtype=dtype, device=device)

        transpose_kernel_int32_003(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 1, 2)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_int32_004(self):
        device = "cpu"
        dtype = torch.int32
        shape = (2, 3, 4, 5)

        input_tensor = torch.randint(0, 100, shape, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, (shape[0], shape[1], shape[3], shape[2]), dtype=dtype, device=device)

        transpose_kernel_int32_004(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 2, 3)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_int32_005(self):
        device = "cpu"
        dtype = torch.int32
        shape = (4, 5)

        input_tensor = torch.randint(0, 100, shape, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape[::-1], dtype=dtype, device=device)

        transpose_kernel_int32_005(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 0, 1)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_int16(
    input_tensor: pypto.Tensor([...], pypto.DT_INT16),
    out_tensor: pypto.Tensor([...], pypto.DT_INT16),
):
    pypto.set_vec_tile_shapes(1, 16)
    out_tensor[:] = pypto.transpose(input_tensor, 0, 1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_int16_003(
    input_tensor: pypto.Tensor([3, 4, 5], pypto.DT_INT16),
    out_tensor: pypto.Tensor([...], pypto.DT_INT16),
):
    pypto.set_vec_tile_shapes(1, 2, 16)
    out_tensor[:] = pypto.transpose(input_tensor, 1, 2)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_int16_004(
    input_tensor: pypto.Tensor([2, 3, 4], pypto.DT_INT16),
    out_tensor: pypto.Tensor([...], pypto.DT_INT16),
):
    pypto.set_vec_tile_shapes(1, 3, 16)
    out_tensor[:] = pypto.transpose(input_tensor, 0, 2)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_int16_005(
    input_tensor: pypto.Tensor([2, 3, 4, 5], pypto.DT_INT16),
    out_tensor: pypto.Tensor([...], pypto.DT_INT16),
):
    pypto.set_vec_tile_shapes(1, 1, 2, 16)
    out_tensor[:] = pypto.transpose(input_tensor, 2, 3)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_int16_006(
    input_tensor: pypto.Tensor([2, 3, 4, 5], pypto.DT_INT16),
    out_tensor: pypto.Tensor([...], pypto.DT_INT16),
):
    pypto.set_vec_tile_shapes(1, 3, 1, 16)
    out_tensor[:] = pypto.transpose(input_tensor, 1, 2)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_int16_007(
    input_tensor: pypto.Tensor([2, 3, 4, 5, 6], pypto.DT_INT16),
    out_tensor: pypto.Tensor([...], pypto.DT_INT16),
):
    pypto.set_vec_tile_shapes(1, 1, 1, 2, 16)
    out_tensor[:] = pypto.transpose(input_tensor, 3, 4)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_int16_008(
    input_tensor: pypto.Tensor([4, 5], pypto.DT_INT16),
    out_tensor: pypto.Tensor([...], pypto.DT_INT16),
):
    pypto.set_vec_tile_shapes(2, 16)
    out_tensor[:] = pypto.transpose(input_tensor, 0, 1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_int16_009(
    input_tensor: pypto.Tensor([2, 5, 6], pypto.DT_INT16),
    out_tensor: pypto.Tensor([...], pypto.DT_INT16),
):
    pypto.set_vec_tile_shapes(1, 5, 16)
    out_tensor[:] = pypto.transpose(input_tensor, 1, 2)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_int16_010(
    input_tensor: pypto.Tensor([3, 2, 4, 5], pypto.DT_INT16),
    out_tensor: pypto.Tensor([...], pypto.DT_INT16),
):
    pypto.set_vec_tile_shapes(1, 2, 2, 16)
    out_tensor[:] = pypto.transpose(input_tensor, 2, 3)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def transpose_kernel_int16_011(
    input_tensor: pypto.Tensor([3, 2, 4, 5], pypto.DT_INT16),
    out_tensor: pypto.Tensor([...], pypto.DT_INT16),
):
    pypto.set_vec_tile_shapes(2, 2, 2, 16)
    out_tensor[:] = pypto.transpose(input_tensor, 2, 3)


class TestLiteNPUTransposeInt16(unittest.TestCase):
    def test_transpose_int16_001(self):
        device = "cpu"
        dtype = torch.int16
        shape = (2, 3)

        input_tensor = torch.randint(0, 100, shape, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape[::-1], dtype=dtype, device=device)

        transpose_kernel_int16(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 0, 1)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_int16_002(self):
        device = "cpu"
        dtype = torch.int16
        shape = (3, 4)

        input_tensor = torch.randint(0, 100, shape, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape[::-1], dtype=dtype, device=device)

        transpose_kernel_int16(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 0, 1)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_int16_003(self):
        device = "cpu"
        dtype = torch.int16
        shape = (3, 4, 5)

        input_tensor = torch.randint(0, 100, shape, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, (shape[0], shape[2], shape[1]), dtype=dtype, device=device)

        transpose_kernel_int16_003(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 1, 2)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_int16_004(self):
        device = "cpu"
        dtype = torch.int16
        shape = (2, 3, 4)

        input_tensor = torch.randint(0, 100, shape, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, (shape[2], shape[1], shape[0]), dtype=dtype, device=device)

        transpose_kernel_int16_004(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 0, 2)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_int16_005(self):
        device = "cpu"
        dtype = torch.int16
        shape = (2, 3, 4, 5)

        input_tensor = torch.randint(0, 100, shape, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, (shape[0], shape[1], shape[3], shape[2]), dtype=dtype, device=device)

        transpose_kernel_int16_005(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 2, 3)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_int16_006(self):
        device = "cpu"
        dtype = torch.int16
        shape = (2, 3, 4, 5)

        input_tensor = torch.randint(0, 100, shape, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, (shape[0], shape[2], shape[1], shape[3]), dtype=dtype, device=device)

        transpose_kernel_int16_006(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 1, 2)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_int16_007(self):
        device = "cpu"
        dtype = torch.int16
        shape = (2, 3, 4, 5, 6)

        input_tensor = torch.randint(0, 100, shape, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, (shape[0], shape[1], shape[2], shape[4], shape[3]), dtype=dtype, device=device)

        transpose_kernel_int16_007(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 3, 4)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_int16_008(self):
        device = "cpu"
        dtype = torch.int16
        shape = (4, 5)

        input_tensor = torch.randint(0, 100, shape, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape[::-1], dtype=dtype, device=device)

        transpose_kernel_int16_008(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 0, 1)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_int16_009(self):
        device = "cpu"
        dtype = torch.int16
        shape = (2, 5, 6)

        input_tensor = torch.randint(0, 100, shape, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, (shape[0], shape[2], shape[1]), dtype=dtype, device=device)

        transpose_kernel_int16_009(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 1, 2)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_int16_010(self):
        device = "cpu"
        dtype = torch.int16
        shape = (3, 2, 4, 5)

        input_tensor = torch.randint(0, 100, shape, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, (shape[0], shape[1], shape[3], shape[2]), dtype=dtype, device=device)

        transpose_kernel_int16_010(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 2, 3)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_transpose_int16_011(self):
        device = "cpu"
        dtype = torch.int16
        shape = (3, 2, 4, 5)

        input_tensor = torch.randint(0, 100, shape, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, (shape[0], shape[1], shape[3], shape[2]), dtype=dtype, device=device)

        transpose_kernel_int16_011(input_tensor, out_tensor)

        golden_out = torch.transpose(input_tensor, 2, 3)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)


if __name__ == '__main__':
    unittest.main()
