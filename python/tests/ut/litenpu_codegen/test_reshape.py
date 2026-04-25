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
def compare_cos_add(davinci1_input, davinci2_input):
    davinci1_input = davinci1_input.reshape(-1).astype(np.float64)
    davinci2_input = davinci2_input.reshape(-1).astype(np.float64)
    davinci1_input = davinci1_input - 1.0
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
def reshape_kernel_fp16(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 16)
    out_tensor[:] = pypto.reshape(input_tensor, [1, 2, 1, 2])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_fp16_002(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(2, 16)
    out_tensor[:] = pypto.reshape(input_tensor, [16])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_fp16_003(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 1, 16)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 12])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_fp16_004(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 1, 1, 16)
    out_tensor[:] = pypto.reshape(input_tensor, [4, 4])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_fp16_005(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(16)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 4])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_fp16_006(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 16)
    out_tensor[:] = pypto.reshape(input_tensor, [3, 3, 3])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_fp16_007(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 1, 16)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 2, 2, 3])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_fp16_008(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(2, 2, 2, 16)
    out_tensor[:] = pypto.reshape(input_tensor, [256])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_fp16_009(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(16)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 2, 3])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
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

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_fp16_002(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (4, 4)
        shape_out = (16,)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp16_002(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_fp16_003(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (2, 3, 4)
        shape_out = (2, 12)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp16_003(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_fp16_004(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (2, 2, 2, 2)
        shape_out = (4, 4)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp16_004(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_fp16_005(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (8,)
        shape_out = (2, 4)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp16_005(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_fp16_006(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (3, 9)
        shape_out = (3, 3, 3)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp16_006(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_fp16_007(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (2, 2, 6)
        shape_out = (2, 2, 2, 3)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp16_007(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_fp16_008(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (4, 4, 4, 4)
        shape_out = (256,)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp16_008(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_fp16_009(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (12,)
        shape_out = (2, 2, 3)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp16_009(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_fp16_010(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (2, 6)
        shape_out = (2, 2, 3, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp16_010(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_fp32(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 8)
    out_tensor[:] = pypto.reshape(input_tensor, [1, 2, 1, 2])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_fp32_002(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(2, 8)
    out_tensor[:] = pypto.reshape(input_tensor, [16])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_fp32_003(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 1, 8)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 12])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_fp32_004(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 1, 1, 8)
    out_tensor[:] = pypto.reshape(input_tensor, [4, 4])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_fp32_005(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(8)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 4])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_fp32_006(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 8)
    out_tensor[:] = pypto.reshape(input_tensor, [3, 3, 3])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_fp32_007(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 1, 8)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 2, 2, 3])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_fp32_008(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(2, 2, 2, 8)
    out_tensor[:] = pypto.reshape(input_tensor, [256])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_fp32_009(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(8)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 2, 3])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
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

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_fp32_002(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (4, 4)
        shape_out = (16,)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp32_002(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_fp32_003(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 3, 4)
        shape_out = (2, 12)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp32_003(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_fp32_004(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 2, 2, 2)
        shape_out = (4, 4)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp32_004(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_fp32_005(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (8,)
        shape_out = (2, 4)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp32_005(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_fp32_006(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (3, 9)
        shape_out = (3, 3, 3)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp32_006(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_fp32_007(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 2, 6)
        shape_out = (2, 2, 2, 3)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp32_007(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_fp32_008(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (4, 4, 4, 4)
        shape_out = (256,)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp32_008(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_fp32_009(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (12,)
        shape_out = (2, 2, 3)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp32_009(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_fp32_010(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 6)
        shape_out = (2, 2, 3, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_fp32_010(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_inplace_fp32(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 8)
    out_tensor_tmp = pypto.add(input_tensor, 1.0)
    out_tensor[:] = pypto.reshape(out_tensor_tmp, [1, 2, 1, 2], inplace=True)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_inplace_fp32_002(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(2, 8)
    out_tensor_tmp = pypto.add(input_tensor, 1.0)
    out_tensor[:] = pypto.reshape(out_tensor_tmp, [16], True)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_inplace_fp32_003(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 1, 8)
    out_tensor_tmp = pypto.add(input_tensor, 1.0)
    out_tensor[:] = pypto.reshape(out_tensor_tmp, [2, 12], True)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_inplace_fp32_004(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 1, 1, 8)
    out_tensor_tmp = pypto.add(input_tensor, 1.0)
    out_tensor[:] = pypto.reshape(out_tensor_tmp, [4, 4], True)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_inplace_fp32_005(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(8)
    out_tensor_tmp = pypto.add(input_tensor, 1.0)
    out_tensor[:] = pypto.reshape(out_tensor_tmp, [2, 4], True)


class TestLiteNPUReshapeInplaceFP32(unittest.TestCase):
    def test_reshape_inplace_fp32_001(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 2)
        shape_out = (1, 2, 1, 2)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_inplace_fp32(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos_add(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_inplace_fp32_002(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (4, 4)
        shape_out = (16,)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_inplace_fp32_002(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos_add(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_inplace_fp32_003(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 3, 4)
        shape_out = (2, 12)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_inplace_fp32_003(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos_add(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_inplace_fp32_004(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 2, 2, 2)
        shape_out = (4, 4)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_inplace_fp32_004(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos_add(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_inplace_fp32_005(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (8,)
        shape_out = (2, 4)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        reshape_kernel_inplace_fp32_005(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos_add(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_int8(
    input_tensor: pypto.Tensor([...], pypto.DT_INT8),
    out_tensor: pypto.Tensor([...], pypto.DT_INT8),
):
    pypto.set_vec_tile_shapes(1, 32)
    out_tensor[:] = pypto.reshape(input_tensor, [1, 2, 1, 2])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_int8_002(
    input_tensor: pypto.Tensor([...], pypto.DT_INT8),
    out_tensor: pypto.Tensor([...], pypto.DT_INT8),
):
    pypto.set_vec_tile_shapes(2, 32)
    out_tensor[:] = pypto.reshape(input_tensor, [16])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_int8_003(
    input_tensor: pypto.Tensor([...], pypto.DT_INT8),
    out_tensor: pypto.Tensor([...], pypto.DT_INT8),
):
    pypto.set_vec_tile_shapes(32)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 4])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_int8_004(
    input_tensor: pypto.Tensor([...], pypto.DT_INT8),
    out_tensor: pypto.Tensor([...], pypto.DT_INT8),
):
    pypto.set_vec_tile_shapes(1, 1, 32)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 12])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
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

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_int8_002(self):
        device = "cpu"
        dtype = torch.int8
        shape_input = (4, 4)
        shape_out = (16,)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        reshape_kernel_int8_002(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_int8_003(self):
        device = "cpu"
        dtype = torch.int8
        shape_input = (8,)
        shape_out = (2, 4)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        reshape_kernel_int8_003(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_int8_004(self):
        device = "cpu"
        dtype = torch.int8
        shape_input = (2, 3, 4)
        shape_out = (2, 12)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        reshape_kernel_int8_004(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_int8_005(self):
        device = "cpu"
        dtype = torch.int8
        shape_input = (12,)
        shape_out = (2, 2, 3)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        reshape_kernel_int8_005(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_int16(
    input_tensor: pypto.Tensor([...], pypto.DT_INT16),
    out_tensor: pypto.Tensor([...], pypto.DT_INT16),
):
    pypto.set_vec_tile_shapes(1, 16)
    out_tensor[:] = pypto.reshape(input_tensor, [1, 2, 1, 2])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_int16_002(
    input_tensor: pypto.Tensor([...], pypto.DT_INT16),
    out_tensor: pypto.Tensor([...], pypto.DT_INT16),
):
    pypto.set_vec_tile_shapes(2, 16)
    out_tensor[:] = pypto.reshape(input_tensor, [16])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_int16_003(
    input_tensor: pypto.Tensor([...], pypto.DT_INT16),
    out_tensor: pypto.Tensor([...], pypto.DT_INT16),
):
    pypto.set_vec_tile_shapes(16)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 4])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_int16_004(
    input_tensor: pypto.Tensor([...], pypto.DT_INT16),
    out_tensor: pypto.Tensor([...], pypto.DT_INT16),
):
    pypto.set_vec_tile_shapes(1, 1, 16)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 12])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
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

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_int16_002(self):
        device = "cpu"
        dtype = torch.int16
        shape_input = (4, 4)
        shape_out = (16,)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        reshape_kernel_int16_002(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_int16_003(self):
        device = "cpu"
        dtype = torch.int16
        shape_input = (8,)
        shape_out = (2, 4)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        reshape_kernel_int16_003(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_int16_004(self):
        device = "cpu"
        dtype = torch.int16
        shape_input = (2, 3, 4)
        shape_out = (2, 12)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        reshape_kernel_int16_004(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_int16_005(self):
        device = "cpu"
        dtype = torch.int16
        shape_input = (12,)
        shape_out = (2, 2, 3)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        reshape_kernel_int16_005(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_int32(
    input_tensor: pypto.Tensor([...], pypto.DT_INT32),
    out_tensor: pypto.Tensor([...], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(1, 8)
    out_tensor[:] = pypto.reshape(input_tensor, [1, 2, 1, 2])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_int32_002(
    input_tensor: pypto.Tensor([...], pypto.DT_INT32),
    out_tensor: pypto.Tensor([...], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(2, 8)
    out_tensor[:] = pypto.reshape(input_tensor, [16])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_int32_003(
    input_tensor: pypto.Tensor([...], pypto.DT_INT32),
    out_tensor: pypto.Tensor([...], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(8)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 4])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel_int32_004(
    input_tensor: pypto.Tensor([...], pypto.DT_INT32),
    out_tensor: pypto.Tensor([...], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(1, 1, 8)
    out_tensor[:] = pypto.reshape(input_tensor, [2, 12])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
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

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_int32_002(self):
        device = "cpu"
        dtype = torch.int32
        shape_input = (4, 4)
        shape_out = (16,)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        reshape_kernel_int32_002(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_int32_003(self):
        device = "cpu"
        dtype = torch.int32
        shape_input = (8,)
        shape_out = (2, 4)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        reshape_kernel_int32_003(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_int32_004(self):
        device = "cpu"
        dtype = torch.int32
        shape_input = (2, 3, 4)
        shape_out = (2, 12)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        reshape_kernel_int32_004(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_reshape_int32_005(self):
        device = "cpu"
        dtype = torch.int32
        shape_input = (12,)
        shape_out = (2, 2, 3)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        reshape_kernel_int32_005(input_tensor, out_tensor)

        golden_out = input_tensor.reshape(shape_out)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)


if __name__ == '__main__':
    unittest.main()
