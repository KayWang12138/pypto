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
Test amin codegen
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
def amin_kernel_fp16_001(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(48)
    out_tensor[:] = pypto.amin(input_tensor, -1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp16_002(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(96)
    out_tensor[:] = pypto.amin(input_tensor, -1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp16_003(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(2, 32)
    out_tensor[:] = pypto.amin(input_tensor, -1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp16_004(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 128)
    out_tensor[:] = pypto.amin(input_tensor, -1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp16_005(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 2, 32)
    out_tensor[:] = pypto.amin(input_tensor, -1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp16_006(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 2, 128)
    out_tensor[:] = pypto.amin(input_tensor, -1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp16_007(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 5, 32)
    out_tensor[:] = pypto.amin(input_tensor, -1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp16_008(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 3, 160)
    out_tensor[:] = pypto.amin(input_tensor, -1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp16_009(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(2, 1, 2, 16)
    out_tensor[:] = pypto.amin(input_tensor, -1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp16_010(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 1, 1, 128)
    out_tensor[:] = pypto.amin(input_tensor, -1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp16_011(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 1, 5, 32)
    out_tensor[:] = pypto.amin(input_tensor, -1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp16_012(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(2, 2, 3, 32)
    out_tensor[:] = pypto.amin(input_tensor, -1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp16_013(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(6, 2, 4, 128)
    out_tensor[:] = pypto.amin(input_tensor, -1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp16_014(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 2, 1, 128)
    out_tensor[:] = pypto.amin(input_tensor, -1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp16_015(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(3, 3, 5, 32)
    out_tensor[:] = pypto.amin(input_tensor, -1)


class TestLiteNPUAminFP16(unittest.TestCase):
    def test_amin_fp16_001(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (112,)
        shape_out = (1,)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp16_001(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_amin_fp16_002(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (100,)
        shape_out = (1,)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp16_002(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_amin_fp16_003(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (4, 128)
        shape_out = (4, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp16_003(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_amin_fp16_004(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (4, 130)
        shape_out = (4, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp16_004(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_amin_fp16_005(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (2, 4, 160)
        shape_out = (2, 4, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp16_005(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_amin_fp16_006(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (2, 4, 140)
        shape_out = (2, 4, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp16_006(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_amin_fp16_007(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (2, 5, 152)
        shape_out = (2, 5, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp16_007(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_amin_fp16_008(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (2, 3, 170)
        shape_out = (2, 3, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp16_008(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_amin_fp16_009(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (5, 2, 4, 176)
        shape_out = (5, 2, 4, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp16_009(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_amin_fp16_010(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (5, 2, 4, 130)
        shape_out = (5, 2, 4, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp16_010(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_amin_fp16_011(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (2, 3, 5, 134)
        shape_out = (2, 3, 5, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp16_011(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_amin_fp16_012(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (4, 2, 6, 135)
        shape_out = (4, 2, 6, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp16_012(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_amin_fp16_013(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (6, 2, 4, 130)
        shape_out = (6, 2, 4, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp16_013(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_amin_fp16_014(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (3, 2, 3, 139)
        shape_out = (3, 2, 3, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp16_014(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_amin_fp16_015(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (6, 3, 5, 141)
        shape_out = (6, 3, 5, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp16_015(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp32_001(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(48)
    out_tensor[:] = pypto.amin(input_tensor, -1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp32_002(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(96)
    out_tensor[:] = pypto.amin(input_tensor, -1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp32_003(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(2, 32)
    out_tensor[:] = pypto.amin(input_tensor, -1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp32_004(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 128)
    out_tensor[:] = pypto.amin(input_tensor, -1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp32_005(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 2, 32)
    out_tensor[:] = pypto.amin(input_tensor, -1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp32_006(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 2, 128)
    out_tensor[:] = pypto.amin(input_tensor, -1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp32_007(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 5, 32)
    out_tensor[:] = pypto.amin(input_tensor, -1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp32_008(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 3, 168)
    out_tensor[:] = pypto.amin(input_tensor, -1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp32_009(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(2, 1, 2, 16)
    out_tensor[:] = pypto.amin(input_tensor, -1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp32_010(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 1, 1, 128)
    out_tensor[:] = pypto.amin(input_tensor, -1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp32_011(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 1, 5, 32)
    out_tensor[:] = pypto.amin(input_tensor, -1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp32_012(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(2, 2, 3, 32)
    out_tensor[:] = pypto.amin(input_tensor, -1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp32_013(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 1, 4, 128)
    out_tensor[:] = pypto.amin(input_tensor, -1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp32_014(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 2, 1, 136)
    out_tensor[:] = pypto.amin(input_tensor, -1)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def amin_kernel_fp32_015(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(3, 3, 5, 32)
    out_tensor[:] = pypto.amin(input_tensor, -1)


class TestLiteNPUAminFP32(unittest.TestCase):
    def test_amin_fp32_001(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (112,)
        shape_out = (1,)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp32_001(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_amin_fp32_002(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (100,)
        shape_out = (1,)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp32_002(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_amin_fp32_003(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (4, 128)
        shape_out = (4, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp32_003(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_amin_fp32_004(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (4, 130)
        shape_out = (4, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp32_004(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_amin_fp32_005(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 4, 160)
        shape_out = (2, 4, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp32_005(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_amin_fp32_006(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 4, 140)
        shape_out = (2, 4, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp32_006(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_amin_fp32_007(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 5, 152)
        shape_out = (2, 5, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp32_007(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_amin_fp32_008(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 3, 170)
        shape_out = (2, 3, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp32_008(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_amin_fp32_009(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (5, 2, 4, 176)
        shape_out = (5, 2, 4, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp32_009(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_amin_fp32_010(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (5, 2, 4, 130)
        shape_out = (5, 2, 4, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp32_010(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_amin_fp32_011(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 3, 5, 134)
        shape_out = (2, 3, 5, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp32_011(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_amin_fp32_012(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (4, 2, 6, 135)
        shape_out = (4, 2, 6, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp32_012(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_amin_fp32_013(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (6, 2, 4, 130)
        shape_out = (6, 2, 4, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp32_013(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_amin_fp32_014(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (3, 2, 3, 139)
        shape_out = (3, 2, 3, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp32_014(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_amin_fp32_015(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (6, 3, 5, 141)
        shape_out = (6, 3, 5, 1)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        amin_kernel_fp32_015(input_tensor, out_tensor)

        golden_out = torch.amin(input_tensor, dim=-1, keepdim=True)
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)


if __name__ == '__main__':
    unittest.main()
