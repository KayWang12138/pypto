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
Test assemble codegen
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

def pypto_assemble_in_torch(source, target, offsets=None):
    """
    source: 可以是 (tensor, offset) 的列表，或者是单个 tensor
    target: 目标 tensor (out)
    offsets: 如果 source 是单个 tensor，则需要指定偏移
    """
    if isinstance(source, list):
        for s_tensor, s_off in source:
            # 生成切片索引
            slices = []
            for i, off in enumerate(s_off):
                slices.append(slice(off, off + s_tensor.shape[i]))
            target[tuple(slices)] = s_tensor
    else:
        # 生成切片索引
        slices = []
        for i, off in enumerate(offsets):
            slices.append(slice(off, off + source.shape[i]))
        target[tuple(slices)] = source

@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_fp16(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 16)
    pypto.assemble(input_tensor, [0, 0], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_fp16_002(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 16)
    pypto.assemble(input_tensor, [1, 1], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_fp16_003(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 1, 16)
    pypto.assemble(input_tensor, [0, 0, 0], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_fp16_004(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(16)
    pypto.assemble(input_tensor, [1], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_fp16_005(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 16)
    pypto.assemble(input_tensor, [0, 2], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_fp16_006(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 1, 16)
    pypto.assemble(input_tensor, [0, 1, 1], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_fp16_007(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 1, 1, 16)
    pypto.assemble(input_tensor, [0, 0, 0, 0], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_fp16_008(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(16)
    pypto.assemble(input_tensor, [0], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_fp16_009(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 16)
    pypto.assemble(input_tensor, [0, 0], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_fp16_010(
    input1: pypto.Tensor([...], pypto.DT_FP16),
    input2: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 16)
    pypto.assemble([(input1, [0, 0]), (input2, [2, 2])], out_tensor, True)


class TestLiteNPUAssembleFP16(unittest.TestCase):
    def test_assemble_fp16_001(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (2, 2)
        shape_out = (4, 4)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.zeros(shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [0, 0])

        assemble_kernel_fp16(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_fp16_002(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (3, 3)
        shape_out = (5, 5)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.zeros(shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [1, 1])

        assemble_kernel_fp16_002(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_fp16_003(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (2, 2, 2)
        shape_out = (3, 3, 3)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.zeros(shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [0, 0, 0])

        assemble_kernel_fp16_003(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_fp16_004(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (4,)
        shape_out = (6,)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.zeros(shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [1])

        assemble_kernel_fp16_004(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_fp16_005(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (2, 4)
        shape_out = (3, 6)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.zeros(shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [0, 2])

        assemble_kernel_fp16_005(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_fp16_006(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (1, 2, 2)
        shape_out = (2, 3, 3)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.zeros(shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [0, 1, 1])

        assemble_kernel_fp16_006(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_fp16_007(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (2, 2, 2, 2)
        shape_out = (3, 3, 3, 3)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.zeros(shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [0, 0, 0, 0])

        assemble_kernel_fp16_007(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_fp16_008(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (3,)
        shape_out = (5,)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.zeros(shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [0])

        assemble_kernel_fp16_008(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_fp16_009(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (1, 3)
        shape_out = (2, 5)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.zeros(shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [0, 0])

        assemble_kernel_fp16_009(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_fp16_010(self):
        device = "cpu"
        dtype = torch.float16
        shape_input1 = (2, 2)
        shape_input2 = (2, 2)
        shape_out = (4, 4)

        input1 = torch.rand(shape_input1, dtype=dtype, device=device)
        input2 = torch.rand(shape_input2, dtype=dtype, device=device)
        out_tensor = torch.zeros(shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch([(input1, [0, 0]), (input2, [2, 2])], golden_out)

        assemble_kernel_fp16_010(input1, input2, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_fp32(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 8)
    pypto.assemble(input_tensor, [0, 0], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_fp32_002(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 8)
    pypto.assemble(input_tensor, [1, 1], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_fp32_003(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 1, 8)
    pypto.assemble(input_tensor, [0, 0, 0], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_fp32_004(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(8)
    pypto.assemble(input_tensor, [1], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_fp32_005(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 8)
    pypto.assemble(input_tensor, [0, 2], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_fp32_006(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 1, 8)
    pypto.assemble(input_tensor, [0, 1, 1], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_fp32_007(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 1, 1, 8)
    pypto.assemble(input_tensor, [0, 0, 0, 0], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_fp32_008(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(8)
    pypto.assemble(input_tensor, [0], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_fp32_009(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 8)
    pypto.assemble(input_tensor, [0, 0], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_fp32_010(
    input1: pypto.Tensor([...], pypto.DT_FP32),
    input2: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 8)
    pypto.assemble([(input1, [0, 0]), (input2, [2, 2])], out_tensor, True)


class TestLiteNPUAssembleFP32(unittest.TestCase):
    def test_assemble_fp32_001(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 2)
        shape_out = (4, 4)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.zeros(shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [0, 0])

        assemble_kernel_fp32(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_fp32_002(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (3, 3)
        shape_out = (5, 5)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.zeros(shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [1, 1])

        assemble_kernel_fp32_002(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_fp32_003(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 2, 2)
        shape_out = (3, 3, 3)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.zeros(shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [0, 0, 0])

        assemble_kernel_fp32_003(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_fp32_004(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (4,)
        shape_out = (6,)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.zeros(shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [1])

        assemble_kernel_fp32_004(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_fp32_005(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 4)
        shape_out = (3, 6)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.zeros(shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [0, 2])

        assemble_kernel_fp32_005(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_fp32_006(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (1, 2, 2)
        shape_out = (2, 3, 3)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.zeros(shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [0, 1, 1])

        assemble_kernel_fp32_006(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_fp32_007(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 2, 2, 2)
        shape_out = (3, 3, 3, 3)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.zeros(shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [0, 0, 0, 0])

        assemble_kernel_fp32_007(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_fp32_008(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (3,)
        shape_out = (5,)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.zeros(shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [0])

        assemble_kernel_fp32_008(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_fp32_009(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (1, 3)
        shape_out = (2, 5)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.zeros(shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [0, 0])

        assemble_kernel_fp32_009(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_fp32_010(self):
        device = "cpu"
        dtype = torch.float32
        shape_input1 = (2, 2)
        shape_input2 = (2, 2)
        shape_out = (4, 4)

        input1 = torch.rand(shape_input1, dtype=dtype, device=device)
        input2 = torch.rand(shape_input2, dtype=dtype, device=device)
        out_tensor = torch.zeros(shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch([(input1, [0, 0]), (input2, [2, 2])], golden_out)

        assemble_kernel_fp32_010(input1, input2, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_int8(
    input_tensor: pypto.Tensor([...], pypto.DT_INT8),
    out_tensor: pypto.Tensor([...], pypto.DT_INT8),
):
    pypto.set_vec_tile_shapes(1, 32)
    pypto.assemble(input_tensor, [0, 0], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_int8_002(
    input_tensor: pypto.Tensor([...], pypto.DT_INT8),
    out_tensor: pypto.Tensor([...], pypto.DT_INT8),
):
    pypto.set_vec_tile_shapes(1, 32)
    pypto.assemble(input_tensor, [1, 1], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_int8_003(
    input_tensor: pypto.Tensor([...], pypto.DT_INT8),
    out_tensor: pypto.Tensor([...], pypto.DT_INT8),
):
    pypto.set_vec_tile_shapes(32)
    pypto.assemble(input_tensor, [1], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_int8_004(
    input_tensor: pypto.Tensor([...], pypto.DT_INT8),
    out_tensor: pypto.Tensor([...], pypto.DT_INT8),
):
    pypto.set_vec_tile_shapes(1, 32)
    pypto.assemble(input_tensor, [0, 2], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_int8_005(
    input1: pypto.Tensor([...], pypto.DT_INT8),
    input2: pypto.Tensor([...], pypto.DT_INT8),
    out_tensor: pypto.Tensor([...], pypto.DT_INT8),
):
    pypto.set_vec_tile_shapes(1, 32)
    pypto.assemble([(input1, [0, 0]), (input2, [2, 2])], out_tensor, True)


class TestLiteNPUAssembleInt8(unittest.TestCase):
    def test_assemble_int8_001(self):
        device = "cpu"
        dtype = torch.int8
        shape_input = (2, 2)
        shape_out = (4, 4)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.zerosint(0, 100, shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [0, 0])

        assemble_kernel_int8(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_int8_002(self):
        device = "cpu"
        dtype = torch.int8
        shape_input = (3, 3)
        shape_out = (5, 5)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.zerosint(0, 100, shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [1, 1])

        assemble_kernel_int8_002(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_int8_003(self):
        device = "cpu"
        dtype = torch.int8
        shape_input = (4,)
        shape_out = (6,)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.zerosint(0, 100, shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [1])

        assemble_kernel_int8_003(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_int8_004(self):
        device = "cpu"
        dtype = torch.int8
        shape_input = (2, 4)
        shape_out = (3, 6)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.zerosint(0, 100, shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [0, 2])

        assemble_kernel_int8_004(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_int8_005(self):
        device = "cpu"
        dtype = torch.int8
        shape_input1 = (2, 2)
        shape_input2 = (2, 2)
        shape_out = (4, 4)

        input1 = torch.randint(0, 100, shape_input1, dtype=dtype, device=device)
        input2 = torch.randint(0, 100, shape_input2, dtype=dtype, device=device)
        out_tensor = torch.zerosint(0, 100, shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch([(input1, [0, 0]), (input2, [2, 2])], golden_out)

        assemble_kernel_int8_005(input1, input2, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_int16(
    input_tensor: pypto.Tensor([...], pypto.DT_INT16),
    out_tensor: pypto.Tensor([...], pypto.DT_INT16),
):
    pypto.set_vec_tile_shapes(1, 16)
    pypto.assemble(input_tensor, [0, 0], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_int16_002(
    input_tensor: pypto.Tensor([...], pypto.DT_INT16),
    out_tensor: pypto.Tensor([...], pypto.DT_INT16),
):
    pypto.set_vec_tile_shapes(1, 16)
    pypto.assemble(input_tensor, [1, 1], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_int16_003(
    input_tensor: pypto.Tensor([...], pypto.DT_INT16),
    out_tensor: pypto.Tensor([...], pypto.DT_INT16),
):
    pypto.set_vec_tile_shapes(16)
    pypto.assemble(input_tensor, [1], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_int16_004(
    input_tensor: pypto.Tensor([...], pypto.DT_INT16),
    out_tensor: pypto.Tensor([...], pypto.DT_INT16),
):
    pypto.set_vec_tile_shapes(1, 16)
    pypto.assemble(input_tensor, [0, 2], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_int16_005(
    input1: pypto.Tensor([...], pypto.DT_INT16),
    input2: pypto.Tensor([...], pypto.DT_INT16),
    out_tensor: pypto.Tensor([...], pypto.DT_INT16),
):
    pypto.set_vec_tile_shapes(1, 16)
    pypto.assemble([(input1, [0, 0]), (input2, [2, 2])], out_tensor, True)


class TestLiteNPUAssembleInt16(unittest.TestCase):
    def test_assemble_int16_001(self):
        device = "cpu"
        dtype = torch.int16
        shape_input = (2, 2)
        shape_out = (4, 4)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.zerosint(0, 100, shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [0, 0])

        assemble_kernel_int16(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_int16_002(self):
        device = "cpu"
        dtype = torch.int16
        shape_input = (3, 3)
        shape_out = (5, 5)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.zerosint(0, 100, shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [1, 1])

        assemble_kernel_int16_002(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_int16_003(self):
        device = "cpu"
        dtype = torch.int16
        shape_input = (4,)
        shape_out = (6,)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.zerosint(0, 100, shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [1])

        assemble_kernel_int16_003(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_int16_004(self):
        device = "cpu"
        dtype = torch.int16
        shape_input = (2, 4)
        shape_out = (3, 6)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.zerosint(0, 100, shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [0, 2])

        assemble_kernel_int16_004(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_int16_005(self):
        device = "cpu"
        dtype = torch.int16
        shape_input1 = (2, 2)
        shape_input2 = (2, 2)
        shape_out = (4, 4)

        input1 = torch.randint(0, 100, shape_input1, dtype=dtype, device=device)
        input2 = torch.randint(0, 100, shape_input2, dtype=dtype, device=device)
        out_tensor = torch.zerosint(0, 100, shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch([(input1, [0, 0]), (input2, [2, 2])], golden_out)

        assemble_kernel_int16_005(input1, input2, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_int32(
    input_tensor: pypto.Tensor([...], pypto.DT_INT32),
    out_tensor: pypto.Tensor([...], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(1, 8)
    pypto.assemble(input_tensor, [0, 0], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_int32_002(
    input_tensor: pypto.Tensor([...], pypto.DT_INT32),
    out_tensor: pypto.Tensor([...], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(1, 8)
    pypto.assemble(input_tensor, [1, 1], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_int32_003(
    input_tensor: pypto.Tensor([...], pypto.DT_INT32),
    out_tensor: pypto.Tensor([...], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(8)
    pypto.assemble(input_tensor, [1], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_int32_004(
    input_tensor: pypto.Tensor([...], pypto.DT_INT32),
    out_tensor: pypto.Tensor([...], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(1, 8)
    pypto.assemble(input_tensor, [0, 2], out_tensor)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_int32_005(
    input1: pypto.Tensor([...], pypto.DT_INT32),
    input2: pypto.Tensor([...], pypto.DT_INT32),
    out_tensor: pypto.Tensor([...], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(1, 8)
    pypto.assemble([(input1, [0, 0]), (input2, [2, 2])], out_tensor, True)


class TestLiteNPUAssembleInt32(unittest.TestCase):
    def test_assemble_int32_001(self):
        device = "cpu"
        dtype = torch.int32
        shape_input = (2, 2)
        shape_out = (4, 4)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.zerosint(0, 100, shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [0, 0])

        assemble_kernel_int32(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_int32_002(self):
        device = "cpu"
        dtype = torch.int32
        shape_input = (3, 3)
        shape_out = (5, 5)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.zerosint(0, 100, shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [1, 1])

        assemble_kernel_int32_002(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_int32_003(self):
        device = "cpu"
        dtype = torch.int32
        shape_input = (4,)
        shape_out = (6,)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.zerosint(0, 100, shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [1])

        assemble_kernel_int32_003(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_int32_004(self):
        device = "cpu"
        dtype = torch.int32
        shape_input = (2, 4)
        shape_out = (3, 6)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.zerosint(0, 100, shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch(input_tensor, golden_out, [0, 2])

        assemble_kernel_int32_004(input_tensor, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_int32_005(self):
        device = "cpu"
        dtype = torch.int32
        shape_input1 = (2, 2)
        shape_input2 = (2, 2)
        shape_out = (4, 4)

        input1 = torch.randint(0, 100, shape_input1, dtype=dtype, device=device)
        input2 = torch.randint(0, 100, shape_input2, dtype=dtype, device=device)
        out_tensor = torch.zerosint(0, 100, shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch([(input1, [0, 0]), (input2, [2, 2])], golden_out)

        assemble_kernel_int32_005(input1, input2, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_list_fp32_001(
    input1: pypto.Tensor([...], pypto.DT_FP32),
    input2: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 8)
    pypto.assemble([(input1, [0, 0]), (input2, [2, 2])], out_tensor, False)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_list_fp32_002(
    input1: pypto.Tensor([...], pypto.DT_FP32),
    input2: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(8)
    pypto.assemble([(input1, [1]), (input2, [3])], out_tensor, False)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_list_fp32_003(
    input1: pypto.Tensor([...], pypto.DT_FP32),
    input2: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 1, 8)
    pypto.assemble([(input1, [0, 0, 0]), (input2, [1, 1, 1])], out_tensor, False)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_list_multi_shape_001(
    input1: pypto.Tensor([...], pypto.DT_FP32),
    input2: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 8)
    pypto.assemble([(input1, [0, 0]), (input2, [2, 2])], out_tensor, False)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_list_multi_shape_002(
    input1: pypto.Tensor([...], pypto.DT_FP32),
    input2: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 8)
    pypto.assemble([(input1, [0, 0]), (input2, [2, 0])], out_tensor, False)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def assemble_kernel_list_multi_shape_003(
    input1: pypto.Tensor([...], pypto.DT_FP32),
    input2: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(10, 80)
    pypto.assemble([(input1, [0, 0]), (input2, [2, 0])], out_tensor, False)


class TestLiteNPUAssembleList(unittest.TestCase):
    def test_assemble_list_fp32_001(self):
        device = "cpu"
        dtype = torch.float32
        shape_input1 = (2, 2)
        shape_input2 = (2, 2)
        shape_out = (4, 4)

        input1 = torch.rand(shape_input1, dtype=dtype, device=device)
        input2 = torch.rand(shape_input2, dtype=dtype, device=device)
        out_tensor = torch.zeros(shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch([(input1, [0, 0]), (input2, [2, 2])], golden_out)

        assemble_kernel_list_fp32_001(input1, input2, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_list_fp32_002(self):
        device = "cpu"
        dtype = torch.float32
        shape_input1 = (4,)
        shape_input2 = (4,)
        shape_out = (6,)

        input1 = torch.rand(shape_input1, dtype=dtype, device=device)
        input2 = torch.rand(shape_input2, dtype=dtype, device=device)
        out_tensor = torch.zeros(shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch([(input1, [1]), (input2, [3])], golden_out)

        assemble_kernel_list_fp32_002(input1, input2, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_list_fp32_003(self):
        device = "cpu"
        dtype = torch.float32
        shape_input1 = (2, 2, 2)
        shape_input2 = (2, 2, 2)
        shape_out = (3, 3, 3)

        input1 = torch.rand(shape_input1, dtype=dtype, device=device)
        input2 = torch.rand(shape_input2, dtype=dtype, device=device)
        out_tensor = torch.zeros(shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch([(input1, [0, 0, 0]), (input2, [1, 1, 1])], golden_out)

        assemble_kernel_list_fp32_003(input1, input2, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_list_multi_shape_001(self):
        device = "cpu"
        dtype = torch.float32
        shape_input1 = (2, 2)
        shape_input2 = (2, 3)
        shape_out = (4, 6)

        input1 = torch.rand(shape_input1, dtype=dtype, device=device)
        input2 = torch.rand(shape_input2, dtype=dtype, device=device)
        out_tensor = torch.zeros(shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch([(input1, [0, 0]), (input2, [2, 2])], golden_out)

        assemble_kernel_list_multi_shape_001(input1, input2, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_list_multi_shape_002(self):
        device = "cpu"
        dtype = torch.float32
        shape_input1 = (3, 2)
        shape_input2 = (3, 2)
        shape_out = (5, 4)

        input1 = torch.rand(shape_input1, dtype=dtype, device=device)
        input2 = torch.rand(shape_input2, dtype=dtype, device=device)
        out_tensor = torch.zeros(shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch([(input1, [0, 0]), (input2, [2, 0])], golden_out)

        assemble_kernel_list_multi_shape_002(input1, input2, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)

    def test_assemble_list_multi_shape_003(self):
        device = "cpu"
        dtype = torch.float32
        shape_input1 = (300, 200)
        shape_input2 = (300, 200)
        shape_out = (500, 400)

        input1 = torch.rand(shape_input1, dtype=dtype, device=device)
        input2 = torch.rand(shape_input2, dtype=dtype, device=device)
        out_tensor = torch.zeros(shape_out, dtype=dtype, device=device)
        golden_out = out_tensor.clone()
        pypto_assemble_in_torch([(input1, [0, 0]), (input2, [2, 0])], golden_out)

        assemble_kernel_list_multi_shape_003(input1, input2, out_tensor)

        cos = compare_cos(out_tensor, golden_out)
        print(f"cosine similarity: {cos}")
        self.assertGreaterEqual(cos, 0.9999)


if __name__ == '__main__':
    unittest.main()
