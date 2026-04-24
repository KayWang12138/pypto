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
Test view codegen
"""

import pypto
import torch
import torch.nn.functional as F
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
def pypto_view_in_torch(x, shape=None, offsets=None, valid_shape=None, dtype=None):
    # 1. 处理 dtype 转换 (位重解释/Bitcast)
    if dtype is not None:
        x = x.view(dtype)
    
    # 如果仅转换 dtype，没有 shape 要求，直接返回
    if shape is None:
        return x

    # 2. 处理 offsets 缺省值
    if offsets is None:
        offsets = [0] * len(shape)
    
    # 3. 确定有效数据的截取范围 (取 valid_shape 或 shape)
    take_shape = valid_shape if valid_shape is not None else shape
    
    # 4. 执行多维切片 (任意维度支持)
    # 构造 slice 对象列表，例如: x[off0:off0+take0, off1:off1+take1, ...]
    indices = []
    for i in range(len(take_shape)):
        indices.append(slice(offsets[i], offsets[i] + take_shape[i]))
    
    y = x[tuple(indices)]
    
    # 5. 补零填充 (如果 shape > valid_shape)
    # F.pad 的 padding 列表顺序是从最后一个维度往前数，且每个维度有 (left, right)
    if list(y.shape) != list(shape):
        padding = []
        # 从最后一个维度向第一个维度遍历
        for i in range(len(shape) - 1, -1, -1):
            pad_after = shape[i] - y.shape[i]
            padding.extend([0, pad_after]) # [前填充, 后填充]
        
        y = F.pad(y, tuple(padding), mode='constant', value=0)
        
    return y

@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_fp16(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(2, 16)
    out_tensor[:] = input_tensor.view([4, 4], [0, 4])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_fp16_002(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(2, 16)
    out_tensor[:] = input_tensor.view([2, 4], [0, 0])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_fp16_003(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 2, 16)
    out_tensor[:] = input_tensor.view([2, 4, 4], [0, 0, 4])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_fp16_004(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 2, 16)
    out_tensor[:] = input_tensor.view([2, 2, 4], [0, 0, 0])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_fp16_005(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(16)
    out_tensor[:] = input_tensor.view([4], [4])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_fp16_006(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 16)
    out_tensor[:] = input_tensor.view([3, 3], [0, 3])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_fp16_007(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 1, 16)
    out_tensor[:] = input_tensor.view([2, 3, 3], [0, 0, 3])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_fp16_008(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 1, 1, 16)
    out_tensor[:] = input_tensor.view([2, 2, 2, 2], [0, 0, 0, 2])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_fp16_009(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(16)
    out_tensor[:] = input_tensor.view([4], [2])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_fp16_010(
    input_tensor: pypto.Tensor([...], pypto.DT_FP16),
    out_tensor: pypto.Tensor([...], pypto.DT_FP16),
):
    pypto.set_vec_tile_shapes(1, 16)
    out_tensor[:] = input_tensor.view([2, 4], [0, 4])


class TestLiteNPUViewFP16(unittest.TestCase):
    def test_view_fp16_001(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (4, 8)
        shape_out = (4, 4)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        view_kernel_fp16(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[4, 4], offsets=[0, 4])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_view_fp16_002(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (4, 8)
        shape_out = (2, 4)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        view_kernel_fp16_002(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[2, 4], offsets=[0, 0])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_view_fp16_003(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (2, 4, 8)
        shape_out = (2, 4, 4)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        view_kernel_fp16_003(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[2, 4, 4], offsets=[0, 0, 4])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_view_fp16_004(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (2, 4, 8)
        shape_out = (2, 2, 4)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        view_kernel_fp16_004(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[2, 2, 4], offsets=[0, 0, 0])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_view_fp16_005(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (8,)
        shape_out = (4,)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        view_kernel_fp16_005(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[4], offsets=[4])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_view_fp16_006(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (3, 6)
        shape_out = (3, 3)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        view_kernel_fp16_006(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[3, 3], offsets=[0, 3])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_view_fp16_007(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (2, 3, 6)
        shape_out = (2, 3, 3)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        view_kernel_fp16_007(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[2, 3, 3], offsets=[0, 0, 3])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_view_fp16_008(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (2, 2, 2, 4)
        shape_out = (2, 2, 2, 2)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        view_kernel_fp16_008(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[2, 2, 2, 2], offsets=[0, 0, 0, 2])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_view_fp16_009(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (6,)
        shape_out = (4,)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        view_kernel_fp16_009(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[4], offsets=[2])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_view_fp16_010(self):
        device = "cpu"
        dtype = torch.float16
        shape_input = (2, 8)
        shape_out = (2, 4)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        view_kernel_fp16_010(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[2, 4], offsets=[0, 4])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_fp32(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(2, 8)
    out_tensor[:] = input_tensor.view([4, 4], [0, 4])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_fp32_002(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(2, 8)
    out_tensor[:] = input_tensor.view([2, 4], [0, 0])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_fp32_003(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 2, 8)
    out_tensor[:] = input_tensor.view([2, 4, 4], [0, 0, 4])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_fp32_004(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 2, 8)
    out_tensor[:] = input_tensor.view([2, 2, 4], [0, 0, 0])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_fp32_005(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(8)
    out_tensor[:] = input_tensor.view([4], [4])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_fp32_006(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 8)
    out_tensor[:] = input_tensor.view([3, 3], [0, 3])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_fp32_007(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 1, 8)
    out_tensor[:] = input_tensor.view([2, 3, 3], [0, 0, 3])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_fp32_008(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 1, 1, 8)
    out_tensor[:] = input_tensor.view([2, 2, 2, 2], [0, 0, 0, 2])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_fp32_009(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(8)
    out_tensor[:] = input_tensor.view([4], [2])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_fp32_010(
    input_tensor: pypto.Tensor([...], pypto.DT_FP32),
    out_tensor: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 8)
    out_tensor[:] = input_tensor.view([2, 4], [0, 4])


class TestLiteNPUViewFP32(unittest.TestCase):
    def test_view_fp32_001(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (4, 8)
        shape_out = (4, 4)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        view_kernel_fp32(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[4, 4], offsets=[0, 4])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_view_fp32_002(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (4, 8)
        shape_out = (2, 4)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        view_kernel_fp32_002(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[2, 4], offsets=[0, 0])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_view_fp32_003(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 4, 8)
        shape_out = (2, 4, 4)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        view_kernel_fp32_003(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[2, 4, 4], offsets=[0, 0, 4])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_view_fp32_004(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 4, 8)
        shape_out = (2, 2, 4)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        view_kernel_fp32_004(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[2, 2, 4], offsets=[0, 0, 0])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_view_fp32_005(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (8,)
        shape_out = (4,)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        view_kernel_fp32_005(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[4], offsets=[4])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_view_fp32_006(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (3, 6)
        shape_out = (3, 3)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        view_kernel_fp32_006(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[3, 3], offsets=[0, 3])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_view_fp32_007(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 3, 6)
        shape_out = (2, 3, 3)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        view_kernel_fp32_007(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[2, 3, 3], offsets=[0, 0, 3])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_view_fp32_008(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 2, 2, 4)
        shape_out = (2, 2, 2, 2)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        view_kernel_fp32_008(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[2, 2, 2, 2], offsets=[0, 0, 0, 2])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_view_fp32_009(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (6,)
        shape_out = (4,)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        view_kernel_fp32_009(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[4], offsets=[2])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_view_fp32_010(self):
        device = "cpu"
        dtype = torch.float32
        shape_input = (2, 8)
        shape_out = (2, 4)

        input_tensor = torch.rand(shape_input, dtype=dtype, device=device)
        out_tensor = torch.rand(shape_out, dtype=dtype, device=device)

        view_kernel_fp32_010(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[2, 4], offsets=[0, 4])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_int32(
    input_tensor: pypto.Tensor([...], pypto.DT_INT32),
    out_tensor: pypto.Tensor([...], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(2, 8)
    out_tensor[:] = input_tensor.view([4, 4], [0, 4])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_int32_002(
    input_tensor: pypto.Tensor([...], pypto.DT_INT32),
    out_tensor: pypto.Tensor([...], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(8)
    out_tensor[:] = input_tensor.view([4], [4])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_int32_003(
    input_tensor: pypto.Tensor([...], pypto.DT_INT32),
    out_tensor: pypto.Tensor([...], pypto.DT_INT32),
):
    pypto.set_vec_tile_shapes(1, 2, 8)
    out_tensor[:] = input_tensor.view([2, 4, 4], [0, 0, 4])


class TestLiteNPUViewInt32(unittest.TestCase):
    def test_view_int32_001(self):
        device = "cpu"
        dtype = torch.int32
        shape_input = (4, 8)
        shape_out = (4, 4)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        view_kernel_int32(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[4, 4], offsets=[0, 4])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_view_int32_002(self):
        device = "cpu"
        dtype = torch.int32
        shape_input = (8,)
        shape_out = (4,)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        view_kernel_int32_002(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[4], offsets=[4])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_view_int32_003(self):
        device = "cpu"
        dtype = torch.int32
        shape_input = (2, 4, 8)
        shape_out = (2, 4, 4)

        input_tensor = torch.randint(0, 100, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(0, 100, shape_out, dtype=dtype, device=device)

        view_kernel_int32_003(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[2, 4, 4], offsets=[0, 0, 4])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_int8(
    input_tensor: pypto.Tensor([...], pypto.DT_INT8),
    out_tensor: pypto.Tensor([...], pypto.DT_INT8),
):
    pypto.set_vec_tile_shapes(2, 32)
    out_tensor[:] = input_tensor.view([4, 4], [0, 4])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_int8_002(
    input_tensor: pypto.Tensor([...], pypto.DT_INT8),
    out_tensor: pypto.Tensor([...], pypto.DT_INT8),
):
    pypto.set_vec_tile_shapes(32)
    out_tensor[:] = input_tensor.view([4], [4])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_int8_003(
    input_tensor: pypto.Tensor([...], pypto.DT_INT8),
    out_tensor: pypto.Tensor([...], pypto.DT_INT8),
):
    pypto.set_vec_tile_shapes(1, 2, 32)
    out_tensor[:] = input_tensor.view([2, 4, 4], [0, 0, 4])


class TestLiteNPUViewInt8(unittest.TestCase):
    def test_view_int8_001(self):
        device = "cpu"
        dtype = torch.int8
        shape_input = (4, 8)
        shape_out = (4, 4)

        input_tensor = torch.randint(-128, 127, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(-128, 127, shape_out, dtype=dtype, device=device)

        view_kernel_int8(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[4, 4], offsets=[0, 4])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_view_int8_002(self):
        device = "cpu"
        dtype = torch.int8
        shape_input = (8,)
        shape_out = (4,)

        input_tensor = torch.randint(-128, 127, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(-128, 127, shape_out, dtype=dtype, device=device)

        view_kernel_int8_002(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[4], offsets=[4])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_view_int8_003(self):
        device = "cpu"
        dtype = torch.int8
        shape_input = (2, 4, 8)
        shape_out = (2, 4, 4)

        input_tensor = torch.randint(-128, 127, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(-128, 127, shape_out, dtype=dtype, device=device)

        view_kernel_int8_003(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[2, 4, 4], offsets=[0, 0, 4])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_int16(
    input_tensor: pypto.Tensor([...], pypto.DT_INT16),
    out_tensor: pypto.Tensor([...], pypto.DT_INT16),
):
    pypto.set_vec_tile_shapes(2, 16)
    out_tensor[:] = input_tensor.view([4, 4], [0, 4])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_int16_002(
    input_tensor: pypto.Tensor([...], pypto.DT_INT16),
    out_tensor: pypto.Tensor([...], pypto.DT_INT16),
):
    pypto.set_vec_tile_shapes(16)
    out_tensor[:] = input_tensor.view([4], [4])


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"}, runtime_options={"run_mode": pypto.RunMode.NPU})
def view_kernel_int16_003(
    input_tensor: pypto.Tensor([...], pypto.DT_INT16),
    out_tensor: pypto.Tensor([...], pypto.DT_INT16),
):
    pypto.set_vec_tile_shapes(1, 2, 16)
    out_tensor[:] = input_tensor.view([2, 4, 4], [0, 0, 4])


class TestLiteNPUViewInt16(unittest.TestCase):
    def test_view_int16_001(self):
        device = "cpu"
        dtype = torch.int16
        shape_input = (4, 8)
        shape_out = (4, 4)

        input_tensor = torch.randint(-32768, 32767, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(-32768, 32767, shape_out, dtype=dtype, device=device)

        view_kernel_int16(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[4, 4], offsets=[0, 4])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_view_int16_002(self):
        device = "cpu"
        dtype = torch.int16
        shape_input = (8,)
        shape_out = (4,)

        input_tensor = torch.randint(-32768, 32767, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(-32768, 32767, shape_out, dtype=dtype, device=device)

        view_kernel_int16_002(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[4], offsets=[4])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

    def test_view_int16_003(self):
        device = "cpu"
        dtype = torch.int16
        shape_input = (2, 4, 8)
        shape_out = (2, 4, 4)

        input_tensor = torch.randint(-32768, 32767, shape_input, dtype=dtype, device=device)
        out_tensor = torch.randint(-32768, 32767, shape_out, dtype=dtype, device=device)

        view_kernel_int16_003(input_tensor, out_tensor)

        golden_out = pypto_view_in_torch(input_tensor, shape=[2, 4, 4], offsets=[0, 0, 4])
        cos_value = compare_cos(np.array(out_tensor.cpu()), np.array(golden_out.cpu()))
        self.assertGreaterEqual(cos_value, 0.9999)

if __name__ == '__main__':
    unittest.main()
