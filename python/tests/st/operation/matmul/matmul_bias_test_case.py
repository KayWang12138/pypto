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
pypto.matmul_bias ST测试用例配置
用于System Test自动化测试框架
"""
from dataclasses import dataclass

import pypto
import torch


@dataclass
class MatmulBiasConfig:
    shape: tuple[int, int, int]
    tile_shape: tuple[list, list, list]
    view_shape: tuple[int, int]
    out_dtype: pypto.DataType
    bias_dtype: pypto.DataType
    bias_shape: tuple[int, int]
    a_trans: bool = False
    b_trans: bool = False
    relu_type: int = 0

    DTYPE_CONFIG = {
        "DT_FP16": {"pto": pypto.DT_FP16, "torch": torch.float16, "atol": 1e-3, "rtol": 1e-3},
        "DT_FP32": {"pto": pypto.DT_FP32, "torch": torch.float32, "atol": 1e-3, "rtol": 1e-3},
        "DT_BF16": {"pto": pypto.DT_BF16, "torch": torch.bfloat16, "atol": 1e-2, "rtol": 1e-2},
        "DT_INT8": {"pto": pypto.DT_INT8, "torch": torch.int8, "atol": 0, "rtol": 0},
        "DT_INT32": {"pto": pypto.DT_INT32, "torch": torch.int32, "atol": 0, "rtol": 0},
    }

    @classmethod
    def from_test_case(cls, case: dict) -> "MatmulBiasConfig":
        bias_info = case.get("bias_info", {})
        relu_type = 1 if bias_info.get("relu_type") == "RELU" else 0
        
        return cls(
            shape=(case["m"], case["k"], case["n"]),
            tile_shape=tuple(case["tileshape"]),
            view_shape=tuple(case["viewshape"]),
            out_dtype=cls.DTYPE_CONFIG[case["c_dtype"]]["pto"],
            bias_dtype=cls.DTYPE_CONFIG[bias_info.get("dtype", case["c_dtype"])]["pto"],
            bias_shape=tuple(bias_info.get("shape", [1, case["n"]])),
            a_trans=case["a_trans"],
            b_trans=case["b_trans"],
            relu_type=relu_type,
        )

    @classmethod
    def get_torch_dtype(cls, dtype_str: str) -> torch.dtype:
        return cls.DTYPE_CONFIG[dtype_str]["torch"]

    @classmethod
    def get_tolerance(cls, dtype_str: str) -> tuple[float, float]:
        info = cls.DTYPE_CONFIG[dtype_str]
        return info["atol"], info["rtol"]


BIAS_TESTS = [
    {
        "id": "B01",
        "name": "fp16_bias_fp16",
        "desc": "FP16输入FP16 Bias",
        "m": 127, "k": 255, "n": 511,
        "a_dtype": "DT_FP16",
        "b_dtype": "DT_FP16",
        "c_dtype": "DT_FP16",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": False,
        "b_trans": False,
        "viewshape": [128, 256],
        "tileshape": [[64, 64], [64, 128], [128, 128]],
        "bias_info": {"dtype": "DT_FP16", "shape": [1, 511]},
        "extend_params": {},
        "products": ["950", "910"],
    },
    {
        "id": "B02",
        "name": "fp16_bias_fp32_trans_a",
        "desc": "FP16输入FP32 Bias+A转置",
        "m": 129, "k": 257, "n": 513,
        "a_dtype": "DT_FP16",
        "b_dtype": "DT_FP16",
        "c_dtype": "DT_FP32",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": True,
        "b_trans": False,
        "viewshape": [128, 256],
        "tileshape": [[128, 128], [128, 128], [256, 256]],
        "bias_info": {"dtype": "DT_FP32", "shape": [1, 513]},
        "extend_params": {},
        "products": ["950", "910"],
    },
    {
        "id": "B03",
        "name": "bf16_bias_fp32_trans_b",
        "desc": "BF16输入FP32 Bias+B转置",
        "m": 129, "k": 255, "n": 511,
        "a_dtype": "DT_BF16",
        "b_dtype": "DT_BF16",
        "c_dtype": "DT_FP32",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": False,
        "b_trans": True,
        "viewshape": [64, 256],
        "tileshape": [[64, 64], [64, 128], [128, 128]],
        "bias_info": {"dtype": "DT_FP32", "shape": [1, 511]},
        "extend_params": {},
        "products": ["950", "910"],
    },
    {
        "id": "B04",
        "name": "fp32_bias_fp32_trans_both",
        "desc": "FP32输入FP32 Bias+双转置",
        "m": 127, "k": 255, "n": 513,
        "a_dtype": "DT_FP32",
        "b_dtype": "DT_FP32",
        "c_dtype": "DT_FP32",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": True,
        "b_trans": True,
        "viewshape": [128, 256],
        "tileshape": [[128, 128], [64, 128], [256, 256]],
        "bias_info": {"dtype": "DT_FP32", "shape": [1, 513]},
        "extend_params": {},
        "products": ["950", "910"],
    },
    {
        "id": "B05",
        "name": "int8_bias_int32",
        "desc": "INT8输入INT32 Bias",
        "m": 129, "k": 257, "n": 511,
        "a_dtype": "DT_INT8",
        "b_dtype": "DT_INT8",
        "c_dtype": "DT_INT32",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": False,
        "b_trans": False,
        "viewshape": [128, 256],
        "tileshape": [[128, 128], [64, 128], [128, 128]],
        "bias_info": {"dtype": "DT_INT32", "shape": [1, 511]},
        "extend_params": {},
        "products": ["950", "910"],
    },
]

BIAS_RELU_TESTS = [
    {
        "id": "R01",
        "name": "fp16_bias_relu",
        "desc": "FP16 Bias+ReLU",
        "m": 127, "k": 257, "n": 511,
        "a_dtype": "DT_FP16",
        "b_dtype": "DT_FP16",
        "c_dtype": "DT_FP16",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": False,
        "b_trans": False,
        "viewshape": [64, 256],
        "tileshape": [[64, 64], [64, 128], [128, 128]],
        "bias_info": {"dtype": "DT_FP16", "shape": [1, 511], "relu_type": "RELU"},
        "extend_params": {},
        "products": ["950", "910"],
    },
    {
        "id": "R02",
        "name": "fp16_bias_fp32_relu_trans_a",
        "desc": "FP16 Bias FP32+ReLU+A转置",
        "m": 127, "k": 257, "n": 513,
        "a_dtype": "DT_FP16",
        "b_dtype": "DT_FP16",
        "c_dtype": "DT_FP32",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": True,
        "b_trans": False,
        "viewshape": [128, 256],
        "tileshape": [[128, 128], [128, 128], [256, 256]],
        "bias_info": {"dtype": "DT_FP32", "shape": [1, 513], "relu_type": "RELU"},
        "extend_params": {},
        "products": ["950", "910"],
    },
    {
        "id": "R03",
        "name": "int8_bias_int32_relu_trans_b",
        "desc": "INT8 Bias INT32+ReLU+B转置",
        "m": 129, "k": 255, "n": 513,
        "a_dtype": "DT_INT8",
        "b_dtype": "DT_INT8",
        "c_dtype": "DT_INT32",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": False,
        "b_trans": True,
        "viewshape": [64, 256],
        "tileshape": [[64, 64], [64, 128], [128, 128]],
        "bias_info": {"dtype": "DT_INT32", "shape": [1, 513], "relu_type": "RELU"},
        "extend_params": {},
        "products": ["950", "910"],
    },
]