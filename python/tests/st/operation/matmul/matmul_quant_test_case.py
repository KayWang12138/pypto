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
pypto.matmul_quant ST测试用例配置
用于System Test自动化测试框架
"""
from dataclasses import dataclass

import pypto
import torch


@dataclass
class MatmulQuantConfig:
    shape: tuple[int, int, int]
    tile_shape: tuple[list, list, list]
    view_shape: tuple[int, int]
    out_dtype: pypto.DataType
    quant_type: int
    relu_type: int = 0
    a_trans: bool = False
    b_trans: bool = False

    DTYPE_CONFIG = {
        "DT_FP16": {"pto": pypto.DT_FP16, "torch": torch.float16, "atol": 1e-3, "rtol": 1e-3},
        "DT_FP32": {"pto": pypto.DT_FP32, "torch": torch.float32, "atol": 1e-3, "rtol": 1e-3},
        "DT_BF16": {"pto": pypto.DT_BF16, "torch": torch.bfloat16, "atol": 1e-2, "rtol": 1e-2},
        "DT_INT8": {"pto": pypto.DT_INT8, "torch": torch.int8, "atol": 0, "rtol": 0},
        "DT_INT32": {"pto": pypto.DT_INT32, "torch": torch.int32, "atol": 0, "rtol": 0},
    }

    @classmethod
    def from_test_case(cls, case: dict) -> "MatmulQuantConfig":
        fixpipe_info = case.get("fixpipe_info", {})
        relu_type = 1 if fixpipe_info.get("relu_type") == "RELU" else 0
        
        return cls(
            shape=(case["m"], case["k"], case["n"]),
            tile_shape=tuple(case["tileshape"]),
            view_shape=tuple(case["viewshape"]),
            out_dtype=cls.DTYPE_CONFIG[case["c_dtype"]]["pto"],
            quant_type=fixpipe_info.get("quant_type", 1),
            relu_type=relu_type,
            a_trans=case["a_trans"],
            b_trans=case["b_trans"],
        )

    @classmethod
    def get_torch_dtype(cls, dtype_str: str) -> torch.dtype:
        return cls.DTYPE_CONFIG[dtype_str]["torch"]

    @classmethod
    def get_tolerance(cls, dtype_str: str) -> tuple[float, float]:
        info = cls.DTYPE_CONFIG[dtype_str]
        return info["atol"], info["rtol"]


PERTENSOR_TESTS = [
    {
        "id": "Q01",
        "name": "int8_pertensor_fp16",
        "desc": "INT8 PerTensor量化输出FP16",
        "m": 129, "k": 255, "n": 511,
        "a_dtype": "DT_INT8",
        "b_dtype": "DT_INT8",
        "c_dtype": "DT_FP16",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": False,
        "b_trans": False,
        "viewshape": [64, 256],
        "tileshape": [[64, 64], [64, 128], [128, 128]],
        "fixpipe_info": {"quant_type": 1, "relu_type": "NO_RELU"},
        "extend_params": {},
        "products": ["950", "910"],
    },
    {
        "id": "Q02",
        "name": "int8_pertensor_fp16_trans_a",
        "desc": "INT8 PerTensor+A转置",
        "m": 127, "k": 257, "n": 513,
        "a_dtype": "DT_INT8",
        "b_dtype": "DT_INT8",
        "c_dtype": "DT_FP16",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": True,
        "b_trans": False,
        "viewshape": [128, 256],
        "tileshape": [[128, 128], [128, 128], [256, 256]],
        "fixpipe_info": {"quant_type": 1, "relu_type": "NO_RELU"},
        "extend_params": {},
        "products": ["950", "910"],
    },
    {
        "id": "Q03",
        "name": "int8_pertensor_fp32_trans_b",
        "desc": "INT8 PerTensor+B转置",
        "m": 129, "k": 255, "n": 513,
        "a_dtype": "DT_INT8",
        "b_dtype": "DT_INT8",
        "c_dtype": "DT_FP32",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": False,
        "b_trans": True,
        "viewshape": [64, 256],
        "tileshape": [[64, 64], [64, 128], [128, 128]],
        "fixpipe_info": {"quant_type": 1, "relu_type": "NO_RELU"},
        "extend_params": {},
        "products": ["950", "910"],
    },
]

PERCHANNEL_TESTS = [
    {
        "id": "Q04",
        "name": "int8_perchannel_fp16",
        "desc": "INT8 PerChannel量化输出FP16",
        "m": 129, "k": 257, "n": 511,
        "a_dtype": "DT_INT8",
        "b_dtype": "DT_INT8",
        "c_dtype": "DT_FP16",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": False,
        "b_trans": False,
        "viewshape": [128, 256],
        "tileshape": [[128, 128], [64, 128], [128, 128]],
        "fixpipe_info": {"quant_type": 2, "relu_type": "NO_RELU"},
        "extend_params": {},
        "products": ["950", "910"],
    },
    {
        "id": "Q05",
        "name": "int8_perchannel_fp16_trans_a",
        "desc": "INT8 PerChannel+A转置",
        "m": 127, "k": 255, "n": 513,
        "a_dtype": "DT_INT8",
        "b_dtype": "DT_INT8",
        "c_dtype": "DT_FP16",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": True,
        "b_trans": False,
        "viewshape": [128, 256],
        "tileshape": [[128, 128], [128, 128], [256, 256]],
        "fixpipe_info": {"quant_type": 2, "relu_type": "NO_RELU"},
        "extend_params": {},
        "products": ["950", "910"],
    },
    {
        "id": "Q06",
        "name": "int8_perchannel_fp32_trans_b",
        "desc": "INT8 PerChannel+B转置",
        "m": 129, "k": 255, "n": 513,
        "a_dtype": "DT_INT8",
        "b_dtype": "DT_INT8",
        "c_dtype": "DT_FP32",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": False,
        "b_trans": True,
        "viewshape": [64, 256],
        "tileshape": [[64, 64], [64, 128], [128, 128]],
        "fixpipe_info": {"quant_type": 2, "relu_type": "NO_RELU"},
        "extend_params": {},
        "products": ["950", "910"],
    },
]

QUANT_RELU_TESTS = [
    {
        "id": "R01",
        "name": "int8_pertensor_relu",
        "desc": "INT8 PerTensor+ReLU",
        "m": 127, "k": 257, "n": 511,
        "a_dtype": "DT_INT8",
        "b_dtype": "DT_INT8",
        "c_dtype": "DT_FP16",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": False,
        "b_trans": False,
        "viewshape": [64, 256],
        "tileshape": [[64, 64], [64, 128], [128, 128]],
        "fixpipe_info": {"quant_type": 1, "relu_type": "RELU"},
        "extend_params": {},
        "products": ["950", "910"],
    },
    {
        "id": "R02",
        "name": "int8_pertensor_relu_trans_a",
        "desc": "INT8 PerTensor+ReLU+A转置",
        "m": 129, "k": 255, "n": 513,
        "a_dtype": "DT_INT8",
        "b_dtype": "DT_INT8",
        "c_dtype": "DT_FP16",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": True,
        "b_trans": False,
        "viewshape": [128, 256],
        "tileshape": [[128, 128], [128, 128], [256, 256]],
        "fixpipe_info": {"quant_type": 1, "relu_type": "RELU"},
        "extend_params": {},
        "products": ["950", "910"],
    },
    {
        "id": "R03",
        "name": "int8_perchannel_relu_trans_b",
        "desc": "INT8 PerChannel+ReLU+B转置",
        "m": 129, "k": 255, "n": 511,
        "a_dtype": "DT_INT8",
        "b_dtype": "DT_INT8",
        "c_dtype": "DT_FP16",
        "a_format": "ND",
        "b_format": "ND",
        "a_trans": False,
        "b_trans": True,
        "viewshape": [64, 256],
        "tileshape": [[64, 64], [64, 128], [128, 128]],
        "fixpipe_info": {"quant_type": 2, "relu_type": "RELU"},
        "extend_params": {},
        "products": ["950", "910"],
    },
]