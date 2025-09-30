#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import numpy as np
import torch
from bfloat16 import bfloat16


def parse_list_str(input_str: str):
    if input_str is None:
        raise ValueError("Can't convert None to list.")
    input_str = input_str.replace(" ", "")
    if input_str.startswith("[") and input_str.endswith("]"):
        input_str = input_str[1:-1]

    ret_list = []
    element_split_ident = " "
    if "{" in input_str:
        element_split_ident = "},{"
    if "[" in input_str:
        element_split_ident = "],["
    if element_split_ident in input_str:
        for sub_str in input_str.split(element_split_ident):
            ret_list.append(parse_list_str(sub_str))
    else:
        for sub_str in input_str.split(","):
            is_num = sub_str.isdecimal() or (
                (sub_str.startswith("-") or sub_str.startswith("+"))
                and sub_str[1:].isdecimal()
            )
            ret_list.append(int(sub_str) if is_num else sub_str)
    return ret_list


def get_dtype_by_name(name: str, is_torch: bool = False):
    str_to_dtype = {
        "int8": [np.int8, torch.int8],
        "int16": [np.int16, torch.int16],
        "int32": [np.int32, torch.int32],
        "int64": [np.int64, torch.int64],
        "fp16": [np.float16, torch.float16],
        "fp32": [np.float32, torch.float32],
        "fp64": [np.float64, torch.float64],
        "uint8": [np.uint8, torch.uint8],
        "uint16": [np.uint16, None],
        "uint32": [np.uint32, None],
        "uint64": [np.uint64, None],
        "bool": [np.bool_, torch.bool],
        "double": [np.float64, torch.double],
        "complex64": [np.complex64, torch.complex64],
        "complex128": [np.complex128, torch.complex64],
        "bf16": [bfloat16, torch.bfloat16],
    }
    return str_to_dtype.get(name, [np.float32, torch.float32])[is_torch]
