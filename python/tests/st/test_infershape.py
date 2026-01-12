#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
import sys
import pypto
import pytest
import torch
import torch_npu


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../../examples/models/deepseek_v32_exp/utils/c'))
from compare import compare


num, d, eps = 4, 512, 1e-6
num2 = (2 + num) * num

def gen_data(t = 16):
    print("t is ", t)
    x_ori = torch.empty((t, num2), dtype = torch.bfloat16).uniform_(-1, 1)
    scale = torch.empty((3, ), dtype = torch.float32).uniform_(-1, 1)
    hc_base_ori = torch.empty((num2, ), dtype = torch.float32).uniform_(-1, 1)

    base = hc_base_ori.reshape(1, num2)
    x = x_ori.to(torch.float32) - 0
    pre = x[:, :num] * scale[0] + base[:, :num]  # (t, 4)