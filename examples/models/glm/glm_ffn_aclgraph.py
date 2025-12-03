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
"""
"""
from torch._subclasses.fake_tensor import FakeTensor
import torch
from torch._dynamo import allow_in_graph
import pypto
from glm_ffn_router_expert_quant import moe_router_expert_main as moe_router_expert_kernel
from glm_ffn_share_expert_quant import moe_main as moe_share_expert_kernel
from glm_ffn_dense_quant import moe_main as dense_quant_kernel


@allow_in_graph
def graph_ffn_router_expert_quant(inputs, outputs):
    if isinstance(inputs[0], FakeTensor):
        return
    moe_router_expert_kernel(inputs, outputs)
    pypto.runtime._device_synchronize()


@allow_in_graph
def graph_ffn_share_expert_quant(inputs, outputs):
    if isinstance(inputs[0], FakeTensor):
        return
    moe_share_expert_kernel(inputs, outputs)
    pypto.runtime._device_synchronize()


@allow_in_graph
def graph_ffn_dense_quant(inputs, outputs):
    if isinstance(inputs[0], FakeTensor):
        return
    dense_quant_kernel(inputs, outputs)
    pypto.runtime._device_synchronize()