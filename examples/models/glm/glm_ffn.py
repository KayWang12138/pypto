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
import torch
import torch_npu
import pypto
from glm_ffn_group_list_cumsum import group_list_cumsum as group_list_cumsum_kernel
from glm_ffn_router_expert_quant import moe_router_expert_main as moe_router_expert_kernel
from glm_ffn_share_expert_quant import moe_main as moe_share_expert_kernel
from glm_ffn_dense_quant import moe_main as dense_quant_kernel


def ffn_router_expert_quant(hidden_states: torch.Tensor,
                            pertoken_scale: torch.Tensor,
                            group_list: torch.Tensor,
                            w1: torch.Tensor,
                            w1_scale: torch.Tensor,
                            w2: torch.Tensor,
                            w2_scale: torch.Tensor
)-> torch.Tensor:
    x_dtype = w2_scale.dtype
    group_list_int32 = group_list.to(torch.int32)
    # group_list_cumsum = (torch.cumsum(group_list_int32, dim=0) - group_list_int32).to(torch.int32)
    group_list_cumsum = torch.zeros_like(group_list_int32, device=f'{group_list_int32.device}')
    inputs = [group_list_int32]
    outputs = [group_list_cumsum]
    group_list_cumsum_kernel(inputs, outputs)
    pypto.runtime._device_synchronize()

    b_s_topk, hidden_size = hidden_states.shape[0:2]
    out_tensor = torch.zeros((b_s_topk, hidden_size), dtype=x_dtype, device=f'{hidden_states.device}')
    inputs = [hidden_states, pertoken_scale, group_list_int32, group_list_cumsum, w1, w1_scale, w2, w2_scale]
    outputs = [out_tensor]
    moe_router_expert_kernel(inputs, outputs)
    pypto.runtime._device_synchronize()
    return out_tensor


def ffn_share_expert_quant(hidden_states: torch.Tensor,
                           w1: torch.Tensor,
                           w1_scale: torch.Tensor,
                           w2: torch.Tensor,
                           w2_scale: torch.Tensor
)-> torch.Tensor:
    x_dtype = hidden_states.dtype
    b_s_topk, hidden_size = hidden_states.shape[0:2]
    out_tensor = torch.zeros((b_s_topk, hidden_size), dtype=x_dtype, device=f'{hidden_states.device}')
    inputs = [hidden_states, w1, w1_scale, w2, w2_scale]
    outputs = [out_tensor]
    moe_share_expert_kernel(inputs, outputs)
    pypto.runtime._device_synchronize()
    return out_tensor


def ffn_dense_quant(hidden_states: torch.Tensor,
                    w1: torch.Tensor,
                    w1_scale: torch.Tensor,
                    w2: torch.Tensor
)-> torch.Tensor:
    x_dtype = hidden_states.dtype
    b_s_topk, hidden_size = hidden_states.shape[0:2]
    out_tensor = torch.zeros((b_s_topk, hidden_size), dtype=x_dtype, device=f'{hidden_states.device}')
    inputs = [hidden_states, w1, w1_scale, w2]
    outputs = [out_tensor]
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
    dense_quant_kernel(pto_inputs, pto_outputs)
    pypto.runtime._device_synchronize()
    return out_tensor