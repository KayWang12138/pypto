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
from glm_ffn_router_expert_quant import glm_router_expert_quant
from glm_ffn_share_expert_quant import glm_share_expert_quant
from glm_ffn_dense_quant import glm_dense_quant


# ffn_router_expert
def ffn_router_expert_quant(hidden_states: torch.Tensor,
                            pertoken_scale: torch.Tensor,
                            group_list: torch.Tensor,
                            w13: torch.Tensor,
                            w13_scale: torch.Tensor,
                            w2: torch.Tensor,
                            w2_scale: torch.Tensor
)-> torch.Tensor:
    return glm_router_expert_quant(hidden_states, pertoken_scale, group_list, w13, w13_scale, w2, w2_scale)


# ffn_share_expert
def ffn_share_expert_quant(hidden_states: torch.Tensor,
                           w13: torch.Tensor,
                           w13_scale: torch.Tensor,
                           w2: torch.Tensor,
                           w2_scale: torch.Tensor
)-> torch.Tensor:
    return glm_share_expert_quant(hidden_states, w13, w13_scale, w2, w2_scale)


# ffn_dense
def ffn_dense_quant(hidden_states: torch.Tensor,
                    w13: torch.Tensor,
                    w13_scale: torch.Tensor,
                    w2: torch.Tensor
)-> torch.Tensor:
    return glm_dense_quant(hidden_states, w13, w13_scale, w2)