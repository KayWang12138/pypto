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
"""
"""
import os
import numpy as np
import torch
import pypto
from st.pypto_test import TestBuilder


@pypto.jit(
    host_options={"only_codegen": True},
    codegen_options={"support_dynamic_unaligned": True}
)
def op_select_experts(input_tensors, output_tensors, params):

    bs, ne, topk, renormalize_flag = params
    router_logits = input_tensors[0]
    ids_k = output_tensors[0]
    weight_k = output_tensors[1]

    pypto.mark_dynamic(router_logits, 0)
    pypto.mark_dynamic(ids_k, 0)
    pypto.mark_dynamic(weight_k, 0)

    view_shape = (1024, ne)
    bs_loop = (bs + view_shape[0] - 1) // view_shape[0]

    with pypto.function("MOEGATE", [router_logits], [ids_k, weight_k]):
        def inside_select_experts():
            for bs_idx in pypto.loop(bs_loop, name="LOOP_MOEGATE_L0", idx_name="bs_idx", unroll_List={1}):
                def bs_loop_func(bs_idx):
                    tile_logits = pypto.view(router_logits, view_shape,
                        [bs_idx * view_shape[0], 0],
                        valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]), ne])
                    pypto.set_vec_tile_shapes(64, 128)

                    tile_logits_fp32 = pypto.cast(tile_logits, pypto.DT_FP32)
                    softmax_out = pypto.softmax(tile_logits_fp32, -1)
                    topk_weight_tmp, topk_ids_tmp = pypto.topk(softmax_out, topk, -1, True)

                    pypto.set_vec_tile_shapes(128, 8)
                    if pypto.cond(pypto.symbolic_scalar(renormalize_flag)):
                        denominator = pypto.sum(topk_weight_tmp, -1, True)
                        topk_weight2 = pypto.div(topk_weight_tmp, denominator)
                    else:
                        denominator = topk_weight_tmp
                        topk_weight2 = denominator
                    topk_weight2_f16 = pypto.cast(topk_weight2, weight_k.dtype)

                    ids_k[bs_idx * pypto.symbolic_scalar(view_shape[0]):, pypto.symbolic_scalar(0):] = topk_ids_tmp
                    weight_k[bs_idx * pypto.symbolic_scalar(view_shape[0]):, pypto.symbolic_scalar(0):] = topk_weight2_f16
                bs_loop_func(bs_idx)
        inside_select_experts()


def golden_select_experts(params, router_logits, topk_ids_tensor_list, topk_weight_2_tensor_list):
    _, _, topk, _ = params
    result = torch.softmax(router_logits.to(torch.float32), dim=-1)
    topk_weight_tensor, topk_ids_tensor = torch.topk(result, topk, dim=-1, largest=True, sorted=True)
    topk_ids_tensor_list = topk_ids_tensor.to(torch.int32)


def golden_select_experts(params, router_logits, topk_ids_tensor_list, topk_weight_2_tensor_list):
    _, _, topk, _ = params
    result = torch.softmax(router_logits.to(torch.float32), dim=-1)
    topk_weight_tensor, topk_ids_tensor = torch.topk(result, topk, dim=-1, largest=True, sorted=True)
    topk_ids_tensor_list = topk_ids_tensor.to(torch.int32)

    denominator_g = torch.sum(topk_weight_tensor, dim=-1, keepdim=True)
    topk_weight_2_tensor = torch.div(topk_weight_tensor, denominator_g).to(torch.float16)
    topk_weight_2_tensor_list = topk_weight_2_tensor
    return topk_ids_tensor_list, topk_weight_2_tensor_list


class SelectExpertsTest(TestBuilder):
    def __init__(self, params: tuple, kernel, kernel_golden, tiling: int):
        super().__init__(params, kernel, kernel_golden, tiling)
        self.set_tol(rtol=5e-3, atol=5e-3)
        self.device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
        
    def get_input_from_param(self):
        bs, ne, top_k, renormalize = self.params
        np.random.seed(0)
        router_logits = torch.rand((bs, ne), dtype=torch.float16, device=f'npu:{self.device_id}')
        self.setup_inputs_jit(router_logits)
        return (router_logits, )


def test():
    bs = 4959
    ne = 128
    top_k = 8
    renormalize = True

    for i in range(0, 2):
        if (i == 1):
            bs = 1
        params = (bs, ne, top_k, renormalize)
        st = SelectExpertsTest(params, op_select_experts, golden_select_experts, tiling=32)
        st(jit=True)
