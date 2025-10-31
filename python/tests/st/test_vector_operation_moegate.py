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
import pto
import pytest
import torch
import numpy as np
from numpy.testing import assert_allclose

@pytest.mark.skip(reason="Dep operation interface")
def test_select_experts_vllm():
    renormalize_flag = True
    input_dtype = pto.DT_FP16
    idx_dtype = pto.DT_INT32
    calc_dtype = pto.DT_FP32
    bs = 15
    ne = 128
    k = 8
    shape = (bs, ne)
    logits_shape = (bs, ne)
    view_shape = (1, ne)
    pto.runtime._device_init()
    logits_input = pto.tensor(logits_shape, input_dtype, "MATMUL_TENSOR_a")
    weight_k = pto.tensor((bs, k), input_dtype, "MATMUL_TENSOR_weight_k")
    idx_k = pto.tensor((bs, k), idx_dtype, "MATMUL_TENSOR_idx_k")

    with pto.function("MATMUL", [logits_input], [idx_k, weight_k]):
        for bs_idx in pto.loop(pto.ceildiv(bs, view_shape[0])):
            def inner(batch_idx):
                tile_logits = pto.view(logits_input, view_shape,
                    [bs_idx * view_shape[0], 0],
                    valid_shape=[pto.min(pto.symbolic_scalar(bs) -
                    bs_idx * view_shape[0], pto.symbolic_scalar(bs)), ne])
                # cast to fp32
                pto.set_vec_tile_shapes(min(8, bs), min(128, ne))
                tile_logits_fp32 = pto.cast(tile_logits, calc_dtype)
                # softmax
                pto.set_vec_tile_shapes(min(8, bs), min(128, ne))
                softmax_out = pto.softmax(tile_logits_fp32)
                # topK
                pto.set_vec_tile_shapes(min(8, bs), min(128, ne))
                topk_weight_tmp, topk_idx_tmp = pto.topk(softmax_out, k, -1, True)
                if renormalize_flag:
                    # sum
                    pto.set_vec_tile_shapes(min(8, bs), min(8, k))
                    denominator = pto.sum(topk_weight_tmp)
                    # div
                    pto.set_vec_tile_shapes(min(8, bs), min(1, k))
                    # shape: (b*s, k) (b*s, 1)
                    topk_weight2 = pto.div(topk_weight_tmp, denominator)
                else:
                    denominator = topk_weight_tmp
                    topk_weight2 = denominator
                # weight cast
                pto.set_vec_tile_shapes(min(8, bs), min(1, k))
                topk_weight2_f16 = pto.cast(topk_weight2, input_dtype)
                pto.assemble(topk_weight2_f16, [batch_idx * view_shape[0], 0], weight_k)
                pto.assemble(topk_idx_tmp, [batch_idx * view_shape[0], 0], idx_k)
            inner(bs_idx)

    np.random.seed(0)
    logits_tensor = np.random.randn(bs, ne).astype(np.float32) * 20.0 - 10.0
    logits_data = logits_tensor.flatten().tolist()
    weight_data = list([0] * bs * k)
    idx_data = list([0] * bs * k)

    pto.runtime._device_run_once_data_from_host([logits_data], [idx_data, weight_data])
    result = torch.softmax(torch.tensor(logits_tensor).to(torch.float32), dim=-1)
    topk_weight_tensor, topk_idx_tensor = torch.topk(result, k, dim=-1, largest=True, sorted=True)
    topk_weight_tensor_list = topk_weight_tensor.flatten().tolist()
    topk_idx_tensor_list = topk_idx_tensor.flatten().tolist()

    denominator_g = torch.sum(topk_weight_tensor, dim=-1, keepdim=True)
    topk_weight_2_tensor = torch.div(topk_weight_tensor, denominator_g).to(torch.float16)
    topk_weight_2_tensor_list = topk_weight_2_tensor.flatten().tolist()

    # idx result
    assert_allclose(np.array(idx_data),
                np.array(topk_idx_tensor_list),
                rtol=5e-3, atol=5e-3)

    # final weight result
    assert_allclose(np.array(weight_data),
                np.array(topk_weight_2_tensor_list),
                rtol=5e-3, atol=5e-3)
    pto.runtime._device_fini()
