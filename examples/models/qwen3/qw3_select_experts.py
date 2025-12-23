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
import os
import pypto
import pytest
import torch
import torch_npu
import numpy as np
from numpy.testing import assert_allclose


def main():
    test_select_experts()


# 1. 添加支持动态的config
@pypto.jit(
    host_options={"only_codegen": True},
)
def select_experts(logits_input, ids_k, weight_k, renormalize_flag):
    # 3. 得到动态tensor的shape
    bs = logits_input.shape[0]
    ne = logits_input.shape[1]
    idx_k_shape = ids_k.shape
    topk = idx_k_shape[1]
    view_shape = (1024, ne)
    bs_loop = (bs + view_shape[0] - 1) // view_shape[0]

    # 4. 实现kernel逻辑，循环展开BS动态轴
    for bs_idx in pypto.loop(bs_loop, name="LOOP_MOEGATE_L0", idx_name="bs_idx", unroll_List={1}):
        def bs_loop_func(bs_idx):
            # 5. 通过view得到tile_logits
            tile_logits = pypto.view(logits_input, view_shape,
                [bs_idx * view_shape[0], 0],
                valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]), ne])

            # 6. 按照计算图实现运算逻辑，设置set_vec_tile_shapes时应尽可能用满UB，但不要超过UB的大小。
            pypto.set_vec_tile_shapes(64, 128)

            # cast to fp32
            tile_logits_fp32 = pypto.cast(tile_logits, pypto.DT_FP32)
            # softmax
            softmax_out = pypto.softmax(tile_logits_fp32, -1)
            # topK
            topk_weight_tmp, topk_ids_tmp = pypto.topk(softmax_out, topk, -1, True)

            pypto.set_vec_tile_shapes(128, 8)
            if pypto.cond(pypto.symbolic_scalar(renormalize_flag)):
                # sum
                denominator = pypto.sum(topk_weight_tmp, -1, True)
                # div
                # for shape (b*s, k) (b*s, 1)
                topk_weight2 = pypto.div(topk_weight_tmp, denominator)
            else:
                denominator = topk_weight_tmp
                topk_weight2 = denominator
            # weight cast
            topk_weight2_f16 = pypto.cast(topk_weight2, weight_k.dtype)

            # 7. 将结果搬运到输出tensor上
            weight_k[bs_idx * pypto.symbolic_scalar(view_shape[0]):, pypto.symbolic_scalar(0):] = topk_weight2_f16
            ids_k[bs_idx * pypto.symbolic_scalar(view_shape[0]):, pypto.symbolic_scalar(0):] = topk_ids_tmp
        bs_loop_func(bs_idx)


def test_select_experts():
    # 1. 设置参数
    bs = 4959
    ne = 128
    top_k = 8
    renormalize = True
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    # 2. 构造多种shape，测试动态case
    for i in range(0, 2):
        if (i == 1):
            bs = 1

        # 3. 准备测试数据
        np.random.seed(0)
        router_logits = torch.rand((bs, ne), dtype=torch.float16, device=f'npu:{device_id}')
        topk_weights = torch.zeros((bs, top_k), dtype=torch.float16, device=f'npu:{device_id}')
        topk_ids = torch.zeros((bs, top_k), dtype=torch.int32, device=f'npu:{device_id}')

        # 4. 执行kernel并获取结果
        inputs = {
            router_logits: [0]
        }
        outputs = {
            topk_ids: [0],
            topk_weights: [0]
        }
        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
        pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
        select_experts(*pto_inputs, *pto_outputs, renormalize)
        pypto.runtime._device_synchronize()

        # 5. 与PyTorch参考实现对比
        result = torch.softmax(router_logits.to(torch.float32), dim=-1)
        topk_weight_tensor, topk_ids_tensor = torch.topk(result, top_k, dim=-1, largest=True, sorted=True)
        topk_ids_tensor_list = topk_ids_tensor.flatten().tolist()

        denominator_g = torch.sum(topk_weight_tensor, dim=-1, keepdim=True)
        topk_weight_2_tensor = torch.div(topk_weight_tensor, denominator_g).to(torch.float16)
        topk_weight_2_tensor_list = topk_weight_2_tensor.flatten().tolist()

        # idx result
        assert_allclose(np.array(topk_ids.cpu().flatten().tolist()),
                    np.array(topk_ids_tensor_list),
                    rtol=5e-3, atol=5e-3)

        # weight result
        assert_allclose(np.array(topk_weights.cpu().flatten().tolist()),
                    np.array(topk_weight_2_tensor_list),
                    rtol=5e-3, atol=5e-3)


if __name__ == "__main__":
    main()
