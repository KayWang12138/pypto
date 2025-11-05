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
import torch_npu
import numpy as np
from numpy.testing import assert_allclose


def main():
    test_select_experts()


@pto.jit
def select_experts(in_tensors, out_tensors, renormalize_flag):
    # 1. 添加支持动态的config
    pto.set_codegen_option("support_dynamic_unaligned", True)
    pto.set_host_option("only_codegen", True)

    # 2. 从入参拿到输入和输出tensor
    logits_input = in_tensors[0]
    ids_k = out_tensors[0]
    weight_k = out_tensors[1]

    # 3. 设置axis=0为动态shape
    pto.mark_dynamic(logits_input, 0)
    pto.mark_dynamic(ids_k, 0)
    pto.mark_dynamic(weight_k, 0)

    # 4. 得到动态tensor的shape
    bs = logits_input.shape[0]
    ne = logits_input.shape[1]
    idx_k_shape = ids_k.shape
    topk = idx_k_shape[1]
    view_shape = (1, ne)
    bs_loop = (bs + view_shape[0] - 1) // view_shape[0]

    # 5. 定义动态函数
    with pto.function("MOEGATE", [logits_input], [ids_k, weight_k]):
        def inside_select_experts():
            # 6. 实现kernel逻辑，循环展开BS动态轴
            for bs_idx in pto.loop(bs_loop, name="LOOP_MOEGATE_L0", idx_name="bs_idx"):
                def bs_loop_func(bs_idx):
                    # 7. 通过view得到tile_logits
                    tile_logits = pto.view(logits_input, view_shape,
                        [bs_idx * view_shape[0], 0],
                        valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]), ne])

                    # 8. 按照计算图实现运算逻辑，设置set_vec_tile_shapes时应尽可能用满UB，但不要超过UB的大小。
                    pto.set_vec_tile_shapes(1, 128)
                    # cast to fp32
                    tile_logits_fp32 = pto.cast(tile_logits, pto.DT_FP32)
                    # softmax
                    softmax_out = pto.softmax(tile_logits_fp32, -1)
                    # topK
                    topk_weight_tmp, topk_ids_tmp = pto.topk(softmax_out, topk, -1, True)

                    if pto.cond(pto.symbolic_scalar(renormalize_flag)):
                        # sum
                        pto.set_vec_tile_shapes(1, 8)
                        denominator = pto.sum(topk_weight_tmp)
                        # div
                        pto.set_vec_tile_shapes(1, 1)
                        # for shape (b*s, k) (b*s, 1)
                        topk_weight2 = pto.div(topk_weight_tmp, denominator)
                    else:
                        denominator = topk_weight_tmp
                        topk_weight2 = denominator
                    # weight cast
                    pto.set_vec_tile_shapes(1, 1)
                    topk_weight2_f16 = pto.cast(topk_weight2, weight_k.dtype)

                    # 9. 将结果搬运到输出tensor上
                    weight_k[bs_idx * pto.symbolic_scalar(view_shape[0]):, pto.symbolic_scalar(0):] = topk_weight2_f16
                    ids_k[bs_idx * pto.symbolic_scalar(view_shape[0]):, pto.symbolic_scalar(0):] = topk_ids_tmp
                bs_loop_func(bs_idx)
        inside_select_experts()
    assert isinstance(ids_k, pto.tensor)
    assert isinstance(weight_k, pto.tensor)


def test_select_experts():
    # 1. 设置参数
    bs = 32
    ne = 128
    top_k = 8
    renormalize = True
    device_id = int(os.environ.get('TILE_FWK_STEST_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    # 2. 构造多种shape，测试动态case
    for i in range(0, 4):
        if (i == 2):
            bs = 3

        # 3. 准备测试数据
        np.random.seed(0)
        router_logits = torch.rand((bs, ne), dtype=torch.float16, device=f'npu:{device_id}')
        topk_weights = torch.zeros((bs, top_k), dtype=torch.float16, device=f'npu:{device_id}')
        topk_ids = torch.zeros((bs, top_k), dtype=torch.int32, device=f'npu:{device_id}')

        # 4. 执行kernel并获取结果
        inputs = [router_logits]
        outputs = [topk_ids, topk_weights]
        select_experts(inputs, outputs, renormalize)
        pto.runtime._device_synchronize()

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
