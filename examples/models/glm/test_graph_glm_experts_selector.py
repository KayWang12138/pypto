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
import torch
import numpy as np
from numpy.testing import assert_allclose
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph


def powers_of_2(n: int) -> set[int]:
    assert n > 0, "n must be positive"
    result = set()
    power = 0
    while True:
        current = 1 << power  # 计算2的power次方
        if current > n:
            break
        result.add(current)
        power += 1
    return result


def main():
    test_select_experts()


@allow_in_graph
def graph_select_experts_glm(inputs, outputs, renormalize_flag, topk_group, num_expert_group, row_ids_flag):
    if isinstance(inputs[0], FakeTensor):
        return
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
    select_experts_glm(pto_inputs, pto_outputs, renormalize_flag, topk_group, num_expert_group, row_ids_flag)
    pypto.runtime._device_synchronize()


def select_experts_pto(router_logits: torch.Tensor,
                       top_k: int,  # number of top k experts.
                       # Whether to renormalize the routing weights.
                       renormalize: bool,
                       # Number of expert groups to select from.
                       topk_group: int,
                       # Number of experts in each group.
                       num_expert_group: int,
                       # Correction bias to apply to expert scores.
                       e_score_correction_bias: torch.Tensor,
                       ) -> tuple[torch.Tensor, torch.Tensor]:  # topk_weights, topk_ids, row_idx
    row_ids_flag = False
    bs = router_logits.shape[0]
    device_info = router_logits.device
    topk_weights = torch.zeros(
        (bs, top_k), dtype=router_logits.dtype, device=device_info)
    topk_ids = torch.zeros((bs, top_k), dtype=torch.int32, device=device_info)
    row_idx = torch.zeros((bs, top_k), dtype=torch.int32, device=device_info)

    # 4. 执行kernel并获取结果
    inputs = [router_logits, e_score_correction_bias]
    outputs = [topk_weights, topk_ids, row_idx]
    graph_select_experts_glm(inputs, outputs, renormalize,
                             topk_group, num_expert_group, row_ids_flag)
    return topk_weights, topk_ids


@pypto.jit
def select_experts_glm(in_tensors, out_tensors, renormalize_flag, topk_group, num_expert_group, row_ids_flag):
    # 1. 添加支持动态的config
    pypto.set_codegen_options(support_dynamic_unaligned=True)
    pypto.set_host_options(only_codegen=True)
    pypto.set_runtime_options(cfgcache_device_task_num=100)
    pypto.set_runtime_options(cfgcache_root_task_num=100)
    pypto.set_runtime_options(cfgcache_leaf_task_num=10000)
    # 泳道图使能  pypto.set_option('profile_enable', True)

    # 2. 从入参拿到输入和输出tensor
    logits_input = in_tensors[0]
    e_score_bias_input = in_tensors[1]
    weight_k = out_tensors[0]
    ids_k = out_tensors[1]
    row_idx = out_tensors[2]

    # 3. 设置axis = 0为动态shape
    pypto.mark_dynamic(logits_input, 0)
    pypto.mark_dynamic(weight_k, 0)
    pypto.mark_dynamic(ids_k, 0)

    # 4. 得到动态tensor的shape
    bs = logits_input.shape[0]
    ne = logits_input.shape[1]
    idx_k_shape = ids_k.shape
    topk = idx_k_shape[1]
    view_shape = (1, ne)
    view_first = 1
    bs_loop = (bs + view_shape[0] - 1) // view_shape[0]

    pypto.set_runtime_options(estimated_stitch_task_max_loop_num=32)
    pypto.set_runtime_options(workspace_recycle_period=32)
    # 5. 定义动态函数

    for _ in pypto.loop(1, name="LOOP_RESHAPE_INPLACE", idx_name="_"):
        pypto.set_vec_tile_shapes(ne)
        e_score_bias_2d = pypto.reshape(e_score_bias_input, [1, ne], inplace=True)  # (160) -> (1,160)

    # 6. 实现kernel逻辑，循环展开BS动态轴
    for bs_idx in pypto.loop(bs_loop, name="LOOP_MOEGATE_L0", idx_name="bs_idx"):
        # 7. 通过view得到tile_logits
        tile_logits = pypto.view(logits_input, view_shape,
                                    [bs_idx * view_shape[0], 0],
                                    valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]), ne])

        # 8. 按照计算图实现运算逻辑，设置set_vec_tile_shapes时应尽可能用满UB，但不要超过UB的大小。
        pypto.set_vec_tile_shapes(view_first, ne)
        e_score_bias_2d_cast = pypto.cast(e_score_bias_2d, tile_logits.dtype)

        # sigmoid
        topk_weights = pypto.sigmoid(tile_logits)  # (bs, ne) fp32

        # add
        topk_weights_add = pypto.add(topk_weights, e_score_bias_2d_cast)  # (8, 160) fp32
        # reshape
        group_unit = ne // num_expert_group
        r1 = pypto.reshape(topk_weights_add,
                            [view_shape[0], num_expert_group, group_unit],
                            valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]), num_expert_group,
                                        group_unit])

        # amax
        pypto.set_vec_tile_shapes(view_first, num_expert_group, group_unit)
        max1 = pypto.amax(r1, -1, False)
        group_weight = max1

        # topk
        pypto.set_vec_tile_shapes(view_first, num_expert_group)
        _, topk_group_indices = pypto.topk(group_weight, topk_group, -1, True)  # (2, topk_group) int32

        # zeros -> full(0)
        topk_group_mask = pypto.full([view_shape[0], num_expert_group], 0.0, group_weight.dtype,
                                        valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]),
                                                    num_expert_group])  # (16, 1)

        # scatter 尾轴不能切
        topk_group_mask_scatter_trans = pypto.scatter_(topk_group_mask, 1, topk_group_indices, 1.0)

        # unsqueeze
        twm_unsqueeze = pypto.unsqueeze(topk_group_mask_scatter_trans, -1)  # (1, 1, 1) fp32

        # expand
        pypto.set_vec_tile_shapes(view_first, num_expert_group, ne)  # ne时 可以切成一块
        twm_expand = pypto.expand_clone(twm_unsqueeze, [view_shape[0], num_expert_group, group_unit],
                                        valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]),
                                                        num_expert_group, group_unit])

        # reshape
        pypto.set_vec_tile_shapes(view_first, num_expert_group, group_unit)  # (1,1,160)
        twm_reshape = pypto.reshape(twm_expand,
                                    [view_shape[0], ne],
                                    valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]), ne])

        # logical_not
        pypto.set_vec_tile_shapes(view_first, ne)
        twm_not = pypto.logical_not(twm_reshape)

        # where
        topk_weights_maskfill = pypto.where(twm_not, 0.0, topk_weights_add)

        # topk2
        _, topk_ids = pypto.topk(topk_weights_maskfill, topk, -1, True)  # (bs, topk) int32

        # tw_gather
        tw_gather = pypto.gather(topk_weights, 1, topk_ids)  # (bs, 8)

        # sum & div
        pypto.set_vec_tile_shapes(view_first, topk)
        if pypto.cond(pypto.symbolic_scalar(renormalize_flag)):
            # sum
            denominator = pypto.sum(tw_gather, -1, True)  # (bs, 1)
            # div for shape (b*s, topk) (b*s, 1)
            topk_weight_out = pypto.div(tw_gather, denominator)  # (bs, topk)
        else:
            denominator = tw_gather
            topk_weight_out = denominator

        # # 9. 将结果搬运到输出tensor上
        weight_k[bs_idx * view_shape[0]:, 0:] = topk_weight_out
        ids_k[bs_idx * view_shape[0]:, 0:] = topk_ids



def gen_row_idx_gloden(hidden_states, top_k):
    num_tokens = hidden_states.shape[0]
    row_idx_len = num_tokens * top_k
    row_idx = (torch.arange(0, row_idx_len, dtype=torch.int32,
                            device=hidden_states.device).view(top_k, -1).permute(1, 0).contiguous())
    return row_idx


def test_select_experts():
    # 1. 设置参数
    bs = 8
    ne = 160
    top_k = 8
    topk_group = 1
    num_expert_group = 1
    renormalize = True
    row_ids_flag = False
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    # 2. 构造多种shape，测试动态case
    for i in range(0, 3):
        if (i == 1):
            bs = 16
        if (i == 2):
            bs = 1037

        # 3. 准备测试数据
        torch.manual_seed(0)
        np.random.seed(0)
        router_logits = torch.rand(
            (bs, ne), dtype=torch.float32, device=f'npu:{device_id}')
        e_score_bias = torch.rand(
            (ne), dtype=torch.bfloat16, device=f'npu:{device_id}')
        topk_weights = torch.zeros(
            (bs, top_k), dtype=torch.float32, device=f'npu:{device_id}')
        topk_ids = torch.zeros(
            (bs, top_k), dtype=torch.int32, device=f'npu:{device_id}')
        row_idx = torch.zeros(
            (bs, top_k), dtype=torch.int32, device=f'npu:{device_id}')

        # 4. 执行kernel并获取结果
        inputs = [router_logits, e_score_bias]
        outputs = [topk_weights, topk_ids, row_idx]
        pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(inputs)]
        pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(outputs)]
        select_experts_glm(pto_inputs, pto_outputs, renormalize, topk_group, num_expert_group, row_ids_flag)
        pypto.runtime._device_synchronize()

        # 5. 与PyTorch参考实现对比
        original_weights = router_logits.sigmoid()
        bias_2d = e_score_bias.unsqueeze(0)
        topk_weights_g_add = original_weights + bias_2d
        tw_view = topk_weights_g_add.view(bs, num_expert_group, -1)
        grouped_weights = tw_view.max(dim=-1).values

        topk_group_indices_g = torch.topk(grouped_weights.to(torch.float32),
                                          k=topk_group,
                                          dim=-1,
                                          sorted=False)[1]
        topk_group_mask = torch.zeros_like(grouped_weights)

        topk_group_mask.scatter_(1, topk_group_indices_g, 1)
        tgm_unsquee = topk_group_mask.unsqueeze(-1)
        tgm_expand = tgm_unsquee.expand(
            bs, num_expert_group, ne // num_expert_group)
        topk_weight_mask = tgm_expand.reshape(bs, -1)
        logical_not_tmp = ~topk_weight_mask.bool()
        topk_weights_fill = topk_weights_g_add.masked_fill(
            logical_not_tmp, 0.0)

        topk_ids_int64 = torch.topk(topk_weights_fill.to(torch.float32),
                                    k=top_k,
                                    dim=-1,
                                    sorted=False)[1]
        topk_ids_int32 = topk_ids_int64.to(torch.int32)

        topk_weights_gather = original_weights.gather(1, topk_ids_int64)

        if renormalize:
            topk_weights_out = topk_weights_gather / \
                topk_weights_gather.sum(dim=-1, keepdim=True)
        else:
            topk_weights_out = topk_weights_gather

        if row_ids_flag:
            golden_row_idx = gen_row_idx_gloden(router_logits, top_k)
        else:
            golden_row_idx = row_idx

        topk_weight_2_tensor_list = topk_weights_out.cpu().flatten().tolist()
        topk_ids_tensor_list = topk_ids_int32.cpu().flatten().tolist()
        golden_row_idx_list = golden_row_idx.cpu().flatten().tolist()

        # weight result
        assert_allclose(np.array(topk_weights.cpu().flatten().tolist()),
                        np.array(topk_weight_2_tensor_list),
                        rtol=1e-5, atol=1e-5)

        # idx result
        assert_allclose(np.array(topk_ids.cpu().flatten().tolist()),
                        np.array(topk_ids_tensor_list),
                        rtol=1e-5, atol=1e-5)

        # row_idx result
        assert_allclose(np.array(row_idx.cpu().flatten().tolist()),
                        np.array(golden_row_idx_list),
                        rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    main()
