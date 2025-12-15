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
import torch
import pypto
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
    test_select_experts_mm()


@allow_in_graph
def graph_select_experts_mm(hidden_states, gate_weight, router_logits_out):
    if isinstance(hidden_states, FakeTensor):
        return
    inputs = {
        hidden_states: [0],
        gate_weight: []
    }
    outputs = {
        router_logits_out: [0]
    }
    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
    select_experts_mm(pto_inputs, pto_outputs)
    pypto.runtime._device_synchronize()


def gate(gate_weight: torch.Tensor,  # gate matmul weights
             hidden_states: torch.Tensor  # Hidden states of shape (num_tokens, hidden_size).
             ) -> torch.Tensor:
    bs = hidden_states.shape[0]
    ne = gate_weight.shape[0]
    router_logits_out = torch.empty((bs, ne), dtype=gate_weight.dtype, device=hidden_states.device)

    graph_select_experts_mm(hidden_states, gate_weight, router_logits_out)
    return router_logits_out


@pypto.jit(
    runtime_options={
    "cfgcache_device_task_num": 100,
    "cfgcache_root_task_num": 1000,
    "cfgcache_leaf_task_num": 10000},
    host_options={"only_codegen": True},
    codegen_options={"support_dynamic_unaligned": True}
)
def select_experts_mm(in_tensors, out_tensors):
    # 泳道图使能  pypto.set_option('profile_enable', True)

    # 2. 从入参拿到输入和输出tensor
    hidden_states = in_tensors[0]
    mm_weight = in_tensors[1]
    router_logits_out = out_tensors[0]

    # 3. 得到动态tensor的shape
    bs = hidden_states.shape[0]
    ne = mm_weight.shape[0]
    h_num = hidden_states.shape[1]

    view_shape = (32, h_num)

    bs_loop = (bs + view_shape[0] - 1) // view_shape[0]

    # 4. 实现kernel逻辑，循环展开BS动态轴
    for bs_idx in pypto.loop(bs_loop, name="LOOP_MOE_MM_L0", idx_name="bs_idx"):

        # 5. 通过view得到tile_logits
        tile_hidden_states = pypto.view(hidden_states, view_shape,
                                        [bs_idx * view_shape[0], 0],
                                        valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]),
                                                        h_num])

        pypto.set_cube_tile_shapes([32, 32], [512, 1024], [16, 16])

        res = pypto.matmul(tile_hidden_states, mm_weight, tile_hidden_states.dtype, b_trans=True)

        # 6. 将结果搬运到输出tensor上
        router_logits_out[bs_idx * view_shape[0]:, 0:] = res


def test_select_experts_mm():
    # 1. 设置参数
    bs = 64
    ne = 160
    h_num = 5120

    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    # 2. 构造多种shape，测试动态case
    for i in range(0, 2):
        if i == 1:
            bs = 1026
        # 3. 准备测试数据
        torch.manual_seed(0)
        np.random.seed(0)
        hidden_states = torch.rand((bs, h_num), dtype=torch.float32, device=f'npu:{device_id}')
        mm_weight = torch.rand((ne, h_num), dtype=torch.float32, device=f'npu:{device_id}')
        router_logits_out = torch.empty((bs, ne), dtype=torch.float32, device=f'npu:{device_id}')

        # 4. 执行kernel并获取结果
        inputs = {
            hidden_states: [0],
            mm_weight: []
        }
        outputs = {
            router_logits_out: [0]
        }
        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
        pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            select_experts_mm(pto_inputs, pto_outputs)
        g.replay()
        pypto.runtime._device_synchronize()

        # 5. 与PyTorch参考实现对比
        result = torch.matmul(hidden_states, mm_weight.t())
        result_list = result.cpu().flatten().tolist()

        # weight result
        assert_allclose(np.array(router_logits_out.cpu().flatten().tolist()),
                        np.array(result_list),
                        rtol=5e-3, atol=5e-3)


if __name__ == "__main__":
    main()
