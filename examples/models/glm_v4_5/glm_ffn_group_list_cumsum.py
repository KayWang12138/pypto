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
import torch_npu
import pypto
import pytest
import numpy as np
from numpy.testing import assert_allclose
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph

def main():
    test_group_list_cumsum()


def get_token_acc_table(group_list):
    assert len(group_list.shape) == 1
    token_acc_table = torch.zeros_like(group_list)
    for i in range(1, group_list.shape[0]):
        token_acc_table[i] = torch.sum(group_list[0:i])
    return token_acc_table


@pypto.jit(
    host_options={"only_codegen": True},
    codegen_options={"support_dynamic_unaligned": True}
)
def moe_group_list_cumsum(inputs, outputs):
    group_list = inputs[0]
    group_list_cumsum = outputs[0]

    expert_num = group_list.shape[0]
    pypto.set_vec_tile_shapes(32)
    # 计算每个专家的token的偏移地址
    for _ in pypto.loop(0, 1, 1, name="LOOP_init", idx_name="idx"):
        def loop_for_init_offset():
            group_list_cumsum[0, ] = 0
        loop_for_init_offset()
    for exp_idx in pypto.loop(1, expert_num, 1, name="LOOP_expert", idx_name="exp_idx", submit_before_loop=True):
        def loop_for_offset(exp_idx):
            pypto.set_vec_tile_shapes(32)
            view_shape = [pypto.min(exp_idx, expert_num),]
            tmp_view = pypto.view(group_list, [16,], [0,], valid_shape=view_shape)
            tmp_cast = pypto.cast(tmp_view, pypto.DT_FP32)
            tmp_acc = pypto.sum(tmp_cast, -1, True)
            tmp_int = pypto.cast(tmp_acc, pypto.DT_INT32)
            pypto.assemble(tmp_int, [(exp_idx),], group_list_cumsum)
        loop_for_offset(exp_idx)



@allow_in_graph
def glm_router_expert_cumsum(group_list_input):
    group_list = group_list_input.to(torch.int32)
    group_list_cumsum = torch.empty_like(group_list, device=group_list_input.device)
    inputs = {
        group_list: [0]
    }
    outputs = {
        group_list_cumsum: [0]
    }
    if not isinstance(group_list_input, FakeTensor):
        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
        pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
        moe_group_list_cumsum(pto_inputs, pto_outputs)
        pypto.runtime._device_synchronize()
    return group_list_cumsum


def test_group_list_cumsum():
    bs = 3
    per_expert_num = 20
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    for i in range(0, 2):
        if (i == 0):
            bs = 2
        if (i == 1):
            bs = 1
        np.random.seed(0)
        group_list = torch.randint(0, bs, (per_expert_num,), dtype = torch.int32, device = f'npu:{device_id}')
        group_list_cumsum = torch.empty_like(group_list, device = f'npu:{device_id}')

        inputs = {
            group_list: [0]
        }
        outputs = {
            group_list_cumsum: [0]
        }
        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
        pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
        moe_group_list_cumsum(pto_inputs, pto_outputs)
        pypto.runtime._device_synchronize()

        # golden
        token_acc_table_tensor = get_token_acc_table(group_list)
        assert_allclose(np.array(group_list_cumsum.cpu().flatten().tolist()),\
                        np.array(token_acc_table_tensor.cpu().flatten().tolist()), rtol=0.005, atol=0.005)


if __name__ == "__main__":
    main()