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
    test_expert_offset_table()


@pto.jit
def get_table_main(inputs, outputs):
    pto.set_host_option("only_codegen", True)
    pto.set_codegen_option("support_dynamic_unaligned", True)
    expert_tokens = inputs[0]
    expert_offset = outputs[0]

    pto.mark_dynamic(inputs[0], 0)

    with pto.function("GET_TOKEN_TABLE", [expert_tokens], [expert_offset]):
        def inside_main_function():
            expert_num = expert_tokens.shape[0]
            pto.set_vec_tile_shapes(32)
            # 计算每个专家的token的偏移地址
            for _ in pto.loop(0, 1, 1, name="LOOP_init", idx_name="idx"):
                def loop_for_init_offset():
                    pto.set_vec_tile_shapes(32)
                    tmp = pto.full([32,], 0, pto.DT_INT32)
                    pto.assemble(tmp, [0,], expert_offset)
                loop_for_init_offset()
            for exp_idx in pto.loop(1, expert_num, 1, name="LOOP_expert", idx_name="exp_idx", submit_before_loop=True):
                def loop_for_offset(exp_idx):
                    pto.set_vec_tile_shapes(32)
                    view_shape = [pto.min(exp_idx, expert_num),]
                    tmp_view = pto.view(expert_tokens, [16,], [0,], valid_shape=view_shape)
                    tmp_cast = pto.cast(tmp_view, pto.DT_FP32)
                    tmp_acc = pto.sum(tmp_cast, -1, True)
                    tmp_int = pto.cast(tmp_acc, pto.DT_INT32)
                    pto.assemble(tmp_int, [(exp_idx),], expert_offset)
                loop_for_offset(exp_idx)
        inside_main_function()


def get_token_acc_table(expert_tokens):
    assert len(expert_tokens.shape) == 1
    token_acc_table = torch.zeros_like(expert_tokens)
    for i in range(1, expert_tokens.shape[0]):
        token_acc_table[i] = torch.sum(expert_tokens[0:i])
    return token_acc_table


def test_expert_offset_table():
    bs = 16
    per_expert_num = 8
    device_id = int(os.environ.get('TILE_FWK_STEST_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    np.random.seed(0)
    expert_tokens = torch.randint(0, bs, (per_expert_num,), dtype = torch.int32, device = f'npu:{device_id}')
    expert_offset = torch.zeros_like(expert_tokens, device = f'npu:{device_id}')

    inputs = [expert_tokens]
    outputs = [expert_offset]
    get_table_main(inputs, outputs)
    pto.runtime._device_synchronize()

    # golden
    token_acc_table_tensor = get_token_acc_table(expert_tokens)
    assert_allclose(np.array(expert_offset.cpu().flatten().tolist()), np.array(token_acc_table_tensor.cpu().flatten().tolist()), rtol=0.005, atol=0.005)

if __name__ == "__main__":
    main()