#!/usr/bin/env python3
# coding: utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""
"""
import os
import math
import copy
import numpy as np
import torch
import pto
import pytest
import torch_npu


def test_gather_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    b = 23
    s = 29
    axis = 0
    idx0 = 4
    idx1 = 4
    src_shape = (b, s)
    index_shape = (idx0, idx1)
    view_shape = (b, 4)
    tile_shape = (b, 4)

    pto.runtime._device_init()

    src_tensor = pto.tensor(src_shape, pto.DataType.DT_INT32, "PTO_TENSOR_SRC")
    index_tensor = pto.tensor(
        index_shape, pto.DataType.DT_INT32, "PTO_TENSOR_INDEX")
    dst_tensor = pto.tensor(
        index_shape, pto.DataType.DT_INT32, "PTO_TENSOR_DST")

    b_loop_num = math.ceil(index_shape[0] / view_shape[0])
    s_loop_num = math.ceil(index_shape[1] / view_shape[1])
    pto.set_codegen_options(support_dynamic_unaligned=True)
    with pto.function("GATHER", [src_tensor, index_tensor], [dst_tensor]):
        for b_idx in pto.loop(b_loop_num, name="LOOP_DIV_L0", idx_name="b_idx"):
            for s_idx in pto.loop(s_loop_num, name="LOOP_SIV_L0", idx_name="s_idx"):
                pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])

                view_tensor_src = pto.view(src_tensor, view_shape,
                                           [b_idx * view_shape[0],
                                            s_idx * view_shape[1]],
                                           valid_shape=[
                                               pto.min(pto.symbolic_scalar(src_shape[0]) - b_idx * view_shape[0],
                                                       pto.symbolic_scalar(view_shape[0])),
                                               pto.min(pto.symbolic_scalar(src_shape[1]) - s_idx * view_shape[1],
                                                       pto.symbolic_scalar(view_shape[1]))]
                                           )
                view_tensor_index = pto.view(index_tensor, view_shape,
                                             [b_idx * view_shape[0],
                                              s_idx * view_shape[1]],
                                             valid_shape=[
                                                 pto.min(pto.symbolic_scalar(src_shape[0]) - b_idx * view_shape[0],
                                                         pto.symbolic_scalar(view_shape[0])),

                                                 pto.min(pto.symbolic_scalar(src_shape[1]) - s_idx * view_shape[1],
                                                         pto.symbolic_scalar(view_shape[1]))]
                                             )
                tmp_dst_tensor = pto.tensor()
                tmp_dst_tensor.move(pto.gather(
                    view_tensor_src, axis, view_tensor_index))
                pto.assemble(tmp_dst_tensor, [
                             b_idx * view_shape[0], 0], dst_tensor)
                del view_tensor_src, view_tensor_index, tmp_dst_tensor
    assert isinstance(dst_tensor, pto.tensor)

    input0_tensor = torch.randint(1, 100, src_shape, dtype=torch.int32)
    input1_tensor = torch.randint(
        0, src_shape[axis], index_shape, dtype=torch.int32)
    result_tensor = torch.zeros(index_shape, dtype=torch.int32)

    pto.runtime._device_run_once_data_from_host(
        [input0_tensor, input1_tensor], [result_tensor])

    result = torch.zeros(index_shape, dtype=torch.int32)
    for i in range(index_shape[0]):
        for j in range(index_shape[1]):
            result[i][j] = input0_tensor[input1_tensor[i][j]][j]

    assert torch.equal(result_tensor, result)
    pto.runtime._device_fini
