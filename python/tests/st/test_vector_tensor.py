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

import os
import math
import copy
import pto
import pytest
from numpy.testing import assert_allclose
import torch
import torch_npu
import numpy as np


def test_exp_tensor_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pto.DT_FP32
    tiling = 32
    n, m = tiling * 1, tiling * 1
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pto.set_codegen_option("support_dynamic_unaligned", True)
    pto.runtime._device_init()
    a = pto.tensor(shape, dtype, "SQRT_TENSOR_a")
    b = pto.tensor(shape, dtype, "SQRT_TENSOR_b")

    with pto.function("EXP", [a], [b]):
        for b_idx in pto.loop(int(math.ceil(n / view_shape[0])), name="LOOP_SQRT_L0", idx_name="b_idx"):
            for s_idx in pto.loop(int(math.ceil(m / view_shape[1])), name="LOOP_SQRT_L1", idx_name="s_idx"):
                tile_a = pto.view(a, view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[(pto.symbolic_scalar(n) -
                    b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                    (pto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))])
                pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tile_a.move(tile_a.exp())
                pto.assemble(tile_a, [b_idx * view_shape[0], s_idx * view_shape[1]], b)
                del tile_a
    a_tensor = torch.rand(n, m, dtype=torch.float32) * 100
    b_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto.runtime._device_run_once_data_from_host([a_tensor], [b_tensor])

    expected = torch.exp(a_tensor)
    assert_allclose(b_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pto.runtime._device_fini()


def test_scatterupdate_tensor_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    b = 1
    s = 1
    n = 1
    d = 8
    block_num = 1
    block_size = 1
    src_shape = (b, s, n, d)
    index_shape = (b, s)
    dst_shape = (block_num, block_size, n, d)

    view_shape = (b, s, n, d)
    tile_shape = (b, s, n, d)
    pto.runtime._device_init()

    src_tensor = pto.tensor(src_shape, pto.DataType.DT_INT32, "PTO_TENSOR_SRC")
    index_tensor = pto.tensor(
        index_shape, pto.DataType.DT_INT32, "PTO_TENSOR_INDEX")
    update_tensor = pto.tensor(
        dst_shape, pto.DataType.DT_INT32, "PTO_TENSOR_DST")
    dst_tensor = pto.tensor(dst_shape, pto.DataType.DT_INT32, "PTO_TENSOR_DST")

    b_loop_num = math.ceil(src_shape[0] / view_shape[0])
    s_loop_num = math.ceil(src_shape[1] / view_shape[1])
    pto.set_codegen_option("support_dynamic_unaligned", True)
    with pto.function("MAIN", [src_tensor, index_tensor, update_tensor], [dst_tensor]):
        for b_idx in pto.loop(b_loop_num, name="b0", idx_name="bidx"):
            for s_idx in pto.loop(s_loop_num, name="s0", idx_name="sidx"):
                tmp_dst_tensor = pto.tensor(
                    dst_shape, pto.DataType.DT_INT32, "PTO_TENSOR_TMP")
                view_tensor_src = pto.view(src_tensor, view_shape,
                                           [b_idx * view_shape[0], s_idx *
                                               view_shape[1], 0, 0],
                                           valid_shape=[
                                               pto.min(pto.symbolic_scalar(
                                                   src_shape[0]) - b_idx * view_shape[0],
                                                   pto.symbolic_scalar(view_shape[0])),
                                               pto.min(pto.symbolic_scalar(
                                                   src_shape[1]) - s_idx * view_shape[1],
                                                   pto.symbolic_scalar(view_shape[1])),
                                               n, d
                                           ]
                                           )
                view_tensor_index = pto.view(index_tensor, [view_shape[0], view_shape[1]],
                                             [b_idx * view_shape[0],
                                                 s_idx * view_shape[1]],
                                             valid_shape=[
                    pto.min(pto.symbolic_scalar(index_shape[0]) - b_idx * view_shape[0],
                            pto.symbolic_scalar(view_shape[0])),
                    pto.min(pto.symbolic_scalar(index_shape[1]) - s_idx * view_shape[1],
                            pto.symbolic_scalar(view_shape[1])),
                ],

                )
                view_tensor_dst = pto.view(update_tensor, dst_shape, [0, 0, 0, 0])
                pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1], tile_shape[2], tile_shape[3])
                tmp_dst_tensor.move(view_tensor_dst.scatter_update(-2, view_tensor_index, view_tensor_src))
                pto.set_vec_tile_shapes(1, 64, n, d)
                pto.assemble(tmp_dst_tensor, [0, 0, 0, 0], dst_tensor)
                del view_tensor_dst, view_tensor_src, view_tensor_index, tmp_dst_tensor
    assert isinstance(dst_tensor, pto.tensor)

    input0_tensor = np.random.uniform(2, 3, src_shape).astype(np.int32)
    input1_tensor = np.random.choice(
        range(0, dst_shape[0] * dst_shape[1]), index_shape, replace=False).astype(np.int32)
    input2_tensor = np.random.uniform(1, 2, dst_shape).astype(np.int32)
    result = copy.copy(input2_tensor)
    d_data = np.zeros(dst_shape[0] * dst_shape[1]
                      * dst_shape[2] * dst_shape[3]).astype(np.int32)

    a_tensor = torch.from_numpy(input0_tensor)
    b_tensor = torch.from_numpy(input1_tensor)
    c_tensor = torch.from_numpy(input2_tensor)
    d_tensor = torch.from_numpy(d_data)
    pto.runtime._device_run_once_data_from_host(
        [a_tensor, b_tensor, c_tensor], [d_tensor])

    for _b in range(b):
        for _s in range(s):
            result[input1_tensor[_b][_s] // block_size][input1_tensor[_b][_s] % block_size][:] \
                = input0_tensor[_b][_s][:]
    result_t = torch.from_numpy(result).to(d_tensor.device).to(d_tensor.dtype)
    result_t = result_t.reshape_as(d_tensor)
    assert (d_tensor == result_t).all().item()
    pto.runtime._device_fini()


class ScatterParamInfo:
    def __init__(self, sdata: float, axis: int, b, s, idx0, idx1):
        self.src_shape = (b, s)
        self.indices_shape = (idx0, idx1)
        self.view_shape = (b, s)
        self.tile_shape = (b, s)
        self.sdata = sdata
        self.axis = axis


def scatter_2dim_tensor_proc(scatter_para, is_inplace):
    pto.runtime._device_init()
    src_shape = scatter_para.src_shape
    indices_shape = scatter_para.indices_shape
    view_shape = scatter_para.view_shape
    tile_shape = scatter_para.tile_shape

    self_tensor = pto.tensor(src_shape, pto.DT_FP32, "PTO_TENSOR_SRC")
    indices_tensor = pto.tensor(indices_shape, pto.DT_INT64, "PTO_TENSOR_INDEX")
    dst_tensor = pto.tensor(src_shape, pto.DT_FP32, "PTO_TENSOR_DST")
    src = scatter_para.sdata

    b_loop_num = math.ceil(indices_shape[0] / view_shape[0])
    s_loop_num = math.ceil(indices_shape[1] / view_shape[1])
    pto.set_codegen_option("support_dynamic_unaligned", True)
    with pto.function("MAIN", [self_tensor, indices_tensor], [dst_tensor]):
        for b_idx in pto.loop(b_loop_num, name="b0", idx_name="bidx"):
            for s_idx in pto.loop(s_loop_num, name="s0", idx_name="sidx"):
                tmp_dst_tensor = pto.tensor(view_shape, pto.DT_FP32, "PTO_TENSOR_TMP")
                view_tensor_src = pto.view(self_tensor, view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[(pto.symbolic_scalar(src_shape[0]) -
                        b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                        (pto.symbolic_scalar(src_shape[1]) -
                        s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))])
                view_tensor_index = pto.view(indices_tensor, view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[(pto.symbolic_scalar(indices_shape[0]) -
                        b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                        (pto.symbolic_scalar(indices_shape[1]) -
                        s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1]))])
                pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                if is_inplace == True:
                    tmp_dst_tensor.move(view_tensor_src.scatter_(scatter_para.axis, view_tensor_index, src))
                else:
                    tmp_dst_tensor.move(view_tensor_src.scatter(scatter_para.axis, view_tensor_index, src))
                pto.assemble(tmp_dst_tensor, [b_idx * view_shape[0], s_idx * view_shape[1]], dst_tensor)
                del view_tensor_src, view_tensor_index, tmp_dst_tensor
    assert isinstance(dst_tensor, pto.tensor)

    input0_tensor = torch.rand(*src_shape, dtype=torch.float32) * 2 - 1
    input1_tensor = torch.randint(0, src_shape[scatter_para.axis], indices_shape, dtype=torch.int64)
    c_tensor = torch.zeros_like(input0_tensor)

    pto.runtime._device_run_once_data_from_host([input0_tensor, input1_tensor], [c_tensor])

    result = input0_tensor.clone()
    for i in range(indices_shape[0]):
        for j in range(indices_shape[1]):
            if scatter_para.axis == 0:
                result[input1_tensor[i, j], j] = scatter_para.sdata
            else:
                result[i, input1_tensor[i, j]] = scatter_para.sdata

    assert torch.equal(c_tensor, result)
    pto.runtime._device_fini()


def test_scatter__tensor_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    b = 4
    s = 5
    idx0 = 2
    idx1 = 5
    scatter_para = ScatterParamInfo(2.0, 0, b, s, idx0, idx1)

    scatter_2dim_tensor_proc(scatter_para, True)


def test_scatter_tensor_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    b = 4
    s = 4
    idx0 = 3
    idx1 = 4
    scatter_para = ScatterParamInfo(2.0, 1, b, s, idx0, idx1)

    scatter_2dim_tensor_proc(scatter_para, False)
