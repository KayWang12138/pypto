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
import pytest
import numpy as np
import pto


class ScatterParamInfo:
    def __init__(self, sdata: float, axis: int, b, s, idx0, idx1):
        self.src_shape = (b, s)
        self.indices_shape = (idx0, idx1)
        self.view_shape = (b, s)
        self.tile_shape = (b, s)
        self.sdata = sdata
        self.axis = axis


def scatter_2dim_comm_proc(scatter_para, scatter_func):
    pto.runtime._device_init()
    src_shape = scatter_para.src_shape
    indices_shape = scatter_para.indices_shape
    view_shape = scatter_para.view_shape
    tile_shape = scatter_para.tile_shape

    self_tensor = pto.tensor(src_shape, pto.DT_FP32, "PTO_TENSOR_SRC")
    indices_tensor = pto.tensor(indices_shape, pto.DT_INT64, "PTO_TENSOR_INDEX")
    dst_tensor = pto.tensor(src_shape, pto.DT_FP32, "PTO_TENSOR_DST")
    src = pto.element(pto.DT_FP32, scatter_para.sdata)

    b_loop_num = math.ceil(indices_shape[0] / view_shape[0])
    s_loop_num = math.ceil(indices_shape[1] / view_shape[1])
    pto.set_codegen_option("support_dynamic_unaligned", True)
    with pto.function("MAIN", [self_tensor, indices_tensor], [dst_tensor]):
        with pto.loop_function("b0", "bidx", pto.loop_range(b_loop_num)) as bloop:
            with pto.loop_function("s0", "sidx", pto.loop_range(s_loop_num)) as sloop:
                for b_idx in bloop:
                    for s_idx in sloop:
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
                        tmp_dst_tensor.move(scatter_func(view_tensor_src, view_tensor_index, src, scatter_para.axis))
                        pto.assemble(tmp_dst_tensor, [b_idx * view_shape[0], s_idx * view_shape[1]], dst_tensor)
                        del view_tensor_src, view_tensor_index, tmp_dst_tensor
    assert isinstance(dst_tensor, pto.tensor)

    input0_tensor = np.random.uniform(-1, 1, src_shape).astype(np.float32)
    input1_tensor = np.random.uniform(0, src_shape[scatter_para.axis], indices_shape).astype(np.int64)
    a_data = input0_tensor.reshape(src_shape[0] * src_shape[1]).tolist()
    b_data = input1_tensor.reshape(indices_shape[0] * indices_shape[1]).tolist()
    c_data = list([0] * src_shape[0] * src_shape[1])
    pto.runtime._device_run_once_data_from_host([a_data, b_data], [c_data])

    result = copy.copy(input0_tensor)
    for i in range(indices_shape[0]):
        for j in range(indices_shape[1]):
            if scatter_para.axis == 0:
                result[input1_tensor[i][j]][j] = scatter_para.sdata
            else:
                result[i][input1_tensor[i][j]] = scatter_para.sdata

    assert c_data == result.reshape(src_shape[0] * src_shape[1]).tolist()
    pto.runtime._device_fini()


@pytest.mark.skip(reason="Dep operation interface")
def test_scatter__onboard():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    b = 4
    s = 5
    idx0 = 2
    idx1 = 5
    scatter_para = ScatterParamInfo(2.0, 0, b, s, idx0, idx1)

    scatter_2dim_comm_proc(scatter_para, pto.scatter_)


@pytest.mark.skip(reason="Dep operation interface")
def test_scatter_onboard():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    b = 4
    s = 4
    idx0 = 3
    idx1 = 4
    scatter_para = ScatterParamInfo(2.0, 0, b, s, idx0, idx1)

    scatter_2dim_comm_proc(scatter_para, pto.scatter)
