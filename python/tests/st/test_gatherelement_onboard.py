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
import numpy as np

import pto


def test_gatherelement_onboard():
    device_id = os.environ.get('TILE_FWK_STEST_DEVICE_ID', 0)
    b = 4
    s = 4
    idx0 = 4
    idx1 = 4
    sdata = 2
    src_shape = (b, s)
    index_shape = (idx0, idx1)
    view_shape = (4, 4)
    tile_shape = (4, 4)

    pto.device_init()

    src_tensor = pto.tensor(src_shape, pto.DataType.DT_INT32, "PTO_TENSOR_SRC")
    index_tensor = pto.tensor(index_shape, pto.DataType.DT_INT32, "PTO_TENSOR_INDEX")
    axis = 0
    dst_tensor = pto.tensor(index_shape, pto.DataType.DT_INT32, "PTO_TENSOR_DST")

    b_loop_num = math.ceil(index_shape[0] / view_shape[0])
    s_loop_num = math.ceil(index_shape[1] / view_shape[1])
    pto.set_codegen_option("support_dynamic_unaligned", True)
    with pto.function("MAIN", [src_tensor, index_tensor], [dst_tensor]):
        with pto.loop_function("b0", "bidx", pto.loop_range(b_loop_num)) as bloop:
            with pto.loop_function("s0", "sidx", pto.loop_range(s_loop_num)) as sloop:
                for b_idx in bloop:
                    for s_idx in sloop:
                        tmp_dst_tensor = pto.tensor(view_shape, pto.DataType.DT_INT32, "PTO_TENSOR_TMP")
                        view_tensor_src = pto.view(src_tensor, view_shape,
                            [
                                (pto.symbolic_scalar(src_shape[0]) -
                                    b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                                (pto.symbolic_scalar(src_shape[1]) -
                                    s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1])),
                            ],
                            [b_idx * view_shape[0], s_idx * view_shape[1]],
                        )
                        view_tensor_index = pto.view(index_tensor, view_shape,
                            [
                                (pto.symbolic_scalar(index_shape[0]) -
                                    b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                                (pto.symbolic_scalar(index_shape[1]) -
                                    s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1])),
                            ],
                            [b_idx * view_shape[0], s_idx * view_shape[1]],
                        )
                        pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                        tmp_dst_tensor.move(pto.gather_element(view_tensor_src, view_tensor_index, axis))
                        pto.assemble(tmp_dst_tensor, [b_idx * view_shape[0], s_idx * view_shape[1]], dst_tensor)
                        del view_tensor_src, view_tensor_index, tmp_dst_tensor
    assert isinstance(dst_tensor, pto.tensor)

    input0_tensor = np.random.uniform(1, 100, src_shape).astype(np.int32)
    input1_tensor = np.random.uniform(0, 0, index_shape).astype(np.int32)
    for i in range(idx1):
        input1_tensor[i] = np.random.uniform(0, src_shape[0], index_shape[0]).astype(np.int32)
    input1_tensor = input1_tensor.transpose()

    a_data = input0_tensor.reshape(src_shape[0] * src_shape[1]).tolist()
    b_data = input1_tensor.reshape(index_shape[0] * index_shape[1]).tolist()
    c_data = list([0] * index_shape[0] * index_shape[1])
    pto.device_run_once_data_from_host([a_data, b_data], [c_data])

    result = np.random.uniform(0, 0, index_shape).astype(np.int32)
    for i in range(index_shape[0]):
        for j in range(index_shape[1]):
            result[i][j] = input0_tensor[input1_tensor[i][j]][j]

    assert c_data == result.reshape(index_shape[0] * index_shape[1]).tolist()
    pto.device_fini()