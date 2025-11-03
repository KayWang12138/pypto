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


def test_scatterupdate_onboard():
    device_id = os.environ.get('TILE_FWK_STEST_DEVICE_ID', 0)
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
    pto.device_init()

    src_tensor = pto.tensor(src_shape, pto.DataType.DT_INT32, "PTO_TENSOR_SRC")
    index_tensor = pto.tensor(index_shape, pto.DataType.DT_INT32, "PTO_TENSOR_INDEX")
    update_tensor = pto.tensor(dst_shape, pto.DataType.DT_INT32, "PTO_TENSOR_DST")
    dst_tensor = pto.tensor(dst_shape, pto.DataType.DT_INT32, "PTO_TENSOR_DST")

    b_loop_num = math.ceil(src_shape[0] / view_shape[0])
    s_loop_num = math.ceil(src_shape[1] / view_shape[1])
    pto.set_codegen_option("support_dynamic_unaligned", True)
    with pto.function("MAIN", [src_tensor, index_tensor, update_tensor], [dst_tensor]):
        with pto.loop_function("b0", "bidx", pto.loop_range(b_loop_num)) as bloop:
            with pto.loop_function("s0", "sidx", pto.loop_range(s_loop_num)) as sloop:
                for b_idx in bloop:
                    for s_idx in sloop:
                        tmp_dst_tensor = pto.tensor(dst_shape, pto.DataType.DT_INT32, "PTO_TENSOR_TMP")
                        view_tensor_src = pto.view(src_tensor, view_shape, 
                            [
                                (pto.symbolic_scalar(src_shape[0]) -
                                    b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                                (pto.symbolic_scalar(src_shape[1]) -
                                    s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1])),
                                n, d
                            ],
                            [b_idx * view_shape[0], s_idx * view_shape[1], 0, 0],
                        )
                        view_tensor_index = pto.view(index_tensor, [view_shape[0], view_shape[1]],
                            [
                                (pto.symbolic_scalar(index_shape[0]) -
                                    b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                                (pto.symbolic_scalar(index_shape[1]) -
                                    s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1])),
                            ],
                            [b_idx * view_shape[0], s_idx * view_shape[1]],
                        )
                        view_tensor_dst = pto.view(update_tensor, dst_shape, [0, 0, 0, 0])
                        pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1], tile_shape[2], tile_shape[3])
                        tmp_dst_tensor.move(pto.scatter_update(
                            view_tensor_dst, view_tensor_index, view_tensor_src, -2, "PA_BSND", 1))
                        pto.set_vec_tile_shapes(1, 64, n, d)
                        pto.assemble(tmp_dst_tensor, [0, 0, 0, 0], dst_tensor)
                        del view_tensor_dst, view_tensor_src, view_tensor_index, tmp_dst_tensor
    assert isinstance(dst_tensor, pto.tensor)


    input0_tensor = np.random.uniform(2, 3, src_shape).astype(np.int32)
    input1_tensor = np.random.choice(
        range(0, dst_shape[0] * dst_shape[1]), index_shape, replace=False).astype(np.int32) 
    input2_tensor = np.random.uniform(1, 2, dst_shape).astype(np.int32)
    result = copy.copy(input2_tensor)

    a_data = input0_tensor.reshape(src_shape[0] * src_shape[1] * src_shape[2] * src_shape[3]).tolist()
    b_data = input1_tensor.reshape(index_shape[0] * index_shape[1]).tolist()
    c_data = input2_tensor.reshape(dst_shape[0] * dst_shape[1] * dst_shape[2] * dst_shape[3]).tolist()
    d_data = np.zeros(dst_shape[0] * dst_shape[1] * dst_shape[2] * dst_shape[3]).astype(np.int32).tolist()

    pto.device_run_once_data_from_host([a_data, b_data, c_data], [d_data])

    for _b in range(b):
        for _s in range(s):
            result[input1_tensor[_b][_s] // block_size][input1_tensor[_b][_s] % block_size][:] \
                = input0_tensor[_b][_s][:]
    
    assert d_data == result.reshape(block_num * block_size * n * d).tolist()
    pto.device_fini()
