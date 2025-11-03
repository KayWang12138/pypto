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
import pytest

@pytest.mark.skip(reason="error case.")
def test_gather_onboard():
    device_id = os.environ.get('TILE_FWK_STEST_DEVICE_ID', 0)
    b = 4
    s = 4
    axis = 0
    index_shape = 2
    src_shape = (b, s)
    view_shape = (index_shape, s)
    tile_shape = (index_shape, s)

    pto.runtime._device_init()

    src_tensor = pto.tensor(src_shape, pto.DataType.DT_INT32, "PTO_TENSOR_SRC")
    index_tensor = pto.tensor([index_shape], pto.DataType.DT_INT32, "PTO_TENSOR_INDEX")
    dst_tensor = pto.tensor((index_shape, s), pto.DataType.DT_INT32, "PTO_TENSOR_DST")

    b_loop_num = math.ceil(index_shape / view_shape[0])
    pto.set_codegen_option("support_dynamic_unaligned", True)
    with pto.function("MAIN", [src_tensor, index_tensor], [dst_tensor]):
        with pto.loop_function("b0", "bidx", pto.loop_range(b_loop_num)) as bloop:
            for b_idx in bloop:
                tmp_dst_tensor = pto.tensor((index_shape, s), pto.DataType.DT_INT32, "PTO_TENSOR_TMP")
                view_tensor_src = pto.view(src_tensor, src_shape, [0, 0])
                view_tensor_index = pto.view(index_tensor, [index_shape], [b_idx * view_shape[0]])
                pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tmp_dst_tensor.move(pto.gather(view_tensor_src, view_tensor_index, axis))
                pto.assemble(tmp_dst_tensor, [b_idx * view_shape[0], 0], dst_tensor)
                del view_tensor_src, view_tensor_index, tmp_dst_tensor
    assert isinstance(dst_tensor, pto.tensor)

    input0_tensor = np.random.uniform(1, 100, src_shape).astype(np.int32)
    input1_tensor = np.random.uniform(0, b, index_shape).astype(np.int32)

    a_data = input0_tensor.reshape(src_shape[0] * src_shape[1]).tolist()
    b_data = input1_tensor.tolist()
    c_data = list([0] * index_shape * s)
    pto.runtime._device_run_once_data_from_host([a_data, b_data], [c_data])

    result = np.random.uniform(0, 0, (index_shape, s)).astype(np.int32)
    for i in range(index_shape):
        result[i] = input0_tensor[input1_tensor[i]]

    assert c_data == result.reshape(index_shape * s).tolist()
    pto.device_fini()