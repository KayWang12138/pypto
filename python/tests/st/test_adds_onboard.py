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
import pto
import pytest
import torch_npu

@pytest.mark.skip(reason="error case.")
def test_adds_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    shape = (72, 71)
    view_shape = (32, 32)
    tile_shape = (32, 32)
    sdata = 1
    pto.runtime._device_init()

    input1 = pto.tensor(shape, pto.DataType.DT_INT32, "PTO_TENSOR_input1")
    input2 = pto.element(pto.DataType.DT_INT32, sdata)
    output = pto.tensor(shape, pto.DataType.DT_INT32, "PTO_TENSOR_output")

    b_loop_num = math.ceil(shape[0] / view_shape[0])
    s_loop_num = math.ceil(shape[1] / view_shape[1])
    pto.set_codegen_option("support_dynamic_unaligned", True)
    with pto.function("MAIN", [input1], [output]):
        for b_idx in pto.loop(b_loop_num, name="b0", idx_name="bidx"):
            for s_idx in pto.loop(s_loop_num, name="s0", idx_name="sidx"):
                view_tensor_a = pto.view(input1, view_shape,
                    [
                        (pto.symbolic_scalar(shape[0]) -
                            b_idx * view_shape[0]).min(pto.symbolic_scalar(view_shape[0])),
                        (pto.symbolic_scalar(shape[1]) -
                            s_idx * view_shape[1]).min(pto.symbolic_scalar(view_shape[1])),
                    ],
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                )
                pto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                view_tensor_a.move(pto.add_s(view_tensor_a, input2))
                pto.assemble(view_tensor_a, [b_idx * view_shape[0], s_idx * view_shape[1]], output)
                del view_tensor_a
    assert isinstance(output, pto.tensor)

    a_data = list(range(shape[0] * shape[1]))
    b_data = list([0] * shape[0] * shape[1])

    pto.runtime._device_run_once_data_from_host([a_data], [b_data])

    assert b_data == [v + sdata for v in a_data]
    pto.device_fini()