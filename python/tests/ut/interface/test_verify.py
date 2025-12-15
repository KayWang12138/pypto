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
import pypto
import pytest
import torch


def add(inputs, outputs):
    a, b = inputs
    c = outputs[0]
    with pypto.function("add", a, b, c):
        for _ in pypto.loop(1):
            pypto.set_vec_tile_shapes(16, 16)
            c[:] = a + b


@pytest.mark.skip(reason="Flatten inputs and outputs")
def test_verify_default():
    a = torch.ones((64, 64))
    b = torch.ones((64, 64))
    c = torch.zeros((64, 64))

    pypto.verify(add, [a, b], [c], [a + b])


@pytest.mark.skip(reason="Flatten inputs and outputs")
def test_verify_full_options():
    a = torch.ones((64, 64))
    b = torch.ones((64, 64))
    c = torch.zeros((64, 64))

    pypto.verify(add, [a, b], [c], [a + b],
                 host_options={"only_codegen": True},
                 codegen_options={"support_dynamic_unaligned": True},
                 verify_options={"verify_tensor_graph": True,
                                 "verify_pass": True,
                                 "check_precision": True,
                                 "dump_tensor": True,
                                 "dump_operation": True,
                                 "profile_enable": True
                                 }
                 )
