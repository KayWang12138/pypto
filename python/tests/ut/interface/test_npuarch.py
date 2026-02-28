#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
import pypto

import torch
import torch_npu
import numpy as np
from numpy.testing import assert_allclose


def runtime_options_list():
    # 910
    if pypto.platform.npuarch == 'DAV_1001':
        return {
            "stitch_function_inner_memory": 8192,
            "stitch_function_outcast_memory": 4096,
            "stitch_function_num_initial": 128,
            "device_sched_mode": 3
        }
    # 910B/910C
    elif pypto.platform.npuarch == 'DAV_2201':
        return {
            "stitch_function_inner_memory": 4096,
            "stitch_function_outcast_memory": 4096,
            "stitch_function_num_initial": 128,
            "device_sched_mode": 3
        }
    # 950
    elif pypto.platform.npuarch == 'DAV_3510':
        return {
            "stitch_function_inner_memory": 4096,
            "stitch_function_outcast_memory": 4096,
            "stitch_function_num_initial": 128,
            "device_sched_mode": 1
        }
    else:
        return {
            "stitch_function_inner_memory": 4096,
            "stitch_function_outcast_memory": 4096,
            "stitch_function_num_initial": 128,
            "device_sched_mode": 1
        }


def test_npuarch():
    assert isinstance(pypto.platform.npuarch, str)
    assert pypto.platform.npuarch in ['DAV_1001', 'DAV_2201', 'DAV_3510']
