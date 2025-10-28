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
import pto

def test_runtime():
    """Test set_runtime_option and get_runtime_option"""
    
    keys = ["machine_sched_mode","machine_sched_mode","machine_sched_mode","machine_sched_mode"]
    pto.set_runtime_option(keys[0], "val1")
    assert pto.get_runtime_option(keys[0]) == "val1"
    pto.set_runtime_option(keys[1], 1)
    assert pto.get_runtime_option(keys[1]) == 1
    pto.set_runtime_option(keys[2], [1,2,3])
    assert pto.get_runtime_option(keys[2]) == [1,2,3]
    pto.set_runtime_option(keys[3], {10:23})
    assert pto.get_runtime_option(keys[3]) == {10:23}
