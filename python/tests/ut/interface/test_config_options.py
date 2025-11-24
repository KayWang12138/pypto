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


def test_print_options():
    pypto.set_print_options(1, 2, 3, 4)


def test_pass_option():
    # int
    pypto.set_pass_options(l1_reuse=1)
    pass_option = pypto.get_pass_options()
    assert pass_option["l1_reuse"] == 1
    # map
    pypto.set_pass_options(cube_nbuffer_map={3: 4})
    pass_option = pypto.get_pass_options()
    assert pass_option["cube_nbuffer_map"] == {3: 4}


def test_host_option():
    pypto.set_host_options(only_codegen=True)
    host_option = pypto.get_host_options()
    assert host_option["only_codegen"] == True


def test_runtime_option():
    pypto.set_runtime_options(stitch_callop_max_num=30000)
    runtime_option = pypto.get_runtime_options()
    assert runtime_option["stitch_callop_max_num"] == 30000


def test_reset_option():
    pypto.set_runtime_options(first_stitch_task_loop_num=23)
    runtime_option = pypto.get_runtime_options()
    assert runtime_option["first_stitch_task_loop_num"] == 23
    pypto.set_host_options(only_codegen=True)
    host_option = pypto.get_host_options()
    assert host_option["only_codegen"] == True
    pypto.reset_options()
    runtime_option = pypto.get_runtime_options()
    host_option = pypto.get_host_options()
    assert runtime_option["first_stitch_task_loop_num"] == 30
    assert host_option["only_codegen"] == False


def test_option():
    pypto.set_option("profile_enable", True)
    option = pypto.get_option("profile_enable")
    assert option == True
