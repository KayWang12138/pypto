#!/usr/bin/env python3
# coding: utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""
"""
import pto


def test_print_options():
    pto.set_print_options(1, 2, 3, 4)


def test_pass_option():
    # int
    pto.set_pass_options(l1_reuse=1)
    pass_option = pto.get_pass_options()
    assert pass_option["l1_reuse"] == 1
    # map
    pto.set_pass_options(cube_nbuffer_map={3: 4})
    pass_option = pto.get_pass_options()
    assert pass_option["cube_nbuffer_map"] == {3: 4}


def test_host_option():
    pto.set_host_options(only_codegen=True)
    host_option = pto.get_host_options()
    assert host_option["only_codegen"] == True


def test_runtime_option():
    pto.set_runtime_options(first_stitch_task_loop_num=33)
    runtime_option = pto.get_runtime_options()
    assert runtime_option["first_stitch_task_loop_num"] == 33


def test_reset_option():
    pto.set_runtime_options(first_stitch_task_loop_num=23)
    runtime_option = pto.get_runtime_options()
    assert runtime_option["first_stitch_task_loop_num"] == 23
    pto.set_host_options(only_codegen=True)
    host_option = pto.get_host_options()
    assert host_option["only_codegen"] == True
    pto.reset_options()
    runtime_option = pto.get_runtime_options()
    host_option = pto.get_host_options()
    assert runtime_option["first_stitch_task_loop_num"] == 30
    assert host_option["only_codegen"] == False


def test_option():
    pto.set_option("profile_enable", True)
    option = pto.get_option("profile_enable")
    assert option == True
