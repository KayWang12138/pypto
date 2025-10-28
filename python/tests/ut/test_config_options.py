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


def test_print_options():
    with pto.function("MAIN", [], []):
        pto.set_print_options(1, 2, 3, 4)


def test_pass_option():
    with pto.function("MAIN", [], []):
        # int
        pto.set_pass_option("l1_reuse", 1)
        pass_option = pto.get_pass_option("l1_reuse")
        assert pass_option == 1
        # map
        pto.set_pass_option("cube_nbuffer_map", {3: 4})
        pass_option = pto.get_pass_option("cube_nbuffer_map")
        assert pass_option == {3: 4}


def test_host_option():
    with pto.function("MAIN", [], []):
        pto.set_host_option("only_codegen", True)
        host_option = pto.get_host_option("only_codegen")
        assert host_option == True
