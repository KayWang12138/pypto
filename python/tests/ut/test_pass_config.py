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


def test_pass_config():
    assert pto.get_pass_default_config(pto.PassConfigKey.KEY_DUMP_FUNCTION_GRAPH_BEFORE_PASS, True) is False
    pto.set_pass_default_config(pto.PassConfigKey.KEY_DUMP_FUNCTION_GRAPH_BEFORE_PASS, True)
    assert pto.get_pass_default_config(pto.PassConfigKey.KEY_DUMP_FUNCTION_GRAPH_BEFORE_PASS, False) is True

    assert pto.get_pass_default_config(pto.PassConfigKey.KEY_DUMP_FUNCTION_GRAPH_AFTER_PASS, True) is False
    pto.set_pass_default_config(pto.PassConfigKey.KEY_DUMP_FUNCTION_GRAPH_AFTER_PASS, True)
    assert pto.get_pass_default_config(pto.PassConfigKey.KEY_DUMP_FUNCTION_GRAPH_AFTER_PASS, False) is True

    pto.set_pass_config("PVC2_OOO", "ExpandFunction", pto.PassConfigKey.KEY_DUMP_FUNCTION_GRAPH_BEFORE_PASS, True)
    assert pto.get_pass_config("PVC2_OOO", "ExpandFunction", pto.PassConfigKey.KEY_DUMP_FUNCTION_GRAPH_BEFORE_PASS,
            False) is True

    pto.set_pass_config("PVC2_OOO", "ExpandFunction", pto.PassConfigKey.KEY_DUMP_FUNCTION_GRAPH_AFTER_PASS, True)
    assert pto.get_pass_config("PVC2_OOO", "ExpandFunction", pto.PassConfigKey.KEY_DUMP_FUNCTION_GRAPH_AFTER_PASS,
            False) is True

    configs = pto.get_pass_configs("PVC2_OOO", "ExpandFunction")
    assert configs.dumpFunctionGraphBeforePass is True
    assert configs.dumpFunctionGraphAfterPass is True