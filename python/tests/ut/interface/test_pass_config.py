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


def test_pass_config():
    assert pypto.get_pass_default_config(pypto.PassConfigKey.KEY_DUMP_FUNCTION_GRAPH_BEFORE_PASS, True) is False
    pypto.set_pass_default_config(pypto.PassConfigKey.KEY_DUMP_FUNCTION_GRAPH_BEFORE_PASS, True)
    assert pypto.get_pass_default_config(pypto.PassConfigKey.KEY_DUMP_FUNCTION_GRAPH_BEFORE_PASS, False) is True

    assert pypto.get_pass_default_config(pypto.PassConfigKey.KEY_DUMP_FUNCTION_GRAPH_AFTER_PASS, True) is False
    pypto.set_pass_default_config(pypto.PassConfigKey.KEY_DUMP_FUNCTION_GRAPH_AFTER_PASS, True)
    assert pypto.get_pass_default_config(pypto.PassConfigKey.KEY_DUMP_FUNCTION_GRAPH_AFTER_PASS, False) is True

    pypto.set_pass_config("PVC2_OOO", "ExpandFunction", pypto.PassConfigKey.KEY_DUMP_FUNCTION_GRAPH_BEFORE_PASS, True)
    assert pypto.get_pass_config("PVC2_OOO", "ExpandFunction", 
                                 pypto.PassConfigKey.KEY_DUMP_FUNCTION_GRAPH_BEFORE_PASS, False) is True

    pypto.set_pass_config("PVC2_OOO", "ExpandFunction", pypto.PassConfigKey.KEY_DUMP_FUNCTION_GRAPH_AFTER_PASS, True)
    assert pypto.get_pass_config("PVC2_OOO", "ExpandFunction", 
                                 pypto.PassConfigKey.KEY_DUMP_FUNCTION_GRAPH_AFTER_PASS, False) is True

    configs = pypto.get_pass_configs("PVC2_OOO", "ExpandFunction")
    assert configs.dumpFunctionGraphBeforePass is True
    assert configs.dumpFunctionGraphAfterPass is True