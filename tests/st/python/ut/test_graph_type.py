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


def test_graph_type():
    # Make sure all graph types are defined
    assert isinstance(pto.graph_type.TENSOR_GRAPH, pto.graph_type)
    assert isinstance(pto.graph_type.TILE_GRAPH, pto.graph_type)
    assert isinstance(pto.graph_type.EXECUTE_GRAPH, pto.graph_type)
    assert isinstance(pto.graph_type.BLOCK_GRAPH, pto.graph_type)
    assert isinstance(pto.graph_type.LEAF_VF_GRAPH, pto.graph_type)
    assert isinstance(pto.graph_type.INVALID, pto.graph_type)
