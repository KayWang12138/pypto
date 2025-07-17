# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set(_CaseFilterList
        # 框架相关用例
        "TestMemoryAttribute.MemorySizeLimitTest"
        "TestMemoryAttribute.MemorySizeTest"
        #"TestIncastOverlap.AddSubExp_12_8"
        #"SerializationTest.TestAddSub"
        "TestSuite_Backend.AihacBackend_Err1"
        "TestSuite_Backend.SimulationBackend_Err1"
)

set(_Target ${TileFwkUTestNamePrefix}_interface_machine)

file(GLOB_RECURSE _Sources src/*.cpp)

TileFwk_UTest_AddCaseLib(
        TARGET ${_Target}
        SOURCES ${_Sources}
        PUBLIC_LINK_LIBRARIES tile_fwk_interface
        GTEST_FILTER_LIST ${_CaseFilterList}
)
