# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set(_CaseFilterList
        "TestRange/RangeOperationTest.TestRange/0"
        "TestRange/RangeOperationTest.TestRange/1"
        "TestAdd/AddOperationTest.TestAdd/10"
        "TestAdd/AddOperationTest.TestAdd/11"
        "TestConcat/ConcatOperationTest.TestConcat/0"
        "TestConcat/ConcatOperationTest.TestConcat/1"
        "TestAdds/AddsOperationTest.TestAdds/20"
        "TestAdds/AddsOperationTest.TestAdds/21"
        "TestAdds/AddsOperationTest.TestAdds/22"
        "TestCast/CastOperationTest.TestCast/10"
        "TestCast/CastOperationTest.TestCast/11"
        "TestCast/CastOperationTest.TestCast/12"
        "TestDiv/DivOperationTest.TestDiv/13"
        "TestDiv/DivOperationTest.TestDiv/14"
        "TestDivs/DivsOperationTest.TestDivs/0"
        "TestDivs/DivsOperationTest.TestDivs/1"
        "TestDivs/DivsOperationTest.TestDivs/2"
        "TestExp/ExpOperationTest.TestExp/18"
        "TestExp/ExpOperationTest.TestExp/19"
        "TestMul/MulOperationTest.TestMul/21"
        "TestMul/MulOperationTest.TestMul/22"
        "TestMul/MulOperationTest.TestMul/25"
        "TestMul/MulOperationTest.TestMul/26"
        "TestMul/MulOperationTest.TestMul/27"
        "TestMuls/MulsOperationTest.TestMuls/13"
        "TestMuls/MulsOperationTest.TestMuls/14"
        "TestMuls/MulsOperationTest.TestMuls/15"
        "TestRowMinSingle/RowMinSingleOperationTest.TestRowMinSingle/0"
        "TestRowMinSingle/RowMinSingleOperationTest.TestRowMinSingle/1"
        "TestRowMaxSingle/RowMaxSingleOperationTest.TestRowMaxSingle/0"
        "TestRowMaxSingle/RowMaxSingleOperationTest.TestRowMaxSingle/1"
        "TestRowSumSingle/RowSumSingleOperationTest.TestRowSumSingle/0"
        "TestRowSumSingle/RowSumSingleOperationTest.TestRowSumSingle/1"
        "TestScatter_/Scatter_OperationTest.TestScatter_/0"
        "TestScatter_/Scatter_OperationTest.TestScatter_/1"
        "TestScatter/ScatterOperationTest.TestScatter/0"
        "TestScatter/ScatterOperationTest.TestScatter/2"
        "TestScatterUpdate/ScatterUpdateOperationTest.TestScatterUpdate/0"
        "TestScatterUpdate/ScatterUpdateOperationTest.TestScatterUpdate/1"
        "TestRsqrt/RsqrtOperationTest.TestRsqrt/0"
        "TestRsqrt/RsqrtOperationTest.TestRsqrt/2"
        "TestRsqrt/RsqrtOperationTest.TestRsqrt/3"
        "TestSqrt/SqrtOperationTest.TestSqrt/0"
        "TestLog/LogOperationTest.TestLog/2"
        "TestLog/LogOperationTest.TestLog/5"
        "TestSub/SubOperationTest.TestSub/2"
        "TestSub/SubOperationTest.TestSub/3"
        "TestSubs/SubsOperationTest.TestSubs/0"
        "TestSubs/SubsOperationTest.TestSubs/1"
        "TestTopK/TopKOperationTest.TestTopK/0"
        "TestTopK/TopKOperationTest.TestTopK/4"
        "TestTopK/TopKOperationTest.TestTopK/6"
        "TestTranspose/TransposeOperationTest.TestTranspose/0"
        "TestTranspose/TransposeOperationTest.TestTranspose/2"
        "TestTranspose/TransposeOperationTest.TestTranspose/13"
        "TestTranspose/TransposeOperationTest.TestTranspose/15"
        "TestTranspose/TransposeOperationTest.TestTranspose/18"
        "TestVectorDuplicate/VectorDuplicateOperationTest.TestVectorDuplicate/0"
        "TestVectorDuplicate/VectorDuplicateOperationTest.TestVectorDuplicate/5"
        "TestVectorDuplicate/VectorDuplicateOperationTest.TestVectorDuplicate/9"
        "TestLogicalNot/LogicalNotOperationTest.TestLogicalNot/1"
        "TestLogicalNot/LogicalNotOperationTest.TestLogicalNot/4" 
)

set(_Target ${PTO_Fwk_STestNamePrefix}_operation)

file(GLOB_RECURSE _Sources src/*.cpp)

PTO_Fwk_STest_AddLib(
        TARGET ${_Target}
        SOURCES ${_Sources}
        PUBLIC_LINK_LIBRARIES tile_fwk_interface tile_fwk_simulation tile_fwk_simulation_ca
        GTEST_FILTER_LIST ${_CaseFilterList}
        LD_LIBRARIES_EXT tile_fwk_compiler tile_fwk_simulation tile_fwk_simulation_ca tile_fwk_simulation_pv
        GOLDEN_SCRIPT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python
)
add_custom_command(
        TARGET ${_Target} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PTO_Fwk_STestExePath}/scripts/
        COMMENT "Copy ${_Target} dependence scripts file."
)