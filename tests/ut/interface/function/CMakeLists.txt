# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set(_CaseFilterList
        # Tensor
        "TestTensor.*"
        # Function
        "SubFunctionTest.*"
        # "FunctionUtilsTest.TestRemoveOperationCase1"
        "FunctionCoverageTest.*"
        # Dynamic Function
        "FunctionWithPass.*"
        # Platform
        "PlatformTest.TestPlatfromCase1"
        "SubFunctionTest.*"
        #workflow verify tools
        "DynamicOpsTest.*"
        # Dynamic Function
        "DynamicFunctionTest.TestGetInputDataInt32Dim3"
        "DynamicFunctionTest.TestSymbolic"
        "DynamicFunctionTest.TestLoopRange"
        "DynamicFunctionTest.TestOnlyExpression"
        "DynamicFunctionTest.TestOnlySymbol"
        "DynamicFunctionTest.TestStaticLoopStatic"
        "DynamicFunctionTest.TestHybridLoopIf"
        "DynamicFunctionTest.TestLoopWithRank"
        "DynamicFunctionTest.TestLoopIfWithRank"
        "DynamicFunctionTest.TestLoopWithManualRank"
        "DynamicFunctionTest.TestSymbolicScalarDumpLoad"
        "DynamicFunctionTest.TestDynOffset"
        "DynamicFunctionTest.MopCall"
        "ExpectedValueTest.TestCheck"
        "DynamicPATest.TestDD"
        "DynamicPATest.dynamic_pa_low_lantency_unroll"
        "DynamicPATest.dynamic_pa_low_lantency_manual_unroll"
        "DynamicPATest.dynamic_pa_high_throughput_only_batch_loop"
        "DynamicAttentionUtTest.dynamic_attention_low"
        "DynamicAttentionUtTest.dynamic_attention_low_nz"
        "DynamicPATest.AllocatorCoverage"
        "DynamicKvSlcUtTest.test_kv_slc"
        "TensorElement.*"
        "ThreadPoolTest.*"
        "StringUtils.*"
        "DynamicTestGenAtten.TestOnboardGenAttenTest_FP16"
)

set(_Target ${TileFwkUTestNamePrefix}_interface_function)

file(GLOB_RECURSE _Sources src/*.cpp)

TileFwk_UTest_AddCaseLib(
        TARGET ${_Target}
        SOURCES ${_Sources}
        PUBLIC_LINK_LIBRARIES tile_fwk_interface
        GTEST_FILTER_LIST ${_CaseFilterList}
)
# 可执行程序运行依赖的配置, 脚本配置
add_custom_command(
        TARGET ${_Target} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${TileFwkUTestExePath}/stubs/compiler/data/platform_config/
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/stubs/compiler/data/platform_config/Ascend910B1.ini
                                         ${TileFwkUTestExePath}/stubs/compiler/data/platform_config/
        COMMENT "Copy ${_Target} dependence config file."
)
