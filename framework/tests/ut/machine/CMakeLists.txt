# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set(_CaseFilterList
        # Dynamic Function
        "spscqueue.normal"
        "TestDynamicDeviceRunner.*"
        # Machine
        "HostMachineCompileTest.test_MlaProlog_float16_32_2_1_256_256_512"
        "HostMachineCompileTest.test_codegen_by_json"
        "DeviceMachineTest.test_get_task_time"
        "TestDeviceRunner.test_set_pmu_event"
        "TestDeviceRunner.test_ini_device_runner"
        # AstApi
        "TestAstApi.test_ast"
        "TestAstOpCompile.test_compile_add_op"
        "TestAstOpCompile.test_dynamic_ViewAssemble"
        "TestAstOpCompile.test_compile_fatbin"
        # Others
        "RuntimeTest.Runtime01"
        "TestDeviceRunner.test_device_runner_get_task_time"
        "OnBoardTestAstInGraph.test_fa_all2all_128"
        "OnBoardTestAstInGraph.test_fa_all2all_128_2"
        "OnBoardPaCostTest.test_page_attention_low_latency"
        "OnBoardPaCostTest.test_page_attention_hight_throughput"
        # device UT
        "TestPro.test_ini"
        "TestAicoreDump.test_aicore_dump_special"
        "TestAicoreDump.test_aicore_dump_normal"
        # binary cache
        "CacheManagerUnitTest.test_*"
        # encode
        "DevEncode.*"
        # distributed
        "CommWaitFlagTest.CommWaitFlag_EnqueueOp_and_PollCompleted_success"
        "ShmemWaitUntilTest.*"
        "SlabWsAllocatorTest.*"
        # Platform
        "PlatformTest.TestPlatfromCase1"
        # 框架相关用例
        "TestMemoryAttribute.MemorySizeTest"
        #"TestIncastOverlap.AddSubExp_12_8"
        #"SerializationTest.TestAddSub"
        "TestSuite_Backend.AihacBackend_Err1"
        "TestSuite_Backend.SimulationBackend_Err1"
        "DynamicMatmulUTest.mm_A_B_ND_bf16"
)

set(_Target ${PTO_Fwk_UTestNamePrefix}_machine)

file(GLOB_RECURSE _Sources src/*.cpp)
list(APPEND _Sources ${PTO_FWK_SRC_ROOT}/framework/src/machine/device/aicore_prof.cpp)
list(APPEND _Sources ${PTO_FWK_SRC_ROOT}/framework/src/machine/device/device_machine.cpp)
list(APPEND _Sources ${PTO_FWK_SRC_ROOT}/framework/src/machine/device/device_log.cpp)
list(APPEND _Sources ${PTO_FWK_SRC_ROOT}/framework/src/machine/device/distributed/comm_wait_flag.cpp)
list(APPEND _Sources ${PTO_FWK_SRC_ROOT}/framework/src/machine/device/distributed/shmem_wait_until.cpp)

PTO_Fwk_UTest_AddCaseLib(
        TARGET ${_Target}
        SOURCES ${_Sources}
        PUBLIC_LINK_LIBRARIES tile_fwk_compiler tile_fwk_runtime tile_fwk_interface dl
        GTEST_FILTER_LIST ${_CaseFilterList}
)
target_compile_definitions(${_Target}
        PRIVATE
            NO_NEON_SUPPORT
)
# 可执行程序运行依赖的配置, 脚本配置
add_custom_command(
        TARGET ${_Target} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PTO_Fwk_UTestExePath}/stubs/compiler/data/platform_config/
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/stubs/compiler/data/platform_config/Ascend910B1.ini
                                         ${PTO_Fwk_UTestExePath}/stubs/compiler/data/platform_config/
        COMMENT "Copy ${_Target} dependence config file."
)
