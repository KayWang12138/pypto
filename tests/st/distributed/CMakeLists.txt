# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_subdirectory(ops)

add_subdirectory(framework)

# 用例可执行程序(编译)
set(_Target ${PTO_Fwk_STestNamePrefix}_distributed)
file(GLOB_RECURSE _Sources src/*.cpp)
set(_PrivateLinkLibraries
        tile_fwk_compiler
        tile_fwk_interface
        tile_fwk_simulation
        ${PTO_Fwk_STestNamePrefix}_utils
        ${PTO_Fwk_STestNamePrefix}_distributed_ops
        ${PTO_Fwk_STestNamePrefix}_distributed_framework
)
PTO_Fwk_GTest_AddExe(
        TARGET                  ${_Target}
        SOURCES                 ${_Sources}
        PRIVATE_LINK_LIBRARIES  ${_PrivateLinkLibraries}
)

# PyPTO, 依赖 CI 任务拆分, 先通过依赖确保上板顺序
if (TARGET ${PTO_Fwk_STestNamePrefix}_python)
    message(STATUS "Add Dependencies ${_Target} -> ${PTO_Fwk_STestNamePrefix}_python")
    add_dependencies(${_Target} ${PTO_Fwk_STestNamePrefix}_python)
endif ()

# 用例可执行程序(执行)
# aicpuWaitFlag_single_test_reduce_scatter_int32_32_32_4 "4"
# aicpuWaitFlag和aiv两方式_单Tensor和多Tensor_算子名称_数据类型_row_col_卡数 卡数
set(_DistributedGTestCaseFilterConfig
        "DistributedTest.aicpuWaitFlag_single_test_reduce_scatter_int32_32_32_4" "4"
        "DistributedTest.aicpuWaitFlag_single_test_all_gather_bfloat16_256_256_4" "4"
        "DistributedTest.aicpuWaitFlag_multi_test_reduce_scatter_float32_128_256_4" "4"
        "DistributedTest.aicpuWaitFlag_multi_test_all_gather_float16_32_32_4" "4"
        "DistributedTest.aivWaitFlag_single_test_reduce_scatter_int32_128_256_4" "4"
        "DistributedTest.aivWaitFlag_single_test_all_gather_bfloat16_256_128_4" "4"
        "DistributedTest.aivWaitFlag_multi_test_reduce_scatter_float32_128_256_4" "4"
        "DistributedTest.aivWaitFlag_multi_test_all_gather_float16_32_32_4" "4"
        "DistributedTest.aivWaitFlag_single_test_moe_dispatch_bfloat16_rank_size_4" "4"
        "DistributedTest.aivWaitFlag_single_test_moe_combine_bfloat16_rank_size_4" "4"
        "DistributedTest.allgather_attn_post_reducescatter_b64_s1_n32_lora256_dim128_h128_rank4_bf16" "4"
        "DistributedTest.shmem_all_gather_int32_128_256_4" "4"
        "DistributedTest.shmem_reduce_scatter_int32_128_256_4" "4"
        "DistributedTest.shmem_allgather_matmul_reducescatter_int32_128_256_4" "4"
)
set(_LdLibrariesExt
        tile_fwk_compiler
        ${PTO_Fwk_STestNamePrefix}_distributed_framework
)
set(_EnvSetupExt
        "TILE_FWK_STEST_GOLDEN_PATH=${ENABLE_TESTS_STEST_GOLDEN_PATH}"
)
PTO_Fwk_STest_Distributed_RunExe(
       TARGET                  ${_Target}
       LD_LIBRARIES_EXT        ${_LdLibrariesExt}
       ENV_LINES_EXT           ${_EnvSetupExt}
       GTEST_FILTER_CONFIG     ${_DistributedGTestCaseFilterConfig}
       GOLDEN_SCRIPT_DIR       ${CMAKE_CURRENT_SOURCE_DIR}/ops/script
)
