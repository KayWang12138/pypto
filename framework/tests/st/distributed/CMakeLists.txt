# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_subdirectory(ops)

add_subdirectory(framework)

# 用例可执行程序(编译)
set(_Target ${PTO_Fwk_STestNamePrefix}_distributed)
file(GLOB_RECURSE _Sources src/*.cpp)
set(_PrivateLinkLibraries
        tile_fwk_utils
        tile_fwk_interface
        tile_fwk_codegen
        tile_fwk_compiler
        tile_fwk_runtime
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

set(TEST_CASE_SRC_DIR "${PTO_FWK_SRC_ROOT}/framework/tests/st/distributed/ops/test_case")
set(TEST_CASE_DST_DIR "$<TARGET_FILE_DIR:${_Target}>/test_case")

add_custom_command(TARGET ${_Target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${TEST_CASE_SRC_DIR}"
        "${TEST_CASE_DST_DIR}"
    COMMENT "Copying test case data files to output directory..."
    VERBATIM
)
target_compile_definitions(${_Target} PRIVATE
    TEST_CASE_RELATIVE_PATH="test_case"
    TEST_CASE_EXE_DIR="${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

# PyPTO, 依赖 CI 任务拆分, 先通过依赖确保上板顺序
if (TARGET ${PTO_Fwk_STestNamePrefix}_python)
    message(STATUS "Add Dependencies ${_Target} -> ${PTO_Fwk_STestNamePrefix}_python")
    add_dependencies(${_Target} ${PTO_Fwk_STestNamePrefix}_python)
endif ()

# 用例可执行程序(执行)
#
# Test case index map (alphabetical JSON file order, 0-based):
#   0  Allgather_AttnPost_Reducescatter  (1 case)
#   1  Allgather                         (1 case)
#   2  Allreduce_Add_Allreduce           (1 case)
#   3  Allreduce  test_1  (twoshot)      ─┐
#   4  Allreduce  test_2  (oneshot)      ─┘ 2 cases
#   5  AllreduceIRCheck                  (1 case)  ── oneshot
#   6  AllreduceV2                       (1 case)  ── oneshot
#   7  AllreduceV3                       (1 case)  ── oneshot
#   8  AllreduceV4                       (1 case)  ── oneshot
#   9  AllreduceV5MultiRank  test_1      (2 ranks) ── oneshot multirank
#  10  AllreduceV5MultiRank  test_2      (8 ranks) ── oneshot multirank
#  11  AllreduceV5                       (1 case)  ── oneshot
#  12  MoeDispatch                       (1 case)
#  13  MoeDistributedCombine  test_1     ─┐
#  14  MoeDistributedCombine  test_2     ─┘ 2 cases
#  15  Reducescatter  test_1             ─┐
#  16  Reducescatter  test_2              │
#  17  Reducescatter  test_3             ─┘ 3 cases
#  18  TwoShotAllReduceIRCheck           (1 case)  ── twoshot
#  19  TwoShotAllReduceV2               (1 case)  ── twoshot
#  20  TwoShotAllReduceV3               (1 case)  ── twoshot
#  21  TwoShotAllReduceV4               (1 case)  ── twoshot
#  22  TwoShotAllReduceV5               (1 case)  ── twoshot
#
# Usage:
#   python3 build_ci.py --stest_distributed              # all tests (default)
#   python3 build_ci.py --stest_distributed=oneshot       # OneShot AllReduce variants only
#   python3 build_ci.py --stest_distributed=twoshot       # TwoShot AllReduce variants only
#   python3 build_ci.py --stest_distributed=multirank     # Multi-rank AllReduceV5 only (2 & 8 ranks)
#
set(_num_ranks "4")
set(_num_cases "22")

# Per-index rank-count overrides for multi-rank tests
set(_rank_override_9 "2")
set(_rank_override_10 "8")

# Build filter config for a list of indices, respecting per-index rank overrides
macro(_distributed_filter_config _output)
    set(${_output} "")
    foreach(_i ${ARGN})
        set(_r "${_num_ranks}")
        if (DEFINED _rank_override_${_i})
            set(_r "${_rank_override_${_i}}")
        endif()
        list(APPEND ${_output}
            "TestDistributedOps/DistributedTest.TestOps/${_i}" "${_r}"
        )
    endforeach()
endmacro()

# Default: all test cases
set(_all_indices "")
foreach(i RANGE 0 ${_num_cases})
    list(APPEND _all_indices ${i})
endforeach()
_distributed_filter_config(_DistributedGTestCaseFilterConfig ${_all_indices})

# Subset filtering: override filter config when a subset keyword is requested
if ("${ENABLE_STEST_DISTRIBUTED}" STREQUAL "oneshot")
    message(STATUS "Distributed subset: oneshot AllReduce variants (indices 5-11)")
    _distributed_filter_config(_DistributedGTestCaseFilterConfig 5 6 7 8 9 10 11)
    set(ENABLE_STEST_DISTRIBUTED "ON")
elseif ("${ENABLE_STEST_DISTRIBUTED}" STREQUAL "twoshot")
    message(STATUS "Distributed subset: twoshot AllReduce variants (indices 18-22)")
    _distributed_filter_config(_DistributedGTestCaseFilterConfig 18 19 20 21 22)
    set(ENABLE_STEST_DISTRIBUTED "ON")
elseif ("${ENABLE_STEST_DISTRIBUTED}" STREQUAL "multirank")
    message(STATUS "Distributed subset: multi-rank AllReduceV5 (indices 9-10)")
    _distributed_filter_config(_DistributedGTestCaseFilterConfig 9 10)
    set(ENABLE_STEST_DISTRIBUTED "ON")
endif()

set(_LdLibrariesExt
        tile_fwk_compiler
        ${PTO_Fwk_STestNamePrefix}_distributed_framework
)
set(_EnvSetupExt
        "TILE_FWK_STEST_GOLDEN_PATH=${ENABLE_STEST_GOLDEN_PATH}"
)
PTO_Fwk_STest_Distributed_RunExe(
       TARGET                  ${_Target}
       LD_LIBRARIES_EXT        ${_LdLibrariesExt}
       ENV_LINES_EXT           ${_EnvSetupExt}
       GTEST_FILTER_CONFIG     ${_DistributedGTestCaseFilterConfig}
       GOLDEN_SCRIPT_DIR       ${CMAKE_CURRENT_SOURCE_DIR}/ops/script
)
