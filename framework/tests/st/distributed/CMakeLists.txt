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
# Test case index map (case-insensitive alphabetical JSON file order, 0-based):
#   0  Allgather_AttnPost_Reducescatter  (1 case)
#   1  Allgather                         (1 case)
#   2  Allreduce_Add_Allreduce           (1 case)
#   3  Allreduce  test_1  (twoshot)      ─┐
#   4  Allreduce  test_2  (oneshot)      ─┘ 2 cases
#   5  AllreduceV10                      (1 case)  ── oneshot, per-group GE signaling
#   6  AllreduceV10PipeGe                (1 case)  ── oneshot, pipelined GE signaling
#   7  MoeDispatch                       (1 case)
#   8  MoeDistributedCombine  test_1    ─┐
#   9  MoeDistributedCombine  test_2    ─┘ 2 cases
#  10  Reducescatter  test_1             ─┐
#  11  Reducescatter  test_2              │
#  12  Reducescatter  test_3             ─┘ 3 cases
#  13  TwoShotAllReduceIRCheck           (1 case)  ── twoshot
#  14  TwoShotAllReduceV2                (1 case)  ── twoshot
#  15  TwoShotAllReduceV3                (1 case)  ── twoshot
#  16  TwoShotAllReduceV4                (1 case)  ── twoshot
#  17  TwoShotAllReduceV5                (1 case)  ── twoshot
#
# Note: sort order is case-insensitive (see GetOpMetaDataFromDir in test_distributed.h).
# 'AllreduceV10' sorts before 'AllreduceV10PipeGe' because '1' < 'P'.
#
# Usage:
#   python3 build_ci.py --stest_distributed              # all tests (default)
#   python3 build_ci.py --stest_distributed=oneshot       # OneShot AllReduce variants only
#   python3 build_ci.py --stest_distributed=twoshot       # TwoShot AllReduce variants only
#   python3 build_ci.py --stest_distributed=ircheck       # TwoShot IR equivalence check
#   python3 build_ci.py --stest_distributed=twoshotir     # TwoShot IR equivalence check only
#   python3 build_ci.py --stest_distributed=v10           # AllReduceV10 only
#   python3 build_ci.py --stest_distributed=v10pipege     # AllReduceV10PipeGe only
#
set(_num_ranks "4")
set(_num_cases "17")

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
    message(STATUS "Distributed subset: oneshot AllReduce variants (indices 5-6)")
    _distributed_filter_config(_DistributedGTestCaseFilterConfig 5 6)
    set(ENABLE_STEST_DISTRIBUTED "ON")
elseif ("${ENABLE_STEST_DISTRIBUTED}" STREQUAL "twoshot")
    message(STATUS "Distributed subset: twoshot AllReduce variants (indices 13-17)")
    _distributed_filter_config(_DistributedGTestCaseFilterConfig 13 14 15 16 17)
    set(ENABLE_STEST_DISTRIBUTED "ON")
elseif ("${ENABLE_STEST_DISTRIBUTED}" STREQUAL "ircheck")
    message(STATUS "Distributed subset: TwoShot IR equivalence check (index 13)")
    _distributed_filter_config(_DistributedGTestCaseFilterConfig 13)
    set(ENABLE_STEST_DISTRIBUTED "ON")
elseif ("${ENABLE_STEST_DISTRIBUTED}" STREQUAL "twoshotir")
    message(STATUS "Distributed subset: TwoShot IR equivalence check (index 13)")
    _distributed_filter_config(_DistributedGTestCaseFilterConfig 13)
    set(ENABLE_STEST_DISTRIBUTED "ON")
elseif ("${ENABLE_STEST_DISTRIBUTED}" STREQUAL "v10")
    message(STATUS "Distributed subset: AllReduceV10 only (index 5)")
    _distributed_filter_config(_DistributedGTestCaseFilterConfig 5)
    set(ENABLE_STEST_DISTRIBUTED "ON")
elseif ("${ENABLE_STEST_DISTRIBUTED}" STREQUAL "v10pipege")
    message(STATUS "Distributed subset: AllReduceV10PipeGe only (index 6)")
    _distributed_filter_config(_DistributedGTestCaseFilterConfig 6)
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
