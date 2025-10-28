# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
cmake_minimum_required(VERSION 3.16)

project(PTO_Framework
        LANGUAGES C CXX
        VERSION 2.0
)

########################################################################################################################
# 基础配置
########################################################################################################################

# 路径配置
get_filename_component(PTO_FWK_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" REALPATH)
get_filename_component(PTO_FWK_BIN_ROOT "${CMAKE_CURRENT_BINARY_DIR}" REALPATH)

message(STATUS "PTO_FWK_SRC_ROOT=${PTO_FWK_SRC_ROOT}")
message(STATUS "PTO_FWK_BIN_ROOT=${PTO_FWK_BIN_ROOT}")

# 公开编译开关(build)
option(BUILD_OPEN_PROJECT "Build in open scenario." ON)
option(BUILD_WITH_CANN "Build with CANN." ON)  # Need to install CANN package and set corresponding environment variables.
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(ENABLE_GCOV "Enable GNU Coverage Instrumentation Tool" OFF)

# 公开编译开关(Feature 特性级开关)
option(ENABLE_FEATURE_PYTHON_FRONT_END "Enable Python Front-end" OFF)
option(ENABLE_TORCH_VERIFIER "Flow verify with libtorch" OFF)

# 公开编译开关(tests)
option(ENABLE_TESTS_EXECUTE "Enable auto execute tests executable" OFF)
option(ENABLE_TESTS_EXECUTE_PARALLEL "Auto execute tests executable parallel" OFF)
option(ENABLE_TESTS_EXECUTE_PARALLEL_TIMEOUT "Auto execute tests executable parallel timeout" OFF)
option(ENABLE_UTEST "Enable UTest" OFF)
option(ENABLE_STEST_GOLDEN_PATH "Specific STest Golden Path" OFF)
option(ENABLE_STEST_GOLDEN_PATH_CLEAN "Clean STest Golden" OFF)
option(ENABLE_STEST_EXECUTE_DEVICE_ID "Auto execute tests executable with Device Id" OFF)
option(ENABLE_STEST_DUMP_JSON "Enable STest Dump Json" OFF)
option(ENABLE_STEST_INTERPRETER_CONFIG "Enable STest Interpreter Config" OFF)
option(ENABLE_STEST_BINARY_CACHE "Enable STest Binary Cache" OFF)
option(ENABLE_STEST "Enable STest" OFF)
option(ENABLE_STEST_DISTRIBUTED "Enable STest(Distributed)" OFF)

# 公开编译开关(stest.tools)
option(ENABLE_STEST_TOOLS_CASE_FILE "Specify STest Tools Case file path" OFF)
option(ENABLE_STEST_TOOLS_INTERCEPT "Enable STest Tools Intercept if has any failed case" OFF)
option(ENABLE_STEST_TOOLS_OUTPUT_CLEAN "Clean STest Tools Output Directory" OFF)
option(ENABLE_STEST_TOOLS_PROF "Enable STest Tools(Profiling)" OFF)
option(ENABLE_STEST_TOOLS_PROF_LEVEL "Specify STest Tools Profiling Level" OFF)
option(ENABLE_STEST_TOOLS_PROF_WARN_UP_CNT "Specify STest Tools Profiling Warn-up cnt" OFF)
option(ENABLE_STEST_TOOLS_PROF_TRY_CNT "Specify STest Tools Profiling Try cnt" OFF)
option(ENABLE_STEST_TOOLS_PROF_MAX_CNT "Specify STest Tools Profiling Max cnt" OFF)

# 兼容性
if (DEFINED AC_ENABLE_FRAMEWORK_WITHOUT_CANN)
    if (AC_ENABLE_FRAMEWORK_WITHOUT_CANN)
        set(BUILD_WITH_CANN OFF)
    else ()
        set(BUILD_WITH_CANN ON)
    endif ()
endif ()
if (DEFINED ENABLE_TESTS_EXECUTE_DEVICE_ID)
    set(ENABLE_STEST_EXECUTE_DEVICE_ID ${ENABLE_TESTS_EXECUTE_DEVICE_ID})
endif ()
if (DEFINED ENABLE_TESTS_UTEST)
    set(ENABLE_UTEST ${ENABLE_TESTS_UTEST})
endif ()
if (DEFINED ENABLE_TESTS_STEST)
    set(ENABLE_STEST ${ENABLE_TESTS_STEST})
endif ()
if (DEFINED ENABLE_TESTS_STEST_DISTRIBUTED)
    set(ENABLE_STEST_DISTRIBUTED ${ENABLE_TESTS_STEST_DISTRIBUTED})
endif ()
if (DEFINED ENABLE_TESTS_STEST_GOLDEN_PATH)
    set(ENABLE_STEST_GOLDEN_PATH ${ENABLE_TESTS_STEST_GOLDEN_PATH})
endif ()
if (DEFINED ENABLE_TESTS_STEST_GOLDEN_PATH_CLEAN)
    set(ENABLE_STEST_GOLDEN_PATH_CLEAN ${ENABLE_TESTS_STEST_GOLDEN_PATH_CLEAN})
endif ()
if (DEFINED ENABLE_TESTS_STEST_BINARY_CACHE)
    set(ENABLE_STEST_BINARY_CACHE ${ENABLE_TESTS_STEST_BINARY_CACHE})
endif ()
if (DEFINED ENABLE_TESTS_STEST_DUMP_JSON)
    set(ENABLE_STEST_DUMP_JSON ${ENABLE_TESTS_STEST_DUMP_JSON})
endif ()
if (DEFINED ENABLE_TESTS_STEST_INTERPRETER_CONFIG)
    set(ENABLE_STEST_INTERPRETER_CONFIG ${ENABLE_TESTS_STEST_INTERPRETER_CONFIG})
endif ()
if (DEFINED ENABLE_TESTS_STEST_TOOLS_OUTPUT_CLEAN)
    set(ENABLE_STEST_TOOLS_OUTPUT_CLEAN ${ENABLE_TESTS_STEST_TOOLS_OUTPUT_CLEAN})
endif ()
if (DEFINED ENABLE_TESTS_STEST_TOOLS_INTERCEPT)
    set(ENABLE_STEST_TOOLS_INTERCEPT ${ENABLE_TESTS_STEST_TOOLS_INTERCEPT})
endif ()
if (DEFINED ENABLE_TESTS_STEST_TOOLS_CASE_FILE)
    set(ENABLE_STEST_TOOLS_CASE_FILE ${ENABLE_TESTS_STEST_TOOLS_CASE_FILE})
endif ()
if (DEFINED ENABLE_TESTS_STEST_TOOLS_PROF)
    set(ENABLE_STEST_TOOLS_PROF ${ENABLE_TESTS_STEST_TOOLS_PROF})
endif ()
if (DEFINED ENABLE_TESTS_STEST_TOOLS_PROF_LEVEL)
    set(ENABLE_STEST_TOOLS_PROF_LEVEL ${ENABLE_TESTS_STEST_TOOLS_PROF_LEVEL})
endif ()
if (DEFINED ENABLE_TESTS_STEST_TOOLS_PROF_WARN_UP_CNT)
    set(ENABLE_STEST_TOOLS_PROF_WARN_UP_CNT ${ENABLE_TESTS_STEST_TOOLS_PROF_WARN_UP_CNT})
endif ()
if (DEFINED ENABLE_TESTS_STEST_TOOLS_PROF_TRY_CNT)
    set(ENABLE_STEST_TOOLS_PROF_TRY_CNT ${ENABLE_TESTS_STEST_TOOLS_PROF_TRY_CNT})
endif ()
if (DEFINED ENABLE_TESTS_STEST_TOOLS_PROF_MAX_CNT)
    set(ENABLE_STEST_TOOLS_PROF_MAX_CNT ${ENABLE_TESTS_STEST_TOOLS_PROF_MAX_CNT})
endif ()

# 缺省编译配置, 编译函数
include(ExternalProject)
include(cmake/func.cmake)
include(cmake/config.cmake)
include(cmake/intf.cmake)


########################################################################################################################
# 编译流程
########################################################################################################################

add_subdirectory(src)
add_subdirectory(python)

if (ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()

if (ENABLE_FEATURE_PYTHON_FRONT_END)
    install(DIRECTORY include/tileop/ DESTINATION pto/lib/include/tileop OPTIONAL)

    install(FILES include/tilefwk/aicpu_common.h        DESTINATION pto/lib/include/tilefwk)
    install(FILES include/tilefwk/aicpu_distributed.h   DESTINATION pto/lib/include/tilefwk)
    install(FILES include/tilefwk/aicpu_runtime.h       DESTINATION pto/lib/include/tilefwk)

    install(FILES include/tilefwk/aicore_data.h         DESTINATION pto/lib/include/tilefwk)
    install(FILES include/tilefwk/aicore_print.h        DESTINATION pto/lib/include/tilefwk)
    install(FILES include/tilefwk/aicore_runtime.h      DESTINATION pto/lib/include/tilefwk)
    install(FILES include/tilefwk/core_func_data.h      DESTINATION pto/lib/include/tilefwk)
else ()
    install(DIRECTORY include/                          DESTINATION lib/tilefwk/ OPTIONAL)
endif ()
