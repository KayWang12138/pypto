# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
cmake_minimum_required(VERSION 3.16)

project(TileFramework
        LANGUAGES C CXX
        VERSION 2.0
)

########################################################################################################################
# 基础配置
########################################################################################################################

# 路径配置
get_filename_component(TILE_FWK_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" REALPATH)
get_filename_component(TILE_FWK_BIN_ROOT "${CMAKE_CURRENT_BINARY_DIR}" REALPATH)
# 公开编译开关(common)
option(BUILD_OPEN_PROJECT   "Build open Tile Framework project."  ON)
if (DEFINED AC_ENABLE_FRAMEWORK_WITHOUT_CANN)
    if (AC_ENABLE_FRAMEWORK_WITHOUT_CANN)
        option(ENABLE_BUILD_WITH_CANN "Enable build with CANN." OFF)
    else ()
        option(ENABLE_BUILD_WITH_CANN "Enable build with CANN." ON)
    endif ()
else ()
    option(ENABLE_BUILD_WITH_CANN "Enable build with CANN." ON)
endif ()
# 公开编译开关(tests)
option(ENABLE_TESTS_EXECUTE "Enable auto execute tests executable"  OFF)
option(ENABLE_TESTS_EXECUTE_DEVICE_ID "Auto execute tests executable with Device Id" OFF)
option(ENABLE_TESTS_EXECUTE_PARALLEL "Auto execute tests executable parallel" OFF)
option(ENABLE_TESTS_EXECUTE_PARALLEL_TIMEOUT "Auto execute tests executable parallel timeout" OFF)
option(ENABLE_TESTS_UTEST "Enable UTest"  OFF)
option(ENABLE_TESTS_STEST "Enable STest"  OFF)
option(ENABLE_TESTS_STEST_DISTRIBUTED "Enable STest(Distributed)"  OFF)
option(ENABLE_TESTS_STEST_GOLDEN_PATH "Specific STest Golden Path"  OFF)
option(ENABLE_TESTS_STEST_GOLDEN_PATH_CLEAN "Clean STest Golden"  OFF)
option(ENABLE_TESTS_STEST_BINARY_CACHE "Enable STest Binary Cache" OFF)
option(ENABLE_TESTS_STEST_DUMP_JSON "Enable STest Dump Json"  OFF)
# 公开编译开关(tools_build)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(ENABLE_GCOV "Enable GNU Coverage Instrumentation Tool" OFF)
option(ENABLE_TESTS_STEST_EXPERIMENT_COPY_AICPU_BINARY "Enable Copy AICPU binary to device auto(Experiment)" OFF)
# 公开编译开关(tools_profiling/tools_precision)
option(ENABLE_TESTS_STEST_TOOLS_OUTPUT_CLEAN "Clean STest Tools Output Directory" OFF)
option(ENABLE_TESTS_STEST_TOOLS_INTERCEPT "Enable STest Tools Intercept if has any failed case" OFF)
option(ENABLE_TESTS_STEST_TOOLS_CASE_FILE "Specify STest Tools Case file path" OFF)
option(ENABLE_TESTS_STEST_TOOLS_PROF "Enable STest Tools(Profiling)" OFF)
option(ENABLE_TESTS_STEST_TOOLS_PROF_LEVEL "Specify STest Tools Profiling Level" OFF)
option(ENABLE_TESTS_STEST_TOOLS_PROF_WARN_UP_CNT "Specify STest Tools Profiling Warn-up cnt" OFF)
option(ENABLE_TESTS_STEST_TOOLS_PROF_TRY_CNT "Specify STest Tools Profiling Try cnt" OFF)
option(ENABLE_TESTS_STEST_TOOLS_PROF_MAX_CNT "Specify STest Tools Profiling Max cnt" OFF)

# 缺省编译配置, 编译函数
include(ExternalProject)
include(cmake/config.cmake)
include(cmake/intf.cmake)
include(cmake/func.cmake)

########################################################################################################################
# 编译流程
########################################################################################################################

add_subdirectory(src)

if (ENABLE_TESTS_UTEST OR ENABLE_TESTS_STEST OR ENABLE_TESTS_STEST_DISTRIBUTED)
    enable_testing()
    add_subdirectory(tests)
endif ()

install(DIRECTORY include/
    DESTINATION lib/tilefwk/ OPTIONAL
)
