# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================


########################################################################################################################
# 预处理
########################################################################################################################

# 预定义变量
set(TileFwkUTestNamePrefix tile_fwk_utest)
set(TileFwkSTestNamePrefix tile_fwk_stest)

set(TileFwkStestExecuteDeviceIdList)
if (NOT ENABLE_TESTS_EXECUTE_DEVICE_ID)
    set(TileFwkStestExecuteDeviceIdList 0)
else ()
    string(REPLACE ":" ";" TileFwkStestExecuteDeviceIdList "${ENABLE_TESTS_EXECUTE_DEVICE_ID}")
endif ()
list(GET TileFwkStestExecuteDeviceIdList 0 TileFwkStestExecuteDeviceIdPref)

# 预定义路径
get_filename_component(TileFwkUTestExePath "${CMAKE_CURRENT_BINARY_DIR}/ut" REALPATH)
get_filename_component(TileFwkSTestExePath "${CMAKE_CURRENT_BINARY_DIR}/st" REALPATH)

if (ENABLE_TESTS_STEST_GOLDEN_PATH)
    get_filename_component(ENABLE_TESTS_STEST_GOLDEN_PATH "${ENABLE_TESTS_STEST_GOLDEN_PATH}" REALPATH)
else ()
    get_filename_component(ENABLE_TESTS_STEST_GOLDEN_PATH "${TileFwkSTestExePath}/golden" REALPATH)
endif ()

# 环境变量 PATH
if ((NOT BUILD_OPEN_PROJECT) AND ENABLE_TESTS_UTEST)
    set(TILE_FWK_EXPORT_ENV_PATH "PATH=$ENV{PATH}:${CCEC_PATH}")
endif()


########################################################################################################################
# 三方库
########################################################################################################################

# GTest
if (BUILD_OPEN_PROJECT AND (ENABLE_TESTS_UTEST OR ENABLE_TESTS_STEST OR ENABLE_TESTS_STEST_DISTRIBUTED))
    find_package(GTest CONFIG)
    if (NOT ${GTest_FOUND})
        if (DEFINED ENV{ASCEND_3RD_LIB_PATH} AND NOT "${ASCEND_3RD_LIB_PATH}x" STREQUAL "x")
            get_filename_component(ASCEND_3RD_LIB_PATH "$ENV{ASCEND_3RD_LIB_PATH}" REALPATH)
            if (EXISTS "${ASCEND_3RD_LIB_PATH}/cmake/modules")
                list(APPEND CMAKE_MODULE_PATH ${ASCEND_3RD_LIB_PATH}/cmake/modules)
            endif ()
            if (EXISTS "${ASCEND_3RD_LIB_PATH}/gtest/lib/cmake/GTest")
                list(APPEND CMAKE_PREFIX_PATH ${ASCEND_3RD_LIB_PATH}/gtest/lib/cmake/GTest)
            endif ()
        endif ()
        find_package(GTest CONFIG)
    endif ()
    if (NOT ${GTest_FOUND})
        message(FATAL_ERROR "No GTest found, please refer to the ReadMe of this project for installation instructions.")
    endif ()
    message(STATUS "Use GTest from ${GTest_DIR}")
endif ()
