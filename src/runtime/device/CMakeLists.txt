# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

cmake_minimum_required(VERSION 3.16)

project(ascendcpp_server LANGUAGES C CXX)


########################################################################################################################
# 基础配置
########################################################################################################################

# 路径配置
get_filename_component(ASCENDCPP_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

# 公开编译开关
option(BUILD_OPEN_PROJECT   "Build open Ascend C++ project."  ON)

# 缺省编译配置, 编译函数
set(CMAKE_SKIP_RPATH TRUE)

########################################################################################################################
# 编译流程
########################################################################################################################

add_library(ascendcpp_server SHARED)
target_sources(ascendcpp_server
        PRIVATE
            ${ASCENDCPP_SRC_ROOT}/src/stubs/stubs.cpp
)
target_compile_options(ascendcpp_server
        PRIVATE
            -Wall
            -Wextra
            -std=gnu++17
            -rdynamic
            -g
            -O3
            -shared
            -fPIC
            -D__DEVICE__
            -Werror
)
target_include_directories(ascendcpp_server
        PRIVATE
            ${ASCENDCPP_SRC_ROOT}/src
            ${ASCENDCPP_SRC_ROOT}/include
            ${ASCENDCPP_SRC_ROOT}/src/stubs
            $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include>
)
target_link_directories(ascendcpp_server
        PRIVATE
            $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib/linux/aarch64/>
)
target_link_libraries(ascendcpp_server
        PRIVATE
            c_sec
            -ldl
)
set_target_properties(ascendcpp_server PROPERTIES OUTPUT_NAME ASTServer)


install(TARGETS ascendcpp_server
        LIBRARY DESTINATION lib OPTIONAL
)
