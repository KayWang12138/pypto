# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

cmake_minimum_required(VERSION 3.16)

project(tile_fwk_server LANGUAGES C CXX)


########################################################################################################################
# 基础配置
########################################################################################################################
if(BUILD_OPEN_PROJECT)
    add_subdirectory(stub)
endif()

# 路径配置
get_filename_component(TILE_FWK_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

# 公开编译开关
option(BUILD_OPEN_PROJECT   "Build open Tile Framework project."  ON)

# 缺省编译配置, 编译函数
set(CMAKE_SKIP_RPATH TRUE)

########################################################################################################################
# 编译流程
########################################################################################################################

add_library(tile_fwk_server SHARED)
target_sources(tile_fwk_server
        PRIVATE
            device_machine.cpp
            aicore_prof.cpp
            distributed/comm_wait_flag.cpp
            dynamic/device_machine.cpp
            dynamic/aicore_prof.cpp
            device_log.cpp
)
target_compile_options(tile_fwk_server
        PRIVATE
            -Wall
            -Wextra
            -std=gnu++17
            -rdynamic
            -O3
            -shared
            -fPIC
            -D__DEVICE__
            -Werror
)
target_include_directories(tile_fwk_server
        PRIVATE
            ${TILE_FWK_SRC_ROOT}/src
            ${TILE_FWK_SRC_ROOT}/src/include
            ${TILE_FWK_SRC_ROOT}/include
            ${TILE_FWK_SRC_ROOT}/src/cost_model
            $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include>
)
target_link_directories(tile_fwk_server
        PRIVATE
            $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib/linux/aarch64/>
            $<$<BOOL:${ENABLE_BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/lib64>
)
target_link_libraries(tile_fwk_server
        PRIVATE
            -Wl,--no-as-needed
            c_sec
            $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,slog_stub,slog>
            -ldl
)
set_target_properties(tile_fwk_server PROPERTIES OUTPUT_NAME tilefwk_backend_server)


install(TARGETS tile_fwk_server
        LIBRARY DESTINATION lib OPTIONAL
)

