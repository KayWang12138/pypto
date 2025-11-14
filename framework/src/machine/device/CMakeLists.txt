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

get_filename_component(PTO_FWK_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

# 缺省编译配置, 编译函数
set(CMAKE_SKIP_RPATH TRUE)
if (BUILD_WITH_CANN)
    set(CMAKE_C_COMPILER "${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-gcc")
    set(CMAKE_CXX_COMPILER "${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++")
endif ()

message(STATUS "CMAKE_GENERATOR=${CMAKE_GENERATOR}")
message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

########################################################################################################################
# 编译流程
########################################################################################################################

file(GLOB_RECURSE _Device_Srcs *.cpp)
file(GLOB_RECURSE _Stub_Srcs stub/*.cpp)
list(REMOVE_ITEM _Device_Srcs ${_Stub_Srcs})

add_library(tile_fwk_server SHARED)
target_sources(tile_fwk_server PRIVATE ${_Device_Srcs})
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
            -fno-common
)
target_include_directories(tile_fwk_server
        PRIVATE
            ${PTO_FWK_SRC_ROOT}/src
            ${PTO_FWK_SRC_ROOT}/src/interface
            ${PTO_FWK_SRC_ROOT}/src/include
            ${PTO_FWK_SRC_ROOT}/include
            ${PTO_FWK_SRC_ROOT}/src/cost_model
            $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include>
)
target_link_directories(tile_fwk_server
        PRIVATE
            $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib/linux/aarch64/>
            $<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/lib64>
)
target_link_libraries(tile_fwk_server
        PRIVATE
            -Wl,--no-as-needed
            c_sec
            $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,slog_stub,slog>
            -ldl
            $<$<TARGET_EXISTS:intf_pub>:$<BUILD_INTERFACE:intf_pub>>
)
set_target_properties(tile_fwk_server PROPERTIES OUTPUT_NAME tilefwk_backend_server)

if (ENABLE_FEATURE_PYTHON_FRONT_END)
else ()
    install(TARGETS tile_fwk_server
            LIBRARY DESTINATION lib OPTIONAL
    )
endif ()
