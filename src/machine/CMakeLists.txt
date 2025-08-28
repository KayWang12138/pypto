# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================


if (ENABLE_BUILD_WITH_CANN)
    add_subdirectory(kernel)
endif ()


if (ENABLE_BUILD_WITH_CANN)
    ExternalProject_Add(tile_fwk_server
            SOURCE_DIR ${TILE_FWK_SRC_ROOT}/src/machine/device
            CONFIGURE_COMMAND ${CMAKE_COMMAND}
                -G ${CMAKE_GENERATOR}
                -DBUILD_OPEN_PROJECT=${BUILD_OPEN_PROJECT}
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -DASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}
                -DCMAKE_C_COMPILER=${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-gcc
                -DCMAKE_CXX_COMPILER=${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++
                -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
                -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                <SOURCE_DIR>
            INSTALL_COMMAND ${CMAKE_COMMAND} -E copy <BINARY_DIR>/libtilefwk_backend_server.so ${CMAKE_BINARY_DIR}/
            BUILD_ALWAYS TRUE
            BUILD_BYPRODUCTS
                ${TILE_FWK_BIN_ROOT}/libtilefwk_backend_server.so
    )
endif ()

file(GLOB_RECURSE _Compiler_Srcs
        host/*.cpp
        kernel/aicore_compiler.cpp
        dump/*.cpp
        platform/*.cpp
        utils/*.cpp
        cache_manager/*.cpp)

add_library(tile_fwk_compiler SHARED)
target_sources(tile_fwk_compiler PRIVATE ${_Compiler_Srcs})
target_include_directories(tile_fwk_compiler
        PUBLIC  # 编译 CANN 场景依赖
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${ENABLE_BUILD_WITH_CANN}>>:${TOP_DIR}/inc>
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${ENABLE_BUILD_WITH_CANN}>>:${TOP_DIR}/abl/msprof/inc>
        PUBLIC  # 使用 CANN 场景依赖
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${ENABLE_BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include>
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${ENABLE_BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${ENABLE_BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(tile_fwk_compiler
        PUBLIC
            $<$<BOOL:${ENABLE_BUILD_WITH_CANN}>:ENABLE_BUILD_WITH_CANN>
        PRIVATE
            SRCPATH="${TILE_FWK_SRC_ROOT}"
            BISHENG_LD="${BISHENG_LD}"
            BISHENG_PROGRAM="${BISHENG_PROGRAM}"
)
target_link_directories(tile_fwk_compiler
        PUBLIC
            $<$<BOOL:${ENABLE_BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/lib64>
)
target_link_libraries(tile_fwk_compiler
        PUBLIC
            tile_fwk_interface
            tile_fwk_codegen
            $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,nlohmann_json::nlohmann_json,json>
        PRIVATE
            intf_pub_cxx17
            tile_fwk_intf_pub
            dl
            $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,boundscheck,c_sec>
)

TileFwk_AnalysisTargetHeaderFiles(TARGET tile_fwk_compiler)
TileFwk_AnalysisTargetSymbols(TARGET tile_fwk_compiler)

install(TARGETS tile_fwk_compiler
        LIBRARY DESTINATION lib OPTIONAL
)
