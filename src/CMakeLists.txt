# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================


function(AscendCpp_Stubs_AddShared)
    cmake_parse_arguments(
            TMP
            ""
            "TARGET"
            ""
            ""
            ${ARGN}
    )
    add_library(${TMP_TARGET} SHARED)
    target_sources(${TMP_TARGET}
            PRIVATE
                ${ASCENDCPP_SRC_ROOT}/src/stubs/stubs.cpp
)
    target_include_directories(${TMP_TARGET}
            PUBLIC
                ${ASCENDCPP_SRC_ROOT}/src/stubs/
    )
    target_link_libraries(${TMP_TARGET}
            PUBLIC
                intf_pub_cxx17
                ascendcpp_intf_pub
    )
endfunction()

add_library(ascendcpp_stubs STATIC)

target_sources(ascendcpp_stubs
        PRIVATE
            stubs/stubs.cpp
)

target_include_directories(ascendcpp_stubs
        PUBLIC
            stubs
)

target_link_libraries(ascendcpp_stubs
        PUBLIC
            intf_pub_cxx17
            ascendcpp_intf_pub
)

# Stubs
AscendCpp_Stubs_AddShared(TARGET ascendcpp_interface)
AscendCpp_Stubs_AddShared(TARGET ascendcpp_codegen)
AscendCpp_Stubs_AddShared(TARGET ascendcpp_simulation)
AscendCpp_Stubs_AddShared(TARGET ascendcpp_runtime)
AscendCpp_Stubs_AddShared(TARGET get_cce_simulation)

if (NOT AC_ENABLE_FRAMEWORK_WITHOUT_CANN)
    ExternalProject_Add(ascendcpp_server
            SOURCE_DIR ${ASCENDCPP_SRC_ROOT}/src/runtime/device
            CONFIGURE_COMMAND ${CMAKE_COMMAND}
                -G ${CMAKE_GENERATOR} 
                -DBUILD_OPEN_PROJECT=${BUILD_OPEN_PROJECT}
                -DASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}
                -DCMAKE_C_COMPILER=${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-gcc
                -DCMAKE_CXX_COMPILER=${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++
                -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
                -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                <SOURCE_DIR>
            INSTALL_COMMAND ${CMAKE_COMMAND} -E copy <BINARY_DIR>/libASTServer.so ${CMAKE_BINARY_DIR}/
            BUILD_ALWAYS TRUE
            BUILD_BYPRODUCTS
                ${ASCENDCPP_BIN_ROOT}/libASTServer.so
    )
endif ()
