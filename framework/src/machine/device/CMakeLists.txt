# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.3)

project(tile_fwk_server LANGUAGES C CXX)

########################################################################################################################
# 基础配置
########################################################################################################################
if (BUILD_OPEN_PROJECT)
    add_subdirectory(stub)
else ()
    set(CMAKE_SKIP_RPATH TRUE)
    get_filename_component(PTO_FWK_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../" ABSOLUTE)
endif()

add_subdirectory(dynamic)

# 缺省编译配置, 编译函数
if (BUILD_WITH_CANN)
    set(CMAKE_C_COMPILER "${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-gcc")
    set(CMAKE_CXX_COMPILER "${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++")
endif ()

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
            ${PTO_FWK_SRC_ROOT}/framework/include
            ${PTO_FWK_SRC_ROOT}/framework/src
            ${PTO_FWK_SRC_ROOT}/framework/src/include
            ${PTO_FWK_SRC_ROOT}/framework/src/interface
            ${PTO_FWK_SRC_ROOT}/framework/src/interface/machine/device
            ${PTO_FWK_SRC_ROOT}/framework/src/cost_model
        PRIVATE  # 使用 CANN 场景依赖
            $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include>
        PRIVATE  # 使用 CANN 整包
            $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include/toolchain>
        PRIVATE  # 使用 CANN 分包
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_SUB}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base>>
)
target_link_directories(tile_fwk_server
        PRIVATE
            $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib/linux/aarch64/>
            $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib/device/>
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

if (NOT ENABLE_FEATURE_PYTHON_FRONT_END)
    install(TARGETS tile_fwk_server LIBRARY DESTINATION lib OPTIONAL)
endif ()

set(TILE_FWK_KERNEL_TAR_GZ_NAME "transformer_tile_fwk_aicpu_kernel.tar.gz")
if (BUILD_OPEN_PROJECT)
    if (ENABLE_SIGN)
        set(SIGN_CONFIG ${PTO_FWK_SRC_ROOT}/scripts/sign/sign_check_cfg.xml)
        if (NOT DEFINED SIGN_SCRIPT OR "${SIGN_SCRIPT}" STREQUAL "")
            # using community sign script as default
            set(SIGN_SCRIPT ${PTO_FWK_SRC_ROOT}/scripts/sign/community_sign_build.py)
        endif ()
        set(SIGN_DIR ${PTO_FWK_SRC_ROOT}/scripts/sign_package)
        add_custom_command(
            OUTPUT ${TILE_FWK_KERNEL_TAR_GZ_NAME}
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${SIGN_DIR}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${SIGN_DIR}/aicpu_kernels_device/tile_fwk_machine
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tile_fwk_server> ${SIGN_DIR}/aicpu_kernels_device/tile_fwk_machine
            COMMAND tar czf ${SIGN_DIR}/${TILE_FWK_KERNEL_TAR_GZ_NAME} ${SIGN_DIR}/aicpu_kernels_device
            COMMAND python3 ${PTO_FWK_SRC_ROOT}/scripts/sign/add_header_sign.py ${SIGN_DIR} true --bios_check_cfg=${SIGN_CONFIG} --sign_script=${SIGN_SCRIPT} --version=${SIGN_VERSION}
            COMMENT "Create tar.gz file from $<TARGET_FILE:tile_fwk_server>, and sign for it."
            DEPENDS tile_fwk_server
        )
        add_custom_target(tile_fwk_aicpu_kernel_tar_gz_pkg ALL DEPENDS ${TILE_FWK_KERNEL_TAR_GZ_NAME})
        if (NOT ENABLE_FEATURE_PYTHON_FRONT_END)
            install(FILES ${PTO_FWK_BIN_OUTPUT_ROOT}/${TILE_FWK_KERNEL_TAR_GZ_NAME} DESTINATION lib OPTIONAL)
        endif ()
    endif ()
else ()
    add_custom_command(TARGET tile_fwk_server POST_BUILD
        COMMENT "Begin to compress ${TILE_FWK_KERNEL_TAR_GZ_NAME} from $<TARGET_FILE:tile_fwk_server>"
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/aicpu_kernels_device/tile_fwk_machine
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tile_fwk_server> ${CMAKE_CURRENT_BINARY_DIR}/aicpu_kernels_device/tile_fwk_machine
        COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND tar czf ${TILE_FWK_KERNEL_TAR_GZ_NAME} aicpu_kernels_device
        COMMENT "${CMAKE_CURRENT_BINARY_DIR}/${TILE_FWK_KERNEL_TAR_GZ_NAME} has been generated."
    )
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${TILE_FWK_KERNEL_TAR_GZ_NAME} DESTINATION lib OPTIONAL)
endif()
