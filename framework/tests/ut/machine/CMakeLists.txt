# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_Target ${PTO_Fwk_UTestNamePrefix}_machine)

file(GLOB_RECURSE _Sources src/*.cpp)
list(APPEND _Sources ${PTO_FWK_SRC_ROOT}/framework/src/machine/device/dynamic/aicore_prof.cpp)
list(APPEND _Sources ${PTO_FWK_SRC_ROOT}/framework/src/machine/device/device_machine.cpp)
list(APPEND _Sources ${PTO_FWK_SRC_ROOT}/framework/src/machine/device/distributed/shmem_wait_until.cpp)
list(APPEND _Sources ${PTO_FWK_SRC_ROOT}/framework/src/machine/device/machine_interface/pypto_aicpu_interface.cpp)
set(_Include ${CMAKE_CURRENT_SOURCE_DIR}/include)

PTO_Fwk_UTest_AddCaseLib(
        TARGET ${_Target}
        SOURCES ${_Sources}
        PRIVATE_INCLUDE_DIRECTORIES ${_Include}
        LINK_LIBRARIES tile_fwk_runtime tile_fwk_operator dl
)
target_compile_definitions(${_Target}
        PRIVATE
            NO_NEON_SUPPORT
)
if (BUILD_WITH_CANN)
    target_include_directories(${_Target}
            PRIVATE  # 使用 CANN 场景依赖
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include>>
            PRIVATE  # 使用 CANN 整包
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include/base>>
                $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_WITH_CANN_SUB}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>>
                $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_WITH_CANN_SUB}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/hccl>>
                $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_WITH_CANN_SUB}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/ascend_hal>>
                $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_WITH_CANN_SUB}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime>>
                $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_WITH_CANN_SUB}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime/runtime>>
                $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_WITH_CANN_SUB}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof>>
                $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_WITH_CANN_SUB}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof/toolchain>>
            PRIVATE  # 使用 CANN 分包
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_SUB}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_SUB}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_SUB}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_SUB}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_SUB}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling>>
    )
endif ()
# 可执行程序运行依赖的配置, 脚本配置
add_custom_command(
        TARGET ${_Target} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/stubs/compiler/data/platform_config/
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/stubs/compiler/data/platform_config/Ascend910B1.ini
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/stubs/compiler/data/platform_config/
        COMMENT "Copy ${_Target} dependence config file to ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/stubs/compiler/data/platform_config/"
)
