# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


if (BUILD_WITH_CANN)
    add_subdirectory(kernel)
    add_subdirectory(device)
endif ()

file(GLOB_RECURSE _Compiler_Srcs
        host/*.cpp
        kernel/aicore_compiler.cpp
        kernel/gen_aicore_code.cpp
        dump/*.cpp
        platform/*.cpp
        utils/*.cpp
        runtime/*.cpp
        device/*.cpp
        cache_manager/*.cpp)

add_library(tile_fwk_compiler SHARED)
target_sources(tile_fwk_compiler PRIVATE ${_Compiler_Srcs})
target_include_directories(tile_fwk_compiler
        PUBLIC  # 编译 CANN 场景依赖
            $<BUILD_INTERFACE:$<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${BUILD_WITH_CANN}>>:${TOP_DIR}/inc>>
            $<BUILD_INTERFACE:$<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${BUILD_WITH_CANN}>>:${TOP_DIR}/abl/msprof/inc>>
            $<BUILD_INTERFACE:$<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${BUILD_WITH_CANN}>>:${TOP_DIR}/abl/msprof/inc/toolchain>>
            $<BUILD_INTERFACE:$<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${BUILD_WITH_CANN}>>:${TOP_DIR}/abl/slog/inc/toolchain>>
            $<BUILD_INTERFACE:$<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${BUILD_WITH_CANN}>>:${TOP_DIR}/ace/npuruntime/inc>>
            $<BUILD_INTERFACE:$<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${BUILD_WITH_CANN}>>:${TOP_DIR}/ace/npuruntime/inc/runtime>>
            $<BUILD_INTERFACE:$<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${BUILD_WITH_CANN}>>:${TOP_DIR}/ace/npuruntime/acl/inc/external>>
            $<BUILD_INTERFACE:$<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${BUILD_WITH_CANN}>>:${TOP_DIR}/ace/comop/inc>>
            $<BUILD_INTERFACE:$<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${BUILD_WITH_CANN}>>:${TOP_DIR}/ace/comop/inc/external>>
            $<BUILD_INTERFACE:$<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${BUILD_WITH_CANN}>>:${TOP_DIR}/ace/comop/hccl/open_source/pkg_inc>>
        PUBLIC  # 使用 CANN 场景依赖
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include/base>>
            $<BUILD_INTERFACE:$<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include>>
            $<BUILD_INTERFACE:$<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>>
            $<BUILD_INTERFACE:$<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/hccl>>
            $<BUILD_INTERFACE:$<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/ascend_hal>>
            $<BUILD_INTERFACE:$<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime>>
            $<BUILD_INTERFACE:$<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime/runtime>>
            $<BUILD_INTERFACE:$<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof>>
            $<BUILD_INTERFACE:$<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof/toolchain>>
        PUBLIC  # 使用 CANN 分包
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_SUB}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_SUB}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_SUB}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_SUB}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime>>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_SUB}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling>>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(tile_fwk_compiler
        PUBLIC
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:BUILD_WITH_CANN>>
            $<BUILD_INTERFACE:$<$<BOOL:${ENABLE_STEST_INTERPRETER_CONFIG}>:ENABLE_STEST_INTERPRETER_CONFIG>>
        PRIVATE
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_SUB}>:BUILD_WITH_CANN_SUB>>
)
target_link_directories(tile_fwk_compiler
        PUBLIC
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/lib64>>
            $<BUILD_INTERFACE:/usr/local/Ascend/driver/lib64/driver>
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib>>
)
target_link_libraries(tile_fwk_compiler
        PRIVATE
            $<$<BOOL:${ENABLE_UTEST}>:tile_fwk_interface>
            $<$<BOOL:${ENABLE_UTEST}>:tile_fwk_codegen>
            $<BUILD_INTERFACE:ascendcl>
            $<BUILD_INTERFACE:runtime>
            $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_WITH_CANN_SUB}>>:hccl>>
            $<BUILD_INTERFACE:profapi>
            json
        PRIVATE
            intf_pub_cxx17
            tile_fwk_intf_pub
            $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:kernel_tiling_headers>
            dl
            c_sec
)
if ((BUILD_WITH_CANN) AND (NOT PRODUCT_SIDE STREQUAL "device"))
    add_dependencies(tile_fwk_compiler tile_fwk_core)
    target_link_libraries(tile_fwk_compiler
            PRIVATE
                ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/kernel/kernel_host.o
    )
endif ()

PTO_Fwk_AnalysisTargetHeaderFiles(TARGET tile_fwk_compiler)
PTO_Fwk_AnalysisTargetSymbols(TARGET tile_fwk_compiler)
