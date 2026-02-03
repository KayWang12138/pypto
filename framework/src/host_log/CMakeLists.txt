# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB _Log_Srcs
        log_api.cpp
        log_manager.cpp)

add_library(tile_fwk_log SHARED)
target_sources(tile_fwk_log PRIVATE ${_Log_Srcs})
target_include_directories(tile_fwk_log
        PUBLIC  # 编译 CANN 场景依赖
            $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/inc>>
            $<BUILD_INTERFACE:$<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/abl/msprof/inc>>
        PUBLIC  # 使用 CANN 场景依赖
            $<BUILD_INTERFACE:$<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include>>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(tile_fwk_log
        PRIVATE
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:BUILD_WITH_CANN>>
)
target_link_directories(tile_fwk_log
        PUBLIC
            $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/lib64>>
)
target_link_libraries(tile_fwk_log
        PRIVATE
            intf_pub_cxx17
            tile_fwk_intf_pub
            dl
            c_sec
)

PTO_Fwk_AnalysisTargetHeaderFiles(TARGET tile_fwk_log)
PTO_Fwk_AnalysisTargetSymbols(TARGET tile_fwk_log)