# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================


file(GLOB_RECURSE _Executor_Srcs *.cpp)
add_library(tile_fwk_executor SHARED)
target_sources(tile_fwk_executor PRIVATE ${_Executor_Srcs})
target_include_directories(tile_fwk_executor
        PUBLIC  # yellow zone
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${ENABLE_BUILD_WITH_CANN}>>:${TOP_DIR}/inc>
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${ENABLE_BUILD_WITH_CANN}>>:${TOP_DIR}/abl/msprof/inc>
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${ENABLE_BUILD_WITH_CANN}>>:${TOP_DIR}/ace/npuruntime/inc>
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${ENABLE_BUILD_WITH_CANN}>>:${TOP_DIR}/metadef/inc>
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${ENABLE_BUILD_WITH_CANN}>>:${TOP_DIR}/metadef/inc/external>
        PUBLIC  # blue zone
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${ENABLE_BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include>
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${ENABLE_BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${ENABLE_BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof>
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${ENABLE_BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime>
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${ENABLE_BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/ascend_hal>
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${ENABLE_BUILD_WITH_CANN}>>:${TILE_FWK_SRC_ROOT}/third_party/metadef/inc>
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${ENABLE_BUILD_WITH_CANN}>>:${TILE_FWK_SRC_ROOT}/third_party/metadef/inc/external>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_directories(tile_fwk_executor
        PUBLIC
            $<$<BOOL:${ENABLE_BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/lib64>
)
target_link_libraries(tile_fwk_executor
        PUBLIC
            graph
            register
            runtime
            $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,boundscheck,c_sec>
            $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,nlohmann_json::nlohmann_json,json>
        PRIVATE
            intf_pub_cxx17
            tile_fwk_intf_pub 
)

TileFwk_AnalysisTargetHeaderFiles(TARGET tile_fwk_executor)
TileFwk_AnalysisTargetSymbols(TARGET tile_fwk_executor)

install(TARGETS tile_fwk_executor
        LIBRARY DESTINATION lib OPTIONAL
)
