# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

file(GLOB_RECURSE _Passes_Srcs *.cpp)
#TileFwk_Debug_List(NAME "_Passes_Srcs" LIST ${_Passes_Srcs} DETAIL OFF)

add_library(tile_fwk_passes SHARED)
target_sources(tile_fwk_passes PRIVATE ${_Passes_Srcs})
target_compile_definitions(tile_fwk_passes
        PRIVATE
            SRCPATH="${TILE_FWK_SRC_ROOT}"
)
target_link_libraries(tile_fwk_passes
        PUBLIC
            $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,nlohmann_json::nlohmann_json,json>
            $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,boundscheck,c_sec>
        PRIVATE
            intf_pub_cxx17
            tile_fwk_intf_pub
            dl
            Threads::Threads
)
TileFwk_AnalysisTargetHeaderFiles(TARGET tile_fwk_passes)
#TileFwk_AnalysisTargetSymbols(TARGET tile_fwk_passes)

install(TARGETS tile_fwk_passes
        LIBRARY DESTINATION lib OPTIONAL
)
