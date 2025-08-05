# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================


file(GLOB_RECURSE _Pv_Model_Srcs *.cpp)
#TileFwk_Debug_List(NAME "_Pv_Model_Srcs" LIST ${_Pv_Model_Srcs} DETAIL ON)

add_library(tile_fwk_simulation_pv SHARED)
target_sources(tile_fwk_simulation_pv PRIVATE ${_Pv_Model_Srcs})
target_include_directories(tile_fwk_simulation_pv
        PRIVATE
            ${TILE_FWK_SRC_ROOT}/src/cost_model
)
target_compile_definitions(tile_fwk_simulation_pv
        PRIVATE
            SRCPATH="${TILE_FWK_SRC_ROOT}"
)
target_link_libraries(tile_fwk_simulation_pv
        PUBLIC
            tile_fwk_codegen
            tile_fwk_simulation
            $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,nlohmann_json::nlohmann_json,json>
        PRIVATE
            intf_pub_cxx17
            tile_fwk_intf_pub
            $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,boundscheck,c_sec>
)
TileFwk_AnalysisTargetHeaderFiles(TARGET tile_fwk_simulation_pv)
TileFwk_AnalysisTargetSymbols(TARGET tile_fwk_simulation_pv)

install(TARGETS tile_fwk_simulation_pv
        LIBRARY DESTINATION lib OPTIONAL
)
