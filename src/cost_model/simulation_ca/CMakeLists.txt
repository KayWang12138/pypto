# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================


file(GLOB_RECURSE _Ca_Model_Srcs *.cpp)
#PTO_Fwk_Debug_Source_List(NAME "_Ca_Model_Srcs" LIST ${_Ca_Model_Srcs} DETAIL ON)

add_library(tile_fwk_simulation_ca SHARED)
target_sources(tile_fwk_simulation_ca PRIVATE ${_Ca_Model_Srcs})
target_include_directories(tile_fwk_simulation_ca
        PRIVATE
            ${PTO_FWK_SRC_ROOT}/src/cost_model
)
target_compile_definitions(tile_fwk_simulation_ca
        PRIVATE
            SRCPATH="${PTO_FWK_SRC_ROOT}"
)
target_link_libraries(tile_fwk_simulation_ca
        PUBLIC
            tile_fwk_codegen
            tile_fwk_simulation
            $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,nlohmann_json::nlohmann_json,json>
        PRIVATE
            intf_pub_cxx17
            tile_fwk_intf_pub
            $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,boundscheck,c_sec>
)
TileFwk_AnalysisTargetHeaderFiles(TARGET tile_fwk_simulation_ca)
PTO_Fwk_AnalysisTargetSymbols(TARGET tile_fwk_simulation_ca)

install(TARGETS tile_fwk_simulation_ca
        LIBRARY DESTINATION lib OPTIONAL
)


# 新增CCE前仿二进制

add_executable(get_cce_simulation)
target_sources(get_cce_simulation PRIVATE tools/GenSimulation.cpp)
target_include_directories(get_cce_simulation
        PRIVATE
            ${PTO_FWK_SRC_ROOT}/src/cost_model
)
target_link_libraries(get_cce_simulation
        PRIVATE
            intf_pub_cxx17
            tile_fwk_intf_pub
            tile_fwk_simulation_ca
            dl
            Threads::Threads
)

install(TARGETS get_cce_simulation
       RUNTIME DESTINATION bin OPTIONAL
)
