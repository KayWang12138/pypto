# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

file(GLOB_RECURSE _Interface_Srcs
        # 前端框架
        configs/*.cpp
        tensor/*.cpp
        program/*.cpp
        function/*.cpp
        interpreter/*.cpp
        utils/*.cpp

        # Operator
        ${TILE_FWK_SRC_ROOT}/src/models/llama/*.cpp
        ${TILE_FWK_SRC_ROOT}/src/models/deepseek/*.cpp
        ${TILE_FWK_SRC_ROOT}/src/models/nsa/*.cpp
        operator/activation/*.cpp
        operator/convolution/*.cpp
        operator/geometric/*.cpp
        operator/logic/*.cpp
        operator/matmul/*.cpp
        operator/normalization/*.cpp
        operator/quantize/*.cpp

        # Operation
        operation/*.cpp
        operation/distributed/*.cpp

        # Machine
        cache/*.cpp
        machine/*.cpp
        registry/*.cpp
)
#TileFwk_Debug_List(NAME "_Interface_Srcs" LIST ${_Interface_Srcs} DETAIL OFF)

add_library(tile_fwk_interface SHARED)
target_sources(tile_fwk_interface PRIVATE ${_Interface_Srcs})
target_compile_definitions(tile_fwk_interface
        PRIVATE
            SRCPATH="${TILE_FWK_SRC_ROOT}"
)
target_link_libraries(tile_fwk_interface
        PUBLIC
            $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,nlohmann_json::nlohmann_json,json>
            $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,boundscheck,c_sec>
        PRIVATE
            intf_pub_cxx17
            tile_fwk_intf_pub
            tile_fwk_passes
            dl
            Threads::Threads
)
TileFwk_AnalysisTargetHeaderFiles(TARGET tile_fwk_interface)
TileFwk_AnalysisTargetSymbols(TARGET tile_fwk_interface)

install(TARGETS tile_fwk_interface
        LIBRARY DESTINATION lib OPTIONAL
)

install(FILES configs/config.json
        DESTINATION lib/config/ OPTIONAL
)
