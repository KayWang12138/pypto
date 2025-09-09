# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

file(GLOB_RECURSE _Operator_Srcs *.cpp)

#PTO_Fwk_Debug_Source_List(NAME "_Operator_Srcs" LIST ${_Operator_Srcs} DETAIL OFF)

add_library(tile_fwk_operator SHARED)
target_sources(tile_fwk_operator PRIVATE ${_Operator_Srcs})
target_link_libraries(tile_fwk_operator
        PUBLIC
            $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,nlohmann_json::nlohmann_json,json>
            $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,boundscheck,c_sec>
        PRIVATE
            intf_pub_cxx17
            tile_fwk_intf_pub
            tile_fwk_interface
)
TileFwk_AnalysisTargetHeaderFiles(TARGET tile_fwk_operator)
PTO_Fwk_AnalysisTargetSymbols(TARGET tile_fwk_operator)

install(TARGETS tile_fwk_operator
        LIBRARY DESTINATION lib OPTIONAL
)
