# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================


file(GLOB_RECURSE _CodeGen_Srcs *.cpp)
#PTO_Fwk_Debug_Source_List(NAME "_CodeGen_Srcs" LIST ${_CodeGen_Srcs} DETAIL OFF)

add_library(tile_fwk_codegen SHARED)
target_sources(tile_fwk_codegen PRIVATE ${_CodeGen_Srcs})
target_link_libraries(tile_fwk_codegen
        PUBLIC
            tile_fwk_interface
            $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,nlohmann_json::nlohmann_json,json>
        PRIVATE
            intf_pub_cxx17
            tile_fwk_intf_pub
            $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,boundscheck,c_sec>
)
PTO_Fwk_AnalysisTargetHeaderFiles(TARGET tile_fwk_codegen)
PTO_Fwk_AnalysisTargetSymbols(TARGET tile_fwk_codegen)

if (BUILD_OPEN_PROJECT AND ENABLE_BUILD_PTO_FWK_PYTHON)
        install(TARGETS tile_fwk_codegen LIBRARY DESTINATION pto/lib OPTIONAL)
else ()
        install(TARGETS tile_fwk_codegen
                LIBRARY DESTINATION lib OPTIONAL
        )
        install(DIRECTORY tileop
                DESTINATION lib/tilefwk/ OPTIONAL
        )
endif()
