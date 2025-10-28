# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

file(GLOB _Interface_Srcs
        # 前端框架
        schema/*.cpp
        configs/*.cpp
        tensor/*.cpp
        program/*.cpp
        function/*.cpp
        interpreter/*.cpp
        utils/*.cpp

        # Operation
        operation/*.cpp
        operation/distributed/*.cpp

        # Machine
        cache/*.cpp
        machine/*.cpp
        machine/host/*.cpp
        registry/*.cpp
)
#PTO_Fwk_Debug_Source_List(NAME "_Interface_Srcs" LIST ${_Interface_Srcs} DETAIL OFF)

add_library(tile_fwk_interface SHARED)
target_sources(tile_fwk_interface PRIVATE ${_Interface_Srcs})
target_compile_definitions(tile_fwk_interface
        PRIVATE
            SRCPATH="${PTO_FWK_SRC_ROOT}"
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
PTO_Fwk_AnalysisTargetHeaderFiles(TARGET tile_fwk_interface)
PTO_Fwk_AnalysisTargetSymbols(TARGET tile_fwk_interface)

if (ENABLE_TORCH_VERIFIER)
    add_library(tile_fwk_calculator SHARED)
    target_link_libraries(tile_fwk_interface
            PUBLIC
                tile_fwk_calculator
    )
    target_sources(tile_fwk_calculator
            PRIVATE
                interpreter/calculator/calc_torch.cpp
    )
    target_compile_definitions(tile_fwk_calculator
            PRIVATE
                -D_GLIBCXX_USE_CXX11_ABI=${TORCH_ABI_VERSION}
                -D__DEVICE__
            PUBLIC
                -DENABLE_VERIFIER=1
    )
    target_include_directories(tile_fwk_calculator
            SYSTEM PRIVATE
                ${TORCH_ROOT_PATH}/include
                ${TORCH_ROOT_PATH}/include/torch/csrc/api/include
                ${PTO_FWK_SRC_ROOT}/include
                ${PTO_FWK_SRC_ROOT}/src/
    )
    target_link_directories(tile_fwk_calculator
            PUBLIC
                ${TORCH_ROOT_PATH}/lib
                ${TORCH_ROOT_PATH}/../torch.libs
    )
    target_link_libraries(tile_fwk_calculator
            PRIVATE
                $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,nlohmann_json::nlohmann_json,json>
                $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,boundscheck,c_sec>
                torch
    )
    target_link_options(tile_fwk_calculator
            PRIVATE
                -Wl,-no-as-needed,${TORCH_ROOT_PATH}/lib/libtorch_cpu.so -Wl,-as-needed
                -Wl,-no-as-needed,${TORCH_ROOT_PATH}/lib/libc10.so -Wl,-as-needed
    )
    install(TARGETS tile_fwk_calculator
            LIBRARY DESTINATION lib OPTIONAL
    )
endif() ## ENABLE_TORCH_VERIFIER

if (ENABLE_FEATURE_PYTHON_FRONT_END)
        install(TARGETS tile_fwk_interface LIBRARY DESTINATION pto/lib OPTIONAL)
        install(FILES configs/tile_fwk_config.json DESTINATION pto/configs)
        install(FILES configs/tile_fwk_config_schema.json DESTINATION pto/configs)
else ()
        install(TARGETS tile_fwk_interface
                LIBRARY DESTINATION lib OPTIONAL
        )

        add_custom_target(
                tile_fwk_config.json ALL
                DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/tile_fwk_config.json
                DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/tile_fwk_config_schema.json
        )

        add_custom_command(
                OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tile_fwk_config.json
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_LIST_DIR}/configs/tile_fwk_config.json ${CMAKE_CURRENT_BINARY_DIR}
        )

        add_custom_target(
                tile_fwk_config_schema_json ALL
                DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/tile_fwk_config_schema.json
        )

        add_custom_command(
                OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tile_fwk_config_schema.json
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_LIST_DIR}/configs/tile_fwk_config_schema.json ${CMAKE_CURRENT_BINARY_DIR}
        )

        install(FILES
                ${CMAKE_CURRENT_BINARY_DIR}/tile_fwk_config.json
                ${CMAKE_CURRENT_BINARY_DIR}/tile_fwk_config_schema.json
                DESTINATION lib OPTIONAL
        )
endif()
