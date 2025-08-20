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

if (ENABLE_TORCH_VERIFIER)
execute_process(
    COMMAND ${TILE_FWK_PYTHON3_EXE} -c "
import torch
print(torch.utils.cmake_prefix_path)
print(int(torch._C._GLIBCXX_USE_CXX11_ABI))
"
    OUTPUT_VARIABLE TORCH_ENV_OUTPUT
)
string(REPLACE "\n" ";" TORCH_ENV_LIST "${TORCH_ENV_OUTPUT}")
list(GET TORCH_ENV_LIST 0 TORCH_ROOT_PATH)
list(GET TORCH_ENV_LIST 1 TORCH_ABI_VERSION)
get_filename_component(TORCH_ROOT_PATH "${TORCH_ROOT_PATH}/../.." REALPATH)

add_library(tile_fwk_calculator SHARED)
target_link_libraries(tile_fwk_interface
PUBLIC
    tile_fwk_calculator
)

target_sources(tile_fwk_calculator PRIVATE
    interpreter/calculator/calc_torch.cpp
)

target_compile_definitions(tile_fwk_calculator
PRIVATE
    -D_GLIBCXX_USE_CXX11_ABI=${TORCH_ABI_VERSION}
PUBLIC
    -DENABLE_VERIFIER=1
)

target_include_directories(tile_fwk_calculator SYSTEM PRIVATE
    ${TORCH_ROOT_PATH}/include
    ${TORCH_ROOT_PATH}/include/torch/csrc/api/include
    ${TILE_FWK_SRC_ROOT}/include
    ${TILE_FWK_SRC_ROOT}/src/
)

target_link_directories(tile_fwk_calculator PUBLIC
    ${TORCH_ROOT_PATH}/lib
    ${TORCH_ROOT_PATH}/../torch.libs
)

target_link_libraries(tile_fwk_calculator PRIVATE
    $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,nlohmann_json::nlohmann_json,json>
    $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,boundscheck,c_sec>
    torch
)

target_link_options(tile_fwk_calculator PRIVATE
    -Wl,-no-as-needed,${TORCH_ROOT_PATH}/lib/libtorch_cpu.so -Wl,-as-needed
    -Wl,-no-as-needed,${TORCH_ROOT_PATH}/lib/libc10.so -Wl,-as-needed
)

install(TARGETS tile_fwk_calculator
        LIBRARY DESTINATION lib OPTIONAL
)
endif() ## ENABLE_TORCH_VERIFIER

install(TARGETS tile_fwk_interface
        LIBRARY DESTINATION lib OPTIONAL
)

add_custom_target(
        tile_fwk_config.json ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/tile_fwk_config.json
)

add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tile_fwk_config.json
        COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/configs/tile_fwk_config.json ${CMAKE_CURRENT_BINARY_DIR}
)

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/tile_fwk_config.json
        DESTINATION lib OPTIONAL
)
