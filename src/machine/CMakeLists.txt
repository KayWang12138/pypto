# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================


if (NOT AC_ENABLE_FRAMEWORK_WITHOUT_CANN)
    add_subdirectory(kernel)
endif ()


if (NOT AC_ENABLE_FRAMEWORK_WITHOUT_CANN)
    ExternalProject_Add(tile_fwk_server
            SOURCE_DIR ${TILE_FWK_SRC_ROOT}/src/machine/device
            CONFIGURE_COMMAND ${CMAKE_COMMAND}
                -G ${CMAKE_GENERATOR} 
                -DBUILD_OPEN_PROJECT=${BUILD_OPEN_PROJECT}
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -DASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}
                -DCMAKE_C_COMPILER=${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-gcc
                -DCMAKE_CXX_COMPILER=${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++
                -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
                -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                <SOURCE_DIR>
            INSTALL_COMMAND ${CMAKE_COMMAND} -E copy <BINARY_DIR>/libASTServer.so ${CMAKE_BINARY_DIR}/
            BUILD_ALWAYS TRUE
            BUILD_BYPRODUCTS
                ${TILE_FWK_BIN_ROOT}/libASTServer.so
    )
endif ()


file(GLOB_RECURSE _Runtime_Srcs *.cpp)
file(GLOB_RECURSE _AiCore_Srcs kernel/*)
list(REMOVE_ITEM _Runtime_Srcs ${_AiCore_Srcs})
file(GLOB_RECURSE _Device_Srcs device/*)
# list(REMOVE_ITEM _Runtime_Srcs ${_Device_Srcs})
#TileFwk_Debug_List(NAME "_Runtime_Srcs" LIST ${_Runtime_Srcs} DETAIL OFF)

add_library(tile_fwk_runtime SHARED)
target_sources(tile_fwk_runtime PRIVATE ${_Runtime_Srcs})
target_include_directories(tile_fwk_runtime
        PUBLIC  # 编译 CANN 场景依赖
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>>:${TOP_DIR}/inc>
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>>:${TOP_DIR}/abl/msprof/inc>
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>>:${TOP_DIR}/ace/npuruntime/inc>
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>>:${TOP_DIR}/ace/npuruntime/acl/inc/external>
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>>:${TOP_DIR}/ace/comop/inc>
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>>:${TOP_DIR}/ace/comop/inc/external>
        PUBLIC  # 使用 CANN 场景依赖
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>>:${ASCEND_CANN_PACKAGE_PATH}/include>
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/ascend_hal>
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime>
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof>
        PRIVATE
            ${TILE_FWK_SRC_ROOT}/tests/cost_model
)
target_compile_definitions(tile_fwk_runtime
        PUBLIC
            $<$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>:AC_ENABLE_FRAMEWORK_WITHOUT_CANN>
            $<$<BOOL:${ENABLE_HCCL_STUB}>:ENABLE_HCCL_STUB>
)
target_link_directories(tile_fwk_runtime
        PUBLIC
            $<$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/lib64>
            $<$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>:/usr/local/Ascend/driver/lib64/driver>
            $<$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib>
)
target_link_libraries(tile_fwk_runtime
        PUBLIC
            tile_fwk_interface
            tile_fwk_codegen
            $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,nlohmann_json::nlohmann_json,json>
            $<$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>:ascendcl>
            $<$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>:runtime>
            $<$<AND:$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>,$<NOT:$<BOOL:${ENABLE_HCCL_STUB}>>>:hccl>
        PRIVATE
            intf_pub_cxx17
            tile_fwk_intf_pub
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>>:kernel_tiling_headers>
            dl
            $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,boundscheck,c_sec>
)
if ((NOT AC_ENABLE_FRAMEWORK_WITHOUT_CANN) AND (NOT PRODUCT_SIDE STREQUAL "device"))
    add_dependencies(tile_fwk_runtime tile_fwk_core)
    target_link_libraries(tile_fwk_runtime
            PRIVATE
                ${PROJECT_BINARY_DIR}/src/machine/kernel/kernel_host.o
    )
endif ()
TileFwk_AnalysisTargetHeaderFiles(TARGET tile_fwk_runtime)
TileFwk_AnalysisTargetSymbols(TARGET tile_fwk_runtime)

install(TARGETS tile_fwk_runtime
        LIBRARY DESTINATION lib OPTIONAL
)
