# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

file(GLOB_RECURSE _Runtime_Srcs *.cpp)
file(GLOB_RECURSE _Device_Srcs ${PTO_FWK_SRC_ROOT}/framework/src/machine/device/*.cpp)
add_library(tile_fwk_runtime SHARED)
target_sources(tile_fwk_runtime PRIVATE ${_Runtime_Srcs} ${_Device_Srcs})
target_include_directories(tile_fwk_runtime
        PUBLIC
            ${PTO_FWK_SRC_ROOT}/framework/src/machine/runtime
        PUBLIC  # 编译 CANN 场景依赖
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${BUILD_WITH_CANN}>>:${TOP_DIR}/inc>
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${BUILD_WITH_CANN}>>:${TOP_DIR}/abl/msprof/inc>
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${BUILD_WITH_CANN}>>:${TOP_DIR}/ace/npuruntime/inc>
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${BUILD_WITH_CANN}>>:${TOP_DIR}/ace/npuruntime/acl/inc/external>
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${BUILD_WITH_CANN}>>:${TOP_DIR}/ace/comop/inc>
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${BUILD_WITH_CANN}>>:${TOP_DIR}/ace/comop/inc/external>
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${BUILD_WITH_CANN}>>:${TOP_DIR}/ace/comop/hccl/open_source/pkg_inc>
        PUBLIC  # 使用 CANN 场景依赖
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include>
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/hccl>
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/ascend_hal>
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime>
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<BOOL:${BUILD_WITH_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof>
        PRIVATE
            ${PTO_FWK_SRC_ROOT}/framework/src/cost_model
)
target_compile_definitions(tile_fwk_runtime
        PUBLIC
            $<$<BOOL:${BUILD_WITH_CANN}>:BUILD_WITH_CANN>
            $<$<BOOL:${ENABLE_STEST_INTERPRETER_CONFIG}>:ENABLE_STEST_INTERPRETER_CONFIG>
        PRIVATE
           $<$<OR:$<BOOL:${ENABLE_UTEST}>,$<BOOL:${ENABLE_STEST}>,$<BOOL:${ENABLE_STEST_DISTRIBUTED}>>:SRCPATH="${PTO_FWK_SRC_ROOT}">
)
target_link_directories(tile_fwk_runtime
        PUBLIC
            $<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/lib64>
            $<$<BOOL:${BUILD_WITH_CANN}>:/usr/local/Ascend/driver/lib64/driver>
            $<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib>
)
target_link_libraries(tile_fwk_runtime
        PUBLIC
            tile_fwk_interface
            tile_fwk_compiler
            $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,nlohmann_json::nlohmann_json,json>
            $<$<BOOL:${BUILD_WITH_CANN}>:ascendcl>
            $<$<BOOL:${BUILD_WITH_CANN}>:runtime>
            $<$<BOOL:${BUILD_WITH_CANN}>:hccl>
        PRIVATE
            intf_pub_cxx17
            tile_fwk_intf_pub
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<BOOL:${BUILD_WITH_CANN}>>:kernel_tiling_headers>
            dl
            $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,boundscheck,c_sec>
)
if ((BUILD_WITH_CANN) AND (NOT PRODUCT_SIDE STREQUAL "device"))
    add_dependencies(tile_fwk_runtime tile_fwk_core)
    target_link_libraries(tile_fwk_runtime
            PRIVATE
                ${PROJECT_BINARY_DIR}/framework/src/machine/kernel/kernel_host.o
    )
endif ()
PTO_Fwk_AnalysisTargetHeaderFiles(TARGET tile_fwk_runtime)
PTO_Fwk_AnalysisTargetSymbols(TARGET tile_fwk_runtime)

if (ENABLE_FEATURE_PYTHON_FRONT_END)
    install(TARGETS tile_fwk_runtime LIBRARY DESTINATION pto/lib OPTIONAL)
else ()
    install(TARGETS tile_fwk_runtime
            LIBRARY DESTINATION lib OPTIONAL
    )
endif ()
