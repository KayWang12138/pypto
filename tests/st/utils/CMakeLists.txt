# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set(_Target ${TileFwkSTestNamePrefix}_utils)
add_library(${_Target} INTERFACE)
target_include_directories(${_Target}
        INTERFACE
            include
        INTERFACE  # 编译 CANN 场景依赖
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>>:${TOP_DIR}/inc>
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>>:${TOP_DIR}/abl/msprof/inc>
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>>:${TOP_DIR}/ace/npuruntime/inc>
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>>:${TOP_DIR}/ace/npuruntime/acl/inc/external>
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>>:${TOP_DIR}/ace/comop/inc>
            $<$<AND:$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>,$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>>:${TOP_DIR}/ace/comop/inc/external>
        INTERFACE  # 使用 CANN 场景依赖
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>>:${ASCEND_CANN_PACKAGE_PATH}/include>
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/ascend_hal>
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime>
            $<$<AND:$<BOOL:${BUILD_OPEN_PROJECT}>,$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof>
        INTERFACE
            ${TILE_FWK_SRC_ROOT}/tests/cost_model
)
target_link_directories(${_Target}
        INTERFACE
            $<$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/lib64>
            $<$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>:/usr/local/Ascend/driver/lib64/driver>
            $<$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>:${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib>
)
target_link_libraries(${_Target}
        INTERFACE
            ${TileFwkSTestNamePrefix}_intf_pub
            $<$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>:ascendcl>
            $<$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>:runtime>
            $<$<NOT:$<BOOL:${AC_ENABLE_FRAMEWORK_WITHOUT_CANN}>>:hccl>
)
