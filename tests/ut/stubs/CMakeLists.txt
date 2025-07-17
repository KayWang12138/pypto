# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

if (NOT AC_ENABLE_FRAMEWORK_WITHOUT_CANN)
    # stub ascendcl
    add_library(ascendcl SHARED)
    target_sources(ascendcl
            PRIVATE
                src/acl_stubs.cpp
    )
    target_link_libraries(ascendcl
            PRIVATE
                intf_pub_cxx17
                tile_fwk_intf_pub
    )
    target_include_directories(ascendcl
            PRIVATE  # 编译 CANN 场景依赖
                $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/inc>
                $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/abl/msprof/inc>
                $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/ace/npuruntime/inc>
                $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/ace/npuruntime/acl/inc/external>
                $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/ace/comop/inc>
                $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/ace/comop/inc/external>
            PRIVATE  # 使用 CANN 场景依赖
                $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include>
                $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>
                $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/ascend_hal>
                $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime>
                $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof>
    )

    # stub runtime
    add_library(runtime SHARED src/runtime_stubs.cpp)
    target_link_libraries(runtime
            PRIVATE
                intf_pub_cxx17
                tile_fwk_intf_pub
                $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,boundscheck,c_sec>
    )
    target_include_directories(runtime
            PRIVATE  # 编译 CANN 场景依赖
                $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/inc>
                $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/abl/msprof/inc>
                $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/ace/npuruntime/inc>
                $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/ace/npuruntime/acl/inc/external>
                $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/ace/comop/inc>
                $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/ace/comop/inc/external>
            PRIVATE  # 使用 CANN 场景依赖
                $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include>
                $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>
                $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/ascend_hal>
                $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime>
                $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof>
    )

    # stub hccl
    add_library(hccl SHARED)
    target_sources(hccl
            PRIVATE
                src/hccl_stubs.cpp
    )
    target_link_libraries(hccl
            PRIVATE
                intf_pub_cxx17
                tile_fwk_intf_pub
                $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,boundscheck,c_sec>
    )
    target_include_directories(hccl
            PRIVATE  # 编译 CANN 场景依赖
                $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/inc>
                $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/abl/msprof/inc>
                $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/ace/npuruntime/inc>
                $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/ace/npuruntime/acl/inc/external>
                $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/ace/comop/inc>
                $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/ace/comop/inc/external>
            PRIVATE  # 使用 CANN 场景依赖
                $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include>
                $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>
                $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/ascend_hal>
                $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime>
                $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof>
    )

    # 追加 Stubs 相关配置, 需保证:
    # 1. stubs 下相关代码编译至可执行文件内, 保证用例内直调 Stubs 内相关函数走进 Stubs;
    # 2. stubs 下相关 target 创建, 保证业务代码(如 tile_fwk_runtime) 链接正常;
    set(_Target ${TileFwkUTestNamePrefix}_stubs)
    aux_source_directory(src _Sources)

    add_library(${_Target} STATIC)
    target_sources(${_Target} PRIVATE ${_Sources})
    target_include_directories(${_Target}
            PUBLIC  # 编译 CANN 场景依赖
                $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/inc>
                $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/abl/msprof/inc>
                $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/ace/npuruntime/inc>
                $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/ace/npuruntime/acl/inc/external>
                $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/ace/comop/inc>
                $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/ace/comop/inc/external>
            PUBLIC  # 使用 CANN 场景依赖
                $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include>
                $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>
                $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/ascend_hal>
                $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime>
                $<$<BOOL:${BUILD_OPEN_PROJECT}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof>
    )
    target_link_libraries(${_Target}
            PUBLIC
                $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,boundscheck,c_sec>
    )
endif()
