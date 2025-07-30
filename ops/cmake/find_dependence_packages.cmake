# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

if (BUILD_OPEN_PROJECT OR ENABLE_BUILD_BINARY)
    include(cmake/function.cmake)
    include(cmake/intf_pub.cmake)
endif()

if (DEFINED ENV{ASCEND_3RD_LIB_PATH} AND NOT "$ENV{ASCEND_3RD_LIB_PATH}x" STREQUAL "x")
    get_filename_component(ASCEND_3RD_LIB_PATH "$ENV{ASCEND_3RD_LIB_PATH}" REALPATH)
    if (EXISTS "${ASCEND_3RD_LIB_PATH}/cmake/modules")
        list(APPEND CMAKE_MODULE_PATH ${ASCEND_3RD_LIB_PATH}/cmake/modules)
    endif()
endif()

if (BUILD_OPEN_PROJECT AND ENABLE_BUILD_HOST)
    list(APPEND CMAKE_PREFIX_PATH ${ASCEND_3RD_LIB_PATH}/protoc)
    find_package(protoc MODULE REQUIRED)
    list(APPEND CMAKE_PREFIX_PATH ${ASCEND_3RD_LIB_PATH}/ascend_protobuf)
    find_package(ascend_protobuf_shared MODULE REQUIRED)
    find_package_if_target_not_exists(securec MODULE REQUIRED)
    find_package_if_target_not_exists(slog MODULE REQUIRED)
    find_package_if_target_not_exists(mmpa MODULE REQUIRED)
    find_package_if_target_not_exists(platform MODULE REQUIRED)
    find_package_if_target_not_exists(metadef MODULE REQUIRED)
endif()

if (ENABLE_BUILD_BINARY)
    list(APPEND CMAKE_PREFIX_PATH ${TILE_FWK_OPS_ROOT_PATH}/../output/lib64)
    find_package_if_target_not_exists(tile_fwk_interface MODULE REQUIRED)
    list(APPEND CMAKE_PREFIX_PATH ${ASCEND_3RD_LIB_PATH}/json)
    find_package(nlohmann_json CONFIG)
endif()
