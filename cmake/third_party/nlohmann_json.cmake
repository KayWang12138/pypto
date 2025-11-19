# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set(_TargetVersion 3.11.3)

# 免重入
if (nlohmann_json_FOUND)
    message(STATUS "Use nlohmann_json from ${nlohmann_json_DIR}")
    return()
endif ()

# 直接查找制品, 若找到则直接退出
if (DEFINED ENV{CANN_3RD_LIB_PATH} AND EXISTS "$ENV{CANN_3RD_LIB_PATH}" AND IS_DIRECTORY "$ENV{CANN_3RD_LIB_PATH}")
    get_filename_component(CANN_3RD_LIB_PATH "$ENV{CANN_3RD_LIB_PATH}" REALPATH)
    find_package(nlohmann_json ${_TargetVersion} EXACT CONFIG PATHS ${CANN_3RD_LIB_PATH} NO_DEFAULT_PATH)
    get_filename_component(_TargetTarGzFile "${CANN_3RD_LIB_PATH}/json-${_TargetVersion}.tar.gz" REALPATH)
else ()
    find_package(nlohmann_json ${_TargetVersion} EXACT CONFIG)
    if (NOT ${nlohmann_json_FOUND})
        if (DEFINED ENV{ASCEND_3RD_LIB_PATH} AND EXISTS "$ENV{ASCEND_3RD_LIB_PATH}" AND IS_DIRECTORY "$ENV{ASCEND_3RD_LIB_PATH}")
            get_filename_component(ASCEND_3RD_LIB_PATH "$ENV{ASCEND_3RD_LIB_PATH}" REALPATH)
            if (EXISTS "${ASCEND_3RD_LIB_PATH}/cmake/modules")
                list(APPEND CMAKE_MODULE_PATH ${ASCEND_3RD_LIB_PATH}/cmake/modules)
            endif ()
            if (EXISTS "${ASCEND_3RD_LIB_PATH}/json/share/cmake/nlohmann_json")
                list(APPEND CMAKE_PREFIX_PATH ${ASCEND_3RD_LIB_PATH}/json/share/cmake/nlohmann_json)
            endif ()
        endif ()
        find_package(nlohmann_json ${_TargetVersion} EXACT CONFIG)
    endif ()
    if (NOT ${nlohmann_json_FOUND})
        message(FATAL_ERROR "No nlohmann_json found, please refer to the ReadMe of this project for installation instructions.")
    endif ()
    message(STATUS "Use nlohmann_json from ${nlohmann_json_DIR}")
endif ()
if (nlohmann_json_FOUND)
    message(STATUS "Use nlohmann_json from ${nlohmann_json_DIR}")
    # 重命名目标
    if (NOT TARGET json)
        get_target_property(_JsonInc nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
        add_library(json INTERFACE)
        set_target_properties(json PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_JsonInc}")
    endif ()
    return()
endif ()

# 获取源码路径
if (EXISTS "${_TargetTarGzFile}" AND NOT IS_DIRECTORY "${_TargetTarGzFile}")
    execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf ${_TargetTarGzFile}
            WORKING_DIRECTORY ${CANN_3RD_LIB_PATH}
            RESULT_VARIABLE _Rst
    )
    if (NOT _Rst EQUAL 0)
        message(FATAL_ERROR "Failed to decompress ${_TargetTarGzFile}")
    endif ()
endif ()
get_filename_component(_TargetSourceDir "${CANN_3RD_LIB_PATH}/json-${_TargetVersion}" REALPATH)
if (NOT EXISTS "${_TargetSourceDir}")
    get_filename_component(_TargetSourceDir "${CANN_3RD_LIB_PATH}/json" REALPATH)
endif ()

# 触发编译
include(ExternalProject)
if (EXISTS "${_TargetSourceDir}")
    get_filename_component(_TargetBinaryDir "${CANN_3RD_LIB_PATH}/build/${CMAKE_BUILD_TYPE}/json" REALPATH)
    ExternalProject_Add(ExternalProject_Nlohmann_Json
            PREFIX ${_TargetBinaryDir}
            SOURCE_DIR ${_TargetSourceDir}
            INSTALL_DIR ${CANN_3RD_LIB_PATH}
            CONFIGURE_COMMAND ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR} -S <SOURCE_DIR> -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
                # 编译器相关配置
                -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                # 具体软件相关变量
                -DJSON_MultipleHeaders=ON
                -DJSON_BuildTests=OFF
            BUILD_ALWAYS FALSE
            EXCLUDE_FROM_ALL TRUE
            BUILD_BYPRODUCTS
            ${CANN_3RD_LIB_PATH}/include/nlohmann/
    )
    if (NOT TARGET json)
        add_library(json INTERFACE)
        set_target_properties(json PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CANN_3RD_LIB_PATH}/include")
        add_dependencies(json ExternalProject_Nlohmann_Json)
    endif ()
else ()
    message(FATAL_ERROR "Failed to get nlohmann_json src path.")
endif ()
