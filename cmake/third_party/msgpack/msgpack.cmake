# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (NOT BUILD_OPEN_PROJECT)
    return()
endif ()

set(_TargetNameAlias "msgpackc-cxx")
set(_TargetVersion 7.0.0)

# 免重入
if (TARGET ${_TargetNameAlias})
    return()
endif ()

# 异常拦截
if (NOT PYPTO_THIRD_PARTY_PATH)
    set(_Msg
            "Failed to get msgpack-c source dir, "
            "need to specify its path through the PYPTO_THIRD_PARTY_PATH (via env/CMake option)"
    )
    string(REPLACE ";" "" _Msg "${_Msg}")
    message(FATAL_ERROR ${_Msg})
endif ()

# 直接查找制品, 若找到则直接退出
get_filename_component(_TargetTarGzFile "${PYPTO_THIRD_PARTY_PATH}/msgpack-cxx-${_TargetVersion}.tar.gz" REALPATH)
get_filename_component(_TargetInstallPrefix "${PYPTO_THIRD_PARTY_PATH}/${CMAKE_BUILD_TYPE}" REALPATH)
find_path(_MsgpackIncludeDir msgpack.hpp
        PATHS "${_TargetInstallPrefix}/include"
        NO_DEFAULT_PATH
)
if (NOT _MsgpackIncludeDir)
    # 兼容部分镜像直接存放 msgpack 源码的情况
    get_filename_component(_CandidateDir "${PYPTO_THIRD_PARTY_PATH}/msgpack-cxx-${_TargetVersion}" REALPATH)
    if (EXISTS "${_CandidateDir}/include/msgpack.hpp")
        set(_MsgpackIncludeDir "${_CandidateDir}/include")
    else ()
        get_filename_component(_CandidateDir "${PYPTO_THIRD_PARTY_PATH}/msgpack-c" REALPATH)
        if (EXISTS "${_CandidateDir}/include/msgpack.hpp")
            set(_MsgpackIncludeDir "${_CandidateDir}/include")
        endif ()
    endif ()
endif ()
if (_MsgpackIncludeDir)
    message(STATUS "Use msgpack-c from binary, include=${_MsgpackIncludeDir}")
    add_library(${_TargetNameAlias} INTERFACE)
    target_include_directories(${_TargetNameAlias} SYSTEM INTERFACE "${_MsgpackIncludeDir}")
    target_compile_definitions(${_TargetNameAlias} INTERFACE MSGPACK_NO_BOOST)
    return()
endif ()

# 触发编译 (header-only, 仅需解压+安装头文件)
get_filename_component(_TargetSourceDir "${PYPTO_THIRD_PARTY_PATH}/msgpack-c" REALPATH)
if (NOT EXISTS ${_TargetSourceDir})
    get_filename_component(_TargetSourceDir "${PYPTO_THIRD_PARTY_PATH}/msgpack-cxx-${_TargetVersion}" REALPATH)
endif ()

# 若源码目录不存在但本地有 tar.gz, 在 configure 阶段解压到 build 目录
# (避免 ExternalProject_Add 触发网络下载)
if (NOT EXISTS ${_TargetSourceDir})
    set(_TarGz "${PYPTO_THIRD_PARTY_PATH}/msgpack-cxx-${_TargetVersion}.tar.gz")
    if (EXISTS "${_TarGz}")
        set(_ExtractBase "${CMAKE_CURRENT_BINARY_DIR}/third_party")
        file(MAKE_DIRECTORY "${_ExtractBase}")
        message(STATUS "Extracting local msgpack-c archive: ${_TarGz}")
        execute_process(
                COMMAND ${CMAKE_COMMAND} -E tar xzf "${_TarGz}"
                WORKING_DIRECTORY "${_ExtractBase}"
                RESULT_VARIABLE _ExtractResult
        )
        # 兼容不同压缩包的顶层目录名
        foreach(_Name "msgpack-cxx-${_TargetVersion}" "msgpack-c" "msgpack-c-cpp-${_TargetVersion}")
            if (EXISTS "${_ExtractBase}/${_Name}/include/msgpack.hpp")
                set(_TargetSourceDir "${_ExtractBase}/${_Name}")
                break()
            endif ()
        endforeach()
        if (NOT EXISTS "${_TargetSourceDir}/include/msgpack.hpp")
            file(GLOB _ExtractedDirs "${_ExtractBase}/*")
            message(FATAL_ERROR "msgpack-c headers not found after extracting ${_TarGz}. "
                    "Extracted contents: ${_ExtractedDirs}")
        endif ()
    endif ()
endif ()

PTO_Fwk_CleanEmptyDir(DIR ${_TargetSourceDir})

set(_ExtArgs)
if (NOT EXISTS ${_TargetSourceDir})
    list(APPEND _ExtArgs
            URL "https://gitcode.com/cann-src-third-party/msgpack-c/releases/download/cpp-${_TargetVersion}/msgpack-cxx-${_TargetVersion}.tar.gz"
            DOWNLOAD_DIR ${PYPTO_THIRD_PARTY_PATH}
    )
endif ()

ExternalProject_Add(ExternalProject_msgpack_cxx   ${_ExtArgs}
        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/third_party/msgpack-cxx-${_TargetVersion}
        SOURCE_DIR ${_TargetSourceDir}
        INSTALL_DIR ${_TargetInstallPrefix}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include <INSTALL_DIR>/include
        BUILD_ALWAYS FALSE
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        TLS_VERIFY OFF
        EXCLUDE_FROM_ALL TRUE
        BUILD_BYPRODUCTS
            ${_TargetInstallPrefix}/include/msgpack.hpp
)
file(MAKE_DIRECTORY ${_TargetInstallPrefix}/include)
add_library(${_TargetNameAlias} INTERFACE)
target_include_directories(${_TargetNameAlias} SYSTEM INTERFACE "${_TargetInstallPrefix}/include")
target_compile_definitions(${_TargetNameAlias} INTERFACE MSGPACK_NO_BOOST)
add_dependencies(${_TargetNameAlias} ExternalProject_msgpack_cxx)
message(STATUS "Use msgpack-c from source: ${_TargetSourceDir}")
