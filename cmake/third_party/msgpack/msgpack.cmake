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

# msgpack-c is a header-only library, look for it in third_party_path

# 优先查找预编译制品
get_filename_component(_TargetInstallPrefix "${PYPTO_THIRD_PARTY_PATH}/${CMAKE_BUILD_TYPE}" REALPATH)
find_path(_MsgpackIncludeDir msgpack.hpp
        PATHS
            "${_TargetInstallPrefix}/include"
            "${PYPTO_THIRD_PARTY_PATH}/include"
        NO_DEFAULT_PATH
)
if (_MsgpackIncludeDir)
    message(STATUS "Use msgpack-c from binary, include=${_MsgpackIncludeDir}")
    add_library(${_TargetNameAlias} INTERFACE)
    target_include_directories(${_TargetNameAlias} SYSTEM INTERFACE "${_MsgpackIncludeDir}")
    target_compile_definitions(${_TargetNameAlias} INTERFACE MSGPACK_NO_BOOST)
    return()
endif ()

# 查找源码目录
get_filename_component(_MsgpackSourceDir "${PYPTO_THIRD_PARTY_PATH}/msgpack-c" REALPATH)
if (NOT EXISTS "${_MsgpackSourceDir}/include/msgpack.hpp")
    get_filename_component(_MsgpackSourceDir "${PYPTO_THIRD_PARTY_PATH}/msgpack-cxx-${_TargetVersion}" REALPATH)
endif ()

if (NOT EXISTS "${_MsgpackSourceDir}/include/msgpack.hpp")
    set(_MsgpackTarGz "${PYPTO_THIRD_PARTY_PATH}/msgpack-cxx-${_TargetVersion}.tar.gz")

    # 若本地已有 tar.gz 则跳过下载, 直接解压
    if (NOT EXISTS "${_MsgpackTarGz}")
        set(_MsgpackUrl "https://gitcode.com/cann-src-third-party/msgpack-c/releases/download/cpp-${_TargetVersion}/msgpack-cxx-${_TargetVersion}.tar.gz")
        message(STATUS "Downloading msgpack-c ${_TargetVersion} from ${_MsgpackUrl}")
        file(DOWNLOAD ${_MsgpackUrl} ${_MsgpackTarGz}
                TLS_VERIFY OFF
                STATUS _DownloadStatus
        )
        list(GET _DownloadStatus 0 _DownloadStatusCode)
        if (NOT _DownloadStatusCode EQUAL 0)
            list(GET _DownloadStatus 1 _DownloadStatusMsg)
            message(FATAL_ERROR "Failed to download msgpack-c: ${_DownloadStatusMsg}")
        endif ()
    else ()
        message(STATUS "Found local msgpack-c archive: ${_MsgpackTarGz}")
    endif ()

    message(STATUS "Extracting msgpack-c ${_TargetVersion}")
    execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf ${_MsgpackTarGz}
            WORKING_DIRECTORY ${PYPTO_THIRD_PARTY_PATH}
            RESULT_VARIABLE _ExtractResult
    )
    if (NOT _ExtractResult EQUAL 0)
        message(FATAL_ERROR "Failed to extract msgpack-c")
    endif ()

    # 解压后查找源码目录 (兼容不同压缩包的顶层目录名)
    set(_MsgpackSourceDir "")
    foreach(_Candidate
            "${PYPTO_THIRD_PARTY_PATH}/msgpack-cxx-${_TargetVersion}"
            "${PYPTO_THIRD_PARTY_PATH}/msgpack-c"
            "${PYPTO_THIRD_PARTY_PATH}/msgpack-c-cpp-${_TargetVersion}")
        if (EXISTS "${_Candidate}/include/msgpack.hpp")
            get_filename_component(_MsgpackSourceDir "${_Candidate}" REALPATH)
            break()
        endif ()
    endforeach()
    if (NOT _MsgpackSourceDir)
        file(GLOB _MsgpackDirs "${PYPTO_THIRD_PARTY_PATH}/msgpack*")
        message(FATAL_ERROR "msgpack-c not found after extraction. "
                "Searched: msgpack-cxx-${_TargetVersion}, msgpack-c, msgpack-c-cpp-${_TargetVersion}. "
                "Existing msgpack* entries: ${_MsgpackDirs}")
    endif ()
endif ()

# Create imported interface target for header-only library
add_library(${_TargetNameAlias} INTERFACE)
target_include_directories(${_TargetNameAlias} SYSTEM INTERFACE "${_MsgpackSourceDir}/include")

# Disable Boost dependency in msgpack (use standalone mode)
target_compile_definitions(${_TargetNameAlias} INTERFACE MSGPACK_NO_BOOST)

message(STATUS "msgpack-c configured from: ${_MsgpackSourceDir}")
