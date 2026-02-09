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
get_filename_component(_MsgpackSourceDir "${PYPTO_THIRD_PARTY_PATH}/msgpack-c" REALPATH)
if (NOT EXISTS "${_MsgpackSourceDir}/include/msgpack.hpp")
    get_filename_component(_MsgpackSourceDir "${PYPTO_THIRD_PARTY_PATH}/msgpack-c-cpp-${_TargetVersion}" REALPATH)
endif ()

if (NOT EXISTS "${_MsgpackSourceDir}/include/msgpack.hpp")
    # 触发下载
    set(_MsgpackTarGz "${PYPTO_THIRD_PARTY_PATH}/msgpack-c-cpp-${_TargetVersion}.tar.gz")
    set(_MsgpackUrl "https://github.com/msgpack/msgpack-c/archive/refs/tags/cpp-${_TargetVersion}.tar.gz")

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

    message(STATUS "Extracting msgpack-c ${_TargetVersion}")
    execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf ${_MsgpackTarGz}
            WORKING_DIRECTORY ${PYPTO_THIRD_PARTY_PATH}
            RESULT_VARIABLE _ExtractResult
    )
    if (NOT _ExtractResult EQUAL 0)
        message(FATAL_ERROR "Failed to extract msgpack-c")
    endif ()

    get_filename_component(_MsgpackSourceDir "${PYPTO_THIRD_PARTY_PATH}/msgpack-c-cpp-${_TargetVersion}" REALPATH)
    if (NOT EXISTS "${_MsgpackSourceDir}/include/msgpack.hpp")
        message(FATAL_ERROR "msgpack-c not found after download at ${_MsgpackSourceDir}. "
                "Please place msgpack-c source in ${PYPTO_THIRD_PARTY_PATH}/msgpack-c/")
    endif ()
endif ()

# Create imported interface target for header-only library
add_library(${_TargetNameAlias} INTERFACE)
target_include_directories(${_TargetNameAlias} SYSTEM INTERFACE "${_MsgpackSourceDir}/include")

# Disable Boost dependency in msgpack (use standalone mode)
target_compile_definitions(${_TargetNameAlias} INTERFACE MSGPACK_NO_BOOST)

message(STATUS "msgpack-c configured from: ${_MsgpackSourceDir}")
