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

# msgpack-cxx 是 header-only 库, 只需定位到 include 目录即可

set(_MsgpackIncludeDir "")

# 1) 查找已安装的头文件 (预编译制品)
get_filename_component(_TargetInstallPrefix "${PYPTO_THIRD_PARTY_PATH}/${CMAKE_BUILD_TYPE}" REALPATH)
find_path(_MsgpackIncludeDir msgpack.hpp
        PATHS "${_TargetInstallPrefix}/include"
        NO_DEFAULT_PATH
)

# 2) 查找已解压的源码目录
if (NOT _MsgpackIncludeDir)
    foreach(_Name "msgpack-cxx-${_TargetVersion}" "msgpack-c")
        if (EXISTS "${PYPTO_THIRD_PARTY_PATH}/${_Name}/include/msgpack.hpp")
            set(_MsgpackIncludeDir "${PYPTO_THIRD_PARTY_PATH}/${_Name}/include")
            break()
        endif ()
    endforeach()
endif ()

# 3) 从本地 tar.gz 解压到 build 目录
if (NOT _MsgpackIncludeDir)
    set(_TarGz "${PYPTO_THIRD_PARTY_PATH}/msgpack-cxx-${_TargetVersion}.tar.gz")
    if (NOT EXISTS "${_TarGz}")
        # 诊断: 列出 PYPTO_THIRD_PARTY_PATH 下所有 msgpack 相关文件
        file(GLOB _MsgpackFiles "${PYPTO_THIRD_PARTY_PATH}/msgpack*")
        message(STATUS "msgpack tar.gz not found at: ${_TarGz}")
        message(STATUS "PYPTO_THIRD_PARTY_PATH=${PYPTO_THIRD_PARTY_PATH}")
        message(STATUS "msgpack* files in PYPTO_THIRD_PARTY_PATH: ${_MsgpackFiles}")
    endif ()
    if (EXISTS "${_TarGz}")
        set(_ExtractBase "${CMAKE_CURRENT_BINARY_DIR}/third_party/msgpack")
        file(MAKE_DIRECTORY "${_ExtractBase}")
        message(STATUS "Extracting local msgpack-c archive: ${_TarGz} -> ${_ExtractBase}")
        execute_process(
                COMMAND ${CMAKE_COMMAND} -E tar xzf "${_TarGz}"
                WORKING_DIRECTORY "${_ExtractBase}"
                RESULT_VARIABLE _ExtractResult
        )
        if (NOT _ExtractResult EQUAL 0)
            message(WARNING "Failed to extract ${_TarGz}, result=${_ExtractResult}")
        endif ()
        # 兼容不同压缩包的顶层目录名
        foreach(_Name "msgpack-cxx-${_TargetVersion}" "msgpack-c" "msgpack-c-cpp-${_TargetVersion}")
            if (EXISTS "${_ExtractBase}/${_Name}/include/msgpack.hpp")
                set(_MsgpackIncludeDir "${_ExtractBase}/${_Name}/include")
                break()
            endif ()
        endforeach()
        # 若以上都没匹配, 尝试直接在解压目录下查找 (压缩包可能没有顶层目录)
        if (NOT _MsgpackIncludeDir AND EXISTS "${_ExtractBase}/include/msgpack.hpp")
            set(_MsgpackIncludeDir "${_ExtractBase}/include")
        endif ()
        if (NOT _MsgpackIncludeDir)
            file(GLOB _ExtractedEntries "${_ExtractBase}/*")
            message(WARNING "msgpack-c headers not found after extracting ${_TarGz}. "
                    "Extracted contents: ${_ExtractedEntries}")
        endif ()
    endif ()
endif ()

# 4) 最终兜底: 下载 tar.gz 并解压
if (NOT _MsgpackIncludeDir)
    set(_TarGz "${CMAKE_CURRENT_BINARY_DIR}/third_party/msgpack-cxx-${_TargetVersion}.tar.gz")
    set(_MsgpackUrl "https://gitcode.com/cann-src-third-party/msgpack-c/releases/download/cpp-${_TargetVersion}/msgpack-cxx-${_TargetVersion}.tar.gz")
    message(STATUS "Downloading msgpack-c ${_TargetVersion} from ${_MsgpackUrl}")
    file(DOWNLOAD ${_MsgpackUrl} ${_TarGz}
            TLS_VERIFY OFF
            STATUS _DownloadStatus
    )
    list(GET _DownloadStatus 0 _DownloadStatusCode)
    if (NOT _DownloadStatusCode EQUAL 0)
        list(GET _DownloadStatus 1 _DownloadStatusMsg)
        message(FATAL_ERROR "Failed to download msgpack-c: ${_DownloadStatusMsg}")
    endif ()

    set(_ExtractBase "${CMAKE_CURRENT_BINARY_DIR}/third_party/msgpack")
    file(MAKE_DIRECTORY "${_ExtractBase}")
    execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf "${_TarGz}"
            WORKING_DIRECTORY "${_ExtractBase}"
            RESULT_VARIABLE _ExtractResult
    )
    if (NOT _ExtractResult EQUAL 0)
        message(FATAL_ERROR "Failed to extract msgpack-c")
    endif ()
    foreach(_Name "msgpack-cxx-${_TargetVersion}" "msgpack-c" "msgpack-c-cpp-${_TargetVersion}")
        if (EXISTS "${_ExtractBase}/${_Name}/include/msgpack.hpp")
            set(_MsgpackIncludeDir "${_ExtractBase}/${_Name}/include")
            break()
        endif ()
    endforeach()
    if (NOT _MsgpackIncludeDir AND EXISTS "${_ExtractBase}/include/msgpack.hpp")
        set(_MsgpackIncludeDir "${_ExtractBase}/include")
    endif ()
endif ()

if (NOT _MsgpackIncludeDir)
    message(FATAL_ERROR "Failed to find msgpack-c headers. "
            "Please place msgpack-cxx source or tar.gz in ${PYPTO_THIRD_PARTY_PATH}/")
endif ()

# header-only: 直接创建 interface target, 无需编译
add_library(${_TargetNameAlias} INTERFACE)
target_include_directories(${_TargetNameAlias} SYSTEM INTERFACE "${_MsgpackIncludeDir}")
target_compile_definitions(${_TargetNameAlias} INTERFACE MSGPACK_NO_BOOST)
message(STATUS "msgpack-c configured, include=${_MsgpackIncludeDir}")
