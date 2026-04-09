# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_TargetVersion "1.0")

if (TARGET libbacktrace)
    return()
endif ()

if (NOT PYPTO_THIRD_PARTY_PATH)
    set(_Msg
            "Failed to get libbacktrace source dir, "
            "need to specify its path through the PYPTO_THIRD_PARTY_PATH (via env/CMake option)"
    )
    string(REPLACE ";" "" _Msg "${_Msg}")
    message(FATAL_ERROR "${_Msg}")
endif ()

get_filename_component(_TargetInstallPrefix "${PYPTO_THIRD_PARTY_PATH}/${CMAKE_BUILD_TYPE}" REALPATH)
set(LIBBACKTRACE_INSTALL_DIR "${_TargetInstallPrefix}")

if (EXISTS "${_TargetInstallPrefix}/lib/libbacktrace.a" AND EXISTS "${_TargetInstallPrefix}/include/backtrace.h")
    add_library(libbacktrace STATIC IMPORTED)
    set_target_properties(libbacktrace PROPERTIES
            IMPORTED_LOCATION "${_TargetInstallPrefix}/lib/libbacktrace.a"
            INTERFACE_INCLUDE_DIRECTORIES "${_TargetInstallPrefix}/include"
    )
    message(STATUS "Use libbacktrace from binary, prefix=${_TargetInstallPrefix}")
else ()
    # 查找已有源码目录（支持 libbacktrace/ 和 libbacktrace-1.0/ 两种命名）
    set(_TargetSourceDir "")
    foreach (_DirName "libbacktrace" "libbacktrace-${_TargetVersion}")
        get_filename_component(_Dir "${PYPTO_THIRD_PARTY_PATH}/${_DirName}" REALPATH)
        if (EXISTS "${_Dir}/configure")
            set(_TargetSourceDir "${_Dir}")
            break ()
        endif ()
    endforeach ()

    # 清理残留的空目录
    foreach (_DirName "libbacktrace" "libbacktrace-${_TargetVersion}")
        get_filename_component(_Dir "${PYPTO_THIRD_PARTY_PATH}/${_DirName}" REALPATH)
        PTO_Fwk_CleanEmptyDir(DIR ${_Dir})
    endforeach ()

    get_filename_component(_TargetBinaryDir "${PYPTO_THIRD_PARTY_PATH}/${CMAKE_BUILD_TYPE}/build/libbacktrace-${_TargetVersion}" REALPATH)
    file(MAKE_DIRECTORY "${_TargetInstallPrefix}/include")
    file(MAKE_DIRECTORY "${_TargetInstallPrefix}/lib")

    set(_ExtArgs)
    if ("${_TargetSourceDir}" STREQUAL "")
        # 源码不存在，使用 git clone 自动下载（目录名 libbacktrace，与 submodule 一致）
        get_filename_component(_TargetSourceDir "${PYPTO_THIRD_PARTY_PATH}/libbacktrace" REALPATH)
        list(APPEND _ExtArgs
                GIT_REPOSITORY "https://github.com/Hzfengsy/libbacktrace.git"
                GIT_TAG "macho-bundle-support"
                GIT_SHALLOW TRUE
        )
    endif ()

    ExternalProject_Add(ExternalProject_libbacktrace  ${_ExtArgs}
            PREFIX "${CMAKE_CURRENT_BINARY_DIR}/third_party/libbacktrace-${_TargetVersion}"
            SOURCE_DIR "${_TargetSourceDir}"
            BINARY_DIR "${_TargetBinaryDir}"
            CONFIGURE_COMMAND "${_TargetSourceDir}/configure" "--prefix=${_TargetInstallPrefix}" --with-pic
            BUILD_COMMAND make
            INSTALL_COMMAND make install
            BUILD_ALWAYS FALSE
            EXCLUDE_FROM_ALL TRUE
            BUILD_BYPRODUCTS
                "${_TargetInstallPrefix}/lib/libbacktrace.a"
                "${_TargetInstallPrefix}/include/backtrace.h"
    )

    add_library(libbacktrace STATIC IMPORTED)
    set_target_properties(libbacktrace PROPERTIES
            IMPORTED_LOCATION "${_TargetInstallPrefix}/lib/libbacktrace.a"
            INTERFACE_INCLUDE_DIRECTORIES "${_TargetInstallPrefix}/include"
    )
    add_dependencies(libbacktrace ExternalProject_libbacktrace)
    message(STATUS "Use libbacktrace from source: ${_TargetSourceDir}")
endif ()

function(pypto_add_apple_dsymutil target_name)
    if(APPLE)
        find_program(DSYMUTIL dsymutil)
        mark_as_advanced(DSYMUTIL)
        if(DSYMUTIL)
            add_custom_command(
                TARGET ${target_name}
                POST_BUILD
                COMMAND ${DSYMUTIL} ARGS $<TARGET_FILE:${target_name}>
                COMMENT "Generating dSYM for ${target_name}"
                VERBATIM
            )
        endif()
    endif()
endfunction()
