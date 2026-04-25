# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_TargetVersion "7.0.0")

if (TARGET msgpackc-cxx)
    return()
endif ()

if (NOT PYPTO_THIRD_PARTY_PATH)
    set(_Msg
            "Failed to get msgpack-c source dir, "
            "need to specify its path through the PYPTO_THIRD_PARTY_PATH (via env/CMake option)"
    )
    string(REPLACE ";" "" _Msg "${_Msg}")
    message(FATAL_ERROR "${_Msg}")
endif ()

get_filename_component(_TargetInstallPrefix "${PYPTO_THIRD_PARTY_PATH}/${CMAKE_BUILD_TYPE}" REALPATH)

if (EXISTS "${_TargetInstallPrefix}/include/msgpack.hpp")
    add_library(msgpackc-cxx INTERFACE)
    target_include_directories(msgpackc-cxx SYSTEM INTERFACE "${_TargetInstallPrefix}/include")
    target_compile_features(msgpackc-cxx INTERFACE cxx_std_17)
    target_compile_definitions(msgpackc-cxx INTERFACE MSGPACK_NO_BOOST)
    message(STATUS "Use msgpack-c from binary, prefix=${_TargetInstallPrefix}")
else ()
    # Find an existing source tree if one is already present.
    set(_TargetSourceDir "")
    foreach (_DirName "msgpack-c" "msgpack-c-cpp-${_TargetVersion}")
        get_filename_component(_Dir "${PYPTO_THIRD_PARTY_PATH}/${_DirName}" REALPATH)
        if (EXISTS "${_Dir}/include/msgpack.hpp")
            set(_TargetSourceDir "${_Dir}")
            break ()
        endif ()
    endforeach ()

    # Clean up empty leftover directories before deciding whether to clone.
    foreach (_DirName "msgpack-c" "msgpack-c-cpp-${_TargetVersion}")
        get_filename_component(_Dir "${PYPTO_THIRD_PARTY_PATH}/${_DirName}" REALPATH)
        PTO_Fwk_CleanEmptyDir(DIR ${_Dir})
    endforeach ()

    set(_ExtArgs)
    if ("${_TargetSourceDir}" STREQUAL "")
        get_filename_component(_TargetSourceDir "${PYPTO_THIRD_PARTY_PATH}/msgpack-c" REALPATH)
        list(APPEND _ExtArgs
                GIT_REPOSITORY "https://github.com/msgpack/msgpack-c.git"
                GIT_TAG "cpp-7.0.0"
                GIT_SHALLOW TRUE
        )
    endif ()

    get_filename_component(_TargetBinaryDir "${PYPTO_THIRD_PARTY_PATH}/${CMAKE_BUILD_TYPE}/build/msgpack-c-${_TargetVersion}" REALPATH)
    ExternalProject_Add(ExternalProject_msgpack_c  ${_ExtArgs}
            PREFIX "${CMAKE_CURRENT_BINARY_DIR}/third_party/msgpack-c-${_TargetVersion}"
            SOURCE_DIR "${_TargetSourceDir}"
            BINARY_DIR "${_TargetBinaryDir}"
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND
                ${CMAKE_COMMAND} -E make_directory "${_TargetInstallPrefix}/include"
                COMMAND ${CMAKE_COMMAND} -E copy_directory "<SOURCE_DIR>/include" "${_TargetInstallPrefix}/include"
            BUILD_ALWAYS FALSE
            EXCLUDE_FROM_ALL TRUE
            BUILD_BYPRODUCTS "${_TargetInstallPrefix}/include/msgpack.hpp"
    )

    add_library(msgpackc-cxx INTERFACE)
    target_include_directories(msgpackc-cxx SYSTEM INTERFACE "${_TargetInstallPrefix}/include")
    target_compile_features(msgpackc-cxx INTERFACE cxx_std_17)
    target_compile_definitions(msgpackc-cxx INTERFACE MSGPACK_NO_BOOST)
    add_dependencies(msgpackc-cxx ExternalProject_msgpack_c)
    message(STATUS "Use msgpack-c from source: ${_TargetSourceDir}")
endif ()
