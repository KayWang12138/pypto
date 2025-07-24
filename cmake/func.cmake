# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================


function(TileFwk_Debug_List)
    cmake_parse_arguments(
            TMP
            ""
            "NAME;DETAIL"
            "LIST"
            ""
            ${ARGN}
    )
    list(LENGTH TMP_LIST _Len)
    message(STATUS "${TMP_NAME}: Length: ${_Len}")
    if (TMP_DETAIL)
        foreach (_f ${TMP_LIST})
            message(STATUS "${_f}")
        endforeach ()
    endif ()
endfunction()

function(TileFwk_AnalysisTargetSymbols)
    cmake_parse_arguments(
            TMP
            "DF;IGNORE_UDF_SELF;IGNORE_UDF_PASSED"
            "TARGET"
            ""
            ""
            ${ARGN}
    )
    if (BUILD_OPEN_PROJECT
            AND (ENABLE_TESTS_UTEST OR ENABLE_TESTS_STEST OR ENABLE_TESTS_STEST_DISTRIBUTED)
            AND (CMAKE_GENERATOR STREQUAL "Unix Makefiles")
            AND (CMAKE_C_COMPILER_ID STREQUAL "GNU"))
        set(_file $<TARGET_FILE:${TMP_TARGET}>)
        get_filename_component(_PyScript "${TILE_FWK_SRC_ROOT}/cmake/scripts/analysis_binary_symbol.py" REALPATH)
        set(_Args "-f=${_file}")
        if (TMP_DF)
            list(APPEND _Args "--print_defined_relations")
        endif ()
        if (TMP_IGNORE_UDF_SELF)
            list(APPEND _Args "--ignore_undefined_symbols_self")
        endif ()
        if (NOT TMP_IGNORE_UDF_PASSED)
            list(APPEND _Args "--ignore_undefined_symbols_pass")
        endif ()
        add_custom_command(
                TARGET ${TMP_TARGET} POST_BUILD
                COMMAND ${TILE_FWK_PYTHON3_EXE} ${_PyScript} ARGS ${_Args}
                COMMENT "Analysis symbol of ${TMP_TARGET}"
        )
    endif ()
endfunction()

function(TileFwk_AnalysisTargetHeaderFiles)
    cmake_parse_arguments(
            TMP
            ""
            "TARGET"
            ""
            ""
            ${ARGN}
    )
    if (BUILD_OPEN_PROJECT
            AND (ENABLE_TESTS_UTEST OR ENABLE_TESTS_STEST OR ENABLE_TESTS_STEST_DISTRIBUTED)
            AND (CMAKE_GENERATOR STREQUAL "Unix Makefiles")
            AND (CMAKE_C_COMPILER_ID STREQUAL "GNU"))
        set(_file $<TARGET_FILE:${TMP_TARGET}>)
        set(_objects $<TARGET_OBJECTS:${TMP_TARGET}>)
        get_filename_component(_PyScript "${TILE_FWK_SRC_ROOT}/cmake/scripts/analysis_binary_header_files.py" REALPATH)
        get_filename_component(_JsonCfg "${TILE_FWK_SRC_ROOT}/cmake/scripts/analysis_binary_header_files.json" REALPATH)
        set(_Args
                "-s=${TILE_FWK_SRC_ROOT}"
                "-b=${TILE_FWK_BIN_ROOT}"
                "-t=${_file}"
                "-o='${_objects}'"
                "-j=${_JsonCfg}"
        )
        # 获取 gcc 默认头文件搜索路径
        execute_process(
                COMMAND ${CMAKE_C_COMPILER} --print-sysroot
                RESULT_VARIABLE _RST
                OUTPUT_VARIABLE _SUFFIX
                ERROR_QUIET
        )
        list(APPEND _Args "-f=${SYS_ROOT}/usr/include")
        list(APPEND _Args "-f=${SYS_ROOT}/usr/lib")
        # CANN
        if (NOT AC_ENABLE_FRAMEWORK_WITHOUT_CANN)
            list(APPEND _Args "-f=${ASCEND_CANN_PACKAGE_PATH}/include")
        endif ()
        # OpenSource
        if (BUILD_OPEN_PROJECT)
            get_target_property(json_inc nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
            get_target_property(c_sec_inc boundscheck INTERFACE_INCLUDE_DIRECTORIES)
        else ()
            get_target_property(json_inc json INTERFACE_INCLUDE_DIRECTORIES)
            get_target_property(c_sec_inc c_sec INTERFACE_INCLUDE_DIRECTORIES)
        endif ()
        set(OpenSourceInc ${json_inc} ${c_sec_inc})
        foreach (_Inc ${OpenSourceInc})
            list(APPEND _Args "-f=${_Inc}")
        endforeach ()
        list(REMOVE_DUPLICATES _Args)

        add_custom_command(
                TARGET ${TMP_TARGET} POST_BUILD
                COMMAND ${TILE_FWK_PYTHON3_EXE} ${_PyScript} ARGS ${_Args}
                COMMENT "Analysis Header-File of ${TMP_TARGET}"
        )
    endif ()
endfunction()
