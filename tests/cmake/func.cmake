# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

# GTest 生成覆盖率
#[[
Parameters:
  one_value_keywords:
      TARGET             : [Required] 指定所依赖的目标(POST_BUILD)
  multi_value_keywords:
      FILTER_DIRECTORIES : [Optional] 覆盖率结果过滤目录
]]
function(TileFwk_GTest_GenerateCoverage)
    cmake_parse_arguments(
            TMP
            ""
            "TARGET"
            "FILTER_DIRECTORIES"
            ""
            ${ARGN}
    )
    if (ENABLE_TESTS_EXECUTE AND ENABLE_GCOV AND BUILD_OPEN_PROJECT)
        # 获取 gcc 默认头文件搜索路径
        execute_process(
                COMMAND ${CMAKE_C_COMPILER} --print-sysroot
                RESULT_VARIABLE _RST
                OUTPUT_VARIABLE _SUFFIX
                ERROR_QUIET
        )
        if (_RST)
            get_filename_component(SYS_ROOT "/usr/include" REALPATH)
        else ()
            get_filename_component(SYS_ROOT "${_SUFFIX}/usr/include" REALPATH)
        endif ()

        # 参数组织
        find_program(LCOV lcov REQUIRED)
        get_filename_component(GenCoveragePy ${TILE_FWK_SRC_ROOT}/tests/cmake/scripts/python/gen_coverage.py REALPATH)
        get_filename_component(GenCoverageDataDir "${TILE_FWK_BIN_ROOT}" REALPATH)
        set(_Args "-s=${TILE_FWK_SRC_ROOT}" "-c=${GenCoverageDataDir}")

        get_target_property(GTest_GTest_Inc     GTest::gtest           INTERFACE_INCLUDE_DIRECTORIES)
        get_target_property(GTest_GTestMain_Inc GTest::gtest_main      INTERFACE_INCLUDE_DIRECTORIES)
        if (BUILD_OPEN_PROJECT)
            get_target_property(Json_Inc nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
        else ()
            get_target_property(Json_Inc json                         INTERFACE_INCLUDE_DIRECTORIES)
        endif ()
        set(Filter_Dirs
                ${TILE_FWK_SRC_ROOT}/tests
                ${TILE_FWK_SRC_ROOT}/third_party
                ${GTest_GTest_Inc}
                ${GTest_GTestMain_Inc}
                ${Json_Inc}
                ${SYS_ROOT}
                ${TMP_FILTER_DIRECTORIES}
        )
        foreach (_dir ${Filter_Dirs})
            list(APPEND _Args "-f=${_dir}")
        endforeach ()
        list(REMOVE_DUPLICATES _Args)

        add_custom_command(
                TARGET ${TMP_TARGET} POST_BUILD
                COMMAND ${TILE_FWK_PYTHON3_EXE} ${GenCoveragePy} ARGS ${_Args}
                COMMENT "Generate coverage for ${TMP_TARGET}"
        )
    endif ()
endfunction()

# GTest 获取可执行程序在执行前需要的命令行配置和环境变量配置
#[[
Parameters:
  one_value_keywords:
      TARGET             : [Required] 用于指定具体 GTest 可执行目标
  multi_value_keywords:
      PY_CMD_SETUP       : [Required] 输出 Python 场景命令行配置
      PY_ENV_LINES       : [Required] 输出 Python 场景环境变量(按照 "K=V" 格式组织)
      BASH_CMD_SETUP     : [Required] 输出 bash   场景命令行配置(内部包含命令行配置+环境变量配置)

      LD_LIBRARIES_EXT   : [Optional] 需要在执行时将所在路径配置到环境变量 LD_LIBRARY_PATH 中的 Libraries
      CMD_SETUP_EXT      : [Optional] 附加命令行配置, 要求调用者设置 export 及多命令行配置间的 && 连接
      ENV_LINES_EXT      : [Optional] 附加环境变量配置(按照 "K=V" 格式组织)
]]
function(TileFwk_GTest_RunExe_GetPreExecSetup PY_CMD_SETUP PY_ENV_LINES BASH_CMD_SETUP)
    cmake_parse_arguments(
            TMP
            ""
            "TARGET"
            "LD_LIBRARIES_EXT;CMD_SETUP_EXT;ENV_LINES_EXT"
            ""
            ${ARGN}
    )

    # 命令行
    set(CmdSetup)
    # 处理变量 CMD_SETUP_EXT
    if (NOT "${TMP_CMD_SETUP_EXT}x" STREQUAL "x")
        list(APPEND CmdSetup ${TMP_CMD_SETUP_EXT})
    endif ()
    # 处理变量内部处理(XSan 相关处理)
    if (ENABLE_ASAN OR ENABLE_UBSAN)
        if (NOT "${CmdSetup}x" STREQUAL "x")
            list(APPEND CmdSetup &&)
        endif ()
        list(APPEND CmdSetup ulimit -s 32768)
    endif ()

    # 环境变量
    set(EnvLines)
    # 处理变量 LD_LIBRARIES_EXT 及环境变量 LD_LIBRARY_PATH
    set(LD_LIBRARY_PATH_EXT)
    foreach (LIBRARY ${TMP_LD_LIBRARIES_EXT})
        add_dependencies(${TMP_TARGET} ${LIBRARY})
        list(APPEND LD_LIBRARY_PATH_EXT "$<TARGET_FILE_DIR:${LIBRARY}>")
    endforeach ()
    string(REPLACE ";" ":" LD_LIBRARY_PATH_EXT "${LD_LIBRARY_PATH_EXT}")
    set(LD_LIBRARY_PATH "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}")
    if (NOT "${LD_LIBRARY_PATH_EXT}x" STREQUAL "x")
        set(LD_LIBRARY_PATH "LD_LIBRARY_PATH=${LD_LIBRARY_PATH_EXT}:$ENV{LD_LIBRARY_PATH}")
    endif ()
    list(APPEND EnvLines ${LD_LIBRARY_PATH})
    # 处理环境变量 PATH
    if ((NOT BUILD_OPEN_PROJECT) AND ENABLE_TESTS_UTEST)
        list(APPEND EnvLines "PATH=$ENV{PATH}:${CCEC_PATH}")
    endif()
    # 处理变量 ENV_SETUP_EXT
    list(REMOVE_ITEM TMP_ENV_LINES_EXT export)
    list(REMOVE_ITEM TMP_ENV_LINES_EXT &)
    list(REMOVE_ITEM TMP_ENV_LINES_EXT &&)
    if (NOT "${TMP_ENV_LINES_EXT}x" STREQUAL "x")
        list(APPEND EnvLines ${TMP_ENV_LINES_EXT})
    endif ()
    # 处理 ASAN / UBSAN 场景
    if (ENABLE_ASAN OR ENABLE_UBSAN)
        if (NOT "${XSAN_LD_PRELOAD}x" STREQUAL "x")
            list(APPEND EnvLines ${XSAN_LD_PRELOAD})
        endif ()
    endif ()

    # 输出处理
    set(PyCmdSetup ${CmdSetup})
    if (NOT "${PyCmdSetup}x" STREQUAL "x")
        list(APPEND PyCmdSetup &&)
    endif ()
    set(${PY_CMD_SETUP} ${PyCmdSetup} PARENT_SCOPE)

    # 输出处理
    set(XSan_Options)
    if (ENABLE_ASAN OR ENABLE_UBSAN)
        set(XSan_Options ${ASAN_OPTIONS} ${UBSAN_OPTIONS})
    endif ()
    set(PyEnvLines ${EnvLines} ${XSan_Options})
    set(${PY_ENV_LINES} ${PyEnvLines} PARENT_SCOPE)

    # 输出处理
    set(BashSetup)
    if (NOT "${EnvLines}x" STREQUAL "x")
        foreach (_line ${EnvLines})
            if (NOT "${BashSetup}x" STREQUAL "x")
                list(APPEND BashSetup &&)
            endif ()
            list(APPEND BashSetup export ${_line})
        endforeach ()
    endif ()
    if (NOT "${CmdSetup}x" STREQUAL "x")
        if (NOT "${BashSetup}x" STREQUAL "x")
            list(APPEND BashSetup &&)
        endif ()
        list(APPEND BashSetup ${CmdSetup})
    endif ()
    # XSan 特殊处理(XSan_Options)
    # 1. 当其存在时, 其需要在 BashSetup 尾部, 其前需要 && 与前述命令行连接, 其后不需要补充 && 连接符
    # 2. 当不存在时, BashSetup 尾部需要补充 && 连接符;
    if (NOT "${BashSetup}x" STREQUAL "x")
        list(APPEND BashSetup &&)
    endif ()
    if (NOT "${XSan_Options}x" STREQUAL "x")
        list(APPEND BashSetup ${XSan_Options})
    endif ()
    set(${BASH_CMD_SETUP} ${BashSetup} PARENT_SCOPE)
endfunction()

# GTest 添加可执行程序
#[[
Parameters:
  one_value_keywords:
      TARGET                        : [Required] 用于指定具体 GTest 可执行目标, 用例会在该目标编译完成后(POST_BUILD)启动执行
  multi_value_keywords:
      SOURCES                       : [Optional] 额外的编译源码
      PRIVATE_INCLUDE_DIRECTORIES   : [Optional] Private 头文件查找路径
      PRIVATE_LINK_LIBRARIES        : [Optional] Private 链接库
]]
function(TileFwk_GTest_AddExe)
    cmake_parse_arguments(
            TMP
            ""
            "TARGET"
            "SOURCES;PRIVATE_INCLUDE_DIRECTORIES;PRIVATE_LINK_LIBRARIES"
            ""
            ${ARGN}
    )
    add_executable(${TMP_TARGET})
    target_sources(${TMP_TARGET}
            PRIVATE
                ${TMP_SOURCES}
                ${TILE_FWK_SRC_ROOT}/tests/main.cpp
    )
    target_include_directories(${TMP_TARGET}
            PRIVATE
                ${TMP_PRIVATE_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(${TMP_TARGET}
            PRIVATE
                GTest::gtest
                -Wl,--no-as-needed
                -Wl,--whole-archive
                ${TMP_PRIVATE_LINK_LIBRARIES}
                -Wl,--as-needed
                -Wl,--no-whole-archive
                -rdynamic
    )
    add_custom_command(
        TARGET ${TMP_TARGET} POST_BUILD
        COMMAND mkdir -p "${TILE_FWK_BIN_ROOT}/src/conf"
        COMMAND ln -sf "${TILE_FWK_SRC_ROOT}/src/interface/configs/tile_fwk_config.json" "${TILE_FWK_BIN_ROOT}/src/conf/tile_fwk_config.json"
        COMMAND ln -sf "${TILE_FWK_SRC_ROOT}/src/passes/pass_config/tile_fwk_platform_info.json" "${TILE_FWK_BIN_ROOT}/src/conf/tile_fwk_platform_info.json"
        COMMENT "Soft link of tile_fwk_config.json and tile_fwk_platform_info.json has been created at ${TILE_FWK_BIN_ROOT}/src/conf"
    )
endfunction()
