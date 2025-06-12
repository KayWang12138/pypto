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
function(AscendCpp_GTest_GenerateCoverage)
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
        get_filename_component(GenCoveragePy ${ASCENDCPP_SRC_ROOT}/tests/cmake/scripts/gen_coverage.py REALPATH)
        get_filename_component(GenCoverageDataDir "${ASCENDCPP_BIN_ROOT}" REALPATH)
        set(_Args "-s=${ASCENDCPP_SRC_ROOT}" "-c=${GenCoverageDataDir}")

        get_target_property(GTest_GTest_Inc     GTest::gtest           INTERFACE_INCLUDE_DIRECTORIES)
        get_target_property(GTest_GTestMain_Inc GTest::gtest_main      INTERFACE_INCLUDE_DIRECTORIES)
        if (BUILD_OPEN_PROJECT)
            get_target_property(Json_Inc nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
        else ()
            get_target_property(Json_Inc json                         INTERFACE_INCLUDE_DIRECTORIES)
        endif ()
        set(Filter_Dirs
                ${ASCENDCPP_SRC_ROOT}/tests
                ${ASCENDCPP_SRC_ROOT}/thirdparty
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
                COMMAND ${ASCENDCPP_PYTHON3} ${GenCoveragePy} ARGS ${_Args}
                COMMENT "Generate coverage for ${TMP_TARGET}"
        )
    endif ()
endfunction()

# GTest 获取可执行程序在执行前需要的命令行配置
#[[
Parameters:
  one_value_keywords:
      TARGET             : [Required] 用于指定具体 GTest 可执行目标
  multi_value_keywords:
      LD_LIBRARIES_EXT   : [Optional] 需要在执行时将所在路径配置到环境变量 LD_LIBRARY_PATH 中的 Libraries
      CMD_SETUP_EXT      : [Optional] 附加命令行配置
      CMD_SETUP          : [Required] 输出命令行配置
Attention:
    1. 函数内按照环境变量 LD_LIBRARY_PATH, LD_LIBRARIES_EXT 指定内容, 环境变量 PATH, CMD_SETUP_EXT 顺序处理产生最终 CMD_SETUP;
]]
function(AscendCpp_GTest_RunExe_GetPreExecCmdSetup CMD_SETUP)
    cmake_parse_arguments(
            TMP
            ""
            "TARGET"
            "LD_LIBRARIES_EXT;CMD_SETUP_EXT"
            ""
            ${ARGN}
    )
    set(CmdSetup)

    # 环境变量 LD_LIBRARY_PATH 及变量 LD_LIBRARIES_EXT 处理
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
    list(APPEND CmdSetup export ${LD_LIBRARY_PATH})

    # 环境变量 PATH 处理
    if (NOT "${ASCENDCPP_EXPORT_ENV_PATH}x" STREQUAL "x")
        list(APPEND CmdSetup && export ${ASCENDCPP_EXPORT_ENV_PATH})
    endif ()

    # CMD_SETUP_EXT 处理
    if (NOT "${TMP_CMD_SETUP_EXT}x" STREQUAL "x")
        list(APPEND CmdSetup && ${TMP_CMD_SETUP_EXT})
    endif ()

    # ASAN / UBSAN
    if (ENABLE_ASAN OR ENABLE_UBSAN)
        list(APPEND CmdSetup && ulimit -s 32768)
        if (NOT "${XSAN_LD_PRELOAD}x" STREQUAL "x")
            list(APPEND CmdSetup && export ${XSAN_LD_PRELOAD})
        endif ()
    endif ()

    set(${CMD_SETUP} ${CmdSetup} PARENT_SCOPE)
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
function(AscendCpp_GTest_AddExe)
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
                ${ASCENDCPP_SRC_ROOT}/tests/main.cpp
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
    )
endfunction()
