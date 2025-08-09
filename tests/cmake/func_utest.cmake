# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set(TileFwkUTestCaseLibraries         "" CACHE INTERNAL "" FORCE)     # UTest 各模块 用例实现二进制
set(TileFwkUTestCaseLdLibrariesExt    "" CACHE INTERNAL "" FORCE)     # UTest 各模块 额外 Load 二进制
set(TileFwkUTestCaseGTestFilterList   "" CACHE INTERNAL "" FORCE)     # UTest 各模块 GTestFilter 配置

# UTest 添加测试用例二进制库
#[[
Parameters:
  one_value_keywords:
      TARGET                      : [Required] 具体测试用例二进制库名称
  multi_value_keywords:
      SOURCES                     : [Required] 编译源码
      PRIVATE_INCLUDE_DIRECTORIES : [Optional] 头文件搜索路径(PRIVATE)
      PUBLIC_LINK_LIBRARIES       : [Optional] 链接库(PUBLIC)
      GTEST_FILTER_LIST           : [Optional] GTestFilter 配置, Filter 间以 ';' 分割
      LD_LIBRARIES_EXT            : [Optional] 需要在执行时将所在路径配置到环境变量 LD_LIBRARY_PATH 中的 Libraries
Attention:
    1. 单次调用本函数时, 可以通过在 GTEST_FILTER_LIST 中配置多个过滤条件('gtest_filter') 以实现执行多用例;
    2. 一般 LD_LIBRARIES_EXT 内配置的二进制, 在正常 source CANN 包环境变量后, LD_LIBRARY_PATH 内也应包含其所在路径;
]]
function(TileFwk_UTest_AddCaseLib)
    cmake_parse_arguments(
            TMP
            ""
            "TARGET"
            "SOURCES;PRIVATE_INCLUDE_DIRECTORIES;PUBLIC_LINK_LIBRARIES;GTEST_FILTER_LIST;LD_LIBRARIES_EXT"
            ""
            ${ARGN}
    )
    add_Library(${TMP_TARGET} STATIC)
    target_sources(${TMP_TARGET} PRIVATE ${TMP_SOURCES})
    target_include_directories(${TMP_TARGET} PRIVATE ${TMP_PRIVATE_INCLUDE_DIRECTORIES})
    target_link_libraries(${TMP_TARGET}
            PUBLIC
                ${TMP_PUBLIC_LINK_LIBRARIES}
            PRIVATE
                ${TileFwkUTestNamePrefix}_intf_pub
                GTest::gtest
    )
    # 后检查
    TileFwk_AnalysisTargetHeaderFiles(TARGET ${TMP_TARGET})

    set(TileFwkUTestCaseLibraries       ${TileFwkUTestCaseLibraries}       ${TMP_TARGET}            CACHE INTERNAL "" FORCE)
    set(TileFwkUTestCaseLdLibrariesExt  ${TileFwkUTestCaseLdLibrariesExt}  ${TMP_LD_LIBRARIES_EXT}  CACHE INTERNAL "" FORCE)
    set(TileFwkUTestCaseGTestFilterList ${TileFwkUTestCaseGTestFilterList} ${TMP_GTEST_FILTER_LIST} CACHE INTERNAL "" FORCE)
endfunction()

# UTest 执行可执行程序
#[[
Parameters:
  one_value_keywords:
      TARGET             : [Required] 用于指定具体 GTest 可执行目标, 用例会在该目标编译完成后(POST_BUILD)启动执行
  multi_value_keywords:
      LD_LIBRARIES_EXT   : [Optional] 需要在执行时将所在路径配置到环境变量 LD_LIBRARY_PATH 中的 Libraries
      CMD_SETUP_EXT      : [Optional] 附加命令行配置
      GTEST_FILTER_LIST  : [Optional] GTestFilter 配置, Filter 间以 ';' 分割
Attention:
    1. 可以多次调用本函数以添加多个'执行任务'; 单次调用本函数时, 可以通过在 GTEST_FILTER_LIST 中配置多个过滤条件('gtest_filter') 以实现执行多用例;
]]
function(TileFwk_UTest_RunExe)
    cmake_parse_arguments(
            TMP
            ""
            "TARGET"
            "LD_LIBRARIES_EXT;CMD_SETUP_EXT;GTEST_FILTER_LIST"
            ""
            ${ARGN}
    )
    if (ENABLE_TESTS_EXECUTE)
        # 命令行参数处理
        TileFwk_GTest_RunExe_GetPreExecCmdSetup(
                CmdSetup
                TARGET              ${TMP_TARGET}
                LD_LIBRARIES_EXT    ${TMP_LD_LIBRARIES_EXT}
                CMD_SETUP_EXT       ${TMP_CMD_SETUP_EXT}
        )
        # 执行流程
        list(LENGTH TMP_GTEST_FILTER_LIST GtestFilterListLen)
        string(REPLACE ";" ":" GtestFilterStr "${TMP_GTEST_FILTER_LIST}")
        message(STATUS "Run GTest(${TMP_TARGET}), XSAN(ASAN:${ENABLE_ASAN} UBSAN:${ENABLE_UBSAN}), GTestFilter(${GtestFilterListLen})=${GtestFilterStr}")
        set(Comment "Run GTest(${TMP_TARGET}), XSAN(ASAN:${ENABLE_ASAN} UBSAN:${ENABLE_UBSAN})")

        if (TMP_GTEST_FILTER_LIST)
            if (ENABLE_TESTS_EXECUTE_PARALLEL)
                # 仅在使能并行执行全局开关, 且需要做 filter 时才进行执行加速
                set(_File $<TARGET_FILE:${TMP_TARGET}>)
                set(_Args "-t=${_File}" "-c=${GtestFilterStr}" "--xsan_options=${XSAN_OPTIONS}" "--halt_on_error")
                if (ENABLE_TESTS_EXECUTE_PARALLEL_TIMEOUT)
                    list(APPEND _Args "--timeout=${ENABLE_TESTS_EXECUTE_PARALLEL_TIMEOUT}")
                endif ()
                get_filename_component(ParallelPy "${TILE_FWK_SRC_ROOT}/tests/cmake/scripts/utest_accelerate.py" REALPATH)
                add_custom_command(
                        TARGET ${TMP_TARGET} POST_BUILD
                        COMMAND ${CmdSetup} && ${TILE_FWK_PYTHON3_EXE} ${ParallelPy} ARGS ${_Args}
                        COMMENT "${Comment} With Parallel Execute Accelerate"
                )
            else ()
                set(GtestFilterListIdx 1)
                foreach (Filter ${TMP_GTEST_FILTER_LIST})
                    add_custom_command(
                            TARGET ${TMP_TARGET} POST_BUILD
                            COMMAND ${CmdSetup} && ${XSAN_OPTIONS} ./${TMP_TARGET} ARGS '--gtest_filter=${Filter}'
                            COMMENT "${Comment} [${GtestFilterListIdx}/${GtestFilterListLen}] With --gtest_filter=${Filter}"
                    )
                    math(EXPR GtestFilterListIdx "${GtestFilterListIdx} + 1")
                endforeach ()
            endif ()
        else ()
            add_custom_command(
                    TARGET ${TMP_TARGET} POST_BUILD
                    COMMAND ${CmdSetup} && ${XSAN_OPTIONS} ./${TMP_TARGET}
                    COMMENT "${Comment}"
            )
        endif ()
    endif ()
endfunction()

# UTest 添加并执行可执行程序
#[[
Parameters:
  one_value_keywords:
      TARGET                        : [Required] 用于指定具体 GTest 可执行目标, 用例会在该目标编译完成后(POST_BUILD)启动执行
Attention:
    1. 执行本函数后, 会产生名称为 TARGET 内容指定的构建目标, 外部可根据该目标名设置其他 custom_command 或处理依赖关系;
    2. 串行执行场景下 TARGET 内容指定最终 executable 名称;
    3. 并行执行场景下 TARGET 内容指定中间 custom_target 名称, 并行执行的各 executable 会依赖该 custom_target;
]]
function(TileFwk_UTest_AddExe_RunExe)
    cmake_parse_arguments(
            TMP
            ""
            "TARGET"
            ""
            ""
            ${ARGN}
    )
    set(_Sources ${CMAKE_CURRENT_BINARY_DIR}/${TileFwkUTestNamePrefix}_main_stub.cpp)
    execute_process(COMMAND touch ${_Sources})

    set(_PrivateLinkLibraries
            ${TileFwkUTestNamePrefix}_intf_pub
            $<$<BOOL:${ENABLE_BUILD_WITH_CANN}>:${TileFwkUTestNamePrefix}_stubs>
    )

    # 支持由 ENABLE_TESTS_UTEST 传入指定的 Filter
    set(GTestFilterList ${TileFwkUTestCaseGTestFilterList})
    if (NOT "${ENABLE_TESTS_UTEST}" STREQUAL "ON")
        set(GTestFilterList ${ENABLE_TESTS_UTEST})
        string(REPLACE ":" ";" GTestFilterList "${GTestFilterList}")
    endif ()

    list(FILTER TileFwkUTestCaseLibraries         EXCLUDE REGEX "PARALLEL_SEPARATOR")
    list(FILTER TileFwkUTestCaseLdLibrariesExt    EXCLUDE REGEX "PARALLEL_SEPARATOR")
    list(FILTER GTestFilterList                     EXCLUDE REGEX "PARALLEL_SEPARATOR")
    list(REMOVE_DUPLICATES TileFwkUTestCaseLibraries)
    list(REMOVE_DUPLICATES TileFwkUTestCaseLdLibrariesExt)
    list(REMOVE_DUPLICATES GTestFilterList)

    TileFwk_GTest_AddExe(
            TARGET                      ${TMP_TARGET}
            SOURCES                     ${_Sources}
            PRIVATE_INCLUDE_DIRECTORIES ${TMP_PRIVATE_INCLUDE_DIRECTORIES}
            PRIVATE_LINK_LIBRARIES      ${_PrivateLinkLibraries} ${TileFwkUTestCaseLibraries}
    )
    TileFwk_UTest_RunExe(
            TARGET              ${TMP_TARGET}
            CMD_SETUP_EXT       ${TMP_CMD_SETUP_EXT}
            LD_LIBRARIES_EXT    ${TileFwkUTestCaseLdLibrariesExt}
            GTEST_FILTER_LIST   ${GTestFilterList}
    )

    # 生成覆盖率
    TileFwk_GTest_GenerateCoverage(TARGET ${TMP_TARGET})
endfunction()
