# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set(PTO_Fwk_UTestCaseLibraries         "" CACHE INTERNAL "" FORCE)     # UTest 各模块 用例实现二进制
set(PTO_Fwk_UTestCaseLdLibrariesExt    "" CACHE INTERNAL "" FORCE)     # UTest 各模块 额外 Load 二进制
set(PTO_Fwk_UTestCaseGTestFilterList   "" CACHE INTERNAL "" FORCE)     # UTest 各模块 GTestFilter 配置

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
function(PTO_Fwk_UTest_AddCaseLib)
    cmake_parse_arguments(
            ARG
            ""
            "TARGET"
            "SOURCES;PRIVATE_INCLUDE_DIRECTORIES;PUBLIC_LINK_LIBRARIES;GTEST_FILTER_LIST;LD_LIBRARIES_EXT"
            ""
            ${ARGN}
    )
    add_Library(${ARG_TARGET} STATIC)
    target_sources(${ARG_TARGET} PRIVATE ${ARG_SOURCES})
    target_include_directories(${ARG_TARGET} PRIVATE ${ARG_PRIVATE_INCLUDE_DIRECTORIES})
    target_link_libraries(${ARG_TARGET}
            PUBLIC
                ${ARG_PUBLIC_LINK_LIBRARIES}
            PRIVATE
                ${PTO_Fwk_UTestNamePrefix}_intf_pub
                GTest::gtest
    )
    # 后检查
    TileFwk_AnalysisTargetHeaderFiles(TARGET ${ARG_TARGET})

    set(PTO_Fwk_UTestCaseLibraries       ${PTO_Fwk_UTestCaseLibraries}       ${ARG_TARGET}            CACHE INTERNAL "" FORCE)
    set(PTO_Fwk_UTestCaseLdLibrariesExt  ${PTO_Fwk_UTestCaseLdLibrariesExt}  ${ARG_LD_LIBRARIES_EXT}  CACHE INTERNAL "" FORCE)
    set(PTO_Fwk_UTestCaseGTestFilterList ${PTO_Fwk_UTestCaseGTestFilterList} ${ARG_GTEST_FILTER_LIST} CACHE INTERNAL "" FORCE)
endfunction()

# UTest 执行可执行程序
#[[
Parameters:
  one_value_keywords:
      TARGET             : [Required] 用于指定具体 GTest 可执行目标, 用例会在该目标编译完成后(POST_BUILD)启动执行
  multi_value_keywords:
      LD_LIBRARIES_EXT   : [Optional] 需要在执行时将所在路径配置到环境变量 LD_LIBRARY_PATH 中的 Libraries
      ENV_LINES_EXT      : [Optional] 需要额外配置的环境变量, 按照 "K=V" 格式组织
      GTEST_FILTER_LIST  : [Optional] GTestFilter 配置, Filter 间以 ';' 分割
Attention:
    1. 可以多次调用本函数以添加多个'执行任务'; 单次调用本函数时, 可以通过在 GTEST_FILTER_LIST 中配置多个过滤条件('gtest_filter') 以实现执行多用例;
]]
function(PTO_Fwk_UTest_RunExe)
    cmake_parse_arguments(
            ARG
            ""
            "TARGET"
            "LD_LIBRARIES_EXT;ENV_LINES_EXT;GTEST_FILTER_LIST"
            ""
            ${ARGN}
    )
    if (ENABLE_TESTS_EXECUTE)
        # 命令行参数处理
        PTO_Fwk_GTest_RunExe_GetPreExecSetup(PyCmdSetup PyEnvLines BashCmdSetup
                TARGET              ${ARG_TARGET}
                ENV_LINES_EXT       ${ARG_ENV_LINES_EXT}
                LD_LIBRARIES_EXT    ${ARG_LD_LIBRARIES_EXT}
        )
        # 执行流程
        list(LENGTH ARG_GTEST_FILTER_LIST GtestFilterListLen)
        string(REPLACE ";" ":" GtestFilterStr "${ARG_GTEST_FILTER_LIST}")
        message(STATUS "Run GTest(${ARG_TARGET}), XSAN(ASAN:${ENABLE_ASAN} UBSAN:${ENABLE_UBSAN}), GTestFilter(${GtestFilterListLen})=${GtestFilterStr}")
        set(Comment "Run GTest(${ARG_TARGET}), XSAN(ASAN:${ENABLE_ASAN} UBSAN:${ENABLE_UBSAN})")

        if (ARG_GTEST_FILTER_LIST)
            if (ENABLE_TESTS_EXECUTE_PARALLEL)
                # 仅在使能并行执行全局开关, 且需要做 filter 时才进行执行加速
                set(_File $<TARGET_FILE:${ARG_TARGET}>)
                set(_Args "-t=${_File}" "--gtest_filter=${GtestFilterStr}" "--halt_on_error")
                if (ENABLE_TESTS_EXECUTE_PARALLEL_TIMEOUT)
                    list(APPEND _Args "--timeout=${ENABLE_TESTS_EXECUTE_PARALLEL_TIMEOUT}")
                endif ()
                if (PyEnvLines)
                    list(APPEND _Args "--env" "${PyEnvLines}")
                endif ()
                get_filename_component(ParallelPy    "${PTO_FWK_SRC_ROOT}/tests/cmake/scripts/python/utest_accelerate.py" REALPATH)
                get_filename_component(ParallelPyCwd "${PTO_FWK_SRC_ROOT}/tests/cmake/scripts/python" REALPATH)
                add_custom_command(
                        TARGET ${ARG_TARGET} POST_BUILD
                        COMMAND ${PyCmdSetup} ${Python3_EXECUTABLE} ${ParallelPy} ARGS ${_Args}
                        COMMENT "${Comment} With Parallel Execute Accelerate"
                        WORKING_DIRECTORY ${ParallelPyCwd}
                )
            else ()
                set(GtestFilterListIdx 1)
                foreach (Filter ${ARG_GTEST_FILTER_LIST})
                    add_custom_command(
                            TARGET ${ARG_TARGET} POST_BUILD
                            COMMAND ${BashCmdSetup} ./${ARG_TARGET} ARGS '--gtest_filter=${Filter}'
                            COMMENT "${Comment} [${GtestFilterListIdx}/${GtestFilterListLen}] With --gtest_filter=${Filter}"
                    )
                    math(EXPR GtestFilterListIdx "${GtestFilterListIdx} + 1")
                endforeach ()
            endif ()
        else ()
            add_custom_command(
                    TARGET ${ARG_TARGET} POST_BUILD
                    COMMAND ${BashCmdSetup} ./${ARG_TARGET}
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
function(PTO_Fwk_UTest_AddExe_RunExe)
    cmake_parse_arguments(
            ARG
            ""
            "TARGET"
            ""
            ""
            ${ARGN}
    )
    set(_Sources ${CMAKE_CURRENT_BINARY_DIR}/${PTO_Fwk_UTestNamePrefix}_main_stub.cpp)
    execute_process(COMMAND touch ${_Sources})

    set(_PrivateLinkLibraries
            ${PTO_Fwk_UTestNamePrefix}_intf_pub
            $<$<BOOL:${ENABLE_BUILD_WITH_CANN}>:${PTO_Fwk_UTestNamePrefix}_stubs>
    )

    # 支持由 ENABLE_TESTS_UTEST 传入指定的 Filter
    set(GTestFilterList ${PTO_Fwk_UTestCaseGTestFilterList})
    if (NOT "${ENABLE_TESTS_UTEST}" STREQUAL "ON")
        set(GTestFilterList ${ENABLE_TESTS_UTEST})
        string(REPLACE ":" ";" GTestFilterList "${GTestFilterList}")
    endif ()

    list(FILTER PTO_Fwk_UTestCaseLibraries         EXCLUDE REGEX "PARALLEL_SEPARATOR")
    list(FILTER PTO_Fwk_UTestCaseLdLibrariesExt    EXCLUDE REGEX "PARALLEL_SEPARATOR")
    list(FILTER GTestFilterList                     EXCLUDE REGEX "PARALLEL_SEPARATOR")
    list(REMOVE_DUPLICATES PTO_Fwk_UTestCaseLibraries)
    list(REMOVE_DUPLICATES PTO_Fwk_UTestCaseLdLibrariesExt)
    list(REMOVE_DUPLICATES GTestFilterList)

    if (NOT "$ENV{GTEST_START}" STREQUAL "")
        list(FIND GTestFilterList $ENV{GTEST_START} idx)
        if (NOT ${idx} EQUAL -1)
            list(SUBLIST GTestFilterList ${idx} -1 GTestFilterList)
        endif ()
    endif ()

    PTO_Fwk_GTest_AddExe(
            TARGET                      ${ARG_TARGET}
            SOURCES                     ${_Sources}
            PRIVATE_INCLUDE_DIRECTORIES ${ARG_PRIVATE_INCLUDE_DIRECTORIES}
            PRIVATE_LINK_LIBRARIES      ${_PrivateLinkLibraries} ${PTO_Fwk_UTestCaseLibraries}
    )
    PTO_Fwk_UTest_RunExe(
            TARGET              ${ARG_TARGET}
            LD_LIBRARIES_EXT    ${PTO_Fwk_UTestCaseLdLibrariesExt}
            GTEST_FILTER_LIST   ${GTestFilterList}
    )

    # 生成覆盖率
    PTO_Fwk_GTest_GenerateCoverage(TARGET ${ARG_TARGET})
endfunction()

# UTest 以 pytest 方式触发 python 用例执行
#[[
Parameters:
  one_value_keywords:
      PYTEST_INI                    : [Optional] 指定具体 pytest.ini 文件
  multi_value_keywords:
      PYTHON_PATH_EXT               : [Optional] 额外需要配置的 PYTHONPATH
      PYTHON_PATH_LIBRARIES         : [Optional] 需要配置在 PYTHONPATH 中的二进制
]]
function(PTO_Fwk_UTest_RunPytest)
    cmake_parse_arguments(
            ARG
            ""
            "PYTEST_INI"
            "PYTHON_PATH_EXT;PYTHON_PATH_LIBRARIES"
            ""
            ${ARGN}
    )
    # 执行
    PTO_Fwk_GTest_RunPytest(
            TARGET_NAME_PREFIX      ${PTO_Fwk_UTestNamePrefix}
            PYTEST_INI              ${ARG_PYTEST_INI}
    )
endfunction()
