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
function(PTO_Fwk_GTest_GenerateCoverage)
    cmake_parse_arguments(
            ARG
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
        get_filename_component(GenCoveragePy ${PTO_FWK_SRC_ROOT}/tests/cmake/scripts/python/gen_coverage.py REALPATH)
        get_filename_component(GenCoverageDataDir "${PTO_FWK_BIN_ROOT}" REALPATH)
        set(_Args "-s=${PTO_FWK_SRC_ROOT}" "-c=${GenCoverageDataDir}")

        get_target_property(GTest_GTest_Inc     GTest::gtest           INTERFACE_INCLUDE_DIRECTORIES)
        get_target_property(GTest_GTestMain_Inc GTest::gtest_main      INTERFACE_INCLUDE_DIRECTORIES)
        if (BUILD_OPEN_PROJECT)
            get_target_property(Json_Inc nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
        else ()
            get_target_property(Json_Inc json                         INTERFACE_INCLUDE_DIRECTORIES)
        endif ()
        set(Filter_Dirs
                ${PTO_FWK_SRC_ROOT}/tests
                ${PTO_FWK_SRC_ROOT}/third_party
                ${GTest_GTest_Inc}
                ${GTest_GTestMain_Inc}
                ${Json_Inc}
                ${SYS_ROOT}
                ${ARG_FILTER_DIRECTORIES}
        )
        foreach (_dir ${Filter_Dirs})
            list(APPEND _Args "-f=${_dir}")
        endforeach ()
        list(REMOVE_DUPLICATES _Args)

        add_custom_command(
                TARGET ${ARG_TARGET} POST_BUILD
                COMMAND ${Python3_EXECUTABLE} ${GenCoveragePy} ARGS ${_Args}
                COMMENT "Generate coverage for ${ARG_TARGET}"
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
function(PTO_Fwk_GTest_RunExe_GetPreExecSetup PY_CMD_SETUP PY_ENV_LINES BASH_CMD_SETUP)
    cmake_parse_arguments(
            ARG
            ""
            "TARGET"
            "LD_LIBRARIES_EXT;CMD_SETUP_EXT;ENV_LINES_EXT"
            ""
            ${ARGN}
    )

    # 命令行
    set(CmdSetup)
    # 处理变量 CMD_SETUP_EXT
    if (NOT "${ARG_CMD_SETUP_EXT}x" STREQUAL "x")
        list(APPEND CmdSetup ${ARG_CMD_SETUP_EXT})
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
    foreach (LIBRARY ${ARG_LD_LIBRARIES_EXT})
        add_dependencies(${ARG_TARGET} ${LIBRARY})
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
    list(REMOVE_ITEM ARG_ENV_LINES_EXT export)
    list(REMOVE_ITEM ARG_ENV_LINES_EXT &)
    list(REMOVE_ITEM ARG_ENV_LINES_EXT &&)
    if (NOT "${ARG_ENV_LINES_EXT}x" STREQUAL "x")
        list(APPEND EnvLines ${ARG_ENV_LINES_EXT})
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
function(PTO_Fwk_GTest_AddExe)
    cmake_parse_arguments(
            ARG
            ""
            "TARGET"
            "SOURCES;PRIVATE_INCLUDE_DIRECTORIES;PRIVATE_LINK_LIBRARIES"
            ""
            ${ARGN}
    )
    add_executable(${ARG_TARGET})
    target_sources(${ARG_TARGET}
            PRIVATE
                ${ARG_SOURCES}
                ${PTO_FWK_SRC_ROOT}/tests/main.cpp
    )
    target_include_directories(${ARG_TARGET}
            PRIVATE
                ${ARG_PRIVATE_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(${ARG_TARGET}
            PRIVATE
                GTest::gtest
                -Wl,--no-as-needed
                -Wl,--whole-archive
                ${ARG_PRIVATE_LINK_LIBRARIES}
                -Wl,--as-needed
                -Wl,--no-whole-archive
                -rdynamic
    )
    add_custom_command(
        TARGET ${ARG_TARGET} POST_BUILD
        COMMAND mkdir -p "${PTO_FWK_BIN_ROOT}/src/conf"
        COMMAND ln -sf "${PTO_FWK_SRC_ROOT}/src/interface/configs/tile_fwk_config.json" "${PTO_FWK_BIN_ROOT}/src/conf/tile_fwk_config.json"
        COMMAND ln -sf "${PTO_FWK_SRC_ROOT}/src/passes/pass_config/tile_fwk_platform_info.json" "${PTO_FWK_BIN_ROOT}/src/conf/tile_fwk_platform_info.json"
        COMMENT "Soft link of tile_fwk_config.json and tile_fwk_platform_info.json has been created at ${PTO_FWK_BIN_ROOT}/src/conf"
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${PTO_FWK_BIN_ROOT}/src/include
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PTO_FWK_BIN_ROOT}/src/include
        COMMAND ln -sf ${PTO_FWK_SRC_ROOT}/include ${PTO_FWK_BIN_ROOT}/src/include/tile_fwk
        COMMENT "Soft link include directory has been created at ${PTO_FWK_BIN_ROOT}/src/include/tile_fwk"
    )
endfunction()

# GTest 以 pytest 方式触发 python 用例执行
#[[
Parameters:
  one_value_keywords:
      STUB_TARGET_NAME              : [Required] 用于指定桩目标的名称
      PYTEST_INI                    : [Optional] 指定具体 pytest.ini 文件
  multi_value_keywords:
      PYTEST_PARAM_EXT              : [Optional] pytest 额外补充参数
]]
function(PTO_Fwk_GTest_RunPytest OUT_TARGET)
    cmake_parse_arguments(
            ARG
            ""
            "TARGET_NAME_PREFIX;PYTEST_INI"
            "PYTEST_PARAM_EXT"
            ""
            ${ARGN}
    )
    # 编译 桩目标
    set(_MainStub ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET_NAME_PREFIX}_main_stub_python.cpp)
    execute_process(COMMAND touch ${_MainStub})
    set(_Target ${ARG_TARGET_NAME_PREFIX}_python)
    add_library(${_Target} SHARED)
    target_sources(${_Target} PRIVATE ${_MainStub})
    add_dependencies(${_Target} pto_impl)
    set(${OUT_TARGET} ${_Target} PARENT_SCOPE)

    # 处理 pytest.ini
    if (NOT ARG_PYTEST_INI)
        get_filename_component(_PytestIniIn "${CMAKE_CURRENT_SOURCE_DIR}/pytest.ini.in" REALPATH)
        if (EXISTS "${_PytestIniIn}")
            get_filename_component(PTO_FWK_PYTHON_TESTS_PATH "${CMAKE_CURRENT_SOURCE_DIR}" REALPATH)  # config pytest.ini.in 所需
            get_filename_component(ARG_PYTEST_INI "${CMAKE_CURRENT_BINARY_DIR}/pytest.ini" REALPATH)
            configure_file(${_PytestIniIn} ${ARG_PYTEST_INI} @ONLY)
        endif ()
    endif ()
    if (NOT EXISTS "${ARG_PYTEST_INI}")
        message(FATAL_ERROR "Can't get ${ARG_STUB_TARGET_NAME} 's pytest.ini[${ARG_PYTEST_INI}]")
    endif ()

    # 执行用例
    if (ENABLE_TESTS_EXECUTE)
        PTO_Fwk_AnalysisPython3Environ(pytest_FOUND JUDGE_PYTEST_INSTALLED)
        if ("${pytest_FOUND}x" STREQUAL "x")
            message(WARNING "pytest not installed, python test won't run.")
        else ()
            # 重新 touch 源码, 保证可重复执行
            add_custom_command(
                    TARGET ${_Target} PRE_BUILD
                    COMMAND touch ${_MainStub}
            )

            # 安装 pypto whl 包
            get_filename_component(_PyPTOInstallPath ${PTO_FWK_BIN_ROOT}/pypto_install REALPATH)
            add_custom_command(
                    TARGET ${_Target} POST_BUILD
                    COMMAND find ./pto/ -name "*.so" -delete
                    COMMAND find ./pto/ -name "*.json" -delete
                    COMMAND ${CMAKE_COMMAND} -E copy ${PTO_FWK_SRC_ROOT}/src/interface/configs/tile_fwk_config.json         ./pto/
                    COMMAND ${CMAKE_COMMAND} -E copy ${PTO_FWK_SRC_ROOT}/src/interface/configs/tile_fwk_config_schema.json  ./pto/
                    COMMAND ${CMAKE_COMMAND} -E copy ${PTO_FWK_SRC_ROOT}/src/passes/pass_config/tile_fwk_platform_info.json ./pto/
                    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tile_fwk_operator>       ./pto/
                    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tile_fwk_interface>      ./pto/
                    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tile_fwk_passes>         ./pto/
                    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tile_fwk_compiler>       ./pto/
                    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tile_fwk_codegen>        ./pto/
                    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tile_fwk_simulation>     ./pto/
                    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tile_fwk_simulation_ca>  ./pto/
                    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tile_fwk_runtime>        ./pto/
                    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:pto_impl>                ./pto/
                    WORKING_DIRECTORY ${PTO_FWK_SRC_ROOT}/python/src/tensor_op
            )
            if (TARGET tile_fwk_calculator)
                add_custom_command(
                        TARGET ${_Target} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tile_fwk_calculator>     ./pto/
                        WORKING_DIRECTORY ${PTO_FWK_SRC_ROOT}/python/src/tensor_op
                )
            endif ()
            add_custom_command(
                    TARGET ${_Target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E remove_directory ${_PyPTOInstallPath}
                    COMMAND ${CMAKE_COMMAND} -E make_directory ${_PyPTOInstallPath}
                    COMMAND ${Python3_EXECUTABLE} -m pip install --target=${_PyPTOInstallPath} -v --force-reinstall .
                    COMMAND ${CMAKE_COMMAND} -E remove_directory ${_PyPTOInstallPath}/pto/include/
                    COMMAND ${CMAKE_COMMAND} -E copy_directory  ${PTO_FWK_SRC_ROOT}/include/ ${_PyPTOInstallPath}/pto/include/
                    COMMENT "Install to ${_PyPTOInstallPath}"
                    WORKING_DIRECTORY ${PTO_FWK_SRC_ROOT}/python/src/tensor_op
            )

            # 拼接 pytest 执行参数和环境变量, 并执行
            set(_PytestParamExt)
            PTO_Fwk_AnalysisPython3Environ(pytest_forked_FOUND JUDGE_PYTEST_FORKED_INSTALLED)
            if (NOT "${pytest_forked_FOUND}x" STREQUAL "x")
                list(APPEND _PytestParamExt --forked)
            else ()
                message(FATAL_ERROR "pytest_forked not installed")
            endif ()
            PTO_Fwk_GTest_RunExe_GetPreExecSetup(PyCmdSetup PyEnvLines BashCmdSetup
                    TARGET ${_Target}
            )
            set(_PytestCmd
                    ${PyEnvLines}
                    PYTHONPATH="${_PyPTOInstallPath}:$ENV{PYTHONPATH}"
                    LD_LIBRARY_PATH="$ENV{LD_LIBRARY_PATH}:${TORCH_ROOT_PATH}/lib"
                    TILEFWK_CONFIG_PATH=${_PyPTOInstallPath}/pto/tile_fwk_config.json
                    PLATFORM_CONFIG_PATH=${_PyPTOInstallPath}/pto/tile_fwk_platform_info.json
                    ${Python3_EXECUTABLE} -m pytest -s -c ${ARG_PYTEST_INI} ${_PytestParamExt} ${PYTEST_PARAM_EXT}
            )
            add_custom_command(
                    TARGET ${_Target} POST_BUILD
                    COMMAND ${_PytestCmd}
                    COMMENT "Run pytest With ${_PytestCmd}"
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
        endif ()
    endif ()
endfunction ()
