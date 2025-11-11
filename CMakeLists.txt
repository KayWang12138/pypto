# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
cmake_minimum_required(VERSION 3.16)

project(PTO_Framework
        LANGUAGES C CXX
        VERSION 1.0.0
)


########################################################################################################################
# 基础配置
########################################################################################################################

# 路径配置
get_filename_component(PTO_FWK_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" REALPATH)
get_filename_component(PTO_FWK_BIN_ROOT "${CMAKE_CURRENT_BINARY_DIR}" REALPATH)

message(STATUS "PTO_FWK_SRC_ROOT=${PTO_FWK_SRC_ROOT}")
message(STATUS "PTO_FWK_BIN_ROOT=${PTO_FWK_BIN_ROOT}")

# 公开编译开关(build)
option(BUILD_OPEN_PROJECT "Build in open scenario." ON)
option(BUILD_WITH_CANN "Build with CANN." ON)  # Need to install CANN package and set corresponding environment variables.
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(ENABLE_GCOV "Enable GNU Coverage Instrumentation Tool" OFF)

# 公开编译开关(Feature 特性级开关)
option(ENABLE_FEATURE_PYTHON_FRONT_END "Enable Python Front-end, Specify whl name" OFF)
option(ENABLE_TORCH_VERIFIER "Flow verify with libtorch" OFF)

# 公开编译开关(tests)
option(ENABLE_TESTS_EXECUTE "Enable auto execute tests executable" OFF)
option(ENABLE_TESTS_EXECUTE_PARALLEL "Auto execute tests executable parallel" OFF)
option(ENABLE_TESTS_EXECUTE_PARALLEL_TIMEOUT "Auto execute tests executable parallel timeout" OFF)
option(ENABLE_TESTS_EXECUTE_CHANGED_FILE "Specify tests changed file" OFF)
option(ENABLE_UTEST "Enable UTest" OFF)
option(ENABLE_STEST_GOLDEN_PATH "Specific STest Golden Path" OFF)
option(ENABLE_STEST_GOLDEN_PATH_CLEAN "Clean STest Golden" OFF)
option(ENABLE_STEST_EXECUTE_DEVICE_ID "Auto execute tests executable with Device Id" OFF)
option(ENABLE_STEST_DUMP_JSON "Enable STest Dump Json" OFF)
option(ENABLE_STEST_INTERPRETER_CONFIG "Enable STest Interpreter Config" OFF)
option(ENABLE_STEST_BINARY_CACHE "Enable STest Binary Cache" OFF)
option(ENABLE_STEST "Enable STest" OFF)
option(ENABLE_STEST_GROUP "Enable STest Group" OFF)
option(ENABLE_STEST_DISTRIBUTED "Enable STest(Distributed)" OFF)

# 公开编译开关(stest.tools)
option(ENABLE_STEST_TOOLS_CASE_FILE "Specify STest Tools Case file path" OFF)
option(ENABLE_STEST_TOOLS_INTERCEPT "Enable STest Tools Intercept if has any failed case" OFF)
option(ENABLE_STEST_TOOLS_OUTPUT_CLEAN "Clean STest Tools Output Directory" OFF)
option(ENABLE_STEST_TOOLS_PROF "Enable STest Tools(Profiling)" OFF)
option(ENABLE_STEST_TOOLS_PROF_LEVEL "Specify STest Tools Profiling Level" OFF)
option(ENABLE_STEST_TOOLS_PROF_WARN_UP_CNT "Specify STest Tools Profiling Warn-up cnt" OFF)
option(ENABLE_STEST_TOOLS_PROF_TRY_CNT "Specify STest Tools Profiling Try cnt" OFF)
option(ENABLE_STEST_TOOLS_PROF_MAX_CNT "Specify STest Tools Profiling Max cnt" OFF)

# 缺省编译配置, 编译函数
include(ExternalProject)
include(cmake/func.cmake)
include(cmake/config.cmake)
include(cmake/intf.cmake)
include(CMakePackageConfigHelpers)


########################################################################################################################
# 编译流程
########################################################################################################################

add_subdirectory(framework)
add_subdirectory(python)


########################################################################################################################
# 安装流程
########################################################################################################################

if (ENABLE_FEATURE_PYTHON_FRONT_END)
    # 安装 二进制文件
    set(InstallExport ${PROJECT_NAME}Targets)
    set(InstallTargets tile_fwk_interface tile_fwk_codegen tile_fwk_compiler tile_fwk_runtime)
    PTO_Fwk_InstallBinaries(
            EXPORT  ${InstallExport}
            TARGETS ${InstallTargets}
    )
    install(TARGETS tile_fwk_server LIBRARY DESTINATION "${ENABLE_FEATURE_PYTHON_FRONT_END}/${CMAKE_INSTALL_LIBDIR}")
    PTO_Fwk_InstallBinaries(
            TARGETS         pto_impl
            INSTALL_LIBDIR  ${CMAKE_INSTALL_PREFIX}/${ENABLE_FEATURE_PYTHON_FRONT_END}
    )

    # 安装 配置文件
    install(FILES
                framework/src/interface/configs/tile_fwk_config.json
                framework/src/interface/configs/tile_fwk_config_schema.json
                framework/src/passes/pass_config/tile_fwk_platform_info.json
            DESTINATION ${CMAKE_INSTALL_PREFIX}/${ENABLE_FEATURE_PYTHON_FRONT_END}/configs
    )

    # 安装 头文件
    install(FILES
                framework/include/tilefwk/comm_group_recorder.h
                framework/include/tilefwk/config.h
                framework/include/tilefwk/data_type.h
                framework/include/tilefwk/element.h
                framework/include/tilefwk/error.h
                framework/include/tilefwk/function.h
                framework/include/tilefwk/gentask_utils.h
                framework/include/tilefwk/lazy.h
                framework/include/tilefwk/op_registry.h
                framework/include/tilefwk/symbolic_distributed.h
                framework/include/tilefwk/symbolic_scalar.h
                framework/include/tilefwk/tensor.h
                framework/include/tilefwk/tile_shape.h
                framework/include/tilefwk/tilefwk.h
                framework/include/tilefwk/tilefwk_op.h
            DESTINATION ${CMAKE_INSTALL_PREFIX}/${ENABLE_FEATURE_PYTHON_FRONT_END}/${CMAKE_INSTALL_INCLUDEDIR}/tilefwk
    )
    install(DIRECTORY framework/include/tileop/
            DESTINATION ${CMAKE_INSTALL_PREFIX}/${ENABLE_FEATURE_PYTHON_FRONT_END}/${CMAKE_INSTALL_LIBDIR}/include/tileop
            PATTERN "*.h"
            PATTERN "*.hpp"
            PATTERN ".svn" EXCLUDE
            PATTERN ".git" EXCLUDE
    )
    install(FILES
                framework/include/tilefwk/aicore_data.h
                framework/include/tilefwk/aicore_print.h
                framework/include/tilefwk/aicore_runtime.h
                framework/include/tilefwk/aicpu_common.h
                framework/include/tilefwk/aicpu_distributed.h
                framework/include/tilefwk/aicpu_runtime.h
                framework/include/tilefwk/core_func_data.h
            DESTINATION ${CMAKE_INSTALL_PREFIX}/${ENABLE_FEATURE_PYTHON_FRONT_END}/${CMAKE_INSTALL_LIBDIR}/include/tilefwk
    )

    # 生成和安装配置文件
    PTO_Fwk_InstallCMakeConfig(
            EXPORT              ${InstallExport}
            CMAKE_PARENT_DIR    ${ENABLE_FEATURE_PYTHON_FRONT_END}/${CMAKE_INSTALL_LIBDIR}
            TARGETS             ${InstallTargets}
    )

    # 设置包信息
    set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
    set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
    set(CPACK_PACKAGE_DESCRIPTION ${PROJECT_NAME})
    include(CPack)
else ()
    install(DIRECTORY framework/include/                          DESTINATION lib/tilefwk/ OPTIONAL)
endif ()
