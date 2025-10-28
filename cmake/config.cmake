# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================


########################################################################################################################
# 环境检查
########################################################################################################################

# Python3
if (NOT DEFINED Python3_EXECUTABLE)
    # 当外部未指定 Python3 时, 一般是从 CMake 为入口触发的编译, 此时直接 find
    find_package(Python3 COMPONENTS Interpreter Development)
    if ("${Python3_EXECUTABLE}x" STREQUAL "x")
        message(FATAL_ERROR "Can't find python3 Interpreter.")
    endif ()
else ()
    # 当外部指定 Python3 时, 此时强制使用外部指定的 Python3 对应版本, 若外部未指定 Python3 版本, 尝试重新获取
    if (NOT DEFINED Python3_FIND_VERSION)
        PTO_Fwk_AnalysisPython3Environ(Python3_FIND_VERSION GET_PYTHON_V)
    endif ()
    find_package(Python3 ${Python3_FIND_VERSION} EXACT COMPONENTS Development)
endif ()

if (Python3_Development_FOUND)
    PTO_Fwk_AnalysisPython3Environ(pybind11_DIR GET_PYBIND11_DIR)
    message(STATUS "pybind11_DIR=${pybind11_DIR}")
    if (NOT "${pybind11_DIR}x" STREQUAL "x")
        find_package(pybind11 CONFIG REQUIRED PATHS ${pybind11_DIR} NO_DEFAULT_PATH)
    endif ()
    if (NOT pybind11_FOUND)
        set(ENABLE_FEATURE_PYTHON_FRONT_END OFF)
        message(WARNING "Can't get pybind11, Auto turn off ENABLE_FEATURE_PYTHON_FRONT_END.")
    endif ()
else ()
    set(ENABLE_FEATURE_PYTHON_FRONT_END OFF)
    message(WARNING "Can't get python3-dev, Auto turn off ENABLE_FEATURE_PYTHON_FRONT_END.")
endif ()


# 获取 CANN 路径
if (CUSTOM_ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH  ${CUSTOM_ASCEND_CANN_PACKAGE_PATH})
elseif (DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH  $ENV{ASCEND_HOME_PATH})
elseif (DEFINED ENV{ASCEND_OPP_PATH})
    get_filename_component(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_OPP_PATH}/.." ABSOLUTE)
else()
    set(ASCEND_CANN_PACKAGE_PATH  "/usr/local/Ascend/latest")
endif ()
message(STATUS "ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}")

########################################################################################################################
# CMake 选项, 缺省参数设置
#   按 CMake 构建过程对 CMake 选项, CMake 缺省参数进行配置
#   CMake 构建过程: 1) 配置阶段(Configure); 2) 构建阶段(Build); 3) 安装阶段(Install);
########################################################################################################################

# 构建阶段(Build)
#   构建类型
#       CMake中的Generator(生成器)是用于生成本地/本机构建系统的工具. 一般分为两种:
#       1. 单配置生成器(Single-configuration generator):
#          在配置(Configuration)阶段, 仅允许指定一种构建类型, 通过变量 CMAKE_BUILD_TYPE 指定;
#          在构建阶段(Build)无法更改构建类型, 仅允许使用配置(Configuration)阶段通过变量 CMAKE_BUILD_TYPE 指定的构建类型;
#          常见的此类型生成器有: Ninja, Unix Makefiles
#       2. 多配置生成器(Multi-configuration generator) :
#          在配置(Configuration)阶段, 仅指定构建阶段(Build)可用的构建类型列表, 通过变量 CMAKE_CONFIGURATION_TYPES 指定;
#          在构建阶段(Build)通过 "--config" 参数, 指定构建阶段具体的构建类型;
#          常见的此类型生成器有: Xcode, Visual Studio
#       所以:
#           1. 单配置生成器(Single-configuration generator)场景下, 如果构建类型(CMAKE_BUILD_TYPE)未指定, 则默认为 Debug ;
#           2. 多配置生成器(Multi-configuration generator)场景下, 如果构建阶段可选的构建类型(CMAKE_CONFIGURATION_TYPES)未指定,
#              则默认将其指定为CMake允许的构建类型全集 [Debug;Release;MinSizeRel;RelWithDebInfo]
get_property(GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if (GENERATOR_IS_MULTI_CONFIG)
    if (NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_CONFIGURATION_TYPES "Debug;Release;MinSizeRel;RelWithDebInfo" CACHE STRING "Configuration Build type" FORCE)
    endif ()
else ()
    if (NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE          "Debug"                                   CACHE STRING "Build type(default Debug)" FORCE)
    endif ()
endif ()
message(STATUS "CMAKE_GENERATOR=${CMAKE_GENERATOR}")

# 构建阶段(Build)
#   可执行文件运行时库文件搜索路径 RPATH
#       在 UTest 及 STest 场景不略去 RPATH
string(REPLACE "," ":" ENABLE_UTEST "${ENABLE_UTEST}")
string(REPLACE "," ":" ENABLE_STEST "${ENABLE_STEST}")
string(REPLACE "," ":" ENABLE_STEST_DISTRIBUTED "${ENABLE_STEST_DISTRIBUTED}")
if (ENABLE_UTEST OR ENABLE_STEST OR ENABLE_STEST_DISTRIBUTED)
    set(ENABLE_TESTS ON)
else ()
    set(ENABLE_TESTS OFF)
endif ()

if (ENABLE_TESTS)
    set(CMAKE_SKIP_RPATH FALSE)
else ()
    set(CMAKE_SKIP_RPATH TRUE)
endif ()

# 构建阶段(Build)
#   语言标准
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# 构建阶段(Build)
#   CCACHE 配置
if (BUILD_OPEN_PROJECT)
    find_program(CCACHE_PROGRAM ccache)
    if (CCACHE_PROGRAM)
        set(CMAKE_C_COMPILER_LAUNCHER   ${CCACHE_PROGRAM} CACHE PATH "C cache Compiler")
        set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM} CACHE PATH "CXX cache Compiler")
        if (NOT DEFINED ENV{CCACHE_BASEDIR})
            set(ENV{CCACHE_BASEDIR} ${PTO_FWK_SRC_ROOT})
        endif ()
        message(STATUS "Use ccache, CCACHE_BASEDIR=$ENV{CCACHE_BASEDIR}")
    else ()
        message(STATUS "ccache not found.")
    endif ()
endif()

# 安装阶段(Install)
#   安装路径
#       未显示设置 CMAKE_INSTALL_PREFIX (即 CMAKE_INSTALL_PREFIX 取缺省值)时,
#       修正其取值与构建树根目录 CMAKE_CURRENT_BINARY_DIR 平级
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    get_filename_component(_Install_Path_Prefix "${CMAKE_CURRENT_BINARY_DIR}/../output" REALPATH)
    set(CMAKE_INSTALL_PREFIX    "${_Install_Path_Prefix}"  CACHE STRING "Install path" FORCE)
endif ()


########################################################################################################################
# 预处理
########################################################################################################################
if (BUILD_OPEN_PROJECT)
    set(BISHENG_PROGRAM bisheng)
    set(BISHENG_LD ld.lld)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
else()
    set(BISHENG_PROGRAM ${CCEC_PATH}/bisheng)
    set(BISHENG_LD ${CCEC_PATH}/ld.lld)
endif()

if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang" AND ENABLE_GCOV)
    set(ENABLE_GCOV OFF)
    message(STATUS "GCov only supported in GNU Compiler, Current Compiler is Clang, Auto turn off it.")
endif ()


# ASAN / UBSAN 场景随编译执行用例场景下, 将相关检查在编译前执行, 避免出现编译完成后又无法执行的情况, 影响使用体验.
if ((ENABLE_ASAN OR ENABLE_UBSAN) AND ENABLE_TESTS_EXECUTE)
    # LD_PRELOAD, 仅 GNU 编译器需要设置
    set(XSAN_LD_PRELOAD)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if (ENABLE_ASAN)
            # libasan.so
            execute_process(COMMAND ${CMAKE_C_COMPILER} --print-file-name=libasan.so
                    RESULT_VARIABLE _RST
                    OUTPUT_VARIABLE ASAN_SHARED_PATH)
            if (_RST)
                message(FATAL_ERROR "Can't get libasan.so path with ${CMAKE_C_COMPILER}")
            endif ()
            get_filename_component(ASAN_SHARED_PATH "${ASAN_SHARED_PATH}" DIRECTORY)
            get_filename_component(ASAN_SHARED_PATH "${ASAN_SHARED_PATH}/libasan.so" REALPATH)
            if (NOT EXISTS ${ASAN_SHARED_PATH})
                message(FATAL_ERROR "ASAN_SHARED_PATH=${ASAN_SHARED_PATH} not exist.")
            endif ()
            list(APPEND XSAN_LD_PRELOAD ${ASAN_SHARED_PATH})
        endif ()
        if (ENABLE_UBSAN)
            # libubsan.so
            execute_process(COMMAND ${CMAKE_C_COMPILER} --print-file-name=libubsan.so
                    RESULT_VARIABLE _RST
                    OUTPUT_VARIABLE UBSAN_SHARED_PATH)
            if (_RST)
                message(FATAL_ERROR "Can't get libubsan.so path with ${CMAKE_C_COMPILER}")
            endif ()
            get_filename_component(UBSAN_SHARED_PATH "${UBSAN_SHARED_PATH}" DIRECTORY)
            get_filename_component(UBSAN_SHARED_PATH "${UBSAN_SHARED_PATH}/libubsan.so" REALPATH)
            if (NOT EXISTS ${UBSAN_SHARED_PATH})
                message(FATAL_ERROR "UBSAN_SHARED_PATH=${UBSAN_SHARED_PATH} not exist.")
            endif ()
            list(APPEND XSAN_LD_PRELOAD ${UBSAN_SHARED_PATH})
        endif ()
        # libstdc++.so
        execute_process(COMMAND ${CMAKE_C_COMPILER} --print-file-name=libstdc++.so
                RESULT_VARIABLE _RST
                OUTPUT_VARIABLE STDC_SHARED_PATH)
        if (_RST)
            message(FATAL_ERROR "Can't get libstdc++.so path with ${CMAKE_C_COMPILER}")
        endif ()
        get_filename_component(STDC_SHARED_PATH "${STDC_SHARED_PATH}" DIRECTORY)
        get_filename_component(STDC_SHARED_PATH "${STDC_SHARED_PATH}/libstdc++.so" REALPATH)
        if (NOT EXISTS ${STDC_SHARED_PATH})
            message(FATAL_ERROR "STDC_SHARED_PATH=${STDC_SHARED_PATH} not exist.")
        endif ()
        list(APPEND XSAN_LD_PRELOAD ${STDC_SHARED_PATH})
        # 结果修正
        string(REPLACE ";" ":" XSAN_LD_PRELOAD "${XSAN_LD_PRELOAD}")
        set(XSAN_LD_PRELOAD "LD_PRELOAD=${XSAN_LD_PRELOAD}")
    endif ()

    set(ASAN_OPTIONS)
    if (ENABLE_ASAN)
        # 谨慎修改 ASAN_OPTIONS 取值, 当前出现告警会使 GTest 失败.
        # halt_on_error=1, 出现告警时停止运行进而触发构建失败, 避免主进程或 fork 出的子进程出现错误无法发现的情况
        # detect_stack_use_after_return=1, 栈空间返回后使用检测
        # check_initialization_order, 尝试捕获初始化顺序问题
        # strict_init_order, 动态初始化器永远不能访问来自其他模块的全局变量, 及时或者已经初始化
        # strict_string_checks, 检查字符串参数是否正确以 null 终止
        # detect_leaks=1, 内存泄漏检测
        set(ASAN_OPTIONS "ASAN_OPTIONS=halt_on_error=0,detect_stack_use_after_return=1,check_initialization_order=1,strict_init_order=1,strict_string_checks=1,detect_leaks=1")
    endif ()

    set(UBSAN_OPTIONS)
    if (ENABLE_UBSAN)
        # 谨慎修改 UBSAN_OPTIONS 取值, 当前出现告警会使 UT 失败.
        # halt_on_error=1, 出现告警时停止运行进而触发构建失败, 避免主进程或 fork 出的子进程出现错误无法发现的情况
        # print_stacktrace=1, 出错时打印调用栈
        set(UBSAN_OPTIONS "UBSAN_OPTIONS=halt_on_error=0,print_stacktrace=1")
    endif ()
endif ()


########################################################################################################################
# 三方库
########################################################################################################################

# nlohmann_json::nlohmann_json
if (BUILD_OPEN_PROJECT)
    find_package(nlohmann_json CONFIG)
    if (NOT ${nlohmann_json_FOUND})
        if (DEFINED ENV{ASCEND_3RD_LIB_PATH} AND NOT "$ENV{ASCEND_3RD_LIB_PATH}x" STREQUAL "x")
            get_filename_component(ASCEND_3RD_LIB_PATH "$ENV{ASCEND_3RD_LIB_PATH}" REALPATH)
            if (EXISTS "${ASCEND_3RD_LIB_PATH}/cmake/modules")
                list(APPEND CMAKE_MODULE_PATH ${ASCEND_3RD_LIB_PATH}/cmake/modules)
            endif ()
            if (EXISTS "${ASCEND_3RD_LIB_PATH}/json/share/cmake/nlohmann_json")
                list(APPEND CMAKE_PREFIX_PATH ${ASCEND_3RD_LIB_PATH}/json/share/cmake/nlohmann_json)
            endif ()
        endif ()
        find_package(nlohmann_json CONFIG)
    endif ()
    if (NOT ${nlohmann_json_FOUND})
        message(FATAL_ERROR "No nlohmann_json::nlohmann_json found, please refer to the ReadMe of this project for installation instructions.")
    endif ()
    message(STATUS "Use nlohmann_json::nlohmann_json from ${nlohmann_json_DIR}")
endif ()

# SecureC
if (BUILD_OPEN_PROJECT)
    set(BoundsCheck_DirName "libboundscheck-v1.1.16")
    get_filename_component(BoundsCheck_Dir "${PTO_FWK_SRC_ROOT}/third_party/${BoundsCheck_DirName}" REALPATH)
    if (NOT (EXISTS "${BoundsCheck_Dir}" AND EXISTS "${BoundsCheck_Dir}/CMakeLists.txt"))
        message(WARNING "Can't get BoundsCheck/HwSecureC Source, Please make sure BoundsCheck has been installed.")
    else ()
        message(STATUS "Use BoundsCheck/HwSecureC from ${BoundsCheck_Dir}")

        get_filename_component(BoundsCheck_Prefix_Dir "${CMAKE_CURRENT_BINARY_DIR}/third_party/${BoundsCheck_DirName}" REALPATH)
        get_filename_component(BoundsCheck_Source_Dir "${BoundsCheck_Dir}" REALPATH)
        get_filename_component(BoundsCheck_Install_Dir "${BoundsCheck_Prefix_Dir}/output" REALPATH)
        ExternalProject_Add(ExternalProject_BoundsCheck
                PREFIX ${BoundsCheck_Prefix_Dir}
                SOURCE_DIR ${BoundsCheck_Source_Dir}
                INSTALL_DIR ${BoundsCheck_Install_Dir}
                CONFIGURE_COMMAND ${CMAKE_COMMAND}
                    -G ${CMAKE_GENERATOR}
                    -S <SOURCE_DIR>
                    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                    -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
                    -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
                BUILD_ALWAYS FALSE
                EXCLUDE_FROM_ALL TRUE
                BUILD_BYPRODUCTS
                    ${BoundsCheck_Install_Dir}/include
                    ${BoundsCheck_Install_Dir}/lib/libc_sec.so
        )

        add_library(boundscheck_shared SHARED IMPORTED)
        set_target_properties(boundscheck_shared PROPERTIES
                IMPORTED_LOCATION ${BoundsCheck_Install_Dir}/lib/libc_sec.so
        )
        add_library(boundscheck INTERFACE)
        set_target_properties(boundscheck PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${BoundsCheck_Install_Dir}/include"
                INTERFACE_LINK_LIBRARIES "boundscheck_shared"
        )
        add_dependencies(boundscheck ExternalProject_BoundsCheck)
        if (ENABLE_FEATURE_PYTHON_FRONT_END)
            install(FILES ${BoundsCheck_Install_Dir}/lib/libc_sec.so DESTINATION pto/lib)
        endif ()
    endif ()
endif ()

# torch optional
if (ENABLE_TESTS)
    PTO_Fwk_AnalysisPython3Environ(torch_Version GET_TORCH_VERSION)
    message(STATUS "Torch=${torch_Version}")
    if ("${torch_Version}" STRGREATER_EQUAL "2.1.0")
        set(ENABLE_TORCH_VERIFIER ON)
        execute_process(
                COMMAND ${Python3_EXECUTABLE} -c "import torch;print(torch.utils.cmake_prefix_path);print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
                OUTPUT_VARIABLE TORCH_ENV_OUTPUT
        )
        string(REPLACE "\n" ";" _TORCH_ENV_LIST "${TORCH_ENV_OUTPUT}")
        list(GET _TORCH_ENV_LIST 0 TORCH_ROOT_PATH)
        list(GET _TORCH_ENV_LIST 1 TORCH_ABI_VERSION)
        get_filename_component(TORCH_ROOT_PATH "${TORCH_ROOT_PATH}/../.." REALPATH)
    endif()
endif()
