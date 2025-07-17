# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_library(tile_fwk_intf_pub INTERFACE)
target_include_directories(tile_fwk_intf_pub
        INTERFACE   # 源码依赖
            ${TILE_FWK_SRC_ROOT}/include
            ${TILE_FWK_SRC_ROOT}/src
)
target_compile_options(tile_fwk_intf_pub
        INTERFACE
            # 安全编译选项
            $<$<CONFIG:Release>:-O2 -D_FORTIFY_SOURCE=2>
            # 基础要求选项
            $<$<CONFIG:Debug>:-g>
            # 告警增强选项
            -Wextra
            -Wundef
            -Wunused
            -Wcast-qual
            -Wpointer-arith
            -Wdate-time
            -Wunused-macros
            -Wfloat-equal
            -Wformat=2
            -Wshadow
            -Wsign-compare
            -Wunused-macros
            -Wvla
            -Wdisabled-optimization
            -Wempty-body
            -Wignored-qualifiers
            $<$<CXX_COMPILER_ID:GNU>:-Wimplicit-fallthrough=3>
            -Wtype-limits
            -Wshift-negative-value
            -Wswitch-default
            -Wframe-larger-than=$<IF:$<OR:$<BOOL:${ENABLE_ASAN}>,$<BOOL:${ENABLE_UBSAN}>>,131072,32768>
            -Woverloaded-virtual
            -Wnon-virtual-dtor
            $<$<CXX_COMPILER_ID:GNU>:-Wshift-overflow=2>
            -Wshift-count-overflow
            -Wwrite-strings
            -Wmissing-format-attribute
            -Wformat-nonliteral
            -Wdelete-non-virtual-dtor
            $<$<CXX_COMPILER_ID:GNU>:-Wduplicated-cond>
            $<$<CXX_COMPILER_ID:GNU>:-Wtrampolines>
            $<$<CXX_COMPILER_ID:GNU>:-Wsized-deallocation>
            $<$<CXX_COMPILER_ID:GNU>:-Wlogical-op>
            $<$<CXX_COMPILER_ID:GNU>:-Wsuggest-attribute=format>
            $<$<COMPILE_LANGUAGE:C>:-Wnested-externs>
            $<$<CXX_COMPILER_ID:GNU>:-Wduplicated-branches>
            -Wmissing-include-dirs
            $<$<CXX_COMPILER_ID:GNU>:-Wformat-signedness>
            $<$<CXX_COMPILER_ID:GNU>:-Wreturn-local-addr>
            -Wredundant-decls
            -Wfloat-conversion
            $<$<CXX_COMPILER_ID:Clang>:-Wno-tautological-unsigned-enum-zero-compare>
            -fno-common
            -fno-strict-aliasing
            # 放在最后
            $<$<CONFIG:Release>:-Wno-return-type>
            $<$<CONFIG:Release>:-Wno-array-bounds>
            $<$<CONFIG:Release>:-Wno-maybe-uninitialized>
            $<$<CONFIG:Release>:-Wno-unused-but-set-variable>
            $<$<CONFIG:Release>:-Wno-unused-variable>
            $<$<CONFIG:Release>:-Wno-unused-parameter>
            $<$<CONFIG:Release>:-Wno-unused-result>
            -Werror
            # 依赖分析选项
            $<$<OR:$<BOOL:${ENABLE_TESTS_UTEST}>,$<BOOL:${ENABLE_TESTS_STEST}>,$<BOOL:${ENABLE_TESTS_STEST_DISTRIBUTED}>>:-MMD>
)
target_link_options(tile_fwk_intf_pub
        INTERFACE
            # 安全编译选项
            $<$<CONFIG:Release>:-s>
)

if (BUILD_OPEN_PROJECT)
    add_library(intf_pub_cxx17 INTERFACE)
    target_compile_options(intf_pub_cxx17
            INTERFACE
                # 安全编译选项
                -fPIC
                $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:-pie>
                $<$<CXX_COMPILER_ID:GNU>:$<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>>
                $<$<CXX_COMPILER_ID:Clang>:$<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},10.0.0>,-fstack-protector-strong,-fstack-protector-all>>
                # 基础要求选项
                -Wall
                # GCOV
                $<$<BOOL:${ENABLE_GCOV}>:$<$<CXX_COMPILER_ID:GNU>:--coverage -fprofile-arcs -ftest-coverage>>
                # ASAN
                $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address -fsanitize-address-use-after-scope -fsanitize=leak>
                # UBSAN
                # 在 Clang 编译器场景下 使能 -fsanitize=undefined 会默认开启基本所有的 UBSAN 检查项, 只有以下检查项不会开启
                #   float-divide-by-zero, unsigned-integer-overflow, implicit-conversion, local-bounds 及 nullability-* 类检查.
                # 故在 Clang 编译器使能 UBSAN 场景下, 需开启 -fsanitize=undefined 使能时仍未开启的对应检查项
                # 在 GNU 编译器场景下, 官方文档并未对使能 -fsanitize=undefined 时开启的默认检查项范围进行说明, 故手工开启常用基本检查项, 避免能力遗漏
                $<$<BOOL:${ENABLE_UBSAN}>:-fsanitize=undefined -fsanitize=float-divide-by-zero -fno-sanitize=alignment>
                $<$<BOOL:${ENABLE_UBSAN}>:$<$<CXX_COMPILER_ID:Clang>:-fsanitize=unsigned-integer-overflow>>    # GNU 不支持这些检查项
                $<$<BOOL:${ENABLE_UBSAN}>:$<$<CXX_COMPILER_ID:Clang>:$<$<VERSION_GREATER_EQUAL:${CMAKE_C_COMPILER_VERSION},10.0.0>:-fsanitize=implicit-conversion>>>    # GNU 不支持这些检查项, Clang高版本才支持这些检查项
                $<$<BOOL:${ENABLE_UBSAN}>:$<$<CXX_COMPILER_ID:GNU>:-fsanitize=shift>>
                $<$<BOOL:${ENABLE_UBSAN}>:$<$<CXX_COMPILER_ID:GNU>:-fsanitize=integer-divide-by-zero>>
                $<$<BOOL:${ENABLE_UBSAN}>:$<$<CXX_COMPILER_ID:GNU>:-fsanitize=signed-integer-overflow>>
                $<$<BOOL:${ENABLE_UBSAN}>:$<$<CXX_COMPILER_ID:GNU>:-fsanitize=float-divide-by-zero>>
                $<$<BOOL:${ENABLE_UBSAN}>:$<$<CXX_COMPILER_ID:GNU>:-fsanitize=float-cast-overflow>>
                $<$<BOOL:${ENABLE_UBSAN}>:$<$<CXX_COMPILER_ID:GNU>:-fsanitize=bool>>
                $<$<BOOL:${ENABLE_UBSAN}>:$<$<CXX_COMPILER_ID:GNU>:-fsanitize=enum>>
                $<$<BOOL:${ENABLE_UBSAN}>:$<$<CXX_COMPILER_ID:GNU>:-fsanitize=vptr>>
                # ASAN/UBSAN 公共
                $<$<OR:$<BOOL:${ENABLE_ASAN}>,$<BOOL:${ENABLE_UBSAN}>>:-fno-omit-frame-pointer -fsanitize-recover=all>
    )
    target_compile_definitions(intf_pub_cxx17
            INTERFACE
                $<$<COMPILE_LANGUAGE:CXX>:_GLIBCXX_USE_CXX11_ABI=0>    # 必须设置, 以保证与 CANN 包内其他 C++ 二进制兼容
    )
    target_link_libraries(intf_pub_cxx17
            INTERFACE
                $<$<BOOL:${ENABLE_GCOV}>:$<$<CXX_COMPILER_ID:GNU>:gcov>>
    )
    target_link_options(intf_pub_cxx17
            INTERFACE
                # 安全编译选项
                -Wl,-z,relro
                -Wl,-z,now
                -Wl,-z,noexecstack
                # GCOV
                $<$<BOOL:${ENABLE_GCOV}>:$<$<CXX_COMPILER_ID:GNU>:-fprofile-arcs -ftest-coverage>>
                # ASAN
                $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address>
                # UBSAN
                $<$<BOOL:${ENABLE_UBSAN}>:-fsanitize=undefined>
    )
endif()
