# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

cmake_minimum_required(VERSION 3.14)
project(tile_fwk_ops)

option(BUILD_OPEN_PROJECT "Build open ascend ops project."  FALSE)
option(ENABLE_BUILD_HOST "Build host."  TRUE)
option(ENABLE_BUILD_BINARY "Build binary."  FALSE)

set(TILE_FWK_OPS_ROOT_PATH ${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_VERBOSE_MAKEFILE True)
set(INSTALL_LIBRARY_DIR lib)

if (NOT DEFINED CMAKE_MODULE_PATH)
        set(CMAKE_MODULE_PATH
                ${CMAKE_CURRENT_LIST_DIR}/cmake/modules
        )
endif()

if (NOT DEFINED CMAKE_PREFIX_PATH)
        set(CMAKE_PREFIX_PATH
                ${ASCEND_HOME_PATH}
        )
endif()

# cann packages
include(cmake/find_dependence_packages.cmake)

if (ENABLE_BUILD_BINARY)
        # tile_fwk_impl
        add_library(tile_fwk_impl SHARED)

        target_compile_options(tile_fwk_impl PRIVATE
                -fvisibility=hidden
        )

        target_link_directories(tile_fwk_impl PRIVATE
                ${ASCEND_HOME_PATH}/lib64
        )

        target_link_libraries(tile_fwk_impl PRIVATE
                $<BUILD_INTERFACE:intf_pub_cxx17>
                c_sec
                ascendalog
                tile_fwk_interface
                nlohmann_json::nlohmann_json
        )

        target_include_directories(tile_fwk_impl PRIVATE
                ${ASCEND_HOME_PATH}/include
        )

        install(TARGETS tile_fwk_impl OPTIONAL
                EXPORT tile_fwk_impl-targets
                LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
        )

        # nsa op kernel binary
        string(REPLACE "," ";" COMPUTE_UNIT_LIST "${ASCEND_COMPUTE_UNIT}")
        foreach(COMPUTE_UNIT IN LISTS COMPUTE_UNIT_LIST)
                add_custom_command(
                        TARGET tile_fwk_impl POST_BUILD
                        COMMAND bash ${TILE_FWK_OPS_ROOT_PATH}/scripts/kernels_compile.sh
                        ${COMPUTE_UNIT}
                        ${CMAKE_CURRENT_SOURCE_DIR}/native_sparse_attention/op_kernel/bin_param
                        ${CMAKE_CURRENT_SOURCE_DIR}/python/tile_fwk_compiler.py
                        ${CMAKE_CURRENT_BINARY_DIR}/libtile_fwk_impl.so
                        ${TILE_FWK_OPS_ROOT_PATH}/../output/lib64/libtile_fwk_compiler.so
                        ${BINARY_OUTPUT_PATH}/${COMPUTE_UNIT}/native_sparse_attention
                )
        endforeach()
endif()

if (BUILD_OPEN_PROJECT AND ENABLE_BUILD_HOST)
        set(METADEF_PROTO_DIR ${TILE_FWK_OPS_ROOT_PATH}/../../../../metadef/proto)
        file(GLOB_RECURSE PROTO_LIST RELATIVE ${CMAKE_CURRENT_LIST_DIR}
                "${METADEF_PROTO_DIR}/task.proto"
                "${METADEF_PROTO_DIR}/ge_ir.proto"
        )
        protobuf_generate(tile_fwk_proto PROTO_SRCS PROTO_HDRS ${PROTO_LIST} TARGET)

        # op proto
        add_library(opsproto_rt2.0 SHARED
                ${PROTO_HDRS}
                ${PROTO_SRCS}
        )
        target_include_directories(opsproto_rt2.0 PRIVATE
                ${CMAKE_BINARY_DIR}/proto/tile_fwk_proto
                ${ASCEND_HOME_PATH}/include/experiment/runtime
                ${ASCEND_HOME_PATH}/include/experiment/msprof
        )
        target_compile_options(opsproto_rt2.0 PRIVATE
                -fvisibility=hidden
        )
        target_compile_definitions(opsproto_rt2.0 PRIVATE
                LOG_CPP
                PROCESS_LOG
        )
        target_link_libraries(opsproto_rt2.0 PRIVATE
                $<BUILD_INTERFACE:intf_pub>
                $<$<BOOL:${alog_FOUND}>:$<BUILD_INTERFACE:alog_headers>>
                -Wl,--whole-archive
                rt2_registry_static
                -Wl,--no-whole-archive
                -Wl,--no-as-needed
                exe_graph
                graph
                graph_base
                register
                alog
                error_manager
                platform
                -Wl,--as-needed
                c_sec
                ascend_protobuf
        )
        set_target_properties(opsproto_rt2.0 PROPERTIES OUTPUT_NAME
                cust_opsproto_rt2.0
        )
        install(TARGETS opsproto_rt2.0
                LIBRARY DESTINATION op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR}
        )

        # op proto
        add_library(opmaster_ct SHARED
                ${PROTO_HDRS}
                ${PROTO_SRCS}
        )
        target_include_directories(opmaster_ct PRIVATE
                ${CMAKE_BINARY_DIR}/proto/tile_fwk_proto
                ${ASCEND_HOME_PATH}/include/experiment/runtime
                ${ASCEND_HOME_PATH}/include/experiment/msprof
        )
        target_compile_options(opmaster_ct PRIVATE
                -fvisibility=hidden
        )
        target_compile_definitions(opmaster_ct PRIVATE
                LOG_CPP
                PROCESS_LOG
        )
        target_link_libraries(opmaster_ct PRIVATE
                $<BUILD_INTERFACE:intf_pub>
                $<$<BOOL:${alog_FOUND}>:$<BUILD_INTERFACE:alog_headers>>
                -Wl,--whole-archive
                rt2_registry_static
                -Wl,--no-whole-archive
                -Wl,--no-as-needed
                exe_graph
                graph
                graph_base
                register
                alog
                error_manager
                platform
                -Wl,--as-needed
                c_sec
                ascend_protobuf
        )
        set_target_properties(opmaster_ct PROPERTIES OUTPUT_NAME
                cust_opmaster_ct
        )
        install(TARGETS opmaster_ct
                LIBRARY DESTINATION op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR}
        )

        # op tiling
        add_library(opmaster_rt2.0 SHARED)
        target_compile_options(opmaster_rt2.0 PRIVATE
                -fvisibility=hidden
        )
        target_compile_definitions(opmaster_rt2.0 PRIVATE
                LOG_CPP
                PROCESS_LOG
        )
        target_link_libraries(opmaster_rt2.0 PRIVATE
                $<BUILD_INTERFACE:intf_pub>
                $<$<BOOL:${alog_FOUND}>:$<BUILD_INTERFACE:alog_headers>>
                -Wl,--whole-archive
                rt2_registry_static
                -Wl,--no-whole-archive
                -Wl,--no-as-needed
                graph
                graph_base
                exe_graph
                platform
                register
                alog
                error_manager
                -Wl,--as-needed
                -Wl,--whole-archive
                -Wl,--no-whole-archive
                mmpa
                c_sec
        )
        set_target_properties(opmaster_rt2.0 PROPERTIES OUTPUT_NAME
                cust_opmaster_rt2.0
        )
        install(TARGETS opmaster_rt2.0
                LIBRARY DESTINATION op_impl/ai_core/tbe/op_tiling/lib/linux/${CMAKE_SYSTEM_PROCESSOR}
        )
endif ()

if (ENABLE_BUILD_HOST)
        target_sources(opsproto_rt2.0 PRIVATE
                src/utils/tile_fwk_op_utils.cpp
        )

        target_include_directories(opsproto_rt2.0 PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/src/
        )

        target_sources(opmaster_ct PRIVATE
                src/utils/tile_fwk_op_utils.cpp
        )

        target_include_directories(opmaster_ct PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/src/
        )
endif ()

add_subdirectory(native_sparse_attention)
