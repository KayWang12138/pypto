# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

if (metadef_FOUND)
  message(STATUS "Package metadef has been found.")
  return()
endif()

set(_cmake_targets_defined "")
set(_cmake_targets_not_defined "")
set(_cmake_expected_targets "")
foreach(_cmake_expected_target IN ITEMS exe_graph lowering error_manager graph graph_base flow_graph register rt2_registry_static metadef_headers)
  list(APPEND _cmake_expected_targets "${_cmake_expected_target}")
  if(TARGET "${_cmake_expected_target}")
    list(APPEND _cmake_targets_defined "${_cmake_expected_target}")
  else()
    list(APPEND _cmake_targets_not_defined "${_cmake_expected_target}")
  endif()
endforeach()
unset(_cmake_expected_target)
if(_cmake_targets_defined STREQUAL _cmake_expected_targets)
  unset(_cmake_targets_defined)
  unset(_cmake_targets_not_defined)
  unset(_cmake_expected_targets)
  return()
endif()
if(NOT _cmake_targets_defined STREQUAL "")
  string(REPLACE ";" ", " _cmake_targets_defined_text "${_cmake_targets_defined}")
  string(REPLACE ";" ", " _cmake_targets_not_defined_text "${_cmake_targets_not_defined}")
  message(FATAL_ERROR "Some (but not all) targets in this export set were already defined.\nTargets Defined: ${_cmake_targets_defined_text}\nTargets not yet defined: ${_cmake_targets_not_defined_text}\n")
endif()
unset(_cmake_targets_defined)
unset(_cmake_targets_not_defined)
unset(_cmake_expected_targets)

# Compute the installation prefix relative to this file.
set(metadef_COMPILER_PREFIX ${ASCEND_HOME_PATH}/compiler)
set(metadef_RUNTIME_PREFIX ${ASCEND_HOME_PATH}/runtime)
set(metadef_PREFIX ${TILE_FWK_OPS_ROOT_PATH}/../../../../metadef)

include(CMakePrintHelpers)
cmake_print_variables(metadef_COMPILER_PREFIX)
cmake_print_variables(metadef_RUNTIME_PREFIX)
cmake_print_variables(metadef_PREFIX)

add_library(exe_graph SHARED IMPORTED)

add_library(lowering SHARED IMPORTED)

add_library(error_manager SHARED IMPORTED)

set_target_properties(error_manager PROPERTIES
  INTERFACE_LINK_LIBRARIES "metadef_headers"
)

add_library(graph SHARED IMPORTED)

set_target_properties(graph PROPERTIES
  INTERFACE_LINK_LIBRARIES "metadef_headers"
)

add_library(graph_base SHARED IMPORTED)

set_target_properties(graph_base PROPERTIES
  INTERFACE_LINK_LIBRARIES "metadef_headers"
)

add_library(flow_graph SHARED IMPORTED)

set_target_properties(flow_graph PROPERTIES
  INTERFACE_LINK_LIBRARIES "metadef_headers"
)

add_library(register SHARED IMPORTED)

set_target_properties(register PROPERTIES
  INTERFACE_LINK_LIBRARIES "metadef_headers"
)

add_library(rt2_registry_static STATIC IMPORTED)

set_target_properties(rt2_registry_static PROPERTIES
  INTERFACE_LINK_LIBRARIES "metadef_headers"
)

add_library(metadef_headers INTERFACE IMPORTED)

target_include_directories(metadef_headers INTERFACE
    ${metadef_PREFIX}
    ${metadef_PREFIX}/inc
    ${metadef_PREFIX}/inc/common
    ${metadef_PREFIX}/inc/exe_graph
    ${metadef_PREFIX}/inc/external
    ${metadef_PREFIX}/inc/external/exe_graph/runtime
    ${metadef_PREFIX}/inc/external/ge
    ${metadef_PREFIX}/inc/graph
    ${metadef_PREFIX}/inc/graph/utils
    ${metadef_PREFIX}/inc/register
    ${metadef_PREFIX}/third_party/transformer
    ${metadef_PREFIX}/third_party/transformer/inc
    ${metadef_PREFIX}/graph
    ${metadef_PREFIX}/graph/debug
    ${metadef_PREFIX}/register
    ${metadef_PREFIX}/register/op_tiling
)

if(CMAKE_VERSION VERSION_LESS 3.0.0)
  message(FATAL_ERROR "This file relies on consumers using CMake 3.0.0 or greater.")
endif()

set_property(TARGET exe_graph APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(exe_graph PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "c_sec;slog"
  IMPORTED_LOCATION_DEBUG "${metadef_COMPILER_PREFIX}/lib64/libexe_graph.so"
  IMPORTED_SONAME_DEBUG "libexe_graph.so"
)

set_property(TARGET lowering APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(lowering PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "c_sec;slog;ascend_protobuf;graph;register"
  IMPORTED_LOCATION_DEBUG "${metadef_COMPILER_PREFIX}/lib64/liblowering.so"
  IMPORTED_SONAME_DEBUG "liblowering.so"
)

set_property(TARGET error_manager APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(error_manager PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "c_sec;slog"
  IMPORTED_LOCATION_DEBUG "${metadef_COMPILER_PREFIX}/lib64/liberror_manager.so"
  IMPORTED_SONAME_DEBUG "liberror_manager.so"
)

set_property(TARGET graph APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(graph PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "graph_base;c_sec;slog;platform;error_manager"
  IMPORTED_LOCATION_DEBUG "${metadef_COMPILER_PREFIX}/lib64/libgraph.so"
  IMPORTED_SONAME_DEBUG "libgraph.so"
)

set_property(TARGET graph_base APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(graph_base PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "c_sec;slog;ascend_protobuf;error_manager"
  IMPORTED_LOCATION_DEBUG "${metadef_COMPILER_PREFIX}/lib64/libgraph_base.so"
  IMPORTED_SONAME_DEBUG "libgraph_base.so"
)

set_property(TARGET flow_graph APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(flow_graph PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "graph_base;graph;c_sec;slog;ascend_protobuf;error_manager"
  IMPORTED_LOCATION_DEBUG "${metadef_COMPILER_PREFIX}/lib64/libflow_graph.so"
  IMPORTED_SONAME_DEBUG "libflow_graph.so"
)

set_property(TARGET register APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(register PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "ascend_protobuf;c_sec;slog;platform;graph"
  IMPORTED_LOCATION_DEBUG "${metadef_COMPILER_PREFIX}/lib64/libregister.so"
  IMPORTED_SONAME_DEBUG "libregister.so"
)

set_property(TARGET rt2_registry_static APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(rt2_registry_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${metadef_RUNTIME_PREFIX}/lib64/librt2_registry.a"
)

# Cleanup temporary variables.
set(metadef_COMPILER_PREFIX)
