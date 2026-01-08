#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

cmake_minimum_required(VERSION 3.16)
project(npucomm LANGUAGES CXX)

# NPU Communication Library (NPUCOMM)
# Based on NVSHMEM design principles, optimized for NPU fine-grain communication

# Dependencies
find_package(Threads REQUIRED)

# Locate tile_fwk_interface when building standalone (outside main project)
set(TILE_FWK_INTERFACE_SEARCH_PATHS
    $ENV{TILE_FWK_INTERFACE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../build/framework/src/interface
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../build/framework/src/interface
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../build/output/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../build/output/lib
    ${CMAKE_BINARY_DIR}/../interface        # when built from project root
)

find_library(TILE_FWK_INTERFACE_LIB
    NAMES tile_fwk_interface
    HINTS ${TILE_FWK_INTERFACE_SEARCH_PATHS}
)

if(NOT TILE_FWK_INTERFACE_LIB)
    message(FATAL_ERROR "tile_fwk_interface not found. "
        "Set TILE_FWK_INTERFACE_DIR to the directory containing libtile_fwk_interface.so "
        "or build from the project root so the target is available.")
endif()

# NPU Communication Library (NPUCOMM)
# Based on NVSHMEM design principles, optimized for NPU fine-grain communication

set(NPUCOMM_SOURCES
    npucomm.cpp
    nvshmem.cpp
)

set(NPUCOMM_HEADERS
    npucomm.h
    nvshmem.h
)

# Create NPUCOMM library
add_library(npucomm SHARED ${NPUCOMM_SOURCES} ${NPUCOMM_HEADERS})

# Set library properties
set_target_properties(npucomm PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
    OUTPUT_NAME npucomm
)

# Include directories
target_include_directories(npucomm
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link dependencies
target_link_libraries(npucomm
    PUBLIC
        ${TILE_FWK_INTERFACE_LIB}
    PRIVATE
        Threads::Threads
)

# Compiler features and options
target_compile_features(npucomm PUBLIC cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(npucomm PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -O3
        -march=native
    )
elseif(MSVC)
    target_compile_options(npucomm PRIVATE
        /W4
        /O2
    )
endif()

# Installation
install(TARGETS npucomm
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(FILES ${NPUCOMM_HEADERS}
    DESTINATION include
)

# Also install nvshmem.h in a separate location for compatibility
install(FILES nvshmem.h
    DESTINATION include/nvshmem
)

# Optional: Build examples and tests
option(BUILD_NPUCOMM_EXAMPLES "Build NPUCOMM examples" ON)
option(BUILD_NPUCOMM_TESTS "Build NPUCOMM tests" ON)

if(BUILD_NPUCOMM_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_NPUCOMM_TESTS)
    add_subdirectory(tests)
endif()

# Print configuration summary
message(STATUS "NPUCOMM Configuration:")
message(STATUS "  Version: 1.0.0")
message(STATUS "  Build examples: ${BUILD_NPUCOMM_EXAMPLES}")
message(STATUS "  Build tests: ${BUILD_NPUCOMM_TESTS}")
message(STATUS "  NPU optimization: Enabled")
message(STATUS "  PGAS support: Enabled")
message(STATUS "  Fine-grain communication: Enabled")
