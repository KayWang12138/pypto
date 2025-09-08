# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

find_package(PythonLibs REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(pto_fwk_py
        MODULE
            src/pybind11.cpp
            src/bindings/enum.cpp
            src/bindings/tensor.cpp
            src/bindings/symbolic_scalar.cpp
            src/bindings/controller.cpp
            src/bindings/operation.cpp
)

target_include_directories(pto_fwk_py
        PRIVATE
            src
)

target_link_libraries(pto_fwk_py
        PRIVATE
            c_sec
            tile_fwk_passes
            tile_fwk_interface
            tile_fwk_codegen
            tile_fwk_operator
            tile_fwk_compiler
            tile_fwk_simulation
        PRIVATE
            intf_pub_cxx17
            tile_fwk_intf_pub
)

set_target_properties(pto_fwk_py
        PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
            CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE
)
