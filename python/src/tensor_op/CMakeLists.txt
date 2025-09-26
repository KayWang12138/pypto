# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

pybind11_add_module(pto_impl
        MODULE
            src/pybind11.cpp
            src/bindings/enum.cpp
            src/bindings/tensor.cpp
            src/bindings/symbolic_scalar.cpp
            src/bindings/controller.cpp
            src/bindings/operation.cpp
            src/bindings/runtime.cpp
)

target_include_directories(pto_impl
        PRIVATE
            src
            ${PTO_FWK_SRC_ROOT}/tests
)

target_link_libraries(pto_impl
        PRIVATE
            $<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,boundscheck,c_sec>
            tile_fwk_passes
            tile_fwk_interface
            tile_fwk_codegen
            tile_fwk_operator
            tile_fwk_compiler
            tile_fwk_simulation
            $<$<AND:$<BOOL:${ENABLE_BUILD_WITH_CANN}>,$<BOOL:${ENABLE_TESTS}>>:tile_fwk_runtime>
        PRIVATE
            intf_pub_cxx17
            tile_fwk_intf_pub
)
