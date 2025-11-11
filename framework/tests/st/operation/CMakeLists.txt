# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set(_Target ${PTO_Fwk_STestNamePrefix}_operation)

file(GLOB_RECURSE _Sources src/*.cpp)

PTO_Fwk_STest_AddLib(
        TARGET ${_Target}
        SOURCES ${_Sources}
        LD_LIBRARIES_EXT tile_fwk_compiler tile_fwk_simulation tile_fwk_simulation_ca tile_fwk_simulation_pv
        GOLDEN_SCRIPT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python
)
add_custom_command(
        TARGET ${_Target} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PTO_Fwk_STestExePath}/scripts/
        COMMENT "Copy ${_Target} dependence scripts file."
)