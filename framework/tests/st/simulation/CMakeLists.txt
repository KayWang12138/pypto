# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================


set(_Target ${PTO_Fwk_STestNamePrefix}_simulation)

file(GLOB_RECURSE _Sources src/*.cpp)

PTO_Fwk_STest_AddLib(
        TARGET ${_Target}
        SOURCES ${_Sources}
        LD_LIBRARIES_EXT tile_fwk_compiler tile_fwk_simulation tile_fwk_simulation_ca
)
# 可执行程序运行依赖的配置, 脚本配置
add_custom_command(
        TARGET ${_Target} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PTO_Fwk_STestExePath}/config/
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/config/* ${PTO_Fwk_STestExePath}/config/
        COMMENT "Copy ${_Target} dependence config file."
)
add_custom_command(
        TARGET ${_Target} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PTO_Fwk_STestExePath}/scripts/
        COMMAND ${CMAKE_COMMAND} -E copy ${PTO_FWK_SRC_ROOT}/framework/src/cost_model/simulation/scripts/print_swim_lane.py ${PTO_Fwk_STestExePath}/scripts/
        COMMAND ${CMAKE_COMMAND} -E copy ${PTO_FWK_SRC_ROOT}/framework/src/cost_model/simulation/scripts/draw_comm_swim_lane_png.py ${PTO_Fwk_STestExePath}/scripts/
        COMMAND ${CMAKE_COMMAND} -E copy ${PTO_FWK_SRC_ROOT}/framework/src/cost_model/simulation/scripts/draw_pipe_swim_lane.py ${PTO_Fwk_STestExePath}/scripts/
        COMMAND ${CMAKE_COMMAND} -E copy ${PTO_FWK_SRC_ROOT}/tools/draw_swim_lane.py ${PTO_Fwk_STestExePath}/scripts/
        COMMAND ${CMAKE_COMMAND} -E copy ${PTO_FWK_SRC_ROOT}/tools/function_json_convert.py ${PTO_Fwk_STestExePath}/scripts/
        COMMAND ${CMAKE_COMMAND} -E copy ${PTO_FWK_SRC_ROOT}/tools/parse_pipe_time_trace.py ${PTO_Fwk_STestExePath}/scripts/
        COMMENT "Copy ${_Target} dependence scripts file."
)
