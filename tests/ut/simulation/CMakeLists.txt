# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set(_CaseFilterList
        # CostModelUT
        #"CostModelTest.TestAttentionPost"
        "CostModelTest.TestAttentionPostAccuracy1"
        "CostModelTest.TestAttentionPostAccuracy2"
        "CostModelTest.TestAttentionPostAccuracyFromJson"
        "CostModelTest.TestAttentionPostFunctional"
        "CostModelTest.TestFixedLatencyTasks"
        "CostModelTest.TestErrorInput"
        "CostModelTest.TestGenCalendarSchedule"
        # "CostModelTest.TestGenCalendarScheduleFronJson"
        "CostModelTest.TestAttentionPostCVMIXMode"
        "CostModelTest.TestAttentionPostSimulationSchedule"
        "CostModelTest.TestAttentionPostL2Cache"
        "CostModelTest.TestBuildBasedOnConfigs"
        "CostModelTest.TestCoreMachineDeadlock"
        #"CostModelTest.TestReplayDispatch"
        #"CostModelTest.TestReplayAll"
        "CostModelTest.TestComm"
        "CostModelTest.TestGlobalCalendar"
        "CostModelTest.TestLeafFunctionMode"
        #"CostModelTest.TestAttentionPostBf16Real"
        "CostModelTest.TestAttentionPostPvModel"
        "PvModelTest.TestAddGlobalAttr"
        "PvModelTest.TestDynFactory"
        "PvModelTest.TestDynImpl"
        "PvModelTest.TestDynCodegen"
        "CostModelDynTest.TestDD"
)

set(_Target ${TileFwkUTestNamePrefix}_simulation)

file(GLOB_RECURSE _Sources src/*.cpp)

TileFwk_UTest_AddCaseLib(
        TARGET ${_Target}
        SOURCES ${_Sources}
        PRIVATE_INCLUDE_DIRECTORIES ${TILE_FWK_SRC_ROOT}/src/cost_model

        PUBLIC_LINK_LIBRARIES tile_fwk_simulation tile_fwk_passes tile_fwk_interface dl
        GTEST_FILTER_LIST ${_CaseFilterList}
        LD_LIBRARIES_EXT tile_fwk_simulation tile_fwk_simulation_ca tile_fwk_simulation_pv
)
# 可执行程序运行依赖的配置, 脚本配置
add_custom_command(
        TARGET ${_Target} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${TileFwkUTestExePath}/config/
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/config/* ${TileFwkUTestExePath}/config/
        COMMENT "Copy ${_Target} dependence config file."
        COMMAND ${CMAKE_COMMAND} -E make_directory ${TileFwkUTestExePath}/scripts/
        COMMAND ${CMAKE_COMMAND} -E copy ${TILE_FWK_SRC_ROOT}/src/cost_model/simulation/scripts/print_swim_lane.py ${TileFwkUTestExePath}/scripts/
        COMMAND ${CMAKE_COMMAND} -E copy ${TILE_FWK_SRC_ROOT}/src/cost_model/simulation/scripts/draw_comm_swim_lane_png.py ${TileFwkUTestExePath}/scripts/
        COMMAND ${CMAKE_COMMAND} -E copy ${TILE_FWK_SRC_ROOT}/src/cost_model/simulation/scripts/draw_pipe_swim_lane.py ${TileFwkUTestExePath}/scripts/
        COMMAND ${CMAKE_COMMAND} -E copy ${TILE_FWK_SRC_ROOT}/tools/draw_swim_lane.py ${TileFwkUTestExePath}/scripts/
        COMMAND ${CMAKE_COMMAND} -E copy ${TILE_FWK_SRC_ROOT}/tools/function_json_convert.py ${TileFwkUTestExePath}/scripts/
        COMMAND ${CMAKE_COMMAND} -E copy ${TILE_FWK_SRC_ROOT}/tools/parse_pipe_time_trace.py ${TileFwkUTestExePath}/scripts/
        COMMENT "Copy ${_Target} dependence scripts file."
)
