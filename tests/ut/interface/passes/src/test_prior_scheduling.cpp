/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_prior_scheduling.cpp
 * \brief Unit test for PriorScheduling.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "passes/execute_graph_pass/prior_scheduling.h"
#include "ut_json/ut_json_tool.h"

namespace npu::tile_fwk {
class TestPriorScheduling : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "PriorSchedulingTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
        PassManager &passManager = PassManager::Instance();
        passManager.RegisterStrategy("PriorSchedulingTestStrategy", {
            {   "RemoveRedundentReshape",   "RemoveRedundentReshape",  PassType::TYPE_TENSOR_GRAPH},
            {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
            {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
            {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
            {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
            {          "InsertConvertOp",          "InsertConvertOp",    PassType::TYPE_TILE_GRAPH},
            {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},
            {       "SplitReshapeOpPVC2",       "SplitReshapeOpPVC2",    PassType::TYPE_TILE_GRAPH},
            {        "RemoveRedundentOp",        "RemoveRedundentOp",    PassType::TYPE_TILE_GRAPH},
            {           "GenerateMoveOp",           "GenerateMoveOp",    PassType::TYPE_TILE_GRAPH},
            {             "GraphInitPass",           "GraphInitPass",    PassType::TYPE_TILE_GRAPH},
            {           "PartitionVCPass",         "PartitionVCPass",    PassType::TYPE_TILE_GRAPH},
            {        "GraphPartitionPass",        "GraphPartitionPass",    PassType::TYPE_TILE_GRAPH},
            {         "NBufferMergePass",         "NBufferMergePass",    PassType::TYPE_TILE_GRAPH},
            {          "UpdateMemoryMap",          "UpdateMemoryMap",    PassType::TYPE_TILE_GRAPH},
            {   "SplitLargeLocalRawPass",   "SplitLargeLocalRawPass",    PassType::TYPE_TILE_GRAPH},
            {         "InsertCopyOpPass",         "InsertCopyOpPass",    PassType::TYPE_TILE_GRAPH},
            {      "L1CopyInReusePass",       "L1CopyInReusePass",       PassType::TYPE_TILE_GRAPH},
            { "CommonOperationEliminate", "CommonOperationEliminate",    PassType::TYPE_TILE_GRAPH},
            {             "PreGraphPass",             "PreGraphPass",    PassType::TYPE_TILE_GRAPH},
            {           "PadLocalBuffer",           "PadLocalBuffer",    PassType::TYPE_TILE_GRAPH},
            {       "SubgraphToFunction",       "SubgraphToFunction", PassType::TYPE_EXECUTE_GRAPH},
            {    "SrcDstBufferMergePass",    "SrcDstBufferMergePass", PassType::TYPE_EXECUTE_GRAPH},
            {             "AddAllocPass",             "AddAllocPass", PassType::TYPE_EXECUTE_GRAPH},
            {          "OoOSchedulePass",          "OoOSchedulePass", PassType::TYPE_EXECUTE_GRAPH},
            {          "RemoveAllocPass",          "RemoveAllocPass", PassType::TYPE_EXECUTE_GRAPH}
        });
    }
    void TearDown() override {}
};

TEST_F(TestPriorScheduling, TestMainSchedule) {
    std::vector<int> shape{64, 64};
    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, shape, "b");
    Tensor c(DT_FP32, shape, "c");
    constexpr int TILE_SHAPE = 32;
    Program::GetInstance().GetTileShape().SetVecTileShapes(TILE_SHAPE,TILE_SHAPE);
    FUNCTION("A") {
        c = Add(a, b);
    }

    std::string jsonFilePath = "./config/pass/json/prior_scheduling.json";
    bool dumpJsonFlag = true;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
    }
    Json readData = LoadJsonFile(jsonFilePath);
    Program::GetInstance().LoadJson(readData);
    Function* currentFunction = Program::GetInstance().GetCurrentFunction();

    Program testProgram(HostMachineMode::SERVER);
    PriorScheduling priorScheduling;
    priorScheduling.RunOnFunction(*currentFunction);
    // PostCheck Test
    EXPECT_EQ(priorScheduling.PostCheck(*currentFunction), SUCCESS);
}

} // namespace acend
