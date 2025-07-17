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
 * \file test_remove_alloc.cpp
 * \brief Unit test for Remove Alloc pass.
 */

#include "gtest/gtest.h"
#include "operation/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"
#include "passes/execute_graph_pass/remove_alloc.h"
#include <vector>
#include <string>

using namespace npu::tile_fwk;

class RemoveAllocTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "RemoveAllocTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

TEST_F(RemoveAllocTest, RemoveAlloc) {
    PROGRAM("RemoveAllocTest") {
        int N = 2;
        int T = 8;
        std::vector<int> shape{N * T, N * T};
        Program::GetInstance().GetTileShape().SetVecTileShapes({2, 2});

        Tensor a(DT_FP32, shape, "a");
        Tensor b(DT_FP32, shape, "b");
        Tensor mulResult(DT_FP32, {T, T}, "mulResult");
        PassManager &passManager = PassManager::Instance();
        passManager.RegisterStrategy("RemoveAllocTestStrategy", {
            {   "RemoveRedundentReshape",   "RemoveRedundentReshape",  PassType::TYPE_TENSOR_GRAPH},
            {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
            {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
            {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
            {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
            {       "InsertConvertOp_01",          "InsertConvertOp",    PassType::TYPE_TILE_GRAPH},
            {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},
            {       "SplitReshapeOpPVC2",       "SplitReshapeOpPVC2",    PassType::TYPE_TILE_GRAPH},
            {        "RemoveRedundentOp",        "RemoveRedundentOp",    PassType::TYPE_TILE_GRAPH},
            {        "GenerateMoveOp_01",           "GenerateMoveOp",    PassType::TYPE_TILE_GRAPH},
            {            "GraphInitPass",            "GraphInitPass",    PassType::TYPE_TILE_GRAPH},
            {          "PartitionVCPass",          "PartitionVCPass",    PassType::TYPE_TILE_GRAPH},
            {        "GraphPartitionPass",        "GraphPartitionPass",    PassType::TYPE_TILE_GRAPH},
            {         "NBufferMergePass",         "NBufferMergePass",    PassType::TYPE_TILE_GRAPH},
            {          "UpdateMemoryMap",          "UpdateMemoryMap",    PassType::TYPE_TILE_GRAPH},
            {       "InsertConvertOp_02",          "InsertConvertOp",    PassType::TYPE_TILE_GRAPH},
            {        "GenerateMoveOp_02",           "GenerateMoveOp",    PassType::TYPE_TILE_GRAPH},
            {   "SplitLargeLocalRawPass",   "SplitLargeLocalRawPass",    PassType::TYPE_TILE_GRAPH},
            {         "InsertCopyOpPass",         "InsertCopyOpPass",    PassType::TYPE_TILE_GRAPH},
            { "CommonOperationEliminate", "CommonOperationEliminate",    PassType::TYPE_TILE_GRAPH},
            {        "L1CopyInReusePass",        "L1CopyInReusePass",    PassType::TYPE_TILE_GRAPH},
            {             "PreGraphPass",             "PreGraphPass",    PassType::TYPE_TILE_GRAPH},
            {           "PadLocalBuffer",           "PadLocalBuffer",    PassType::TYPE_TILE_GRAPH},
            {       "SubgraphToFunction",       "SubgraphToFunction", PassType::TYPE_EXECUTE_GRAPH},
            {    "SrcDstBufferMergePass",    "SrcDstBufferMergePass", PassType::TYPE_EXECUTE_GRAPH},
            {             "AddAllocPass",             "AddAllocPass", PassType::TYPE_EXECUTE_GRAPH},
            {          "OoOSchedulePass",          "OoOSchedulePass", PassType::TYPE_EXECUTE_GRAPH},
            {              "MemoryReuse",              "MemoryReuse", PassType::TYPE_EXECUTE_GRAPH},

        });
        ConfigManager::Instance();

        std::vector<int> originOpmagic;
        FUNCTION("PerfectlyMatch") {
            config::SetPassStrategy("RemoveAllocTestStrategy");

            auto resultTmp = View(a, {10, 10}, {2, 2});
            auto c = View(resultTmp, {4, 4}, {0, 0});
            auto d = View(resultTmp, {4, 4}, {0, 4});
            auto e = View(resultTmp, {4, 4}, {4, 0});
            auto f = View(resultTmp, {4, 4}, {4, 4});

            auto addResult0 = Add(c, d);
            Program::GetInstance().GetTileShape().SetVecTileShapes({2, 2});
            auto addResult1 = Add(e, f);
            mulResult = Mul(addResult0, addResult1);
        }
        std::string jsonFilePath = "./config/pass/json/remove_alloc.json";
        bool dumpJsonFlag = true;
        if (dumpJsonFlag) {
            auto programJson = Program::GetInstance().DumpJson();
            DumpJsonFile(programJson, jsonFilePath);
        }
        Json readData = LoadJsonFile(jsonFilePath);
        Program::GetInstance().LoadJson(readData);

        // Call the pass
        Function* func = Program::GetInstance().GetCurrentFunction();
        npu::tile_fwk::RemoveAllocPass removeAllocPass;
        removeAllocPass.PreCheck(*func);
        removeAllocPass.RunOnFunction(*func);
        removeAllocPass.PostCheck(*func);

        // ================== Verify Pass Effect ==================
        auto updatedOperations = Program::GetInstance().GetFunctionByRawName("TENSOR_PerfectlyMatch")->Operations();
        int allocNum = 0;
        for (size_t i = 0; i < updatedOperations.size(); i++) {
            if (updatedOperations[i].GetOpcodeStr().find("ALLOC") != std::string::npos) {
                allocNum++;
            }
        }
        constexpr int allocNumExpected = 0;
        EXPECT_EQ(allocNum, allocNumExpected) << "0 alloc operations";
    }
}
