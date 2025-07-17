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
 * \file test_split_large_local_raw.cpp
 * \brief Unit test for SplitLargeLocalRaw pass.
 */

#include "gtest/gtest.h"
#include "operation/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"
#include "passes/tile_graph_pass/split_large_local_raw.h"
#include <vector>

using namespace npu::tile_fwk;

class SplitLargeLocalRawTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "SplitLargeLocalRawTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

TEST_F(SplitLargeLocalRawTest, SplitLocalRaw) {
    PROGRAM("SplitLargeLocalRawTest") {
        PassManager &passManager = PassManager::Instance();
        passManager.RegisterStrategy("SplitLargeLocalRawTestStrategy", {
            {   "RemoveRedundentReshape",   "RemoveRedundentReshape",  PassType::TYPE_TENSOR_GRAPH},
            {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
            {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
            {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
            {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
            {          "InsertConvertOp",          "InsertConvertOp",    PassType::TYPE_TILE_GRAPH},
            {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},
            {       "SplitReshapeOpPVC2",       "SplitReshapeOpPVC2",    PassType::TYPE_TILE_GRAPH},
            {             "GraphInitPass",           "GraphInitPass",    PassType::TYPE_TILE_GRAPH},
            {           "PartitionVCPass",         "PartitionVCPass",    PassType::TYPE_TILE_GRAPH},
            {        "GraphPartitionPass",        "GraphPartitionPass",    PassType::TYPE_TILE_GRAPH},

        });
        ConfigManager::Instance();

        int b = 1;
        int n = 2;
        int s = 128;
        int d = 512;
        int v_head =128;
        int tileSize0 = 128;
        int tileSize1 = 32;
        std::vector<int> inShape = {b * s, n, d};
        Tensor attnPostIn(DT_FP32, inShape, "attnPostIn");
        Tensor kvBProjWV(DT_FP32, {n, d, v_head}, "kvBProjWV");
        Tensor atten_output;
        FUNCTION("SplitLocalRaw") {
            config::SetPassStrategy("SplitLargeLocalRawTestStrategy");

            DataType dType = attnPostIn->Datatype();
            Program::GetInstance().GetTileShape().SetVecTileShapes({32, 1, d});
            Tensor atten_res2 = Transpose(attnPostIn, {0, 1});
        Program::GetInstance().GetTileShape().SetVecTileShapes(tileSize0, tileSize0);
            Program::GetInstance().GetTileShape().SetCubeTileShapes({tileSize1, tileSize1}, {tileSize0, tileSize0}, {tileSize0, tileSize0});
            atten_output = Matrix::BatchMatmul(dType, atten_res2, kvBProjWV);
        }
        std::string jsonFilePath = "./config/pass/json/split_large_local_raw.json";
        bool dumpJsonFlag = true;
        if (dumpJsonFlag) {
            auto programJson = Program::GetInstance().DumpJson();
            DumpJsonFile(programJson, jsonFilePath);
        }
        Json readData = LoadJsonFile(jsonFilePath);
        Program::GetInstance().LoadJson(readData);

        // Call the pass
        Function* func = Program::GetInstance().GetCurrentFunction();
        npu::tile_fwk::SplitLargeLocalRawPass splitLargeLocalRawPass;
        splitLargeLocalRawPass.PreCheck(*func);
        splitLargeLocalRawPass.RunOnFunction(*func);
        splitLargeLocalRawPass.PostCheck(*func);

        // ================== Verify Pass Effect ==================
        auto updatedOperations = Program::GetInstance().GetFunctionByRawName("TENSOR_SplitLocalRaw")->Operations();
        for (const auto &updatedOperation : updatedOperations) {
            if(updatedOperation.GetOpcode() == Opcode::OP_ASSEMBLE)
            {
                auto shape = updatedOperation.GetIOperands()[0]->GetShape();
                auto rawShape = updatedOperation.GetIOperands()[0]->GetRawTensor()->GetRawShape();
                EXPECT_EQ(shape.size(), rawShape.size());
                for (size_t j = 0; j < shape.size(); j++) {
                    EXPECT_EQ(shape[j], rawShape[j]);
                }
            }
        }
    }
}
