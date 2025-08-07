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
 * \file test_insert_copy_op.cpp
 * \brief Unit test for Insert Copy Op pass.
 */

#include "gtest/gtest.h"
#include "operation/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"
#include "passes/tile_graph_pass/insert_copy_op.h"
#include <vector>

using namespace npu::tile_fwk;

class InsertCopyOpTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "InsertCopyOpTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

TEST_F(InsertCopyOpTest, InsertCopy) {
    PROGRAM("InsertCopyOpTest") {
        std::vector<int> shape{128, 128};
        Program::GetInstance().GetTileShape().SetVecTileShapes({128, 128});
        Program::GetInstance().GetTileShape().SetCubeTileShapes({128, 128}, {128, 128}, {128, 128});
        Tensor inputA(DT_FP32, shape, "A");
        Tensor inputB(DT_FP32, shape, "B");
        Tensor inputC(DT_FP32, shape, "C");
        Tensor inputD(DT_FP32, shape, "D");
        Tensor output(DT_FP32, shape, "output");

        PassManager &passManager = PassManager::Instance();
        passManager.RegisterStrategy("InsertCopyOpTestStrategy", {
            {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
            {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
            {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},
            {           "GenerateMoveOp",           "GenerateMoveOp",    PassType::TYPE_TILE_GRAPH},
            {        "GraphPartitionPass",        "GraphPartitionPass",    PassType::TYPE_TILE_GRAPH},
            {          "UpdateMemoryMap",          "UpdateMemoryMap",    PassType::TYPE_TILE_GRAPH},

        });
        ConfigManager::Instance();

        std::vector<int> originOpmagic;
        FUNCTION("A_MUL_Bt") {
            config::SetPassStrategy("InsertCopyOpTestStrategy");

            auto tmpA = View(inputA, shape, {0,0});
            auto tmpB = View(inputB, shape, {0,0});

            Tensor tmpOutput(DT_FP32, shape, "tmpOutput");
            tmpOutput->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
            tmpOutput = Matrix::Matmul<false, true>(DataType::DT_FP32, tmpA, tmpB);

            auto addResult = Add(inputC, inputD);

            output = Add(addResult, tmpOutput);
        }
        std::string jsonFilePath = "./config/pass/json/insert_copy.json";
        bool dumpJsonFlag = true;
        if (dumpJsonFlag) {
            auto programJson = Program::GetInstance().DumpJson();
            DumpJsonFile(programJson, jsonFilePath);
        }
        Json readData = LoadJsonFile(jsonFilePath);
        Program::GetInstance().LoadJson(readData);

        Function* func = Program::GetInstance().GetCurrentFunction();

        Program testProgram(HostMachineMode::SERVER);
        npu::tile_fwk::InsertCopyOpPass insertCopyOpPass;
        insertCopyOpPass.PreCheck(*func);
        insertCopyOpPass.RunOnFunction(*func);
        insertCopyOpPass.PostCheck(*func);

        // ================== Verify Pass Effect ==================
        auto updatedOperations = Program::GetInstance().GetFunctionByRawName("TENSOR_A_MUL_Bt")->Operations();
        constexpr int expectOperationsNum = 10;
        EXPECT_EQ(updatedOperations.size(), expectOperationsNum) << "10 operations should remain";
        int copyInNum = 0;
        int copyOutNum = 0;
        for (size_t i = 0; i < updatedOperations.size(); i++) {
            if(updatedOperations[i].GetOpcode() == Opcode::OP_COPY_IN)
                copyInNum++;
            else if(updatedOperations[i].GetOpcode() == Opcode::OP_COPY_OUT)
                copyOutNum++;
        }
        constexpr int expectCopyInNum = 5;
        constexpr int expectCopyOutNum = 2;
        EXPECT_EQ(copyOutNum, expectCopyOutNum) << "2 operations should be OP_COPY_OUT";
        EXPECT_EQ(copyInNum, expectCopyInNum) << "5 operations should be OP_COPY_IN";
    }
}
