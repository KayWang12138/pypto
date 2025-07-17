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
 * \file test_merge_compile_function.cpp
 * \brief Unit test for Merge Compile Function pass.
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
#include "passes/tile_graph_pass/merge_compile_function.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <string>
using namespace npu::tile_fwk;

class MergeCompileFunctionTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "MergeCompileFunctionTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

TEST_F(MergeCompileFunctionTest, A_MUL_Bt) {
    PROGRAM("MergeCompileFunctionTest") {
        std::vector<int> shape{128, 128};
        Program::GetInstance().GetTileShape().SetVecTileShapes({128, 128});
        Program::GetInstance().GetTileShape().SetCubeTileShapes({128, 128}, {128, 128}, {128, 128});
        Tensor inputA(DT_FP32, shape, "A");
        Tensor inputB(DT_FP32, shape, "B");
        Tensor inputC(DT_FP32, shape, "C");
        Tensor inputD(DT_FP32, shape, "D");
        Tensor output(DT_FP32, shape, "output");

        PassManager &passManager = PassManager::Instance();
        passManager.RegisterStrategy("MergeCompileFunctionTestStrategy", {
            {   "RemoveRedundentReshape",   "RemoveRedundentReshape",  PassType::TYPE_TENSOR_GRAPH},
            {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
            {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
            {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
            {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
            {          "InsertConvertOp",          "InsertConvertOp",    PassType::TYPE_TILE_GRAPH},
            {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},
            {           "GenerateMoveOp",           "GenerateMoveOp",   PassType::TYPE_TILE_GRAPH},
            {        "GroupOperationsOp",        "GroupOperationsOp",   PassType::TYPE_TILE_GRAPH},
            {            "GraphInitPass",            "GraphInitPass",   PassType::TYPE_TILE_GRAPH},

        });
        ConfigManager::Instance();

        std::vector<int> originOpmagic;
        FUNCTION("A_MUL_Bt") {
            config::SetPassStrategy("MergeCompileFunctionTestStrategy");

            auto tmpA = View(inputA, shape, {0,0});
            auto tmpB = View(inputB, shape, {0,0});

            Tensor tmpOutput(DT_FP32, shape, "tmpOutput");
            tmpOutput->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
            tmpOutput = Matrix::Matmul<false, true>(DataType::DT_FP32, tmpA, tmpB);

            auto addResult = Add(inputC, inputD);

            output = Add(addResult, tmpOutput);
        }
        std::string jsonFilePath = "./config/pass/json/merge_compile_function_a_mul_bt.json";
        bool dumpJsonFlag = true;
        if (dumpJsonFlag) {
            auto programJson = Program::GetInstance().DumpJson();
            DumpJsonFile(programJson, jsonFilePath);
        }
        Json readData = LoadJsonFile(jsonFilePath);
        Program::GetInstance().LoadJson(readData);

        Function* func = Program::GetInstance().GetCurrentFunction();

        Program testProgram(HostMachineMode::SERVER);
        npu::tile_fwk::MergeCompileFunction mergeCompileFunction;
        mergeCompileFunction.PreCheck(*func);
        mergeCompileFunction.RunOnFunction(*func);
        mergeCompileFunction.PostCheck(*func);

        // ================== Verify Pass Effect ==================
        auto updatedOperations = Program::GetInstance().GetFunctionByRawName("TENSOR_A_MUL_Bt")->Operations();
        std::vector<int> subGraphIDs;
        constexpr int leastSubGraphIDNum = 2;
        for (const auto &updatedOperation : updatedOperations) {
            auto iter = std::find(subGraphIDs.begin(), subGraphIDs.end(), updatedOperation.GetSubgraphID());
            if(iter == subGraphIDs.end()) {
                subGraphIDs.push_back(updatedOperation.GetSubgraphID());
            }
        }
        EXPECT_TRUE(subGraphIDs.size() >= leastSubGraphIDNum) << "at least 2 subGraphIDs should be found";
    }
}
