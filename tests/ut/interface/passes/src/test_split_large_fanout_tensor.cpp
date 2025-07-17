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
 * \file test_split_large_fanout_tensor.cpp
 * \brief Unit test for Split Large Fanout Tensor pass.
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
#include "passes/tile_graph_pass/split_large_fanout_tensor.h"
#include <vector>

using namespace npu::tile_fwk;

class SplitLargeFanoutTensorTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "SplitLargeFanoutTensorTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

TEST_F(SplitLargeFanoutTensorTest, PerfectlyMatch) {
    PROGRAM("SplitLargeFanoutTensorTest") {
        int N = 2;
        int T = 8;
        std::vector<int> shape{N * T, N * T};
        Program::GetInstance().GetTileShape().SetVecTileShapes({128, 128});

        Tensor inputA(DT_FP32, shape, "a");
        Tensor inputB(DT_FP32, shape, "b");
        Tensor result(DT_FP32, {T, T}, "result");
        PassManager &passManager = PassManager::Instance();
        passManager.RegisterStrategy("SplitLargeFanoutTensorTestStrategy", {
            {   "RemoveRedundentReshape",   "RemoveRedundentReshape",  PassType::TYPE_TENSOR_GRAPH},
            {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
            {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
            {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
            {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
            {          "InsertConvertOp",          "InsertConvertOp",    PassType::TYPE_TILE_GRAPH},
        });
        ConfigManager::Instance();

        std::vector<int> originOpmagic;
        FUNCTION("PerfectlyMatch") {
            config::SetPassStrategy("SplitLargeFanoutTensorTestStrategy");

            std::vector<std::pair<Tensor, std::vector<int>>> aggregation;
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++) {
                    auto partialResult = Sub(View(inputA, {T, T}, {i * T, j * T}), View(inputB, {T, T}, {j * T, i * T}));
                    aggregation.emplace_back(partialResult, std::vector<int>{i * T, j * T});
                }
            auto gatherResult = Assemble(aggregation); // 10
            auto inputC = View(gatherResult, {T, T}, {8, 0});
            auto inputD = View(gatherResult, {T, T}, {8, T});

            result = Add(inputC, inputD);
        }

        std::string jsonFilePath = "./config/pass/json/split_large_fanout_tensor_perfectly_match.json";
        bool dumpJsonFlag = false;
        if (dumpJsonFlag) {
            auto programJson = Program::GetInstance().DumpJson();
            DumpJsonFile(programJson, jsonFilePath);
            Json readData = LoadJsonFile(jsonFilePath);
            Program::GetInstance().LoadJson(readData);
        }

        // Call the pass
        Function* func = Program::GetInstance().GetCurrentFunction();
        npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
        splitLargeFanoutTensor.PreCheck(*func);
        splitLargeFanoutTensor.RunOnFunction(*func);
        splitLargeFanoutTensor.PostCheck(*func);

        // ================== Verify Pass Effect ==================
        auto updatedOperations = Program::GetInstance().GetFunctionByRawName("TENSOR_PerfectlyMatch")->Operations();
        constexpr int expectOperationsSize = 20;
        EXPECT_EQ(updatedOperations.size(), expectOperationsSize) << "18 operations should remain";
        int viewNum = 0;
        int assembleNum = 0;
        for (const auto &updatedOperation : updatedOperations) {
            if(updatedOperation.GetOpcode() == Opcode::OP_VIEW)
                viewNum++;
            else if(updatedOperation.GetOpcode() == Opcode::OP_ASSEMBLE)
                assembleNum++;
        }
        constexpr int expectViewNum = 10;
        constexpr int expectAssembleNum = 5;
        EXPECT_EQ(viewNum, expectViewNum) << "10 operations should be OP_VIEW";
        EXPECT_EQ(assembleNum, expectAssembleNum) << "3 operations should be OP_ASSEMBLE";
    }
}

TEST_F(SplitLargeFanoutTensorTest, BeCovered) {
    PROGRAM("SplitLargeFanoutTensorTest") {
        int N = 2;
        int T = 8;
        std::vector<int> shape{N * T, N * T};
        Program::GetInstance().GetTileShape().SetVecTileShapes({128, 128});

        Tensor inputA(DT_FP32, shape, "a");
        Tensor inputB(DT_FP32, shape, "b");
        Tensor result(DT_FP32, {N, N}, "result");
        PassManager &passManager = PassManager::Instance();
        passManager.RegisterStrategy("SplitLargeFanoutTensorTestStrategy", {
            {   "RemoveRedundentReshape",   "RemoveRedundentReshape",  PassType::TYPE_TENSOR_GRAPH},
            {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
            {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
            {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
            {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
            {          "InsertConvertOp",          "InsertConvertOp",    PassType::TYPE_TILE_GRAPH},
        });
        ConfigManager::Instance();

        std::vector<int> originOpmagic;
        FUNCTION("BeCovered") {
            config::SetPassStrategy("SplitLargeFanoutTensorTestStrategy");

            std::vector<std::pair<Tensor, std::vector<int>>> aggregation;
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++) {
                    auto partialResult = Sub(View(inputA, {T, T}, {i * T, j * T}), View(inputB, {T, T}, {j * T, i * T}));
                    aggregation.emplace_back(partialResult, std::vector<int>{i * T, j * T});
                }
            auto gatherResult = Assemble(aggregation); // 10
            auto inputC = View(gatherResult, {N, N}, {8, 0});
            auto inputD = View(gatherResult, {N, N}, {8, T});

            result = Add(inputC, inputD);
        }

        std::string jsonFilePath = "./config/pass/json/split_large_fanout_tensor_be_covered.json";
        bool dumpJsonFlag = false;
        if (dumpJsonFlag) {
            auto programJson = Program::GetInstance().DumpJson();
            DumpJsonFile(programJson, jsonFilePath);
            Json readData = LoadJsonFile(jsonFilePath);
            Program::GetInstance().LoadJson(readData);
        }

        // Call the pass
        Function* func = Program::GetInstance().GetCurrentFunction();
        npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
        splitLargeFanoutTensor.PreCheck(*func);
        splitLargeFanoutTensor.RunOnFunction(*func);
        splitLargeFanoutTensor.PostCheck(*func);

        // ================== Verify Pass Effect ==================
        auto updatedOperations = Program::GetInstance().GetFunctionByRawName("TENSOR_BeCovered")->Operations();
        constexpr int expectOperationsSize = 20;
        EXPECT_EQ(updatedOperations.size(), expectOperationsSize) << "18 operations should remain";
        int viewNum = 0;
        int assembleNum = 0;
        for (const auto &updatedOperation : updatedOperations) {
            if(updatedOperation.GetOpcode() == Opcode::OP_VIEW)
                viewNum++;
            else if(updatedOperation.GetOpcode() == Opcode::OP_ASSEMBLE)
                assembleNum++;
        }
        constexpr int expectViewNum = 10;
        constexpr int expectAssembleNum = 5;
        EXPECT_EQ(viewNum, expectViewNum) << "10 operations should be OP_VIEW";
        EXPECT_EQ(assembleNum, expectAssembleNum) << "3 operations should be OP_ASSEMBLE";
    }
}

TEST_F(SplitLargeFanoutTensorTest, PerfectlyMatchWithAll) {
    PROGRAM("SplitLargeFanoutTensorTest") {
        int N = 2;
        int T = 8;
        std::vector<int> shape{N * T, N * T};
        Program::GetInstance().GetTileShape().SetVecTileShapes({128, 128});

        Tensor inputA(DT_FP32, shape, "a");
        Tensor inputB(DT_FP32, shape, "b");
        Tensor result(DT_FP32, {T, T}, "result");
        PassManager &passManager = PassManager::Instance();
        passManager.RegisterStrategy("SplitLargeFanoutTensorTestStrategy", {
            {   "RemoveRedundentReshape",   "RemoveRedundentReshape",  PassType::TYPE_TENSOR_GRAPH},
            {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
            {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
            {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
            {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
            {          "InsertConvertOp",          "InsertConvertOp",    PassType::TYPE_TILE_GRAPH},
        });
        ConfigManager::Instance();

        std::vector<int> originOpmagic;
        FUNCTION("PerfectlyMatchWithAll") {
            config::SetPassStrategy("SplitLargeFanoutTensorTestStrategy");

            std::vector<std::pair<Tensor, std::vector<int>>> aggregation;
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++) {
                    auto partialResult = Sub(View(inputA, {T, T}, {i * T, j * T}), View(inputB, {T, T}, {j * T, i * T}));
                    aggregation.emplace_back(partialResult, std::vector<int>{i * T, j * T});
                }
            auto gatherResult = Assemble(aggregation); // 10
            auto inputC = View(gatherResult, {2*T, T}, {0, 0});
            auto inputD = View(gatherResult, {2*T, T}, {0, 8});

            result = Add(inputC, inputD);
        }

        std::string jsonFilePath = "./config/pass/json/split_large_fanout_tensor_perfectly_match_withall.json";
        bool dumpJsonFlag = false;
        if (dumpJsonFlag) {
            auto programJson = Program::GetInstance().DumpJson();
            DumpJsonFile(programJson, jsonFilePath);
            Json readData = LoadJsonFile(jsonFilePath);
            Program::GetInstance().LoadJson(readData);
        }

        // Call the pass
        Function* func = Program::GetInstance().GetCurrentFunction();
        npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
        splitLargeFanoutTensor.PreCheck(*func);
        splitLargeFanoutTensor.RunOnFunction(*func);
        splitLargeFanoutTensor.PostCheck(*func);

        // ================== Verify Pass Effect ==================
        auto updatedOperations = Program::GetInstance().GetFunctionByRawName("TENSOR_PerfectlyMatchWithAll")->Operations();
        constexpr int expectOperationsSize = 20;
        EXPECT_EQ(updatedOperations.size(), expectOperationsSize) << "20 operations should remain";
        int viewNum = 0;
        int assembleNum = 0;
        for (size_t i = 0; i < updatedOperations.size(); i++) {
            if(updatedOperations[i].GetOpcode() == Opcode::OP_VIEW)
                viewNum++;
            else if(updatedOperations[i].GetOpcode() == Opcode::OP_ASSEMBLE)
                assembleNum++;
        }
        constexpr int expectViewNum = 10;
        constexpr int expectAssembleNum = 5;
        EXPECT_EQ(viewNum, expectViewNum) << "10 operations should be OP_VIEW";
        EXPECT_EQ(assembleNum, expectAssembleNum) << "5 operations should be OP_ASSEMBLE";
    }
}

TEST_F(SplitLargeFanoutTensorTest, MultiView) {
    PROGRAM("SplitLargeFanoutTensorTest") {

        Tensor qRope(DT_FP32, {33, 64}, "a");
        Tensor qNope(DT_FP32, {33, 512}, "b");
        Tensor result(DT_FP32, {32, 576}, "result");
        PassManager &passManager = PassManager::Instance();
        passManager.RegisterStrategy("SplitLargeFanoutTensorTestStrategy", {
            {   "RemoveRedundentReshape",   "RemoveRedundentReshape",  PassType::TYPE_TENSOR_GRAPH},
            {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
            {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
            {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
            {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
            {          "InsertConvertOp",          "InsertConvertOp",    PassType::TYPE_TILE_GRAPH},
        });
        ConfigManager::Instance();

        std::vector<int> originOpmagic;
        FUNCTION("MultiView") {
            config::SetPassStrategy("SplitLargeFanoutTensorTestStrategy");
            auto qr = View(qRope, {32, 64}, {1, 0});
            auto qn = View(qNope, {32, 512}, {1, 0});
            auto qi = Assemble({
                {qn, {0,0}},
                {qr, {0,512}}

            });
        Program::GetInstance().GetTileShape().SetVecTileShapes({32, 64});
            result = AddS(qi, Element(DataType::DT_FP32, 0.0));
        }

        std::string jsonFilePath = "./config/pass/json/split_large_fanout_tensor_multi_view.json";
        bool dumpJsonFlag = false;
        if (dumpJsonFlag) {
            auto programJson = Program::GetInstance().DumpJson();
            DumpJsonFile(programJson, jsonFilePath);
            Json readData = LoadJsonFile(jsonFilePath);
            Program::GetInstance().LoadJson(readData);
        }

        // Call the pass
        Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_MultiView");
        for (auto &op : func->Operations()) {
            if (op.GetOpcode() != Opcode::OP_VIEW) {
                continue;
            }
            auto output = op.oOperand.front();
            auto viewAttr = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
            auto consumers = output->GetConsumers();
            bool allChildrenView = std::all_of(consumers.begin(), consumers.end(),
                [](const Operation *opNext) { return opNext->GetOpcode() == Opcode::OP_VIEW; });
            if (allChildrenView) {
                for (auto &consumer : consumers) {
                    ASSERT(consumer->GetOpcode() == Opcode::OP_VIEW);
                    auto newOffset = viewAttr->GetFromOffset();
                    auto nextViewAttr = dynamic_cast<ViewOpAttribute *>(consumer->GetOpAttribute().get());
                    auto nextViewOffset = nextViewAttr->GetFromOffset();
                    auto& newDynOffset = viewAttr->GetFromDynOffset();
                    for (size_t i = 0; i < newOffset.size(); ++i) {
                        newOffset[i] = newOffset[i] + nextViewOffset[i];
                    }
                    if (newDynOffset.empty()) {
                        for (size_t i = 0; i < newOffset.size(); ++i) {
                            newDynOffset.push_back(SymbolicScalar(0));
                        }
                    }
                }
            }
        }
        npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
        splitLargeFanoutTensor.PreCheck(*func);
        splitLargeFanoutTensor.RunOnFunction(*func);
        splitLargeFanoutTensor.PostCheck(*func);

        auto loop_set = func->LoopCheck();
        std::cout << "loop_set.size() = " << loop_set.size() << std::endl;
        for (auto i : loop_set) {
            std::cout << "in loop_set, loop_set[i] number = " << i << std::endl;
        }
        auto updatedOperations = func->Operations();
        std::cout << "updatedOperations.size() = " << updatedOperations.size() << std::endl;
        for (auto &updated_operation : updatedOperations) {
            std::cout << "updatedOperations[i].GetOpcode() = " << updated_operation.GetOpcodeStr() << std::endl;
            std::cout << "updatedOperations[i].opmagic = " << updated_operation.opmagic << std::endl;
            std::cout << "updatedOperations[i].GetSubgraphID() = " << updated_operation.GetSubgraphID() << std::endl;
            if (updated_operation.GetSubgraphID() != 0) {
            }
        }
        EXPECT_EQ(loop_set.size(), 0) << "SubGraphInCycle number is 0";
        constexpr int expectOperationsSize = 27;
        EXPECT_EQ(updatedOperations.size(), expectOperationsSize) << "20 operations should remain";
        int viewNum = 0;
        int assembleNum = 0;
        for (size_t i = 0; i < updatedOperations.size(); i++) {
            if(updatedOperations[i].GetOpcode() == Opcode::OP_VIEW)
                viewNum++;
            else if(updatedOperations[i].GetOpcode() == Opcode::OP_ASSEMBLE)
                assembleNum++;
        }
        constexpr int expectViewNum = 9;
        constexpr int expectAssembleNum = 9;
        EXPECT_EQ(viewNum, expectViewNum) << "9 operations should be OP_VIEW";
        EXPECT_EQ(assembleNum, expectAssembleNum) << "9 operations should be OP_ASSEMBLE";
    }
}
