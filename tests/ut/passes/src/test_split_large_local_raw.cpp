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
#include "tilefwk/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"
#include "passes/tile_graph_pass/split_large_local_raw.h"
#include "computational_graph_builder.h"
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
            {   "RemoveRedundantReshape",   "RemoveRedundantReshape",  PassType::TYPE_TENSOR_GRAPH},
            {      "InferMemoryConflict",      "InferMemoryConflict",  PassType::TYPE_TENSOR_GRAPH},
            {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
            {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
            {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
            {             "SplitReshape",             "SplitReshape",    PassType::TYPE_TILE_GRAPH},
            {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},
            {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
            {        "RemoveRedundantOp",        "RemoveRedundantOp",    PassType::TYPE_TILE_GRAPH},
            {        "GraphPartition",        "GraphPartition",    PassType::TYPE_TILE_GRAPH},

        });
        ConfigManager::Instance();

        int b = 1;
        int n = 2;
        int s = 128;
        int d = 512;
        int v_head =128;
        int tileSize0 = 128;
        int tileSize1 = 32;
        std::vector<int64_t> inShape = {b * s, n, d};
        Tensor attnPostIn(DT_FP32, inShape, "attnPostIn");
        Tensor kvBProjWV(DT_FP32, {n, d, v_head}, "kvBProjWV");
        Tensor atten_output;
        FUNCTION("SplitLocalRaw") {
            config::SetPassStrategy("SplitLargeLocalRawTestStrategy");

            DataType dType = attnPostIn->Datatype();
            TileShape::Current().SetVecTile({32, 1, d});
            Tensor atten_res2 = Transpose(attnPostIn, {0, 1});
            TileShape::Current().SetVecTile(tileSize0, tileSize0);
            TileShape::Current().SetCubeTile({tileSize1, tileSize1}, {tileSize0, tileSize0}, {tileSize0, tileSize0});
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
        npu::tile_fwk::SplitLargeLocalRawTensor splitLargeLocalRawPass;
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

TEST_F(SplitLargeLocalRawTest, GraphBoundaryOnUb) {
    int NUM_64 = 64;
    int NUM_128 = 128;
    int SUBGRAPH_ID_1 = 1;
    int SUBGRAPH_ID_2 = 2;
    std::vector<int64_t> shape0{NUM_64, NUM_64};
    std::vector<int64_t> shape1{NUM_64, NUM_128};
    ComputationalGraphBuilder G;

    G.AddTensor(DataType::DT_FP32, shape0, "a"); // [64, 64]
    auto a = G.GetTensor("a");
    a->UpdateSubgraphID(SUBGRAPH_ID_1);
    G.AddTensor(DataType::DT_FP32, shape0, "b"); // [64, 64]
    auto b = G.GetTensor("b");
    b->UpdateSubgraphID(SUBGRAPH_ID_1);

    G.AddTensor(DataType::DT_FP32, shape1, "oriTensor"); // [64, 128]
    auto oriTensor = G.GetTensor("oriTensor");
    G.AddTensor(DataType::DT_FP32, shape1, "assembledTensor"); // [64, 128]
    auto assembledTensor = G.GetTensor("assembledTensor");
    assembledTensor->UpdateSubgraphID(SUBGRAPH_ID_2);
    assembledTensor->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

    // a --> View --> tiledA --> Assemble --> assembledTensor
    G.AddTensor(DataType::DT_FP32, shape0, "tiledA"); // [64, 64]
    auto tiledA = G.GetTensor("tiledA");
    tiledA->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    tiledA->UpdateSubgraphID(SUBGRAPH_ID_1);
    tiledA->tensor = oriTensor->tensor;
    tiledA->UpdateOffset({0, 0});
    G.AddOp(Opcode::OP_VIEW, {"a"}, {"tiledA"}, "View_A");
    auto viewA = G.GetOp("View_A");
    viewA->UpdateSubgraphID(SUBGRAPH_ID_1);
    G.AddOp(Opcode::OP_ASSEMBLE, {"tiledA"}, {"assembledTensor"}, "Assemble_A");
    auto attrAssembleA = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int64_t> {0, 0});
    auto assembleA = G.GetOp("Assemble_A");
    assembleA->SetOpAttribute(attrAssembleA);
    assembleA->UpdateSubgraphID(SUBGRAPH_ID_1);

    // b --> View --> tiledB --> Assemble --> assembledTensor
    G.AddTensor(DataType::DT_FP32, shape0, "tiledB"); // [64, 64]
    auto tiledB = G.GetTensor("tiledB");
    tiledB->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    tiledB->UpdateSubgraphID(SUBGRAPH_ID_2);
    tiledB->tensor = oriTensor->tensor;
    tiledB->UpdateOffset({0, NUM_64});
    G.AddOp(Opcode::OP_VIEW, {"b"}, {"tiledB"}, "View_B");
    auto viewB = G.GetOp("View_B");
    viewB->UpdateSubgraphID(SUBGRAPH_ID_1);
    G.AddOp(Opcode::OP_ASSEMBLE, {"tiledB"}, {"assembledTensor"}, "Assemble_B");
    auto attrAssembleB = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int64_t> {0, NUM_64});
    auto assembleB = G.GetOp("Assemble_B");
    assembleB->SetOpAttribute(attrAssembleB);
    assembleB->UpdateSubgraphID(SUBGRAPH_ID_2);

    G.AddTensor(DataType::DT_FP32, shape1, "exp_out");
    G.AddOp(Opcode::OP_EXP, {"assembledTensor"}, {"exp_out"}, "Exp");
    auto expOut = G.GetTensor("exp_out");
    expOut->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    expOut->UpdateSubgraphID(SUBGRAPH_ID_2);
    auto expOp = G.GetOp("Exp");
    expOp->UpdateSubgraphID(SUBGRAPH_ID_2);

    Function *function = G.GetFunction();
    // 确认构图完毕
    /*
    dump graph before Pass
    function->DumpJsonFile(jsonFilePath);
    */

    // Call the pass
    npu::tile_fwk::SplitLargeLocalRawTensor splitLargeLocalRawPass;
    splitLargeLocalRawPass.PreCheck(*function);
    splitLargeLocalRawPass.RunOnFunction(*function);
    splitLargeLocalRawPass.PostCheck(*function);

    // 校验正确性
    for (auto &op : function->Operations()) {
        std::cout << op.GetOpcodeStr() << " " << op.GetOpMagic() << std::endl;
        for (auto &out : op.GetOOperands()) {
            auto shape = out->GetShape();
            auto rawShape = out->GetRawTensor()->GetRawShape();
            EXPECT_EQ(shape.size(), rawShape.size());
            for (size_t j = 0; j < shape.size(); j++) {
                EXPECT_EQ(shape[j], rawShape[j]);
            }
        }
    }
}