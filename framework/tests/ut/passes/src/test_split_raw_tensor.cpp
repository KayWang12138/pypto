/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
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
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"
#include "passes/tile_graph_pass/graph_optimization/split_raw.h"
#include "computational_graph_builder.h"
#include <vector>

using namespace npu::tile_fwk;

class SplitLargeLocalRawTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "SplitLargeLocalRawTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

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
    npu::tile_fwk::SplitRawTensor splitLargeLocalRawPass;
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

void AddTensor(ComputationalGraphBuilder &G, const std::string name, const Shape &shape,
    const Shape &rawShape, const Offset &offset, const MemoryType memType) {
    std::string nameRaw = name + "_raw";
    G.AddTensor(DataType::DT_FP32, rawShape, nameRaw);
    auto tensorRaw = G.GetTensor(nameRaw);
    tensorRaw->SetMemoryTypeBoth(memType, true);
    G.AddTensor(DataType::DT_FP32, shape, name);
    auto tensorLogic = G.GetTensor(name);
    tensorLogic->SetMemoryTypeBoth(memType, true);
    tensorLogic->tensor = tensorRaw->tensor;
    tensorLogic->UpdateOffset(offset);
}

// input -> view1 -> a -> view2 -> b -> assemble1 -> output
TEST_F(SplitLargeLocalRawTest, TestSplitRawTensorCheceker) {
    int NUM_64 = 64;
    int NUM_128 = 128;
    int NUM_256 = 256;
    int NUM_512 = 512;
    std::vector<int64_t> shape1{NUM_64, NUM_64};
    std::vector<int64_t> shape2{NUM_128, NUM_128};
    std::vector<int64_t> shape3{NUM_256, NUM_256};
    std::vector<int64_t> shape4{NUM_512, NUM_512};
    ComputationalGraphBuilder G;

    AddTensor(G, "input", shape3, shape4, shape3, MemoryType::MEM_DEVICE_DDR);
    AddTensor(G, "a", shape2, shape3, shape2, MemoryType::MEM_DEVICE_DDR);
    AddTensor(G, "b", shape1, shape2, shape1, MemoryType::MEM_UB);
    AddTensor(G, "output", shape3, shape4, shape3, MemoryType::MEM_DEVICE_DDR);

    G.AddOp(Opcode::OP_VIEW, {"input"}, {"a"}, "view1");
    G.GetOp("view1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(shape3, MemoryType::MEM_DEVICE_DDR));
    G.AddOp(Opcode::OP_VIEW, {"a"}, {"b"}, "view2");
    G.GetOp("view2")->SetOpAttribute(std::make_shared<ViewOpAttribute>(shape2, MemoryType::MEM_UB));
    G.AddOp(Opcode::OP_ASSEMBLE, {"b"}, {"output"}, "assemble1");
    G.GetOp("assemble1")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, shape2));
    
    G.SetInCast({"input"});
    G.SetOutCast({"output"});

    Function *function = G.GetFunction();
    npu::tile_fwk::SplitRawTensor splitLargeLocalRawPass;
    EXPECT_EQ(splitLargeLocalRawPass.PreCheck(*function), SUCCESS);
    EXPECT_EQ(splitLargeLocalRawPass.PostCheck(*function), FAILED);
    EXPECT_EQ(splitLargeLocalRawPass.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(splitLargeLocalRawPass.PostCheck(*function), SUCCESS);
}