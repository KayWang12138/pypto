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
 * \file test_subgraph_to_function_check.cpp
 * \brief Unit test for SubgraphToFunction preCheck and postCheck.
 */

#include "gtest/gtest.h"
#include <algorithm>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "tilefwk/data_type.h"
#include "interface/inner/tile_shape.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/configs/config_manager.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/program/program.h"
#define private public
#include "passes/pass_check/subgraph_to_function_checker.h"
#undef private
#include "computational_graph_builder.h"

using namespace npu::tile_fwk;
using namespace std;

class SubgraphToFunctionCheckTest : public testing::Test {
public:
    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
    }

    void TearDown() override {}
};

TEST_F(SubgraphToFunctionCheckTest, TestPrePostCheck) {
    constexpr int kTileSize = 32;
    constexpr int kVectorSize = 64;
    config::SetPassConfig("PVC2_OOO", "SubgraphToFunction", KEY_PRE_CHECK, true);
    config::SetPassConfig("PVC2_OOO", "SubgraphToFunction", KEY_POST_CHECK, true);
    TileShape::Current().SetVecTile(kTileSize, kTileSize);
    TileShape::Current().SetCubeTile({kTileSize, kTileSize}, {kTileSize, kTileSize}, {kTileSize, kTileSize});

    std::vector<int64_t> shape = {kVectorSize, kVectorSize};
    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, shape, "b");
    Tensor c1(DT_FP32, shape, "c1");
    Tensor c2(DT_FP32, shape, "c2");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(a, 1.0f),
        RawTensorData::CreateConstantTensor<float>(b, 2.0f),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(c1, 0.0f),
        RawTensorData::CreateConstantTensor<float>(c2, 0.0f),
    });

    FUNCTION("SimpleTest", {a, b}, {c1, c2}) {
        Tensor temp1 = Add(a, b);
        temp1 = Mul(temp1, a);
        c1 = Sub(temp1, b);

        Tensor temp2 = Add(a, b);
        temp2 = Mul(temp2, a);
        c2 = Sub(temp2, b);
    }
    auto mainFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_SimpleTest_2");
    EXPECT_NE(mainFunc, nullptr);
    ALOG_INFO_F("Pre/Post check test completed");
}

// 覆盖 NOPCheck: 非 NOP 的 op -> line 24 (subgraph_to_function_checker.cpp 29 对应 IOperands，此处覆盖 24)
TEST_F(SubgraphToFunctionCheckTest, NOPCheck_NonNOP_Fail) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    auto f = std::make_shared<Function>(Program::GetInstance(), "NOPCheckTest", "NOPCheckTest", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("NOPCheckTest", f);
    ComputationalGraphBuilder G(f.get());
    std::vector<int64_t> shape = {8, 8};
    EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, shape, "a"));
    EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, shape, "b"));
    EXPECT_TRUE(G.AddOp(Opcode::OP_ADD, {"a"}, {"b"}, "add_op"));
    SubGraphToFuncChecker checker;
    const Operation& op = G.GetFunction()->Operations()[0];
    EXPECT_NE(checker.NOPCheck(op), SUCCESS);
}

// 覆盖 CheckSubGraphTopo: totalSubGraphNum <= 0 且 operations.size() > 0 -> line 59
TEST_F(SubgraphToFunctionCheckTest, CheckSubGraphTopo_InvalidTotalSubGraphNum_Fail) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    auto f = std::make_shared<Function>(Program::GetInstance(), "TopoTest", "TopoTest", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("TopoTest", f);
    ComputationalGraphBuilder G(f.get());
    std::vector<int64_t> shape = {8, 8};
    EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, shape, "x"));
    EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, shape, "y"));
    EXPECT_TRUE(G.AddOp(Opcode::OP_ADD, {"x"}, {"y"}, "add_op"));
    G.GetOp("add_op")->UpdateSubgraphID(0);
    G.GetFunction()->SetTotalSubGraphCount(0);
    SubGraphToFuncChecker checker;
    EXPECT_NE(checker.DoPreCheck(*G.GetFunction()), SUCCESS);
}

// 覆盖 CheckSubGraphTopo: subGraphId < 0 且 NOPCheck 失败 -> line 63；NOP 带 IOperands -> line 29
TEST_F(SubgraphToFunctionCheckTest, CheckSubGraphTopo_NegativeSubGraphId_NotNOP_Fail) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    auto f = std::make_shared<Function>(Program::GetInstance(), "NegIdTest", "NegIdTest", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("NegIdTest", f);
    ComputationalGraphBuilder G(f.get());
    std::vector<int64_t> shape = {8, 8};
    EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, shape, "a"));
    EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, shape, "b"));
    EXPECT_TRUE(G.AddOp(Opcode::OP_ADD, {"a"}, {"b"}, "add_op"));
    G.GetOp("add_op")->UpdateSubgraphID(-1);
    G.GetFunction()->SetTotalSubGraphCount(1);
    SubGraphToFuncChecker checker;
    EXPECT_NE(checker.DoPreCheck(*G.GetFunction()), SUCCESS);
}

// 覆盖 NOPCheck: NOP 带 IOperands -> line 29（subGraphId=-1 时触发 NOPCheck）
TEST_F(SubgraphToFunctionCheckTest, NOPCheck_NOPWithIOperands_Fail) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    auto f = std::make_shared<Function>(Program::GetInstance(), "NOPIOpTest", "NOPIOpTest", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("NOPIOpTest", f);
    ComputationalGraphBuilder G(f.get());
    std::vector<int64_t> shape = {8, 8};
    EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, shape, "a"));
    EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, shape, "b"));
    EXPECT_TRUE(G.AddOp(Opcode::OP_NOP, {"a"}, {"b"}, "nop_op"));
    G.GetOp("nop_op")->UpdateSubgraphID(-1);
    G.GetFunction()->SetTotalSubGraphCount(1);
    SubGraphToFuncChecker checker;
    EXPECT_NE(checker.DoPreCheck(*G.GetFunction()), SUCCESS);
}

// 覆盖 CheckSubGraphTopo: subGraphId >= totalSubGraphNum -> line 72
TEST_F(SubgraphToFunctionCheckTest, CheckSubGraphTopo_SubGraphIdOutOfRange_Fail) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    auto f = std::make_shared<Function>(Program::GetInstance(), "OutOfRangeTest", "OutOfRangeTest", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("OutOfRangeTest", f);
    ComputationalGraphBuilder G(f.get());
    std::vector<int64_t> shape = {8, 8};
    EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, shape, "a"));
    EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, shape, "b"));
    EXPECT_TRUE(G.AddOp(Opcode::OP_ADD, {"a"}, {"b"}, "add_op"));
    G.GetOp("add_op")->UpdateSubgraphID(1);
    G.GetFunction()->SetTotalSubGraphCount(1);
    SubGraphToFuncChecker checker;
    EXPECT_NE(checker.DoPreCheck(*G.GetFunction()), SUCCESS);
}

// 覆盖 InAndOutGraphConsistencyCheck: parentSeqNo 导致 nodeColIdx 越界 -> line 188
TEST_F(SubgraphToFunctionCheckTest, InAndOutGraphConsistencyCheck_ParentSeqNoExceeds_Fail) {
    SubGraphToFuncChecker checker;
    std::vector<std::vector<size_t>> inGraph = {{0, 0}};
    std::vector<std::vector<size_t>> outGraph = {{0}};
    EXPECT_NE(checker.InAndOutGraphConsistencyCheck(inGraph, outGraph), SUCCESS);
}

// 覆盖 InAndOutGraphConsistencyCheck: outEdgeGraph[parentSeqNo][nodeColIdx] != i -> line 194
TEST_F(SubgraphToFunctionCheckTest, InAndOutGraphConsistencyCheck_NodeMismatch_Fail) {
    SubGraphToFuncChecker checker;
    std::vector<std::vector<size_t>> inGraph = {{0}};
    std::vector<std::vector<size_t>> outGraph = {{1}};
    EXPECT_NE(checker.InAndOutGraphConsistencyCheck(inGraph, outGraph), SUCCESS);
}

// 覆盖 ColorOutGraphCheck: colorOutGraph_ 中缺少 outGraph_ 的边 -> line 356 (通过 VerifyRedundantEdge 失败)
TEST_F(SubgraphToFunctionCheckTest, ColorOutGraphCheck_EdgeMissedInColorOutGraph_Fail) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    auto f = std::make_shared<Function>(Program::GetInstance(), "ColorTest", "ColorTest", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("ColorTest", f);
    ComputationalGraphBuilder G(f.get());
    std::vector<int64_t> shape = {8, 8};
    EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, shape, "a"));
    EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, shape, "b"));
    EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, shape, "c"));
    EXPECT_TRUE(G.AddOp(Opcode::OP_ADD, {"a"}, {"b"}, "add1"));
    EXPECT_TRUE(G.AddOp(Opcode::OP_ADD, {"b"}, {"c"}, "add2"));
    G.GetOp("add1")->UpdateSubgraphID(0);
    G.GetOp("add2")->UpdateSubgraphID(1);
    G.GetFunction()->SetTotalSubGraphCount(2);
    G.GetFunction()->SetFunctionType(FunctionType::STATIC);
    SubGraphToFuncChecker checker;
    EXPECT_EQ(checker.DoPreCheck(*G.GetFunction()), SUCCESS);
    std::vector<std::vector<int>> colorInGraph = {{}, {0}};
    std::vector<std::vector<int>> colorOutGraph = {{}, {}};
    checker.SetColorGraph(colorInGraph, colorOutGraph);
    EXPECT_NE(checker.DoPostCheck(*G.GetFunction()), SUCCESS);
}

// 覆盖 ColorOutGraphCheck: colorOutGraph_ 中的边在 outGraph_ 中无对应 -> line 366
TEST_F(SubgraphToFunctionCheckTest, ColorOutGraphCheck_EdgeInColorNotInOutGraph_Fail) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    auto f = std::make_shared<Function>(Program::GetInstance(), "ColorTest2", "ColorTest2", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("ColorTest2", f);
    ComputationalGraphBuilder G(f.get());
    std::vector<int64_t> shape = {8, 8};
    EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, shape, "a"));
    EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, shape, "b"));
    EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, shape, "c"));
    EXPECT_TRUE(G.AddOp(Opcode::OP_ADD, {"a"}, {"b"}, "add1"));
    EXPECT_TRUE(G.AddOp(Opcode::OP_ADD, {"b"}, {"c"}, "add2"));
    G.GetOp("add1")->UpdateSubgraphID(0);
    G.GetOp("add2")->UpdateSubgraphID(1);
    G.GetFunction()->SetTotalSubGraphCount(2);
    G.GetFunction()->SetFunctionType(FunctionType::STATIC);
    SubGraphToFuncChecker checker;
    EXPECT_EQ(checker.DoPreCheck(*G.GetFunction()), SUCCESS);
    std::vector<std::vector<int>> colorInGraph = {{}, {0}, {0}};
    std::vector<std::vector<int>> colorOutGraph = {{1, 2}, {}, {}};
    checker.SetColorGraph(colorInGraph, colorOutGraph);
    EXPECT_NE(checker.DoPostCheck(*G.GetFunction()), SUCCESS);
}