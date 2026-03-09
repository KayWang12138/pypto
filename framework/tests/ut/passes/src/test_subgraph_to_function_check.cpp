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
#include "tilefwk/tilefwk_op.h"

using namespace npu::tile_fwk;
using namespace std;

static const std::vector<int64_t> kShape88 = {8, 8};

// 公共：创建 Function + 单 op 图，执行 DoPreCheck 并断言
static void RunPreCheckTest(const std::string& funcName, Opcode opcode,
    const std::vector<std::string>& iops, const std::vector<std::string>& oops, const std::string& opName,
    int opSubGraphId, int totalSubGraphCount, Status expectedStatus) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    auto f = std::make_shared<Function>(Program::GetInstance(), funcName, funcName, nullptr);
    Program::GetInstance().InsertFuncToFunctionMap(funcName, f);
    ComputationalGraphBuilder G(f.get());
    for (const auto& t : iops) EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, kShape88, t));
    for (const auto& t : oops) EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, kShape88, t));
    EXPECT_TRUE(G.AddOp(opcode, iops, oops, opName));
    G.GetOp(opName)->UpdateSubgraphID(opSubGraphId);
    G.GetFunction()->SetTotalSubGraphCount(totalSubGraphCount);
    SubGraphToFuncChecker checker;
    EXPECT_EQ(checker.DoPreCheck(*G.GetFunction()), expectedStatus);
}

// 公共：创建两 op 链 a->b->c 或三 op 叉 a->b->c, a->b->d，设置 boundary/memory，执行 color 检查
static void RunColorOutGraphCheckTest(const std::string& funcName,
    const std::vector<std::vector<int>>& colorInGraph,
    const std::vector<std::vector<int>>& colorOutGraph,
    Status expectedPostCheckStatus,
    bool threeOpFork = false) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    auto f = std::make_shared<Function>(Program::GetInstance(), funcName, funcName, nullptr);
    Program::GetInstance().InsertFuncToFunctionMap(funcName, f);
    ComputationalGraphBuilder G(f.get());
    for (const auto& t : {"a", "b", "c"}) EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, kShape88, t));
    EXPECT_TRUE(G.AddOp(Opcode::OP_ADD, {"a"}, {"b"}, "add1"));
    EXPECT_TRUE(G.AddOp(Opcode::OP_ADD, {"b"}, {"c"}, "add2"));
    if (threeOpFork) {
        EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, kShape88, "d"));
        EXPECT_TRUE(G.AddOp(Opcode::OP_ADD, {"b"}, {"d"}, "add3"));
        G.GetOp("add3")->UpdateSubgraphID(2);
        G.GetTensor("d")->subGraphID = 2;
    }
    G.GetOp("add1")->UpdateSubgraphID(0);
    G.GetOp("add2")->UpdateSubgraphID(1);
    G.GetTensor("a")->subGraphID = G.GetTensor("b")->subGraphID = 0;
    G.GetTensor("c")->subGraphID = 1;
    G.GetTensor("b")->isSubGraphBoundary = true;
    for (const auto& t : {"a", "b", "c"}) G.GetTensor(t)->SetMemoryTypeBoth(MemoryType::MEM_UB);
    if (threeOpFork) G.GetTensor("d")->SetMemoryTypeBoth(MemoryType::MEM_UB);
    G.GetFunction()->SetTotalSubGraphCount(threeOpFork ? 3 : 2);
    G.GetFunction()->SetFunctionType(FunctionType::STATIC);
    SubGraphToFuncChecker checker;
    EXPECT_EQ(checker.DoPreCheck(*G.GetFunction()), SUCCESS);
    checker.SetColorGraph(colorInGraph, colorOutGraph);
    EXPECT_EQ(checker.DoPostCheck(*G.GetFunction()), expectedPostCheckStatus);
}

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

// 覆盖 NOPCheck: 非 NOP 的 op -> line 24
TEST_F(SubgraphToFunctionCheckTest, NOPCheck_NonNOP_Fail) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    auto f = std::make_shared<Function>(Program::GetInstance(), "NOPCheckTest", "NOPCheckTest", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("NOPCheckTest", f);
    ComputationalGraphBuilder G(f.get());
    EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, kShape88, "a"));
    EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, kShape88, "b"));
    EXPECT_TRUE(G.AddOp(Opcode::OP_ADD, {"a"}, {"b"}, "add_op"));
    SubGraphToFuncChecker checker;
    EXPECT_NE(checker.NOPCheck(G.GetFunction()->Operations()[0]), SUCCESS);
}

// 覆盖 CheckSubGraphTopo: totalSubGraphNum <= 0 -> line 59
TEST_F(SubgraphToFunctionCheckTest, CheckSubGraphTopo_InvalidTotalSubGraphNum_Fail) {
    RunPreCheckTest("TopoTest", Opcode::OP_ADD, {"x"}, {"y"}, "add_op", 0, 0, FAILED);
}

// 覆盖 CheckSubGraphTopo: subGraphId < 0 且 NOPCheck 失败 -> line 63
TEST_F(SubgraphToFunctionCheckTest, CheckSubGraphTopo_NegativeSubGraphId_NotNOP_Fail) {
    RunPreCheckTest("NegIdTest", Opcode::OP_ADD, {"a"}, {"b"}, "add_op", -1, 1, FAILED);
}

// 覆盖 NOPCheck: NOP 带 IOperands -> line 29
TEST_F(SubgraphToFunctionCheckTest, NOPCheck_NOPWithIOperands_Fail) {
    RunPreCheckTest("NOPIOpTest", Opcode::OP_NOP, {"a"}, {"b"}, "nop_op", -1, 1, FAILED);
}

// 覆盖 CheckSubGraphTopo: subGraphId >= totalSubGraphNum -> line 61-64
TEST_F(SubgraphToFunctionCheckTest, CheckSubGraphTopo_SubGraphIdOutOfRange_Fail) {
    RunPreCheckTest("OutOfRangeTest", Opcode::OP_ADD, {"a"}, {"b"}, "add_op", 1, 1, FAILED);
}

// 公共：创建两 op 链 a->b->c，可指定每 op 的 subGraphId，执行 DoPreCheck
static void RunPreCheck2OpTest(const std::string& funcName,
    int add1SgId, int add2SgId, int totalSubGraphCount, Status expectedStatus) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    auto f = std::make_shared<Function>(Program::GetInstance(), funcName, funcName, nullptr);
    Program::GetInstance().InsertFuncToFunctionMap(funcName, f);
    ComputationalGraphBuilder G(f.get());
    for (const auto& t : {"a", "b", "c"}) EXPECT_TRUE(G.AddTensor(DataType::DT_FP32, kShape88, t));
    EXPECT_TRUE(G.AddOp(Opcode::OP_ADD, {"a"}, {"b"}, "add1"));
    EXPECT_TRUE(G.AddOp(Opcode::OP_ADD, {"b"}, {"c"}, "add2"));
    G.GetOp("add1")->UpdateSubgraphID(add1SgId);
    G.GetOp("add2")->UpdateSubgraphID(add2SgId);
    G.GetFunction()->SetTotalSubGraphCount(totalSubGraphCount);
    SubGraphToFuncChecker checker;
    EXPECT_EQ(checker.DoPreCheck(*G.GetFunction()), expectedStatus);
}

// 覆盖 CheckSubGraphTopo: parentSubGraphId > subGraphId -> line 72
TEST_F(SubgraphToFunctionCheckTest, CheckSubGraphTopo_ParentSubGraphIdGreater_Fail) {
    RunPreCheck2OpTest("ParentGreaterTest", 1, 0, 2, FAILED);
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

// 覆盖 ColorOutGraphCheck: colorOutGraph_ 中缺少 outGraph_ 的边 -> line 356
TEST_F(SubgraphToFunctionCheckTest, ColorOutGraphCheck_EdgeMissedInColorOutGraph_Fail) {
    RunColorOutGraphCheckTest("ColorTest_EdgeMissed", {{}, {0}, {}}, {{1}, {}, {}}, FAILED, true);
}

// 覆盖 ColorOutGraphCheck: colorOutGraph_ 中的边在 outGraph_ 中无对应 -> line 366
TEST_F(SubgraphToFunctionCheckTest, ColorOutGraphCheck_EdgeInColorNotInOutGraph_Fail) {
    RunColorOutGraphCheckTest("ColorTest_EdgeInColor", {{}, {0}, {0}}, {{1, 2}, {}, {}}, FAILED, true);
}