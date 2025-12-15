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
 * \file test_intra_subgraph_adapter.cpp
 * \brief Unit test for intra_subgraph_adapter pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "passes/tile_graph_pass/data_path/intra_subgraph_adapter.h"
#include <fstream>
#include <vector>
#include <string>
#include "computational_graph_builder.h"

namespace npu {
namespace tile_fwk{
constexpr int SUB_GRAPH_ID_NUM0 = 0;
constexpr int SUB_GRAPH_ID_NUM1 = 1;
constexpr int SUB_GRAPH_ID_NUM2 = 2;
constexpr int SUB_GRAPH_ID_NUM3 = 3;

class IntraSubGraphAdapterTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    }

    void TearDown() override {}
};

// 没有主子图，且只有一个producer
TEST_F(IntraSubGraphAdapterTest, TestNoMainSubGraph) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "IntraSubGraphAdapterTest", "IntraSubGraphAdapterTest", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {8, 16};

    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor1->SetMemoryTypeBoth(MEM_UB, true);
    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto &exp_op0 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {incast1}, {tensor1});
    auto &exp_op1 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor1}, {outcast1});
    exp_op0.UpdateSubgraphID(SUB_GRAPH_ID_NUM0);
    exp_op1.UpdateSubgraphID(SUB_GRAPH_ID_NUM1);
    auto &exp_op2 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor1}, {outcast2});
    exp_op2.UpdateSubgraphID(SUB_GRAPH_ID_NUM2);

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast1);
    currFunctionPtr->outCasts_.push_back(outcast2);

    // Call the pass
    IntraSubgraphAdapter ISA;
    size_t subGraphCount = 4;
    currFunctionPtr->SetTotalSubGraphCount(subGraphCount);
    EXPECT_EQ(ISA.RunOnFunction(*currFunctionPtr), SUCCESS);
}

// 没有主子图，且有多个producer
TEST_F(IntraSubGraphAdapterTest, TestNoMainSubGraphAndMulityProducer) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "IntraSubGraphAdapterTest", "IntraSubGraphAdapterTest", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {8, 16};

    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto incast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor1->SetMemoryTypeBoth(MEM_UB, true);
    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    auto &exp_op0 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {incast1}, {tensor1});
    exp_op0.UpdateSubgraphID(SUB_GRAPH_ID_NUM0);
    auto &exp_op1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {incast2}, {tensor1});
    exp_op1.UpdateSubgraphID(SUB_GRAPH_ID_NUM1);

    auto &exp_op2 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor1}, {outcast1});
    exp_op2.UpdateSubgraphID(SUB_GRAPH_ID_NUM2);
    auto &exp_op3 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor1}, {outcast2});
    exp_op3.UpdateSubgraphID(SUB_GRAPH_ID_NUM3);

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outcast1);
    currFunctionPtr->outCasts_.push_back(outcast2);

    // Call the pass
    IntraSubgraphAdapter ISA;
    size_t subGraphCount = 4;
    currFunctionPtr->SetTotalSubGraphCount(subGraphCount);
    EXPECT_EQ(ISA.RunOnFunction(*currFunctionPtr), SUCCESS);
}

// 有主子图，且只有一个producer
TEST_F(IntraSubGraphAdapterTest, TestHasMainSubGraph) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "IntraSubGraphAdapterTest", "IntraSubGraphAdapterTest", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {8, 16};

    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor1->SetMemoryTypeBoth(MEM_UB, true);
    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto &exp_op0 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {incast1}, {tensor1});
    auto &exp_op1 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor1}, {outcast1});
    exp_op0.UpdateSubgraphID(SUB_GRAPH_ID_NUM0);
    exp_op1.UpdateSubgraphID(SUB_GRAPH_ID_NUM0);
    auto &exp_op2 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor1}, {outcast2});
    exp_op2.UpdateSubgraphID(SUB_GRAPH_ID_NUM1);
    auto &exp_op3 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor1}, {outcast3});
    exp_op3.UpdateSubgraphID(SUB_GRAPH_ID_NUM2);

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast1);
    currFunctionPtr->outCasts_.push_back(outcast2);
    currFunctionPtr->outCasts_.push_back(outcast3);

    // Call the pass
    IntraSubgraphAdapter ISA;
    size_t subGraphCount = 4;
    currFunctionPtr->SetTotalSubGraphCount(subGraphCount);
    EXPECT_EQ(ISA.RunOnFunction(*currFunctionPtr), SUCCESS);
}

// 有主子图，且有多个producer
TEST_F(IntraSubGraphAdapterTest, TestHasMainSubGraphAndMulityProducer) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "IntraSubGraphAdapterTest", "IntraSubGraphAdapterTest", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {8, 16};

    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto incast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor1->SetMemoryTypeBoth(MEM_UB, true);
    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto &exp_op0 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {incast1}, {tensor1});
    exp_op0.UpdateSubgraphID(SUB_GRAPH_ID_NUM0);

    auto &exp_op1 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {incast2}, {tensor1});
    exp_op1.UpdateSubgraphID(SUB_GRAPH_ID_NUM1);
    auto &exp_op_o1 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor1}, {outcast1});
    exp_op_o1.UpdateSubgraphID(SUB_GRAPH_ID_NUM1);

    auto &exp_op2 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor1}, {outcast2});
    exp_op2.UpdateSubgraphID(SUB_GRAPH_ID_NUM2);
    auto &exp_op3 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor1}, {outcast3});
    exp_op3.UpdateSubgraphID(SUB_GRAPH_ID_NUM3);

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outcast1);
    currFunctionPtr->outCasts_.push_back(outcast2);
    currFunctionPtr->outCasts_.push_back(outcast3);

    // Call the pass
    IntraSubgraphAdapter ISA;
    size_t subGraphCount = 4;
    currFunctionPtr->SetTotalSubGraphCount(subGraphCount);
    EXPECT_EQ(ISA.RunOnFunction(*currFunctionPtr), SUCCESS);
}
}
}