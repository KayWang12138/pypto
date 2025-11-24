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
 * \file test_n_buffer_merge.cpp
 * \brief Unit test for n_buffer_merge pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "passes/tile_graph_pass/graph_partition/n_buffer_merge.h"
#include <fstream>
#include <vector>
#include <string>
#include "computational_graph_builder.h"

namespace npu {
namespace tile_fwk{

class NBufferMergeTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

        config::SetPassOption(NBUFFER_MERGE_MODE, 2);
        config::SetPassOption(VEC_NBUFFER_MAP, std::map<int64_t, int64_t>{{-1, 2}});
    }

    void TearDown() override {}

    Status TestNBufferMergeWithDifferentVecBufferMap(std::map<int64_t, int64_t> vecNBufferMap);
};

TEST_F(NBufferMergeTest, TestNBufferMerge) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestNBufferMerge", "TestNBufferMerge", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    constexpr int subGraphID0 = 0;
    constexpr int subGraphID1 = 1;
    std::vector<int64_t> shape = {8, 16};
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    incast1->subGraphID = subGraphID0;
    auto incast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    incast2->subGraphID = subGraphID1;
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor1->subGraphID = subGraphID0;
    auto tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor2->subGraphID = subGraphID0;
    auto tensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor3->subGraphID = subGraphID1;
    auto tensor4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor4->subGraphID = subGraphID1;

    auto &copy_op1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast1}, {tensor1});
    copy_op1.UpdateSubgraphID(subGraphID0);
    auto &copy_op2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast2}, {tensor3});
    copy_op2.UpdateSubgraphID(subGraphID1);
    auto &copy_out1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {tensor1}, {tensor2});
    copy_out1.UpdateSubgraphID(subGraphID0);
    auto &copy_out2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {tensor3}, {tensor4});
    copy_out2.UpdateSubgraphID(subGraphID1);

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(tensor2);
    currFunctionPtr->outCasts_.push_back(tensor4);

    // Call the pass
    NBufferMerge nPass;
    Pass &nbufferPass = nPass;
    nbufferPass.PreCheck(*currFunctionPtr);
    nbufferPass.Run(*currFunctionPtr, "", "");
    nbufferPass.PostCheck(*currFunctionPtr);
}

TEST_F(NBufferMergeTest, TestMode0) {
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int vecParallelNum = 6;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"incast0", "incast1", "outcast"}), true);
    EXPECT_EQ(G.AddOps({Opcode::OP_COPY_IN}, {{"incast0"}}, {{"incast1"}}, {"copy_in"}, true), true);
    G.GetOp("copy_in")->UpdateSubgraphID(0);
    const int subGraphNum = 20;
    for (int i = 1; i < subGraphNum; i++) {
        std::string strID = std::to_string(i);
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"tensor1" + strID, "tensor2" + strID, "tensor3" + strID}), true);
        std::vector<Opcode> opLists{Opcode::OP_ABS, Opcode::OP_EXP, Opcode::OP_ADDS, Opcode::OP_ASSEMBLE};
        std::vector<std::vector<std::string>> iOperands{{"incast1"}, {"tensor1" + strID}, {"tensor2" + strID}, {"tensor3" + strID}};
        std::vector<std::vector<std::string>> oOperands{{"tensor1" + strID}, {"tensor2" + strID}, {"tensor3" + strID}, {"outcast"}};
        std::vector<std::string> opNames{"ABS_" + strID, "EXP_" + strID, "ADDS_" + strID, "ASSEMBLE_" + strID};
        EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);
        G.GetOp("ABS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("EXP_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ADDS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ASSEMBLE_" + strID)->UpdateSubgraphID(i);
    }
    EXPECT_EQ(G.SetInCast({"incast0"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast"}), true);
    Function *function = G.GetFunction();
    function->paramConfigs_.nBufferMergeMode = 0;
    function->paramConfigs_.sgVecParallelNum = vecParallelNum;
    function->SetTotalSubGraphCount(subGraphNum);
    NBufferMerge NBM;
    EXPECT_EQ(NBM.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), subGraphNum);
}

TEST_F(NBufferMergeTest, TestMode1) {
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int vecParallelNum = 6;
    const int result = 8;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"incast0", "incast1", "outcast"}), true);
    EXPECT_EQ(G.AddOps({Opcode::OP_COPY_IN}, {{"incast0"}}, {{"incast1"}}, {"copy_in"}, true), true);
    G.GetOp("copy_in")->UpdateSubgraphID(0);
    const int subGraphNum = 20;
    for (int i = 1; i < subGraphNum; i++) {
        std::string strID = std::to_string(i);
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"tensor1" + strID, "tensor2" + strID, "tensor3" + strID}), true);
        std::vector<Opcode> opLists{Opcode::OP_ABS, Opcode::OP_EXP, Opcode::OP_ADDS, Opcode::OP_ASSEMBLE};
        std::vector<std::vector<std::string>> iOperands{{"incast1"}, {"tensor1" + strID}, {"tensor2" + strID}, {"tensor3" + strID}};
        std::vector<std::vector<std::string>> oOperands{{"tensor1" + strID}, {"tensor2" + strID}, {"tensor3" + strID}, {"outcast"}};
        std::vector<std::string> opNames{"ABS_" + strID, "EXP_" + strID, "ADDS_" + strID, "ASSEMBLE_" + strID};
        EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);
        G.GetOp("ABS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("EXP_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ADDS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ASSEMBLE_" + strID)->UpdateSubgraphID(i);
    }
    EXPECT_EQ(G.SetInCast({"incast0"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast"}), true);
    Function *function = G.GetFunction();
    function->paramConfigs_.nBufferMergeMode = 1;
    function->paramConfigs_.sgVecParallelNum = vecParallelNum;
    function->SetTotalSubGraphCount(subGraphNum);
    NBufferMerge NBM;
    EXPECT_EQ(NBM.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), result);
}

TEST_F(NBufferMergeTest, TestMode2) {
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int vecParallelNum = 6;
    const int result = 11;
    const int manualMode = 2;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"incast0", "incast1", "outcast"}), true);
    EXPECT_EQ(G.AddOps({Opcode::OP_COPY_IN}, {{"incast0"}}, {{"incast1"}}, {"copy_in"}, true), true);
    G.GetOp("copy_in")->UpdateSubgraphID(0);
    const int subGraphNum = 20;
    for (int i = 1; i < subGraphNum; i++) {
        std::string strID = std::to_string(i);
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"tensor1" + strID, "tensor2" + strID, "tensor3" + strID}), true);
        std::vector<Opcode> opLists{Opcode::OP_ABS, Opcode::OP_EXP, Opcode::OP_ADDS, Opcode::OP_ASSEMBLE};
        std::vector<std::vector<std::string>> iOperands{{"incast1"}, {"tensor1" + strID}, {"tensor2" + strID}, {"tensor3" + strID}};
        std::vector<std::vector<std::string>> oOperands{{"tensor1" + strID}, {"tensor2" + strID}, {"tensor3" + strID}, {"outcast"}};
        std::vector<std::string> opNames{"ABS_" + strID, "EXP_" + strID, "ADDS_" + strID, "ASSEMBLE_" + strID};
        EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);
        G.GetOp("ABS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("EXP_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ADDS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ASSEMBLE_" + strID)->UpdateSubgraphID(i);
    }
    EXPECT_EQ(G.SetInCast({"incast0"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast"}), true);
    Function *function = G.GetFunction();
    function->paramConfigs_.nBufferMergeMode = manualMode;
    function->paramConfigs_.vecNBufferMap = {{-1, 4}, {1, 2}};
    function->paramConfigs_.sgVecParallelNum = vecParallelNum;
    function->SetTotalSubGraphCount(subGraphNum);
    NBufferMerge NBM;
    EXPECT_EQ(NBM.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), result);
}

TEST_F(NBufferMergeTest, TestMode3) {
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int vecParallelNum = 6;
    const int noneMode = 3;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"incast0", "incast1", "outcast"}), true);
    EXPECT_EQ(G.AddOps({Opcode::OP_COPY_IN}, {{"incast0"}}, {{"incast1"}}, {"copy_in"}, true), true);
    G.GetOp("copy_in")->UpdateSubgraphID(0);
    const int subGraphNum = 20;
    for (int i = 1; i < subGraphNum; i++) {
        std::string strID = std::to_string(i);
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"tensor1" + strID, "tensor2" + strID, "tensor3" + strID}), true);
        std::vector<Opcode> opLists{Opcode::OP_ABS, Opcode::OP_EXP, Opcode::OP_ADDS, Opcode::OP_ASSEMBLE};
        std::vector<std::vector<std::string>> iOperands{{"incast1"}, {"tensor1" + strID}, {"tensor2" + strID}, {"tensor3" + strID}};
        std::vector<std::vector<std::string>> oOperands{{"tensor1" + strID}, {"tensor2" + strID}, {"tensor3" + strID}, {"outcast"}};
        std::vector<std::string> opNames{"ABS_" + strID, "EXP_" + strID, "ADDS_" + strID, "ASSEMBLE_" + strID};
        EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);
        G.GetOp("ABS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("EXP_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ADDS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ASSEMBLE_" + strID)->UpdateSubgraphID(i);
    }
    EXPECT_EQ(G.SetInCast({"incast0"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast"}), true);
    Function *function = G.GetFunction();
    function->paramConfigs_.nBufferMergeMode = noneMode;
    function->paramConfigs_.sgVecParallelNum = vecParallelNum;
    function->SetTotalSubGraphCount(subGraphNum);
    NBufferMerge NBM;
    EXPECT_EQ(NBM.RunOnFunction(*function), FAILED);
}

TEST_F(NBufferMergeTest, TestMode2NoMap) {
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int vecParallelNum = 6;
    const int manualMode = 2;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"incast0", "incast1", "outcast"}), true);
    EXPECT_EQ(G.AddOps({Opcode::OP_COPY_IN}, {{"incast0"}}, {{"incast1"}}, {"copy_in"}, true), true);
    G.GetOp("copy_in")->UpdateSubgraphID(0);
    const int subGraphNum = 20;
    for (int i = 1; i < subGraphNum; i++) {
        std::string strID = std::to_string(i);
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"tensor1" + strID, "tensor2" + strID, "tensor3" + strID}), true);
        std::vector<Opcode> opLists{Opcode::OP_ABS, Opcode::OP_EXP, Opcode::OP_ADDS, Opcode::OP_ASSEMBLE};
        std::vector<std::vector<std::string>> iOperands{{"incast1"}, {"tensor1" + strID}, {"tensor2" + strID}, {"tensor3" + strID}};
        std::vector<std::vector<std::string>> oOperands{{"tensor1" + strID}, {"tensor2" + strID}, {"tensor3" + strID}, {"outcast"}};
        std::vector<std::string> opNames{"ABS_" + strID, "EXP_" + strID, "ADDS_" + strID, "ASSEMBLE_" + strID};
        EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);
        G.GetOp("ABS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("EXP_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ADDS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ASSEMBLE_" + strID)->UpdateSubgraphID(i);
    }
    EXPECT_EQ(G.SetInCast({"incast0"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast"}), true);
    Function *function = G.GetFunction();
    function->paramConfigs_.nBufferMergeMode = manualMode;
    function->paramConfigs_.sgVecParallelNum = vecParallelNum;
    function->SetTotalSubGraphCount(subGraphNum);
    NBufferMerge NBM;
    EXPECT_EQ(NBM.RunOnFunction(*function), FAILED);
}

TEST_F(NBufferMergeTest, TestMode1Map) {
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int vecParallelNum = 6;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"incast0", "incast1", "outcast"}), true);
    EXPECT_EQ(G.AddOps({Opcode::OP_COPY_IN}, {{"incast0"}}, {{"incast1"}}, {"copy_in"}, true), true);
    G.GetOp("copy_in")->UpdateSubgraphID(0);
    const int subGraphNum = 20;
    for (int i = 1; i < subGraphNum; i++) {
        std::string strID = std::to_string(i);
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"tensor1" + strID, "tensor2" + strID, "tensor3" + strID}), true);
        std::vector<Opcode> opLists{Opcode::OP_ABS, Opcode::OP_EXP, Opcode::OP_ADDS, Opcode::OP_ASSEMBLE};
        std::vector<std::vector<std::string>> iOperands{{"incast1"}, {"tensor1" + strID}, {"tensor2" + strID}, {"tensor3" + strID}};
        std::vector<std::vector<std::string>> oOperands{{"tensor1" + strID}, {"tensor2" + strID}, {"tensor3" + strID}, {"outcast"}};
        std::vector<std::string> opNames{"ABS_" + strID, "EXP_" + strID, "ADDS_" + strID, "ASSEMBLE_" + strID};
        EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);
        G.GetOp("ABS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("EXP_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ADDS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ASSEMBLE_" + strID)->UpdateSubgraphID(i);
    }
    EXPECT_EQ(G.SetInCast({"incast0"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast"}), true);
    Function *function = G.GetFunction();
    function->paramConfigs_.nBufferMergeMode = 1;
    function->paramConfigs_.vecNBufferMap = {{-1, 2}};
    function->paramConfigs_.sgVecParallelNum = vecParallelNum;
    function->SetTotalSubGraphCount(subGraphNum);
    NBufferMerge NBM;
    EXPECT_EQ(NBM.RunOnFunction(*function), FAILED);
}

TEST_F(NBufferMergeTest, TestMode2AndVecNBufferMapKeyLessThanMinValue) {
    EXPECT_EQ(TestNBufferMergeWithDifferentVecBufferMap({{-2, 4}, {1, 2}}), FAILED);
}

TEST_F(NBufferMergeTest, TestMode2AndVecNBufferMapKeyMoreThanMaxValue) {
    EXPECT_EQ(TestNBufferMergeWithDifferentVecBufferMap({{-1, 4}, {100, 2}}), FAILED);
}

TEST_F(NBufferMergeTest, TestMode2AndVecNBufferMapValueLessThanMinValue) {
    EXPECT_EQ(TestNBufferMergeWithDifferentVecBufferMap({{-1, 4}, {1, 0}}), FAILED);
}

TEST_F(NBufferMergeTest, TestMode2AndVecNBufferMapValueMoreThanMaxValue) {
    EXPECT_EQ(TestNBufferMergeWithDifferentVecBufferMap({{-1, 4}, {1, INT64_MAX}}), FAILED);
}

Status NBufferMergeTest::TestNBufferMergeWithDifferentVecBufferMap(std::map<int64_t, int64_t> vecNBufferMap) {
    std::vector<int64_t> tileShape{16, 16};
    const int vecParallelNum = 3;
    const int manualMode = 2;
    ComputationalGraphBuilder G;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"incast0", "incast1", "outcast"}), true);
    EXPECT_EQ(G.AddOps({Opcode::OP_COPY_IN}, {{"incast0"}}, {{"incast1"}}, {"copy_in"}, true), true);
    G.GetOp("copy_in")->UpdateSubgraphID(0);
    const int subGraphNum = 10;
    for (int i = 1; i < subGraphNum; i++) {
        std::string strID = std::to_string(i);
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"tensor1" + strID, "tensor2" + strID, "tensor3" + strID}), true);
        std::vector<std::vector<std::string>> iOperands{{"incast1"}, {"tensor1" + strID}, {"tensor2" + strID}, {"tensor3" + strID}};
        std::vector<std::vector<std::string>> oOperands{{"tensor1" + strID}, {"tensor2" + strID}, {"tensor3" + strID}, {"outcast"}};
        std::vector<std::string> opNames{"ABS_" + strID, "EXP_" + strID, "ADDS_" + strID, "ASSEMBLE_" + strID};
        std::vector<Opcode> opLists{Opcode::OP_ABS, Opcode::OP_EXP, Opcode::OP_ADDS, Opcode::OP_ASSEMBLE};
        EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);
        G.GetOp("EXP_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ADDS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ASSEMBLE_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ABS_" + strID)->UpdateSubgraphID(i);
    }
    EXPECT_EQ(G.SetInCast({"incast0"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast"}), true);
    Function *function = G.GetFunction();
    function->paramConfigs_.nBufferMergeMode = manualMode;
    function->paramConfigs_.vecNBufferMap = vecNBufferMap;
    function->paramConfigs_.sgVecParallelNum = vecParallelNum;
    function->SetTotalSubGraphCount(subGraphNum);
    NBufferMerge NBM;
    return NBM.RunOnFunction(*function);
}
}
}