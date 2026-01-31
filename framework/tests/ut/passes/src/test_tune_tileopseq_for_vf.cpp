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
* \file test_tune_tileop_for_vf.cpp
* \brief Unit test for TuneTileOpSeqForVF.
*/
#include <gtest/gtest.h>
#include "tilefwk/platform.h"
#include "passes/block_graph_pass/tune_tileopseq_for_vf.h"
#define private public

namespace npu {
namespace tile_fwk {

class TuneTileopseqForVFTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, HOST_COMPILE_END);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        Platform::Instance().ObtainPlatformInfo();
    }
    void TearDown() override {}
};

void BuildGraphForTest(std::shared_ptr<Function> currFunctionPtr, std::vector<Operation *> &opListPtr) {
    std::vector<int64_t> shape = {16, 16};
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor1->memoryrange.start = 0;
    tensor1->memoryrange.end = 10;
    auto tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor2->memoryrange.start = 10;
    tensor2->memoryrange.end = 20;
    auto tensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor3->memoryrange.start = 20;
    tensor3->memoryrange.end = 30;
    auto tensor4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor4->memoryrange.start = 30;
    tensor4->memoryrange.end = 40;
    auto tensor5 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor5->memoryrange.start = 40;
    tensor5->memoryrange.end = 50;
    auto tensor6 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor6->memoryrange.start = 50;
    tensor6->memoryrange.end = 60;
    auto tensor7 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor7->memoryrange.start = 60;
    tensor7->memoryrange.end = 70;
    auto tensor8 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor8->memoryrange.start = 70;
    tensor8->memoryrange.end = 80;
    auto tensor9 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor9->memoryrange.start = 80;
    tensor9->memoryrange.end = 90;
    auto tensor10 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor10->memoryrange.start = 90;
    tensor10->memoryrange.end = 100;
    auto &vecop1 = currFunctionPtr->AddRawOperation(Opcode::OP_EXP, {tensor1}, {tensor2});
    opListPtr.emplace_back(&vecop1);
    auto &vecop2 = currFunctionPtr->AddRawOperation(Opcode::OP_SQRT, {tensor3}, {tensor4});
    opListPtr.emplace_back(&vecop2);
    auto &vecop3 = currFunctionPtr->AddRawOperation(Opcode::OP_RECIPROCAL, {tensor5}, {tensor6});
    opListPtr.emplace_back(&vecop3);
    auto &op1 = currFunctionPtr->AddRawOperation(Opcode::OP_TRANSPOSE_MOVEIN, {tensor7}, {tensor8});
    opListPtr.emplace_back(&op1);
    auto &op2 = currFunctionPtr->AddRawOperation(Opcode::OP_TRANSPOSE_MOVEOUT, {tensor6}, {tensor9});
    opListPtr.emplace_back(&op2);
    auto &vecop4 = currFunctionPtr->AddRawOperation(Opcode::OP_EXPAND, {tensor8}, {tensor10});
    opListPtr.emplace_back(&vecop4);
}

TEST_F(TuneTileopseqForVFTest, TestMergeForTuneTileop) {
    // Build Graph
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestFindDep", "TestFindDep", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestFindDepLeaf", "TestFindDepLeaf", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());
    std::vector<Operation *> opListPtr;
    BuildGraphForTest(currFunctionPtr, opListPtr);
    TuneTileOpSeqForVF tuneTileop;
    PipeSync ps;
    tuneTileop.opList_ = opListPtr;
    for (auto &op : tuneTileop.opList_) {
        op->SetAIVCore(AIVCore::AIV0);
    }
    tuneTileop.ChangeOpSeq(ps, false);
    EXPECT_EQ(tuneTileop.opList_[0]->GetOpcode(), Opcode::OP_TRANSPOSE_MOVEIN);
    EXPECT_EQ(tuneTileop.opList_[5]->GetOpcode(), Opcode::OP_TRANSPOSE_MOVEOUT);
}

TEST_F(TuneTileopseqForVFTest, TestNotMergeForTuneTileop) {
    // Build Graph
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestFindDep", "TestFindDep", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestFindDepLeaf", "TestFindDepLeaf", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());
    std::vector<Operation *> opListPtr;
    BuildGraphForTest(currFunctionPtr, opListPtr);
    opListPtr[3]->GetIOperands()[0]->memoryrange.start = 50;
    opListPtr[3]->GetIOperands()[0]->memoryrange.end = 60;
    opListPtr[3]->GetOOperands()[0]->memoryrange.start = 60;
    opListPtr[3]->GetOOperands()[0]->memoryrange.end = 70;
    opListPtr[4]->GetIOperands()[0]->memoryrange.start = 60;
    opListPtr[4]->GetIOperands()[0]->memoryrange.end = 70;
    opListPtr[4]->GetOOperands()[0]->memoryrange.start = 70;
    opListPtr[4]->GetOOperands()[0]->memoryrange.end = 80;
    TuneTileOpSeqForVF tuneTileop;
    PipeSync ps;
    tuneTileop.opList_ = opListPtr;
    for (auto &op : tuneTileop.opList_) {
        op->SetAIVCore(AIVCore::AIV0);
    }
    tuneTileop.ChangeOpSeq(ps, false);
    EXPECT_EQ(tuneTileop.opList_[0]->GetOpcode(), Opcode::OP_EXP);
    EXPECT_EQ(tuneTileop.opList_[5]->GetOpcode(), Opcode::OP_EXPAND);
}

} // namespace tile_fwk
} // namespace npu

#undef private