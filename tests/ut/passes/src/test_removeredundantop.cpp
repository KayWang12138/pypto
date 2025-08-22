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
 * \file test_expand_function.cpp
 * \brief Unit test for ExpandFunction pass.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "ut_json/ut_json_tool.h"
#include "passes/pass_manager.h"
#include "interface/configs/config_manager.h"

#define private public
#include "passes/tile_graph_pass/remove_redundant_op.h"

namespace npu {
namespace tile_fwk{
static const size_t kSizeZero = 0UL;
static const size_t kSizeOne = 1UL;
static const size_t kSizeSeven = 7UL;
static const size_t kSizeEight = 8UL;
static const size_t kSizeTen = 10UL;
static const size_t kSizeEleven = 11UL;
static const size_t kSizeThirteen = 13UL;
static const size_t kSizeForteen = 14UL;
static const uint16_t kNumZero = 0u;
static const uint16_t kNumOne = 1u;
static const uint16_t kNumTwo = 2u;
static const uint16_t kNumThree = 3u;
static const uint16_t kNumFour = 4u;
static const uint16_t kNumFive = 5u;
static const uint16_t kNumEight = 8u;
static const uint16_t kNumExpFour = 16u;
static const uint16_t kNumExpFive = 32u;
static const uint16_t kNumExpSix = 64u;
static const uint16_t kNumExpSeven = 128u;
static const uint16_t kNumExpEight = 256u;

class TestRemoveRedundantOpPass : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "ExpandFunctionTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

/*
TESTRemoveDummyExpand
inCast{8,16}->expand->ubTensor{8,16}->exp->outCast1{8,16}
                                    ->sqrt->outCast2{8,16}
                                    ->reciprocal->outCast3{8,16}
inCast{8,16}->exp->outCast1
            ->sqrt->outCast2
            ->reciprocal->outCast3
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest1) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp", "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int> shape = {kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    
    currFunctionPtr->AddOperation(Opcode::OP_EXPAND, {inCast}, {ubTensor});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor}, {outCast1});
    currFunctionPtr->AddOperation(Opcode::OP_SQRT, {ubTensor}, {outCast2});
    currFunctionPtr->AddOperation(Opcode::OP_RECIPROCAL, {ubTensor}, {outCast3});
    
    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->outCasts_.push_back(outCast3);

    RemoveRedundantOp removeredundantpass;
    EXPECT_NE(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t expand_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_EXPAND) {
            ++expand_num;
        } else if (op.GetOpcode() == Opcode::OP_SQRT) {
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(op.GetInputOperand(kSizeZero), inCast);
        } else if (op.GetOpcode() == Opcode::OP_EXP) {
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(op.GetInputOperand(kSizeZero), inCast);
        } else if (op.GetOpcode() == Opcode::OP_RECIPROCAL) {
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(op.GetInputOperand(kSizeZero), inCast);
        }
    }
    EXPECT_EQ(expand_num, kNumZero);
}

/*
TESTRemoveDummyRegCopy
inCast{8,16}->regcopy->ubTensor1{16,8}->regcopy->ubTensor2{16,8}->exp->outCast1{16,8}
inCast{8,16}->regcopy->ubTensor1{16,8}->exp->outCast1{16,8}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest2) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp", "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int> shape1 = {kNumEight, kNumExpFour};
    std::vector<int> shape2 = {kNumExpFour, kNumEight};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    
    auto &regcopy = currFunctionPtr->AddOperation(Opcode::OP_REGISTER_COPY, {inCast}, {ubTensor1});
    currFunctionPtr->AddOperation(Opcode::OP_REGISTER_COPY, {ubTensor1}, {ubTensor2});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor2}, {outCast});
    
    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp removeredundantpass;
    EXPECT_NE(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t regcopy_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_REGISTER_COPY) {
            EXPECT_EQ(op.GetOpMagic(), regcopy.GetOpMagic());
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(op.GetInputOperand(kSizeZero), inCast);
            ++regcopy_num;
        } else if (op.GetOpcode() == Opcode::OP_EXP) {
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(op.GetInputOperand(kSizeZero), ubTensor1);
        }
    }
    EXPECT_EQ(regcopy_num, kNumOne);
}

/*
TESTRemoveDummyAssembleDDR
inCast{8,16}->exp->ubTensor1{8,16}  ->sqrt->ubTensor2{8,16}->assmble->ubTensor3{8,16}->exp->outCast2{8,16}
                                    ->assemble->outCast1{8,16}

inCast{8,16}->exp->outCast1{8,16}->sqrt->ubTensor2{8,16}->exp->outCast2{8,16}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest3) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp", "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int> shape = {kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor3->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto outCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape, "outCast1", NodeType::OUTCAST);
    outCast1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    
    auto &exp1 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {inCast}, {ubTensor1});
    currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {ubTensor1}, {outCast1});
    auto &sqrt = currFunctionPtr->AddOperation(Opcode::OP_SQRT, {ubTensor1}, {ubTensor2});
    currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {ubTensor2}, {ubTensor3});
    auto &exp2 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor3}, {outCast3});
    
    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);

    RemoveRedundantOp removeredundantpass;
    EXPECT_NE(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t assemble_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assemble_num;
        }
    }
    EXPECT_EQ(assemble_num, kNumZero);
    EXPECT_EQ(exp1.GetOutputOperandSize(), kSizeOne);
    EXPECT_EQ(exp1.GetOutputOperand(kSizeZero), outCast1);
    EXPECT_EQ(sqrt.GetInputOperandSize(), kSizeOne);
    EXPECT_EQ(sqrt.GetInputOperand(kSizeZero), outCast1);
    EXPECT_EQ(exp2.GetInputOperandSize(), kSizeOne);
    EXPECT_EQ(exp2.GetInputOperand(kSizeZero), ubTensor2);
}

/*
TESTRemoveDummyAssembleDDRSpecialCase(WARNING CASE)
inCast{8,16}->exp(any legal op)->ddrTensor1{8,16}  ->exp->outCast3{8,16}
                                    ->assemble->outCast1{8,16}
                                    ->assemble->outCast2{8,16}

inCast{8,16}->exp->outCast1/outCast2{8,16}->exp->outCast3{8,16}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest4) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp", "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int> shape = {kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto outCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape, "outCast1", NodeType::OUTCAST);
    outCast1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape, "outCast2", NodeType::OUTCAST);
    outCast2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto outCast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    
    auto &exp1 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {inCast}, {ubTensor});
    currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {ubTensor}, {outCast1});
    currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {ubTensor}, {outCast2});
    auto &exp2 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor}, {outCast3});
    
    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->outCasts_.push_back(outCast3);

    RemoveRedundantOp removeredundantpass;
    EXPECT_NE(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);
    EXPECT_NE(removeredundantpass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t assemble_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assemble_num;
        }
    }
    EXPECT_EQ(assemble_num, kNumZero);
    EXPECT_EQ(exp1.GetOutputOperandSize(), kSizeOne);
    EXPECT_EQ(exp2.GetInputOperandSize(), kSizeOne);
}

/*
TESTRemoveDummyAssembleUB
inCast{8,16}->assemble->ubTensor1{8,16}->exp->outCast1{8,16}
            ->assemble->ubTensor2{4,16}->exp->outCast2{4,16}
inCast{8,16}->exp->outCast1{8,16}
            ->assemble->ubTensor2{4,16}->exp->outCast2{4,16}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest5) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp", "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int> shape1 = {kNumEight, kNumExpFour};
    std::vector<int> shape2 = {kNumFour, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    inCast->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    auto outCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    
    currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {inCast}, {ubTensor1});
    currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {inCast}, {ubTensor2});
    auto &exp1 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor1}, {outCast1});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor2}, {outCast2});
    
    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);

    RemoveRedundantOp removeredundantpass;
    EXPECT_NE(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    
    uint32_t assemble_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assemble_num;
        }
    }
    EXPECT_EQ(assemble_num, kNumOne);
    EXPECT_EQ(exp1.GetInputOperandSize(), kSizeOne);
    EXPECT_EQ(exp1.GetInputOperand(kSizeZero), inCast);
}

/*
TESTRemoveDummyView(WARNING CASE)
inCast{8,16}->exp->ddrTensor1{8,16}->exp->ubTensor2{8,16}->view->ubTensor3{8,16}->exp->outCast2{8,16}
                                  ->view->outCast1{8,16}                       ->reciprocal->outCast3{8,16}
                                                                               ->sqrt->outCast4{8,16}
            
inCast{8,16}->exp->outCast1{8,16}
ddrTensor1{8,16} ->exp->ddrTensor2{8,16}->exp->outCast2{8,16}
                ->reciprocal->outCast3{8,16}
                ->sqrt->outCast4{8,16}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest6) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp", "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int> shape1 = {kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto outCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto outCast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto outCast4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto &exp1 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {inCast}, {ubTensor1});
    currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor1}, {outCast1});
    auto &exp2 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor1}, {ubTensor2});
    currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {ubTensor3});
    auto &exp3 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor3}, {outCast2});
    auto &reci = currFunctionPtr->AddOperation(Opcode::OP_RECIPROCAL, {ubTensor3}, {outCast3});
    auto &sqrt = currFunctionPtr->AddOperation(Opcode::OP_SQRT, {ubTensor3}, {outCast4});
    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->outCasts_.push_back(outCast3);
    currFunctionPtr->outCasts_.push_back(outCast4);
    RemoveRedundantOp removeredundantpass;
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);
    
    uint32_t view_num = kNumZero;
    uint32_t output_ubTensor1 = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++view_num;
        }
        if (op.GetOutputOperandSize() == 1 && op.GetOutputOperand(0) == ubTensor1) {
            ++output_ubTensor1;
        }
    }
    EXPECT_EQ(view_num, kNumZero);
    EXPECT_EQ(output_ubTensor1, kNumZero);
    EXPECT_EQ(exp1.GetOutputOperand(kSizeZero), outCast1);
    EXPECT_EQ(exp2.GetInputOperand(kSizeZero), ubTensor1);
    EXPECT_EQ(exp3.GetInputOperand(kSizeZero), ubTensor2);
    EXPECT_EQ(reci.GetInputOperand(kSizeZero), ubTensor2);
    EXPECT_EQ(sqrt.GetInputOperand(kSizeZero), ubTensor2);
}

/*
TESTRemoveDummyViewComm
inCast{8,16}->view->ubTensor1{4, 16}->wait_flag->outCast1{4,16}

inCast{8,16}->wait_flag->outCast1{4,16}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest7) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp", "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int> shape1 = {kNumEight, kNumExpFour};
    std::vector<int> shape2 = {kNumFour, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    
    currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast}, {ubTensor});
    auto &wait_flag = currFunctionPtr->AddOperation(Opcode::OP_COMM_WAIT_FLAG, {ubTensor}, {outCast});
    
    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp removeredundantpass;
    EXPECT_NE(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t view_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++view_num;
        }
    }
    EXPECT_EQ(view_num, kNumZero);
    EXPECT_EQ(wait_flag.GetInputOperandSize(), kSizeOne);
    EXPECT_EQ(wait_flag.GetInputOperand(kSizeZero), inCast);
}

/*
TESTRemoveDummyCopyIn
inCast{8,16}->view->ubTensor1{1,8,16}->copyin->ubTensor2{1,8,16}->exp->outCast{1,8,16}

inCast{8,16}->view->ubTensor1{1,8,16}->exp->outCast{1,8,16}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest8) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp", "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int> shape1 = {kNumEight, kNumExpFour};
    std::vector<int> shape2 = {kNumOne, kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_L1, false);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    
    auto op_attr = std::make_shared<ViewOpAttribute>(std::vector<int>{0, 0});
    auto &view = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast}, {ubTensor1});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {ubTensor1}, {ubTensor2});
    auto &exp = currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor2}, {outCast});
    view.SetOpAttribute(op_attr);

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp removeredundantpass;
    EXPECT_NE(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t copy_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_COPY_IN) {
            ++copy_num;
        }
    }
    EXPECT_EQ(copy_num, kNumZero);
    EXPECT_EQ(ubTensor1->GetMemoryTypeOriginal(), MemoryType::MEM_L1);
    EXPECT_EQ(ubTensor1->GetMemoryTypeToBe(), MemoryType::MEM_L1);
    EXPECT_EQ(op_attr->GetTo(), MemoryType::MEM_L1);
    EXPECT_EQ(exp.GetInputOperandSize(), kSizeOne);
    EXPECT_EQ(exp.GetInputOperand(kSizeZero), ubTensor1);
}

/*
TESTCannotRemoveDummyCopyIn
inCast{8,16}->view->ubTensor1{8,16}->copyin->ubTensor2{8,16}->view->outCast{1,8,16}

inCast{8,16}->view->ubTensor1{8,16}->copyin->ubTensor2{8,16}->view->outCast{1,8,16}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest9) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp", "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int> shape1 = {kNumEight, kNumExpFour};
    std::vector<int> shape2 = {kNumOne, kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_L1, false);
    
    auto op_attr = std::make_shared<ViewOpAttribute>(std::vector<int>{0, 0});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {inCast}, {ubTensor1});
    auto &copy = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {ubTensor1}, {ubTensor2});
    currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp removeredundantpass;
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t copy_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_COPY_IN) {
            ++copy_num;
        }
    }
    EXPECT_EQ(copy_num, kNumOne);
    auto inputOffset = ubTensor1->GetOffset();
    auto attr = std::static_pointer_cast<CopyOpAttribute>(copy.GetOpAttribute());
    auto offset = attr->GetFromOffset();
    std::vector<OpImmediate> newOffset;
    for (size_t i = 0; i < ubTensor2->shape.size(); i++) {
        EXPECT_EQ(offset[i].Dump(), OpImmediate::Specified(SymbolicScalar(inputOffset[i])).Dump());
    }
}

/*
TESTRemoveDummyView(WARNING CASE)
inCast{8,16}->view->ddrTensor{8,16}->assemble->outCast{1,8,16}
inCast{8,16}->assemble->outCast{1,8,16} (FAILED)
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest10) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp", "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int> shape = {kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ddrTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    
    currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast}, {ddrTensor});
    currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {ddrTensor}, {outCast});
    
    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp removeredundantpass;
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_NE(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);
}

/*
TESTRemoveDummyRegCopy
inCast{8,16}/{a0,16}->regcopy->ubTensor1{8,16}/{a1,16}->regcopy->ubTensor2{16,8}/{a1,16}->exp->outCast1{16,8}
inCast{8,16}/{a0,16}->regcopy->ubTensor1{8,16}/{a1,16}->exp->outCast1{16,8}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest11) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantOp", "TestRemoveRedundantOp", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int> shape1 = {kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    inCast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    ubTensor1->SetMemoryTypeBoth(MemoryType::MEM_UB);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    ubTensor2->SetMemoryTypeBoth(MemoryType::MEM_UB);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    outCast->SetMemoryTypeBoth(MemoryType::MEM_UB);
    
    auto &regcopy = currFunctionPtr->AddOperation(Opcode::OP_REGISTER_COPY, {inCast}, {ubTensor1});
    currFunctionPtr->AddOperation(Opcode::OP_REGISTER_COPY, {ubTensor1}, {ubTensor2});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor2}, {outCast});
    
    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantOp removeredundantpass;
    EXPECT_NE(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.PostCheck(*currFunctionPtr), SUCCESS);

    uint32_t regcopy_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_REGISTER_COPY) {
            EXPECT_EQ(op.GetOpMagic(), regcopy.GetOpMagic());
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(op.GetInputOperand(kSizeZero), inCast);
            ++regcopy_num;
        } else if (op.GetOpcode() == Opcode::OP_EXP) {
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(op.GetInputOperand(kSizeZero), ubTensor1);
        }
    }
    EXPECT_EQ(regcopy_num, kNumOne);
}

/*
view->exp(end assemble)->view(end assemble)->expand(end assemble)->exp(end assemble)
                                                                 ->exp(end assemble)

exp(end assemble*3) ->exp(end assemble)
                    ->exp(end assemble)
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpSTest1) {
    //Define the shape of the Tensors
    std::vector<int> shape = {kNumExpSix, kNumExpSix};
    
    PassManager &passManager = PassManager::Instance();

    Tensor input(DT_FP32, shape, "input");
    Tensor exp(DT_FP32, shape, "exp");
    Tensor view(DT_FP32, shape, "view");
    Tensor expand(DT_FP32, shape, "expand");
    Tensor output1(DT_FP32, shape, "output1");
    Tensor output2(DT_FP32, shape, "output2");
    
    FUNCTION("STCase1") {
        exp = Exp(input);
        view = View(exp, shape, {kNumZero, kNumZero});
        expand = Expand(view, shape);
        output1 = Exp(expand);
        output2 = Exp(expand);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase1");
    EXPECT_EQ(func->Operations().size(), kSizeEleven);

    passManager.RegisterStrategy("RemoveRedundantOpTestStrategy", {
        {   "RemoveRedundantOp",   "RemoveRedundantOp",  PassType::TYPE_TILE_GRAPH},
    });
    auto ret = passManager.RunPass(Program::GetInstance(), *func, "RemoveRedundantOpTestStrategy");
    EXPECT_EQ(ret, SUCCESS);
    
    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations();

    int view_num = kNumZero;
    int expand_num = kNumZero;
    EXPECT_EQ(updated_operations.size(), kSizeEight);
    for (const auto &op : updated_operations) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_num++;
        } else if (op.GetOpcode() == Opcode::OP_EXPAND) {
            expand_num++;
        } 
    }
    EXPECT_EQ(view_num, kNumZero);
    EXPECT_EQ(expand_num, kNumZero);
}

/*
view->exp(end assemble)->view(end assemble)->expand(end assemble)->exp(end assemble)
                                                                 ->exp(end assemble)

exp(end assemble)->view(end assemble)->expand(end assemble) ->exp(end assemble)
                                                            ->exp(end assemble)
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpSTest2) {
    //Define the shape of the Tensors
    std::vector<int> shape = {kNumExpSix, kNumExpSix};
    std::vector<int> shape2 = {kNumExpFive, kNumExpSeven};
    std::vector<int> shape3 = {kNumExpFour, kNumExpEight};
    
    PassManager &passManager = PassManager::Instance();

    Tensor input(DT_FP32, shape, "input");
    Tensor exp(DT_FP32, shape, "exp");
    Tensor view(DT_FP32, shape2, "view");
    Tensor expand(DT_FP32, shape3, "expand");
    Tensor output1(DT_FP32, shape3, "output1");
    Tensor output2(DT_FP32, shape3, "output2");
    
    FUNCTION("STCase2") {
        exp = Exp(input);
        view = View(exp, shape2, {kNumZero, kNumZero});
        expand = Expand(view, DT_FP16, shape3);
        output1 = Exp(expand);
        output2 = Exp(expand);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase2");
    EXPECT_EQ(func->Operations().size(), kSizeEleven);

    passManager.RegisterStrategy("RemoveRedundantOpTestStrategy", {
        {   "RemoveRedundantOp",   "RemoveRedundantOp",  PassType::TYPE_TILE_GRAPH},
    });
    auto ret = passManager.RunPass(Program::GetInstance(), *func, "RemoveRedundantOpTestStrategy");
    EXPECT_EQ(ret, SUCCESS);
    
    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations();

    int view_num = kNumZero;
    int expand_num = kNumZero;
    EXPECT_EQ(updated_operations.size(), kSizeTen);
    for (const auto &op : updated_operations) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_num++;
        } else if (op.GetOpcode() == Opcode::OP_EXPAND) {
            expand_num++;
        } 
    }
    EXPECT_EQ(view_num, kNumOne);
    EXPECT_EQ(expand_num, kNumOne);
}

/*
view{64,64} ->exp{64,64} ->assemble{64, 64}
view{64,64} ->view{32,64} ->exp{64, 64} ->assemble{32, 64} ->assemble{64, 64}
            ->view{32,64} ->exp{64, 64} ->assemble{32, 64}
view{64,64} ->view{32,64} ->exp{64, 64} ->assemble{32, 64}
            ->view{32,64} ->exp{64, 64} ->assemble{32, 64}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpSTest3) {
    //Define the shape of the Tensors
    std::vector<int> shape = {kNumExpSix, kNumExpSix};
    std::vector<int> tile_shape = {kNumExpFive, kNumExpSix};
    
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("ExpandFunctionTestStrategy", {
        {   "ExpandFunction",   "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
        {   "AssignMemoryType", "AssignMemoryType",  PassType::TYPE_TILE_GRAPH},
    });

    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");

    FUNCTION("STCase3") {
        Program::GetInstance().GetTileShape().SetVecTileShapes(tile_shape);
        output = Exp(input);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase3");
    EXPECT_EQ(func->Operations().size(), kSizeEight);
    int assemble_before = kNumZero;
    for (const auto &op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assemble_before++;
        }
    }
    EXPECT_EQ(assemble_before, kNumThree);

    passManager.RegisterStrategy("RemoveRedundantOpTestStrategy", {
        {   "RemoveRedundantOp",   "RemoveRedundantOp",  PassType::TYPE_TILE_GRAPH},
    });
    EXPECT_EQ(passManager.RunPass(Program::GetInstance(), *func, "RemoveRedundantOpTestStrategy"), SUCCESS);
    EXPECT_EQ(func->Operations().size(), kSizeSeven);

    // ================== Verify the effect of the Pass ==================
    int assemble_after = kNumZero;
    for (const auto &op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assemble_after++;
        }
    }
    EXPECT_EQ(assemble_after, kNumTwo);
    EXPECT_NE(assemble_after, assemble_before);
}

/*
view{64,64} ->exp{64,64} ->assemble{64, 64}

view{64,64} ->view{32,64} ->exp{64, 64} ->assemble{32, 64} ->assemble{64, 64}
            ->view{32,64} ->exp{64, 64} ->assemble{32, 64}
            ->view{32,64} ->exp{64, 64} ->assemble{32, 64}
            ->view{32,64} ->exp{64, 64} ->assemble{32, 64}

view{32,64} ->exp{64, 64} ->assemble{32, 64} ->assemble{32, 64}
view{32,64} ->exp{64, 64} ->assemble{32, 64}
view{32,64} ->exp{64, 64} ->assemble{32, 64}
view{32,64} ->exp{64, 64} ->assemble{32, 64}
*/
TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpSTest4) {
    //Define the shape of the Tensors
    std::vector<int> shape = {kNumExpSix, kNumExpSix};
    std::vector<int> tile_shape = {kNumExpFive, kNumExpFive};
    
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("ExpandFunctionTestStrategy", {
        {   "ExpandFunction",   "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
    });

    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");

    FUNCTION("STCase4") {
        Program::GetInstance().GetTileShape().SetVecTileShapes(tile_shape);
        output = Exp(input);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase4");
    EXPECT_EQ(func->Operations().size(), kSizeForteen);
    int assemble_before = kNumZero;
    for (const auto &op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assemble_before++;
        }
    }
    EXPECT_EQ(assemble_before, kNumFive);

    passManager.RegisterStrategy("RemoveRedundantOpTestStrategy", {
        {   "RemoveRedundantOp",   "RemoveRedundantOp",  PassType::TYPE_TILE_GRAPH},
    });
    EXPECT_EQ(passManager.RunPass(Program::GetInstance(), *func, "RemoveRedundantOpTestStrategy"), SUCCESS);
    EXPECT_EQ(func->Operations().size(), kSizeThirteen);

    // ================== Verify the effect of the Pass ==================
    int assemble_after = kNumZero;
    for (const auto &op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assemble_after++;
        }
    }
    EXPECT_EQ(assemble_after, kNumFive);
    EXPECT_EQ(assemble_after, assemble_before);
}
}
}