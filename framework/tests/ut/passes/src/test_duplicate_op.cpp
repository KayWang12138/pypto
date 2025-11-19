/**
 * Copyright (c)  Huawei Technologies Co., Ltd. 2025. 
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_duplicate_op.cpp
 * \brief Unit test for DuplicateOp pass.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "ut_json/ut_json_tool.h"
#include "interface/configs/config_manager.h"

#define private public
#include "passes/tile_graph_pass/graph_optimization/duplicate_op.h"
namespace npu {
namespace tile_fwk{
static const size_t kSizeZero = 0UL;
static const size_t kSizeOne = 1UL;
static const uint16_t kNumZero = 0u;
static const uint16_t kNumOne = 1u;
static const uint16_t kNumTwo = 2u;
static const uint16_t KNumThree = 3u;
static const uint16_t kNumFour = 4u;
static const uint16_t kNumFive = 5u;
static const uint16_t kNumSix = 6u;
static const uint16_t kNumSeven = 7u;
static const uint16_t kNumEight = 8u;
static const uint16_t kNumEleven = 11u;
static const uint16_t kNumForteen = 14u;
static const uint16_t kNumExpFour = 16u;
static const uint16_t kNumExpFive = 32u;
static const uint16_t kNumExpSix = 64u;
static const uint16_t kNumExpSeven = 128u;

class TestDuplicateOpPass : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "DuplicateOpTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

/*
TESTDuplicateViewSingleConsumer
inCast{8,16}->view->ubTensor{1,8,16}->exp->outCast{1,8,16}

inCast{8,16}->view->ubTensor{1,8,16}->exp->outCast{1,8,16}
*/
TEST_F(TestDuplicateOpPass, DuplicateViewUTest1) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumOne, kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);

    auto &veiwOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast}, {ubTensor});
    auto &tensorOffset = inCast->GetTensorOffset();
    veiwOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset.GetOffset(), tensorOffset.GetDynOffset(), ubTensor->GetDynValidShape()));
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    DuplicateOp duplicateoppass;
    auto status = duplicateoppass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    uint32_t view_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++view_num;
        }
    }
    EXPECT_EQ(view_num, kNumOne);
}

/*
TESTDuplicateViewThreeConsumer
inCast{8,16}->view->ubTensor{1,8,16}->exp->outCast1{1,8,16}
                                    ->view->outCast2{8,16}
                                    ->sqrt->outCast3{1,8,16}
inCast{8,16}->view->ubTensor{1,8,16}->view->outCast2{8,16}
            ->view->viewTensor1{1,8,16}->exp->outCast1{1,8,16}                        
            ->view->viewTensor2{1,8,16}->sqrt->outCast3{1,8,16}
*/
TEST_F(TestDuplicateOpPass, DuplicateViewUTest2) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumOne, kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto outCast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);

    auto &veiwOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast}, {ubTensor});
    auto &tensorOffset = inCast->GetTensorOffset();
    veiwOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset.GetOffset(), tensorOffset.GetDynOffset(), ubTensor->GetDynValidShape()));
    auto &view_op = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor}, {outCast2});
    auto &tensorOffset1 = ubTensor->GetTensorOffset();
    view_op.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset1.GetOffset(), tensorOffset1.GetDynOffset(), outCast2->GetDynValidShape()));
    auto &sqrt_op = currFunctionPtr->AddOperation(Opcode::OP_SQRT, {ubTensor}, {outCast3});
    auto &exp_op = currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor}, {outCast1});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->outCasts_.push_back(outCast3);
    currFunctionPtr->outCasts_.push_back(outCast1);

    DuplicateOp duplicateoppass;
    auto status = duplicateoppass.RunOnFunction(*currFunctionPtr);

    EXPECT_EQ(status, SUCCESS);

    uint32_t view_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++view_num;
        }
    }
    // 旧的两个 + 新的两个
    EXPECT_EQ(view_num, kNumFour);
    EXPECT_EQ(exp_op.GetInputOperandSize(), kNumOne);
    EXPECT_NE(exp_op.GetInputOperand(kSizeZero), ubTensor);
    EXPECT_EQ(view_op.GetInputOperandSize(), kNumOne);
    EXPECT_EQ(view_op.GetInputOperand(kSizeZero), ubTensor);
    EXPECT_EQ(sqrt_op.GetInputOperandSize(), kNumOne);
    EXPECT_NE(sqrt_op.GetInputOperand(kSizeZero), ubTensor);
}

/*
TESTDuplicateViewAlternativeConsumer
inCast{8,16}->view->ubTensor1{1,8,16}
            ->view->ubTensor2{1,8,16}
ubTensor1+ubTensor1->div->outCast1{1,8,16}
ubTensor1+ubTensor2->div->outCast2{1,8,16}
ubTensor2+ubTensor2->div->outCast3{1,8,16}

inCast{8,16}->view->ubTensor1'{1,8,16}
            ->view->ubTensor2'{1,8,16}
            ->view->ubTensor3'{1,8,16}
            ->view->ubTensor4'{1,8,16}
            ->view->ubTensor5'{1,8,16}
            ->view->ubTensor6'{1,8,16}
ubTensor1'+ubTensor2'->div->outCast1{1,8,16}
ubTensor3'+ubTensor4'->div->outCast2{1,8,16}
ubTensor5'+ubTensor6'->div->outCast3{1,8,16}
*/
TEST_F(TestDuplicateOpPass, DuplicateViewUTest3) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumOne, kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto &veiwOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast}, {ubTensor1});
    auto &tensorOffset = inCast->GetTensorOffset();
    veiwOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset.GetOffset(), tensorOffset.GetDynOffset(), ubTensor1->GetDynValidShape()));
    auto &veiwOp1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast}, {ubTensor2});
    auto &tensorOffset1 = inCast->GetTensorOffset();
    veiwOp1.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset1.GetOffset(), tensorOffset1.GetDynOffset(), ubTensor2->GetDynValidShape()));
    auto &div1 = currFunctionPtr->AddOperation(Opcode::OP_DIV, {ubTensor1, ubTensor1}, {outCast1});
    auto &div2 = currFunctionPtr->AddOperation(Opcode::OP_DIV, {ubTensor1, ubTensor2}, {outCast2});
    auto &div3 = currFunctionPtr->AddOperation(Opcode::OP_DIV, {ubTensor2, ubTensor2}, {outCast3});
    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast3);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->outCasts_.push_back(outCast1);
    DuplicateOp duplicateoppass;
    auto status = duplicateoppass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    uint32_t view_num = 0;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++view_num;
        }
    }
    EXPECT_EQ(view_num, kNumSix);
    EXPECT_EQ(div1.GetInputOperandSize(), kNumTwo);
    EXPECT_EQ(div2.GetInputOperandSize(), kNumTwo);
    EXPECT_EQ(div3.GetInputOperandSize(), kNumTwo);
    auto div1_input1 = div1.GetInputOperand(kSizeZero);
    auto div1_input2 = div1.GetInputOperand(kSizeOne);
    EXPECT_NE(div1_input1, ubTensor1);
    EXPECT_EQ(div1_input1, div1_input2);
    auto div2_input1 = div2.GetInputOperand(kSizeZero);
    auto div2_input2 = div2.GetInputOperand(kSizeOne);
    EXPECT_NE(div2_input1, ubTensor1);
    EXPECT_NE(div2_input2, ubTensor2);
    EXPECT_NE(div2_input1, div2_input2);
    auto div3_input1 = div3.GetInputOperand(kSizeZero);
    auto div3_input2 = div3.GetInputOperand(kSizeOne);
    EXPECT_NE(div3_input2, ubTensor2);
    EXPECT_EQ(div3_input1, div3_input2);
}

/*
incast    ->view  -> tensor1 -> exp  -> outcast1
                             -> exp  -> outcast2
incast    ->view  -> tensor1 -> exp  -> outcast1
                             -> exp  -> outcast2
*/
TEST_F(TestDuplicateOpPass, TestDupViewL1) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDupViewL1", "TestDupViewL1", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {8, 16};
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto &view_op = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {incast}, {tensor1});
    auto &exp_op1 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor1}, {outcast1});
    auto &exp_op2 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor1}, {outcast2});
    (void) exp_op1;
    (void) exp_op2;
    auto view_attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, 
                                                       MEM_L1, 
                                                       std::vector<SymbolicScalar>(), 
                                                       std::vector<SymbolicScalar>());
    view_op.SetOpAttribute(view_attr);
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast1);
    currFunctionPtr->outCasts_.push_back(outcast2);
    DuplicateOp duplicateoppass;
    duplicateoppass.RunOnFunction(*currFunctionPtr);
    int view_num = 0;
    auto opList = currFunctionPtr->Operations();
    for (auto &op : opList) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_num++;
        }
    }
    EXPECT_EQ(view_num, 1);
}

/*
incast -> view -> tensor1 -> view -> tensor2 -> view -> outcast4
               -> exp -> outcast2 -> exp -> outcast3
incast -> view -> tensor1 -> view -> tensor2 -> view -> outcast4
               -> exp -> tensor1' -> exp -> outcast3 view -> testor2' -> exp -> outcast3
*/
TEST_F(TestDuplicateOpPass, TestContinuousView) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {8, 16};
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto &view_op1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {incast}, {tensor1});
    auto &view_op2 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {tensor1}, {tensor2});
    auto &view_op3 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {tensor2}, {outcast2});
    [[maybe_unused]] auto &exp_op = currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor1}, {outcast1});
    [[maybe_unused]] auto &exp_op1 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {tensor2}, {outcast3});

    auto view_attr1 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0},
                                                        MEM_UNKNOWN, 
                                                        std::vector<SymbolicScalar>(), 
                                                        std::vector<SymbolicScalar>());
    view_op1.SetOpAttribute(view_attr1);

    auto view_attr2 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0},
                                                        MEM_UNKNOWN, 
                                                        std::vector<SymbolicScalar>(), 
                                                        std::vector<SymbolicScalar>());
    view_op2.SetOpAttribute(view_attr2);

    auto view_attr3 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0},
                                                        MEM_UNKNOWN, 
                                                        std::vector<SymbolicScalar>(), 
                                                        std::vector<SymbolicScalar>());
    view_op3.SetOpAttribute(view_attr3);

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast1);
    currFunctionPtr->outCasts_.push_back(outcast2);
    currFunctionPtr->outCasts_.push_back(outcast3);

    DuplicateOp duplicateoppass;
    duplicateoppass.RunOnFunction(*currFunctionPtr);
    int view_num = 0;
    auto opList = currFunctionPtr->Operations();
    for (const auto& op : opList)
    {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_num++;
        }
    }
    const int res = 5;
    EXPECT_EQ(view_num, res);
}

/*
view -> view(+end assemble) -> view(+end assemble)
                  -> sqrt(+end assemble)
                  -> exp(+end assemble)
                  -> view(+end assemble)
view -> view -> view(+end assemble)
         view -> sqrt(+end assemble)
         view -> exp(+end assemble)
         view-> view(+end assemble)
*/
TEST_F(TestDuplicateOpPass, DuplicateViewSTest1) {
    // define the shape of the tensors
    std::vector<int64_t> shape1 = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> shape2 = {kNumExpSeven, kNumExpSeven};
    std::vector<int64_t> shape3 = {kNumExpSeven, kNumExpSeven};

    PassManager &passManager = PassManager::Instance();
    Tensor input(DT_FP32, shape1, "input");
    Tensor view1(DT_FP32, shape2, "reshape1");
    Tensor output1(DT_FP32, shape3, "reshape1");
    Tensor output2(DT_FP32, shape2, "reshape2");
    Tensor output3(DT_FP32, shape2, "reshape3");
    Tensor output4(DT_FP32, shape1, "reshape4");

    FUNCTION("STCase1") {
        TileShape::Current().SetVecTile({64, 64});
        view1 = View(input, shape2, {kNumZero, kNumZero});
        output1 = View(view1, shape3, {kNumZero, kNumZero});
        output2 = Exp(view1);
        output3 = Sqrt(view1);
        output4 = View(view1, shape1, {kNumZero, kNumZero});
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase1");
    EXPECT_EQ(func->Operations().size(), kNumEleven);
    passManager.RegisterStrategy("DuplicateViewTestStrategy", {
        {"DuplicateOp", "DuplicateOp"}
    });

    auto ret = passManager.RunPass(Program::GetInstance(), *func, "DuplicateViewTestStrategy");
    EXPECT_EQ(ret, SUCCESS);
    func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase1");
    // =======================Verify the effect of the pass ================
    auto updated_operations = func->Operations();

    int view_num = 0;
    EXPECT_EQ(updated_operations.size(), kNumForteen);
    for (const auto &op : updated_operations) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_num++;
        }
    }
    EXPECT_EQ(view_num, kNumSeven);
}

/*
TESTDuplicateGatherinSingleConsumer
inCast{8,16}->Gatherin->ubTensor{8,16}->exp->outCast{8,16}

inCast{8,16}->Gatherin->ubTensor{8,16}->exp->outCast{8,16}
*/
TEST_F(TestDuplicateOpPass, DuplicateGatherInUTest1) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);

    currFunctionPtr->AddOperation(Opcode::OP_GATHER_IN_L1, {inCast}, {ubTensor});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    DuplicateOp duplicateoppass;
    auto status = duplicateoppass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    uint32_t gatherin_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_GATHER_IN_L1) {
            ++gatherin_num;
        }
    }
    EXPECT_EQ(gatherin_num, kNumOne);
}

/*
TESTDuplicateGatherinThreeConsumer
inCast{8,16}->Gatherin->ubTensor{1,8,16}->exp->outCast1{1,8,16}
                                    ->view->outCast2{8,16}
                                    ->sqrt->outCast3{1,8,16}
inCast{8,16}->Gatherin->ubTensor{1,8,16}->view->outCast2{8,16}
            ->Gatherin->GatherinTensor1{1,8,16}->exp->outCast1{1,8,16}
            ->Gatherin->GatherinTensor2{1,8,16}->sqrt->outCast3{1,8,16}
*/
TEST_F(TestDuplicateOpPass, DuplicateGatherInUTest2) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    int64_t j = 0;
    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumOne, kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto outCast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);

    auto &gatherinOp = currFunctionPtr->AddOperation(Opcode::OP_GATHER_IN_L1, {inCast}, {ubTensor});
    gatherinOp.SetAttribute(OpAttributeKey::startOffset,j);
    auto &exp_op = currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor}, {outCast1});
    auto &view_op = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor}, {outCast2});
    auto &tensorOffset1 = ubTensor->GetTensorOffset();
    view_op.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset1.GetOffset(), tensorOffset1.GetDynOffset(), outCast2->GetDynValidShape()));
    auto &sqrt_op = currFunctionPtr->AddOperation(Opcode::OP_SQRT, {ubTensor}, {outCast3});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->outCasts_.push_back(outCast3);

    DuplicateOp duplicateoppass;
    auto status = duplicateoppass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    uint32_t gatherinnum = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_GATHER_IN_L1) {
            ++gatherinnum;
        }
    }
    // 旧的两个 + 新的两个
    EXPECT_EQ(gatherinnum, KNumThree);
    EXPECT_EQ(exp_op.GetInputOperandSize(), kNumOne);
    EXPECT_EQ(exp_op.GetInputOperand(kSizeZero), ubTensor);
    EXPECT_EQ(view_op.GetInputOperandSize(), kNumOne);
    EXPECT_NE(view_op.GetInputOperand(kSizeZero), ubTensor);
    EXPECT_EQ(sqrt_op.GetInputOperandSize(), kNumOne);
    EXPECT_NE(sqrt_op.GetInputOperand(kSizeZero), ubTensor);
}

/*
TESTDuplicateViewAlternativeConsumer
inCast{8,16}->gatherin->ubTensor1{1,8,16}
            ->gatherin->ubTensor2{1,8,16}
ubTensor1+ubTensor1->div->outCast1{1,8,16}
ubTensor1+ubTensor2->div->outCast2{1,8,16}
ubTensor2+ubTensor2->div->outCast3{1,8,16}

inCast{8,16}->gatherin->ubTensor1'{1,8,16}
            ->gatherin->ubTensor2'{1,8,16}
            ->gatherin->ubTensor3'{1,8,16}
            ->gatherin->ubTensor4'{1,8,16}
ubTensor1'+ubTensor1'->div->outCast1{1,8,16}
ubTensor2'+ubTensor3'->div->outCast2{1,8,16}
ubTensor4'+ubTensor4'->div->outCast3{1,8,16}
*/
TEST_F(TestDuplicateOpPass, DuplicateGatherInUTest3) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    int64_t i = 0;
    int64_t j = 1;
    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumOne, kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);

    auto &gatherinOp = currFunctionPtr->AddOperation(Opcode::OP_GATHER_IN_L1, {inCast}, {ubTensor1});
    gatherinOp.SetAttribute(OpAttributeKey::startOffset, i);
    auto &gatherinOp1 = currFunctionPtr->AddOperation(Opcode::OP_GATHER_IN_L1, {inCast}, {ubTensor2});
    gatherinOp1.SetAttribute(OpAttributeKey::startOffset, j);
    auto &div_op1 = currFunctionPtr->AddOperation(Opcode::OP_DIV, {ubTensor1, ubTensor1}, {outCast1});
    auto &div_op2 = currFunctionPtr->AddOperation(Opcode::OP_DIV, {ubTensor1, ubTensor2}, {outCast2});
    auto &div_op3 = currFunctionPtr->AddOperation(Opcode::OP_DIV, {ubTensor2, ubTensor2}, {outCast3});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->outCasts_.push_back(outCast3);

    DuplicateOp duplicateoppass;
    auto status = duplicateoppass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    uint32_t gatherin_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_GATHER_IN_L1) {
            ++gatherin_num;
        }
    }
    EXPECT_EQ(gatherin_num, kNumFour);
    EXPECT_EQ(div_op1.GetInputOperandSize(), kNumTwo);
    EXPECT_EQ(div_op2.GetInputOperandSize(), kNumTwo);
    EXPECT_EQ(div_op3.GetInputOperandSize(), kNumTwo);

    auto div1_input1 = div_op1.GetInputOperand(kSizeZero);
    auto div1_input2 = div_op1.GetInputOperand(kSizeOne);
    EXPECT_EQ(div1_input2, ubTensor1);
    EXPECT_EQ(div1_input1, div1_input2);

    auto div2_input1 = div_op2.GetInputOperand(kSizeZero);
    auto div2_input2 = div_op2.GetInputOperand(kSizeOne);
    EXPECT_NE(div2_input1, ubTensor1);
    EXPECT_EQ(div2_input2, ubTensor2);
    EXPECT_NE(div2_input1, div2_input2);

    auto div3_input1 = div_op3.GetInputOperand(kSizeZero);
    auto div3_input2 = div_op3.GetInputOperand(kSizeOne);
    EXPECT_NE(div3_input2, ubTensor2);
    EXPECT_EQ(div3_input1, div3_input2);
}

/*
TESTDuplicateGatherinConsumerGatherin(ERROR)
inCast{8,16}->Gatherin->ubTensor{1,8,16}->Gatherin->outCast{1,8,16}
*/
TEST_F(TestDuplicateOpPass, DuplicateGatherInUTest4) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumOne, kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);

    currFunctionPtr->AddOperation(Opcode::OP_GATHER_IN_L1, {inCast}, {ubTensor});
    currFunctionPtr->AddOperation(Opcode::OP_GATHER_IN_L1, {ubTensor}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    DuplicateOp duplicateoppass;
    auto status = duplicateoppass.RunOnFunction(*currFunctionPtr);
    EXPECT_NE(status, SUCCESS);
}

/*
TESTDuplicateGatherinViewnNormal
inCast{8,16}->Gatherin->ubTensor1{8,16}->exp->outCast1{8,16}
                      ->view->ubtensor2 ->exp->outcast2
                                        ->view->outcast3
inCast{8,16}->Gatherin->ubTensor1{8,16}->exp->outCast1{8,16}
            ->Gatherin->ubTensor1'{8,16}->view->ubtensor2 ->exp->outcast2
            ->Gatherin->ubTensor1'{8,16}->view->ubtensor2 ->view->outcast3
*/
TEST_F(TestDuplicateOpPass, DuplicateViewGatherInUTest1) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    int64_t i = 0;
    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto &gatherin = currFunctionPtr->AddOperation(Opcode::OP_GATHER_IN_L1, {inCast}, {ubTensor1});
    gatherin.SetAttribute(OpAttributeKey::startOffset, i);
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor1}, {outCast1});
    auto &view_op = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor1}, {ubTensor2});
    auto &tensorOffset1 = ubTensor1->GetTensorOffset();
    view_op.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset1.GetOffset(), tensorOffset1.GetDynOffset(), ubTensor2->GetDynValidShape()));
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor2}, {outCast2});
    auto &view_op1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {outCast3});
    auto &tensorOffset2 = ubTensor2->GetTensorOffset();
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset2.GetOffset(), tensorOffset2.GetDynOffset(), outCast3->GetDynValidShape()));
    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->outCasts_.push_back(outCast3);

    DuplicateOp duplicateoppass;
    auto status = duplicateoppass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    uint32_t gatherin_num = kNumZero;
    uint32_t view_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++view_num;
        }
    }
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_GATHER_IN_L1) {
            ++gatherin_num;
        }
    }
    EXPECT_EQ(gatherin_num, KNumThree);
    EXPECT_EQ(view_num, KNumThree);
}

/*
TESTDuplicateGatherinView(special1)
inCast{8,16}->Gatherin->ubtensor0 ->view->ubtensor1 ->gatherIn->ubtensor2->sqrt->outcast1
                                                             ->exp->outcast2
inCast{8,16}->Gatherin->ubtensor0 ->view->ubtensor1 ->gatherIn->ubtensor2->sqrt->outcast1
            ->Gatherin->ubtensor0'->view->ubtensor1'->gatherIn->ubtensor2'->exp->outcast2
*/
TEST_F(TestDuplicateOpPass, DuplicateViewGatherInUTest2) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    int64_t i = 0;
    int64_t j = 0;
    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor0 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto &gatherin = currFunctionPtr->AddOperation(Opcode::OP_GATHER_IN_L1, {inCast}, {ubTensor0});
    gatherin.SetAttribute(OpAttributeKey::startOffset, i);
    auto &view_op = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor0}, {ubTensor1});
    auto &tensorOffset1 = ubTensor0->GetTensorOffset();
    view_op.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset1.GetOffset(), tensorOffset1.GetDynOffset(), ubTensor1->GetDynValidShape()));
    auto &gatherin1 = currFunctionPtr->AddOperation(Opcode::OP_GATHER_IN_L1, {ubTensor1}, {ubTensor2});
    gatherin1.SetAttribute(OpAttributeKey::startOffset, j);
    currFunctionPtr->AddOperation(Opcode::OP_SQRT, {ubTensor2}, {outCast1});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor2}, {outCast2});
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast2);
    
    DuplicateOp duplicateoppass;
    auto status = duplicateoppass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    uint32_t gatherin_num = kNumZero;
    uint32_t view_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_GATHER_IN_L1) {
            ++gatherin_num;
        }
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++view_num;
        }
    }
    EXPECT_EQ(gatherin_num, kNumFive);
    EXPECT_EQ(view_num, KNumThree);
}

/*
TESTDuplicateGatherinView(special2)
inCast{8,16}->View->ubtensor0 ->GatherIn->ubtensor1 ->View->ubtensor2->sqrt->outcast1
                                                                      ->exp->outcast2
inCast{8,16}->View->ubtensor0 ->GatherIn->ubtensor1 ->View->ubtensor2->sqrt->outcast1
            ->View->ubtensor0' ->GatherIn->ubtensor1' ->View->ubtensor2'->exp->outcast2
*/
TEST_F(TestDuplicateOpPass, DuplicateViewGatherInUTest3) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    int64_t i = 0;
    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor0 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto &view_op = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast}, {ubTensor0});
    auto &tensorOffset1 = inCast->GetTensorOffset();
    view_op.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset1.GetOffset(), tensorOffset1.GetDynOffset(), ubTensor0->GetDynValidShape()));
    auto &gatherin = currFunctionPtr->AddOperation(Opcode::OP_GATHER_IN_L1, {ubTensor0}, {ubTensor1});
    gatherin.SetAttribute(OpAttributeKey::startOffset, i);
    auto &view_op1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor1}, {ubTensor2});
    auto &tensorOffset2 = ubTensor1->GetTensorOffset();
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(tensorOffset1.GetOffset(), tensorOffset2.GetDynOffset(), ubTensor2->GetDynValidShape()));

    currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor2}, {outCast2});
    currFunctionPtr->AddOperation(Opcode::OP_SQRT, {ubTensor2}, {outCast1});
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->inCasts_.push_back(inCast);

    DuplicateOp duplicateoppass;
    auto status = duplicateoppass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    uint32_t view_num = kNumZero;
    uint32_t gatherin_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++view_num;
        }
        if (op.GetOpcode() == Opcode::OP_GATHER_IN_L1) {
            ++gatherin_num;
        }
    }
    EXPECT_EQ(view_num, kNumSeven);
    EXPECT_EQ(gatherin_num, KNumThree);
}
}
}