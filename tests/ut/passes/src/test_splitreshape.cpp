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
#include "passes/pass_manager.h"
#include "interface/configs/config_manager.h"

#define private public
#include "passes/tile_graph_pass/split_reshape.h"

namespace npu {
namespace tile_fwk{
static const uint32_t kNumZero = 0u;
static const uint32_t kNumOne = 1u;
static const uint32_t kNumTwo = 2u;
static const uint32_t kNumThree = 3u;
static const uint32_t kNumFour = 4u;
static const uint32_t kNumSix = 6u;
static const uint32_t kNumEight = 8u;
static const uint32_t kExpFour = 16u;
static const uint32_t kExpFive = 32u;
static const uint32_t kExpSix = 64u;
static const uint32_t kNumNineSix = 96u;
static const uint32_t kExpSeven = 128u;
static const uint32_t kExpEight = 256u;
static const size_t kSizeZero = 0UL;
static const size_t kSizeOne = 1UL;
static const size_t kSizeTwo = 2UL;
static const size_t kSizeFour = 4UL;

class TestSplitReshapePass : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "SplitReshapeTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

TEST_F(TestSplitReshapePass, SplitReshapeUTest1) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    
    SplitReshape pass;
    auto status = pass.Init();
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.copyOutSources.size(), kSizeZero);
    EXPECT_EQ(pass.reshapeSources.size(), kSizeZero);
    EXPECT_EQ(pass.mapOffset.size(), kSizeZero);
    EXPECT_EQ(pass.assembles.size(), kSizeZero);
    EXPECT_EQ(pass.reshapes.size(), kSizeZero);
    EXPECT_EQ(pass.redundentViewops.size(), kSizeZero);
    EXPECT_EQ(pass.reshapeRawOutputs.size(), kSizeZero);
}

TEST_F(TestSplitReshapePass, SplitReshapeUTest2) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int> shape = {kNumTwo, kNumOne, kNumEight};
    std::vector<int> offset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int> offset2 = {kNumOne, kNumZero, kNumZero};
    std::vector<int> shape1 = {kNumOne, kNumOne, kNumEight};
    std::vector<int> shape2 = {kNumTwo, kNumOne, kNumEight};
    std::vector<int> shape3 = {kNumTwo, kNumEight};
    
    std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input1 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor, offset1, shape1);
    auto input2 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor, offset2, shape1);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    
    auto &assemble_op1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input1}, {ubTensor});
    auto assemble_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assemble_op1.SetOpAttribute(assemble_Attr1);
    
    auto &assemble_op2 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input2}, {ubTensor});
    auto assemble_Attr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assemble_op2.SetOpAttribute(assemble_Attr2);

    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor}, {output});
    
    SplitReshape pass;
    auto status = pass.CollectCopyOut(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.reshapeSources.size(), kSizeOne);
    auto iter1 = pass.reshapeSources.find(output->tensor->rawmagic);
    EXPECT_NE(iter1, pass.reshapeSources.end());
    EXPECT_EQ(iter1->second, ubTensor);

    EXPECT_EQ(pass.copyOutSources.size(), kSizeOne);
    auto iter2 = pass.copyOutSources.find(ubTensor->tensor->rawmagic);
    EXPECT_NE(iter2, pass.copyOutSources.end());
    EXPECT_EQ(iter2->second.size(), kNumTwo);
    EXPECT_EQ(iter2->second.count(input1), kNumOne);
    EXPECT_EQ(iter2->second.count(input2), kNumOne);

    EXPECT_EQ(pass.mapOffset.size(), kSizeTwo);
    EXPECT_EQ(pass.mapOffset.count(input1->magic), kNumOne);
    EXPECT_EQ(pass.mapOffset[input1->magic].count(ubTensor->magic), kNumOne);
    EXPECT_EQ(pass.mapOffset[input1->magic][ubTensor->magic], offset1);
    EXPECT_EQ(pass.mapOffset.count(input2->magic), kNumOne);
    EXPECT_EQ(pass.mapOffset[input2->magic].count(ubTensor->magic), kNumOne);
    EXPECT_EQ(pass.mapOffset[input2->magic][ubTensor->magic], offset2);
}

TEST_F(TestSplitReshapePass, SplitReshapeUTest3) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int> shape = {kNumTwo, kNumOne, kNumEight};
    std::vector<int> offset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int> offset2 = {kNumOne, kNumZero, kNumZero};
    std::vector<int> shape1 = {kNumOne, kNumOne, kNumEight};
    std::vector<int> shape2 = {kNumTwo, kNumOne, kNumEight};
    std::vector<int> shape3 = {kNumTwo, kNumEight};
    
    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape);
    
    auto case1Input1 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor1, offset1, shape1);
    auto case1Input2 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor1, offset2, shape1);
    auto case1UbTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto case1Output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    auto &assemble_op1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {case1Input1}, {case1UbTensor});
    auto assemble_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assemble_op1.SetOpAttribute(assemble_Attr1);
    auto &assemble_op2 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {case1Input2}, {case1UbTensor});
    auto assemble_Attr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assemble_op2.SetOpAttribute(assemble_Attr2);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {case1UbTensor}, {case1Output});
    
    auto case2Input = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto case2Output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {case2Input}, {case2Output});

    auto case3Input1 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor1, offset1, shape1);
    auto case3Input2 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor2, offset2, shape1);
    auto case3UbTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto case3Output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    auto &assemble_op3 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {case3Input1}, {case3UbTensor});
    auto assemble_Attr3 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assemble_op3.SetOpAttribute(assemble_Attr3);
    auto &assemble_op4 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {case3Input2}, {case3UbTensor});
    auto assemble_Attr4 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assemble_op4.SetOpAttribute(assemble_Attr4);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {case3UbTensor}, {case3Output});
       
    auto case4Input1 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor1, offset1, shape1);
    auto case4UbTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto case4Output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    auto &assemble_op5 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {case4Input1}, {case4UbTensor});
    auto assemble_Attr5 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assemble_op5.SetOpAttribute(assemble_Attr5);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {case4UbTensor}, {case4Output});

    SplitReshape pass;
    auto status = pass.CollectCopyOut(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(pass.CheckSplit(case1UbTensor), true);
    EXPECT_EQ(pass.CheckSplit(case2Input), true);
    EXPECT_EQ(pass.CheckSplit(case3UbTensor), false);
    EXPECT_EQ(pass.CheckSplit(case4UbTensor), true);
}

TEST_F(TestSplitReshapePass, SplitReshapeUTest4) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    SplitReshape pass;
    Status status;
    std::vector<int32_t> inputShape;
    std::vector<int32_t> outputShape;
    std::vector<int32_t> alignedShape;
    std::vector<int32_t> expectedShape;

    alignedShape.clear();
    inputShape = {kExpSix, kExpSix};
    outputShape = {kExpFive, kExpSeven};
    expectedShape = {kExpFive, kNumTwo, kExpSix};
    status = pass.ShapeAlign(inputShape, outputShape, alignedShape);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(alignedShape, expectedShape);
    
    alignedShape.clear();
    inputShape = {kNumTwo, kExpFive, kExpSix, kExpSeven};
    outputShape = {kExpSix, kExpSix, kExpSeven};
    expectedShape = inputShape;
    status = pass.ShapeAlign(inputShape, outputShape, alignedShape);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(alignedShape, expectedShape);
    
    alignedShape.clear();
    inputShape = {kExpSix, kExpSix, kExpSeven};
    outputShape = {kNumTwo, kExpFive, kExpSix, kExpSeven};
    expectedShape = outputShape;
    status = pass.ShapeAlign(inputShape, outputShape, alignedShape);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(alignedShape, expectedShape);
    
    inputShape = {kNumNineSix, kNumFour};
    outputShape = {kExpSix, kNumSix};
    status = pass.ShapeAlign(inputShape, outputShape, alignedShape);
    EXPECT_EQ(status, WARNING);
}

TEST_F(TestSplitReshapePass, SplitReshapeUTest5) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    SplitReshape pass;
    Status status;
    std::vector<int32_t> rawShape;
    std::vector<int32_t> alignedShape;
    std::vector<int32_t> tileOffset;
    std::vector<int32_t> tileShape;
    std::vector<int32_t> expectOffset;
    std::vector<int32_t> expectShape;
    std::vector<int32_t> newOffset;
    std::vector<int32_t> newShape;

    ReshapeTilePara shapePara;

    rawShape = {kNumEight};
    alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    tileOffset = {kNumTwo};
    tileShape = {kNumTwo};
    shapePara = {rawShape, alignedShape, tileOffset, tileShape};
    status = pass.RawToAlign(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectOffset = {kNumZero, kNumOne, kNumZero};
    expectShape = {kNumOne, kNumOne, kNumTwo};
    EXPECT_EQ(newShape, expectShape);
    EXPECT_EQ(newOffset, expectOffset);

    rawShape = {kExpFive, kNumTwo, kExpSeven};
    alignedShape = {kExpFive, kNumTwo, kNumFour, kExpFive};
    tileOffset = {kNumZero, kNumZero, kNumZero};
    tileShape = {kNumOne, kNumOne, kExpSix};
    shapePara = {rawShape, alignedShape, tileOffset, tileShape};
    status = pass.RawToAlign(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectOffset = {kNumZero, kNumZero, kNumZero, kNumZero};
    expectShape = {kNumOne, kNumOne, kNumTwo, kExpFive};
    EXPECT_EQ(newShape, expectShape);
    EXPECT_EQ(newOffset, expectOffset);

    rawShape = {kExpFive, kNumTwo, kExpEight};
    alignedShape = {kExpFive, kNumTwo, kNumEight, kExpFive};
    tileOffset = {kNumZero, kNumOne, kExpSeven};
    tileShape = {kNumOne, kNumOne, kExpSix};
    shapePara = {rawShape, alignedShape, tileOffset, tileShape};
    status = pass.RawToAlign(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectOffset = {kNumZero, kNumOne, kNumFour, kNumZero};
    expectShape = {kNumOne, kNumOne, kNumTwo, kExpFive};
    EXPECT_EQ(newShape, expectShape);
    EXPECT_EQ(newOffset, expectOffset);

    rawShape = {kExpFive, kNumFour, kExpEight};
    alignedShape = {kExpFive, kNumFour, kNumEight, kExpFive};
    tileOffset = {kNumOne, kNumOne, kExpSeven};
    tileShape = {kNumOne, kNumTwo, kExpSix};
    shapePara = {rawShape, alignedShape, tileOffset, tileShape};
    status = pass.RawToAlign(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectOffset = {kNumOne, kNumOne, kNumFour, kNumZero};
    expectShape = {kNumOne, kNumTwo, kNumTwo, kExpFive};
    EXPECT_EQ(newShape, expectShape);
    EXPECT_EQ(newOffset, expectOffset);
}

TEST_F(TestSplitReshapePass, SplitReshapeUTest6) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    SplitReshape pass;
    Status status;
    std::vector<int32_t> alignedShape;
    std::vector<int32_t> rawShape;
    std::vector<int32_t> tileOffset;
    std::vector<int32_t> tileShape;
    std::vector<int32_t> expectOffset;
    std::vector<int32_t> expectShape;
    std::vector<int32_t> newOffset;
    std::vector<int32_t> newShape;

    ReshapeTilePara shapePara;

    rawShape = {kNumSix};
    alignedShape = {kNumTwo, kNumThree};
    tileOffset = {kNumOne, kNumZero};
    tileShape = {kNumOne, kNumThree};
    shapePara = {alignedShape, rawShape, tileOffset, tileShape};
    status = pass.AlignToRaw(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectOffset = {kNumThree};
    expectShape = {kNumThree};
    EXPECT_EQ(newShape, expectShape);
    EXPECT_EQ(newOffset, expectOffset);
    
    rawShape = {kExpFive, kNumTwo, kExpSeven};
    alignedShape = {kExpFive, kNumTwo, kNumFour, kExpFive};
    tileOffset = {kNumZero, kNumZero, kNumZero, kNumZero};
    tileShape = {kNumOne, kNumOne, kNumTwo, kExpFive};
    shapePara = {alignedShape, rawShape, tileOffset, tileShape};
    status = pass.AlignToRaw(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectOffset = {kNumZero, kNumZero, kNumZero};
    expectShape = {kNumOne, kNumOne, kExpSix};
    EXPECT_EQ(newShape, expectShape);
    EXPECT_EQ(newOffset, expectOffset);
    
    rawShape = {kExpFive, kNumTwo, kExpEight};
    alignedShape = {kExpFive, kNumTwo, kNumEight, kExpFive};
    tileOffset = {kNumZero, kNumOne, kNumFour, kNumZero};
    tileShape = {kNumOne, kNumOne, kNumTwo, kExpFive};
    shapePara = {alignedShape, rawShape, tileOffset, tileShape};
    status = pass.AlignToRaw(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectOffset = {kNumZero, kNumOne, kExpSeven};
    expectShape = {kNumOne, kNumOne, kExpSix};
    EXPECT_EQ(newShape, expectShape);
    EXPECT_EQ(newOffset, expectOffset);
    
    rawShape = {kExpFive, kNumFour, kExpEight};
    alignedShape = {kExpFive, kNumFour, kNumEight, kExpFive};
    tileOffset = {kNumOne, kNumOne, kNumFour, kNumZero};
    tileShape = {kNumOne, kNumTwo, kNumTwo, kExpFive};
    shapePara = {alignedShape, rawShape, tileOffset, tileShape};
    status = pass.AlignToRaw(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectOffset = {kNumOne, kNumOne, kExpSeven};
    expectShape = {kNumOne, kNumTwo, kExpSix};
    EXPECT_EQ(newShape, expectShape);
    EXPECT_EQ(newOffset, expectOffset);
}

TEST_F(TestSplitReshapePass, SplitReshapeUTest7) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    SplitReshape pass;
    Status status;
    std::vector<int32_t> alignedShape;
    std::vector<int32_t> rawShape;
    std::vector<int32_t> tileOffset;
    std::vector<int32_t> tileShape;
    std::vector<int32_t> newOffset;
    std::vector<int32_t> newShape;

    ReshapeTilePara shapePara;

    rawShape = {kNumEight};
    alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    tileOffset = {kNumZero, kNumTwo, kNumOne};
    tileShape = {kNumTwo, kNumTwo, kNumTwo};
    shapePara = {alignedShape, rawShape, tileOffset, tileShape};
    status = pass.AlignToRaw(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(newShape.empty());
    EXPECT_TRUE(newOffset.empty());

    rawShape = {kExpFive, kNumTwo, kExpEight};
    alignedShape = {kExpFive, kNumTwo, kNumEight, kExpFive};
    tileShape = {kNumOne, kNumOne, kNumTwo, kExpFive};
    tileOffset = {kNumZero, kNumOne, kNumZero, kExpFive};
    shapePara = {alignedShape, rawShape, tileOffset, tileShape};
    status = pass.AlignToRaw(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(newShape.empty());
    EXPECT_TRUE(newOffset.empty());
}

/*
多对一场景
rawShape = {2, 4}
{2, 4} -> assemble -> {2, 4} -> reshape -> {2, 2, 2}
*/
TEST_F(TestSplitReshapePass, SplitReshapeUTest8) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int> shape = {kNumTwo, kNumFour};
    std::vector<int> offset1 = {kNumZero, kNumZero};
    std::vector<int> offset2 = {kNumZero, kNumTwo};
    std::vector<int> shape1 = {kNumTwo, kNumTwo};
    std::vector<int> shape2 = {kNumTwo, kNumFour};
    std::vector<int> shape3 = {kNumTwo, kNumTwo, kNumTwo};
    
    std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input1 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor, offset1, shape1);
    auto input2 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor, offset2, shape1);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);

    auto &assemble_op1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input1}, {ubTensor});
    auto assemble_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assemble_op1.SetOpAttribute(assemble_Attr1);
    auto &assemble_op2 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input2}, {ubTensor});
    auto assemble_Attr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assemble_op2.SetOpAttribute(assemble_Attr2);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor}, {output});

    SplitReshape pass;
    LogicalTensors overlaps;
    LogicalTensors newOverlaps;
    std::vector<SymbolicScalar> validShape;
    std::vector<int> newOutputTileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int> newOutputTileShape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    auto newOutput = std::make_shared<LogicalTensor>(*currFunctionPtr, output->tensor, newOutputTileOffset, newOutputTileShape, validShape);
    copyOutTilePara copyOutTile = {ubTensor, output, newOutput, alignedShape, validShape};
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.ObtainCopyOutTile(*currFunctionPtr, copyOutTile, overlaps, newOverlaps), SUCCESS);
    EXPECT_EQ(overlaps.size(), kSizeTwo);
    EXPECT_NE(std::find(overlaps.begin(), overlaps.end(), input1), overlaps.end());
    EXPECT_NE(std::find(overlaps.begin(), overlaps.end(), input2), overlaps.end());
    EXPECT_EQ(newOverlaps.size(), kSizeTwo);
}

/*
一对一场景
rawShape = {2, 2, 2}
{2, 2, 2} -> assemble -> {2, 2, 2} -> reshape -> {4, 2}
*/
TEST_F(TestSplitReshapePass, SplitReshapeUTest9) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int> shape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> offset = {kNumZero, kNumZero, kNumZero};
    std::vector<int> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> shape2 = {kNumFour, kNumTwo};
    
    std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor, offset, shape1);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);

    auto &assemble_op = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input}, {ubTensor});
    auto assemble_Attr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset);
    assemble_op.SetOpAttribute(assemble_Attr);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor}, {output});

    SplitReshape pass;
    LogicalTensors overlaps;
    LogicalTensors newOverlaps;
    std::vector<SymbolicScalar> validShape;
    std::vector<int> newOutputTileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int> newOutputTileShape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    auto newOutput = std::make_shared<LogicalTensor>(*currFunctionPtr, output->tensor, newOutputTileOffset, newOutputTileShape, validShape);
    copyOutTilePara copyOutTile = {ubTensor, output, newOutput, alignedShape, validShape};
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.ObtainCopyOutTile(*currFunctionPtr, copyOutTile, overlaps, newOverlaps), SUCCESS);
    EXPECT_EQ(overlaps.size(), kSizeOne);
    EXPECT_NE(std::find(overlaps.begin(), overlaps.end(), input), overlaps.end());
    EXPECT_EQ(newOverlaps.size(), kSizeOne);
    EXPECT_EQ(newOverlaps[0]->GetOffset(), newOutputTileOffset);
    EXPECT_EQ(newOverlaps[0]->GetShape(), newOutputTileShape);
}

/*
验证一对一场景下ub数据的处理
rawShape = {2, 2, 2}
{2, 2, 2}(ub) -> assemble -> {2, 2, 2} -> reshape -> {4, 2} -> view -> {4,2}(ub) -> OP
{2, 2, 2}(ub) -> reshape(一个ReshapeOp成员) -> {4, 2}(ub) -> OP
*/
TEST_F(TestSplitReshapePass, SplitReshapeUTest10) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int> shape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> offset = {kNumZero, kNumZero, kNumZero};
    std::vector<int> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> shape2 = {kNumFour, kNumTwo};
    
    std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor, offset, shape1);
    input->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    output->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    auto opOutput = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);

    auto &assemble_op = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input}, {ubTensor1});
    auto assemble_Attr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset);
    assemble_op.SetOpAttribute(assemble_Attr);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &view_op = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output});
    auto &post_op = currFunctionPtr->AddOperation(Opcode::OP_NOP, {output}, {opOutput});

    CalcOverlapPara para;
    std::vector<SymbolicScalar> validShape;
    std::vector<int> newTileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int> newTileShape = {kNumTwo, kNumTwo, kNumTwo};
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input};
    para.newOverlaps = {std::make_shared<LogicalTensor>(*currFunctionPtr, input->tensor, newTileOffset, newTileShape, validShape)};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output;
    para.inputView = ubTensor2;
    
    SplitReshape pass;
    EXPECT_EQ(pass.redundentViewops.size(), kSizeZero);
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.UpdateForPerfectlyMatch(*currFunctionPtr, view_op, para), SUCCESS);
    EXPECT_EQ(pass.redundentViewops.size(), kSizeOne);
    EXPECT_NE(pass.redundentViewops.find(&view_op), pass.redundentViewops.end());
    auto newReshapeOutput = post_op.GetInputOperand(kSizeZero);
    EXPECT_NE(newReshapeOutput, output);
    EXPECT_EQ(newReshapeOutput->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(pass.reshapes.size(), kSizeOne);
    auto reshape = pass.reshapes.begin()->second;
    EXPECT_EQ(reshape->input, input);
    EXPECT_EQ(reshape->output, newReshapeOutput);
}

/*
验证一对一场景下ddr数据的处理
rawShape = {2, 2, 2}
{2, 2, 2}(ddr) -> assemble -> {2, 2, 2} -> reshape -> {4, 2}(unknown) -> view -> {4,2}(ub) -> OP
{2, 2, 2}(ddr) -> reshape(一个ReshapeOp成员) -> {4, 2}(unknown) -> view -> {4,2}(ub) -> OP
*/
TEST_F(TestSplitReshapePass, SplitReshapeUTest11) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int> shape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> offset = {kNumZero, kNumZero, kNumZero};
    std::vector<int> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> shape2 = {kNumFour, kNumTwo};
    
    std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor, offset, shape1);
    input->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    output->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    
    auto &assemble_op = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input}, {ubTensor1});
    auto assemble_Attr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset);
    assemble_op.SetOpAttribute(assemble_Attr);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &view_op = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output});
    std::vector<int> view_offset = {0, 0};
    auto view_Attr = std::make_shared<ViewOpAttribute>(view_offset);
    view_op.SetOpAttribute(view_Attr);

    CalcOverlapPara para;
    std::vector<SymbolicScalar> validShape;
    std::vector<int> newTileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int> newTileShape = {kNumTwo, kNumTwo, kNumTwo};
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input};
    para.newOverlaps = {std::make_shared<LogicalTensor>(*currFunctionPtr, input->tensor, newTileOffset, newTileShape, validShape)};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output;
    para.inputView = ubTensor2;
    
    SplitReshape pass;
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.UpdateForPerfectlyMatch(*currFunctionPtr, view_op, para), SUCCESS);
    auto newReshapeOutput = view_op.GetInputOperand(kSizeZero);
    EXPECT_NE(newReshapeOutput, ubTensor2);
    EXPECT_EQ(newReshapeOutput->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
    EXPECT_EQ(pass.reshapes.size(), kSizeOne);
    auto reshape = pass.reshapes.begin()->second;
    EXPECT_EQ(reshape->input, input);
    EXPECT_EQ(reshape->output, newReshapeOutput);
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(view_op.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute->GetFromOffset(), view_offset);
}

/*
验证一对一场景下其他数据的处理
rawShape = {2, 2, 2}
{2, 2, 2}(ddr) -> assemble -> {2, 2, 2}(unknown) -> reshape -> {4, 2}(unknown) -> view -> {4,2}(ddr) -> OP
{2, 2, 2}(ddr) -> assemble -> {2, 2, 2}(unknown) -> reshape(一个ReshapeOp成员) -> {4, 2}(unknown) -> view -> {4,2}(ddr) -> OP
*/
TEST_F(TestSplitReshapePass, SplitReshapeUTest12) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int> shape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> offset = {kNumZero, kNumZero, kNumZero};
    std::vector<int> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> shape2 = {kNumFour, kNumTwo};
    
    std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor, offset, shape1);
    input->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    output->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    
    auto &assemble_op = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input}, {ubTensor1});
    auto assemble_Attr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset);
    assemble_op.SetOpAttribute(assemble_Attr);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &view_op = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output});
    std::vector<int> view_offset = {0, 0};
    auto view_Attr = std::make_shared<ViewOpAttribute>(view_offset);
    view_op.SetOpAttribute(view_Attr);

    CalcOverlapPara para;
    std::vector<SymbolicScalar> validShape;
    std::vector<int> newTileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int> newTileShape = {kNumTwo, kNumTwo, kNumTwo};
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input};
    para.newOverlaps = {std::make_shared<LogicalTensor>(*currFunctionPtr, input->tensor, newTileOffset, newTileShape, validShape)};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output;
    para.inputView = ubTensor2;
    
    SplitReshape pass;
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.UpdateForPerfectlyMatch(*currFunctionPtr, view_op, para), SUCCESS);
    auto newReshapeOutput = view_op.GetInputOperand(kSizeZero);
    EXPECT_NE(newReshapeOutput, ubTensor2);
    EXPECT_EQ(newReshapeOutput->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
    EXPECT_EQ(pass.reshapes.size(), kSizeOne);
    auto reshape = pass.reshapes.begin()->second;
    EXPECT_EQ(pass.assembles.size(), kSizeOne);
    auto assemble = pass.assembles.begin();
    auto newReshapeSource = reshape->input;
    EXPECT_EQ(reshape->output, newReshapeOutput);
    EXPECT_EQ(newReshapeSource->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(view_op.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute->GetFromOffset(), view_offset);
    EXPECT_EQ(assemble->from, MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(assemble->toOffset, offset);
    EXPECT_EQ(assemble->input, input);
    EXPECT_EQ(assemble->output, newReshapeSource);
}

/*
验证一对多场景下ub数据的处理
rawShape = {2, 2, 2}
{2, 2, 2}(ub) -> assemble -> {2, 2, 2} -> reshape -> {2, 4} -> view -> {2, 2}(ub)
                                                            -> view -> {2, 2}(ub) 
{2, 2, 2}(ub) -> reshape -> {2, 4}(ub) -> view -> {2, 2}
                                       -> view -> {2, 2}           
*/
TEST_F(TestSplitReshapePass, SplitReshapeUTest13) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> shape2 = {kNumTwo, kNumFour};
    std::vector<int> shape3 = {kNumTwo, kNumTwo};
    std::vector<int> offset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int> offset2 = {kNumZero, kNumZero};
    std::vector<int> view_offset1 = {kNumZero, kNumZero};
    std::vector<int> view_offset2 = {kNumZero, kNumTwo};

    std::shared_ptr<RawTensor> RawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor1, offset1, shape1);
    input->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    std::shared_ptr<RawTensor> RawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape2);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor2, offset2, shape2);
    auto output1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    output1->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    auto output2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    output2->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    
    auto &assemble_op = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input}, {ubTensor1});
    auto assemble_Attr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assemble_op.SetOpAttribute(assemble_Attr);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &view_op1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output1});
    auto view_Attr1 = std::make_shared<ViewOpAttribute>(view_offset1);
    view_op1.SetOpAttribute(view_Attr1);
    auto &view_op2 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output2});
    auto view_Attr2 = std::make_shared<ViewOpAttribute>(view_offset2);
    view_op2.SetOpAttribute(view_Attr2);

    CalcOverlapPara para;
    std::vector<SymbolicScalar> validShape;
    
    SplitReshape pass;
    std::vector<int> newCopyOutTileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int> newCopyOutTileShape = {kNumTwo, kNumTwo, kNumTwo};
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input};
    para.newOverlaps = {std::make_shared<LogicalTensor>(*currFunctionPtr, input->tensor, newCopyOutTileOffset, newCopyOutTileShape, validShape)};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output1;
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    
    std::vector<int> viewOffset = {kNumZero, kNumZero};
    auto inputView = std::make_shared<LogicalTensor>(*currFunctionPtr, ubTensor2->tensor, viewOffset, shape2);
    para.inputView = inputView;
    para.newInputViewTileShape = {kNumTwo, kNumOne, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumZero, kNumZero};
    EXPECT_EQ(pass.UpdateForBeCovered(*currFunctionPtr, view_op1, para), SUCCESS);
    std::vector<int> view2Offset = {kNumZero, kNumTwo};
    para.newInputViewTileShape = {kNumTwo, kNumOne, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumOne, kNumZero};
    EXPECT_EQ(pass.UpdateForBeCovered(*currFunctionPtr, view_op2, para), SUCCESS);
    
    EXPECT_EQ(pass.reshapes.size(), kSizeOne);
    auto newReshape = pass.reshapes.begin()->second;
    EXPECT_EQ(newReshape->input, input);
    EXPECT_NE(newReshape->output, ubTensor2);
    EXPECT_EQ(newReshape->output->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    auto viewOpAttribute1 = dynamic_cast<ViewOpAttribute *>(view_op1.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute1->GetFromOffset(), view_offset1);
    auto viewOpAttribute2 = dynamic_cast<ViewOpAttribute *>(view_op2.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute2->GetFromOffset(), view_offset2);
}

/*
验证一对多场景下其他数据的处理
rawShape = {2, 2, 2}
{2, 2, 2}(ddr) -> assemble -> {2, 2, 2} -> reshape -> {2, 4}(unknown) -> view -> {2, 2}(ddr)
                                                                      -> view -> {2, 2}(ddr) 
{2, 2, 2}(ddr) -> assemble -> {2, 2, 2} -> reshape -> {2, 4}(unknown) -> view -> {2, 2}
                                                                      -> view -> {2, 2}           
*/
TEST_F(TestSplitReshapePass, SplitReshapeUTest14) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> shape2 = {kNumTwo, kNumFour};
    std::vector<int> shape3 = {kNumTwo, kNumTwo};
    std::vector<int> offset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int> offset2 = {kNumZero, kNumZero};
    std::vector<int> view_offset1 = {kNumZero, kNumZero};
    std::vector<int> view_offset2 = {kNumZero, kNumTwo};

    std::shared_ptr<RawTensor> RawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor1, offset1, shape1);
    input->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    std::shared_ptr<RawTensor> RawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape2);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor2, offset2, shape2);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto output1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    output1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    output2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    
    auto &assemble_op = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input}, {ubTensor1});
    auto assemble_Attr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assemble_op.SetOpAttribute(assemble_Attr);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &view_op1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output1});
    auto view_Attr1 = std::make_shared<ViewOpAttribute>(view_offset1);
    view_op1.SetOpAttribute(view_Attr1);
    auto &view_op2 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output2});
    auto view_Attr2 = std::make_shared<ViewOpAttribute>(view_offset2);
    view_op2.SetOpAttribute(view_Attr2);

    CalcOverlapPara para;
    std::vector<SymbolicScalar> validShape;
    
    SplitReshape pass;
    std::vector<int> newCopyOutTileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int> newCopyOutTileShape = {kNumTwo, kNumTwo, kNumTwo};
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input};
    para.newOverlaps = {std::make_shared<LogicalTensor>(*currFunctionPtr, input->tensor, newCopyOutTileOffset, newCopyOutTileShape, validShape)};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output1;
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    
    std::vector<int> viewOffset = {kNumZero, kNumZero};
    auto inputView = std::make_shared<LogicalTensor>(*currFunctionPtr, ubTensor2->tensor, viewOffset, shape2);
    para.inputView = inputView;
    para.newInputViewTileShape = {kNumTwo, kNumOne, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumZero, kNumZero};
    EXPECT_EQ(pass.UpdateForBeCovered(*currFunctionPtr, view_op1, para), SUCCESS);
    std::vector<int> view2Offset = {kNumZero, kNumTwo};
    para.newInputViewTileShape = {kNumTwo, kNumOne, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumOne, kNumZero};
    EXPECT_EQ(pass.UpdateForBeCovered(*currFunctionPtr, view_op2, para), SUCCESS);
    
    EXPECT_EQ(pass.reshapes.size(), kSizeOne);
    auto newReshape = pass.reshapes.begin()->second;
    auto newReshapeResource = newReshape->input;
    EXPECT_NE(newReshape->output, ubTensor2);
    EXPECT_EQ(newReshape->output->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
    EXPECT_EQ(pass.assembles.size(), kSizeOne);
    auto newAssemble = pass.assembles.begin();
    EXPECT_EQ(newAssemble->from, MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(newAssemble->toOffset, offset1);
    EXPECT_EQ(newAssemble->input, input);
    EXPECT_EQ(newAssemble->output, newReshapeResource);
    auto viewOpAttribute1 = dynamic_cast<ViewOpAttribute *>(view_op1.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute1->GetFromOffset(), view_offset1);
    auto viewOpAttribute2 = dynamic_cast<ViewOpAttribute *>(view_op2.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute2->GetFromOffset(), view_offset2);
}

/*
验证多对一场景下ub数据的处理
rawShape = {2, 4}
{2, 2}(ub) -> assemble -> {2, 4}(unknown) -> reshape -> {2, 2, 2} -> view -> {2, 2, 2}(ub)
{2, 2}(ub) -> assemble ->
{2, 2}(ub) -> {2, 4}(unknown) -> reshape -> {2, 2, 2}(ub)        
{2, 2}(ub) -> 
*/
TEST_F(TestSplitReshapePass, SplitReshapeUTest15) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int> shape1 = {kNumTwo, kNumFour};
    std::vector<int> shape2 = {kNumTwo, kNumTwo};
    std::vector<int> shape3 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> offset1 = {kNumZero, kNumZero};
    std::vector<int> offset2 = {kNumZero, kNumTwo};
    std::vector<int> view_offset = {kNumZero, kNumZero, kNumZero};

    std::shared_ptr<RawTensor> RawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto input1 = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor1, offset1, shape2);
    input1->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    auto input2 = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor1, offset2, shape2);
    input2->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    output->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    auto postOutput = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    
    auto &assemble_op1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input1}, {ubTensor1});
    auto assemble_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_UB, offset1);
    assemble_op1.SetOpAttribute(assemble_Attr1);
    auto &assemble_op2 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input2}, {ubTensor1});
    auto assemble_Attr2 = std::make_shared<AssembleOpAttribute>(MEM_UB, offset2);
    assemble_op2.SetOpAttribute(assemble_Attr2);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &view_op = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output});
    auto view_Attr = std::make_shared<ViewOpAttribute>(view_offset);
    view_op.SetOpAttribute(view_Attr);
    auto &post_op = currFunctionPtr->AddOperation(Opcode::OP_NOP, {output}, {postOutput});

    CalcOverlapPara para;
    std::vector<SymbolicScalar> validShape;
    
    SplitReshape pass;
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input1, input2};
    std::vector<int> newInput1TileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int> newInput1TileShape = {kNumTwo, kNumOne, kNumTwo};
    auto newInput1 = std::make_shared<LogicalTensor>(*currFunctionPtr, input1->tensor, newInput1TileOffset, newInput1TileShape, validShape);
    std::vector<int> newInput2TileOffset = {kNumZero, kNumOne, kNumZero};
    std::vector<int> newInput2TileShape = {kNumTwo, kNumOne, kNumTwo};
    auto newInput2 = std::make_shared<LogicalTensor>(*currFunctionPtr, input2->tensor, newInput2TileOffset, newInput2TileShape, validShape);
    para.newOverlaps = {newInput1, newInput2};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output;
    para.newInputViewTileShape = {kNumTwo, kNumTwo, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumZero, kNumZero};
    auto inputView = std::make_shared<LogicalTensor>(*currFunctionPtr, ubTensor2->tensor, view_offset, shape3);
    para.inputView = inputView;
    EXPECT_EQ(pass.redundentViewops.size(), kSizeZero);
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.UpdateForPerfectlyMatchWithAll(*currFunctionPtr, view_op, para), SUCCESS);
    EXPECT_EQ(pass.redundentViewops.size(), kSizeOne);
    EXPECT_NE(pass.redundentViewops.find(&view_op), pass.redundentViewops.end());
    auto newReshapeOutput = post_op.GetInputOperand(kSizeZero);
    EXPECT_NE(newReshapeOutput, output);
    EXPECT_EQ(pass.reshapes.size(), kSizeOne);
    auto reshape = pass.reshapes.begin()->second;
    auto newReshapeSource = reshape->input;
    EXPECT_EQ(reshape->output, newReshapeOutput);
    EXPECT_EQ(newReshapeSource->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(newReshapeOutput->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
}

/*
验证多对一场景下其他数据的处理
rawShape = {2, 4}
{2, 2}(ddr) -> assemble -> {2, 4}(unknown) -> reshape -> {2, 2, 2}(ddr) -> view -> {2, 2, 2}(ddr)
{2, 2}(ddr) -> assemble ->
{2, 2}(ddr) -> {2, 4}(unknown) -> reshape -> {2, 2, 2}(ddr) -> view -> {2, 2}          
{2, 2}(ddr) -> 
*/
TEST_F(TestSplitReshapePass, SplitReshapeUTest16) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int> shape1 = {kNumTwo, kNumFour};
    std::vector<int> shape2 = {kNumTwo, kNumTwo};
    std::vector<int> shape3 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> offset1 = {kNumZero, kNumZero};
    std::vector<int> offset2 = {kNumZero, kNumTwo};
    std::vector<int> view_offset = {kNumZero, kNumZero, kNumZero};

    std::shared_ptr<RawTensor> RawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto input1 = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor1, offset1, shape2);
    input1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto input2 = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor1, offset2, shape2);
    input2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    output->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    
    auto &assemble_op1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input1}, {ubTensor1});
    auto assemble_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assemble_op1.SetOpAttribute(assemble_Attr1);
    auto &assemble_op2 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input2}, {ubTensor1});
    auto assemble_Attr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assemble_op2.SetOpAttribute(assemble_Attr2);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &view_op = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output});
    auto view_Attr = std::make_shared<ViewOpAttribute>(view_offset);
    view_op.SetOpAttribute(view_Attr);
    
    CalcOverlapPara para;
    std::vector<SymbolicScalar> validShape;
    
    SplitReshape pass;
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input1, input2};
    std::vector<int> newInput1TileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int> newInput1TileShape = {kNumTwo, kNumOne, kNumTwo};
    auto newInput1 = std::make_shared<LogicalTensor>(*currFunctionPtr, input1->tensor, newInput1TileOffset, newInput1TileShape, validShape);
    std::vector<int> newInput2TileOffset = {kNumZero, kNumOne, kNumZero};
    std::vector<int> newInput2TileShape = {kNumTwo, kNumOne, kNumTwo};
    auto newInput2 = std::make_shared<LogicalTensor>(*currFunctionPtr, input2->tensor, newInput2TileOffset, newInput2TileShape, validShape);
    para.newOverlaps = {newInput1, newInput2};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output;
    para.newInputViewTileShape = {kNumTwo, kNumTwo, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumZero, kNumZero};
    auto inputView = std::make_shared<LogicalTensor>(*currFunctionPtr, ubTensor2->tensor, view_offset, shape3);
    para.inputView = inputView;
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.UpdateForPerfectlyMatchWithAll(*currFunctionPtr, view_op, para), SUCCESS);
    EXPECT_EQ(pass.redundentViewops.size(), kSizeZero);
    EXPECT_EQ(pass.reshapes.size(), kSizeOne);
    auto reshape = pass.reshapes.begin()->second;
    auto newReshapeSource = reshape->input;
    auto newReshapeOutput = view_op.GetInputOperand(kSizeZero);
    EXPECT_EQ(reshape->output, newReshapeOutput);
    EXPECT_EQ(newReshapeSource->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
    EXPECT_EQ(newReshapeOutput->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(view_op.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute->GetFromOffset(), inputView->offset);
}

/*
验证多对一兜底场景下ub数据的处理
rawShape = {2, 4}
{2, 2}(ub) -> assemble -> {2, 4}(unknown) -> reshape -> {2, 2, 2} -> view -> {2, 2, 2}(ub)
{2, 2}(ub) -> assemble ->
{2, 2}(ub) -> reshape -> {2, 1, 2}(ub) -> assemble -> {2, 2, 2}(ub)        
{2, 2}(ub) -> reshape -> {2, 1, 2}(ub) -> assemble
*/
TEST_F(TestSplitReshapePass, SplitReshapeUTest17) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int> shape1 = {kNumTwo, kNumFour};
    std::vector<int> shape2 = {kNumTwo, kNumTwo};
    std::vector<int> shape3 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> offset1 = {kNumZero, kNumZero};
    std::vector<int> offset2 = {kNumZero, kNumTwo};
    std::vector<int> view_offset = {kNumZero, kNumZero, kNumZero};

    std::shared_ptr<RawTensor> RawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto input1 = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor1, offset1, shape2);
    input1->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    auto input2 = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor1, offset2, shape2);
    input2->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    output->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    
    auto &assemble_op1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input1}, {ubTensor1});
    auto assemble_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_UB, offset1);
    assemble_op1.SetOpAttribute(assemble_Attr1);
    auto &assemble_op2 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input2}, {ubTensor1});
    auto assemble_Attr2 = std::make_shared<AssembleOpAttribute>(MEM_UB, offset2);
    assemble_op2.SetOpAttribute(assemble_Attr2);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &view_op = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output});
    auto view_Attr = std::make_shared<ViewOpAttribute>(view_offset);
    view_op.SetOpAttribute(view_Attr);
    
    CalcOverlapPara para;
    std::vector<SymbolicScalar> validShape;
    
    SplitReshape pass;
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input1, input2};
    std::vector<int> newInput1TileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int> newInput1TileShape = {kNumTwo, kNumOne, kNumTwo};
    auto newInput1 = std::make_shared<LogicalTensor>(*currFunctionPtr, input1->tensor, newInput1TileOffset, newInput1TileShape, validShape);
    std::vector<int> newInput2TileOffset = {kNumZero, kNumOne, kNumZero};
    std::vector<int> newInput2TileShape = {kNumTwo, kNumOne, kNumTwo};
    auto newInput2 = std::make_shared<LogicalTensor>(*currFunctionPtr, input2->tensor, newInput2TileOffset, newInput2TileShape, validShape);
    para.newOverlaps = {newInput1, newInput2};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output;
    para.newInputViewTileShape = {kNumTwo, kNumOne, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumOne, kNumZero};
    std::vector<int> InputViewTileShape = {kNumTwo, kNumTwo, kNumTwo};
    auto inputView = std::make_shared<LogicalTensor>(*currFunctionPtr, ubTensor2->tensor, view_offset, InputViewTileShape);
    para.inputView = inputView;
    EXPECT_EQ(pass.redundentViewops.size(), kSizeZero);
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.UpdateForAssembleAfterReshape(*currFunctionPtr, view_op, para), SUCCESS);
    EXPECT_EQ(pass.redundentViewops.size(), kSizeOne);
    EXPECT_EQ(pass.reshapes.size(), kSizeTwo);
    EXPECT_EQ(pass.assembles.size(), kSizeTwo);

    AssembleOp newAssembleOp1;
    AssembleOp newAssembleOp2;
    auto newReshapeOp1 = std::make_shared<ReshapeOp>(nullptr, nullptr);
    auto newReshapeOp2 = std::make_shared<ReshapeOp>(nullptr, nullptr);
    int reshapeCnt = 0;
    int assembleCnt = 0;
    std::vector<int> assemble_offset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int> assemble_offset2 = {kNumZero, kNumOne, kNumZero};
    for (const auto &assemble : pass.assembles) {
        EXPECT_EQ(assemble.from, MemoryType::MEM_UB);
        EXPECT_EQ(assemble.output, output);
        if (assemble.toOffset == assemble_offset1) {
            newAssembleOp1 = assemble;
            assembleCnt += 1;
        } else if (assemble.toOffset == assemble_offset2) {
            newAssembleOp2 = assemble;
            assembleCnt += 10;
        }
    }
    EXPECT_EQ(assembleCnt, 11);
    auto newReshapeOutput1 = newAssembleOp1.input;
    EXPECT_EQ(newReshapeOutput1->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    auto newReshapeOutput2 = newAssembleOp2.input;
    EXPECT_EQ(newReshapeOutput2->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    for (const auto &reshape : pass.reshapes) {
        auto reshapeOp = reshape.second;
        if (reshapeOp->output == newReshapeOutput1) {
            newReshapeOp1 = reshapeOp;
            reshapeCnt += 1;
        } else if (reshapeOp->output == newReshapeOutput2) {
            newReshapeOp2 = reshapeOp;
            reshapeCnt += 10;
        }
    }
    EXPECT_EQ(reshapeCnt, 11);
    EXPECT_EQ(newReshapeOp1->input, input1);
    EXPECT_EQ(newReshapeOp2->input, input2);
}

/*
验证多对一兜底场景下ddr数据的处理
rawShape = {2, 4}
{2, 2}(ddr) -> assemble -> {2, 4}(unknown) -> reshape -> {2, 2, 2}(unknown) -> view -> {2, 2, 2}(ub)
{2, 2}(ddr) -> assemble ->
{2, 2}(ddr) -> reshape -> {2, 1, 2}(unknown) -> assemble -> {2, 2, 2} -> view -> {2, 2, 2}(ub)
{2, 2}(ddr) -> reshape -> {2, 1, 2}(unknown) -> assemble
*/
TEST_F(TestSplitReshapePass, SplitReshapeUTest18) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int> shape1 = {kNumTwo, kNumFour};
    std::vector<int> shape2 = {kNumTwo, kNumTwo};
    std::vector<int> shape3 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> offset1 = {kNumZero, kNumZero};
    std::vector<int> offset2 = {kNumZero, kNumTwo};
    std::vector<int> view_offset = {kNumZero, kNumZero, kNumZero};

    std::shared_ptr<RawTensor> RawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto input1 = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor1, offset1, shape2);
    input1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto input2 = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor1, offset2, shape2);
    input2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    output->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    
    auto &assemble_op1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input1}, {ubTensor1});
    auto assemble_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_UB, offset1);
    assemble_op1.SetOpAttribute(assemble_Attr1);
    auto &assemble_op2 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input2}, {ubTensor1});
    auto assemble_Attr2 = std::make_shared<AssembleOpAttribute>(MEM_UB, offset2);
    assemble_op2.SetOpAttribute(assemble_Attr2);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &view_op = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output});
    auto view_Attr = std::make_shared<ViewOpAttribute>(view_offset);
    view_op.SetOpAttribute(view_Attr);
    
    CalcOverlapPara para;
    std::vector<SymbolicScalar> validShape;
    
    SplitReshape pass;
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input1, input2};
    std::vector<int> newInput1TileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int> newInput1TileShape = {kNumTwo, kNumOne, kNumTwo};
    auto newInput1 = std::make_shared<LogicalTensor>(*currFunctionPtr, input1->tensor, newInput1TileOffset, newInput1TileShape, validShape);
    std::vector<int> newInput2TileOffset = {kNumZero, kNumOne, kNumZero};
    std::vector<int> newInput2TileShape = {kNumTwo, kNumOne, kNumTwo};
    auto newInput2 = std::make_shared<LogicalTensor>(*currFunctionPtr, input2->tensor, newInput2TileOffset, newInput2TileShape, validShape);
    para.newOverlaps = {newInput1, newInput2};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output;
    para.newInputViewTileShape = {kNumTwo, kNumOne, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumOne, kNumZero};
    std::vector<int> InputViewTileShape = {kNumTwo, kNumTwo, kNumTwo};
    auto inputView = std::make_shared<LogicalTensor>(*currFunctionPtr, ubTensor2->tensor, view_offset, InputViewTileShape);
    para.inputView = inputView;
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.UpdateForAssembleAfterReshape(*currFunctionPtr, view_op, para), SUCCESS);
    EXPECT_EQ(pass.redundentViewops.size(), kSizeZero);
    EXPECT_EQ(pass.reshapes.size(), kSizeTwo);
    EXPECT_EQ(pass.assembles.size(), kSizeTwo);

    AssembleOp newAssembleOp1;
    AssembleOp newAssembleOp2;
    auto newReshapeOp1 = std::make_shared<ReshapeOp>(nullptr, nullptr);
    auto newReshapeOp2 = std::make_shared<ReshapeOp>(nullptr, nullptr);
    int reshapeCnt = 0;
    int assembleCnt = 0;
    std::vector<int> assemble_offset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int> assemble_offset2 = {kNumZero, kNumOne, kNumZero};
    for (const auto &assemble : pass.assembles) {
        EXPECT_EQ(assemble.from, MemoryType::MEM_UNKNOWN);
        EXPECT_EQ(assemble.output, view_op.GetInputOperand(kSizeZero));
        if (assemble.toOffset == assemble_offset1) {
            newAssembleOp1 = assemble;
            assembleCnt += 1;
        } else if (assemble.toOffset == assemble_offset2) {
            newAssembleOp2 = assemble;
            assembleCnt += 10;
        }
    }
    EXPECT_EQ(assembleCnt, 11);
    auto newReshapeOutput1 = newAssembleOp1.input;
    EXPECT_EQ(newReshapeOutput1->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
    auto newReshapeOutput2 = newAssembleOp2.input;
    EXPECT_EQ(newReshapeOutput2->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
    for (const auto &reshape : pass.reshapes) {
        auto reshapeOp = reshape.second;
        if (reshapeOp->output == newReshapeOutput1) {
            newReshapeOp1 = reshapeOp;
            reshapeCnt += 1;
        } else if (reshapeOp->output == newReshapeOutput2) {
            newReshapeOp2 = reshapeOp;
            reshapeCnt += 10;
        }
    }
    EXPECT_EQ(reshapeCnt, 11);
    EXPECT_EQ(newReshapeOp1->input, input1);
    EXPECT_EQ(newReshapeOp2->input, input2);
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(view_op.GetOpAttribute().get());
    EXPECT_NE(viewOpAttribute, nullptr);
    EXPECT_EQ(viewOpAttribute->GetFromOffset(), view_offset);
}

/*
验证多对一兜底场景下其他数据的处理
rawShape = {2, 4}
{2, 2}(ddr) -> assemble -> {2, 4}(unknown) -> reshape -> {2, 2, 2}(unknown) -> view -> {2, 2, 2}(ddr)
{2, 2}(ddr) -> assemble ->
{2, 2}(ddr) -> assemble -> {2, 2}(unknown) -> reshape -> {2, 1, 2}(ub) -> assemble -> {2, 2, 2}(unknown) -> view -> {2, 2, 2}(ddr)
{2, 2}(ddr) -> assemble -> {2, 2}(unknown) -> reshape -> {2, 1, 2}(ub) -> assemble -> 
*/
TEST_F(TestSplitReshapePass, SplitReshapeUTest19) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDuplicateView", "TestDuplicateView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int> shape1 = {kNumTwo, kNumFour};
    std::vector<int> shape2 = {kNumTwo, kNumTwo};
    std::vector<int> shape3 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> offset1 = {kNumZero, kNumZero};
    std::vector<int> offset2 = {kNumZero, kNumTwo};
    std::vector<int> view_offset = {kNumZero, kNumZero, kNumZero};

    std::shared_ptr<RawTensor> RawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto input1 = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor1, offset1, shape2);
    input1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto input2 = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor1, offset2, shape2);
    input2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    output->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    
    auto &assemble_op1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input1}, {ubTensor1});
    auto assemble_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_UB, offset1);
    assemble_op1.SetOpAttribute(assemble_Attr1);
    auto &assemble_op2 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input2}, {ubTensor1});
    auto assemble_Attr2 = std::make_shared<AssembleOpAttribute>(MEM_UB, offset2);
    assemble_op2.SetOpAttribute(assemble_Attr2);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &view_op = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output});
    auto view_Attr = std::make_shared<ViewOpAttribute>(view_offset);
    view_op.SetOpAttribute(view_Attr);
    
    CalcOverlapPara para;
    std::vector<SymbolicScalar> validShape;
    
    SplitReshape pass;
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input1, input2};
    std::vector<int> newInput1TileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int> newInput1TileShape = {kNumTwo, kNumOne, kNumTwo};
    auto newInput1 = std::make_shared<LogicalTensor>(*currFunctionPtr, input1->tensor, newInput1TileOffset, newInput1TileShape, validShape);
    std::vector<int> newInput2TileOffset = {kNumZero, kNumOne, kNumZero};
    std::vector<int> newInput2TileShape = {kNumTwo, kNumOne, kNumTwo};
    auto newInput2 = std::make_shared<LogicalTensor>(*currFunctionPtr, input2->tensor, newInput2TileOffset, newInput2TileShape, validShape);
    para.newOverlaps = {newInput1, newInput2};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output;
    para.newInputViewTileShape = {kNumTwo, kNumOne, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumOne, kNumZero};
    std::vector<int> InputViewTileShape = {kNumTwo, kNumTwo, kNumTwo};
    auto inputView = std::make_shared<LogicalTensor>(*currFunctionPtr, ubTensor2->tensor, view_offset, InputViewTileShape);
    para.inputView = inputView;
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.UpdateForAssembleAfterReshape(*currFunctionPtr, view_op, para), SUCCESS);
    EXPECT_EQ(pass.redundentViewops.size(), kSizeZero);
    EXPECT_EQ(pass.reshapes.size(), kSizeTwo);
    EXPECT_EQ(pass.assembles.size(), kSizeFour);

    AssembleOp assembleBeforeReshape1;
    AssembleOp assembleAfterReshape1;
    AssembleOp assembleBeforeReshape2;
    AssembleOp assembleAfterReshape2;
    auto newReshapeOp1 = std::make_shared<ReshapeOp>(nullptr, nullptr);
    auto newReshapeOp2 = std::make_shared<ReshapeOp>(nullptr, nullptr);
    int assembleCnt = 0;
    int reshapeCnt = 0;
    std::vector<int> assemble_offset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int> assemble_offset2 = {kNumZero, kNumOne, kNumZero};
    for (const auto &assemble : pass.assembles) {
        if (assemble.toOffset == assemble_offset1) {
            EXPECT_EQ(assemble.from, MemoryType::MEM_UNKNOWN);
            EXPECT_EQ(assemble.output, view_op.GetInputOperand(kSizeZero));
            assembleAfterReshape1 = assemble;
            assembleCnt += 1;
        } else if (assemble.toOffset == assemble_offset2) {
            EXPECT_EQ(assemble.from, MemoryType::MEM_UNKNOWN);
            EXPECT_EQ(assemble.output, view_op.GetInputOperand(kSizeZero));
            assembleAfterReshape2 = assemble;
            assembleCnt += 10;
        } else if (assemble.input == input1) {
            EXPECT_EQ(assemble.from, MemoryType::MEM_DEVICE_DDR);
            assembleBeforeReshape1 = assemble;
            assembleCnt += 100;
        } else if (assemble.input == input2) {
            EXPECT_EQ(assemble.from, MemoryType::MEM_DEVICE_DDR);
            assembleBeforeReshape2 = assemble;
            assembleCnt += 1000;
        }
    }
    EXPECT_EQ(assembleCnt, 1111);
    for (const auto &reshape : pass.reshapes) {
        auto reshapeOp = reshape.second;
        if (reshapeOp->input == assembleBeforeReshape1.output) {
            EXPECT_EQ(reshapeOp->output, assembleAfterReshape1.input);
            EXPECT_EQ(reshapeOp->input->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
            reshapeCnt += 1;
        } else if (reshapeOp->input == assembleBeforeReshape2.output) {
            EXPECT_EQ(reshapeOp->output, assembleAfterReshape2.input);
            EXPECT_EQ(reshapeOp->input->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
            reshapeCnt += 10;
        }
    }
    EXPECT_EQ(reshapeCnt, 11);
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(view_op.GetOpAttribute().get());
    EXPECT_NE(viewOpAttribute, nullptr);
    EXPECT_EQ(viewOpAttribute->GetFromOffset(), view_offset);
}

/*
校验一对一场景(使用expandfunction作为前序pass)
1) 用例设置：
{2,2,4} -> exp -> {2,2,4} -> reshape -> {2,2,1,4} -> exp -> {2,2,1,4}
tileshape = {2,2,2,2}
2) expandfunction：
exp -> {2,2,2} -> assemble -> reshape -> {2,2,1,4} -> view -> {2,2,1,2} -> exp -> {2,2,1,2}
exp -> {2,2,2} -> assemble                         -> view -> {2,2,1,2} -> exp -> {2,2,1,2}
3) splitreshape
exp -> {2,2,2} -> assemble -> reshape -> {2,2,1,2} -> view -> {2,2,1,2} -> exp -> {2,2,1,2}
exp -> {2,2,2} -> assemble -> reshape -> {2,2,1,2} -> view -> {2,2,1,2} -> exp -> {2,2,1,2}
*/
TEST_F(TestSplitReshapePass, SplitReshapeSTest1) {
    //Define the shape of the Tensors
    std::vector<int> origShape = {kNumTwo, kNumTwo, kNumFour};
    std::vector<int> reshapeShape = {kNumTwo, kNumTwo, kNumOne, kNumFour};
    std::vector<int> tiledShape = {kNumTwo, kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> tiledorigShape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> tiledreshapeShape = {kNumTwo, kNumTwo, kNumOne, kNumTwo};

    Program::GetInstance().GetTileShape().SetVecTileShapes(tiledShape);
    Tensor input(DT_FP32, origShape, "input");
    Tensor output(DT_FP32, reshapeShape, "output");

    FUNCTION("STCase1") {
        Tensor exp = Exp(input);
        Tensor reshape = Reshape(exp, reshapeShape);
        output = Exp(reshape);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase1");
    
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("ExpandFunctionStrategy", {
        {   "ExpandFunction",   "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
    });
    EXPECT_EQ(passManager.RunPass(Program::GetInstance(), *func, "ExpandFunctionStrategy"), SUCCESS);
    
    int reshapeOp = 0;
    for (auto &op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            auto reshapeInput = op.GetInputOperand(kSizeZero);
            EXPECT_NE(reshapeInput, nullptr);
            EXPECT_EQ(reshapeInput->shape, origShape);
            EXPECT_EQ(reshapeInput->GetProducers().size(), kSizeTwo);
            for (const auto &producer : reshapeInput->GetProducers()) {
                EXPECT_EQ(producer->GetOpcode(), Opcode::OP_ASSEMBLE);
            }
            EXPECT_EQ(op.GetOutputOperandSize(), kSizeOne);
            auto reshapeOutput = op.GetOutputOperand(kSizeZero);
            EXPECT_NE(reshapeOutput, nullptr);
            EXPECT_EQ(reshapeOutput->shape, reshapeShape);
            EXPECT_EQ(reshapeOutput->GetConsumers().size(), kSizeTwo);
            for (const auto &consumer : reshapeOutput->GetConsumers()) {
                EXPECT_EQ(consumer->GetOpcode(), Opcode::OP_VIEW);
            }
            reshapeOp++;
        }
    }
    EXPECT_EQ(reshapeOp, kNumOne);

    passManager.RegisterStrategy("SplitReshapeTestStrategy", {
        {   "SplitReshape",   "SplitReshape",  PassType::TYPE_TILE_GRAPH},
    });
    EXPECT_EQ(passManager.RunPass(Program::GetInstance(), *func, "SplitReshapeTestStrategy"), SUCCESS);
    
    reshapeOp = 0;
    for (auto &op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            auto reshapeInput = op.GetInputOperand(kSizeZero);
            EXPECT_NE(reshapeInput, nullptr);
            EXPECT_EQ(reshapeInput->shape, tiledorigShape);
            EXPECT_EQ(reshapeInput->GetProducers().size(), kSizeOne);
            for (const auto &producer : reshapeInput->GetProducers()) {
                EXPECT_EQ(producer->GetOpcode(), Opcode::OP_ASSEMBLE);
            }
            EXPECT_EQ(op.GetOutputOperandSize(), kSizeOne);
            auto reshapeOutput = op.GetOutputOperand(kSizeZero);
            EXPECT_NE(reshapeOutput, nullptr);
            EXPECT_EQ(reshapeOutput->shape, tiledreshapeShape);
            EXPECT_EQ(reshapeOutput->GetConsumers().size(), kSizeOne);
            for (const auto &consumer : reshapeOutput->GetConsumers()) {
                EXPECT_EQ(consumer->GetOpcode(), Opcode::OP_VIEW);
            }
            reshapeOp++;
        }
    }
    EXPECT_EQ(reshapeOp, kNumTwo);
}

/*
校验一对多场景(使用expandfunction作为前序pass)
1) 用例设置：
{4,2,2} -> exp -> {4,2,2} -> reshape -> {4,4} -> exp -> {4,4}
tileshape = {2,2,2}
2) expandfunction：
exp -> {2,2,2} -> assemble -> reshape -> {4,4} -> view -> {2,2} -> exp -> {2,2}
exp -> {2,2,2} -> assemble                     -> view -> {2,2} -> exp -> {2,2}
                                               -> view -> {2,2} -> exp -> {2,2}
                                               -> view -> {2,2} -> exp -> {2,2}
3) splitreshape
                                               -> view -> {2,2} -> exp -> {2,2}
exp -> {2,2,2} -> assemble -> reshape -> {2,4} -> view -> {2,2} -> exp -> {2,2}
exp -> {2,2,2} -> assemble -> reshape -> {2,4} -> view -> {2,2} -> exp -> {2,2}
                                               -> view -> {2,2} -> exp -> {2,2}
*/
TEST_F(TestSplitReshapePass, SplitReshapeSTest2) {
    //Define the shape of the Tensors
    std::vector<int> origShape = {kNumFour, kNumTwo, kNumTwo};
    std::vector<int> reshapeShape = {kNumFour, kNumFour};
    std::vector<int> tiledShape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> tiledorigShape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> tiledreshapeShape = {kNumTwo, kNumFour};
    std::vector<int> tiledviewShape = {kNumTwo, kNumTwo};

    Program::GetInstance().GetTileShape().SetVecTileShapes(tiledShape);
    Tensor input(DT_FP32, origShape, "input");
    Tensor output(DT_FP32, reshapeShape, "output");

    FUNCTION("STCase2") {
        Tensor exp = Exp(input);
        Tensor reshape = Reshape(exp, reshapeShape);
        output = Exp(reshape);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase2");
    
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("ExpandFunctionStrategy", {
        {   "ExpandFunction",   "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
    });
    EXPECT_EQ(passManager.RunPass(Program::GetInstance(), *func, "ExpandFunctionStrategy"), SUCCESS);
    
    int reshapeOp = 0;
    for (auto &op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            auto reshapeInput = op.GetInputOperand(kSizeZero);
            EXPECT_NE(reshapeInput, nullptr);
            EXPECT_EQ(reshapeInput->shape, origShape);
            EXPECT_EQ(reshapeInput->GetProducers().size(), kSizeTwo);
            for (const auto &producer : reshapeInput->GetProducers()) {
                EXPECT_EQ(producer->GetOpcode(), Opcode::OP_ASSEMBLE);
            }
            EXPECT_EQ(op.GetOutputOperandSize(), kSizeOne);
            auto reshapeOutput = op.GetOutputOperand(kSizeZero);
            EXPECT_NE(reshapeOutput, nullptr);
            EXPECT_EQ(reshapeOutput->shape, reshapeShape);
            EXPECT_EQ(reshapeOutput->GetConsumers().size(), kSizeFour);
            for (const auto &consumer : reshapeOutput->GetConsumers()) {
                EXPECT_EQ(consumer->GetOpcode(), Opcode::OP_VIEW);
            }
            reshapeOp++;
        }
    }
    EXPECT_EQ(reshapeOp, kNumOne);

    passManager.RegisterStrategy("SplitReshapeTestStrategy", {
        {   "SplitReshape",   "SplitReshape",  PassType::TYPE_TILE_GRAPH},
    });
    EXPECT_EQ(passManager.RunPass(Program::GetInstance(), *func, "SplitReshapeTestStrategy"), SUCCESS);
    
    reshapeOp = 0;
    for (auto &op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            auto reshapeInput = op.GetInputOperand(kSizeZero);
            EXPECT_NE(reshapeInput, nullptr);
            EXPECT_EQ(reshapeInput->shape, tiledorigShape);
            EXPECT_EQ(reshapeInput->GetProducers().size(), kSizeOne);
            for (const auto &producer : reshapeInput->GetProducers()) {
                EXPECT_EQ(producer->GetOpcode(), Opcode::OP_ASSEMBLE);
            }
            EXPECT_EQ(op.GetOutputOperandSize(), kSizeOne);
            auto reshapeOutput = op.GetOutputOperand(kSizeZero);
            EXPECT_NE(reshapeOutput, nullptr);
            EXPECT_EQ(reshapeOutput->shape, tiledreshapeShape);
            EXPECT_EQ(reshapeOutput->GetConsumers().size(), kSizeTwo);
            for (const auto &consumer : reshapeOutput->GetConsumers()) {
                EXPECT_EQ(consumer->GetOpcode(), Opcode::OP_VIEW);
                EXPECT_EQ(consumer->GetOutputOperandSize(), kSizeOne);
                EXPECT_EQ(consumer->GetOutputOperand(kSizeZero)->shape, tiledviewShape);
            }
            reshapeOp++;
        }
    }
    EXPECT_EQ(reshapeOp, kNumTwo);
}

/*
校验多对一场景(使用expandfunction作为前序pass)
1) 用例设置：
{2,4,4} -> exp -> {2,4,4} -> reshape -> {2,4,2,2} -> exp -> {2,4,2,2}
tileshape = {2,2,2,2}
2) expandfunction：
exp -> {2,2,2} -> assemble -> reshape -> {2,4,2,2} -> view -> {2,2,2,2} -> exp -> {2,2,2,2}
exp -> {2,2,2} -> assemble                         -> view -> {2,2,2,2} -> exp -> {2,2,2,2}
exp -> {2,2,2} -> assemble
exp -> {2,2,2} -> assemble
3) splitreshape
exp -> {2,2,2} -> assemble ->
exp -> {2,2,2} -> assemble -> reshape -> {2,2,2,2} -> view -> {2,2,2,2} -> exp -> {2,2,2,2}
exp -> {2,2,2} -> assemble -> reshape -> {2,2,2,2} -> view -> {2,2,2,2} -> exp -> {2,2,2,2}
exp -> {2,2,2} -> assemble ->
*/
TEST_F(TestSplitReshapePass, SplitReshapeSTest3) {
    //Define the shape of the Tensors
    std::vector<int> origShape = {kNumTwo, kNumFour, kNumFour};
    std::vector<int> reshapeShape = {kNumTwo, kNumFour, kNumTwo, kNumTwo};
    std::vector<int> tiledShape = {kNumTwo, kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> tiledassembleShape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> tiledreshapeShape = {kNumTwo, kNumTwo, kNumFour};
    std::vector<int> tiledviewShape = {kNumTwo, kNumTwo, kNumTwo, kNumTwo};

    Program::GetInstance().GetTileShape().SetVecTileShapes(tiledShape);
    Tensor input(DT_FP32, origShape, "input");
    Tensor output(DT_FP32, reshapeShape, "output");

    FUNCTION("STCase3") {
        Tensor exp = Exp(input);
        Tensor reshape = Reshape(exp, reshapeShape);
        output = Exp(reshape);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase3");
    
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("ExpandFunctionStrategy", {
        {   "ExpandFunction",   "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
    });
    EXPECT_EQ(passManager.RunPass(Program::GetInstance(), *func, "ExpandFunctionStrategy"), SUCCESS);
    
    int reshapeOp = 0;
    for (auto &op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            auto reshapeInput = op.GetInputOperand(kSizeZero);
            EXPECT_NE(reshapeInput, nullptr);
            EXPECT_EQ(reshapeInput->shape, origShape);
            EXPECT_EQ(reshapeInput->GetProducers().size(), kSizeFour);
            for (const auto &producer : reshapeInput->GetProducers()) {
                EXPECT_EQ(producer->GetOpcode(), Opcode::OP_ASSEMBLE);
            }
            EXPECT_EQ(op.GetOutputOperandSize(), kSizeOne);
            auto reshapeOutput = op.GetOutputOperand(kSizeZero);
            EXPECT_NE(reshapeOutput, nullptr);
            EXPECT_EQ(reshapeOutput->shape, reshapeShape);
            EXPECT_EQ(reshapeOutput->GetConsumers().size(), kSizeTwo);
            for (const auto &consumer : reshapeOutput->GetConsumers()) {
                EXPECT_EQ(consumer->GetOpcode(), Opcode::OP_VIEW);
            }
            reshapeOp++;
        }
    }
    EXPECT_EQ(reshapeOp, kNumOne);

    passManager.RegisterStrategy("SplitReshapeTestStrategy", {
        {   "SplitReshape",   "SplitReshape",  PassType::TYPE_TILE_GRAPH},
    });
    EXPECT_EQ(passManager.RunPass(Program::GetInstance(), *func, "SplitReshapeTestStrategy"), SUCCESS);
    
    reshapeOp = 0;
    for (auto &op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            auto reshapeInput = op.GetInputOperand(kSizeZero);
            EXPECT_NE(reshapeInput, nullptr);
            EXPECT_EQ(reshapeInput->shape, tiledreshapeShape);
            EXPECT_EQ(reshapeInput->GetProducers().size(), kSizeTwo);
            for (const auto &producer : reshapeInput->GetProducers()) {
                EXPECT_EQ(producer->GetOpcode(), Opcode::OP_ASSEMBLE);
                EXPECT_EQ(producer->GetInputOperandSize(), kSizeOne);
                EXPECT_EQ(producer->GetInputOperand(kSizeZero)->shape, tiledassembleShape);
            }
            EXPECT_EQ(op.GetOutputOperandSize(), kSizeOne);
            auto reshapeOutput = op.GetOutputOperand(kSizeZero);
            EXPECT_NE(reshapeOutput, nullptr);
            EXPECT_EQ(reshapeOutput->shape, tiledviewShape);
            EXPECT_EQ(reshapeOutput->GetConsumers().size(), kSizeOne);
            for (const auto &consumer : reshapeOutput->GetConsumers()) {
                EXPECT_EQ(consumer->GetOpcode(), Opcode::OP_VIEW);
                EXPECT_EQ(consumer->GetOutputOperandSize(), kSizeOne);
                EXPECT_EQ(consumer->GetOutputOperand(kSizeZero)->shape, tiledviewShape);
            }
            reshapeOp++;
        }
    }
    EXPECT_EQ(reshapeOp, kNumTwo);
}

/*
校验多对一兜底场景(使用expandfunction作为前序pass)
1) 用例设置：
{1,16} -> exp -> {1,16} -> reshape -> {1,1,2,8} -> exp -> {1,1,2,8}
tileshape1 = {1,4}
tileshape2 = {2,1,2,4}
2) expandfunction：
exp -> {1,4} -> assemble -> {1,16} -> reshape -> {1,1,2,8} -> view -> {1,1,2,4} -> exp -> {1,1,2,4}
exp -> {1,4} -> assemble                                   -> view -> {1,1,2,4} -> exp -> {1,1,2,4}
exp -> {1,4} -> assemble
exp -> {1,4} -> assemble
3) splitreshape
exp -> {1,4} -> assemble -> {1,4} -> reshape -> {1,1,2,2} -> assemble 
exp -> {1,4} -> assemble -> {1,4} -> reshape -> {1,1,2,2} -> assemble -> {1,1,2,4} -> view -> {1,1,2,4} -> exp -> {1,1,2,4}
exp -> {1,4} -> assemble -> {1,4} -> reshape -> {1,1,2,2} -> assemble -> {1,1,2,4} -> view -> {1,1,2,4} -> exp -> {1,1,2,4}
exp -> {1,4} -> assemble -> {1,4} -> reshape -> {1,1,2,2} -> assemble
*/
TEST_F(TestSplitReshapePass, SplitReshapeSTest4) {
    //Define the shape of the Tensors
    std::vector<int> origShape = {kNumOne, kExpFour};
    std::vector<int> reshapeShape = {kNumOne, kNumOne, kNumTwo, kNumEight};
    std::vector<int> tiledShape1 = {kNumOne, kNumFour};
    std::vector<int> tiledShape2 = {kNumTwo, kNumOne, kNumTwo, kNumFour};
    std::vector<int> tiledreshapeShape = {kNumOne, kNumFour};
    std::vector<int> tiledassembleShape = {kNumOne, kNumOne, kNumOne, kNumFour};
    std::vector<int> tiledviewShape = {kNumOne, kNumOne, kNumTwo, kNumFour};
    
    Tensor input(DT_FP32, origShape, "input");
    Tensor output(DT_FP32, reshapeShape, "output");

    FUNCTION("STCase4") {
        Program::GetInstance().GetTileShape().SetVecTileShapes(tiledShape1);
        Tensor exp = Exp(input);
        Tensor reshape = Reshape(exp, reshapeShape);
        Program::GetInstance().GetTileShape().SetVecTileShapes(tiledShape2);
        output = Exp(reshape);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase4");
    
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("ExpandFunctionStrategy", {
        {   "ExpandFunction",   "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
    });
    EXPECT_EQ(passManager.RunPass(Program::GetInstance(), *func, "ExpandFunctionStrategy"), SUCCESS);
    
    int reshapeOp = 0;
    for (auto &op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            auto reshapeInput = op.GetInputOperand(kSizeZero);
            EXPECT_NE(reshapeInput, nullptr);
            EXPECT_EQ(reshapeInput->shape, origShape);
            EXPECT_EQ(reshapeInput->GetProducers().size(), kSizeFour);
            for (const auto &producer : reshapeInput->GetProducers()) {
                EXPECT_EQ(producer->GetOpcode(), Opcode::OP_ASSEMBLE);
            }
            EXPECT_EQ(op.GetOutputOperandSize(), kSizeOne);
            auto reshapeOutput = op.GetOutputOperand(kSizeZero);
            EXPECT_NE(reshapeOutput, nullptr);
            EXPECT_EQ(reshapeOutput->shape, reshapeShape);
            EXPECT_EQ(reshapeOutput->GetConsumers().size(), kSizeTwo);
            for (const auto &consumer : reshapeOutput->GetConsumers()) {
                EXPECT_EQ(consumer->GetOpcode(), Opcode::OP_VIEW);
                EXPECT_EQ(consumer->GetOutputOperandSize(), kSizeOne);
                EXPECT_EQ(consumer->GetOutputOperand(kSizeZero)->shape, tiledviewShape);
            }
            reshapeOp++;
        }
    }

    passManager.RegisterStrategy("SplitReshapeTestStrategy", {
        {   "SplitReshape",   "SplitReshape",  PassType::TYPE_TILE_GRAPH},
    });
    EXPECT_EQ(passManager.RunPass(Program::GetInstance(), *func, "SplitReshapeTestStrategy"), SUCCESS);
    
    reshapeOp = 0;
    for (auto &op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            auto reshapeInput = op.GetInputOperand(kSizeZero);
            EXPECT_NE(reshapeInput, nullptr);
            EXPECT_EQ(reshapeInput->shape, tiledreshapeShape);
            EXPECT_EQ(reshapeInput->GetProducers().size(), kSizeOne);
            for (const auto &producer : reshapeInput->GetProducers()) {
                EXPECT_EQ(producer->GetOpcode(), Opcode::OP_ASSEMBLE);
                EXPECT_EQ(producer->GetInputOperandSize(), kSizeOne);
                EXPECT_EQ(producer->GetInputOperand(kSizeZero)->shape, tiledreshapeShape);
            }
            EXPECT_EQ(op.GetOutputOperandSize(), kSizeOne);
            auto reshapeOutput = op.GetOutputOperand(kSizeZero);
            EXPECT_NE(reshapeOutput, nullptr);
            EXPECT_EQ(reshapeOutput->shape, tiledassembleShape);
            EXPECT_EQ(reshapeOutput->GetConsumers().size(), kSizeOne);
            for (const auto &consumer : reshapeOutput->GetConsumers()) {
                EXPECT_EQ(consumer->GetOpcode(), Opcode::OP_ASSEMBLE);
                EXPECT_EQ(consumer->GetOutputOperandSize(), kSizeOne);
                auto viewInput = consumer->GetOutputOperand(kSizeZero);
                EXPECT_EQ(viewInput->shape, tiledviewShape);
                EXPECT_EQ(viewInput->GetProducers().size(), kSizeTwo);
            }
            reshapeOp++;
        }
    }
}

/*
splitreshape pass不起作用的场景
一对多场景(使用expandfunction作为前序pass)
assemble的tile输入无法被映射为一个完整inputView的tile
1) 用例设置：
{1,1,2,8} -> exp -> {1,1,2,8} -> reshape -> {1,16} -> exp -> {1,16}
tileshape1 = {2,1,2,4}
tileshape2 = {1,4}
2) expandfunction：
exp -> {1,1,2,4} -> assemble -> reshape -> {1,16} -> view -> {1,4} -> exp -> {1,4}
exp -> {1,1,2,4} -> assemble                      -> view -> {1,4} -> exp -> {1,4}
                                                  -> view -> {1,4} -> exp -> {1,4}
                                                  -> view -> {1,4} -> exp -> {1,4}
*/
TEST_F(TestSplitReshapePass, SplitReshapeSTest5) {
    //Define the shape of the Tensors
    std::vector<int> origShape = {kNumOne, kNumOne, kNumTwo, kNumEight};
    std::vector<int> reshapeShape = {kNumOne, kExpFour};
    std::vector<int> tiledShape1 = {kNumTwo, kNumOne, kNumTwo, kNumFour};
    std::vector<int> tiledShape2 = {kNumOne, kNumFour};
    
    Tensor input(DT_FP32, origShape, "input");
    Tensor output(DT_FP32, reshapeShape, "output");

    FUNCTION("STCase5") {
        Program::GetInstance().GetTileShape().SetVecTileShapes(tiledShape1);
        Tensor exp = Exp(input);
        Tensor reshape = Reshape(exp, reshapeShape);
        Program::GetInstance().GetTileShape().SetVecTileShapes(tiledShape2);
        output = Exp(reshape);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase5");
    
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("ExpandFunctionStrategy", {
        {   "ExpandFunction",   "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
    });
    EXPECT_EQ(passManager.RunPass(Program::GetInstance(), *func, "ExpandFunctionStrategy"), SUCCESS);
    
    int reshapeOp = 0;
    int OpNum = 0;
    for (auto &op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            reshapeOp++;
        }
        OpNum++;
    }
    EXPECT_EQ(reshapeOp, kNumOne);

    passManager.RegisterStrategy("SplitReshapeTestStrategy", {
        {   "SplitReshape",   "SplitReshape",  PassType::TYPE_TILE_GRAPH},
    });
    EXPECT_EQ(passManager.RunPass(Program::GetInstance(), *func, "SplitReshapeTestStrategy"), SUCCESS);
    
    reshapeOp = 0;
    int AfterOpNum = 0;
    for (auto &op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            reshapeOp++;
        }
        AfterOpNum++;
    }
    EXPECT_EQ(reshapeOp, kNumOne);
    EXPECT_EQ(AfterOpNum, OpNum);
}

/*
splitreshape pass不起作用的场景
使用expandfunction作为前序pass
无法计算reshape前后的加细
1) 用例设置：
{64,6} -> exp -> {64,6} -> reshape -> {96,4} -> exp -> {96,4}
tileshape = {32,2}
2) expandfunction：
exp -> {32,2} -> assemble -> {64,6} -> reshape -> {96,4} -> view -> {32,2} -> exp -> {32,2}
exp -> {32,2} -> assemble                                -> view -> {32,2} -> exp -> {32,2}
exp -> {32,2} -> assemble                                -> view -> {32,2} -> exp -> {32,2}
exp -> {32,2} -> assemble                                -> view -> {32,2} -> exp -> {32,2}                                                  
exp -> {32,2} -> assemble                                -> view -> {32,2} -> exp -> {32,2}
exp -> {32,2} -> assemble                                -> view -> {32,2} -> exp -> {32,2}
*/
TEST_F(TestSplitReshapePass, SplitReshapeSTest6) {
    //Define the shape of the Tensors
    std::vector<int> origShape = {kExpSix, kNumSix};
    std::vector<int> reshapeShape = {kNumNineSix, kNumFour};
    std::vector<int> tiledShape = {kExpFive, kNumTwo};
    
    Program::GetInstance().GetTileShape().SetVecTileShapes(tiledShape);
    Tensor input(DT_FP32, origShape, "input");
    Tensor output(DT_FP32, reshapeShape, "output");

    FUNCTION("STCase6") {
        Tensor exp = Exp(input);
        Tensor reshape = Reshape(exp, reshapeShape);
        output = Exp(reshape);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase6");
    
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("ExpandFunctionStrategy", {
        {   "ExpandFunction",   "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
    });
    EXPECT_EQ(passManager.RunPass(Program::GetInstance(), *func, "ExpandFunctionStrategy"), SUCCESS);
    
    int reshapeOp = 0;
    for (auto &op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            reshapeOp++;
        }
    }
    EXPECT_EQ(reshapeOp, kNumOne);

    passManager.RegisterStrategy("SplitReshapeTestStrategy", {
        {   "SplitReshape",   "SplitReshape",  PassType::TYPE_TILE_GRAPH},
    });
    EXPECT_EQ(passManager.RunPass(Program::GetInstance(), *func, "SplitReshapeTestStrategy"), SUCCESS);
    
    reshapeOp = 0;
    for (auto &op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            reshapeOp++;
        }
    }
    EXPECT_EQ(reshapeOp, kNumOne);
}

/*
splitreshape pass不起作用的场景
多对多的场景，前后的tile即非cover也非covered
1) 用例设置：
{8,8} -> exp -> {8,8} -> reshape -> {16,4} -> exp -> {16,4}
tileshape1 = {2,4}
tileshape1 = {4,2}
2) expandfunction：
exp -> {2,4} -> assemble -> {8,8} -> reshape -> {16,4} -> view -> {4,2} -> exp -> {4,2}
exp -> {2,4} -> assemble                               -> view -> {4,2} -> exp -> {4,2}
exp -> {2,4} -> assemble                               -> view -> {4,2} -> exp -> {4,2}
exp -> {2,4} -> assemble                               -> view -> {4,2} -> exp -> {4,2}                                                  
exp -> {2,4} -> assemble                               -> view -> {4,2} -> exp -> {4,2}
exp -> {2,4} -> assemble                               -> view -> {4,2} -> exp -> {4,2}
*/
TEST_F(TestSplitReshapePass, SplitReshapeSTest7) {
    //Define the shape of the Tensors
    std::vector<int> origShape = {kNumEight, kNumEight};
    std::vector<int> reshapeShape = {kExpFour, kNumFour};
    std::vector<int> tiledShape1 = {kNumTwo, kNumFour};
    std::vector<int> tiledShape2 = {kNumFour, kNumTwo};
    
    Tensor input(DT_FP32, origShape, "input");
    Tensor output(DT_FP32, reshapeShape, "output");

    FUNCTION("STCase7") {
        Program::GetInstance().GetTileShape().SetVecTileShapes(tiledShape1);
        Tensor exp = Exp(input);
        Tensor reshape = Reshape(exp, reshapeShape);
        Program::GetInstance().GetTileShape().SetVecTileShapes(tiledShape2);
        output = Exp(reshape);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase7");
    
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("ExpandFunctionStrategy", {
        {   "ExpandFunction",   "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
    });
    EXPECT_EQ(passManager.RunPass(Program::GetInstance(), *func, "ExpandFunctionStrategy"), SUCCESS);
    
    int reshapeOp = 0;
    for (auto &op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            reshapeOp++;
        }
    }
    EXPECT_EQ(reshapeOp, kNumOne);

    passManager.RegisterStrategy("SplitReshapeTestStrategy", {
        {   "SplitReshape",   "SplitReshape",  PassType::TYPE_TILE_GRAPH},
    });
    EXPECT_EQ(passManager.RunPass(Program::GetInstance(), *func, "SplitReshapeTestStrategy"), SUCCESS);
    
    reshapeOp = 0;
    for (auto &op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            reshapeOp++;
        }
    }
    EXPECT_EQ(reshapeOp, kNumOne);
}
}
}