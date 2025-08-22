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
static const uint32_t kNumTwelve = 12u;
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

TEST_F(TestSplitReshapePass, TestInit) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    
    SplitReshape pass;
    auto status = pass.Init();
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.copyOutSources.size(), kSizeZero);
    EXPECT_EQ(pass.reshapeSources.size(), kSizeZero);
    EXPECT_EQ(pass.mapOffset.size(), kSizeZero);
    EXPECT_EQ(pass.assembles.size(), kSizeZero);
    EXPECT_EQ(pass.reshapes.size(), kSizeZero);
    EXPECT_EQ(pass.redundantViewops.size(), kSizeZero);
    EXPECT_EQ(pass.reshapeRawOutputs.size(), kSizeZero);
}

TEST_F(TestSplitReshapePass, TestCollectCopyOut) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
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

TEST_F(TestSplitReshapePass, TestDynCollectCopyOut) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int> shape = {kNumTwo, kNumOne, kNumEight};
    std::vector<int> offset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int> offset2 = {kNumOne, kNumZero, kNumZero};
    std::vector<SymbolicScalar> dynOffset = {SymbolicScalar("a"), kNumZero, SymbolicScalar("b")};
    std::vector<int> shape1 = {kNumOne, kNumOne, kNumEight};
    std::vector<int> shape2 = {kNumTwo, kNumOne, kNumEight};
    std::vector<int> shape3 = {kNumTwo, kNumEight};
    
    std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input1 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor, offset1, shape1);
    auto input2 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor, offset2, shape1);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    
    auto &assemble_op1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input1}, {ubTensor});
    auto assemble_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1, dynOffset);
    assemble_op1.SetOpAttribute(assemble_Attr1);
    
    auto &assemble_op2 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input2}, {ubTensor});
    auto assemble_Attr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2, dynOffset);
    assemble_op2.SetOpAttribute(assemble_Attr2);

    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor}, {output});
    
    SplitReshape pass;
    auto status = pass.CollectCopyOut(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.dynMapOffset.size(), kSizeTwo);
    EXPECT_EQ(pass.dynMapOffset.count(input1->magic), kNumOne);
    EXPECT_EQ(pass.dynMapOffset[input1->magic].count(ubTensor->magic), kNumOne);
    EXPECT_EQ(pass.dynMapOffset[input1->magic][ubTensor->magic].size(), dynOffset.size());
    for (size_t i = 0; i < dynOffset.size(); ++i) {
        EXPECT_EQ(pass.dynMapOffset[input1->magic][ubTensor->magic][i].Dump(), dynOffset[i].Dump());
    }
    EXPECT_EQ(pass.dynMapOffset.count(input2->magic), kNumOne);
    EXPECT_EQ(pass.dynMapOffset[input2->magic].count(ubTensor->magic), kNumOne);
    EXPECT_EQ(pass.dynMapOffset[input2->magic][ubTensor->magic].size(), dynOffset.size());
    for (size_t i = 0; i < dynOffset.size(); ++i) {
        EXPECT_EQ(pass.dynMapOffset[input2->magic][ubTensor->magic][i].Dump(), dynOffset[i].Dump());
    }
}

TEST_F(TestSplitReshapePass, TestCheckSplit) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
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

TEST_F(TestSplitReshapePass, TestCheckDynStatus) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int> shape = {kNumTwo, kNumOne, kNumEight};
    std::vector<int> offset = {kNumZero, kNumZero, kNumZero};
    std::vector<int> shape1 = {kNumOne, kNumOne, kNumEight};
    std::vector<int> shape2 = {kNumTwo, kNumOne, kNumEight};
    std::vector<int> shape3 = {kNumTwo, kNumEight};
    
    std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    
    auto case1Input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor, offset, shape1);
    auto case1UbTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto case1Output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    auto &assemble_op1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {case1Input}, {case1UbTensor});
    auto assemble_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset);
    assemble_op1.SetOpAttribute(assemble_Attr1);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {case1UbTensor}, {case1Output});

    auto case2Input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor, offset, shape1);
    auto case2UbTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto case2Output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    std::vector<SymbolicScalar> case2DynUbShape = {SymbolicScalar("a1"), SymbolicScalar(kNumOne), SymbolicScalar(kNumEight)};
    std::vector<SymbolicScalar> case2DynUbOffset = {SymbolicScalar("b1"), SymbolicScalar(kNumOne), SymbolicScalar(kNumEight)};
    std::vector<SymbolicScalar> case2DynOutputShape = {SymbolicScalar("a2"), SymbolicScalar(kNumEight)};
    std::vector<SymbolicScalar> case2DynOutputOffset = {SymbolicScalar("b2"), SymbolicScalar(kNumEight)};
    TensorOffset UbOffset({}, case2DynUbOffset);
    TensorOffset OutputOffset({}, case2DynOutputOffset);
    case2UbTensor->UpdateDynValidShape(case2DynUbShape);
    case2UbTensor->UpdateOffset(UbOffset);
    case2Output->UpdateDynValidShape(case2DynOutputShape);
    case2Output->UpdateOffset(OutputOffset);
    auto &assemble_op2 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {case2Input}, {case2UbTensor});
    auto assemble_Attr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset);
    assemble_op2.SetOpAttribute(assemble_Attr2);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {case2UbTensor}, {case2Output});

    auto case3Input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor, offset, shape1);
    auto case3UbTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto case3Output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    std::vector<SymbolicScalar> case3DynUbShape = {SymbolicScalar("s0"), SymbolicScalar(kNumOne), SymbolicScalar(kNumEight)};
    std::vector<SymbolicScalar> case3DynOutputShape = {SymbolicScalar("s0"), SymbolicScalar(kNumEight)};
    case3UbTensor->UpdateDynValidShape(case3DynUbShape);
    case3Output->UpdateDynValidShape(case3DynOutputShape);
    auto &assemble_op3 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {case3Input}, {case3UbTensor});
    auto assemble_Attr3 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset);
    assemble_op3.SetOpAttribute(assemble_Attr3);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {case3UbTensor}, {case3Output});

    SplitReshape pass;
    auto status = pass.CollectCopyOut(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(pass.CheckDynStatus(case1UbTensor, case1Output), true);
    EXPECT_EQ(pass.CheckDynStatus(case2UbTensor, case2Output), true);
    EXPECT_EQ(pass.CheckDynStatus(case3UbTensor, case3Output), true);
}

TEST_F(TestSplitReshapePass, TestShapeAlign) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
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

TEST_F(TestSplitReshapePass, TestRawToAlign) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
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

    rawShape = {kExpFive, kNumFour, kNumNineSix};
    alignedShape = {kExpFive, kNumFour, kNumFour, kExpFive};
    tileOffset = {kNumOne, kNumOne, kExpSeven};
    tileShape = {kNumOne, kNumTwo, kExpSix};
    shapePara = {rawShape, alignedShape, tileOffset, tileShape};
    status = pass.RawToAlign(shapePara, newOffset, newShape);
    EXPECT_EQ(status, WARNING);
}

TEST_F(TestSplitReshapePass, TestAlignToRaw) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
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

TEST_F(TestSplitReshapePass, TestDynRawToAlign) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    SplitReshape pass;
    Status status;
    std::vector<int32_t> rawShape;
    std::vector<int32_t> alignedShape;
    std::vector<SymbolicScalar> dynOffset;
    std::vector<SymbolicScalar> dynShape;
    std::vector<SymbolicScalar> expectOffset;
    std::vector<SymbolicScalar> expectShape;
    std::vector<SymbolicScalar> newOffset;
    std::vector<SymbolicScalar> newShape;

    DynReshapeTilePara shapePara;

    rawShape = {kNumTwo, kNumEight};
    alignedShape = {kNumTwo, kNumTwo, kNumTwo, kNumTwo};
    dynOffset = {SymbolicScalar("b"), kNumZero};
    dynShape = {SymbolicScalar("a"), kNumTwo};
    shapePara = {rawShape, alignedShape, dynOffset, dynShape};
    status = pass.DynRawToAlign(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectOffset = {SymbolicScalar("b"), SymbolicScalar(kNumZero), SymbolicScalar(kNumZero), SymbolicScalar(kNumZero)};
    expectShape = {SymbolicScalar("a"), SymbolicScalar(kNumOne), SymbolicScalar(kNumOne), SymbolicScalar(kNumTwo)};
    EXPECT_EQ(newShape.size(), expectShape.size());
    for (size_t i = 0; i < newShape.size(); ++i) {
        EXPECT_EQ(newShape[i].Dump(), expectShape[i].Dump());
    }
    EXPECT_EQ(newOffset.size(), expectOffset.size());
    for (size_t i = 0; i < newOffset.size(); ++i) {
        EXPECT_EQ(newOffset[i].Dump(), expectOffset[i].Dump());
    }

    rawShape = {kNumEight};
    alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    dynOffset = {SymbolicScalar("b")};
    dynShape = {SymbolicScalar("a")};
    shapePara = {rawShape, alignedShape, dynOffset, dynShape};
    status = pass.DynRawToAlign(shapePara, newOffset, newShape);
    EXPECT_EQ(status, WARNING);
}

TEST_F(TestSplitReshapePass, TestDynAlignToRaw) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    SplitReshape pass;
    Status status;
    std::vector<int32_t> rawShape;
    std::vector<int32_t> alignedShape;
    std::vector<SymbolicScalar> dynOffset;
    std::vector<SymbolicScalar> dynShape;
    std::vector<SymbolicScalar> expectOffset;
    std::vector<SymbolicScalar> expectShape;
    std::vector<SymbolicScalar> newOffset;
    std::vector<SymbolicScalar> newShape;

    DynReshapeTilePara shapePara;

    alignedShape = {kNumTwo, kNumTwo, kNumTwo, kNumTwo};
    rawShape = {kNumTwo, kNumEight};
    dynShape = {SymbolicScalar("a"), kNumOne, kNumTwo, kNumTwo};
    dynOffset = {SymbolicScalar("b"), kNumOne, kNumOne, kNumZero};
    shapePara = {alignedShape, rawShape, dynOffset, dynShape};
    status = pass.DynAlignToRaw(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectOffset = {SymbolicScalar("b") * 1, SymbolicScalar(kNumSix)};
    expectShape = {SymbolicScalar("a") * 1, SymbolicScalar(kNumFour)};
    EXPECT_EQ(newShape.size(), expectShape.size());
    for (size_t i = 0; i < newShape.size(); ++i) {
        EXPECT_EQ(newShape[i].Dump(), expectShape[i].Dump());
    }
    EXPECT_EQ(newOffset.size(), expectOffset.size());
    for (size_t i = 0; i < newOffset.size(); ++i) {
        EXPECT_EQ(newOffset[i].Dump(), expectOffset[i].Dump());
    }
    
    alignedShape = {kNumTwo, kNumTwo, kNumTwo, kNumTwo};
    rawShape = {kNumTwo, kNumEight};
    dynShape = {kNumOne, SymbolicScalar("a"), kNumTwo, kNumTwo};
    dynOffset = {kNumOne, SymbolicScalar("b"), kNumZero, kNumZero};
    shapePara = {alignedShape, rawShape, dynOffset, dynShape};
    status = pass.DynAlignToRaw(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectShape = {SymbolicScalar(kNumOne), SymbolicScalar("a") * 4};
    expectOffset = {SymbolicScalar(kNumOne), SymbolicScalar("b") * 4};
    EXPECT_EQ(newShape.size(), expectShape.size());
    for (size_t i = 0; i < newShape.size(); ++i) {
        EXPECT_EQ(newShape[i].Dump(), expectShape[i].Dump());
    }
    EXPECT_EQ(newOffset.size(), expectOffset.size());
    for (size_t i = 0; i < newOffset.size(); ++i) {
        EXPECT_EQ(newOffset[i].Dump(), expectOffset[i].Dump());
    }
    
    alignedShape = {kNumTwo, kNumTwo, kNumTwo, kNumTwo};
    rawShape = {kNumTwo, kNumEight};
    dynShape = {kNumOne, kNumOne, SymbolicScalar("a"), kNumTwo};
    dynOffset = {kNumOne, kNumOne, SymbolicScalar("b"), kNumZero};
    shapePara = {alignedShape, rawShape, dynOffset, dynShape};
    status = pass.DynAlignToRaw(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectShape = {SymbolicScalar(kNumOne), SymbolicScalar("a") * 2};
    expectOffset = {SymbolicScalar(kNumOne), SymbolicScalar("b") * 2 + 4};
    EXPECT_EQ(newShape.size(), expectShape.size());
    for (size_t i = 0; i < newShape.size(); ++i) {
        EXPECT_EQ(newShape[i].Dump(), expectShape[i].Dump());
    }
    EXPECT_EQ(newOffset.size(), expectOffset.size());
    for (size_t i = 0; i < newOffset.size(); ++i) {
        EXPECT_EQ(newOffset[i].Dump(), expectOffset[i].Dump());
    }
    
    alignedShape = {kNumTwo, kNumTwo, kNumTwo, kNumTwo};
    rawShape = {kNumTwo, kNumEight};
    dynShape = {kNumOne, kNumOne, kNumOne, SymbolicScalar("a")};
    dynOffset = {kNumOne, kNumOne, kNumOne, SymbolicScalar("b")};
    shapePara = {alignedShape, rawShape, dynOffset, dynShape};
    status = pass.DynAlignToRaw(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectShape = {SymbolicScalar(kNumOne), SymbolicScalar("a") * 1};
    expectOffset = {SymbolicScalar(kNumOne), SymbolicScalar("b") * 1 + 6};
    EXPECT_EQ(newShape.size(), expectShape.size());
    for (size_t i = 0; i < newShape.size(); ++i) {
        EXPECT_EQ(newShape[i].Dump(), expectShape[i].Dump());
    }
    EXPECT_EQ(newOffset.size(), expectOffset.size());
    for (size_t i = 0; i < newOffset.size(); ++i) {
        EXPECT_EQ(newOffset[i].Dump(), expectOffset[i].Dump());
    }
    
    alignedShape = {kNumTwo, kNumTwo, kNumTwo, kNumTwo};
    rawShape = {kNumTwo, kNumEight};
    dynShape = {kNumOne, kNumOne, SymbolicScalar("a"), kNumTwo};
    dynOffset = {kNumOne, kNumOne, SymbolicScalar("b"), kNumOne};
    shapePara = {alignedShape, rawShape, dynOffset, dynShape};
    status = pass.DynAlignToRaw(shapePara, newOffset, newShape);
    EXPECT_EQ(status, FAILED);
    
    alignedShape = {kNumTwo, kNumTwo, kNumTwo, kNumTwo};
    rawShape = {kNumTwo, kNumEight};
    dynShape = {kNumOne, kNumTwo, SymbolicScalar("a"), kNumTwo};
    dynOffset = {kNumOne, kNumOne, SymbolicScalar("b"), kNumZero};
    shapePara = {alignedShape, rawShape, dynOffset, dynShape};
    status = pass.DynAlignToRaw(shapePara, newOffset, newShape);
    EXPECT_EQ(status, FAILED);
    
    alignedShape = {kNumTwo, kNumTwo, kNumTwo, kNumTwo};
    rawShape = {kNumTwo, kNumEight};
    dynShape = {kNumOne, kNumOne, SymbolicScalar("a"), kNumOne};
    dynOffset = {kNumOne, kNumOne, SymbolicScalar("b"), kNumZero};
    shapePara = {alignedShape, rawShape, dynOffset, dynShape};
    status = pass.DynAlignToRaw(shapePara, newOffset, newShape);
    EXPECT_EQ(status, FAILED);
}

TEST_F(TestSplitReshapePass, TestAlignToRawSpecialCase) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
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
TEST_F(TestSplitReshapePass, TestObtainCopyOutTileBeCovered) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
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
    copyOutTilePara copyOutTile = {ubTensor, output, newOutput, alignedShape};
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
TEST_F(TestSplitReshapePass, TestObtainCopyOutTilePerfectlyMatched) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
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
    copyOutTilePara copyOutTile = {ubTensor, output, newOutput, alignedShape};
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.ObtainCopyOutTile(*currFunctionPtr, copyOutTile, overlaps, newOverlaps), SUCCESS);
    EXPECT_EQ(overlaps.size(), kSizeOne);
    EXPECT_NE(std::find(overlaps.begin(), overlaps.end(), input), overlaps.end());
    EXPECT_EQ(newOverlaps.size(), kSizeOne);
    EXPECT_EQ(newOverlaps[0]->GetOffset(), newOutputTileOffset);
    EXPECT_EQ(newOverlaps[0]->GetShape(), newOutputTileShape);
}


/*
多对一场景
rawShape = {2, 2, 2}
{2, 2, 2} -> assemble -> {2, 2, 2} -> reshape -> {4, 2}
*/
TEST_F(TestSplitReshapePass, TestDynObtainCopyOutTile) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int> shape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> offset = {kNumZero, kNumZero, kNumZero};
    std::vector<SymbolicScalar> dynShape = {kNumTwo, kNumTwo, SymbolicScalar("a")};
    std::vector<SymbolicScalar> dynOffset = {kNumZero, kNumZero, SymbolicScalar("b")};
    std::vector<int> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> shape2 = {kNumFour, kNumTwo};
    
    std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor, offset, shape1, dynShape);
    input->UpdateOffset(TensorOffset(offset, dynOffset));
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);

    auto &assemble_op = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input}, {ubTensor});
    auto assemble_Attr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset, dynOffset);
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
    copyOutTilePara copyOutTile = {ubTensor, output, newOutput, alignedShape};
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.ObtainCopyOutTile(*currFunctionPtr, copyOutTile, overlaps, newOverlaps), SUCCESS);
    EXPECT_EQ(overlaps.size(), kSizeOne);
    EXPECT_NE(std::find(overlaps.begin(), overlaps.end(), input), overlaps.end());
    EXPECT_EQ(newOverlaps.size(), kSizeOne);
    std::vector<SymbolicScalar> expectedDynShape = {kNumTwo, kNumTwo, SymbolicScalar("a")};
    std::vector<SymbolicScalar> expectedDynOffset = {kNumZero, kNumZero, SymbolicScalar("b")};
    for (const auto &copyOutSource : newOverlaps) {
        EXPECT_EQ(copyOutSource->GetDynValidShape().size(), kNumThree);
        EXPECT_EQ(copyOutSource->GetDynOffset().size(), kNumThree);
        for (size_t i = 0; i < expectedDynShape.size(); ++i) {
            EXPECT_EQ(copyOutSource->GetDynValidShape()[i].Dump(), expectedDynShape[i].Dump());
        }
        for (size_t i = 0; i < expectedDynOffset.size(); ++i) {
            EXPECT_EQ(copyOutSource->GetDynOffset()[i].Dump(), expectedDynOffset[i].Dump());
        }
    }
}

/*
验证一对一场景下ub数据的处理
rawShape = {2, 2, 2}
{2, 2, 2}(ub) -> assemble -> {2, 2, 2} -> reshape -> {4, 2} -> view -> {4,2}(ub) -> OP
{2, 2, 2}(ub) -> reshape(一个ReshapeOp成员) -> {4, 2}(ub) -> OP
*/
TEST_F(TestSplitReshapePass, TestUpdateForPerfectlyMatchForUB) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
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
    EXPECT_EQ(pass.redundantViewops.size(), kSizeZero);
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.UpdateForPerfectlyMatch(*currFunctionPtr, view_op, para), SUCCESS);
    EXPECT_EQ(pass.redundantViewops.size(), kSizeOne);
    EXPECT_NE(pass.redundantViewops.find(&view_op), pass.redundantViewops.end());
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
TEST_F(TestSplitReshapePass, TestUpdateForPerfectlyMatchForDDR) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
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
TEST_F(TestSplitReshapePass, TestUpdateForPerfectlyMatchOtherCase) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
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
验证一对一场景下ub动态shape数据的处理
rawShape = {2, 2, 2}
{2, 2, 2}(ub) -> assemble -> {2, 2, 2} -> reshape -> {4, 2} -> view -> {4,2}(ub) -> OP
{2, 2, a}/{0, 0, b}          {2, 2, a}/{0, 0, b}     {4, a}/{0, b}     {4, a}/{0, b}
{2, 2, 2}(ub) -> reshape(一个ReshapeOp成员) -> {4, 2}(ub) -> OP
{2, 2, a}/{0, 0, b}                            {4, a}/{0, b}
*/
TEST_F(TestSplitReshapePass, TestDynUpdateForPerfectlyMatchForUB) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int> shape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> offset = {kNumZero, kNumZero, kNumZero};
    std::vector<SymbolicScalar> dynShape = {kNumTwo, kNumTwo, SymbolicScalar("a")};
    std::vector<SymbolicScalar> dynOffset = {kNumZero, kNumZero, SymbolicScalar("b")};
    std::vector<int> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> shape2 = {kNumFour, kNumTwo};
    
    std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor, offset, shape1, dynShape);
    input->UpdateOffset(TensorOffset(offset, dynOffset));
    input->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    output->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    auto opOutput = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);

    auto &assemble_op = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input}, {ubTensor1});
    auto assemble_Attr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset, dynOffset);
    assemble_op.SetOpAttribute(assemble_Attr);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &view_op = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output});
    auto &post_op = currFunctionPtr->AddOperation(Opcode::OP_NOP, {output}, {opOutput});

    CalcOverlapPara para;
    std::vector<int> newTileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int> newTileShape = {kNumTwo, kNumTwo, kNumTwo};
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input};
    auto newOverlap = std::make_shared<LogicalTensor>(*currFunctionPtr, input->tensor, newTileOffset, newTileShape, dynShape);
    newOverlap->UpdateOffset(TensorOffset(newTileOffset, dynOffset));
    para.newOverlaps = {newOverlap};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output;
    para.inputView = ubTensor2;
    
    SplitReshape pass;
    EXPECT_EQ(pass.redundantViewops.size(), kSizeZero);
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.UpdateForPerfectlyMatch(*currFunctionPtr, view_op, para), SUCCESS);
    EXPECT_EQ(pass.redundantViewops.size(), kSizeOne);
    EXPECT_NE(pass.redundantViewops.find(&view_op), pass.redundantViewops.end());
    auto newReshapeOutput = post_op.GetInputOperand(kSizeZero);
    EXPECT_NE(newReshapeOutput, output);
    EXPECT_EQ(newReshapeOutput->GetDynValidShape().size(), kNumTwo);
    EXPECT_EQ(newReshapeOutput->GetDynOffset().size(), kNumTwo);
    std::vector<SymbolicScalar> expectDynShape = {kNumFour, SymbolicScalar("a") * 1};
    std::vector<SymbolicScalar> expectDynOffset = {kNumZero, SymbolicScalar("b") * 1};
    for (size_t i = 0; i < expectDynShape.size(); ++i) {
        EXPECT_EQ(newReshapeOutput->GetDynValidShape()[i].Dump(), expectDynShape[i].Dump());
        EXPECT_EQ(newReshapeOutput->GetDynOffset()[i].Dump(), expectDynOffset[i].Dump());
    }
    EXPECT_EQ(newReshapeOutput->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(pass.reshapes.size(), kSizeOne);
    auto reshape = pass.reshapes.begin()->second;
    EXPECT_EQ(reshape->input, input);
    EXPECT_EQ(reshape->output, newReshapeOutput);
}

/*
验证一对一场景下动态shape的ddr数据的处理
rawShape = {2, 2, 2}
{2, 2, 2}(ddr) -> assemble -> {2, 2, 2} -> reshape -> {4, 2}(unknown) -> view -> {4,2}(ub) -> OP
{2, 2, a}/{0, 0, b}          {2, 2, a}/{0, 0, b}     {4, a}/{0, b}     {4, a}/{0, b}
{2, 2, 2}(ddr) -> reshape(一个ReshapeOp成员) -> {4, 2}(unknown) -> view -> {4,2}(ub) -> OP
{2, 2, a}/{0, 0, b}                            {4, a}/{0, b}              {4,a}/{0, b}
*/
TEST_F(TestSplitReshapePass, TestDynUpdateForPerfectlyMatchForDDR) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int> shape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> offset = {kNumZero, kNumZero, kNumZero};
    std::vector<SymbolicScalar> dynShape = {kNumTwo, kNumTwo, SymbolicScalar("a")};
    std::vector<SymbolicScalar> dynOffset = {kNumZero, kNumZero, SymbolicScalar("b")};
    std::vector<int> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> shape2 = {kNumFour, kNumTwo};
    std::vector<SymbolicScalar> dynViewOffset = {kNumZero, SymbolicScalar("b") * 1};
    
    std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor, offset, shape1);
    input->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    ubTensor2->UpdateOffset(TensorOffset(ubTensor2->GetOffset(), dynViewOffset));
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
    std::vector<int> newTileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int> newTileShape = {kNumTwo, kNumTwo, kNumTwo};
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input};
    auto newOverlap = std::make_shared<LogicalTensor>(*currFunctionPtr, input->tensor, newTileOffset, newTileShape, dynShape);
    newOverlap->UpdateOffset(TensorOffset(newTileOffset, dynOffset));
    para.newOverlaps = {newOverlap};
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
    EXPECT_EQ(newReshapeOutput->GetDynValidShape().size(), kNumTwo);
    EXPECT_EQ(newReshapeOutput->GetDynOffset().size(), kNumTwo);
    std::vector<SymbolicScalar> expectDynShape = {kNumFour, SymbolicScalar("a") * 1};
    std::vector<SymbolicScalar> expectDynOffset = {kNumZero, SymbolicScalar("b") * 1};
    for (size_t i = 0; i < expectDynShape.size(); ++i) {
        EXPECT_EQ(newReshapeOutput->GetDynValidShape()[i].Dump(), expectDynShape[i].Dump());
        EXPECT_EQ(newReshapeOutput->GetDynOffset()[i].Dump(), expectDynOffset[i].Dump());
    }
    EXPECT_EQ(pass.reshapes.size(), kSizeOne);
    auto reshape = pass.reshapes.begin()->second;
    EXPECT_EQ(reshape->input, input);
    EXPECT_EQ(reshape->output, newReshapeOutput);
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(view_op.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute->GetFromOffset(), view_offset);
    EXPECT_EQ(viewOpAttribute->GetFromDynOffset().size(), kNumTwo);
    for (size_t i = 0; i < dynViewOffset.size(); ++i) {
        EXPECT_EQ(viewOpAttribute->GetFromDynOffset()[i].Dump(), dynViewOffset[i].Dump());
    }
}

/*
验证一对一场景下动态shape的其他数据的处理
rawShape = {2, 2, 2}
{2, 2, 2}(ddr) -> assemble -> {2, 2, 2}(unknown) -> reshape -> {4, 2}(unknown) -> view -> {4,2}(ddr) -> OP
{2, 2, a}/{0, 0, b}           {2, 2, a}/{0, 0, b}              {4, a}/{0, b}              {2,a}/{0,b}
{2, 2, 2}(ddr) -> assemble -> {2, 2, 2}(unknown) -> reshape(一个ReshapeOp成员) -> {4, 2}(unknown) -> view -> {4,2}(ddr) -> OP
{2, 2, a}/{0, 0, b}           {2, 2, a}/{0, 0, b}                                 {4, a}/{0, b}             {4, a}/{0, b}
*/
TEST_F(TestSplitReshapePass, TestDynUpdateForPerfectlyMatchOtherCase) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int> shape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> offset = {kNumZero, kNumZero, kNumZero};
    std::vector<SymbolicScalar> dynShape = {kNumTwo, kNumTwo, SymbolicScalar("a")};
    std::vector<SymbolicScalar> dynOffset = {kNumZero, kNumZero, SymbolicScalar("b")};
    std::vector<int> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> shape2 = {kNumFour, kNumTwo};
    std::vector<SymbolicScalar> dynViewOffset = {kNumZero, SymbolicScalar("b") * 1};
    
    std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor, offset, shape1, dynShape);
    input->UpdateOffset(TensorOffset(offset, dynOffset));
    input->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    ubTensor2->UpdateOffset(TensorOffset(ubTensor2->GetOffset(), dynViewOffset));
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    output->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    
    auto &assemble_op = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input}, {ubTensor1});
    auto assemble_Attr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset, dynOffset);
    assemble_op.SetOpAttribute(assemble_Attr);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &view_op = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output});
    std::vector<int> view_offset = {0, 0};
    auto view_Attr = std::make_shared<ViewOpAttribute>(view_offset);
    view_op.SetOpAttribute(view_Attr);

    CalcOverlapPara para;
    std::vector<int> newTileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int> newTileShape = {kNumTwo, kNumTwo, kNumTwo};
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input};
    auto newOverlap = std::make_shared<LogicalTensor>(*currFunctionPtr, input->tensor, newTileOffset, newTileShape, dynShape);
    newOverlap->UpdateOffset(TensorOffset(newTileOffset, dynOffset));
    para.newOverlaps = {newOverlap};
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
    EXPECT_EQ(newReshapeOutput->GetDynValidShape().size(), kNumTwo);
    EXPECT_EQ(newReshapeOutput->GetDynOffset().size(), kNumTwo);
    std::vector<SymbolicScalar> expectDynShape = {kNumFour, SymbolicScalar("a") * 1};
    std::vector<SymbolicScalar> expectDynOffset = {kNumZero, SymbolicScalar("b") * 1};
    for (size_t i = 0; i < expectDynShape.size(); ++i) {
        EXPECT_EQ(newReshapeOutput->GetDynValidShape()[i].Dump(), expectDynShape[i].Dump());
        EXPECT_EQ(newReshapeOutput->GetDynOffset()[i].Dump(), expectDynOffset[i].Dump());
    }
    EXPECT_EQ(pass.reshapes.size(), kSizeOne);
    auto reshape = pass.reshapes.begin()->second;
    EXPECT_EQ(pass.assembles.size(), kSizeOne);
    auto assemble = pass.assembles.begin();
    auto newReshapeSource = reshape->input;
    EXPECT_EQ(reshape->output, newReshapeOutput);
    EXPECT_EQ(newReshapeSource->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
    EXPECT_EQ(newReshapeSource->GetDynValidShape().size(), kNumThree);
    EXPECT_EQ(newReshapeSource->GetDynOffset().size(), kNumThree);
    std::vector<SymbolicScalar> expectDynSourceShape = {kNumTwo, kNumTwo, SymbolicScalar("a") * 1};
    std::vector<SymbolicScalar> expectDynSourceOffset = {kNumZero, kNumZero, SymbolicScalar("b") * 1};
    for (size_t i = 0; i < expectDynShape.size(); ++i) {
        EXPECT_EQ(newReshapeSource->GetDynValidShape()[i].Dump(), expectDynSourceShape[i].Dump());
        EXPECT_EQ(newReshapeSource->GetDynOffset()[i].Dump(), expectDynSourceOffset[i].Dump());
    }
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(view_op.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute->GetFromOffset(), view_offset);
    EXPECT_EQ(viewOpAttribute->GetFromDynOffset().size(), kNumTwo);
    for (size_t i = 0; i < dynViewOffset.size(); ++i) {
        EXPECT_EQ(viewOpAttribute->GetFromDynOffset()[i].Dump(), dynViewOffset[i].Dump());
    }
    EXPECT_EQ(assemble->from, MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(assemble->toOffset, offset);
    for (size_t i = 0; i < expectDynShape.size(); ++i) {
        EXPECT_EQ(assemble->toDynOffset[i].Dump(), expectDynSourceOffset[i].Dump());
    }
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
TEST_F(TestSplitReshapePass, TestUpdateForBeCoveredForUB) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
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
TEST_F(TestSplitReshapePass, TestUpdateForBeCoveredOtherCase) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
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
验证一对多场景下动态shape ub数据的处理
rawShape = {2, 2, 2}
{2, 2, 2}(ub) -> assemble -> {2, 2, 2} -> reshape -> {2, 4} -> view -> {2, 2}(ub)
                                                            -> view -> {2, 2}(ub) 
{a, 2, 2}/{b, 0, 0}          {a, 2, 2}/{b, 0, 0}     {a, 4}/{b, 0}     {a, 2}/{b, 0}
                                                                       {a, 2}/{b, 0}
{2, 2, 2}(ub) -> reshape -> {2, 4}(ub) -> view -> {2, 2}
                                       -> view -> {2, 2}
{a, 2, 2}/{b, 0, 0}         {a, 4}/{b, 0}         {a, 2}/{b, 0}
                                                  {a, 2}/{b, 0}
*/
TEST_F(TestSplitReshapePass, TestDynUpdateForBeCoveredForUB) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> shape2 = {kNumTwo, kNumFour};
    std::vector<int> shape3 = {kNumTwo, kNumTwo};
    std::vector<int> offset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int> offset2 = {kNumZero, kNumZero};
    std::vector<SymbolicScalar> dynSrcShape = {SymbolicScalar("a"), kNumTwo, kNumTwo};
    std::vector<SymbolicScalar> dynSrcOffset = {SymbolicScalar("b"), kNumZero, kNumZero};
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
    SplitReshape pass;
    std::vector<int> newCopyOutTileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int> newCopyOutTileShape = {kNumTwo, kNumTwo, kNumTwo};
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input};
    auto newOverlap = std::make_shared<LogicalTensor>(*currFunctionPtr, input->tensor, newCopyOutTileOffset, newCopyOutTileShape, dynSrcShape);
    newOverlap->UpdateOffset(TensorOffset(newCopyOutTileOffset, dynSrcOffset));
    para.newOverlaps = {newOverlap};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output1;
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    
    std::vector<int> viewOffset = {kNumZero, kNumZero};
    auto inputView = std::make_shared<LogicalTensor>(*currFunctionPtr, ubTensor2->tensor, viewOffset, shape2);
    para.inputView = inputView;
    para.newInputViewTileShape = {kNumTwo, kNumOne, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumZero, kNumZero};
    para.newInputViewDynShape = {SymbolicScalar("a"), kNumOne, kNumTwo};
    para.newInputViewDynOffset = {SymbolicScalar("b"), kNumZero, kNumZero};
    EXPECT_EQ(pass.UpdateForBeCovered(*currFunctionPtr, view_op1, para), SUCCESS);
    std::vector<int> view2Offset = {kNumZero, kNumTwo};
    para.newInputViewTileShape = {kNumTwo, kNumOne, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumOne, kNumZero};
    para.newInputViewDynShape = {SymbolicScalar("a"), kNumOne, kNumTwo};
    para.newInputViewDynOffset = {SymbolicScalar("b"), kNumOne, kNumZero};
    EXPECT_EQ(pass.UpdateForBeCovered(*currFunctionPtr, view_op2, para), SUCCESS);
    
    std::vector<SymbolicScalar> expectShape = {SymbolicScalar("a") * 1, kNumFour};
    std::vector<SymbolicScalar> expectOffset1 = {SymbolicScalar("b") * 1, kNumZero};
    std::vector<SymbolicScalar> expectOffset2 = {SymbolicScalar("b") * 1, kNumTwo};
    EXPECT_EQ(pass.reshapes.size(), kSizeOne);
    auto newReshape = pass.reshapes.begin()->second;
    EXPECT_EQ(newReshape->input, input);
    EXPECT_NE(newReshape->output, ubTensor2);
    auto reshapeOutput = newReshape->output;
    EXPECT_EQ(reshapeOutput->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    for (size_t i = 0; i < expectOffset1.size(); ++i) {
        EXPECT_EQ(reshapeOutput->GetDynOffset()[i].Dump(), expectOffset1[i].Dump());
    }
    for (size_t i = 0; i < expectShape.size(); ++i) {
        EXPECT_EQ(reshapeOutput->GetDynValidShape()[i].Dump(), expectShape[i].Dump());
    }
    auto viewOpAttribute1 = dynamic_cast<ViewOpAttribute *>(view_op1.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute1->GetFromOffset(), view_offset1);
    for (size_t i = 0; i < expectOffset1.size(); ++i) {
        EXPECT_EQ(viewOpAttribute1->GetFromDynOffset()[i].Dump(), expectOffset1[i].Dump());
    }
    auto viewOpAttribute2 = dynamic_cast<ViewOpAttribute *>(view_op2.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute2->GetFromOffset(), view_offset2);
    for (size_t i = 0; i < expectOffset2.size(); ++i) {
        EXPECT_EQ(viewOpAttribute2->GetFromDynOffset()[i].Dump(), expectOffset2[i].Dump());
    }
}

/*
验证一对多场景下动态shape其他数据的处理
rawShape = {2, 2, 2}
{2, 2, 2}(ddr) -> assemble -> {2, 2, 2} -> reshape -> {2, 4}(unknown) -> view -> {2, 2}(ddr)
                                                                      -> view -> {2, 2}(ddr) 
{a, 2, 2}/{b, 0, 0}           {a, 2, 2}/{b, 0, 0}     {a, 4}/{b, 0}              {a, 2}/{b, 0}
                                                                                 {a, 2}/{b, 0}
{2, 2, 2}(ddr) -> assemble -> {2, 2, 2} -> reshape -> {2, 4}(unknown) -> view -> {2, 2}
                                                                      -> view -> {2, 2}
{a, 2, 2}/{b, 0, 0}           {a, 2, 2}/{b, 0, 0}     {a, 4}/{b, 0}              {a, 2}/{b, 0}
                                                                                 {a, 2}/{b, 0}          
*/
TEST_F(TestSplitReshapePass, TestDynUpdateForBeCoveredOtherCase) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> shape2 = {kNumTwo, kNumFour};
    std::vector<int> shape3 = {kNumTwo, kNumTwo};
    std::vector<int> offset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int> offset2 = {kNumZero, kNumZero};
    std::vector<SymbolicScalar> dynSrcShape = {SymbolicScalar("a"), kNumTwo, kNumTwo};
    std::vector<SymbolicScalar> dynSrcOffset = {SymbolicScalar("b"), kNumZero, kNumZero};
    std::vector<int> view_offset1 = {kNumZero, kNumZero};
    std::vector<int> view_offset2 = {kNumZero, kNumTwo};

    std::shared_ptr<RawTensor> RawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor1, offset1, shape1, dynSrcShape);
    input->UpdateOffset(TensorOffset(offset1, dynSrcOffset));
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
    auto assemble_Attr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1, dynSrcOffset);
    assemble_op.SetOpAttribute(assemble_Attr);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &view_op1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output1});
    auto view_Attr1 = std::make_shared<ViewOpAttribute>(view_offset1);
    view_op1.SetOpAttribute(view_Attr1);
    auto &view_op2 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output2});
    auto view_Attr2 = std::make_shared<ViewOpAttribute>(view_offset2);
    view_op2.SetOpAttribute(view_Attr2);

    CalcOverlapPara para;
    SplitReshape pass;
    std::vector<int> newCopyOutTileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int> newCopyOutTileShape = {kNumTwo, kNumTwo, kNumTwo};
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input};
    auto newOverlap = std::make_shared<LogicalTensor>(*currFunctionPtr, input->tensor, newCopyOutTileOffset, newCopyOutTileShape, dynSrcShape);
    newOverlap->UpdateOffset(TensorOffset(newCopyOutTileOffset, dynSrcOffset));
    para.newOverlaps = {newOverlap};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output1;
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    
    std::vector<int> viewOffset = {kNumZero, kNumZero};
    auto inputView = std::make_shared<LogicalTensor>(*currFunctionPtr, ubTensor2->tensor, viewOffset, shape2);
    para.inputView = inputView;
    para.newInputViewTileShape = {kNumTwo, kNumOne, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumZero, kNumZero};
    para.newInputViewDynShape = {SymbolicScalar("a"), kNumOne, kNumTwo};
    para.newInputViewDynOffset = {SymbolicScalar("b"), kNumZero, kNumZero};
    EXPECT_EQ(pass.UpdateForBeCovered(*currFunctionPtr, view_op1, para), SUCCESS);
    std::vector<int> view2Offset = {kNumZero, kNumTwo};
    para.newInputViewTileShape = {kNumTwo, kNumOne, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumOne, kNumZero};
    para.newInputViewDynShape = {SymbolicScalar("a"), kNumOne, kNumTwo};
    para.newInputViewDynOffset = {SymbolicScalar("b"), kNumOne, kNumZero};
    EXPECT_EQ(pass.UpdateForBeCovered(*currFunctionPtr, view_op2, para), SUCCESS);
    
    std::vector<SymbolicScalar> expectShape = {SymbolicScalar("a") * 1, kNumFour};
    std::vector<SymbolicScalar> expectOffset1 = {SymbolicScalar("b") * 1, kNumZero};
    std::vector<SymbolicScalar> expectOffset2 = {SymbolicScalar("b") * 1, kNumTwo};
    EXPECT_EQ(pass.reshapes.size(), kSizeOne);
    auto newReshape = pass.reshapes.begin()->second;
    auto newReshapeResource = newReshape->input;
    EXPECT_NE(newReshape->output, ubTensor2);
    auto reshapeOutput = newReshape->output;
    EXPECT_EQ(reshapeOutput->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
    for (size_t i = 0; i < expectOffset1.size(); ++i) {
        EXPECT_EQ(reshapeOutput->GetDynOffset()[i].Dump(), expectOffset1[i].Dump());
    }
    for (size_t i = 0; i < expectShape.size(); ++i) {
        EXPECT_EQ(reshapeOutput->GetDynValidShape()[i].Dump(), expectShape[i].Dump());
    }
    EXPECT_EQ(pass.assembles.size(), kSizeOne);
    auto newAssemble = pass.assembles.begin();
    EXPECT_EQ(newAssemble->from, MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(newAssemble->toOffset, offset1);
    for (size_t i = 0; i < dynSrcOffset.size(); ++i) {
        EXPECT_EQ(newAssemble->toDynOffset[i].Dump(), dynSrcOffset[i].Dump());
    }
    EXPECT_EQ(newAssemble->input, input);
    EXPECT_EQ(newAssemble->output, newReshapeResource);
    auto viewOpAttribute1 = dynamic_cast<ViewOpAttribute *>(view_op1.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute1->GetFromOffset(), view_offset1);
    for (size_t i = 0; i < expectOffset1.size(); ++i) {
        EXPECT_EQ(viewOpAttribute1->GetFromDynOffset()[i].Dump(), expectOffset1[i].Dump());
    }
    auto viewOpAttribute2 = dynamic_cast<ViewOpAttribute *>(view_op2.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute2->GetFromOffset(), view_offset2);
    for (size_t i = 0; i < expectOffset2.size(); ++i) {
        EXPECT_EQ(viewOpAttribute2->GetFromDynOffset()[i].Dump(), expectOffset2[i].Dump());
    }
}

/*
验证多对一场景下ub数据的处理
rawShape = {2, 4}
{2, 2}(ub) -> assemble -> {2, 4}(unknown) -> reshape -> {2, 2, 2} -> view -> {2, 2, 2}(ub)
{2, 2}(ub) -> assemble ->
{2, 2}(ub) -> {2, 4}(unknown) -> reshape -> {2, 2, 2}(ub)        
{2, 2}(ub) -> 
*/
TEST_F(TestSplitReshapePass, TestUpdateForPerfectlyMatchWithAllForUB) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
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
    EXPECT_EQ(pass.redundantViewops.size(), kSizeZero);
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.UpdateForPerfectlyMatchWithAll(*currFunctionPtr, view_op, para), SUCCESS);
    EXPECT_EQ(pass.redundantViewops.size(), kSizeOne);
    EXPECT_NE(pass.redundantViewops.find(&view_op), pass.redundantViewops.end());
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
TEST_F(TestSplitReshapePass, TestUpdateForPerfectlyMatchWithAllOtherCase) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
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
    EXPECT_EQ(pass.redundantViewops.size(), kSizeZero);
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
验证多对一场景下动态shape ub数据的处理
rawShape = {2, 4}
{2, 2}(ub) -> assemble -> {2, 4}(unknown) -> reshape -> {2, 2, 2} -> view -> {2, 2, 2}(ub)
{2, 2}(ub) -> assemble ->
{a, 2}/{b, 0}             {a, 4}/{b, 0}                 {a, 2, 2}/{b, 0, 0}  {a, 2, 2}/{b, 0, 0}
{a, 2}/{b, 2}
{2, 2}(ub) -> {2, 4}(unknown) -> reshape -> {2, 2, 2}(ub)        
{2, 2}(ub) -> 
{a, 2}/{b, 0} {a, 4}/{b, 0}                 {a, 2, 2}/{b, 0, 0}
{a, 2}/{b, 2}
*/
TEST_F(TestSplitReshapePass, TestDynUpdateForPerfectlyMatchWithAllForUB) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
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
    SplitReshape pass;
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input1, input2};
    std::vector<int> newInput1TileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int> newInput1TileShape = {kNumTwo, kNumOne, kNumTwo};
    std::vector<SymbolicScalar> newInput1DynOffset = {SymbolicScalar("b"), kNumZero, kNumZero};
    std::vector<SymbolicScalar> newInput1DynShape = {SymbolicScalar("a"), kNumOne, kNumTwo};
    auto newInput1 = std::make_shared<LogicalTensor>(*currFunctionPtr, input1->tensor, newInput1TileOffset, newInput1TileShape, newInput1DynShape);
    newInput1->UpdateOffset(TensorOffset(newInput1TileOffset, newInput1DynOffset));
    std::vector<int> newInput2TileOffset = {kNumZero, kNumOne, kNumZero};
    std::vector<int> newInput2TileShape = {kNumTwo, kNumOne, kNumTwo};
    std::vector<SymbolicScalar> newInput2DynOffset = {SymbolicScalar("b"), kNumOne, kNumZero};
    std::vector<SymbolicScalar> newInput2DynShape = {SymbolicScalar("a"), kNumOne, kNumTwo};
    auto newInput2 = std::make_shared<LogicalTensor>(*currFunctionPtr, input2->tensor, newInput2TileOffset, newInput2TileShape, newInput2DynShape);
    newInput2->UpdateOffset(TensorOffset(newInput2TileOffset, newInput2DynOffset));
    para.newOverlaps = {newInput1, newInput2};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output;
    para.newInputViewTileShape = {kNumTwo, kNumTwo, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumZero, kNumZero};
    para.newInputViewDynShape = {SymbolicScalar("a"), kNumTwo, kNumTwo};
    para.newInputViewDynOffset = {SymbolicScalar("b"), kNumZero, kNumZero};
    std::vector<SymbolicScalar> InputViewDynShape = {SymbolicScalar("a") * 1, kNumTwo, kNumTwo};
    std::vector<SymbolicScalar> InputViewDynOffset = {SymbolicScalar("b") * 1, kNumZero, kNumZero};
    auto inputView = std::make_shared<LogicalTensor>(*currFunctionPtr, ubTensor2->tensor, view_offset, shape3, InputViewDynShape);
    inputView->UpdateOffset(TensorOffset(view_offset, InputViewDynOffset));
    para.inputView = inputView;
    EXPECT_EQ(pass.redundantViewops.size(), kSizeZero);
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.UpdateForPerfectlyMatchWithAll(*currFunctionPtr, view_op, para), SUCCESS);
    EXPECT_EQ(pass.redundantViewops.size(), kSizeOne);
    EXPECT_NE(pass.redundantViewops.find(&view_op), pass.redundantViewops.end());
    auto newReshapeOutput = post_op.GetInputOperand(kSizeZero);
    EXPECT_NE(newReshapeOutput, output);
    EXPECT_EQ(pass.reshapes.size(), kSizeOne);
    auto reshape = pass.reshapes.begin()->second;
    auto newReshapeSource = reshape->input;
    EXPECT_EQ(newReshapeSource->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    std::vector<SymbolicScalar> expectSrcDynShape = {SymbolicScalar("a") * 1, kNumFour};
    std::vector<SymbolicScalar> expectSrcDynOffset = {SymbolicScalar("b") * 1, kNumZero};
    EXPECT_EQ(newReshapeSource->GetDynValidShape().size(), kNumTwo);
    EXPECT_EQ(newReshapeSource->GetDynOffset().size(), kNumTwo);
    for (size_t i = 0; i < expectSrcDynShape.size(); ++ i) {
        EXPECT_EQ(newReshapeSource->GetDynValidShape()[i].Dump(), expectSrcDynShape[i].Dump());
        EXPECT_EQ(newReshapeSource->GetDynOffset()[i].Dump(), expectSrcDynOffset[i].Dump());    
    }
    EXPECT_EQ(reshape->output, newReshapeOutput);
    EXPECT_EQ(newReshapeOutput->GetDynValidShape().size(), kNumThree);
    EXPECT_EQ(newReshapeOutput->GetDynOffset().size(), kNumThree);
    EXPECT_EQ(newReshapeOutput->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    for (size_t i = 0; i < InputViewDynOffset.size(); ++ i) {
        EXPECT_EQ(newReshapeOutput->GetDynValidShape()[i].Dump(), InputViewDynShape[i].Dump());
        EXPECT_EQ(newReshapeOutput->GetDynOffset()[i].Dump(), InputViewDynOffset[i].Dump());    
    }
}

/*
验证多对一场景下动态shape其他数据的处理
rawShape = {2, 4}
{2, 2}(ddr) -> assemble -> {2, 4}(unknown) -> reshape -> {2, 2, 2}(ddr) -> view -> {2, 2, 2}(ddr)
{2, 2}(ddr) -> assemble ->
{a, 2}/{b, 0}              {a, 4}/{b, 0}                 {a, 2, 2}/{b, 0, 0}  {a, 2, 2}/{b, 0, 0}
{a, 2}/{b, 2}
{2, 2}(ddr) -> {2, 4}(unknown) -> reshape -> {2, 2, 2}(ddr) -> view -> {2, 2}          
{2, 2}(ddr) -> 
{a, 2}/{b, 0}  {a, 4}/{b, 0}                 {a, 2, 2}/{b, 0, 0}       {a, 2, 2}/{b, 0, 0}
{a, 2}/{b, 2}
*/
TEST_F(TestSplitReshapePass, TestDynUpdateForPerfectlyMatchWithAllOtherCase) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
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
    SplitReshape pass;
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input1, input2};
    std::vector<int> newInput1TileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int> newInput1TileShape = {kNumTwo, kNumOne, kNumTwo};
    std::vector<SymbolicScalar> newInput1DynOffset = {SymbolicScalar("b"), kNumZero, kNumZero};
    std::vector<SymbolicScalar> newInput1DynShape = {SymbolicScalar("a"), kNumOne, kNumTwo};
    auto newInput1 = std::make_shared<LogicalTensor>(*currFunctionPtr, input1->tensor, newInput1TileOffset, newInput1TileShape, newInput1DynShape);
    newInput1->UpdateOffset(TensorOffset(newInput1TileOffset, newInput1DynOffset));
    std::vector<int> newInput2TileOffset = {kNumZero, kNumOne, kNumZero};
    std::vector<int> newInput2TileShape = {kNumTwo, kNumOne, kNumTwo};
    std::vector<SymbolicScalar> newInput2DynOffset = {SymbolicScalar("b"), kNumOne, kNumZero};
    std::vector<SymbolicScalar> newInput2DynShape = {SymbolicScalar("a"), kNumOne, kNumTwo};
    auto newInput2 = std::make_shared<LogicalTensor>(*currFunctionPtr, input2->tensor, newInput2TileOffset, newInput2TileShape, newInput2DynShape);
    newInput2->UpdateOffset(TensorOffset(newInput2TileOffset, newInput2DynOffset));
    para.newOverlaps = {newInput1, newInput2};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output;
    para.newInputViewTileShape = {kNumTwo, kNumTwo, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumZero, kNumZero};
    para.newInputViewDynShape = {SymbolicScalar("a"), kNumTwo, kNumTwo};
    para.newInputViewDynOffset = {SymbolicScalar("b"), kNumZero, kNumZero};
    std::vector<SymbolicScalar> InputViewDynShape = {SymbolicScalar("a") * 1, kNumTwo, kNumTwo};
    std::vector<SymbolicScalar> InputViewDynOffset = {SymbolicScalar("b") * 1, kNumZero, kNumZero};
    auto inputView = std::make_shared<LogicalTensor>(*currFunctionPtr, ubTensor2->tensor, view_offset, shape3, InputViewDynShape);
    inputView->UpdateOffset(TensorOffset(view_offset, InputViewDynOffset));
    para.inputView = inputView;
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.UpdateForPerfectlyMatchWithAll(*currFunctionPtr, view_op, para), SUCCESS);
    EXPECT_EQ(pass.redundantViewops.size(), kSizeZero);
    EXPECT_EQ(pass.reshapes.size(), kSizeOne);
    auto reshape = pass.reshapes.begin()->second;
    auto newReshapeSource = reshape->input;
    auto newReshapeOutput = view_op.GetInputOperand(kSizeZero);
    EXPECT_EQ(reshape->output, newReshapeOutput);
    EXPECT_EQ(newReshapeSource->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
    std::vector<SymbolicScalar> expectSrcDynShape = {SymbolicScalar("a") * 1, kNumFour};
    std::vector<SymbolicScalar> expectSrcDynOffset = {SymbolicScalar("b") * 1, kNumZero};
    EXPECT_EQ(newReshapeSource->GetDynValidShape().size(), kNumTwo);
    EXPECT_EQ(newReshapeSource->GetDynOffset().size(), kNumTwo);
    for (size_t i = 0; i < expectSrcDynShape.size(); ++ i) {
        EXPECT_EQ(newReshapeSource->GetDynValidShape()[i].Dump(), expectSrcDynShape[i].Dump());
        EXPECT_EQ(newReshapeSource->GetDynOffset()[i].Dump(), expectSrcDynOffset[i].Dump());    
    }
    EXPECT_EQ(newReshapeOutput->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(newReshapeOutput->GetDynValidShape().size(), kNumThree);
    EXPECT_EQ(newReshapeOutput->GetDynOffset().size(), kNumThree);
    for (size_t i = 0; i < InputViewDynOffset.size(); ++ i) {
        EXPECT_EQ(newReshapeOutput->GetDynValidShape()[i].Dump(), InputViewDynShape[i].Dump());
        EXPECT_EQ(newReshapeOutput->GetDynOffset()[i].Dump(), InputViewDynOffset[i].Dump());    
    }
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(view_op.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute->GetFromOffset(), inputView->offset);
    EXPECT_EQ(viewOpAttribute->GetFromDynOffset().size(), kNumThree);
    for (size_t i = 0; i < InputViewDynOffset.size(); ++ i) {
        EXPECT_EQ(viewOpAttribute->GetFromDynOffset()[i].Dump(), InputViewDynOffset[i].Dump());    
    } 
}

/*
验证多对一兜底场景下ub数据的处理
rawShape = {2, 4}
{2, 2}(ub) -> assemble -> {2, 4}(unknown) -> reshape -> {2, 2, 2} -> view -> {2, 2, 2}(ub)
{2, 2}(ub) -> assemble ->
{2, 2}(ub) -> reshape -> {2, 1, 2}(ub) -> assemble -> {2, 2, 2}(ub)        
{2, 2}(ub) -> reshape -> {2, 1, 2}(ub) -> assemble
*/
TEST_F(TestSplitReshapePass, TestUpdateForAssembleAfterReshapeForUB) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
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
    EXPECT_EQ(pass.redundantViewops.size(), kSizeZero);
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.UpdateForAssembleAfterReshape(*currFunctionPtr, view_op, para), SUCCESS);
    EXPECT_EQ(pass.redundantViewops.size(), kSizeOne);
    EXPECT_EQ(pass.reshapes.size(), kSizeTwo);
    EXPECT_EQ(pass.assembles.size(), kSizeTwo);

    DynAssembleOp newAssembleOp1;
    DynAssembleOp newAssembleOp2;
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
TEST_F(TestSplitReshapePass, TestUpdateForAssembleAfterReshapeForDDR) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
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
    EXPECT_EQ(pass.redundantViewops.size(), kSizeZero);
    EXPECT_EQ(pass.reshapes.size(), kSizeTwo);
    EXPECT_EQ(pass.assembles.size(), kSizeTwo);

    DynAssembleOp newAssembleOp1;
    DynAssembleOp newAssembleOp2;
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
TEST_F(TestSplitReshapePass, TestUpdateForAssembleAfterReshapeOtherCase) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
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
    EXPECT_EQ(pass.redundantViewops.size(), kSizeZero);
    EXPECT_EQ(pass.reshapes.size(), kSizeTwo);
    EXPECT_EQ(pass.assembles.size(), kSizeFour);

    DynAssembleOp assembleBeforeReshape1;
    DynAssembleOp assembleAfterReshape1;
    DynAssembleOp assembleBeforeReshape2;
    DynAssembleOp assembleAfterReshape2;
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
验证多对一兜底场景下动态shape ub数据的处理
rawShape = {2, 4}
{2, 2}(ub) -> assemble -> {2, 4}(unknown) -> reshape -> {2, 2, 2} -> view -> {2, 2, 2}(ub)
{2, 2}(ub) -> assemble ->
{a, 2}/{b, 0}             {a, 4}/{b, 0}                 {a, 2, 2}/{b, 0, 0}  {a, 2, 2}/{b, 0, 0}
{a, 2}/{b, 2}
{2, 2}(ub) -> reshape -> {2, 1, 2}(ub) -> assemble -> {2, 2, 2}(ub)        
{2, 2}(ub) -> reshape -> {2, 1, 2}(ub) -> assemble
{a, 2}/{b, 0}            {a, 1, 2}/{b, 0, 0}          {a, 2, 2}/{b, 0, 0}
{a, 2}/{b, 2}            {a, 1, 2}/{b, 1, 0}
*/
TEST_F(TestSplitReshapePass, TestDynUpdateForAssembleAfterReshapeForUB) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int> shape1 = {kNumTwo, kNumFour};
    std::vector<int> shape2 = {kNumTwo, kNumTwo};
    std::vector<int> shape3 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> offset1 = {kNumZero, kNumZero};
    std::vector<int> offset2 = {kNumZero, kNumTwo};
    std::vector<SymbolicScalar> dynShape = {SymbolicScalar("a"), kNumTwo};
    std::vector<SymbolicScalar> dynOffset1 = {SymbolicScalar("b"), kNumZero};
    std::vector<SymbolicScalar> dynOffset2 = {SymbolicScalar("b"), kNumTwo};
    std::vector<int> view_offset = {kNumZero, kNumZero, kNumZero};

    std::shared_ptr<RawTensor> RawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto input1 = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor1, offset1, shape2, dynShape);
    input1->UpdateOffset(TensorOffset(offset1, dynOffset1));
    input1->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    auto input2 = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor1, offset2, shape2, dynShape);
    input2->UpdateOffset(TensorOffset(offset2, dynOffset2));
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
    SplitReshape pass;
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input1, input2};
    std::vector<int> newInput1TileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int> newInput1TileShape = {kNumTwo, kNumOne, kNumTwo};
    std::vector<SymbolicScalar> newInput1DynOffset = {SymbolicScalar("b"), kNumZero, kNumZero};
    std::vector<SymbolicScalar> newInput1DynShape = {SymbolicScalar("a"), kNumOne, kNumTwo};
    auto newInput1 = std::make_shared<LogicalTensor>(*currFunctionPtr, input1->tensor, newInput1TileOffset, newInput1TileShape, newInput1DynShape);
    newInput1->UpdateOffset(TensorOffset(newInput1TileOffset, newInput1DynOffset));
    std::vector<int> newInput2TileOffset = {kNumZero, kNumOne, kNumZero};
    std::vector<int> newInput2TileShape = {kNumTwo, kNumOne, kNumTwo};
    std::vector<SymbolicScalar> newInput2DynOffset = {SymbolicScalar("b"), kNumOne, kNumZero};
    std::vector<SymbolicScalar> newInput2DynShape = {SymbolicScalar("a"), kNumOne, kNumTwo};
    auto newInput2 = std::make_shared<LogicalTensor>(*currFunctionPtr, input2->tensor, newInput2TileOffset, newInput2TileShape, newInput2DynShape);
    newInput2->UpdateOffset(TensorOffset(newInput2TileOffset, newInput2DynOffset));
    para.newOverlaps = {newInput1, newInput2};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output;
    para.newInputViewTileShape = {kNumTwo, kNumOne, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumOne, kNumZero};
    para.newInputViewDynShape = {SymbolicScalar("a"), kNumOne, kNumTwo};
    para.newInputViewDynOffset = {SymbolicScalar("b"), kNumOne, kNumZero};
    std::vector<int> InputViewTileShape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<SymbolicScalar> InputViewDynShape = {SymbolicScalar("a"), kNumTwo, kNumTwo};
    std::vector<SymbolicScalar> InputViewDynOffset = {SymbolicScalar("b"), kNumZero, kNumZero};
    auto inputView = std::make_shared<LogicalTensor>(*currFunctionPtr, ubTensor2->tensor, view_offset, InputViewTileShape);
    inputView->UpdateOffset(TensorOffset(view_offset, InputViewDynOffset));
    para.inputView = inputView;
    EXPECT_EQ(pass.redundantViewops.size(), kSizeZero);
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.UpdateForAssembleAfterReshape(*currFunctionPtr, view_op, para), SUCCESS);
    EXPECT_EQ(pass.redundantViewops.size(), kSizeOne);
    EXPECT_EQ(pass.reshapes.size(), kSizeTwo);
    EXPECT_EQ(pass.assembles.size(), kSizeTwo);

    DynAssembleOp newAssembleOp1;
    DynAssembleOp newAssembleOp2;
    auto newReshapeOp1 = std::make_shared<ReshapeOp>(nullptr, nullptr);
    auto newReshapeOp2 = std::make_shared<ReshapeOp>(nullptr, nullptr);
    int reshapeCnt = 0;
    int assembleCnt = 0;
    std::vector<SymbolicScalar> dynAssembleOffset1 = {SymbolicScalar("b") * 1 - SymbolicScalar("b"), kNumZero, kNumZero};
    std::vector<SymbolicScalar> dynAssembleOffset2 = {SymbolicScalar("b") * 1 - SymbolicScalar("b"), kNumOne, kNumZero};
    std::vector<int> assemble_offset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int> assemble_offset2 = {kNumZero, kNumOne, kNumZero};
    for (const auto &assemble : pass.assembles) {
        EXPECT_EQ(assemble.from, MemoryType::MEM_UB);
        EXPECT_EQ(assemble.output, output);
        if (assemble.toOffset == assemble_offset1) {
            EXPECT_EQ(assemble.toDynOffset.size(), kNumThree);
            for (size_t i = 0; i < dynAssembleOffset1.size(); ++i) {
                EXPECT_EQ(assemble.toDynOffset[i].Dump(), dynAssembleOffset1[i].Dump());
            }
            newAssembleOp1 = assemble;
            assembleCnt += 1;
        } else if (assemble.toOffset == assemble_offset2) {
            EXPECT_EQ(assemble.toDynOffset.size(), kNumThree);
            for (size_t i = 0; i < dynAssembleOffset2.size(); ++i) {
                EXPECT_EQ(assemble.toDynOffset[i].Dump(), dynAssembleOffset2[i].Dump());
            }
            newAssembleOp2 = assemble;
            assembleCnt += 10;
        }
    }
    EXPECT_EQ(assembleCnt, 11);
    auto newReshapeOutput1 = newAssembleOp1.input;
    std::vector<SymbolicScalar> expectDynShape = {SymbolicScalar("a") * 1, kNumOne, kNumTwo};
    std::vector<SymbolicScalar> expectDynOffset1 = {SymbolicScalar("b") * 1, kNumZero, kNumZero};
    std::vector<SymbolicScalar> expectDynOffset2 = {SymbolicScalar("b") * 1, kNumOne, kNumZero};
    EXPECT_EQ(newReshapeOutput1->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(newReshapeOutput1->GetDynValidShape().size(), kNumThree);
    EXPECT_EQ(newReshapeOutput1->GetDynOffset().size(), kNumThree);
    for (size_t i = 0; i < expectDynShape.size(); ++i) {
        EXPECT_EQ(newReshapeOutput1->GetDynValidShape()[i].Dump(), expectDynShape[i].Dump());
        EXPECT_EQ(newReshapeOutput1->GetDynOffset()[i].Dump(), expectDynOffset1[i].Dump());
    }
    auto newReshapeOutput2 = newAssembleOp2.input;
    EXPECT_EQ(newReshapeOutput2->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(newReshapeOutput2->GetDynValidShape().size(), kNumThree);
    EXPECT_EQ(newReshapeOutput2->GetDynOffset().size(), kNumThree);
    for (size_t i = 0; i < expectDynShape.size(); ++i) {
        EXPECT_EQ(newReshapeOutput2->GetDynValidShape()[i].Dump(), expectDynShape[i].Dump());
        EXPECT_EQ(newReshapeOutput2->GetDynOffset()[i].Dump(), expectDynOffset2[i].Dump());
    }
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
{a, 2}/{b, 0}              {a, 4}/{b, 0}                 {a, 2, 2}/{b, 0, 0}           {a, 2, 2}/{b, 0, 0}
{a, 2}/{b, 2}
{2, 2}(ddr) -> reshape -> {2, 1, 2}(unknown) -> assemble -> {2, 2, 2} -> view -> {2, 2, 2}(ub)
{2, 2}(ddr) -> reshape -> {2, 1, 2}(unknown) -> assemble
{a, 2}/{b, 0}             {a, 1, 2}/{b, 0, 0}               {a, 2, 2}/{b, 0, 0}  {a, 2, 2}/{b, 0, 0}
{a, 2}/{b, 2}             {a, 1, 2}/{b, 1, 0}
*/
TEST_F(TestSplitReshapePass, TestDynUpdateForAssembleAfterReshapeForDDR) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int> shape1 = {kNumTwo, kNumFour};
    std::vector<int> shape2 = {kNumTwo, kNumTwo};
    std::vector<int> shape3 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> offset1 = {kNumZero, kNumZero};
    std::vector<int> offset2 = {kNumZero, kNumTwo};
    std::vector<int> offset3 = {kNumZero, kNumZero, kNumZero};
    std::vector<SymbolicScalar> dynShape = {SymbolicScalar("a"), kNumTwo};
    std::vector<SymbolicScalar> dynOffset1 = {SymbolicScalar("b"), kNumZero};
    std::vector<SymbolicScalar> dynOffset2 = {SymbolicScalar("b"), kNumTwo};
    std::vector<SymbolicScalar> dynOutputShape = {SymbolicScalar("a"), kNumTwo, kNumTwo};
    std::vector<SymbolicScalar> dynOutputOffset = {SymbolicScalar("b"), kNumZero, kNumZero};
    std::vector<int> view_offset = {kNumZero, kNumZero, kNumZero};

    std::shared_ptr<RawTensor> RawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto input1 = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor1, offset1, shape2, dynShape);
    input1->UpdateOffset(TensorOffset(offset1, dynOffset1));
    input1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto input2 = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor1, offset2, shape2, dynShape);
    input2->UpdateOffset(TensorOffset(offset1, dynOffset2));
    input2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3, dynOutputShape);
    output->UpdateOffset(TensorOffset(offset3, dynOutputOffset));
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
    SplitReshape pass;
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input1, input2};
    std::vector<int> newInput1TileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int> newInput1TileShape = {kNumTwo, kNumOne, kNumTwo};
    std::vector<SymbolicScalar> newInput1DynOffset = {SymbolicScalar("b"), kNumZero, kNumZero};
    std::vector<SymbolicScalar> newInput1DynShape = {SymbolicScalar("a"), kNumOne, kNumTwo};
    auto newInput1 = std::make_shared<LogicalTensor>(*currFunctionPtr, input1->tensor, newInput1TileOffset, newInput1TileShape, newInput1DynShape);
    newInput1->UpdateOffset(TensorOffset(newInput1TileOffset, newInput1DynOffset));
    std::vector<int> newInput2TileOffset = {kNumZero, kNumOne, kNumZero};
    std::vector<int> newInput2TileShape = {kNumTwo, kNumOne, kNumTwo};
    std::vector<SymbolicScalar> newInput2DynOffset = {SymbolicScalar("b"), kNumOne, kNumZero};
    std::vector<SymbolicScalar> newInput2DynShape = {SymbolicScalar("a"), kNumOne, kNumTwo};
    auto newInput2 = std::make_shared<LogicalTensor>(*currFunctionPtr, input2->tensor, newInput2TileOffset, newInput2TileShape, newInput2DynShape);
    newInput2->UpdateOffset(TensorOffset(newInput2TileOffset, newInput2DynOffset));
    para.newOverlaps = {newInput1, newInput2};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output;
    para.newInputViewTileShape = {kNumTwo, kNumOne, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumOne, kNumZero};
    para.newInputViewDynShape = {SymbolicScalar("a"), kNumOne, kNumTwo};
    para.newInputViewDynOffset = {SymbolicScalar("b"), kNumOne, kNumZero};
    std::vector<int> InputViewTileShape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<SymbolicScalar> InputViewDynShape = {SymbolicScalar("a"), kNumTwo, kNumTwo};
    std::vector<SymbolicScalar> InputViewDynOffset = {SymbolicScalar("b"), kNumZero, kNumZero};
    auto inputView = std::make_shared<LogicalTensor>(*currFunctionPtr, ubTensor2->tensor, view_offset, InputViewTileShape);
    inputView->UpdateOffset(TensorOffset(view_offset, InputViewDynOffset));
    para.inputView = inputView;
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.UpdateForAssembleAfterReshape(*currFunctionPtr, view_op, para), SUCCESS);
    EXPECT_EQ(pass.redundantViewops.size(), kSizeZero);
    EXPECT_EQ(pass.reshapes.size(), kSizeTwo);
    EXPECT_EQ(pass.assembles.size(), kSizeTwo);

    DynAssembleOp newAssembleOp1;
    DynAssembleOp newAssembleOp2;
    auto newReshapeOp1 = std::make_shared<ReshapeOp>(nullptr, nullptr);
    auto newReshapeOp2 = std::make_shared<ReshapeOp>(nullptr, nullptr);
    int reshapeCnt = 0;
    int assembleCnt = 0;
    std::vector<int> assemble_offset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int> assemble_offset2 = {kNumZero, kNumOne, kNumZero};
    std::vector<SymbolicScalar> dynAssembleOffset1 = {SymbolicScalar("b") * 1 - SymbolicScalar("b"), kNumZero, kNumZero};
    std::vector<SymbolicScalar> dynAssembleOffset2 = {SymbolicScalar("b") * 1 - SymbolicScalar("b"), kNumOne, kNumZero};
    for (const auto &assemble : pass.assembles) {
        EXPECT_EQ(assemble.from, MemoryType::MEM_UNKNOWN);
        EXPECT_EQ(assemble.output, view_op.GetInputOperand(kSizeZero));
        if (assemble.toOffset == assemble_offset1) {
            EXPECT_EQ(assemble.toDynOffset.size(), kNumThree);
            for (size_t i = 0; i < dynAssembleOffset1.size(); ++i) {
                EXPECT_EQ(assemble.toDynOffset[i].Dump(), dynAssembleOffset1[i].Dump());
            }
            newAssembleOp1 = assemble;
            assembleCnt += 1;
        } else if (assemble.toOffset == assemble_offset2) {
            EXPECT_EQ(assemble.toDynOffset.size(), kNumThree);
            for (size_t i = 0; i < dynAssembleOffset2.size(); ++i) {
                EXPECT_EQ(assemble.toDynOffset[i].Dump(), dynAssembleOffset2[i].Dump());
            }
            newAssembleOp2 = assemble;
            assembleCnt += 10;
        }
    }
    EXPECT_EQ(assembleCnt, 11);
    auto newReshapeOutput1 = newAssembleOp1.input;
    std::vector<SymbolicScalar> expectDynShape = {SymbolicScalar("a") * 1, kNumOne, kNumTwo};
    std::vector<SymbolicScalar> expectDynOffset1 = {SymbolicScalar("b") * 1, kNumZero, kNumZero};
    std::vector<SymbolicScalar> expectDynOffset2 = {SymbolicScalar("b") * 1, kNumOne, kNumZero};
    EXPECT_EQ(newReshapeOutput1->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
    EXPECT_EQ(newReshapeOutput1->GetDynValidShape().size(), kNumThree);
    EXPECT_EQ(newReshapeOutput1->GetDynOffset().size(), kNumThree);
    for (size_t i = 0; i < expectDynShape.size(); ++i) {
        EXPECT_EQ(newReshapeOutput1->GetDynValidShape()[i].Dump(), expectDynShape[i].Dump());
        EXPECT_EQ(newReshapeOutput1->GetDynOffset()[i].Dump(), expectDynOffset1[i].Dump());
    }
    auto newReshapeOutput2 = newAssembleOp2.input;
    EXPECT_EQ(newReshapeOutput2->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
    EXPECT_EQ(newReshapeOutput2->GetDynValidShape().size(), kNumThree);
    EXPECT_EQ(newReshapeOutput2->GetDynOffset().size(), kNumThree);
    for (size_t i = 0; i < expectDynShape.size(); ++i) {
        EXPECT_EQ(newReshapeOutput2->GetDynValidShape()[i].Dump(), expectDynShape[i].Dump());
        EXPECT_EQ(newReshapeOutput2->GetDynOffset()[i].Dump(), expectDynOffset2[i].Dump());
    }
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
    EXPECT_EQ(viewOpAttribute->GetFromDynOffset().size(), kNumThree);
    for (size_t i = 0; i < dynOutputOffset.size(); ++i) {
        EXPECT_EQ(viewOpAttribute->GetFromDynOffset()[i].Dump(), dynOutputOffset[i].Dump());
    }
    auto newInput = view_op.GetInputOperand(kNumZero);
    EXPECT_EQ(newInput->shape, shape3);
    for (size_t i = 0; i < dynOutputOffset.size(); ++i) {
        EXPECT_EQ(newInput->dynValidShape_[i].Dump(), dynOutputShape[i].Dump());
        EXPECT_EQ(newInput->dynOffset_[i].Dump(), dynOutputOffset[i].Dump());
    }
}

/*
验证多对一兜底场景下其他数据的处理
rawShape = {2, 4}
{2, 2}(ddr) -> assemble -> {2, 4}(unknown) -> reshape -> {2, 2, 2}(unknown) -> view -> {2, 2, 2}(ddr)
{2, 2}(ddr) -> assemble ->
{2, 2}(ddr) -> assemble -> {2, 2}(unknown) -> reshape -> {2, 1, 2}(ub) -> assemble -> {2, 2, 2}(unknown) -> view -> {2, 2, 2}(ddr)
{2, 2}(ddr) -> assemble -> {2, 2}(unknown) -> reshape -> {2, 1, 2}(ub) -> assemble -> 
*/
TEST_F(TestSplitReshapePass, TestDynUpdateForAssembleAfterReshapeOtherCase) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int> shape1 = {kNumTwo, kNumFour};
    std::vector<int> shape2 = {kNumTwo, kNumTwo};
    std::vector<int> shape3 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> offset1 = {kNumZero, kNumZero};
    std::vector<int> offset2 = {kNumZero, kNumTwo};
    std::vector<int> offset3 = {kNumZero, kNumZero, kNumZero};
    std::vector<SymbolicScalar> dynShape = {SymbolicScalar("a"), kNumTwo};
    std::vector<SymbolicScalar> dynOffset1 = {SymbolicScalar("b"), kNumZero};
    std::vector<SymbolicScalar> dynOffset2 = {SymbolicScalar("b"), kNumTwo};
    std::vector<SymbolicScalar> dynOutputShape = {SymbolicScalar("a"), kNumTwo, kNumTwo};
    std::vector<SymbolicScalar> dynOutputOffset = {SymbolicScalar("b"), kNumZero, kNumZero};
    std::vector<int> view_offset = {kNumZero, kNumZero, kNumZero};

    std::shared_ptr<RawTensor> RawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto input1 = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor1, offset1, shape2, dynShape);
    input1->UpdateOffset(TensorOffset(offset1, dynOffset1));
    input1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto input2 = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor1, offset2, shape2, dynShape);
    input2->UpdateOffset(TensorOffset(offset2, dynOffset2));
    input2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3, dynOutputShape);
    output->UpdateOffset(TensorOffset(offset3, dynOutputOffset));
    output->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    
    std::vector<SymbolicScalar> assembleOffset1 = {SymbolicScalar("b"), kNumZero};
    std::vector<SymbolicScalar> assembleOffset2 = {SymbolicScalar("b"), kNumTwo};
    auto &assemble_op1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input1}, {ubTensor1});
    auto assemble_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_UB, offset1, assembleOffset1);
    assemble_op1.SetOpAttribute(assemble_Attr1);
    auto &assemble_op2 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input2}, {ubTensor1});
    auto assemble_Attr2 = std::make_shared<AssembleOpAttribute>(MEM_UB, offset2, assembleOffset2);
    assemble_op2.SetOpAttribute(assemble_Attr2);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &view_op = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output});
    auto view_Attr = std::make_shared<ViewOpAttribute>(view_offset);
    view_op.SetOpAttribute(view_Attr);
    
    CalcOverlapPara para;
    SplitReshape pass;
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input1, input2};
    std::vector<int> newInput1TileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int> newInput1TileShape = {kNumTwo, kNumOne, kNumTwo};
    std::vector<SymbolicScalar> newInput1DynOffset = {SymbolicScalar("b"), kNumZero, kNumZero};
    std::vector<SymbolicScalar> newInput1DynShape = {SymbolicScalar("a"), kNumOne, kNumTwo};
    auto newInput1 = std::make_shared<LogicalTensor>(*currFunctionPtr, input1->tensor, newInput1TileOffset, newInput1TileShape, newInput1DynShape);
    newInput1->UpdateOffset(TensorOffset(newInput1TileOffset, newInput1DynOffset));
    std::vector<int> newInput2TileOffset = {kNumZero, kNumOne, kNumZero};
    std::vector<int> newInput2TileShape = {kNumTwo, kNumOne, kNumTwo};
    std::vector<SymbolicScalar> newInput2DynOffset = {SymbolicScalar("b"), kNumOne, kNumZero};
    std::vector<SymbolicScalar> newInput2DynShape = {SymbolicScalar("a"), kNumOne, kNumTwo};
    auto newInput2 = std::make_shared<LogicalTensor>(*currFunctionPtr, input2->tensor, newInput2TileOffset, newInput2TileShape, newInput2DynShape);
    newInput2->UpdateOffset(TensorOffset(newInput2TileOffset, newInput2DynOffset));
    para.newOverlaps = {newInput1, newInput2};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output;
    para.newInputViewTileShape = {kNumTwo, kNumOne, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumOne, kNumZero};
    para.newInputViewDynShape = {SymbolicScalar("a"), kNumOne, kNumTwo};
    para.newInputViewDynOffset = {SymbolicScalar("b"), kNumOne, kNumZero};
    std::vector<int> InputViewTileShape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<SymbolicScalar> InputViewDynShape = {SymbolicScalar("a"), kNumTwo, kNumTwo};
    std::vector<SymbolicScalar> InputViewDynOffset = {SymbolicScalar("b"), kNumZero, kNumZero};
    auto inputView = std::make_shared<LogicalTensor>(*currFunctionPtr, ubTensor2->tensor, view_offset, InputViewTileShape);
    inputView->UpdateOffset(TensorOffset(view_offset, InputViewDynOffset));
    para.inputView = inputView;
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.UpdateForAssembleAfterReshape(*currFunctionPtr, view_op, para), SUCCESS);
    EXPECT_EQ(pass.redundantViewops.size(), kSizeZero);
    EXPECT_EQ(pass.reshapes.size(), kSizeTwo);
    EXPECT_EQ(pass.assembles.size(), kSizeFour);

    LogicalTensorPtr newReshapeSource1;
    LogicalTensorPtr newReshapeSource2;
    LogicalTensorPtr newReshapeOutput1;
    LogicalTensorPtr newReshapeOutput2;
    DynAssembleOp assembleBeforeReshape1;
    DynAssembleOp assembleAfterReshape1;
    DynAssembleOp assembleBeforeReshape2;
    DynAssembleOp assembleAfterReshape2;
    auto newReshapeOp1 = std::make_shared<ReshapeOp>(nullptr, nullptr);
    auto newReshapeOp2 = std::make_shared<ReshapeOp>(nullptr, nullptr);
    int assembleCnt = 0;
    int reshapeCnt = 0;
    std::vector<int> assemble_offset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int> assemble_offset2 = {kNumZero, kNumOne, kNumZero};
    std::vector<SymbolicScalar> dynSrcAssembleOffset1 = {SymbolicScalar("b"), kNumZero};
    std::vector<SymbolicScalar> dynSrcAssembleOffset2 = {SymbolicScalar("b"), kNumTwo};
    std::vector<SymbolicScalar> dynDstAssembleOffset1 = {SymbolicScalar("b") * 1 - SymbolicScalar("b"), kNumZero, kNumZero};
    std::vector<SymbolicScalar> dynDstAssembleOffset2 = {SymbolicScalar("b") * 1 - SymbolicScalar("b"), kNumOne, kNumZero};
    for (const auto &assemble : pass.assembles) {
        if (assemble.toOffset == assemble_offset1) {
            EXPECT_EQ(assemble.from, MemoryType::MEM_UNKNOWN);
            EXPECT_EQ(assemble.output, view_op.GetInputOperand(kSizeZero));
            EXPECT_EQ(assemble.toDynOffset.size(), kNumThree);
            for (size_t i = 0; i < dynDstAssembleOffset1.size(); ++i) {
                EXPECT_EQ(assemble.toDynOffset[i].Dump(), dynDstAssembleOffset1[i].Dump());
            }
            assembleAfterReshape1 = assemble;
            newReshapeOutput1 = assembleAfterReshape1.input;
            assembleCnt += 1;
        } else if (assemble.toOffset == assemble_offset2) {
            EXPECT_EQ(assemble.from, MemoryType::MEM_UNKNOWN);
            EXPECT_EQ(assemble.output, view_op.GetInputOperand(kSizeZero));
            EXPECT_EQ(assemble.toDynOffset.size(), kNumThree);
            for (size_t i = 0; i < dynDstAssembleOffset2.size(); ++i) {
                EXPECT_EQ(assemble.toDynOffset[i].Dump(), dynDstAssembleOffset2[i].Dump());
            }
            assembleAfterReshape2 = assemble;
            newReshapeOutput2 = assembleAfterReshape2.input;
            assembleCnt += 10;
        } else if (assemble.input == input1) {
            EXPECT_EQ(assemble.from, MemoryType::MEM_DEVICE_DDR);
            EXPECT_EQ(assemble.toDynOffset.size(), kNumTwo);
            for (size_t i = 0; i < dynSrcAssembleOffset1.size(); ++i) {
                EXPECT_EQ(assemble.toDynOffset[i].Dump(), dynSrcAssembleOffset1[i].Dump());
            }
            assembleBeforeReshape1 = assemble;
            newReshapeSource1 = assembleBeforeReshape1.output;
            assembleCnt += 100;
        } else if (assemble.input == input2) {
            EXPECT_EQ(assemble.from, MemoryType::MEM_DEVICE_DDR);
            EXPECT_EQ(assemble.toDynOffset.size(), kNumTwo);
            for (size_t i = 0; i < dynSrcAssembleOffset2.size(); ++i) {
                EXPECT_EQ(assemble.toDynOffset[i].Dump(), dynSrcAssembleOffset2[i].Dump());
            }
            assembleBeforeReshape2 = assemble;
            newReshapeSource2 = assembleBeforeReshape2.output;
            assembleCnt += 1000;
        }
    }
    EXPECT_EQ(assembleCnt, 1111);
    for (const auto &reshape : pass.reshapes) {
        auto reshapeOp = reshape.second;
        if (reshapeOp->input == assembleBeforeReshape1.output) {
            EXPECT_EQ(reshapeOp->input, newReshapeSource1);
            EXPECT_EQ(reshapeOp->output, newReshapeOutput1);
            EXPECT_EQ(reshapeOp->input->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
            reshapeCnt += 1;
        } else if (reshapeOp->input == assembleBeforeReshape2.output) {
            EXPECT_EQ(reshapeOp->input, newReshapeSource2);
            EXPECT_EQ(reshapeOp->output, newReshapeOutput2);
            EXPECT_EQ(reshapeOp->input->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
            reshapeCnt += 10;
        }
    }
    EXPECT_EQ(reshapeCnt, 11);
    
    std::vector<int> srcShape = {kNumTwo, kNumTwo};
    std::vector<int> dstShape = {kNumTwo, kNumOne, kNumTwo};
    std::vector<int> srcOffset1 = {kNumZero, kNumZero};
    std::vector<int> srcOffset2 = {kNumZero, kNumTwo};
    std::vector<int> dstOffset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int> dstOffset2 = {kNumZero, kNumOne, kNumZero};
    
    std::vector<SymbolicScalar> dynSrcShape = {SymbolicScalar("a"), kNumTwo};
    std::vector<SymbolicScalar> dynDstShape = {SymbolicScalar("a") * 1, kNumOne, kNumTwo};
    std::vector<SymbolicScalar> dynSrcOffset1 = {SymbolicScalar("b"), kNumZero};
    std::vector<SymbolicScalar> dynSrcOffset2 = {SymbolicScalar("b"), kNumTwo};
    std::vector<SymbolicScalar> dynDstOffset1 = {SymbolicScalar("b") * 1, kNumZero, kNumZero};
    std::vector<SymbolicScalar> dynDstOffset2 = {SymbolicScalar("b") * 1, kNumOne, kNumZero};
    EXPECT_EQ(newReshapeSource1->GetShape(), srcShape);
    EXPECT_EQ(newReshapeSource1->GetOffset(), srcOffset1);
    EXPECT_EQ(newReshapeSource1->GetDynOffset().size(), kNumTwo);
    EXPECT_EQ(newReshapeSource1->GetDynValidShape().size(), kNumTwo);
    for (size_t i = 0; i < kNumTwo; ++i) {
        EXPECT_EQ(newReshapeSource1->GetDynOffset()[i].Dump(), dynSrcOffset1[i].Dump());
        EXPECT_EQ(newReshapeSource1->GetDynValidShape()[i].Dump(), dynSrcShape[i].Dump());
    }
    EXPECT_EQ(newReshapeSource2->GetShape(), srcShape);
    EXPECT_EQ(newReshapeSource2->GetOffset(), srcOffset2);
    EXPECT_EQ(newReshapeSource2->GetDynOffset().size(), kNumTwo);
    EXPECT_EQ(newReshapeSource2->GetDynValidShape().size(), kNumTwo);
    for (size_t i = 0; i < kNumTwo; ++i) {
        EXPECT_EQ(newReshapeSource2->GetDynOffset()[i].Dump(), dynSrcOffset2[i].Dump());
        EXPECT_EQ(newReshapeSource2->GetDynValidShape()[i].Dump(), dynSrcShape[i].Dump());
    }
    EXPECT_EQ(newReshapeOutput1->GetShape(), dstShape);
    EXPECT_EQ(newReshapeOutput1->GetOffset(), dstOffset1);
    EXPECT_EQ(newReshapeOutput1->GetDynOffset().size(), kNumThree);
    EXPECT_EQ(newReshapeOutput1->GetDynValidShape().size(), kNumThree);
    for (size_t i = 0; i < kNumThree; ++i) {
        EXPECT_EQ(newReshapeOutput1->GetDynOffset()[i].Dump(), dynDstOffset1[i].Dump());
        EXPECT_EQ(newReshapeOutput1->GetDynValidShape()[i].Dump(), dynDstShape[i].Dump());
    }
    EXPECT_EQ(newReshapeOutput2->GetShape(), dstShape);
    EXPECT_EQ(newReshapeOutput2->GetOffset(), dstOffset2);
    EXPECT_EQ(newReshapeOutput2->GetDynOffset().size(), kNumThree);
    EXPECT_EQ(newReshapeOutput2->GetDynValidShape().size(), kNumThree);
    for (size_t i = 0; i < kNumThree; ++i) {
        EXPECT_EQ(newReshapeOutput2->GetDynOffset()[i].Dump(), dynDstOffset2[i].Dump());
        EXPECT_EQ(newReshapeOutput2->GetDynValidShape()[i].Dump(), dynDstShape[i].Dump());
    }

    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(view_op.GetOpAttribute().get());
    EXPECT_NE(viewOpAttribute, nullptr);
    EXPECT_EQ(viewOpAttribute->GetFromOffset(), view_offset);
    EXPECT_EQ(viewOpAttribute->GetFromDynOffset().size(), kNumThree);
    for (size_t i = 0; i < dynOutputOffset.size(); ++i) {
        EXPECT_EQ(viewOpAttribute->GetFromDynOffset()[i].Dump(), dynOutputOffset[i].Dump());
    }
    auto newInput = view_op.GetInputOperand(kNumZero);
    EXPECT_EQ(newInput->shape, shape3);
    for (size_t i = 0; i < dynOutputOffset.size(); ++i) {
        EXPECT_EQ(newInput->dynValidShape_[i].Dump(), dynOutputShape[i].Dump());
        EXPECT_EQ(newInput->dynOffset_[i].Dump(), dynOutputOffset[i].Dump());
    }
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
TEST_F(TestSplitReshapePass, TestPerfectlyMatchedSTest) {
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
TEST_F(TestSplitReshapePass, TestBeCoveredSTest) {
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
TEST_F(TestSplitReshapePass, TestPerfectlyMatchedWithallSTest) {
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
TEST_F(TestSplitReshapePass, TestPerfectlyMatchedWithallAssembleSTest) {
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
验证一对一场景下动态shape的兜底策略
因为缺乏宏构建策略，手动构造expandfunction的输出构图
1) 用例设置：
{2,2,2} -> assemble -> {2,2,4} -> reshape -> {2,2,1,4} -> view -> {2,2,1,2}
{2,2,2} -> assemble                                    -> view -> {2,2,1,2}
{a0,a1,2}/{b0,b1,0}    {a0,a1,4}/{b0,b1,0}   {a0,a1,1,4}/{b0,b1,0,0}  {a0,a1,1,2}/{b0,b1,0,0}
{a0,a1,2}/{b0,b1,2}                                                   {a0,a1,1,2}/{b0,b1,0,2}
2) splitreshape
{2,2,2} -> assemble -> {2,2,2} -> reshape -> {2,2,1,2} -> view -> {2,2,1,2}
{2,2,2} -> assemble -> {2,2,2} -> reshape -> {2,2,1,2} -> view -> {2,2,1,2}
*/
TEST_F(TestSplitReshapePass, TestDynPerfectlyMatchSTest) {
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(func != nullptr);

    std::vector<int> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> shape2 = {kNumTwo, kNumTwo, kNumFour};
    std::vector<int> shape3 = {kNumTwo, kNumTwo, kNumOne, kNumFour};
    std::vector<int> shape4 = {kNumTwo, kNumTwo, kNumOne, kNumTwo};
    std::vector<int> assembleOffset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int> assembleOffset2 = {kNumZero, kNumZero, kNumTwo};
    std::vector<int> viewOffset1 = {kNumZero, kNumZero, kNumZero, kNumZero};
    std::vector<int> viewOffset2 = {kNumZero, kNumZero, kNumZero, kNumTwo};
    
    std::vector<SymbolicScalar> dynInputShape = {SymbolicScalar("a0"), SymbolicScalar("a1"), kNumTwo};
    std::vector<SymbolicScalar> dynInputOffset1 = {SymbolicScalar("b0"), SymbolicScalar("b1"), kNumZero};
    std::vector<SymbolicScalar> dynInputOffset2 = {SymbolicScalar("b0"), SymbolicScalar("b1"), kNumTwo};
    
    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape2);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape3);
    auto input1 = std::make_shared<LogicalTensor>(*func, ddrRawTensor1, assembleOffset1, shape1, dynInputShape);
    input1->UpdateOffset(TensorOffset(assembleOffset1, dynInputOffset1));
    input1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto input2 = std::make_shared<LogicalTensor>(*func, ddrRawTensor1, assembleOffset2, shape1, dynInputShape);
    input2->UpdateOffset(TensorOffset(assembleOffset2, dynInputOffset2));
    input2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*func, DT_FP32, shape2);
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*func, DT_FP32, shape3);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output1 = std::make_shared<LogicalTensor>(*func, ddrRawTensor2, viewOffset1, shape4);
    output1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output2 = std::make_shared<LogicalTensor>(*func, ddrRawTensor2, viewOffset2, shape4);
    output2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    
    auto &assemble_op1 = func->AddOperation(Opcode::OP_ASSEMBLE, {input1}, {ubTensor1});
    auto assemble_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset1, dynInputOffset1);
    assemble_op1.SetOpAttribute(assemble_Attr1);
    auto &assemble_op2 = func->AddOperation(Opcode::OP_ASSEMBLE, {input2}, {ubTensor1});
    auto assemble_Attr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset2, dynInputOffset2);
    assemble_op2.SetOpAttribute(assemble_Attr2);
    func->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &view_op1 = func->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output1});
    auto view_Attr1 = std::make_shared<ViewOpAttribute>(viewOffset1);
    view_op1.SetOpAttribute(view_Attr1);
    auto &view_op2 = func->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output2});
    auto view_Attr2 = std::make_shared<ViewOpAttribute>(viewOffset2);
    view_op2.SetOpAttribute(view_Attr2);
    
    func->inCasts_.push_back(input1);
    func->inCasts_.push_back(input2);
    func->outCasts_.push_back(output1);
    func->outCasts_.push_back(output2);

    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("SplitReshapeTestStrategy", {
        {   "SplitReshape",   "SplitReshape",  PassType::TYPE_TILE_GRAPH},
    });
    EXPECT_EQ(passManager.RunPass(Program::GetInstance(), *func, "SplitReshapeTestStrategy"), SUCCESS);
    
    int reshapeOp = 0;
    int assembleOp = 0;
    int viewOp = 0;
    Operation *newAssemble1;
    Operation *newAssemble2;
    for (auto &op : func->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_RESHAPE) {
            reshapeOp++;
        } else if (op->GetOpcode() == Opcode::OP_ASSEMBLE) {
            if (op->GetInputOperand(kSizeZero) == input1) {
                newAssemble1 = op;
                assembleOp += 1;
            } else if (op->GetInputOperand(kSizeZero) == input2) {
                newAssemble2 = op;
                assembleOp += 10;
            }
        } else if (op->GetOpcode() == Opcode::OP_VIEW) {
            viewOp++;
        }
    }
    EXPECT_EQ(reshapeOp, kNumTwo);
    EXPECT_EQ(viewOp, kNumTwo);

    EXPECT_EQ(assembleOp, 11);
    auto assembleAttr1 = dynamic_cast<AssembleOpAttribute *>(newAssemble1->GetOpAttribute().get());
    EXPECT_EQ(assembleAttr1->GetToDynOffset().size(), kNumThree);
    auto assembleAttr2 = dynamic_cast<AssembleOpAttribute *>(newAssemble2->GetOpAttribute().get());
    EXPECT_EQ(assembleAttr2->GetToDynOffset().size(), kNumThree); 
    for (size_t i = 0; i < kNumThree; ++i) {
        EXPECT_EQ(assembleAttr1->GetToDynOffset()[i].Dump(), dynInputOffset1[i].Dump());
        EXPECT_EQ(assembleAttr2->GetToDynOffset()[i].Dump(), dynInputOffset2[i].Dump());
    }

    std::vector<SymbolicScalar> dynOutputShape = {SymbolicScalar("a0") * 1, SymbolicScalar("a1") * 1, kNumOne, kNumTwo};
    std::vector<SymbolicScalar> dynOutputOffset1 = {SymbolicScalar("b0") * 1, SymbolicScalar("b1") * 1, kNumZero, kNumZero};
    std::vector<SymbolicScalar> dynOutputOffset2 = {SymbolicScalar("b0") * 1, SymbolicScalar("b1") * 1, kNumZero, kNumTwo};
    
    auto reshapeSource1 = newAssemble1->GetOutputOperand(kSizeZero);
    auto reshapeSource2 = newAssemble2->GetOutputOperand(kSizeZero);
    EXPECT_NE(reshapeSource1, reshapeSource2);
    EXPECT_EQ(reshapeSource1->dynOffset_.size(), kNumThree);
    EXPECT_EQ(reshapeSource1->dynValidShape_.size(), kNumThree);
    EXPECT_EQ(reshapeSource2->dynOffset_.size(), kNumThree);
    EXPECT_EQ(reshapeSource2->dynValidShape_.size(), kNumThree);
    for (size_t i = 0; i < kNumThree; ++i) {
        EXPECT_EQ(reshapeSource1->dynValidShape_[i].Dump(), dynInputShape[i].Dump());
        EXPECT_EQ(reshapeSource1->dynOffset_[i].Dump(), dynInputOffset1[i].Dump());
        EXPECT_EQ(reshapeSource2->dynValidShape_[i].Dump(), dynInputShape[i].Dump());
        EXPECT_EQ(reshapeSource2->dynOffset_[i].Dump(), dynInputOffset2[i].Dump());
    }
    auto reshape1 = *(reshapeSource1->GetConsumers().begin());
    auto reshape2 = *(reshapeSource2->GetConsumers().begin());
    EXPECT_NE(reshape1, reshape2);
    auto reshapeOutput1 = reshape1->GetOutputOperand(kSizeZero);
    auto reshapeOutput2 = reshape2->GetOutputOperand(kSizeZero);
    EXPECT_NE(reshapeOutput1, reshapeOutput2);
    EXPECT_EQ(reshapeOutput1->dynOffset_.size(), kNumFour);
    EXPECT_EQ(reshapeOutput1->dynValidShape_.size(), kNumFour);
    EXPECT_EQ(reshapeOutput2->dynOffset_.size(), kNumFour);
    EXPECT_EQ(reshapeOutput2->dynValidShape_.size(), kNumFour);
    for (size_t i = 0; i < kNumFour; ++i) {
        EXPECT_EQ(reshapeOutput1->dynValidShape_[i].Dump(), dynOutputShape[i].Dump());
        EXPECT_EQ(reshapeOutput1->dynOffset_[i].Dump(), dynOutputOffset1[i].Dump());
        EXPECT_EQ(reshapeOutput2->dynValidShape_[i].Dump(), dynOutputShape[i].Dump());
        EXPECT_EQ(reshapeOutput2->dynOffset_[i].Dump(), dynOutputOffset2[i].Dump());
    }
    auto view1 = *(reshapeOutput1->GetConsumers().begin());
    auto view2 = *(reshapeOutput2->GetConsumers().begin());
    EXPECT_NE(view1, view2);

    auto viewOpAttribute1 = dynamic_cast<ViewOpAttribute *>(view1->GetOpAttribute().get());
    auto viewOpAttribute2 = dynamic_cast<ViewOpAttribute *>(view2->GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute1->GetFromDynOffset().size(), kNumFour);
    EXPECT_EQ(viewOpAttribute2->GetFromDynOffset().size(), kNumFour);
    for (size_t i = 0; i < dynOutputOffset1.size(); ++i) {
        EXPECT_EQ(viewOpAttribute1->GetFromDynOffset()[i].Dump(), dynOutputOffset1[i].Dump());
        EXPECT_EQ(viewOpAttribute2->GetFromDynOffset()[i].Dump(), dynOutputOffset2[i].Dump());
    }
}

/*
验证一对多场景下动态shape的兜底策略
因为缺乏宏构建策略，手动构造expandfunction的输出构图
1) 用例设置：
{2,2,2} -> assemble -> {2,2,4} -> reshape -> {4,4} -> view -> {2,2}
{2,2,2} -> assemble                                -> view -> {2,2}
                                                   -> view -> {2,2}
                                                   -> view -> {2,2}
{2,2,a}/{0,0,b}        {2,2,a}/{0,0,b}       {4,a}/{0,b}      {2,a}/{0,b}
{2,2,a}/{0,0,b}                                               {2,a}/{2,b}
                                                              {2,a}/{0,b}
                                                              {2,a}/{2,b}
2) splitreshape
                                                   -> view -> {2,2}
{2,2,2} -> assemble -> {2,2,2} -> reshape -> {2,4} -> view -> {2,2}
{2,2,2} -> assemble -> {2,2,2} -> reshape -> {2,4} -> view -> {2,2}
                                                   -> view -> {2,2}
*/
TEST_F(TestSplitReshapePass, TestDynBeCoveredSTest) {
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(func != nullptr);

    std::vector<int> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> shape2 = {kNumTwo, kNumTwo, kNumFour};
    std::vector<int> shape3 = {kNumFour, kNumFour};
    std::vector<int> shape4 = {kNumTwo, kNumTwo};
    std::vector<int> assembleOffset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int> assembleOffset2 = {kNumZero, kNumZero, kNumTwo};
    std::vector<int> viewOffset1 = {kNumZero, kNumZero};
    std::vector<int> viewOffset2 = {kNumZero, kNumTwo};
    std::vector<int> viewOffset3 = {kNumTwo, kNumZero};
    std::vector<int> viewOffset4 = {kNumTwo, kNumTwo};
    
    std::vector<SymbolicScalar> dynInputShape = {kNumTwo, kNumTwo, SymbolicScalar("a")};
    std::vector<SymbolicScalar> dynInputOffset = {kNumZero, kNumZero, SymbolicScalar("b")};
    std::vector<SymbolicScalar> dynOutputShape = {kNumTwo, SymbolicScalar("a")};
    std::vector<SymbolicScalar> dynOutputOffset1 = {kNumZero, SymbolicScalar("b")};
    std::vector<SymbolicScalar> dynOutputOffset2 = {kNumTwo, SymbolicScalar("b")};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape2);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape3);
    auto input1 = std::make_shared<LogicalTensor>(*func, ddrRawTensor1, assembleOffset1, shape1, dynInputShape);
    input1->UpdateOffset(TensorOffset(assembleOffset1, dynInputOffset));
    input1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto input2 = std::make_shared<LogicalTensor>(*func, ddrRawTensor1, assembleOffset2, shape1, dynInputShape);
    input2->UpdateOffset(TensorOffset(assembleOffset2, dynInputOffset));
    input2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*func, DT_FP32, shape2);
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*func, DT_FP32, shape3);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output1 = std::make_shared<LogicalTensor>(*func, ddrRawTensor2, viewOffset1, shape4, dynOutputShape);
    output1->UpdateOffset(TensorOffset(viewOffset1, dynOutputOffset1));
    output1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output2 = std::make_shared<LogicalTensor>(*func, ddrRawTensor2, viewOffset2, shape4, dynOutputShape);
    output2->UpdateOffset(TensorOffset(viewOffset2, dynOutputOffset1));
    output2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output3 = std::make_shared<LogicalTensor>(*func, ddrRawTensor2, viewOffset3, shape4, dynOutputShape);
    output3->UpdateOffset(TensorOffset(viewOffset3, dynOutputOffset2));
    output3->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output4 = std::make_shared<LogicalTensor>(*func, ddrRawTensor2, viewOffset4, shape4, dynOutputShape);
    output4->UpdateOffset(TensorOffset(viewOffset4, dynOutputOffset2));
    output4->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    
    auto &assemble_op1 = func->AddOperation(Opcode::OP_ASSEMBLE, {input1}, {ubTensor1});
    auto assemble_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset1, dynInputOffset);
    assemble_op1.SetOpAttribute(assemble_Attr1);
    auto &assemble_op2 = func->AddOperation(Opcode::OP_ASSEMBLE, {input2}, {ubTensor1});
    auto assemble_Attr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset2, dynInputOffset);
    assemble_op2.SetOpAttribute(assemble_Attr2);
    func->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &view_op1 = func->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output1});
    auto view_Attr1 = std::make_shared<ViewOpAttribute>(viewOffset1);
    view_op1.SetOpAttribute(view_Attr1);
    auto &view_op2 = func->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output2});
    auto view_Attr2 = std::make_shared<ViewOpAttribute>(viewOffset2);
    view_op2.SetOpAttribute(view_Attr2);
    auto &view_op3 = func->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output3});
    auto view_Attr3 = std::make_shared<ViewOpAttribute>(viewOffset3);
    view_op3.SetOpAttribute(view_Attr3);
    auto &view_op4 = func->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output4});
    auto view_Attr4 = std::make_shared<ViewOpAttribute>(viewOffset4);
    view_op4.SetOpAttribute(view_Attr4);
    
    func->inCasts_.push_back(input1);
    func->inCasts_.push_back(input2);
    func->outCasts_.push_back(output1);
    func->outCasts_.push_back(output2);
    func->outCasts_.push_back(output3);
    func->outCasts_.push_back(output4);

    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("SplitReshapeTestStrategy", {
        {   "SplitReshape",   "SplitReshape",  PassType::TYPE_TILE_GRAPH},
    });
    EXPECT_EQ(passManager.RunPass(Program::GetInstance(), *func, "SplitReshapeTestStrategy"), SUCCESS);
    
    int reshapeOp = 0;
    int assembleOp = 0;
    int viewOp = 0;
    Operation *newAssemble1;
    Operation *newAssemble2;
    for (auto &op : func->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_RESHAPE) {
            reshapeOp++;
        } else if (op->GetOpcode() == Opcode::OP_ASSEMBLE) {
            if (op->GetInputOperand(kSizeZero) == input1) {
                newAssemble1 = op;
                assembleOp += 1;
            } else if (op->GetInputOperand(kSizeZero) == input2) {
                newAssemble2 = op;
                assembleOp += 10;
            }
        } else if (op->GetOpcode() == Opcode::OP_VIEW) {
            viewOp++;
        }
    }
    EXPECT_EQ(reshapeOp, kNumTwo);
    EXPECT_EQ(viewOp, kNumFour);

    EXPECT_EQ(assembleOp, 11);
    auto assembleAttr1 = dynamic_cast<AssembleOpAttribute *>(newAssemble1->GetOpAttribute().get());
    EXPECT_EQ(assembleAttr1->GetToDynOffset().size(), kNumThree);
    auto assembleAttr2 = dynamic_cast<AssembleOpAttribute *>(newAssemble2->GetOpAttribute().get());
    EXPECT_EQ(assembleAttr2->GetToDynOffset().size(), kNumThree); 
    for (size_t i = 0; i < kNumThree; ++i) {
        EXPECT_EQ(assembleAttr1->GetToDynOffset()[i].Dump(), dynInputOffset[i].Dump());
        EXPECT_EQ(assembleAttr2->GetToDynOffset()[i].Dump(), dynInputOffset[i].Dump());
    }

    std::vector<SymbolicScalar> dynReshapeOutputShape = {kNumFour, SymbolicScalar("a") * 1};
    std::vector<SymbolicScalar> dynReshapeOutputOffset = {kNumZero, SymbolicScalar("b") * 1};

    auto reshapeSource1 = newAssemble1->GetOutputOperand(kSizeZero);
    auto reshapeSource2 = newAssemble2->GetOutputOperand(kSizeZero);
    EXPECT_NE(reshapeSource1, reshapeSource2);
    EXPECT_EQ(reshapeSource1->dynOffset_.size(), kNumThree);
    EXPECT_EQ(reshapeSource1->dynValidShape_.size(), kNumThree);
    EXPECT_EQ(reshapeSource2->dynOffset_.size(), kNumThree);
    EXPECT_EQ(reshapeSource2->dynValidShape_.size(), kNumThree);
    for (size_t i = 0; i < kNumThree; ++i) {
        EXPECT_EQ(reshapeSource1->dynValidShape_[i].Dump(), dynInputShape[i].Dump());
        EXPECT_EQ(reshapeSource1->dynOffset_[i].Dump(), dynInputOffset[i].Dump());
        EXPECT_EQ(reshapeSource2->dynValidShape_[i].Dump(), dynInputShape[i].Dump());
        EXPECT_EQ(reshapeSource2->dynOffset_[i].Dump(), dynInputOffset[i].Dump());
    }
    auto reshape1 = *(reshapeSource1->GetConsumers().begin());
    auto reshape2 = *(reshapeSource2->GetConsumers().begin());
    EXPECT_NE(reshape1, reshape2);
    auto reshapeOutput1 = reshape1->GetOutputOperand(kSizeZero);
    auto reshapeOutput2 = reshape2->GetOutputOperand(kSizeZero);
    EXPECT_NE(reshapeOutput1, reshapeOutput2);
    EXPECT_EQ(reshapeOutput1->dynOffset_.size(), kNumTwo);
    EXPECT_EQ(reshapeOutput1->dynValidShape_.size(), kNumTwo);
    EXPECT_EQ(reshapeOutput2->dynOffset_.size(), kNumTwo);
    EXPECT_EQ(reshapeOutput2->dynValidShape_.size(), kNumTwo);
    for (size_t i = 0; i < kNumTwo; ++i) {
        EXPECT_EQ(reshapeOutput1->dynValidShape_[i].Dump(), dynReshapeOutputShape[i].Dump());
        EXPECT_EQ(reshapeOutput1->dynOffset_[i].Dump(), dynReshapeOutputOffset[i].Dump());
        EXPECT_EQ(reshapeOutput2->dynValidShape_[i].Dump(), dynReshapeOutputShape[i].Dump());
        EXPECT_EQ(reshapeOutput2->dynOffset_[i].Dump(), dynReshapeOutputOffset[i].Dump());
    }
    EXPECT_EQ(reshapeOutput1->GetConsumers().size(), kNumTwo);
    EXPECT_EQ(reshapeOutput2->GetConsumers().size(), kNumTwo);
    auto view1 = *(reshapeOutput1->GetConsumers().begin());
    auto view2 = *(++(reshapeOutput1->GetConsumers().begin()));
    auto view3 = *(reshapeOutput2->GetConsumers().begin());
    auto view4 = *(++(reshapeOutput2->GetConsumers().begin()));
    EXPECT_NE(view1, view2);
    EXPECT_NE(view1, view3);
    EXPECT_NE(view1, view4);
    
    std::vector<SymbolicScalar> dynViewOffset1 = {kNumZero, SymbolicScalar("b") * 1};
    std::vector<SymbolicScalar> dynViewOffset2 = {kNumTwo, SymbolicScalar("b") * 1};
    auto viewOpAttribute1 = dynamic_cast<ViewOpAttribute *>(view1->GetOpAttribute().get());
    auto viewOpAttribute2 = dynamic_cast<ViewOpAttribute *>(view2->GetOpAttribute().get());
    auto viewOpAttribute3 = dynamic_cast<ViewOpAttribute *>(view3->GetOpAttribute().get());
    auto viewOpAttribute4 = dynamic_cast<ViewOpAttribute *>(view4->GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute1->GetFromDynOffset().size(), kNumTwo);
    EXPECT_EQ(viewOpAttribute2->GetFromDynOffset().size(), kNumTwo);
    EXPECT_EQ(viewOpAttribute3->GetFromDynOffset().size(), kNumTwo);
    EXPECT_EQ(viewOpAttribute4->GetFromDynOffset().size(), kNumTwo);
    for (size_t i = 0; i < dynOutputOffset1.size(); ++i) {
        EXPECT_EQ(viewOpAttribute1->GetFromDynOffset()[i].Dump(), dynViewOffset1[i].Dump());
        EXPECT_EQ(viewOpAttribute2->GetFromDynOffset()[i].Dump(), dynViewOffset2[i].Dump());
        EXPECT_EQ(viewOpAttribute3->GetFromDynOffset()[i].Dump(), dynViewOffset1[i].Dump());
        EXPECT_EQ(viewOpAttribute4->GetFromDynOffset()[i].Dump(), dynViewOffset2[i].Dump());
    }
}

/*
验证多对一场景下动态shape的兜底策略
因为缺乏宏构建策略，手动构造expandfunction的输出构图
1) 用例设置：
{2,2,2} -> assemble -> {8,2,2} -> reshape -> {2,4,2,2} -> view -> {2,2,2,2}
{2,2,2} -> assemble                                    -> view -> {2,2,2,2}
{2,2,2} -> assemble
{2,2,2} -> assemble
{2,2,a}/{0,0,b}        {2,8,a}/{0,0,b}       {2,4,2,a}/{0,0,0,b}  {2,2,2,a}/{0,0,0,b}
{2,2,a}/{2,0,b}                                                   {2,2,2,a}/{0,2,0,b}
{2,2,a}/{4,0,b}
{2,2,a}/{6,0,b}
2) splitreshape
{2,2,2} -> assemble ->
{2,2,2} -> assemble -> reshape -> {2,2,2,2} -> view -> {2,2,2,2}
{2,2,2} -> assemble -> reshape -> {2,2,2,2} -> view -> {2,2,2,2}
{2,2,2} -> assemble ->
*/
TEST_F(TestSplitReshapePass, TestDynPerfectlyMatchWithAllSTest) {
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(func != nullptr);

    std::vector<int> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> shape2 = {kNumTwo, kNumEight, kNumTwo};
    std::vector<int> shape3 = {kNumTwo, kNumFour, kNumTwo, kNumTwo};
    std::vector<int> shape4 = {kNumTwo, kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> assembleOffset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int> assembleOffset2 = {kNumZero, kNumTwo, kNumZero};
    std::vector<int> assembleOffset3 = {kNumZero, kNumFour, kNumZero};
    std::vector<int> assembleOffset4 = {kNumZero, kNumSix, kNumZero};
    std::vector<int> viewOffset1 = {kNumZero, kNumZero, kNumZero, kNumZero};
    std::vector<int> viewOffset2 = {kNumZero, kNumTwo, kNumZero, kNumZero};
    
    std::vector<SymbolicScalar> dynInputShape = {kNumTwo, kNumTwo, SymbolicScalar("a")};
    std::vector<SymbolicScalar> dynInputOffset1 = {kNumZero, kNumZero, SymbolicScalar("b")};
    std::vector<SymbolicScalar> dynInputOffset2 = {kNumZero, kNumTwo, SymbolicScalar("b")};
    std::vector<SymbolicScalar> dynInputOffset3 = {kNumZero, kNumFour, SymbolicScalar("b")};
    std::vector<SymbolicScalar> dynInputOffset4 = {kNumZero, kNumSix, SymbolicScalar("b")};
    std::vector<SymbolicScalar> dynOutputShape = {kNumTwo, kNumTwo, kNumTwo, SymbolicScalar("a")};
    std::vector<SymbolicScalar> dynOutputOffset1 = {kNumZero, kNumZero, kNumZero, SymbolicScalar("b")};
    std::vector<SymbolicScalar> dynOutputOffset2 = {kNumZero, kNumTwo, kNumZero, SymbolicScalar("b")};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape2);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape3);
    auto input1 = std::make_shared<LogicalTensor>(*func, ddrRawTensor1, assembleOffset1, shape1, dynInputShape);
    input1->UpdateOffset(TensorOffset(assembleOffset1, dynInputOffset1));
    input1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto input2 = std::make_shared<LogicalTensor>(*func, ddrRawTensor1, assembleOffset2, shape1, dynInputShape);
    input2->UpdateOffset(TensorOffset(assembleOffset2, dynInputOffset2));
    input2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto input3 = std::make_shared<LogicalTensor>(*func, ddrRawTensor1, assembleOffset3, shape1, dynInputShape);
    input3->UpdateOffset(TensorOffset(assembleOffset3, dynInputOffset3));
    input3->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto input4 = std::make_shared<LogicalTensor>(*func, ddrRawTensor1, assembleOffset4, shape1, dynInputShape);
    input4->UpdateOffset(TensorOffset(assembleOffset4, dynInputOffset4));
    input4->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*func, DT_FP32, shape2);
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*func, DT_FP32, shape3);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output1 = std::make_shared<LogicalTensor>(*func, ddrRawTensor2, viewOffset1, shape4, dynOutputShape);
    output1->UpdateOffset(TensorOffset(viewOffset1, dynOutputOffset1));
    output1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output2 = std::make_shared<LogicalTensor>(*func, ddrRawTensor2, viewOffset2, shape4, dynOutputShape);
    output2->UpdateOffset(TensorOffset(viewOffset2, dynOutputOffset2));
    output2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    
    auto &assemble_op1 = func->AddOperation(Opcode::OP_ASSEMBLE, {input1}, {ubTensor1});
    auto assemble_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset1, dynInputOffset1);
    assemble_op1.SetOpAttribute(assemble_Attr1);
    auto &assemble_op2 = func->AddOperation(Opcode::OP_ASSEMBLE, {input2}, {ubTensor1});
    auto assemble_Attr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset2, dynInputOffset2);
    assemble_op2.SetOpAttribute(assemble_Attr2);
    auto &assemble_op3 = func->AddOperation(Opcode::OP_ASSEMBLE, {input3}, {ubTensor1});
    auto assemble_Attr3 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset3, dynInputOffset3);
    assemble_op3.SetOpAttribute(assemble_Attr3);
    auto &assemble_op4 = func->AddOperation(Opcode::OP_ASSEMBLE, {input4}, {ubTensor1});
    auto assemble_Attr4 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset4, dynInputOffset4);
    assemble_op4.SetOpAttribute(assemble_Attr4);
    func->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &view_op1 = func->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output1});
    auto view_Attr1 = std::make_shared<ViewOpAttribute>(viewOffset1);
    view_op1.SetOpAttribute(view_Attr1);
    auto &view_op2 = func->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output2});
    auto view_Attr2 = std::make_shared<ViewOpAttribute>(viewOffset2);
    view_op2.SetOpAttribute(view_Attr2);
    
    func->inCasts_.push_back(input1);
    func->inCasts_.push_back(input2);
    func->inCasts_.push_back(input3);
    func->inCasts_.push_back(input4);
    func->outCasts_.push_back(output1);
    func->outCasts_.push_back(output2);

    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("SplitReshapeTestStrategy", {
        {   "SplitReshape",   "SplitReshape",  PassType::TYPE_TILE_GRAPH},
    });
    EXPECT_EQ(passManager.RunPass(Program::GetInstance(), *func, "SplitReshapeTestStrategy"), SUCCESS);
    
    int reshapeOp = 0;
    int assembleOp = 0;
    int viewOp = 0;
    Operation *newAssemble1;
    Operation *newAssemble2;
    Operation *newAssemble3;
    Operation *newAssemble4;
    for (auto &op : func->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_RESHAPE) {
            reshapeOp++;
        } else if (op->GetOpcode() == Opcode::OP_ASSEMBLE) {
            if (op->GetInputOperand(kSizeZero) == input1) {
                newAssemble1 = op;
                assembleOp += 1;
            } else if (op->GetInputOperand(kSizeZero) == input2) {
                newAssemble2 = op;
                assembleOp += 10;
            } else if (op->GetInputOperand(kSizeZero) == input3) {
                newAssemble3 = op;
                assembleOp += 100;
            } else if (op->GetInputOperand(kSizeZero) == input4) {
                newAssemble4 = op;
                assembleOp += 1000;
            }
        } else if (op->GetOpcode() == Opcode::OP_VIEW) {
            viewOp++;
        }
    }
    EXPECT_EQ(reshapeOp, kNumTwo);
    EXPECT_EQ(viewOp, kNumTwo);

    EXPECT_EQ(assembleOp, 1111);
    std::vector<SymbolicScalar> dynAssembleOffset1 = {kNumZero, kNumZero, SymbolicScalar("b")};
    std::vector<SymbolicScalar> dynAssembleOffset2 = {kNumZero, kNumTwo, SymbolicScalar("b")};
    std::vector<SymbolicScalar> dynAssembleOffset3 = {kNumZero, kNumFour, SymbolicScalar("b")};
    std::vector<SymbolicScalar> dynAssembleOffset4 = {kNumZero, kNumSix, SymbolicScalar("b")};
    auto assembleAttr1 = dynamic_cast<AssembleOpAttribute *>(newAssemble1->GetOpAttribute().get());
    EXPECT_EQ(assembleAttr1->GetToDynOffset().size(), kNumThree);
    auto assembleAttr2 = dynamic_cast<AssembleOpAttribute *>(newAssemble2->GetOpAttribute().get());
    EXPECT_EQ(assembleAttr2->GetToDynOffset().size(), kNumThree); 
    auto assembleAttr3 = dynamic_cast<AssembleOpAttribute *>(newAssemble3->GetOpAttribute().get());
    EXPECT_EQ(assembleAttr3->GetToDynOffset().size(), kNumThree); 
    auto assembleAttr4 = dynamic_cast<AssembleOpAttribute *>(newAssemble4->GetOpAttribute().get());
    EXPECT_EQ(assembleAttr4->GetToDynOffset().size(), kNumThree); 
    for (size_t i = 0; i < kNumThree; ++i) {
        EXPECT_EQ(assembleAttr1->GetToDynOffset()[i].Dump(), dynAssembleOffset1[i].Dump());
        EXPECT_EQ(assembleAttr2->GetToDynOffset()[i].Dump(), dynAssembleOffset2[i].Dump());
        EXPECT_EQ(assembleAttr3->GetToDynOffset()[i].Dump(), dynAssembleOffset3[i].Dump());
        EXPECT_EQ(assembleAttr4->GetToDynOffset()[i].Dump(), dynAssembleOffset4[i].Dump());
    }

    std::vector<SymbolicScalar> dynReshapeInputShape = {kNumTwo, kNumFour, SymbolicScalar("a") * 1};
    std::vector<SymbolicScalar> dynReshapeInputOffset1 = {kNumZero, kNumZero, SymbolicScalar("b") * 1};
    std::vector<SymbolicScalar> dynReshapeInputOffset2 = {kNumZero, kNumFour, SymbolicScalar("b") * 1};
    std::vector<SymbolicScalar> dynReshapeOutputShape = {kNumTwo, kNumTwo, kNumTwo, SymbolicScalar("a")};
    std::vector<SymbolicScalar> dynReshapeOutputOffset1 = {kNumZero, kNumZero, kNumZero, SymbolicScalar("b")};
    std::vector<SymbolicScalar> dynReshapeOutputOffset2 = {kNumZero, kNumTwo, kNumZero, SymbolicScalar("b")};

    auto reshapeSource1 = newAssemble1->GetOutputOperand(kSizeZero);
    auto reshapeSource2 = newAssemble4->GetOutputOperand(kSizeZero);
    EXPECT_NE(reshapeSource1, reshapeSource2);
    EXPECT_EQ(reshapeSource1->dynOffset_.size(), kNumThree);
    EXPECT_EQ(reshapeSource1->dynValidShape_.size(), kNumThree);
    EXPECT_EQ(reshapeSource2->dynOffset_.size(), kNumThree);
    EXPECT_EQ(reshapeSource2->dynValidShape_.size(), kNumThree);
    for (size_t i = 0; i < kNumThree; ++i) {
        EXPECT_EQ(reshapeSource1->dynValidShape_[i].Dump(), dynReshapeInputShape[i].Dump());
        EXPECT_EQ(reshapeSource1->dynOffset_[i].Dump(), dynReshapeInputOffset1[i].Dump());
        EXPECT_EQ(reshapeSource2->dynValidShape_[i].Dump(), dynReshapeInputShape[i].Dump());
        EXPECT_EQ(reshapeSource2->dynOffset_[i].Dump(), dynReshapeInputOffset2[i].Dump());
    }
    auto reshape1 = *(reshapeSource1->GetConsumers().begin());
    auto reshape2 = *(reshapeSource2->GetConsumers().begin());
    EXPECT_NE(reshape1, reshape2);
    auto reshapeOutput1 = reshape1->GetOutputOperand(kSizeZero);
    auto reshapeOutput2 = reshape2->GetOutputOperand(kSizeZero);
    EXPECT_NE(reshapeOutput1, reshapeOutput2);
    EXPECT_EQ(reshapeOutput1->dynOffset_.size(), kNumFour);
    EXPECT_EQ(reshapeOutput1->dynValidShape_.size(), kNumFour);
    EXPECT_EQ(reshapeOutput2->dynOffset_.size(), kNumFour);
    EXPECT_EQ(reshapeOutput2->dynValidShape_.size(), kNumFour);
    for (size_t i = 0; i < kNumFour; ++i) {
        EXPECT_EQ(reshapeOutput1->dynValidShape_[i].Dump(), dynReshapeOutputShape[i].Dump());
        EXPECT_EQ(reshapeOutput1->dynOffset_[i].Dump(), dynReshapeOutputOffset1[i].Dump());
        EXPECT_EQ(reshapeOutput2->dynValidShape_[i].Dump(), dynReshapeOutputShape[i].Dump());
        EXPECT_EQ(reshapeOutput2->dynOffset_[i].Dump(), dynReshapeOutputOffset2[i].Dump());
    }
    EXPECT_EQ(reshapeOutput1->GetConsumers().size(), kNumOne);
    EXPECT_EQ(reshapeOutput2->GetConsumers().size(), kNumOne);
    auto view1 = *(reshapeOutput1->GetConsumers().begin());
    auto view2 = *(reshapeOutput2->GetConsumers().begin());
    EXPECT_NE(view1, view2);
    
    std::vector<SymbolicScalar> dynViewOffset1 = {kNumZero, kNumZero, kNumZero, SymbolicScalar("b")};
    std::vector<SymbolicScalar> dynViewOffset2 = {kNumZero, kNumTwo, kNumZero, SymbolicScalar("b")};
    auto viewOpAttribute1 = dynamic_cast<ViewOpAttribute *>(view1->GetOpAttribute().get());
    auto viewOpAttribute2 = dynamic_cast<ViewOpAttribute *>(view2->GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute1->GetFromDynOffset().size(), kNumFour);
    EXPECT_EQ(viewOpAttribute2->GetFromDynOffset().size(), kNumFour);
    for (size_t i = 0; i < kNumFour; ++i) {
        EXPECT_EQ(viewOpAttribute1->GetFromDynOffset()[i].Dump(), dynViewOffset1[i].Dump());
        EXPECT_EQ(viewOpAttribute2->GetFromDynOffset()[i].Dump(), dynViewOffset2[i].Dump());
    }
}

/*
验证多对一场景下动态shape的完整兜底策略
因为缺乏宏构建策略，手动构造expandfunction的输出构图
1) 用例设置：
{2,4} -> assemble -> {2,16} -> reshape -> {1,2,2,8} -> view -> {1,2,2,4}
{2,4} -> assemble                                   -> view -> {1,2,2,4}
{2,4} -> assemble
{2,4} -> assemble
2) splitreshape
{2,4} -> assemble -> {2,4} -> reshape -> {1,2,1,4} -> assemble 
{2,4} -> assemble -> {2,4} -> reshape -> {1,2,1,4} -> assemble -> {1,2,2,4} -> view -> {1,2,2,4}
{2,4} -> assemble -> {2,4} -> reshape -> {1,2,1,4} -> assemble -> {1,2,2,4} -> view -> {1,2,2,4}
{2,4} -> assemble -> {2,4} -> reshape -> {1,2,1,4} -> assemble
*/
TEST_F(TestSplitReshapePass, TestDynAssembleAfterReshapeSTest) {
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(func != nullptr);

    std::vector<int> shape1 = {kNumTwo, kNumFour};
    std::vector<int> shape2 = {kNumTwo, kExpFour};
    std::vector<int> shape3 = {kNumOne, kNumTwo, kNumTwo, kNumEight};
    std::vector<int> shape4 = {kNumOne, kNumTwo, kNumTwo, kNumFour};
    std::vector<int> assembleOffset1 = {kNumZero, kNumZero};
    std::vector<int> assembleOffset2 = {kNumZero, kNumFour};
    std::vector<int> assembleOffset3 = {kNumZero, kNumEight};
    std::vector<int> assembleOffset4 = {kNumZero, kNumTwelve};
    std::vector<int> viewOffset1 = {kNumZero, kNumZero, kNumZero, kNumZero};
    std::vector<int> viewOffset2 = {kNumZero, kNumZero, kNumZero, kNumFour};
    
    std::vector<SymbolicScalar> dynInputShape = {SymbolicScalar("a"), kNumFour};
    std::vector<SymbolicScalar> dynInputOffset1 = {SymbolicScalar("b"), kNumZero};
    std::vector<SymbolicScalar> dynInputOffset2 = {SymbolicScalar("b"), kNumFour};
    std::vector<SymbolicScalar> dynInputOffset3 = {SymbolicScalar("b"), kNumEight};
    std::vector<SymbolicScalar> dynInputOffset4 = {SymbolicScalar("b"), kNumTwelve};
    std::vector<SymbolicScalar> dynOutputShape = {kNumOne, SymbolicScalar("a"), kNumTwo, kNumFour};
    std::vector<SymbolicScalar> dynOutputOffset1 = {kNumZero, SymbolicScalar("b"), kNumZero, kNumZero};
    std::vector<SymbolicScalar> dynOutputOffset2 = {kNumZero, SymbolicScalar("b"), kNumZero, kNumFour};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape2);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape3);
    auto input1 = std::make_shared<LogicalTensor>(*func, ddrRawTensor1, assembleOffset1, shape1, dynInputShape);
    input1->UpdateOffset(TensorOffset(assembleOffset1, dynInputOffset1));
    input1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto input2 = std::make_shared<LogicalTensor>(*func, ddrRawTensor1, assembleOffset2, shape1, dynInputShape);
    input2->UpdateOffset(TensorOffset(assembleOffset2, dynInputOffset2));
    input2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto input3 = std::make_shared<LogicalTensor>(*func, ddrRawTensor1, assembleOffset3, shape1, dynInputShape);
    input3->UpdateOffset(TensorOffset(assembleOffset3, dynInputOffset3));
    input3->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto input4 = std::make_shared<LogicalTensor>(*func, ddrRawTensor1, assembleOffset4, shape1, dynInputShape);
    input4->UpdateOffset(TensorOffset(assembleOffset4, dynInputOffset4));
    input4->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*func, DT_FP32, shape2);
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*func, DT_FP32, shape3);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output1 = std::make_shared<LogicalTensor>(*func, ddrRawTensor2, viewOffset1, shape4, dynOutputShape);
    output1->UpdateOffset(TensorOffset(viewOffset1, dynOutputOffset1));
    output1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output2 = std::make_shared<LogicalTensor>(*func, ddrRawTensor2, viewOffset2, shape4, dynOutputShape);
    output2->UpdateOffset(TensorOffset(viewOffset2, dynOutputOffset2));
    output2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    
    auto &assemble_op1 = func->AddOperation(Opcode::OP_ASSEMBLE, {input1}, {ubTensor1});
    auto assemble_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset1, dynInputOffset1);
    assemble_op1.SetOpAttribute(assemble_Attr1);
    auto &assemble_op2 = func->AddOperation(Opcode::OP_ASSEMBLE, {input2}, {ubTensor1});
    auto assemble_Attr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset2, dynInputOffset2);
    assemble_op2.SetOpAttribute(assemble_Attr2);
    auto &assemble_op3 = func->AddOperation(Opcode::OP_ASSEMBLE, {input3}, {ubTensor1});
    auto assemble_Attr3 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset3, dynInputOffset3);
    assemble_op3.SetOpAttribute(assemble_Attr3);
    auto &assemble_op4 = func->AddOperation(Opcode::OP_ASSEMBLE, {input4}, {ubTensor1});
    auto assemble_Attr4 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset4, dynInputOffset4);
    assemble_op4.SetOpAttribute(assemble_Attr4);
    func->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &view_op1 = func->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output1});
    auto view_Attr1 = std::make_shared<ViewOpAttribute>(viewOffset1);
    view_op1.SetOpAttribute(view_Attr1);
    auto &view_op2 = func->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output2});
    auto view_Attr2 = std::make_shared<ViewOpAttribute>(viewOffset2);
    view_op2.SetOpAttribute(view_Attr2);
    
    func->inCasts_.push_back(input1);
    func->inCasts_.push_back(input2);
    func->inCasts_.push_back(input3);
    func->inCasts_.push_back(input4);
    func->outCasts_.push_back(output1);
    func->outCasts_.push_back(output2);

    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("SplitReshapeTestStrategy", {
        {   "SplitReshape",   "SplitReshape",  PassType::TYPE_TILE_GRAPH},
    });
    EXPECT_EQ(passManager.RunPass(Program::GetInstance(), *func, "SplitReshapeTestStrategy"), SUCCESS);
    
    int reshapeOp = 0;
    int assembleOp = 0;
    int assembleCnt = 0;
    int viewOp = 0;
    Operation *newAssemble1;
    Operation *newAssemble2;
    Operation *newAssemble3;
    Operation *newAssemble4;
    for (auto &op : func->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_RESHAPE) {
            reshapeOp++;
        } else if (op->GetOpcode() == Opcode::OP_ASSEMBLE) {
            if (op->GetInputOperand(kSizeZero) == input1) {
                newAssemble1 = op;
                assembleOp += 1;
            } else if (op->GetInputOperand(kSizeZero) == input2) {
                newAssemble2 = op;
                assembleOp += 10;
            } else if (op->GetInputOperand(kSizeZero) == input3) {
                newAssemble3 = op;
                assembleOp += 100;
            } else if (op->GetInputOperand(kSizeZero) == input4) {
                newAssemble4 = op;
                assembleOp += 1000;
            }
            assembleCnt++;
        } else if (op->GetOpcode() == Opcode::OP_VIEW) {
            viewOp++;
        }
    }
    EXPECT_EQ(assembleCnt, kNumEight);
    EXPECT_EQ(reshapeOp, kNumFour);
    EXPECT_EQ(viewOp, kNumTwo);

    EXPECT_EQ(assembleOp, 1111);
    std::vector<SymbolicScalar> dynAssembleOffset1 = {SymbolicScalar("b"), kNumZero};
    std::vector<SymbolicScalar> dynAssembleOffset2 = {SymbolicScalar("b"), kNumFour};
    std::vector<SymbolicScalar> dynAssembleOffset3 = {SymbolicScalar("b"), kNumEight};
    std::vector<SymbolicScalar> dynAssembleOffset4 = {SymbolicScalar("b"), kNumTwelve};
    auto assembleAttr1 = dynamic_cast<AssembleOpAttribute *>(newAssemble1->GetOpAttribute().get());
    EXPECT_EQ(assembleAttr1->GetToDynOffset().size(), kNumTwo);
    auto assembleAttr2 = dynamic_cast<AssembleOpAttribute *>(newAssemble2->GetOpAttribute().get());
    EXPECT_EQ(assembleAttr2->GetToDynOffset().size(), kNumTwo); 
    auto assembleAttr3 = dynamic_cast<AssembleOpAttribute *>(newAssemble3->GetOpAttribute().get());
    EXPECT_EQ(assembleAttr3->GetToDynOffset().size(), kNumTwo); 
    auto assembleAttr4 = dynamic_cast<AssembleOpAttribute *>(newAssemble4->GetOpAttribute().get());
    EXPECT_EQ(assembleAttr4->GetToDynOffset().size(), kNumTwo); 
    for (size_t i = 0; i < kNumTwo; ++i) {
        EXPECT_EQ(assembleAttr1->GetToDynOffset()[i].Dump(), dynAssembleOffset1[i].Dump());
        EXPECT_EQ(assembleAttr2->GetToDynOffset()[i].Dump(), dynAssembleOffset2[i].Dump());
        EXPECT_EQ(assembleAttr3->GetToDynOffset()[i].Dump(), dynAssembleOffset3[i].Dump());
        EXPECT_EQ(assembleAttr4->GetToDynOffset()[i].Dump(), dynAssembleOffset4[i].Dump());
    }

    std::vector<SymbolicScalar> dynReshapeInputShape = {SymbolicScalar("a"), kNumFour};
    std::vector<SymbolicScalar> dynReshapeInputOffset1 = {SymbolicScalar("b"), kNumZero};
    std::vector<SymbolicScalar> dynReshapeInputOffset2 = {SymbolicScalar("b"), kNumFour};
    std::vector<SymbolicScalar> dynReshapeInputOffset3 = {SymbolicScalar("b"), kNumEight};
    std::vector<SymbolicScalar> dynReshapeInputOffset4 = {SymbolicScalar("b"), kNumTwelve};
    std::vector<SymbolicScalar> dynReshapeOutputShape = {kNumOne, SymbolicScalar("a") * 1, kNumOne, kNumFour};
    std::vector<SymbolicScalar> dynReshapeOutputOffset1 = {kNumZero, SymbolicScalar("b") * 1, kNumZero, kNumZero};
    std::vector<SymbolicScalar> dynReshapeOutputOffset2 = {kNumZero, SymbolicScalar("b") * 1, kNumZero, kNumFour};
    std::vector<SymbolicScalar> dynReshapeOutputOffset3 = {kNumZero, SymbolicScalar("b") * 1, kNumOne, kNumZero};
    std::vector<SymbolicScalar> dynReshapeOutputOffset4 = {kNumZero, SymbolicScalar("b") * 1, kNumOne, kNumFour};

    auto reshapeSource1 = newAssemble1->GetOutputOperand(kSizeZero);
    auto reshapeSource2 = newAssemble2->GetOutputOperand(kSizeZero);
    auto reshapeSource3 = newAssemble3->GetOutputOperand(kSizeZero);
    auto reshapeSource4 = newAssemble4->GetOutputOperand(kSizeZero);
    EXPECT_NE(reshapeSource1, reshapeSource2);
    EXPECT_NE(reshapeSource1, reshapeSource3);
    EXPECT_NE(reshapeSource1, reshapeSource4);
    EXPECT_EQ(reshapeSource1->dynOffset_.size(), kNumTwo);
    EXPECT_EQ(reshapeSource1->dynValidShape_.size(), kNumTwo);
    EXPECT_EQ(reshapeSource2->dynOffset_.size(), kNumTwo);
    EXPECT_EQ(reshapeSource2->dynValidShape_.size(), kNumTwo);
    EXPECT_EQ(reshapeSource3->dynOffset_.size(), kNumTwo);
    EXPECT_EQ(reshapeSource3->dynValidShape_.size(), kNumTwo);
    EXPECT_EQ(reshapeSource4->dynOffset_.size(), kNumTwo);
    EXPECT_EQ(reshapeSource4->dynValidShape_.size(), kNumTwo);
    for (size_t i = 0; i < kNumTwo; ++i) {
        EXPECT_EQ(reshapeSource1->dynValidShape_[i].Dump(), dynReshapeInputShape[i].Dump());
        EXPECT_EQ(reshapeSource1->dynOffset_[i].Dump(), dynReshapeInputOffset1[i].Dump());
        EXPECT_EQ(reshapeSource2->dynValidShape_[i].Dump(), dynReshapeInputShape[i].Dump());
        EXPECT_EQ(reshapeSource2->dynOffset_[i].Dump(), dynReshapeInputOffset2[i].Dump());
        EXPECT_EQ(reshapeSource3->dynValidShape_[i].Dump(), dynReshapeInputShape[i].Dump());
        EXPECT_EQ(reshapeSource3->dynOffset_[i].Dump(), dynReshapeInputOffset3[i].Dump());
        EXPECT_EQ(reshapeSource4->dynValidShape_[i].Dump(), dynReshapeInputShape[i].Dump());
        EXPECT_EQ(reshapeSource4->dynOffset_[i].Dump(), dynReshapeInputOffset4[i].Dump());
    }
    auto reshape1 = *(reshapeSource1->GetConsumers().begin());
    auto reshape2 = *(reshapeSource2->GetConsumers().begin());
    auto reshape3 = *(reshapeSource3->GetConsumers().begin());
    auto reshape4 = *(reshapeSource4->GetConsumers().begin());
    EXPECT_NE(reshape1, reshape2);
    EXPECT_NE(reshape1, reshape3);
    EXPECT_NE(reshape1, reshape4);
    auto reshapeOutput1 = reshape1->GetOutputOperand(kSizeZero);
    auto reshapeOutput2 = reshape2->GetOutputOperand(kSizeZero);
    auto reshapeOutput3 = reshape3->GetOutputOperand(kSizeZero);
    auto reshapeOutput4 = reshape4->GetOutputOperand(kSizeZero);
    EXPECT_NE(reshapeOutput1, reshapeOutput2);
    EXPECT_NE(reshapeOutput1, reshapeOutput3);
    EXPECT_NE(reshapeOutput1, reshapeOutput4);
    EXPECT_EQ(reshapeOutput1->dynOffset_.size(), kNumFour);
    EXPECT_EQ(reshapeOutput1->dynValidShape_.size(), kNumFour);
    EXPECT_EQ(reshapeOutput2->dynOffset_.size(), kNumFour);
    EXPECT_EQ(reshapeOutput2->dynValidShape_.size(), kNumFour);
    EXPECT_EQ(reshapeOutput3->dynOffset_.size(), kNumFour);
    EXPECT_EQ(reshapeOutput3->dynValidShape_.size(), kNumFour);
    EXPECT_EQ(reshapeOutput4->dynOffset_.size(), kNumFour);
    EXPECT_EQ(reshapeOutput4->dynValidShape_.size(), kNumFour);
    for (size_t i = 0; i < kNumFour; ++i) {
        EXPECT_EQ(reshapeOutput1->dynValidShape_[i].Dump(), dynReshapeOutputShape[i].Dump());
        EXPECT_EQ(reshapeOutput1->dynOffset_[i].Dump(), dynReshapeOutputOffset1[i].Dump());
        EXPECT_EQ(reshapeOutput2->dynValidShape_[i].Dump(), dynReshapeOutputShape[i].Dump());
        EXPECT_EQ(reshapeOutput2->dynOffset_[i].Dump(), dynReshapeOutputOffset2[i].Dump());
        EXPECT_EQ(reshapeOutput3->dynValidShape_[i].Dump(), dynReshapeOutputShape[i].Dump());
        EXPECT_EQ(reshapeOutput3->dynOffset_[i].Dump(), dynReshapeOutputOffset3[i].Dump());
        EXPECT_EQ(reshapeOutput4->dynValidShape_[i].Dump(), dynReshapeOutputShape[i].Dump());
        EXPECT_EQ(reshapeOutput4->dynOffset_[i].Dump(), dynReshapeOutputOffset4[i].Dump());
    }
    auto afterAssemble1 = *(reshapeOutput1->GetConsumers().begin());
    auto afterAssemble2 = *(reshapeOutput2->GetConsumers().begin());
    auto afterAssemble3 = *(reshapeOutput3->GetConsumers().begin());
    auto afterAssemble4 = *(reshapeOutput4->GetConsumers().begin());
    auto assembleOpAttribute1 = dynamic_cast<AssembleOpAttribute *>(afterAssemble1->GetOpAttribute().get());
    auto assembleOpAttribute2 = dynamic_cast<AssembleOpAttribute *>(afterAssemble2->GetOpAttribute().get());
    auto assembleOpAttribute3 = dynamic_cast<AssembleOpAttribute *>(afterAssemble3->GetOpAttribute().get());
    auto assembleOpAttribute4 = dynamic_cast<AssembleOpAttribute *>(afterAssemble4->GetOpAttribute().get());
    EXPECT_EQ(assembleOpAttribute1->GetToDynOffset().size(), kNumFour);
    EXPECT_EQ(assembleOpAttribute2->GetToDynOffset().size(), kNumFour);
    EXPECT_EQ(assembleOpAttribute3->GetToDynOffset().size(), kNumFour);
    EXPECT_EQ(assembleOpAttribute4->GetToDynOffset().size(), kNumFour);
    std::vector<SymbolicScalar> dynAssembleAttrOffset1 = {kNumZero, SymbolicScalar("b") * 1 - SymbolicScalar("b"), kNumZero, kNumZero};
    std::vector<SymbolicScalar> dynAssembleAttrOffset2 = {kNumZero, SymbolicScalar("b") * 1 - SymbolicScalar("b"), kNumOne, kNumZero};
    for (size_t i = 0; i < kNumFour; ++i) {
        EXPECT_EQ(assembleOpAttribute1->GetToDynOffset()[i].Dump(), dynAssembleAttrOffset1[i].Dump());
        EXPECT_EQ(assembleOpAttribute2->GetToDynOffset()[i].Dump(), dynAssembleAttrOffset1[i].Dump());
        EXPECT_EQ(assembleOpAttribute3->GetToDynOffset()[i].Dump(), dynAssembleAttrOffset2[i].Dump());
        EXPECT_EQ(assembleOpAttribute4->GetToDynOffset()[i].Dump(), dynAssembleAttrOffset2[i].Dump());
    }
    auto assembleOutput1 = afterAssemble1->GetOutputOperand(kSizeZero);
    auto assembleOutput2 = afterAssemble2->GetOutputOperand(kSizeZero);
    auto assembleOutput3 = afterAssemble3->GetOutputOperand(kSizeZero);
    auto assembleOutput4 = afterAssemble4->GetOutputOperand(kSizeZero);
    EXPECT_EQ(assembleOutput1, assembleOutput3);
    EXPECT_EQ(assembleOutput2, assembleOutput4);
    EXPECT_EQ(assembleOutput1->GetConsumers().size(), kNumOne);
    EXPECT_EQ(assembleOutput2->GetConsumers().size(), kNumOne);
    EXPECT_EQ(assembleOutput1->dynOffset_.size(), kNumFour);
    EXPECT_EQ(assembleOutput1->dynValidShape_.size(), kNumFour);
    EXPECT_EQ(assembleOutput2->dynOffset_.size(), kNumFour);
    EXPECT_EQ(assembleOutput2->dynValidShape_.size(), kNumFour);
    for (size_t i = 0; i < kNumFour; ++i) {
        EXPECT_EQ(assembleOutput1->dynValidShape_[i].Dump(), dynOutputShape[i].Dump());
        EXPECT_EQ(assembleOutput1->dynOffset_[i].Dump(), dynOutputOffset1[i].Dump());
        EXPECT_EQ(assembleOutput2->dynValidShape_[i].Dump(), dynOutputShape[i].Dump());
        EXPECT_EQ(assembleOutput2->dynOffset_[i].Dump(), dynOutputOffset2[i].Dump());
    }
    auto view1 = *(assembleOutput1->GetConsumers().begin());
    auto view2 = *(assembleOutput2->GetConsumers().begin());
    EXPECT_NE(view1, view2);
    std::vector<SymbolicScalar> dynViewOffset1 = {kNumZero, SymbolicScalar("b"), kNumZero, kNumZero};
    std::vector<SymbolicScalar> dynViewOffset2 = {kNumZero, SymbolicScalar("b"), kNumZero, kNumFour};
    auto viewOpAttribute1 = dynamic_cast<ViewOpAttribute *>(view1->GetOpAttribute().get());
    auto viewOpAttribute2 = dynamic_cast<ViewOpAttribute *>(view2->GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute1->GetFromDynOffset().size(), kNumFour);
    EXPECT_EQ(viewOpAttribute2->GetFromDynOffset().size(), kNumFour);
    for (size_t i = 0; i < kNumFour; ++i) {
        EXPECT_EQ(viewOpAttribute1->GetFromDynOffset()[i].Dump(), dynViewOffset1[i].Dump());
        EXPECT_EQ(viewOpAttribute2->GetFromDynOffset()[i].Dump(), dynViewOffset2[i].Dump());
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
TEST_F(TestSplitReshapePass, TestExceptionCase1) {
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
TEST_F(TestSplitReshapePass, TestExceptionCase2) {
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
TEST_F(TestSplitReshapePass, TestExceptionCase3) {
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

/*
splitreshape pass不起作用的场景
动态shape位于变化轴
{2,2,2} -> assemble -> {2,2,4} -> reshape -> {2,2,2,2} -> view -> {2,2,1,2}
{2,2,2} -> assemble                                    -> view -> {2,2,1,2}
{2,2,a}/{0,0,b1}
{2,2,a}/{0,0,b2}  
*/
TEST_F(TestSplitReshapePass, TestExceptionCase4) {
    //Define the shape of the Tensors
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(func != nullptr);

    std::vector<int> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> shape2 = {kNumTwo, kNumTwo, kNumFour};
    std::vector<int> shape3 = {kNumTwo, kNumTwo, kNumTwo, kNumTwo};
    std::vector<int> shape4 = {kNumTwo, kNumTwo, kNumOne, kNumTwo};
    std::vector<int> assembleOffset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int> assembleOffset2 = {kNumZero, kNumZero, kNumTwo};
    std::vector<int> viewOffset1 = {kNumZero, kNumZero, kNumZero, kNumZero};
    std::vector<int> viewOffset2 = {kNumZero, kNumZero, kNumOne, kNumZero};
    
    std::vector<SymbolicScalar> dynInputShape = {kNumTwo, kNumTwo, SymbolicScalar("a")};
    std::vector<SymbolicScalar> dynInputOffset1 = {kNumZero, kNumZero, SymbolicScalar("b0")};
    std::vector<SymbolicScalar> dynInputOffset2 = {kNumZero, kNumZero, SymbolicScalar("b1")};
    
    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape2);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape3);
    auto input1 = std::make_shared<LogicalTensor>(*func, ddrRawTensor1, assembleOffset1, shape1, dynInputShape);
    input1->UpdateOffset(TensorOffset(assembleOffset1, dynInputOffset1));
    input1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto input2 = std::make_shared<LogicalTensor>(*func, ddrRawTensor1, assembleOffset2, shape1, dynInputShape);
    input2->UpdateOffset(TensorOffset(assembleOffset2, dynInputOffset2));
    input2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*func, DT_FP32, shape2);
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*func, DT_FP32, shape3);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output1 = std::make_shared<LogicalTensor>(*func, ddrRawTensor2, viewOffset1, shape4);
    output1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output2 = std::make_shared<LogicalTensor>(*func, ddrRawTensor2, viewOffset2, shape4);
    output2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    
    auto &assemble_op1 = func->AddOperation(Opcode::OP_ASSEMBLE, {input1}, {ubTensor1});
    auto assemble_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset1, dynInputOffset1);
    assemble_op1.SetOpAttribute(assemble_Attr1);
    auto &assemble_op2 = func->AddOperation(Opcode::OP_ASSEMBLE, {input2}, {ubTensor1});
    auto assemble_Attr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset2, dynInputOffset2);
    assemble_op2.SetOpAttribute(assemble_Attr2);
    func->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &view_op1 = func->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output1});
    auto view_Attr1 = std::make_shared<ViewOpAttribute>(viewOffset1);
    view_op1.SetOpAttribute(view_Attr1);
    auto &view_op2 = func->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output2});
    auto view_Attr2 = std::make_shared<ViewOpAttribute>(viewOffset2);
    view_op2.SetOpAttribute(view_Attr2);
    
    func->inCasts_.push_back(input1);
    func->inCasts_.push_back(input2);
    func->outCasts_.push_back(output1);
    func->outCasts_.push_back(output2);

    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("SplitReshapeTestStrategy", {
        {   "SplitReshape",   "SplitReshape",  PassType::TYPE_TILE_GRAPH},
    });

    int reshapeOp = 0;
    for (auto &op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            reshapeOp++;
        }
    }
    EXPECT_EQ(reshapeOp, kNumOne);

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