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
 * \file test_infer_shape.cpp
 * \brief
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/operation/op_infer_shape_impl.h"
#include "passes/tile_graph_pass/infer_dyn_shape.h"
#include "interface/operation/attribute.h"

using namespace npu::tile_fwk;

class InferShapeTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
    }

    void TearDown() override {}
};

TEST_F(InferShapeTest, TestAdd) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestAddInferShape", "TestAddInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto incast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    outCast->UpdateDynValidShape({SymbolicScalar("output_0_Dim_0"), SymbolicScalar("output_0_Dim_1")});

    auto &copy_op1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate(SymbolicScalar("Input_0_Dim_0")), OpImmediate(SymbolicScalar("Input_0_Dim_1"))};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetOpAttribute(copyin1Attr);

    auto &copy_op2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast2}, {ubTensor2});
    auto copyin2Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape1 = {OpImmediate(SymbolicScalar("Input_1_Dim_0")), OpImmediate(SymbolicScalar("Input_1_Dim_1"))};
    copyin2Attr->SetToDynValidShape(toValidShape1);
    copy_op2.SetOpAttribute(copyin2Attr);

    auto& add_op = currFunctionPtr->AddOperation(Opcode::OP_ADD, {ubTensor1, ubTensor2}, {ubTensor3});
    (void) add_op;
    auto& copy_out_op = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {ubTensor3}, {outCast});
    (void) copy_out_op;
    

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outCast);

    InferDynShapePass inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestAddAlignCase) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestAddInferShape", "TestAddInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto incast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    auto &copy_op1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    copy_op1.SetOpAttribute(copyin1Attr);

    auto &copy_op2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast2}, {ubTensor2});
    auto copyin2Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    copy_op2.SetOpAttribute(copyin2Attr);

    auto& add_op = currFunctionPtr->AddOperation(Opcode::OP_ADD, {ubTensor1, ubTensor2}, {ubTensor3});
    (void) add_op;
    auto& copy_out_op = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {ubTensor3}, {outCast});
    auto copyoutAttr = std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}), shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    copy_out_op.SetOpAttribute(copyoutAttr);
    

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outCast);

    InferDynShapePass inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestAddExp) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestAddInferShape", "TestAddInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto incast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    outCast->UpdateDynValidShape({SymbolicScalar("output_0_Dim_0"), SymbolicScalar("output_0_Dim_1")});

    auto &copy_op1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate(SymbolicScalar("Input_0_Dim_0")), OpImmediate(SymbolicScalar("Input_0_Dim_1"))};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetOpAttribute(copyin1Attr);

    auto &copy_op2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast2}, {ubTensor2});
    auto copyin2Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme);
    std::vector<npu::tile_fwk::OpImmediate> toValidShape1 = {OpImmediate(SymbolicScalar("Input_1_Dim_0")), OpImmediate(SymbolicScalar("Input_1_Dim_1"))};
    copyin2Attr->SetToDynValidShape(toValidShape1);
    copy_op2.SetOpAttribute(copyin2Attr);

    auto& add_op = currFunctionPtr->AddOperation(Opcode::OP_ADD, {ubTensor1, ubTensor2}, {ubTensor3});
    (void) add_op;
    auto tmpCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto& copy_out_op = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {ubTensor3}, {tmpCast});
    auto copyout1Attr = std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}), shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    copy_out_op.SetOpAttribute(copyout1Attr);

    auto ubTensor4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto &copy_op3 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {tmpCast}, {ubTensor4});
    auto copyin3Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    copy_op3.SetOpAttribute(copyin3Attr);

    auto ubTensor5 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto &exp = currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor4}, {ubTensor5});
    (void) exp;

    auto& copy_out_op1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {ubTensor5}, {outCast});
    (void) copy_out_op1;

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outCast);

    InferDynShapePass inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}
