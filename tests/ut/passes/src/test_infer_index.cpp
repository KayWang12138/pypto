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
 * \file test_infer_index.cpp
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
#include "passes/block_graph_pass/infer_param_index.h"

namespace npu {
namespace tile_fwk {
class InferIndexTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
    }

    void TearDown() override {}
};

TEST_F(InferIndexTest, TestReset) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(),
                                                      "TestReset",
                                                      "TestReset",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {8, 16};
    std::vector<int64_t> outshape = {0, 0};
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inshape);
    auto outcast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, outshape);

    auto &copyin_op = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {outcast});
    (void) copyin_op;

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferParamIndex inferParamIndex;
    std::cout << currFunctionPtr->Dump() << std::endl;
    std::cout << copyin_op.GetOOperands()[0]->Dump() << std::endl;
    EXPECT_EQ(inferParamIndex.ResetDynValidShape(*currFunctionPtr), SUCCESS);
}

TEST_F(InferIndexTest, TestResetNoneOp) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(),
                                                      "TestReset",
                                                      "TestReset",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    InferParamIndex inferParamIndex;
    EXPECT_EQ(inferParamIndex.ResetDynValidShape(*currFunctionPtr), SUCCESS);
}

TEST_F(InferIndexTest, TestResetNoneOut) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(),
                                                      "TestReset",
                                                      "TestReset",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {8, 16};
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inshape);

    auto &copyin_op = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {});
    (void) copyin_op;
    currFunctionPtr->inCasts_.push_back(incast);

    InferParamIndex inferParamIndex;
    EXPECT_EQ(inferParamIndex.ResetDynValidShape(*currFunctionPtr), FAILED);
}

TEST_F(InferIndexTest, TestResetView) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(),
                                                      "TestReset",
                                                      "TestReset",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {8, 16};
    std::vector<int64_t> offset = {2, 0};

    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inshape);
    auto outcast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inshape);
    incast->UpdateDynValidShape({SymbolicScalar("input_0_Dim_0"), SymbolicScalar("input_0_Dim_1")});

    auto &view_op = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {incast}, {outcast});
    auto viewAttr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>(),
                                                      MEM_UNKNOWN,
                                                      std::vector<SymbolicScalar>(),
                                                      std::vector<SymbolicScalar>());
    viewAttr->SetFromOffset(std::vector<int64_t>(),
                            {SymbolicScalar("Offset_0_Dim_0"),
                             SymbolicScalar("Offset_0_Dim_1")});
    view_op.SetOpAttribute(viewAttr);

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferParamIndex inferParamIndex;
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferParamIndex.ResetDynValidShape(*currFunctionPtr), SUCCESS);
}


TEST_F(InferIndexTest, TestInferShape) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(),
                                                      "TestInferShape",
                                                      "TestInferShape",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {8, 16};
    auto shapeImme = OpImmediate::Specified(inshape);
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inshape);
    auto outcast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inshape);

    auto &copyin_op = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {outcast});
    auto copyin_attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}),
                                                         MEM_UB,
                                                         shapeImme,
                                                         shapeImme,
                                                         std::vector<OpImmediate>());
    std::vector<OpImmediate> toValidShape = {OpImmediate(SymbolicScalar("Input_0_Dim_0")),
                                             OpImmediate(SymbolicScalar("Input_0_Dim_1"))};
    copyin_op.SetOpAttribute(copyin_attr);

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);
    std::cout << currFunctionPtr->Dump() << std::endl;
    InferParamIndex inferIndexTest;
    EXPECT_EQ(inferIndexTest.InferShape(*currFunctionPtr), SUCCESS);
}

TEST_F(InferIndexTest, TestInferShapeNoneOp) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(),
                                                      "TestInferShape",
                                                      "TestInferShape",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    InferParamIndex inferIndexTest;
    EXPECT_EQ(inferIndexTest.InferShape(*currFunctionPtr), FAILED);
}

}
}