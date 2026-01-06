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
 * \file test_expand_function.cpp
 * \brief Unit test for ExpandFunction pass.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "tilefwk/tilefwk.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "passes/pass_mgr/pass_manager.h"

#define private public
#include "passes/block_graph_pass/loopaxes_proc.h"

namespace npu {
namespace tile_fwk {
static const int kKeepOut = -1;
static const int kNum0 = 0;
static const int kNum1 = 1;
static const int kNum2 = 2;
static const int kNum4 = 4;
static const int kNum16 = 2;

class TestLoopaxesProcPass : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "ExpandFunctionTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

TEST_F(TestLoopaxesProcPass, LoopaxesProcUTest1) {
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestLoopaxesProcPass", "TestLoopaxesProcPass", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestLoopaxesProcPassLeaf", "TestLoopaxesProcPassLeaf", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());
    rootFuncPtr->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    rootFuncPtr->SetUnderDynamicFunction(true);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNum16};
    std::vector<int64_t> shape2 = {kNum2, kNum2, kNum4};
    std::vector<int64_t> shape3 = {kNum4, kNum4};
    auto inCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto inCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto ubTensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto ubTensor4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);

    auto &view1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast1}, {ubTensor1});
    auto &view2 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast2}, {ubTensor2});
    auto &reshape1 = currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor3});
    auto &add = currFunctionPtr->AddOperation(Opcode::OP_ADD, {ubTensor2, ubTensor3}, {ubTensor4});
    auto &reshape2 = currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor4}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast1);
    currFunctionPtr->inCasts_.push_back(inCast2);
    currFunctionPtr->outCasts_.push_back(outCast);

    LoopaxesProc loopaxesprocpass;
    auto status = loopaxesprocpass.RunOnFunction(*rootFuncPtr);
    EXPECT_EQ(status, SUCCESS);

    int loopGroup;
    std::vector<int64_t> expectedLoopAxis;

    EXPECT_TRUE(view1.HasAttr(OpAttributeKey::loopGroup));
    EXPECT_EQ(view1.GetIntAttribute(OpAttributeKey::loopGroup), kKeepOut);
    EXPECT_TRUE(view1.HasAttr(OpAttributeKey::loopAxes));
    EXPECT_TRUE(view1.GetVectorIntAttribute(OpAttributeKey::loopAxes).empty());
    
    expectedLoopAxis = {kNum2, kNum2};
    EXPECT_TRUE(view2.HasAttr(OpAttributeKey::loopGroup));
    EXPECT_EQ(view2.GetIntAttribute(OpAttributeKey::loopGroup), kNum0);
    EXPECT_TRUE(view2.HasAttr(OpAttributeKey::loopAxes));
    EXPECT_EQ(view2.GetVectorIntAttribute(OpAttributeKey::loopAxes), expectedLoopAxis);
    
    expectedLoopAxis = {kNum2, kNum2};
    EXPECT_TRUE(reshape1.HasAttr(OpAttributeKey::loopGroup));
    EXPECT_EQ(reshape1.GetIntAttribute(OpAttributeKey::loopGroup), kNum0);
    EXPECT_TRUE(reshape1.HasAttr(OpAttributeKey::loopAxes));
    EXPECT_EQ(reshape1.GetVectorIntAttribute(OpAttributeKey::loopAxes), expectedLoopAxis);
    
    expectedLoopAxis = {kNum2, kNum2};
    EXPECT_TRUE(add.HasAttr(OpAttributeKey::loopGroup));
    EXPECT_EQ(add.GetIntAttribute(OpAttributeKey::loopGroup), kNum0);
    EXPECT_TRUE(add.HasAttr(OpAttributeKey::loopAxes));
    EXPECT_EQ(add.GetVectorIntAttribute(OpAttributeKey::loopAxes), expectedLoopAxis);
    
    expectedLoopAxis = {kNum4};
    EXPECT_TRUE(reshape2.HasAttr(OpAttributeKey::loopGroup));
    EXPECT_EQ(reshape2.GetIntAttribute(OpAttributeKey::loopGroup), kNum1);
    EXPECT_TRUE(reshape2.HasAttr(OpAttributeKey::loopAxes));
    EXPECT_EQ(reshape2.GetVectorIntAttribute(OpAttributeKey::loopAxes), expectedLoopAxis);
}
}
}