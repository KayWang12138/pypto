/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_last_use.cpp
 * \brief
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/block_graph_pass/schedule_ooo/schedule_ooo.h"
#include "interface/operation/attribute.h"
#include "interface/function/function.h"

using namespace npu::tile_fwk;

class GetParamIdxTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
    }

    void TearDown() override {}
};

TEST_F(SetLastUse, TestNormal) {
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestNormal", "TestNormal", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestNormal", "TestNormal", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {ubTensor1});
    auto &op2 = currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor1}, {ubTensor2});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {ubTensor1}, {outcast});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);
    OoOSchedule pass;
    pass.RecordLastUseMemory(*rootFuncPtr);
    std::vector<long int> res = {0, 1};
    EXPECT_EQ(op2.GetVectorIntAttribute(OpAttributeKey::lastUse), res);
}