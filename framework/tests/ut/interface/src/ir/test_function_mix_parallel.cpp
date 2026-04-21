/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_function_mix_parallel.cpp
 * \brief Unit tests for mix-split callop grouped parallel execution in FunctionInterpreter.
 */

#include <gtest/gtest.h>

#include "interface/inner/tilefwk.h"
#include "interface/interpreter/function.h"
#include "interface/interpreter/raw_tensor_data.h"

namespace npu::tile_fwk {
namespace {

class FunctionMixParallelTest : public testing::Test {
public:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
    }
};

TEST_F(FunctionMixParallelTest, GroupedMixSplitCallOpsExecuteTwoCalleeFrames)
{
    auto rootFunc =
        std::make_shared<Function>(Program::GetInstance(), "mix_parallel_root", "mix_parallel_root", nullptr);
    rootFunc->SetFunctionType(FunctionType::STATIC);
    rootFunc->SetGraphType(GraphType::BLOCK_GRAPH);

    auto calleeFunc1 = std::make_shared<Function>(
        Program::GetInstance(), "mix_parallel_callee1", "mix_parallel_callee1", rootFunc.get());
    auto calleeFunc2 = std::make_shared<Function>(
        Program::GetInstance(), "mix_parallel_callee2", "mix_parallel_callee2", rootFunc.get());
    calleeFunc1->SetFunctionType(FunctionType::STATIC);
    calleeFunc2->SetFunctionType(FunctionType::STATIC);
    calleeFunc1->SetGraphType(GraphType::BLOCK_GRAPH);
    calleeFunc2->SetGraphType(GraphType::BLOCK_GRAPH);

    std::vector<int64_t> shape = {1};
    auto rootIn1 = std::make_shared<LogicalTensor>(*rootFunc, DT_FP32, shape);
    auto rootOut1 = std::make_shared<LogicalTensor>(*rootFunc, DT_FP32, shape);
    auto rootIn2 = std::make_shared<LogicalTensor>(*rootFunc, DT_FP32, shape);
    auto rootOut2 = std::make_shared<LogicalTensor>(*rootFunc, DT_FP32, shape);
    rootFunc->inCasts_ = {rootIn1, rootIn2};
    rootFunc->outCasts_ = {rootOut1, rootOut2};

    auto calleeIn1 = std::make_shared<LogicalTensor>(*calleeFunc1, DT_FP32, shape);
    auto calleeOut1 = std::make_shared<LogicalTensor>(*calleeFunc1, DT_FP32, shape);
    calleeFunc1->inCasts_ = {calleeIn1};
    calleeFunc1->outCasts_ = {calleeOut1};
    auto calleeIn2 = std::make_shared<LogicalTensor>(*calleeFunc2, DT_FP32, shape);
    auto calleeOut2 = std::make_shared<LogicalTensor>(*calleeFunc2, DT_FP32, shape);
    calleeFunc2->inCasts_ = {calleeIn2};
    calleeFunc2->outCasts_ = {calleeOut2};

    auto leafAttr1 = std::make_shared<LeafFuncAttribute>();
    auto leafAttr2 = std::make_shared<LeafFuncAttribute>();
    leafAttr1->mixId = 1;
    leafAttr2->mixId = 1;
    calleeFunc1->SetLeafFuncAttribute(leafAttr1);
    calleeFunc2->SetLeafFuncAttribute(leafAttr2);

    auto calleeHash1 = calleeFunc1->ComputeHash();
    auto calleeHash2 = calleeFunc2->ComputeHash();
    auto& callOp1 = rootFunc->AddRawOperation(Opcode::OP_CALL, {rootIn1}, {rootOut1}, false);
    auto& callOp2 = rootFunc->AddRawOperation(Opcode::OP_CALL, {rootIn2}, {rootOut2}, false);

    std::map<int, SymbolicScalar> emptyOutExpr;
    std::vector<std::vector<SymbolicScalar>> emptyArgList;
    auto callAttr1 =
        std::dynamic_pointer_cast<CallOpAttribute>(calleeFunc1->CreateCallOpAttribute(emptyArgList, emptyOutExpr));
    auto callAttr2 =
        std::dynamic_pointer_cast<CallOpAttribute>(calleeFunc2->CreateCallOpAttribute(emptyArgList, emptyOutExpr));
    ASSERT_NE(callAttr1, nullptr);
    ASSERT_NE(callAttr2, nullptr);
    callAttr1->SetCalleeHash(calleeHash1);
    callAttr2->SetCalleeHash(calleeHash2);
    callAttr1->wrapId = 100;
    callAttr2->wrapId = 100;
    callOp1.SetOpAttribute(callAttr1);
    callOp2.SetOpAttribute(callAttr2);

    FunctionInterpreter interpreter;
    interpreter.calleeHashDict[calleeHash1.GetHash()] = calleeFunc1.get();
    interpreter.calleeHashDict[calleeHash2.GetHash()] = calleeFunc2.get();

    Tensor inTensor(DT_FP32, shape);
    Tensor outTensor(DT_FP32, shape);
    auto inView1 = std::make_shared<LogicalTensorData>(RawTensorData::CreateConstantTensor<float>(inTensor, 1.0f));
    auto inView2 = std::make_shared<LogicalTensorData>(RawTensorData::CreateConstantTensor<float>(inTensor, 2.0f));
    auto outView1 = std::make_shared<LogicalTensorData>(RawTensorData::CreateConstantTensor<float>(outTensor, 0.0f));
    auto outView2 = std::make_shared<LogicalTensorData>(RawTensorData::CreateConstantTensor<float>(outTensor, 0.0f));
    auto inoutDataPair = std::make_shared<FunctionIODataPair>(
        std::vector<std::shared_ptr<LogicalTensorData>>{inView1, inView2},
        std::vector<std::shared_ptr<LogicalTensorData>>{outView1, outView2});

    auto rootCallAttr =
        std::dynamic_pointer_cast<CallOpAttribute>(calleeFunc1->CreateCallOpAttribute(emptyArgList, emptyOutExpr));
    ASSERT_NE(rootCallAttr, nullptr);
    FunctionFrame rootFrame(rootFunc.get(), nullptr, rootCallAttr, inoutDataPair, 0);

    std::vector<std::shared_ptr<FunctionFrame>> capturedFrames;
    interpreter.captureFrameList = &capturedFrames;
    interpreter.DumpBegin();

    auto operations = rootFunc->Operations();
    size_t opIdx = 0;
    bool handled = interpreter.TryExecuteMixSplitCallOps(rootFrame, operations, opIdx, operations.at(0));
    interpreter.DumpEnd();

    EXPECT_TRUE(handled);
    EXPECT_EQ(opIdx, 2U);
    ASSERT_EQ(capturedFrames.size(), 2U);

    std::set<const Operation*> executedCallops;
    for (const auto& frame : capturedFrames) {
        ASSERT_NE(frame, nullptr);
        ASSERT_NE(frame->callop, nullptr);
        executedCallops.insert(frame->callop);
    }
    EXPECT_EQ(executedCallops.size(), 2U);
    EXPECT_TRUE(executedCallops.count(&callOp1) > 0);
    EXPECT_TRUE(executedCallops.count(&callOp2) > 0);
}

} // namespace
} // namespace npu::tile_fwk
