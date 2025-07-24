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
 * \file test_sqrt_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace ascend::test_operation;
namespace {
struct SqrtOpFuncArgs : public OpFuncArgs {
    SqrtOpFuncArgs(std::vector<int> inputShape0, std::vector<int> viewShape, std::vector<int> vecTileShape, DataType dType) :
        inputShape0_(inputShape0), viewShape_(viewShape), vecTileShape_(vecTileShape), dType_(dType) {}
    int num_;
    std::vector<int> inputShape0_;
    std::vector<int> viewShape_;
    std::vector<int> vecTileShape_;
    DataType dType_;
};

struct SqrtOperationMetadata {
    SqrtOperationMetadata(std::vector<int> inputShape0, std::vector<int> viewShape, std::vector<int> vecTileShape, DataType dType, OpFunc opFunc) :
        args_(inputShape0, viewShape, vecTileShape, dType), opFunc_(opFunc) {}
    SqrtOpFuncArgs args_;
    OpFunc opFunc_;
};

static void SqrtOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                const OpFuncArgs* opArgs) {
    const SqrtOpFuncArgs  *SqrtInfo =  static_cast<const SqrtOpFuncArgs*>(opArgs);
    Program::GetInstance().GetTileShape().SetVecTileShapes(SqrtInfo->vecTileShape_);
    const int firstl0LoopLengthTile = SqrtInfo->viewShape_[0];
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, CeilDiv(firstDim, firstl0LoopLengthTile), 1)) {
            auto tileTensor0 = DViewPad(inputs[0], {firstl0LoopLengthTile, secondDim},
                {std::min(firstDim - bIdx * firstl0LoopLengthTile, firstl0LoopLengthTile), secondDim},
                {bIdx * firstl0LoopLengthTile, 0});
            auto res = Sqrt(tileTensor0);
            DAssemble(res, {bIdx * firstl0LoopLengthTile, 0}, outputs[0]);
        }
    }
}

static void SqrtOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                const OpFuncArgs* opArgs) {
    const SqrtOpFuncArgs  *SqrtInfo =  static_cast<const SqrtOpFuncArgs*>(opArgs);
    Program::GetInstance().GetTileShape().SetVecTileShapes(SqrtInfo->vecTileShape_);
    const int firstViewShape = SqrtInfo->viewShape_[0];
    const int secondViewShape = SqrtInfo->viewShape_[1];
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, CeilDiv(firstDim, firstViewShape), 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, CeilDiv(secondDim, secondViewShape), 1)) {
                Tensor tileTensor0 = DViewPad(inputs[0], {firstViewShape, secondViewShape},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                            std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                        {bIdx * firstViewShape, sIdx * secondViewShape});
                auto res = Sqrt(tileTensor0);
                DAssemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

static const SqrtOperationMetadata testDataLists[] = {
    SqrtOperationMetadata{{16, 1}, {32, 32}, {16, 16}, DataType::DT_FP32, SqrtOperationExeFunc},
    SqrtOperationMetadata{{32, 1}, {32, 1}, {32, 1}, DataType::DT_FP32, SqrtOperationExeFunc2Dims},
    SqrtOperationMetadata{{32, 128}, {32, 16}, {16, 16}, DataType::DT_FP32, SqrtOperationExeFunc},
    SqrtOperationMetadata{{32, 1}, {32, 1}, {16, 1}, DataType::DT_FP32, SqrtOperationExeFunc2Dims},
    SqrtOperationMetadata{{1, 1}, {32, 32}, {16, 16}, DataType::DT_FP32, SqrtOperationExeFunc2Dims},
    SqrtOperationMetadata{{32, 32}, {32, 32}, {16, 16}, DataType::DT_FP32, SqrtOperationExeFunc},
    SqrtOperationMetadata{{128, 128}, {32, 32}, {64, 64}, DataType::DT_FP16, SqrtOperationExeFunc2Dims},
    SqrtOperationMetadata{{16, 1}, {16, 1}, {16, 1}, DataType::DT_FP32, SqrtOperationExeFunc2Dims},
    SqrtOperationMetadata{{16, 1}, {16, 1}, {16, 1}, DataType::DT_FP32, SqrtOperationExeFunc2Dims},
    SqrtOperationMetadata{{128, 128}, {32, 32}, {16, 16}, DataType::DT_FP32, SqrtOperationExeFunc2Dims},
    SqrtOperationMetadata{{32, 32}, {16, 16}, {32, 32}, DataType::DT_FP32, SqrtOperationExeFunc},
    SqrtOperationMetadata{{32, 32}, {32, 32}, {32, 32}, DataType::DT_FP32, SqrtOperationExeFunc},
    SqrtOperationMetadata{{32, 32}, {32, 32}, {32, 32}, DataType::DT_FP32, SqrtOperationExeFunc},
    SqrtOperationMetadata{{32, 32}, {32, 32}, {32, 32}, DataType::DT_FP32, SqrtOperationExeFunc},
    SqrtOperationMetadata{{64 * 48, 128 * 3}, {128, 127}, {32, 32}, DataType::DT_FP32, SqrtOperationExeFunc}
};

class SqrtOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<SqrtOperationMetadata> {};

INSTANTIATE_TEST_SUITE_P(
    TestSqrt,
    SqrtOperationTest,
    ::testing::ValuesIn(testDataLists)
);

TEST_P(SqrtOperationTest, TestSqrt) {
    TestCaseDesc testCase;
    testCase.inputTensors = {
        Tensor(GetParam().args_.dType_, GetParam().args_.inputShape0_, "input0")
    };
    testCase.outputTensors = {
        Tensor(GetParam().args_.dType_, GetParam().args_.inputShape0_, "output")
    };
    testCase.inputPaths = {GetGoldenDir() + "/x.bin"};
    testCase.goldenPaths = {GetGoldenDir() + "/res.bin"};
    testCase.args = &(GetParam().args_);
    testCase.opFunc = GetParam().opFunc_;
    TestExecutor::runTest(testCase);
}
} // namespace