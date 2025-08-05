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
 * \file test_reduce_max_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct ReduceMaxOpFuncArgs : public OpFuncArgs {
    ReduceMaxOpFuncArgs(std::vector<int> shape, std::vector<int> outputShape, std::vector<std::string> attrs,
        std::vector<int> vecTileShapes, DataType dType) :
        shape_(shape), outputShape_(outputShape), attrs_(attrs), vecTileShapes_(vecTileShapes), dType_(dType) {}
    std::vector<int> shape_;
    std::vector<int> outputShape_;
    std::vector<std::string> attrs_;
    std::vector<int> vecTileShapes_;
    DataType dType_;
};

struct ReduceMaxOperationMetadata {
    ReduceMaxOperationMetadata(std::vector<int> shape, std::vector<int> outputShape, std::vector<std::string> attrs,
        std::vector<int> vecTileShapes, DataType dType, OpFunc opFunc) :
        args_(shape, outputShape, attrs, vecTileShapes, dType), opFunc_(opFunc) {}
    ReduceMaxOpFuncArgs args_;
    OpFunc opFunc_;
};

void ReduceMaxOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                const OpFuncArgs* opArgs) {
    int dim = std::atoi((static_cast<const ReduceMaxOpFuncArgs*>(opArgs))->attrs_[0].c_str());
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        const int firstViewShape = 128;
        int batch = (firstDim + firstViewShape -1) / firstViewShape;

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(batch)) {
            auto viewTensor = DViewPad(inputs[0], {firstViewShape, secondDim},
                {std::min(firstDim - bIdx * firstViewShape, firstViewShape), secondDim},
                {bIdx * firstViewShape, 0});
            Program::GetInstance().GetTileShape().SetVecTileShapes(
                (static_cast<const ReduceMaxOpFuncArgs*>(opArgs))->vecTileShapes_);
            auto res = RowMaxSingle(viewTensor, dim);
            DAssemble(res, {bIdx * firstViewShape, 0}, outputs[0]);
        }
    }
}

static const ReduceMaxOperationMetadata testDataLists[] = {
    ReduceMaxOperationMetadata({512, 128}, {512, 1}, {"1"}, {64, 128}, DataType::DT_FP32, ReduceMaxOperationExeFunc),
    ReduceMaxOperationMetadata({4, 1000}, {4, 1}, {"1"}, {4, 200}, DataType::DT_FP32, ReduceMaxOperationExeFunc),
};

class ReduceMaxOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<ReduceMaxOperationMetadata> {};

INSTANTIATE_TEST_SUITE_P(
    TestReduceMax,
    ReduceMaxOperationTest,
    ::testing::ValuesIn(testDataLists)
);

TEST_P(ReduceMaxOperationTest, test_reduce_max) {
    TestCaseDesc testCase;
    testCase.inputTensors = {
        Tensor(GetParam().args_.dType_, GetParam().args_.shape_, "input0"),
    };
    testCase.outputTensors = {
        Tensor(GetParam().args_.dType_, GetParam().args_.outputShape_, "output")
    };
    testCase.inputPaths = {GetGoldenDir() + "/x.bin"};
    testCase.goldenPaths = {GetGoldenDir() + "/res.bin"};
    testCase.args = &(GetParam().args_);
    testCase.opFunc = GetParam().opFunc_;
    TestExecutor::runTest(testCase);
}

void ReduceMax3DOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                const OpFuncArgs* opArgs) {
    int dim = std::atoi((static_cast<const ReduceMaxOpFuncArgs*>(opArgs))->attrs_[0].c_str());
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        SymbolicScalar lastDim = inputs[0]->shape[2];
        const int firstViewShape = 48;
        const int secondViewShape = 32;
        int batch = (firstDim + firstViewShape -1) / firstViewShape;

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(batch)) {
            LOOP("LOOP_L1_bIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange((secondDim + secondViewShape -1) / secondViewShape)) {
                auto viewTensor = DViewPad(inputs[0], {firstViewShape, secondViewShape, lastDim},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                    std::min(secondDim - sIdx * secondViewShape, secondViewShape), lastDim},
                    {bIdx * firstViewShape, sIdx * secondViewShape, 0});
                Program::GetInstance().GetTileShape().SetVecTileShapes(
                    (static_cast<const ReduceMaxOpFuncArgs*>(opArgs))->vecTileShapes_);
                auto res = RowMaxSingle(viewTensor, dim);
                DAssemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, 0}, outputs[0]);
            }
        }
    }
}

static const ReduceMaxOperationMetadata testDataLists3D[] = {
    ReduceMaxOperationMetadata({48, 32, 32}, {48, 32, 1}, {"2"}, {8, 32, 32}, DataType::DT_FP32, ReduceMax3DOperationExeFunc),
};

class ReduceMax3DOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<ReduceMaxOperationMetadata> {};

INSTANTIATE_TEST_SUITE_P(
    TestReduceMax,
    ReduceMax3DOperationTest,
    ::testing::ValuesIn(testDataLists3D)
);

TEST_P(ReduceMax3DOperationTest, test_reduce_max) {
    TestCaseDesc testCase;
    testCase.inputTensors = {
        Tensor(GetParam().args_.dType_, GetParam().args_.shape_, "input0"),
    };
    testCase.outputTensors = {
        Tensor(GetParam().args_.dType_, GetParam().args_.outputShape_, "output")
    };
    testCase.inputPaths = {GetGoldenDir() + "/x.bin"};
    testCase.goldenPaths = {GetGoldenDir() + "/res.bin"};
    testCase.args = &(GetParam().args_);
    testCase.opFunc = GetParam().opFunc_;
    TestExecutor::runTest(testCase);
}

void ReduceMax4DOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                const OpFuncArgs* opArgs) {
    int dim = std::atoi((static_cast<const ReduceMaxOpFuncArgs*>(opArgs))->attrs_[0].c_str());
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        SymbolicScalar threeDim = inputs[0]->shape[2];
        SymbolicScalar lastDim = inputs[0]->shape[3];
        const int firstViewShape = 48;
        int batch = (firstDim + firstViewShape -1) / firstViewShape;

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(batch)) {
            auto viewTensor = DViewPad(inputs[0], {firstViewShape, secondDim, threeDim, lastDim},
                {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                secondDim, threeDim, lastDim},
                {bIdx * firstViewShape, 0, 0, 0});
            Program::GetInstance().GetTileShape().SetVecTileShapes(
                (static_cast<const ReduceMaxOpFuncArgs*>(opArgs))->vecTileShapes_);
            auto res = RowMaxSingle(viewTensor, dim);
            DAssemble(res, {bIdx * firstViewShape, 0, 0, 0}, outputs[0]);
        }
    }
}

static const ReduceMaxOperationMetadata testDataLists4D[] = {
    ReduceMaxOperationMetadata({8, 6, 10, 8}, {8, 6, 10, 1}, {"3"}, {4,4,4,16}, DataType::DT_FP32, ReduceMax4DOperationExeFunc),
};

class ReduceMax4DOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<ReduceMaxOperationMetadata> {};

INSTANTIATE_TEST_SUITE_P(
    TestReduceMax,
    ReduceMax4DOperationTest,
    ::testing::ValuesIn(testDataLists4D)
);

TEST_P(ReduceMax4DOperationTest, test_reduce_max) {
    TestCaseDesc testCase;
    testCase.inputTensors = {
        Tensor(GetParam().args_.dType_, GetParam().args_.shape_, "input0"),
    };
    testCase.outputTensors = {
        Tensor(GetParam().args_.dType_, GetParam().args_.outputShape_, "output")
    };
    testCase.inputPaths = {GetGoldenDir() + "/x.bin"};
    testCase.goldenPaths = {GetGoldenDir() + "/res.bin"};
    testCase.args = &(GetParam().args_);
    testCase.opFunc = GetParam().opFunc_;
    TestExecutor::runTest(testCase);
}

} // namespace