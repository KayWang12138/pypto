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
 * \file test_topk_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct TopKOpFuncArgs : public OpFuncArgs {
    TopKOpFuncArgs(std::vector<int> shape, std::vector<int> outputShape, std::vector<std::string> attrs,
        std::vector<int> veiwShape, std::vector<int> vecTileShape, DataType dType) :
        shape_(shape), outputShape_(outputShape), attrs_(attrs), veiwShape_(veiwShape), vecTileShape_(vecTileShape), dType_(dType) {}
    std::vector<int> shape_;
    std::vector<int> outputShape_;
    std::vector<std::string> attrs_;
    std::vector<int> veiwShape_;
    std::vector<int> vecTileShape_;
    DataType dType_;
};

struct TopKOperationMetadata {
    TopKOperationMetadata(std::vector<int> shape, std::vector<int> outputShape, std::vector<std::string> attrs,
        std::vector<int> veiwShape, std::vector<int> vecTileShape, DataType dType, OpFunc opFunc) :
        args_(shape, outputShape, attrs, veiwShape, vecTileShape, dType), opFunc_(opFunc) {}
    TopKOpFuncArgs args_;
    OpFunc opFunc_;
};

void TopKOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                const OpFuncArgs* opArgs) {
    int k = std::atoi((static_cast<const TopKOpFuncArgs*>(opArgs))->attrs_[0].c_str());
    int dim = std::atoi((static_cast<const TopKOpFuncArgs*>(opArgs))->attrs_[1].c_str());
    const char* str = (static_cast<const TopKOpFuncArgs*>(opArgs))->attrs_[2].c_str();
    bool isLargest = (strcmp(str, "true") == 0) || (strcmp(str, "1") == 0);
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0], outputs[1]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        auto args = static_cast<const TopKOpFuncArgs*>(opArgs);
        const int firstViewShape = args->veiwShape_[0];
        const int bloop = CeilDiv(firstDim, firstViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(bloop)) {
            auto viewTensor = DViewPad(inputs[0], {firstViewShape, secondDim},
                {std::min(firstDim - bIdx * firstViewShape, firstViewShape), secondDim},
                {bIdx * firstViewShape, 0});
            Program::GetInstance().GetTileShape().SetVecTileShapes(
                (static_cast<const TopKOpFuncArgs*>(opArgs))->vecTileShape_);
            auto res = TopK(viewTensor, k, dim, isLargest);
            DAssemble(std::get<0>(res), {bIdx * firstViewShape, 0}, outputs[0]);
            DAssemble(std::get<1>(res), {bIdx * firstViewShape, 0}, outputs[1]);
        }
    }
}

static const TopKOperationMetadata testDataLists[] = {
    TopKOperationMetadata({128, 32}, {128, 8}, {"8", "-1", "true"}, {128, 32}, {8, 8}, DataType::DT_FP32, TopKOperationExeFunc),
};

class TopKOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<TopKOperationMetadata> {};

INSTANTIATE_TEST_SUITE_P(
    TestTopK,
    TopKOperationTest,
    ::testing::ValuesIn(testDataLists)
);

TEST_P(TopKOperationTest, test_topk) {
    TestCaseDesc testCase;
    testCase.inputTensors = {
        Tensor(GetParam().args_.dType_, GetParam().args_.shape_, "input0"),
    };
    testCase.outputTensors = {
        Tensor(GetParam().args_.dType_, GetParam().args_.outputShape_, "output0"),
        Tensor(DataType::DT_INT32, GetParam().args_.outputShape_, "output1")
    };
    testCase.inputPaths = {GetGoldenDir() + "/x.bin"};
    testCase.goldenPaths = {GetGoldenDir() + "/value.bin", GetGoldenDir() + "/index.bin"};
    testCase.args = &(GetParam().args_);
    testCase.opFunc = GetParam().opFunc_;
    TestExecutor::runTest(testCase);
}

void TopK3DOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                const OpFuncArgs* opArgs) {
    int k = std::atoi((static_cast<const TopKOpFuncArgs*>(opArgs))->attrs_[0].c_str());
    int dim = std::atoi((static_cast<const TopKOpFuncArgs*>(opArgs))->attrs_[1].c_str());
    const char* str = (static_cast<const TopKOpFuncArgs*>(opArgs))->attrs_[2].c_str();
    bool isLargest = (strcmp(str, "true") == 0) || (strcmp(str, "1") == 0);
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0], outputs[1]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        SymbolicScalar thirdDim = inputs[0]->shape[2];
        auto args = static_cast<const TopKOpFuncArgs*>(opArgs);
        const int firstViewShape = args->veiwShape_[0];
        const int secondViewShape = args->veiwShape_[1];
        const int thirdViewShape = args->veiwShape_[2];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        const int nloop = CeilDiv(thirdDim, thirdViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(bloop)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(sloop)) {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(nloop)) {
                    auto viewTensor = DViewPad(inputs[0], {firstViewShape, secondViewShape, thirdViewShape},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                        std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                        {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});
                    Program::GetInstance().GetTileShape().SetVecTileShapes(
                        (static_cast<const TopKOpFuncArgs*>(opArgs))->vecTileShape_);
                    auto res = TopK(viewTensor, k, dim, isLargest);
                    DAssemble(std::get<0>(res), {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                    DAssemble(std::get<1>(res), {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[1]);
                }
            }
        }
    }
}

static const TopKOperationMetadata test3DDataLists[] = {
    TopKOperationMetadata({8, 2, 8}, {8, 2, 8}, {"8", "-1", "true"}, {8, 2, 8}, {8, 2, 8}, DataType::DT_FP32, TopK3DOperationExeFunc),
};

class TopK3DOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<TopKOperationMetadata> {};

INSTANTIATE_TEST_SUITE_P(
    TestTopK,
    TopK3DOperationTest,
    ::testing::ValuesIn(test3DDataLists)
);

TEST_P(TopK3DOperationTest, test_topk) {
    TestCaseDesc testCase;
    testCase.inputTensors = {
        Tensor(GetParam().args_.dType_, GetParam().args_.shape_, "input0"),
    };
    testCase.outputTensors = {
        Tensor(GetParam().args_.dType_, GetParam().args_.outputShape_, "output0"),
        Tensor(DataType::DT_INT32, GetParam().args_.outputShape_, "output1")
    };
    testCase.inputPaths = {GetGoldenDir() + "/x.bin"};
    testCase.goldenPaths = {GetGoldenDir() + "/value.bin", GetGoldenDir() + "/index.bin"};
    testCase.args = &(GetParam().args_);
    testCase.opFunc = GetParam().opFunc_;
    TestExecutor::runTest(testCase);
}

void TopK4DOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                const OpFuncArgs* opArgs) {
    int k = std::atoi((static_cast<const TopKOpFuncArgs*>(opArgs))->attrs_[0].c_str());
    int dim = std::atoi((static_cast<const TopKOpFuncArgs*>(opArgs))->attrs_[1].c_str());
    const char* str = (static_cast<const TopKOpFuncArgs*>(opArgs))->attrs_[2].c_str();
    bool isLargest = (strcmp(str, "true") == 0) || (strcmp(str, "1") == 0);
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0], outputs[1]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        SymbolicScalar thirdDim = inputs[0]->shape[2];
        SymbolicScalar fourDim = inputs[0]->shape[3];
        auto args = static_cast<const TopKOpFuncArgs*>(opArgs);
        const int firstViewShape = args->veiwShape_[0];
        const int secondViewShape = args->veiwShape_[1];
        const int thirdViewShape = args->veiwShape_[2];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        const int nloop = CeilDiv(thirdDim, thirdViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(bloop)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(sloop)) {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(nloop)) {
                    auto viewTensor = DViewPad(inputs[0], {firstViewShape, secondViewShape, thirdViewShape, fourDim},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                        std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape), fourDim},
                        {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape, 0});
                    Program::GetInstance().GetTileShape().SetVecTileShapes(
                        (static_cast<const TopKOpFuncArgs*>(opArgs))->vecTileShape_);
                    auto res = TopK(viewTensor, k, dim, isLargest);
                    DAssemble(std::get<0>(res), {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape, 0}, outputs[0]);
                    DAssemble(std::get<1>(res), {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape, 0}, outputs[1]);
                }
            }
        }
    }
}

static const TopKOperationMetadata test4DDataLists[] = {
    TopKOperationMetadata({2, 8, 2, 8}, {2, 8, 2, 8}, {"8", "-1", "true"}, {2, 8, 2, 8}, {2, 8, 2, 8}, DataType::DT_FP32, TopK4DOperationExeFunc),
};

class TopK4DOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<TopKOperationMetadata> {};

INSTANTIATE_TEST_SUITE_P(
    TestTopK,
    TopK4DOperationTest,
    ::testing::ValuesIn(test4DDataLists)
);

TEST_P(TopK4DOperationTest, test_topk) {
    TestCaseDesc testCase;
    testCase.inputTensors = {
        Tensor(GetParam().args_.dType_, GetParam().args_.shape_, "input0"),
    };
    testCase.outputTensors = {
        Tensor(GetParam().args_.dType_, GetParam().args_.outputShape_, "output0"),
        Tensor(DataType::DT_INT32, GetParam().args_.outputShape_, "output1")
    };
    testCase.inputPaths = {GetGoldenDir() + "/x.bin"};
    testCase.goldenPaths = {GetGoldenDir() + "/value.bin", GetGoldenDir() + "/index.bin"};
    testCase.args = &(GetParam().args_);
    testCase.opFunc = GetParam().opFunc_;
    TestExecutor::runTest(testCase);
}

} // namespace