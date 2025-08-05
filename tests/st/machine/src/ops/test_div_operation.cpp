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
 * \file test_div_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct DivOpFuncArgs : public OpFuncArgs {
    DivOpFuncArgs(const std::vector<int> &input0Shape, const std::vector<int> &input1Shape,
        const std::vector<int> &viewShape, const std::vector<int> tileShape, const DataType &dType)
        : input0Shape_(input0Shape),
          input1Shape_(input1Shape),
          viewShape_(viewShape),
          tileShape_(tileShape),
          dType_(dType) {}

    std::vector<int> input0Shape_;
    std::vector<int> input1Shape_;
    std::vector<int> viewShape_;
    std::vector<int> tileShape_;
    DataType dType_;
};

struct DivOpMetaData {
    DivOpMetaData(int case_index, const DivOpFuncArgs &args, const OpFunc &opFunc)
        : case_index_(case_index), args_(args), opFunc_(opFunc) {}

    int case_index_;
    DivOpFuncArgs args_;
    OpFunc opFunc_;
};

static void DivOperationExeFuncDoubleCut(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0], inputs[1]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        auto args = static_cast<const DivOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];

        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                auto tileTensor0 = DViewPad(inputs[0], {firstViewShape, secondViewShape},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                    {bIdx * firstViewShape, sIdx * secondViewShape});
                auto tileTensor1 = DViewPad(inputs[1], {firstViewShape, secondViewShape},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                    {bIdx * firstViewShape, sIdx * secondViewShape});
                Program::GetInstance().GetTileShape().SetVecTileShapes(args->tileShape_);
                auto res = Div(tileTensor0, tileTensor1);
                DAssemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

static void DivOperationExeFuncTripleCut(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0], inputs[1]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        SymbolicScalar thirdDim = inputs[0]->shape[2];
        auto args = static_cast<const DivOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];

        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        const int nloop = CeilDiv(thirdDim, thirdViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1)) {
                    auto tileTensor0 = DViewPad(inputs[0], {firstViewShape, secondViewShape, thirdViewShape},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                            std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                            std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                        {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});
                    auto tileTensor1 = DViewPad(inputs[1], {firstViewShape, secondViewShape, thirdViewShape},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                            std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                            std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                        {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});
                    Program::GetInstance().GetTileShape().SetVecTileShapes(args->tileShape_);
                    auto res = Div(tileTensor0, tileTensor1);
                    DAssemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

static const DivOpMetaData testDataLists[] = {
    DivOpMetaData{ 0,             DivOpFuncArgs{{1024, 128}, {1024, 128}, {128, 512}, {64, 64}, DataType::DT_FP32},
                  DivOperationExeFuncDoubleCut},
    DivOpMetaData{ 1,             DivOpFuncArgs{{1024, 128}, {1024, 128}, {128, 512}, {64, 64}, DataType::DT_FP32},
                  DivOperationExeFuncDoubleCut},
    DivOpMetaData{ 2,             DivOpFuncArgs{{1024, 128}, {1024, 128}, {128, 512}, {64, 64}, DataType::DT_FP32},
                  DivOperationExeFuncDoubleCut},
    DivOpMetaData{ 3,             DivOpFuncArgs{{1024, 128}, {1024, 128}, {128, 512}, {64, 64}, DataType::DT_FP32},
                  DivOperationExeFuncDoubleCut},
    DivOpMetaData{ 4,             DivOpFuncArgs{{1024, 128}, {1024, 128}, {128, 512}, {64, 64}, DataType::DT_FP32},
                  DivOperationExeFuncDoubleCut},
    DivOpMetaData{ 5,             DivOpFuncArgs{{1024, 128}, {1024, 128}, {128, 512}, {64, 64}, DataType::DT_FP32},
                  DivOperationExeFuncDoubleCut},
    DivOpMetaData{ 6,             DivOpFuncArgs{{1024, 128}, {1024, 128}, {128, 512}, {64, 64}, DataType::DT_FP32},
                  DivOperationExeFuncDoubleCut},
    DivOpMetaData{ 7,             DivOpFuncArgs{{1024, 128}, {1024, 128}, {128, 512}, {64, 64}, DataType::DT_FP32},
                  DivOperationExeFuncDoubleCut},
    DivOpMetaData{ 8,             DivOpFuncArgs{{1024, 128}, {1024, 128}, {128, 512}, {64, 64}, DataType::DT_FP32},
                  DivOperationExeFuncDoubleCut},
    DivOpMetaData{ 9,             DivOpFuncArgs{{1024, 128}, {1024, 128}, {128, 512}, {64, 64}, DataType::DT_FP32},
                  DivOperationExeFuncDoubleCut},
    DivOpMetaData{10, DivOpFuncArgs{{496, 128, 64}, {496, 128, 64}, {64, 32, 32}, {16, 32, 32}, DataType::DT_FP32},
                  DivOperationExeFuncTripleCut},
    DivOpMetaData{11,     DivOpFuncArgs{{128 + 17, 128}, {128 + 17, 128}, {128, 128}, {32, 32}, DataType::DT_FP32},
                  DivOperationExeFuncDoubleCut},
    DivOpMetaData{12,         DivOpFuncArgs{{90 + 17, 128}, {90 + 17, 128}, {45, 64}, {32, 32}, DataType::DT_FP32},
                  DivOperationExeFuncDoubleCut},
};

class DivOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<DivOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestDiv, DivOperationTest, ::testing::ValuesIn(testDataLists));

TEST_P(DivOperationTest, TestDiv) {
    TestCaseDesc testCase;
    auto args = GetParam().args_;
    testCase.inputTensors = {
        Tensor(args.dType_, args.input0Shape_, "input0"), Tensor(args.dType_, args.input1Shape_, "input1")};
    testCase.outputTensors = {Tensor(args.dType_, args.input0Shape_, "output")};
    testCase.args = &(GetParam().args_);
    testCase.opFunc = GetParam().opFunc_;
    testCase.inputPaths = {GetGoldenDir() + "/x.bin", GetGoldenDir() + "/y.bin"};
    testCase.goldenPaths = {GetGoldenDir() + "/res.bin"};
    TestExecutor::runTest(testCase);
}
} // namespace
