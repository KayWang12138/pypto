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
 * \file test_sub_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct SubOpFuncArgs : public OpFuncArgs {
    SubOpFuncArgs(const std::vector<int> &input0Shape, const std::vector<int> &input1Shape,
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

struct SubOperationMetadata {
    SubOperationMetadata(int case_index, const SubOpFuncArgs &args, const OpFunc &opFunc)
        : case_index_(case_index), args_(args), opFunc_(opFunc) {}

    int case_index_;
    SubOpFuncArgs args_;
    OpFunc opFunc_;
};

static void SubOperationExeFuncDoubleCut(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0], inputs[1]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];

        auto args = static_cast<const SubOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[0];
        int bloop = CeilDiv(firstDim, firstViewShape);
        int sloop = CeilDiv(secondDim, secondViewShape);

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
                auto res = Sub(tileTensor0, tileTensor1);
                DAssemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

static const SubOperationMetadata testDataLists[] = {
    // shape0,shape1,viewshape,tileshape,dtype
    SubOperationMetadata(
        3, SubOpFuncArgs{{128 + 17, 128}, {128 + 17, 128}, {128, 128}, {32, 32}, DataType::DT_FP32}, SubOperationExeFuncDoubleCut),
    SubOperationMetadata(
        4, SubOpFuncArgs{{90 + 17, 128}, {90 + 17, 128}, {45, 46}, {32, 32}, DataType::DT_FP32}, SubOperationExeFuncDoubleCut),

};

class SubOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<SubOperationMetadata> {};

INSTANTIATE_TEST_SUITE_P(TestSub, SubOperationTest, ::testing::ValuesIn(testDataLists));

TEST_P(SubOperationTest, test_sub) {
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