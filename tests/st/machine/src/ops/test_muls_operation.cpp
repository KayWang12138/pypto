/**MulsOperationTest
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_muls_operation.cpp
 * \brief
 */

#include <cmath>
#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct MulsOpFuncArgs : public OpFuncArgs {
    MulsOpFuncArgs(const std::vector<int> &input0Shape, const std::vector<int> &viewShape,
        const std::vector<int> tileShape, const DataType &dType)
        : input0Shape_(input0Shape), viewShape_(viewShape), tileShape_(tileShape), dType_(dType) {}

    std::vector<int> input0Shape_;
    std::vector<int> viewShape_;
    std::vector<int> tileShape_;
    DataType dType_;
};

struct MulsOperationMetadata {
    MulsOperationMetadata(int case_index, const MulsOpFuncArgs &args, const OpFunc &opFunc)
        : case_index_(case_index), args_(args), opFunc_(opFunc) {}

    int case_index_;
    MulsOpFuncArgs args_;
    OpFunc opFunc_;
};

static void MulsOperationExeFuncDoubleCut(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];

        auto args = static_cast<const MulsOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[0];
        int bloop = CeilDiv(firstDim, firstViewShape);
        int sloop = CeilDiv(secondDim, secondViewShape);

        std::vector<float> inputData(1, 0);
        readInput<float>(GetGoldenDir() + "/y.bin", inputData);
        Element value(DataType::DT_FP32, inputData[0]);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                auto tileTensor0 = DViewPad(inputs[0], {firstViewShape, secondViewShape},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                    {bIdx * firstViewShape, sIdx * secondViewShape});
                Program::GetInstance().GetTileShape().SetVecTileShapes(args->tileShape_);
                auto res = MulS(tileTensor0, value);
                DAssemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

static void MulsOperationExeFuncTripleCut(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        SymbolicScalar thirdDim = inputs[0]->shape[2];

        auto args = static_cast<const MulsOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[0];
        const int thirdViewShape = args->viewShape_[2];
        int bloop = CeilDiv(firstDim, firstViewShape);
        int sloop = CeilDiv(secondDim, secondViewShape);
        int nloop = CeilDiv(thirdDim, thirdViewShape);

        std::vector<float> inputData(1, 0);
        readInput<float>(GetGoldenDir() + "/y.bin", inputData);
        Element value(DataType::DT_FP32, inputData[0]);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1)) {
                    auto tileTensor0 = DViewPad(inputs[0], {firstViewShape, secondViewShape, thirdViewShape},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                            std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                            std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                        {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});
                    Program::GetInstance().GetTileShape().SetVecTileShapes(args->tileShape_);
                    auto res = MulS(tileTensor0, value);
                    DAssemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

static void MulsOperationExeFuncQuadrupleCut(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        SymbolicScalar thirdDim = inputs[0]->shape[2];
        SymbolicScalar fourthDim = inputs[0]->shape[3];

        auto args = static_cast<const MulsOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[0];
        const int thirdViewShape = args->viewShape_[2];
        const int fourthViewShape = args->viewShape_[3];
        int bloop = CeilDiv(firstDim, firstViewShape);
        int sloop = CeilDiv(secondDim, secondViewShape);
        int nloop = CeilDiv(thirdDim, thirdViewShape);
        int qloop = CeilDiv(fourthDim, fourthViewShape);

        std::vector<float> inputData(1, 0);
        readInput<float>(GetGoldenDir() + "/y.bin", inputData);
        Element value(DataType::DT_FP32, inputData[0]);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1)) {
                    LOOP("LOOP_L3_qIdx", FunctionType::DYNAMIC_LOOP, qIdx, LoopRange(0, qloop, 1)) {
                        auto tileTensor0 =
                            DViewPad(inputs[0], {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                    std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                    std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape),
                                    std::min(fourthDim - qIdx * fourthViewShape, fourthViewShape)},
                                {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                    qIdx * fourthViewShape});
                        Program::GetInstance().GetTileShape().SetVecTileShapes(args->tileShape_);
                        auto res = MulS(tileTensor0, value);
                        DAssemble(res,
                            {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                qIdx * fourthViewShape},
                            outputs[0]);
                    }
                }
            }
        }
    }
}

static const MulsOperationMetadata testDataLists[] = {

    MulsOperationMetadata(
        0, MulsOpFuncArgs{        {64 * 48, 512},       {128, 128},     {32, 32}, DataType::DT_FP32},
         MulsOperationExeFuncDoubleCut),
    MulsOperationMetadata(
        1, MulsOpFuncArgs{    {64 * 48, 128 * 3},       {128, 128},     {32, 32}, DataType::DT_FP32},
         MulsOperationExeFuncDoubleCut),
    MulsOperationMetadata(
        2, MulsOpFuncArgs{     {64 * 32, 16 * 3},        {128, 64},     {32, 32}, DataType::DT_FP32},
         MulsOperationExeFuncDoubleCut),
    MulsOperationMetadata(3, MulsOpFuncArgs{{64 * 48 + 3, 128 * 3},       {128, 128},     {32, 32}, DataType::DT_FP32},
        MulsOperationExeFuncDoubleCut),
    MulsOperationMetadata(
        4, MulsOpFuncArgs{    {64 * 48, 128 * 3},       {128, 127},     {32, 32}, DataType::DT_FP32},
         MulsOperationExeFuncDoubleCut),
    MulsOperationMetadata(5, MulsOpFuncArgs{    {48 * 128, 64, 64},     {64, 64, 64}, {32, 32, 32}, DataType::DT_FP32},
        MulsOperationExeFuncTripleCut),
    MulsOperationMetadata(6, MulsOpFuncArgs{ {16, 16, 128, 64 / 2}, {16, 16, 16, 16}, {8, 8, 8, 8}, DataType::DT_FP32},
        MulsOperationExeFuncQuadrupleCut),
    MulsOperationMetadata(
        7, MulsOpFuncArgs{    {64 * 48, 128 * 3},       {128, 128},     {32, 32}, DataType::DT_FP32},
         MulsOperationExeFuncDoubleCut),
    MulsOperationMetadata(
        8, MulsOpFuncArgs{    {64 * 48, 128 * 3},       {128, 128},     {32, 32}, DataType::DT_FP32},
         MulsOperationExeFuncDoubleCut),
    MulsOperationMetadata(
        9, MulsOpFuncArgs{          {64 * 48, 0},       {128, 128},     {32, 32}, DataType::DT_FP32},
         MulsOperationExeFuncDoubleCut),
    MulsOperationMetadata(
        11, MulsOpFuncArgs{                {1, 1},         {32, 32},     {16, 16}, DataType::DT_FP32},
         MulsOperationExeFuncDoubleCut),
    MulsOperationMetadata(
        12, MulsOpFuncArgs{              {32, 32},         {32, 32},     {16, 16}, DataType::DT_FP32},
         MulsOperationExeFuncDoubleCut),
    MulsOperationMetadata(
        13, MulsOpFuncArgs{            {128, 128},         {32, 32},     {64, 64}, DataType::DT_FP32},
         MulsOperationExeFuncDoubleCut)
};

class MulsOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<MulsOperationMetadata> {};

INSTANTIATE_TEST_SUITE_P(TestMuls, MulsOperationTest, ::testing::ValuesIn(testDataLists));

TEST_P(MulsOperationTest, test_muls) {
    TestCaseDesc testCase;
    auto args = GetParam().args_;
    testCase.inputTensors = {Tensor(args.dType_, args.input0Shape_, "input0")};
    testCase.outputTensors = {Tensor(args.dType_, args.input0Shape_, "output")};
    testCase.args = &(GetParam().args_);
    testCase.opFunc = GetParam().opFunc_;
    testCase.inputPaths = {GetGoldenDir() + "/x.bin"};
    testCase.goldenPaths = {GetGoldenDir() + "/res.bin"};
    TestExecutor::runTest(testCase);
}
} // namespace