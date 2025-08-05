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
 * \file test_add_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct AddOpFuncArgs : public OpFuncArgs {
    AddOpFuncArgs(std::vector<int> shape, std::vector<int> vecTileShapes, DataType dType) :
        shape_(shape), vecTileShapes_(vecTileShapes), dType_(dType) {}
    std::vector<int> shape_;
    std::vector<int> vecTileShapes_;
    DataType dType_;
};

struct AddOperationMetadata {
    AddOperationMetadata(std::vector<int> shape, std::vector<int> vecTileShapes, DataType dType, OpFunc opFunc) :
        args_(shape, vecTileShapes, dType), opFunc_(opFunc) {}
    AddOpFuncArgs args_;
    OpFunc opFunc_;
};

static void AddOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                const OpFuncArgs* opArgs) {
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0], inputs[1]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        const int firstl0LoopLengthTile = 128;

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, CeilDivSymbolicScalar(firstDim, firstl0LoopLengthTile), 1)) {
            auto tileTensor0 = DViewPad(inputs[0], {firstl0LoopLengthTile, secondDim},
                {std::min(firstDim - bIdx * firstl0LoopLengthTile, firstl0LoopLengthTile), secondDim},
                {bIdx * firstl0LoopLengthTile, 0});
            auto tileTensor1 = DViewPad(inputs[1], {firstl0LoopLengthTile, secondDim},
                {std::min(firstDim - bIdx * firstl0LoopLengthTile, firstl0LoopLengthTile), secondDim},
                {bIdx * firstl0LoopLengthTile, 0});
            Program::GetInstance().GetTileShape().SetVecTileShapes(
                (static_cast<const AddOpFuncArgs*>(opArgs))->vecTileShapes_);
            auto res = Add(tileTensor0, tileTensor1);
            DAssemble(res, {bIdx * firstl0LoopLengthTile, 0}, outputs[0]);
        }
    }
}

static void AddOperationExeFuncDoubleCut(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                         const OpFuncArgs* opArgs) {
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0], inputs[1]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        const int firstViewShape = 128;
        const int secondViewShape = 128;

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, CeilDivSymbolicScalar(firstDim, firstViewShape), 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, CeilDivSymbolicScalar(secondDim, secondViewShape), 1)) {
                auto tileTensor0 = DViewPad(inputs[0], {firstViewShape, secondViewShape},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                    {bIdx * firstViewShape, sIdx * secondViewShape});
                auto tileTensor1 = DViewPad(inputs[1], {firstViewShape, secondViewShape},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                    {bIdx * firstViewShape, sIdx * secondViewShape});
                Program::GetInstance().GetTileShape().SetVecTileShapes(
                    (static_cast<const AddOpFuncArgs*>(opArgs))->vecTileShapes_);
                auto res = Add(tileTensor0, tileTensor1);
                DAssemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

[[maybe_unused]] static void AddOperationExeFuncDoubleCutWithBrocast(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0], inputs[1]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        const int firstViewShape = 128;
        const int secondViewShape = 128;

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx,
            LoopRange(0, CeilDivSymbolicScalar(firstDim, firstViewShape), 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx,
                LoopRange(0, CeilDivSymbolicScalar(secondDim, secondViewShape), 1)) {
                auto tileTensor0 = DViewPad(inputs[0], {firstViewShape, secondViewShape},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                    {bIdx * firstViewShape, sIdx * secondViewShape});
                auto tileTensor1 = DViewPad(inputs[1], {firstViewShape, 1},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape), 1}, {bIdx * firstViewShape, 0});
                Program::GetInstance().GetTileShape().SetVecTileShapes(
                    (static_cast<const AddOpFuncArgs *>(opArgs))->vecTileShapes_);
                auto res = Add(tileTensor0, tileTensor1);
                DAssemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

[[maybe_unused]] static void AddOperationExeFuncFourCut(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0], inputs[1]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        SymbolicScalar thirdDim = inputs[0]->shape[2];
        SymbolicScalar fourthDim = inputs[0]->shape[3];
        const int firstViewShape = 128;
        const int secondViewShape = 128;
        const int thirdViewShape = 128;
        const int fourthViewShape = 128;

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx,
            LoopRange(0, CeilDivSymbolicScalar(firstDim, firstViewShape), 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx,
                LoopRange(0, CeilDivSymbolicScalar(secondDim, secondViewShape), 1)) {
                LOOP("LOOP_L2_sIdx", FunctionType::DYNAMIC_LOOP, nIdx,
                    LoopRange(0, CeilDivSymbolicScalar(thirdDim, thirdViewShape), 1)) {
                    LOOP("LOOP_L3_sIdx", FunctionType::DYNAMIC_LOOP, dIdx,
                        LoopRange(0, CeilDivSymbolicScalar(fourthDim, fourthViewShape), 1)) {
                        auto tileTensor0 =
                            DViewPad(inputs[0], {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                    std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                    std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape),
                                    std::min(fourthDim - dIdx * fourthViewShape, fourthViewShape)},
                                {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape, dIdx * fourthViewShape});
                        auto tileTensor1 = DViewPad(inputs[1], {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape),
                                std::min(fourthDim - dIdx * fourthViewShape, fourthViewShape)},
                            {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape, dIdx * fourthViewShape});
                        Program::GetInstance().GetTileShape().SetVecTileShapes(
                            (static_cast<const AddOpFuncArgs *>(opArgs))->vecTileShapes_);
                        auto res = Add(tileTensor0, tileTensor1);
                        DAssemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape, dIdx * fourthViewShape}, outputs[0]);
                    }
                }
            }
        }
    }
}

[[maybe_unused]] static void AddOperationExeFuncFourCutWithLastDimBrocast(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0], inputs[1]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        SymbolicScalar thirdDim = inputs[0]->shape[2];
        SymbolicScalar fourthDim = inputs[0]->shape[3];
        const int firstViewShape = 128;
        const int secondViewShape = 128;
        const int thirdViewShape = 128;
        const int fourthViewShape = 128;

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx,
            LoopRange(0, CeilDivSymbolicScalar(firstDim, firstViewShape), 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx,
                LoopRange(0, CeilDivSymbolicScalar(secondDim, secondViewShape), 1)) {
                LOOP("LOOP_L2_sIdx", FunctionType::DYNAMIC_LOOP, nIdx,
                    LoopRange(0, CeilDivSymbolicScalar(thirdDim, thirdViewShape), 1)) {
                    LOOP("LOOP_L3_sIdx", FunctionType::DYNAMIC_LOOP, dIdx,
                        LoopRange(0, CeilDivSymbolicScalar(fourthDim, fourthViewShape), 1)) {
                        auto tileTensor0 =
                            DViewPad(inputs[0], {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                    std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                    std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape),
                                    std::min(fourthDim - dIdx * fourthViewShape, fourthViewShape)},
                                {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                    dIdx * fourthViewShape});
                        auto tileTensor1 = DViewPad(inputs[1], {firstViewShape, secondViewShape, thirdViewShape, 1},
                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape), 1},
                            {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape, 0});
                        Program::GetInstance().GetTileShape().SetVecTileShapes(
                            (static_cast<const AddOpFuncArgs *>(opArgs))->vecTileShapes_);
                        auto res = Add(tileTensor0, tileTensor1);
                        DAssemble(res,
                            {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape,
                                dIdx * fourthViewShape},
                            outputs[0]);
                    }
                }
            }
        }
    }
}

static const AddOperationMetadata testDataLists[] = {
    AddOperationMetadata({512, 128}, {64, 128}, DataType::DT_FP32, AddOperationExeFunc),
    AddOperationMetadata({1024, 256}, {64, 128}, DataType::DT_FP32, AddOperationExeFunc),
    AddOperationMetadata({512, 256}, {64, 64}, DataType::DT_FP32, AddOperationExeFuncDoubleCut),
    AddOperationMetadata({512, 128}, {64, 64}, DataType::DT_FP32, AddOperationExeFuncDoubleCut),
    AddOperationMetadata({128 + 17, 128}, {32, 32}, DataType::DT_FP32, AddOperationExeFuncDoubleCut),
    AddOperationMetadata({90 + 17, 128}, {32, 32}, DataType::DT_FP32, AddOperationExeFuncDoubleCut),
    AddOperationMetadata({90 + 17, 128 + 17}, {16, 16}, DataType::DT_FP32, AddOperationExeFuncDoubleCut),
    AddOperationMetadata({1, 1}, {16, 16}, DataType::DT_FP32, AddOperationExeFuncDoubleCut),
    AddOperationMetadata({32, 32}, {16, 16}, DataType::DT_FP32, AddOperationExeFuncDoubleCut),
    AddOperationMetadata({128, 128}, {64, 64}, DataType::DT_FP32, AddOperationExeFuncDoubleCut),
    AddOperationMetadata({192, 512}, {32, 32}, DataType::DT_FP32, AddOperationExeFuncDoubleCut),
};

class AddOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<AddOperationMetadata> {};

INSTANTIATE_TEST_SUITE_P(
    TestAdd,
    AddOperationTest,
    ::testing::ValuesIn(testDataLists)
);

TEST_P(AddOperationTest, test_add) {
    TestCaseDesc testCase;
    testCase.inputTensors = {
        Tensor(GetParam().args_.dType_, GetParam().args_.shape_, "input0"),
        Tensor(GetParam().args_.dType_, GetParam().args_.shape_, "input1")
    };
    testCase.outputTensors = {
        Tensor(GetParam().args_.dType_, GetParam().args_.shape_, "output")
    };
    testCase.inputPaths = {GetGoldenDir() + "/x.bin", GetGoldenDir() + "/y.bin"};
    testCase.goldenPaths = {GetGoldenDir() + "/res.bin"};
    testCase.args = &(GetParam().args_);
    testCase.opFunc = GetParam().opFunc_;
    TestExecutor::runTest(testCase);
}
} // namespace