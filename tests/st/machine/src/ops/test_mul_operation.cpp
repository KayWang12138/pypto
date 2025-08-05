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
 * \file test_mul_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct MulOpFuncArgs : public OpFuncArgs {
    MulOpFuncArgs(std::vector<std::vector<int>> inputShapes, std::vector<int> viewShape, std::vector<int> vecTileShape, DataType dType) :
        inputShapes_(inputShapes), viewShape_(viewShape), vecTileShape_(vecTileShape), dType_(dType) {}
    int num_;
    std::vector<std::vector<int>> inputShapes_;
    std::vector<int> viewShape_;
    std::vector<int> vecTileShape_;
    DataType dType_;
};

struct MulOperationMetadata {
    MulOperationMetadata(std::vector<std::vector<int>> inputShapes, std::vector<int> viewShape, std::vector<int> vecTileShape,
        DataType dType, OpFunc opFunc) : args_(inputShapes, viewShape, vecTileShape, dType), opFunc_(opFunc) {}
    MulOpFuncArgs args_;
    OpFunc opFunc_;
};

static void MulOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                const OpFuncArgs* opArgs) {
    const MulOpFuncArgs  *mulInfos =  static_cast<const MulOpFuncArgs*>(opArgs);
    Program::GetInstance().GetTileShape().SetVecTileShapes(mulInfos->vecTileShape_);
    const int viewshape0 = mulInfos->viewShape_[0];
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0], inputs[1]}, {outputs[0]}) {
        SymbolicScalar firstDim1 = inputs[0]->shape[0];
        SymbolicScalar secondDim1 = inputs[0]->shape[1];
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, CeilDiv(firstDim1, viewshape0), 1)) {
            auto tileTensor0 = DViewPad(inputs[0], {viewshape0, secondDim1},
                {std::min(firstDim1 - bIdx * viewshape0, viewshape0), secondDim1},
                {bIdx * viewshape0, 0});
            auto tileTensor1 = DViewPad(inputs[1], {viewshape0, secondDim1},
                {std::min(firstDim1 - bIdx * viewshape0, viewshape0), secondDim1},
                {bIdx * viewshape0, 0});
            auto res = Mul(tileTensor0, tileTensor1);
            DAssemble(res, {bIdx * viewshape0, 0}, outputs[0]);
        }
    }
}

/* broadcast [32, 1]场景 */
static void MulOperationExeFuncBroadcastLastAxis(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                const OpFuncArgs* opArgs) {
    const MulOpFuncArgs  *mulInfo =  static_cast<const MulOpFuncArgs*>(opArgs);
    Program::GetInstance().GetTileShape().SetVecTileShapes(mulInfo->vecTileShape_);
    const int firstViewshape = mulInfo->viewShape_[0];
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0], inputs[1]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, CeilDiv(firstDim, firstViewshape), 1)) {
            auto tileTensor0 = DViewPad(inputs[0], {firstViewshape, secondDim},
                {std::min(firstDim - bIdx * firstViewshape, firstViewshape), secondDim},
                {bIdx * firstViewshape, 0});
            auto tileTensor1 = DViewPad(inputs[1], {firstViewshape, 1},
                {std::min(firstDim - bIdx * firstViewshape, firstViewshape), 1},
                {bIdx * firstViewshape, 0});
            auto res = Mul(tileTensor0, tileTensor1);
            DAssemble(res, {bIdx * firstViewshape, 0}, outputs[0]);
        }
    }
}

/* broadcast [1, 32]场景 */
static void MulOperationExeFuncBroadcastFirstAxis(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                const OpFuncArgs* opArgs) {
    const MulOpFuncArgs  *mulInfo =  static_cast<const MulOpFuncArgs*>(opArgs);
    Program::GetInstance().GetTileShape().SetVecTileShapes(mulInfo->vecTileShape_);
    const int firstl0LoopLengthTile = mulInfo->viewShape_[0];
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0], inputs[1]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, CeilDiv(firstDim, firstl0LoopLengthTile), 1)) {
            auto tileTensor0 = DViewPad(inputs[0], {firstl0LoopLengthTile, secondDim},
                {std::min(firstDim - bIdx * firstl0LoopLengthTile, firstl0LoopLengthTile), secondDim},
                {bIdx * firstl0LoopLengthTile, 0});
            auto tileTensor1 = DViewPad(inputs[1], {1, secondDim},
                {1, secondDim},
                {0, 0});
            auto res = Mul(tileTensor0, tileTensor1);
            DAssemble(res, {bIdx * firstl0LoopLengthTile, 0}, outputs[0]);
        }
    }
}

static void MulOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                const OpFuncArgs* opArgs) {
    const MulOpFuncArgs  *mulInfo =  static_cast<const MulOpFuncArgs*>(opArgs);
    Program::GetInstance().GetTileShape().SetVecTileShapes(mulInfo->vecTileShape_);
    const int firstViewShape = mulInfo->viewShape_[0];
    const int secondViewShape = mulInfo->viewShape_[1];
    const int broadcastFlag = 1;
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0], inputs[1]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, CeilDiv(firstDim, firstViewShape), 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, CeilDiv(secondDim, secondViewShape), 1)) {
                Tensor tileTensor0;
                Tensor tileTensor1;
                IF(inputs[0]->shape[1] != broadcastFlag && inputs[1]->shape[1] == broadcastFlag) {
                    tileTensor0 = DViewPad(inputs[0], {firstViewShape, secondViewShape},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                            std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                        {bIdx * firstViewShape, sIdx * secondViewShape});
                    tileTensor1 = DViewPad(inputs[1], {firstViewShape, 1},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape), 1},
                        {bIdx * firstViewShape, 0});
                } ELSE IF(inputs[0]->shape[0] != broadcastFlag && inputs[1]->shape[0] == broadcastFlag) {
                    tileTensor0 = DViewPad(inputs[0], {firstViewShape, secondViewShape},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                            std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                        {bIdx * firstViewShape, sIdx * secondViewShape});
                    tileTensor1 = DViewPad(inputs[1], {1, secondViewShape},
                        {1, std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                        {0, sIdx * secondViewShape});
                } ELSE {
                    tileTensor0 = DViewPad(inputs[0], {firstViewShape, secondViewShape},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                            std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                        {bIdx * firstViewShape, sIdx * secondViewShape});
                    tileTensor1 = DViewPad(inputs[1], {firstViewShape, secondViewShape},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                            std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                        {bIdx * firstViewShape, sIdx * secondViewShape});
                }
                auto res = Mul(tileTensor0, tileTensor1);
                DAssemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

static void MulOperationExeFunc4Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                const OpFuncArgs* opArgs) {
    const MulOpFuncArgs  *mulInfo =  static_cast<const MulOpFuncArgs*>(opArgs);
    Program::GetInstance().GetTileShape().SetVecTileShapes(mulInfo->vecTileShape_);
    const int firstViewShape    = mulInfo->viewShape_[0];
    const int secondViewShape   = mulInfo->viewShape_[1];
    const int thirdViewShape    = mulInfo->viewShape_[2];
    const int fourthViewShape   = mulInfo->viewShape_[3];
    const int broadcastFlag = 1;
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0], inputs[1]}, {outputs[0]}) {
        SymbolicScalar firstDim     = inputs[0]->shape[0];
        SymbolicScalar secondDim    = inputs[0]->shape[1];
        SymbolicScalar thirdDim     = inputs[0]->shape[2];
        SymbolicScalar fourthDim    = inputs[0]->shape[3];

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, CeilDiv(firstDim, firstViewShape), 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, CeilDiv(secondDim, secondViewShape), 1)) {
                LOOP("LOOP_L2_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, CeilDiv(thirdDim, thirdViewShape), 1)) {
                    LOOP("LOOP_L3_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, CeilDiv(fourthDim, fourthViewShape), 1)) {
                        Tensor tileTensor0;
                        Tensor tileTensor1;
                        IF(inputs[1]->shape[2] == broadcastFlag && inputs[0]->shape[2] != broadcastFlag) {
                            // case 26 [16, 16, 1, 16] broadcast场景
                            // case 27 [1, 1, 1, 16] broadcast场景
                            tileTensor0 = DViewPad(inputs[0], {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                 std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                 std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape),
                                 std::min(fourthDim - nIdx * fourthViewShape, fourthViewShape)},
                                {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape, nIdx * fourthViewShape});
                            tileTensor1 = DViewPad(inputs[1], {firstViewShape, secondViewShape, 1, fourthViewShape},
                                {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                 std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                 1,
                                 std::min(fourthDim - nIdx * fourthViewShape, fourthViewShape)},
                                {bIdx * firstViewShape, sIdx * secondViewShape, 0, nIdx * fourthViewShape});
                        } ELSE {
                            tileTensor0 = DViewPad(inputs[0], {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                 std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                 std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape),
                                 std::min(fourthDim - nIdx * fourthViewShape, fourthViewShape)},
                                {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape, nIdx * fourthViewShape});
                            tileTensor1 = DViewPad(inputs[1], {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                                {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                 std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                 std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape),
                                 std::min(fourthDim - nIdx * fourthViewShape, fourthViewShape)},
                                {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape, nIdx * fourthViewShape});
                        }
                        auto res = Mul(tileTensor0, tileTensor1);
                        DAssemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, mIdx * thirdViewShape, nIdx * fourthViewShape}, outputs[0]);
                    }
                }
            }
        }
    }
}

static const MulOperationMetadata testDataLists[] = {
    MulOperationMetadata{{{32, 1536}, {32, 1536}}, {32, 32}, {16, 16}, DataType::DT_FP32, MulOperationExeFunc},
    MulOperationMetadata{{{71, 576}, {71, 576}}, {32, 32}, {16, 16}, DataType::DT_FP32, MulOperationExeFunc2Dims},
    MulOperationMetadata{{{31, 1}, {31, 1}}, {32, 32}, {16, 16}, DataType::DT_FP32, MulOperationExeFunc2Dims},
    MulOperationMetadata{{{8, 128}, {8, 128}}, {32, 32}, {16, 16}, DataType::DT_FP32, MulOperationExeFunc2Dims},
    MulOperationMetadata{{{1, 1}, {1, 1}}, {32, 32}, {16, 16}, DataType::DT_FP32, MulOperationExeFunc2Dims},
    MulOperationMetadata{{{107, 145}, {107, 145}}, {32, 32}, {16, 16}, DataType::DT_FP32, MulOperationExeFunc2Dims},
    MulOperationMetadata{{{32, 32}, {32, 32}}, {32, 32}, {16, 16}, DataType::DT_FP32, MulOperationExeFunc2Dims},
    MulOperationMetadata{{{128, 128}, {128, 128}}, {32, 32}, {64, 64}, DataType::DT_FP32, MulOperationExeFunc2Dims},
    MulOperationMetadata{{{192, 512}, {192, 512}}, {32, 32}, {32, 32}, DataType::DT_FP32, MulOperationExeFunc2Dims},
    MulOperationMetadata{{{2, 2, 2, 65}, {2, 2, 2, 65}}, {32, 32, 32, 32}, {8, 8, 8, 8}, DataType::DT_FP32, MulOperationExeFunc4Dims},
    MulOperationMetadata{{{32, 128, 64, 192}, {32, 128, 64, 192}}, {32, 32, 32, 32}, {8, 8, 8, 8}, DataType::DT_FP32, MulOperationExeFunc4Dims},
    MulOperationMetadata{{{32, 32}, {32, 32}}, {32, 32}, {32, 32}, DataType::DT_FP32, MulOperationExeFunc2Dims},
    MulOperationMetadata{{{32, 32}, {32, 32}}, {32, 32}, {32, 32}, DataType::DT_FP32, MulOperationExeFunc2Dims},
    MulOperationMetadata{{{32, 32}, {32, 32}}, {32, 32}, {32, 32}, DataType::DT_FP32, MulOperationExeFunc},
    MulOperationMetadata{{{32, 32}, {32, 32}}, {32, 32}, {32, 32}, DataType::DT_FP32, MulOperationExeFunc},
    MulOperationMetadata{{{32, 32}, {32, 1}}, {32, 32}, {32, 32}, DataType::DT_FP32, MulOperationExeFuncBroadcastLastAxis},
    MulOperationMetadata{{{32, 32}, {1, 32}}, {32, 32}, {32, 32}, DataType::DT_FP32, MulOperationExeFuncBroadcastFirstAxis},
    MulOperationMetadata{{{1, 1, 16, 16}, {1, 1, 1, 16}}, {1, 1, 16, 16}, {1, 1, 16, 16}, DataType::DT_FP32, MulOperationExeFunc4Dims},
    MulOperationMetadata{{{16, 16, 16, 16}, {16, 16, 1, 16}}, {16, 16, 16, 16}, {8, 8, 8, 8}, DataType::DT_FP32, MulOperationExeFunc4Dims},
    MulOperationMetadata{{{16, 128, 10, 2}, {16, 128, 10, 2}}, {32, 32, 32, 32}, {8, 8, 8, 8}, DataType::DT_FP32, MulOperationExeFunc4Dims},
};

class MulOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<MulOperationMetadata> {};

INSTANTIATE_TEST_SUITE_P(
    TestMul,
    MulOperationTest,
    ::testing::ValuesIn(testDataLists)
);

TEST_P(MulOperationTest, TestMul) {
    TestCaseDesc testCase;
    testCase.inputTensors = {
        Tensor(GetParam().args_.dType_, GetParam().args_.inputShapes_[0], "input0"),
        Tensor(GetParam().args_.dType_, GetParam().args_.inputShapes_[1], "input1")
    };
    testCase.outputTensors = {
        Tensor(GetParam().args_.dType_, GetParam().args_.inputShapes_[0], "output")
    };
    testCase.inputPaths = {GetGoldenDir() + "/x.bin", GetGoldenDir() + "/y.bin"};
    testCase.goldenPaths = {GetGoldenDir() + "/res.bin"};
    testCase.args = &(GetParam().args_);
    testCase.opFunc = GetParam().opFunc_;
    TestExecutor::runTest(testCase);
}
} // namespace