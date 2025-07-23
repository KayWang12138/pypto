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

using namespace ascend::test_operation;
namespace {
struct AddsOpFuncArgs : public OpFuncArgs {
    AddsOpFuncArgs(std::vector<int> shape, std::vector<int> vecViewShapes, std::vector<int> vecTileShapes,
        DataType dType) : shape_(shape), vecViewShapes_(vecViewShapes), vecTileShapes_(vecTileShapes), dType_(dType) {}
    std::vector<int> shape_;
    std::vector<int> vecViewShapes_;
    std::vector<int> vecTileShapes_;
    DataType dType_;
};

struct AddsOperationMetadata {
    AddsOperationMetadata(int caseIndex, AddsOpFuncArgs args, OpFunc opFunc) :
        caseIndex_(caseIndex), args_(args), opFunc_(opFunc) {}
    int caseIndex_;
    AddsOpFuncArgs args_;
    OpFunc opFunc_;
};

static void AddsOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                 const OpFuncArgs* opArgs) {                               
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        const struct AddsOpFuncArgs *args = static_cast<const AddsOpFuncArgs*>(opArgs);
        const int firstl0LoopLengthTile = args->vecViewShapes_[0];
        const int bloop = CeilDiv(firstDim, firstl0LoopLengthTile);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            auto tileTensor0 = DViewPad(inputs[0], {firstl0LoopLengthTile, secondDim},
                {std::min(firstDim - bIdx * firstl0LoopLengthTile, firstl0LoopLengthTile), secondDim},
                {bIdx * firstl0LoopLengthTile, 0});
            Program::GetInstance().GetTileShape().SetVecTileShapes(args->vecTileShapes_);
            auto res = AddS(tileTensor0, Element(DataType::DT_FP32, 1e-5f));
            DAssemble(res, {bIdx * firstl0LoopLengthTile, 0}, outputs[0]);
        }
    }
}

static void AddsOperationExeFuncDoubleCut(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                          const OpFuncArgs* opArgs) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);                                
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        std::vector<float> inputData(1, 0);
        readInput<float>(GetGoldenDir() + "/y.bin", inputData);
        Element value(DataType::DT_FP32, inputData[0]);
        auto *args = static_cast<const AddsOpFuncArgs*>(opArgs);
        const int firstViewShape = args->vecViewShapes_[0];
        const int secondViewShape = args->vecViewShapes_[1];
        int bloop = CeilDiv(firstDim, firstViewShape);
        int sloop = CeilDiv(secondDim, secondViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                auto tileTensor0 = DViewPad(inputs[0], {firstViewShape, secondViewShape},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                    {bIdx * firstViewShape, sIdx * secondViewShape});
                Program::GetInstance().GetTileShape().SetVecTileShapes(args->vecTileShapes_);
                auto res = AddS(tileTensor0, value);
                DAssemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

static void AddsOperationExeFuncTripleCut(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                          const OpFuncArgs* opArgs) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);                              
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        SymbolicScalar thirdDim = inputs[0]->shape[2];
        const struct AddsOpFuncArgs *args = static_cast<const AddsOpFuncArgs*>(opArgs);
        const int firstViewShape = args->vecViewShapes_[0];
        const int secondViewShape = args->vecViewShapes_[1];
        const int thirdViewShape = args->vecViewShapes_[2];
        int bloop = CeilDiv(firstDim, firstViewShape);
        int sloop = CeilDiv(secondDim, secondViewShape);
        int nloop = CeilDiv(thirdDim, thirdViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1)) {
                    auto tileTensor0 = DViewPad(inputs[0], {firstViewShape, secondViewShape, thirdViewShape},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                            std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                        {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});
                    Program::GetInstance().GetTileShape().SetVecTileShapes(args->vecTileShapes_);
                    auto res = AddS(tileTensor0, Element(DataType::DT_FP32, 1.0f/24));
                    DAssemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

static void AddsOperationExeFuncQuadrupleCut(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                             const OpFuncArgs* opArgs) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);                                  
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        SymbolicScalar thirdDim = inputs[0]->shape[2];
        SymbolicScalar fourthDim = inputs[0]->shape[3];
        const struct AddsOpFuncArgs *args = static_cast<const AddsOpFuncArgs*>(opArgs);
        const int firstViewShape = args->vecViewShapes_[0];
        const int secondViewShape = args->vecViewShapes_[1];
        const int thirdViewShape = args->vecViewShapes_[2];
        const int fourthViewShape = args->vecViewShapes_[3];
        int bloop = CeilDiv(firstDim, firstViewShape);
        int sloop = CeilDiv(secondDim, secondViewShape);
        int nloop = CeilDiv(thirdDim, thirdViewShape);
        int qloop = CeilDiv(fourthDim, fourthViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1)) {
                    LOOP("LOOP_L3_qIdx", FunctionType::DYNAMIC_LOOP, qIdx, LoopRange(0, qloop, 1)) {
                        auto tileTensor0 = DViewPad(inputs[0], {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                    std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape),
                                        std::min(fourthDim - qIdx * fourthViewShape, fourthViewShape)},
                            {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape, qIdx * fourthViewShape});
                        Program::GetInstance().GetTileShape().SetVecTileShapes(args->vecTileShapes_);
                        auto res = AddS(tileTensor0, Element(DataType::DT_FP32, 1.0f));
                        DAssemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape, qIdx * fourthViewShape}, outputs[0]);
                    }
                }
            }
        }
    }
}

static const AddsOperationMetadata testDataLists[] = {
    AddsOperationMetadata{0, AddsOpFuncArgs{{64*48, 1}, {128, 1}, {32, 32}, DataType::DT_FP32},
                          AddsOperationExeFunc},
    AddsOperationMetadata{1, AddsOpFuncArgs{{64*48, 128*3}, {128, 128}, {32, 32}, DataType::DT_FP32},
                          AddsOperationExeFuncDoubleCut},
    AddsOperationMetadata{2, AddsOpFuncArgs{{64*32, 16*3}, {128, 128}, {32, 32}, DataType::DT_FP32},
                          AddsOperationExeFuncDoubleCut},
    AddsOperationMetadata{3, AddsOpFuncArgs{{64*32 + 3, 16*3}, {128, 128}, {32, 32}, DataType::DT_FP32},
                          AddsOperationExeFuncDoubleCut},
    AddsOperationMetadata{4, AddsOpFuncArgs{{64*48, 128*3}, {128, 127}, {32, 32}, DataType::DT_FP32},
                          AddsOperationExeFuncDoubleCut},
    AddsOperationMetadata{5, AddsOpFuncArgs{{48*128, 64, 64}, {64, 64, 64}, {32, 32, 32}, DataType::DT_FP32},
                          AddsOperationExeFuncTripleCut},
    AddsOperationMetadata{6, AddsOpFuncArgs{{1, 1}, {32, 32}, {16, 16}, DataType::DT_FP32},
                          AddsOperationExeFuncDoubleCut},
    AddsOperationMetadata{7, AddsOpFuncArgs{{32, 32}, {32, 32}, {16, 16}, DataType::DT_FP32},
                          AddsOperationExeFuncDoubleCut},
    AddsOperationMetadata{8, AddsOpFuncArgs{{128, 128}, {32, 32}, {64, 64}, DataType::DT_FP32},
                          AddsOperationExeFuncDoubleCut},
    AddsOperationMetadata{9, AddsOpFuncArgs{{48, 64, 128, 32}, {48, 64, 32, 16}, {16, 16, 16, 8}, DataType::DT_FP32},
                          AddsOperationExeFuncQuadrupleCut},
    AddsOperationMetadata{10, AddsOpFuncArgs{{64*48, 0}, {128, 128}, {32, 32}, DataType::DT_FP32},
                          AddsOperationExeFuncDoubleCut},
    AddsOperationMetadata{11, AddsOpFuncArgs{{0, 128*3}, {128, 128}, {32, 32}, DataType::DT_FP32},
                          AddsOperationExeFuncDoubleCut},
    AddsOperationMetadata{12, AddsOpFuncArgs{{64*48, 128*3}, {128, 128}, {32, 32}, DataType::DT_FP32},
                          AddsOperationExeFuncDoubleCut},
    AddsOperationMetadata{13, AddsOpFuncArgs{{96*1024, 128*3}, {12*512, 96}, {128, 32}, DataType::DT_FP32},
                          AddsOperationExeFuncDoubleCut},
    AddsOperationMetadata{14, AddsOpFuncArgs{{64, 64*576}, {32, 1024}, {32, 128}, DataType::DT_FP32},
                          AddsOperationExeFuncDoubleCut},
    AddsOperationMetadata{15, AddsOpFuncArgs{{64, 64*512}, {32, 1024}, {32, 128}, DataType::DT_FP32},
                          AddsOperationExeFuncDoubleCut},
    AddsOperationMetadata{16, AddsOpFuncArgs{{64*32, 7168*4}, {1024, 1024}, {32, 512}, DataType::DT_FP32},
                          AddsOperationExeFuncDoubleCut},
    AddsOperationMetadata{17, AddsOpFuncArgs{{96, 64*576}, {32, 1024}, {32, 128}, DataType::DT_FP32},
                          AddsOperationExeFuncDoubleCut},
    AddsOperationMetadata{18, AddsOpFuncArgs{{96, 64*512}, {32, 1024}, {32, 128}, DataType::DT_FP32},
                          AddsOperationExeFuncDoubleCut},
    AddsOperationMetadata{19, AddsOpFuncArgs{{768*1024, 128*3}, {32*1024, 128}, {128, 32}, DataType::DT_FP32},
                          AddsOperationExeFuncDoubleCut},
};

class AddsOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<AddsOperationMetadata> {};

INSTANTIATE_TEST_SUITE_P(
    TestAdds,
    AddsOperationTest,
    ::testing::ValuesIn(testDataLists)
);

TEST_P(AddsOperationTest, test_adds) {
    TestCaseDesc testAddsCase;
    testAddsCase.inputTensors = {
        Tensor(GetParam().args_.dType_, GetParam().args_.shape_, "input0")
    };
    testAddsCase.outputTensors = {
        Tensor(GetParam().args_.dType_, GetParam().args_.shape_, "output")
    };
    testAddsCase.inputPaths = {GetGoldenDir() + "/x.bin"};
    testAddsCase.goldenPaths = {GetGoldenDir() + "/res.bin"};
    testAddsCase.args = &(GetParam().args_);
    testAddsCase.opFunc = GetParam().opFunc_;
    TestExecutor::runTest(testAddsCase);
}
} // namespace