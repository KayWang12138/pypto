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
 * \file test_divs_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace ascend::test_operation;
namespace {
struct DivsOpFuncArgs : public OpFuncArgs {
    DivsOpFuncArgs(std::vector<int> shape, std::vector<int> vecTileShape, std::vector<int> tileShape, DataType dType) :
        shape_(shape), vecTileShape_(vecTileShape), tileShape_(tileShape), dType_(dType) {}
    std::vector<int> shape_;
    std::vector<int> vecTileShape_;
    std::vector<int> tileShape_;
    DataType dType_;
};
struct DivsOperationMetaData {
    DivsOperationMetaData(std::vector<int> shape, std::vector<int> vecTileShape, std::vector<int> tileShape,
        DataType dType, OpFunc opFunc) :
        args_(shape, vecTileShape, tileShape, dType), opFunc_(opFunc) {}
    DivsOpFuncArgs args_;
    OpFunc opFunc_;
};

static void DivsOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs, const OpFuncArgs *opArgs) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0]}) {
        const DivsOpFuncArgs* args = static_cast<const DivsOpFuncArgs*>(opArgs);
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        std::vector<float> divNum(1, 0);
        readInput<float>(GetGoldenDir() + "/y.bin", divNum);
        const int firstViewShape = args->vecTileShape_[0];
        const int secondViewShape = args->vecTileShape_[1];
        int bloop = CeilDiv(firstDim, firstViewShape);
        int sloop = CeilDiv(secondDim, secondViewShape);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                auto tileTensor0 = DViewPad(inputs[0], {firstViewShape, secondViewShape},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                    {bIdx * firstViewShape, sIdx * secondViewShape});
                if (args->tileShape_[0] != 0 && args->tileShape_[1] != 0) {
                    Program::GetInstance().GetTileShape().SetVecTileShapes(args->tileShape_);
                }
                auto res = DivS(tileTensor0, Element(args->dType_, divNum[0]));
                DAssemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

static void DivsOperationExeFuncDim3(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs, const OpFuncArgs *opArgs) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);                              
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0]}) {
        const DivsOpFuncArgs* args = static_cast<const DivsOpFuncArgs*>(opArgs);
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        SymbolicScalar thirdDim = inputs[0]->shape[2];
        std::vector<float> divNum(1, 0);
        readInput<float>(GetGoldenDir() + "/y.bin", divNum);
        const int firstViewShape = args->vecTileShape_[0];
        const int secondViewShape = args->vecTileShape_[1];
        const int thirdViewShape = args->vecTileShape_[2];
        int bloop = CeilDiv(firstDim, firstViewShape);
        int sloop = CeilDiv(secondDim, secondViewShape);
        int nloop = CeilDiv(thirdDim, thirdViewShape);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1)) {
                    auto tileTensor0 = DViewPad(inputs[0], {firstViewShape, secondViewShape, thirdViewShape},
                        {
                            std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                            std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                            std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)
                        },
                        {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});
                    Program::GetInstance().GetTileShape().SetVecTileShapes(args->tileShape_);
                    auto res = DivS(tileTensor0, Element(args->dType_, divNum[0]));
                    DAssemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

static void DivsOperationExeFuncDim4(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs, const OpFuncArgs *opArgs) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);                                  
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0]}) {
        const DivsOpFuncArgs* args = static_cast<const DivsOpFuncArgs*>(opArgs);
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        SymbolicScalar thirdDim = inputs[0]->shape[2];
        SymbolicScalar fourthDim = inputs[0]->shape[3];
        std::vector<float> divNum(1, 0);
        readInput<float>(GetGoldenDir() + "/y.bin", divNum);
        const int firstViewShape = args->vecTileShape_[0];
        const int secondViewShape = args->vecTileShape_[1];
        const int thirdViewShape = args->vecTileShape_[2];
        const int fourthViewShape = args->vecTileShape_[3];
        int bloop = CeilDiv(firstDim, firstViewShape);
        int sloop = CeilDiv(secondDim, secondViewShape);
        int nloop = CeilDiv(thirdDim, thirdViewShape);
        int qloop = CeilDiv(fourthDim, fourthViewShape);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1)) {
                    LOOP("LOOP_L3_qIdx", FunctionType::DYNAMIC_LOOP, qIdx, LoopRange(0, qloop, 1)) {
                        auto tileTensor0 = DViewPad(inputs[0], {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                            {
                                std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape),
                                std::min(fourthDim - qIdx * fourthViewShape, fourthViewShape)
                            },
                            {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape, qIdx * fourthViewShape});
                        Program::GetInstance().GetTileShape().SetVecTileShapes(args->tileShape_);
                        auto res = DivS(tileTensor0, Element(args->dType_, divNum[0]));
                        DAssemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape, qIdx * fourthViewShape}, outputs[0]);
                    }
                }
            }
        }
    }
}

static const DivsOperationMetaData testDataLists[] = {
    DivsOperationMetaData{{64*48, 512}, {128, 128}, {32, 32}, DataType::DT_FP32, DivsOperationExeFunc},
    DivsOperationMetaData{{64*48, 128*3}, {128, 128}, {32, 32}, DataType::DT_FP32, DivsOperationExeFunc},
    DivsOperationMetaData{{64*32, 16*3}, {128, 128}, {32, 32}, DataType::DT_FP32, DivsOperationExeFunc},
    DivsOperationMetaData{{64*32+3, 16*3}, {128, 128}, {32, 32}, DataType::DT_FP32, DivsOperationExeFunc},
    DivsOperationMetaData{{64*48, 128*3}, {128, 127}, {32, 32}, DataType::DT_FP32, DivsOperationExeFunc},
    DivsOperationMetaData{{48*128, 64, 64}, {64, 64, 64}, {32, 32, 32}, DataType::DT_FP32, DivsOperationExeFuncDim3},
    DivsOperationMetaData{{16, 16, 128, 32}, {16, 16, 16, 16}, {8, 8, 8, 8}, DataType::DT_FP32, DivsOperationExeFuncDim4},
    DivsOperationMetaData{{64*48, 128*3}, {128, 128}, {32, 32}, DataType::DT_FP32, DivsOperationExeFunc},
    DivsOperationMetaData{{64*48, 128*3}, {128, 128}, {32, 32}, DataType::DT_FP32, DivsOperationExeFunc},
    DivsOperationMetaData{{64*48, 0}, {128, 128}, {32, 32}, DataType::DT_FP32, DivsOperationExeFunc},
    DivsOperationMetaData{{64*48, 128*3}, {128, 128}, {32, 32}, DataType::DT_FP32, DivsOperationExeFunc},
    DivsOperationMetaData{{64*48, 128*3}, {128, 128}, {32, 32}, DataType::DT_FP32, DivsOperationExeFunc},
    DivsOperationMetaData{{64*48, 128*3}, {128, 128}, {32, 32}, DataType::DT_FP32, DivsOperationExeFunc},
    DivsOperationMetaData{{64*48, 128*3}, {128, 128}, {32, 32}, DataType::DT_FP32, DivsOperationExeFunc},
    DivsOperationMetaData{{64*48, 128*3}, {128, 128}, {32, 32}, DataType::DT_FP32, DivsOperationExeFunc},
    DivsOperationMetaData{{1, 1}, {32, 32}, {16, 16}, DataType::DT_FP32, DivsOperationExeFunc},
    DivsOperationMetaData{{32, 32}, {32, 32}, {16, 16}, DataType::DT_FP32, DivsOperationExeFunc},
    DivsOperationMetaData{{128, 128}, {32, 32}, {64, 64}, DataType::DT_FP32, DivsOperationExeFunc},
    DivsOperationMetaData{{16, 1536}, {32, 32}, {64, 64}, DataType::DT_FP32, DivsOperationExeFunc},
    DivsOperationMetaData{{16, 512}, {32, 32}, {64, 64}, DataType::DT_FP32, DivsOperationExeFunc},
    DivsOperationMetaData{{16, 128*3}, {32, 32}, {64, 64}, DataType::DT_FP32, DivsOperationExeFunc},
};

class DivsOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<DivsOperationMetaData> {};

INSTANTIATE_TEST_SUITE_P(
    TestDivs,
    DivsOperationTest,
    ::testing::ValuesIn(testDataLists)
);

TEST_P(DivsOperationTest, test_divs) {
    TestCaseDesc testCase;
    testCase.inputTensors = {
        Tensor(GetParam().args_.dType_, GetParam().args_.shape_, "input0")
    };
    testCase.outputTensors = {
        Tensor(GetParam().args_.dType_, GetParam().args_.shape_, "output")
    };
    testCase.inputPaths = {GetGoldenDir() + "/x.bin"};
    testCase.goldenPaths = {GetGoldenDir() + "/res.bin"};
    testCase.args = &GetParam().args_;
    testCase.opFunc = GetParam().opFunc_;
    TestExecutor::runTest(testCase);
}
} // namespace