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
 * \file test_exp_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct ExpOpFuncArgs : public OpFuncArgs {
    ExpOpFuncArgs(std::vector<int> shape, std::vector<int> vecViewShapes, std::vector<int> vecTileShapes,
        DataType dType) : shape_(shape), vecViewShapes_(vecViewShapes), vecTileShapes_(vecTileShapes), dType_(dType) {}
    std::vector<int> shape_;
    std::vector<int> vecViewShapes_;
    std::vector<int> vecTileShapes_;
    DataType dType_;
};

struct ExpOperationMetadata {
    ExpOperationMetadata(std::vector<int> shape, std::vector<int> vecViewShapes,
        std::vector<int> vecTileShapes, DataType dType, OpFunc opFunc) :
        args_(shape, vecViewShapes, vecTileShapes, dType), opFunc_(opFunc) {}
    ExpOpFuncArgs args_;
    OpFunc opFunc_;
};

static void ExpOperationExeFuncView1Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                const OpFuncArgs* opArgs)
{
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        const struct ExpOpFuncArgs *args = static_cast<const ExpOpFuncArgs*>(opArgs);
        const int firstViewShape = args->vecViewShapes_[0];
        const int bloop = CeilDiv(firstDim, firstViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            /* view切分 {firstViewShape，secondDim} = viewshape  */
            auto tileTensor = DViewPad(inputs[0], {firstViewShape, secondDim},
                {std::min(firstDim - bIdx * firstViewShape, firstViewShape), secondDim},
                {bIdx * firstViewShape, 0});
            Program::GetInstance().GetTileShape().SetVecTileShapes(args->vecTileShapes_);
            auto res = Exp(tileTensor);
            DAssemble(res, {bIdx * firstViewShape, 0}, outputs[0]);
        }
    }
}

static void ExpOperationExeFunc2Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                    const OpFuncArgs* opArgs)
{
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        const struct ExpOpFuncArgs *args = static_cast<const ExpOpFuncArgs*>(opArgs);
        const int firstViewShape = args->vecViewShapes_[0];
        const int secondViewShape = args->vecViewShapes_[1];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                auto tileTensor = DViewPad(inputs[0], {firstViewShape, secondViewShape},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                    {bIdx * firstViewShape, sIdx * secondViewShape});
                Program::GetInstance().GetTileShape().SetVecTileShapes(args->vecTileShapes_);
                auto res = Exp(tileTensor);
                DAssemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

static void ExpOperationExeFunc3Dims(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                    const OpFuncArgs* opArgs)
{
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        SymbolicScalar thirdDim = inputs[0]->shape[2];
        const struct ExpOpFuncArgs *args = static_cast<const ExpOpFuncArgs*>(opArgs);
        const int firstViewShape = args->vecViewShapes_[0];
        const int secondViewShape = args->vecViewShapes_[1];
        const int thirdViewShape = args->vecViewShapes_[2];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        const int nloop = CeilDiv(thirdDim, secondViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L3_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1)) {
                    auto tileTensor = DViewPad(inputs[0], {firstViewShape, secondViewShape, thirdViewShape},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                        std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                        {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});
                    Program::GetInstance().GetTileShape().SetVecTileShapes(args->vecTileShapes_);
                    auto res = Exp(tileTensor);
                    DAssemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

static const ExpOperationMetadata testDataLists[] = {
    ExpOperationMetadata{{64 * 48, 1}, {128, 1}, {32, 32}, DataType::DT_FP32, ExpOperationExeFuncView1Dims}, // 0: 1dims
    ExpOperationMetadata{{64 * 48, 128 * 3}, {128, 128}, {32, 32}, DataType::DT_FP32, ExpOperationExeFunc2Dims}, // 1: 2dims
    ExpOperationMetadata{{64 * 48, 16 * 3}, {128, 128}, {32, 32}, DataType::DT_FP32, ExpOperationExeFunc2Dims}, // 2: tile shape nonaligned
    ExpOperationMetadata{{64 * 48, 128 * 3}, {128, 127}, {32, 32}, DataType::DT_FP32, ExpOperationExeFunc2Dims, }, // 3: view shape nonaligned
    ExpOperationMetadata{{48 * 128, 64, 64}, {64, 64, 64}, {32, 32, 32}, DataType::DT_FP32, ExpOperationExeFunc3Dims}, // 5: 3dims
    ExpOperationMetadata{{64 * 48, 128 * 3}, {128, 128}, {32, 32}, DataType::DT_FP16, ExpOperationExeFunc2Dims}, // 7: dtype:fp16
    ExpOperationMetadata{{64 * 48, 128 * 3}, {128, 128}, {32, 32}, DataType::DT_FP32, ExpOperationExeFunc2Dims}, // 8: inf
    ExpOperationMetadata{{64 * 48, 128 * 3}, {128, 128}, {32, 32}, DataType::DT_FP32, ExpOperationExeFunc2Dims}, // 9: nan
    ExpOperationMetadata{{64 * 48, 0}, {128, 128}, {32, 32}, DataType::DT_FP32, ExpOperationExeFunc2Dims}, // 10: empty tensor
    ExpOperationMetadata{{64 * 48, 128 * 3}, {128, 128}, {32, 32}, DataType::DT_FP32, ExpOperationExeFunc2Dims}, // 11: max
    ExpOperationMetadata{{64 * 48, 128 * 3}, {128, 128}, {32, 32}, DataType::DT_FP32, ExpOperationExeFunc2Dims}, // 12: min
    ExpOperationMetadata{{1, 1}, {32, 32}, {16, 16}, DataType::DT_FP32, ExpOperationExeFunc2Dims}, // 14: oshape < vshape
    ExpOperationMetadata{{32, 32}, {32, 32}, {16, 16}, DataType::DT_FP32, ExpOperationExeFunc2Dims}, // 15: oshape = vshape
    ExpOperationMetadata{{128, 128}, {32, 32}, {64, 64}, DataType::DT_FP32, ExpOperationExeFunc2Dims}, // 16: vshape < tshape
    ExpOperationMetadata{{1, 64*192}, {128, 128}, {32, 32}, DataType::DT_FP32, ExpOperationExeFunc2Dims}, // 19: k_dim 192 b min
    ExpOperationMetadata{{96, 64*192}, {128, 128}, {32, 32}, DataType::DT_FP32, ExpOperationExeFunc2Dims}, // 20: k_dim 192 b max
    ExpOperationMetadata{{1, 64*128}, {128, 128}, {32, 32}, DataType::DT_FP32, ExpOperationExeFunc2Dims}, // 21: v_dim 128 b min
    ExpOperationMetadata{{96, 64*128}, {128, 128}, {32, 32}, DataType::DT_FP32, ExpOperationExeFunc2Dims}, // 22: v_dim 128 b max
};

class ExpOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<ExpOperationMetadata> {};

INSTANTIATE_TEST_SUITE_P(
    TestExp,
    ExpOperationTest,
    ::testing::ValuesIn(testDataLists)
);

TEST_P(ExpOperationTest, TestExp) {
    TestCaseDesc testCase;
    testCase.inputTensors = {
        Tensor(GetParam().args_.dType_, GetParam().args_.shape_, "input"),
    };
    testCase.outputTensors = {
        Tensor(GetParam().args_.dType_, GetParam().args_.shape_, "output")
    };
    testCase.inputPaths = {GetGoldenDir() + "/x.bin"};
    testCase.goldenPaths = {GetGoldenDir() + "/res.bin"};
    testCase.args = &(GetParam().args_);
    testCase.opFunc = GetParam().opFunc_;
    TestExecutor::runTest(testCase);
}
} //namespace