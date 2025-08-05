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
struct VectorDupOpFuncArgs : public OpFuncArgs {
    VectorDupOpFuncArgs(std::vector<int> shape, std::vector<int> vecViewShapes, std::vector<int> vecTileShapes,
        DataType dType) : shape_(shape), vecViewShapes_(vecViewShapes), vecTileShapes_(vecTileShapes), dType_(dType) {}
    std::vector<int> shape_;
    std::vector<int> vecViewShapes_;
    std::vector<int> vecTileShapes_;
    DataType dType_;
};

struct VectorDupOperationMetadata {
    VectorDupOperationMetadata(int caseIndex, VectorDupOpFuncArgs args, OpFunc opFunc) :
        caseIndex_(caseIndex), args_(args), opFunc_(opFunc) {}
    int caseIndex_;
    VectorDupOpFuncArgs args_;
    OpFunc opFunc_;
};

static void VectorDupOperationExeFunc(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                 const OpFuncArgs* opArgs) {                               
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        const struct VectorDupOpFuncArgs *args = static_cast<const VectorDupOpFuncArgs*>(opArgs);
        const int firstl0LoopLengthTile = args->vecViewShapes_[0];
        const int bloop = CeilDiv(firstDim, firstl0LoopLengthTile);
        Program::GetInstance().GetTileShape().SetVecTileShapes(args->vecTileShapes_);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            auto tileTensor0 = VectorDuplicate(Element(DataType::DT_FP32, 1e-5f), DT_FP32, {firstl0LoopLengthTile, secondDim},
                {std::min(firstDim - bIdx * firstl0LoopLengthTile, firstl0LoopLengthTile), secondDim});
            DAssemble(tileTensor0, {bIdx * firstl0LoopLengthTile, 0}, outputs[0]);
        }
    }
}

static void VectorDupOperationExeFuncDoubleCut(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                          const OpFuncArgs* opArgs) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);                                
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        std::vector<float> inputData(1, 0);
        readInput<float>(GetGoldenDir() + "/x.bin", inputData);
        Element value(DataType::DT_FP32, inputData[0]);
        auto args = static_cast<const VectorDupOpFuncArgs*>(opArgs);
        const int firstViewShape = args->vecViewShapes_[0];
        const int secondViewShape = args->vecViewShapes_[1];
        int bloop = CeilDiv(firstDim, firstViewShape);
        int sloop = CeilDiv(secondDim, secondViewShape);
        Program::GetInstance().GetTileShape().SetVecTileShapes(args->vecTileShapes_);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                auto tileTensor0 = VectorDuplicate(value, DT_FP32, {firstViewShape, secondViewShape},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)});
                DAssemble(tileTensor0, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

static void VectorDupOperationExeFuncTripleCut(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                          const OpFuncArgs* opArgs) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);                              
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        SymbolicScalar thirdDim = inputs[0]->shape[2];
        auto args = static_cast<const VectorDupOpFuncArgs*>(opArgs);
        const int firstViewShape = args->vecViewShapes_[0];
        const int secondViewShape = args->vecViewShapes_[1];
        const int thirdViewShape = args->vecViewShapes_[2];
        int bloop = CeilDiv(firstDim, firstViewShape);
        int sloop = CeilDiv(secondDim, secondViewShape);
        int nloop = CeilDiv(thirdDim, thirdViewShape);
        Program::GetInstance().GetTileShape().SetVecTileShapes(args->vecTileShapes_);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1)) {
                    auto tileTensor0 = VectorDuplicate(Element(DataType::DT_FP32, 2.0f), DT_FP32, {firstViewShape, secondViewShape, thirdViewShape},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                            std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)});
                    DAssemble(tileTensor0, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

static void VectorDupOperationExeFuncQuadrupleCut(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                             const OpFuncArgs* opArgs) {
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);                                  
    FUNCTION("main", FunctionType::DYNAMIC, {inputs[0]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0]->shape[0];
        SymbolicScalar secondDim = inputs[0]->shape[1];
        SymbolicScalar thirdDim = inputs[0]->shape[2];
        SymbolicScalar fourthDim = inputs[0]->shape[3];
        auto args = static_cast<const VectorDupOpFuncArgs*>(opArgs);
        const int firstViewShape = args->vecViewShapes_[0];
        const int secondViewShape = args->vecViewShapes_[1];
        const int thirdViewShape = args->vecViewShapes_[2];
        const int fourthViewShape = args->vecViewShapes_[3];
        int bloop = CeilDiv(firstDim, firstViewShape);
        int sloop = CeilDiv(secondDim, secondViewShape);
        int nloop = CeilDiv(thirdDim, thirdViewShape);
        int qloop = CeilDiv(fourthDim, fourthViewShape);
        Program::GetInstance().GetTileShape().SetVecTileShapes(args->vecTileShapes_);
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1)) {
                    LOOP("LOOP_L3_qIdx", FunctionType::DYNAMIC_LOOP, qIdx, LoopRange(0, qloop, 1)) {
                        auto tileTensor0 = VectorDuplicate(Element(DataType::DT_FP32, 2.0f), DT_FP32, {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                            {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                    std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape),
                                        std::min(fourthDim - qIdx * fourthViewShape, fourthViewShape)});
                        DAssemble(tileTensor0, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape, qIdx * fourthViewShape}, outputs[0]);
                    }
                }
            }
        }
    }
}

static const VectorDupOperationMetadata testDataLists[] = {
    VectorDupOperationMetadata{0, VectorDupOpFuncArgs{{64*48, 1}, {128, 1}, {32, 32}, DataType::DT_FP32},
                               VectorDupOperationExeFunc},
    VectorDupOperationMetadata{1, VectorDupOpFuncArgs{{64*48, 128*3}, {128, 128}, {32, 32}, DataType::DT_FP32},
                               VectorDupOperationExeFuncDoubleCut},
    VectorDupOperationMetadata{2, VectorDupOpFuncArgs{{64*32, 16*3}, {128, 128}, {32, 32}, DataType::DT_FP32},
                               VectorDupOperationExeFuncDoubleCut},
    VectorDupOperationMetadata{3, VectorDupOpFuncArgs{{64*32 + 3, 16*3}, {128, 128}, {32, 32}, DataType::DT_FP32},
                               VectorDupOperationExeFuncDoubleCut},
    VectorDupOperationMetadata{4, VectorDupOpFuncArgs{{64*48, 128*3}, {128, 127}, {32, 32}, DataType::DT_FP32},
                               VectorDupOperationExeFuncDoubleCut},
    VectorDupOperationMetadata{5, VectorDupOpFuncArgs{{48*128, 64, 64}, {64, 64, 64}, {32, 32, 32}, DataType::DT_FP32},
                               VectorDupOperationExeFuncTripleCut},
    VectorDupOperationMetadata{6, VectorDupOpFuncArgs{{1, 1}, {32, 32}, {16, 16}, DataType::DT_FP32},
                               VectorDupOperationExeFuncDoubleCut},
    VectorDupOperationMetadata{7, VectorDupOpFuncArgs{{32, 32}, {32, 32}, {16, 16}, DataType::DT_FP32},
                               VectorDupOperationExeFuncDoubleCut},
    VectorDupOperationMetadata{8, VectorDupOpFuncArgs{{128, 128}, {32, 32}, {64, 64}, DataType::DT_FP32},
                               VectorDupOperationExeFuncDoubleCut},
    VectorDupOperationMetadata{9,
                               VectorDupOpFuncArgs{{48, 64, 128, 32}, {48, 64, 32, 16}, {16, 16, 16, 8},
                                                   DataType::DT_FP32},
                               VectorDupOperationExeFuncQuadrupleCut},
    VectorDupOperationMetadata{10, VectorDupOpFuncArgs{{64*48, 0}, {128, 128}, {32, 32}, DataType::DT_FP32},
                               VectorDupOperationExeFuncDoubleCut},
    VectorDupOperationMetadata{11, VectorDupOpFuncArgs{{0, 128*3}, {128, 128}, {32, 32}, DataType::DT_FP32},
                               VectorDupOperationExeFuncDoubleCut},
    VectorDupOperationMetadata{12, VectorDupOpFuncArgs{{64*48, 1536}, {256, 256}, {32, 32}, DataType::DT_FP32},
                               VectorDupOperationExeFuncDoubleCut},
    VectorDupOperationMetadata{13, VectorDupOpFuncArgs{{64*48, 512}, {256, 128}, {32, 32}, DataType::DT_FP32},
                               VectorDupOperationExeFuncDoubleCut},
    VectorDupOperationMetadata{14, VectorDupOpFuncArgs{{64*48, 7168*4}, {256, 4096}, {32, 512}, DataType::DT_FP32},
                               VectorDupOperationExeFuncDoubleCut},
};

class VectorDupOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<VectorDupOperationMetadata> {};

INSTANTIATE_TEST_SUITE_P(
    TestVectorDup,
    VectorDupOperationTest,
    ::testing::ValuesIn(testDataLists)
);

TEST_P(VectorDupOperationTest, test_vector_dup) {
    TestCaseDesc testVectorDupCase;
    testVectorDupCase.inputTensors = {
        Tensor(GetParam().args_.dType_, GetParam().args_.shape_, "input0")
    };
    testVectorDupCase.outputTensors = {
        Tensor(GetParam().args_.dType_, GetParam().args_.shape_, "output")
    };
    testVectorDupCase.inputPaths = {GetGoldenDir() + "/x.bin"};
    testVectorDupCase.goldenPaths = {GetGoldenDir() + "/res.bin"};
    testVectorDupCase.args = &(GetParam().args_);
    testVectorDupCase.opFunc = GetParam().opFunc_;
    TestExecutor::runTest(testVectorDupCase);
}
} // namespace