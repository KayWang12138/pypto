/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redisRemainderSbute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_RemainderS_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
const unsigned IDX_DIM0 = 0;
const unsigned IDX_DIM1 = 1;
const unsigned IDX_DIM2 = 2;
const unsigned IDX_DIM3 = 3;
const unsigned IDX_DIM4 = 4;
struct RemainderSOpFuncArgs : public OpFuncArgs {
    RemainderSOpFuncArgs(const std::vector<int64_t> &viewShape, const std::vector<int64_t> &tileShape, Element &scalar,
        bool reverseOperand)
        : viewShape_(viewShape), tileShape_(tileShape), scalar_(scalar), reverseOperand_(reverseOperand) {}
    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    bool reverseOperand_;
    Element scalar_;
};

struct RemainderSOpMetaData {
    explicit RemainderSOpMetaData(const OpFunc &opFunc, const nlohmann::json &test_data)
        : opFunc_(opFunc), test_data_(test_data) {}
    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void RemainderSOperationExeFunc2Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    size_t shapeSize = inputs[0].GetShape().size();
    SymbolicScalar src_firstDim = inputs[0].GetShape()[0];
    SymbolicScalar src_secondDim = inputs[0].GetShape()[1];
    auto args = static_cast<const RemainderSOpFuncArgs *>(opArgs);
    const std::vector<int64_t> viewShape = args->viewShape_;
    bool reverseOperand = args->reverseOperand_;
    const int bloop = CeilDiv(src_firstDim, viewShape[0]);
    const int sloop = CeilDiv(src_secondDim, viewShape[1]);

    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        LOOP("LOOP_L1_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                std::vector<SymbolicScalar> dynOffsets = {bIdx * viewShape[0], sIdx * viewShape[1]};
                auto tileTensor = View(inputs[0], viewShape,
                    {std::min(src_firstDim - bIdx * viewShape[0], viewShape[0]),
                        std::min(src_secondDim - sIdx * viewShape[1], viewShape[1])},
                    dynOffsets);
                TileShape::Current().SetVecTile(args->tileShape_);
                Tensor res =
                    reverseOperand ? RemainderS(args->scalar_, tileTensor) : RemainderS(tileTensor, args->scalar_);
                Assemble(res, dynOffsets, outputs[0]);
            }
        }
    }
}

static void RemainderSOperationExeFunc3Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    size_t shapeSize = inputs[0].GetShape().size();
    SymbolicScalar src_firstDim = inputs[0].GetShape()[0];
    SymbolicScalar src_secondDim = inputs[0].GetShape()[1];
    SymbolicScalar src_thirdDim = inputs[0].GetShape()[2];
    auto args = static_cast<const RemainderSOpFuncArgs *>(opArgs);
    std::vector<int64_t> viewShape = args->viewShape_;
    bool reverseOperand = args->reverseOperand_;
    const int loop[] = {
        CeilDiv(src_firstDim, viewShape[0]), CeilDiv(src_secondDim, viewShape[1]), CeilDiv(src_thirdDim, viewShape[2])};

    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop[IDX_DIM0])) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(loop[IDX_DIM1])) {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(loop[IDX_DIM2])) {
                    std::vector<SymbolicScalar> dynOffsets = {
                        bIdx * viewShape[0], sIdx * viewShape[1], nIdx * viewShape[2]};
                    auto tileTensor = View(inputs[0], viewShape,
                        {std::min(src_firstDim - bIdx * viewShape[0], viewShape[0]),
                            std::min(src_secondDim - sIdx * viewShape[1], viewShape[1]),
                            std::min(src_thirdDim - nIdx * viewShape[2], viewShape[2])},
                        dynOffsets);
                    TileShape::Current().SetVecTile(args->tileShape_);
                    Tensor res =
                        reverseOperand ? RemainderS(args->scalar_, tileTensor) : RemainderS(tileTensor, args->scalar_);
                    Assemble(res, dynOffsets, outputs[0]);
                }
            }
        }
    }
}

static void RemainderSOperationExeFunc4Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    size_t shapeSize = inputs[0].GetShape().size();
    SymbolicScalar src_firstDim = inputs[0].GetShape()[0];
    SymbolicScalar src_secondDim = inputs[0].GetShape()[1];
    SymbolicScalar src_thirdDim = inputs[0].GetShape()[2];
    SymbolicScalar src_forthDim = inputs[0].GetShape()[3];

    auto args = static_cast<const RemainderSOpFuncArgs *>(opArgs);
    std::vector<int64_t> viewShape = args->viewShape_;
    bool reverseOperand = args->reverseOperand_;
    const int loop[] = {CeilDiv(src_firstDim, viewShape[0]), CeilDiv(src_secondDim, viewShape[1]),
        CeilDiv(src_thirdDim, viewShape[2]), CeilDiv(src_forthDim, viewShape[3])};
    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop[IDX_DIM0])) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(loop[IDX_DIM1])) {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(loop[IDX_DIM2])) {
                    LOOP("LOOP_L3_qIdx", FunctionType::DYNAMIC_LOOP, qIdx, LoopRange(loop[IDX_DIM3])) {
                        std::vector<SymbolicScalar> dynOffsets = {
                            bIdx * viewShape[0], sIdx * viewShape[1], nIdx * viewShape[2], qIdx * viewShape[3]};
                        auto tileTensor = View(inputs[0], viewShape,
                            {std::min(src_firstDim - bIdx * viewShape[0], viewShape[0]),
                                std::min(src_secondDim - sIdx * viewShape[1], viewShape[1]),
                                std::min(src_thirdDim - nIdx * viewShape[2], viewShape[2]),
                                std::min(src_forthDim - qIdx * viewShape[3], viewShape[3])},
                            dynOffsets);
                        TileShape::Current().SetVecTile(args->tileShape_);
                        Tensor res = reverseOperand ? RemainderS(args->scalar_, tileTensor) :
                                                      RemainderS(tileTensor, args->scalar_);
                        Assemble(res, dynOffsets, outputs[0]);
                    }
                }
            }
        }
    }
}

static void RemainderSOperationExeFunc5Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    size_t shapeSize = inputs[0].GetShape().size();
    SymbolicScalar src_firstDim = inputs[0].GetShape()[0];
    SymbolicScalar src_secondDim = inputs[0].GetShape()[1];
    SymbolicScalar src_thirdDim = inputs[0].GetShape()[2];
    SymbolicScalar src_forthDim = inputs[0].GetShape()[3];
    SymbolicScalar src_fifthDim = inputs[0].GetShape()[4];

    auto args = static_cast<const RemainderSOpFuncArgs *>(opArgs);
    std::vector<int64_t> viewShape = args->viewShape_;
    bool reverseOperand = args->reverseOperand_;
    const int loop[] = {CeilDiv(src_firstDim, viewShape[0]), CeilDiv(src_secondDim, viewShape[1]),
        CeilDiv(src_thirdDim, viewShape[2]), CeilDiv(src_forthDim, viewShape[3]), CeilDiv(src_fifthDim, viewShape[4])};
    FUNCTION("main", {inputs[0]}, {outputs[0]}) {
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop[IDX_DIM0])) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(loop[IDX_DIM1])) {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(loop[IDX_DIM2])) {
                    LOOP("LOOP_L3_qIdx", FunctionType::DYNAMIC_LOOP, qIdx, LoopRange(loop[IDX_DIM3])) {
                        LOOP("LOOP_L4_rIdx", FunctionType::DYNAMIC_LOOP, rIdx, LoopRange(loop[IDX_DIM4])) {
                            std::vector<SymbolicScalar> dynOffsets = {bIdx * viewShape[0], sIdx * viewShape[1],
                                nIdx * viewShape[2], qIdx * viewShape[3], rIdx * viewShape[4]};
                            auto tileTensor = View(inputs[0], viewShape,
                                {std::min(src_firstDim - bIdx * viewShape[0], viewShape[0]),
                                    std::min(src_secondDim - sIdx * viewShape[1], viewShape[1]),
                                    std::min(src_thirdDim - nIdx * viewShape[2], viewShape[2]),
                                    std::min(src_forthDim - qIdx * viewShape[3], viewShape[3]),
                                    std::min(src_fifthDim - rIdx * viewShape[4], viewShape[4])},
                                dynOffsets);
                            TileShape::Current().SetVecTile(args->tileShape_);
                            Tensor res = reverseOperand ? RemainderS(args->scalar_, tileTensor) :
                                                          RemainderS(tileTensor, args->scalar_);
                            Assemble(res, dynOffsets, outputs[0]);
                        }
                    }
                }
            }
        }
    }
}

class Remainder_RSOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<RemainderSOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestRemainder_RS, Remainder_RSOperationTest,
    ::testing::ValuesIn(
        GetOpMetaData<RemainderSOpMetaData>({RemainderSOperationExeFunc2Dims, RemainderSOperationExeFunc3Dims,
                                                RemainderSOperationExeFunc4Dims, RemainderSOperationExeFunc5Dims},
            "Remainder_RS")));

class RemainderSOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<RemainderSOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestRemainderS, RemainderSOperationTest,
    ::testing::ValuesIn(
        GetOpMetaData<RemainderSOpMetaData>({RemainderSOperationExeFunc2Dims, RemainderSOperationExeFunc3Dims,
                                                RemainderSOperationExeFunc4Dims, RemainderSOperationExeFunc5Dims},
            "RemainderS")));

TEST_P(Remainder_RSOperationTest, TestRemainder_RS) {
    auto test_data = GetParam().test_data_;
    bool reverseOperand = true;
    auto dtype = testCase.outputTensors.at(0).GetDataType();
    Element scalar(dtype, GetValueByName<float>(test_data, "scalar"));
    auto args = RemainderSOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data), scalar, reverseOperand);
    auto testCase = CreateTestCaseDesc<RemainderSOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}

TEST_P(RemainderSOperationTest, TestRemainderS) {
    auto test_data = GetParam().test_data_;
    bool reverseOperand = false;
    auto dtype = testCase.outputTensors.at(0).GetDataType();
    Element scalar(dtype, GetValueByName<float>(test_data, "scalar"));
    auto args = RemainderSOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data), scalar, reverseOperand);
    auto testCase = CreateTestCaseDesc<RemainderSOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace