/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_Remainder_operation.cpp
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
struct RemainderOpFuncArgs : public OpFuncArgs {
    RemainderOpFuncArgs(const std::vector<int64_t> &viewShape, const std::vector<int64_t> tileShape)
        : viewShape_(viewShape), tileShape_(tileShape) {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct RemainderOpMetaData {
    explicit RemainderOpMetaData(const OpFunc &opFunc, const nlohmann::json &test_data)
        : opFunc_(opFunc), test_data_(test_data) {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void RemainderOperationExeFunc2Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        auto args = static_cast<const RemainderOpFuncArgs *>(opArgs);
        const std::vector<int> viewShape = args->viewShape_;

        const int loop[] = {CeilDiv(firstDim, viewShape[0]), CeilDiv(secondDim, viewShape[1])};
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop[IDX_DIM0])) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(loop[IDX_DIM1])) {
                std::vector<SymbolicScalar> dynOffsets = {
                    bIdx * viewShape[0], sIdx * viewShape[1], nIdx * viewShape[2]};
                std::vector<SymbolicScalar> validShape = {std::min(firstDim - bIdx * viewShape[0], viewShape[0]),
                    std::min(secondDim - sIdx * viewShape[1], viewShape[1]),
                    std::min(thirdDim - nIdx * viewShape[2], viewShape[2])};
                Tensor tileTensor0 = View(inputs[1], viewShape, validShape, dynOffsets);
                Tensor tileTensor1 = View(inputs[1], viewShape, validShape, dynOffsets);
                IF(inputs[0].GetShape().size() == 1) {
                    tileTensor0 = View(inputs[0], {viewShape[1]}, {validShape[1]}, {dynOffsets[1]});
                }
                ELSE IF(inputs[1].GetShape().size() == 1) {
                    tileTensor1 = View(inputs[1], {viewShape[1]}, {validShape[1]}, {dynOffsets[1]});
                }
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Remainder(tileTensor0, tileTensor1);
                Assemble(res, dynOffsets, outputs[0]);
            }
        }
    }
}

static void RemainderOperationExeFunc3Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        auto args = static_cast<const RemainderOpFuncArgs *>(opArgs);
        const std::vector<int> viewShape = args->viewShape_;

        const int loop[] = {
            CeilDiv(firstDim, viewShape[0]), CeilDiv(secondDim, viewShape[1]), CeilDiv(thirdDim, viewShape[2])};
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop[IDX_DIM0])) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(loop[IDX_DIM1])) {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(loop[IDX_DIM2])) {
                    std::vector<SymbolicScalar> dynOffsets = {
                        bIdx * viewShape[0], sIdx * viewShape[1], nIdx * viewShape[2]};
                    std::vector<SymbolicScalar> validShape = {std::min(firstDim - bIdx * viewShape[0], viewShape[0]),
                        std::min(secondDim - sIdx * viewShape[1], viewShape[1]),
                        std::min(thirdDim - nIdx * viewShape[2], viewShape[2])};
                    Tensor tileTensor0 = View(inputs[1], viewShape, validShape, dynOffsets);
                    Tensor tileTensor1 = View(inputs[1], viewShape, validShape, dynOffsets);
                    IF(inputs[0].GetShape().size() == 2) {
                        tileTensor0 = View(inputs[0], {viewShape[1], viewShape[2]}, {validShape[1], validShape[2]},
                            {dynOffsets[1], dynOffsets[2]});
                    }
                    ELSE IF(inputs[1].GetShape().size() == 2) {
                        tileTensor1 = View(inputs[1], {viewShape[1], viewShape[2]}, {validShape[1], validShape[2]},
                            {dynOffsets[1], dynOffsets[2]});
                    }
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Remainder(tileTensor0, tileTensor1);
                    Assemble(res, dynOffsets, outputs[0]);
                }
            }
        }
    }
}

static void RemainderOperationExeFunc4Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar fourthDim = inputs[0].GetShape()[3];
        auto args = static_cast<const RemainderOpFuncArgs *>(opArgs);
        const std::vector<int> viewShape = args->viewShape_;

        const int loop[] = {CeilDiv(firstDim, viewShape[0]), CeilDiv(secondDim, viewShape[1]),
            CeilDiv(thirdDim, viewShape[2]), CeilDiv(fourthDim, viewShape[3])};
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(loop[IDX_DIM0])) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(loop[IDX_DIM1])) {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(loop[IDX_DIM2])) {
                    LOOP("LOOP_L3_qIdx", FunctionType::DYNAMIC_LOOP, qIdx, LoopRange(loop[IDX_DIM3])) {
                        std::vector<SymbolicScalar> dynOffsets = {
                            bIdx * viewShape[0], sIdx * viewShape[1], nIdx * viewShape[2], qIdx * viewShape[3]};
                        std::vector<SymbolicScalar> validShape = {
                            std::min(firstDim - bIdx * viewShape[0], viewShape[0]),
                            std::min(secondDim - sIdx * viewShape[1], viewShape[1]),
                            std::min(thirdDim - nIdx * viewShape[2], viewShape[2]),
                            std::min(fourthDim - qIdx * viewShape[3], viewShape[3])};
                        Tensor tileTensor0 = View(inputs[1], viewShape, validShape, dynOffsets);
                        Tensor tileTensor1 = View(inputs[1], viewShape, validShape, dynOffsets);
                        IF(inputs[0].GetShape().size() == 3) {
                            tileTensor0 = View(inputs[0], {viewShape[1], viewShape[2], viewshape[3]},
                                {validShape[1], validShape[2], validShape[3]},
                                {dynOffsets[1], dynOffsets[2], dynOffsets[3]});
                        }
                        ELSE IF(inputs[1].GetShape().size() == 3) {
                            tileTensor1 = View(inputs[1], {viewShape[1], viewShape[2], viewshape[3]},
                                {validShape[1], validShape[2], validShape[3]},
                                {dynOffsets[1], dynOffsets[2], dynOffsets[3]});
                        }
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Remainder(tileTensor0, tileTensor1);
                        Assemble(res, dynOffsets, outputs[0]);
                    }
                }
            }
        }
    }
}

class RemainderOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<RemainderOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestRemainder, RemainderOperationTest,
    ::testing::ValuesIn(GetOpMetaData<RemainderOpMetaData>(
        {RemainderOperationExeFunc2Dims, RemainderOperationExeFunc3Dims, RemainderOperationExeFunc4Dims},
        "Remainder")));

TEST_P(RemainderOperationTest, TestRemainder) {
    auto test_data = GetParam().test_data_;
    auto args = RemainderOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = CreateTestCaseDesc<RemainderOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
