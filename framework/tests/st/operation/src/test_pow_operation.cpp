/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_pow_operation.cpp
 * \brief
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
struct PowOpFuncArgs : public OpFuncArgs {
    PowOpFuncArgs(const std::vector<int64_t> &viewShape, const std::vector<int64_t> tileShape)
        : viewShape_(viewShape), tileShape_(tileShape) {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
};

struct PowOpMetaData {
    explicit PowOpMetaData(const OpFunc &opFunc, const nlohmann::json &test_data)
        : opFunc_(opFunc), test_data_(test_data) {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

static void PowOperationExeFunc2Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {

    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        auto args = static_cast<const PowOpFuncArgs *>(opArgs);
        const int64_t firstViewShape = args->viewShape_[0];
        const int64_t secondViewShape = args->viewShape_[1];
        const std::vector<int64_t> shapes = {firstViewShape, secondViewShape};

        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                std::vector<SymbolicScalar> offset = {
                    bIdx * firstViewShape,
                    sIdx * secondViewShape
                };
                std::vector<SymbolicScalar> validShape = {
                    std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                    std::min(secondDim - sIdx * secondViewShape, secondViewShape)
                };
                Tensor tileTensor0 = View(inputs[0], shapes, validShape, offset);
                Tensor tileTensor1 = View(inputs[1], shapes, validShape, offset);
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Pow(tileTensor0, tileTensor1);
                Assemble(res, offset, outputs[0]);
            }
        }
    }
}

static void PowOperationExeFunc3Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {

    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        auto args = static_cast<const PowOpFuncArgs *>(opArgs);
        const int64_t firstViewShape = args->viewShape_[0];
        const int64_t secondViewShape = args->viewShape_[1];
        const int64_t thirdViewShape = args->viewShape_[2];
        const std::vector<int64_t> shapes = {firstViewShape, secondViewShape, thirdViewShape};

        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        const int mloop = CeilDiv(thirdDim, thirdViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L2_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mloop, 1)) {
                    std::vector<SymbolicScalar> offset = {
                        bIdx * firstViewShape,
                        sIdx * secondViewShape,
                        mIdx * thirdViewShape
                    };
                    std::vector<SymbolicScalar> validShape = {
                        std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                        std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape)
                    };
                    Tensor tileTensor0 = View(inputs[0], shapes, validShape, offset);
                    Tensor tileTensor1 = View(inputs[1], shapes, validShape, offset);
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Pow(tileTensor0, tileTensor1);
                    Assemble(res, offset, outputs[0]);
                }
            }
        }
    }
}

static void PowOperationExeFunc4Dims(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {

    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]}) {
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar fourthDim = inputs[0].GetShape()[3];
        auto args = static_cast<const PowOpFuncArgs *>(opArgs);
        const int64_t firstViewShape = args->viewShape_[0];
        const int64_t secondViewShape = args->viewShape_[1];
        const int64_t thirdViewShape = args->viewShape_[2];
        const int64_t fourthViewShape = args->viewShape_[3];
        const std::vector<int64_t> shapes = {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape};

        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        const int mloop = CeilDiv(thirdDim, thirdViewShape);
        const int nloop = CeilDiv(fourthDim, fourthViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L2_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, mloop, 1)) {
                    LOOP("LOOP_L3_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1)) {
                        std::vector<SymbolicScalar> offset = {
                            bIdx * firstViewShape,
                            sIdx * secondViewShape,
                            mIdx * thirdViewShape,
                            nIdx * fourthViewShape
                        };
                        std::vector<SymbolicScalar> validShape = {
                            std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                            std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                            std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape),
                            std::min(fourthDim - nIdx * fourthViewShape, fourthViewShape)
                        };
                        Tensor tileTensor0 = View(inputs[0], shapes, validShape, offset);
                        Tensor tileTensor1 = View(inputs[1], shapes, validShape, offset);
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Pow(tileTensor0, tileTensor1);
                        Assemble(res, offset, outputs[0]);
                    }
                }
            }
        }
    }
}

class PowOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<PowOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestPow, PowOperationTest,
    ::testing::ValuesIn(GetOpMetaData<PowOpMetaData>(
        {PowOperationExeFunc2Dims, PowOperationExeFunc3Dims, PowOperationExeFunc4Dims}, "Pow")));

TEST_P(PowOperationTest, TestPow) {
    auto test_data = GetParam().test_data_;
    auto args = PowOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data));
    auto testCase = CreateTestCaseDesc<PowOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
