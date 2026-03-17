/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_quantize_operation.cpp
 * \brief Test cases for Quantize operation
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;

namespace {

struct QuantizeOpFuncArgs : public OpFuncArgs {
    QuantizeOpFuncArgs(std::vector<int64_t> shape, std::vector<int64_t> vecTileShapes,
                       DataType otype, int axis, bool useZeroPoints = false)
        : viewShape_(shape), tileShape_(vecTileShapes), otype_(otype), axis_(axis),
          useZeroPoints_(useZeroPoints) {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    DataType otype_;
    int axis_;
    bool useZeroPoints_;
};

struct QuantizeOpMetaData {
    explicit QuantizeOpMetaData(const OpFunc &opFunc, const nlohmann::json &test_data)
        : opFunc_(opFunc), test_data_(test_data) {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

// Symmetric quantization (FP32 -> INT8) test
static void QuantizeSymmetricOperationExeFunc(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]}) {
        auto args = static_cast<const QuantizeOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];

        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);

        DataType otype = args->otype_;
        int axis = args->axis_;

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                auto tileTensorInput = View(inputs[0], {firstViewShape, secondViewShape},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                    {bIdx * firstViewShape, sIdx * secondViewShape});

                // Scale shape: [..., row, 1] for axis=-1
                auto tileTensorScale = View(inputs[1], {firstViewShape, 1},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape), 1},
                    {bIdx * firstViewShape, 0});

                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Quantize(tileTensorInput, tileTensorScale, otype, axis);
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

// Asymmetric quantization (FP32 -> UINT8) test with zero_points
static void QuantizeAsymmetricOperationExeFunc(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0], inputs[1], inputs[2]}, {outputs[0]}) {
        auto args = static_cast<const QuantizeOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];

        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);

        DataType otype = args->otype_;
        int axis = args->axis_;

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                auto tileTensorInput = View(inputs[0], {firstViewShape, secondViewShape},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                    {bIdx * firstViewShape, sIdx * secondViewShape});

                // Scale shape: [..., row, 1] for axis=-1
                auto tileTensorScale = View(inputs[1], {firstViewShape, 1},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape), 1},
                    {bIdx * firstViewShape, 0});

                // Zero_points shape: [..., row, 1] for axis=-1
                auto tileTensorZeroPoints = View(inputs[2], {firstViewShape, 1},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape), 1},
                    {bIdx * firstViewShape, 0});

                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Quantize(tileTensorInput, tileTensorScale, otype, axis, tileTensorZeroPoints);
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

// 3D tensor quantization test
static void Quantize3DOperationExeFunc(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]}) {
        auto args = static_cast<const QuantizeOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];

        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        const int nloop = CeilDiv(thirdDim, thirdViewShape);

        DataType otype = args->otype_;
        int axis = args->axis_;

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1)) {
                    auto tileTensorInput = View(inputs[0], {firstViewShape, secondViewShape, thirdViewShape},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                            std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                            std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                        {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});

                    // Scale shape: [..., 1, 1, 1] for axis=-1 on 3D tensor
                    auto tileTensorScale = View(inputs[1], {firstViewShape, 1, 1},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape), 1, 1},
                        {bIdx * firstViewShape, 0, 0});

                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Quantize(tileTensorInput, tileTensorScale, otype, axis);
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

class QuantizeOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<QuantizeOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestQuantize, QuantizeOperationTest,
    ::testing::ValuesIn(GetOpMetaData<QuantizeOpMetaData>(
        {QuantizeSymmetricOperationExeFunc, QuantizeAsymmetricOperationExeFunc,
         Quantize3DOperationExeFunc}, "Quantize")));

TEST_P(QuantizeOperationTest, TestQuantize) {
    auto test_data = GetParam().test_data_;
    auto otype = static_cast<DataType>(GetValueByName<int>(test_data, "otype"));
    auto axis = GetValueByName<int>(test_data, "axis");
    auto useZeroPoints = GetValueByName<bool>(test_data, "use_zero_points", false);

    auto args = QuantizeOpFuncArgs(GetViewShape(test_data), GetTileShape(test_data), otype, axis, useZeroPoints);
    auto testCase = CreateTestCaseDesc<QuantizeOpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}

} // namespace
