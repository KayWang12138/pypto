/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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

// ============================================================
// 2D Tensor Tests
// ============================================================

// Symmetric quantization (FP32 -> INT8) with axis=-1 / -2
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

                // Scale shape: [..., row] for axis=-1
                auto tileTensorScale = View(inputs[1], {firstViewShape},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape)},
                    {bIdx * firstViewShape});

                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Quantize(tileTensorInput, tileTensorScale, otype, axis, Tensor());
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}


// Asymmetric quantization (FP32 -> UINT8) test with zero_points, axis=-1 /-2
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

                // Scale shape: [..., row] for axis=-1
                auto tileTensorScale = View(inputs[1], {firstViewShape},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape)},
                    {bIdx * firstViewShape});

                // Zero_points shape: [..., row] for axis=-1
                auto tileTensorZeroPoints = View(inputs[2], {firstViewShape},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape)},
                    {bIdx * firstViewShape});

                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Quantize(tileTensorInput, tileTensorScale, otype, axis, tileTensorZeroPoints);
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

// ============================================================
// 3D Tensor Tests
// ============================================================

// 3D tensor quantization with axis=-1 / -2
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

                    // Scale shape: [..., H] for axis=-1 on 3D tensor
                    auto tileTensorScale = View(inputs[1], {firstViewShape, secondViewShape},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                            std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                        {bIdx * firstViewShape, sIdx * secondViewShape});

                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Quantize(tileTensorInput, tileTensorScale, otype, axis, Tensor());
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

// 3D tensor asymmetric quantization with axis=-1 /-2
static void Quantize3DAsymmetricOperationExeFunc(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0], inputs[1], inputs[2]}, {outputs[0]}) {
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

                    // Scale shape: [..., H] for axis=-1 on 3D tensor
                    auto tileTensorScale = View(inputs[1], {firstViewShape, secondViewShape},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                            std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                        {bIdx * firstViewShape, sIdx * secondViewShape});

                    // Zero_points shape: [..., H] for axis=-1 on 3D tensor
                    auto tileTensorZeroPoints = View(inputs[2], {firstViewShape, secondViewShape},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                            std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                        {bIdx * firstViewShape, sIdx * secondViewShape});

                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Quantize(tileTensorInput, tileTensorScale, otype, axis, tileTensorZeroPoints);
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

// ============================================================
// 4D Tensor Tests
// ============================================================

// 4D tensor quantization with axis=-1
static void Quantize4DOperationExeFunc(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]}) {
        auto args = static_cast<const QuantizeOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int fourthViewShape = args->viewShape_[3];

        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar fourthDim = inputs[0].GetShape()[3];
        const int loop0 = CeilDiv(firstDim, firstViewShape);
        const int loop1 = CeilDiv(secondDim, secondViewShape);
        const int loop2 = CeilDiv(thirdDim, thirdViewShape);
        const int loop3 = CeilDiv(fourthDim, fourthViewShape);

        DataType otype = args->otype_;
        int axis = args->axis_;

        LOOP("LOOP_L0", FunctionType::DYNAMIC_LOOP, idx0, LoopRange(0, loop0, 1)) {
            LOOP("LOOP_L1", FunctionType::DYNAMIC_LOOP, idx1, LoopRange(0, loop1, 1)) {
                LOOP("LOOP_L2", FunctionType::DYNAMIC_LOOP, idx2, LoopRange(0, loop2, 1)) {
                    LOOP("LOOP_L3", FunctionType::DYNAMIC_LOOP, idx3, LoopRange(0, loop3, 1)) {
                        auto tileTensorInput = View(inputs[0],
                            {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                            {std::min(firstDim - idx0 * firstViewShape, firstViewShape),
                                std::min(secondDim - idx1 * secondViewShape, secondViewShape),
                                std::min(thirdDim - idx2 * thirdViewShape, thirdViewShape),
                                std::min(fourthDim - idx3 * fourthViewShape, fourthViewShape)},
                            {idx0 * firstViewShape, idx1 * secondViewShape, idx2 * thirdViewShape, idx3 * fourthViewShape});

                        // Scale shape: [..., H] for axis=-1 on 4D tensor
                        auto tileTensorScale = View(inputs[1], {firstViewShape, secondViewShape, thirdViewShape},
                            {std::min(firstDim - idx0 * firstViewShape, firstViewShape),
                                std::min(secondDim - idx1 * secondViewShape, secondViewShape),
                                std::min(thirdDim - idx2 * thirdViewShape, thirdViewShape)},
                            {idx0 * firstViewShape, idx1 * secondViewShape, idx2 * thirdViewShape});

                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Quantize(tileTensorInput, tileTensorScale, otype, axis, Tensor());
                        Assemble(res, {idx0 * firstViewShape, idx1 * secondViewShape, idx2 * thirdViewShape, idx3 * fourthViewShape}, outputs[0]);
                    }
                }
            }
        }
    }
}

// 4D tensor quantization with axis=-1
static void Quantize4DAsymmetricOperationExeFunc(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]}) {
        auto args = static_cast<const QuantizeOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int fourthViewShape = args->viewShape_[3];

        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar fourthDim = inputs[0].GetShape()[3];
        const int loop0 = CeilDiv(firstDim, firstViewShape);
        const int loop1 = CeilDiv(secondDim, secondViewShape);
        const int loop2 = CeilDiv(thirdDim, thirdViewShape);
        const int loop3 = CeilDiv(fourthDim, fourthViewShape);

        DataType otype = args->otype_;
        int axis = args->axis_;

        LOOP("LOOP_L0", FunctionType::DYNAMIC_LOOP, idx0, LoopRange(0, loop0, 1)) {
            LOOP("LOOP_L1", FunctionType::DYNAMIC_LOOP, idx1, LoopRange(0, loop1, 1)) {
                LOOP("LOOP_L2", FunctionType::DYNAMIC_LOOP, idx2, LoopRange(0, loop2, 1)) {
                    LOOP("LOOP_L3", FunctionType::DYNAMIC_LOOP, idx3, LoopRange(0, loop3, 1)) {
                        auto tileTensorInput = View(inputs[0],
                            {firstViewShape, secondViewShape, thirdViewShape, fourthViewShape},
                            {std::min(firstDim - idx0 * firstViewShape, firstViewShape),
                                std::min(secondDim - idx1 * secondViewShape, secondViewShape),
                                std::min(thirdDim - idx2 * thirdViewShape, thirdViewShape),
                                std::min(fourthDim - idx3 * fourthViewShape, fourthViewShape)},
                            {idx0 * firstViewShape, idx1 * secondViewShape, idx2 * thirdViewShape, idx3 * fourthViewShape});

                        // Scale shape: [..., H] for axis=-1 on 4D tensor
                        auto tileTensorScale = View(inputs[1], {firstViewShape, secondViewShape, thirdViewShape},
                            {std::min(firstDim - idx0 * firstViewShape, firstViewShape),
                                std::min(secondDim - idx1 * secondViewShape, secondViewShape),
                                std::min(thirdDim - idx2 * thirdViewShape, thirdViewShape)},
                            {idx0 * firstViewShape, idx1 * secondViewShape, idx2 * thirdViewShape});
                        // Offset shape: [..., H] for axis=-1 on 4D tensor
                        auto tileTensorOffset = View(inputs[2], {firstViewShape, secondViewShape, thirdViewShape},
                            {std::min(firstDim - idx0 * firstViewShape, firstViewShape),
                                std::min(secondDim - idx1 * secondViewShape, secondViewShape),
                                std::min(thirdDim - idx2 * thirdViewShape, thirdViewShape)},
                            {idx0 * firstViewShape, idx1 * secondViewShape, idx2 * thirdViewShape});

                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Quantize(tileTensorInput, tileTensorScale, otype, axis, tileTensorOffset);
                        Assemble(res, {idx0 * firstViewShape, idx1 * secondViewShape, idx2 * thirdViewShape, idx3 * fourthViewShape}, outputs[0]);
                    }
                }
            }
        }
    }
}

// Helper function to select the appropriate execution function based on test parameters
OpFunc SelectQuantizeOpFunc(int ndim, int axis, bool useZeroPoints) {
    // Normalize axis to negative indexing
    int normalizedAxis = (axis >= 0) ? axis - ndim : axis;

    if (ndim == 2) {
        return useZeroPoints ? QuantizeAsymmetricOperationExeFunc : QuantizeSymmetricOperationExeFunc;
    } else if (ndim == 3) {
        return useZeroPoints ? Quantize3DAsymmetricOperationExeFunc : Quantize3DOperationExeFunc;
    } else if (ndim == 4) {
        // 4D asymmetric not implemented yet, use symmetric functions
        return useZeroPoints ? Quantize4DAsymmetricOperationExeFunc : Quantize4DOperationExeFunc;
    }

    // Default to 2D symmetric
    return QuantizeSymmetricOperationExeFunc;
}

class QuantizeOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<QuantizeOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestQuantize, QuantizeOperationTest,
    ::testing::ValuesIn(GetOpMetaData<QuantizeOpMetaData>(
        {QuantizeSymmetricOperationExeFunc, QuantizeAsymmetricOperationExeFunc,
        Quantize3DOperationExeFunc, Quantize3DAsymmetricOperationExeFunc,
        Quantize4DOperationExeFunc, Quantize4DAsymmetricOperationExeFunc}, "Quantize")));

TEST_P(QuantizeOperationTest, TestQuantize) {
    auto test_data = GetParam().test_data_;
    auto otype = static_cast<DataType>(GetValueByName<int>(test_data, "otype"));
    auto axis = GetValueByName<int>(test_data, "axis");
    auto useZeroPoints = GetValueByName<bool>(test_data, "use_zero_points");
    auto viewShape = GetViewShape(test_data);
    int ndim = static_cast<int>(viewShape.size());

    // Dynamically select the appropriate execution function
    auto selectedOpFunc = SelectQuantizeOpFunc(ndim, axis, useZeroPoints);

    auto args = QuantizeOpFuncArgs(viewShape, GetTileShape(test_data), otype, axis, useZeroPoints);

    TestCaseDesc testCase;
    testCase.inputTensors = GetInputTensors(test_data);
    testCase.outputTensors = GetOutputTensors(test_data);
    testCase.args = &args;
    testCase.opFunc = selectedOpFunc;
    std::transform(testCase.inputTensors.begin(), testCase.inputTensors.end(), std::back_inserter(testCase.inputPaths),
        [](const auto &tensor) { return GetGoldenDir() + "/" + tensor.GetStorage()->Symbol() + ".bin"; });
    std::transform(testCase.outputTensors.begin(), testCase.outputTensors.end(),
        std::back_inserter(testCase.goldenPaths),
        [](const auto &tensor) { return GetGoldenDir() + "/" + tensor.GetStorage()->Symbol() + ".bin"; });
    auto params_dict = test_data.at("params");
    testCase.onBoard = params_dict.find("on_board") == params_dict.end() || GetValueByName<bool>(test_data, "on_board");

    TestExecutor::runTest(testCase);
}

} // namespace
