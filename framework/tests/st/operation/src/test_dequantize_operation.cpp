/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_dequantize_operation.cpp
 * \brief Test cases for Dequantize operation
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;

namespace {

struct DequantizeOpFuncArgs : public OpFuncArgs {
    DequantizeOpFuncArgs(std::vector<int64_t> shape, std::vector<int64_t> vecTileShapes,
                         DataType inputDtype, int axis, bool useZeroPoints = false)
        : viewShape_(shape), tileShape_(vecTileShapes), inputDtype_(inputDtype), axis_(axis),
          useZeroPoints_(useZeroPoints) {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    DataType inputDtype_;  // INT8 or INT16
    int axis_;
    bool useZeroPoints_;
};

struct DequantizeOpMetaData {
    explicit DequantizeOpMetaData(const OpFunc &opFunc, const nlohmann::json &test_data)
        : opFunc_(opFunc), test_data_(test_data) {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

// ============================================================
// 2D Tensor Tests - INT8
// ============================================================

// Symmetric dequantization (INT8 -> FP32) with axis=-1
static void DequantizeSymmetricInt8OperationExeFunc(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]}) {
        auto args = static_cast<const DequantizeOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];

        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);

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
                auto res = Dequantize(tileTensorInput, tileTensorScale, DataType::DT_FP32, axis, Tensor());
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

// Symmetric dequantization (INT8 -> FP32) with axis=-2 (per-column scaling)
static void Dequantize2DAxis2Int8OperationExeFunc(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]}) {
        auto args = static_cast<const DequantizeOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];

        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);

        int axis = args->axis_;  // axis=-2

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                auto tileTensorInput = View(inputs[0], {firstViewShape, secondViewShape},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                    {bIdx * firstViewShape, sIdx * secondViewShape});

                // Scale shape: [1, col] for axis=-2 (per-column scaling)
                auto tileTensorScale = View(inputs[1], {1, secondViewShape},
                    {1, std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                    {0, sIdx * secondViewShape});

                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Dequantize(tileTensorInput, tileTensorScale, DataType::DT_FP32, axis, Tensor());
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

// Asymmetric dequantization (INT8 -> FP32) with zero_points, axis=-1
static void DequantizeAsymmetricInt8OperationExeFunc(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0], inputs[1], inputs[2]}, {outputs[0]}) {
        auto args = static_cast<const DequantizeOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];

        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);

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
                auto res = Dequantize(tileTensorInput, tileTensorScale, DataType::DT_FP32, axis, tileTensorZeroPoints);
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

// Asymmetric dequantization (INT8 -> FP32) with axis=-2
static void Dequantize2DAxis2AsymmetricInt8OperationExeFunc(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0], inputs[1], inputs[2]}, {outputs[0]}) {
        auto args = static_cast<const DequantizeOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];

        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);

        int axis = args->axis_;  // axis=-2

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                auto tileTensorInput = View(inputs[0], {firstViewShape, secondViewShape},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                    {bIdx * firstViewShape, sIdx * secondViewShape});

                // Scale shape: [1, col] for axis=-2
                auto tileTensorScale = View(inputs[1], {1, secondViewShape},
                    {1, std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                    {0, sIdx * secondViewShape});

                // Zero_points shape: [1, col] for axis=-2
                auto tileTensorZeroPoints = View(inputs[2], {1, secondViewShape},
                    {1, std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                    {0, sIdx * secondViewShape});

                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Dequantize(tileTensorInput, tileTensorScale, DataType::DT_FP32, axis, tileTensorZeroPoints);
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

// ============================================================
// 2D Tensor Tests - INT16
// ============================================================

// Symmetric dequantization (INT16 -> FP32) with axis=-1
static void DequantizeSymmetricInt16OperationExeFunc(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]}) {
        auto args = static_cast<const DequantizeOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];

        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);

        int axis = args->axis_;

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                auto tileTensorInput = View(inputs[0], {firstViewShape, secondViewShape},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                    {bIdx * firstViewShape, sIdx * secondViewShape});

                auto tileTensorScale = View(inputs[1], {firstViewShape, 1},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape), 1},
                    {bIdx * firstViewShape, 0});

                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Dequantize(tileTensorInput, tileTensorScale, DataType::DT_FP32, axis, Tensor());
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

// Symmetric dequantization (INT16 -> FP32) with axis=-2
static void Dequantize2DAxis2Int16OperationExeFunc(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]}) {
        auto args = static_cast<const DequantizeOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];

        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);

        int axis = args->axis_;

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                auto tileTensorInput = View(inputs[0], {firstViewShape, secondViewShape},
                    {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                    {bIdx * firstViewShape, sIdx * secondViewShape});

                auto tileTensorScale = View(inputs[1], {1, secondViewShape},
                    {1, std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                    {0, sIdx * secondViewShape});

                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = Dequantize(tileTensorInput, tileTensorScale, DataType::DT_FP32, axis, Tensor());
                Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape}, outputs[0]);
            }
        }
    }
}

// ============================================================
// 3D Tensor Tests
// ============================================================

// 3D tensor dequantization with axis=-1 (INT8)
static void Dequantize3DInt8OperationExeFunc(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]}) {
        auto args = static_cast<const DequantizeOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];

        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        const int nloop = CeilDiv(thirdDim, thirdViewShape);

        int axis = args->axis_;

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1)) {
                    auto tileTensorInput = View(inputs[0], {firstViewShape, secondViewShape, thirdViewShape},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                            std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                            std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                        {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});

                    auto tileTensorScale = View(inputs[1], {firstViewShape, 1, 1},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape), 1, 1},
                        {bIdx * firstViewShape, 0, 0});

                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Dequantize(tileTensorInput, tileTensorScale, DataType::DT_FP32, axis, Tensor());
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

// 3D tensor dequantization with axis=-2 (INT8)
static void Dequantize3DAxis2Int8OperationExeFunc(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]}) {
        auto args = static_cast<const DequantizeOpFuncArgs *>(opArgs);
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];

        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        const int bloop = CeilDiv(firstDim, firstViewShape);
        const int sloop = CeilDiv(secondDim, secondViewShape);
        const int nloop = CeilDiv(thirdDim, thirdViewShape);

        int axis = args->axis_;

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bloop, 1)) {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sloop, 1)) {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nloop, 1)) {
                    auto tileTensorInput = View(inputs[0], {firstViewShape, secondViewShape, thirdViewShape},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                            std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                            std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                        {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape});

                    auto tileTensorScale = View(inputs[1], {firstViewShape, secondViewShape, 1},
                        {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                            std::min(secondDim - sIdx * secondViewShape, secondViewShape), 1},
                        {bIdx * firstViewShape, sIdx * secondViewShape, 0});

                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = Dequantize(tileTensorInput, tileTensorScale, DataType::DT_FP32, axis, Tensor());
                    Assemble(res, {bIdx * firstViewShape, sIdx * secondViewShape, nIdx * thirdViewShape}, outputs[0]);
                }
            }
        }
    }
}

// ============================================================
// 4D Tensor Tests
// ============================================================

// 4D tensor dequantization with axis=-1 (INT8)
static void Dequantize4DInt8OperationExeFunc(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]}) {
        auto args = static_cast<const DequantizeOpFuncArgs *>(opArgs);
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

                        auto tileTensorScale = View(inputs[1], {firstViewShape, secondViewShape, 1, 1},
                            {std::min(firstDim - idx0 * firstViewShape, firstViewShape),
                                std::min(secondDim - idx1 * secondViewShape, secondViewShape), 1, 1},
                            {idx0 * firstViewShape, idx1 * secondViewShape, 0, 0});

                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Dequantize(tileTensorInput, tileTensorScale, DataType::DT_FP32, axis, Tensor());
                        Assemble(res, {idx0 * firstViewShape, idx1 * secondViewShape, idx2 * thirdViewShape, idx3 * fourthViewShape}, outputs[0]);
                    }
                }
            }
        }
    }
}

// 4D tensor dequantization with axis=-2 (INT8)
static void Dequantize4DAxis2Int8OperationExeFunc(
    const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs, const OpFuncArgs *opArgs) {
    FUNCTION("main", {inputs[0], inputs[1]}, {outputs[0]}) {
        auto args = static_cast<const DequantizeOpFuncArgs *>(opArgs);
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

                        auto tileTensorScale = View(inputs[1], {firstViewShape, secondViewShape, thirdViewShape, 1},
                            {std::min(firstDim - idx0 * firstViewShape, firstViewShape),
                                std::min(secondDim - idx1 * secondViewShape, secondViewShape),
                                std::min(thirdDim - idx2 * thirdViewShape, thirdViewShape), 1},
                            {idx0 * firstViewShape, idx1 * secondViewShape, idx2 * thirdViewShape, 0});

                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = Dequantize(tileTensorInput, tileTensorScale, DataType::DT_FP32, axis, Tensor());
                        Assemble(res, {idx0 * firstViewShape, idx1 * secondViewShape, idx2 * thirdViewShape, idx3 * fourthViewShape}, outputs[0]);
                    }
                }
            }
        }
    }
}

// Helper function to select the appropriate execution function based on test parameters
OpFunc SelectDequantizeOpFunc(int ndim, int axis, DataType inputDtype, bool useZeroPoints) {
    // Normalize axis to negative indexing
    int normalizedAxis = (axis >= 0) ? axis - ndim : axis;

    if (inputDtype == DataType::DT_INT8) {
        if (ndim == 2) {
            if (normalizedAxis == -1) {
                return useZeroPoints ? DequantizeAsymmetricInt8OperationExeFunc : DequantizeSymmetricInt8OperationExeFunc;
            } else {
                return useZeroPoints ? Dequantize2DAxis2AsymmetricInt8OperationExeFunc : Dequantize2DAxis2Int8OperationExeFunc;
            }
        } else if (ndim == 3) {
            return (normalizedAxis == -1) ? Dequantize3DInt8OperationExeFunc : Dequantize3DAxis2Int8OperationExeFunc;
        } else if (ndim == 4) {
            return (normalizedAxis == -1) ? Dequantize4DInt8OperationExeFunc : Dequantize4DAxis2Int8OperationExeFunc;
        }
    } else if (inputDtype == DataType::DT_INT16) {
        if (ndim == 2) {
            return (normalizedAxis == -1) ? DequantizeSymmetricInt16OperationExeFunc : Dequantize2DAxis2Int16OperationExeFunc;
        }
        // For INT16 with higher dimensions, use INT8 functions (type doesn't affect loop structure)
        if (ndim == 3) {
            return (normalizedAxis == -1) ? Dequantize3DInt8OperationExeFunc : Dequantize3DAxis2Int8OperationExeFunc;
        } else if (ndim == 4) {
            return (normalizedAxis == -1) ? Dequantize4DInt8OperationExeFunc : Dequantize4DAxis2Int8OperationExeFunc;
        }
    }

    // Default to 2D symmetric INT8
    return DequantizeSymmetricInt8OperationExeFunc;
}

class DequantizeOperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<DequantizeOpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestDequantize, DequantizeOperationTest,
    ::testing::ValuesIn(GetOpMetaData<DequantizeOpMetaData>(
        {DequantizeSymmetricInt8OperationExeFunc}, "Dequantize")));

TEST_P(DequantizeOperationTest, TestDequantize) {
    auto test_data = GetParam().test_data_;
    auto inputDtype = static_cast<DataType>(GetValueByName<int>(test_data, "input_dtype"));
    auto axis = GetValueByName<int>(test_data, "axis");
    auto useZeroPoints = GetValueByName<bool>(test_data, "use_zero_points");
    auto viewShape = GetViewShape(test_data);
    int ndim = static_cast<int>(viewShape.size());

    // Dynamically select the appropriate execution function
    auto selectedOpFunc = SelectDequantizeOpFunc(ndim, axis, inputDtype, useZeroPoints);

    auto args = DequantizeOpFuncArgs(viewShape, GetTileShape(test_data), inputDtype, axis, useZeroPoints);

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
