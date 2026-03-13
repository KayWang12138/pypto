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
 * \file quantize.cpp
 * \brief Quantize operation implementation
 */

#include "quantize.h"
#include "interface/inner/tilefwk.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"

namespace npu::tile_fwk {

// Normalize axis to negative indexing (-1 for last dim, -2 for second last, etc.)
static int NormalizeAxis(int axis, int ndim) {
    if (axis >= 0) {
        return axis - ndim;
    }
    return axis;
}

void CheckQuantize(const LogicalTensorPtr &input, const LogicalTensorPtr &scale,
                   DataType otype, int axis, const LogicalTensorPtr &zeroPoints) {
    // 1. Input must be FP32
    ASSERT(input->Datatype() == DataType::DT_FP32)
        << "Quantize input must be DT_FP32, but got " << DataType2String(input->Datatype());

    // 2. Dimension check: 2-4 dimensions
    auto shapeSize = input->shape.size();
    ASSERT(shapeSize >= 2 && shapeSize <= 4)
        << "Quantize input must be 2-4 dimensional, but got " << shapeSize << " dimensions";

    // 3. Shape size check
    int64_t totalSize = 1;
    for (auto dim : input->shape) {
        ASSERT(dim > 0) << "Input dimension must be positive";
        totalSize *= dim;
    }
    ASSERT(totalSize <= INT32_MAX)
        << "Input size " << totalSize << " exceeds INT32_MAX " << INT32_MAX;

    // 4. Scale type check
    ASSERT(scale->Datatype() == DataType::DT_FP32)
        << "Scale must be DT_FP32, but got " << DataType2String(scale->Datatype());

    // 5. Axis check: -1, -2, or relative dimensions
    int normalizedAxis = NormalizeAxis(axis, static_cast<int>(shapeSize));
    ASSERT(normalizedAxis == -1 || normalizedAxis == -2)
        << "Quantize axis must be -1 or -2 (last two dimensions), but got " << axis
        << " (normalized to " << normalizedAxis << ")";

    // 6. Output type check
    ASSERT(otype == DataType::DT_INT8 || otype == DataType::DT_UINT8)
        << "Quantize output type must be DT_INT8 or DT_UINT8, but got " << DataType2String(otype);

    // 7. Asymmetric quantization requires zero_points
    if (otype == DataType::DT_UINT8) {
        ASSERT(zeroPoints != nullptr)
            << "Asymmetric quantization (DT_UINT8) requires zero_points";
        ASSERT(zeroPoints->Datatype() == DataType::DT_FP32)
            << "zero_points must be DT_FP32, but got " << DataType2String(zeroPoints->Datatype());
    }

    // 8. Scale shape validation
    // For axis=-1: scale shape should be [..., row, 1] (last dim is 1)
    // For axis=-2: scale shape should be [..., 1, col] (second last dim is 1)
    ASSERT(scale->shape.size() == shapeSize)
        << "Scale must have same rank as input, got scale rank " << scale->shape.size()
        << " vs input rank " << shapeSize;

    for (size_t i = 0; i < shapeSize; ++i) {
        int normalizedIdx = static_cast<int>(i) - static_cast<int>(shapeSize);
        if (normalizedIdx == normalizedAxis) {
            // Scale axis: must be 1 (this is the dimension being quantized/compressed)
            ASSERT(scale->shape[i] == 1)
                << "Scale shape[" << i << "] must be 1 on scale axis, got " << scale->shape[i];
        } else {
            // Non-scale axis: can broadcast (be 1) or match input shape
            ASSERT(scale->shape[i] == 1 || scale->shape[i] == input->shape[i])
                << "Scale shape[" << i << "] must be 1 or match input shape[" << i << "] ("
                << input->shape[i] << "), got " << scale->shape[i];
        }
    }

    // 9. Zero points shape validation (if provided)
    if (zeroPoints != nullptr) {
        ASSERT(zeroPoints->shape.size() == shapeSize)
            << "Zero points must have same rank as input";
        for (size_t i = 0; i < shapeSize; ++i) {
            ASSERT(zeroPoints->shape[i] == 1 || zeroPoints->shape[i] == input->shape[i])
                << "Zero points shape[" << i << "] must be 1 or match input";
        }
    }
}

// Logical tensor operation (non-tiled)
LogicalTensorPtr TensorQuantizeOperation(Function &function, const LogicalTensorPtr &input,
                                         const LogicalTensorPtr &scale, DataType otype,
                                         int axis, const LogicalTensorPtr &zeroPoints) {
    DECLARE_TRACER();

    // Parameter validation
    CheckQuantize(input, scale, otype, axis, zeroPoints);

    // Create result tensor
    auto result = std::make_shared<LogicalTensor>(function, otype, input->shape, input->GetDynValidShape());

    // Prepare inputs
    std::vector<LogicalTensorPtr> inputs = {input, scale};
    if (otype == DataType::DT_UINT8 && zeroPoints != nullptr) {
        inputs.push_back(zeroPoints);
    }

    // Add operation with axis attribute
    Opcode opCode = (otype == DataType::DT_UINT8)
        ? Opcode::OP_QUANTIZE_ASYM
        : Opcode::OP_QUANTIZE_SYM;
    auto &op = function.AddOperation(opCode, inputs, {result});

    // Store axis as operation attribute for TileFunc to retrieve
    int normalizedAxis = NormalizeAxis(axis, static_cast<int>(input->shape.size()));
    op.SetAttribute(OP_ATTR_PREFIX + "AXIS", static_cast<int64_t>(normalizedAxis));

    return result;
}

// Tile operation implementation for 2D tensors with axis=-1 (native TQuant support)
template <QuantizeType quantType>
void TiledQuantize2DAxisLast(Function &function, const TileShape &tileShape,
                             size_t cur, Input &input, Input &scale,
                             const LogicalTensorPtr &result,
                             const LogicalTensorPtr &zeroPoints) {
    if (cur == input.tensor.GetShape().size()) {
        auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto scaleTile = scale.tensor.GetStorage()->View(function, scale.tileInfo.shape, scale.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);

        std::vector<LogicalTensorPtr> inputs = {inputTile, scaleTile};
        if (quantType == QuantizeType::INT8_ASYM && zeroPoints != nullptr) {
            TileInfo zeroTileInfo(zeroPoints->shape.size(), zeroPoints->offset.size());
            for (size_t i = 0; i < zeroTileInfo.shape.size(); ++i) {
                zeroTileInfo.offset[i] = (zeroPoints->shape[i] == 1) ? 0 : input.tileInfo.offset[i];
                zeroTileInfo.shape[i] = (zeroPoints->shape[i] == 1) ? 1 : input.tileInfo.shape[i];
            }
            auto zeroTile = zeroPoints->View(function, zeroTileInfo.shape, zeroTileInfo.offset);
            inputs.push_back(zeroTile);
        }

        function.AddOperation(GetQuantizeOpCode<quantType>(), inputs, {resultTile});
        return;
    }

    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        scale.tileInfo.shape[cur] = (scale.tensor.GetShape()[cur] == 1) ? 1 : std::min(scale.tensor.GetShape()[cur] - i, vecTile[cur]);
        scale.tileInfo.offset[cur] = (scale.tensor.GetShape()[cur] == 1) ? 0 : i;
        TiledQuantize2DAxisLast<quantType>(function, tileShape, cur + 1, input, scale, result, zeroPoints);
    }
}

// Main tiled quantize operation dispatcher
template <QuantizeType quantType>
void TiledQuantizeOperation(Function &function, const TileShape &tileShape, size_t cur,
                            Input &input, Input &scale, const LogicalTensorPtr &result,
                            const LogicalTensorPtr &zeroPoints, int axis) {
    int ndim = static_cast<int>(input.tensor.GetShape().size());
    int normalizedAxis = NormalizeAxis(axis, ndim);

    // For 2D tensor with axis=-1, use native TQuant path
    if (ndim == 2 && normalizedAxis == -1) {
        TiledQuantize2DAxisLast<quantType>(function, tileShape, cur, input, scale, result, zeroPoints);
        return;
    }

    // For other cases (3D/4D or axis=-2), we handle by treating as 2D logically
    // The key insight: TQuant operates on tiles, and we need to ensure scale tiles
    // align correctly with input tiles based on the axis.

    // For axis=-1 on N-D tensor:
    // - Scale shape is [..., 1] broadcastable
    // - We iterate through all dimensions, with scale broadcast on all but last

    // For axis=-2 on N-D tensor:
    // - Scale shape is [..., 1, N] where N matches input's second-last dim
    // - We need special handling for the second-last dimension

    if (cur == input.tensor.GetShape().size()) {
        auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);

        // Calculate scale tile info based on axis
        TileInfo scaleTileInfo(scale.tensor.GetShape().size(), scale.tileInfo.offset.size());
        for (size_t i = 0; i < scaleTileInfo.shape.size(); ++i) {
            int normalizedIdx = static_cast<int>(i) - static_cast<int>(scale.tensor.GetShape().size());

            if (normalizedAxis == -1) {
                // axis=-1: scale broadcasts on all dims except last
                scaleTileInfo.shape[i] = (scale.tensor.GetShape()[i] == 1) ? 1 : input.tileInfo.shape[i];
                scaleTileInfo.offset[i] = (scale.tensor.GetShape()[i] == 1) ? 0 : input.tileInfo.offset[i];
            } else {
                // axis=-2: scale broadcasts except on second-last dim
                if (normalizedIdx == -2) {
                    // This is the scale dimension
                    scaleTileInfo.shape[i] = (scale.tensor.GetShape()[i] == 1) ? 1 : input.tileInfo.shape[i];
                    scaleTileInfo.offset[i] = (scale.tensor.GetShape()[i] == 1) ? 0 : input.tileInfo.offset[i];
                } else {
                    // Broadcast dimension
                    scaleTileInfo.shape[i] = 1;
                    scaleTileInfo.offset[i] = 0;
                }
            }
        }
        auto scaleTile = scale.tensor.GetStorage()->View(function, scaleTileInfo.shape, scaleTileInfo.offset);

        std::vector<LogicalTensorPtr> inputs = {inputTile, scaleTile};
        if (quantType == QuantizeType::INT8_ASYM && zeroPoints != nullptr) {
            TileInfo zeroTileInfo(zeroPoints->shape.size(), zeroPoints->offset.size());
            for (size_t i = 0; i < zeroTileInfo.shape.size(); ++i) {
                int normalizedIdx = static_cast<int>(i) - static_cast<int>(zeroPoints->shape.size());

                if (normalizedAxis == -1) {
                    zeroTileInfo.offset[i] = (zeroPoints->shape[i] == 1) ? 0 : input.tileInfo.offset[i];
                    zeroTileInfo.shape[i] = (zeroPoints->shape[i] == 1) ? 1 : input.tileInfo.shape[i];
                } else {
                    // For axis=-2, zero_points should follow similar pattern to scale
                    if (normalizedIdx == -2) {
                        zeroTileInfo.offset[i] = (zeroPoints->shape[i] == 1) ? 0 : input.tileInfo.offset[i];
                        zeroTileInfo.shape[i] = (zeroPoints->shape[i] == 1) ? 1 : input.tileInfo.shape[i];
                    } else {
                        zeroTileInfo.offset[i] = 0;
                        zeroTileInfo.shape[i] = 1;
                    }
                }
            }
            auto zeroTile = zeroPoints->View(function, zeroTileInfo.shape, zeroTileInfo.offset);
            inputs.push_back(zeroTile);
        }

        // Add attribute for axis in the tile operation
        auto &op = function.AddOperation(GetQuantizeOpCode<quantType>(), inputs, {resultTile});
        op.SetAttribute(OP_ATTR_PREFIX + "AXIS", static_cast<int64_t>(normalizedAxis));
        return;
    }

    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        scale.tileInfo.shape[cur] = (scale.tensor.GetShape()[cur] == 1) ? 1 : std::min(scale.tensor.GetShape()[cur] - i, vecTile[cur]);
        scale.tileInfo.offset[cur] = (scale.tensor.GetShape()[cur] == 1) ? 0 : i;
        TiledQuantizeOperation<quantType>(function, tileShape, cur + 1, input, scale, result, zeroPoints, axis);
    }
}

// Wrapper function for tiled operation
template <QuantizeType quantType>
void TiledQuantizeOperation(Function &function, const TileShape &tileShape,
                            const LogicalTensorPtr &input, const LogicalTensorPtr &scale,
                            const LogicalTensorPtr &result, const LogicalTensorPtr &zeroPoints, int axis) {
    ASSERT(input->shape.size() == input->offset.size()) << "The shape size of input and offset must be equal";

    TileInfo inputTileInfo(result->shape.size(), result->offset.size());
    TileInfo scaleTileInfo(result->shape.size(), result->offset.size());
    auto inputWrap = Input{input, inputTileInfo};
    auto scaleWrap = Input{scale, scaleTileInfo};
    TiledQuantizeOperation<quantType>(function, tileShape, 0, inputWrap, scaleWrap, result, zeroPoints, axis);
}

// Tile func wrapper for registration (symmetric)
void QuantizeSymOperationTileFunc(Function &function, const TileShape &tileShape,
                                  const std::vector<LogicalTensorPtr> &iOperand,
                                  const std::vector<LogicalTensorPtr> &oOperand,
                                  const Operation &op) {
    ASSERT(iOperand.size() == 2) << "Quantize symmetric operation requires 2 inputs (input, scale)";
    ASSERT(oOperand.size() == 1) << "Quantize operation requires 1 output";

    // Get axis from operation attribute (defaults to -1 if not set)
    int axis = -1;
    if (op.HasAttribute(OP_ATTR_PREFIX + "AXIS")) {
        axis = static_cast<int>(op.GetIntAttribute(OP_ATTR_PREFIX + "AXIS"));
    }

    TiledQuantizeOperation<QuantizeType::INT8_SYM>(function, tileShape,
        iOperand[0], iOperand[1], oOperand[0], nullptr, axis);
}

// Tile func wrapper for registration (asymmetric)
void QuantizeAsymOperationTileFunc(Function &function, const TileShape &tileShape,
                                   const std::vector<LogicalTensorPtr> &iOperand,
                                   const std::vector<LogicalTensorPtr> &oOperand,
                                   const Operation &op) {
    ASSERT(iOperand.size() == 3) << "Quantize asymmetric operation requires 3 inputs (input, scale, zero_points)";
    ASSERT(oOperand.size() == 1) << "Quantize operation requires 1 output";

    // Get axis from operation attribute (defaults to -1 if not set)
    int axis = -1;
    if (op.HasAttribute(OP_ATTR_PREFIX + "AXIS")) {
        axis = static_cast<int>(op.GetIntAttribute(OP_ATTR_PREFIX + "AXIS"));
    }

    TiledQuantizeOperation<QuantizeType::INT8_ASYM>(function, tileShape,
        iOperand[0], iOperand[1], oOperand[0], iOperand[2], axis);
}

Tensor Quantize(const Tensor &input, const Tensor &scale, DataType otype, int axis, const Tensor &zeroPoints) {
    DECLARE_TRACER();
    ASSERT(input.GetShape().size() == input.GetStorage()->offset.size())
        << "The shape size of input and offset should be equal";

    // Convert to LogicalTensor pointers
    LogicalTensorPtr zeroPointsPtr = nullptr;
    if (zeroPoints.GetStorage() != nullptr) {
        zeroPointsPtr = zeroPoints.GetStorage();
    }

    RETURN_CALL(QuantizeOperation, *Program::GetInstance().GetCurrentFunction(),
        input.GetStorage(), scale.GetStorage(), otype, axis, zeroPointsPtr);
}

REGISTER_OPERATION_TILED_FUNC(OP_QUANTIZE_SYM, Opcode::OP_QUANTIZE_SYM, QuantizeSymOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_QUANTIZE_ASYM, Opcode::OP_QUANTIZE_ASYM, QuantizeAsymOperationTileFunc);

} // namespace npu::tile_fwk
