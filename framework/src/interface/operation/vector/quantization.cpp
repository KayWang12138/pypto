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
 * \file quantization.cpp
 * \brief Quantization operation implementation for INT8 symmetric and asymmetric quantization
 */

#include "interface/utils/operator_tracer.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/operation/vector/tensor_transformation.h"

namespace npu::tile_fwk {

// =============================================================================
// Symmetric Quantization (FP32 -> INT8)
// =============================================================================

void TiledQuantizeSymmetric(Function &function, const TileShape &tileShape, size_t cur,
    Input &srcInput, Input &scaleInput, Input &dstInput, int64_t axis) {
    if (cur == dstInput.tensor.GetShape().size()) {
        auto srcTile = srcInput.tensor.GetStorage()->View(function, srcInput.tileInfo.shape, srcInput.tileInfo.offset);
        auto scaleTile = scaleInput.tensor.GetStorage()->View(function, scaleInput.tileInfo.shape, scaleInput.tileInfo.offset);
        auto dstTile = dstInput.tensor.GetStorage()->View(function, dstInput.tileInfo.shape, dstInput.tileInfo.offset);

        auto &op = function.AddOperation(Opcode::OP_QUANTIZE_SYM, {srcTile, scaleTile}, {dstTile});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        return;
    }

    auto &vecTile = tileShape.GetVecTile();
    for (int64_t i = 0; i < dstInput.tensor.GetShape()[cur]; i += vecTile[cur]) {
        // Update dst tile info
        dstInput.tileInfo.shape[cur] = std::min(dstInput.tensor.GetShape()[cur] - i, vecTile[cur]);
        dstInput.tileInfo.offset[cur] = i;

        // Update src tile info - src has same shape as dst
        if (cur < srcInput.tensor.GetShape().size()) {
            srcInput.tileInfo.shape[cur] = std::min(srcInput.tensor.GetShape()[cur] - i, vecTile[cur]);
            srcInput.tileInfo.offset[cur] = i;
        }

        // Update scale tile info - scale may have different shape depending on axis
        if (cur < scaleInput.tensor.GetShape().size()) {
            // If scale's dimension is 1 (broadcast dimension), use modulo which gives 0
            // Otherwise use the same index as src/dst
            int64_t scaleIdx = i % scaleInput.tensor.GetShape()[cur];
            scaleInput.tileInfo.shape[cur] = std::min(scaleInput.tensor.GetShape()[cur] - scaleIdx, vecTile[cur]);
            scaleInput.tileInfo.offset[cur] = scaleIdx;
        }

        TiledQuantizeSymmetric(function, tileShape, cur + 1, srcInput, scaleInput, dstInput, axis);
    }
}

void TiledQuantizeSymmetric(Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &src, const LogicalTensorPtr &scale,
    const LogicalTensorPtr &dst, int64_t axis) {
    ASSERT(src->shape.size() == src->offset.size()) << "Source shape size and offset size should be equal";
    ASSERT(dst->shape.size() == dst->offset.size()) << "Destination shape size and offset size should be equal";

    TileInfo srcTileInfo(src->shape.size(), src->offset.size());
    TileInfo scaleTileInfo(scale->shape.size(), scale->offset.size());
    TileInfo dstTileInfo(dst->shape.size(), dst->offset.size());

    auto srcInput = Input{Tensor(src), srcTileInfo};
    auto scaleInput = Input{Tensor(scale), scaleTileInfo};
    auto dstInput = Input{Tensor(dst), dstTileInfo};

    TiledQuantizeSymmetric(function, tileShape, 0, srcInput, scaleInput, dstInput, axis);
}

// =============================================================================
// Asymmetric Quantization (FP32 -> UINT8)
// =============================================================================

void TiledQuantizeAsymmetric(Function &function, const TileShape &tileShape, size_t cur,
    Input &srcInput, Input &scaleInput, Input &offsetInput, Input &dstInput, int64_t axis) {
    if (cur == dstInput.tensor.GetShape().size()) {
        auto srcTile = srcInput.tensor.GetStorage()->View(function, srcInput.tileInfo.shape, srcInput.tileInfo.offset);
        auto scaleTile = scaleInput.tensor.GetStorage()->View(function, scaleInput.tileInfo.shape, scaleInput.tileInfo.offset);
        auto offsetTile = offsetInput.tensor.GetStorage()->View(function, offsetInput.tileInfo.shape, offsetInput.tileInfo.offset);
        auto dstTile = dstInput.tensor.GetStorage()->View(function, dstInput.tileInfo.shape, dstInput.tileInfo.offset);

        auto &op = function.AddOperation(Opcode::OP_QUANTIZE_ASYM, {srcTile, scaleTile, offsetTile}, {dstTile});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        return;
    }

    auto &vecTile = tileShape.GetVecTile();
    for (int64_t i = 0; i < dstInput.tensor.GetShape()[cur]; i += vecTile[cur]) {
        // Update dst tile info
        dstInput.tileInfo.shape[cur] = std::min(dstInput.tensor.GetShape()[cur] - i, vecTile[cur]);
        dstInput.tileInfo.offset[cur] = i;

        // Update src tile info - src has same shape as dst
        if (cur < srcInput.tensor.GetShape().size()) {
            srcInput.tileInfo.shape[cur] = std::min(srcInput.tensor.GetShape()[cur] - i, vecTile[cur]);
            srcInput.tileInfo.offset[cur] = i;
        }

        // Update scale tile info - scale may have different shape depending on axis
        if (cur < scaleInput.tensor.GetShape().size()) {
            int64_t scaleIdx = i % scaleInput.tensor.GetShape()[cur];
            scaleInput.tileInfo.shape[cur] = std::min(scaleInput.tensor.GetShape()[cur] - scaleIdx, vecTile[cur]);
            scaleInput.tileInfo.offset[cur] = scaleIdx;
        }

        // Update offset tile info - offset has same shape as scale
        if (cur < offsetInput.tensor.GetShape().size()) {
            int64_t offsetIdx = i % offsetInput.tensor.GetShape()[cur];
            offsetInput.tileInfo.shape[cur] = std::min(offsetInput.tensor.GetShape()[cur] - offsetIdx, vecTile[cur]);
            offsetInput.tileInfo.offset[cur] = offsetIdx;
        }

        TiledQuantizeAsymmetric(function, tileShape, cur + 1, srcInput, scaleInput, offsetInput, dstInput, axis);
    }
}

void TiledQuantizeAsymmetric(Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &src, const LogicalTensorPtr &scale, const LogicalTensorPtr &offset,
    const LogicalTensorPtr &dst, int64_t axis) {
    ASSERT(src->shape.size() == src->offset.size()) << "Source shape size and offset size should be equal";
    ASSERT(dst->shape.size() == dst->offset.size()) << "Destination shape size and offset size should be equal";

    TileInfo srcTileInfo(src->shape.size(), src->offset.size());
    TileInfo scaleTileInfo(scale->shape.size(), scale->offset.size());
    TileInfo offsetTileInfo(offset->shape.size(), offset->offset.size());
    TileInfo dstTileInfo(dst->shape.size(), dst->offset.size());

    auto srcInput = Input{Tensor(src), srcTileInfo};
    auto scaleInput = Input{Tensor(scale), scaleTileInfo};
    auto offsetInput = Input{Tensor(offset), offsetTileInfo};
    auto dstInput = Input{Tensor(dst), dstTileInfo};

    TiledQuantizeAsymmetric(function, tileShape, 0, srcInput, scaleInput, offsetInput, dstInput, axis);
}

// =============================================================================
// Tensor-level Quantization Operations
// =============================================================================

LogicalTensorPtr TensorQuantizeSymmetricOperation(Function &function,
    const LogicalTensorPtr &src, const LogicalTensorPtr &scale, int64_t axis) {
    // Output is INT8 for symmetric quantization
    auto result = std::make_shared<LogicalTensor>(function, DataType::DT_INT8, src->shape, src->GetDynValidShape());
    auto &op = function.AddOperation(Opcode::OP_QUANTIZE_SYM, {src, scale}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
    function.UpdateTensorDataUsage(op);
    return result;
}

LogicalTensorPtr TensorQuantizeAsymmetricOperation(Function &function,
    const LogicalTensorPtr &src, const LogicalTensorPtr &scale, const LogicalTensorPtr &offset, int64_t axis) {
    // Output is UINT8 for asymmetric quantization
    auto result = std::make_shared<LogicalTensor>(function, DataType::DT_UINT8, src->shape, src->GetDynValidShape());
    auto &op = function.AddOperation(Opcode::OP_QUANTIZE_ASYM, {src, scale, offset}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
    function.UpdateTensorDataUsage(op);
    return result;
}

// =============================================================================
// Public Quantize API
// =============================================================================

Tensor Quantize(const Tensor &input, const Tensor &scale, DataType otype, int axis, const Tensor &zeroPoints) {
    DECLARE_TRACER();

    // Validate input shapes
    ASSERT(input.GetShape().size() >= SHAPE_DIM2 && input.GetShape().size() <= SHAPE_DIM5)
        << "The shape.size() only support 2~5";

    // Validate data types
    std::vector<DataType> SUPPORT_INPUT_DATATYPES = {DataType::DT_FP32, DataType::DT_FP16, DataType::DT_BF16};
    ASSERT(std::find(SUPPORT_INPUT_DATATYPES.begin(), SUPPORT_INPUT_DATATYPES.end(), input.GetDataType()) !=
           SUPPORT_INPUT_DATATYPES.end())
        << "The input datatype is not supported";

    // Normalize axis to negative indexing
    int ndim = static_cast<int>(input.GetShape().size());
    int normalizedAxis = axis;
    if (axis >= 0) {
        normalizedAxis = axis - ndim;
    }
    ASSERT(normalizedAxis == -1 || normalizedAxis == -2) << "Only axis=-1 (per-row) and axis=-2 (per-column) are supported";

    // Determine quantization type based on zeroPoints presence
    bool isAsymmetric = (zeroPoints.GetStorage() != nullptr);

    // For axis=-2 (per-column quantization), use Transpose to swap last two dimensions
    // Strategy: transpose input and scale -> quantize with axis=-1 -> transpose output back
    if (normalizedAxis == -2) {
        // Swap last two dimensions: [..., H, W] -> [..., W, H]
        int lastDim = ndim - 1;        // -1 in positive index
        int secondLastDim = ndim - 2;  // -2 in positive index

        // Transpose input: [..., H, W] -> [..., W, H]
        Tensor transposedInput = Transpose(input, {secondLastDim, lastDim});

        // Transpose scale: scale shape is [..., 1, W] -> [..., W, 1]
        Tensor transposedScale = Transpose(scale, {secondLastDim, lastDim});

        if (isAsymmetric) {
            // Transpose zeroPoints: [..., 1, W] -> [..., W, 1]
            Tensor transposedZeroPoints = Transpose(zeroPoints, {secondLastDim, lastDim});

            // Asymmetric quantization with axis=-1
            ASSERT(otype == DataType::DT_UINT8 || otype == DataType::DT_INT8)
                << "Asymmetric quantization output type should be UINT8 or INT8";
            Tensor quantizedResult = QuantizeAsymmetricOperation(
                *Program::GetInstance().GetCurrentFunction(),
                transposedInput.GetStorage(), transposedScale.GetStorage(),
                transposedZeroPoints.GetStorage(), -1);

            // Transpose output back: [..., W, H] -> [..., H, W]
            return Transpose(quantizedResult, {secondLastDim, lastDim});
        } else {
            // Symmetric quantization with axis=-1
            ASSERT(otype == DataType::DT_INT8)
                << "Symmetric quantization output type should be INT8";
            Tensor quantizedResult = QuantizeSymmetricOperation(
                *Program::GetInstance().GetCurrentFunction(),
                transposedInput.GetStorage(), transposedScale.GetStorage(), -1);

            // Transpose output back: [..., W, H] -> [..., H, W]
            return Transpose(quantizedResult, {secondLastDim, lastDim});
        }
    }

    // axis=-1 case: direct quantization without transpose
    if (isAsymmetric) {
        // Asymmetric quantization: FP32 -> UINT8
        ASSERT(otype == DataType::DT_UINT8 || otype == DataType::DT_INT8)
            << "Asymmetric quantization output type should be UINT8 or INT8";
        RETURN_CALL(QuantizeAsymmetricOperation, *Program::GetInstance().GetCurrentFunction(),
            input.GetStorage(), scale.GetStorage(), zeroPoints.GetStorage(), normalizedAxis);
    } else {
        // Symmetric quantization: FP32 -> INT8
        ASSERT(otype == DataType::DT_INT8)
            << "Symmetric quantization output type should be INT8";
        RETURN_CALL(QuantizeSymmetricOperation, *Program::GetInstance().GetCurrentFunction(),
            input.GetStorage(), scale.GetStorage(), normalizedAxis);
    }
}

// Convenience function for symmetric quantization
Tensor QuantizeSymmetric(const Tensor &src, const Tensor &scale, int64_t axis) {
    return Quantize(src, scale, DataType::DT_INT8, axis, Tensor());
}

// Convenience function for asymmetric quantization
Tensor QuantizeAsymmetric(const Tensor &src, const Tensor &scale, const Tensor &offset, int64_t axis) {
    return Quantize(src, scale, DataType::DT_UINT8, axis, offset);
}

// =============================================================================
// Tile Function Registration
// =============================================================================

void QuantizeSymmetricOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    const Operation &op) {
    int64_t axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    TiledQuantizeSymmetric(function, tileShape, iOperand[0], iOperand[1], oOperand[0], axis);
}

void QuantizeAsymmetricOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    const Operation &op) {
    int64_t axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    TiledQuantizeAsymmetric(function, tileShape, iOperand[0], iOperand[1], iOperand[2], oOperand[0], axis);
}

REGISTER_OPERATION_TILED_FUNC(OP_QUANTIZE_SYM, Opcode::OP_QUANTIZE_SYM, QuantizeSymmetricOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_QUANTIZE_ASYM, Opcode::OP_QUANTIZE_ASYM, QuantizeAsymmetricOperationTileFunc);

} // namespace npu::tile_fwk
