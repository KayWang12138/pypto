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
 * \file quant.cpp
 * \brief
 */

#include "interface/operation/operation_common.h"
#include "interface/program/program.h"
#include "interface/utils/operator_tracer.h"

namespace npu::tile_fwk {
namespace {
constexpr int64_t QUANT_MX_MIN_RANK = 1;
constexpr int64_t QUANT_MX_MAX_RANK = 4;
constexpr int64_t QUANT_MX_GROUP_COLS = 32;
constexpr int64_t QUANT_MX_SCALE_GROUP_COLS = 64;
constexpr int64_t QUANT_MX_SCALE_PAIR_SIZE = 2;
constexpr int64_t QUANT_MX_TILE_ALIGN_BYTES = 256;

int64_t CeilDiv(int64_t dividend, int64_t divisor) { return (dividend + divisor - 1) / divisor; }

void CheckQuantMXDtype(DataType quantDtype)
{
    ASSERT(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, quantDtype == DataType::DT_FP8E4M3)
        << "QuantMX currently only supports DT_FP8E4M3 output. Current quant dtype: " << DataType2String(quantDtype);
}

void CheckQuantMXMode(DequantScaleRoundingMode mode)
{
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, mode == DequantScaleRoundingMode::ROUND_DOWN)
        << "QuantMX currently only supports ROUND_DOWN (OCP standard) mode.";
}

DequantScaleRoundingMode GetQuantMXMode(const Operation& op)
{
    int64_t modeValue = 0;
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, op.GetAttr(OpAttributeKey::mxQuantMode, modeValue))
        << "QuantMX missing required attribute: " << OpAttributeKey::mxQuantMode;
    return static_cast<DequantScaleRoundingMode>(modeValue);
}

int64_t GetQuantMXAxis(const Operation& op)
{
    int64_t axis = 0;
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, op.GetAttr(OpAttributeKey::mxQuantAxis, axis))
        << "QuantMX missing required attribute: " << OpAttributeKey::mxQuantAxis;
    return axis;
}

int64_t GetQuantMXPerformanceMode(const Operation& op)
{
    int64_t performanceMode = 0;
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, op.GetAttr(OpAttributeKey::mxQuantPerformanceMode, performanceMode))
        << "QuantMX missing required attribute: " << OpAttributeKey::mxQuantPerformanceMode;
    return performanceMode;
}

int64_t NormalizeQuantMXAxis(int64_t axis, size_t rank)
{
    if (axis < 0) {
        axis += static_cast<int64_t>(rank);
    }
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, axis >= 0 && axis < static_cast<int64_t>(rank))
        << "QuantMX axis is out of range. Current axis: " << axis << ", input rank: " << rank;
    return axis;
}

void CheckQuantMXAxis(int64_t axis, size_t rank)
{
    const int64_t normalizedAxis = NormalizeQuantMXAxis(axis, rank);
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, normalizedAxis == static_cast<int64_t>(rank) - 1)
        << "QuantMX currently only supports the last axis. Current axis: " << axis << ", input rank: " << rank;
}

void CheckQuantMXInput(const Tensor& input, DataType quantDtype, DequantScaleRoundingMode mode, int64_t axis)
{
    const auto inputDtype = input.GetDataType();
    ASSERT(
        VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED,
        inputDtype == DataType::DT_FP16 || inputDtype == DataType::DT_BF16 || inputDtype == DataType::DT_FP32)
        << "QuantMX currently only supports DT_FP16, DT_BF16, and DT_FP32 input.";
    CheckQuantMXDtype(quantDtype);
    CheckQuantMXMode(mode);
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, input.Format() == TileOpFormat::TILEOP_ND)
        << "QuantMX only supports TILEOP_ND input.";
    ASSERT(
        VectorErrorCode::ERR_PARAM_INVALID,
        QUANT_MX_MIN_RANK <= input.GetShape().size() && input.GetShape().size() <= QUANT_MX_MAX_RANK)
        << "QuantMX only supports 1D to 4D input.";
    CheckQuantMXAxis(axis, input.GetShape().size());
    const int64_t lastDimBytes = input.GetShape().back() * BytesOf(inputDtype);
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, lastDimBytes % QUANT_MX_TILE_ALIGN_BYTES == 0)
        << "QuantMX view shape's last dim must be 256-byte aligned. Current last dim bytes: " << lastDimBytes;
}

std::vector<int64_t> BuildQuantMXGroupedShape(const std::vector<int64_t>& inputShape)
{
    auto groupedShape = inputShape;
    groupedShape.back() = CeilDiv(groupedShape.back(), QUANT_MX_GROUP_COLS);
    return groupedShape;
}

std::vector<int64_t> BuildQuantMXPerformanceGroupedShape(const std::vector<int64_t>& inputShape)
{
    if (inputShape.size() == 1) {
        return {CeilDiv(inputShape[0], QUANT_MX_GROUP_COLS)};
    }

    std::vector<int64_t> groupedShape;
    groupedShape.reserve(inputShape.size() - 1);
    for (size_t i = 0; i + 2 < inputShape.size(); ++i) {
        groupedShape.push_back(inputShape[i]);
    }
    groupedShape.push_back(inputShape[inputShape.size() - 2] * CeilDiv(inputShape.back(), QUANT_MX_GROUP_COLS));
    return groupedShape;
}

std::vector<int64_t> BuildQuantMXPerformanceVecTile(const std::vector<int64_t>& inputVecTile)
{
    if (inputVecTile.size() == 1) {
        return {CeilDiv(inputVecTile[0], QUANT_MX_GROUP_COLS)};
    }

    std::vector<int64_t> groupedVecTile;
    groupedVecTile.reserve(inputVecTile.size() - 1);
    for (size_t i = 0; i + 2 < inputVecTile.size(); ++i) {
        groupedVecTile.push_back(inputVecTile[i]);
    }
    groupedVecTile.push_back(
        inputVecTile[inputVecTile.size() - 2] * CeilDiv(inputVecTile.back(), QUANT_MX_GROUP_COLS));
    return groupedVecTile;
}

std::vector<int64_t> BuildQuantMXPerformanceGroupedOffset(
    const std::vector<int64_t>& inputOffset, const std::vector<int64_t>& inputShape,
    const std::vector<int64_t>& inputTileShape)
{
    if (inputOffset.size() == 1) {
        return {inputOffset[0] / QUANT_MX_GROUP_COLS};
    }

    std::vector<int64_t> groupedOffset;
    groupedOffset.reserve(inputOffset.size() - 1);
    for (size_t i = 0; i + 2 < inputOffset.size(); ++i) {
        groupedOffset.push_back(inputOffset[i]);
    }
    const int64_t groupCols = CeilDiv(inputShape.back(), QUANT_MX_GROUP_COLS);
    const int64_t tileRows = inputTileShape[inputTileShape.size() - 2];
    groupedOffset.push_back(
        inputOffset[inputOffset.size() - 2] * groupCols + tileRows * (inputOffset.back() / QUANT_MX_GROUP_COLS));
    return groupedOffset;
}

std::vector<SymbolicScalar> BuildQuantMXGroupedValidShape(const std::vector<SymbolicScalar>& inputValidShape)
{
    auto groupedValidShape = inputValidShape;
    groupedValidShape.back() = (groupedValidShape.back() + QUANT_MX_GROUP_COLS - 1) / QUANT_MX_GROUP_COLS;
    return groupedValidShape;
}

std::vector<SymbolicScalar> BuildQuantMXPerformanceGroupedValidShape(const std::vector<SymbolicScalar>& inputValidShape)
{
    if (inputValidShape.size() == 1) {
        return {(inputValidShape[0] + QUANT_MX_GROUP_COLS - 1) / QUANT_MX_GROUP_COLS};
    }

    std::vector<SymbolicScalar> groupedValidShape;
    groupedValidShape.reserve(inputValidShape.size() - 1);
    for (size_t i = 0; i + 2 < inputValidShape.size(); ++i) {
        groupedValidShape.push_back(inputValidShape[i]);
    }
    groupedValidShape.push_back(
        inputValidShape[inputValidShape.size() - 2] *
        ((inputValidShape.back() + QUANT_MX_GROUP_COLS - 1) / QUANT_MX_GROUP_COLS));
    return groupedValidShape;
}

std::vector<int64_t> BuildQuantMXScaleShape(const std::vector<int64_t>& inputShape)
{
    auto scaleShape = inputShape;
    scaleShape.back() = CeilDiv(scaleShape.back(), QUANT_MX_SCALE_GROUP_COLS);
    scaleShape.push_back(QUANT_MX_SCALE_PAIR_SIZE);
    return scaleShape;
}

std::vector<SymbolicScalar> BuildQuantMXScaleValidShape(const std::vector<SymbolicScalar>& inputValidShape)
{
    auto scaleValidShape = inputValidShape;
    scaleValidShape.back() = (scaleValidShape.back() + QUANT_MX_SCALE_GROUP_COLS - 1) / QUANT_MX_SCALE_GROUP_COLS;
    scaleValidShape.push_back(SymbolicScalar(QUANT_MX_SCALE_PAIR_SIZE));
    return scaleValidShape;
}

void CheckQuantMXTileShape(const LogicalTensorPtr& input, const VecTile& vecTile)
{
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, vecTile.size() == input->GetShape().size())
        << "QuantMX tile shape rank must match input rank.";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, vecTile[vecTile.size() - 1] > 0)
        << "QuantMX tile shape last dim must be positive.";

    const int64_t lastDimBytes = vecTile[vecTile.size() - 1] * BytesOf(input->Datatype());
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, lastDimBytes % QUANT_MX_TILE_ALIGN_BYTES == 0)
        << "QuantMX tile shape's last dim must be 256-byte aligned. Current last dim bytes: " << lastDimBytes;
}

void TiledQuantMXOperation(
    Function& function, const TileShape& tileShape, size_t cur, Input& input, const LogicalTensorPtr& dst,
    const LogicalTensorPtr& exp, const LogicalTensorPtr& maxScratch, const LogicalTensorPtr& scalingScratch,
    DequantScaleRoundingMode mode, int64_t axis, int64_t performanceMode)
{
    if (cur == input.tensor.GetShape().size()) {
        const int64_t lastDimBytes = input.tileInfo.shape.back() * BytesOf(input.tensor.GetDataType());
        ASSERT(VectorErrorCode::ERR_PARAM_INVALID, lastDimBytes % QUANT_MX_TILE_ALIGN_BYTES == 0)
            << "QuantMX tile width must be 256-byte aligned. Current last dim bytes: " << lastDimBytes;

        auto addQuantMXTile = [&](const std::vector<int64_t>& quantTileShape, const std::vector<int64_t>& tileOffset,
                                  const std::vector<int64_t>& groupedTileShape,
                                  const std::vector<int64_t>& groupedTileOffset) {
            auto srcTile = input.tensor.GetStorage()->View(function, quantTileShape, tileOffset);
            auto dstTile = dst->View(function, quantTileShape, tileOffset);
            auto expTile = exp->View(function, groupedTileShape, groupedTileOffset);
            auto maxTile = maxScratch->View(function, groupedTileShape, groupedTileOffset);
            auto scalingTile = scalingScratch->View(function, quantTileShape, tileOffset);
            auto& tiledOp =
                function.AddOperation(Opcode::OP_QUANT_MX, {srcTile}, {dstTile, expTile, maxTile, scalingTile});
            tiledOp.SetAttribute(OpAttributeKey::mxQuantMode, static_cast<int64_t>(mode));
            tiledOp.SetAttribute(OpAttributeKey::mxQuantAxis, axis);
            tiledOp.SetAttribute(OpAttributeKey::mxQuantPerformanceMode, performanceMode);
        };

        if (performanceMode == 0) {
            auto groupedTileShape = input.tileInfo.shape;
            groupedTileShape.back() = CeilDiv(groupedTileShape.back(), QUANT_MX_GROUP_COLS);
            auto groupedTileOffset = input.tileInfo.offset;
            groupedTileOffset.back() /= QUANT_MX_GROUP_COLS;
            addQuantMXTile(input.tileInfo.shape, input.tileInfo.offset, groupedTileShape, groupedTileOffset);
            return;
        }

        addQuantMXTile(
            input.tileInfo.shape, input.tileInfo.offset, BuildQuantMXPerformanceGroupedShape(input.tileInfo.shape),
            BuildQuantMXPerformanceGroupedOffset(input.tileInfo.offset, input.tensor.GetShape(), input.tileInfo.shape));
        return;
    }

    const auto& vecTile = tileShape.GetVecTile();
    int64_t step = std::max<int64_t>(1, std::min<int64_t>(vecTile[cur], input.tensor.GetShape()[cur]));

    for (int64_t i = 0; i < input.tensor.GetShape()[cur]; i += step) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, step);
        input.tileInfo.offset[cur] = i;
        TiledQuantMXOperation(
            function, tileShape, cur + 1, input, dst, exp, maxScratch, scalingScratch, mode, axis, performanceMode);
    }
}

void QuantMXTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, iOperand.size() == 1) << "QuantMX expects 1 input tensor.";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, oOperand.size() == 4) << "QuantMX expects 4 output tensors.";

    const auto& src = iOperand[0];
    const auto& dst = oOperand[0];
    const auto& exp = oOperand[1];
    const auto& maxScratch = oOperand[2];
    const auto& scalingScratch = oOperand[3];
    CheckQuantMXDtype(dst->Datatype());
    auto mode = GetQuantMXMode(op);
    auto axis = GetQuantMXAxis(op);
    auto performanceMode = GetQuantMXPerformanceMode(op);
    CheckQuantMXMode(mode);
    CheckQuantMXAxis(axis, src->GetShape().size());
    CheckQuantMXTileShape(src, tileShape.GetVecTile());
    TileInfo inputTileInfo(src->shape.size(), src->offset.size());
    auto input = Input{Tensor(src), inputTileInfo};
    TiledQuantMXOperation(
        function, tileShape, 0, input, dst, exp, maxScratch, scalingScratch, mode, axis, performanceMode);
}
} // namespace

std::tuple<Tensor, Tensor> QuantMX(
    const Tensor& input, DataType quantDtype, DequantScaleRoundingMode mode, int64_t axis, bool performanceMode)
{
    DECLARE_TRACER();
    CheckQuantMXInput(input, quantDtype, mode, axis);

    const auto& inputShape = input.GetShape();
    const int64_t normalizedAxis = NormalizeQuantMXAxis(axis, inputShape.size());
    const std::vector<int64_t> groupedShape = performanceMode ? BuildQuantMXPerformanceGroupedShape(inputShape) :
                                                                BuildQuantMXGroupedShape(inputShape);
    const std::vector<int64_t> scaleShape = BuildQuantMXScaleShape(inputShape);

    const auto scratchDtype = input.GetDataType();
    auto quantized = Tensor(quantDtype, inputShape, "", TileOpFormat::TILEOP_ND);
    auto exp = Tensor(DataType::DT_FP8E8M0, groupedShape, "", TileOpFormat::TILEOP_ND);
    auto maxScratch = Tensor(scratchDtype, groupedShape, "", TileOpFormat::TILEOP_ND);
    auto scalingScratch = Tensor(scratchDtype, inputShape, "", TileOpFormat::TILEOP_ND);

    std::vector<SymbolicScalar> scaleValidShape;
    const auto& inputValidShape = input.GetStorage()->GetDynValidShape();
    if (!inputValidShape.empty()) {
        quantized.GetStorage()->UpdateDynValidShape(inputValidShape);
        const auto groupedValidShape = performanceMode ? BuildQuantMXPerformanceGroupedValidShape(inputValidShape) :
                                                         BuildQuantMXGroupedValidShape(inputValidShape);
        exp.GetStorage()->UpdateDynValidShape(groupedValidShape);
        maxScratch.GetStorage()->UpdateDynValidShape(groupedValidShape);
        scalingScratch.GetStorage()->UpdateDynValidShape(inputValidShape);
        scaleValidShape = BuildQuantMXScaleValidShape(inputValidShape);
    }

    auto& op = Program::GetInstance().GetCurrentFunction()->AddOperation(
        Opcode::OP_QUANT_MX, {input.GetStorage()},
        {quantized.GetStorage(), exp.GetStorage(), maxScratch.GetStorage(), scalingScratch.GetStorage()});
    op.SetAttribute(OpAttributeKey::mxQuantMode, static_cast<int64_t>(mode));
    op.SetAttribute(OpAttributeKey::mxQuantAxis, normalizedAxis);
    op.SetAttribute(OpAttributeKey::mxQuantPerformanceMode, static_cast<int64_t>(performanceMode ? 1 : 0));
    const auto oldVecTile = TileShape::Current().GetVecTile();
    if (performanceMode && !oldVecTile.tile.empty()) {
        TileShape::Current().SetVecTile(BuildQuantMXPerformanceVecTile(oldVecTile.tile));
    }
    auto scale = Reshape(exp, scaleShape, scaleValidShape);
    if (performanceMode && !oldVecTile.tile.empty()) {
        TileShape::Current().SetVecTile(oldVecTile);
    }
    return std::tie(quantized, scale);
}

REGISTER_OPERATION_TILED_FUNC(QuantMX, Opcode::OP_QUANT_MX, QuantMXTileFunc);
} // namespace npu::tile_fwk
