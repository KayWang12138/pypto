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
constexpr int64_t QUANT_MX_MIN_RANK = 2;
constexpr int64_t QUANT_MX_MAX_RANK = 4;
constexpr int64_t QUANT_MX_GROUP_COLS = 32;
constexpr int64_t QUANT_MX_TILE_ALIGN_BYTES = 256;

int64_t CeilDiv(int64_t dividend, int64_t divisor) { return (dividend + divisor - 1) / divisor; }

bool IsQuantMXFp4Dtype(DataType dtype)
{
    return dtype == DataType::DT_FP4_E2M1X2 || dtype == DataType::DT_FP4_E1M2X2;
}

void CheckQuantMXDtype(DataType quantDtype)
{
    ASSERT(
        VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED,
        quantDtype == DataType::DT_FP8E4M3 || IsQuantMXFp4Dtype(quantDtype))
        << "QuantMX only supports DT_FP8E4M3 output currently, and reserves DT_FP4_E2M1X2/DT_FP4_E1M2X2 "
           "for future MXFP4 support. Current quant dtype: "
        << DataType2String(quantDtype);
    ASSERT(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, quantDtype == DataType::DT_FP8E4M3)
        << "QuantMX does not support MXFP4 output yet. Current quant dtype: " << DataType2String(quantDtype);
}

void CheckQuantMXInput(const Tensor& input, DataType quantDtype)
{
    const auto inputDtype = input.GetDataType();
    ASSERT(
        VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED,
        inputDtype == DataType::DT_FP16 || inputDtype == DataType::DT_BF16 || inputDtype == DataType::DT_FP32)
        << "QuantMX only supports DT_FP16, DT_BF16, and DT_FP32 input.";
    CheckQuantMXDtype(quantDtype);
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, input.Format() == TileOpFormat::TILEOP_ND)
        << "QuantMX only supports TILEOP_ND input.";
    ASSERT(
        VectorErrorCode::ERR_PARAM_INVALID,
        QUANT_MX_MIN_RANK <= input.GetShape().size() && input.GetShape().size() <= QUANT_MX_MAX_RANK)
        << "QuantMX only supports 2D to 4D input.";
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

std::vector<SymbolicScalar> BuildQuantMXGroupedValidShape(const std::vector<SymbolicScalar>& inputValidShape)
{
    auto groupedValidShape = inputValidShape;
    groupedValidShape.back() = (groupedValidShape.back() + QUANT_MX_GROUP_COLS - 1) / QUANT_MX_GROUP_COLS;
    return groupedValidShape;
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
    const LogicalTensorPtr& exp, const LogicalTensorPtr& maxScratch, const LogicalTensorPtr& scalingScratch)
{
    if (cur == input.tensor.GetShape().size()) {
        const int64_t lastDimBytes = input.tileInfo.shape.back() * BytesOf(input.tensor.GetDataType());
        ASSERT(VectorErrorCode::ERR_PARAM_INVALID, lastDimBytes % QUANT_MX_TILE_ALIGN_BYTES == 0)
            << "QuantMX tile width must be 256-byte aligned. Current last dim bytes: " << lastDimBytes;

        auto groupedTileShape = input.tileInfo.shape;
        groupedTileShape.back() = CeilDiv(groupedTileShape.back(), QUANT_MX_GROUP_COLS);
        auto groupedTileOffset = input.tileInfo.offset;
        groupedTileOffset.back() /= QUANT_MX_GROUP_COLS;

        auto srcTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto dstTile = dst->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto expTile = exp->View(function, groupedTileShape, groupedTileOffset);
        auto maxTile = maxScratch->View(function, groupedTileShape, groupedTileOffset);
        auto scalingTile = scalingScratch->View(function, input.tileInfo.shape, input.tileInfo.offset);
        function.AddOperation(Opcode::OP_QUANT_MX, {srcTile}, {dstTile, expTile, maxTile, scalingTile});
        return;
    }

    const auto& vecTile = tileShape.GetVecTile();
    int64_t step = std::max<int64_t>(1, std::min<int64_t>(vecTile[cur], input.tensor.GetShape()[cur]));

    for (int64_t i = 0; i < input.tensor.GetShape()[cur]; i += step) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, step);
        input.tileInfo.offset[cur] = i;
        TiledQuantMXOperation(function, tileShape, cur + 1, input, dst, exp, maxScratch, scalingScratch);
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
    CheckQuantMXTileShape(src, tileShape.GetVecTile());
    TileInfo inputTileInfo(src->shape.size(), src->offset.size());
    auto input = Input{Tensor(src), inputTileInfo};
    TiledQuantMXOperation(function, tileShape, 0, input, dst, exp, maxScratch, scalingScratch);
}
} // namespace

std::tuple<Tensor, Tensor> QuantMX(const Tensor& input, DataType quantDtype)
{
    DECLARE_TRACER();
    CheckQuantMXInput(input, quantDtype);

    const auto& inputShape = input.GetShape();
    const std::vector<int64_t> groupedShape = BuildQuantMXGroupedShape(inputShape);

    auto quantized = Tensor(quantDtype, inputShape, "", TileOpFormat::TILEOP_ND);
    auto exp = Tensor(DataType::DT_FP8E8M0, groupedShape, "", TileOpFormat::TILEOP_ND);
    auto maxScratch = Tensor(DataType::DT_FP32, groupedShape, "", TileOpFormat::TILEOP_ND);
    auto scalingScratch = Tensor(DataType::DT_FP32, inputShape, "", TileOpFormat::TILEOP_ND);

    const auto& inputValidShape = input.GetStorage()->GetDynValidShape();
    if (!inputValidShape.empty()) {
        quantized.GetStorage()->UpdateDynValidShape(inputValidShape);
        const auto groupedValidShape = BuildQuantMXGroupedValidShape(inputValidShape);
        exp.GetStorage()->UpdateDynValidShape(groupedValidShape);
        maxScratch.GetStorage()->UpdateDynValidShape(groupedValidShape);
        scalingScratch.GetStorage()->UpdateDynValidShape(inputValidShape);
    }

    Program::GetInstance().GetCurrentFunction()->AddOperation(
        Opcode::OP_QUANT_MX, {input.GetStorage()},
        {quantized.GetStorage(), exp.GetStorage(), maxScratch.GetStorage(), scalingScratch.GetStorage()});
    return std::tie(quantized, exp);
}

REGISTER_OPERATION_TILED_FUNC(QuantMX, Opcode::OP_QUANT_MX, QuantMXTileFunc);
} // namespace npu::tile_fwk
