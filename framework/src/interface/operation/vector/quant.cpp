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
constexpr int64_t QUANT_MX_RANK = 2;
constexpr int64_t QUANT_MX_GROUP_COLS = 32;
constexpr int64_t QUANT_MX_INPUT_ALIGN = 64;

void CheckQuantMXInput(const Tensor& input)
{
    ASSERT(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, input.GetDataType() == DataType::DT_FP32)
        << "QuantMX only supports DT_FP32 input.";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, input.Format() == TileOpFormat::TILEOP_ND)
        << "QuantMX only supports TILEOP_ND input.";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, input.GetShape().size() == QUANT_MX_RANK)
        << "QuantMX only supports 2D input.";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, input.GetShape()[1] % QUANT_MX_INPUT_ALIGN == 0)
        << "QuantMX requires the last dimension to be aligned to 64.";
}

std::vector<int64_t> BuildQuantMXGroupedShape(const std::vector<int64_t>& inputShape)
{
    return {inputShape[0], inputShape[1] / QUANT_MX_GROUP_COLS};
}

std::vector<SymbolicScalar> BuildQuantMXGroupedValidShape(const std::vector<SymbolicScalar>& inputValidShape)
{
    return {inputValidShape[0], (inputValidShape[1] + QUANT_MX_GROUP_COLS - 1) / QUANT_MX_GROUP_COLS};
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

    const auto& vecTile = tileShape.GetVecTile();
    const int64_t rowTile = std::max<int64_t>(1, vecTile[0]);
    int64_t colTile = (vecTile[1] / QUANT_MX_INPUT_ALIGN) * QUANT_MX_INPUT_ALIGN;
    if (colTile == 0) {
        colTile = src->shape[1];
    }
    colTile = std::min<int64_t>(colTile, src->shape[1]);

    for (int64_t row = 0; row < src->shape[0]; row += rowTile) {
        const int64_t curRows = std::min<int64_t>(src->shape[0] - row, rowTile);
        for (int64_t col = 0; col < src->shape[1]; col += colTile) {
            const int64_t curCols = std::min<int64_t>(src->shape[1] - col, colTile);
            ASSERT(VectorErrorCode::ERR_PARAM_INVALID, curCols % QUANT_MX_INPUT_ALIGN == 0)
                << "QuantMX tile width must be aligned to 64.";

            auto srcTile = src->View(function, {curRows, curCols}, {row, col});
            auto dstTile = dst->View(function, {curRows, curCols}, {row, col});
            auto expTile =
                exp->View(function, {curRows, curCols / QUANT_MX_GROUP_COLS}, {row, col / QUANT_MX_GROUP_COLS});
            auto maxTile =
                maxScratch->View(function, {curRows, curCols / QUANT_MX_GROUP_COLS}, {row, col / QUANT_MX_GROUP_COLS});
            auto scalingTile = scalingScratch->View(function, {curRows, curCols}, {row, col});
            function.AddOperation(Opcode::OP_QUANT_MX, {srcTile}, {dstTile, expTile, maxTile, scalingTile});
        }
    }
}
} // namespace

std::tuple<Tensor, Tensor> QuantMX(const Tensor& input)
{
    DECLARE_TRACER();
    CheckQuantMXInput(input);

    const auto& inputShape = input.GetShape();
    const std::vector<int64_t> groupedShape = BuildQuantMXGroupedShape(inputShape);

    auto quantized = Tensor(DataType::DT_FP8E4M3, inputShape, "", TileOpFormat::TILEOP_ND);
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
