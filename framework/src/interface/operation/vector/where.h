/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file where.h
 * \brief
 */

#pragma once
#include <string>
#include "interface/utils/common.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "tensor_transformation.h"

namespace npu::tile_fwk {

template <typename U, typename W>
void TiledWhereOperation(Function &function, const TileShape &tileShape, size_t cur, Input &condition, U &input,
    W &other, const LogicalTensorPtr &result, TileInfo &resultTileInfo) {
    if (cur == result->shape.size()) {
        auto inputDatatype = DT_FP32;
        if constexpr (std::is_same_v<U, Input>) {
            inputDatatype = input.tensor.GetDataType();
        } else if constexpr (std::is_same_v<U, const Element>) {
            inputDatatype = input.GetDataType();
        }
        unsigned COUNT_MAX_BYTE = 4096;
        auto conditionTile =
            condition.tensor.GetStorage()->View(function, condition.tileInfo.shape, condition.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        std::vector<int64_t> castConditionShape({static_cast<int64_t>(COUNT_MAX_BYTE / BytesOf(DT_FP32))});
        auto castConditionTensor = std::make_shared<LogicalTensor>(function, DT_FP16, castConditionShape);
        std::vector<int64_t> compareConditionShape({static_cast<int64_t>(COUNT_MAX_BYTE / BytesOf(DT_FP32))});
        auto compareConditionTensor = std::make_shared<LogicalTensor>(function, DT_FP16, compareConditionShape);
        std::vector<int64_t> vcmpBitResultShape({static_cast<int64_t>(COUNT_MAX_BYTE / BytesOf(DT_FP32) / 8)});
        auto vcmpBitResultTensor = std::make_shared<LogicalTensor>(function, DT_INT8, vcmpBitResultShape);
        std::vector<int64_t> startAddrUBShape({1});
        auto startAddrUBTensor = std::make_shared<LogicalTensor>(function, DT_UINT64, startAddrUBShape);
        std::vector<int64_t> inputTempShape({static_cast<int64_t>(COUNT_MAX_BYTE / BytesOf(DT_FP32))});
        auto inputTempTensor = std::make_shared<LogicalTensor>(function, inputDatatype, inputTempShape);
        std::vector<int64_t> otherTempShape({static_cast<int64_t>(COUNT_MAX_BYTE / BytesOf(DT_FP32))});
        auto otherTempTensor = std::make_shared<LogicalTensor>(function, inputDatatype, otherTempShape);
        if constexpr (std::is_same_v<U, Input> && std::is_same_v<W, Input>) {
            auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
            auto otherTile = other.tensor.GetStorage()->View(function, other.tileInfo.shape, other.tileInfo.offset);
            function.AddOperation(Opcode::OP_WHERE_TT, {conditionTile, inputTile, otherTile},
                {resultTile, castConditionTensor, compareConditionTensor, vcmpBitResultTensor, startAddrUBTensor,
                    inputTempTensor, otherTempTensor});
        } else if constexpr (std::is_same_v<U, Input> && std::is_same_v<W, const Element>) {
            auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
            auto &op = function.AddOperation(Opcode::OP_WHERE_TS, {conditionTile, inputTile},
                {resultTile, castConditionTensor, compareConditionTensor, vcmpBitResultTensor, startAddrUBTensor,
                    inputTempTensor, otherTempTensor});
            op.SetAttribute(OpAttributeKey::scalar, other);
        } else if constexpr (std::is_same_v<U, const Element> && std::is_same_v<W, Input>) {
            auto otherTile = other.tensor.GetStorage()->View(function, other.tileInfo.shape, other.tileInfo.offset);
            auto &op = function.AddOperation(Opcode::OP_WHERE_ST, {conditionTile, otherTile},
                {resultTile, castConditionTensor, compareConditionTensor, vcmpBitResultTensor, startAddrUBTensor,
                    inputTempTensor, otherTempTensor});
            op.SetAttribute(OpAttributeKey::scalar, input);
        } else if constexpr (std::is_same_v<U, const Element> && std::is_same_v<W, const Element>) {
            auto &op = function.AddOperation(Opcode::OP_WHERE_SS, {conditionTile},
                {resultTile, castConditionTensor, compareConditionTensor, vcmpBitResultTensor, startAddrUBTensor,
                    inputTempTensor, otherTempTensor});
            op.SetAttribute(OpAttributeKey::scalar, input);
            op.SetAttribute(OpAttributeKey::dynScalar, other);
        }
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        condition.tileInfo.offset[cur] = i % condition.tensor.GetStorage()->shape[cur];
        condition.tileInfo.shape[cur] =
            std::min(condition.tensor.GetStorage()->shape[cur] - condition.tileInfo.offset[cur], vecTile[cur]);
        if constexpr (std::is_same_v<U, Input>) {
            input.tileInfo.offset[cur] = i % input.tensor.GetStorage()->shape[cur];
            input.tileInfo.shape[cur] =
                std::min(input.tensor.GetStorage()->shape[cur] - input.tileInfo.offset[cur], vecTile[cur]);
        }
        if constexpr (std::is_same_v<W, Input>) {
            other.tileInfo.offset[cur] = i % other.tensor.GetStorage()->shape[cur];
            other.tileInfo.shape[cur] =
                std::min(other.tensor.GetStorage()->shape[cur] - other.tileInfo.offset[cur], vecTile[cur]);
        }
        TiledWhereOperation(function, tileShape, cur + 1, condition, input, other, result, resultTileInfo);
    }
}

template <typename U, typename W>
void TiledWhereOperation(Function &function, const TileShape &tileShape, const LogicalTensorPtr &condition,
    const U &input, const W &other, const LogicalTensorPtr &result) {
    LogicalTensorPtr conditionPtr = condition;
    LogicalTensorPtr inputPtr = nullptr;
    LogicalTensorPtr otherPtr = nullptr;
    if constexpr (std::is_same_v<U, LogicalTensorPtr>) {
        inputPtr = input;
    }
    if constexpr (std::is_same_v<W, LogicalTensorPtr>) {
        otherPtr = other;
    }

    if (condition->shape != result->shape) {
        auto targetShape = result->shape;
        auto tmp = std::make_shared<LogicalTensor>(function, condition->Datatype(), targetShape);
        Expand(function, tileShape, condition, {inputPtr, otherPtr}, tmp);
        conditionPtr = tmp;
    }
    if constexpr (std::is_same_v<U, LogicalTensorPtr>) {
        if (input->shape != result->shape) {
            auto targetShape = result->shape;
            auto tmp = std::make_shared<LogicalTensor>(function, input->Datatype(), targetShape);
            Expand(function, tileShape, inputPtr, {condition, otherPtr}, tmp);
            inputPtr = tmp;
        }
    }
    if constexpr (std::is_same_v<W, LogicalTensorPtr>) {
        if (other->shape != result->shape) {
            auto targetShape = result->shape;
            auto tmp = std::make_shared<LogicalTensor>(function, other->Datatype(), targetShape);
            Expand(function, tileShape, otherPtr, {inputPtr, condition}, tmp);
            otherPtr = tmp;
        }
    }

    TileInfo tileInfoCondition(result->shape.size(), result->offset.size());
    auto inputCondition = Input{conditionPtr, tileInfoCondition};
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    if constexpr (std::is_same_v<U, LogicalTensorPtr> && std::is_same_v<W, LogicalTensorPtr>) {
        TileInfo tileInfoInput(result->shape.size(), result->offset.size());
        TileInfo tileInfoOther(result->shape.size(), result->offset.size());
        auto inputInput = Input{inputPtr, tileInfoInput};
        auto inputOther = Input{otherPtr, tileInfoOther};
        TiledWhereOperation(function, tileShape, 0, inputCondition, inputInput, inputOther, result, resultTileInfo);
    } else if constexpr (std::is_same_v<U, LogicalTensorPtr> && std::is_same_v<W, Element>) {
        TileInfo tileInfoInput(result->shape.size(), result->offset.size());
        auto inputInput = Input{inputPtr, tileInfoInput};
        TiledWhereOperation(function, tileShape, 0, inputCondition, inputInput, other, result, resultTileInfo);
    } else if constexpr (std::is_same_v<U, Element> && std::is_same_v<W, LogicalTensorPtr>) {
        TileInfo tileInfoOther(result->shape.size(), result->offset.size());
        auto inputOther = Input{otherPtr, tileInfoOther};
        TiledWhereOperation(function, tileShape, 0, inputCondition, input, inputOther, result, resultTileInfo);
    } else if constexpr (std::is_same_v<U, Element> && std::is_same_v<W, Element>) {
        TiledWhereOperation(function, tileShape, 0, inputCondition, input, other, result, resultTileInfo);
    }
}

} // namespace npu::tile_fwk
