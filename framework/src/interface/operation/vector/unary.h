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
 * \file unary.h
 * \brief
 */

#pragma once
#include <string>
#include "interface/utils/common.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"

namespace npu::tile_fwk {

enum class UnaryOpType {
    EXP,
    RSQRT,
    SQRT,
    RECIPROCAL,
    DUPLICATE,
    ABS,
    LN,
};

template <UnaryOpType T>
std::string GetUnaryOpName() {
    switch (T) {
        case UnaryOpType::EXP: return "EXP";
        case UnaryOpType::RSQRT: return "RSQRT";
        case UnaryOpType::SQRT: return "SQRT";
        case UnaryOpType::RECIPROCAL: return "RECIPROCAL";
        case UnaryOpType::DUPLICATE: return "DUPLICATE";
        case UnaryOpType::ABS: return "ABS";
        case UnaryOpType::LN: return "LN";
        default: ASSERT(false && "unknown unary op type"); return "";
    }
}

template <UnaryOpType T>
Opcode GetUnaryOpNameCode() {
#define CASE(X) \
    case UnaryOpType::X: return Opcode::OP_##X
    switch (T) {
        CASE(EXP);
        CASE(RSQRT);
        CASE(SQRT);
        CASE(RECIPROCAL);
        CASE(DUPLICATE);
        CASE(ABS);
        CASE(LN);
        default: ASSERT(false && "unknown unary op type");
    }
#undef CASE
}

inline void UnaryOperationOperandCheck(const std::vector<LogicalTensorPtr> &iOperand,
    const std::vector<LogicalTensorPtr> &oOperand) {
    ASSERT(iOperand.size() == 1);
    ASSERT(oOperand.size() == 1);
}

template <UnaryOpType T>
void TiledUnaryOperation(
    Function &function, const TileShape &tileShape, size_t cur, Input &input, const LogicalTensorPtr &result) {
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);
        function.AddOperation(GetUnaryOpNameCode<T>(), {tile}, {resultTile});
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledUnaryOperation<T>(function, tileShape, cur + 1, input, result);
    }
}

template <UnaryOpType T>
void TiledUnaryOperation(
    Function &function, const TileShape &tileShape, const LogicalTensorPtr &operand, const LogicalTensorPtr &result) {
    ASSERT(operand->shape.size() == operand->offset.size());

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};
    TiledUnaryOperation<T>(function, tileShape, 0, input, result);
}

template <UnaryOpType T>
LogicalTensorPtr TensorUnaryOperation(Function &function, LogicalTensorPtr operand) {
    auto opName = GetUnaryOpName<T>();
    CheckTensorShape(operand, opName);
    auto result = std::make_shared<LogicalTensor>(
        function, operand->tensor->datatype, operand->shape, operand->GetDynValidShape());
    function.AddOperation(GetUnaryOpNameCode<T>(), {operand}, {result});
    return result;
}

} // namespace npu::tile_fwk
