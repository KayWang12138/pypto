/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved
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

void UnaryOperationOperandCheck(
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand);

template <UnaryOpType T>
LogicalTensorPtr TensorUnaryOperation(Function &function, LogicalTensorPtr operand) {
    auto opName = GetUnaryOpName<T>();
    CheckTensorShape(operand, opName);
    auto result = std::make_shared<LogicalTensor>(
        function, operand->tensor->datatype, operand->shape, operand->GetDynValidShape(), operand->Format());
    function.AddOperation(GetUnaryOpNameCode<T>(), {operand}, {result});
    return result;
}

} // namespace npu::tile_fwk
