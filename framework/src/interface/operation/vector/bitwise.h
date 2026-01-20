/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bitwise.h
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

namespace npu::tile_fwk {

enum class BitwiseOpType {
    BITWISEAND,
};

template <BitwiseOpType T>
std::string GetBitwiseOpName() {
    switch (T) {
        case BitwiseOpType::BITWISEAND: return "BITWISEAND";
        default: ASSERT(false && "unknown bitwise op type"); return "";
    }
}

template <BitwiseOpType T, bool WithElement = false, bool WithBrc = false>
Opcode GetBitwiseOpNameCode() {
    if constexpr (WithElement) {
#define CASE(X) \
    case BitwiseOpType::X: return Opcode::OP_##X##S
        switch (T) {
            default: ASSERT(false && "unknown bitwise op type");
        }
#undef CASE
    }

#define CASE(X) \
    case BitwiseOpType::X: return Opcode::OP_##X
    switch (T) {
        CASE(BITWISEAND);
        default: ASSERT(false && "unknown bitwise op type");
    }
#undef CASE
}

std::vector<int64_t> BitwiseOperationResultShape(LogicalTensorPtr operand1, LogicalTensorPtr operand2);
LogicalTensorPtr BitwiseOperationBroadCast(const LogicalTensorPtr &operand, const std::vector<int> &broadCastShape);
void CheckBinOpOperandsValid(const LogicalTensorPtr &operand1, const LogicalTensorPtr &operand2);
void BitwiseOperationOperandCheck(
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand);
void CheckBitwiseInputTensors(const LogicalTensorPtr &tensor1, const LogicalTensorPtr &tensor2, std::string &op);

// OP_BITWISEAND
template <BitwiseOpType T>
LogicalTensorPtr TensorBitwiseOperation(Function &function, const Tensor &operand1, const Tensor &operand2) {
    auto oprandT1 = operand1.GetStorage();
    auto oprandT2 = operand2.GetStorage();
    if (oprandT1->shape.size() != oprandT2->shape.size()) {
        std::vector<int> broadCastShape = GetBroadCastShape(oprandT1, oprandT2);
        oprandT1 = BitwiseOperationBroadCast(oprandT1, broadCastShape);
        oprandT2 = BitwiseOperationBroadCast(oprandT2, broadCastShape);
    }
    auto opName = GetBitwiseOpName<T>();
    CheckBitwiseInputTensors(oprandT1, oprandT2, opName);

    std::vector<SymbolicScalar> resultValidShape;
    std::vector<int64_t> resultShape = BitwiseOperationResultShape(oprandT1, oprandT2);
    if ((!oprandT1->GetDynValidShape().empty()) && (!oprandT2->GetDynValidShape().empty())) {
        for (size_t i = 0; i < resultShape.size(); ++i) {
            if (resultShape[i] == oprandT1->shape[i]) {
                resultValidShape.push_back(operand1.GetStorage()->GetDynValidShape()[i]);
            } else {
                resultValidShape.push_back(operand2.GetStorage()->GetDynValidShape()[i]);
            }
        }
    }
    auto result = std::make_shared<LogicalTensor>(
        function, oprandT1->Datatype(), resultShape, resultValidShape, oprandT1->Format());
    function.AddOperation(GetBitwiseOpNameCode<T>(), {oprandT1, oprandT2}, {result});
    return result;
}

// OP_BITWISEANDS
template <BitwiseOpType T>
LogicalTensorPtr TensorBitwiseOperationScalar(Function &function, LogicalTensorPtr operand1, const Element &value) {
    auto opName = GetBitwiseOpName<T>();
    CheckTensorShape(operand1, opName);
    auto result =
        std::make_shared<LogicalTensor>(function, operand1->Datatype(), operand1->shape, operand1->GetDynValidShape());
    auto &op = function.AddOperation(GetBitwiseOpNameCode<T, true>(), {operand1}, {result});
    op.SetAttribute(OpAttributeKey::scalar, value);
    return result;
}

} // namespace npu::tile_fwk
