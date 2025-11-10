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
 * \file binary.cpp
 * \brief
 */

#include "binary.h"
#include "interface/utils/operator_tracer.h"

namespace npu::tile_fwk {

std::vector<int64_t> BinaryOperationResultShape(LogicalTensorPtr operand1, LogicalTensorPtr operand2) {
    std::vector<int64_t> resultShape(operand1->shape.size());
    for (size_t i = 0; i < resultShape.size(); i++) {
        resultShape[i] = std::max(operand1->shape[i], operand2->shape[i]);
    }
    return resultShape;
}

LogicalTensorPtr BinaryOperationBroadCast(const LogicalTensorPtr &operand, const std::vector<int> &broadCastShape) {
    if (operand->shape.size() < broadCastShape.size()) {
        auto broadCastDims = broadCastShape.size() - operand->shape.size();
        std::vector<int64_t> unsqueezeShape(operand->shape);
        unsqueezeShape.insert(unsqueezeShape.begin(), broadCastDims, 1);
        auto tmpOperand = Reshape(operand, unsqueezeShape).GetStorage();
        return tmpOperand;
    }
    return operand;
}

Tensor Add(const Tensor &self, const Tensor &other) {
    DECLARE_TRACER();
    RETURN_CALL(BinaryOperation<BinaryOpType::ADD>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor Sub(const Tensor &self, const Tensor &other) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperation<BinaryOpType::SUB>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor Mul(const Tensor &self, const Tensor &other) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor Div(const Tensor &self, const Tensor &other) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperation<BinaryOpType::DIV>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor Maximum(const Tensor &operand1, const Tensor &operand2) {
    DECLARE_TRACER();

    RETURN_CALL(
        BinaryOperation<BinaryOpType::MAXIMUM>, *Program::GetInstance().GetCurrentFunction(), operand1, operand2);
}

Tensor Minimum(const Tensor &operand1, const Tensor &operand2) {
    DECLARE_TRACER();

    RETURN_CALL(
        BinaryOperation<BinaryOpType::MINIMUM>, *Program::GetInstance().GetCurrentFunction(), operand1, operand2);
}

Tensor Add(const Tensor &self, const Element &other) {
    DECLARE_TRACER();
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::ADD>, *Program::GetInstance().GetCurrentFunction(),
        self.GetStorage(), other);
}

Tensor Sub(const Tensor &self, const Element &other) {
    DECLARE_TRACER();
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::SUB>, *Program::GetInstance().GetCurrentFunction(),
        self.GetStorage(), other);
}

Tensor Mul(const Tensor &self, const Element &other) {
    DECLARE_TRACER();
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(),
        self.GetStorage(), other);
}

Tensor Div(const Tensor &self, const Element &other) {
    DECLARE_TRACER();
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::DIV>, *Program::GetInstance().GetCurrentFunction(),
        self.GetStorage(), other);
}

Tensor MaxS(const Tensor &operand1, const Element &operand2) {
    DECLARE_TRACER();
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::MAX>, *Program::GetInstance().GetCurrentFunction(),
        operand1.GetStorage(), operand2);
}

Tensor MinS(const Tensor &operand1, const Element &operand2) {
    DECLARE_TRACER();
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::MIN>, *Program::GetInstance().GetCurrentFunction(),
        operand1.GetStorage(), operand2);
}

Tensor ScalarAddS(const Tensor &operand, const Element &value, bool reverseOperand) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_ADD>, *Program::GetInstance().GetCurrentFunction(),
        operand.GetStorage(), value, reverseOperand);
}

Tensor ScalarSubS(const Tensor &operand, const Element &value, bool reverseOperand) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_SUB>, *Program::GetInstance().GetCurrentFunction(),
        operand.GetStorage(), value, reverseOperand);
}

Tensor ScalarMulS(const Tensor &operand, const Element &value, bool reverseOperand) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_MUL>, *Program::GetInstance().GetCurrentFunction(),
        operand.GetStorage(), value, reverseOperand);
}

Tensor ScalarDivS(const Tensor &operand, const Element &value, bool reverseOperand) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_DIV>, *Program::GetInstance().GetCurrentFunction(),
        operand.GetStorage(), value, reverseOperand);
}

Tensor ScalarMaxS(const Tensor &operand, const Element &value, bool reverseOperand) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_MAX>, *Program::GetInstance().GetCurrentFunction(),
        operand.GetStorage(), value, reverseOperand);
}

Tensor ScalarAdd(const Tensor &operand1, const Tensor &operand2) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_ADD>, *Program::GetInstance().GetCurrentFunction(),
        operand1.GetStorage(), operand2.GetStorage());
}
Tensor ScalarSub(const Tensor &operand1, const Tensor &operand2) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_SUB>, *Program::GetInstance().GetCurrentFunction(),
        operand1.GetStorage(), operand2.GetStorage());
}

Tensor ScalarMul(const Tensor &operand1, const Tensor &operand2) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_MUL>, *Program::GetInstance().GetCurrentFunction(),
        operand1.GetStorage(), operand2.GetStorage());
}

Tensor ScalarDiv(const Tensor &operand1, const Tensor &operand2) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_DIV>, *Program::GetInstance().GetCurrentFunction(),
        operand1.GetStorage(), operand2.GetStorage());
}

Tensor ScalarMax(const Tensor &operand1, const Tensor &operand2) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_MAX>, *Program::GetInstance().GetCurrentFunction(),
        operand1.GetStorage(), operand2.GetStorage());
}

REGISTER_OPERATION_TILED_FUNC(OP_ADD, Opcode::OP_ADD, BinaryOperationTileFunc<BinaryOpType::ADD>);
REGISTER_OPERATION_TILED_FUNC(OP_SUB, Opcode::OP_SUB, BinaryOperationTileFunc<BinaryOpType::SUB>);
REGISTER_OPERATION_TILED_FUNC(OP_MUL, Opcode::OP_MUL, BinaryOperationTileFunc<BinaryOpType::MUL>);
REGISTER_OPERATION_TILED_FUNC(OP_DIV, Opcode::OP_DIV, BinaryOperationTileFunc<BinaryOpType::DIV>);
REGISTER_OPERATION_TILED_FUNC(OP_MAXIMUM, Opcode::OP_MAXIMUM, BinaryOperationTileFunc<BinaryOpType::MAXIMUM>);
REGISTER_OPERATION_TILED_FUNC(OP_MINIMUM, Opcode::OP_MINIMUM, BinaryOperationTileFunc<BinaryOpType::MINIMUM>);

REGISTER_OPERATION_TILED_FUNC(OP_ADDS, Opcode::OP_ADDS, BinaryOperationScalarTileFunc<BinaryOpType::ADD>);
REGISTER_OPERATION_TILED_FUNC(OP_SUBS, Opcode::OP_SUBS, BinaryOperationScalarTileFunc<BinaryOpType::SUB>);
REGISTER_OPERATION_TILED_FUNC(OP_MULS, Opcode::OP_MULS, BinaryOperationScalarTileFunc<BinaryOpType::MUL>);
REGISTER_OPERATION_TILED_FUNC(OP_DIVS, Opcode::OP_DIVS, BinaryOperationScalarTileFunc<BinaryOpType::DIV>);
REGISTER_OPERATION_TILED_FUNC(OP_MAXS, Opcode::OP_MAXS, BinaryOperationScalarTileFunc<BinaryOpType::MAX>);
REGISTER_OPERATION_TILED_FUNC(OP_MINS, Opcode::OP_MINS, BinaryOperationScalarTileFunc<BinaryOpType::MIN>);

REGISTER_OPERATION_TILED_FUNC(OP_S_ADDS, Opcode::OP_S_ADDS, BinaryOperationAllScalarResTileFunc<BinaryOpType::S_ADD>);
REGISTER_OPERATION_TILED_FUNC(OP_S_SUBS, Opcode::OP_S_SUBS, BinaryOperationAllScalarResTileFunc<BinaryOpType::S_SUB>);
REGISTER_OPERATION_TILED_FUNC(OP_S_MULS, Opcode::OP_S_MULS, BinaryOperationAllScalarResTileFunc<BinaryOpType::S_MUL>);
REGISTER_OPERATION_TILED_FUNC(OP_S_DIVS, Opcode::OP_S_DIVS, BinaryOperationAllScalarResTileFunc<BinaryOpType::S_DIV>);
REGISTER_OPERATION_TILED_FUNC(OP_S_MAXS, Opcode::OP_S_MAXS, BinaryOperationAllScalarResTileFunc<BinaryOpType::S_MAX>);
REGISTER_OPERATION_TILED_FUNC(OP_S_MINS, Opcode::OP_S_MINS, BinaryOperationAllScalarResTileFunc<BinaryOpType::S_MIN>);

REGISTER_OPERATION_TILED_FUNC(OP_S_ADD, Opcode::OP_S_ADD, BinaryOperationAllScalarTileFunc<BinaryOpType::S_ADD>);
REGISTER_OPERATION_TILED_FUNC(OP_S_SUB, Opcode::OP_S_SUB, BinaryOperationAllScalarTileFunc<BinaryOpType::S_SUB>);
REGISTER_OPERATION_TILED_FUNC(OP_S_MUL, Opcode::OP_S_MUL, BinaryOperationAllScalarTileFunc<BinaryOpType::S_MUL>);
REGISTER_OPERATION_TILED_FUNC(OP_S_DIV, Opcode::OP_S_DIV, BinaryOperationAllScalarTileFunc<BinaryOpType::S_DIV>);
REGISTER_OPERATION_TILED_FUNC(OP_S_MAX, Opcode::OP_S_MAX, BinaryOperationAllScalarTileFunc<BinaryOpType::S_MAX>);
REGISTER_OPERATION_TILED_FUNC(OP_S_MIN, Opcode::OP_S_MIN, BinaryOperationAllScalarTileFunc<BinaryOpType::S_MIN>);

} // namespace npu::tile_fwk