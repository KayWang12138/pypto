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
 * \file basic.cpp
 * \brief
 */

#include "where.h"
#include "binary.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"

namespace npu::tile_fwk {

LogicalTensorPtr TensorWhereOperation(
    Function &function, const Tensor &condition, const Tensor &input, const Tensor &other) {
    DECLARE_TRACER();
    ASSERT(condition.GetShape().size() == condition.GetStorage()->offset.size());
    ASSERT(input.GetShape().size() == input.GetStorage()->offset.size());
    ASSERT(other.GetShape().size() == other.GetStorage()->offset.size());
    auto conditionT0 = condition.GetStorage();
    auto inputT1 = input.GetStorage();
    auto otherT2 = other.GetStorage();
    std::vector<int> broadCastShape;
    if (inputT1->shape.size() != otherT2->shape.size()) {
        broadCastShape = GetBroadCastShape(inputT1, otherT2);
        inputT1 = BinaryOperationBroadCast(inputT1, broadCastShape);
    }
    broadCastShape = GetBroadCastShape(conditionT0, inputT1);
    conditionT0 = BinaryOperationBroadCast(conditionT0, broadCastShape);
    inputT1 = BinaryOperationBroadCast(inputT1, broadCastShape);
    otherT2 = BinaryOperationBroadCast(otherT2, broadCastShape);
    CheckBinOpOperandsValid(inputT1, otherT2);
    std::vector<int64_t> resultShape = BinaryOperationResultShape(inputT1, otherT2);
    auto result = std::make_shared<LogicalTensor>(function, input.GetStorage()->Datatype(), resultShape);
    GraphUtils::AddDynOperation(function, Opcode::OP_WHERE_TT, {conditionT0, inputT1, otherT2}, {result});
    return result;
}

LogicalTensorPtr TensorWhereOperation(
    Function &function, const Tensor &condition, const Tensor &input, const Element &other) {
    ASSERT(condition.GetShape().size() == condition.GetStorage()->offset.size());
    ASSERT(input.GetShape().size() == input.GetStorage()->offset.size());
    auto conditionT0 = condition.GetStorage();
    auto inputT1 = input.GetStorage();
    std::vector<int> broadCastShape;

    if (condition.GetStorage()->Datatype() == DT_BOOL) {
        broadCastShape = GetBroadCastShape(conditionT0, inputT1);
        conditionT0 = BinaryOperationBroadCast(conditionT0, broadCastShape);
    } else {
        ALOG_ERROR_F("condition Datatype must be bool.");
    }
    inputT1 = BinaryOperationBroadCast(inputT1, broadCastShape);
    std::vector<int64_t> resultShape = BinaryOperationResultShape(inputT1, inputT1);
    auto result = std::make_shared<LogicalTensor>(function, inputT1->Datatype(), resultShape);
    auto &op = GraphUtils::AddDynOperation(function, Opcode::OP_WHERE_TS, {conditionT0, inputT1}, {result});
    op.SetAttribute(OpAttributeKey::scalar, other);
    return result;
}

LogicalTensorPtr TensorWhereOperation(
    Function &function, const Tensor &condition, const Element &input, const Tensor &other) {
    ASSERT(condition.GetShape().size() == condition.GetStorage()->offset.size());
    ASSERT(other.GetShape().size() == other.GetStorage()->offset.size());
    auto conditionT0 = condition.GetStorage();
    auto otherT1 = other.GetStorage();
    std::vector<int> broadCastShape;

    if (condition.GetStorage()->Datatype() == DT_BOOL) {
        broadCastShape = GetBroadCastShape(conditionT0, otherT1);
        conditionT0 = BinaryOperationBroadCast(conditionT0, broadCastShape);
    } else {
        ALOG_ERROR_F("condition Datatype must be bool.");
    }
    otherT1 = BinaryOperationBroadCast(otherT1, broadCastShape);
    std::vector<int64_t> resultShape = BinaryOperationResultShape(otherT1, otherT1);
    auto result = std::make_shared<LogicalTensor>(function, otherT1->Datatype(), resultShape);
    auto &op = GraphUtils::AddDynOperation(function, Opcode::OP_WHERE_ST, {conditionT0, otherT1}, {result});
    op.SetAttribute(OpAttributeKey::scalar, input);
    return result;
}

LogicalTensorPtr TensorWhereOperation(
    Function &function, const Tensor &condition, const Element &input, const Element &other) {
    ASSERT(condition.GetShape().size() == condition.GetStorage()->offset.size());
    auto conditionT0 = condition.GetStorage();
    std::vector<int64_t> resultShape = BinaryOperationResultShape(conditionT0, conditionT0);
    auto result = std::make_shared<LogicalTensor>(function, input.GetDataType(), resultShape);
    auto &op = GraphUtils::AddDynOperation(function, Opcode::OP_WHERE_SS, {conditionT0}, {result});
    op.SetAttribute(OpAttributeKey::scalar, input);
    op.SetAttribute(OpAttributeKey::dynScalar, other);
    return result;
}

Tensor Where(const Tensor &condition, const Tensor &input, const Tensor &other) {
    DECLARE_TRACER();
    RETURN_CALL(WhereOperation, *Program::GetInstance().GetCurrentFunction(), condition, input, other);
}

Tensor Where(const Tensor &condition, const Tensor &input, const Element &otherValue) {
    DECLARE_TRACER();
    RETURN_CALL(WhereOperation, *Program::GetInstance().GetCurrentFunction(), condition, input, otherValue);
}

Tensor Where(const Tensor &condition, const Element &inputValue, const Tensor &other) {
    DECLARE_TRACER();
    RETURN_CALL(WhereOperation, *Program::GetInstance().GetCurrentFunction(), condition, inputValue, other);
}

Tensor Where(const Tensor &condition, const Element &inputValue, const Element &otherValue) {
    DECLARE_TRACER();
    RETURN_CALL(WhereOperation, *Program::GetInstance().GetCurrentFunction(), condition, inputValue, otherValue);
}

void WhereOperationTileFuncTT(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    TiledWhereOperation(function, tileShape, iOperand[0], iOperand[1], iOperand[2], oOperand[0]);
}

void WhereOperationTileFuncTS(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    TiledWhereOperation(
        function, tileShape, iOperand[0], iOperand[1], op.GetElementAttribute(OpAttributeKey::scalar), oOperand[0]);
}

void WhereOperationTileFuncST(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    TiledWhereOperation(
        function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar), iOperand[1], oOperand[0]);
}

void WhereOperationTileFuncSS(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    TiledWhereOperation(function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar),
        op.GetElementAttribute(OpAttributeKey::dynScalar), oOperand[0]);
}

REGISTER_OPERATION_TILED_FUNC(OP_WHERE_TT, Opcode::OP_WHERE_TT, WhereOperationTileFuncTT);
REGISTER_OPERATION_TILED_FUNC(OP_WHERE_TS, Opcode::OP_WHERE_TS, WhereOperationTileFuncTS);
REGISTER_OPERATION_TILED_FUNC(OP_WHERE_ST, Opcode::OP_WHERE_ST, WhereOperationTileFuncST);
REGISTER_OPERATION_TILED_FUNC(OP_WHERE_SS, Opcode::OP_WHERE_SS, WhereOperationTileFuncSS);

} // namespace npu::tile_fwk