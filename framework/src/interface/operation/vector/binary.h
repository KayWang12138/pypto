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
#include "interface/configs/config_manager.h"
#include "tensor_transformation.h"

namespace npu::tile_fwk {

enum class BinaryOpType {
    ADD,
    SUB,
    MUL,
    DIV,
    MAX,
    MIN,
    ADD_BRC,
    SUB_BRC,
    MUL_BRC,
    DIV_BRC,
    MAX_BRC,
    MIN_BRC,
    S_ADD,
    S_SUB,
    S_MUL,
    S_DIV,
    S_MAX,
    S_MIN,
    MAXIMUM,
    MINIMUM,
    CMP,
};

template <BinaryOpType T>
std::string GetBinaryOpName() {
    switch (T) {
        case BinaryOpType::ADD: return "ADD";
        case BinaryOpType::SUB: return "SUB";
        case BinaryOpType::MUL: return "MUL";
        case BinaryOpType::DIV: return "DIV";
        case BinaryOpType::MAX: return "MAX";
        case BinaryOpType::MIN: return "MIN";
        case BinaryOpType::MAXIMUM: return "MAXIMUM";
        case BinaryOpType::MINIMUM: return "MINIMUM";
        default: ASSERT(false && "unknown binary op type"); return "";
    }
}

template <BinaryOpType T, bool WithElement = false, bool WithBrc = false>
Opcode GetBinaryOpNameCode() {
    if constexpr (WithElement) {
#define CASE(X) \
    case BinaryOpType::X: return Opcode::OP_##X##S
        switch (T) {
            CASE(ADD);
            CASE(SUB);
            CASE(MUL);
            CASE(DIV);
            CASE(MAX);
            CASE(MIN);
            CASE(S_ADD);
            CASE(S_SUB);
            CASE(S_MUL);
            CASE(S_DIV);
            CASE(S_MAX);
            CASE(S_MIN);
            default: ASSERT(false && "unknown binary op type");
        }
#undef CASE
    }

    if constexpr (WithBrc) {
#define CASE(X) \
    case BinaryOpType::X: return Opcode::OP_##X##_BRC
        switch (T) {
            CASE(ADD);
            CASE(SUB);
            CASE(MUL);
            CASE(DIV);
            CASE(MAX);
            CASE(MIN);
            default: ASSERT(false && "unknown binary op type");
        }
#undef CASE
    }

#define CASE(X) \
    case BinaryOpType::X: return Opcode::OP_##X
    switch (T) {
        CASE(ADD);
        CASE(SUB);
        CASE(MUL);
        CASE(DIV);
        CASE(S_ADD);
        CASE(S_SUB);
        CASE(S_MUL);
        CASE(S_DIV);
        CASE(S_MAX);
        CASE(S_MIN);
        CASE(MAXIMUM);
        CASE(MINIMUM);
        default: ASSERT(false && "unknown binary op type");
    }
#undef CASE
}

std::vector<int64_t> BinaryOperationResultShape(LogicalTensorPtr operand1, LogicalTensorPtr operand2);
LogicalTensorPtr BinaryOperationBroadCast(const LogicalTensorPtr &operand, const std::vector<int> &broadCastShape);

inline void CheckOperandsValid(const Tensor &operand1, const Tensor &operand2) {
    ASSERT(operand1.GetShape().size() == operand2.GetShape().size());
    ASSERT(operand1.GetShape().size() == operand1.GetStorage()->offset.size());
    ASSERT(operand2.GetShape().size() == operand2.GetStorage()->offset.size());
}

inline void CheckBinOpOperandsValid(const LogicalTensorPtr &operand1, const LogicalTensorPtr &operand2) {
    CheckOperandsValid(operand1, operand2);
    for (size_t i = 0; i < operand1->shape.size(); ++i) {
        if (operand1->shape[i] != operand2->shape[i] && (operand1->shape[i] != 1 && operand2->shape[i] != 1)) {
            ASSERT(false && "shape not support binary operation");
        }
    }
}

inline void CheckBinaryInputTensors(const LogicalTensorPtr &tensor1, const LogicalTensorPtr &tensor2, std::string &op) {
    CheckTensorShape(tensor1, op);
    CheckTensorShape(tensor2, op);
    CheckBinOpOperandsValid(tensor1, tensor2);
    if (tensor1->Datatype() != tensor2->Datatype()) {
        ASSERT(false && "The dtype of input tensors are not same.");
    }
    if (tensor1->Format() != tensor2->Format()) {
        ASSERT(false && "The format of input tensors are not same.");
    }
}

inline void BinaryOperationOperandCheck(
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand) {
    constexpr size_t inOpSize = 2;
    constexpr size_t outOpSize = 1;
    ASSERT(iOperand.size() == inOpSize && "iOperand size should be 2");
    ASSERT(oOperand.size() == outOpSize && "oOperand size should be 1");
}

// [m,n] + [m, 1]
inline bool CallBrcBinOp(LogicalTensorPtr operand1, LogicalTensorPtr operand2) {
    ASSERT(operand1->shape.size() == operand2->shape.size() && "Dims not match");
    size_t shapeSize = operand1->shape.size();
    for (size_t i = 0; i < shapeSize - 1; ++i) {
        if (operand1->shape[i] != operand2->shape[i]) {
            return false;
        }
    }

    return (operand1->shape[shapeSize - 1] != 1) && (operand2->shape[shapeSize - 1] == 1);
}

template <BinaryOpType T>
void TiledBinaryOperation(Function &function, const TileShape &tileShape, size_t cur, Input &input1, Input &input2,
    const LogicalTensorPtr &result, TileInfo &resultTileInfo, bool withBrc) {
    if (cur == input1.tensor.GetShape().size()) {
        auto inputTile1 = input1.tensor.GetStorage()->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto inputTile2 = input2.tensor.GetStorage()->View(function, input2.tileInfo.shape, input2.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        if (withBrc) {
            std::vector<int64_t> tmpShape(input1.tileInfo.shape);
            tmpShape[input1.tileInfo.shape.size() - 1] = BLOCK_SIZE / BytesOf(input2.tensor.GetDataType());
            auto tempTensor =
                std::make_shared<LogicalTensor>(function, input2.tensor.GetStorage()->Datatype(), tmpShape);
            function.AddOperation(
                GetBinaryOpNameCode<T, false, true>(), {inputTile1, inputTile2}, {resultTile, tempTensor});
        } else {
            function.AddOperation(GetBinaryOpNameCode<T, false, false>(), {inputTile1, inputTile2}, {resultTile});
        }
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.offset[cur] = i % input1.tensor.GetShape()[cur];
        input1.tileInfo.shape[cur] =
            std::min(input1.tensor.GetShape()[cur] - input1.tileInfo.offset[cur], vecTile[cur]);
        input2.tileInfo.offset[cur] = i % input2.tensor.GetShape()[cur];
        input2.tileInfo.shape[cur] =
            std::min(input2.tensor.GetShape()[cur] - input2.tileInfo.offset[cur], vecTile[cur]);
        TiledBinaryOperation<T>(function, tileShape, cur + 1, input1, input2, result, resultTileInfo, withBrc);
    }
}

template <BinaryOpType T>
void TiledBinaryOperation(Function &function, const TileShape &tileShape, LogicalTensorPtr operand1,
    LogicalTensorPtr operand2, const LogicalTensorPtr &result) {
    CheckBinOpOperandsValid(operand1, operand2);
    bool withBrc =
        CallBrcBinOp(operand1, operand2) && ConfigManager::Instance().GetOperationConfig("FORCE_COMBINE_AXIS", false);
    // nolast brc will be inline
    if (!withBrc) {
        if (operand1->shape != result->shape) {
            auto targetShape = result->shape;
            auto tmp = std::make_shared<LogicalTensor>(function, operand1->Datatype(), targetShape);
            Expand(function, tileShape, operand1, {operand2}, tmp);
            operand1 = tmp;
        }

        if (operand2->shape != result->shape) {
            auto targetShape = result->shape;
            auto tmp = std::make_shared<LogicalTensor>(function, operand2->Datatype(), targetShape);
            Expand(function, tileShape, operand2, {operand1}, tmp);
            operand2 = tmp;
        }
    }

    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo tileInfo2(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = Input{operand1, tileInfo1};
    auto input2 = Input{operand2, tileInfo2};
    TiledBinaryOperation<T>(function, tileShape, 0, input1, input2, result, resultTileInfo, withBrc);
}

// OP_ADD OP_SUB OP_MUL OP_DIV OP_MAX
template <BinaryOpType T>
void BinaryOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    BinaryOperationOperandCheck(iOperand, oOperand);
    TiledBinaryOperation<T>(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
}

// OP_ADD OP_SUB OP_MUL OP_DIV OP_MAX
template <BinaryOpType T>
LogicalTensorPtr TensorBinaryOperation(Function &function, const Tensor &operand1, const Tensor &operand2) {
    auto oprandT1 = operand1.GetStorage();
    auto oprandT2 = operand2.GetStorage();
    if (oprandT1->shape.size() != oprandT2->shape.size()) {
        std::vector<int> broadCastShape = GetBroadCastShape(oprandT1, oprandT2);
        oprandT1 = BinaryOperationBroadCast(oprandT1, broadCastShape);
        oprandT2 = BinaryOperationBroadCast(oprandT2, broadCastShape);
    }
    auto opName = GetBinaryOpName<T>();
    CheckBinaryInputTensors(oprandT1, oprandT2, opName);

    std::vector<SymbolicScalar> resultValidShape;
    std::vector<int64_t> resultShape = BinaryOperationResultShape(oprandT1, oprandT2);
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
    function.AddOperation(GetBinaryOpNameCode<T>(), {oprandT1, oprandT2}, {result});
    return result;
}

template <BinaryOpType T>
void TiledBinaryOperationScalar(Function &function, const TileShape &tileShape, size_t cur, Input &input1,
    Element &value, const LogicalTensorPtr &result, TileInfo &resultTileInfo, bool reverseOperand) {
    if (cur == input1.tensor.GetShape().size()) {
        auto inputTile1 = input1.tensor.GetStorage()->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        // 确认接口
        auto &op = function.AddOperation(GetBinaryOpNameCode<T, true>(), {inputTile1}, {resultTile});
        op.SetAttribute(OpAttributeKey::scalar, value);
        op.SetAttribute(OP_ATTR_PREFIX + "reverseOperand", reverseOperand);
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.offset[cur] = i % input1.tensor.GetShape()[cur];
        input1.tileInfo.shape[cur] =
            std::min(input1.tensor.GetShape()[cur] - input1.tileInfo.offset[cur], vecTile[cur]);

        TiledBinaryOperationScalar<T>(
            function, tileShape, cur + 1, input1, value, result, resultTileInfo, reverseOperand);
    }
}

template <BinaryOpType T>
void TiledBinaryOperationScalar(Function &function, const TileShape &tileShape, LogicalTensorPtr operand1,
    Element value, const LogicalTensorPtr &result, bool reverseOperand = false) {
    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = Input{operand1, tileInfo1};
    TiledBinaryOperationScalar<T>(function, tileShape, 0, input1, value, result, resultTileInfo, reverseOperand);
}

// OP_ADDS OP_SUBS OP_MULS OP_DIVS OP_MAXS OP_MINS
template <BinaryOpType T>
void BinaryOperationScalarTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    TiledBinaryOperationScalar<T>(
        function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar), oOperand[0]);
}

// OP_ADDS OP_SUBS OP_MULS OP_DIVS OP_MAXS OP_MINS
template <BinaryOpType T>
LogicalTensorPtr TensorBinaryOperationScalar(Function &function, LogicalTensorPtr operand1, const Element &value) {
    auto opName = GetBinaryOpName<T>();
    CheckTensorShape(operand1, opName);
    auto result =
        std::make_shared<LogicalTensor>(function, operand1->Datatype(), operand1->shape, operand1->GetDynValidShape());
    auto &op = function.AddOperation(GetBinaryOpNameCode<T, true>(), {operand1}, {result});
    op.SetAttribute(OpAttributeKey::scalar, value);
    return result;
}

template <BinaryOpType T>
void TiledBinaryOperationAllScalar(Function &function, const TileShape &tileShape, size_t cur, Input &input1,
    Element &value, const LogicalTensorPtr &result, TileInfo &resultTileInfo, bool reverseOperand) {
    if (cur == input1.tensor.GetShape().size()) {
        auto inputTile1 = input1.tensor.GetStorage()->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        // 确认接口
        auto &op = function.AddOperation(GetBinaryOpNameCode<T, true>(), {inputTile1}, {resultTile});
        op.SetAttribute(OpAttributeKey::scalar, value);
        op.SetAttribute(OP_ATTR_PREFIX + "reverseOperand", reverseOperand);
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.offset[cur] = i % input1.tensor.GetShape()[cur];
        input1.tileInfo.shape[cur] =
            std::min(input1.tensor.GetShape()[cur] - input1.tileInfo.offset[cur], vecTile[cur]);

        TiledBinaryOperationScalar<T>(
            function, tileShape, cur + 1, input1, value, result, resultTileInfo, reverseOperand);
    }
}

template <BinaryOpType T>
void TiledBinaryOperationAllScalar(Function &function, const TileShape &tileShape, LogicalTensorPtr operand1,
    Element value, const LogicalTensorPtr &result, bool reverseOperand) {
    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = Input{operand1, tileInfo1};
    TiledBinaryOperationAllScalar<T>(function, tileShape, 0, input1, value, result, resultTileInfo, reverseOperand);
}

// OP_S_ADDS OP_S_SUBS OP_S_MULS OP_S_DIVS OP_S_MAXS
template <BinaryOpType T>
void BinaryOperationAllScalarResTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    TiledBinaryOperationAllScalar<T>(function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar),
        oOperand[0], op.GetBoolAttribute(OP_ATTR_PREFIX + "reverseOperand"));
}

// OP_S_ADDS OP_S_SUBS OP_S_MULS OP_S_DIVS OP_S_MAXS
template <BinaryOpType T>
LogicalTensorPtr TensorBinaryOperationAllScalar(
    Function &function, const Tensor &operand1, const Element &value, bool reverseOperand) {
    auto result = std::make_shared<LogicalTensor>(function, operand1.GetStorage()->Datatype(), operand1.GetShape());
    auto &op = function.AddOperation(GetBinaryOpNameCode<T, true>(), {operand1.GetStorage()}, {result});
    op.SetAttribute(OpAttributeKey::scalar, value);
    op.SetAttribute(OP_ATTR_PREFIX + "reverseOperand", reverseOperand);
    return result;
}

template <BinaryOpType T>
void TiledBinaryOperationAllScalar(Function &function, const TileShape &tileShape, size_t cur, Input &input1,
    Input &input2, const LogicalTensorPtr &result, TileInfo &resultTileInfo) {
    if (cur == input1.tensor.GetShape().size()) {
        auto inputTile1 = input1.tensor.GetStorage()->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto inputTile2 = input2.tensor.GetStorage()->View(function, input2.tileInfo.shape, input2.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        function.AddOperation(GetBinaryOpNameCode<T, false>(), {inputTile1, inputTile2}, {resultTile});
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.offset[cur] = i % input1.tensor.GetShape()[cur];
        input1.tileInfo.shape[cur] =
            std::min(input1.tensor.GetShape()[cur] - input1.tileInfo.offset[cur], vecTile[cur]);
        input2.tileInfo.offset[cur] = i % input2.tensor.GetShape()[cur];
        input2.tileInfo.shape[cur] =
            std::min(input2.tensor.GetShape()[cur] - input2.tileInfo.offset[cur], vecTile[cur]);
        TiledBinaryOperationAllScalar<T>(function, tileShape, cur + 1, input1, input2, result, resultTileInfo);
    }
}

template <BinaryOpType T>
void TiledBinaryOperationAllScalar(Function &function, const TileShape &tileShape, LogicalTensorPtr operand1,
    LogicalTensorPtr operand2, const LogicalTensorPtr &result) {
    CheckBinOpOperandsValid(operand1, operand2);

    if (operand1->shape != result->shape) {
        auto targetShape = result->shape;
        auto tmp = std::make_shared<LogicalTensor>(function, operand1->Datatype(), targetShape);
        Expand(function, tileShape, operand1, {operand2}, tmp);
        operand1 = tmp;
    }

    if (operand2->shape != result->shape) {
        auto targetShape = result->shape;
        auto tmp = std::make_shared<LogicalTensor>(function, operand2->Datatype(), targetShape);
        Expand(function, tileShape, operand2, {operand1}, tmp);
        operand2 = tmp;
    }

    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo tileInfo2(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = Input{operand1, tileInfo1};
    auto input2 = Input{operand2, tileInfo2};
    TiledBinaryOperationAllScalar<T>(function, tileShape, 0, input1, input2, result, resultTileInfo);
}

// OP_S_ADD OP_S_SUB OP_S_MUL OP_S_DIV OP_S_MAX
template <BinaryOpType T>
void BinaryOperationAllScalarTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    BinaryOperationOperandCheck(iOperand, oOperand);
    TiledBinaryOperationAllScalar<T>(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
}

// OP_S_ADD OP_S_SUB OP_S_MUL OP_S_DIV OP_S_MAX
template <BinaryOpType T>
LogicalTensorPtr TensorBinaryOperationAllScalar(Function &function, const Tensor &operand1, const Tensor &operand2) {
    auto opName = GetBinaryOpName<T>();
    CheckBinaryInputTensors(operand1.GetStorage(), operand2.GetStorage(), opName);
    auto result = std::make_shared<LogicalTensor>(function, operand1.GetStorage()->Datatype(), operand1.GetShape());
    function.AddOperation(GetBinaryOpNameCode<T, false>(), {operand1.GetStorage(), operand2.GetStorage()}, {result});
    return result;
}

} // namespace npu::tile_fwk
