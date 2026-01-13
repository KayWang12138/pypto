/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bitwise_shift.cpp
 * \brief
 */

#include "binary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "interface/utils/common.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {

void CheckBitwiseShiftDtype(const DataType &selfType, const DataType &otherType) {
    std::vector<DataType> BITWISRSHIFT_SUPPORT_DATATYPES = {DataType::DT_INT16, DataType::DT_INT32};
    bool selfSupport = (std::find(BITWISRSHIFT_SUPPORT_DATATYPES.begin(), BITWISRSHIFT_SUPPORT_DATATYPES.end(), selfType) != 
           BITWISRSHIFT_SUPPORT_DATATYPES.end());
    bool otherSupport = (std::find(BITWISRSHIFT_SUPPORT_DATATYPES.begin(), BITWISRSHIFT_SUPPORT_DATATYPES.end(), otherType) != 
           BITWISRSHIFT_SUPPORT_DATATYPES.end());
    ASSERT(selfSupport && otherSupport) << "Inputs datatype not supported";
}
struct LogicalInput {
    const LogicalTensorPtr tensor;
    TileInfo tileInfo;
};

template <BinaryOpType T>
void TiledBitwiseShiftOperation(Function &function, const TileShape &tileShape, size_t cur, LogicalInput &input1,
    LogicalInput &input2, const LogicalTensorPtr &result, TileInfo &resultTileInfo) {
    if (cur == input1.tensor->GetShape().size()) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto inputTile2 = input2.tensor->View(function, input2.tileInfo.shape, input2.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        
        Shape tmpShape = (resultTileInfo.shape.size() > NUM2) ?
                    Shape(resultTileInfo.shape.end() - NUM2, resultTileInfo.shape.end()) : resultTileInfo.shape;
        auto alignSize = BLOCK_SIZE / static_cast<int64_t>(BytesOf(result->Datatype()));
        tmpShape[tmpShape.size() - 1] = AlignUp(tmpShape[tmpShape.size() - 1], alignSize);
        auto tempTensor = std::make_shared<LogicalTensor>(function, result->Datatype(), tmpShape);
        function.AddOperation(GetBinaryOpNameCode<T, false, false>(), {inputTile1, inputTile2}, {resultTile, tempTensor});
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        input1.tileInfo.offset[cur] = i % input1.tensor->GetShape()[cur];
        input2.tileInfo.offset[cur] = i % input2.tensor->GetShape()[cur];
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.shape[cur] = std::min(input1.tensor->GetShape()[cur] - input1.tileInfo.offset[cur], vecTile[cur]);
        input2.tileInfo.shape[cur] = std::min(input2.tensor->GetShape()[cur] - input2.tileInfo.offset[cur], vecTile[cur]);
        TiledBitwiseShiftOperation<T>(function, tileShape, cur + 1, input1, input2, result, resultTileInfo);
    }
}


template <BinaryOpType T>
void TiledBitwiseShiftOperation(Function &function, const TileShape &tileShape, LogicalTensorPtr operand1, LogicalTensorPtr operand2, const LogicalTensorPtr &result) {
    CheckBinOpOperandsValid(operand1, operand2);
    BroadcastOperandTensor(operand1, operand2, result, function, tileShape);
    BroadcastOperandTensor(operand2, operand1, result, function, tileShape);

    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo tileInfo2(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = LogicalInput{operand1, tileInfo1};
    auto input2 = LogicalInput{operand2, tileInfo2};
    // 如果使能了Combine Axis逻辑，需要将withbrc置为false，避免后续走OP_XX_BRC逻辑
    TiledBitwiseShiftOperation<T>(function, tileShape, 0, input1, input2, result, resultTileInfo);
}

template <BinaryOpType T>
void TiledBitwiseShiftOperationScalar(Function &function, const TileShape &tileShape, size_t cur, LogicalInput &input1,
    Element &value, const LogicalTensorPtr &result, TileInfo &resultTileInfo, bool reverseOperand) {
    if (cur == input1.tensor->GetShape().size()) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        // 确认接口
        auto &op = function.AddOperation(GetBinaryOpNameCode<T, true, false>(), {inputTile1}, {resultTile});
        op.SetAttribute(OpAttributeKey::scalar, value);
        op.SetAttribute(OP_ATTR_PREFIX + "reverseOperand", reverseOperand);
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.offset[cur] = i % input1.tensor->GetShape()[cur];
        input1.tileInfo.shape[cur] =
            std::min(input1.tensor->GetShape()[cur] - input1.tileInfo.offset[cur], vecTile[cur]);

        TiledBitwiseShiftOperationScalar<T>(
            function, tileShape, cur + 1, input1, value, result, resultTileInfo, reverseOperand);
    }
}

template <BinaryOpType T>
void TiledBitwiseShiftOperationScalar(Function &function, const TileShape &tileShape, LogicalTensorPtr operand1,
    Element value, const LogicalTensorPtr &result, bool reverseOperand = false) {
    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = LogicalInput{operand1, tileInfo1};
    TiledBitwiseShiftOperationScalar<T>(function, tileShape, 0, input1, value, result, resultTileInfo, reverseOperand);
}

Tensor BitwiseRightShift(const Tensor &self, const Tensor &other) {
    DECLARE_TRACER();
    CheckBitwiseShiftDtype(self.GetDataType(), other.GetDataType());
    RETURN_CALL(BinaryOperation<BinaryOpType::BITWISERIGHTSHIFT>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor BitwiseRightShift(const Tensor &self, const Element &other) {
    DECLARE_TRACER();
    CheckBitwiseShiftDtype(self.GetDataType(), other.GetDataType());
    Element newOther = other;
    if (self.GetDataType() != other.GetDataType()) {
        newOther = Element(self.GetDataType(), other.Cast<int32_t>());
    }
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::BITWISERIGHTSHIFT>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), newOther);
}

Tensor BitwiseRightShift(const Element &self, const Tensor &other) {
    DECLARE_TRACER();
    CheckBitwiseShiftDtype(self.GetDataType(), other.GetDataType());
    Tensor newSelf = Full(self, other.GetDataType(), other.GetShape(), other.GetStorage()->GetDynValidShape());
    RETURN_CALL(BinaryOperation<BinaryOpType::BITWISERIGHTSHIFT>, *Program::GetInstance().GetCurrentFunction(), newSelf, other);
}
Tensor BitwiseLeftShift(const Tensor &self, const Tensor &other) {
    DECLARE_TRACER();
    CheckBitwiseShiftDtype(self.GetDataType(), other.GetDataType());
    RETURN_CALL(BinaryOperation<BinaryOpType::BITWISELEFTSHIFT>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor BitwiseLeftShift(const Tensor &self, const Element &other) {
    DECLARE_TRACER();
    CheckBitwiseShiftDtype(self.GetDataType(), other.GetDataType());
    Element newOther = other;
    if (self.GetDataType() != other.GetDataType()) {
        newOther = Element(self.GetDataType(), other.Cast<int32_t>());
    }
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::BITWISELEFTSHIFT>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), newOther);
}

Tensor BitwiseLeftShift(const Element &self, const Tensor &other) {
    DECLARE_TRACER();
    CheckBitwiseShiftDtype(self.GetDataType(), other.GetDataType());
    auto newSelf = Full(self, self.GetDataType(), other.GetShape(), other.GetStorage()->GetDynValidShape());
    RETURN_CALL(BinaryOperation<BinaryOpType::BITWISELEFTSHIFT>, *Program::GetInstance().GetCurrentFunction(), newSelf, other);
}

template <BinaryOpType T>
void BitwiseShiftOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    BinaryOperationOperandCheck(iOperand, oOperand);
    TiledBitwiseShiftOperation<T>(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
}

template <BinaryOpType T>
void BitwiseShiftOperationScalarTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    TiledBitwiseShiftOperationScalar<T>(function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar), oOperand[0]);
}

REGISTER_OPERATION_TILED_FUNC(OP_BITWISERIGHTSHIFT, Opcode::OP_BITWISERIGHTSHIFT, BitwiseShiftOperationTileFunc<BinaryOpType::BITWISERIGHTSHIFT>);
REGISTER_OPERATION_TILED_FUNC(OP_BITWISELEFTSHIFT, Opcode::OP_BITWISELEFTSHIFT, BitwiseShiftOperationTileFunc<BinaryOpType::BITWISELEFTSHIFT>);

REGISTER_OPERATION_TILED_FUNC(OP_BITWISERIGHTSHIFTS, Opcode::OP_BITWISERIGHTSHIFTS, BitwiseShiftOperationScalarTileFunc<BinaryOpType::BITWISERIGHTSHIFT>);
REGISTER_OPERATION_TILED_FUNC(OP_BITWISELEFTSHIFTS, Opcode::OP_BITWISELEFTSHIFTS, BitwiseShiftOperationScalarTileFunc<BinaryOpType::BITWISELEFTSHIFT>);

} // namespace npu::tile_fwk