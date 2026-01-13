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

enum class BitwiseShiftOpType {
    BITWISERIGHTSHIFT,
    BITWISELEFTSHIFT,
};

template <BitwiseShiftOpType T>
std::string GetBitwiseShiftOpName() {
    switch (T) {
        case BitwiseShiftOpType::BITWISERIGHTSHIFT: return "BITWISERIGHTSHIFT";
        case BitwiseShiftOpType::BITWISELEFTSHIFT: return "BITWISELEFTSHIFT";
        default: ASSERT(false && "unknown binary op type"); return "";
    }
}

template <BitwiseShiftOpType T, bool WithElement = false>
Opcode GetBitwiseShiftOpNameCode() {
    if constexpr (WithElement) {
#define CASE(X) \
    case BitwiseShiftOpType::X: return Opcode::OP_##X##S
        switch (T) {
            CASE(BITWISERIGHTSHIFT);
            CASE(BITWISELEFTSHIFT);
            default: ASSERT(false && "unknown binary op type");
        }
#undef CASE
    }

#define CASE(X) \
    case BitwiseShiftOpType::X: return Opcode::OP_##X
    switch (T) {
        CASE(BITWISERIGHTSHIFT);
        CASE(BITWISELEFTSHIFT);
        default: ASSERT(false && "unknown binary op type");
    }
#undef CASE
}

void CheckBitwiseShiftDtype(const DataType &selfType, const DataType &otherType) {
    std::vector<DataType> BITWISRSHIFT_SUPPORT_DATATYPES = {DataType::DT_INT16, DataType::DT_INT32};
    bool selfSupport = (std::find(BITWISRSHIFT_SUPPORT_DATATYPES.begin(), BITWISRSHIFT_SUPPORT_DATATYPES.end(), selfType) != 
           BITWISRSHIFT_SUPPORT_DATATYPES.end());
    bool otherSupport = (std::find(BITWISRSHIFT_SUPPORT_DATATYPES.begin(), BITWISRSHIFT_SUPPORT_DATATYPES.end(), otherType) != 
           BITWISRSHIFT_SUPPORT_DATATYPES.end());
    ASSERT(selfSupport && otherSupport) << "Inputs datatype not supported";
}

static const DataType promotedTable[DTYPE_NUM][DTYPE_NUM] = {
    //                DT_INT8    DT_UINT8   DT_INT16   DT_INT32
    /* DT_INT8   */  {DT_INT8,   DT_INT16,  DT_INT16,  DT_INT32},
    /* DT_UINT8  */  {DT_INT16,  DT_UINT8,  DT_INT16,  DT_INT32},
    /* DT_INT16  */  {DT_INT16,  DT_INT16,  DT_INT16,  DT_INT32},
    /* DT_INT32 */   {DT_INT32,  DT_INT32,  DT_INT32,  DT_INT32},
};
struct LogicalInput {
    const LogicalTensorPtr tensor;
    TileInfo tileInfo;
};

template <BitwiseShiftOpType T>
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
        function.AddOperation(GetBitwiseShiftOpNameCode<T, false>(), {inputTile1, inputTile2}, {resultTile, tempTensor});
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


template <BitwiseShiftOpType T>
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

template <BitwiseShiftOpType T>
LogicalTensorPtr TensorBitwiseShiftOperation(Function &function, const Tensor &self, const Tensor &other) {
    if (ConfigManager::Instance().GetOperationConfig(KEY_COMBINE_AXIS, false)) {
        ConfigManager::Instance().SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
    }
    auto operand1 = self.GetStorage();
    auto operand2 = other.GetStorage();
    if (operand1->shape.size() != operand2->shape.size()) {
        std::vector<int> broadCastShape = GetBroadCastShape(operand1, operand2);
        operand1 = BinaryOperationBroadCast(operand1, broadCastShape);
        operand2 = BinaryOperationBroadCast(operand2, broadCastShape);
    }
    if (operand1->Format() != operand2->Format()) {
        ASSERT(false && "The format of input tensors are not same.");
    }
    auto opName = GetBitwiseShiftOpName<T>();
    CheckTensorShape(operand1, opName);
    CheckTensorShape(operand2, opName);
    CheckBinOpOperandsValid(operand1, operand2);

    std::vector<SymbolicScalar> resultValidShape;
    std::vector<int64_t> resultShape = BinaryOperationResultShape(operand1, operand2);
    if ((!operand1->GetDynValidShape().empty()) && (!operand2->GetDynValidShape().empty())) {
        for (size_t i = 0; i < resultShape.size(); ++i) {
            if (resultShape[i] == operand1->shape[i]) {
                resultValidShape.push_back(self.GetStorage()->GetDynValidShape()[i]);
            } else {
                resultValidShape.push_back(other.GetStorage()->GetDynValidShape()[i]);
            }
        }
    }
    DataType promotedDtype = promotedTable[operand1->Datatype()][operand2->Datatype()];
    auto result = std::make_shared<LogicalTensor>(function, promotedDtype, resultShape, resultValidShape, operand1->Format());
    function.AddOperation(GetBitwiseShiftOpNameCode<T>(), {operand1, operand2}, {result});
    return result;
}

template <BitwiseShiftOpType T>
void TiledBitwiseShiftOperationScalar(Function &function, const TileShape &tileShape, size_t cur, LogicalInput &input1,
    Element &value, const LogicalTensorPtr &result, TileInfo &resultTileInfo, bool reverseOperand) {
    if (cur == input1.tensor->GetShape().size()) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        // 确认接口
        auto &op = function.AddOperation(GetBitwiseShiftOpNameCode<T, true>(), {inputTile1}, {resultTile});
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

template <BitwiseShiftOpType T>
void TiledBitwiseShiftOperationScalar(Function &function, const TileShape &tileShape, LogicalTensorPtr operand1,
    Element value, const LogicalTensorPtr &result, bool reverseOperand = false) {
    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = LogicalInput{operand1, tileInfo1};
    TiledBitwiseShiftOperationScalar<T>(function, tileShape, 0, input1, value, result, resultTileInfo, reverseOperand);
}

template <BitwiseShiftOpType T>
LogicalTensorPtr TensorBitwiseShiftOperationScalar(Function &function, const LogicalTensorPtr& self, const Element &other) {
    auto opName = GetBitwiseShiftOpName<T>();
    CheckTensorShape(self, opName);
    auto result = std::make_shared<LogicalTensor>(function, self->Datatype(), self->shape, self->GetDynValidShape());
    auto &op = function.AddOperation(GetBitwiseShiftOpNameCode<T, true>(), {self}, {result});
    op.SetAttribute(OpAttributeKey::scalar, other);
    return result;
}

Tensor BitwiseRightShift(const Tensor &self, const Tensor &other) {
    DECLARE_TRACER();
    CheckBitwiseShiftDtype(self.GetDataType(), other.GetDataType());
    RETURN_CALL(BitwiseShiftOperation<BitwiseShiftOpType::BITWISERIGHTSHIFT>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor BitwiseRightShift(const Tensor &self, const Element &other) {
    DECLARE_TRACER();
    CheckBitwiseShiftDtype(self.GetDataType(), other.GetDataType());
    Element newOther = other;
    if (self.GetDataType() != other.GetDataType()) {
        newOther = Element(self.GetDataType(), other.Cast<int32_t>());
    }
    RETURN_CALL(BitwiseShiftOperationScalar<BitwiseShiftOpType::BITWISERIGHTSHIFT>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), newOther);
}

Tensor BitwiseRightShift(const Element &self, const Tensor &other) {
    DECLARE_TRACER();
    CheckBitwiseShiftDtype(self.GetDataType(), other.GetDataType());
    Tensor newSelf = Full(self, other.GetDataType(), other.GetShape(), other.GetStorage()->GetDynValidShape());
    RETURN_CALL(BitwiseShiftOperation<BitwiseShiftOpType::BITWISERIGHTSHIFT>, *Program::GetInstance().GetCurrentFunction(), newSelf, other);
}
Tensor BitwiseLeftShift(const Tensor &self, const Tensor &other) {
    DECLARE_TRACER();
    CheckBitwiseShiftDtype(self.GetDataType(), other.GetDataType());
    RETURN_CALL(BitwiseShiftOperation<BitwiseShiftOpType::BITWISELEFTSHIFT>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor BitwiseLeftShift(const Tensor &self, const Element &other) {
    DECLARE_TRACER();
    CheckBitwiseShiftDtype(self.GetDataType(), other.GetDataType());
    Element newOther = other;
    if (self.GetDataType() != other.GetDataType()) {
        newOther = Element(self.GetDataType(), other.Cast<int32_t>());
    }
    RETURN_CALL(BitwiseShiftOperationScalar<BitwiseShiftOpType::BITWISELEFTSHIFT>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), newOther);
}

Tensor BitwiseLeftShift(const Element &self, const Tensor &other) {
    DECLARE_TRACER();
    CheckBitwiseShiftDtype(self.GetDataType(), other.GetDataType());
    auto newSelf = Full(self, self.GetDataType(), other.GetShape(), other.GetStorage()->GetDynValidShape());
    RETURN_CALL(BitwiseShiftOperation<BitwiseShiftOpType::BITWISELEFTSHIFT>, *Program::GetInstance().GetCurrentFunction(), newSelf, other);
}

template <BitwiseShiftOpType T>
void BitwiseShiftOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    BinaryOperationOperandCheck(iOperand, oOperand);
    TiledBitwiseShiftOperation<T>(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
}

template <BitwiseShiftOpType T>
void BitwiseShiftOperationScalarTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    TiledBitwiseShiftOperationScalar<T>(function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar), oOperand[0]);
}

REGISTER_OPERATION_TILED_FUNC(OP_BITWISERIGHTSHIFT, Opcode::OP_BITWISERIGHTSHIFT, BitwiseShiftOperationTileFunc<BitwiseShiftOpType::BITWISERIGHTSHIFT>);
REGISTER_OPERATION_TILED_FUNC(OP_BITWISELEFTSHIFT, Opcode::OP_BITWISELEFTSHIFT, BitwiseShiftOperationTileFunc<BitwiseShiftOpType::BITWISELEFTSHIFT>);

REGISTER_OPERATION_TILED_FUNC(OP_BITWISERIGHTSHIFTS, Opcode::OP_BITWISERIGHTSHIFTS, BitwiseShiftOperationScalarTileFunc<BitwiseShiftOpType::BITWISERIGHTSHIFT>);
REGISTER_OPERATION_TILED_FUNC(OP_BITWISELEFTSHIFTS, Opcode::OP_BITWISELEFTSHIFTS, BitwiseShiftOperationScalarTileFunc<BitwiseShiftOpType::BITWISELEFTSHIFT>);

} // namespace npu::tile_fwk