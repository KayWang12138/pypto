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
 * \file unary.cpp
 * \brief
 */

#include "binary.h"
#include "unary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"

#include <cmath>

namespace npu::tile_fwk {

void UnaryOperationOperandCheck(
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand) {
    ASSERT(iOperand.size() == 1) << "The input operand size should be 1";
    ASSERT(oOperand.size() == 1) << "The output operand size should be 1";
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
    ASSERT(operand->shape.size() == operand->offset.size()) << "The shape size of operand and offset must be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};
    TiledUnaryOperation<T>(function, tileShape, 0, input, result);
}

Tensor Exp(const Tensor &self) {
    DECLARE_TRACER();

    RETURN_CALL(UnaryOperation<UnaryOpType::EXP>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor Ln(const Tensor &operand) {
    DECLARE_TRACER();

    RETURN_CALL(UnaryOperation<UnaryOpType::LN>, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage());
}

Tensor Rsqrt(const Tensor &self) {
    DECLARE_TRACER();

    auto castSelf = self.GetStorage();
    if (self.GetDataType() != DataType::DT_FP32) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(),
            DataType::DT_FP32, CastMode::CAST_NONE);
    }
    auto sqrtSelf = CALL(UnaryOperation<UnaryOpType::SQRT>, *Program::GetInstance().GetCurrentFunction(), castSelf);
    auto ones = CALL(FullOperation, *Program::GetInstance().GetCurrentFunction(), Element(DataType::DT_FP32, 1.0),
        SymbolicScalar(), DataType::DT_FP32, self.GetShape(), self.GetStorage()->GetDynValidShape());
    auto result = CALL(BinaryOperation<BinaryOpType::DIV>, *Program::GetInstance().GetCurrentFunction(), ones, sqrtSelf);
    if (self.GetDataType() != DataType::DT_FP32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), result,
            self.GetDataType(), CastMode::CAST_NONE);
    }
    return result;
}

Tensor Sqrt(const Tensor &self) {
    DECLARE_TRACER();

    RETURN_CALL(UnaryOperation<UnaryOpType::SQRT>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor Reciprocal(const Tensor &operand) {
    DECLARE_TRACER();

    RETURN_CALL(
        UnaryOperation<UnaryOpType::RECIPROCAL>, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage());
}

Tensor Abs(const Tensor &self) {
    DECLARE_TRACER();

    RETURN_CALL(UnaryOperation<UnaryOpType::ABS>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor Hub(const Tensor &self) {
    DECLARE_TRACER();

    RETURN_CALL(UnaryOperation<UnaryOpType::HUB>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor Duplicate(const Tensor &operand) {
    DECLARE_TRACER();

    RETURN_CALL(
        UnaryOperation<UnaryOpType::DUPLICATE>, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage());
}

Tensor Round(const Tensor &self, const int &decimals) {
    DECLARE_TRACER();

    auto shapeSize = self.GetShape().size();
    auto dataType = self.GetDataType();

    ASSERT(SHAPE_DIM2 <= shapeSize && shapeSize <= SHAPE_DIM4) << "The shape.size() only support 2~4";
    std::vector<DataType> ROUND_SUPPORT_DATATYPES = {
        DataType::DT_FP32, DataType::DT_FP16, DataType::DT_BF16, DataType::DT_INT32, DataType::DT_INT16};
    ASSERT(std::find(ROUND_SUPPORT_DATATYPES.begin(), ROUND_SUPPORT_DATATYPES.end(), dataType) !=
           ROUND_SUPPORT_DATATYPES.end())
        << "The datatype is not supported";

    /*auto floatSelf = self.GetStorage();
    if (self.GetDataType() != DataType::DT_FP32) {
        floatSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
            self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    }

    float powDecimals = static_cast<float>(pow(static_cast<float>(10), static_cast<float>(decimals)));
    auto powDecimalsTensor =
        CALL(FullOperation, *Program::GetInstance().GetCurrentFunction(), Element(DataType::DT_FP32, powDecimals),
            SymbolicScalar(), DataType::DT_FP32, self.GetShape(), self.GetStorage()->GetDynValidShape());

    auto mulSelf = Tensor(DataType::DT_FP32, self.GetShape());
    mulSelf = CALL(
        BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), floatSelf, powDecimalsTensor);
    auto roundSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
        mulSelf.GetStorage(), DataType::DT_FP32, CastMode::CAST_ROUND);

    auto result = CALL(
        BinaryOperation<BinaryOpType::DIV>, *Program::GetInstance().GetCurrentFunction(), roundSelf, powDecimalsTensor);
    if (self.GetDataType() != DataType::DT_FP32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), result,
            self.GetDataType(), CastMode::CAST_NONE);
    }
    return result;*/

	RETURN_CALL(UnaryOperation<UnaryOpType::ROUND>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

void RoundOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::ROUND>(function, tileShape, iOperand[0], oOperand[0]);
}

void ExpOperationTileFunc(Function &function, const TileShape &tileShape, const std::vector<LogicalTensorPtr> &iOperand,
    const std::vector<LogicalTensorPtr> &oOperand, [[maybe_unused]] const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::EXP>(function, tileShape, iOperand[0], oOperand[0]);
}

void RsqrtOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::RSQRT>(function, tileShape, iOperand[0], oOperand[0]);
}

void SqrtOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::SQRT>(function, tileShape, iOperand[0], oOperand[0]);
}

void ReciprocalOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::RECIPROCAL>(function, tileShape, iOperand[0], oOperand[0]);
}

void AbsOperationTileFunc(Function &function, const TileShape &tileShape, const std::vector<LogicalTensorPtr> &iOperand,
    const std::vector<LogicalTensorPtr> &oOperand, [[maybe_unused]] const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::ABS>(function, tileShape, iOperand[0], oOperand[0]);
}

void LnOperationTileFunc(Function &function, const TileShape &tileShape, const std::vector<LogicalTensorPtr> &iOperand,
    const std::vector<LogicalTensorPtr> &oOperand, [[maybe_unused]] const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::LN>(function, tileShape, iOperand[0], oOperand[0]);
}

void HubOperationTileFunc(Function &function, const TileShape &tileShape, const std::vector<LogicalTensorPtr> &iOperand,
    const std::vector<LogicalTensorPtr> &oOperand, [[maybe_unused]] const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::HUB>(function, tileShape, iOperand[0], oOperand[0]);
}

REGISTER_OPERATION_TILED_FUNC(OP_EXP, Opcode::OP_EXP, ExpOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ROUND, Opcode::OP_ROUND, RoundOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_RSQRT, Opcode::OP_RSQRT, RsqrtOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_SQRT, Opcode::OP_SQRT, SqrtOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_RECIPROCAL, Opcode::OP_RECIPROCAL, ReciprocalOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ABS, Opcode::OP_ABS, AbsOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_LN, Opcode::OP_LN, LnOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_HUB, Opcode::OP_HUB, HubOperationTileFunc);

} // namespace npu::tile_fwk