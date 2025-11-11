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
 * \file unary.cpp
 * \brief
 */

#include "unary.h"
#include "interface/utils/operator_tracer.h"

namespace npu::tile_fwk {

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

    RETURN_CALL(UnaryOperation<UnaryOpType::RSQRT>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
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

Tensor Duplicate(const Tensor &operand) {
    DECLARE_TRACER();

    RETURN_CALL(
        UnaryOperation<UnaryOpType::DUPLICATE>, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage());
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

REGISTER_OPERATION_TILED_FUNC(OP_EXP, Opcode::OP_EXP, ExpOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_RSQRT, Opcode::OP_RSQRT, RsqrtOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_SQRT, Opcode::OP_SQRT, SqrtOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_RECIPROCAL, Opcode::OP_RECIPROCAL, ReciprocalOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ABS, Opcode::OP_ABS, AbsOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_LN, Opcode::OP_LN, LnOperationTileFunc);

} // namespace npu::tile_fwk