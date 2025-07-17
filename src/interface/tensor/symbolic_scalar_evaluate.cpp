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
 * \file symbolic_scalar_evaluate.cpp
 * \brief
 */

#include "interface/tensor/symbolic_scalar_evaluate.h"

namespace npu::tile_fwk {

ScalarImmediateType EvaluateSymbol::EvaluateSymbolicScalar(const RawSymbolicScalarPtr &ss) {
    ScalarImmediateType result{0};
    switch (ss->Kind()) {
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE: {
            std::shared_ptr<RawSymbolicImmediate> imm = std::static_pointer_cast<RawSymbolicImmediate>(ss);
            result = imm->Immediate();
        } break;
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL: {
            std::shared_ptr<RawSymbolicSymbol> sym = std::static_pointer_cast<RawSymbolicSymbol>(ss);
            ASSERT(symbolDict_.count(sym->Name()));
            result = symbolDict_[sym->Name()];
        } break;
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION: {
            std::shared_ptr<RawSymbolicExpression> expr = std::static_pointer_cast<RawSymbolicExpression>(ss);
            if (expr->Opcode() == SymbolicOpcode::T_MOP_CALL) {
                std::vector<ScalarImmediateType> dataList;
                for (size_t i = 1; i < expr->OperandList().size(); i++) {
                    dataList.emplace_back(EvaluateSymbolicScalar(expr->OperandList()[i]));
                }
                std::string name = std::static_pointer_cast<RawSymbolicSymbol>(expr->OperandList()[0])->Name();
                result = EvaluateSymbolicCall(name, dataList);
            } else {
                std::vector<ScalarImmediateType> dataList;
                for (size_t i = 0; i < expr->OperandList().size(); i++) {
                    dataList.emplace_back(EvaluateSymbolicScalar(expr->OperandList()[i]));
                }
                if (SymbolicOpcode::T_UOP_BEGIN <= expr->Opcode() && expr->Opcode() < SymbolicOpcode::T_UOP_END) {
                    result = RawSymbolicExpression::GetSymbolicCalcUnary(expr->Opcode())(dataList[0]);
                } else if (SymbolicOpcode::T_BOP_BEGIN <= expr->Opcode() &&
                           expr->Opcode() < SymbolicOpcode::T_BOP_END) {
                    result = dataList[0];
                    for (size_t i = 1; i < dataList.size(); i++) {
                        result = RawSymbolicExpression::GetSymbolicCalcBinary(expr->Opcode())(result, dataList[i]);
                    }
                }
            }
        } break;
        default: ASSERT(false); break;
    }
    return result;
}

} // namespace npu::tile_fwk