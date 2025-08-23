/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "interface/interpreter/function.h"
#include "interface/utils/log.h"
#include "interface/interpreter/operation.h"

namespace npu::tile_fwk {

static int GetAsParameterCoaIndex(const RawSymbolicScalarPtr &value) {
    if (value->IsExpressionCall("RUNTIME_COA_GET_PARAM_OFFSET")) {
        auto &operands = value->GetExpressionOperandList();
        auto base = operands[RUNTIME_GET_PARAM_OFFSET_OPERAND_INDEX_COA_INDEX]->GetImmediateValue();
        auto dimIdx = operands[RUNTIME_GET_PARAM_OFFSET_OPERAND_INDEX_DIM_INDEX]->GetImmediateValue();
        return base + COA_INDEX_DIM_BASE + dimIdx;
    } else if (value->IsExpressionCall("RUNTIME_COA_GET_PARAM_VALID_SHAPE")) {
        auto &operands = value->GetExpressionOperandList();
        auto dim = operands[RUNTIME_GET_PARAM_OFFSET_OPERAND_INDEX_DIM_SIZE_INDEX]->GetImmediateValue();
        auto base = operands[RUNTIME_GET_PARAM_OFFSET_OPERAND_INDEX_COA_INDEX]->GetImmediateValue();
        auto dimIdx = operands[RUNTIME_GET_PARAM_OFFSET_OPERAND_INDEX_DIM_INDEX]->GetImmediateValue();
        return base + COA_INDEX_DIM_BASE + dim * 3 + dimIdx;
    }
    return -1;
}

std::vector<int> OperationInterpreter::EvaluateOpImmediate(
    FunctionFrame *frame, const std::vector<OpImmediate> &opImmList) {
    std::vector<int> result;
    for (auto &opImm : opImmList) {
        int res = 0;
        if (opImm.IsSpecified()) {
            auto opImmValue = opImm.GetSpecifiedValue();
            auto coaIndex = GetAsParameterCoaIndex(opImmValue.Raw());
            if (coaIndex != -1) {
                auto attr = frame->callopAttr->GetLinearArgList()[coaIndex];
                res = EvaluateSymbolicScalar(attr);
            } else {
                res = EvaluateSymbolicScalar(opImm.GetSpecifiedValue());
            }
        } else {
            int index = opImm.GetParameterIndex();
            auto attr = frame->callopAttr->GetLinearArgList()[index];
            res = EvaluateSymbolicScalar(attr);
        }
        result.push_back(res);
    }
    return result;
}

void OperationInterpreter::ExecuteOperation(ExecuteOperationContext *ctx) {
    auto iOperands = OperationInterpreter::GetValidDataView(*ctx->ioperandDataViewList);
    auto oOperands = OperationInterpreter::GetValidDataView(*ctx->ooperandInplaceDataViewList);
    ExecuteOperationContext ctxValid = {ctx->frame, this, ctx->op, &iOperands, {}, &oOperands};
    OperationInterpreter::CallOperationInterpreterFunc(&ctxValid);
}
}