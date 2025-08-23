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
    using ExecEntry = void (OperationInterpreter::*)(ExecuteOperationContext *ctx);
    // clang-format off
    static std::unordered_map<Opcode, ExecEntry> execEntryDict = {
        {Opcode::OP_CAST, &OperationInterpreter::ExecuteOpCast},

        {Opcode::OP_EXP, &OperationInterpreter::ExecuteOpUnary<Opcode::OP_EXP>},
        {Opcode::OP_SQRT, &OperationInterpreter::ExecuteOpUnary<Opcode::OP_SQRT>},
        {Opcode::OP_ABS, &OperationInterpreter::ExecuteOpUnary<Opcode::OP_ABS>},

        {Opcode::OP_ADD, &OperationInterpreter::ExecuteOpBinary<Opcode::OP_ADD>},
        {Opcode::OP_ADD_BRC, &OperationInterpreter::ExecuteOpBinary<Opcode::OP_ADD_BRC>},
        {Opcode::OP_SUB, &OperationInterpreter::ExecuteOpBinary<Opcode::OP_SUB>},
        {Opcode::OP_SUB_BRC, &OperationInterpreter::ExecuteOpBinary<Opcode::OP_SUB_BRC>},
        {Opcode::OP_MUL, &OperationInterpreter::ExecuteOpBinary<Opcode::OP_MUL>},
        {Opcode::OP_MUL_BRC, &OperationInterpreter::ExecuteOpBinary<Opcode::OP_MUL_BRC>},
        {Opcode::OP_DIV, &OperationInterpreter::ExecuteOpBinary<Opcode::OP_DIV>},
        {Opcode::OP_DIV_BRC, &OperationInterpreter::ExecuteOpBinary<Opcode::OP_DIV_BRC>},
        {Opcode::OP_S_ADD, &OperationInterpreter::ExecuteOpBinary<Opcode::OP_ADD>},
        {Opcode::OP_S_SUB, &OperationInterpreter::ExecuteOpBinary<Opcode::OP_SUB>},
        {Opcode::OP_S_MUL, &OperationInterpreter::ExecuteOpBinary<Opcode::OP_MUL>},
        {Opcode::OP_S_DIV, &OperationInterpreter::ExecuteOpBinary<Opcode::OP_DIV>},
        {Opcode::OP_S_MAX, &OperationInterpreter::ExecuteOpBinary<Opcode::OP_S_MAX>},
        {Opcode::OP_PAIRMAX, &OperationInterpreter::ExecuteOpBinary<Opcode::OP_PAIRMAX>},
        {Opcode::OP_PAIRMIN, &OperationInterpreter::ExecuteOpBinary<Opcode::OP_PAIRMIN>},
        {Opcode::OP_PAIRSUM, &OperationInterpreter::ExecuteOpBinary<Opcode::OP_PAIRSUM>},
        {Opcode::OP_S_MIN, &OperationInterpreter::ExecuteOpBinary<Opcode::OP_S_MIN>},
        {Opcode::OP_MAXIMUM, &OperationInterpreter::ExecuteOpBinary<Opcode::OP_S_MAX>},

        {Opcode::OP_ADDS, &OperationInterpreter::ExecuteOpBinaryScalar<Opcode::OP_ADDS>},
        {Opcode::OP_SUBS, &OperationInterpreter::ExecuteOpBinaryScalar<Opcode::OP_SUBS>},
        {Opcode::OP_MULS, &OperationInterpreter::ExecuteOpBinaryScalar<Opcode::OP_MULS>},
        {Opcode::OP_DIVS, &OperationInterpreter::ExecuteOpBinaryScalar<Opcode::OP_DIVS>},
        {Opcode::OP_S_ADDS, &OperationInterpreter::ExecuteOpBinaryScalar<Opcode::OP_ADDS>},
        {Opcode::OP_S_SUBS, &OperationInterpreter::ExecuteOpBinaryScalar<Opcode::OP_SUBS>},
        {Opcode::OP_S_MULS, &OperationInterpreter::ExecuteOpBinaryScalar<Opcode::OP_MULS>},
        {Opcode::OP_S_DIVS, &OperationInterpreter::ExecuteOpBinaryScalar<Opcode::OP_DIVS>},
        {Opcode::OP_S_MAXS, &OperationInterpreter::ExecuteOpBinaryScalar<Opcode::OP_S_MAXS>},
        {Opcode::OP_S_MIN, &OperationInterpreter::ExecuteOpBinaryScalar<Opcode::OP_S_MINS>},

        {Opcode::OP_A_MUL_B, &OperationInterpreter::ExecuteOpAMulB},
        {Opcode::OP_A_MULACC_B, &OperationInterpreter::ExecuteOpAMulB},
        {Opcode::OP_A_MUL_BT, &OperationInterpreter::ExecuteOpAMulB},
        {Opcode::OP_A_MULACC_BT, &OperationInterpreter::ExecuteOpAMulB},

        {Opcode::OP_VEC_DUP, &OperationInterpreter::ExecuteOpVecDup},
        {Opcode::OP_ROWSUM_SINGLE, &OperationInterpreter::ExecuteOpReduce<Opcode::OP_ROWSUM_SINGLE>},
        {Opcode::OP_ROWSUMLINE, &OperationInterpreter::ExecuteOpReduce<Opcode::OP_ROWSUMLINE>},
        {Opcode::OP_ROWMAX_SINGLE, &OperationInterpreter::ExecuteOpReduce<Opcode::OP_ROWMAX_SINGLE>},
        {Opcode::OP_ROWMIN_SINGLE, &OperationInterpreter::ExecuteOpReduce<Opcode::OP_ROWMIN_SINGLE>},
        {Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE, &OperationInterpreter::ExecuteOpReduce<Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE>},
        {Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE, &OperationInterpreter::ExecuteOpReduce<Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE>},

        {Opcode::OP_RESHAPE, &OperationInterpreter::ExecuteOpReshape},
        {Opcode::OP_EXPAND, &OperationInterpreter::ExecuteOpExpand},
        {Opcode::OP_TRANSPOSE_MOVEOUT, &OperationInterpreter::ExecuteOpTransposeDataMove},
        {Opcode::OP_TRANSPOSE_VNCHWCONV, &OperationInterpreter::ExecuteOpTranspose},
        {Opcode::OP_INDEX_OUTCAST, &OperationInterpreter::ExecuteOpIndexOutcast},

        {Opcode::OP_REDUCE_ACC, &OperationInterpreter::ExecuteOpReduceAcc},
        {Opcode::OP_PHASE1, &OperationInterpreter::ExecuteOpNone},
        {Opcode::OP_PHASE2, &OperationInterpreter::ExecuteOpNone},
        {Opcode::OP_SYNC_SRC, &OperationInterpreter::ExecuteOpNone},
        {Opcode::OP_SYNC_DST, &OperationInterpreter::ExecuteOpNone},
        {Opcode::OP_BAR_V, &OperationInterpreter::ExecuteOpNone},
        {Opcode::OP_BAR_M, &OperationInterpreter::ExecuteOpNone},
        {Opcode::OP_UB_ALLOC, &OperationInterpreter::ExecuteOpAlloc},
        {Opcode::OP_L0A_ALLOC, &OperationInterpreter::ExecuteOpAlloc},
        {Opcode::OP_L0B_ALLOC, &OperationInterpreter::ExecuteOpAlloc},
        {Opcode::OP_L0C_ALLOC, &OperationInterpreter::ExecuteOpAlloc},
        {Opcode::OP_L1_ALLOC, &OperationInterpreter::ExecuteOpAlloc},
        {Opcode::OP_L1_TO_L0A, &OperationInterpreter::ExecuteDuplicate},
        {Opcode::OP_L1_TO_L0B, &OperationInterpreter::ExecuteDuplicate},
        {Opcode::OP_L1_TO_L0_AT, &OperationInterpreter::ExecuteDuplicate},
        {Opcode::OP_L1_TO_L0_BT, &OperationInterpreter::ExecuteDuplicate},
        {Opcode::OP_CONVERT, &OperationInterpreter::ExecuteDuplicate},

        {Opcode::OP_ASSEMBLE, &OperationInterpreter::ExecuteOpAssemble},
        {Opcode::OP_VIEW, &OperationInterpreter::ExecuteOpView},
        {Opcode::OP_COPY_OUT, &OperationInterpreter::ExecuteOpCopyOut},
        {Opcode::OP_COPY_IN, &OperationInterpreter::ExecuteOpCopyIn},
        {Opcode::OP_REGISTER_COPY, &OperationInterpreter::ExecuteOpCopy},
        {Opcode::OP_PRINT, &OperationInterpreter::ExecutePrint},
    };
    // clang-format on

    auto iOperands = GetValidDataView(*ctx->ioperandDataViewList);
    auto oOperands = GetValidDataView(*ctx->ooperandInplaceDataViewList);
    ExecuteOperationContext ctxValid = {ctx->frame, ctx->op, &iOperands, {}, &oOperands};

    ASSERT(execEntryDict.count(ctx->op->GetOpcode()));
    auto execEntry = execEntryDict[ctx->op->GetOpcode()];
    (this->*execEntry)(&ctxValid);
}
}