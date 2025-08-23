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

void ExecuteOpAMulB(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ctx->ioperandDataViewList->size() == SIZE_TWO || ctx->ioperandDataViewList->size() == SIZE_THREE);
    auto ret = ctx->ooperandInplaceDataViewList->at(0);
    auto lhs = ctx->ioperandDataViewList->at(0);
    auto rhs = ctx->ioperandDataViewList->at(1);
    if (ret->GetDataType() != lhs->GetDataType()) {
        // HACK: matmul type might be different between ioperand and ooperand
        auto lhsCast = LogicalTensorData::CreateEmpty(ret->GetDataType(), lhs->GetShape(), lhs->GetValidShape());
        auto rhsCast = LogicalTensorData::CreateEmpty(ret->GetDataType(), rhs->GetShape(), rhs->GetValidShape());
        Calculator::CalcCast(lhsCast.get(), lhs.get(), ret->GetDataType(), CastMode::CAST_NONE, ctx->opInter->GetPoolPtr());
        Calculator::CalcCast(rhsCast.get(), rhs.get(), ret->GetDataType(), CastMode::CAST_NONE, ctx->opInter->GetPoolPtr());
        lhs = lhsCast;
        rhs = rhsCast;
    }

    int k1 = ctx->op->GetTileShape().K(1);
    int k2 = ctx->op->GetTileShape().K(2);
    int kStep = lhs->GetShape()[1];
    if (k1 != 0 && k2 != 0) {
        kStep = std::gcd(k1, k2);
    }
    switch (ctx->op->GetOpcode()) {
        case Opcode::OP_A_MUL_B: Calculator::CalcMatMul(ret.get(), lhs.get(), rhs.get(), kStep, ctx->opInter->GetPoolPtr()); break;
        case Opcode::OP_A_MULACC_B:
            Calculator::CalcMatMulAcc(
                ret.get(), lhs.get(), rhs.get(), kStep, ctx->ioperandDataViewList->at(SIZE_TWO).get(), ctx->opInter->GetPoolPtr());
            break;
        case Opcode::OP_A_MUL_BT: Calculator::CalcMatMulTrans(ret.get(), lhs.get(), rhs.get(), kStep, ctx->opInter->GetPoolPtr()); break;
        case Opcode::OP_A_MULACC_BT:
            Calculator::CalcMatMulTransAcc(
                ret.get(), lhs.get(), rhs.get(), kStep, ctx->ioperandDataViewList->at(SIZE_TWO).get(), ctx->opInter->GetPoolPtr());
            break;
        default: ASSERT(false); break;
    }
}
REGISTER_CLACOP_FUNC(OP_A_MUL_B, Opcode::OP_A_MUL_B, ExecuteOpAMulB);
REGISTER_CLACOP_FUNC(OP_A_MULACC_B, Opcode::OP_A_MULACC_B, ExecuteOpAMulB);
REGISTER_CLACOP_FUNC(OP_A_MUL_BT, Opcode::OP_A_MUL_BT, ExecuteOpAMulB);
REGISTER_CLACOP_FUNC(OP_A_MULACC_BT, Opcode::OP_A_MULACC_BT, ExecuteOpAMulB);

void ExecuteOpAlloc(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() <= 1);
    ASSERT(ctx->ioperandDataViewList->size() == 0);
}
REGISTER_CLACOP_FUNC(OP_UB_ALLOC, Opcode::OP_UB_ALLOC, ExecuteOpAlloc);
REGISTER_CLACOP_FUNC(OP_L0A_ALLOC, Opcode::OP_L0A_ALLOC, ExecuteOpAlloc);
REGISTER_CLACOP_FUNC(OP_L0B_ALLOC, Opcode::OP_L0B_ALLOC, ExecuteOpAlloc);
REGISTER_CLACOP_FUNC(OP_L0C_ALLOC, Opcode::OP_L0C_ALLOC, ExecuteOpAlloc);
REGISTER_CLACOP_FUNC(OP_L1_ALLOC, Opcode::OP_L1_ALLOC, ExecuteOpAlloc);

void ExecuteDuplicate(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto &ret = ctx->ooperandInplaceDataViewList->at(0);
    auto &oper = ctx->ioperandDataViewList->at(0);
    Opcode opCode = ctx->op->GetOpcode();
    if (opCode == Opcode::OP_L1_TO_L0_BT || opCode == Opcode::OP_L1_TO_L0_AT) {
        std::vector<int> axises = {0, 1};
        Calculator::CalcTransposeAdjDim(ret.get(), oper.get(), axises[0], ctx->opInter->GetPoolPtr());
    } else {
        Calculator::CalcCopy(ret.get(), oper.get(), ctx->opInter->GetPoolPtr());
    }
}
REGISTER_CLACOP_FUNC(OP_L1_TO_L0A, Opcode::OP_L1_TO_L0A, ExecuteDuplicate);
REGISTER_CLACOP_FUNC(OP_L1_TO_L0B, Opcode::OP_L1_TO_L0B, ExecuteDuplicate);
REGISTER_CLACOP_FUNC(OP_L1_TO_L0_AT, Opcode::OP_L1_TO_L0_AT, ExecuteDuplicate);
REGISTER_CLACOP_FUNC(OP_L1_TO_L0_BT, Opcode::OP_L1_TO_L0_BT, ExecuteDuplicate);
REGISTER_CLACOP_FUNC(OP_CONVERT, Opcode::OP_CONVERT, ExecuteDuplicate);
}