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
 * \file calc_cube.cpp
 * \brief
 */

#include "interface/interpreter/function.h"
#include "interface/utils/log.h"
#include "interface/interpreter/operation.h"
#include "calc.h"
#include "interface/operation/operation_impl.h"

namespace npu::tile_fwk {

void ExecuteOpAMulB(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    auto ret = ctx->ooperandInplaceDataViewList->at(0);
    auto lhs = ctx->ioperandDataViewList->at(0);
    auto rhs = ctx->ioperandDataViewList->at(1);
    auto bias = (ctx->op->GetBoolAttribute(Matrix::A_MUL_B_BIAS_ATTR)) ? ctx->ioperandDataViewList->at(2) : nullptr;
    auto &cubeTile = ctx->op->GetTileShape().GetCubeTile();
    int k1 = cubeTile.k[1];
    int k2 = cubeTile.k[2];
    int kStep = std::gcd(k1, k2);
    bool transA = (ctx->op->HasAttr(Matrix::A_MUL_B_TRANS_A)) ? ctx->op->GetBoolAttribute(Matrix::A_MUL_B_TRANS_A) : false;
    bool transB = (ctx->op->HasAttr(Matrix::A_MUL_B_TRANS_B)) ? ctx->op->GetBoolAttribute(Matrix::A_MUL_B_TRANS_B) : false;
    uint64_t scale = (ctx->op->HasAttr(Matrix::A_MUL_B_SCALE_ATTR)) ? ctx->op->GetElementAttribute(Matrix::A_MUL_B_SCALE_ATTR).GetUnsignedData() : 0;
    int relu = (ctx->op->HasAttr(Matrix::A_MUL_B_RELU_ATTR)) ? ctx->op->GetIntAttribute(Matrix::A_MUL_B_RELU_ATTR) : 0;
    LogicalTensorDataPtr scalePtr = nullptr;
    if (lhs->GetDataType() == DataType::DT_INT8 && ret->GetDataType() == DataType::DT_FP16 && scale == 0) {
        for (int idx = 0; idx < (int) ctx->ioperandDataViewList->size(); idx++) {
            if (ctx->ioperandDataViewList->at(idx)->GetDataType() == DataType::DT_UINT64) {
                scalePtr = ctx->ioperandDataViewList->at(idx);
            }
        }
    }
    MatMulParam param = {transA, transB, kStep, scale, relu, scalePtr};
    switch (ctx->op->GetOpcode()) {
        case Opcode::OP_A_MUL_B: {
            calc::MatMul(ret, lhs, rhs, bias, param);
        } break;
        case Opcode::OP_A_MULACC_B: {
            auto acc = ctx->ioperandDataViewList->at(2);
            if (lhs->GetDataType() == DataType::DT_INT8 && acc->GetDataType() == DataType::DT_FP32) {
                throw std::runtime_error("pass customized part, cannot restore the computation logic.");
            }
            calc::AccMatMul(ret, lhs, rhs, acc, param);
        } break;
        default: ASSERT(false); break;
    }
}
REGISTER_CALC_OP(OP_A_MUL_B, Opcode::OP_A_MUL_B, ExecuteOpAMulB);
REGISTER_CALC_OP(OP_A_MULACC_B, Opcode::OP_A_MULACC_B, ExecuteOpAMulB);
REGISTER_CALC_OP(OP_A_MUL_BT, Opcode::OP_A_MUL_BT, ExecuteOpAMulB);
REGISTER_CALC_OP(OP_A_MULACC_BT, Opcode::OP_A_MULACC_BT, ExecuteOpAMulB);
REGISTER_CALC_OP(OP_AT_MUL_B, Opcode::OP_AT_MUL_B, ExecuteOpAMulB);
REGISTER_CALC_OP(OP_AT_MUL_BT, Opcode::OP_AT_MUL_BT, ExecuteOpAMulB);

void ExecuteOpAlloc(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() <= 1);
    ASSERT(ctx->ioperandDataViewList->size() == 0);
}
REGISTER_CALC_OP(OP_UB_ALLOC, Opcode::OP_UB_ALLOC, ExecuteOpAlloc);
REGISTER_CALC_OP(OP_L0A_ALLOC, Opcode::OP_L0A_ALLOC, ExecuteOpAlloc);
REGISTER_CALC_OP(OP_L0B_ALLOC, Opcode::OP_L0B_ALLOC, ExecuteOpAlloc);
REGISTER_CALC_OP(OP_L0C_ALLOC, Opcode::OP_L0C_ALLOC, ExecuteOpAlloc);
REGISTER_CALC_OP(OP_L1_ALLOC, Opcode::OP_L1_ALLOC, ExecuteOpAlloc);
REGISTER_CALC_OP(OP_FIX_ALLOC, Opcode::OP_FIX_ALLOC, ExecuteOpAlloc);
REGISTER_CALC_OP(OP_BT_ALLOC, Opcode::OP_BT_ALLOC, ExecuteOpAlloc);

void ExecuteDuplicate(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto &ret = ctx->ooperandInplaceDataViewList->at(0);
    auto &oper = ctx->ioperandDataViewList->at(0);
    Opcode opCode = ctx->op->GetOpcode();
    bool trans = opCode == Opcode::OP_L1_TO_L0_BT || opCode == Opcode::OP_L1_TO_L0_AT;
    auto copyin = std::static_pointer_cast<CopyOpAttribute>(ctx->op->GetOpAttribute()); // 获取attr
    if (opCode == Opcode::OP_L1_TO_L0A || opCode == Opcode::OP_L1_TO_L0B || opCode == Opcode::OP_L1_TO_L0_AT || opCode == Opcode::OP_L1_TO_L0_BT) {
        if (copyin == nullptr) {
            calc::Copy(ret, oper, trans);
            return;
        }
        std::vector<int64_t> fromOffset = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyin->GetFromOffset());
        auto iop = oper->View(ret->GetShape(), fromOffset);
        calc::Copy(ret, iop, trans);
    } else if (opCode == Opcode::OP_L0C_TO_L1) {
        // fixpipe
        bool quant = oper->GetDataType() == DataType::DT_INT32 && ret->GetDataType() == DataType::DT_FP16;
        uint64_t scale = (ctx->op->HasAttr(Matrix::A_MUL_B_SCALE_ATTR)) ? ctx->op->GetElementAttribute(Matrix::A_MUL_B_SCALE_ATTR).GetUnsignedData() : 0;
        int relu = (ctx->op->HasAttr(Matrix::A_MUL_B_RELU_ATTR)) ? ctx->op->GetIntAttribute(Matrix::A_MUL_B_RELU_ATTR) : 0;
        LogicalTensorDataPtr scalePtr = nullptr;
        if (ctx->ioperandDataViewList->size() > 1) {
            scalePtr = ctx->ioperandDataViewList->at(1);
        }
        std::vector<int64_t> shape = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyin->GetShape());
        std::vector<int64_t> fromOffset = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyin->GetFromOffset());
        std::vector<int64_t> toOffset = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyin->GetToOffset());
        if (oper->GetShape()[0] > ret->GetShape()[0] || oper->GetShape()[1] > ret->GetShape()[1]) {
            auto iop = oper->View(ret->GetShape(), fromOffset);
            if (scalePtr != nullptr) {
                auto scaleOp = scalePtr->View({1, ret->GetShape()[1]}, {0, fromOffset[1]});
            }
            calc::Copy(ret, iop);
            if (quant) {
                calc::Fixpipe(ret, iop, scalePtr, scale, relu);
            }
        } else {
            auto iop = ret->View(oper->GetShape(), toOffset);
            calc::Copy(iop, oper);
            if (quant) {
                calc::Fixpipe(ret, oper, scalePtr, scale, relu);
            }
        }
    } else {
        calc::Copy(ret, oper, trans);
    }
}
REGISTER_CALC_OP(OP_L1_TO_L0A, Opcode::OP_L1_TO_L0A, ExecuteDuplicate);
REGISTER_CALC_OP(OP_L1_TO_L0B, Opcode::OP_L1_TO_L0B, ExecuteDuplicate);
REGISTER_CALC_OP(OP_L1_TO_L0_AT, Opcode::OP_L1_TO_L0_AT, ExecuteDuplicate);
REGISTER_CALC_OP(OP_L1_TO_L0_BT, Opcode::OP_L1_TO_L0_BT, ExecuteDuplicate);
REGISTER_CALC_OP(OP_CONVERT, Opcode::OP_CONVERT, ExecuteDuplicate);
REGISTER_CALC_OP(OP_L1_TO_FIX_QUANT_PRE, Opcode::OP_L1_TO_FIX_QUANT_PRE, ExecuteDuplicate);
REGISTER_CALC_OP(OP_L1_TO_BT, Opcode::OP_L1_TO_BT, ExecuteDuplicate);
REGISTER_CALC_OP(OP_L0C_TO_L1, Opcode::OP_L0C_TO_L1, ExecuteDuplicate);
} // namespace npu::tile_fwk

        