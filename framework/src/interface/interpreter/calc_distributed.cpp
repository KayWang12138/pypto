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
 * \file calc_distributed.cpp
 * \brief
 */

#include <memory>
#include "interface/interpreter/operation.h"
#include "interface/interpreter/calc.h"
#include "tensor/symbolic_scalar.h"
#include "tilefwk/error.h"
#include "interface\operation\distributed\distributed_common.h"

namespace npu::tile_fwk {

std::vector<uint64_t> UnBind(ExecuteOperationContext *ctx, SymbolicScalar attr) {
    std::shared_ptr<RawSymbolicExpression> expr = std::static_pointer_cast<RawSymbolicExpression>(attr.Raw());
    ASSERT(expr->Opcode() == SymbolicOpcode::T_MOP_CALL);
    std::vector<uint64_t> parameters;
    for (size_t i = 1; i < expr->OperandList().size(); i++) {
        ScalarImmediateType value = ctx->opInter->EvaluateSymbolicScalar(SymbolicScalar(expr->OperandList()[i]));
        parameters.emplace_back(value);
    }
    return parameters;
}

void ExecuteOpBindTensor(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == 0);
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    auto out = ctx->ooperandInplaceDataViewList->at(0);
    SymbolicScalar attr = ctx->op->GetSymbolicScalarAttribute(OpAttributeKey::bindTensor);
    std::vector<uint64_t> parameters = UnBind(ctx, attr);
    uint64_t groupIndex = parameters[0];
    uint64_t memType = parameters[1];
    uint64_t slotSize = parameters[2];
    calc::BindTensor(out, groupIndex, memType, slotSize);
}
REGISTER_CALC_OP(OP_BIND_TENSOR, Opcode::OP_BIND_TENSOR, ExecuteOpBindTensor);

void ExecuteOpShmemPut(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == 3);
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    auto in = ctx->ioperandDataViewList->at(0);
    Distributed::ShmemPutAttr attr;
    ctx->op->GetAttr(OpAttributeKey::distOpAttr, attr);
    int dstRank = ctx->opInter->EvaluateSymbolicScalar(attr.ownerRank);
    calc::Put(in, dstRank);
}
REGISTER_CALC_OP(OP_SHMEM_PUT, Opcode::OP_SHMEM_PUT, ExecuteOpShmemPut);

void ExecuteOpShmemGet(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == 2);
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    auto out = ctx->ooperandInplaceDataViewList->at(0);
    Distributed::ShmemGetAttr attr;
    ctx->op->GetAttr(OpAttributeKey::distOpAttr, attr);
    int srcRank = ctx->opInter->EvaluateSymbolicScalar(attr.ownerRank);
    calc::Get(out, srcRank);
}
REGISTER_CALC_OP(OP_SHMEM_GET, Opcode::OP_SHMEM_GET, ExecuteOpShmemGet);

void ExecuteOpShmemWaitUntil(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == 2);
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    auto out = ctx->ooperandInplaceDataViewList->at(0);
    Distributed::ShmemWaitUntilAttr attr;
    ctx->op->GetAttr(OpAttributeKey::distOpAttr, attr);
    int srcRank = ctx->opInter->EvaluateSymbolicScalar(attr.ownerRank);
    calc::WaitUntil(srcRank);
}
REGISTER_CALC_OP(OP_SHMEM_WAIT_UNTIL, Opcode::OP_SHMEM_WAIT_UNTIL, ExecuteOpShmemWaitUntil);

void ExecuteOpShmemWaitUntil(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == 2);
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    auto out = ctx->ooperandInplaceDataViewList->at(0);
    Distributed::ShmemWaitUntilAttr attr;
    ctx->op->GetAttr(OpAttributeKey::distOpAttr, attr);
    int dstRank = ctx->opInter->EvaluateSymbolicScalar(attr.ownerRank);
    calc::Signal(dstRank);
}
REGISTER_CALC_OP(OP_SHMEM_SIGNAL, Opcode::OP_SHMEM_SIGNAL, ExecuteOpShmemSignal);

void ExecuteOpShmemSet(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == 2);
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    auto out = ctx->ooperandInplaceDataViewList->at(0);
    Distributed::ShmemSetAttr attr;
    ctx->op->GetAttr(OpAttributeKey::distOpAttr, attr);
    int dstRank = ctx->opInter->EvaluateSymbolicScalar(attr.ownerRank);
    calc::Set(dstRank);
}
REGISTER_CALC_OP(OP_SHMEM_SET, Opcode::OP_SHMEM_SET, ExecuteOpShmemSet);

}