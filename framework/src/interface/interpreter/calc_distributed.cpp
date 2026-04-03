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
#include "tilefwk/comm_group_recorder.h"
#include "communication.h"


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
    auto &out = ctx->ooperandInplaceDataViewList->at(0);
    SymbolicScalar attr = ctx->op->GetSymbolicScalarAttribute(OpAttributeKey::bindTensor);
    std::vector<uint64_t> parameters = UnBind(ctx, attr);
    uint64_t groupIndex = parameters[0];
    uint64_t memType = parameters[1];
    uint64_t slotSize = parameters[2];
    const auto &groupNames = Distributed::CommGroupRecorder::GetInstance().Output();
    ASSERT(groupIndex < static_cast<uint64_t>(groupNames.size()));
    const std::string &groupName = groupNames[groupIndex];
    std::cout << "groupName: " << groupName << " memType: " << memType << " slotSize: " << slotSize << std::endl;
    if (memType == 1) {
        out = SimulationCommManager::Instance().Alloc(groupName, slotSize);
    }
    if (memType == 0) {
        out = SimulationCommManager::Instance().AllocSignal(groupName, slotSize);
    }
}
REGISTER_CALC_OP(OP_BIND_TENSOR, Opcode::OP_BIND_TENSOR, ExecuteOpBindTensor);

void ExecuteOpShmemSet(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == 2);
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1 || ctx->ooperandInplaceDataViewList->size() == 2);
    auto &in = ctx->ioperandDataViewList->at(0);

    ShmemSetAttr attr;
    ctx->op->GetAttribute(OpAttributeKey::distOpAttr, attr);
    
    std::shared_ptr<SimulationCommContext> context = SimulationCommContext::Instance().GetCommContext(attr.group);
    bool isSignal = (attr.setType == 1);
    if (!isSignal) {
        context->Signal(context->GetRank(), 0, in->GetSize() * BytesOf(in->GetDataType()));
    } else {
        context->Set(context->GetRank(), 0, in->GetSize() * BytesOf(in->GetDataType()));
    }
}
REGISTER_CALC_OP(OP_SHMEM_SET, Opcode::OP_SHMEM_SET, ExecuteOpShmemSet);

void ExecuteOpShmemPut(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == 3);
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    auto &in = ctx->ioperandDataViewList->at(1);
    auto &out = ctx->ooperandInplaceDataViewList->at(0);

    ShmemPutAttr attr;
    ctx->op->GetAttribute(OpAttributeKey::distOpAttr, attr);
    
    std::shared_ptr<SimulationCommContext> context = SimulationCommContext::Instance().GetCommContext(attr.group);
    int dstRank = ctx->opInter->EvaluateSymbolicScalar(attr.ownerRank);
    int atomicType = 0;
    if (attr.atomicType == AtomicType.ADD) {
        atomicType = 0;
    }
    if (attr.atomicType == AtomicType.ADD) {
        atomicType = 1;
    }
    context->Put(dstRank, 0, in->GetSize() * BytesOf(in->GetDataType()), atomicType);
}
REGISTER_CALC_OP(OP_SHMEM_PUT, Opcode::OP_SHMEM_PUT, ExecuteOpShmemPut);

void ExecuteOpShmemSignal(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == 2);
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 2 || ctx->ooperandInplaceDataViewList->size() == 1);
    auto &in = ctx->ioperandDataViewList->at(1);

    ShmemSignalAttr attr;
    ctx->op->GetAttribute(OpAttributeKey::distOpAttr, attr);
    
    std::shared_ptr<SimulationCommContext> context = SimulationCommContext::Instance().GetCommContext(attr.group);
    int dstRank = ctx->opInter->EvaluateSymbolicScalar(attr.ownerRank);
    int atomicType = 0;
    if (attr.atomicType == AtomicType.ADD) {
        atomicType = 0;
    }
    if (attr.atomicType == AtomicType.ADD) {
        atomicType = 1;
    }
    int value = attr.signalValue;
    bool notifyAll = attr.notifyAll;
    context->Signal(dstRank, value, in->GetSize() * BytesOf(in->GetDataType()), atomicType, notifyAll);
}
REGISTER_CALC_OP(OP_SHMEM_SIGNAL, Opcode::OP_SHMEM_SIGNAL, ExecuteOpShmemSignal);

void ExecuteOpShmemWaitUntil(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == 2);
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    auto &in = ctx->ioperandDataViewList->at(1);

    ShmemWaitUntilAttr attr;
    ctx->op->GetAttribute(OpAttributeKey::distOpAttr, attr);
    
    std::shared_ptr<SimulationCommContext> context = SimulationCommContext::Instance().GetCommContext(attr.group);
    int srcRank = context->GetRank();
    int expect = attr.expectedSum;
    bool reset = attr.resetSignal;
    context->Wait(srcRank, expect, in->GetSize() * BytesOf(in->GetDataType()), reset);
}
REGISTER_CALC_OP(OP_SHMEM_WAIT_UNTIL, Opcode::OP_SHMEM_WAIT_UNTIL, ExecuteOpShmemWaitUntil);

void ExecuteOpShmemGet(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == 2);
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 2 || ctx->ooperandInplaceDataViewList->size() == 1);
    auto &in = ctx->ioperandDataViewList->at(1);

    ShmemGetAttr attr;
    ctx->op->GetAttribute(OpAttributeKey::distOpAttr, attr);
    
    std::shared_ptr<SimulationCommContext> context = SimulationCommContext::Instance().GetCommContext(attr.group);
    int srcRank = ctx->opInter->EvaluateSymbolicScalar(attr.srcRank);
    context->Get(srcRank, in->GetSize() * BytesOf(in->GetDataType()));
}
REGISTER_CALC_OP(OP_SHMEM_GET, Opcode::OP_SHMEM_GET, ExecuteOpShmemGet);

}