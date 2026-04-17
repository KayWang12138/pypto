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
#include <iostream>
#include "interface/interpreter/operation.h"
#include "tensor/symbolic_scalar.h"
#include "tilefwk/error.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/comm_group_recorder.h"
#include "calc.h"
#include "communication.h"
#include "interface/operation/distributed/distributed_common.h"


namespace npu::tile_fwk {

LogicalTensorDataPtr ConvertTensorData(LogicalTensorDataPtr src, const std::vector<int64_t>& targetShape, DataType targetDtype) {
    if (src->GetShape() == targetShape && src->GetDataType() == targetDtype) {
        return src;
    }

    size_t srcSize = src->GetSize() * BytesOf(src->GetDataType());
    size_t targetSize = 1;
    for (auto dim : targetShape) {
        targetSize *= dim;
    }
    targetSize *= BytesOf(targetDtype);

    ASSERT(srcSize == targetSize) << "Source and target tensor sizes do not match! srcSize: " << srcSize << ", targetSize: " << targetSize;

    RawTensorDataPtr targetData = std::make_shared<RawTensorData>(targetDtype, targetShape);
    // StringUtils::DataCopy(targetData->data(), targetSize, src->GetData()->data(), srcSize);
    targetData->assign(src->GetData()->data(), src->GetData()->data() + srcSize);

    return std::make_shared<LogicalTensorData>(targetData);
}

void ExecuteOpBindTensor(ExecuteOperationContext *ctx) {
    // std::cout << "=== ExecuteOpBindTensor running ..." << std::endl;
    // ASSERT(ctx->ioperandDataViewList->size() == 0);
    // ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    // LogicalTensorDataPtr out = ctx->ooperandInplaceDataViewList->at(0);

    // SymbolicScalar attr = ctx->op->GetSymbolicScalarAttribute(OpAttributeKey::bindTensor);
    // std::vector<uint64_t> parameters = UnBind(ctx, attr);
    // uint64_t groupIndex = parameters[0];
    // uint64_t memType = parameters[1];
    // uint64_t slotSize = parameters[2];
    // const auto &groupNames = Distributed::CommGroupRecorder::GetInstance().Output();
    // ASSERT(groupIndex < static_cast<uint64_t>(groupNames.size()));
    // const std::string &groupName = groupNames[groupIndex];
    // if (memType == 1) {
    //     std::cout << "Alloc " << slotSize << "B for " << groupName << std::endl;
    //     RawTensorDataPtr tmp = SimulationCommManager::Instance().Alloc(groupName, slotSize);
    //     *out = LogicalTensorData(tmp, out->GetShape(), out->GetValidShape(), out->GetOffset());
    // }
    // if (memType == 0) {
    //     std::cout << "AllocSignal " << slotSize << "B for " << groupName << std::endl;
    //     RawTensorDataPtr tmp = SimulationCommManager::Instance().AllocSignal(groupName, slotSize);
    //     *out = LogicalTensorData(tmp, out->GetShape(), out->GetValidShape(), out->GetOffset());
    // }
    // std::cout << "=== ExecuteOpBindTensor exited." << std::endl;
    (void) ctx;
}
REGISTER_CALC_OP(OP_BIND_TENSOR, Opcode::OP_BIND_TENSOR, ExecuteOpBindTensor);

void ExecuteOpShmemSet(ExecuteOperationContext *ctx) {
    std::cout << "=== ExecuteOpShmemSet running ..." << std::endl;

    ASSERT(ctx->ioperandDataViewList->size() == 2);
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1 || ctx->ooperandInplaceDataViewList->size() == 2);
    auto &shm = ctx->ioperandDataViewList->at(1);

    Distributed::ShmemSetAttr attr;
    ctx->op->GetAttr(OpAttributeKey::distOpAttr, attr);
    
    std::shared_ptr<SimulationCommContext> context = SimulationCommManager::Instance().GetCommContext(attr.group);
    size_t slotSize = shm->GetSize() * BytesOf(shm->GetDataType());
    if (!attr.isSetData) {
        std::cout << "Set rank " << context->GetRank() << "'s signal as 0 from " << shm->GetStorageOffset() << " to " << shm->GetStorageOffset() + slotSize << std::endl;
        context->Signal(context->GetRank(), 0, slotSize, shm->GetStorageOffset());
    } else {
        std::cout << "Set rank " << context->GetRank() << "'s data as 0 from " << shm->GetStorageOffset() << " to " << shm->GetStorageOffset() + slotSize << std::endl;
        context->Set(context->GetRank(), 0, slotSize, shm->GetStorageOffset());
    }

    std::cout << "=== ExecuteOpShmemSet exited ..." << std::endl;
}
REGISTER_CALC_OP(OP_SHMEM_SET, Opcode::OP_SHMEM_SET, ExecuteOpShmemSet);

void ExecuteOpShmemPut(ExecuteOperationContext *ctx) {
    std::cout << "=== ExecuteOpShmemPut running ..." << std::endl;

    ASSERT(ctx->ioperandDataViewList->size() == 3);
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    auto &in = ctx->ioperandDataViewList->at(1);
    auto &shm = ctx->ioperandDataViewList->at(2);

    Distributed::ShmemPutAttr attr;
    ctx->op->GetAttr(OpAttributeKey::distOpAttr, attr);
    
    std::shared_ptr<SimulationCommContext> context = SimulationCommManager::Instance().GetCommContext(attr.group);
    int dstRank = ctx->opInter->EvaluateSymbolicScalar(attr.ownerRank);
    int atomicType = 0;
    if (attr.atomicType == Distributed::AtomicType::SET) {
        atomicType = 0;
    }
    if (attr.atomicType == Distributed::AtomicType::ADD) {
        atomicType = 1;
    }
    
    auto castedIn = LogicalTensorData::CreateEmpty(shm->GetDataType(), shm->GetShape(), shm->GetValidShape(), shm->GetShape());
    calc::Cast(castedIn, in);

    std::cout << "Put data " << in <<" to dstRank " << dstRank << " from " << shm->GetStorageOffset() << " to " << shm->GetStorageOffset() + castedIn->GetSize() * BytesOf(castedIn->GetDataType()) << " , atomic type: " << atomicType << std::endl;
    context->Put(castedIn, dstRank, shm->GetStorageOffset(), atomicType);

    std::cout << "=== ExecuteOpShmemPut exited ..." << std::endl;
}
REGISTER_CALC_OP(OP_SHMEM_PUT, Opcode::OP_SHMEM_PUT, ExecuteOpShmemPut);

void ExecuteOpShmemSignal(ExecuteOperationContext *ctx) {
    std::cout << "=== ExecuteOpShmemSignal running ..." << std::endl;

    ASSERT(ctx->ioperandDataViewList->size() == 2);
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 2 || ctx->ooperandInplaceDataViewList->size() == 1);
    auto &shm = ctx->ioperandDataViewList->at(1);

    Distributed::ShmemSignalAttr attr;
    ctx->op->GetAttr(OpAttributeKey::distOpAttr, attr);
    
    std::shared_ptr<SimulationCommContext> context = SimulationCommManager::Instance().GetCommContext(attr.group);
    int dstRank = ctx->opInter->EvaluateSymbolicScalar(attr.ownerRank);
    int atomicType = 0;
    if (attr.atomicType == Distributed::AtomicType::SET) {
        atomicType = 0;
    }
    if (attr.atomicType == Distributed::AtomicType::ADD) {
        atomicType = 1;
    }
    int value = attr.signalValue;
    bool notifyAll = attr.notifyAll;
    size_t slotSize = shm->GetSize() * BytesOf(shm->GetDataType());
    std::cout << "Signal " << value << " to rank " << dstRank << " from offset " << shm->GetStorageOffset() << " to " << shm->GetStorageOffset() + slotSize << "; atomicType: " << atomicType << ", notifyAll: " << notifyAll << std::endl;
    context->Signal(dstRank, value, slotSize, shm->GetStorageOffset(), atomicType, notifyAll);

    std::cout << "=== ExecuteOpShmemSignal exited ..." << std::endl;
}
REGISTER_CALC_OP(OP_SHMEM_SIGNAL, Opcode::OP_SHMEM_SIGNAL, ExecuteOpShmemSignal);

void ExecuteOpShmemWaitUntil(ExecuteOperationContext *ctx) {
    std::cout << "=== ExecuteOpShmemWaitUntil running ..." << std::endl;

    ASSERT(ctx->ioperandDataViewList->size() == 2);
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    auto &shm = ctx->ioperandDataViewList->at(1);

    Distributed::ShmemWaitUntilAttr attr;
    ctx->op->GetAttr(OpAttributeKey::distOpAttr, attr);
    
    std::shared_ptr<SimulationCommContext> context = SimulationCommManager::Instance().GetCommContext(attr.group);
    int srcRank = context->GetRank();
    int expect = attr.expectedSum;
    bool reset = attr.resetSignal;
    size_t slotSize =  shm->GetSize() * BytesOf(shm->GetDataType());

    std::cout << "Rank " << srcRank << " is waiting for " << expect << " from " << shm->GetStorageOffset() << " to " << shm->GetStorageOffset() + slotSize << " reset: " << reset << std::endl;
    uint64_t taskId = context->WaitAsync(srcRank, expect, slotSize, shm->GetStorageOffset(), reset);
    std::cout << "WaitUntil add task " << taskId << std::endl;
    
    // Register the WaitUntil task with the operation for dependency resolution
    SimulationCommManager::RegisterWaitTask(ctx->op, context, taskId);

    std::cout << "=== ExecuteOpShmemWaitUntil exited ..." << std::endl;
}
REGISTER_CALC_OP(OP_SHMEM_WAIT_UNTIL, Opcode::OP_SHMEM_WAIT_UNTIL, ExecuteOpShmemWaitUntil);

void ExecuteOpShmemGet(ExecuteOperationContext *ctx) {
    std::cout << "=== ExecuteOpShmemGet running ..." << std::endl;

    ASSERT(ctx->ioperandDataViewList->size() == 2);
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 2 || ctx->ooperandInplaceDataViewList->size() == 1);
    auto &shm = ctx->ioperandDataViewList->at(1);
    auto out = ctx->ooperandInplaceDataViewList->at(0);

    Distributed::ShmemGetAttr attr;
    ctx->op->GetAttr(OpAttributeKey::distOpAttr, attr);
    
    std::shared_ptr<SimulationCommContext> context = SimulationCommManager::Instance().GetCommContext(attr.group);
    int srcRank = ctx->opInter->EvaluateSymbolicScalar(attr.ownerRank);
    size_t slotSize = out->GetSize() * BytesOf(out->GetDataType());

    std::cout << "Get " << srcRank << "'s data from " << srcRank << " from " << shm->GetStorageOffset() << " to " << shm->GetStorageOffset() + slotSize << std::endl;
    LogicalTensorDataPtr tmp = context->Get(srcRank, slotSize, shm->GetStorageOffset());
    calc::Copy(out, tmp);

    std::cout << "=== ExecuteOpShmemGet exited ..." << std::endl;
}
REGISTER_CALC_OP(OP_SHMEM_GET, Opcode::OP_SHMEM_GET, ExecuteOpShmemGet);
REGISTER_CALC_OP(OP_SHMEM_GET_GM2UB, Opcode::OP_SHMEM_GET_GM2UB, ExecuteOpShmemGet);

}