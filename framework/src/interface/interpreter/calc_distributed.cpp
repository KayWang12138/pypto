/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file calc_distributed.cpp
 * \brief Distributed communication operations implementation for precision tool
 */

#include "interface/interpreter/function.h"
#include "interface/interpreter/operation.h"
#include "interface/interpreter/rank_info.h"
#include "interface/interpreter/shmem_manager.h"
#include "interface/interpreter/verify_error.h"
#include "tilefwk/pypto_fwk_log.h"
#include <cstring>
#include <vector>
#include <atomic>

namespace npu::tile_fwk {

// Helper function to calculate byte size
static size_t GetByteSize(DataType dataType, const std::vector<int64_t>& shape) {
    size_t elementSize = 0;
    switch (dataType) {
        case DT_FP32: elementSize = sizeof(float); break;
        case DT_FP16: elementSize = sizeof(uint16_t); break;
        case DT_INT32: elementSize = sizeof(int32_t); break;
        case DT_INT16: elementSize = sizeof(int16_t); break;
        case DT_INT8: elementSize = sizeof(int8_t); break;
        case DT_UINT8: elementSize = sizeof(uint8_t); break;
        case DT_BOOL: elementSize = sizeof(bool); break;
        case DT_BF16: elementSize = sizeof(uint16_t); break;
        default: ASSERT(ExecuteOperationScene::UNSUPPORTED_DATATYPE, false);
    }
    
    size_t totalSize = elementSize;
    for (size_t i = 0; i < shape.size(); ++i) {
        totalSize *= shape[i];
    }
    return totalSize;
}

// Helper function for atomic add
static void AtomicAdd(void* dstAddr, void* srcAddr, size_t byteSize, DataType dataType) {
    switch (dataType) {
        case DT_FP32: {
            std::atomic<float>* dstAtomic = static_cast<std::atomic<float>*>(dstAddr);
            float* src = static_cast<float*>(srcAddr);
            size_t count = byteSize / sizeof(float);
            for (size_t i = 0; i < count; ++i) {
                float expected = dstAtomic[i].load(std::memory_order_relaxed);
                float desired = expected + src[i];
                while (!dstAtomic[i].compare_exchange_weak(expected, desired,
                    std::memory_order_acq_rel, std::memory_order_relaxed)) {
                    desired = expected + src[i];
                }
            }
            break;
        }
        case DT_INT32: {
            std::atomic<int32_t>* dstAtomic = static_cast<std::atomic<int32_t>*>(dstAddr);
            int32_t* src = static_cast<int32_t*>(srcAddr);
            size_t count = byteSize / sizeof(int32_t);
            for (size_t i = 0; i < count; ++i) {
                dstAtomic[i].fetch_add(src[i], std::memory_order_acq_rel);
            }
            break;
        }
        default:
            // For other types, use memcpy
            memcpy(dstAddr, srcAddr, byteSize);
            break;
    }
}

// OP_SHMEM_PUT
void ExecuteOpShmemPut(ExecuteOperationContext* ctx) {
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH,
           ctx->ioperandDataViewList->size() >= 2);
    
    auto inputData = ctx->ioperandDataViewList->at(0);
    auto shmemData = ctx->ioperandDataViewList->at(1);
    auto outputData = ctx->ooperandInplaceDataViewList->size() > 0 ?
                       ctx->ooperandInplaceDataViewList->at(0) : nullptr;

    // Get attributes
    std::string groupName = ctx->op->GetStringAttribute("group_name");
    int dstRankId = ctx->op->GetIntAttribute("dst_rank");
    std::vector<int64_t> offset = ctx->op->GetIntListAttribute("offset");
    std::vector<int64_t> shape = ctx->op->GetIntListAttribute("shape");
    AtomicType atomicType = ctx->op->GetEnumAttribute<AtomicType>("atomic_type");

    // Get shared memory manager
    ShmemManager* shmemManager = ShmemManager::GetInstance();
    void* dstAddr = shmemManager->GetDataPtr(groupName, dstRankId, offset);

    // Execute data copy
    void* srcAddr = inputData->GetData()->data();
    size_t byteSize = GetByteSize(inputData->GetDataType(), shape);

    if (atomicType == AtomicType::ADD) {
        AtomicAdd(dstAddr, srcAddr, byteSize, inputData->GetDataType());
    } else {
        memcpy(dstAddr, srcAddr, byteSize);
    }

    // Set output (pred token)
    if (outputData != nullptr) {
        memcpy(outputData->GetData()->data(), srcAddr, byteSize);
    }
}

// OP_SHMEM_GET
void ExecuteOpShmemGet(ExecuteOperationContext* ctx) {
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH,
           ctx->ioperandDataViewList->size() >= 1);
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH,
           ctx->ooperandInplaceDataViewList->size() >= 1);
    
    auto outputData = ctx->ooperandInplaceDataViewList->at(0);
    auto shmemData = ctx->ioperandDataViewList->at(0);

    // Get attributes
    std::string groupName = ctx->op->GetStringAttribute("group_name");
    int srcRankId = ctx->op->GetIntAttribute("src_rank");
    std::vector<int64_t> offset = ctx->op->GetIntListAttribute("offset");
    std::vector<int64_t> shape = ctx->op->GetIntListAttribute("shape");
    AtomicType atomicType = ctx->op->GetEnumAttribute<AtomicType>("atomic_type");

    // Get shared memory manager
    ShmemManager* shmemManager = ShmemManager::GetInstance();
    void* srcAddr = shmemManager->GetDataPtr(groupName, srcRankId, offset);

    // Execute data copy
    void* dstAddr = outputData->GetData()->data();
    size_t byteSize = GetByteSize(outputData->GetDataType(), shape);

    if (atomicType == AtomicType::ADD) {
        AtomicAdd(dstAddr, srcAddr, byteSize, outputData->GetDataType());
    } else {
        memcpy(dstAddr, srcAddr, byteSize);
    }
}

// OP_SHMEM_PUT_UB2GM
void ExecuteOpShmemPutUb2Gm(ExecuteOperationContext* ctx) {
    // Similar to OP_SHMEM_PUT but from UB
    ExecuteOpShmemPut(ctx);
}

// OP_SHMEM_GET_GM2UB
void ExecuteOpShmemGetGm2Ub(ExecuteOperationContext* ctx) {
    // Similar to OP_SHMEM_GET but to UB
    ExecuteOpShmemGet(ctx);
}

// OP_SHMEM_SIGNAL
void ExecuteOpShmemSignal(ExecuteOperationContext* ctx) {
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH,
           ctx->ioperandDataViewList->size() >= 1);
    
    auto shmemSignal = ctx->ioperandDataViewList->at(0);

    // Get attributes
    std::string groupName = ctx->op->GetStringAttribute("group_name");
    auto dstPeList = ctx->op->GetIntListAttribute("dst_pe");
    int dstRankId = dstPeList[0];
    std::vector<int64_t> offset = ctx->op->GetIntListAttribute("offset");
    int32_t signalValue = ctx->op->GetIntAttribute("signal");
    AtomicType atomicType = ctx->op->GetEnumAttribute<AtomicType>("sig_op");

    // Get shared memory manager
    RankInfo* rankInfo = RankInfo::GetInstance();
    ShmemManager* shmemManager = ShmemManager::GetInstance();
    void* signalAddr = shmemManager->GetSignalPtr(
        groupName, rankInfo->GetRankId(), dstRankId, offset);

    // Execute signal operation
    if (atomicType == AtomicType::ADD) {
        shmemManager->AtomicAdd(signalAddr, signalValue);
    } else {
        shmemManager->AtomicSet(signalAddr, signalValue);
    }
}

// OP_SHMEM_WAIT_UNTIL
void ExecuteOpShmemWaitUntil(ExecuteOperationContext* ctx) {
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH,
           ctx->ioperandDataViewList->size() >= 1);
    
    auto shmemSignal = ctx->ioperandDataViewList->at(0);

    // Get attributes
    std::string groupName = ctx->op->GetStringAttribute("group_name");
    std::vector<int64_t> offset = ctx->op->GetIntListAttribute("offset");
    int32_t expectedValue = ctx->op->GetIntAttribute("cmp_value");
    bool resetSignal = ctx->op->GetBoolAttribute("clear_signal");

    // Get shared memory manager
    RankInfo* rankInfo = RankInfo::GetInstance();
    ShmemManager* shmemManager = ShmemManager::GetInstance();
    void* signalAddr = shmemManager->GetSignalPtr(
        groupName, rankInfo->GetRankId(), rankInfo->GetRankId(), offset);

    // Execute wait operation
    shmemManager->WaitUntil(signalAddr, expectedValue, resetSignal);
}

// OP_SHMEM_SET
void ExecuteOpShmemSet(ExecuteOperationContext* ctx) {
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH,
           ctx->ioperandDataViewList->size() >= 1);
    
    auto shmemData = ctx->ioperandDataViewList->at(0);

    // Get attributes
    std::string groupName = ctx->op->GetStringAttribute("group_name");
    std::vector<int64_t> offset = ctx->op->GetIntListAttribute("offset");
    std::vector<int64_t> shape = ctx->op->GetIntListAttribute("shape");

    // Get shared memory manager
    ShmemManager* shmemManager = ShmemManager::GetInstance();
    void* dataAddr = shmemManager->GetDataPtr(groupName, 0, offset);

    // Clear memory
    size_t byteSize = GetByteSize(shmemData->GetDataType(), shape);
    memset(dataAddr, 0, byteSize);
}

// Register distributed operations
REGISTER_CALC_OP(OP_SHMEM_PUT, Opcode::OP_SHMEM_PUT, ExecuteOpShmemPut);
REGISTER_CALC_OP(OP_SHMEM_GET, Opcode::OP_SHMEM_GET, ExecuteOpShmemGet);
REGISTER_CALC_OP(OP_SHMEM_PUT_UB2GM, Opcode::OP_SHMEM_PUT_UB2GM, ExecuteOpShmemPutUb2Gm);
REGISTER_CALC_OP(OP_SHMEM_GET_GM2UB, Opcode::OP_SHMEM_GET_GM2UB, ExecuteOpShmemGetGm2Ub);
REGISTER_CALC_OP(OP_SHMEM_SIGNAL, Opcode::OP_SHMEM_SIGNAL, ExecuteOpShmemSignal);
REGISTER_CALC_OP(OP_SHMEM_WAIT_UNTIL, Opcode::OP_SHMEM_WAIT_UNTIL, ExecuteOpShmemWaitUntil);
REGISTER_CALC_OP(OP_SHMEM_SET, Opcode::OP_SHMEM_SET, ExecuteOpShmemSet);

} // namespace npu::tile_fwk
