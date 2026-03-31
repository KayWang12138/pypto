/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to * License for details. You may not use this file except in compliance with * License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in * root of the software repository for the full text of the License.
 */

/*!
 * \file calc_distributed.cpp
 * \brief Distributed operations interpreter for verification tool
 */

#include "operation.h"
#include "shared_memory_context.h"
#include "calc.h"
#include <thread>

namespace npu::tile_fwk {

void HandleBindTensor(ExecuteOperationContext* ctx) {
    auto& shmemCtx = SharedMemoryContext::GetInstance();
    
    if (!ctx->op->HasAttribute(OpAttributeKey::bindTensor)) {
        VERIFY_LOGE("OP_BIND_TENSOR missing bindTensor attribute");
        return;
    }
    
    auto bindTensor = ctx->op->GetSymbolicScalarAttribute(OpAttributeKey::bindTensor);
    
    VERIFY_LOGI("HandleBindTensor: bindTensor=%lu", bindTensor.GetValue());
    
    uint64_t groupIndex = (bindTensor.GetValue() >> 54) & 0x3;
    uint64_t memType = (bindTensor.GetValue() >> 56) & 0x3;
    uint64_t size = bindTensor.GetValue() & ((1ULL << 54) - 1);
    
    if (!shmemCtx.IsInitialized()) {
        VERIFY_LOGI("Shared memory context not initialized, skipping OP_BIND_TENSOR");
        return;
    }
    
    void* vaddr = shmemCtx.AllocateShmem(
        static_cast<int>(groupIndex), 
        static_cast<ShmemMemType>(memType), 
        size);
    
    VERIFY_LOGI("BindTensor allocated: groupIndex=%lu, memType=%lu, size=%lu, vaddr=%p",
                groupIndex, memType, size, vaddr);
    
    // 获取输出 tensor 的 LogicalTensorData
    if (ctx->op->GetOOperands().empty()) {
        VERIFY_LOGE("OP_BIND_TENSOR: no output operands");
        return;
    }
    
    auto outputTensor = ctx->op->GetOOperands()[0];
    
    // 从 ooperandDataViewList 获取已创建的 LogicalTensorData
    if (ctx->ooperandDataViewList == nullptr || ctx->ooperandDataViewList->empty()) {
        VERIFY_LOGE("OP_BIND_TENSOR: no output data view");
        return;
    }
    
    auto outputDataView = ctx->ooperandDataViewList->at(0);
    auto rawData = outputDataView->GetData();
    
    // 计算共享内存 buffer 的实际指针
    void* shmemPtr = nullptr;
    
    if (memType == 0) { // DATA
        shmemPtr = shmemCtx.GetDataPtr(static_cast<int>(groupIndex));
    } else { // STATUS
        shmemPtr = shmemCtx.GetStatusPtr(static_cast<int>(groupIndex));
    }
    
    if (shmemPtr != nullptr) {
        // 解析 vaddr 获取 offset
        uint64_t offset = vaddr & ((1ULL << 54) - 1);
        uint8_t* bufferPtr = static_cast<uint8_t*>(shmemPtr) + offset;
        
        // 设置外部 buffer
        rawData->SetExternalBuffer(bufferPtr, size);
        
        VERIFY_LOGI("BindTensor: set external buffer for output tensor, groupIndex=%lu, memType=%lu, size=%lu",
                    groupIndex, memType, size);
    }
}

void HandleShmemPut(ExecuteOperationContext* ctx) {
    auto& shmemCtx = SharedMemoryContext::GetInstance();
    
    if (ctx->ioperandDataViewList == nullptr || ctx->ioperandDataViewList->size() < 3) {
        VERIFY_LOGE("OP_SHMEM_PUT: insufficient input operands");
        return;
    }
    
    auto src = ctx->ioperandDataViewList->at(0);
    auto dst = ctx->ioperandDataViewList->at(2);
    
    if (src == nullptr || dst == nullptr) {
        VERIFY_LOGE("OP_SHMEM_PUT: null data view");
        return;
    }
    
    int dstPe = 0;
    if (ctx->op->HasAttribute(OpAttributeKey::dstPe)) {
        dstPe = ctx->op->GetIntAttribute(OpAttributeKey::dstPe);
    }
    
    if (!shmemCtx.IsInitialized()) {
        VERIFY_LOGI("Shared memory context not initialized, OP_SHMEM_PUT using local copy");
        std::copy(src->GetData()->begin(), src->GetData()->end(),
                  dst->GetData()->begin());
        return;
    }
    
    if (dstPe == shmemCtx.GetCurrentRank()) {
        VERIFY_LOGI("OP_SHMEM_PUT: local copy (dstPe=%d, currentRank=%d)", 
                    dstPe, shmemCtx.GetCurrentRank());
        calc::Copy(dst, src);
    } else {
        VERIFY_LOGI("OP_SHMEM_PUT: cross-rank copy to shared memory (dstPe=%d, currentRank=%d)",
                    dstPe, shmemCtx.GetCurrentRank());
        
        if (dst->GetData()->UsesExternalBuffer()) {
            uint8_t* dstPtr = static_cast<uint8_t*>(dst->GetData()->GetExternalBuffer());
            size_t dstSize = dst->GetData()->GetExternalBufferSize();
            
            auto shmemData = std::make_shared<RawTensorData>(
                src->GetDataType(), 
                src->GetShape()
            );
            shmemData->SetExternalBuffer(dstPtr, dstSize);
            
            auto shmemView = std::make_shared<LogicalTensorData>(
                shmemData, 
                src->GetShape(), 
                src->GetShape(), 
                std::vector<int64_t>(src->GetShape().size(), 0)
            );
            
            calc::Copy(shmemView, src);
            VERIFY_LOGI("OP_SHMEM_PUT: copied %zu bytes to shared memory", dstSize);
        } else {
            VERIFY_LOGE("OP_SHMEM_PUT: destination tensor is not using shared memory");
        }
    }
}

void HandleShmemGet(ExecuteOperationContext* ctx) {
    auto& shmemCtx = SharedMemoryContext::GetInstance();
    
    if (ctx->ioperandDataViewList == nullptr || ctx->ioperandDataViewList->empty()) {
        VERIFY_LOGE("OP_SHMEM_GET: insufficient input operands");
        return;
    }
    
    if (ctx->ooperandDataViewList == nullptr || ctx->ooperandDataViewList->empty()) {
        VERIFY_LOGE("OP_SHMEM_GET: insufficient output operands");
        return;
    }
    
    auto src = ctx->ioperandDataViewList->at(0);
    auto output = ctx->ooperandDataViewList->at(0);
    
    if (src == nullptr || output == nullptr) {
        VERIFY_LOGE("OP_SHMEM_GET: null data view");
        return;
    }
    
    int srcPe = 0;
    if (ctx->op->HasAttribute(OpAttributeKey::srcPe)) {
        srcPe = ctx->op->GetIntAttribute(OpAttributeKey::srcPe);
    }
    
    if (!shmemCtx.IsInitialized()) {
        VERIFY_LOGI("Shared memory context not initialized, OP_SHMEM_GET using local copy");
        std::copy(src->GetData()->begin(), src->GetData()->end(),
                  output->GetData()->begin());
        return;
    }
    
    if (srcPe == shmemCtx.GetCurrentRank()) {
        VERIFY_LOGI("OP_SHMEM_GET: local copy (srcPe=%d, currentRank=%d)",
                    srcPe, shmemCtx.GetCurrentRank());
        calc::Copy(output, src);
    } else {
        VERIFY_LOGI("OP_SHMEM_GET: cross-rank copy from shared memory (srcPe=%d, currentRank=%d)",
                    srcPe, shmemCtx.GetCurrentRank());
        
        if (src->GetData()->UsesExternalBuffer()) {
            uint8_t* srcPtr = static_cast<uint8_t*>(src->GetData()->GetExternalBuffer());
            size_t srcSize = src->GetData()->GetExternalBufferSize();
            
            auto shmemData = std::make_shared<RawTensorData>(
                output->GetDataType(), 
                output->GetShape()
            );
            shmemData->SetExternalBuffer(srcPtr, srcSize);
            
            auto shmemView = std::make_shared<LogicalTensorData>(
                shmemData, 
                output->GetShape(), 
                output->GetShape(), 
                std::vector<int64_t>(output->GetShape().size(), 0)
            );
            
            calc::Copy(output, shmemView);
            VERIFY_LOGI("OP_SHMEM_GET: copied %zu bytes from shared memory", srcSize);
        } else {
            VERIFY_LOGE("OP_SHMEM_GET: source tensor is not using shared memory");
        }
    }
}

void HandleShmemSignal(ExecuteOperationContext* ctx) {
    auto& shmemCtx = SharedMemoryContext::GetInstance();
    
    if (ctx->ioperandDataViewList == nullptr || ctx->ioperandDataViewList->empty()) {
        VERIFY_LOGE("OP_SHMEM_SIGNAL: insufficient input operands");
        return;
    }
    
    auto dst = ctx->ioperandDataViewList->at(0);
    
    if (dst == nullptr) {
        VERIFY_LOGE("OP_SHMEM_SIGNAL: null data view");
        return;
    }
    
    int signal = 0;
    if (ctx->op->HasAttribute(OpAttributeKey::signal)) {
        signal = ctx->op->GetIntAttribute(OpAttributeKey::signal);
    }
    
    if (!shmemCtx.IsInitialized()) {
        VERIFY_LOGI("Shared memory context not initialized, skipping OP_SHMEM_SIGNAL");
        return;
    }
    
    if (dst->GetData()->UsesExternalBuffer()) {
        uint8_t* dstPtr = static_cast<uint8_t*>(dst->GetData()->GetExternalBuffer());
        
        int32_t* signalPtr = reinterpret_cast<int32_t*>(dstPtr);
        *signalPtr = signal;
        
        VERIFY_LOGI("OP_SHMEM_SIGNAL: wrote signal=%d to shared memory", signal);
    } else {
        VERIFY_LOGE("OP_SHMEM_SIGNAL: destination tensor is not using shared memory");
    }
}

void HandleShmemWaitUntil(ExecuteOperationContext* ctx) {
    auto& shmemCtx = SharedMemoryContext::GetInstance();
    
    if (ctx->ioperandDataViewList == nullptr || ctx->ioperandDataViewList->empty()) {
        VERIFY_LOGE("OP_SHMEM_WAIT_UNTIL: insufficient input operands");
        return;
    }
    
    auto src = ctx->ioperandDataViewList->at(0);
    
    if (src == nullptr) {
        VERIFY_LOGE("OP_SHMEM_WAIT_UNTIL: null data view");
        return;
    }
    
    int expectedValue = 0;
    if (ctx->op->HasAttribute(OpAttributeKey::expectedValue)) {
        expectedValue = ctx->op->GetIntAttribute(OpAttributeKey::expectedValue);
    }
    
    if (!shmemCtx.IsInitialized()) {
        VERIFY_LOGI("Shared memory context not initialized, skipping OP_SHMEM_WAIT_UNTIL");
        return;
    }
    
    if (src->GetData()->UsesExternalBuffer()) {
        uint8_t* srcPtr = static_cast<uint8_t*>(src->GetData()->GetExternalBuffer());
        
        int32_t* signalPtr = reinterpret_cast<int32_t*>(srcPtr);
        
        int maxWaitCycles = 1000000;
        int waitCycles = 0;
        while (*signalPtr != expectedValue && waitCycles < maxWaitCycles) {
            std::this_thread::yield();
            waitCycles++;
        }
        
        if (waitCycles >= maxWaitCycles) {
            VERIFY_LOGE("OP_SHMEM_WAIT_UNTIL: timeout waiting for signal=%d, current=%d",
                        expectedValue, *signalPtr);
        } else {
            VERIFY_LOGI("OP_SHMEM_WAIT_UNTIL: signal reached expected value=%d after %d cycles",
                        expectedValue, waitCycles);
        }
    } else {
        VERIFY_LOGE("OP_SHMEM_WAIT_UNTIL: source tensor is not using shared memory");
    }
}

REGISTER_CALC_OP(BindTensor, Opcode::OP_BIND_TENSOR, HandleBindTensor);
REGISTER_CALC_OP(ShmemPut, Opcode::OP_SHMEM_PUT, HandleShmemPut);
REGISTER_CALC_OP(ShmemGet, Opcode::OP_SHMEM_GET, HandleShmemGet);
REGISTER_CALC_OP(ShmemSignal, Opcode::OP_SHMEM_SIGNAL, HandleShmemSignal);
REGISTER_CALC_OP(ShmemWaitUntil, Opcode::OP_SHMEM_WAIT_UNTIL, HandleShmemWaitUntil);

} 
