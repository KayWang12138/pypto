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
 * \file test_shmem_wait_until.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <atomic>

#include "machine/device/distributed/common.h"
#include "machine/device/distributed/shmem_wait_until.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "tileop/distributed/hccl_context.h"
#include "machine/device/dynamic/aicore_manager.h"

namespace {

class TestResourceHolder {
private:
    struct TensorResources {
        std::vector<int32_t> rawAddr;
        uint32_t rawShape0, rawShape1, rawShape2, rawShape3;
        npu::tile_fwk::Distributed::TensorInfo info;
    };

    struct CodeResources {
        std::unique_ptr<int32_t[]> data;
        npu::tile_fwk::dynamic::DevRelocVector<int32_t> aicpuCode;
    };

    struct TaskResources {
        std::unique_ptr<void, decltype(&free)> buffer;
        npu::tile_fwk::DynFuncData* funcData;
        std::unique_ptr<npu::tile_fwk::dynamic::DynDeviceTask> task;
    };

    struct FuncDataResources {
        std::unique_ptr<uint64_t[]> exprTbl;
        std::unique_ptr<TileOp::HcclCombinOpParam> hcclParam;
        std::unique_ptr<uint64_t[]> rawTensorAddrHolder;
        std::unique_ptr<npu::tile_fwk::DevRawTensorDesc[]> rawTensorDescHolder;
        std::unique_ptr<uint64_t[]> opAttrs;
    };

    TensorResources tensorRes_;
    CodeResources codeRes_;
    TaskResources taskRes_;
    FuncDataResources funcDataRes_;
    std::unique_ptr<npu::tile_fwk::dynamic::DeviceWorkspaceAllocator> allocator_;
    std::unique_ptr<npu::tile_fwk::Distributed::ShmemWaitUntil> shmemWaitUntil_;
    std::unique_ptr<npu::tile_fwk::dynamic::AicpuTaskManager> aicpuTaskManager_;
    std::unique_ptr<npu::tile_fwk::dynamic::AiCoreManager> aicoreManager_;

public:
    TestResourceHolder(uint32_t rankSize) {
        initializeTensorResources(rankSize);
        initializeCodeResources(rankSize);
        initializeAllocatorAndManagers();
        initializeTaskResources();
        initializeFuncDataResources();
        initializeShmemWaitUntil();
    }

    npu::tile_fwk::Distributed::ShmemWaitUntil* getShmemWaitUntil() { 
        return shmemWaitUntil_.get(); 
    }
    
    npu::tile_fwk::dynamic::AiCoreManager* getAiCoreManager() { 
        return aicoreManager_.get(); 
    }
    
    npu::tile_fwk::DynFuncData* getFuncData() { 
        return taskRes_.funcData; 
    }
    
    const npu::tile_fwk::dynamic::DevRelocVector<int32_t>& getAicpuCode() { 
        return codeRes_.aicpuCode; 
    }
    
    uint64_t* getOpAttrs() { 
        return funcDataRes_.opAttrs.get(); 
    }

private:
    void initializeTensorResources(uint32_t rankSize) {
        tensorRes_.rawShape0 = rankSize;
        tensorRes_.rawShape1 = rankSize;
        tensorRes_.rawShape2 = 4;
        tensorRes_.rawShape3 = 8;
        
        tensorRes_.rawAddr.resize(tensorRes_.rawShape1 * tensorRes_.rawShape2 * tensorRes_.rawShape3, 0);
        
        tensorRes_.info.offset = {0, 1, 0, 0};
        tensorRes_.info.dim = 3;
        tensorRes_.info.expectedSum = 0;
        tensorRes_.info.resetSignal = false;
        tensorRes_.info.rawIndex = 0;
        tensorRes_.info.rawAddr = reinterpret_cast<uint64_t>(tensorRes_.rawAddr.data());

        if (tensorRes_.rawAddr.size() > 0) {
            size_t offset = tensorRes_.offset[1] * tensorRes_.rawShape2 * tensorRes_.rawShape3 + 
                           tensorRes_.offset[2] * tensorRes_.rawShape3 + tensorRes_.offset[3];
            if (offset < tensorRes_.rawAddr.size()) {
                tensorRes_.rawAddr[offset] = 1;
            }
        }
    }

    void initializeCodeResources(uint32_t rankSize) {
        constexpr size_t codeSize = 17;
        codeRes_.data = std::make_unique<int32_t[]>(codeSize);
        uint32_t initData[codeSize] = {153, 2, 2, 44, 4, 2, 18, 4, 0, 2, 1, 0, 4, rankSize, rankSize, 4, 8};
        std::copy(initData, initData + codeSize, codeRes_.data.get());
        codeRes_.aicpuCode = npu::tile_fwk::dynamic::DevRelocVector<int32_t>(codeSize, codeRes_.data.get());
    }

    void initializeAllocatorAndManagers() {
        allocator_ = std::make_unique<npu::tile_fwk::dynamic::DeviceWorkspaceAllocator>();
        aicpuTaskManager_ = std::make_unique<npu::tile_fwk::dynamic::AicpuTaskManager>();
        aicoreManager_ = std::make_unique<npu::tile_fwk::dynamic::AiCoreManager>(*aicpuTaskManager_);
    }

    void initializeTaskResources() {
        size_t headerSize = sizeof(npu::tile_fwk::DynFuncHeader);
        size_t dataSize = sizeof(npu::tile_fwk::DynFuncData);
        
        void* bufferPtr = malloc(headerSize + dataSize);
        if (!bufferPtr) {
            throw std::runtime_error("Failed to allocate task buffer");
        }
        
        taskRes_.buffer = {bufferPtr, free};
        auto* header = new(taskRes_.buffer.get()) npu::tile_fwk::DynFuncHeader();
        taskRes_.funcData = new(header + 1) npu::tile_fwk::DynFuncData();

        taskRes_.task = std::make_unique<npu::tile_fwk::dynamic::DynDeviceTask>(*allocator_);
        taskRes_.task->dynFuncDataList = header;
        taskRes_.task->dynFuncDataList[0].seqNo = 1;
        taskRes_.task->dynFuncDataList[0].funcNum = 1;
        taskRes_.task->dynFuncDataList[0].funcSize = 1u;
        taskRes_.task->dynFuncDataList[0].cceBinary = nullptr;
    }

    void initializeFuncDataResources() {
        constexpr size_t exprTblSize = 50;
        constexpr size_t opAttrsLength = 17;
        
        funcDataRes_.exprTbl = std::make_unique<uint64_t[]>(exprTblSize);
        funcDataRes_.hcclParam = std::make_unique<TileOp::HcclCombinOpParam>();
        funcDataRes_.rawTensorAddrHolder = std::make_unique<uint64_t[]>(1);
        funcDataRes_.rawTensorDescHolder = std::make_unique<npu::tile_fwk::DevRawTensorDesc[]>(1);
        funcDataRes_.opAttrs = std::make_unique<uint64_t[]>(opAttrsLength);
        
        funcDataRes_.hcclParam->rankNum = 0;
        funcDataRes_.hcclParam->windowsIn[0] = tensorRes_.info.rawAddr;
        
        uint64_t initAttrs[opAttrsLength] = {0, 0, 1, 0, 0, 1, 1, 1, 8, 
            tensorRes_.rawShape0, tensorRes_.rawShape1, tensorRes_.rawShape2, tensorRes_.rawShape3, 0, 0, 0, 0};
        std::copy(initAttrs, initAttrs + opAttrsLength, funcDataRes_.opAttrs.get());
        
        taskRes_.funcData->exprTbl = funcDataRes_.exprTbl.get();
        taskRes_.funcData->hcclContext[0] = reinterpret_cast<uint64_t>(funcDataRes_.hcclParam.get());
        taskRes_.funcData->rawTensorAddr = funcDataRes_.rawTensorAddrHolder.get();
        taskRes_.funcData->rawTensorDesc = funcDataRes_.rawTensorDescHolder.get();
        
        if (funcDataRes_.rawTensorAddrHolder) {
            funcDataRes_.rawTensorAddrHolder[0] = 0;
        }
        if (funcDataRes_.rawTensorDescHolder) {
            funcDataRes_.rawTensorDescHolder[0] = {0, 0};
        }
    }

    void initializeShmemWaitUntil() {
        shmemWaitUntil_ = std::make_unique<npu::tile_fwk::Distributed::ShmemWaitUntil>();
        shmemWaitUntil_->Init(taskRes_.task.get());
    }
};

class ThreadSafeTaskProcessor {
private:
    std::mutex mutex_;
    std::atomic<bool> initialized_{false};
    
public:
    struct TaskAttributes {
        std::unique_ptr<int32_t[]> opAtrrOffsets;
        std::unique_ptr<uint64_t[]> opAttrsCopy;
    };

    std::vector<TaskAttributes> prepareTasks(
        uint32_t tileOpCount,
        npu::tile_fwk::Distributed::ShmemWaitUntil* shmemWaitUntil,
        const npu::tile_fwk::dynamic::DevRelocVector<int32_t>& aicpuCode,
        npu::tile_fwk::DynFuncData* funcData,
        uint64_t* opAttrsPtr) {
        
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<TaskAttributes> taskAttrs;
        constexpr size_t opAttrsLength = 17;

        for (uint32_t taskId = 0; taskId < tileOpCount; ++taskId) {
            TaskAttributes attr;
            
            attr.opAtrrOffsets = std::make_unique<int32_t[]>(taskId + 1);
            if (!attr.opAtrrOffsets) {
                throw std::runtime_error("Failed to allocate opAtrrOffsets");
            }
            attr.opAtrrOffsets[taskId] = 0;

            int opAttrsSize = 1 + attr.opAtrrOffsets[taskId] + opAttrsLength;
            attr.opAttrsCopy = std::make_unique<uint64_t[]>(opAttrsSize);
            if (!attr.opAttrsCopy) {
                throw std::runtime_error("Failed to allocate opAttrsCopy");
            }

            if (opAttrsPtr) {
                std::copy(opAttrsPtr, opAttrsPtr + opAttrsLength, 
                         attr.opAttrsCopy.get() + attr.opAtrrOffsets[taskId]);
            }

            funcData->opAtrrOffsets = attr.opAtrrOffsets.get();
            funcData->opAttrs = attr.opAttrsCopy.get();

            if (shmemWaitUntil) {
                shmemWaitUntil->PrepareTask(taskId, aicpuCode);
            }

            taskAttrs.push_back(std::move(attr));
        }

        initialized_ = true;
        return taskAttrs;
    }

    void runTests(uint32_t tileOpCount,
                 npu::tile_fwk::Distributed::ShmemWaitUntil* shmemWaitUntil,
                 npu::tile_fwk::dynamic::AiCoreManager* aicoreManager) {
        if (!initialized_.load()) {
            throw std::runtime_error("Tasks not prepared before running");
        }

        std::lock_guard<std::mutex> lock(mutex_);
        for (uint32_t taskId = 0; taskId < tileOpCount; ++taskId) {
            if (shmemWaitUntil) {
                shmemWaitUntil->EnqueueOp(taskId);
                shmemWaitUntil->PollCompleted(*aicoreManager);
            }
        }
    }
};

void TestShmemWaitUntil(uint32_t tileOpCount) {
    const uint32_t rankSize = 4;
    
    try {
        TestResourceHolder resourceHolder(rankSize);
        ThreadSafeTaskProcessor taskProcessor;
        auto taskAttrs = taskProcessor.prepareTasks(
            tileOpCount,
            resourceHolder.getShmemWaitUntil(),
            resourceHolder.getAicpuCode(),
            resourceHolder.getFuncData(),
            resourceHolder.getOpAttrs());
        
        taskProcessor.runTests(
            tileOpCount,
            resourceHolder.getShmemWaitUntil(),
            resourceHolder.getAiCoreManager());
            
    } catch (const std::exception& e) {
        GTEST_FAIL() << "Test failed with exception: " << e.what();
    }
}

TEST(ShmemWaitUntilTest, BasicFunctionality) {
    constexpr uint32_t tileOpCount = 1;
    TestShmemWaitUntil(tileOpCount);
}
} // namespace