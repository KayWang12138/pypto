/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file shmem_manager.cpp
 * \brief Shared memory manager implementation for distributed precision tool
 */

#include "interface/interpreter/shmem_manager.h"
#include "interface/interpreter/rank_info.h"
#include "tilefwk/error.h"
#include "tilefwk/data_type.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <string>
#include <vector>

namespace npu::tile_fwk {

ShmemManager* ShmemManager::GetInstance() {
    static ShmemManager instance;
    return &instance;
}

size_t ShmemManager::GetDataTypeSize(uint32_t dataType) const {
    switch (dataType) {
        case DT_FP32: return sizeof(float);
        case DT_FP16: return sizeof(uint16_t);
        case DT_INT32: return sizeof(int32_t);
        case DT_INT16: return sizeof(int16_t);
        case DT_INT8: return sizeof(int8_t);
        case DT_UINT8: return sizeof(uint8_t);
        case DT_BOOL: return sizeof(bool);
        case DT_BF16: return sizeof(uint16_t);
        default: ASSERT(ExecuteOperationScene::UNSUPPORTED_DATATYPE, false);
    }
    return 0;
}

ShmemManager::ShmemRegion* ShmemManager::CreateRegion(
    const std::string& groupName, int worldSize, int rankId,
    uint32_t dataType, const std::vector<int64_t>& shape) {
    
    std::lock_guard<std::mutex> lock(mutex_);

    // 检查是否已存在
    if (regions_.find(groupName) != regions_.end()) {
        return &regions_[groupName];
    }

    ShmemRegion region;
    region.worldSize = worldSize;
    region.creatorRank = rankId;
    region.isCreator = (rankId == 0);  // rank 0 创建共享内存

    // 计算数据区域大小
    size_t elementSize = GetDataTypeSize(dataType);
    size_t dataElementCount = worldSize;
    for (auto dim : shape) {
        dataElementCount *= dim;
    }
    region.dataSize = dataElementCount * elementSize;

    // 计算信号区域大小
    size_t signalElementCount = worldSize * worldSize;
    for (auto dim : shape) {
        signalElementCount *= dim;
    }
    region.signalSize = signalElementCount * sizeof(int32_t);

    // 生成共享内存名称
    RankInfo* rankInfo = RankInfo::GetInstance();
    region.dataShmName = rankInfo->GenerateShmemName(groupName + "_data");
    region.signalShmName = rankInfo->GenerateShmemName(groupName + "_signal");

    // 创建共享内存
    if (region.isCreator) {
        // rank 0 创建共享内存
        region.dataShmFd = shm_open(region.dataShmName.c_str(),
                                       O_CREAT | O_RDWR, 0666);
        ASSERT(ExecuteOperationScene::SHMEM_CREATE_FAILED, region.dataShmFd != -1)
            << "Failed to create shared memory: " << region.dataShmName;
        ftruncate(region.dataShmFd, region.dataSize);

        region.signalShmFd = shm_open(region.signalShmName.c_str(),
                                         O_CREAT | O_RDWR, 0666);
        ASSERT(ExecuteOperationScene::SHMEM_CREATE_FAILED, region.signalShmFd != -1)
            << "Failed to create shared memory: " << region.signalShmName;
        ftruncate(region.signalShmFd, region.signalSize);
    } else {
        // 其他进程打开共享内存
        region.dataShmFd = shm_open(region.dataShmName.c_str(), O_RDWR, 0);
        ASSERT(ExecuteOperationScene::SHMEM_OPEN_FAILED, region.dataShmFd != -1)
            << "Failed to open shared memory: " << region.dataShmName;
        
        region.signalShmFd = shm_open(region.signalShmName.c_str(), O_RDWR, 0);
        ASSERT(ExecuteOperationScene::SHMEM_OPEN_FAILED, region.signalShmFd != -1)
            << "Failed to open shared memory: " << region.signalShmName;
    }

    // 映射共享内存
    region.dataPtr = mmap(nullptr, region.dataSize,
                              PROT_READ | PROT_WRITE, MAP_SHARED,
                              region.dataShmFd, 0);
    ASSERT(ExecuteOperationScene::SHMEM_MMAP_FAILED, region.dataPtr != MAP_FAILED)
        << "Failed to mmap shared memory: " << region.dataShmName;

    region.signalPtr = mmap(nullptr, region.signalSize,
                                PROT_READ | PROT_WRITE, MAP_SHARED,
                                region.signalShmFd, 0);
    ASSERT(ExecuteOperationScene::SHMEM_MMAP_FAILED, region.signalPtr != MAP_FAILED)
        << "Failed to mmap shared memory: " << region.signalShmName;

    // 初始化信号区域（仅创建者）
    if (region.isCreator) {
        memset(region.signalPtr, 0, region.signalSize);
    }

    regions_[groupName] = region;
    return &regions_[groupName];
}

void ShmemManager::DestroyRegion(const std::string& groupName) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = regions_.find(groupName);
    if (it == regions_.end()) {
        return;
    }

    ShmemRegion& region = it->second;

    // 仅创建者销毁共享内存
    if (region.isCreator) {
        munmap(region.dataPtr, region.dataSize);
        munmap(region.signalPtr, region.signalSize);
        close(region.dataShmFd);
        close(region.signalShmFd);
        shm_unlink(region.dataShmName.c_str());
        shm_unlink(region.signalShmName.c_str());
    }

    regions_.erase(it);
}

void* ShmemManager::GetDataPtr(const std::string& groupName, int rankId,
                                const std::vector<int64_t>& offset) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = regions_.find(groupName);
    ASSERT(ExecuteOperationScene::SHMEM_REGION_NOT_FOUND, it != regions_.end())
        << "Shared memory region not found: " << groupName;

    ShmemRegion& region = it->second;

    // 计算偏移量
    size_t rankDataSize = region.dataSize / region.worldSize;
    size_t byteOffset = rankId * rankDataSize;
    for (size_t i = 0; i < offset.size(); ++i) {
        byteOffset += offset[i];
    }

    return static_cast<char*>(region.dataPtr) + byteOffset;
}

void* ShmemManager::GetSignalPtr(const std::string& groupName, int srcRank, int dstRank,
                                  const std::vector<int64_t>& offset) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = regions_.find(groupName);
    ASSERT(ExecuteOperationScene::SHMEM_REGION_NOT_FOUND, it != regions_.end())
        << "Shared memory region not found: " << groupName;

    ShmemRegion& region = it->second;

    // 计算偏移量（信号矩阵：[srcRank, dstRank, ...]）
    size_t rankSignalSize = region.signalSize / (region.worldSize * region.worldSize);
    size_t byteOffset = (srcRank * region.worldSize + dstRank) * rankSignalSize;
    for (size_t i = 0; i < offset.size(); ++i) {
        byteOffset += offset[i];
    }

    return static_cast<char*>(region.signalPtr) + byteOffset;
}

void ShmemManager::AtomicSet(void* addr, int32_t value) {
    std::atomic<int32_t>* atomicAddr = static_cast<std::atomic<int32_t>*>(addr);
    atomicAddr->store(value, std::memory_order_release);
}

void ShmemManager::AtomicAdd(void* addr, int32_t value) {
    std::atomic<int32_t>* atomicAddr = static_cast<std::atomic<int32_t>*>(addr);
    atomicAddr->fetch_add(value, std::memory_order_acq_rel);
}

void ShmemManager::WaitUntil(void* addr, int32_t expectedValue, bool resetSignal) {
    std::atomic<int32_t>* atomicAddr = static_cast<std::atomic<int32_t>*>(addr);

    // 轮询等待（可以优化为futex）
    int retryCount = 0;
    const int maxRetry = 1000000;  // 防止死锁
    
    while (atomicAddr->load(std::memory_order_acquire) != expectedValue) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        retryCount++;
        
        ASSERT(ExecuteOperationScene::SHMEM_WAIT_TIMEOUT, retryCount < maxRetry)
            << "WaitUntil timeout after " << maxRetry << " retries";
    }

    if (resetSignal) {
        atomicAddr->store(0, std::memory_order_release);
    }
}

void ShmemManager::CleanupAll() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& [groupName, region] : regions_) {
        if (region.isCreator) {
            munmap(region.dataPtr, region.dataSize);
            munmap(region.signalPtr, region.signalSize);
            close(region.dataShmFd);
            close(region.signalShmFd);
            shm_unlink(region.dataShmName.c_str());
            shm_unlink(region.signalShmName.c_str());
        }
    }

    regions_.clear();
}

ShmemManager::~ShmemManager() {
    CleanupAll();
}

} // namespace npu::tile_fwk
