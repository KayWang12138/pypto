/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under * terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to * License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file shared_memory_context.cpp
 * \brief Shared memory context implementation for distributed verification tool
 */

#include "shared_memory_context.h"
#include "tilefwk/pypto_fwk_log.h"
#include <algorithm>
#include <cstring>

namespace npu::tile_fwk {

SharedMemoryContext& SharedMemoryContext::GetInstance() {
    static SharedMemoryContext instance;
    return instance;
}

void SharedMemoryContext::Initialize() {
    if (initialized_) {
        return;
    }
    
    InitializeRankInfo();
    commDomainCount_ = DetectCommDomainCount();
    
    if (commDomainCount_ > 0) {
        VERIFY_LOGI("Initializing shared memory context: commDomainCount=%d, currentRank=%d, worldSize=%d",
                    commDomainCount_, currentRank_, worldSize_);
        
        for (int i = 0; i < commDomainCount_; ++i) {
            if (!CreateCommDomainShmem(i)) {
                VERIFY_LOGE("Failed to create shared memory for comm domain %d", i);
                commDomainCount_ = 0;
                return;
            }
        }
        
        initialized_ = true;
        VERIFY_LOGI("Shared memory context initialized successfully");
    } else {
        VERIFY_LOGI("No communication domains detected, skipping shared memory initialization");
    }
}

void* SharedMemoryContext::AllocateShmem(int groupIndex, ShmemMemType memType, size_t size) {
    // ASSERT(ExecuteOperationScene::INVALID_GROUP_INDEX,
    //        groupIndex >= 0 && groupIndex < commDomainCount_);
    
    auto& domain = commDomains_[groupIndex];
    
    size_t alignedSize = (size + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
    
    void* resultPtr = nullptr;
    
    if (memType == ShmemMemType::DATA) {
        std::lock_guard<std::mutex> lock(domain->dataMutex);
        
        if (domain->dataOffset + alignedSize > domain->dataSize) {
            VERIFY_LOGE("Shared memory out of memory: groupIndex=%d, memType=DATA, "
                        "requested=%zu, available=%zu, used=%zu",
                        groupIndex, alignedSize, 
                        domain->dataSize - domain->dataOffset, domain->dataOffset);
            ASSERT(ExecuteOperationScene::SHMEM_OUT_OF_MEMORY, false);
        }
        
        domain->dataOffset += alignedSize;
        size_t offset = domain->dataSize - domain->dataOffset;
        
        resultPtr = static_cast<uint8_t*>(domain->dataPtr) + offset;
        
        VERIFY_LOGI("Allocated shared memory: groupIndex=%d, memType=DATA, size=%zu, offset=%zu",
                    groupIndex, alignedSize, offset);
    } else {
        std::lock_guard<std::mutex> lock(domain->statusMutex);
        
        if (domain->statusOffset + alignedSize > domain->statusSize) {
            VERIFY_LOGE("Shared memory out of memory: groupIndex=%d, memType=STATUS, "
                        "requested=%zu, available=%zu, used=%zu",
                        groupIndex, alignedSize, 
                        domain->statusSize - domain->statusOffset, domain->statusOffset);
            ASSERT(ExecuteOperationScene::SHMEM_OUT_OF_MEMORY, false);
        }
        
        domain->statusOffset += alignedSize;
        size_t offset = domain->statusSize - domain->statusOffset;
        
        resultPtr = static_cast<uint8_t*>(domain->statusPtr) + offset;
        
        VERIFY_LOGI("Allocated shared memory: groupIndex=%d, memType=STATUS, size=%zu, offset=%zu",
                    groupIndex, alignedSize, offset);
    }
    
    constexpr uint64_t OFFSET_BITS = 54UL;
    constexpr uint64_t GROUP_BITS = 2UL;
    constexpr uint64_t MEMTYPE_BITS = 2UL;
    constexpr uint64_t GROUP_SHIFT = OFFSET_BITS;
    constexpr uint64_t MEMTYPE_SHIFT = GROUP_SHIFT + GROUP_BITS;
    constexpr uint64_t FILL_SHIFT = MEMTYPE_SHIFT + MEMTYPE_BITS;
    
    size_t offset = static_cast<uint8_t*>(resultPtr) - 
                    (memType == ShmemMemType::DATA ? domain->dataPtr : domain->statusPtr);
    
    uint64_t vaddr = offset | 
                      (static_cast<uint64_t>(groupIndex) << GROUP_SHIFT) | 
                      (static_cast<uint64_t>(memType) << MEMTYPE_SHIFT) | 
                      (1UL << FILL_SHIFT);
    
    return reinterpret_cast<void*>(vaddr);
}

void* SharedMemoryContext::GetDataPtr(int groupIndex) const {
    // ASSERT(ExecuteOperationScene::INVALID_GROUP_INDEX,
    //        groupIndex >= 0 && groupIndex < commDomainCount_);
    return commDomains_[groupIndex]->dataPtr;
}

void* SharedMemoryContext::GetStatusPtr(int groupIndex) const {
    // ASSERT(ExecuteOperationScene::INVALID_GROUP_INDEX,
    //        groupIndex >= 0 && groupIndex < commDomainCount_);
    return commDomains_[groupIndex]->statusPtr;
}

void SharedMemoryContext::Cleanup() {
    if (!initialized_) {
        return;
    }
    
    VERIFY_LOGI("Cleaning up shared memory context");
    
    for (auto& domain : commDomains_) {
        if (domain && domain->dataPtr != nullptr) {
            munmap(domain->dataPtr, domain->totalSize);
            domain->dataPtr = nullptr;
        }
    }
    
    for (auto& [groupIndex, fd] : shmFdMap_) {
        close(fd);
        
        if (currentRank_ == 0) {
            std::string name = GenerateShmemName(groupIndex);
            shm_unlink(name.c_str());
        }
    }
    
    shmFdMap_.clear();
    commDomains_.clear();
    initialized_ = false;
    
    VERIFY_LOGI("Shared memory context cleaned up");
}

SharedMemoryContext::~SharedMemoryContext() {
    Cleanup();
}

bool SharedMemoryContext::CreateCommDomainShmem(int groupIndex) {
    std::string shmemName = GenerateShmemName(groupIndex);
    size_t totalSize = SHMEM_SIZE_PER_DOMAIN;
    size_t dataSize = totalSize - STATUS_REGION_SIZE;
    size_t statusSize = STATUS_REGION_SIZE;
    
    VERIFY_LOGI("Creating shared memory for comm domain %d: name=%s, totalSize=%zu, dataSize=%zu, statusSize=%zu",
                groupIndex, shmemName.c_str(), totalSize, dataSize, statusSize);
    
    int fd = shm_open(shmemName.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd == -1) {
        VERIFY_LOGE("Failed to create shared memory %s: %s", shmemName.c_str(), strerror(errno));
        return false;
    }
    
    shmFdMap_[groupIndex] = fd;
    
    if (ftruncate(fd, totalSize) == -1) {
        VERIFY_LOGE("Failed to truncate shared memory %s: %s", shmemName.c_str(), strerror(errno));
        close(fd);
        return false;
    }
    
    void* ptr = mmap(nullptr, totalSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        VERIFY_LOGE("Failed to map shared memory %s: %s", shmemName.c_str(), strerror(errno));
        close(fd);
        return false;
    }
    
    std::memset(ptr, 0, totalSize);
    
    auto domain = std::make_unique<CommDomainShmemInfo>(groupIndex, shmemName, totalSize, dataSize, statusSize);
    domain->dataPtr = ptr;
    domain->statusPtr = static_cast<uint8_t*>(ptr) + dataSize;
    
    commDomains_.push_back(std::move(domain));
    
    VERIFY_LOGI("Shared memory created successfully for comm domain %d", groupIndex);
    return true;
}

std::string SharedMemoryContext::GenerateShmemName(int groupIndex) const {
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    
    pid_t pid = getpid();
    
    return std::string("/pypto_verify_") + hostname + "_" + std::to_string(pid) + "_group" + std::to_string(groupIndex);
}

void SharedMemoryContext::InitializeRankInfo() {
    const char* rankEnv = std::getenv("PYPTO_VERIFY_RANK");
    if (rankEnv != nullptr) {
        currentRank_ = std::atoi(rankEnv);
    }
    
    const char* worldSizeEnv = std::getenv("PYPTO_VERIFY_WORLD_SIZE");
    if (worldSizeEnv != nullptr) {
        worldSize_ = std::atoi(worldSizeEnv);
    }
    
    VERIFY_LOGI("Rank info initialized: currentRank=%d, worldSize=%d", currentRank_, worldSize_);
}

int SharedMemoryContext::DetectCommDomainCount() {
    const char* commCountEnv = std::getenv("PYPTO_VERIFY_COMM_DOMAIN_COUNT");
    if (commCountEnv != nullptr) {
        int count = std::atoi(commCountEnv);
        if (count >= 0 && count <= 2) {
            VERIFY_LOGI("Comm domain count from env: %d", count);
            return count;
        }
    }
    
    VERIFY_LOGI("No comm domain count specified, defaulting to 0");
    return 0;
}

} 
