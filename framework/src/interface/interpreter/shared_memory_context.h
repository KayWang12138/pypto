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
 * \file shared_memory_context.h
 * \brief Shared memory context for distributed verification tool
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <atomic>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include "verify_error.h"

namespace npu::tile_fwk {

enum class ShmemMemType {
    DATA = 0,
    STATUS = 1
};

struct CommDomainShmemInfo {
    int groupIndex;
    std::string shmemName;
    void* dataPtr{nullptr};
    void* statusPtr{nullptr};
    size_t totalSize;
    size_t dataSize;
    size_t statusSize;
    
    std::atomic<size_t> dataOffset{0};
    std::atomic<size_t> statusOffset{0};
    
    std::mutex dataMutex;
    std::mutex statusMutex;
    
    CommDomainShmemInfo(int gi, const std::string& name, size_t total, size_t ds, size_t ss)
        : groupIndex(gi), shmemName(name), totalSize(total), dataSize(ds), statusSize(ss) {}
};

class SharedMemoryContext {
public:
    static SharedMemoryContext& GetInstance();
    
    void Initialize();
    bool IsInitialized() const { return initialized_; }
    
    int GetCommDomainCount() const { return commDomainCount_; }
    
    void* AllocateShmem(int groupIndex, ShmemMemType memType, size_t size);
    
    void* GetDataPtr(int groupIndex) const;
    void* GetStatusPtr(int groupIndex) const;
    
    void Cleanup();
    
    int GetCurrentRank() const { return currentRank_; }
    int GetWorldSize() const { return worldSize_; }
    
private:
    SharedMemoryContext() = default;
    ~SharedMemoryContext();
    
    SharedMemoryContext(const SharedMemoryContext&) = delete;
    SharedMemoryContext& operator=(const SharedMemoryContext&) = delete;
    
    bool CreateCommDomainShmem(int groupIndex);
    std::string GenerateShmemName(int groupIndex) const;
    void InitializeRankInfo();
    int DetectCommDomainCount();
    
private:
    bool initialized_{false};
    int commDomainCount_{0};
    int currentRank_{0};
    int worldSize_{1};
    
    std::vector<std::unique_ptr<CommDomainShmemInfo>> commDomains_;
    std::unordered_map<int, int> shmFdMap_;
    
    static constexpr size_t SHMEM_SIZE_PER_DOMAIN = 50 * 1024 * 1024;
    static constexpr size_t STATUS_REGION_SIZE = 1 * 1024 * 1024;
    static constexpr size_t ALIGNMENT = 512;
};

} 
