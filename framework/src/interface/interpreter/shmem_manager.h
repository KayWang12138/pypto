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
 * \file shmem_manager.h
 * \brief Shared memory manager for distributed precision tool
 */

#ifndef SHMEM_MANAGER_H
#define SHMEM_MANAGER_H

#include <string>
#include <map>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
#include <cstdint>
#include <vector>

namespace npu::tile_fwk {

class ShmemManager {
public:
    struct ShmemRegion {
        void* dataPtr;
        void* signalPtr;
        size_t dataSize;
        size_t signalSize;
        std::string dataShmName;
        std::string signalShmName;
        int dataShmFd;
        int signalShmFd;
        int worldSize;
        int creatorRank;
        bool isCreator;
    };

    static ShmemManager* GetInstance();

    ShmemRegion* CreateRegion(const std::string& groupName, int worldSize,
                             int rankId, uint32_t dataType,
                             const std::vector<int64_t>& shape);

    void DestroyRegion(const std::string& groupName);

    void* GetDataPtr(const std::string& groupName, int rankId,
                     const std::vector<int64_t>& offset);

    void* GetSignalPtr(const std::string& groupName, int srcRank, int dstRank,
                       const std::vector<int64_t>& offset);

    void AtomicSet(void* addr, int32_t value);
    void AtomicAdd(void* addr, int32_t value);
    void WaitUntil(void* addr, int32_t expectedValue, bool resetSignal);

    void CleanupAll();

private:
    ShmemManager() = default;
    ~ShmemManager();

    size_t GetDataTypeSize(uint32_t dataType) const;

    std::map<std::string, ShmemRegion> regions_;
    std::mutex mutex_;
};

} // namespace npu::tile_fwk

#endif // SHMEM_MANAGER_H
