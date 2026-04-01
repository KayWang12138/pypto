/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file communication.cpp
 * \brief
 */

#include "communication.h"
#include <thread>
#include <cstring>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <mutex>

namespace npu::tile_fwk {

int GetRankId() {
    const char *rank = std::getenv("RANK");
    if (rank != nullptr) {
        return std::atoi(rank);
    }
    return -1;
}

int GetWorldSize() {
    const char *worldSize = std::getenv("WORLD_SIZE");
    if (worldSize != nullptr) {
        return std::atoi(worldSize);
    }
    return -1;
}

// ============================== CommContext
CommContext::~RemoteRank() {
    if (dataBase) {
        munmap(dataBase, WIN_IN_SIZE);
        dataBase = nullptr;
    }
    if (ctrlBase) {
        munmap(ctrlBase, WIN_IN_SIZE);
        ctrlBase = nullptr;
    }
}

void CommContext::Init(const std::string &groupName, int rank, int worldSize) {
    groupName_ = groupName;
    rank_ = rank;
    worldSize_ = worldSize;
}

void CommContext::PreAlloc(bool isSignal) {
    if (isSignal && allocatedSignal_) {
        return;
    }
    if (!isSignal && allocatedData_) {
        return;
    }
    std::string handler = CommManager::GetHandler(groupName_, rank_, isSignal);
    int fd = shm_open(handler.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd == -1) {
        throw std::runtime_error("shm_open error!");
    }
    size_t size = isSignal ? WIN_EXP_SIZE: WIN_IN_SIZE;
    if (ftruncate(fd, size) == -1) {
        close(fd);
        throw std::runtime_error("ftruncate error!");
    }
    uint8_t *base = (uint8_t *) mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("mmap error!");
    }
    close(fd);

    if (isSignal) {
        ctrlBase_ = base;
        allocatedSignal_ = true;
        memset(ctrlBase_, 0, WIN_EXP_SIZE);
    } else {
        dataBase_ = base;
        allocatedData_ = true;
        memset(dataBase_, 0, WIN_IN_SIZE);
    }
}

LogicalTensorDataPtr CommContext::Alloc(size_t slotSize) {
    std::lock_guard<std::mutex> lock(allocMutex_);

    if (!allocatedData_) {
        throw std::runtime_error("data area not pre-allocated!");
    }

    size_t beforeSize = dataShmSize_.load();
    size_t shmSize = beforeSize + slotSize;
    if (shmSize > WIN_IN_SIZE) {
        throw std::runtime_error("Out of pre-allocated memory!");
    }
    dataShmSize_.store(shmSize);
    return RawTensorData::CreateTensor(DT_INT8, {1, slotSize}, dataBase_ + beforeSize);
}

LogicalTensorDataPtr CommContext::AllocSignal(size_t slotSize) {
    std::lock_guard<std::mutex> lock(allocMutex_);
    if (!allocatedSignal_) {
        throw std::runtime_error("signal area not pre-allocated!");
    }

    size_t beforeSize = ctrlShmSize_.load();
    size_t shmSize = beforeSize + slotSize;
    if (ctrlShmSize_ > WIN_EXP_SIZE) {
        throw std::runtime_error("Out of pre-allocated memory!");
    }
    ctrlShmSize_.store(shmSize);
    return RawTensorData::CreateTensor(DT_INT8, {1, slotSize}, ctrlBase_ + beforeSize);
}

uint8_t *CommContext::GetRemoteRank(int dstRank, bool isSignal) {
    if (dstRank == rank_) {
        return isSignal ? ctrlBase_ : dataBase_;
    }

    if (dstRank < 0 || dstRank >= worldSize_) {
        throw std::runtime_error("Invalid remote rank " + std::to_string(dstRank) +
                                ", world size: " + std::to_string(worldSize_));
    }

    std::lock_guard<std::mutex> lock(remoteMutex_);

    auto it = remoteRanks_.find(dstRank);
    if (it != remoteRanks_.end()) {
        return isSignal ? it->second->ctrlBase : it->second->dataBase;
    }
    auto remote = std::make_unique<RemoteRank>();
    if (!isSignal) {
        std::string dataHandler = CommManager::GetHandler(groupName_, dstRank, false);
        int fd = shm_open(dataHandler.c_str(), O_RDWR, 0666);
        if (fd == -1) {
            close(fd);
            throw std::runtime_error("GetRemoteRank shm_open error!");
        }
        remote->dataBase = (uint8_t *) mmap(nullptr, WIN_IN_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (remote->dataBase == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("GetRemoteRank mmap error!");
        }
        close(fd);
    } else {
        std::string ctrlHandler = CommManager::GetHandler(groupName_, dstRank, true);
        fd = shm_open(ctrlHandler.c_str(), O_RDWR, 0666);
        if (fd == -1) {
            close(fd);
            throw std::runtime_error("GetRemoteRank shm_open error!");
        }
        remote->ctrlBase = (uint8_t *) mmap(nullptr, WIN_EXP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (remote->ctrlBase == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("GetRemoteRank mmap error!");
        }
        close(fd);
    }

    uint8_t *result = isSignal ? remote->ctrlBase : remote->dataBase;
    remoteRanks_[dstRank] = std::move(remote);
    return result;
}

void CommContext::Put(LogicalTensorDataPtr data, int dstRank, uint64_t offset, int atomicType = 0) {
    uint8_t *base = GetRemoteRank(dstRank, false);
    size_t dataSize = data->GetSize() * BytesOf(data->GetDataType());
    if (offset + dataSize > WIN_IN_SIZE) {
        throw std::runtime_error("Put operation would exceed shared memory bounds!");
    }
    memcpy(base + offset, data->GetData()->GetDevPtr(), dataSize);
}

void CommContext::Signal(int dstRank, int value, size_t slotSize, int atomicType = 0, bool notifyAll = false) {
    uint8_t *base = GetRemoteRank(dstRank, true);
    if (slotSize > WIN_EXP_SIZE) {
        throw std::runtime_error("Signal operation would exceed shared memory bounds!");
    }
    std::atomic_thread_fence(std::memory_order_release);
    memset(base, value, slotSize);
}

void CommContext::Wait(int srcRank, int expect, size_t slotSize, bool reset = false) {
    volatile uint8_t *base = reinterpret_cast<volatile uint8_t *>(GetRemoteRank(srcRank, true));
    uint8_t targetValue = static_cast<uint8_t>(expect);
    if (slotSize == 0 || slotSize >= WIN_EXP_SIZE) {
        throw std::runtime_error("Invalid slotSize in Wait operation!");
    }
    while(base[slotSize - 1] != targetValue) {
        std::this_thread::yield();
        std::atomic_thread_fence(std::memory_order_acquire);
    }
    if (reset) {
        volatile uint8_t *vbase = base;
        for (size_t i = 0; i < slotSize; i++) {
            const_cast<volatile uint8_t *>(vbase)[i] = 0;
        }
        std::atomic_thread_fence(std::memory_order_release);
    }
}

LogicalTensorDataPtr CommContext::Get(int srcRank, size_t slotSize, uint64_t offset = 0) {
    uint8_t *base = GetRemoteRank(srcRank, false);
    if (offset + slotSize > WIN_IN_SIZE) {
        throw std::runtime_error("Get operation would exceed shared memory bound!");
    }
    RawTensorDataPtr result = RawTensorData::CreateTensor(DT_UINT8, {1, slotSize}, base + offset);
    return result;
}

void CommContext::Destroy() {
    if (ctrlBase_) {
        munmap(ctrlBase_, WIN_EXP_SIZE);
        ctrlBase_ = nullptr;
    }

    if (dataBase_) {
        munmap(dataBase_, WIN_IN_SIZE);
        dataBase_ = nullptr;
    }
    
    allocatedData_ = false;
    allocatedSignal_ = false;
    dataShmSize_ = 0;
    ctrlShmSize_ = 0;
}

void CommContext::~CommContext() {
    Destroy();
}

// ============================== CommManager
void CommManager::CreateCommContext(const std::string &groupName, int rank, int worldSize) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (contexts_.find(groupName) != contexts_.end()) {
        throw std::runtime_error("CommContex for group " + groupName + " already exist!");
    }

    auto context = std::make_shared<CommContext>();
    context->Init(groupName, rank, worldSize);
    context->PreAlloc(true);
    context->PreAlloc(false);
    contexts_[groupName] = context;
}

std::string CommManager::GetHandler(const std::string &groupName, int rank, bool isSignal) {
    std::string suffix = isSignal ? "_ctrl" : "_data";
    return groupName + std::to_string(rank) + suffix;
}

/* Alloc a new tensor in WIN area, and record the offset.*/
LogicalTensorDataPtr CommManager::Alloc(const std::string &groupName, size_t slotSize) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = contexts_.find(groupName);
    if (it == contexts_.end()) {
        throw std::runtime_error("CommContext for group " + groupName + " not found!");
    }
    auto result = it->second->Alloc(slotSize);
    return result;
}

LogicalTensorDataPtr CommManager::AllocSignal(const std::string &groupName, size_t slotSize) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = contexts_.find(groupName);
    if (it == contexts_.end()) {
        throw std::runtime_error("CommContext for group " + groupName + " not found!");
    }
    auto result = it->second->AllocSignal(slotSize);
    return result;
}

} // namespace npu:tile_fwk