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

#include <chrono>
#include <thread>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <mutex>
#include <iostream>
#include "interface/tileop/distributed/comm_context.h"
#include "machine/runtime/distributed/distributed_context.h"
#include "interface/tensor/float.h"
#include "communication.h"

namespace npu::tile_fwk {

void CheckNotNullPtr(uint8_t *ptr, const char *message) {
    if (!ptr) {
        throw std::runtime_error(message);
    }
}

int GetRankId(const std::string &groupName) {
    const char* rankStr = std::getenv("RANK");
    if (rankStr != nullptr) {
        int rankId = std::atoi(rankStr);
        std::cout << "current rank is " << rankId << std::endl;
        return rankId;
    }

    auto it = g_context.find(groupName);
    if (it == g_context.end()) {
        return -1;
    }
    TileOp::CommContext *context = (TileOp::CommContext *)(it->second.second);
    int rankId = context->rankId;
    std::cout << "current rank is " << rankId << std::endl;
    return rankId;
}

int GetWorldSize(const std::string &groupName) {
    const char* worldSizeStr = std::getenv("WORLD_SIZE");
    if (worldSizeStr != nullptr) {
        int worldSize = std::atoi(worldSizeStr);
        std::cout << "current world_size is " << worldSize << std::endl;
        return worldSize;
    }

    auto it = g_context.find(groupName);
    if (it == g_context.end()) {
        return -1;
    }
    TileOp::CommContext *context = (TileOp::CommContext *)(it->second.second);
    int worldSize = context->rankNum;
    std::cout << "current world_size is " << worldSize << std::endl;
    return worldSize;
}

// ============================== SimulationCommContext
SimulationCommContext::RemoteRank::~RemoteRank() {
    if (dataBase && dataBase != MAP_FAILED) {
        munmap(dataBase, WIN_IN_SIZE);
        dataBase = nullptr;
    }
    if (ctrlBase && ctrlBase != MAP_FAILED) {
        munmap(ctrlBase, WIN_EXP_SIZE);
        ctrlBase = nullptr;
    }
}

void SimulationCommContext::Init(const std::string &groupName, int rank, int worldSize, uint32_t round) {
    groupName_ = groupName;
    rank_ = rank;
    worldSize_ = worldSize;
    round_ = round;
}

void SimulationCommContext::PreAlloc(bool isSignal) {
    if (isSignal && allocatedSignal_) {
        return;
    }
    if (!isSignal && allocatedData_) {
        return;
    }
    std::string handler = SimulationCommManager::GetHandler(groupName_, rank_, isSignal, round_);
    int fd = shm_open(handler.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd == -1) {
        throw std::runtime_error("shm_open " + handler + " error!");
    }
    size_t size = isSignal ? WIN_EXP_SIZE: WIN_IN_SIZE;
    if (ftruncate(fd, size) == -1) {
        close(fd);
        throw std::runtime_error("ftruncate error!");
    }
    uint8_t *base = (uint8_t *) mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED || base == nullptr) {
        close(fd);
        throw std::runtime_error("mmap error!");
    }
    close(fd);

    if (isSignal) {
        ctrlBase_ = base;
        allocatedSignal_ = true;
        ctrlName_ = handler;
        memset(ctrlBase_, 0, WIN_EXP_SIZE);
    } else {
        dataBase_ = base;
        allocatedData_ = true;
        dataName_ = handler;
        memset(dataBase_, 0, WIN_IN_SIZE);
    }
    std::cout << "round " << round_ << " prealloc for " << handler << std::endl;
}

RawTensorDataPtr SimulationCommContext::Alloc(DataType dataType, const Shape& shape) {
    std::lock_guard<std::mutex> lock(allocMutex_);

    if (!allocatedData_) {
        throw std::runtime_error("data area not pre-allocated!");
    }

    size_t slotSize = BytesOf(dataType) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    size_t beforeSize = dataShmSize_.load();
    size_t shmSize = beforeSize + slotSize;
    if (shmSize > WIN_IN_SIZE) {
        throw std::runtime_error("Out of pre-allocated memory!");
    }
    dataShmSize_.store(shmSize);
    std::cout << "round " << round_ << " alloc " << slotSize << "B for rank " << rank_ << std::endl;

    auto result = RawTensorData::CreateTensor(dataType, shape, dataBase_ + beforeSize);
    result->SetShmOffset(beforeSize);
    return result;
}

RawTensorDataPtr SimulationCommContext::AllocSignal(DataType dataType, const Shape& shape) {
    std::lock_guard<std::mutex> lock(allocMutex_);
    if (!allocatedSignal_) {
        throw std::runtime_error("signal area not pre-allocated!");
    }
    size_t slotSize = BytesOf(dataType) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

    size_t beforeSize = ctrlShmSize_.load();
    size_t shmSize = beforeSize + slotSize;
    if (shmSize > WIN_EXP_SIZE) {
        throw std::runtime_error("Out of pre-allocated memory!");
    }
    ctrlShmSize_.store(shmSize);
    std::cout << "round " << round_ << " allocSignal " << slotSize << "B for rank " << rank_ << std::endl;

    auto result = RawTensorData::CreateTensor(dataType, shape, ctrlBase_ + beforeSize);
    result->SetShmOffset(beforeSize);
    return result;
}

uint8_t *SimulationCommContext::GetRemoteRank(int dstRank, bool isSignal) {
    if (dstRank == rank_) {
        uint8_t *result = isSignal ? ctrlBase_ : dataBase_;
        CheckNotNullPtr(result, "base is nullptr!");
        return result;
    }

    if (dstRank < 0 || dstRank >= worldSize_) {
        throw std::runtime_error("Invalid remote rank " + std::to_string(dstRank) +
                                ", world size: " + std::to_string(worldSize_));
    }

    auto openWithRetry = [&](const std::string &handler) {
        int fd = -1;
        int retries = 1000;
        while (retries--) {
            fd = shm_open(handler.c_str(), O_RDWR, 0666);
            if (fd != -1) {
                return fd;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        throw std::runtime_error("GetRemoteRank shm_open " + handler + " error!");
    };

    std::lock_guard<std::mutex> lock(remoteMutex_);

    auto it = remoteRanks_.find(dstRank);
    if (it != remoteRanks_.end()) {
        uint8_t *result = isSignal ? it->second->ctrlBase : it->second->dataBase;
        CheckNotNullPtr(result, "found in remoteRanks, but base is nullptr!");
        return result;
    }
    auto remote = std::make_unique<RemoteRank>();
    std::string dataHandler = SimulationCommManager::GetHandler(groupName_, dstRank, false, round_);
    int fd = openWithRetry(dataHandler);
    remote->dataBase = (uint8_t *) mmap(nullptr, WIN_IN_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (remote->dataBase == MAP_FAILED || remote->dataBase == nullptr) {
        close(fd);
        throw std::runtime_error("GetRemoteRank mmap " + dataHandler + " error!");
    }
    close(fd);

    std::string ctrlHandler = SimulationCommManager::GetHandler(groupName_, dstRank, true, round_);
    fd = openWithRetry(ctrlHandler);
    remote->ctrlBase = (uint8_t *) mmap(nullptr, WIN_EXP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (remote->ctrlBase == MAP_FAILED || remote->ctrlBase == nullptr) {
        close(fd);
        throw std::runtime_error("GetRemoteRank mmap " + ctrlHandler + " error!");
    }
    close(fd);

    uint8_t *result = isSignal ? remote->ctrlBase : remote->dataBase;
    CheckNotNullPtr(result, "Created, but base is nullptr!");
    remoteRanks_[dstRank] = std::move(remote);
    return result;
}

template<typename T>
void AtomicAddArray(T *dst, const T *src, size_t count) {
    for (size_t i = 0; i < count; i++) {
        auto *atomicPtr = reinterpret_cast<std::atomic<T>*>(&dst[i]);
        T old = atomicPtr->load(std::memory_order_relaxed);
        T newV;
        do {
            newV = old + src[i];
        } while (!atomicPtr->compare_exchange_weak(old, newV));
    }
}

template<>
void AtomicAddArray<int32_t>(int32_t *dst, const int32_t *src, size_t count) {
    for (size_t i = 0; i < count; i++) {
        __sync_fetch_and_add(&dst[i], src[i]);
    }
}

void SimulationCommContext::Put(LogicalTensorDataPtr data, int dstRank, uint64_t offset, int atomicType) {
    uint8_t *base = GetRemoteRank(dstRank, false);
    size_t slotSize = data->GetSize() * BytesOf(data->GetDataType());
    if (offset + slotSize > WIN_IN_SIZE) {
        throw std::runtime_error("Put operation would exceed shared memory bounds!");
    }
    std::atomic_thread_fence(std::memory_order_release);
    if (atomicType == 0) {
        std::cout << "entered atomicType: 0" << std::endl;
        memcpy(base + offset, data->GetData()->data(), slotSize);
    }
    if (atomicType == 1) {
        std::cout << "entered atomicType: 1" << std::endl;
        void *src = reinterpret_cast<void *>(data->GetData()->data());
        void *dst = reinterpret_cast<void *>(base + offset);
        switch (data->GetDataType()) {
            case DT_UINT8:
                AtomicAddArray<uint8_t>(static_cast<uint8_t *>(dst), static_cast<uint8_t *>(src), data->GetSize());
                break;
            case DT_UINT16:
                AtomicAddArray<uint16_t>(static_cast<uint16_t *>(dst), static_cast<uint16_t *>(src), data->GetSize());
                break;
            case DT_UINT32:
                AtomicAddArray<uint32_t>(static_cast<uint32_t *>(dst), static_cast<uint32_t *>(src), data->GetSize());
                break;
            case DT_INT8:
                AtomicAddArray<int8_t>(static_cast<int8_t *>(dst), static_cast<int8_t *>(src), data->GetSize());
                break;
            case DT_INT16:
                AtomicAddArray<int16_t>(static_cast<int16_t *>(dst), static_cast<int16_t *>(src), data->GetSize());
                break;
            case DT_INT32:
                AtomicAddArray<int32_t>(static_cast<int32_t *>(dst), static_cast<int32_t *>(src), data->GetSize());
                break;
            case DT_FP32:
                AtomicAddArray<float>(static_cast<float *>(dst), static_cast<float *>(src), data->GetSize());
                break;
            case DT_FP16:
                AtomicAddArray<float16>(static_cast<float16 *>(dst), static_cast<float16 *>(src), data->GetSize());
                break;
            case DT_BF16:
                AtomicAddArray<bfloat16>(static_cast<bfloat16 *>(dst), static_cast<bfloat16 *>(src), data->GetSize());
                break;
            default:
                throw std::runtime_error("Unsupported atomic add data type!");
        }
    }
    std::cout << "round " << round_ << " put " << data << "to rank " << dstRank << " offset:" << offset << std::endl;
}

void SimulationCommContext::Set(int dstRank, int value, size_t slotSize, uint64_t offset) {
    uint8_t *base = GetRemoteRank(dstRank, false);
    if (slotSize > WIN_IN_SIZE) {
        throw std::runtime_error("Set operation would exceed shared memory bounds!");
    }
    std::atomic_thread_fence(std::memory_order_release);
    std::cout << "round " << round_ << " alloc " << slotSize << "B for rank " << rank_ << std::endl;
    memset(base + offset, value, slotSize);
}

void SimulationCommContext::SignalSingle(int dstRank, int value, size_t slotSize, uint64_t offset, int atomicType) {
    uint8_t *base = GetRemoteRank(dstRank, true);
    if (slotSize > WIN_EXP_SIZE) {
        throw std::runtime_error("Signal operation would exceed shared memory bounds!");
    }
    std::atomic_thread_fence(std::memory_order_release);
    int32_t *ctrlBase = reinterpret_cast<int32_t *>(base);
    slotSize = slotSize / (sizeof(int32_t) / sizeof(uint8_t));
    offset = offset / (sizeof(int32_t) / sizeof(uint8_t));
    if (atomicType == 0) {
        memset(ctrlBase + offset, value, slotSize);
    }
    if (atomicType == 1) {
        for (size_t i = 0; i < slotSize; i++) {
            __sync_fetch_and_add(&ctrlBase[offset + i], value);
        }
    }
    std::cout << "round " << round_ << " send signal " << value << " to rank " << dstRank << std::endl;
}

void SimulationCommContext::Signal(int dstRank, int value, size_t slotSize, uint64_t offset, int atomicType, bool notifyAll) {
    if (!notifyAll) {
        SignalSingle(dstRank, value, slotSize, offset, atomicType);
        return;
    }
    for (int rank = 0; rank < worldSize_; rank++) {
        SignalSingle(rank, value, slotSize, offset, atomicType);
    }
}

void SimulationCommContext::Wait(int srcRank, int expect, size_t slotSize, uint64_t offset, bool reset) {
    volatile uint8_t *base = reinterpret_cast<volatile uint8_t *>(GetRemoteRank(srcRank, true));
    int32_t targetValue = static_cast<int32_t>(expect);
    if (slotSize == 0 || slotSize >= WIN_EXP_SIZE) {
        throw std::runtime_error("Invalid slotSize in Wait operation!");
    }
    volatile int32_t *ctrlBase = reinterpret_cast<volatile int32_t *>(base);
    offset = offset / (sizeof(int32_t) / sizeof(uint8_t));
    slotSize = slotSize / (sizeof(int32_t) / sizeof(uint8_t));
    std::cout << "round " << round_ << " waitting " << targetValue << " for rank " << srcRank << std::endl;
    while(ctrlBase[offset + slotSize - 1] != targetValue) {
        std::this_thread::yield();
        std::atomic_thread_fence(std::memory_order_acquire);
    }
    if (reset) {
        volatile int32_t *vbase = ctrlBase;
        for (size_t i = offset; i < offset + slotSize; i++) {
            const_cast<volatile int32_t *>(vbase)[i] = 0;
        }
        std::atomic_thread_fence(std::memory_order_release);
    }
}

bool SimulationCommContext::CheckWaitCondition(int srcRank, int expect, size_t slotSize, uint64_t offset) {
    volatile uint8_t *base = reinterpret_cast<volatile uint8_t *>(GetRemoteRank(srcRank, true));
    int32_t targetValue = static_cast<int32_t>(expect);
    if (slotSize == 0 || slotSize >= WIN_EXP_SIZE) {
        throw std::runtime_error("Invalid slotSize in Wait operation!");
    }
    volatile int32_t *ctrlBase = reinterpret_cast<volatile int32_t *>(base);
    offset = offset / (sizeof(int32_t) / sizeof(uint8_t));
    slotSize = slotSize / (sizeof(int32_t) / sizeof(uint8_t));
    std::atomic_thread_fence(std::memory_order_acquire);
    return ctrlBase[offset + slotSize - 1] == targetValue;
}

LogicalTensorDataPtr SimulationCommContext::Get(int srcRank, DataType datatype, const Shape &shape, uint64_t offset) {
    uint8_t *base = GetRemoteRank(srcRank, false);
    size_t slotSize = BytesOf(datatype) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    if (offset + slotSize > WIN_IN_SIZE) {
        throw std::runtime_error("Get operation would exceed shared memory bound!");
    }
    std::cout << "round " << round_ << " get " << slotSize << "B data from rank " << srcRank << std::endl;
    RawTensorDataPtr result = RawTensorData::CreateTensor(datatype, shape, base + offset);
    return std::make_shared<LogicalTensorData>(result);
}

void SimulationCommContext::Destroy() {
    if (ctrlBase_) {
        munmap(ctrlBase_, WIN_EXP_SIZE);
        ctrlBase_ = nullptr;
    }
    if (!ctrlName_.empty()) {
        shm_unlink(ctrlName_.c_str());
        ctrlName_.clear();
    }

    if (dataBase_) {
        munmap(dataBase_, WIN_IN_SIZE);
        dataBase_ = nullptr;
    }
    if (!dataName_.empty()) {
        shm_unlink(dataName_.c_str());
        dataName_.clear();
    }
    std::cout << "round " << round_ << " released rank " << rank_ << std::endl;

    allocatedData_ = false;
    allocatedSignal_ = false;
    dataShmSize_ = 0;
    ctrlShmSize_ = 0;
}

SimulationCommContext::~SimulationCommContext() {
    Destroy();
}

// ============================== SimulationCommManager
void SimulationCommManager::CreateSimulationCommContext(const std::string &groupName, uint32_t round) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (contexts_.find(groupName) != contexts_.end()) {
        throw std::runtime_error("CommContext for group " + groupName + " already exist!");
    }

    int rank = GetRankId(groupName);
    int worldSize = GetWorldSize(groupName);
    if (rank == -1) {
        throw std::runtime_error("Can not get rank for group " + groupName + "!");
    }
    if (worldSize == -1) {
        throw std::runtime_error("Can not get worldSize for group " + groupName + "!");
    }

    auto context = std::make_shared<SimulationCommContext>();
    context->Init(groupName, rank, worldSize, round);
    context->PreAlloc(true);
    context->PreAlloc(false);
    contexts_[groupName] = context;
}

void SimulationCommManager::DestroySimulationCommContext(const std::string &groupName) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = contexts_.find(groupName);
    if (it == contexts_.end()) {
        return;
    }
    contexts_.erase(it);
}

std::shared_ptr<SimulationCommContext> SimulationCommManager::GetCommContext(const std::string &groupName) {
    auto it = contexts_.find(groupName);
    if (it == contexts_.end()) {
        throw std::runtime_error("There is no group named " + groupName + " in contexts!");
    }
    return it->second;
}

std::string SimulationCommManager::GetHandler(const std::string &groupName, int rank, bool isSignal, uint32_t round) {
    std::string suffix = isSignal ? "_ctrl" : "_data";
    return "round_" + std::to_string(round) + "_" + groupName + "_" + std::to_string(rank) + suffix;
}


/* Alloc a new tensor in WIN area, and record the offset.*/
RawTensorDataPtr SimulationCommManager::Alloc(const std::string &groupName, DataType dataType, const Shape& shape) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = contexts_.find(groupName);
    if (it == contexts_.end()) {
        throw std::runtime_error("SimulationCommContext for group " + groupName + " not found!");
    }
    auto result = it->second->Alloc(dataType, shape);
    return result;
}

RawTensorDataPtr SimulationCommManager::AllocSignal(const std::string &groupName, DataType dataType, const Shape& shape) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = contexts_.find(groupName);
    if (it == contexts_.end()) {
        throw std::runtime_error("SimulationCommContext for group " + groupName + " not found!");
    }
    auto result = it->second->AllocSignal(dataType, shape);
    return result;
}

} // namespace npu:tile_fwk