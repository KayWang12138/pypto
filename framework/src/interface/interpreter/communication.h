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
 * \file communication.h
 * \brief
 */

#pragma once

#include <string>
#include <atomic>
#include <unordered_map>
#include <mutex>
#include <memory>
#include "raw_tensor_data.h"


namespace npu::tile_fwk {
class SimulationCommManager;

int GetRankId(const std::string &groupName);

int GetWorldSize(const std::string &groupName);

class SimulationCommContext {
public:
    static constexpr size_t WIN_IN_SIZE = 200 * 1024 * 1024;
    static constexpr size_t WIN_EXP_SIZE = 1 * 1024 * 1024;
    void Init(const std::string &groupName, int rank, int worldSize);
    LogicalTensorDataPtr Alloc(size_t slotSize);
    LogicalTensorDataPtr AllocSignal(size_t slotSize);

    int GetRank() const {return rank_;};
    int GetWorldSize() const {return worldSize_;};
    std::string GetGroupName() {return groupName_;};
    void Put(LogicalTensorDataPtr data, int dstRank, uint64_t offset, [[maybe_unused]] int atomicType = 0);
    void Signal(int dstRank, int value, size_t slotSize, [[maybe_unused]] int atomicType = 0, [[maybe_unused]] bool notifyAll = false);
    void Wait(int srcRank, int expect, size_t slotSize, bool reset = false);
    LogicalTensorDataPtr Get(int srcRank, size_t slotSize, uint64_t offset = 0);
private:
    friend class SimulationCommManager;
    SimulationCommContext() = default;
    SimulationCommContext(const SimulationCommContext &) = delete;
    SimulationCommContext& operator=(const SimulationCommContext &) = delete;
    SimulationCommContext(SimulationCommContext &&other) noexcept
        : groupName_(std::move(other.groupName_)),
          rank_(other.rank_),
          worldSize_(other.worldSize_),
          dataBase_(other.dataBase_),
          ctrlBase_(other.ctrlBase_),
          dataShmSize_(other.dataShmSize_.load()),
          ctrlShmSize_(other.ctrlShmSize_.load()),
          allocatedData_(other.allocatedData_),
          allocatedSignal_(other.allocatedSignal_),
          ctrlName_(std::move(other.ctrlName_)),
          dataName_(std::move(other.dataName_)),
          remoteMutex_(),
          allocMutex_(),
          remoteRanks_(std::move(other.remoteRanks_)) {
            other.dataBase_ = nullptr;
            other.ctrlBase_ = nullptr;
            other.allocatedData_ = false;
            other.allocatedSignal_ = false;
            other.dataShmSize_ = 0;
            other.ctrlShmSize_ = 0;
          }
    SimulationCommContext& operator=(SimulationCommContext &&other) noexcept {
        if (this != &other) {
            Destroy();
            groupName_ = std::move(other.groupName_);
            rank_ = other.rank_;
            worldSize_ = other.worldSize_;
            dataBase_ = other.dataBase_;
            ctrlBase_ = other.ctrlBase_;
            dataShmSize_ = other.dataShmSize_.load();
            ctrlShmSize_ = other.ctrlShmSize_.load();
            allocatedData_ = other.allocatedData_;
            allocatedSignal_ = other.allocatedSignal_;
            ctrlName_ = std::move(other.ctrlName_);
            dataName_ = std::move(other.dataName_);
            remoteRanks_ = std::move(other.remoteRanks_);

            other.dataBase_ = nullptr;
            other.ctrlBase_ = nullptr;
            other.allocatedData_ = false;
            other.allocatedSignal_ = false;
            other.dataShmSize_ = 0;
            other.ctrlShmSize_ = 0;
        }
        return *this;
    }
    ~SimulationCommContext();

    struct RemoteRank {
        uint8_t *dataBase = nullptr;
        uint8_t *ctrlBase = nullptr;
        ~RemoteRank();
    };

    uint8_t *GetRemoteRank(int dstRank, bool isSignal);
    void PreAlloc(bool isSignal);
    void Destroy();

    std::string groupName_;
    int rank_ = -1;
    int worldSize_ = -1;

    uint8_t *dataBase_ = nullptr;
    uint8_t *ctrlBase_ = nullptr;

    std::atomic<size_t> dataShmSize_ = 0;
    std::atomic<size_t> ctrlShmSize_ = 0;
    bool allocatedData_ = false;
    bool allocatedSignal_ = false;

    std::string ctrlName_;
    std::string dataName_;
    std::mutex remoteMutex_;
    std::mutex allocMutex_;
    std::unordered_map<int, std::unique_ptr<RemoteRank>> remoteRanks_;
};

class SimulationCommManager {
public:
    static SimulationCommManager &Instance() {
        static SimulationCommManager instance;
        return instance;
    }
    void CreateSimulationCommContext(const std::string &groupName);
    LogicalTensorDataPtr Alloc(const std::string &groupName, size_t slotSize);
    LogicalTensorDataPtr AllocSignal(const std::string &groupName, size_t slotSize);
    static std::string GetHandler(const std::string &groupName, int rank, bool isSignal);
private:
    SimulationCommManager() = default;
    ~SimulationCommManager() = default;
    std::unordered_map<std::string, std::shared_ptr<SimulationCommContext>> contexts_;
    std::mutex mutex_;
};
}  // namespace npu::tile_fwk