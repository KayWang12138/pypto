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
 * \file shmem_wait_until.h
 * \brief
 */

#ifndef SHMEM_WAIT_UNTIL_H
#define SHMEM_WAIT_UNTIL_H

#include <vector>

#include "common.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/utils/dynamic/device_task.h"
#include "machine/device/dynamic/device_utils.h"
#include "interface/utils/distributed_error.h"

namespace npu::tile_fwk::Distributed {
struct SignalTileOp {
    void Init(uint64_t taskId, int32_t* addr, int32_t expectedSum, int32_t cmpType, bool resetSignal)
    {
        taskId_ = taskId;
        addr_ = addr;
        expectedSum_ = expectedSum;
        cmpType_ = cmpType;
        resetSignal_ = resetSignal;
    }
    bool PollCompleted() const;

    SignalTileOp* next{nullptr};
    uint64_t taskId_{0};
    int32_t* addr_{nullptr};
    int32_t expectedSum_{0};
    int32_t cmpType_{static_cast<int32_t>(OpType::EQ)};
    bool resetSignal_{false};
    TaskStat* profData_{nullptr};
};

class HashMap {
public:
    void Init()
    {
        (void)memset_s(&taskArray, sizeof(taskArray), 0, sizeof(taskArray));
        (void)memset_s(&hashTable, sizeof(hashTable), 0, sizeof(hashTable));
        taskCount = 0;
    }

    uint32_t Hash(uint32_t taskId) { return taskId & AICPU_TASK_ARRAY_SIZE_MOD; }

    SignalTileOp* CreateTaskData(uint32_t taskId, int32_t* addr, int32_t expectSum, int32_t cmpType, bool resetSignal)
    {
        if (taskCount >= AICPU_TASK_ARRAY_SIZE) {
            DEV_ERROR(
                DistributedErrorCode::AICPU_TASK_NUM_EXCEED_LIMIT,
                "ctrl.task.pre.task.create#: taskCount=%u >= AICPU_TASK_ARRAY_SIZE=%lu", taskCount,
                AICPU_TASK_ARRAY_SIZE);
            return nullptr;
        }
        SignalTileOp* newTask = &taskArray[taskCount];
        newTask->Init(taskId, addr, expectSum, cmpType, resetSignal);
        taskCount++;
        return newTask;
    }

    int32_t InsertTask(uint32_t taskId, int32_t* addr, int32_t expectSum, int32_t cmpType, bool resetSignal)
    {
        SignalTileOp* newTask = CreateTaskData(taskId, addr, expectSum, cmpType, resetSignal);
        if (newTask == nullptr) {
            return dynamic::DEVICE_MACHINE_ERROR;
        }
        uint32_t index = Hash(taskId);
        SignalTileOp* current = hashTable[index];
        hashTable[index] = newTask;
        newTask->next = current;
        return dynamic::DEVICE_MACHINE_OK;
    }

    SignalTileOp* FindTask(uint32_t taskId)
    {
        uint32_t index = Hash(taskId);
        SignalTileOp* current = hashTable[index];
        while (current != nullptr) {
            if (current->taskId_ == taskId) {
                return current;
            }
            current = current->next;
        }
        return nullptr;
    }

private:
    SignalTileOp taskArray[AICPU_TASK_ARRAY_SIZE];
    uint32_t taskCount{0};
    SignalTileOp* hashTable[AICPU_TASK_ARRAY_SIZE];
};

class CircularQueue {
public:
    CircularQueue() = default;

    inline int32_t Enqueue(SignalTileOp* task)
    {
        queue_[rear_] = task;
        rear_ = (rear_ + 1) & AICPU_TASK_ARRAY_SIZE_MOD;
        if (rear_ == front_) {
            DEV_ERROR(
                DistributedErrorCode::AICPU_TASK_NUM_EXCEED_LIMIT,
                "ctrl.task.pre.task.enqueue#: SignalTileOp queue_ is full, front=%u, rear=%u", front_, rear_);
            return dynamic::DEVICE_MACHINE_ERROR;
        }
        return dynamic::DEVICE_MACHINE_OK;
    }

    inline bool IsEmpty() const { return front_ == rear_; }

    inline int32_t Dequeue()
    {
        if (IsEmpty()) {
            DEV_ERROR(DistributedErrorCode::AICPU_TASK_QUEUE_EMPTY, "sche.task.end.task.dequeue#: Queue is empty.");
            return dynamic::DEVICE_MACHINE_ERROR;
        }
        front_ = (front_ + 1) & AICPU_TASK_ARRAY_SIZE_MOD;
        return dynamic::DEVICE_MACHINE_OK;
    }

    inline const SignalTileOp* operator[](uint16_t index) const { return queue_[index]; }

    inline int32_t Remove(uint16_t index)
    {
        queue_[index] = queue_[front_];
        return Dequeue();
    }

    int32_t PollCompleted(std::function<int32_t(SignalTileOp*)> processor)
    {
        uint16_t current = front_;
        uint16_t end = rear_;
        if (current > end) {
            end += AICPU_TASK_ARRAY_SIZE;
        }
        for (uint16_t i = current; i < end; ++i) {
            uint16_t actualIndex = i & AICPU_TASK_ARRAY_SIZE_MOD;
            SignalTileOp* task = queue_[actualIndex];
            if (task->PollCompleted()) {
                if (task->profData_ != nullptr) {
                    task->profData_->execEnd = dynamic::GetCycles();
                }
                int32_t ret = processor(task);
                if (ret != dynamic::DEVICE_MACHINE_OK) {
                    return ret;
                }
                ret = Remove(actualIndex);
                if (ret != dynamic::DEVICE_MACHINE_OK) {
                    return ret;
                }
            }
        }
        return dynamic::DEVICE_MACHINE_OK;
    }

private:
    SignalTileOp* queue_[AICPU_TASK_ARRAY_SIZE];
    uint16_t front_{0};
    uint16_t rear_{0};
};

class ShmemWaitUntilImpl {
public:
    inline void Init(npu::tile_fwk::dynamic::DynDeviceTask* dynDeviceTask)
    {
        dynDeviceTask_ = dynDeviceTask;
        funcDataList_ = reinterpret_cast<DynFuncData*>(&dynDeviceTask->GetDynFuncDataList()->At(0));
        hcclContextAddr_ = funcDataList_->startArgs->commContexts;
        commGroupNum_ = funcDataList_->startArgs->commGroupNum;
        hashMap_.Init();
    }

    inline int32_t EnqueueOp(uint64_t taskId, TaskStat* taskStat)
    {
        SignalTileOp* task = hashMap_.FindTask(taskId);
        if (task == nullptr) {
            DEV_ERROR(
                DistributedErrorCode::AICPU_TASKID_NOT_IN_MAP, "ctrl.task.pre.task.enqueue#: taskId=%lu not found",
                taskId);
            return dynamic::DEVICE_MACHINE_ERROR;
        }
        if (taskStat != nullptr) {
            task->profData_ = taskStat;
            task->profData_->taskId = static_cast<int32_t>(taskId);
            task->profData_->execStart = dynamic::GetCycles();
        }
        return runingTaskQueue_.Enqueue(task);
    }

    inline int32_t PrepareTask(uint64_t taskId, const npu::tile_fwk::dynamic::DevRelocVector<int32_t>& aicpuCode)
    {
        paramInfo_ = DecodeAicpuCode(aicpuCode);
        TensorInfo info = ShmemWaitUntilImpl::GetTensorInfo(taskId, aicpuCode);
        const int32_t expectedSum = info.expectedSum;
        const int32_t cmpType = info.cmpType;
        const bool resetSignal = info.resetSignal;
        int32_t stride = static_cast<int32_t>(paramInfo_.bufferStride);
        if (stride <= 0) {
            DEV_ERROR(
                DistributedErrorCode::INVALID_SHMEM_TENSOR,
                "ctrl.task.pre.task.prepare#: invalid buffer stride=%d (signalStride=%d)", stride, info.signalStride);
            return dynamic::DEVICE_MACHINE_ERROR;
        }
        if (info.offset.size() < 3) {
            DEV_ERROR(
                DistributedErrorCode::INVALID_SHMEM_TENSOR,
                "ctrl.task.pre.task.prepare#: invalid signal offset dim=%lu, expected >= 3",
                info.offset.size());
            return dynamic::DEVICE_MACHINE_ERROR;
        }

        const uint64_t ownerRankIndex = 0;
        uint64_t rowOffsetIndex = info.offset.size() - 2;
        uint64_t colOffsetIndex = info.offset.size() - 1;

        int32_t tileCols = (paramInfo_.rawShapeCol + paramInfo_.tileShapeCol - 1) / paramInfo_.tileShapeCol;
        int32_t tileRows = (paramInfo_.rawShapeRow + paramInfo_.tileShapeRow - 1) / paramInfo_.tileShapeRow;
        int32_t tileRow = info.offset[rowOffsetIndex] / paramInfo_.tileShapeRow;
        int32_t tileCol = info.offset[colOffsetIndex] / paramInfo_.tileShapeCol;
        int32_t tileIndex = tileRow * tileCols + tileCol;
        int32_t totalTileNum = tileRows * tileCols;
        if (tileIndex < 0 || tileIndex >= totalTileNum) {
            DEV_ERROR(
                DistributedErrorCode::INVALID_SHMEM_TENSOR,
                "ctrl.task.pre.task.prepare#: invalid tileIndex=%d, totalTileNum=%d, offset=[%u,%u], tile=[%u,%u],"
                " rawShape=[%u,%u]",
                tileIndex, totalTileNum, info.offset[rowOffsetIndex], info.offset[colOffsetIndex], paramInfo_.tileShapeRow,
                paramInfo_.tileShapeCol, paramInfo_.rawShapeRow, paramInfo_.rawShapeCol);
            return dynamic::DEVICE_MACHINE_ERROR;
        }
        uint64_t rankNum = GetRankNum(hcclContextAddr_, info.vaddr);
        if (info.offset[ownerRankIndex] >= rankNum) {
            DEV_ERROR(
                DistributedErrorCode::INVALID_SHMEM_TENSOR,
                "ctrl.task.pre.task.prepare#: ownerRank=%u out of range rankNum=%lu",
                info.offset[ownerRankIndex], rankNum);
            return dynamic::DEVICE_MACHINE_ERROR;
        }
        uint64_t maxTileNum = TileOp::Distributed::DecodeShmemAddrMaxTileNum(info.vaddr);
        if (static_cast<uint64_t>(totalTileNum) > maxTileNum) {
            DEV_ERROR(
                DistributedErrorCode::INVALID_SHMEM_TENSOR,
                "ctrl.task.pre.task.prepare#: totalTileNum=%d exceeds maxTileNum=%lu", totalTileNum, maxTileNum);
            return dynamic::DEVICE_MACHINE_ERROR;
        }
        uint64_t logicalSignalIndex =
            (static_cast<uint64_t>(info.offset[ownerRankIndex]) * maxTileNum +
                static_cast<uint64_t>(tileIndex)) * static_cast<uint64_t>(stride);
        uint64_t logicalSignalCapacity = rankNum * maxTileNum * static_cast<uint64_t>(stride);
        if (logicalSignalIndex >= logicalSignalCapacity) {
            DEV_ERROR(
                DistributedErrorCode::INVALID_SHMEM_TENSOR,
                "ctrl.task.pre.task.prepare#: signal index OOB index=%lu capacity=%lu (ownerRank=%u,"
                " tileIndex=%d, totalTileNum=%d, stride=%d, rankNum=%lu, maxTileNum=%lu)",
                logicalSignalIndex, logicalSignalCapacity, info.offset[ownerRankIndex], tileIndex, totalTileNum,
                stride, rankNum, maxTileNum);
            return dynamic::DEVICE_MACHINE_ERROR;
        }

        DEV_DEBUG(
            "ShmemWaitUntilImpl::EnqueueOp logical rawShape=[%u, %u],"
            "logical tile=[%u, %u], logical offset=[%u, %u], ownerRank=%u,"
            "actual rawShape=[%lu, %d], actual offset=[%u, %d], buffer maxTileNum=%lu, bufferStride=%u",
            paramInfo_.rawShapeRow, paramInfo_.rawShapeCol, paramInfo_.tileShapeRow, paramInfo_.tileShapeCol,
            info.offset[rowOffsetIndex], info.offset[colOffsetIndex], info.offset[ownerRankIndex], rankNum, totalTileNum,
            info.offset[ownerRankIndex], tileIndex, maxTileNum, paramInfo_.bufferStride);

        int32_t* addr = reinterpret_cast<int32_t*>(info.rawAddr) +
            (info.offset[ownerRankIndex] * maxTileNum + tileIndex) * stride;
        return hashMap_.InsertTask(taskId, addr, expectedSum, cmpType, resetSignal);
    }

    int32_t PollCompleted(npu::tile_fwk::dynamic::AiCoreManager* aiCoreManager);

    CircularQueue runingTaskQueue_;

private:
    HashMap hashMap_;
    uint32_t signalTileOpCount_{0};

    npu::tile_fwk::dynamic::DynDeviceTask* dynDeviceTask_;
    npu::tile_fwk::DynFuncData* funcDataList_;
    int64_t* hcclContextAddr_;
    uint64_t commGroupNum_{0};
    AicpuParamInfo paramInfo_;

    TensorInfo GetTensorInfo(uint64_t taskId, const npu::tile_fwk::dynamic::DevRelocVector<int32_t>& aicpuCode);
};

} // namespace npu::tile_fwk::Distributed
#endif // SHMEM_WAIT_UNTIL_H
