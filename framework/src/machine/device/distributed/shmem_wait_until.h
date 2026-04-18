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
    void Init(uint64_t taskId, int32_t* addr, int32_t expectedSum, bool resetSignal)
    {
        taskId_ = taskId;
        addr_ = addr;
        expectedSum_ = expectedSum;
        resetSignal_ = resetSignal;
    }
    bool PollCompleted() const;

    SignalTileOp* next{nullptr};
    uint64_t taskId_{0};
    int32_t* addr_{nullptr};
    int32_t expectedSum_{0};
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

    SignalTileOp* CreateTaskData(uint32_t taskId, int32_t* addr, int32_t expectSum, bool resetSignal)
    {
        if (taskCount >= AICPU_TASK_ARRAY_SIZE) {
            DEV_ERROR(
                DistributedErrorCode::AICPU_TASK_NUM_EXCEED_LIMIT,
                "ctrl.task.pre.task.create#: taskCount=%u >= AICPU_TASK_ARRAY_SIZE=%lu", taskCount,
                AICPU_TASK_ARRAY_SIZE);
            return nullptr;
        }
        SignalTileOp* newTask = &taskArray[taskCount];
        newTask->Init(taskId, addr, expectSum, resetSignal);
        taskCount++;
        return newTask;
    }

    int32_t InsertTask(uint32_t taskId, int32_t* addr, int32_t expectSum, bool resetSignal)
    {
        SignalTileOp* newTask = CreateTaskData(taskId, addr, expectSum, resetSignal);
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
        uint16_t currentRear = __atomic_load_n(&rear_, __ATOMIC_RELAXED);
        uint16_t nextRear = (currentRear + 1) & AICPU_TASK_ARRAY_SIZE_MOD;
        if (nextRear == __atomic_load_n(&front_, __ATOMIC_RELAXED)) {
            DEV_ERROR(
                DistributedErrorCode::AICPU_TASK_NUM_EXCEED_LIMIT,
                "ctrl.task.pre.task.enqueue#: SignalTileOp queue_ is full, front=%u, rear=%u",
                __atomic_load_n(&front_, __ATOMIC_RELAXED), currentRear);
            return dynamic::DEVICE_MACHINE_ERROR;
        }
        queue_[currentRear] = task;
        __atomic_store_n(&rear_, nextRear, __ATOMIC_RELEASE);
        return dynamic::DEVICE_MACHINE_OK;
    }

    inline bool IsEmpty() const {
        return __atomic_load_n(&front_, __ATOMIC_RELAXED) == __atomic_load_n(&rear_, __ATOMIC_RELAXED);
    }

    int32_t PollCompleted(std::function<int32_t(SignalTileOp*)> processor)
    {
        // Fast path: if queue is empty, return immediately
        // Avoid unnecessary iteration on empty queue for performance
        if (IsEmpty()) {
            return dynamic::DEVICE_MACHINE_OK;
        }

        // Multiple consumers can poll concurrently, each processes from front forward
        // Use atomic fetch to claim tasks, no locking needed
        uint16_t currentFront;
        uint16_t end = __atomic_load_n(&rear_, __ATOMIC_ACQUIRE);

        while (true) {
            currentFront = __atomic_load_n(&front_, __ATOMIC_ACQUIRE);
            if (currentFront == end) {
                break;
            }

            SignalTileOp* task = queue_[currentFront];
            if (task->PollCompleted()) {
                if (task->profData_ != nullptr) {
                    task->profData_->execEnd = dynamic::GetCycles();
                }
                // Try to claim this task by advancing front
                if (__sync_bool_compare_and_swap(&front_, currentFront, (currentFront + 1) & AICPU_TASK_ARRAY_SIZE_MOD)) {
                    int32_t ret = processor(task);
                    if (ret != dynamic::DEVICE_MACHINE_OK) {
                        return ret;
                    }
                }
            } else {
                // Task not completed, skip for now
                end = __atomic_load_n(&rear_, __ATOMIC_ACQUIRE);
                if (currentFront + 1 == end) {
                    break;
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
        const bool resetSignal = info.resetSignal;

        int32_t tileCols = (paramInfo_.rawShapeCol + paramInfo_.tileShapeCol - 1) / paramInfo_.tileShapeCol;
        int32_t tileRows = (paramInfo_.rawShapeRow + paramInfo_.tileShapeRow - 1) / paramInfo_.tileShapeRow;
        int32_t tileRow = info.offset[SHMEM_DIM_ROW] / paramInfo_.tileShapeRow;
        int32_t tileCol = info.offset[SHMEM_DIM_COL] / paramInfo_.tileShapeCol;
        int32_t tileIndex = tileRow * tileCols + tileCol;
        int32_t totalTileNum = tileRows * tileCols;

        int32_t* addr =
            reinterpret_cast<int32_t*>(info.rawAddr) +
            CalcLinearOffset(totalTileNum, info.offset[OWNER_RANK_ID_INDEX], tileIndex) * paramInfo_.bufferStride;

        DEV_DEBUG(
            "PrepareTask baseAddr=0x%lx, actualAddr=0x%lx, logical rawShape=[%u, %u], logical tile=[%u, %u],"
            "logical offset=[%u, %u], ownerRank=%u, actual rawShape=[%lu, %d], actual offset=[%u, %d],"
            "buffer maxTileNum=%lu, bufferStride=%u",
            info.rawAddr, reinterpret_cast<uint64_t>(addr), paramInfo_.rawShapeRow, paramInfo_.rawShapeCol,
            paramInfo_.tileShapeRow, paramInfo_.tileShapeCol, info.offset[SHMEM_DIM_ROW], info.offset[SHMEM_DIM_COL],
            info.offset[OWNER_RANK_ID_INDEX], GetRankNum(hcclContextAddr_, info.vaddr), totalTileNum,
            info.offset[OWNER_RANK_ID_INDEX], tileIndex, TileOp::Distributed::DecodeShmemAddrMaxTileNum(info.vaddr),
            paramInfo_.bufferStride);

        return hashMap_.InsertTask(taskId, addr, expectedSum, resetSignal);
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
