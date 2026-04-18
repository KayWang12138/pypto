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

struct alignas(64) SignalTileOp {
    void Init(uint64_t taskId, int32_t* addr, int32_t expectedSum, bool resetSignal)
    {
        taskId_ = taskId;
        addr_ = addr;
        expectedSum_ = expectedSum;
        resetSignal_ = resetSignal;
    }
    bool PollCompleted() const;

    uint64_t taskId_{0};
    int32_t expectedSum_{0};
    int32_t* addr_{nullptr};
    bool resetSignal_{false};
    SignalTileOp* next{nullptr};
    TaskStat* profData_{nullptr};
};

class WaitUntilTaskCache {
public:
    static WaitUntilTaskCache& Instance()
    {
        static WaitUntilTaskCache instance;
        return instance;
    }

    __attribute__((always_inline)) inline void Reset()
    {
        (void)memset_s(&hashTable_, sizeof(hashTable_), 0, sizeof(hashTable_));
        (void)memset_s(&taskCount_, sizeof(taskCount_), 0, sizeof(taskCount_));
    }

    __attribute__((always_inline)) inline uint32_t Hash(uint64_t taskId, uint32_t parallelIdx)
    {
        uint32_t baseIndex = parallelIdx * SLOT_PER_PARALLEL;
        uint32_t offset = ((taskId * 2654435761U) >> 22) & SLOT_PER_PARALLEL_MOD;
        return baseIndex + offset;
    }

    int32_t InsertTask(uint64_t taskId, uint32_t parallelIdx, int32_t* addr, int32_t expectSum, bool resetSignal)
    {
        uint32_t& count = taskCount_[parallelIdx];
        uint32_t poolIndex = parallelIdx * SLOT_PER_PARALLEL + count;

        if (poolIndex >= (parallelIdx + 1) * SLOT_PER_PARALLEL) {
            DEV_ERROR(
                DistributedErrorCode::AICPU_TASK_NUM_EXCEED_LIMIT,
                "ctrl.task.cache.insert#: taskCount_[%u]=%u >= SLOT_PER_PARALLEL=%u", parallelIdx, count,
                SLOT_PER_PARALLEL);
            return dynamic::DEVICE_MACHINE_ERROR;
        }

        SignalTileOp* newTask = &taskArrayPool_[poolIndex];
        newTask->Init(taskId, addr, expectSum, resetSignal);
        count++;

        uint32_t hashIndex = Hash(taskId, parallelIdx);
        SignalTileOp* current = hashTable_[hashIndex];
        hashTable_[hashIndex] = newTask;
        newTask->next = current;

        return dynamic::DEVICE_MACHINE_OK;
    }

    __attribute__((always_inline)) inline SignalTileOp* FindTask(uint64_t taskId, uint32_t parallelIdx)
    {
        uint32_t hashIndex = Hash(taskId, parallelIdx);
        SignalTileOp* current = hashTable_[hashIndex];
        while (current != nullptr) {
            if (current->taskId_ == taskId) {
                return current;
            }
            current = current->next;
        }
        return nullptr;
    }

    __attribute__((always_inline)) inline void ClearByParallelIdx(uint32_t parallelIdx)
    {
        uint32_t startIndex = parallelIdx * SLOT_PER_PARALLEL;
        (void)memset_s(
            &hashTable_[startIndex], SLOT_PER_PARALLEL * sizeof(SignalTileOp*), 0,
            SLOT_PER_PARALLEL * sizeof(SignalTileOp*));
        taskCount_[parallelIdx] = 0;
    }

private:
    WaitUntilTaskCache() = default;

    alignas(64) SignalTileOp* hashTable_[AICPU_TASK_ARRAY_SIZE];
    alignas(64) SignalTileOp taskArrayPool_[AICPU_TASK_ARRAY_SIZE];
    uint32_t taskCount_[npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM];
};

class CircularQueue {
public:
    CircularQueue()
    {
        (void)memset_s(&front_, sizeof(front_), 0, sizeof(front_));
        (void)memset_s(&rear_, sizeof(rear_), 0, sizeof(rear_));
    }

    static CircularQueue& Instance()
    {
        static CircularQueue instance;
        return instance;
    }

    inline int32_t Enqueue(uint32_t parallelIdx, SignalTileOp* task)
    {
        uint32_t startIndex = parallelIdx * SLOT_PER_PARALLEL;
        uint16_t& rear = rear_[parallelIdx];
        uint16_t& front = front_[parallelIdx];

        queue_[startIndex + (rear & SLOT_PER_PARALLEL_MOD)] = task;
        rear++;

        if ((rear & SLOT_PER_PARALLEL_MOD) == (front & SLOT_PER_PARALLEL_MOD)) {
            DEV_ERROR(
                DistributedErrorCode::AICPU_TASK_NUM_EXCEED_LIMIT,
                "ctrl.task.pre.task.enqueue#: SignalTileOp queue_ is full, parallelIdx=%u", parallelIdx);
            return dynamic::DEVICE_MACHINE_ERROR;
        }
        return dynamic::DEVICE_MACHINE_OK;
    }

    inline bool IsEmpty(uint32_t parallelIdx) const { return front_[parallelIdx] == rear_[parallelIdx]; }

    int32_t PollCompleted(uint32_t parallelIdx, std::function<int32_t(SignalTileOp*, uint32_t)> processor)
    {
        uint32_t startIndex = parallelIdx * SLOT_PER_PARALLEL;
        uint16_t& front = front_[parallelIdx];
        uint16_t rear = rear_[parallelIdx];

        if (front == rear)
            return dynamic::DEVICE_MACHINE_OK;

        constexpr uint16_t MAX_POLL_BATCH = 32;
        uint16_t polled = 0;

        for (uint16_t i = front; i < rear && polled < MAX_POLL_BATCH; ++i, ++polled) {
            uint16_t actualIndex = startIndex + (i & SLOT_PER_PARALLEL_MOD);
            SignalTileOp* task = queue_[actualIndex];
            if (task->PollCompleted()) {
                if (task->profData_ != nullptr) {
                    task->profData_->execEnd = dynamic::GetCycles();
                }
                int32_t ret = processor(task, parallelIdx);
                if (ret != dynamic::DEVICE_MACHINE_OK)
                    return ret;

                queue_[actualIndex] = queue_[startIndex + (front & SLOT_PER_PARALLEL_MOD)];
                front++;
            }
        }
        return dynamic::DEVICE_MACHINE_OK;
    }

    inline void ClearByParallelIdx(uint32_t parallelIdx)
    {
        front_[parallelIdx] = 0;
        rear_[parallelIdx] = 0;
    }

private:
    SignalTileOp* queue_[AICPU_TASK_ARRAY_SIZE];
    uint16_t front_[npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM];
    uint16_t rear_[npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM];
};

class ShmemWaitUntilImpl {
public:
    static inline void Init(AicpuTaskContext* ctx, npu::tile_fwk::dynamic::DynDeviceTask* dynDeviceTask)
    {
        ctx->Init(dynDeviceTask);
    }

    static __attribute__((always_inline)) inline int32_t EnqueueOp(
        AicpuTaskContext* ctx, uint32_t parallelIdx, uint64_t taskId, TaskStat* taskStat)
    {
        (void)ctx;
        SignalTileOp* task = WaitUntilTaskCache::Instance().FindTask(taskId, parallelIdx);
        if (task == nullptr) {
            DEV_DEBUG("ctrl.task.pre.task.enqueue#: taskId=%lu not found in cache, may be initializing", taskId);
            return dynamic::DEVICE_MACHINE_OK;
        }
        if (taskStat != nullptr) {
            task->profData_ = taskStat;
            task->profData_->taskId = static_cast<int32_t>(taskId);
            task->profData_->execStart = dynamic::GetCycles();
        }
        return CircularQueue::Instance().Enqueue(parallelIdx, task);
    }

    static inline int32_t PrepareTask(
        AicpuTaskContext* ctx, uint32_t parallelIdx, uint64_t taskId,
        const npu::tile_fwk::dynamic::DevRelocVector<int32_t>& aicpuCode)
    {
        ctx->paramInfo_ = DecodeAicpuCode(aicpuCode);
        TensorInfo info = GetTensorInfo(ctx, taskId, aicpuCode);
        const int32_t expectedSum = info.expectedSum;
        const bool resetSignal = info.resetSignal;

        int32_t tileCols =
            (ctx->paramInfo_.rawShapeCol + ctx->paramInfo_.tileShapeCol - 1) / ctx->paramInfo_.tileShapeCol;
        int32_t tileRows =
            (ctx->paramInfo_.rawShapeRow + ctx->paramInfo_.tileShapeRow - 1) / ctx->paramInfo_.tileShapeRow;
        int32_t tileRow = info.offset[SHMEM_DIM_ROW] / ctx->paramInfo_.tileShapeRow;
        int32_t tileCol = info.offset[SHMEM_DIM_COL] / ctx->paramInfo_.tileShapeCol;
        int32_t tileIndex = tileRow * tileCols + tileCol;
        int32_t totalTileNum = tileRows * tileCols;

        int32_t* addr =
            reinterpret_cast<int32_t*>(info.rawAddr) +
            CalcLinearOffset(totalTileNum, info.offset[OWNER_RANK_ID_INDEX], tileIndex) * ctx->paramInfo_.bufferStride;

        DEV_DEBUG(
            "PrepareTask baseAddr=0x%lx, actualAddr=0x%lx, logical rawShape=[%u, %u], logical tile=[%u, %u],"
            "logical offset=[%u, %u], ownerRank=%u, actual rawShape=[%lu, %d], actual offset=[%u, %d],"
            "buffer maxTileNum=%lu, bufferStride=%u",
            info.rawAddr, reinterpret_cast<uint64_t>(addr), ctx->paramInfo_.rawShapeRow, ctx->paramInfo_.rawShapeCol,
            ctx->paramInfo_.tileShapeRow, ctx->paramInfo_.tileShapeCol, info.offset[SHMEM_DIM_ROW],
            info.offset[SHMEM_DIM_COL], info.offset[OWNER_RANK_ID_INDEX], GetRankNum(ctx->hcclContextAddr_, info.vaddr),
            totalTileNum, info.offset[OWNER_RANK_ID_INDEX], tileIndex,
            TileOp::Distributed::DecodeShmemAddrMaxTileNum(info.vaddr), ctx->paramInfo_.bufferStride);

        return WaitUntilTaskCache::Instance().InsertTask(taskId, parallelIdx, addr, expectedSum, resetSignal);
    }

    static int32_t PollCompleted(
        AicpuTaskContext* ctx, uint32_t parallelIdx, npu::tile_fwk::dynamic::AiCoreManager* aiCoreManager);

private:
    static TensorInfo GetTensorInfo(
        AicpuTaskContext* ctx, uint64_t taskId, const npu::tile_fwk::dynamic::DevRelocVector<int32_t>& aicpuCode);
};

} // namespace npu::tile_fwk::Distributed
#endif // SHMEM_WAIT_UNTIL_H
