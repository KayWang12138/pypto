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
 * \file aicpu_task_manager.h
 * \brief
 */

#pragma once

#include <functional>
#include <atomic>
#include <vector>
#include <array>
#include <malloc.h>
#include <queue>

#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/device/distributed/common.h"
#include "machine/device/distributed/shmem_wait_until.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/device/dynamic/device_sche_context.h"
#include "interface/operation/opcode.h"
#include "interface/utils/common.h"
#include "interface/utils/distributed_error.h"

namespace npu::tile_fwk::dynamic {

class AicpuTaskManager {
public:
    enum TaskType {
        SHMEM_WAIT_UNTIL = 0,
        TASK_TYPE_NUM,
    };

    AicpuTaskManager(){};
    ~AicpuTaskManager(){};

    inline void InitDeviceArgs(DeviceArgs* deviceArgs)
    {
        sharedBuffer_ = deviceArgs->sharedBuffer;
        aicNum_ = deviceArgs->nrAic;
        aivNum_ = deviceArgs->nrAiv;
    }

    inline int32_t Init(SchDeviceTaskContext* devTaskCtx, bool profSwitch)
    {
        auto deviceTask = reinterpret_cast<DynDeviceTask*>(devTaskCtx->GetDeviceTask());
        auto& aicpuCtx = devTaskCtx->GetAicpuTaskCtx();
        uint32_t parallelIdx = devTaskCtx->GetParallelIdx();

        npu::tile_fwk::Distributed::ShmemWaitUntilImpl::Init(&aicpuCtx, deviceTask);

        npu::tile_fwk::Distributed::WaitUntilTaskCache::Instance().ClearByParallelIdx(parallelIdx);
        npu::tile_fwk::Distributed::CircularQueue::Instance().ClearByParallelIdx(parallelIdx);

        if (profSwitch) {
            KernelArgs* args = (KernelArgs*)(sharedBuffer_ + (aicNum_ + aivNum_) * SHARED_BUFFER_SIZE);
            aicpuCtx.aicpuTaskStat_ = (Metrics*)(args->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
        }

        uint8_t expected = static_cast<uint8_t>(AicpuTaskPrepareState::NOT_STARTED);
        if (deviceTask->aicpuTaskPrepareState.compare_exchange_strong(
                expected, static_cast<uint8_t>(AicpuTaskPrepareState::PREPARING), std::memory_order_relaxed,
                std::memory_order_relaxed)) {
            int32_t ret = PrepareAicpuTask(devTaskCtx);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                deviceTask->aicpuTaskPrepareState.store(
                    static_cast<uint8_t>(AicpuTaskPrepareState::NOT_STARTED), std::memory_order_relaxed);
                return ret;
            }
            deviceTask->aicpuTaskPrepareState.store(
                static_cast<uint8_t>(AicpuTaskPrepareState::COMPLETED), std::memory_order_release);
            return DEVICE_MACHINE_OK;
        }

        return DEVICE_MACHINE_OK;
    }

    __attribute__((always_inline)) inline int32_t TaskProcess(SchDeviceTaskContext* devTaskCtx, uint64_t& taskCount)
    {
        auto* readyQueue = devTaskCtx->readyAicpuFunctionQue;

        if (likely(IsReady(devTaskCtx))) {
            if (__atomic_load_n(&readyQueue->tail, __ATOMIC_RELAXED) ==
                __atomic_load_n(&readyQueue->head, __ATOMIC_RELAXED)) {
                return DEVICE_MACHINE_OK;
            }
            ReadyQueueLock(readyQueue);
            uint64_t taskIdx = readyQueue->head;
            taskCount = readyQueue->tail - readyQueue->head;
            readyQueue->head += taskCount;
            ReadyQueueUnLock(readyQueue);

            for (uint32_t i = 0; i < taskCount; ++i) {
                auto ret = TaskDispatch(devTaskCtx, readyQueue->elem[taskIdx + i]);
                if (ret != DEVICE_MACHINE_OK) {
                    return ret;
                }
            }
            return DEVICE_MACHINE_OK;
        }
        return DEVICE_MACHINE_OK;
    }

    __attribute__((always_inline)) inline int32_t TaskPoll(
        SchDeviceTaskContext* devTaskCtx, AiCoreManager* aiCoreManager)
    {
        auto& aicpuCtx = devTaskCtx->GetAicpuTaskCtx();
        uint32_t parallelIdx = devTaskCtx->GetParallelIdx();
        return npu::tile_fwk::Distributed::ShmemWaitUntilImpl::PollCompleted(&aicpuCtx, parallelIdx, aiCoreManager);
    }

    __attribute__((always_inline)) inline int32_t DispatchAndPollTask(
        SchDeviceTaskContext* devTaskCtx, uint64_t& taskCount, AiCoreManager* aiCoreManager)
    {
        if (likely(IsReady(devTaskCtx))) {
            int32_t ret = TaskProcess(devTaskCtx, taskCount);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
            return TaskPoll(devTaskCtx, aiCoreManager);
        }
        return DEVICE_MACHINE_OK;
    }

    inline bool Finished(SchDeviceTaskContext* devTaskCtx)
    {
        uint32_t parallelIdx = devTaskCtx->GetParallelIdx();
        return npu::tile_fwk::Distributed::CircularQueue::Instance().IsEmpty(parallelIdx);
    }

    inline int32_t SyncAicpuTaskFinish(SchDeviceTaskContext* devTaskCtx, AiCoreManager* aiCoreManager)
    {
        int64_t start_cycles = GetCycles();
        while (!Finished(devTaskCtx)) {
            auto ret = TaskPoll(devTaskCtx, aiCoreManager);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
            if (GetCycles() - start_cycles > TIMEOUT_CYCLES) {
                DEV_ERROR(
                    DistributedErrorCode::AICPU_TASK_TIMEOUT,
                    "#sche.task.end.sync.timeout: SyncAicpuTaskFinish timeout.");
                return DEVICE_MACHINE_TIMEOUT_SYNC_AICPU_FINISH;
            }
        }
        return DEVICE_MACHINE_OK;
    }

private:
    uint64_t sharedBuffer_;
    uint32_t aicNum_;
    uint32_t aivNum_;

    inline void ReadyQueueLock(ReadyCoreFunctionQueue* readyQueue)
    {
        while (!__sync_bool_compare_and_swap(&readyQueue->lock, 0, 1)) {
#ifdef __aarch64__
            asm volatile("wfe" ::: "memory");
#else
            asm volatile("pause" ::: "memory");
#endif
        }
    }

    inline void ReadyQueueUnLock(ReadyCoreFunctionQueue* readyQueue)
    {
        __atomic_store_n(&readyQueue->lock, 0, __ATOMIC_RELEASE);
#ifdef __aarch64__
        asm volatile("sev" ::: "memory");
#endif
    }

    inline TaskType GetTaskType(SchDeviceTaskContext* devTaskCtx, uint64_t taskId)
    {
        auto& aicpuCtx = devTaskCtx->GetAicpuTaskCtx();
        auto* deviceTask = aicpuCtx.dynDeviceTask_;
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto callList = deviceTask->dynFuncDataCacheList[funcId].calleeList;
        auto& code = deviceTask->aicpuLeafBinary[callList[opIndex]].aicpuLeafCode;
        auto taskType = TaskType::TASK_TYPE_NUM;
        switch (code[0]) {
            case static_cast<uint32_t>(Opcode::OP_SHMEM_WAIT_UNTIL):
                taskType = TaskType::SHMEM_WAIT_UNTIL;
                break;
            default:
                break;
        }
        return taskType;
    }

    inline int32_t TaskDispatch(SchDeviceTaskContext* devTaskCtx, uint64_t taskId)
    {
        auto& aicpuCtx = devTaskCtx->GetAicpuTaskCtx();
        uint32_t parallelIdx = devTaskCtx->GetParallelIdx();

        int32_t ret = DEVICE_MACHINE_OK;
        auto taskType = GetTaskType(devTaskCtx, taskId);
        if (taskType < TaskType::TASK_TYPE_NUM) {
            TaskStat* taskStat = nullptr;
            if (aicpuCtx.aicpuTaskStat_ != nullptr) {
                taskStat = &(aicpuCtx.aicpuTaskStat_->tasks[aicpuCtx.aicpuTaskStat_->taskCount]);
                ++aicpuCtx.aicpuTaskStat_->taskCount;
            }
            ret = npu::tile_fwk::Distributed::ShmemWaitUntilImpl::EnqueueOp(&aicpuCtx, parallelIdx, taskId, taskStat);
        }
        return ret;
    }

    inline int32_t PrepareAicpuTask(SchDeviceTaskContext* devTaskCtx)
    {
        auto& aicpuCtx = devTaskCtx->GetAicpuTaskCtx();
        uint32_t parallelIdx = devTaskCtx->GetParallelIdx();
        auto* deviceTask = aicpuCtx.dynDeviceTask_;

        for (uint64_t funcId = 0; funcId < deviceTask->dynFuncDataCacheListSize; ++funcId) {
            auto callList = deviceTask->dynFuncDataCacheList[funcId].calleeList;
            for (size_t opIndex = 0; opIndex < deviceTask->dynFuncDataCacheList[funcId].devFunc->GetOperationSize();
                 ++opIndex) {
                auto coreType = deviceTask->cceBinary[callList[opIndex]].coreType;
                if (unlikely(coreType != static_cast<int>(MachineType::AICPU))) {
                    continue;
                }
                uint32_t taskId = MakeTaskID(funcId, opIndex);
                auto& code = deviceTask->aicpuLeafBinary[callList[opIndex]].aicpuLeafCode;
                auto ret =
                    npu::tile_fwk::Distributed::ShmemWaitUntilImpl::PrepareTask(&aicpuCtx, parallelIdx, taskId, code);
                if (ret != DEVICE_MACHINE_OK) {
                    return ret;
                }
            }
        }
        return DEVICE_MACHINE_OK;
    }

    __attribute__((always_inline)) inline bool IsReady(SchDeviceTaskContext* devTaskCtx)
    {
        auto deviceTask = reinterpret_cast<DynDeviceTask*>(devTaskCtx->GetDeviceTask());
        return deviceTask->aicpuTaskPrepareState.load(std::memory_order_relaxed) ==
               static_cast<uint8_t>(AicpuTaskPrepareState::COMPLETED);
    }
};
} // namespace npu::tile_fwk::dynamic
