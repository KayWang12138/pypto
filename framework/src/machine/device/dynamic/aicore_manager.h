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
 * \file aicore_manager.h
 * \brief
 */

#pragma once
#include <cstdint>
#include <sys/ioctl.h>
#include <functional>
#include <vector>
#include <atomic>
#include <array>
#include <semaphore.h>
#include "machine/utils/dynamic/dev_start_args.h"
#include "securec.h"
#include "device_common.h"
#include "tilefwk/config.h"
#include "tilefwk/aicore_print.h"
#include "interface/utils/common.h"
#include "interface/operation/opcode.h"
#include "interface/schema/schema.h"
#include "machine/utils/concurrent_queue.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/utils/dynamic/device_task.h"
#include "machine/utils/dynamic/small_array.h"
#include "machine/utils/dynamic/spsc_queue.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/utils/device_log.h"
#include "machine/device/dynamic/aicore_prof.h"
#include "machine/device/dynamic/aicore_hal.h"
#include "machine/device/dynamic/aicpu_task_manager.h"
#include "machine/device/dynamic/device_utils.h"
#include "machine/device/dynamic/wrap_manager.h"
#include "machine/device/dump/aicore_dump.h"
#include <chrono>

namespace npu::tile_fwk::dynamic {

constexpr uint32_t REG_31_BITS = 0x7FFFFFFF;
constexpr uint32_t REG_32_BITS = 0xFFFFFFFF;
#define REG_LOW_TASK_ID(regVal) (regVal) & REG_31_BITS // 低31位存储的taskid
#define REG_LOW_TASK_STATE(regVal) ((regVal)&REG_32_BITS) >> 31 // 低32位存储的task的状态
constexpr uint32_t TASK_FIN_STATE = 1; // 任务执行完成完成
constexpr uint32_t TASK_ACK_STATE = 0; // 收到任务状态，没执行完成
constexpr uint32_t REG_TASK_NUM = 2; // 一次寄存器task个数

constexpr uint32_t AIV_CORE_COUNT = 48;
constexpr uint32_t AIC_CORE_COUNT = 24;
constexpr uint32_t TOTAL_CORE_COUNT = AIV_CORE_COUNT + AIC_CORE_COUNT;

#define MAX_QUEUED_TASKS 4096

typedef uint64_t aicoreFunction_t;
typedef uint32_t aicoreTask_t;
typedef uint32_t aicoreCore_t;
typedef uint64_t aicorePair_t;

constexpr aicoreFunction_t aicoreNullFunction = 0xFFFFFFFFFFFFFFFFUL;
constexpr aicoreTask_t aicoreNullTask = 0xFFFFFFFFUL;
constexpr aicoreCore_t aicoreNullCore = 0xFFFFFFFFUL;
constexpr aicorePair_t aicoreNullPair = 0xFFFFFFFFFFFFFFFFUL;

// typedef pypto::utils::ConcurrentQueue<aicoreTask_t, aicoreNullTask> taskQueue_t;
typedef pypto::utils::ConcurrentQueue<aicoreCore_t, aicoreNullCore, TOTAL_CORE_COUNT> coreQueue_t;
typedef pypto::utils::ConcurrentQueue<aicorePair_t, aicoreNullPair, TOTAL_CORE_COUNT> pairQueue_t;

struct TaskInfo {
    int coreIdx;
    uint64_t taskId;
    TaskInfo(int idx, uint64_t id) : coreIdx(idx), taskId(id) {}
};

class AiCoreManager {
public:
    AiCoreManager() = default;
    virtual ~AiCoreManager() = default;

    void SendDevTaskModel(DeviceTask *devTask) {
        int64_t funcdata;
        auto dyntask = (DynDeviceTask *)devTask;
        funcdata = static_cast<int64_t>(PtrToValue(dyntask->GetDynFuncDataList()));
        ForEachManageAicore([&](int coreIdx) {
            aicoreHal_.InitTaskData(coreIdx, funcdata, (uint64_t)nullptr);
        });
    }
    inline void SetSchduleContext(SchduleContext * context) {
        this->context_ = context;
    }

    inline bool CheckAndResetReg() {
        if (!validGetPgMask_) {
            return true;
        }
        bool isValid = true;
        if (aicoreHal_.GetRegSprDataMainBase() == DAV_3510::REG_SPR_DATA_MAIN_BASE) {
            return true;
        }
        auto regAddrs = aicoreHal_.GetRegAddrs();
        uint32_t regNum = aicoreHal_.GetregNum();
        for (uint32_t coreIdx = 0; coreIdx < regNum; ++coreIdx) {
            if (regAddrs[coreIdx] == 0) {
                continue;
            }
            uint32_t currentStatus = *(reinterpret_cast<volatile uint32_t*>(regAddrs[coreIdx] + REG_SPR_FAST_PATH_ENABLE));
            if (currentStatus != REG_SPR_FAST_PATH_CLOSE) {
                isValid = false;
                *(reinterpret_cast<volatile uint32_t*>(regAddrs[coreIdx] + REG_SPR_FAST_PATH_ENABLE)) = REG_SPR_FAST_PATH_CLOSE;
            }
        }
        return isValid;
    }

inline void InitDevTask(DeviceTaskCtrl *taskCtrl)
{
        curTaskCtrl_ = taskCtrl;
        curDevTask_ = taskCtrl->devTask;
        curTaskId_ = taskCtrl->taskId;

        if (!preFetchSuccess_) {
            SendDevTaskModel(curDevTask_);
        }

        readyAicCoreFunctionQue_ = reinterpret_cast<ReadyCoreFunctionQueue *>(curDevTask_->readyAicCoreFunctionQue);
        readyAivCoreFunctionQue_ = reinterpret_cast<ReadyCoreFunctionQueue *>(curDevTask_->readyAivCoreFunctionQue);

        // If I am the lead AICPU scheduler, perform initailization steps
        if (isLeaderScheduler_ == true)
        {
            // Allocating queues
            // auto availableVectorTaskQueue = new coreQueue_t();
            // auto availableCubeTaskQueue = new coreQueue_t();
            // curDevTask_->availableVectorTaskQueue = (uint64_t) availableVectorTaskQueue;
            // curDevTask_->availableCubeTaskQueue = (uint64_t) availableCubeTaskQueue;

            // // // Adding initial set of tasks 
            // auto readyAicCoreFunctionQue = (ReadyCoreFunctionQueue *)curDevTask_->readyAicCoreFunctionQue;
            // auto readyAivCoreFunctionQue = (ReadyCoreFunctionQueue *)curDevTask_->readyAivCoreFunctionQue;
            // // DEV_ERROR("[AICPU %d] AIV Tasks: %lu", aicpuIdx_, readyAivCoreFunctionQue->size());
            // // DEV_ERROR("[AICPU %d] AIC Tasks: %lu", aicpuIdx_, readyAicCoreFunctionQue->size());
            // for (size_t i = 0; i < readyAivCoreFunctionQue->Size(); i++) availableVectorTaskQueue->push((uint32_t)readyAivCoreFunctionQue->GetBuffer()[i]);
            // for (size_t i = 0; i < readyAicCoreFunctionQue->Size(); i++) availableCubeTaskQueue->push((uint32_t)readyAicCoreFunctionQue->GetBuffer()[i]);

        //     // Allocating core queues
        //     auto availableVectorCoreQueue = new coreQueue_t(AIV_CORE_COUNT);
        //     auto availableCubeCoreQueue = new coreQueue_t(AIC_CORE_COUNT);
        //     curDevTask_->availableVectorCoreQueue = (uint64_t) availableVectorCoreQueue;
        //     curDevTask_->availableCubeCoreQueue = (uint64_t) availableCubeCoreQueue;
        //     for (uint32_t i = AIC_CORE_COUNT; i < AIC_CORE_COUNT + AIV_CORE_COUNT; i++) availableVectorCoreQueue->push(i);
        //     for (uint32_t i = 0; i < AIC_CORE_COUNT; i++) availableCubeCoreQueue->push(i);

            // Setting task as initialized, allowing others to continue
            curDevTask_->isTaskInitialized = true;
        }
        
        // If I am not a lead AICPU scheduler, wait until initialization is ready
        if (isLeaderScheduler_ == false) while(curDevTask_->isTaskInitialized == false){ /* Busy wait */ };

        freeACoreQueue_[(int)CoreType::AIV] = new coreQueue_t();
        freeACoreQueue_[(int)CoreType::AIC] = new coreQueue_t();
        busyACoreQueue_[(int)CoreType::AIV] = new pairQueue_t();
        busyACoreQueue_[(int)CoreType::AIC] = new pairQueue_t();
        freeBCoreQueue_[(int)CoreType::AIV] = new pairQueue_t();
        freeBCoreQueue_[(int)CoreType::AIC] = new pairQueue_t();
        busyBCoreQueue_[(int)CoreType::AIV] = new pairQueue_t();
        busyBCoreQueue_[(int)CoreType::AIC] = new pairQueue_t();

        for (int i = aivStart_; i < aivEnd_; i++) freeACoreQueue_[(int)CoreType::AIV]->push(i);
        for (int i = aicStart_; i < aicEnd_; i++) freeACoreQueue_[(int)CoreType::AIC]->push(i);

        // DEV_ERROR(0, "[AICPU %d] Vector Ready: %u, (%lu + %lu)", aicpuIdx_, GetReadyCoreNum(CoreType::AIV), freeACoreQueue_[(int)CoreType::AIV]->size(), busyACoreQueue_[(int)CoreType::AIV]->size());
        // DEV_ERROR(0, "[AICPU %d] Cube Ready:   %u, (%lu + %lu)", aicpuIdx_, GetReadyCoreNum(CoreType::AIC), freeACoreQueue_[(int)CoreType::AIC]->size(), busyACoreQueue_[(int)CoreType::AIC]->size());
    }

    inline void CountSendTask(uint64_t &sentAic, uint64_t &sentAiv) {
        sentAic = context_->sendCnt_[static_cast<int>(CoreType::AIC)];
        sentAiv = context_->sendCnt_[static_cast<int>(CoreType::AIV)];
        context_->waitTaskCnt_[static_cast<int>(CoreType::AIC)] += sentAic;
        context_->waitTaskCnt_[static_cast<int>(CoreType::AIV)] += sentAiv;
        context_->sendCnt_[static_cast<int>(CoreType::AIC)] = 0;
        context_->sendCnt_[static_cast<int>(CoreType::AIV)] = 0;
    }

    inline void RunCoreTask(uint64_t& sent) {
        DispatchAiCoreTask(CoreType::AIC, readyAicCoreFunctionQue_);
        DispatchAiCoreTask(CoreType::AIV, readyAivCoreFunctionQue_);

        uint64_t sentAic = 0;
        uint64_t sentAiv = 0;
        CountSendTask(sentAic, sentAiv);

        sent = 0UL;
        sent += (sentAic + sentAiv);
    }

    inline void RunTask(DeviceTaskCtrl *taskCtrl) {
        InitDevTask(taskCtrl);

        const auto t0 = std::chrono::high_resolution_clock::now();

        ExecuteTask(taskCtrl);

        const auto tf = std::chrono::high_resolution_clock::now();
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tf - t0).count();
        DEV_ERROR(0, "[AICPU %d] Running Time: %ldns", aicpuIdx_, ns);
    }

    inline void ExecuteTask(DeviceTaskCtrl *taskCtrl) {
        ProcessTask(taskCtrl);
        ProcessTaskLoop(taskCtrl);
        WaitAllAicoreFinish();
    }


    inline bool checkCoreFinished(const int coreIdx)
    {
        uint64_t finTaskVal = aicoreHal_.GetFinishedTask(coreIdx);
        uint32_t regLFinTaskState = REG_LOW_TASK_STATE(finTaskVal);
        if (regLFinTaskState == TASK_FIN_STATE) return true;
        return false;
    }

    inline int WaitAllAicoreFinish()
    {
        for (auto i = aicStart_; i < aicEnd_; i++) {
            while (checkCoreFinished(i) == false) { /* busy wait */}
            NormalStopSingleCore(i);
        }

        for (auto i = aivStart_; i < aivEnd_; i++) {
            while (checkCoreFinished(i) == false) { /* busy wait */}
            NormalStopSingleCore(i);
        }

        __sync_synchronize();

        return 0;
    }

    inline void ProcessTask(DeviceTaskCtrl *taskCtrl) {
        uint64_t curSent = 0UL;
        if (!taskCtrl->isFirstDevTask) {
            RunCoreTask(curSent);
            taskCtrl->finishedFunctionCnt.fetch_add(curSent, std::memory_order_relaxed);
        }
    }

    inline void ProcessTaskLoop(DeviceTaskCtrl *taskCtrl) {
        uint32_t lastSent = 0;
        uint32_t allSentCnt = taskCtrl->finishedFunctionCnt.load(std::memory_order_relaxed);
        while (allSentCnt < curDevTask_->coreFunctionCnt) {
            uint64_t curSent = 0;
            RunCoreTask(curSent);
            if (likely(curSent == 0)) {
                if (lastSent > 0) {
                    taskCtrl->finishedFunctionCnt.fetch_add(lastSent, std::memory_order_relaxed);
                    lastSent = 0;
                }
            } else {
                lastSent += curSent;
            }

            // To prevent an unnecessary execution of RunCoreTask after the final batch of tasks is sent.
            allSentCnt = taskCtrl->finishedFunctionCnt.load(std::memory_order_relaxed) + lastSent;
        }

        if (lastSent > 0) {
            // Other SCH-AICPU are still waiting for the taskCtrl->finishedFunctionCnt actual value.
            taskCtrl->finishedFunctionCnt.fetch_add(lastSent, std::memory_order_relaxed);
        }
    }

    void ResetRegAll() {
        ForEachManageAicore([this](int coreIdx) {
            if (aicoreHal_.ReadPathReg(coreIdx) == REG_SPR_FAST_PATH_OPEN) {
                aicoreHal_.SetReadyQueue(coreIdx, AICORE_TASK_STOP + 1);
                aicoreHal_.WriteReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
            } else {
                aicoreHal_.SetReadyQueue(coreIdx, AICORE_TASK_STOP + 1);
            }
        });
    }


    inline void RunManager(int threadIdx, DevStartArgs *devStartArgs, DeviceArgs *deviceArgs, int schedIdx) {
        Init(threadIdx, deviceArgs, schedIdx);
        DeviceTaskCtrl *taskCtrl = nullptr;
        prefetchedTaskQueue_ = &(devStartArgs->deviceRuntimeDataDesc.taskQueueList[schedIdx_]);
        if constexpr (IsDeviceMode()) {
            HandShake();
            devStartArgs->syncFlag = 1;
        }
        while (true) {
            taskCtrl = preFetchSuccess_ ? preFetchNextDevTaskCtrl_ : prefetchedTaskQueue_->Dequeue();
            if (taskCtrl == nullptr) {
                break;
            }
            RunTask(taskCtrl);
            taskCtrl->PutTask(0);
        }
    }

private:

    static inline aicorePair_t encodePair(const uint64_t coreId, const uint64_t taskId) { return aicorePair_t ((taskId << 32) + (coreId & 0x00000000FFFFFFFFUL)); }
    static inline uint64_t decodePairTask(const aicorePair_t pair) { return pair >> 32; }
    static inline uint64_t decodePairCore(const aicorePair_t pair) { return pair & 0x00000000FFFFFFFFUL; }

    inline bool PreFetchNextDevTask() {
        preFetchNextDevTaskCtrl_ = nullptr;
        preFetchSuccess_ = prefetchedTaskQueue_->TryDequeue(preFetchNextDevTaskCtrl_);
        return preFetchSuccess_;
    }

    inline void NormalStopSingleCore(int coreIdx) {
        aicoreHal_.SetReadyQueue(coreIdx, AICORE_TASK_STOP + 1);
        aicoreHal_.ResetShakeBuf(coreIdx);
    }

    inline uint32_t GetReadyCoreNum(CoreType type) {
        return freeACoreQueue_[(int)type]->size() + freeBCoreQueue_[(int)type]->size();
    }

    inline uint64_t TryBatchSendTask(CoreType type, ReadyCoreFunctionQueue* readyQue)
    {
        if (__atomic_load_n(&readyQue->tail, __ATOMIC_RELAXED) == __atomic_load_n(&readyQue->head, __ATOMIC_RELAXED)) {
            return 0;
        }
        
        uint32_t ready = GetReadyCoreNum(type);

        if (ready == 0 ) {
            return 0;
        }

        ReadyQueueLock(readyQue);
        uint32_t head = readyQue->head;
        const uint32_t availableTasks = readyQue->Size();
        uint32_t taskCount = std::min(ready, availableTasks);
        readyQue->head += taskCount;
        ReadyQueueUnLock(readyQue);

        BatchSendTask(type, &readyQue->elem[head], taskCount);
        return taskCount;
    }

    inline uint32_t BatchSendTask(CoreType type, uint32_t *newTask, uint32_t taskCount) {
        uint32_t sendCnt = 0;

        const auto freeACoreCount = freeACoreQueue_[(int)type]->size();
        while (sendCnt < taskCount && sendCnt < freeACoreCount)
        {
            uint32_t coreId = freeACoreQueue_[(int)type]->pop();
            SendTaskToAiCore(type, coreId, newTask[sendCnt]);
            auto pair = encodePair(coreId, newTask[sendCnt]);
            busyACoreQueue_[(int)type]->push(pair);
            sendCnt++;
        }

        while (sendCnt < taskCount)
        {
            auto pair = freeBCoreQueue_[(int)type]->pop();
            auto coreId = decodePairCore(pair);
            SendTaskToAiCore(type, coreId, newTask[sendCnt]);
            busyBCoreQueue_[(int)type]->push(pair);
            sendCnt++;
        }

        return sendCnt;
    }

    inline void DispatchAiCoreTask(CoreType type, ReadyCoreFunctionQueue* readyQue) {
        if (context_->waitTaskCnt_[(int)type] > 0) {
            ResolveDepForAllAiCore(type);
        }
        TryBatchSendTask(type, readyQue);
    }

    inline void SendTaskToAiCore(CoreType type, int coreIdx, uint32_t newTask) {
        aicoreHal_.SetReadyQueue(coreIdx, newTask + 1);
        context_->sendCnt_[(int)type]++;
    }

    inline void PushReadyQue(ReadyCoreFunctionQueue *readyQue, void *idList, uint32_t idCnt) const {
        ReadyQueueLock(readyQue);
        memcpy_s(&readyQue->elem[readyQue->tail], idCnt * sizeof(uint32_t), (uint8_t *)idList, idCnt * sizeof(uint32_t));
        readyQue->tail += idCnt;
        ReadyQueueUnLock(readyQue);
    }

    inline void BatchPushReadyQueue() {
        uint32_t aicIndex = static_cast<uint32_t>(CoreType::AIC);
        uint32_t aivIndex = static_cast<uint32_t>(CoreType::AIV);

        if (context_->readyCount[aicIndex] > 0) {
            PushReadyQue(readyAicCoreFunctionQue_, context_->readyIds[aicIndex], context_->readyCount[aicIndex]);
            TryBatchSendTask(CoreType::AIC, readyAicCoreFunctionQue_);
            context_->readyCount[aicIndex] = 0;
        }

        if (context_->readyCount[aivIndex] > 0) {
            PushReadyQue(readyAivCoreFunctionQue_, context_->readyIds[aivIndex], context_->readyCount[aivIndex]);
            TryBatchSendTask(CoreType::AIV, readyAivCoreFunctionQue_);
            context_->readyCount[aivIndex] = 0;
        }
    }

    inline void ResolveDepForAllAiCore(CoreType type)
    {
        // Handling Busy A queue pairings
        busyACoreQueue_[(int)type]->lock();
        size_t busyAQueuePairCount = busyACoreQueue_[(int)type]->size();
        for (size_t i = 0; i < busyAQueuePairCount; i++)
        {
            const auto pair = busyACoreQueue_[(int)type]->pop();
            ResolveBusyAQueuePair(type, pair);
        }
        busyACoreQueue_[(int)type]->unlock();

        // Handling Free B queue pairings
        freeBCoreQueue_[(int)type]->lock();
        size_t freeBQueuePairCount = freeBCoreQueue_[(int)type]->size();
        for (size_t i = 0; i < freeBQueuePairCount; i++)
        {
            const auto pair = freeBCoreQueue_[(int)type]->pop();
            ResolveFreeBQueuePair(type, pair);
        }
        freeBCoreQueue_[(int)type]->unlock();

        // Handling Busy B queue pairings
        busyBCoreQueue_[(int)type]->lock();
        size_t busyBQueuePairCount = busyBCoreQueue_[(int)type]->size();
        for (size_t i = 0; i < busyBQueuePairCount; i++)
        {
            const auto pair = busyBCoreQueue_[(int)type]->pop();
            ResolveBusyBQueuePair(type, pair);
        }
        busyBCoreQueue_[(int)type]->unlock();

        BatchPushReadyQueue();
    }

    inline void ResolveBusyAQueuePair(CoreType type, const aicorePair_t pair)
    {
        const auto coreId = decodePairCore(pair);

        uint64_t finTaskRegVal = aicoreHal_.GetFinishedTask(coreId);
        uint32_t finTaskId = REG_LOW_TASK_ID(finTaskRegVal);
        uint32_t finTaskState = REG_LOW_TASK_STATE(finTaskRegVal);

        const auto taskAId = decodePairTask(pair);

        // We need to make sure the task reported by the core is the one we last assigned to it
        if (finTaskId == taskAId)
        {
            if (finTaskState == TASK_FIN_STATE)
            {
                processFinishedTask(type, taskAId);
                freeACoreQueue_[(int)type]->push(coreId);
                return;
            }

            if (finTaskState == TASK_ACK_STATE)
            {
                freeBCoreQueue_[(int)type]->push(pair);
                return;
            }
        }
        
        // Nothing has changed, put this pair back to its queue
        busyACoreQueue_[(int)type]->push(pair);
    }

    inline void ResolveFreeBQueuePair(CoreType type, const aicorePair_t pair)
    {
        const auto coreId = decodePairCore(pair);

        uint64_t finTaskRegVal = aicoreHal_.GetFinishedTask(coreId);
        uint32_t finTaskId = REG_LOW_TASK_ID(finTaskRegVal);
        uint32_t finTaskState = REG_LOW_TASK_STATE(finTaskRegVal);

        const auto taskAId = decodePairTask(pair);

        // We need to make sure the task reported by the core is the one we last assigned to it
        if (finTaskId == taskAId)
        {
            if (finTaskState == TASK_FIN_STATE)
            {
                processFinishedTask(type, taskAId);
                freeACoreQueue_[(int)type]->push(coreId);
                return;
            }
        }
        
        // Nothing has changed, put this pair back to its queue
        freeBCoreQueue_[(int)type]->push(pair);
    }

    inline void ResolveBusyBQueuePair(CoreType type, const aicorePair_t pair)
    {
        const auto coreId = decodePairCore(pair);

        uint64_t finTaskRegVal = aicoreHal_.GetFinishedTask(coreId);
        uint32_t finTaskId = REG_LOW_TASK_ID(finTaskRegVal);
        uint32_t finTaskState = REG_LOW_TASK_STATE(finTaskRegVal);

        const auto taskAId = decodePairTask(pair);

        // If the reported task is not A, it means A is finished
        if (finTaskId != taskAId)
        {
            // Process A
            processFinishedTask(type, taskAId);

            // Get task B id
            const auto taskBId = finTaskId;

            // If B has also finished, then process it and return the core to the free core queue
            if (finTaskState == TASK_FIN_STATE)
            {
                processFinishedTask(type, taskBId);
                freeACoreQueue_[(int)type]->push(coreId);
                return;
            }

            // Otherwise, set task B as task A and put it into the freeB queue to allow a new task B to be assigned to the core
            const auto newPair = encodePair(coreId, taskBId);
            freeBCoreQueue_[(int)type]->push(newPair);
            return;
        }
        
        // Nothing has changed, put this pair back into its queue
        busyBCoreQueue_[(int)type]->push(pair);
    }

    inline void PushReadyTask(int coreType, uint64_t taskId) {
        context_->readyIds[coreType][context_->readyCount[coreType]++] = taskId;
    }

    inline void ResolveDynStitched(DynDeviceTask *dyntask, int origfunc, int origop) {
        auto &duppedData = dyntask->GetDynFuncDataCacheList()[origfunc].duppedData;
        auto &stitchList = duppedData->GetOperationStitch(origop);
        auto cceBinary = dyntask->cceBinary;

        for (auto *node = stitchList.Head(); node != nullptr; node = node->Next()) {
            uint32_t listSize = node->Size();
            for (uint32_t i = 0; i < listSize; i++) {
                uint32_t id = node->At(i);
                auto funcId = FuncID(id);
                auto opIndex = TaskID(id);
                auto predCounts = dyntask->dynFuncDataCacheList[funcId].predCount;
                bool needProcess = predCounts[opIndex] == 1 ||
                    __atomic_sub_fetch(&predCounts[opIndex], 1, __ATOMIC_RELAXED) == 0;
                if (!needProcess) {
                    continue;
                }

                auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
                auto coreType = cceBinary[callList[opIndex]].coreType;
                if (unlikely(coreType == static_cast<int>(CoreType::HUB))) {
                    ResolveDepDyn(id);
                } else {
                    PushReadyTask(static_cast<int>(coreType), id);
                }
            }
        }
    }

    inline void ResolveDepDyn(uint64_t taskId) {
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);

        auto cceBinary = dyntask->cceBinary;
        auto func = dyntask->dynFuncDataCacheList[funcId].devFunc;
        auto predCounts =  dyntask->dynFuncDataCacheList[funcId].predCount;
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;

        size_t succIndexSize;
        const int *succIndexList = func->GetOperationDepGraphCopyOutResolveSuccIndexAddr(opIndex, succIndexSize);
        size_t succSize;
        auto succList = func->GetOperationDepGraphSuccAddr(opIndex, succSize);
        for (size_t i = succIndexList[0]; i < succSize; i++) {
            auto succIdx = succList[i];
            if (predCounts[succIdx] == 1 ||
                __atomic_sub_fetch(&predCounts[succIdx], 1, __ATOMIC_RELAXED) == 0) {
                auto id = MakeTaskID(funcId, succIdx);
                auto coreType = cceBinary[callList[succIdx]].coreType;
                if (unlikely(coreType == static_cast<int>(CoreType::HUB))) {
                    ResolveDepDyn(id);
                } else {
                    PushReadyTask(static_cast<int>(coreType), id);
                }
            }
        }

        ResolveDynStitched(dyntask, funcId, opIndex);
    }

    inline void processFinishedTask(CoreType type, uint64_t taskId) {
        ResolveDepDyn(taskId);
        context_->waitTaskCnt_[(int)type]--;
    }

    inline void Init(int threadIdx, DeviceArgs *deviceArgs, int schedIdx) {
        aicNum_ = static_cast<int32_t>(deviceArgs->nrAic);
        aivNum_ = static_cast<int32_t>(deviceArgs->nrAiv);
        aicpuNum_ = deviceArgs->scheCpuNum;
        aicpuIdx_ = threadIdx;
        schedIdx_ = schedIdx;
        aicValidNum_ = deviceArgs->nrValidAic;
        aicoreHal_.Init(deviceArgs);
        validGetPgMask_ = deviceArgs->validGetPgMask;
        isSendStop = false;

        // Getting variable controlling who is the scheduler lead. The first to arrive here should take the lead so that this starts as fast as possible
        auto leadSchedulerId = (std::atomic<uint32_t>*) &deviceArgs->leadSchedulerId;
        
        // Putting myself as leader, if nobody has done it yet
        uint32_t expectedValue = AICPU_LEAD_SCHEDULER_NULL;
        isLeaderScheduler_ = leadSchedulerId->compare_exchange_strong(expectedValue, (uint32_t)aicpuIdx_);

        UpdateAiCoreBlockIndexSection();
        aicoreHal_.MapRegistersForAllCores(aicNum_);
        preFetchSuccess_ = false;
        preFetchNextDevTaskCtrl_ = nullptr;
    }

    inline void HandShakeTryPreFetchDevTask() {
        if (!preFetchSuccess_ && PreFetchNextDevTask()) {
            preFetchNextDevTaskCtrl_->isFirstDevTask = true;
            SendDevTaskModel(preFetchNextDevTaskCtrl_->devTask);
            InitDevTask(preFetchNextDevTaskCtrl_);
        }
    }

    inline void HandShakePostProc() {
        if (preFetchSuccess_) {
            uint64_t sentAic = 0;
            uint64_t sentAiv = 0;
            CountSendTask(sentAic, sentAiv);
            if (sentAic + sentAiv > 0) {
                preFetchNextDevTaskCtrl_->finishedFunctionCnt.fetch_add(sentAic + sentAiv, std::memory_order_relaxed);
            }
        }
    }

    inline void HandShake() {
        int handShakeNum = 0;
        int mngAicoreNum = aicEnd_ - aicStart_ + aivEnd_ - aivStart_;
        bool handFlag[MAX_AICORE_NUM] = {false};
        uint64_t start_cycles = GetCycles();
        bool aicAllSuccess = false;
        bool aivAllSuccess = false;
        int aicSucessCnt = 0;
        int aivSucessCnt = 0;
        
        while (handShakeNum < mngAicoreNum) {
            HandShakeTryPreFetchDevTask();

            bool curIterAllAicSuccess = true;
            bool curIterAllAivSuccess = true;
            for (int i = aicEnd_ - 1; (!aicAllSuccess) && i >= aicStart_; i--) {
                if (handFlag[i]) {
                    continue;
                }
                if (aicoreHal_.TryHandShakeByGm(i, dotStatus_)) {
                    handShakeNum++;
                    aicSucessCnt++;
                    handFlag[i] = true;
                } else {
                    curIterAllAicSuccess = false;
                }
            }
            aicAllSuccess = curIterAllAicSuccess;

            for (int i = aivEnd_ - 1; (!aivAllSuccess) && i >= aivStart_; i--) {
                if (handFlag[i]) {
                    continue;
                }
                if (aicoreHal_.TryHandShakeByGm(i, dotStatus_)) {
                    handShakeNum++;
                    aivSucessCnt++;
                    handFlag[i] = true;
                } else {
                    curIterAllAivSuccess = false;
                }
            }
            aivAllSuccess = curIterAllAivSuccess;

            if (GetCycles() - start_cycles > HAND_SHAKE_TIMEOUT) {
                DEV_ERROR(0, "HandShakeByGmWithPreSendTask timeout notHandshakeNum=%d.", mngAicoreNum - handShakeNum);
                return;
            }
        }

        HandShakePostProc();
    }

    /* assign aic and aiv core index section for this aicpu */
    inline void UpdateAiCoreBlockIndexSection() {
        auto f = [](int total, int idx, int part, int &start, int &end) {
            int perCpu = total / part;
            int remain = total % part;
            start = idx * perCpu + ((idx < remain) ? idx : remain);
            end = start + perCpu + ((idx < remain) ? 1 : 0);
        };

        f(aicValidNum_, schedIdx_, aicpuNum_, aicStart_, aicEnd_);
        f(AIV_NUM_PER_AI_CORE * aicValidNum_, schedIdx_, aicpuNum_, aivStart_, aivEnd_);
        aivStart_ += aicValidNum_;
        aivEnd_ += aicValidNum_;

        aicoreHal_.SetMngCoreBlockId(aicStart_, aicEnd_, aivStart_, aivEnd_);
    }

    inline int GetPhyIdByBlockId(int coreIdx) {
        return aicoreHal_.GetPhyIdByBlockId(coreIdx);
    }

    inline void ForEachManageAicore(std::function<void(int coreIdx)> func) const {
        for (int i = aicStart_; i < aicEnd_; ++i) {
            func(i);
        }
        for (int i = aivStart_; i < aivEnd_; ++i) {
            func(i);
        }
    }

    inline void AbnormalStop() {
        ResetRegAll();
        CheckAndResetReg();
    }

    inline int GetAllAiCoreNum() { return aicNum_ + aivNum_; }
    inline void SetDotStatus(int64_t status) { dotStatus_ = status; }
    inline CoreType AicoreType(int coreIdx) const { return coreIdx < aicEnd_ ? CoreType::AIC : CoreType::AIV; }

private:
    AicoreHAL aicoreHal_;
    int aicNum_{0};
    int aivNum_{0};
    int aicValidNum_{0}; // 有效的aic，根据pgmask计算host传过来
    int aicpuIdx_{0};
    int schedIdx_{0};
    int aicpuNum_{MAX_SCHEDULE_AICPU_NUM};
    int aicStart_{0};
    int aicEnd_{0};
    int aivStart_{0};
    int aivEnd_{0};
    bool validGetPgMask_{true};

    DeviceTask* curDevTask_{nullptr};
    DeviceTaskCtrl* curTaskCtrl_{nullptr};
    int curTaskId_{0};

    /* prepare aicore ready task list */
    ReadyCoreFunctionQueue* readyAicCoreFunctionQue_{nullptr};
    ReadyCoreFunctionQueue* readyAivCoreFunctionQue_{nullptr};
    SchduleContext * context_{nullptr};


    bool preFetchSuccess_{false};
    DeviceTaskCtrl* preFetchNextDevTaskCtrl_{nullptr};
    SPSCQueue<DeviceTaskCtrl *, DEFAULT_QUEUE_SIZE> *prefetchedTaskQueue_{nullptr};
    int64_t dotStatus_{0};
    bool isSendStop{false};

    friend class AiCoreProf;

    // Queues for managing tasks, cores and their pairing
    coreQueue_t* freeACoreQueue_[AICORE_TYPE_NUM];
    pairQueue_t* busyACoreQueue_[AICORE_TYPE_NUM];
    pairQueue_t* freeBCoreQueue_[AICORE_TYPE_NUM];
    pairQueue_t* busyBCoreQueue_[AICORE_TYPE_NUM];

    // Variable to scheduler lead and whether or not I am the lead
    bool isLeaderScheduler_;
};
}
