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
#include "machine/utils/dynamic/spsc_queue.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/utils/device_log.h"
#include "aicpu_task_manager.h"
#include "interface/operation/opcode.h"
#include "securec.h"
#include "dynamic/device_utils.h"
#include "aicore_dump.h"
#include "interface/utils/common.h"
#include "machine/device/dynamic/device_utils.h"

namespace npu::tile_fwk {
const uint32_t REG_SPR_FAST_PATH_ENABLE = 0x18;
const uint64_t REG_SPR_FAST_PATH_OPEN = 0xE;
const uint64_t REG_SPR_FAST_PATH_CLOSE = 0xF;
const uint32_t REG_SPR_DATA_MAIN_BASE = 0xA0;
const uint32_t REG_SPR_COND = 0x4C8;

const uint32_t AICORE_STATUS_INIT = 0xFFFFFFFFU;
const uint32_t CORE_NUM_PER_AI_CORE = 3;
const uint32_t AIV_NUM_PER_AI_CORE = 2;
const uint32_t AICORE_TYPE_NUM = 2;

constexpr uint32_t AIV_CORE_COUNT = 48;
constexpr uint32_t AIC_CORE_COUNT = 24;
constexpr uint32_t TOTAL_CORE_COUNT = AIV_CORE_COUNT + AIC_CORE_COUNT;

constexpr uint32_t MAX_STATIC_SCHEDULE_AICPU_NUM = 3;   // 真正负责调度aicore的aicpu个数
constexpr uint32_t AIV_PER_AICPU = AIV_CORE_COUNT / MAX_STATIC_SCHEDULE_AICPU_NUM;
constexpr uint32_t AIC_PER_AICPU = AIC_CORE_COUNT / MAX_STATIC_SCHEDULE_AICPU_NUM;
constexpr int32_t START_STATIC_AICPU_NUM = MAX_STATIC_SCHEDULE_AICPU_NUM;
constexpr uint32_t MAX_AICORE_NUM = 108;
constexpr uint32_t MAX_AIV_TOTAL_NUM = 72;
constexpr uint32_t MAX_AIC_TOTAL_NUM = MAX_AICORE_NUM - MAX_AIV_TOTAL_NUM;
constexpr uint32_t MAX_MANAGER_AIV_NUM = MAX_AIV_TOTAL_NUM;

constexpr uint32_t REG_31_BITS = 0x7FFFFFFF;
constexpr uint32_t REG_32_BITS = 0xFFFFFFFF;
#define REG_LOW_TASK_ID(regVal) (regVal) & REG_31_BITS                     // 低31位存储的taskid
#define REG_LOW_TASK_STATE(regVal) ((regVal)&REG_32_BITS) >> 31            // 低32位存储的task的状态
#define REG_HIGH_TASK_ID(regVal) ((regVal) >> 32) & REG_31_BITS            // 高32位存储的taskid
#define REG_HIGH_TASK_STATE(regVal) (((regVal) >> 32) & REG_32_BITS) >> 31 // 高32位存储的task状态
constexpr uint32_t TASK_FIN_STATE = 1;                                     // 任务执行完成完成
constexpr uint32_t AICORE_COREID_BIT_OFFSET = 32;
constexpr int32_t AICORE_COREID_MASK = 0x0FFF;
constexpr uint32_t DEFAULT_TASK_QUEUE_SIZE = 64;

#define MAX_QUEUED_TASKS 1024

typedef uint32_t aicoreTask_t;
typedef uint32_t aicoreCore_t;
typedef uint64_t aicorePair_t;
 
constexpr aicoreTask_t aicoreNullTask = 0xFFFFFFFFUL;
constexpr aicoreCore_t aicoreNullCore = 0xFFFFFFFFUL;
constexpr aicorePair_t aicoreNullPair = 0xFFFFFFFFFFFFFFFFUL;

struct sdma_l2_cmo_desc {
    unsigned long   src_addr;
    size_t          size;
    char            cmo_opcode;
};

#define SDMA_FILE "/dev/sdma"
#define IOCTL_SDMA_L2_CMO  _IOW('s', 3, struct sdma_l2_cmo_desc)

struct DeviceTaskCtrl {
    int taskType{DEVICE_TASK_TYPE_INVALID};
    uint64_t taskId{0};
    void *devTask{nullptr};
    std::atomic<uint64_t> issuedTaskCount{0};
    std::atomic<int> refcnt{-1};
    void (*finishFunc)(void *devTask){nullptr};
    int retCode{0};

    bool IsFree() { return refcnt == -1; }
    void PutTask(int ret)
    {
        if (ret != 0) retCode = ret;

        auto cnt = refcnt--;
        while (refcnt.load(std::memory_order_relaxed) != 0) ;
        if (cnt == 1)
        {
          if (finishFunc != nullptr) finishFunc(devTask);
          refcnt = -1;
        }
    }
};

inline void SdmaPrefetch(DeviceTask *devTask)
{
    if (devTask == nullptr || devTask->l2Info.prefetchNum == 0) {
      return;
    }
    if (devTask->l2Info.prefetchNum > MAX_PREFETCH_NUM) {
      DEV_ERROR("Prefetch invalid num %ld.\n", devTask->l2Info.prefetchNum);
      return;
    }
    int fd = open(SDMA_FILE, O_RDWR);
    if (fd == -1) {
      return;
    }
    struct sdma_l2_cmo_desc desc;
    desc.cmo_opcode = 0x6;
    int ret = 0;
    for (int64_t i = 0; i < devTask->l2Info.prefetchNum; ++i) {
      desc.src_addr = devTask->l2Info.prefetchAddrs[i];
      desc.size = devTask->l2Info.prefetchSizes[i];
      ret |= ioctl(fd, IOCTL_SDMA_L2_CMO, &desc);
      DEV_DEBUG("Prefetch %lx %lu ret:%d\n", devTask->l2Info.prefetchAddrs[i],
          devTask->l2Info.prefetchSizes[i], ret);
    }
    DEV_INFO("Prefetch tensor num %ld ret %d.\n", devTask->l2Info.prefetchNum, ret);
    close(fd);
    return;
}

typedef pypto::utils::ConcurrentQueue<aicoreTask_t, aicoreNullTask> taskQueue_t;
typedef pypto::utils::ConcurrentQueue<aicoreCore_t, aicoreNullCore> coreQueue_t;
typedef pypto::utils::ConcurrentQueue<aicorePair_t, aicoreNullPair> pairQueue_t;

class AiCoreManager {
public:
    AiCoreManager() = default;
    ~AiCoreManager() = default;

    inline void InitTaskData() {
        readyAicCoreFunctionQue_ = reinterpret_cast<StaticReadyCoreFunctionQueue *>(curDevTask_->readyAicCoreFunctionQue);
        readyAivCoreFunctionQue_ = reinterpret_cast<StaticReadyCoreFunctionQueue *>(curDevTask_->readyAivCoreFunctionQue);

        // If I am the lead AICPU scheduler, perform initailization steps
        if (isLeaderScheduler_ == true)
        {
            // Resetting issued task count -- needed to verify termination
            curTaskCtrl_->issuedTaskCount = 0;

            // Setting the corresponding task kernel to execute for all cores
            for (uint32_t coreIdx = 0; coreIdx < TOTAL_CORE_COUNT; coreIdx++)
                args_[coreIdx]->shakeBuffer[SHAK_BUF_COREFUNC_DATA_INDEX] = (int64_t)&curDevTask_->coreFuncData;

            // Allocating queues
            auto availableTaskQueue = new coreQueue_t(MAX_QUEUED_TASKS);
            curDevTask_->availableTaskQueue = (uint64_t) availableTaskQueue;

            // Adding initial set of tasks 
            for (size_t i = 0; i < readyAivCoreFunctionQue_->wasSize(); i++) availableTaskQueue->push((uint32_t)readyAivCoreFunctionQue_->getBuffer()[i]);
            for (size_t i = 0; i < readyAicCoreFunctionQue_->wasSize(); i++) availableTaskQueue->push((uint32_t)readyAicCoreFunctionQue_->getBuffer()[i]);

            // Allocating core queues
            auto availableVectorCoreQueue = new coreQueue_t(AIV_CORE_COUNT);
            auto availableCubeCoreQueue = new coreQueue_t(AIC_CORE_COUNT);
            curDevTask_->availableVectorCoreQueue = (uint64_t) availableVectorCoreQueue;
            curDevTask_->availableCubeCoreQueue = (uint64_t) availableCubeCoreQueue;
            for (uint32_t i = AIC_CORE_COUNT; i < AIC_CORE_COUNT + AIV_CORE_COUNT; i++) availableVectorCoreQueue->push(i);
            for (uint32_t i = 0; i < AIC_CORE_COUNT; i++) availableCubeCoreQueue->push(i);

            // Allocating pairing queue
            curDevTask_->runningPairQueue = (uint64_t) new pairQueue_t(TOTAL_CORE_COUNT);

            // Setting task as initialized, allowing others to continue
            curDevTask_->isTaskInitialized = true;
        }
        else // If I am not a lead AICPU scheduler, wait until initialization is ready
        {
            while(curDevTask_->isTaskInitialized == false){ /* Busy wait */ };
        }

        availableTaskQueue_ = reinterpret_cast<taskQueue_t*>(curDevTask_->availableTaskQueue);
        runningPairQueue_                          = (pairQueue_t*)curDevTask_->runningPairQueue;
        availableCoreQueue_[(int)MachineType::AIV] = (coreQueue_t*)curDevTask_->availableVectorCoreQueue;
        availableCoreQueue_[(int)MachineType::AIC] = (coreQueue_t*)curDevTask_->availableCubeCoreQueue;
    }

    inline int RunTask(DeviceTaskCtrl *taskCtrl)
    {
        int ret = npu::tile_fwk::dynamic::DEVICE_MACHINE_OK;
        curTaskCtrl_ = taskCtrl;
        InitTaskData();

        const auto t0 = std::chrono::high_resolution_clock::now();

        npu::tile_fwk::dynamic::TimeCheck tm;
        uint32_t globalTasksIssued = 0;
        while (globalTasksIssued < curDevTask_->coreFunctionCnt)
        {
            uint32_t tasksIssued = 0;
            while (tryIssuePendingTasks() > 0) tasksIssued++;
            checkTaskTerminations();
            while (tryIssuePendingTasks() > 0) tasksIssued++;
            globalTasksIssued = curTaskCtrl_->issuedTaskCount.fetch_add(tasksIssued, std::memory_order_relaxed);

            if (npu::tile_fwk::dynamic::CheckTimeOut("wait task send finish.", tm) != 0) return -1;
        }

        WaitAllAicoreFinish();// Waiting in parallel for all compute cores to finish

        const auto tf = std::chrono::high_resolution_clock::now();
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tf - t0).count();
        DEV_ERROR("[AICPU %d] Running Time: %ldns", aicpuIdx_, ns);

        if (isLeaderScheduler_ == true)
        {
            delete availableTaskQueue_;
            delete availableCoreQueue_[(int)MachineType::AIV];
            delete availableCoreQueue_[(int)MachineType::AIC];
            delete runningPairQueue_;
        } 

        return ret;
    }

    inline int Run(int threadIdx, DeviceArgs *deviceArgs, DeviceTaskCtrl *taskCtrl = nullptr)
    {
        aicpuIdx_ = threadIdx;
        curDevTask_ = static_cast<DeviceTask *>(taskCtrl->devTask);

        // Getting variable controlling who is the scheduler lead. The first to arrive here should take the lead so that this starts as fast as possible
        leadSchedulerId_ = (std::atomic<uint32_t>*) &curDevTask_->leadSchedulerId;
        
        // Putting myself as leader, if nobody has done it yet
        uint32_t expectedValue = AICPU_LEAD_SCHEDULER_NULL;
        isLeaderScheduler_ = leadSchedulerId_->compare_exchange_strong(expectedValue, (uint32_t)aicpuIdx_);

        aicNum_ = deviceArgs->nrAic;
        aivNum_ = deviceArgs->nrAiv;
        aicpuNum_ = START_STATIC_AICPU_NUM;
        aicValidNum_ = deviceArgs->nrValidAic;
        regAddrs_ = reinterpret_cast<int64_t *>(deviceArgs->coreRegAddr);
        sharedBuffer_ = deviceArgs->sharedBuffer;

        for (uint32_t idx = 0; idx < MAX_AICORE_NUM; idx++)
        {
            auto baseAddress = (uint64_t) regAddrs_[idx];
            readyRegQueues_[idx]  = (volatile uint64_t *)(baseAddress + (uint64_t)regSprDataMainBase_);
            finishRegQueues_[idx] = (volatile uint64_t *)(baseAddress + (uint64_t)regSprCond_);
        }

        // If I am the lead AICPU scheduler, perform initailization steps
        if (isLeaderScheduler_ == true)
        {
            // Handshake with all AI cores
            for (uint32_t coreIdx = 0; coreIdx < TOTAL_CORE_COUNT; coreIdx++)
            {
                auto args = reinterpret_cast<KernelArgs *>((static_cast<uint64_t>(sharedBuffer_)) + SHARED_BUFFER_SIZE * coreIdx);
                args->taskEntry.reserved[0] = dotStatus_;
                volatile int64_t *handshakeBuffer = args->shakeBuffer;
                npu::tile_fwk::dynamic::TimeCheck tm;
                while ((*handshakeBuffer & 0xFFFFFFFF) != AICORE_SAY_HELLO)
                {
                    if (npu::tile_fwk::dynamic::CheckTimeOut("hand shake", tm) != 0) {
                        DEV_ERROR("hand shake %u timeout.\n", coreIdx);
                        AbnormalStop();
                    }
                }
            }

            // Setting task as initialized, allowing others to continue
            curDevTask_->isDeviceInitialized = true;
        }
        else // If I am not a lead AICPU scheduler, wait until initialization is ready
        {
            while(curDevTask_->isDeviceInitialized == false){ /* Busy wait */ };
        }

        for (uint32_t coreIdx = 0; coreIdx < TOTAL_CORE_COUNT; coreIdx++)
        {
            auto args = reinterpret_cast<KernelArgs *>((static_cast<uint64_t>(sharedBuffer_)) + SHARED_BUFFER_SIZE * coreIdx);
            int64_t *handshakeBuffer = args->shakeBuffer;
            args_[coreIdx] = args;
            blockIdToPhyCoreId_[coreIdx] = (*handshakeBuffer >> AICORE_COREID_BIT_OFFSET) & AICORE_COREID_MASK;
            WriteReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_OPEN);  // Enabling fast path
        }

        /* write to MAINBASE reg need reg 0x18 open first */
        __sync_synchronize();

        int ret = npu::tile_fwk::dynamic::DEVICE_MACHINE_OK;
        if (taskCtrl != nullptr) {
            ret = RunTask(taskCtrl);
        } else {
            while (true) {
                taskCtrl = taskQueue_.Dequeue();
                if (taskCtrl == nullptr)
                    break;
                ret = RunTask(taskCtrl);
                taskCtrl->PutTask(ret);
            }
        }

        if (isLeaderScheduler_ == true) NormalStop();
        return ret;
    }

    void PushTask(DeviceTaskCtrl *taskCtrl) { taskQueue_.Enqueue(taskCtrl); }

private:

    static inline aicorePair_t encodePair(const uint64_t taskId, const uint64_t coreId) { return aicorePair_t ((taskId << 32) + (coreId & 0x00000000FFFFFFFFUL)); }
    static inline uint64_t decodePairTask(const aicorePair_t pair) { return pair >> 32; }
    static inline uint64_t decodePairCore(const aicorePair_t pair) { return pair & 0x00000000FFFFFFFFUL; }

    inline int WaitAllAicoreFinish()
    {
        npu::tile_fwk::dynamic::TimeCheck tm;
        for (uint32_t i = aicpuIdx_; i < TOTAL_CORE_COUNT; i += MAX_STATIC_SCHEDULE_AICPU_NUM) {
            while (checkCoreFinished(i) == false)
            {
                if (npu::tile_fwk::dynamic::CheckTimeOut("wait tail task", tm) != 0) {
                    DEV_ERROR("wait tail task finish timeout coreindx=%u.\n", i);
                    return -1;
                }
            }
        }
        return 0;
    }

    inline uint64_t tryIssuePendingTasks()
    {
        auto taskIdx = (uint64_t)availableTaskQueue_->pop();
        if (taskIdx == aicoreNullTask) return 0;

        // Getting task's type
        const auto readyState = reinterpret_cast<CoreFunctionReadyState *>(curDevTask_->coreFunctionReadyStateAddr);
        const auto coreType = readyState[taskIdx].coreType;
        const auto coreIdx = (uint64_t)availableCoreQueue_[coreType]->pop();
        if (coreIdx == aicoreNullCore)
        {
            availableTaskQueue_->push(taskIdx);
            return 0;
        }

        // DEV_ERROR("AICPU %d - Running Task: %lu", aicpuIdx_, *taskSetAddress);
        SetReadyQueue(coreIdx, taskIdx);
        runningPairQueue_->push(encodePair(taskIdx, coreIdx));
        return 1;
    }

    inline bool checkCoreFinished(const int coreIdx)
    {
        const uint64_t finTaskVal = GetFinishedTask(coreIdx);
        const uint32_t regLFinTaskState = REG_LOW_TASK_STATE(finTaskVal);
        if (regLFinTaskState == TASK_FIN_STATE) return true;
        return false;
    }

    inline void checkTaskTerminations()
    {
        const auto runningPair = runningPairQueue_->pop();
        if (runningPair == aicoreNullPair) return;

        const int coreIdx = (int)decodePairCore(runningPair);

        if (checkCoreFinished(coreIdx)) 
        {
            const uint64_t taskId = (int)decodePairTask(runningPair);
            ResolveDep(taskId);
            const auto readyState = reinterpret_cast<CoreFunctionReadyState *>(curDevTask_->coreFunctionReadyStateAddr);
            const auto coreType = readyState[taskId].coreType; 
            availableCoreQueue_[coreType]->push(coreIdx);
        }
        else
        {
            runningPairQueue_->push(runningPair);
        }
    }


    inline void ResolveVirtualPure(uint64_t dep) {
       const auto virtualFuncInfo = &(reinterpret_cast<CoreFunctionWsAddr *>(curDevTask_->coreFuncData.coreFunctionWsAddr)[dep]);
       const auto topo = reinterpret_cast<CoreFunctionTopo *>(virtualFuncInfo->topoAddr);
        for (uint64_t i = 0 ; i < topo->depNum; i++) {
            const uint64_t depId = topo->depIds[i];
            availableTaskQueue_->push((uint32_t)depId);
        }
    }

    inline void ResolveVirtualMix(uint64_t dep, CoreFunctionReadyState* readyState) {
        const auto virtualFuncInfo = &(reinterpret_cast<CoreFunctionWsAddr *>(curDevTask_->coreFuncData.coreFunctionWsAddr)[dep]);
        const auto topo = reinterpret_cast<CoreFunctionTopo *>(virtualFuncInfo->topoAddr);
        for (uint64_t i = 0 ; i < topo->depNum; i++) {
            const uint64_t depId = topo->depIds[i];
            if (readyState[depId].readyCount == topo->readyCount) {
                availableTaskQueue_->push((uint32_t)depId);
            } else {
                if (__sync_add_and_fetch(&(readyState[depId].readyCount), (-1) * topo->readyCount) == 0) {
                    availableTaskQueue_->push((uint32_t)depId);
                }
            }
        }
    }

    inline void ResolveByCoreType(int coretype, uint64_t depTaskId, CoreFunctionReadyState *readyState) {
        // Compiler optimizations reduce switch-case to O(1), rendering if-else unnecessary in such cases.
        switch (coretype) {
            case static_cast<int>(MachineType::AIV):
            case static_cast<int>(MachineType::AIC): {
                availableTaskQueue_->push((uint32_t)depTaskId);
                break;
            }
            case static_cast<int>(MachineType::HUB): {
                ResolveDep(depTaskId);
                break;
            }
            case static_cast<int>(MachineType::VIRTUAL_PURE): {
                ResolveVirtualPure(depTaskId);
                break;
            }
            case static_cast<int>(MachineType::VIRTUAL_MIX): {
                ResolveVirtualMix(depTaskId, readyState);
                break;
            }
            default: {
                DEV_ERROR("Unsupported task type");
                break;
            }
        }
    }

    inline void ResolveDep(uint64_t finishId) {
        auto readyState = reinterpret_cast<CoreFunctionReadyState *>(curDevTask_->coreFunctionReadyStateAddr);
        const auto funcInfo = &(reinterpret_cast<CoreFunctionWsAddr *>(curDevTask_->coreFuncData.coreFunctionWsAddr)[finishId]);
        const auto topo = reinterpret_cast<CoreFunctionTopo *>(funcInfo->topoAddr);
        for (uint64_t i = 0; i < topo->depNum; i++) {
            const uint64_t dep = topo->depIds[i];
            const int ret = __sync_add_and_fetch(&(readyState[dep].readyCount), 1);
            if (ret != 0) continue;
            ResolveByCoreType(readyState[dep].coreType, dep, readyState);
        }
    }

    inline uint64_t GetFinishedTask(const int coreIdx)
    {
        const auto physicalCoreIdx = blockIdToPhyCoreId_[coreIdx];
         return *(finishRegQueues_[physicalCoreIdx]);
    }

    inline void WriteReg32(const int coreIdx, const int offset, const uint32_t val) {
        const auto physicalCoreIdx = blockIdToPhyCoreId_[coreIdx];
        *(reinterpret_cast<volatile uint32_t*>(regAddrs_[physicalCoreIdx] + offset)) = val;
    }

    inline void SetReadyQueue(const int coreIdx, const uint64_t taskIdx) {
        const auto physicalCoreIdx = blockIdToPhyCoreId_[coreIdx];
        *(readyRegQueues_[physicalCoreIdx]) = taskIdx + 1; // Plus one is a required offset
    }

    inline void AbnormalStop() {
        for (size_t coreIdx = 0; coreIdx < TOTAL_CORE_COUNT; coreIdx++)
        {
            WriteReg32(coreIdx, regSprDataMainBase_, AICORE_TASK_STOP + 1);
            if (isNeedWriteRegForFastPath_) WriteReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE); // write to MAINBASE reg must be done before close 0x18
        }
    }

    inline void NormalStop() {
        for (size_t coreIdx = 0; coreIdx < TOTAL_CORE_COUNT; coreIdx++) SetReadyQueue(coreIdx, AICORE_TASK_STOP);

        __sync_synchronize(); // write to MAINBASE reg must be done before close 0x18 */

        for (size_t coreIdx = 0; coreIdx < TOTAL_CORE_COUNT; coreIdx++)
        {
            if (isNeedWriteRegForFastPath_) WriteReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
            volatile KernelArgs *arg = reinterpret_cast<KernelArgs *>(sharedBuffer_ + coreIdx * SHARED_BUFFER_SIZE);
            arg->shakeBuffer[0] = 0;
            arg->shakeBuffer[SHAK_BUF_COREFUNC_DATA_INDEX] = 0;
        };
    }

    inline void SetDotStatus(int64_t status) { dotStatus_ = status; }

private:
    int aicNum_{0};
    int aivNum_{0};
    int aicValidNum_{0}; // 有效的aic，根据pgmask计算host传过来
    int aicpuIdx_{0};
    int aicpuNum_{MAX_STATIC_SCHEDULE_AICPU_NUM};
    int64_t *regAddrs_{nullptr};
    int64_t sharedBuffer_{0};
    DeviceTask *curDevTask_{nullptr};
    DeviceTaskCtrl* curTaskCtrl_{nullptr};

    std::array<int, MAX_AICORE_NUM> blockIdToPhyCoreId_;
    std::array<volatile uint64_t *, MAX_AICORE_NUM> readyRegQueues_;
    std::array<volatile uint64_t *, MAX_AICORE_NUM> finishRegQueues_;
    std::array<KernelArgs *, MAX_AICORE_NUM> args_;

    SPSCQueue<DeviceTaskCtrl *, DEFAULT_TASK_QUEUE_SIZE> taskQueue_;

    // Variable to scheduler lead and whether or not I am the lead
    std::atomic<uint32_t>* leadSchedulerId_;
    bool isLeaderScheduler_;

    /* prepare aicore ready task list */
    StaticReadyCoreFunctionQueue *readyAicCoreFunctionQue_{nullptr};
    StaticReadyCoreFunctionQueue *readyAivCoreFunctionQue_{nullptr};

    // Queues for managing tasks, cores and their pairing
    pypto::utils::ConcurrentQueue<aicoreTask_t, aicoreNullTask>* availableTaskQueue_;
    pypto::utils::ConcurrentQueue<aicoreCore_t, aicoreNullCore>* availableCoreQueue_[AICORE_TYPE_NUM];
    pypto::utils::ConcurrentQueue<aicorePair_t, aicoreNullPair>* runningPairQueue_; 

    int64_t dotStatus_{0};
    bool isNeedWriteRegForFastPath_{true};
    uint32_t regSprDataMainBase_{REG_SPR_DATA_MAIN_BASE};
    uint32_t regSprCond_{REG_SPR_COND};
};
} // namespace npu::tile_fwk
