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

constexpr uint32_t MAX_STATIC_SCHEDULE_AICPU_NUM = 3;   // 真正负责调度aicore的aicpu个数
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

struct TaskInfo {
    int coreIdx;
    uint64_t taskId;
    TaskInfo(int idx, uint64_t id) : coreIdx(idx), taskId(id) {}
};

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
    uint64_t finishedAicFunctionCnt{0}; // 所有aicpu处理完成的aic function个数，多线程增加修改
    uint64_t finishedAivFunctionCnt{0}; // 所有aicpu处理完成的aiv function个数，多线程增加修改
    uint64_t finishedAicpuFunctionCnt{0}; // 所有aicpu处理完成的aicpu function个数，多线程增加修改
    std::atomic<uint64_t> issuedTaskCount{0};
    std::atomic<int> refcnt{-1};
    void (*finishFunc)(void *devTask){nullptr};
    int retCode{0};
    std::array<std::array<std::atomic<bool>, MAX_STATIC_SCHEDULE_AICPU_NUM>, AICORE_TYPE_NUM>  isAicpuIdle;

    bool IsFree() { return refcnt == -1; }

    void PutTask(int ret) {
        if (ret != 0)
            retCode = ret;

        auto cnt = refcnt--;
        while (refcnt.load(std::memory_order_relaxed) != 0)
            ;

        if (cnt == 1) {
            if (finishFunc) {
                finishFunc(devTask);
            }
            refcnt = -1;
        }
    }
};

void SdmaPrefetch(DeviceTask *devTask);

typedef pypto::utils::ConcurrentQueue<aicoreTask_t, aicoreNullTask> taskQueue_t;
typedef pypto::utils::ConcurrentQueue<aicoreCore_t, aicoreNullCore> coreQueue_t;
typedef pypto::utils::ConcurrentQueue<aicorePair_t, aicoreNullPair> pairQueue_t;

class AiCoreManager {
public:
    AiCoreManager(AicpuTaskManager &aicpuTaskManager) : aicpuTaskManager_(aicpuTaskManager){};
    ~AiCoreManager(){};

    inline void InitTaskData() {
        readyAicCoreFunctionQue_ = reinterpret_cast<StaticReadyCoreFunctionQueue *>(curDevTask_->readyAicCoreFunctionQue);
        readyAivCoreFunctionQue_ = reinterpret_cast<StaticReadyCoreFunctionQueue *>(curDevTask_->readyAivCoreFunctionQue);

        // If I am the lead AICPU scheduler, perform initailization steps
        if (isLeaderScheduler_ == true)
        {
            // Initiaizing core function data prior to execution
            ForAllAicores([this](int coreIdx) {
                volatile int64_t *funcData = &args_[coreIdx]->shakeBuffer[SHAK_BUF_COREFUNC_DATA_INDEX];
                *funcData = reinterpret_cast<int64_t>(&curDevTask_->coreFuncData);
            });

            // Allocating queues
            auto availableTaskQueue = new coreQueue_t(MAX_QUEUED_TASKS);
            curDevTask_->availableTaskQueue = (uint64_t) availableTaskQueue;

            // Adding initial set of tasks and cores
            for (size_t i = 0; i < readyAivCoreFunctionQue_->wasSize(); i++) availableTaskQueue->push((uint32_t)readyAivCoreFunctionQue_->getBuffer()[i]);
            for (size_t i = 0; i < readyAicCoreFunctionQue_->wasSize(); i++) availableTaskQueue->push((uint32_t)readyAicCoreFunctionQue_->getBuffer()[i]);

            // Setting task as initialized, allowing others to continue
            curDevTask_->isTaskInitialized = true;
        }
        else // If I am not a lead AICPU scheduler, wait until initialization is ready
        {
            while(curDevTask_->isTaskInitialized == false){ /* Busy wait */ };
        }

        availableTaskQueue_ = reinterpret_cast<taskQueue_t*>(curDevTask_->availableTaskQueue);
        runningPairQueue_                          = new pairQueue_t(AIV_CORE_COUNT + AIC_CORE_COUNT);
        availableCoreQueue_[(int)MachineType::AIV] = new coreQueue_t(AIV_CORE_COUNT);
        availableCoreQueue_[(int)MachineType::AIC] = new coreQueue_t(AIC_CORE_COUNT);
        for (int i = aivStart_; i < aivEnd_; i++) availableCoreQueue_[(int)MachineType::AIV]->push((uint32_t)i);
        for (int i = aicStart_; i < aicEnd_; i++) availableCoreQueue_[(int)MachineType::AIC]->push((uint32_t)i);
    }
    

    int RunTask(DeviceTaskCtrl *taskCtrl);

    int Run(int threadIdx, DeviceArgs *deviceArgs, DeviceTaskCtrl *taskCtrl = nullptr);
    void PushTask(DeviceTaskCtrl *taskCtrl) { taskQueue_.Enqueue(taskCtrl); }

private:

    static inline aicorePair_t encodePair(const uint64_t taskId, const uint64_t coreId) { return aicorePair_t ((taskId << 32) + (coreId & 0x00000000FFFFFFFFUL)); }
    static inline uint64_t decodePairTask(const aicorePair_t pair) { return pair >> 32; }
    static inline uint64_t decodePairCore(const aicorePair_t pair) { return pair & 0x00000000FFFFFFFFUL; }

    int WaitAllAicoreFinish(int coreIdxStart, int coreIdxEnd);
    uint64_t TryBatchSendTask();
    bool checkCoreFinished(const int coreIdx);
    void ResolveDepForAllAiCore();
    void ResolveVirtualPure(uint64_t dep, CoreFunctionReadyState* readyState);
    void ResolveVirtualMix(uint64_t dep, CoreFunctionReadyState* readyState);
    void ResolveByCoreType(int coretype, uint64_t depTaskId, CoreFunctionReadyState *readyState);
    void ResolveDep(uint64_t finishId);

    inline uint64_t GetFinishedTask(const int coreIdx) { return *(finishRegQueues_[GetPhyIdByBlockId(coreIdx)]); }

    inline int GetPhyIdByBlockId(const int coreIdx) { return blockIdToPhyCoreId_[coreIdx]; }

    inline void ForAllAicores(std::function<void(int coreIdx)> func) const {
        for (size_t i = 0; i < MAX_AIV_TOTAL_NUM; ++i) {
            func(i);
        }
    }

    inline uint32_t ReadReg32(const int coreIdx, const int offset) {
        const auto idx = GetPhyIdByBlockId(coreIdx);
        return *(reinterpret_cast<volatile uint32_t*>(regAddrs_[idx] + offset));
    }

    inline void WriteReg32(const int coreIdx, const int offset, const uint32_t val) {
        const auto idx = GetPhyIdByBlockId(coreIdx);
        *(reinterpret_cast<volatile uint32_t*>(regAddrs_[idx] + offset)) = val;
    }

    inline void SetReadyQueue(const int coreIdx, const uint64_t taskIdx) {
        const auto idx = GetPhyIdByBlockId(coreIdx);
        *(readyRegQueues_[idx]) = taskIdx + 1; // Plus one is a required offset
    }

    inline void WriteReg32ALl(int offset, uint32_t val) {
        for (size_t i = 0; i < MAX_AICORE_NUM; i++)
         *(reinterpret_cast<volatile uint32_t *>(regAddrs_[i] + offset)) = val;
    }

    void AbnormalStop();

    void NormalStop();

    inline void SetDotStatus(int64_t status) { dotStatus_ = status; }
    inline CoreType AicoreType(int coreIdx) const { return coreIdx < aicEnd_ ? CoreType::AIC : CoreType::AIV; }

private:
    int aicNum_{0};
    int aivNum_{0};
    int aicValidNum_{0}; // 有效的aic，根据pgmask计算host传过来
    int aicpuIdx_{0};
    int aicStart_{0};
    int aicpuNum_{MAX_STATIC_SCHEDULE_AICPU_NUM};
    int aicEnd_{0};
    int aivStart_{0};
    int aivEnd_{0};
    int64_t *regAddrs_{nullptr};
    int64_t sharedBuffer_{0};
    DeviceTask *curDevTask_{nullptr};
    DeviceTaskCtrl* curTaskCtrl_{nullptr};
    AicpuTaskManager &aicpuTaskManager_;

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
