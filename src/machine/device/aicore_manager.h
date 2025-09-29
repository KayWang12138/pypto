/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
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
#include "machine/kernel/aicore.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/utils/device_log.h"
#include "aicpu_task_manager.h"
#include "interface/operation/opcode.h"
#include "securec.h"
#include "aicore_prof.h"
#include "dynamic/device_utils.h"
#include "aicore_dump.h"
#include "interface/utils/common.h"

namespace npu::tile_fwk {
const uint32_t REG_SPR_FAST_PATH_ENABLE = 0x18;
const uint64_t REG_SPR_FAST_PATH_OPEN = 0xE;
const uint64_t REG_SPR_FAST_PATH_CLOSE = 0xF;

const uint32_t REG_SPR_DATA_MAIN_BASE = 0xA0; // 0xA0 -> DATA_MAIN_BASE
const uint32_t REG_SPR_COND = 0x4C8;          // 0x4C8 -> COND SPR
const uint32_t REG_SPR_MAGIC = 0x78;
const int INVALID_CORE_IDX = 0xFF;

const uint32_t AICORE_STATUS_INIT = 0xFFFFFFFFU;
const uint32_t CORE_NUM_PER_AI_CORE = 3;
const uint32_t AIV_NUM_PER_AI_CORE = 2;
const uint32_t READY_ID_FIX_CACHE_NUM = 800;
const uint32_t AICORE_TYPE_NUM = 2;

constexpr uint32_t MAX_AICORE_NUM = 75;
constexpr uint32_t NAX_AIV_TOTAL_NUM = 50;
constexpr uint32_t MAX_MANAGER_AIV_NUM = (NAX_AIV_TOTAL_NUM / npu::tile_fwk::dynamic::MAX_SCHEDULE_AICPU_NUM) + 1;

constexpr uint32_t REG_31_BITS = 0x7FFFFFFF;
constexpr uint32_t REG_32_BITS = 0xFFFFFFFF;
#define REG_LOW_TASK_ID(regVal) (regVal) & REG_31_BITS                     // 低31位存储的taskid
#define REG_LOW_TASK_STATE(regVal) ((regVal)&REG_32_BITS) >> 31            // 低32位存储的task的状态
#define REG_HIGH_TASK_ID(regVal) ((regVal) >> 32) & REG_31_BITS            // 高32位存储的taskid
#define REG_HIGH_TASK_STATE(regVal) (((regVal) >> 32) & REG_32_BITS) >> 31 // 高32位存储的task状态
constexpr uint32_t TASK_FIN_STATE = 1;                                     // 任务执行完成完成
constexpr uint32_t TASK_ACK_STATE = 0;                                     // 收到任务状态，没执行完成
constexpr uint32_t REG_TASK_NUM = 2;                                       // 一次寄存器task个数

constexpr uint32_t NUM_ONE = 1;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t NUM_THREE = 3;
constexpr uint32_t NUM_THIRTY_TWO = 32;

constexpr uint32_t DEFAULT_QUEUE_SIZE = 64;

constexpr int32_t AICORE_COREID_MASK = 0x0FFF;
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
    std::atomic<uint64_t> finishedFunctionCnt{0};
    std::atomic<int> refcnt{-1};
    void (*finishFunc)(void *devTask){nullptr};
    int retCode{0};
    std::array<std::array<std::atomic<bool>, npu::tile_fwk::dynamic::MAX_SCHEDULE_AICPU_NUM>, AICORE_TYPE_NUM>  isAicpuIdle;

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

inline void ReadyQueueLock(StaticReadyCoreFunctionQueue *rq) {
    while (!__sync_bool_compare_and_swap(&rq->lock, 0, 1)) {
    }
}

inline void ReadyQueueUnLock(StaticReadyCoreFunctionQueue *rq) {
    while (!__sync_bool_compare_and_swap(&rq->lock, 1, 0)) {
    }
}

inline void SdmaPrefetch(DeviceTask *devTask) {
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

class AiCoreManager {
public:
    AiCoreManager(AicpuTaskManager &aicpuTaskManager) : aicpuTaskManager_(aicpuTaskManager), prof_(*this){};
    ~AiCoreManager(){};

    inline void InitTaskData(DeviceTaskCtrl *taskCtrl) {
        curTaskCtrl_ = taskCtrl;
        curDevTask_ = static_cast<DeviceTask *>(taskCtrl->devTask);
        ForEachManageAicore([this](int coreIdx) {
            volatile int64_t *funcData = &args_[coreIdx]->shakeBuffer[SHAK_BUF_COREFUNC_DATA_INDEX];
            *funcData = reinterpret_cast<int64_t>(&curDevTask_->coreFuncData);
        });
        readyAicCoreFunctionQue_ = reinterpret_cast<StaticReadyCoreFunctionQueue *>(curDevTask_->readyAicCoreFunctionQue);
        readyAivCoreFunctionQue_ = reinterpret_cast<StaticReadyCoreFunctionQueue *>(curDevTask_->readyAivCoreFunctionQue);
    }

    inline void CountSendTask(uint64_t& sentAic, uint64_t& sentAiv) {
        sentAic += sendCnt_[static_cast<int>(CoreType::AIC)];
        sentAiv += sendCnt_[static_cast<int>(CoreType::AIV)];
        waitTaskCnt_[static_cast<int>(CoreType::AIC)] += sentAic;
        waitTaskCnt_[static_cast<int>(CoreType::AIV)] += sentAiv;
        sendCnt_[static_cast<int>(CoreType::AIC)] = 0;
        sendCnt_[static_cast<int>(CoreType::AIV)] = 0;
    }

    template <bool enableAicpuTask = false>
    inline void RunCoreTask(DeviceTaskCtrl *taskCtrl) {
        uint64_t sentAic = 0;
        uint64_t sentAiv = 0;
        DispatchAiCoreTask(CoreType::AIC, readyAicCoreFunctionQue_, aicStart_, aicEnd_);
        CountSendTask(sentAic, sentAiv);
        DispatchAiCoreTask(CoreType::AIV, readyAivCoreFunctionQue_, aivStart_, aivEnd_);
        CountSendTask(sentAic, sentAiv);
        uint64_t sentAicpu = 0UL;
        if constexpr (enableAicpuTask) {
            if (aicpuIdx_ == 1) {
                sentAicpu = ResolveDepForAicpuTask();
            }
        }

        taskCtrl->finishedFunctionCnt.fetch_add(sentAic + sentAiv + reSolveHubCnt_ + sentAicpu,
            std::memory_order_relaxed);
#if DEBUG_SWITCH
        __sync_fetch_and_add(&(taskCtrl->finishedAicFunctionCnt), sentAic);
        __sync_fetch_and_add(&(taskCtrl->finishedAivFunctionCnt), sentAiv);
        __sync_fetch_and_add(&(taskCtrl->finishedAicpuFunctionCnt), sentAicpu);
        procAicCoreFunctionCnt_ += sentAic;
        procAivCoreFunctionCnt_ += sentAiv;
        procAicpuFunctionCnt_ += sentAicpu;
        DEV_DEBUG("finish send  aic task cnt: %lu,  aiv task cnt: %lu, hub task cnt:%lu, aicpu task cnt:%lu, target totalcnt: %lu \n",
            taskCtrl->finishedAicFunctionCnt, taskCtrl->finishedAivFunctionCnt,
            reSolveHubCnt_, taskCtrl->finishedAicpuFunctionCnt, curDevTask_->coreFunctionCnt);
#endif
        reSolveHubCnt_ = 0;
    }

    inline int RunTask(DeviceTaskCtrl *taskCtrl) {
        int ret = 0;
        DEV_INFO("receive new task %lu\n", taskCtrl->taskId);
        InitTaskData(taskCtrl);

        RunCoreTask(taskCtrl);
        if (aicpuIdx_ == 1) {
            aicpuTaskManager_.Init(curDevTask_);
        }

        npu::tile_fwk::dynamic::TimeCheck tm;
        while (taskCtrl->finishedFunctionCnt.load(std::memory_order_relaxed) < curDevTask_->coreFunctionCnt) {
            RunCoreTask<true>(taskCtrl);
            if (npu::tile_fwk::dynamic::CheckTimeOut("wait task send finish.", tm) != 0) {
                return -1;
            }
        }

        DEV_DEBUG("Aicpu %d proc finish send all task .\n", aicpuIdx_);
        ret = WaitAllAicoreFinish(aicStart_, aicEnd_);
        if (ret != npu::tile_fwk::dynamic::DEVICE_MACHINE_OK) {
            DEV_ERROR("wait tail aic task timeout .\n");
        }
        ret = WaitAllAicoreFinish(aivStart_, aivEnd_);
        if (ret != npu::tile_fwk::dynamic::DEVICE_MACHINE_OK) {
            DEV_ERROR("wait tail aiv task timeout .\n");
        }
        if (aicpuIdx_ == 1) {
            while (!aicpuTaskManager_.Finished()) {
                (void)aicpuTaskManager_.TaskProcess();
            }
        }
        return ret;
    }

    inline int Run(int threadIdx, DeviceArgs *deviceArgs, DeviceTaskCtrl *taskCtrl = nullptr) {
        Init(threadIdx, deviceArgs);

        int ret = HandkShake();
        if (ret != npu::tile_fwk::dynamic::DEVICE_MACHINE_OK) {
            DEV_DEBUG("hand shake timeout .\n");
            AbnormalStop();
            return ret;
        }

        prof_.ProfStart();
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
        NormalStop();
        ProfStop();
        DEV_DEBUG("Aicpu %d stop ret = %d, proc aic task cnt: %lu,  aiv task cnt: %lu.\n", aicpuIdx_, ret,
            procAicCoreFunctionCnt_, procAivCoreFunctionCnt_);
        return ret;
    }
    void PushTask(DeviceTaskCtrl *taskCtrl) { taskQueue_.Enqueue(taskCtrl); }

private:
    inline void DumpTaskProf() {
        ForEachManageAicore([this](int coreIdx) {
            volatile KernelArgs *arg = (KernelArgs *)(sharedBuffer_ + coreIdx * SHARED_BUFFER_SIZE);
            volatile Metrics *metric = (Metrics *)(arg->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
            DEV_INFO("aicore %d host alloc metric memory :%p.\n", coreIdx, metric);
            if (metric == nullptr) {
                DEV_INFO("aicore %d Null metric.\n", coreIdx);
                return;
            }
            while (metric->isMetricStop != 1) {
            }; // wait aicore dcci metric data finish
            DEV_DEBUG("Dump core %d prof data , task cnt %ld, metric:%p \n.", coreIdx, metric->taskCount, metric);
            for (int i = 0; i < metric->taskCount; i++) {
                volatile TaskStat *stat = &metric->tasks[i];
                prof_.ProfGet(coreIdx, stat->subGraphId, stat->taskId,
                    &((Metrics *)(arg->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]))->tasks[i]);
                DEV_DEBUG("  Dump prof for task %d, execstart: %ld execend :%ld .\n", stat->taskId, stat->execStart,
                    stat->execEnd);
            }
        });
    }

    inline void ProfStop() {
#if PROF_DFX_HOST_PREPARE_MEMORY_MODE
        DumpTaskProf();
#endif
        prof_.ProfStop();
    }
    inline void DumpAiCoreStatus() const {
#if defined(DEBUG_SWITCH) && DEBUG_SWITCH
        ForEachManageAicore([this](int coreIdx) {
            volatile KernelArgs *arg = (KernelArgs *)(sharedBuffer_ + coreIdx * SHARED_BUFFER_SIZE);
            DEV_INFO("\n!!***********************aicore %d last status **************************!!\n", coreIdx);
            DEV_INFO("hello status %ld\n", arg->shakeBuffer[0]);
            DEV_INFO("last_taskId %ld task status %ld\n", arg->shakeBuffer[NUM_ONE], arg->shakeBuffer[NUM_TWO]);
            for (size_t i = 0; i < sizeof(arg->taskEntry) / sizeof(TaskEntry); i++) {
                DEV_INFO("task req index %lu: taskId %d, subGraphID %d funcAddr %ld\n", i, arg->taskEntry.taskId,
                    arg->taskEntry.subGraphId, arg->taskEntry.funcAddr);
            }

            for (size_t i = 0; i < sizeof(arg->taskStat) / sizeof(TaskStat); i++) {
                DEV_INFO("task rsp index %lu: taskId %d, subGraphID %d execStart %ld execEnd %ld\n", i,
                    arg->taskStat[i].taskId, arg->taskStat[i].subGraphId, arg->taskStat[i].execStart,
                    arg->taskStat[i].execEnd);
            }

            DEV_INFO("reg low task: runningid(%lu) pendingid(%lu) dfxpos(%d)\n", runningIds_[coreIdx],
                pendingIds_[coreIdx], taskDfxStatPos_[coreIdx]);

            DEV_INFO("send task info ~~~~~~~~~~~~~~~~~~~~~~~~~~~~count:%lu~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n",
                sendTask_[coreIdx].size());
            for (size_t i = 0; i < sendTask_[coreIdx].size(); i++) {
                DEV_INFO("send task: seqno %lu, taskId %lx\n", i, sendTask_[coreIdx][i].taskId);
            }

            DEV_INFO("recv finish task info ~~~~~~~~~~~~~~~~~~~~~~~~~count:%lu~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n",
                recvFinTask_[coreIdx].size());
            for (size_t i = 0; i < recvFinTask_[coreIdx].size(); i++) {
                DEV_INFO("recv task: seqno %lu, taskId %lx\n", i, recvFinTask_[coreIdx][i].taskId);
            }

            DEV_INFO("recv ack task info ~~~~~~~~~~~~~~~~~~~~~~~~~~~~count:%lu~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n",
                recvAckTask_[coreIdx].size());
            for (size_t i = 0; i < recvAckTask_[coreIdx].size(); i++) {
                DEV_INFO("recv ack task: seqno %lu, taskId %lx\n", i, recvAckTask_[coreIdx][i].taskId);
            }
        });
#endif
    }

    inline void DumpTaskTensor(int &coreIdx, volatile TaskStat *stat) {
        auto coreId = GetPhyIdByBlockId(coreIdx);
        DEV_DEBUG("Output coreId is %d taskid is %d, with execStart: %ld, execend: %ld \n", coreId, stat->taskId,
            stat->execStart, stat->execEnd);
        aicoreDump_.DumpInit(stat->subGraphId, stat->taskId, coreId, stat->execStart, stat->execEnd);
        auto funcInfo =
            &(reinterpret_cast<CoreFunctionWsAddr *>(curDevTask_->coreFuncData.coreFunctionWsAddr))[stat->taskId];
        aicoreDump_.DoDump(funcInfo->invokeEntryInfo, funcInfo->invokeEntryNum, funcInfo->invokeEntryAddr, "output");
    }

    inline bool CheckTaskFinished(int coreIdx) {
        uint64_t finTaskVal = GetFinishedTask(coreIdx);
        uint32_t regLFinTaskId = REG_LOW_TASK_ID(finTaskVal);
        uint32_t regLFinTaskState = REG_LOW_TASK_STATE(finTaskVal);

        if (regLFinTaskState == TASK_FIN_STATE &&
            (pendingIds_[coreIdx] == regLFinTaskId || runningIds_[coreIdx] == regLFinTaskId)) {
            DfxProcAfterFinishTask(coreIdx, regLFinTaskId);
            pendingIds_[coreIdx] = AICORE_TASK_INIT;
            runningIds_[coreIdx] = AICORE_TASK_INIT;
        }

        return pendingIds_[coreIdx] == AICORE_TASK_INIT && runningIds_[coreIdx] == AICORE_TASK_INIT;
    }

    inline int WaitAllAicoreFinish(int coreIdxStart, int coreIdxEnd) {
        int stopSent = 0;
        bool coreStopped[MAX_MANAGER_AIV_NUM] = {false};
        npu::tile_fwk::dynamic::TimeCheck tm;
        while (stopSent < coreIdxEnd - coreIdxStart) {
            for (int i = coreIdxStart; i < coreIdxEnd; i++) {
                if (coreStopped[i - coreIdxStart]) {
                    continue;
                }
                if (CheckTaskFinished(i)) {
                    stopSent++;
                    coreStopped[i - coreIdxStart] = true;
                    continue;
                }
                if (npu::tile_fwk::dynamic::CheckTimeOut("wait tail task", tm) != 0) {
                    DEV_ERROR("wait tail task finish timeout coreindx=%d.\n", i);
                    return -1;
                }
            }
        }
        uint64_t tmp_task_finish_time;
        prof_.AsmCntvc(tmp_task_finish_time);
        task_end_time_ = std::max(task_end_time_, tmp_task_finish_time);
        DEV_DEBUG("Last task finish with time: %lu\n", task_end_time_);
        return 0;
    }

    inline uint64_t TryBatchSendTask(CoreType type, StaticReadyCoreFunctionQueue* readyQue,
                int coreIdxStart, int coreIdxEnd) {
        if (readyQue->tail - readyQue->head == 0) {
            DEV_DEBUG("AiCpud:%d, can not send task currently. ready Task: %lu \n", aicpuIdx_,
                readyQue->tail - readyQue->head);
            return 0;
        }

        // check other aicpu if idle, Prioritize remaining tasks for scheduling by other AICPUs
        if (coreRunReadyCnt_[static_cast<int>(type)] == 0 && IsExistOtherAicpuIdle(type)) {
            return 0;
        }

        uint64_t ready = corePendReadyCnt_[static_cast<int>(type)];
        if (ready == 0 ) {
            DEV_DEBUG("AiCpud:%d, can not send task currently. ready Core: %lu.\n", aicpuIdx_, ready);
            return 0;
        }
        uint64_t readyId[MAX_MANAGER_AIV_NUM];
        ReadyQueueLock(readyQue);
        uint64_t taskIdx = readyQue->head;
        uint64_t taskCount = std::min(ready, readyQue->tail - readyQue->head);
        if (taskCount == 0) {
            DEV_DEBUG("AiCpud:%d, taskCount is zero \n", aicpuIdx_);
            ReadyQueueUnLock(readyQue);
            return 0;
        }
        if (READY_QUE_LIFO_SWITCH && !firstLock[static_cast<int>(type)]) {
            memcpy_s(readyId, taskCount * sizeof(uint64_t),
                reinterpret_cast<uint8_t *>(&readyQue->elem[readyQue->tail - taskCount]), taskCount * sizeof(uint64_t));
            readyQue->tail -= taskCount;
        } else {
            readyQue->head += taskCount;
        }
        ReadyQueueUnLock((readyQue));
        DEV_DEBUG("AiCpud:%d, pop all new task count: %lu \n", aicpuIdx_, taskCount);
        BatchSendTask(type, (READY_QUE_LIFO_SWITCH && !firstLock[static_cast<int>(type)])? readyId : &readyQue->elem[taskIdx],
            taskCount, coreIdxStart, coreIdxEnd, READY_QUE_LIFO_SWITCH);
        DEV_DEBUG("core ready cnt: %u \n", corePendReadyCnt_[static_cast<int>(type)]);
        firstLock[static_cast<int>(type)] = false;
        return taskCount;
    }

    inline uint32_t BatchSendTask(CoreType type, uint64_t *newTask, uint32_t taskCount,
        int coreIdxStart, int coreIdxEnd, bool isLifo) {
        uint32_t sendCnt = 0;
        uint32_t taskIdx = isLifo ? taskCount : 0;
        uint32_t coreRunReadyCnt = coreRunReadyCnt_[static_cast<int>(type)];
        DEV_DEBUG("Begin Batch send %s task: corerunreadycnt:%u, pendreadyCnt:%u, taskCount:%u \n",
            type == CoreType::AIC ? "AIC": "AIV", coreRunReadyCnt,
            corePendReadyCnt_[static_cast<int>(type)], taskCount);
        while (sendCnt < (uint64_t)coreRunReadyCnt && sendCnt < taskCount) {
            DEV_DEBUG("  ## send task use runready core %u \n",
                runReadyCoreIdx_[static_cast<int>(type)][coreRunReadyCnt_[static_cast<int>(type)] - 1]);
            SendTaskToAiCore(type,
                runReadyCoreIdx_[static_cast<int>(type)][--coreRunReadyCnt_[static_cast<int>(type)]],
                isLifo ? newTask[--taskIdx] : newTask[taskIdx++]);
            sendCnt++;
        }
        corePendReadyCnt_[static_cast<int>(type)] -= sendCnt;

        uint32_t idx = lastPendReadyCoreIdx_[static_cast<int>(type)];
        uint32_t coreNum = coreIdxEnd - coreIdxStart;
        uint32_t lastProcCore = idx;
        DEV_DEBUG("  ## send task left pend ready cnt %u , last core index:%u\n",
            corePendReadyCnt_[static_cast<int>(type)], idx);
        while (corePendReadyCnt_[static_cast<int>(type)] > 0 && sendCnt < taskCount) {
            if (pendingIds_[idx] == AICORE_TASK_INIT) {
                DEV_DEBUG("  ## send task use pendready core %u \n", idx);
                SendTaskToAiCore(type, idx, isLifo ? newTask[--taskIdx] : newTask[taskIdx++]);
                sendCnt++;
                corePendReadyCnt_[static_cast<int>(type)]--;
                lastProcCore = idx;
            }
            idx = coreIdxStart + (idx - coreIdxStart + 1) % coreNum;
        }

        if (lastProcCore != lastPendReadyCoreIdx_[static_cast<int>(type)]) {
            lastPendReadyCoreIdx_[static_cast<int>(type)] = coreIdxStart + (lastProcCore - coreIdxStart + 1) % coreNum;
        }
        DEV_DEBUG("  ## finish send task left runreadycnt:%u pendreadycnt %u, last coreindex:%u \n",
            coreRunReadyCnt_[static_cast<int>(type)], corePendReadyCnt_[static_cast<int>(type)], idx);
        return sendCnt;
    }

    inline uint64_t DispatchAiCoreTask(CoreType type, StaticReadyCoreFunctionQueue* readyQue,
                                       int coreIdxStart, int coreIdxEnd) {
        uint64_t taskCount = TryBatchSendTask(type, readyQue, coreIdxStart, coreIdxEnd);
        if (waitTaskCnt_[static_cast<int>(type)] > 0) {
            ResolveDepForAllAiCore(type, readyQue, coreIdxStart, coreIdxEnd);
            taskCount += TryBatchSendTask(type, readyQue, coreIdxStart, coreIdxEnd);
        }
        if (coreRunReadyCnt_[static_cast<int>(type)] > 0)  {
            AicpuIsIdle(type);
        } else {
            AicpuIsBusy(type);
        }
        return taskCount;
    }

    inline void SendTaskToAiCore(CoreType type, int coreIdx, uint64_t newTask) {
        if (isFirstTaskSend_) {
            prof_.AsmCntvc(task_start_time_);
            DEV_DEBUG("First task start with time: %lu\n", task_start_time_);
            isFirstTaskSend_ = false;
        }
        AddTask(coreIdx, newTask);
        pendingIds_[coreIdx] = newTask;
        sendCnt_[static_cast<int>(type)]++;
        DEV_DEBUG("Send task %lu, at core %d ,type:%d \n", newTask, coreIdx, static_cast<int>(type));
    }

    inline void SetAiCpuStat(int coreIdx, uint64_t taskId) {
        struct AiCpuTaskStat aiCpuTaskStat;
        aiCpuTaskStat.taskId = taskId;
        aiCpuTaskStat.coreId = GetPhyIdByBlockId(coreIdx);
        prof_.AsmCntvc(aiCpuTaskStat.taskGetStart);
        prof_.SetAiCpuTaskStat(taskId, aiCpuTaskStat);
    };

    inline void AddTask(int coreIdx, uint64_t taskId) {
        DEV_DEBUG("CoreIdx: %d, Send new task: %lx.\n", coreIdx, taskId);
        SetReadyQueue(coreIdx, taskId + 1);
#if PERF_AICPU_TEST_SWITCH
        SetAiCpuStat(coreIdx, taskId);
#endif

#if defined(DEBUG_SWITCH) && DEBUG_SWITCH
        DEV_DEBUG("Start to dump input tensor info, num is\n");
        auto funcInfo = &(reinterpret_cast<CoreFunctionWsAddr *>(curDevTask_->coreFuncData.coreFunctionWsAddr))[taskId];
        aicoreDump_.DumpInit(funcInfo->psgId, taskId, GetPhyIdByBlockId(coreIdx));
        aicoreDump_.DoDump(funcInfo->invokeEntryInfo, funcInfo->invokeEntryNum, funcInfo->invokeEntryAddr, "input");
        sendTask_[coreIdx].push_back(TaskInfo(coreIdx, taskId));
#endif
    }

    inline void PushReadyQue(StaticReadyCoreFunctionQueue *readyQue, void *idList, uint32_t idCnt) const {
        ReadyQueueLock(readyQue);
        memcpy_s(
            &readyQue->elem[readyQue->tail], idCnt * sizeof(uint64_t), (uint8_t *)idList, idCnt * sizeof(uint64_t));
        readyQue->tail += idCnt;
        ReadyQueueUnLock(readyQue);
    }

    inline void ResolveDepForAllAiCore(
        CoreType type, StaticReadyCoreFunctionQueue *readyQue, int coreIdxStart, int coreIdxEnd) {
        (void)readyQue;
        for (int i = coreIdxStart; i < coreIdxEnd; i++) {
            if (IsNoTaskDispatch(i)) {
                continue;
            }
            ResolveByRegVal(type, i, GetFinishedTask(i));
            if (readyAicCoreFunctionQue_->tail - readyAicCoreFunctionQue_->head == 0 ||
                readyAivCoreFunctionQue_->tail - readyAivCoreFunctionQue_->head == 0) {
                BatchPushReadyQueue();
            }
        }

        BatchPushReadyQueue();
        return;
    }

    inline void BatchPushReadyQueue() {
        uint32_t aicIndex = static_cast<uint32_t>(CoreType::AIC);
        uint32_t aivIndex = static_cast<uint32_t>(CoreType::AIV);
        if (readyCount[aicIndex] > 0) {
            if (SEND_TASK_IMMEDIATELY_SWITCH) {
                uint32_t sendCnt = BatchSendTask(CoreType::AIC,
                    static_cast<uint64_t*>(readyIds[aicIndex]), readyCount[aicIndex], aicStart_, aicEnd_, true);
                readyCount[aicIndex] -= sendCnt;
            }
            DEV_DEBUG("resolved new task, aic ready count: %lu coretype:%u\n", readyCount[aicIndex], aicIndex);
            if (readyCount[aicIndex] > 0) {
                PushReadyQue(readyAicCoreFunctionQue_, readyIds[aicIndex], readyCount[aicIndex]);
            }
            readyCount[aicIndex] = 0;
        }

        if (readyCount[aivIndex] > 0) {
            if (SEND_TASK_IMMEDIATELY_SWITCH) {
                uint32_t sendCnt = BatchSendTask(CoreType::AIV,
                    static_cast<uint64_t*>(readyIds[aivIndex]), readyCount[aivIndex], aivStart_, aivEnd_, true);
                readyCount[aivIndex] -= sendCnt;
            }
            DEV_DEBUG("resolved new task, aiv ready count: %lu coretype:%u\n", readyCount[aivIndex], aivIndex);
            if (readyCount[aivIndex] > 0) {
                PushReadyQue(readyAivCoreFunctionQue_, readyIds[aivIndex], readyCount[aivIndex]);
            }
            readyCount[aivIndex] = 0;
        }

        if (readyIdsExtend[aicIndex].size() > 0) {
            DEV_DEBUG("resolved new task, extend aic ready count: %lu, coretype:%u\n", readyIdsExtend[aicIndex].size(),
                aicIndex);
            PushReadyQue(readyAicCoreFunctionQue_, readyIdsExtend[aicIndex].data(), readyIdsExtend[aicIndex].size());
            readyIdsExtend[aicIndex].clear();
        }

        if (readyIdsExtend[aivIndex].size() > 0) {
            DEV_DEBUG("resolved new task, extend aiv ready count: %lu, coretype:%u\n", readyIdsExtend[aivIndex].size(),
                aivIndex);
            PushReadyQue(readyAivCoreFunctionQue_, readyIdsExtend[aivIndex].data(), readyIdsExtend[aivIndex].size());
            readyIdsExtend[aivIndex].clear();
        }
    }

    inline uint64_t ResolveDepForAicpuTask() {
        uint64_t taskCount = aicpuTaskManager_.TaskProcess();
        std::vector<uint64_t> completed = aicpuTaskManager_.TaskPoll();
        for (const uint64_t &taskId : completed) {
            ResolveDep(taskId);
            BatchPushReadyQueue();
        }
        return taskCount;
    }

    inline bool IsNoTaskDispatch(int coreIdx) {
        return (runningIds_[coreIdx] == AICORE_TASK_INIT && pendingIds_[coreIdx] == AICORE_TASK_INIT);
    }

    inline bool SendTaskDirectlyWhenCoreRunReady(CoreType type, int coreIdx) {
        if (SEND_TASK_IMMEDIATELY_SWITCH && readyCount[static_cast<int>(type)] > 0) {
            SendTaskToAiCore(type, coreIdx, readyIds[static_cast<int>(type)][--readyCount[static_cast<int>(type)]]);
            return true;
        }
        return false;
    }

    inline void ResolveWhenSyncMode(CoreType type, uint32_t finTaskId, uint32_t finTaskState, int coreIdx)  {
        if (finTaskId == pendingIds_[coreIdx] && finTaskState == TASK_FIN_STATE) {
            DEV_DEBUG("core index: %d, PendingTask Finished."
                " pending: %lx\n", coreIdx, pendingIds_[coreIdx]);
            ResolveDepWithDfx(type, coreIdx, finTaskId);
            pendingIds_[coreIdx] = AICORE_TASK_INIT;
            corePendReadyCnt_[static_cast<int>(type)]++;
            runReadyCoreIdx_[static_cast<int>(type)][coreRunReadyCnt_[static_cast<int>(type)]++] = coreIdx;
        }
    }

    inline void ResolveByRegVal(CoreType type, int coreIdx, uint64_t finTaskRegVal) {
        uint32_t finTaskId = REG_LOW_TASK_ID(finTaskRegVal);
        uint32_t finTaskState = REG_LOW_TASK_STATE(finTaskRegVal);

        DEV_DEBUG("reslove task core index: %d, finishtaskid:%x, finishstate:%u \n", coreIdx, finTaskId, finTaskState);
#if SCHEDULE_USE_PENDING_AND_RUNING_SWITCH
        uint32_t tmpTaskId;
        if (finTaskId == pendingIds_[coreIdx] && finTaskState == TASK_FIN_STATE) {
            DEV_DEBUG("PendingTask Finished.runningid:%lx\n", runningIds_[coreIdx]);
            tmpTaskId = runningIds_[coreIdx];
            runningIds_[coreIdx] = AICORE_TASK_INIT;
            pendingIds_[coreIdx] = AICORE_TASK_INIT;
            if (!SendTaskDirectlyWhenCoreRunReady(type, coreIdx)) {
                runReadyCoreIdx_[static_cast<int>(type)][coreRunReadyCnt_[static_cast<int>(type)]++] = coreIdx;
                corePendReadyCnt_[static_cast<int>(type)]++;
            }
            if (tmpTaskId != AICORE_TASK_INIT) {
                ResolveDepWithDfx(type, coreIdx, tmpTaskId);
            }
            ResolveDepWithDfx(type, coreIdx, finTaskId);
        } else if (finTaskId == pendingIds_[coreIdx] && finTaskState == TASK_ACK_STATE) {
#if defined(DEBUG_SWITCH) && DEBUG_SWITCH
            recvAckTask_[coreIdx].push_back(TaskInfo(coreIdx, finTaskId));
#endif
            DEV_DEBUG("PendingTask Acked. Running task finished.runningid: %lx\n", runningIds_[coreIdx]);
            tmpTaskId = runningIds_[coreIdx];
            runningIds_[coreIdx] = finTaskId;
            pendingIds_[coreIdx] = AICORE_TASK_INIT;
            corePendReadyCnt_[static_cast<int>(type)]++;
            if (tmpTaskId != AICORE_TASK_INIT) {
                ResolveDepWithDfx(type, coreIdx, tmpTaskId);
            }
        } else if (finTaskId == runningIds_[coreIdx] && finTaskState == TASK_FIN_STATE) {
            DEV_DEBUG("core index: %d, RuningTask Finished. pending: %lx, running: %lx\n",
                coreIdx, pendingIds_[coreIdx], runningIds_[coreIdx]);
            runningIds_[coreIdx] = AICORE_TASK_INIT;
            if (pendingIds_[coreIdx] == AICORE_TASK_INIT) {
                if (!SendTaskDirectlyWhenCoreRunReady(type, coreIdx)) {
                    runReadyCoreIdx_[static_cast<int>(type)][coreRunReadyCnt_[static_cast<int>(type)]++] = coreIdx;
                } else {
                    corePendReadyCnt_[static_cast<int>(type)]--;
                }
            }
            ResolveDepWithDfx(type, coreIdx, finTaskId);
        } else {
            DEV_DEBUG("Warning, maybe inconsistent state. coreidx:%d,finTask:%lx,pending:%lx,running:%lx\n",
                coreIdx, finTaskRegVal, pendingIds_[coreIdx], runningIds_[coreIdx]);
        }
#else
        ResolveWhenSyncMode(type, finTaskId, finTaskState, coreIdx);
#endif
    }

    inline void PushAicpuTaskQueue(uint64_t taskId) {
        aicpuTaskManager_.TaskEnqueue(taskId);
    }

    inline int GetNextSendCoreIdx(int coreType) {
        if (coreRunReadyCnt_[coreType] > 0) {
            corePendReadyCnt_[coreType]--;
            return runReadyCoreIdx_[coreType][--coreRunReadyCnt_[coreType]];
        }

        int startIdx;
        int coreNum;
        int idx = lastPendReadyCoreIdx_[coreType];
        if (coreType == static_cast<int>(CoreType::AIC)) {
            startIdx = aicStart_;
            coreNum = aicEnd_ - aicStart_;
        } else {
            startIdx = aivStart_;
            coreNum = aivEnd_ - aivStart_;
        }
        if (corePendReadyCnt_[coreType] > 0) {
            while (pendingIds_[idx] != AICORE_TASK_INIT) {
                idx = startIdx + (idx - startIdx + 1) % (coreNum);
            }
            lastPendReadyCoreIdx_[coreType] = startIdx + (idx - startIdx + 1) % (coreNum);
            corePendReadyCnt_[coreType]--;
            return idx;
        }
        return INVALID_CORE_IDX;
    }

    inline bool SendTaskDirectlyWhenTaskReady(int coreType, int64_t taskId) {
        if (!SEND_TASK_IMMEDIATELY_SWITCH) {
            return false;
        }
        int readyCore = GetNextSendCoreIdx(coreType);
        if (readyCore == INVALID_CORE_IDX) {
            return false;
        }

        SendTaskToAiCore(static_cast<CoreType>(coreType), readyCore, taskId);
        return true;
    }

    inline void PushReadyTask(int coreType, int64_t taskId) {
        if (SendTaskDirectlyWhenTaskReady(coreType, taskId)) {
            return;
        }
        if (readyCount[coreType] < READY_ID_FIX_CACHE_NUM) {
            readyIds[coreType][readyCount[coreType]++] = taskId;
        } else {
            /*  固定缓存不够用了 插入扩展动态缓存 */
            readyIdsExtend[coreType].push_back(taskId);
        }
    }

    inline void ResolveVirtualPure(uint64_t dep, CoreFunctionReadyState* readyState) {
        DEV_DEBUG("new virtual pure task resolved. id: %lu\n", dep);
        auto virtualFuncInfo =
            &(reinterpret_cast<CoreFunctionWsAddr *>(curDevTask_->coreFuncData.coreFunctionWsAddr)[dep]);
        auto topo = reinterpret_cast<CoreFunctionTopo *>(virtualFuncInfo->topoAddr);
        for (uint64_t i = 0 ; i < topo->depNum; i++) {
            uint64_t depId = topo->depIds[i];
            if (readyState[depId].coreType == static_cast<uint64_t>(MachineType::AICPU)) {
                PushAicpuTaskQueue(depId);
                continue;
            }
            PushReadyTask(readyState[depId].coreType, depId);
        }
    }

    inline void ResolveVirtualMix(uint64_t dep, CoreFunctionReadyState* readyState) {
        DEV_DEBUG("new virtual mix task resolved. id: %lu\n", dep);
        auto virtualFuncInfo =
            &(reinterpret_cast<CoreFunctionWsAddr *>(curDevTask_->coreFuncData.coreFunctionWsAddr)[dep]);
        auto topo = reinterpret_cast<CoreFunctionTopo *>(virtualFuncInfo->topoAddr);
        for (uint64_t i = 0 ; i < topo->depNum; i++) {
            uint64_t depId = topo->depIds[i];
            if (readyState[depId].coreType == static_cast<uint64_t>(MachineType::AICPU)) {
                PushAicpuTaskQueue(depId);
                continue;
            }
            if (readyState[depId].readyCount == topo->readyCount) {
                PushReadyTask(readyState[depId].coreType, depId);
            } else {
                if (__sync_add_and_fetch(&(readyState[depId].readyCount), (-1) * topo->readyCount) == 0) {
                    PushReadyTask(readyState[depId].coreType, depId);
                }
            }
        }
    }

    inline void ResolveByCoreType(int coretype, uint64_t depTaskId, CoreFunctionReadyState *readyState) {
        // Compiler optimizations reduce switch-case to O(1), rendering if-else unnecessary in such cases.
        switch (coretype) {
            case static_cast<int>(MachineType::AIV):
            case static_cast<int>(MachineType::AIC): {
                PushReadyTask(coretype, depTaskId);
                break;
            }
            case static_cast<int>(MachineType::MIX): {
                DEV_ERROR("in valid core type mix.");
                break;
            }
            case static_cast<int>(MachineType::AICPU): {
                PushAicpuTaskQueue(depTaskId);
                break;
            }
            case static_cast<int>(MachineType::HUB): {
                reSolveHubCnt_++;
                ResolveDep(depTaskId);
                break;
            }
            case static_cast<int>(MachineType::VIRTUAL_PURE): {
                ResolveVirtualPure(depTaskId, readyState);
                break;
            }
            case static_cast<int>(MachineType::VIRTUAL_MIX): {
                ResolveVirtualMix(depTaskId, readyState);
                break;
            }
            default: {
                break;
            }
        }
        DEV_DEBUG("new task resolved. id: %lu, coretype: %d\n", depTaskId, coretype);
    }

    inline void ResolveDep(uint64_t finishId) {
        auto readyState = reinterpret_cast<CoreFunctionReadyState *>(curDevTask_->coreFunctionReadyStateAddr);
        auto funcInfo =
            &(reinterpret_cast<CoreFunctionWsAddr *>(curDevTask_->coreFuncData.coreFunctionWsAddr)[finishId]);
        auto topo = reinterpret_cast<CoreFunctionTopo *>(funcInfo->topoAddr);
        DEV_DEBUG("resolve %lx, Dep core function num: %lu\n", finishId, topo->depNum);
#if PERF_AICPU_TEST_SWITCH
        struct AiCpuTaskStat aiCpuTaskStat = prof_.GetAiCpuTaskStat(finishId);
        prof_.AsmCntvc(aiCpuTaskStat.execStart);
#endif
        for (uint64_t i = 0; i < topo->depNum; i++) {
            uint64_t dep = topo->depIds[i];
            int ret = __sync_add_and_fetch(&(readyState[dep].readyCount), 1);
            if (ret != 0) {
                continue;
            }
            ResolveByCoreType(readyState[dep].coreType, dep, readyState);
        }
#if PERF_AICPU_TEST_SWITCH // 性能AICPU数据测试
        prof_.AsmCntvc(aiCpuTaskStat.execEnd);
        prof_.ProfGetAiCpuTaskStat(aicpuIdx_, &aiCpuTaskStat);
#endif
    }

    inline void ResolveDepWithDfx(CoreType type, int coreIdx, uint64_t finishId) {
        ResolveDep(finishId);
        DfxProcAfterFinishTask(coreIdx, finishId);
        if (waitTaskCnt_[static_cast<int>(type)] != 0) {
            waitTaskCnt_[static_cast<int>(type)]--;
        }
    }

    inline bool IsExistOtherAicpuIdle(CoreType type) {
        int idx = (aicpuIdx_ + 1) % npu::tile_fwk::dynamic::START_AICPU_NUM;
        while (idx != aicpuIdx_) {
            if (curTaskCtrl_->isAicpuIdle[static_cast<int>(type)][idx].load(std::memory_order_relaxed) == true){
                return true;
            }
            idx = (idx + 1) % npu::tile_fwk::dynamic::START_AICPU_NUM;
        }
        return false;
    }

    inline void AicpuIsBusy(CoreType type) {
        if (curTaskCtrl_->isAicpuIdle[static_cast<int>(type)][aicpuIdx_] != false) {
            curTaskCtrl_->isAicpuIdle[static_cast<int>(type)][aicpuIdx_].store(false, std::memory_order_relaxed);
        }
    }

    inline void AicpuIsIdle(CoreType type) {
        if (curTaskCtrl_->isAicpuIdle[static_cast<int>(type)][aicpuIdx_] != true) {
            curTaskCtrl_->isAicpuIdle[static_cast<int>(type)][aicpuIdx_].store(true, std::memory_order_relaxed);
        }
    }

    inline void BatchGetFinishedTask(uint64_t finTask[], int coreIdxStart, int coreIdxEnd) {
        uint64_t finTaskGet;
        for (int i = coreIdxStart; i < coreIdxEnd; i++) {
            finTaskGet = GetFinishedTask(i);
            finTask[i - coreIdxStart] = finTaskGet;
        }
    }

    inline uint64_t GetFinishedTask(int coreIdx) { return *(finishRegQueues_[GetPhyIdByBlockId(coreIdx)]); }

    inline void ResetCnt() {
        waitTaskCnt_[static_cast<int>(CoreType::AIC)] = 0;
        waitTaskCnt_[static_cast<int>(CoreType::AIV)] = 0;
        readyCount[static_cast<int>(CoreType::AIC)] = 0;
        readyCount[static_cast<int>(CoreType::AIV)] = 0;
        sendCnt_[static_cast<int>(CoreType::AIC)] = 0;
        sendCnt_[static_cast<int>(CoreType::AIV)] = 0;
    }

    inline void Init(int threadIdx, DeviceArgs *deviceArgs) {
        aicNum_ = deviceArgs->nrAic;
        aivNum_ = deviceArgs->nrAiv;
        aicpuNum_ = npu::tile_fwk::dynamic::START_AICPU_NUM;
        aicpuIdx_ = threadIdx;
        aicValidNum_ = deviceArgs->nrValidAic;
        regAddrs_ = reinterpret_cast<int64_t *>(deviceArgs->coreRegAddr);
        sharedBuffer_ = deviceArgs->sharedBuffer;
        runningIds_.fill(AICORE_STATUS_INIT);
        pendingIds_.fill(AICORE_STATUS_INIT);
        taskDfxStatPos_.fill(REG_LOW_TASK_PING);

        blockIdToPhyCoreId_.fill(-1);
        readyRegQueues_.fill(nullptr);
        finishRegQueues_.fill(nullptr);
        UpdateAiCoreBlockIndexSection();
        MapRegistersForAllCores();

        args_.fill(nullptr);
        ResetCnt();
        firstLock[static_cast<int>(CoreType::AIC)] = true;
        firstLock[static_cast<int>(CoreType::AIV)] = true;
        DEV_DEBUG("Init aicore manager aicNum_ %d aivNum_  %d aicpuNum_ %d aicpuIdx_ %d "
                  "aicValidNum_ %d regAddrs_ %p sharedBuffer_ %p \n",
            aicNum_, aivNum_, aicpuNum_, aicpuIdx_, aicValidNum_, regAddrs_, (void *)sharedBuffer_);
        prof_.ProfInit(reinterpret_cast<int64_t *>(deviceArgs->corePmuRegAddr),
                       reinterpret_cast<int64_t *>(deviceArgs->pmuEventAddr));
    }

    inline int HandkShake() {
        DEV_INFO("Aicpu %d handshake start.\n", aicpuIdx_);
        int rc = ForEachManageAicoreWithRet([this](int coreIdx) -> int {
#if PERF_AICPU_TEST_SWITCH
            struct AiCpuHandShakeSta shakeHandStat;
            shakeHandStat.threadId = aicpuIdx_;
            prof_.AsmCntvc(shakeHandStat.shakeStart);
#endif
            int ret = npu::tile_fwk::dynamic::DEVICE_MACHINE_OK;
            auto args =
                reinterpret_cast<KernelArgs *>((static_cast<uint64_t>(sharedBuffer_)) + SHARED_BUFFER_SIZE * coreIdx);
            args->taskEntry.reserved[0] = dotStatus_;
            volatile int64_t *shakeBuffer = args->shakeBuffer;
            npu::tile_fwk::dynamic::TimeCheck tm;
            while ((*shakeBuffer & 0xFFFFFFFF) != AICORE_SAY_HELLO) {
                if (npu::tile_fwk::dynamic::CheckTimeOut("hand shake", tm) != 0) {
                    DEV_ERROR("hand shake %d timeout.\n", coreIdx);
                    return -1;
                }
            }
            args_[coreIdx] = args;
#if PERF_AICPU_TEST_SWITCH
            prof_.AsmCntvc(shakeHandStat.shakeEnd);
            shakeHandStat.coreId = *shakeBuffer >> NUM_THIRTY_TWO;
            prof_.ProGetHandShake(aicpuIdx_, &shakeHandStat);
#endif
            blockIdToPhyCoreId_[coreIdx] = (*shakeBuffer >> NUM_THIRTY_TWO) & AICORE_COREID_MASK;
            DEV_DEBUG("coreidx %d handshake  phycorid %d .\n", coreIdx, blockIdToPhyCoreId_[coreIdx]);
            return ret;
        });
#if PERF_AICPU_TEST_SWITCH
        prof_.ProfStopHandShake();
#endif
        if (rc != npu::tile_fwk::dynamic::DEVICE_MACHINE_OK) {
            DEV_DEBUG("Aicpu %d handshake failed end.\n", aicpuIdx_);
            return rc;
        }

        ForEachManageAicore(
            [this](int coreIdx) { WriteReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_OPEN); });
        /* write to MAINBASE reg need reg 0x18 open first */
        __sync_synchronize();
        DEV_INFO("Aicpu %d handshake sucess end.\n", aicpuIdx_);
        return 0;
    }

    /* assign aic and aiv core index section for this aicpu */
    inline void UpdateAiCoreBlockIndexSection() {
        auto f = [](int total, int idx, int part, int &start, int &end) {
            int perCpu = total / part;
            int remain = total % part;
            start = idx * perCpu + ((idx < remain) ? idx : remain);
            end = start + perCpu + ((idx < remain) ? 1 : 0);
        };

        f(aicValidNum_, aicpuIdx_, aicpuNum_, aicStart_, aicEnd_);
        f(AIV_NUM_PER_AI_CORE * aicValidNum_, aicpuIdx_, aicpuNum_, aivStart_, aivEnd_);
        aivStart_ += aicValidNum_;
        aivEnd_ += aicValidNum_;
        corePendReadyCnt_[static_cast<int>(CoreType::AIC)] = aicEnd_ - aicStart_;
        corePendReadyCnt_[static_cast<int>(CoreType::AIV)] = aivEnd_ - aivStart_;
        coreRunReadyCnt_[static_cast<int>(CoreType::AIC)] = 0;
        coreRunReadyCnt_[static_cast<int>(CoreType::AIV)] = 0;
        ForEachManageAicoreReverse(
            [this](int coreIdx) {
            int coreType = static_cast<int>(AicoreType(coreIdx));
            runReadyCoreIdx_[coreType][coreRunReadyCnt_[coreType]++] = coreIdx;
            });
        lastPendReadyCoreIdx_[static_cast<int>(CoreType::AIV)] = aivStart_;
        lastPendReadyCoreIdx_[static_cast<int>(CoreType::AIC)] = aicStart_;
        DEV_DEBUG("assign core aic coreindex section: start %d end %d.\n", aicStart_, aicEnd_);
        DEV_DEBUG("assign core aiv coreindex section: start %d end %d.\n", aivStart_, aivEnd_);
    }

    inline int GetPhyIdByBlockId(int coreIdx) { return blockIdToPhyCoreId_[coreIdx]; }

    inline void MapRegistersForAllCores() {
        for (uint32_t idx = 0; idx < aicNum_ * CORE_NUM_PER_AI_CORE; idx++) {
            void *addr = reinterpret_cast<void *>(regAddrs_[idx]);
            if (addr == nullptr) {
                continue;
            }
            DEV_DEBUG("phy core %u Addr is %p\n", idx, addr);
            volatile uint64_t *reqQueueReg =
                reinterpret_cast<volatile uint64_t *>(static_cast<uint8_t *>(addr) + REG_SPR_DATA_MAIN_BASE);
            readyRegQueues_[idx] = reqQueueReg;
            volatile uint64_t *finishQueueReg =
                reinterpret_cast<volatile uint64_t *>(static_cast<uint8_t *>(addr) + REG_SPR_COND);
            finishRegQueues_[idx] = finishQueueReg;
        }
    }

    inline void ForEachManageAicore(std::function<void(int coreIdx)> func) const {
        for (int i = aicStart_; i < aicEnd_; ++i) {
            func(i);
        }
        for (int i = aivStart_; i < aivEnd_; ++i) {
            func(i);
        }
    }

    inline void ForEachManageAicoreReverse(std::function<void(int coreIdx)> func) const {
        for (int i = aicEnd_ - 1; i >= aicStart_; i--) {
            func(i);
        }
        for (int i = aivEnd_ -1; i >= aivStart_ ; i--) {
            func(i);
        }
    }

    inline int ForEachManageAicoreWithRet(std::function<int(int coreIdx)> func) const {
        int ret = npu::tile_fwk::dynamic::DEVICE_MACHINE_OK;
        for (int i = aicStart_; i < aicEnd_; ++i) {
            ret = func(i);
            if (ret != npu::tile_fwk::dynamic::DEVICE_MACHINE_OK) {
                DEV_ERROR("proc aicore aic %d failed.\n", i);
                return ret;
            }
        }
        for (int i = aivStart_; i < aivEnd_; ++i) {
            ret = func(i);
            if (ret != npu::tile_fwk::dynamic::DEVICE_MACHINE_OK) {
                DEV_ERROR("proc aicore aiv %d failed.\n", i);
                return ret;
            }
        }
        return ret;
    }

    inline uint32_t ReadReg32(int coreIdx, int offset) {
        return *(reinterpret_cast<volatile uint32_t *>(regAddrs_[GetPhyIdByBlockId(coreIdx)] + offset));
    }

    inline void WriteReg32(int coreIdx, int offset, uint32_t val) {
        *(reinterpret_cast<volatile uint32_t *>(regAddrs_[GetPhyIdByBlockId(coreIdx)] + offset)) = val;
    }

    inline void SetReadyQueue(int coreIdx, uint64_t value) {
        volatile uint64_t *readyQ = readyRegQueues_[GetPhyIdByBlockId(coreIdx)];
        if (readyQ != nullptr) {
            *readyQ = value;
        }
    }

    inline uint64_t GetReadyQueueRegValue(int coreIdx) {
        volatile uint64_t *readyQ = readyRegQueues_[GetPhyIdByBlockId(coreIdx)];
        if (readyQ != nullptr) {
            return *readyQ;
        }
        return 0;
    }

    inline void WriteReg32ALl(int offset, uint32_t val) {
        for (int i = 0; i < aicNum_ + aivNum_; i++) {
            if (regAddrs_[i] != 0) {
                *(reinterpret_cast<volatile uint32_t *>(regAddrs_[i] + offset)) = val;
            }
        }
    }

    inline void AbnormalStop() {
        DEV_INFO("aicore manager %d try abnormal stop\n", aicpuIdx_);
        WriteReg32ALl(REG_SPR_DATA_MAIN_BASE, AICORE_TASK_STOP + 1);
        /* write to MAINBASE reg must be done before close 0x18 */
        WriteReg32ALl(REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
        DEV_INFO("aicore manager %d abnormal stopped\n", aicpuIdx_);
    }

    inline void NormalStop() {
        DEV_DEBUG("aicore manager %d try normal stop\n", aicpuIdx_);
        ForEachManageAicore([this](auto coreIdx) { SetReadyQueue(coreIdx, AICORE_TASK_STOP + 1); });
        /* write to MAINBASE reg must be done before close 0x18 */
        __sync_synchronize();
        ForEachManageAicore([this](auto coreIdx) {
          WriteReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
          volatile KernelArgs *arg = (KernelArgs *)(sharedBuffer_ + coreIdx * SHARED_BUFFER_SIZE);
          arg->shakeBuffer[0] = 0;
          arg->shakeBuffer[SHAK_BUF_COREFUNC_DATA_INDEX] = 0;
        });
        DEV_DEBUG("aicore manager %d normal stopped\n", aicpuIdx_);
    }

    inline int GetAllAiCoreNum() { return aicNum_ + aivNum_; }
    inline void SetDotStatus(int64_t status) { dotStatus_ = status; }
    inline CoreType AicoreType(int coreIdx) const { return coreIdx < aicEnd_ ? CoreType::AIC : CoreType::AIV; }
    inline void SetNextDfxPos(int coreIdx) {
            taskDfxStatPos_[coreIdx] =
                taskDfxStatPos_[coreIdx] == REG_LOW_TASK_PING ? REG_LOW_TASK_PONG : REG_LOW_TASK_PING;
    }
    inline int GetDfxPos(int coreIdx) { return taskDfxStatPos_[coreIdx]; }

    // DFX
    inline void DfxProcAfterFinishTask(int coreIdx, uint64_t taskId) {
        (void)taskId;
        int pos = GetDfxPos(coreIdx);
        volatile TaskStat *stat = &args_[coreIdx]->taskStat[pos];
        (void)stat;

#if PROF_DFX_HOST_PREPARE_MEMORY_MODE != 1
        prof_.ProfGet(coreIdx, stat->subGraphId, stat->taskId, &args_[coreIdx]->taskStat[pos]);
#endif

#if defined(DEBUG_SWITCH) && DEBUG_SWITCH
        DumpTaskTensor(coreIdx, stat);
        recvFinTask_[coreIdx].push_back(TaskInfo(coreIdx, taskId));
#endif

#if PROF_DFX_HOST_PREPARE_MEMORY_MODE != 1
        SetNextDfxPos(coreIdx); // pingpong 存储
#endif
    }
public:
    uint64_t GetTaskStartTime() {
        return task_start_time_;
    }
    uint64_t GetTaskEndTime() {
        return task_end_time_;
    }
private:
    bool isFirstTaskSend_{true};
    bool firstLock[AICORE_TYPE_NUM]{true,true};
    uint64_t task_start_time_{UINT64_MAX};
    uint64_t task_end_time_{0};
    int aicNum_{0};
    int aivNum_{0};
    int aicValidNum_{0}; // 有效的aic，根据pgmask计算host传过来
    int aicpuIdx_{0};
    int aicStart_{0};
    int aicpuNum_{npu::tile_fwk::dynamic::MAX_SCHEDULE_AICPU_NUM};
    int aicEnd_{0};
    int aivStart_{0};
    int aivEnd_{0};
    uint64_t procAicCoreFunctionCnt_{0};
    uint64_t procAivCoreFunctionCnt_{0};
    uint64_t procAicpuFunctionCnt_{0};
    int64_t *regAddrs_{nullptr};
    int64_t sharedBuffer_{0};
    DeviceTask *curDevTask_{nullptr};
    DeviceTaskCtrl* curTaskCtrl_{nullptr};
    AicpuTaskManager &aicpuTaskManager_;

    std::array<uint64_t, MAX_AICORE_NUM> runningIds_;
    std::array<uint64_t, MAX_AICORE_NUM> pendingIds_;

    /* 低32位任务 存储的dfx状态信息， 乒乓存储 0 or 1,
       高32位任务 存储的dfx状态信息， 乒乓存储 2 or 3
    */
    std::array<int, MAX_AICORE_NUM> taskDfxStatPos_;

    std::array<int, MAX_AICORE_NUM> blockIdToPhyCoreId_;
    std::array<volatile uint64_t *, MAX_AICORE_NUM> readyRegQueues_;
    std::array<volatile uint64_t *, MAX_AICORE_NUM> finishRegQueues_;
    std::array<KernelArgs *, MAX_AICORE_NUM> args_;

    SPSCQueue<DeviceTaskCtrl *, DEFAULT_QUEUE_SIZE> taskQueue_;

    /* prepare aicore ready task list */
    StaticReadyCoreFunctionQueue *readyAicCoreFunctionQue_{nullptr};
    StaticReadyCoreFunctionQueue *readyAivCoreFunctionQue_{nullptr};

    AiCoreProf prof_;
    AicoreDump aicoreDump_;
    int64_t dotStatus_{0};
    uint64_t waitTaskCnt_[AICORE_TYPE_NUM]{0,0};
    uint32_t corePendReadyCnt_[AICORE_TYPE_NUM]{0,0};
    uint32_t coreRunReadyCnt_[AICORE_TYPE_NUM]{0,0};
    uint32_t runReadyCoreIdx_[AICORE_TYPE_NUM][MAX_MANAGER_AIV_NUM];
    uint32_t lastPendReadyCoreIdx_[AICORE_TYPE_NUM]{0,0};
    uint64_t reSolveHubCnt_{0};

    uint64_t readyIds[AICORE_TYPE_NUM][READY_ID_FIX_CACHE_NUM];
    uint64_t readyCount[AICORE_TYPE_NUM]{0,0};
    std::vector<uint64_t> readyIdsExtend[AICORE_TYPE_NUM];

    uint32_t sendCnt_[AICORE_TYPE_NUM]{0,0};

#if defined(DEBUG_SWITCH) && DEBUG_SWITCH
    std::vector<TaskInfo> sendTask_[MAX_AICORE_NUM];
    std::vector<TaskInfo> recvFinTask_[MAX_AICORE_NUM];
    std::vector<TaskInfo> recvAckTask_[MAX_AICORE_NUM];
#endif

    friend class AiCoreProf;
};
} // namespace npu::tile_fwk
