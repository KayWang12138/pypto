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
#include "securec.h"
#include "tilefwk/config.h"
#include "tilefwk/aicore_print.h"
#include "interface/utils/common.h"
#include "interface/operation/opcode.h"
#include "interface/schema/schema.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "machine/utils/dynamic/small_array.h"
#include "machine/utils/dynamic/spsc_queue.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/utils/device_log.h"
#include "machine/kernel/aicore.h"
#include "machine/device/distributed/comm_wait_flag.h"
#include "machine/device/aicore_dump.h"
#include "machine/device/dynamic/aicore_prof.h"
#include "machine/device/dynamic/aicore_hal.h"
#include "machine/device/dynamic/aicpu_task_manager.h"
#include "machine/device/dynamic/device_context.h"
#include "machine/device/dynamic/device_utils.h"

namespace npu::tile_fwk::dynamic {
const uint32_t REG_SPR_FAST_PATH_ENABLE = 0x18;
const uint64_t REG_SPR_FAST_PATH_OPEN = 0xE;
const uint64_t REG_SPR_FAST_PATH_CLOSE = 0xF;

const int INVALID_CORE_IDX = 0xFF;

const uint32_t AICORE_STATUS_INIT = 0xFFFFFFFFU;
const uint32_t AIV_NUM_PER_AI_CORE = 2;
const uint32_t READY_ID_FIX_CACHE_NUM = 2048;
const uint32_t AICORE_TYPE_NUM = 2;

constexpr uint32_t  MAX_MANAGER_AIV_NUM = (NAX_AIV_TOTAL_NUM / MAX_SCHEDULE_AICPU_NUM) + 1;

constexpr uint32_t REG_31_BITS = 0x7FFFFFFF;
constexpr uint32_t REG_32_BITS = 0xFFFFFFFF;
#define REG_LOW_TASK_ID(regVal) (regVal) & REG_31_BITS // 低31位存储的taskid
#define REG_LOW_TASK_STATE(regVal) ((regVal)&REG_32_BITS) >> 31 // 低32位存储的task的状态
constexpr uint32_t TASK_FIN_STATE = 1; // 任务执行完成完成
constexpr uint32_t TASK_ACK_STATE = 0; // 收到任务状态，没执行完成
constexpr uint32_t REG_TASK_NUM = 2; // 一次寄存器task个数

constexpr uint32_t DEFAULT_QUEUE_SIZE = 64;

struct TaskInfo {
    int coreIdx;
    uint64_t taskId;
    TaskInfo(int idx, uint64_t id) : coreIdx(idx), taskId(id) {}
};

typedef void (*FinishCallback)(DeviceTask *, void *);

struct sdma_l2_cmo_desc {
    unsigned long   src_addr;
    size_t          size;
    char            cmo_opcode;
};

const std::string SDMA_FILE = "/dev/sdma";

#define IOCTL_SDMA_L2_CMO  _IOW('s', 3, struct sdma_l2_cmo_desc)

struct DeviceTaskCtrl {
    int taskType{0};
    uint64_t taskId{0};
    DeviceTask *devTask{nullptr};
    uint64_t initAicFuncNum{0};
    uint64_t initAivFuncNum{0};
    uint64_t finishedAicFunctionCnt{0}; // 所有aicpu处理完成的aic function个数，多线程增加修改
    uint64_t finishedAivFunctionCnt{0}; // 所有aicpu处理完成的aiv function个数，多线程增加修改
    uint64_t finishedAicpuFunctionCnt{0}; // 所有aicpu处理完成的aicpu function个数，多线程增加修改
    uint64_t finishedHubFunctionCnt{0}; // 所有aicpu处理完成的hub function个数，多线程增加修改
    std::atomic<uint64_t> finishedFunctionCnt{0};
    std::atomic<bool> finishFlag{true};
    std::atomic<int> runcnt{0};
    void *ctx{nullptr};
    FinishCallback finish{nullptr};
    int retCode{0};
    std::array<std::array<std::atomic<bool>, MAX_SCHEDULE_AICPU_NUM>, AICORE_TYPE_NUM>  isAicpuIdle;

    inline bool IsFree() { return finishFlag.load(std::memory_order_acquire); }

    void PutTask(int ret) {
        if (ret != 0)
            retCode = ret;

        // sync point, ensure all aiore_manager threads task finished
        int cnt = runcnt.fetch_sub(1, std::memory_order_acq_rel);
        if (cnt == 1) {
            if (finish) {
                finish(devTask, ctx);
            }
            finishFlag.store(true, std::memory_order_release); // set finish
        } else {
            // wait finish
            while (!finishFlag.load(std::memory_order_acquire)) {}
        }
    }
};


inline void ReadyQueueLock(ReadyCoreFunctionQueue* rq) {
  while (!__sync_bool_compare_and_swap(&rq->lock, 0, 1)) {
  }
}

inline void ReadyQueueUnLock(ReadyCoreFunctionQueue* rq) {
  while (!__sync_bool_compare_and_swap(&rq->lock, 1, 0)) {
  }
}

inline void SdmaPrefetch(DeviceTask *devTask) {
    if (devTask == nullptr || devTask->l2Info.prefetchNum == 0) {
      return;
    }
    if (devTask->l2Info.prefetchNum > MAX_PREFETCH_NUM) {
      DEV_ERROR("Prefetch invalid num %ld.", devTask->l2Info.prefetchNum);
      return;
    }
    int fd = open(SDMA_FILE.c_str(), O_RDWR);
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
      DEV_DEBUG("Prefetch %lx %lu ret:%d.", devTask->l2Info.prefetchAddrs[i],
          devTask->l2Info.prefetchSizes[i], ret);
    }
    DEV_INFO("Prefetch tensor num %ld ret %d.", devTask->l2Info.prefetchNum, ret);
    close(fd);
    return;
}

class AiCoreManager {
public:
    explicit AiCoreManager(AicpuTaskManager &aicpuTaskManager) : aicpuTaskManager_(aicpuTaskManager), aicoreProf_(*this){};
    ~AiCoreManager(){};

    void InitLogger(AicoreLogger *logger) {
        logger_ = logger;
    }

    inline void InitTaskData(DeviceTaskCtrl *taskCtrl) {
        isFirstTaskSend_ = true;
        curTaskCtrl_ = taskCtrl;
        curDevTask_ = taskCtrl->devTask;
        curTaskType_ = taskCtrl->taskType;
        curTaskId_ = taskCtrl->taskId;
        aicoreHal_.SetModel(taskCtrl->devTask->aicoreModel);
        int64_t funcdata;
        auto dyntask = (DynDeviceTask *)curDevTask_;
        funcdata = static_cast<int64_t>(PtrToValue(dyntask->GetDynFuncDataList()));
        ForEachManageAicore([&](int coreIdx) {
            auto logbuf = logger_ ? logger_[coreIdx].GetBuffer() : nullptr;
            aicoreHal_.InitTaskData(coreIdx, funcdata, (uint64_t)logbuf);
        });

        readyAicCoreFunctionQue_ = reinterpret_cast<ReadyCoreFunctionQueue *>(curDevTask_->readyAicCoreFunctionQue);
        readyAivCoreFunctionQue_ = reinterpret_cast<ReadyCoreFunctionQueue *>(curDevTask_->readyAivCoreFunctionQue);
    }

    template <bool enableAicpuTask = false>
    inline uint32_t RunCoreTask(DeviceTaskCtrl *taskCtrl) {
        (void)taskCtrl;
        uint64_t sentAic = 0;
        uint64_t sentAiv = 0;
        DispatchAiCoreTask(CoreType::AIC, readyAicCoreFunctionQue_, aicStart_, aicEnd_);
        DispatchAiCoreTask(CoreType::AIV, readyAivCoreFunctionQue_, aivStart_, aivEnd_);
        sentAic += sendCnt_[static_cast<int>(CoreType::AIC)];
        sentAiv += sendCnt_[static_cast<int>(CoreType::AIV)];
        waitTaskCnt_[static_cast<int>(CoreType::AIC)] += sentAic;
        waitTaskCnt_[static_cast<int>(CoreType::AIV)] += sentAiv;
        sendCnt_[static_cast<int>(CoreType::AIC)] = 0;
        sendCnt_[static_cast<int>(CoreType::AIV)] = 0;

        uint64_t sent = 0UL;
        if constexpr (enableAicpuTask) {
            if (IsNeedProcAicpuTask()) {
                sent = ResolveDepForAicpuTask();
            }
        }

        DEV_IF_VERBOSE_DEBUG {
            __sync_fetch_and_add(&(taskCtrl->finishedAicFunctionCnt), sentAic);
            __sync_fetch_and_add(&(taskCtrl->finishedAivFunctionCnt), sentAiv);
            __sync_fetch_and_add(&(taskCtrl->finishedAicpuFunctionCnt), sent);
            __sync_fetch_and_add(&(taskCtrl->finishedHubFunctionCnt), resolveHubCnt_);
            procAicCoreFunctionCnt_ += sentAic;
            procAivCoreFunctionCnt_ += sentAiv;
            procAicpuFunctionCnt_ += sent;
            DEV_VERBOSE_DEBUG("finish send  aic task cnt: %lu,  aiv task cnt: %lu, hub task cnt:%lu,"
                "aicpu task cnt:%lu, target totalcnt: %lu.",
                taskCtrl->finishedAicFunctionCnt, taskCtrl->finishedAivFunctionCnt,
                taskCtrl->finishedHubFunctionCnt, taskCtrl->finishedAicpuFunctionCnt, curDevTask_->coreFunctionCnt);
        }
        sent += (sentAic + sentAiv + resolveHubCnt_);
        resolveHubCnt_ = 0;
        return sent;
    }

    void DumpAicoreLog(int coreIdx) {
        const int bufSize = 512;
        char buf[bufSize];
        while (logger_[coreIdx].Read(buf, bufSize)) {
            DEV_INFO("core-%d %s", coreIdx, buf);
        }
    }

    inline int RunTask(DeviceTaskCtrl *taskCtrl) {
        int rc, ret = DEVICE_MACHINE_OK;
        seq = taskCtrl->taskId;
        DEV_INFO("receive new task %lu.", taskCtrl->taskId);

        InitTaskData(taskCtrl);
        uint32_t curSent = RunCoreTask(taskCtrl);
        taskCtrl->finishedFunctionCnt.fetch_add(curSent, std::memory_order_relaxed);

        if (IsNeedProcAicpuTask()) {
            aicpuTaskManager_.Init(reinterpret_cast<DynDeviceTask *>(curDevTask_));
        }

        uint64_t start = GetCycles();
        uint32_t lastSent = 0;
        while (taskCtrl->finishedFunctionCnt.load(std::memory_order_relaxed) < curDevTask_->coreFunctionCnt) {
            curSent = RunCoreTask<true>(taskCtrl);
            if (likely(curSent == 0)) {
                if (lastSent > 0) {
                    taskCtrl->finishedFunctionCnt.fetch_add(lastSent, std::memory_order_relaxed);
                    lastSent = 0;
                }
            } else {
                lastSent += curSent;
            }

            if (GetCycles() - start > TIMEOUT_CYCLES) {
                ret = DEVICE_MACHINE_TIMEOUT_CORETASK;
                goto FINISH;
            }
            (void)start;
        }
        PerfMtBegin(PERF_EVT_WAIT_AICORE_FINISH, aicpuIdx_);
        rc = WaitAllAicoreFinish(aicStart_, aicEnd_, DEVICE_MACHINE_TIMEOUT_AIC);
        if (rc != DEVICE_MACHINE_OK) {
            ret = rc;
        }
        rc = WaitAllAicoreFinish(aivStart_, aivEnd_, DEVICE_MACHINE_TIMEOUT_AIV);
        if (rc != DEVICE_MACHINE_OK) {
            ret = rc;
        }
        if (IsNeedProcAicpuTask()) {
            while (!aicpuTaskManager_.Finished()) {
                (void)aicpuTaskManager_.TaskProcess();
            }
        }
        PerfMtEnd(PERF_EVT_WAIT_AICORE_FINISH, aicpuIdx_);
        DEV_DEBUG("Aicpu %d proc finish send all task,aic: %lu, aiv: %lu, aicpu: %lu.",
            aicpuIdx_, procAicCoreFunctionCnt_, procAivCoreFunctionCnt_, procAicpuFunctionCnt_);
    FINISH:
        if (ret != DEVICE_MACHINE_OK) {
            DEV_ERROR("Aicpu %d proc finish %lu %lu %lu, but timeout !.", aicpuIdx_,
                taskCtrl->finishedFunctionCnt.load(), curDevTask_->coreFunctionCnt, taskCtrl->taskId);
            DumpAiCoreStatus();
        }
        return ret;
    }

    inline void DumpLastWord(int coreIdx) {
        uint64_t status = aicoreHal_.GetAicoreStatus(coreIdx);
        if (pendingIds_[coreIdx] != AICORE_TASK_INIT) {
            DEV_ERROR("status %lu,pending taskid: %s,funcdata:  %s.", status, std::to_string(pendingIds_[coreIdx]).c_str(),
            ((DynDeviceTask *)curDevTask_)->DumpTaskData(pendingIds_[coreIdx]).c_str());
        }
        if (runningIds_[coreIdx] != AICORE_TASK_INIT) {
            DEV_ERROR("status %lu,running taskid:%s,funcdata:  %s.", status, std::to_string(runningIds_[coreIdx]).c_str(),
                ((DynDeviceTask *)curDevTask_)->DumpTaskData(runningIds_[coreIdx]).c_str());
        }
    }

    void ResetRegAll() {
      ForEachManageAicore([this](int coreIdx) {
          if (aicoreHal_.ReadReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE) == REG_SPR_FAST_PATH_OPEN) {
            aicoreHal_.WriteReg32(coreIdx, REG_SPR_DATA_MAIN_BASE, AICORE_TASK_STOP + 1);
            aicoreHal_.WriteReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
          }
      });
    }

    inline int Run(int threadIdx, DeviceArgs *deviceArgs, DeviceTaskCtrl *taskCtrl = nullptr) {
        int ret = 0;
        DEV_DEBUG("schedule run: %p", taskCtrl);
        Init(threadIdx, deviceArgs);
        PerfMtTrace(PERF_TRACE_INIT, threadIdx);
        DEV_DEBUG("schedule run init succ");
        if constexpr (IsDeviceMode()) {
            ret = HandShake();
            PerfMtTrace(PERF_TRACE_CORE_HAND_SHAKE, threadIdx);
            if (ret != DEVICE_MACHINE_OK) {
                DEV_ERROR("hand shake timeout.");
                AbnormalStop();
                do {
                    taskCtrl->PutTask(ret);
                } while ((taskCtrl = taskQueue_.Dequeue()));
                return ret;
            }
            aicoreProf_.ProfStart();
        }
        DEV_DEBUG("schedule run start succ");
        if (taskCtrl != nullptr) {
            ret = RunTask(taskCtrl);
        } else {
            uint64_t lastDevTaskFinCycle = 0;
            while (ret == 0) {
                DEV_DEBUG("schedule task wait");
                taskCtrl = taskQueue_.Dequeue();
                DEV_DEBUG("schedule task recv");
                if (taskCtrl == nullptr) {
                    PerfMtTrace(PERF_TRACE_WAIT_ALL_DEV_TASK_FINISH, aicpuIdx_, lastDevTaskFinCycle);
                    break;
                }
                PerfMtTrace(PERF_TRACE_DEV_TASK_RCV, aicpuIdx_);
                PROF_STAGE_BEGIN_MTSAFE(PERF_EVT_STAGE_SCHEDULE, threadIdx, "dispatch.before\n");
                PerfMtBegin(PERF_EVT_RUN_TASK, threadIdx);
                ret = RunTask(taskCtrl);
                lastDevTaskFinCycle = GetCycles();
                PerfMtTrace(PERF_TRACE_DEV_TASK_SCHED_EXEC, aicpuIdx_, lastDevTaskFinCycle);
                PerfMtEnd(PERF_EVT_RUN_TASK, threadIdx);
                DEV_DEBUG("run task finish taskid=%d ret %d.", curTaskId_, ret);
                if (ret != 0)
                    break;

                PerfMtBegin(PERF_EVT_SYNC_AICORE, threadIdx);
                SyncAiCore(taskCtrl->taskId);
                PerfMtTrace(PERF_TRACE_DEV_TASK_SYNC_CORE_STOP, aicpuIdx_);
                PerfMtEnd(PERF_EVT_SYNC_AICORE, threadIdx);
                DEV_DEBUG("sync finish.");
                taskCtrl->PutTask(ret);
                PerfMtTrace(PERF_TRACE_DEV_TASK_RSP, threadIdx);
                PROF_STAGE_END_MTSAFE(PERF_EVT_STAGE_SCHEDULE, threadIdx, "dispatch.after\n");
            }
            if (ret) {
                DEV_ERROR("task %lu execute error %d, skip rest tasks.", taskCtrl->taskId, ret);
                if (IsDeviceMode()) {
                    ForEachManageAicore([&](int coreIdx) {
                        DumpLastWord(coreIdx);
                    });
                }
                do {
                    taskCtrl->PutTask(ret);
                } while ((taskCtrl = taskQueue_.Dequeue()));
            }
        }

        if constexpr (IsDeviceMode()) {
            NormalStop();
            PerfMtTrace(PERF_TRACE_WAIT_CORE_EXIT, aicpuIdx_);
            ProfStop();
        }
        DEV_INFO("Aicpu %d stop ret = %d, proc aic task cnt: %lu,  aiv task cnt: %lu.",
            aicpuIdx_,
            ret,
            procAicCoreFunctionCnt_,
            procAivCoreFunctionCnt_);
        return ret;
    }

    void PushTask(DeviceTaskCtrl *taskCtrl) { taskQueue_.Enqueue(taskCtrl); }

    inline void DumpAicorePerfTrace(std::ostringstream& oss) {
        (void)oss;
#if ENABLE_PERF_TRACE
        for (int i = aicStart_; i < aicEnd_; ++i) {
            int ret = aicoreHal_.DumpAicorePerfTrace(aicpuIdx_, i, CoreType::AIC, oss);
            if (ret == DEVICE_MACHINE_OK) {
                oss << ",";
            }
        }
        for (int i = aivStart_; i < aivEnd_; ++i) {
            int ret = aicoreHal_.DumpAicorePerfTrace(aicpuIdx_, i, CoreType::AIV, oss);
            if (ret == DEVICE_MACHINE_OK) {
                oss << ((i == aivEnd_ - 1) ? "" : ",");
            }
        }
#endif
    }
private:
    inline void DumpTaskProf() {
        ForEachManageAicoreWithRet([this] (int coreIdx) -> int { return aicoreHal_.DumpTaskProf(coreIdx);});
    }

    inline void ProfStop() {
        if (aicoreProf_.ProfIsEnable()) {
#if PROF_DFX_HOST_PREPARE_MEMORY_MODE
            DumpTaskProf();
#endif
        }

        aicoreProf_.ProfStop();
    }

    inline void DumpAiCoreStatus() const {
        DEV_IF_VERBOSE_DEBUG {
            ForEachManageAicore([this](int coreIdx) {
                if constexpr (IsDeviceMode()) {
                    aicoreHal_.DumpAicoreStatus(coreIdx);
                }
                DEV_VERBOSE_DEBUG("reg low task: runningid(%u) pendingid(%u) dfxpos(%d).", runningIds_[coreIdx],
                    pendingIds_[coreIdx], taskDfxStatPos_[coreIdx]);

                DEV_VERBOSE_DEBUG("send task info ~~~~~~~~~~~~~~~~~~~~~~~~~~~~count:%lu~~~~~~~~~~~~~~~~~~~~~~~~~~~~.",
                    sendTask_[coreIdx].size());
                for (size_t i = 0; i < sendTask_[coreIdx].size(); i++) {
                    DEV_VERBOSE_DEBUG("send task: seqno %d, taskId %lx.", (int)i, sendTask_[coreIdx][i].taskId);
                }

                DEV_VERBOSE_DEBUG("recv finish task info ~~~~~~~~~~~~~~~~~~~~~~~~~count:%lu~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~.",
                    recvFinTask_[coreIdx].size());
                for (size_t i = 0; i < recvFinTask_[coreIdx].size(); i++) {
                    DEV_VERBOSE_DEBUG("recv task: seqno %d, taskId %lx.", (int)i, recvFinTask_[coreIdx][i].taskId);
                }

                DEV_VERBOSE_DEBUG("recv ack task info ~~~~~~~~~~~~~~~~~~~~~~~~~~~~count:%lu~~~~~~~~~~~~~~~~~~~~~~~~~~~~.",
                    recvAckTask_[coreIdx].size());
                for (size_t i = 0; i < recvAckTask_[coreIdx].size(); i++) {
                    DEV_VERBOSE_DEBUG("recv ack task: seqno %d, taskId %lx.", static_cast<int>(i), recvAckTask_[coreIdx][i].taskId);
                }
            });
        }
    }

    inline bool CheckTaskFinished(int coreIdx) {
        uint64_t finTaskVal = aicoreHal_.GetFinishedTask(coreIdx);
        uint32_t regLFinTaskId = REG_LOW_TASK_ID(finTaskVal);
        uint32_t regLFinTaskState = REG_LOW_TASK_STATE(finTaskVal);

        int type =static_cast<int>(AicoreType(coreIdx));
        if (regLFinTaskState == TASK_FIN_STATE &&
            (pendingIds_[coreIdx] == regLFinTaskId || runningIds_[coreIdx] == regLFinTaskId)) {
            if (pendingIds_[coreIdx] == regLFinTaskId) {
                runReadyCoreIdx_[type][coreRunReadyCnt_[type]++] = coreIdx;
                corePendReadyCnt_[type]++;
            }

            if (runningIds_[coreIdx] == regLFinTaskId) {
                runReadyCoreIdx_[type][coreRunReadyCnt_[type]++] = coreIdx;
            }
            DfxProcAfterFinishTask(coreIdx, regLFinTaskId);
            pendingIds_[coreIdx] = AICORE_TASK_INIT;
            pendingResolveIndexList_[coreIdx] = 0;
            runningIds_[coreIdx] = AICORE_TASK_INIT;
            runningResolveIndexList_[coreIdx] = 0;
        }

        return pendingIds_[coreIdx] == AICORE_TASK_INIT && runningIds_[coreIdx] == AICORE_TASK_INIT;
    }

    inline int WaitAllAicoreFinish(int coreIdxStart, int coreIdxEnd, int errorCode) {
        int stopSent = 0;
        bool coreStopped[MAX_MANAGER_AIV_NUM] = {false};
        int64_t start_cycles = GetCycles();

        while (stopSent < coreIdxEnd - coreIdxStart) {
            for (int i = coreIdxStart; i < coreIdxEnd; i++) {
                if (coreStopped[i - coreIdxStart]) {
                    continue;
                }
                if (CheckTaskFinished(i)) {
                    stopSent++;
                    coreStopped[i - coreIdxStart] = true;
                } else if (GetCycles() - start_cycles > TIMEOUT_CYCLES) {
                    DEV_ERROR("wait tail task finish timeout coreindx=%d.", i);
                    return errorCode;
                }
            }
        }
        return DEVICE_MACHINE_OK;
    }

    inline uint32_t GetReadyCoreNum(CoreType type) {
        if (enableFairSch_ && IsExistOtherAicpuIdle(type))  {
            return coreRunReadyCnt_[static_cast<int>(type)];
        }
        return corePendReadyCnt_[static_cast<int>(type)];
    }

    inline uint64_t TryBatchSendTask(CoreType type, ReadyCoreFunctionQueue* readyQue,
                int coreIdxStart, int coreIdxEnd) {
        if (__atomic_load_n(&readyQue->tail, __ATOMIC_RELAXED) == __atomic_load_n(&readyQue->head, __ATOMIC_RELAXED)) {
            DEV_VERBOSE_DEBUG("AiCpud:%d, can not send task currently. ready Task: 0.", aicpuIdx_);
            return 0;
        }

        uint32_t ready = GetReadyCoreNum(type);
        if (ready == 0 ) {
            DEV_VERBOSE_DEBUG("AiCpud:%d, can not send task currently. ready Core: %u.", aicpuIdx_, ready);
            return 0;
        }
        PerfMtBegin(PERF_EVT_SEND_AIC_TASK, aicpuIdx_);
        uint32_t readyId[MAX_MANAGER_AIV_NUM];
        ReadyQueueLock(readyQue);
        uint32_t head = __atomic_load_n(&readyQue->head, __ATOMIC_RELAXED);
        uint32_t tail = __atomic_load_n(&readyQue->tail, __ATOMIC_RELAXED);
        uint32_t taskCount = std::min(ready, tail - head);
        if (taskCount == 0) {
            DEV_VERBOSE_DEBUG("AiCpud:%u, taskCount is zero.", head);
            ReadyQueueUnLock(readyQue);
            PerfMtEnd(PERF_EVT_SEND_AIC_TASK, aicpuIdx_);
            return 0;
        }
        bool isRealLifo = (enableL2CacheSch_ && !firstLock[static_cast<int>(type)]);
        if (isRealLifo) {
            memcpy_s(readyId, taskCount * sizeof(uint64_t),
                reinterpret_cast<uint8_t *>(&readyQue->elem[tail - taskCount]), taskCount * sizeof(uint32_t));
            __atomic_fetch_sub(&readyQue->tail, taskCount, std::memory_order_release);
        } else {
            __atomic_fetch_add(&readyQue->head, taskCount, std::memory_order_release);
        }
        ReadyQueueUnLock((readyQue));
        DEV_VERBOSE_DEBUG("AiCpud:%d, pop all new task count: %u.", aicpuIdx_, taskCount);
        BatchSendTask(type, isRealLifo ? &readyId[taskCount - 1] : &readyQue->elem[head],
            taskCount, coreIdxStart, coreIdxEnd, isRealLifo);
        DEV_VERBOSE_DEBUG("core ready cnt: %u.", corePendReadyCnt_[static_cast<int>(type)]);
        firstLock[static_cast<int>(type)] = false;
        PerfMtEnd(PERF_EVT_SEND_AIC_TASK, aicpuIdx_);
        return taskCount;
    }

    inline uint32_t BatchSendTask(CoreType type, uint32_t *newTask, uint32_t taskCount,
        int coreIdxStart, int coreIdxEnd, bool isLifo) {
        uint32_t sendCnt = 0;
        uint32_t coreRunReadyCnt = coreRunReadyCnt_[static_cast<int>(type)];
        DEV_VERBOSE_DEBUG("Begin Batch send %s task: corerunreadycnt:%u, pendreadyCnt:%u, taskCount:%u.",
            type == CoreType::AIC ? "AIC": "AIV", coreRunReadyCnt,
            corePendReadyCnt_[static_cast<int>(type)], taskCount);
        while (sendCnt < static_cast<uint64_t>(coreRunReadyCnt) && sendCnt < taskCount) {
            DEV_VERBOSE_DEBUG("  ## send task use runready core %u.",
                runReadyCoreIdx_[static_cast<int>(type)][coreRunReadyCnt_[static_cast<int>(type)] - 1]);
            SendTaskToAiCore(type,
                runReadyCoreIdx_[static_cast<int>(type)][--coreRunReadyCnt_[static_cast<int>(type)]],
                isLifo ? *newTask-- : *newTask++);
            sendCnt++;
        }
        corePendReadyCnt_[static_cast<int>(type)] -= sendCnt;

        uint32_t idx = lastPendReadyCoreIdx_[static_cast<int>(type)];
        uint32_t coreNum = coreIdxEnd - coreIdxStart;
        uint32_t lastProcCore = idx;
        DEV_VERBOSE_DEBUG("  ## send task left pend ready cnt %u , last core index:%u.",
            corePendReadyCnt_[static_cast<int>(type)], idx);
        while (corePendReadyCnt_[static_cast<int>(type)] > 0 && sendCnt < taskCount) {
            if (pendingIds_[idx] == AICORE_TASK_INIT) {
                DEV_VERBOSE_DEBUG("  ## send task use pendready core %u.", idx);
                SendTaskToAiCore(type, idx, isLifo ? *newTask-- : *newTask++);
                sendCnt++;
                corePendReadyCnt_[static_cast<int>(type)]--;
                lastProcCore = idx;
            }
            idx = coreIdxStart + (idx - coreIdxStart + 1) % coreNum;
        }

        if (lastProcCore != lastPendReadyCoreIdx_[static_cast<int>(type)]) {
            lastPendReadyCoreIdx_[static_cast<int>(type)] = coreIdxStart + (lastProcCore - coreIdxStart + 1) % coreNum;
        }
        DEV_VERBOSE_DEBUG("  ## finish send task left runreadycnt:%u pendreadycnt %u, last coreindex:%u.",
            coreRunReadyCnt_[static_cast<int>(type)], corePendReadyCnt_[static_cast<int>(type)], idx);
        return sendCnt;
    }

    inline uint64_t DispatchAiCoreTask(CoreType type, ReadyCoreFunctionQueue* readyQue,
                                       int coreIdxStart, int coreIdxEnd) {
        if (waitTaskCnt_[static_cast<int>(type)] > 0) {
            ResolveDepForAllAiCore(type, coreIdxStart, coreIdxEnd);
        }
        uint64_t taskCount = TryBatchSendTask(type, readyQue, coreIdxStart, coreIdxEnd);
        if (enableFairSch_) {
            if (coreRunReadyCnt_[static_cast<int>(type)] > 0)  {
                AicpuIsIdle(type);
            } else {
                AicpuIsBusy(type);
            }
        }
        return taskCount;
    }

    inline void SendTaskToAiCore(CoreType type, int coreIdx, uint64_t newTask) {
        DEV_TRACE_DEBUG(LEvent(
            LUid(curTaskCtrl_->taskId, FuncID(newTask), GetRootIndex(newTask), TaskID(newTask), GetLeafIndex(newTask)),
            LActStart(coreIdx)));
        aicoreHal_.SetReadyQueue(coreIdx, (newTask + 1) & 0xFFFFFFFF);
        pendingIds_[coreIdx] = newTask;
        pendingResolveIndexList_[coreIdx] = 0;
        sendCnt_[static_cast<int>(type)]++;

        if (isFirstTaskSend_) {
            PerfMtTrace(PERF_TRACE_DEV_TASK_SEND_FIRST_CALLOP_TASK, aicpuIdx_);
            isFirstTaskSend_ = false;
        }

        DEV_IF_VERBOSE_DEBUG {
            sendTask_[coreIdx].push_back(TaskInfo(coreIdx, newTask));
        }
        DEV_VERBOSE_DEBUG("Send task %lu, at core %d ,type:%d.", newTask, coreIdx, static_cast<int>(type));
    }

    inline void SetAiCpuStat(int coreIdx, uint64_t taskId) {
        struct AiCpuTaskStat aiCpuTaskStat;
        aiCpuTaskStat.taskId = taskId;
        aiCpuTaskStat.coreId = aicoreHal_.GetPhyIdByBlockId(coreIdx);
        aicoreProf_.AsmCntvc(aiCpuTaskStat.taskGetStart);
        aicoreProf_.SetAiCpuTaskStat(taskId, aiCpuTaskStat);
    };

    inline void PushReadyQue(ReadyCoreFunctionQueue *readyQue, void *idList, uint32_t idCnt) const {
        ReadyQueueLock(readyQue);
        memcpy_s(
            &readyQue->elem[readyQue->tail], idCnt * sizeof(uint32_t), (uint8_t *)idList, idCnt * sizeof(uint32_t));
         __atomic_fetch_add(&readyQue->tail, idCnt, std::memory_order_release);
        DEV_IF_NONDEVICE {
            DEV_ASSERT(readyQue->tail <= readyQue->capacity);
        }
        ReadyQueueUnLock(readyQue);
    }

    inline void ResolveDepForAllAiCore(CoreType type, int coreIdxStart, int coreIdxEnd) {
        PerfMtBegin(static_cast<int>(PERF_EVT_RESOLVE_DEPENDENCE), aicpuIdx_);
        for (int i = coreIdxStart; i < coreIdxEnd; i++) {
            if ((runningIds_[i] != AICORE_TASK_INIT || pendingIds_[i] != AICORE_TASK_INIT)) {
                ResolveByRegVal(type, i);
                if (enableFairSch_) {
                    if (readyAicCoreFunctionQue_->tail - readyAicCoreFunctionQue_->head == 0 ||
                        readyAivCoreFunctionQue_->tail - readyAivCoreFunctionQue_->head == 0) {
                        BatchPushReadyQueue();
                    }
                }
            }
        }

        BatchPushReadyQueue();
        PerfMtEnd(static_cast<int>(PERF_EVT_RESOLVE_DEPENDENCE), aicpuIdx_);
        return;
    }

    inline void BatchPushReadyQueue() {
        uint32_t aicIndex = static_cast<uint32_t>(CoreType::AIC);
        uint32_t aivIndex = static_cast<uint32_t>(CoreType::AIV);
        if (readyCount[aicIndex] > 0) {
            uint32_t needSendCnt = std::min(GetReadyCoreNum(CoreType::AIC), readyCount[aicIndex]);
            if (needSendCnt > 0) {
                readyCount[aicIndex] -= BatchSendTask(CoreType::AIC, &readyIds[aicIndex][readyCount[aicIndex] - 1],
                    needSendCnt, aicStart_, aicEnd_, true);
            }
            DEV_VERBOSE_DEBUG("resolved new task, aic ready count: %u coretype:%u.", readyCount[aicIndex], aicIndex);
            if (readyCount[aicIndex] > 0) {
                PushReadyQue(readyAicCoreFunctionQue_, readyIds[aicIndex], readyCount[aicIndex]);
            }
            readyCount[aicIndex] = 0;
        }

        if (readyCount[aivIndex] > 0) {
            uint32_t needSendCnt = std::min(GetReadyCoreNum(CoreType::AIV), readyCount[aivIndex]);
            if (needSendCnt > 0) {
                readyCount[aivIndex] -= BatchSendTask(CoreType::AIV, &readyIds[aivIndex][readyCount[aivIndex] - 1],
                    needSendCnt, aivStart_, aivEnd_, true);
            }
            DEV_VERBOSE_DEBUG("resolved new task, aiv ready count: %u coretype: %u.", readyCount[aivIndex], aivIndex);
            if (readyCount[aivIndex] > 0) {
                PushReadyQue(readyAivCoreFunctionQue_, readyIds[aivIndex], readyCount[aivIndex]);
            }
            readyCount[aivIndex] = 0;
        }
    }

    inline uint64_t ResolveDepForAicpuTask() {
        uint64_t taskCount = aicpuTaskManager_.TaskProcess();
        std::vector<uint64_t> completed = aicpuTaskManager_.TaskPoll();
        for (const uint64_t &taskId : completed) {
            ResolveDepDyn(taskId);
            BatchPushReadyQueue();
        }
        return taskCount;
    }

    inline void ResolveWhenSyncMode(CoreType type, uint32_t finTaskId, uint32_t finTaskState, int coreIdx)  {
        if (finTaskId == pendingIds_[coreIdx] && finTaskState == TASK_FIN_STATE) {
            DEV_VERBOSE_DEBUG("core index: %d, PendingTask Finished."
                " pending: %x.", coreIdx, pendingIds_[coreIdx]);
            ResolveDepWithDfx(type, coreIdx, finTaskId);
            pendingIds_[coreIdx] = AICORE_TASK_INIT;
            pendingResolveIndexList_[coreIdx] = 0;
            corePendReadyCnt_[static_cast<int>(type)]++;
            runReadyCoreIdx_[static_cast<int>(type)][coreRunReadyCnt_[static_cast<int>(type)]++] = coreIdx;
        }
    }

    static uint64_t RuntimeCopyOutResolveCounterDecode(uint64_t aicpuCallCode) {
        return aicpuCallCode & 0xffff;
    }

    inline void ResolveByRegVal(CoreType type, int coreIdx) {
        uint64_t finTaskRegVal = aicoreHal_.GetFinishedTask(coreIdx);
        uint32_t aicpuCallCode = finTaskRegVal >> 32;
        uint32_t finTaskId = REG_LOW_TASK_ID(finTaskRegVal);
        uint32_t finTaskState = REG_LOW_TASK_STATE(finTaskRegVal);
        DEV_VERBOSE_DEBUG("reslove task core index: %d, finishtaskid:%x, finishstate: %u.", coreIdx, finTaskId, finTaskState);
#if SCHEDULE_USE_PENDING_AND_RUNING_SWITCH
        auto &pendingIdRef = pendingIds_[coreIdx];
        auto &pendingResolveIndexBaseRef = pendingResolveIndexList_[coreIdx];
        auto &runningIdRef = runningIds_[coreIdx];
        auto &runningResolveIndexBaseRef = runningResolveIndexList_[coreIdx];
        if (likely(finTaskId == pendingIdRef && finTaskState == TASK_FIN_STATE)) {
            // pending task is finished, resolve both running and pending task.
            DEV_VERBOSE_DEBUG("Pending Finished: core:%d pending:%x,%d running:%x,%d", coreIdx, pendingIdRef, pendingResolveIndexBaseRef, runningIdRef, runningResolveIndexBaseRef);
            uint32_t runningIdValue = runningIdRef;
            int runningResolveIndexBaseValue = runningResolveIndexBaseRef;
            uint32_t pendingIdValue = pendingIdRef;
            int pendingResolveIndexBaseValue = pendingResolveIndexBaseRef;
            runningIdRef = AICORE_TASK_INIT;
            runningResolveIndexBaseRef = 0;
            pendingIdRef = AICORE_TASK_INIT; // ResolveDepWithDfx depend this line
            pendingResolveIndexBaseRef = 0;
            runReadyCoreIdx_[static_cast<int>(type)][coreRunReadyCnt_[static_cast<int>(type)]++] = coreIdx;
            corePendReadyCnt_[static_cast<int>(type)]++;
            if (runningIdValue != AICORE_TASK_INIT) {
                ResolveDepWithDfx(type, coreIdx, runningIdValue, runningResolveIndexBaseValue);
            }
            ResolveDepWithDfx(type, coreIdx, pendingIdValue, pendingResolveIndexBaseValue);
        } else if (unlikely(finTaskId == pendingIdRef && aicpuCallCode != 0)) {
            // pending task is copyout, reolve both running and pending task.
            DEV_VERBOSE_DEBUG("Pending Copyout: core:%d pending:%x,%d running:%x,%d", coreIdx, pendingIdRef, pendingResolveIndexBaseRef, runningIdRef, runningResolveIndexBaseRef);
            uint32_t copyOutResolveCounter = RuntimeCopyOutResolveCounterDecode(aicpuCallCode);
            uint32_t runningIdValue = runningIdRef;
            int runningResolveIndexBaseValue = runningResolveIndexBaseRef;
            uint32_t pendingIdValue = pendingIdRef;
            int pendingResolveIndexBaseValue = pendingResolveIndexBaseRef;
            runningIdRef = pendingIdRef;
            runningResolveIndexBaseRef = copyOutResolveCounter + 1;
            pendingIdRef = AICORE_TASK_INIT; // ResolveDepWithDfx depend this line
            pendingResolveIndexBaseRef = 0;
            corePendReadyCnt_[static_cast<int>(type)]++;
            if (runningIdValue != AICORE_TASK_INIT) {
                ResolveDepWithDfx(type, coreIdx, runningIdValue, runningResolveIndexBaseValue);
            }
            ResolveCopyOutDepDyn(copyOutResolveCounter, pendingIdValue, pendingResolveIndexBaseValue);

        } else if (finTaskId == pendingIdRef && finTaskState == TASK_ACK_STATE) {
            // pending task is acknowledged, resolve running task. And move pending to running
            DEV_VERBOSE_DEBUG("Pending Acknowledged: core:%d pending:%x,%d running:%x,%d", coreIdx, pendingIdRef, pendingResolveIndexBaseRef, runningIdRef, runningResolveIndexBaseRef);
            DEV_IF_VERBOSE_DEBUG {
                recvAckTask_[coreIdx].push_back(TaskInfo(coreIdx, finTaskId));
            }
            uint32_t runningIdValue = runningIdRef;
            int runningResolveIndexBaseValue = runningResolveIndexBaseRef;
            runningIdRef = finTaskId;
            runningResolveIndexBaseRef = pendingResolveIndexBaseRef;
            pendingIdRef = AICORE_TASK_INIT; // ResolveDepWithDfx depend this line
            pendingResolveIndexBaseRef = 0;
            corePendReadyCnt_[static_cast<int>(type)]++;
            if (runningIdValue != AICORE_TASK_INIT) {
                ResolveDepWithDfx(type, coreIdx, runningIdValue, runningResolveIndexBaseValue);
            }
        } else if (finTaskId == runningIdRef && finTaskState == TASK_FIN_STATE) {
            // running task is finished, resolve running task. Pending task is unmodified
            DEV_VERBOSE_DEBUG("Running finished: core:%d pending:%x,%d running:%x,%d", coreIdx, pendingIdRef, pendingResolveIndexBaseRef, runningIdRef, runningResolveIndexBaseRef);
            uint32_t runningIdValue = runningIdRef;
            int runningResolveIndexBaseValue = runningResolveIndexBaseRef;
            runningIdRef = AICORE_TASK_INIT;
            runningResolveIndexBaseRef = 0;
            if (pendingIdRef == AICORE_TASK_INIT) {
                runReadyCoreIdx_[static_cast<int>(type)][coreRunReadyCnt_[static_cast<int>(type)]++] = coreIdx;
            }
            ResolveDepWithDfx(type, coreIdx, runningIdValue, runningResolveIndexBaseValue);
        } else if (unlikely(finTaskId == runningIdRef && aicpuCallCode != 0)) {
            // running task is copyout, resolve running task. Pending task is unmodified
            DEV_VERBOSE_DEBUG("Running copyout: core:%d pending:%x,%d running:%x,%d", coreIdx, pendingIdRef, pendingResolveIndexBaseRef, runningIdRef, runningResolveIndexBaseRef);
            uint32_t copyOutResolveCounter = RuntimeCopyOutResolveCounterDecode(aicpuCallCode);
            uint32_t runningIdValue = runningIdRef;
            int runningResolveIndexBaseValue = runningResolveIndexBaseRef;
            runningResolveIndexBaseRef = copyOutResolveCounter + 1;
            ResolveCopyOutDepDyn(copyOutResolveCounter, runningIdValue, runningResolveIndexBaseValue);
        } else {
            DEV_VERBOSE_DEBUG("Warning, maybe inconsistent state. coreidx: %d,finTask: %lx,pending: %x,running: %x.", coreIdx, finTaskRegVal, pendingIdRef, runningIdRef);
        }
#else
        ResolveWhenSyncMode(type, finTaskId, finTaskState, coreIdx);
#endif
    }

    inline void PushAicpuTaskQueue(uint64_t taskId) {
        aicpuTaskManager_.TaskEnqueue(taskId);
    }

    inline bool TrySendTaskDirectly(int coreType, uint32_t taskId) {
        if (coreRunReadyCnt_[coreType] > 0) {
            corePendReadyCnt_[coreType]--;
            DEV_VERBOSE_DEBUG("Direct send task when task ready %x.", taskId);
            SendTaskToAiCore(static_cast<CoreType>(coreType),
                runReadyCoreIdx_[coreType][--coreRunReadyCnt_[coreType]], taskId);
            return true;
        }

        if (corePendReadyCnt_[coreType] == 0) {
            return false;
        }

        if (enableFairSch_ && IsExistOtherAicpuIdle(static_cast<CoreType>(coreType))) {
            return false;
        }

        int startIdx;
        int coreNum;
        int idx = static_cast<int>(lastPendReadyCoreIdx_[coreType]);
        if (coreType == static_cast<int>(CoreType::AIC)) {
            startIdx = aicStart_;
            coreNum = aicEnd_ - aicStart_;
        } else {
            startIdx = aivStart_;
            coreNum = aivEnd_ - aivStart_;
        }
        while (pendingIds_[idx] != AICORE_TASK_INIT) {
            idx = startIdx + (idx - startIdx + 1) % (coreNum);
        }
        lastPendReadyCoreIdx_[coreType] = static_cast<uint32_t>(startIdx + (idx - startIdx + 1) % (coreNum));
        corePendReadyCnt_[coreType]--;
        DEV_VERBOSE_DEBUG("Direct send task when task ready %x.", taskId);
        SendTaskToAiCore(static_cast<CoreType>(coreType), idx, taskId);
        return true;
    }

    inline void PushReadyTask(int coreType, uint64_t taskId) {
        if (enableL2CacheSch_ && TrySendTaskDirectly(coreType, taskId)) {
            return;
        }

        if (unlikely(readyCount[coreType] == READY_ID_FIX_CACHE_NUM)) {
            ReadyCoreFunctionQueue* readyQue =
                coreType == static_cast<int>(CoreType::AIC) ?  readyAicCoreFunctionQue_ : readyAivCoreFunctionQue_;
            PushReadyQue(readyQue, readyIds[coreType], readyCount[coreType]);
            readyCount[coreType] = 0;
        }
        readyIds[coreType][readyCount[coreType]++] = taskId;
    }

    inline uint64_t GetCostModelTaskTime(uint64_t coreIdx, uint64_t taskId, uint64_t currentTime) {
        auto funcId = FuncID(taskId);
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto costModelData = reinterpret_cast<CostModel::ModelData*>(curDevTask_->costModelData);
        if (costModelData == nullptr) return 0;
        auto source = dyntask->GetDynFuncDataCacheList()[funcId].devFunc;
        auto opIndex = TaskID(taskId);
        auto leafFunctionIdx = source->GetOperationAttrCalleeIndex(opIndex);
        auto timeCost = costModelData->functionTime[leafFunctionIdx];
        auto header = dyntask->GetDynFuncDataList();
        auto dyndata = reinterpret_cast<DynFuncData *>(&header->At(0));
        auto opAttrs = &dyndata->opAttrs[dyndata->opAtrrOffsets[TaskID(taskId)]];
        auto psgId = opAttrs[0];
        // devTaskId - funcId - leaf function Id - psgId
        std::string name = std::to_string(curTaskId_) + '-' + std::to_string(funcId) + '-' +
                           std::to_string(opIndex) + '-' + std::to_string(psgId);
        PerfMtEvent(PERF_EVT_TASK, coreIdx + PERF_AICORE_THREAD_START, currentTime, currentTime + timeCost, name);
        return timeCost;
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
                if (predCounts[opIndex] == 1 ||
                    __atomic_sub_fetch(&predCounts[opIndex], 1, __ATOMIC_RELAXED) == 0) {
                    auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
                    auto coreType = cceBinary[callList[opIndex]].coreType;
                    if (unlikely(coreType == static_cast<int>(CoreType::HUB))) {
                        ResolveDepDyn(id);
                        resolveHubCnt_++;
                    } else if (coreType == static_cast<int>(MachineType::AICPU)){
                        PushAicpuTaskQueue(id);
                    } else {
                        PushReadyTask(static_cast<int>(coreType), id);
                    }
                }
            }
        }
    }

    inline int GetRootIndex(uint32_t taskId) const {
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto func = dyntask->dynFuncDataCacheList[funcId].devFunc;
        return func->GetRootIndex();
    }

    inline int GetLeafIndex(uint32_t taskId) const {
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
        return callList[opIndex];
    }

    inline DevAscendFunctionDuppedData *GetDuppedData(uint32_t taskId) const {
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto funcId = FuncID(taskId);
        return dyntask->dynFuncDataCacheList[funcId].duppedData;
    }

    inline void ResolveDepDyn(uint64_t finishId, size_t resolveIndexBase = 0) {
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto funcId = FuncID(finishId);
        auto opIndex = TaskID(finishId);

        auto cceBinary = dyntask->cceBinary;
        auto func = dyntask->dynFuncDataCacheList[funcId].devFunc;
        auto predCounts =  dyntask->dynFuncDataCacheList[funcId].predCount;
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;

        size_t succIndexSize;
        const int *succIndexList = func->GetOperationDepGraphCopyOutResolveSuccIndexAddr(opIndex, succIndexSize);
        size_t succSize;
        auto succList = func->GetOperationDepGraphSuccAddr(opIndex, succSize);
        for (size_t i = succIndexList[resolveIndexBase]; i < succSize; i++) {
            auto succIdx = succList[i];
            if (predCounts[succIdx] == 1 ||
                __atomic_sub_fetch(&predCounts[succIdx], 1, __ATOMIC_RELAXED) == 0) {
                auto id = MakeTaskID(funcId, succIdx);
                auto coreType = cceBinary[callList[succIdx]].coreType;
                if (unlikely(coreType == static_cast<int>(CoreType::HUB))) {
                    ResolveDepDyn(id);
                    resolveHubCnt_++;
                } else if (unlikely(coreType == static_cast<int>(MachineType::AICPU))){
                    PushAicpuTaskQueue(id);
                } else {
                    PushReadyTask(static_cast<int>(coreType), id);
                }
            }
        }

        ResolveDynStitched(dyntask, funcId, opIndex);
    }

    inline void ResolveCopyOutDepDyn(uint32_t currResolveIndex, uint64_t taskId, uint32_t resolveIndexBase) {
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
        const int *succList = func->GetOperationDepGraphSuccAddr(opIndex, succSize);
        // here we don't use resolveIndexBase + 1, because at the beginning, resolveIndexBase is 0. And we resolve from 0.
        for (int i = succIndexList[resolveIndexBase]; i < succIndexList[currResolveIndex + 1]; i++) {
            auto succIdx = succList[i];
            if (predCounts[succIdx] == 1 ||
                __atomic_sub_fetch(&predCounts[succIdx], 1, __ATOMIC_RELAXED) == 0) {
                auto id = MakeTaskID(funcId, succIdx);
                auto coreType = cceBinary[callList[succIdx]].coreType;
                if (unlikely(coreType == static_cast<int>(CoreType::HUB))) {
                    ResolveDepDyn(id);
                    resolveHubCnt_++;
                } else if (unlikely(coreType == static_cast<int>(MachineType::AICPU))){
                    PushAicpuTaskQueue(id);
                } else {
                    PushReadyTask(static_cast<int>(coreType), id);
                }
            }
        }
    }

    inline void ResolveDepWithDfx(CoreType type, int coreIdx, uint64_t finishId, size_t resolveIndexBase = 0) {
        ResolveDepDyn(finishId, resolveIndexBase);
        DEV_VERBOSE_DEBUG("[Call]: Core %d Dispatch Task: %lu, %u, %u", coreIdx, seq,
                  FuncID(finishId), TaskID(finishId));
        DEV_TRACE_DEBUG(LEvent(
            LUid(curTaskCtrl_->taskId, FuncID(finishId), GetRootIndex(finishId), TaskID(finishId), GetLeafIndex(finishId)),
            LActFinish(coreIdx)));
        DfxProcAfterFinishTask(coreIdx, finishId);
        waitTaskCnt_[static_cast<int>(type)]--;
    }

    inline bool IsExistOtherAicpuIdle(CoreType type) {
        int idx = (aicpuIdx_ + 1) % aicpuNum_;
        while (idx != aicpuIdx_) {
            if (curTaskCtrl_->isAicpuIdle[static_cast<int>(type)][idx].load(std::memory_order_relaxed) == true){
                return true;
            }
            idx = (idx + 1) % aicpuNum_;
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

    inline void ResetCnt() {
        waitTaskCnt_[static_cast<int>(CoreType::AIC)] = 0;
        waitTaskCnt_[static_cast<int>(CoreType::AIV)] = 0;
        readyCount[static_cast<int>(CoreType::AIC)] = 0;
        readyCount[static_cast<int>(CoreType::AIV)] = 0;
        sendCnt_[static_cast<int>(CoreType::AIC)] = 0;
        sendCnt_[static_cast<int>(CoreType::AIV)] = 0;
    }

    inline void Init(int threadIdx, DeviceArgs *deviceArgs) {
        aicNum_ = static_cast<int32_t>(deviceArgs->nrAic);
        aivNum_ = static_cast<int32_t>(deviceArgs->nrAiv);
        aicpuNum_ = CalcSchAicpuNumByBlockDim(deviceArgs->nrValidAic);
        aicpuIdx_ = threadIdx;
        aicValidNum_ = deviceArgs->nrValidAic;
        aicoreHal_.Init(deviceArgs, &aicoreProf_);
        runningIds_.fill(AICORE_STATUS_INIT);
        pendingIds_.fill(AICORE_STATUS_INIT);
        runningResolveIndexList_.fill(0);
        pendingResolveIndexList_.fill(0);
        taskDfxStatPos_.fill(REG_LOW_TASK_PING);
        if (deviceArgs->machineConfig != static_cast<uint8_t>(MachineScheduleConfig::DEFAULT_SCH)) {
            if (aicpuNum_ > 1) {
                enableFairSch_ = static_cast<uint8_t>(deviceArgs->machineConfig) &
                    static_cast<uint8_t>(MachineScheduleConfig::MULTI_CORE_FAIR_SCH);
            }
            enableL2CacheSch_ = static_cast<uint8_t>(deviceArgs->machineConfig) &
                static_cast<uint8_t>(MachineScheduleConfig::L2CACHE_AFFINITY_SCH);
        }
        UpdateAiCoreBlockIndexSection();
        if constexpr (IsDeviceMode()) {
            aicoreHal_.MapRegistersForAllCores(aicNum_);
            aicoreProf_.ProfInit(reinterpret_cast<int64_t *>(deviceArgs->corePmuRegAddr),
               reinterpret_cast<int64_t *>(deviceArgs->pmuEventAddr));
        } else {
            aicoreHal_.SetTaskTimeCost([this](uint64_t coreIdx, uint64_t taskId, uint64_t time)
                {return GetCostModelTaskTime(coreIdx, taskId, time); });
        }
        ResetCnt();
        firstLock[static_cast<int>(CoreType::AIC)] = true;
        firstLock[static_cast<int>(CoreType::AIV)] = true;
        DEV_INFO("Init aicore manager aicNum_ %d aivNum_  %d sch_aicpuNum_ %d aicpuIdx_ %d "
                  "aicValidNum_ %d aicoreHal_.regAddrs_ %p sharedBuffer_ %p machineConfig: %u.",
            aicNum_, aivNum_, aicpuNum_, aicpuIdx_, aicValidNum_, aicoreHal_.GetRegAddrs(),
            (void *)aicoreHal_.GetSharedBuffer(), static_cast<uint8_t>(deviceArgs->machineConfig));
    }

    inline int HandShake() {
        DEV_INFO("Aicpu %d handshake start.", aicpuIdx_);
        int rc = ForEachManageAicoreWithRet([this](int coreIdx) -> int {
            int ret = aicoreHal_.HandShake(coreIdx, dotStatus_);
            DEV_VERBOSE_DEBUG("coreidx %d handshake  phycorid %d.", coreIdx, aicoreHal_.GetPhyIdByBlockId(coreIdx));
            return ret;
        });
        if (rc != DEVICE_MACHINE_OK) {
            DEV_ERROR("Aicpu %d handshake failed end.", aicpuIdx_);
            return rc;
        }

        ForEachManageAicore([this](int coreIdx) {
            aicoreHal_.WriteReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_OPEN);
        });
        /* write to MAINBASE reg need reg 0x18 open first */
        __sync_synchronize();
        DEV_INFO("Aicpu %d handshake sucess end.", aicpuIdx_);
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
        lastPendReadyCoreIdx_[static_cast<int>(CoreType::AIV)] = static_cast<uint32_t>(aivStart_);
        lastPendReadyCoreIdx_[static_cast<int>(CoreType::AIC)] = static_cast<uint32_t>(aicStart_);
        DEV_DEBUG("assign core aic coreindex section: start %d end %d.", aicStart_, aicEnd_);
        DEV_DEBUG("assign core aiv coreindex section: start %d end %d.", aivStart_, aivEnd_);
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

    inline void ForEachManageAicoreReverse(std::function<void(int coreIdx)> func) const {
        for (int i = aicEnd_ - 1; i >= aicStart_; --i) {
            func(i);
        }
        for (int i = aivEnd_ -1; i >= aivStart_ ; --i) {
            func(i);
        }
    }

    void SyncAiCore(uint64_t devTaskId) {
        if constexpr (IsDeviceMode()) {
            uint64_t waitAckStopVal = (devTaskId << REG_HIGH_DTASKID_SHIFT) | (AICORE_FUNC_STOP | AICORE_FIN_MASK);
            aicoreHal_.SetReadyQueue(aicStart_, aicEnd_, AICORE_FUNC_STOP + 1);
            aicoreHal_.SetReadyQueue(aivStart_, aivEnd_, AICORE_FUNC_STOP + 1);
            aicoreHal_.WaitFinQueue(aicStart_, aicEnd_, waitAckStopVal);
            aicoreHal_.WaitFinQueue(aivStart_, aivEnd_, waitAckStopVal);
            aicoreHal_.SetReadyQueue(aicStart_, aicEnd_, 0);
            aicoreHal_.SetReadyQueue(aivStart_, aivEnd_, 0);
        }
        __sync_synchronize();
    }

    inline int ForEachManageAicoreWithRet(std::function<int(int coreIdx)> func) const {
        int ret = DEVICE_MACHINE_OK;
        for (int i = aicStart_; i < aicEnd_; ++i) {
            ret = func(i);
            if (ret != DEVICE_MACHINE_OK) {
                DEV_ERROR("proc aicore aic %d failed.", i);
                return ret;
            }
        }
        for (int i = aivStart_; i < aivEnd_; ++i) {
            ret = func(i);
            if (ret != DEVICE_MACHINE_OK) {
                DEV_ERROR("proc aicore aiv %d failed.", i);
                return ret;
            }
        }
        return ret;
    }

    inline void AbnormalStop() {
        DEV_INFO("aicore manager %d try abnormal stop.", aicpuIdx_);
        aicoreHal_.WriteReg32All(aicNum_, aivNum_, REG_SPR_DATA_MAIN_BASE, AICORE_TASK_STOP + 1);
        /* write to MAINBASE reg must be done before close 0x18 */
        aicoreHal_.WriteReg32All(aicNum_, aivNum_, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
        DEV_INFO("aicore manager %d abnormal stopped.", aicpuIdx_);
    }

    inline void NormalStop() {
        DEV_INFO("aicore manager %d try normal stop.", aicpuIdx_);
        ForEachManageAicore([this](auto coreIdx) { aicoreHal_.SetReadyQueue(coreIdx, AICORE_TASK_STOP + 1) ; });
        /* write to MAINBASE reg must be done before close 0x18 */
        __sync_synchronize();
        ForEachManageAicore([this](auto coreIdx) {
            aicoreHal_.WriteReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
            aicoreHal_.ResetShakeBuf(coreIdx);
        });
        DEV_INFO("aicore manager %d normal stopped.", aicpuIdx_);
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
        if constexpr (!IsDeviceMode())
            return;

#if ENABLE_AICORE_PRINT
        DumpAicoreLog(coreIdx);
#endif

        volatile TaskStat *stat = aicoreHal_.GetTaskStat(coreIdx, 0);

#if PROF_DFX_HOST_PREPARE_MEMORY_MODE != 1
        aicoreProf_.ProfGet(coreIdx, stat->subGraphId, stat->taskId, const_cast<TaskStat*>(stat));
#endif

        DEV_IF_VERBOSE_DEBUG {
            recvFinTask_[coreIdx].push_back(TaskInfo(coreIdx, taskId));
        }

#if PROF_DFX_HOST_PREPARE_MEMORY_MODE != 1
        SetNextDfxPos(coreIdx); // pingpong 存储
#endif
    (void)stat;
    }

    inline bool IsNeedProcAicpuTask() {
        return aicpuIdx_ == 1;
    }
private:
    uint64_t seq;
    AicoreHAL aicoreHal_;
    bool isFirstTaskSend_{true};
    bool firstLock[AICORE_TYPE_NUM]{true,true};
    int aicNum_{0};
    int aivNum_{0};
    int aicValidNum_{0}; // 有效的aic，根据pgmask计算host传过来
    int aicpuIdx_{0};
    int aicpuNum_{MAX_SCHEDULE_AICPU_NUM};
    int aicStart_{0};
    int aicEnd_{0};
    int aivStart_{0};
    int aivEnd_{0};
    uint64_t procAicCoreFunctionCnt_{0};
    uint64_t procAivCoreFunctionCnt_{0};
    uint64_t procAicpuFunctionCnt_{0};
    bool enableL2CacheSch_{false};
    bool enableFairSch_{false};

    DeviceTask* curDevTask_{nullptr};
    DeviceTaskCtrl* curTaskCtrl_{nullptr};
    int curTaskType_{0};
    int curTaskId_{0};

    std::array<uint32_t, MAX_AICORE_NUM> runningIds_;
    std::array<uint32_t, MAX_AICORE_NUM> pendingIds_;
    std::array<int, MAX_AICORE_NUM> runningResolveIndexList_;
    std::array<int, MAX_AICORE_NUM> pendingResolveIndexList_;

    /* prepare aicore ready task list */
    ReadyCoreFunctionQueue* readyAicCoreFunctionQue_{nullptr};
    ReadyCoreFunctionQueue* readyAivCoreFunctionQue_{nullptr};

    uint64_t waitTaskCnt_[AICORE_TYPE_NUM]{0,0};
    uint32_t corePendReadyCnt_[AICORE_TYPE_NUM]{0,0};
    uint32_t coreRunReadyCnt_[AICORE_TYPE_NUM]{0,0};
    uint32_t runReadyCoreIdx_[AICORE_TYPE_NUM][MAX_MANAGER_AIV_NUM];
    uint32_t lastPendReadyCoreIdx_[AICORE_TYPE_NUM]{0,0};
    uint64_t resolveHubCnt_{0};

    uint32_t readyIds[AICORE_TYPE_NUM][READY_ID_FIX_CACHE_NUM];
    uint32_t readyCount[AICORE_TYPE_NUM]{0,0};
    uint32_t sendCnt_[AICORE_TYPE_NUM]{0,0};

    std::array<int, MAX_AICORE_NUM> taskDfxStatPos_;

    SPSCQueue<DeviceTaskCtrl *, DEFAULT_QUEUE_SIZE> taskQueue_;
    AicpuTaskManager &aicpuTaskManager_;
    AiCoreProf aicoreProf_;
    AicoreDump aicoreDump_;
    int64_t dotStatus_{0};

    std::vector<TaskInfo> sendTask_[MAX_AICORE_NUM];
    std::vector<TaskInfo> recvFinTask_[MAX_AICORE_NUM];
    std::vector<TaskInfo> recvAckTask_[MAX_AICORE_NUM];
    AicoreLogger *logger_{nullptr};
    friend class AiCoreProf;
};
}
