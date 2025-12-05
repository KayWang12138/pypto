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
#include "securec.h"
#include "device_common.h"
#include "tilefwk/config.h"
#include "tilefwk/aicore_print.h"
#include "interface/utils/common.h"
#include "interface/operation/opcode.h"
#include "interface/schema/schema.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/utils/dynamic/device_task.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "machine/utils/dynamic/small_array.h"
#include "machine/utils/dynamic/spsc_queue.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/utils/device_log.h"
#include "machine/kernel/aicore.h"
#include "machine/device/distributed/comm_wait_flag.h"
#include "machine/device/dynamic/aicore_prof.h"
#include "machine/device/dynamic/aicore_hal.h"
#include "machine/device/dynamic/aicpu_task_manager.h"
#include "machine/device/dynamic/device_utils.h"

namespace npu::tile_fwk::dynamic {
const uint32_t REG_SPR_FAST_PATH_ENABLE = 0x18;
const uint64_t REG_SPR_FAST_PATH_OPEN = 0xE;
const uint64_t REG_SPR_FAST_PATH_CLOSE = 0xF;

const int INVALID_CORE_IDX = 0xFF;

const uint32_t AICORE_STATUS_INIT = 0xFFFFFFFFU;
const uint32_t AIV_NUM_PER_AI_CORE = 2;
const uint32_t READY_ID_FIX_CACHE_NUM = 2048;


constexpr uint32_t  MAX_MANAGER_AIV_NUM = NAX_AIV_TOTAL_NUM;

constexpr uint32_t REG_31_BITS = 0x7FFFFFFF;
constexpr uint32_t REG_32_BITS = 0xFFFFFFFF;
#define REG_LOW_TASK_ID(regVal) (regVal) & REG_31_BITS // 低31位存储的taskid
#define REG_LOW_TASK_STATE(regVal) ((regVal)&REG_32_BITS) >> 31 // 低32位存储的task的状态
constexpr uint32_t TASK_FIN_STATE = 1; // 任务执行完成完成
constexpr uint32_t TASK_ACK_STATE = 0; // 收到任务状态，没执行完成
constexpr uint32_t REG_TASK_NUM = 2; // 一次寄存器task个数

#ifdef SUPPORT_WRAP
constexpr uint32_t CORE_IDX_AIV = 0;
constexpr uint32_t CORE_IDX_AIC = 1;

enum class MixResourceType {
    MIX_UNKNOWN = 0,
    MIX_1C1V = 1,
    MIX_1C2V = 2
};

inline void WrapInfoQueueLock(WrapInfoQueue* rq) {
  while (!__sync_bool_compare_and_swap(&rq->lock, 0, 1)) {
  }
}

inline void WrapInfoQueueUnLock(WrapInfoQueue* rq) {
  while (!__sync_bool_compare_and_swap(&rq->lock, 1, 0)) {
  }
}
#endif

struct TaskInfo {
    int coreIdx;
    uint64_t taskId;
    TaskInfo(int idx, uint64_t id) : coreIdx(idx), taskId(id) {}
};

inline void ReadyQueueLock(ReadyCoreFunctionQueue* rq) {
  while (!__sync_bool_compare_and_swap(&rq->lock, 0, 1)) {
  }
}

inline void ReadyQueueUnLock(ReadyCoreFunctionQueue* rq) {
  while (!__sync_bool_compare_and_swap(&rq->lock, 1, 0)) {
  }
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

        if (!preFetchSuccess_) {
            int64_t funcdata;
            auto dyntask = (DynDeviceTask *)curDevTask_;
            funcdata = static_cast<int64_t>(PtrToValue(dyntask->GetDynFuncDataList()));
            ForEachManageAicore([&](int coreIdx) {
                auto logbuf = logger_ ? logger_[coreIdx].GetBuffer() : nullptr;
                aicoreHal_.InitTaskData(coreIdx, funcdata, (uint64_t)logbuf);
            });
        }

        readyAicCoreFunctionQue_ = reinterpret_cast<ReadyCoreFunctionQueue *>(curDevTask_->readyAicCoreFunctionQue);
        readyAivCoreFunctionQue_ = reinterpret_cast<ReadyCoreFunctionQueue *>(curDevTask_->readyAivCoreFunctionQue);
#ifdef SUPPORT_WRAP
        readyWrapCoreFunctionQue_ = reinterpret_cast<WrapInfoQueue *>(curDevTask_->readyWrapCoreFunctionQue);
        wrapTasklist_ = reinterpret_cast<uint32_t *>(curDevTask_->wrapTasklist);

        wrapQueueForThread_.head = 0;
        wrapQueueForThread_.tail = 0;
        wrapQueueForThread_.elem = curDevTask_->wrapIdNum == 0 ? nullptr :
            static_cast<uint64_t *>(malloc(curDevTask_->wrapIdNum * sizeof(uint64_t)));
#endif
    }

    template <bool enableAicpuTask = false>
    inline uint32_t RunCoreTask(DeviceTaskCtrl *taskCtrl) {
        (void)taskCtrl;
#ifdef SUPPORT_WRAP
        DispatchMixCoreTask();
#endif
        DispatchAiCoreTask(CoreType::AIC, readyAicCoreFunctionQue_, aicStart_, aicEnd_);
        DispatchAiCoreTask(CoreType::AIV, readyAivCoreFunctionQue_, aivStart_, aivEnd_);

        uint64_t sentAic = sendCnt_[static_cast<int>(CoreType::AIC)];
        uint64_t sentAiv = sendCnt_[static_cast<int>(CoreType::AIV)];
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
        uint32_t allSentCnt = taskCtrl->finishedFunctionCnt.load(std::memory_order_relaxed);
        while (allSentCnt < curDevTask_->coreFunctionCnt) {
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

            // To prevent an unnecessary execution of RunCoreTask after the final batch of tasks is sent.
            allSentCnt = taskCtrl->finishedFunctionCnt.load(std::memory_order_relaxed) + lastSent;
        }
        if (lastSent > 0) {
            // Other SCH-AICPU are still waiting for the taskCtrl->finishedFunctionCnt actual value.
            taskCtrl->finishedFunctionCnt.fetch_add(lastSent, std::memory_order_relaxed);
        }

        PerfMtTrace(PERF_TRACE_DEV_TASK_SCHED_EXEC, aicpuIdx_);
        PerfMtBegin(PERF_EVT_SYNC_AICORE, aicpuIdx_);
        rc = SyncAicoreDevTaskFinish();
        PerfMtTrace(PERF_TRACE_DEV_TASK_SYNC_CORE_STOP, aicpuIdx_);
        if (rc != DEVICE_MACHINE_OK) {
            ret = rc;
        }
        DEV_DEBUG("sync finish ret = %d.", rc);

        if (IsNeedProcAicpuTask()) {
            while (!aicpuTaskManager_.Finished()) {
                (void)aicpuTaskManager_.TaskProcess();
            }
        }
        PerfMtEnd(PERF_EVT_SYNC_AICORE, aicpuIdx_);
        DEV_DEBUG("Aicpu %d proc finish send all task,aic: %lu, aiv: %lu, aicpu: %lu.",
            aicpuIdx_, procAicCoreFunctionCnt_, procAivCoreFunctionCnt_, procAicpuFunctionCnt_);
    FINISH:
#ifdef SUPPORT_WRAP
        if (wrapQueueForThread_.elem != nullptr) {
            free(wrapQueueForThread_.elem);
            wrapQueueForThread_.elem = nullptr;
        }
#endif
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
        uint32_t regDataMainBase = aicoreHal_.GetRegSprDataMainBase();
        if (isNeedWriteRegForFastPath_) {
            if (aicoreHal_.ReadReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE) == REG_SPR_FAST_PATH_OPEN) {
                aicoreHal_.WriteReg32(coreIdx, regDataMainBase, AICORE_TASK_STOP + 1);
                aicoreHal_.WriteReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
            }
        } else {
            aicoreHal_.WriteReg32(coreIdx, regDataMainBase, AICORE_TASK_STOP + 1);
        }
      });
    }

    inline void SetValidCore(std::array<bool, MAX_AICORE_NUM> *validCore) {
        DEV_IF_DEVICE {
            ForEachManageAicore([&](int coreIdx) {
                (*validCore)[GetPhyIdByBlockId(coreIdx)] = true;
                DEV_DEBUG(" Aicore %d is valid.", GetPhyIdByBlockId(coreIdx));
            });
            aicoreHal_.SetValidCore(validCore);
        }
    }

    inline int Run(int threadIdx, DeviceArgs *deviceArgs, bool handShakeByGm = true) {
        int ret = 0;
        DEV_DEBUG("schedule run threadIdx:%d", threadIdx);
        Init(threadIdx, deviceArgs);
        PerfMtTrace(PERF_TRACE_INIT, threadIdx);
        DEV_DEBUG("schedule run init succ");
        DeviceTaskCtrl *taskCtrl = nullptr;
        taskQueue_ = &(reinterpret_cast<SPSCQueue<DeviceTaskCtrl *, DEFAULT_QUEUE_SIZE>*>(deviceArgs->taskQueue)[threadIdx]);
        if constexpr (IsDeviceMode()) {
            ret = HandShake(handShakeByGm);
            PerfMtTrace(PERF_TRACE_CORE_HAND_SHAKE, threadIdx);
            if (ret != DEVICE_MACHINE_OK) {
                DEV_ERROR("hand shake timeout %d.", handShakeByGm);
                AbnormalStop();
                while ((taskCtrl = taskQueue_->Dequeue())) {
                    taskCtrl->PutTask(ret);
                } 
                return ret;
            }
            aicoreProf_.ProfStart();
        }
        DEV_DEBUG("schedule run start succ");
        uint64_t lastDevTaskFinCycle = 0;
        while (ret == 0) {
            DEV_DEBUG("schedule task wait");
            if (preFetchSuccess_) {
                taskCtrl = preFetchNextDevTaskCtrl_;
            } else {
                taskCtrl = taskQueue_->Dequeue();
            }
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
            PerfMtEnd(PERF_EVT_RUN_TASK, threadIdx);
            DEV_DEBUG("run task finish taskid=%d ret %d.", curTaskId_, ret);
            if (ret != 0)
                break;
            taskCtrl->PutTask(ret);
            PerfMtTrace(PERF_TRACE_DEV_TASK_RSP, threadIdx);
            PROF_STAGE_END_MTSAFE(PERF_EVT_STAGE_SCHEDULE, threadIdx, "dispatch.after\n");
        }
        if (ret) {
            DEV_ERROR("task %lu execute error %d, skip rest tasks.", taskCtrl->taskId, ret);
            if constexpr (IsDeviceMode()) {
                ForEachManageAicore([&](int coreIdx) {
                    DumpLastWord(coreIdx);
                });
            }
            do {
                taskCtrl->PutTask(ret);
            } while ((taskCtrl = taskQueue_->Dequeue()));

            if constexpr (IsDeviceMode()) {
                NormalStop(); // some core maybe timeout
            }
        }

        if constexpr (IsDeviceMode()) {
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

    inline bool CheckStopTaskCanBeSent(int coreIdx, bool &needWaitStopRsp) {
        if (pendingIds_[coreIdx] == AICORE_TASK_INIT && runningIds_[coreIdx] == AICORE_TASK_INIT) {
            return true;
        }

        uint64_t finTaskVal = aicoreHal_.GetFinishedTask(coreIdx);
        uint32_t regLFinTaskId = REG_LOW_TASK_ID(finTaskVal);
        uint32_t regLFinTaskState = REG_LOW_TASK_STATE(finTaskVal);
        bool bMatch = false;

        int type =static_cast<int>(AicoreType(coreIdx));
        if (likely(regLFinTaskState == TASK_FIN_STATE)) {
            if (pendingIds_[coreIdx] == regLFinTaskId) {
                bMatch = true;
                runReadyCoreIdx_[type][coreRunReadyCnt_[type]++] = coreIdx;
                corePendReadyCnt_[type]++;
                if (runningIds_[coreIdx] != AICORE_TASK_INIT) {
                    DfxProcAfterFinishTask(coreIdx, runningIds_[coreIdx]);
                }
                DfxProcAfterFinishTask(coreIdx, regLFinTaskId);
                DEV_VERBOSE_DEBUG("rcv final pending task finish, pendtask: %u", regLFinTaskId);
            } else if (runningIds_[coreIdx] == regLFinTaskId && pendingIds_[coreIdx] == AICORE_TASK_INIT) {
                bMatch = true;
                runReadyCoreIdx_[type][coreRunReadyCnt_[type]++] = coreIdx;
                DfxProcAfterFinishTask(coreIdx, regLFinTaskId);
                DEV_VERBOSE_DEBUG("rcv final running task finish, runningtask: %u", regLFinTaskId);
            }
        } else if (regLFinTaskState == TASK_ACK_STATE && pendingIds_[coreIdx] == regLFinTaskId) {
           // The core stop task can be sent once the last task ACK is received, without waiting for finish rsp.
           // The execution of the final task and the sending of the final core stop task can be parallelized.
            bMatch = true;
            needWaitStopRsp = true;
            runReadyCoreIdx_[type][coreRunReadyCnt_[type]++] = coreIdx;
            corePendReadyCnt_[type]++;
            DfxProcAfterFinishTask(coreIdx, regLFinTaskId);
            if (runningIds_[coreIdx] != AICORE_TASK_INIT) {
                DfxProcAfterFinishTask(coreIdx, runningIds_[coreIdx]);
            }
            DEV_VERBOSE_DEBUG("rcv final pending task ack, pendtask: %u", regLFinTaskId);
        }

        if (bMatch) {
            pendingIds_[coreIdx] = AICORE_TASK_INIT;
            pendingResolveIndexList_[coreIdx] = 0;
            runningIds_[coreIdx] = AICORE_TASK_INIT;
            runningResolveIndexList_[coreIdx] = 0;
            return true;
        }

        return false;
    }

    inline void PreFetchNextDevTask() {
        preFetchNextDevTaskCtrl_ = nullptr;
        preFetchSuccess_ = taskQueue_->TryDequeue(preFetchNextDevTaskCtrl_);

        DEV_INFO("Prefetch next dev task : success:%d, devtaskid:%lu",
            preFetchSuccess_, preFetchNextDevTaskCtrl_ != nullptr ? preFetchNextDevTaskCtrl_->taskId : INVALID_DEV_TASK_ID);
        return;
    }

    inline void SendPreFetchNextDevTaskDataToCore(int coreIdx) {
        if (preFetchNextDevTaskCtrl_ == nullptr) {
            return;
        }
        int64_t funcdata;
        auto dyntask = (DynDeviceTask *)preFetchNextDevTaskCtrl_->devTask;
        funcdata = static_cast<int64_t>(PtrToValue(dyntask->GetDynFuncDataList()));
        auto logbuf = logger_ ? logger_[coreIdx].GetBuffer() : nullptr;
        aicoreHal_.InitTaskData(coreIdx, funcdata, (uint64_t)logbuf);
        return;
    }

    enum AicoreStatus {
        CORE_TASK_WAIT_FINISH = 0,
        CORE_SEND_STOP,
        CORE_FINISH_STOP,
    };

    inline AicoreStatus AicoreDevTaskFinishProc(int coreIdx,  AicoreStatus curCoreStatus) {
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        uint64_t waitAckStopVal =
            ((uint64_t)curTaskId_ << REG_HIGH_DTASKID_SHIFT) | (AICORE_FUNC_STOP | AICORE_FIN_MASK);
        DEV_IF_DEVICE {
            if ((curCoreStatus == CORE_SEND_STOP) && (aicoreHal_.GetFinishedTask(coreIdx) == waitAckStopVal)) {
                if (!dyntask->IsLastTask()) {
                    /* With the previous DevTask verified as stopped, the next DevTask can be sent early,
                    bypassing the delay for its control flow response. */
                    SendPreFetchNextDevTaskDataToCore(coreIdx);
                    aicoreHal_.SetReadyQueue(coreIdx, 0);
                    DEV_DEBUG("core %d rsp AICORE_FUNC_STOP ack.", coreIdx);
                } else {
                    DEV_DEBUG("Last devtask ,core %d send AICORE_TASK_STOP.", coreIdx);
                    NormalStopSingleCore(coreIdx);
                }
                return CORE_FINISH_STOP;
            }
        }

        bool needWaitStopRsp = false;
        if (CheckStopTaskCanBeSent(coreIdx, needWaitStopRsp)) {
            DEV_IF_DEVICE {
                if (needWaitStopRsp || !dyntask->IsLastTask()) {
                    /* Sending pre-fetch next devTask should not be called here,
                       as it may result in the nextDevTask being stopped. */
                    aicoreHal_.SetReadyQueue(coreIdx, AICORE_FUNC_STOP + 1);
                    DEV_DEBUG("core %d send AICORE_FUNC_STOP.", coreIdx);
                    return CORE_SEND_STOP;
                } else {
                    DEV_DEBUG("Last devtask ,core %d send AICORE_TASK_STOP.", coreIdx);
                    NormalStopSingleCore(coreIdx);
                    return CORE_FINISH_STOP;
                }
            } else {
                return CORE_FINISH_STOP;
            }
        }

        DEV_DEBUG("core %d have tail task not finish.", coreIdx);
        return curCoreStatus;
    }

    inline int SyncAicoreDevTaskFinish() {
        int stopNum = 0;
        int mngCoreNum = aicEnd_ - aicStart_ + aivEnd_ - aivStart_;
        AicoreStatus coreStatus[MAX_AICORE_NUM] = {CORE_TASK_WAIT_FINISH};
        bool aicAllStop = false;
        bool aivAllStop = false;

        PreFetchNextDevTask();
        int64_t start_cycles = GetCycles();
        while (stopNum < mngCoreNum) {
            bool curIterAicAllStop = true;
            bool curIterAivAllStop = true;
            for (int i = aicStart_; (!aicAllStop) && i < aicEnd_; i++) {
                if (coreStatus[i] == CORE_FINISH_STOP) {
                    continue;
                }

                coreStatus[i] = AicoreDevTaskFinishProc(i, coreStatus[i]);
                if (coreStatus[i] == CORE_FINISH_STOP) {
                    stopNum++;
                } else {
                    curIterAicAllStop = false;
                }
            }
            aicAllStop = curIterAicAllStop;

            for (int i = aivStart_; (!aivAllStop) && i < aivEnd_; i++) {
                if (coreStatus[i] == CORE_FINISH_STOP) {
                    continue;
                }

                coreStatus[i] = AicoreDevTaskFinishProc(i, coreStatus[i]);
                if (coreStatus[i] == CORE_FINISH_STOP) {
                    stopNum++;
                } else {
                    curIterAivAllStop = false;
                }
            }
            aivAllStop = curIterAivAllStop;

            if (GetCycles() - start_cycles > TIMEOUT_CYCLES) {
                DEV_ERROR("SyncAicoreDevTaskFinish timeout notstopNum=%d.", mngCoreNum - stopNum);
                return DEVICE_MACHINE_TIMEOUT_SYNC_CORE_FINISH;
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
#ifdef SUPPORT_WRAP
            DispatchMixCoreTask();
#endif
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

#ifdef SUPPORT_WRAP
    inline uint32_t GetAvailableCoreIdx(MixResourceType mixType = MixResourceType::MIX_UNKNOWN) {
        if (mixType == MixResourceType::MIX_1C1V) {
            for (uint32_t i = 0; i < coreRunReadyCnt_[CORE_IDX_AIC]; i++) {
                for (uint32_t j = 0; j < coreRunReadyCnt_[CORE_IDX_AIV]; j++) {
                    uint32_t aicIdx = runReadyCoreIdx_[CORE_IDX_AIC][i];
                    uint32_t aivIdx = runReadyCoreIdx_[CORE_IDX_AIV][j];
                    if (aicIdx * AIV_NUM_PER_AI_CORE + aicValidNum_ == aivIdx) {
                        return aicIdx;
                    }
                }
            }
            return INVALID_CORE_IDX;
        }

        for (uint32_t i = 0; i < coreRunReadyCnt_[CORE_IDX_AIC]; i++) {
            for (uint32_t j = 0; j < coreRunReadyCnt_[CORE_IDX_AIV]; j++) {
                for (uint32_t k = 0; k < coreRunReadyCnt_[CORE_IDX_AIV]; k++) {
                    uint32_t aicIdx = runReadyCoreIdx_[CORE_IDX_AIC][i];
                    uint32_t aivIdx0 = runReadyCoreIdx_[CORE_IDX_AIV][j];
                    uint32_t aivIdx1 = runReadyCoreIdx_[CORE_IDX_AIV][k];
                    if (aicIdx * AIV_NUM_PER_AI_CORE + aicValidNum_ == aivIdx0 && aivIdx0 + 1 == aivIdx1) {
                        return aicIdx;
                    }
                }
            }
        }
        return INVALID_CORE_IDX;
    }

    inline void RemoveRunReadyCoreIdxForWrap(uint32_t coreIdx, MixResourceType mixType = MixResourceType::MIX_UNKNOWN) {
        coreRunReadyCnt_[CORE_IDX_AIC]--;
        corePendReadyCnt_[CORE_IDX_AIC]--;
        // if coreIdx is at the tail of runReadyCoreIdx_, no processing is need, simply cnt--
        if (runReadyCoreIdx_[CORE_IDX_AIC][coreRunReadyCnt_[CORE_IDX_AIC]] != coreIdx) {
            // if coreIdx isnt at the tail of runReadyCoreIdx_, replace it by tail data
            for (uint32_t i = 0; i < coreRunReadyCnt_[CORE_IDX_AIC]; i++) {
                if (runReadyCoreIdx_[CORE_IDX_AIC][i] == coreIdx) {
                    // swap tail data with coreIdx
                    runReadyCoreIdx_[CORE_IDX_AIC][i] = runReadyCoreIdx_[CORE_IDX_AIC][coreRunReadyCnt_[CORE_IDX_AIC]];
                    runReadyCoreIdx_[CORE_IDX_AIC][coreRunReadyCnt_[CORE_IDX_AIC]] = coreIdx;
                }
            }
        }

        coreRunReadyCnt_[CORE_IDX_AIV]--;
        corePendReadyCnt_[CORE_IDX_AIV]--;
        uint32_t aivIdx0 = coreIdx * AIV_NUM_PER_AI_CORE + aicValidNum_;
        if (runReadyCoreIdx_[CORE_IDX_AIV][coreRunReadyCnt_[CORE_IDX_AIV]] != aivIdx0) {
            for (uint32_t i = 0; i < coreRunReadyCnt_[CORE_IDX_AIV]; i++) {
                if (runReadyCoreIdx_[CORE_IDX_AIV][i] == aivIdx0) {
                    runReadyCoreIdx_[CORE_IDX_AIV][i] = runReadyCoreIdx_[CORE_IDX_AIV][coreRunReadyCnt_[CORE_IDX_AIV]];
                    runReadyCoreIdx_[CORE_IDX_AIV][coreRunReadyCnt_[CORE_IDX_AIV]] = aivIdx0;
                }
            }
        }

        if (mixType != MixResourceType::MIX_1C1V) {
            coreRunReadyCnt_[CORE_IDX_AIV]--;
            corePendReadyCnt_[CORE_IDX_AIV]--;
            uint32_t aivIdx1 = coreIdx * AIV_NUM_PER_AI_CORE + aicValidNum_ + 1;
            if (runReadyCoreIdx_[CORE_IDX_AIV][coreRunReadyCnt_[CORE_IDX_AIV]] != aivIdx1) {
                for (uint32_t i = 0; i < coreRunReadyCnt_[CORE_IDX_AIV]; i++) {
                    if (runReadyCoreIdx_[CORE_IDX_AIV][i] == aivIdx1) {
                        runReadyCoreIdx_[CORE_IDX_AIV][i] = runReadyCoreIdx_[CORE_IDX_AIV][coreRunReadyCnt_[CORE_IDX_AIV]];
                        runReadyCoreIdx_[CORE_IDX_AIV][coreRunReadyCnt_[CORE_IDX_AIV]] = aivIdx1;
                    }
                }
            }
            DEV_VERBOSE_DEBUG("remove coreIdx %u  %u  %u", coreIdx, aivIdx0, aivIdx1);
        } else {
            DEV_VERBOSE_DEBUG("remove coreIdx %u  %u", coreIdx, aivIdx0);
        }
    }

    inline void AddRunReadyCoreIdxForWrap(uint32_t coreIdx, MixResourceType mixType = MixResourceType::MIX_UNKNOWN) {
        runReadyCoreIdx_[CORE_IDX_AIC][coreRunReadyCnt_[CORE_IDX_AIC]++] = coreIdx;
        runReadyCoreIdx_[CORE_IDX_AIV][coreRunReadyCnt_[CORE_IDX_AIV]++] = coreIdx * AIV_NUM_PER_AI_CORE + aicValidNum_;
        corePendReadyCnt_[CORE_IDX_AIC]++;
        corePendReadyCnt_[CORE_IDX_AIV]++;
        if (mixType != MixResourceType::MIX_1C1V) {
            runReadyCoreIdx_[CORE_IDX_AIV][coreRunReadyCnt_[CORE_IDX_AIV]++] = coreIdx * AIV_NUM_PER_AI_CORE + aicValidNum_ + 1;
            DEV_VERBOSE_DEBUG("add coreIdx %u  %u  %u", coreIdx, coreIdx * AIV_NUM_PER_AI_CORE + aicValidNum_,
                coreIdx * AIV_NUM_PER_AI_CORE + aicValidNum_ + 1);
            corePendReadyCnt_[CORE_IDX_AIV]++;
        } else {
            DEV_VERBOSE_DEBUG("add coreIdx %u  %u", coreIdx, coreIdx * AIV_NUM_PER_AI_CORE + aicValidNum_);
        }
    }

    inline void UpdateWrapQueueForThread() {
        // when readyWrapCoreFunctionQueue has valid value and has available wrapCore
        // move wrapId from readyWrapCoreFunctionQueue to wrapQueueForThread, and occpy wrapCore
        WrapInfoQueueLock(readyWrapCoreFunctionQue_);
        uint32_t head = __atomic_load_n(&readyWrapCoreFunctionQue_->head, __ATOMIC_RELAXED);
        uint32_t tail = __atomic_load_n(&readyWrapCoreFunctionQue_->tail, __ATOMIC_RELAXED);
        uint32_t taskCount = tail - head;
        if (taskCount == 0) {
            DEV_VERBOSE_DEBUG("mixcore taskCount is zero.");
            WrapInfoQueueUnLock(readyWrapCoreFunctionQue_);
            return;
        }

        while (taskCount-- > 0) {
            WrapInfo *wrapInfo = &readyWrapCoreFunctionQue_->elem[readyWrapCoreFunctionQue_->head];
            uint32_t wrapId = wrapInfo->wrapId;
            MixResourceType mixType = static_cast<MixResourceType>(wrapInfo->mixResourceType);

            uint32_t avaiCoreIdx = GetAvailableCoreIdx(mixType);
            if (avaiCoreIdx == INVALID_CORE_IDX) {
                DEV_VERBOSE_DEBUG("no available wrap core.");
                WrapInfoQueueUnLock(readyWrapCoreFunctionQue_);
                return;
            }

            DEV_VERBOSE_DEBUG("move wrapId[%u] to wrapQueueForThread. occupy coreIdx[%u]", wrapId, avaiCoreIdx);
            wrapQueueForThread_.elem[wrapQueueForThread_.tail++] = reinterpret_cast<uint64_t>(wrapInfo);
            __atomic_fetch_add(&readyWrapCoreFunctionQue_->head, 1, std::memory_order_release);
            RemoveRunReadyCoreIdxForWrap(avaiCoreIdx, mixType);

            wrapInfo->aicCoreIdx = avaiCoreIdx;
            wrapInfo->aivCoreIdxZero = avaiCoreIdx * AIV_NUM_PER_AI_CORE + aicValidNum_;
            wrapInfo->aivCoreIdxOne = wrapInfo->aivCoreIdxZero + (mixType != MixResourceType::MIX_1C1V ? 1 : 0);
            wrapCoreStatus_[wrapInfo->aicCoreIdx] = 0;
            wrapCoreStatus_[wrapInfo->aivCoreIdxZero] = 0;
            wrapCoreStatus_[wrapInfo->aivCoreIdxOne] = 0;
            DEV_VERBOSE_DEBUG("add wrapInfo, aicCoreIdx = %u, aivCoreIdxZero = %u, aivCoreIdxOne = %u, taskCnt = %u, mixResourceType = %u",
                wrapInfo->aicCoreIdx, wrapInfo->aivCoreIdxZero, wrapInfo->aivCoreIdxOne, wrapInfo->taskCnt, static_cast<uint32_t>(wrapInfo->mixResourceType));
        }
        WrapInfoQueueUnLock(readyWrapCoreFunctionQue_);
    }

    inline void DispatchMixCoreTask() {
        if (curDevTask_->wrapIdNum == 0) {
            return;
        }
        UpdateWrapQueueForThread();
        for (uint32_t idx = wrapQueueForThread_.head; idx < wrapQueueForThread_.tail; idx++) {
            WrapInfo *wrapInfo = reinterpret_cast<WrapInfo *>(wrapQueueForThread_.elem[idx]);
            std::vector<uint32_t> sendTaskIdx;
            ReadyQueueLock(&wrapInfo->tasklist);

            for (uint32_t taskIdx = wrapInfo->tasklist.head; taskIdx < wrapInfo->tasklist.tail; taskIdx++) {
                uint32_t taskId = wrapInfo->tasklist.elem[taskIdx];
                CoreType coreType = GetCoreType(taskId);
                DEV_VERBOSE_DEBUG("try to send wrapId[%u]'s taskIdx[%u] taskId[%u]", wrapInfo->wrapId, taskIdx, taskId);
                if (coreType == CoreType::AIC && wrapCoreStatus_[wrapInfo->aicCoreIdx] == 0) {
                    SendTaskToAiCore(coreType, wrapInfo->aicCoreIdx, taskId);
                    wrapCoreStatus_[wrapInfo->aicCoreIdx] = 1;
                    sendTaskIdx.push_back(taskIdx);
                } else if (coreType == CoreType::AIV) {
                    int32_t wrapVecId = GetWrapVecId(taskId);
                    if (wrapCoreStatus_[wrapInfo->aivCoreIdxZero] == 0 && (wrapVecId == 0 || wrapVecId == -1)) {
                        SendTaskToAiCore(coreType, wrapInfo->aivCoreIdxZero, taskId);
                        wrapCoreStatus_[wrapInfo->aivCoreIdxZero] = 1;
                        sendTaskIdx.push_back(taskIdx);
                    } else if (wrapCoreStatus_[wrapInfo->aivCoreIdxOne] == 0 && (wrapVecId == 1 || wrapVecId == -1)) {
                        SendTaskToAiCore(coreType, wrapInfo->aivCoreIdxOne, taskId);
                        wrapCoreStatus_[wrapInfo->aivCoreIdxOne] = 1;
                        sendTaskIdx.push_back(taskIdx);
                    }
                }
                if ( wrapCoreStatus_[wrapInfo->aicCoreIdx] == 1 && wrapCoreStatus_[wrapInfo->aivCoreIdxZero] == 1 &&
                    wrapCoreStatus_[wrapInfo->aivCoreIdxOne] == 1) {
                        break; // all wrapCore is busy, early exit
                    }
            }
            for (int32_t i = static_cast<int32_t>(sendTaskIdx.size()) - 1; i >= 0; i--) {
                std::swap(wrapInfo->tasklist.elem[sendTaskIdx[i]], wrapInfo->tasklist.elem[--wrapInfo->tasklist.tail]);
            }
            ReadyQueueUnLock(&wrapInfo->tasklist);
        }
    }
#endif

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
#ifdef SUPPORT_WRAP
            if (wrapCoreStatus_[coreIdx] == 0) { // wrapcore doesnt support pending & running yet
                runReadyCoreIdx_[static_cast<int>(type)][coreRunReadyCnt_[static_cast<int>(type)]++] = coreIdx;
                corePendReadyCnt_[static_cast<int>(type)]++;
            }
#else
            runReadyCoreIdx_[static_cast<int>(type)][coreRunReadyCnt_[static_cast<int>(type)]++] = coreIdx;
            corePendReadyCnt_[static_cast<int>(type)]++;
#endif
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
#ifdef SUPPORT_WRAP
            if (wrapCoreStatus_[coreIdx] == 0) {
                corePendReadyCnt_[static_cast<int>(type)]++;
            }
#else
            corePendReadyCnt_[static_cast<int>(type)]++;
#endif
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
#ifdef SUPPORT_WRAP
            if (wrapCoreStatus_[coreIdx] == 0) {
                corePendReadyCnt_[static_cast<int>(type)]++;
            }
#else
            corePendReadyCnt_[static_cast<int>(type)]++;
#endif
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
#ifdef SUPPORT_WRAP
            if (wrapCoreStatus_[coreIdx] == 0 && pendingIdRef == AICORE_TASK_INIT) {
                runReadyCoreIdx_[static_cast<int>(type)][coreRunReadyCnt_[static_cast<int>(type)]++] = coreIdx;
            }
#else
            if (pendingIdRef == AICORE_TASK_INIT) {
                runReadyCoreIdx_[static_cast<int>(type)][coreRunReadyCnt_[static_cast<int>(type)]++] = coreIdx;
            }
#endif
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

#ifdef SUPPORT_WRAP
    inline void PushTaskToTasklist(uint32_t wrapId, uint32_t taskId) {
        WrapInfo *wrapInfo = nullptr;
        WrapInfoQueueLock(readyWrapCoreFunctionQue_);
        for (uint32_t idx = 0; idx < readyWrapCoreFunctionQue_->tail; idx++) {
            if (readyWrapCoreFunctionQue_->elem[idx].wrapId == wrapId) {
                wrapInfo = &readyWrapCoreFunctionQue_->elem[idx];
                break;
            }
        }

        if (wrapInfo == nullptr) {
            // add a new wrapinfo
            wrapInfo = &readyWrapCoreFunctionQue_->elem[readyWrapCoreFunctionQue_->tail];
            wrapInfo->wrapId = wrapId;
            wrapInfo->aicCoreIdx = 0;
            wrapInfo->aivCoreIdxZero = 0;
            wrapInfo->aivCoreIdxOne = 0;
            wrapInfo->taskCnt = GetWrapTaskNum(taskId);
            wrapInfo->mixResourceType = GetMixResourceType(taskId);
            wrapInfo->tasklist.head = 0;
            wrapInfo->tasklist.tail = 0;
            wrapInfo->tasklist.capacity = wrapInfo->taskCnt;
            if (readyWrapCoreFunctionQue_->tail == 0) {
                wrapInfo->tasklist.elem = wrapTasklist_;
            } else {
                auto preQueue = &readyWrapCoreFunctionQue_->elem[readyWrapCoreFunctionQue_->tail - 1];
                wrapInfo->tasklist.elem = preQueue->tasklist.elem + preQueue->tasklist.capacity;
            }
            __atomic_fetch_add(&readyWrapCoreFunctionQue_->tail, 1, std::memory_order_release);
        }
        WrapInfoQueueUnLock(readyWrapCoreFunctionQue_);
        ReadyQueueLock(&wrapInfo->tasklist);
        wrapInfo->tasklist.elem[wrapInfo->tasklist.tail++] = taskId;
        ReadyQueueUnLock(&wrapInfo->tasklist);
    }

    inline void ResolveDepForMixCore(uint32_t taskId) {
        // resolve dep, if has available core, send task directly, else call PushTaskToTasklist, try to send task in next loop
        uint32_t wrapId = GetWrapId(taskId);
        DEV_VERBOSE_DEBUG("taskId = %u, wrapId = %u", taskId, wrapId);

        WrapInfo *wrapInfo = nullptr;
        for (uint32_t idx = wrapQueueForThread_.head; idx < wrapQueueForThread_.tail; idx++) {
            if (reinterpret_cast<WrapInfo *>(wrapQueueForThread_.elem[idx])->wrapId == wrapId) {
                wrapInfo = reinterpret_cast<WrapInfo *>(wrapQueueForThread_.elem[idx]);
                break;
            }
        }

        if (wrapInfo == nullptr) { // the wrap is not in this thread
            DEV_VERBOSE_DEBUG("the wrapId %u is not in this thread, push taskId %u to tasklist", wrapId, taskId);
            PushTaskToTasklist(wrapId, taskId);
            return;
        }

        // if the wrap is in this thread, try to send task directly
        if (GetCoreType(taskId) == CoreType::AIC && wrapCoreStatus_[wrapInfo->aicCoreIdx] == 0) {
            DEV_VERBOSE_DEBUG("directly send taskId %u to cubecore", taskId);
            SendTaskToAiCore(CoreType::AIC, wrapInfo->aicCoreIdx, taskId);
            wrapCoreStatus_[wrapInfo->aicCoreIdx] = 1;
            return;
        }

        if (GetCoreType(taskId) == CoreType::AIV) {
            int32_t wrapVecId = GetWrapVecId(taskId);
            if (wrapCoreStatus_[wrapInfo->aivCoreIdxZero] == 0 && (wrapVecId == 0 || wrapVecId == -1)) {
                DEV_VERBOSE_DEBUG("directly send taskId %u to veccore0", taskId);
                SendTaskToAiCore(CoreType::AIV, wrapInfo->aivCoreIdxZero, taskId);
                wrapCoreStatus_[wrapInfo->aivCoreIdxZero] = 1;
                return;
            } else if (wrapCoreStatus_[wrapInfo->aivCoreIdxOne] == 0 && (wrapVecId == 1 || wrapVecId == -1)) {
                DEV_VERBOSE_DEBUG("directly send taskId %u to veccore1", taskId);
                SendTaskToAiCore(CoreType::AIV, wrapInfo->aivCoreIdxOne, taskId);
                wrapCoreStatus_[wrapInfo->aivCoreIdxOne] = 1;
                return;
            }
        }
        DEV_VERBOSE_DEBUG("there is no available core, push taskId %u to tasklist", taskId);
        PushTaskToTasklist(wrapId, taskId);
    }

    inline void UpdateFinishIdForMixCore(uint32_t finishId, int coreIdx) {
        uint32_t wrapId = GetWrapId(finishId);
        WrapInfo *wrapInfo = nullptr;
        uint32_t wrapIdx = 0;
        for (uint32_t idx = wrapQueueForThread_.head; idx < wrapQueueForThread_.tail; idx++) {
            if (reinterpret_cast<WrapInfo *>(wrapQueueForThread_.elem[idx])->wrapId == wrapId) {
                wrapInfo = reinterpret_cast<WrapInfo *>(wrapQueueForThread_.elem[idx]);
                wrapIdx = idx;
                break;
            }
        }

        if (wrapInfo == nullptr) {
            DEV_ERROR("cant find wrapInfo in wrapQueueForThread!");
            return;
        }
        wrapInfo->taskCnt--;
        if (wrapInfo->taskCnt == 0) { // all tasks for this wrap finish
            DEV_VERBOSE_DEBUG("wrapId %u 's all tasks finish, release wrapcore", wrapId);
            AddRunReadyCoreIdxForWrap(wrapInfo->aicCoreIdx, static_cast<MixResourceType>(wrapInfo->mixResourceType)); // free wrap core
            wrapCoreStatus_[wrapInfo->aicCoreIdx] = 0;
            wrapCoreStatus_[wrapInfo->aivCoreIdxZero] = 0;
            wrapCoreStatus_[wrapInfo->aivCoreIdxOne] = 0;
            std::swap(wrapQueueForThread_.elem[wrapIdx], wrapQueueForThread_.elem[--wrapQueueForThread_.tail]);
        } else {
            DEV_VERBOSE_DEBUG("wrapId %u 's all tasks not finish yet, only set coreIdx[%d] status to 0", wrapId, coreIdx);
            wrapCoreStatus_[coreIdx] = 0;
        }
    }
#endif

    inline void ResolveDepDyn(uint64_t finishId, size_t resolveIndexBase = 0, int coreIdx = 0) {
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto funcId = FuncID(finishId);
        auto opIndex = TaskID(finishId);

#ifdef SUPPORT_WRAP
        if (curDevTask_->wrapIdNum > 0 && GetWrapId(finishId) != -1) {
            UpdateFinishIdForMixCore(finishId, coreIdx);
        }
#endif

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
                    ResolveDepDyn(id, resolveIndexBase, coreIdx);
                    resolveHubCnt_++;
                } else if (unlikely(coreType == static_cast<int>(MachineType::AICPU))){
                    PushAicpuTaskQueue(id);
#ifdef SUPPORT_WRAP
                } else if (curDevTask_->wrapIdNum > 0 && GetWrapId(id) != -1) {
                    ResolveDepForMixCore(id);
#endif
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
        ResolveDepDyn(finishId, resolveIndexBase, coreIdx);
        DEV_VERBOSE_DEBUG("[Call]: Core %d Dispatch Task: %lu, %u, %u", coreIdx, seq,
                  FuncID(finishId), TaskID(finishId));
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
        aicpuNum_ = deviceArgs->scheCpuNum;
        aicpuIdx_ = threadIdx;
        aicValidNum_ = deviceArgs->nrValidAic;
        aicoreHal_.Init(deviceArgs, &aicoreProf_);
        runningIds_.fill(AICORE_STATUS_INIT);
        pendingIds_.fill(AICORE_STATUS_INIT);
        runningResolveIndexList_.fill(0);
        pendingResolveIndexList_.fill(0);
        taskDfxStatPos_.fill(REG_LOW_TASK_PING);

        if (deviceArgs->socVersion == SocVersion::AIC_310) {
            isNeedWriteRegForFastPath_ = false;
        }
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
        preFetchSuccess_ = false;
        preFetchNextDevTaskCtrl_ = nullptr;
        DEV_INFO("Init aicore manager aicNum_ %d aivNum_  %d sch_aicpuNum_ %d aicpuIdx_ %d "
                  "aicValidNum_ %d aicoreHal_.regAddrs_ %p sharedBuffer_ %p machineConfig: %u.",
            aicNum_, aivNum_, aicpuNum_, aicpuIdx_, aicValidNum_, aicoreHal_.GetRegAddrs(),
            (void *)aicoreHal_.GetSharedBuffer(), static_cast<uint8_t>(deviceArgs->machineConfig));
    }

    inline int HandShakeByGm() {
        int rc = ForEachManageAicoreWithRet([this](int coreIdx) -> int {
            int ret = aicoreHal_.HandShakeByGm(coreIdx, dotStatus_);
            DEV_VERBOSE_DEBUG("coreidx %d handshake by gm phycorid %d.",
                coreIdx, aicoreHal_.GetPhyIdByBlockId(coreIdx));
            return ret;
        });

        return rc;
    }

    inline int HandShake(bool isHandShakeByGm) {
        DEV_INFO("Aicpu %d handshake start.", aicpuIdx_);
        int rc = DEVICE_MACHINE_OK;
        if (isHandShakeByGm) {
            rc = HandShakeByGm();
        } else {
            rc = aicoreHal_.HandShakeByReg(dotStatus_);
        }

        if (rc != DEVICE_MACHINE_OK) {
            DEV_ERROR("Aicpu %d handshake failed end.", aicpuIdx_);
            return rc;
        }

        if (isNeedWriteRegForFastPath_) {
            ForEachManageAicore([this](int coreIdx) {
                aicoreHal_.WriteReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_OPEN);
            });
        }
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
        aicoreHal_.SetMngCoreBlockId(aicStart_, aicEnd_, aivStart_, aivEnd_);
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
        ResetRegAll();
        DEV_INFO("aicore manager %d abnormal stopped.", aicpuIdx_);
    }

    inline void NormalStop() {
        DEV_INFO("aicore manager %d try normal stop.", aicpuIdx_);
        ForEachManageAicore([this](auto coreIdx) { aicoreHal_.SetReadyQueue(coreIdx, AICORE_TASK_STOP + 1) ; });
        /* write to MAINBASE reg must be done before close 0x18 */
        __sync_synchronize();
        ForEachManageAicore([this](auto coreIdx) {
            if (isNeedWriteRegForFastPath_) {
                aicoreHal_.WriteReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
            }
            aicoreHal_.ResetShakeBuf(coreIdx);
        });
        DEV_INFO("aicore manager %d normal stopped.", aicpuIdx_);
    }

    inline void NormalStopSingleCore(int coreIdx) {
        aicoreHal_.SetReadyQueue(coreIdx, AICORE_TASK_STOP + 1);
        __sync_synchronize();
        aicoreHal_.WriteReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
        aicoreHal_.ResetShakeBuf(coreIdx);
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
        DEV_TRACE_DEBUG(LEvent(
            LUid(curTaskCtrl_->taskId, FuncID(taskId), GetRootIndex(taskId), TaskID(taskId), GetLeafIndex(taskId)),
            LActFinish(coreIdx)));
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

#ifdef SUPPORT_WRAP
    int32_t GetWrapId(uint32_t taskId) {
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto opWrapList = dyntask->dynFuncDataCacheList[funcId].opWrapList;
        if (opWrapList[opIndex] != -1) {
            return MakeWrapID(funcId, opWrapList[opIndex]);
        } else {
            return -1;
        }
    }

    uint32_t GetWrapTaskNum(uint32_t taskId) {
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto opWrapTaskNumList = dyntask->dynFuncDataCacheList[funcId].opWrapTaskNumList;
        return opWrapTaskNumList[opIndex];
    }

    int32_t GetWrapVecId(uint32_t taskId) {
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto cceBinary = dyntask->cceBinary;
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
        return cceBinary[callList[opIndex]].wrapVecId;
    }

    CoreType GetCoreType(uint32_t taskId) {
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto cceBinary = dyntask->cceBinary;
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
        return static_cast<CoreType>(cceBinary[callList[opIndex]].coreType);
    }

    uint32_t GetMixResourceType(uint32_t taskId) {
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto cceBinary = dyntask->cceBinary;
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
        return cceBinary[callList[opIndex]].mixResourceType;
    }
#endif

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
#ifdef SUPPORT_WRAP
    WrapInfoQueue* readyWrapCoreFunctionQue_{nullptr};
    // Queue managed by each thread, elem is wrapInfo's addr
    StaticReadyCoreFunctionQueue wrapQueueForThread_{0, 0, nullptr, 0};
    uint32_t* wrapTasklist_{nullptr};
    uint32_t wrapCoreStatus_[MAX_AICORE_NUM]{0};
#endif
    uint64_t waitTaskCnt_[AICORE_TYPE_NUM]{0,0};
    uint32_t corePendReadyCnt_[AICORE_TYPE_NUM]{0,0};
    uint32_t coreRunReadyCnt_[AICORE_TYPE_NUM]{0,0};
    uint32_t runReadyCoreIdx_[AICORE_TYPE_NUM][MAX_MANAGER_AIV_NUM];
    uint32_t lastPendReadyCoreIdx_[AICORE_TYPE_NUM]{0,0};
    uint64_t resolveHubCnt_{0};

    uint32_t readyIds[AICORE_TYPE_NUM][READY_ID_FIX_CACHE_NUM];
    uint32_t readyCount[AICORE_TYPE_NUM]{0,0};
    uint32_t sendCnt_[AICORE_TYPE_NUM]{0,0};

    bool preFetchSuccess_{false};
    DeviceTaskCtrl* preFetchNextDevTaskCtrl_{nullptr};

    std::array<int, MAX_AICORE_NUM> taskDfxStatPos_;

    SPSCQueue<DeviceTaskCtrl *, DEFAULT_QUEUE_SIZE> *taskQueue_{nullptr};
    AicpuTaskManager &aicpuTaskManager_;
    AiCoreProf aicoreProf_;
    int64_t dotStatus_{0};

    std::vector<TaskInfo> sendTask_[MAX_AICORE_NUM];
    std::vector<TaskInfo> recvFinTask_[MAX_AICORE_NUM];
    std::vector<TaskInfo> recvAckTask_[MAX_AICORE_NUM];

    bool isNeedWriteRegForFastPath_{true};
    AicoreLogger *logger_{nullptr};
    friend class AiCoreProf;
};
}
