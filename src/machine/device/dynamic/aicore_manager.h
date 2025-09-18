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
#include "device_context.h"
#include <semaphore.h>
#include "device_utils.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "machine/utils/dynamic/small_array.h"
#include "machine/utils/dynamic/spsc_queue.h"
#include "interface/schema/schema.h"
#include "machine/kernel/aicore.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/utils/device_log.h"
#include "machine/device/distributed/comm_wait_flag.h"
#include "machine/device/dynamic/aicpu_task_manager.h"
#include "interface/operation/opcode.h"
#include "securec.h"
#include "machine/device/dynamic/aicore_prof.h"
#include "machine/device/aicore_dump.h"
#include "interface/utils/common.h"
#include "tilefwk/config.h"

namespace npu::tile_fwk::dynamic {
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
const uint32_t READY_ID_FIX_CACHE_NUM = 2048;
const uint32_t AICORE_TYPE_NUM = 2;

constexpr uint32_t MAX_AICORE_NUM = 75;
constexpr uint32_t NAX_AIV_TOTAL_NUM = 50;
constexpr uint32_t  MAX_MANAGER_AIV_NUM = (NAX_AIV_TOTAL_NUM / MAX_SCHEDULE_AICPU_NUM) + 1;

constexpr uint32_t REG_31_BITS = 0x7FFFFFFF;
constexpr uint32_t REG_32_BITS = 0xFFFFFFFF;
#define REG_LOW_TASK_ID(regVal) (regVal) & REG_31_BITS // 低31位存储的taskid
#define REG_LOW_TASK_STATE(regVal) ((regVal)&REG_32_BITS) >> 31 // 低32位存储的task的状态
constexpr uint32_t TASK_FIN_STATE = 1; // 任务执行完成完成
constexpr uint32_t TASK_ACK_STATE = 0; // 收到任务状态，没执行完成
constexpr uint32_t REG_TASK_NUM = 2; // 一次寄存器task个数

constexpr uint32_t NUM_ONE = 1;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t NUM_THREE = 3;
constexpr uint32_t NUM_FOUR = 4;
constexpr uint32_t NUM_FIVE = 5;
constexpr uint32_t NUM_THIRTY_TWO = 32;

constexpr uint32_t DEFAULT_QUEUE_SIZE = 64;

const int32_t CORE_QUEUE_MODE_NUM_8 = 8;
const int32_t CORE_QUEUE_MODE_NUM_7 = 7;
const int32_t CORE_QUEUE_MODE_NUM_6 = 6;
const int32_t CORE_QUEUE_MODE_NUM_5 = 5;
const int32_t CORE_QUEUE_MODE_NUM_4 = 4;
const int32_t CORE_QUEUE_MODE_NUM_3 = 3;
const int32_t CORE_QUEUE_MODE_NUM_2 = 2;
const int32_t CORE_QUEUE_MODE_NUM_1 = 1;
const int32_t CORE_QUEUE_MODE_NUM_0 = 0;

constexpr int32_t AICORE_COREID_MASK = 0x0FFF;
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
    std::atomic<int> refcnt{-1};
    std::atomic<int> runcnt{0};
    void *ctx{nullptr};
    FinishCallback finish{nullptr};
    int retCode{0};
    std::array<std::array<std::atomic<bool>, MAX_SCHEDULE_AICPU_NUM>, AICORE_TYPE_NUM>  isAicpuIdle;

    inline bool IsFree() { return refcnt.load(std::memory_order_relaxed) == -1; }

    void PutTask(int ret) {
        if (ret != 0)
            retCode = ret;

        // sync point, ensure all aiore_manager threads task finished
        runcnt.fetch_sub(1, std::memory_order_relaxed);
        while (runcnt.load(std::memory_order_relaxed) != 0)
            ;

        auto cnt = refcnt.fetch_sub(1, std::memory_order_release);
        if (cnt == 1) {
            if (finish) {
                finish(devTask, ctx);
            }
            refcnt.store(-1, std::memory_order_release);
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

class AicoreHAL {
public:
    inline void Init(DeviceArgs *deviceArgs, AiCoreProf *prof) {
        prof_ = prof;
        sharedBuffer_ = deviceArgs->sharedBuffer;
        regAddrs_ = reinterpret_cast<int64_t *>(deviceArgs->coreRegAddr);
        readyRegQueues_.fill(nullptr);
        finishRegQueues_.fill(nullptr);
        blockIdToPhyCoreId_.fill(-1);
        args_.fill(nullptr);
    }

    inline void SetModel(uint64_t model) {
        model_ = reinterpret_cast<CostModel::AiCoreModel*>(model);
    }

    int64_t *GetRegAddrs() const { return regAddrs_; }

    inline uint32_t ReadReg32(int coreIdx, int offset) {
        auto idx = GetPhyIdByBlockId(coreIdx);
        if (idx != -1) {
          return *(reinterpret_cast<volatile uint32_t*>(regAddrs_[idx] + offset));
        }
        return 0;
    }

    inline void WriteReg32(int coreIdx, int offset, uint32_t val) {
        auto idx = GetPhyIdByBlockId(coreIdx);
        if (idx != -1) {
          *(reinterpret_cast<volatile uint32_t*>(regAddrs_[idx] + offset)) = val;
        }
        return;
    }

    inline void WriteReg32All(int aicNum, int aivNum, int offset, uint32_t val) {
        for (int i = 0; i < aicNum + aivNum; ++i) {
            if (regAddrs_[i] != 0) {
                *(reinterpret_cast<volatile uint32_t *>(regAddrs_[i] + offset)) = val;
            }
        }
    }

    inline bool IsSpecialTask(uint32_t taskId) {
        return taskId == AICORE_TASK_INIT || taskId == AICORE_TASK_STOP || taskId == AICORE_FUNC_STOP;
    }

    inline void SetReadyQueue(int coreIdx, uint64_t value) {
        if constexpr (IsDeviceMode()) {
            *readyRegQueues_[GetPhyIdByBlockId(coreIdx)] = value;
        } else {
            DEV_INFO("set coreidx %d value %lx.", coreIdx, value);
            auto taskId = value - 1;
            if (value == 0 || taskId == AICORE_TASK_STOP || taskId == AICORE_FUNC_STOP) return;
            CostModelSendTask(coreIdx, taskId);
        }
    }

    inline void SetReadyQueue(int coreStart, int coreEnd, uint32_t val) {
        if constexpr (IsDeviceMode()) {
            int i, idx = coreStart;
            int n = coreEnd - coreStart;
            for (i = 0; i < (n & (~CORE_QUEUE_MODE_NUM_7)); i += CORE_QUEUE_MODE_NUM_8) {
                *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
            }
            switch (n & CORE_QUEUE_MODE_NUM_7) {
                case CORE_QUEUE_MODE_NUM_7:
                    *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_6:
                    *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_5:
                    *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_4:
                    *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_3:
                    *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_2:
                    *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_1:
                    *readyRegQueues_[GetPhyIdByBlockId(idx++)] = val;
                    [[fallthrough]];
                default:
                    break;
            }
        }
    }

    inline void SetReadyQueue(const uint32_t *coreIdx, const uint32_t *vals, int n) {
        if constexpr (IsDeviceMode()) {
            for (int i = 0; i < (n & (~CORE_QUEUE_MODE_NUM_7)); i += CORE_QUEUE_MODE_NUM_8) {
                *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
                *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
                *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
                *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
                *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
                *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
                *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
                *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
            }
            switch (n & CORE_QUEUE_MODE_NUM_7) {
                case CORE_QUEUE_MODE_NUM_7:
                    *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_6:
                    *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_5:
                    *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_4:
                    *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_3:
                    *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_2:
                    *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_1:
                    *readyRegQueues_[GetPhyIdByBlockId(*coreIdx++)] = *vals++;
                    [[fallthrough]];
                default:
                    break;
            }
        } else {
            for (int i = 0; i < n; i++) {
                auto taskId = vals[i] - 1;
                if (IsSpecialTask(taskId))
                    continue;
                CostModelSendTask(coreIdx[i], taskId);
            }
        }
    }

    inline void GetFinishQueue(const uint32_t *coreIdx, uint32_t *vals, int n) {
        if constexpr (IsDeviceMode()) {
            for (int i = 0; i < (n & (~CORE_QUEUE_MODE_NUM_7)); i += CORE_QUEUE_MODE_NUM_8) {
                *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
            }
            switch (n & CORE_QUEUE_MODE_NUM_7) {
                case CORE_QUEUE_MODE_NUM_7:
                    *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_6:
                    *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_5:
                    *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_4:
                    *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_3:
                    *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_2:
                    *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                    [[fallthrough]];
                case CORE_QUEUE_MODE_NUM_1:
                    *vals++ = *finishRegQueues_[GetPhyIdByBlockId(*coreIdx++)];
                    [[fallthrough]];
                default:
                    break;
            }
        } else {
            for (int i = 0; i < n; i++) {
                vals[i] = CostModelGetTask(coreIdx[i]);
            }
        }
    }

    inline void WaitFinQueue(int coreStart, int coreEnd, uint64_t val) {
        for (int idx = coreStart; idx < coreEnd; idx++) {
            while (*finishRegQueues_[GetPhyIdByBlockId(idx)] != val)
                ;
        }
    }

    inline uint64_t GetFinishedTask(int coreIdx) {
        if constexpr (IsDeviceMode()) {
            return *(finishRegQueues_[GetPhyIdByBlockId(coreIdx)]);
        } else {
            return CostModelGetTask(coreIdx);
        }
    }

    void SetTaskTimeCost(std::function<uint64_t(uint64_t, uint64_t, uint64_t)> func) { getTaskTimeCost = func; }

    uint64_t CostModelGetTask(int coreIdx) {
        auto currentTime = GetCycles();
        DEV_DEBUG("CostModel AICore polling aicore %d, time %lu.", coreIdx, currentTime);
        if (taskIds[coreIdx].empty()) return AICORE_FUNC_STOP | AICORE_FIN_MASK;
        uint64_t taskId;
        while (!taskIds[coreIdx].empty() && currentTime >= taskTimes[coreIdx].front()) {
            taskId = taskIds[coreIdx].front();
            taskTimes[coreIdx].pop_front();
            taskIds[coreIdx].pop_front();
        }
        if (taskIds[coreIdx].empty()) {
            DEV_DEBUG("CostModel AICore %d finish task 0x%lx, current time %lu.", coreIdx, taskId, currentTime);
            return taskId | AICORE_FIN_MASK;
        }
        DEV_DEBUG("CostModel AICore %d running task 0x%lx, current time %lu, finish time %lu.",
                  coreIdx, taskIds[coreIdx].front(), currentTime, taskTimes[coreIdx].front());
        return taskIds[coreIdx].front();
    }

    void CostModelSendTask(int coreIdx, uint64_t taskId) {
        uint64_t time = taskIds[coreIdx].empty() ? GetCycles() : taskTimes[coreIdx].back();
        uint64_t timeCost = getTaskTimeCost == nullptr ? 0 : getTaskTimeCost(coreIdx, taskId, time);
        taskTimes[coreIdx].push_back(time + timeCost);
        taskIds[coreIdx].push_back(taskId);
        if (model_) {
            model_->SendTask(coreIdx, taskId);
        }
        DEV_DEBUG("CostModel AICore %d add task 0x%lx, new queue size %lu, finish time %lu.",
                  coreIdx, taskId, taskIds[coreIdx].size(), time + timeCost);
    }

    inline void SendTaskBatch(int coreIdx, uint64_t regVal, uint64_t taskData) {
        if constexpr (IsDeviceMode()) {
            volatile KernelArgs *arg = args_[coreIdx];
            arg->shakeBuffer[SHAK_BUF_BATCH_TASK_INDEX] = static_cast<int64_t>(taskData);
        }
        __sync_synchronize();
        SetReadyQueue(coreIdx, regVal);
    }

    int64_t GetSharedBuffer() { return sharedBuffer_; }

    inline void MapRegistersForAllCores(int aicNum) {
        for (uint32_t idx = 0; idx < static_cast<u_int32_t>(aicNum * CORE_NUM_PER_AI_CORE); idx++) {
            void *addr = reinterpret_cast<void *>(regAddrs_[idx]);
            if (addr == nullptr) {
                continue;
            }
            DEV_DEBUG("phy core %u Addr is %p.", idx, addr);
            volatile uint64_t *reqQueueReg =
                reinterpret_cast<volatile uint64_t *>(static_cast<uint8_t *>(addr) + REG_SPR_DATA_MAIN_BASE);
            readyRegQueues_[idx] = reqQueueReg;
            volatile uint64_t *finishQueueReg =
                reinterpret_cast<volatile uint64_t *>(static_cast<uint8_t *>(addr) + REG_SPR_COND);
            finishRegQueues_[idx] = finishQueueReg;
        }
    }

    inline int &GetPhyIdByBlockId(int coreIdx) {
        return blockIdToPhyCoreId_[coreIdx];
    }

    int DumpTaskProf(int coreIdx) {
        volatile KernelArgs *arg = (KernelArgs *)(sharedBuffer_ + coreIdx * SHARED_BUFFER_SIZE);
        volatile Metrics*  metric = (Metrics *)(arg->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
        DEV_INFO("aicore %d host alloc metric memory :%p.", coreIdx, metric);
        if (metric == nullptr) {
            DEV_INFO("aicore %d Null metric.", coreIdx);
           return 0;
        }

        uint64_t cycles_start = GetCycles();
        while (metric->isMetricStop != 1) {
            if (GetCycles() - cycles_start > PROF_DUMP_TIMEOUT_CYCLES) {
                DEV_ERROR("wait metrics done timeout !!!.");
                return DEVICE_MACHINE_ERROR;
            }
        }; // wait aicore dcci metric data finish

        DEV_INFO("Dump core %d prof data , task cnt %ld, metric:%p.", coreIdx, metric->taskCount, metric);
        for (int i = 0; i < metric->taskCount; i++) {
            volatile TaskStat *stat = &metric->tasks[i];
            prof_->ProfGet(coreIdx, stat->subGraphId, stat->taskId,
                          &((Metrics *)(arg->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]))->tasks[i]);
            DEV_INFO("  Dump prof for task %d, execstart: %ld execend :%ld.",
                     stat->taskId, stat->execStart, stat->execEnd);
        }
        return 0;
    }

    void DumpAicoreStatus(int coreIdx) const {
        volatile KernelArgs *arg = (KernelArgs *)(sharedBuffer_ + coreIdx * SHARED_BUFFER_SIZE);
        DEV_INFO("!!***********************aicore %d last status **************************!!", coreIdx);
        DEV_INFO("hello status %ld.", arg->shakeBuffer[0]);
        DEV_INFO("last_taskId %ld task status [%ld, %ld, %ld, %ld].", arg->shakeBuffer[NUM_ONE],
            arg->shakeBuffer[NUM_TWO], arg->shakeBuffer[NUM_THREE], arg->shakeBuffer[NUM_FOUR], arg->shakeBuffer[NUM_FIVE]);

        for (size_t i = 0; i < sizeof(arg->taskStat) / sizeof(TaskStat); i++) {
            DEV_INFO("task rsp index %lu: taskId %d, subGraphID %d execStart %ld execEnd %ld.", i,
                arg->taskStat[i].taskId, arg->taskStat[i].subGraphId,
                arg->taskStat[i].execStart, arg->taskStat[i].execEnd);
        }
    }

    uint64_t GetAicoreStatus(int coreIdx) const {
        volatile KernelArgs *arg = (KernelArgs *)(sharedBuffer_ + coreIdx * SHARED_BUFFER_SIZE);
        return arg->shakeBuffer[0x2];
    }

    inline void InitTaskData(int coreIdx, int64_t funcdata) {
        if constexpr (IsDeviceMode()) {
            volatile KernelArgs *arg = args_[coreIdx];
            arg->shakeBuffer[SHAK_BUF_COREFUNC_DATA_INDEX] = funcdata;
        }else{
            if (model_) {
                model_->InitData(coreIdx, funcdata);
            }
        }
    }

    int HandShake(int coreIdx, int64_t dotStatus) {
        int ret = DEVICE_MACHINE_OK;
        auto args =
            reinterpret_cast<KernelArgs*>((static_cast<uint64_t>(sharedBuffer_)) + SHARED_BUFFER_SIZE * coreIdx);
        args->taskEntry.reserved[0] = static_cast<uint32_t>(dotStatus);
        volatile int64_t *shakeBuffer = args->shakeBuffer;
        uint32_t cycles_start = GetCycles();
        while ((*shakeBuffer & 0xFFFFFFFF) != AICORE_SAY_HELLO) {
            if (GetCycles() - cycles_start > HAND_SHAKE_TIMEOUT) {
                DEV_ERROR("hand shake %d timeout.\n", coreIdx);
                return -1;
            }
        }
        args_[coreIdx] = args;
        GetPhyIdByBlockId(coreIdx) = (*shakeBuffer >> NUM_THIRTY_TWO) & AICORE_COREID_MASK;
        return ret;
    }

    void ResetShakeBuf(int coreIdx) {
        args_[coreIdx]->shakeBuffer[0] = 0;
        args_[coreIdx]->shakeBuffer[SHAK_BUF_COREFUNC_DATA_INDEX] = 0;
        return;
    }

    volatile TaskStat *GetTaskStat(int coreIdx, int pos) {
        volatile TaskStat *stat = &args_[coreIdx]->taskStat[pos];
        return stat;
    }
private:
    int64_t sharedBuffer_;

    int64_t* regAddrs_{nullptr};

    std::array<volatile KernelArgs*, MAX_AICORE_NUM> args_;

    std::array<volatile uint64_t*, MAX_AICORE_NUM> readyRegQueues_;
    std::array<volatile uint64_t*, MAX_AICORE_NUM> finishRegQueues_;

    // cost model aicore
    std::function<uint64_t(uint64_t, uint64_t, uint64_t)> getTaskTimeCost{nullptr};
    std::array<std::deque<uint64_t>, MAX_AICORE_NUM> taskIds;
    std::array<std::deque<uint64_t>, MAX_AICORE_NUM> taskTimes;

    std::array<int, MAX_AICORE_NUM> blockIdToPhyCoreId_;

    AiCoreProf *prof_{nullptr};

    CostModel::AiCoreModel *model_{nullptr};
};
class AiCoreManager {
public:
    explicit AiCoreManager(AicpuTaskManager &aicpuTaskManager) : aicpuTaskManager_(aicpuTaskManager), prof_(*this){};
    ~AiCoreManager(){};

    inline void InitTaskData(DeviceTaskCtrl *taskCtrl) {
        curTaskCtrl_ = taskCtrl;
        curDevTask_ = taskCtrl->devTask;
        curTaskType_ = taskCtrl->taskType;
        curTaskId_ = taskCtrl->taskId;
        aicoreHAL.SetModel(taskCtrl->devTask->aicoreModel);
        int64_t funcdata;
        if (IsStaticFunction()) {
            funcdata = (int64_t)&curDevTask_->coreFuncData;
        } else {
            auto dyntask = (DynDeviceTask *)curDevTask_;
            funcdata = static_cast<int64_t>(PtrToValue(dyntask->dynFuncData));
        }
        ForEachManageAicore([&](int coreIdx) { aicoreHAL.InitTaskData(coreIdx, funcdata); });

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

#if DEBUG_SWITCH
        __sync_fetch_and_add(&(taskCtrl->finishedAicFunctionCnt), sentAic);
        __sync_fetch_and_add(&(taskCtrl->finishedAivFunctionCnt), sentAiv);
        __sync_fetch_and_add(&(taskCtrl->finishedAicpuFunctionCnt), sent);
        __sync_fetch_and_add(&(taskCtrl->finishedHubFunctionCnt), resolveHubCnt_);
        procAicCoreFunctionCnt_ += sentAic;
        procAivCoreFunctionCnt_ += sentAiv;
        procAicpuFunctionCnt_ += sent;
        DEV_DEBUG("finish send  aic task cnt: %lu,  aiv task cnt: %lu, hub task cnt:%lu,"
            "aicpu task cnt:%lu, target totalcnt: %lu.",
            taskCtrl->finishedAicFunctionCnt, taskCtrl->finishedAivFunctionCnt,
            taskCtrl->finishedHubFunctionCnt, taskCtrl->finishedAicpuFunctionCnt, curDevTask_->coreFunctionCnt);
#endif
        sent += (sentAic + sentAiv + resolveHubCnt_);
        resolveHubCnt_ = 0;
        return sent;
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
                ret = DEVICE_MACHINE_ERROR;
                goto FINISH;
            }
            (void)start;
        }
        PerfMtBegin(PERF_EVT_WAIT_AICORE_FINISH, aicpuIdx_);
        rc = WaitAllAicoreFinish(aicStart_, aicEnd_);
        if (rc != DEVICE_MACHINE_OK) {
            ret = rc;
        }
        rc = WaitAllAicoreFinish(aivStart_, aivEnd_);
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
        uint64_t status = aicoreHAL.GetAicoreStatus(coreIdx);
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
          if (aicoreHAL.ReadReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE) == REG_SPR_FAST_PATH_OPEN) {
            aicoreHAL.WriteReg32(coreIdx, REG_SPR_DATA_MAIN_BASE, AICORE_TASK_STOP + 1);
            aicoreHAL.WriteReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
          }
      });
    }

    inline int Run(int threadIdx, DeviceArgs *deviceArgs, DeviceTaskCtrl *taskCtrl = nullptr) {
        int ret = 0;
        Init(threadIdx, deviceArgs);
        if constexpr (IsDeviceMode()) {
            ret = HandShake();
            if (ret != DEVICE_MACHINE_OK) {
                DEV_ERROR("hand shake timeout.");
                AbnormalStop();
                do {
                    taskCtrl->PutTask(ret);
                } while ((taskCtrl = taskQueue_.Dequeue()));
                return ret;
            }
            prof_.ProfStart();
        }
        if (taskCtrl != nullptr) {
            ret = RunTask(taskCtrl);
        } else {
            while (ret == 0) {
                taskCtrl = taskQueue_.Dequeue();
                if (taskCtrl == nullptr)
                    break;

                PROF_STAGE_BEGIN_MTSAFE(PERF_EVT_STAGE_SCHEDULE, threadIdx, "dispatch.before\n");

                PerfMtBegin(PERF_EVT_RUN_TASK, threadIdx);
                ret = RunTask(taskCtrl);
                PerfMtEnd(PERF_EVT_RUN_TASK, threadIdx);
                DEV_DEBUG("run task finish taskid=%d ret %d.", curTaskId_, ret);
                if (ret != 0)
                    break;

                PerfMtBegin(PERF_EVT_SYNC_AICORE, threadIdx);
                SyncAiCore(taskCtrl->taskId);
                PerfMtEnd(PERF_EVT_SYNC_AICORE, threadIdx);
                DEV_DEBUG("sync finish.");
                taskCtrl->PutTask(ret);
                PROF_STAGE_END_MTSAFE(PERF_EVT_STAGE_SCHEDULE, threadIdx, "dispatch.after\n");
            }
            if (ret) {
                DEV_ERROR("task %lu execute errror, skip rest tasks.", taskCtrl->taskId);
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
            ProfStop();
        }
        DEV_DEBUG("Aicpu %d stop ret = %d, proc aic task cnt: %lu,  aiv task cnt: %lu.",
            aicpuIdx_,
            ret,
            procAicCoreFunctionCnt_,
            procAivCoreFunctionCnt_);
        return ret;
    }

    void PushTask(DeviceTaskCtrl *taskCtrl) { taskQueue_.Enqueue(taskCtrl); }
private:
    inline void DumpTaskProf() {
        ForEachManageAicoreWithRet([this] (int coreIdx) -> int { return aicoreHAL.DumpTaskProf(coreIdx);});
    }

    inline void ProfStop() {
        if (prof_.ProfIsEnable()) {
#if PROF_DFX_HOST_PREPARE_MEMORY_MODE
            DumpTaskProf();
#endif
        }

        prof_.ProfStop();
    }

    inline void DumpAiCoreStatus() const {
#if defined(DEBUG_SWITCH) && DEBUG_SWITCH
        ForEachManageAicore([this](int coreIdx) {
            if constexpr (IsDeviceMode()) {
                aicoreHAL.DumpAicoreStatus(coreIdx);
            }
            DEV_INFO("reg low task: runningid(%u) pendingid(%u) dfxpos(%d).", runningIds_[coreIdx],
                pendingIds_[coreIdx], taskDfxStatPos_[coreIdx]);

            DEV_INFO("send task info ~~~~~~~~~~~~~~~~~~~~~~~~~~~~count:%lu~~~~~~~~~~~~~~~~~~~~~~~~~~~~.",
                sendTask_[coreIdx].size());
            for (size_t i = 0; i < sendTask_[coreIdx].size(); i++) {
                DEV_INFO("send task: seqno %d, taskId %lx.", (int)i, sendTask_[coreIdx][i].taskId);
            }

            DEV_INFO("recv finish task info ~~~~~~~~~~~~~~~~~~~~~~~~~count:%lu~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~.",
                recvFinTask_[coreIdx].size());
            for (size_t i = 0; i < recvFinTask_[coreIdx].size(); i++) {
                DEV_INFO("recv task: seqno %d, taskId %lx.", (int)i, recvFinTask_[coreIdx][i].taskId);
            }

            DEV_INFO("recv ack task info ~~~~~~~~~~~~~~~~~~~~~~~~~~~~count:%lu~~~~~~~~~~~~~~~~~~~~~~~~~~~~.",
                recvAckTask_[coreIdx].size());
            for (size_t i = 0; i < recvAckTask_[coreIdx].size(); i++) {
                DEV_INFO("recv ack task: seqno %d, taskId %lx.", static_cast<int>(i), recvAckTask_[coreIdx][i].taskId);
            }
        });
#endif
    }

    inline void DumpTaskTensor(int &coreIdx, volatile TaskStat *stat) {
        if (!IsStaticFunction()) {
            return; // not support currently
        }
        auto coreId = aicoreHAL.GetPhyIdByBlockId(coreIdx);
        DEV_DEBUG("Output coreId is %d taskid is %d, with execStart: %ld, execend: %ld.", coreId, stat->taskId,
            stat->execStart, stat->execEnd);
        aicoreDump_.DumpInit(stat->subGraphId, stat->taskId, coreId, stat->execStart, stat->execEnd);
        auto funcInfo = GetFunctionWsAddr(stat->taskId);
        if (funcInfo) {
            aicoreDump_.DoDump(
                funcInfo->invokeEntryInfo, funcInfo->invokeEntryNum, funcInfo->invokeEntryAddr, "output");
        }
    }

    inline bool CheckTaskFinished(int coreIdx) {
        uint64_t finTaskVal = aicoreHAL.GetFinishedTask(coreIdx);
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
            runningIds_[coreIdx] = AICORE_TASK_INIT;
        }

        return pendingIds_[coreIdx] == AICORE_TASK_INIT && runningIds_[coreIdx] == AICORE_TASK_INIT;
    }

    inline int WaitAllAicoreFinish(int coreIdxStart, int coreIdxEnd) {
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
                    return DEVICE_MACHINE_ERROR;
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
            DEV_DEBUG("AiCpud:%d, can not send task currently. ready Task: 0.", aicpuIdx_);
            return 0;
        }

        uint32_t ready = GetReadyCoreNum(type);
        if (ready == 0 ) {
            DEV_DEBUG("AiCpud:%d, can not send task currently. ready Core: %u.", aicpuIdx_, ready);
            return 0;
        }
        PerfMtBegin(PERF_EVT_SEND_AIC_TASK, aicpuIdx_);
        uint32_t readyId[MAX_MANAGER_AIV_NUM];
        ReadyQueueLock(readyQue);
        uint32_t head = __atomic_load_n(&readyQue->head, __ATOMIC_RELAXED);
        uint32_t tail = __atomic_load_n(&readyQue->tail, __ATOMIC_RELAXED);
        uint32_t taskCount = std::min(ready, tail - head);
        if (taskCount == 0) {
            DEV_DEBUG("AiCpud:%u, taskCount is zero.", head);
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
        DEV_DEBUG("AiCpud:%d, pop all new task count: %u.", aicpuIdx_, taskCount);
        BatchSendTask(type, isRealLifo ? &readyId[taskCount - 1] : &readyQue->elem[head],
            taskCount, coreIdxStart, coreIdxEnd, isRealLifo);
        DEV_DEBUG("core ready cnt: %u.", corePendReadyCnt_[static_cast<int>(type)]);
        firstLock[static_cast<int>(type)] = false;
        PerfMtEnd(PERF_EVT_SEND_AIC_TASK, aicpuIdx_);
        return taskCount;
    }

    inline uint32_t BatchSendTask(CoreType type, uint32_t *newTask, uint32_t taskCount,
        int coreIdxStart, int coreIdxEnd, bool isLifo) {
        uint32_t sendCnt = 0;
        uint32_t coreRunReadyCnt = coreRunReadyCnt_[static_cast<int>(type)];
        DEV_DEBUG("Begin Batch send %s task: corerunreadycnt:%u, pendreadyCnt:%u, taskCount:%u.",
            type == CoreType::AIC ? "AIC": "AIV", coreRunReadyCnt,
            corePendReadyCnt_[static_cast<int>(type)], taskCount);
        while (sendCnt < static_cast<uint64_t>(coreRunReadyCnt) && sendCnt < taskCount) {
            DEV_DEBUG("  ## send task use runready core %u.",
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
        DEV_DEBUG("  ## send task left pend ready cnt %u , last core index:%u.",
            corePendReadyCnt_[static_cast<int>(type)], idx);
        while (corePendReadyCnt_[static_cast<int>(type)] > 0 && sendCnt < taskCount) {
            if (pendingIds_[idx] == AICORE_TASK_INIT) {
                DEV_DEBUG("  ## send task use pendready core %u.", idx);
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
        DEV_DEBUG("  ## finish send task left runreadycnt:%u pendreadycnt %u, last coreindex:%u.",
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
        aicoreHAL.SetReadyQueue(coreIdx, (newTask + 1) & 0xFFFFFFFF);
        pendingIds_[coreIdx] = newTask;
        sendCnt_[static_cast<int>(type)]++;

#if DEBUG_SWITCH
        if (IsStaticFunction()) {
            DEV_DEBUG("Start to dump input tensor info, num is.");
            auto funcInfo = GetFunctionWsAddr(newTask);
            if (funcInfo) {
                aicoreDump_.DumpInit(funcInfo->psgId, newTask, aicoreHAL.GetPhyIdByBlockId(coreIdx));
                aicoreDump_.DoDump(funcInfo->invokeEntryInfo, funcInfo->invokeEntryNum, funcInfo->invokeEntryAddr, "input");
            }
        } else {
            // dynamic function, not support currently
        }

        sendTask_[coreIdx].push_back(TaskInfo(coreIdx, newTask));
#endif
        DEV_DEBUG("Send task %lu, at core %d ,type:%d.", newTask, coreIdx, static_cast<int>(type));
    }

    inline void SetAiCpuStat(int coreIdx, uint64_t taskId) {
        struct AiCpuTaskStat aiCpuTaskStat;
        aiCpuTaskStat.taskId = taskId;
        aiCpuTaskStat.coreId = aicoreHAL.GetPhyIdByBlockId(coreIdx);
        prof_.AsmCntvc(aiCpuTaskStat.taskGetStart);
        prof_.SetAiCpuTaskStat(taskId, aiCpuTaskStat);
    };

    CoreFunctionWsAddr *GetFunctionWsAddr(uint32_t id) {
        if (unlikely(IsStaticFunction())) {
            auto wsAddr = curDevTask_->coreFuncData.coreFunctionWsAddr;
            return &reinterpret_cast<CoreFunctionWsAddr *>(wsAddr)[TaskID(id)];
        } else {
            return nullptr;
        }
    }

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
                ResolveByRegVal(type, i, aicoreHAL.GetFinishedTask(i));
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
            DEV_DEBUG("resolved new task, aic ready count: %u coretype:%u.", readyCount[aicIndex], aicIndex);
            if (readyCount[aicIndex] > 0) {
                PushReadyQue(readyAicCoreFunctionQue_, readyIds[aicIndex], readyCount[aicIndex]);
            }
            readyCount[aicIndex] = 0;
        }

        if (readyCount[aivIndex] > 0) {
            uint32_t needSendCnt = std::min(GetReadyCoreNum(CoreType::AIV), readyCount[aicIndex]);
            if (needSendCnt > 0) {
                readyCount[aivIndex] -= BatchSendTask(CoreType::AIV, &readyIds[aivIndex][readyCount[aivIndex] - 1],
                    needSendCnt, aivStart_, aivEnd_, true);
            }
            DEV_DEBUG("resolved new task, aiv ready count: %u coretype: %u.", readyCount[aivIndex], aivIndex);
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
            DEV_DEBUG("core index: %d, PendingTask Finished."
                " pending: %x.", coreIdx, pendingIds_[coreIdx]);
            ResolveDepWithDfx(type, coreIdx, finTaskId);
            pendingIds_[coreIdx] = AICORE_TASK_INIT;
            corePendReadyCnt_[static_cast<int>(type)]++;
            runReadyCoreIdx_[static_cast<int>(type)][coreRunReadyCnt_[static_cast<int>(type)]++] = coreIdx;
        }
    }

    inline void ResolveByRegVal(CoreType type, int coreIdx, uint64_t finTaskRegVal) {
        uint32_t finTaskId = REG_LOW_TASK_ID(finTaskRegVal);
        uint32_t finTaskState = REG_LOW_TASK_STATE(finTaskRegVal);
        DEV_DEBUG("reslove task core index: %d, finishtaskid:%x, finishstate: %u.", coreIdx, finTaskId, finTaskState);
#if SCHEDULE_USE_PENDING_AND_RUNING_SWITCH
        auto &pendId = pendingIds_[coreIdx];
        auto &runId = runningIds_[coreIdx];
        uint32_t tmpTaskId;
        if (likely(finTaskId == pendId && finTaskState == TASK_FIN_STATE)) {
            DEV_DEBUG("PendingTask Finished.runningid: %x.", runId);
            tmpTaskId = runId;
            runId = AICORE_TASK_INIT;
            pendId = AICORE_TASK_INIT; // ResolveDepWithDfx depend this line
            runReadyCoreIdx_[static_cast<int>(type)][coreRunReadyCnt_[static_cast<int>(type)]++] = coreIdx;
            corePendReadyCnt_[static_cast<int>(type)]++;
            if (tmpTaskId != AICORE_TASK_INIT) {
                ResolveDepWithDfx(type, coreIdx, tmpTaskId);
            }
            ResolveDepWithDfx(type, coreIdx, finTaskId);
        } else if (finTaskId == pendId && finTaskState == TASK_ACK_STATE) {
#if defined(DEBUG_SWITCH) && DEBUG_SWITCH
            recvAckTask_[coreIdx].push_back(TaskInfo(coreIdx, finTaskId));
#endif
            DEV_DEBUG("PendingTask Acked. Running task finished.runningid: %x.", runId);
            tmpTaskId = runId;
            runId = finTaskId;
            pendId = AICORE_TASK_INIT; // ResolveDepWithDfx depend this line
            corePendReadyCnt_[static_cast<int>(type)]++;
            if (tmpTaskId != AICORE_TASK_INIT) {
                ResolveDepWithDfx(type, coreIdx, tmpTaskId);
            }
        } else if (finTaskId == runId && finTaskState == TASK_FIN_STATE) {
            DEV_DEBUG("core index: %d, RuningTask Finished. pending: %x, running: %x.",
            coreIdx, pendId, runId);
            runId = AICORE_TASK_INIT;
            if (pendId == AICORE_TASK_INIT) {
                runReadyCoreIdx_[static_cast<int>(type)][coreRunReadyCnt_[static_cast<int>(type)]++] = coreIdx;
            }
            ResolveDepWithDfx(type, coreIdx, finTaskId);
        } else {
            DEV_DEBUG("Warning, maybe inconsistent state. coreidx: %d,finTask: %lx,pending: %x,running: %x.",
                coreIdx, finTaskRegVal, pendId, runId);
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
            DEV_DEBUG("Direct send task when task ready %x.", taskId);
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
        DEV_DEBUG("Direct send task when task ready %x.", taskId);
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

    inline void ResolveVirtualPure(uint64_t dep, CoreFunctionReadyState* readyState) {
        DEV_DEBUG("new virtual pure task resolved. id: %lu.", dep);
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
        DEV_DEBUG("new virtual mix task resolved. id: %lu.", dep);
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
                resolveHubCnt_++;
                ResolveDepStatic(depTaskId);
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
        DEV_DEBUG("new task resolved. id: %lu, coretype: %d.", depTaskId, coretype);
    }

    inline uint64_t GetCostModelTaskTime(uint64_t coreIdx, uint64_t taskId, uint64_t currentTime) {
        auto funcId = FuncID(taskId);
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto costModelData = reinterpret_cast<CostModel::ModelData*>(curDevTask_->costModelData);
        if (costModelData == nullptr) return 0;
        auto &funcDup = dyntask->stitchedList[funcId];
        auto opIndex = TaskID(taskId);
        auto leafFunctionIdx = funcDup.GetSource()->GetOperationAttrCalleeIndex(opIndex);
        auto timeCost = costModelData->functionTime[leafFunctionIdx];
        auto header = dyntask->dynFuncData;
        auto dyndata = reinterpret_cast<DynFuncData *>(header + 1);
        auto opAttrs = &dyndata->opAttrs[dyndata->opAtrrOffsets[TaskID(taskId)]];
        auto psgId = opAttrs[0];
        // devTaskId - funcId - leaf function Id - psgId
        std::string name = std::to_string(curTaskId_) + '-' + std::to_string(funcId) + '-' +
                           std::to_string(opIndex) + '-' + std::to_string(psgId);
        PerfMtEvent(PERF_EVT_TASK, coreIdx + PERF_AICORE_THREAD_START, currentTime, currentTime + timeCost, name);
        return timeCost;
    }

    inline void ResolveDynStitched(DynDeviceTask *dyntask, int origfunc, int origop) {
        auto &funcDup = dyntask->stitchedList[origfunc];
        auto &stitchList = funcDup.GetOperationStitch(origop);
        auto cceBinary = dyntask->cceBinary;

        for (auto *node = stitchList.Head(); node != nullptr; node = node->Next()) {
            uint32_t listSize = node->Size();
            for (uint32_t i = 0; i < listSize; i++) {
                uint32_t id = node->At(i);
                auto funcId = FuncID(id);
                auto opIndex = TaskID(id);
                auto predCounts = dyntask->cacheList[funcId].predCount;
                if (predCounts[opIndex] == 1 ||
                    __atomic_sub_fetch(&predCounts[opIndex], 1, __ATOMIC_RELAXED) == 0) {
                    auto callList = dyntask->cacheList[funcId].calleList;
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
        auto func = dyntask->cacheList[funcId].devFunc;
        return func->GetRootIndex();
    }

    inline int GetLeafIndex(uint32_t taskId) const {
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto callList = dyntask->cacheList[funcId].calleList;
        return callList[opIndex];
    }

    inline DevAscendFunctionDupped GetDuppedData(uint32_t taskId) const {
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto funcId = FuncID(taskId);
        return dyntask->cacheList[funcId].dup;
    }

    inline void ResolveDepDyn(uint64_t finishId) {
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto funcId = FuncID(finishId);
        auto opIndex = TaskID(finishId);

        auto cceBinary = dyntask->cceBinary;
        auto func = dyntask->cacheList[funcId].devFunc;
        auto predCounts =  dyntask->cacheList[funcId].predCount;
        auto callList = dyntask->cacheList[funcId].calleList;

        size_t succSize;
        auto succList = func->GetOperationDepGraphSuccAddr(opIndex, succSize);
        for (size_t i = 0; i < succSize; i++) {
            auto succIdx = succList[i];
            if (predCounts[succIdx] == 1 ||
                __atomic_sub_fetch(&predCounts[succIdx], 1, __ATOMIC_RELAXED) == 0) {
                auto id = MakeTaskID(funcId, succIdx);
                auto coreType = cceBinary[callList[succIdx]].coreType;
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

        ResolveDynStitched(dyntask, funcId, opIndex);
    }

    inline void ResolveDepStatic(uint64_t finishId) {
        auto readyState = reinterpret_cast<CoreFunctionReadyState *>(curDevTask_->coreFunctionReadyStateAddr);
        auto funcInfo =
            &(reinterpret_cast<CoreFunctionWsAddr *>(curDevTask_->coreFuncData.coreFunctionWsAddr)[finishId]);
        auto topo = reinterpret_cast<CoreFunctionTopo *>(funcInfo->topoAddr);
        DEV_DEBUG("resolve %lx, Dep core function num: %lu.", finishId, topo->depNum);

        for (uint64_t i = 0; i < topo->depNum; i++) {
            uint64_t dep = topo->depIds[i];
            int ret = __sync_add_and_fetch(&(readyState[dep].readyCount), 1);
            if (ret != 0) {
                continue;
            }
            ResolveByCoreType(readyState[dep].coreType, dep, readyState);
        }
    }

    inline void ResolveDepWithDfx(CoreType type, int coreIdx, uint64_t finishId) {
        if (unlikely(IsStaticFunction())) {
            ResolveDepStatic(finishId);
        } else {
            ResolveDepDyn(finishId);
        }
        DEV_DEBUG("[Call]: Core %d Dispatch Task: %lu, %u, %u", coreIdx, seq,
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
        aicoreHAL.Init(deviceArgs, &prof_);
        runningIds_.fill(AICORE_STATUS_INIT);
        pendingIds_.fill(AICORE_STATUS_INIT);
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
            aicoreHAL.MapRegistersForAllCores(aicNum_);
            prof_.ProfInit(reinterpret_cast<int64_t *>(deviceArgs->corePmuRegAddr),
               reinterpret_cast<int64_t *>(deviceArgs->pmuEventAddr));
        } else {
            aicoreHAL.SetTaskTimeCost([this](uint64_t coreIdx, uint64_t taskId, uint64_t time)
                {return GetCostModelTaskTime(coreIdx, taskId, time); });
        }
        ResetCnt();
        firstLock[static_cast<int>(CoreType::AIC)] = true;
        firstLock[static_cast<int>(CoreType::AIV)] = true;
        DEV_DEBUG("Init aicore manager aicNum_ %d aivNum_  %d sch_aicpuNum_ %d aicpuIdx_ %d "
                  "aicValidNum_ %d aicoreHAL.regAddrs_ %p sharedBuffer_ %p machineConfig: %u.",
            aicNum_, aivNum_, aicpuNum_, aicpuIdx_, aicValidNum_, aicoreHAL.GetRegAddrs(),
            (void *)aicoreHAL.GetSharedBuffer(), static_cast<uint8_t>(deviceArgs->machineConfig));
    }

    inline int HandShake() {
        DEV_INFO("Aicpu %d handshake start.", aicpuIdx_);
        int rc = ForEachManageAicoreWithRet([this](int coreIdx) -> int {
            int ret = aicoreHAL.HandShake(coreIdx, dotStatus_);
            DEV_DEBUG("coreidx %d handshake  phycorid %d.", coreIdx, aicoreHAL.GetPhyIdByBlockId(coreIdx));
            return ret;
        });
        if (rc != DEVICE_MACHINE_OK) {
            DEV_DEBUG("Aicpu %d handshake failed end.", aicpuIdx_);
            return rc;
        }

        ForEachManageAicore([this](int coreIdx) {
            aicoreHAL.WriteReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_OPEN);
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
        return aicoreHAL.GetPhyIdByBlockId(coreIdx);
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
            aicoreHAL.SetReadyQueue(aicStart_, aicEnd_, AICORE_FUNC_STOP + 1);
            aicoreHAL.SetReadyQueue(aivStart_, aivEnd_, AICORE_FUNC_STOP + 1);
            aicoreHAL.WaitFinQueue(aicStart_, aicEnd_, waitAckStopVal);
            aicoreHAL.WaitFinQueue(aivStart_, aivEnd_, waitAckStopVal);
            aicoreHAL.SetReadyQueue(aicStart_, aicEnd_, 0);
            aicoreHAL.SetReadyQueue(aivStart_, aivEnd_, 0);
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
        aicoreHAL.WriteReg32All(aicNum_, aivNum_, REG_SPR_DATA_MAIN_BASE, AICORE_TASK_STOP + 1);
        /* write to MAINBASE reg must be done before close 0x18 */
        aicoreHAL.WriteReg32All(aicNum_, aivNum_, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
        DEV_INFO("aicore manager %d abnormal stopped.", aicpuIdx_);
    }

    inline void NormalStop() {
        DEV_DEBUG("aicore manager %d try normal stop.", aicpuIdx_);
        ForEachManageAicore([this](auto coreIdx) { aicoreHAL.SetReadyQueue(coreIdx, AICORE_TASK_STOP + 1) ; });
        /* write to MAINBASE reg must be done before close 0x18 */
        __sync_synchronize();
        ForEachManageAicore([this](auto coreIdx) {
            aicoreHAL.WriteReg32(coreIdx, REG_SPR_FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
            aicoreHAL.ResetShakeBuf(coreIdx);
        });
        DEV_DEBUG("aicore manager %d normal stopped.", aicpuIdx_);
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

        volatile TaskStat *stat = aicoreHAL.GetTaskStat(coreIdx, 0);

#if PROF_DFX_HOST_PREPARE_MEMORY_MODE != 1
        prof_.ProfGet(coreIdx, stat->subGraphId, stat->taskId, const_cast<TaskStat*>(stat));
#endif

#if defined(DEBUG_SWITCH) && DEBUG_SWITCH
        DumpTaskTensor(coreIdx, stat);
        recvFinTask_[coreIdx].push_back(TaskInfo(coreIdx, taskId));
#endif

#if PROF_DFX_HOST_PREPARE_MEMORY_MODE != 1
        SetNextDfxPos(coreIdx); // pingpong 存储
#endif
    (void)stat;
    }

    inline bool IsStaticFunction() {
        return curTaskType_ == DEVICE_TASK_TYPE_STATIC;
    }

    inline bool IsNeedProcAicpuTask() {
        return aicpuIdx_ == 1;
    }
private:
    uint64_t seq;
    AicoreHAL aicoreHAL;
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
    AiCoreProf prof_;
    AicoreDump aicoreDump_;
    int64_t dotStatus_{0};
#if defined(DEBUG_SWITCH) && DEBUG_SWITCH
    std::vector<TaskInfo> sendTask_[MAX_AICORE_NUM];
    std::vector<TaskInfo> recvFinTask_[MAX_AICORE_NUM];
    std::vector<TaskInfo> recvAckTask_[MAX_AICORE_NUM];
#endif

    friend class AiCoreProf;
};
}
