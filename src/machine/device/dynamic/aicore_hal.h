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
 * \file aicore_hal.h
 * \brief
 */

#pragma once

#include "tilefwk/common_def.h"
#include "machine/device/dynamic/aicore_prof.h"
#include "machine/device/dynamic/costmodel_utils.h"

namespace npu::tile_fwk::dynamic {
constexpr uint32_t MAX_AICORE_NUM = 75;
constexpr uint32_t NAX_AIV_TOTAL_NUM = 50;
const uint32_t CORE_NUM_PER_AI_CORE = 3;

constexpr uint32_t NUM_ONE = 1;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t NUM_THREE = 3;
constexpr uint32_t NUM_FOUR = 4;
constexpr uint32_t NUM_FIVE = 5;
constexpr uint32_t NUM_THIRTY_TWO = 32;

const int32_t CORE_QUEUE_MODE_NUM_8 = 8;
const int32_t CORE_QUEUE_MODE_NUM_7 = 7;
const int32_t CORE_QUEUE_MODE_NUM_6 = 6;
const int32_t CORE_QUEUE_MODE_NUM_5 = 5;
const int32_t CORE_QUEUE_MODE_NUM_4 = 4;
const int32_t CORE_QUEUE_MODE_NUM_3 = 3;
const int32_t CORE_QUEUE_MODE_NUM_2 = 2;
const int32_t CORE_QUEUE_MODE_NUM_1 = 1;

const uint32_t REG_SPR_DATA_MAIN_BASE = 0xA0; // 0xA0 -> DATA_MAIN_BASE
const uint32_t REG_SPR_COND = 0x4C8;          // 0x4C8 -> COND SPR
const uint32_t REG_SPR_MAGIC = 0x78;
constexpr int32_t AICORE_COREID_MASK = 0x0FFF;

class AicoreHAL {
public:
    inline void Init(DeviceArgs *deviceArgs, AiCoreProf *aicoreProf) {
        aicoreProf_ = aicoreProf;
        sharedBuffer_ = deviceArgs->sharedBuffer;
        regAddrs_ = reinterpret_cast<int64_t *>(deviceArgs->coreRegAddr);
        readyRegQueues_.fill(nullptr);
        finishRegQueues_.fill(nullptr);
        blockIdToPhyCoreId_.fill(-1);
        args_.fill(nullptr);
    }

    inline void SetModel(uint64_t costModel) {
        costModel_ = reinterpret_cast<CostModel::AiCoreModel*>(costModel);
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
        if (costModel_) {
            costModel_->SendTask(coreIdx, taskId);
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
            aicoreProf_->ProfGet(coreIdx, stat->subGraphId, stat->taskId,
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
        } else {
            if (costModel_) {
                costModel_->InitData(coreIdx, funcdata);
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

    AiCoreProf *aicoreProf_{nullptr};
    CostModel::AiCoreModel *costModel_{nullptr};
};
}
