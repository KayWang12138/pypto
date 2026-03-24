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
 * \file wrap_manager.h
 * \brief
 */

#pragma once
#include <cstdint>
#include "aicore_constants.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/device/dynamic/aicore_hal.h"
#include "machine/device/tilefwk/core_func_data.h"
namespace npu::tile_fwk::dynamic {

using SendTaskToAiCoreFunc = std::function<void(CoreType type, int coreIdx, uint64_t newTask)>;
using AddReadyCoreIdxFunc = std::function<void(int coreIdx, int type)>;

enum class MixResourceType {
    MIX_UNKNOWN = 0,
    MIX_1C1V = 1,
    MIX_1C2V = 2
};

enum class DieId {
    DIE_0 = 0,
    DIE_1 = 1,
    DIE_MIX = 2,
    DIE_UNKNOW
};

inline void WrapInfoQueueLock(WrapInfoQueue* rq) {
    while (!__sync_bool_compare_and_swap(&rq->lock, 0, 1)) {
    }
}

inline void WrapInfoQueueUnLock(WrapInfoQueue* rq) {
    while (!__sync_bool_compare_and_swap(&rq->lock, 1, 0)) {
    }
}

inline uint32_t GetTaskNumByMixResType(MixResourceType mixType) {
    switch (mixType) {
        case static_cast<uint32_t>(MixResourceType::MIX_1C1V):
            return 2;
        case static_cast<uint32_t>(MixResourceType::MIX_1C2V):
            return 3;
        default:
            return 0;
    }
}

inline bool IsMixTaskFinish(WrapInfo* wrapInfo) {
    switch (wrapInfo->mixResourceType) {
        case static_cast<uint32_t>(MixResourceType::MIX_1C1V):
            return wrapInfo->tasklist[0] == AICORE_TASK_STOP && wrapInfo->tasklist[1] == AICORE_TASK_STOP;
        case static_cast<uint32_t>(MixResourceType::MIX_1C2V):
            return wrapInfo->tasklist[0] == AICORE_TASK_STOP && wrapInfo->tasklist[1] == AICORE_TASK_STOP && wrapInfo->tasklist[2] == AICORE_TASK_STOP;
        default:
            DEV_ERROR("unexpected mixResourceType %u.", wrapInfo->mixResourceType);
            return false;
    }
}

#define RETURN_NULL_IF_NOT(val) \
    if (!val) {  \
        return;  \
    }

#define RETURN_RET_IF_NOT(val, ret) \
    if (!val) {  \
        return ret;  \
    }

const std::set<uint32_t> invalidTaskIds = {AICORE_TASK_INIT, AICORE_SAY_ACK, AICORE_TASK_STOP};

class WrapManager {
public:
    ~WrapManager(){};
    WrapManager(){};

    DeviceTask* curDevTask_;
    uint32_t* coreRunReadyCnt_;
    uint32_t* runReadyCoreIdx_[AICORE_TYPE_NUM];
    uint32_t* corePendReadyCnt_;
    uint32_t* pendingIds_;
    uint32_t* runningIds_;

    int aicValidNum_{0};
    int curDie0MaxCpuId_{0};
    int curDie1StartCpuId_{0};
    DieId dieId_{DieId::DIE_MIX};

    uint32_t* coreIdxPosition_{nullptr};
    bool *wrapCoreAvail_{nullptr};
    AddReadyCoreIdxFunc AddReadyCoreIdx{nullptr};

    WrapInfoQueue* readyWrapCoreFunctionQue_{nullptr};
    // Queue managed by each thread, elem is wrapInfo's addr
    StaticReadyCoreFunctionQueue wrapQueueForThread_{0, 0, nullptr, 0};
    uint32_t* wrapTasklist_{nullptr};
    SendTaskToAiCoreFunc SendTaskToAiCore;
    bool isOpenMixSche {false};
    ArchInfo archInfo;

    // for die-to-die shchedule
    ReadyCoreFunctionQueue* readyDieAicFunctionQue_[DIE_NUM] = {nullptr};
    ReadyCoreFunctionQueue* readyDieAivFunctionQue_[DIE_NUM] = {nullptr};

    inline void InitDeviceInfo(DeviceArgs *deviceArgs, int schedIdx) {
        archInfo = deviceArgs->archInfo;
        InitDieMaxCpuId(static_cast<int>(deviceArgs->scheCpuNum));
        InitDieId(schedIdx);
    }

    inline void InitDieMaxCpuId(int scheCpuNum) {
        curDie0MaxCpuId_ = scheCpuNum >> 1;
        // In odd scenes, scheCpuIdx = curDie0MaxCpuId_ is DIE_MIX, else is DIE_1
        curDie1StartCpuId_ = (scheCpuNum & 1) ? curDie0MaxCpuId_ + 1 : curDie0MaxCpuId_;
    }

    inline void InitDieId(int schedIdx) {
        dieId_ = GetDieId(schedIdx);
    }

    inline void GetDieSchedIdRange(int &schedStart, int &schedEnd, int scheCpuNum) {
        if (dieId_ == DieId::DIE_0) {
            schedStart = 0;
            schedEnd = curDie0MaxCpuId_;
        } else if (dieId_ == DieId::DIE_1) {
            schedStart = curDie1StartCpuId_;
            schedEnd = scheCpuNum;
        }
    }

    inline DieId GetDieId(int scheCpuIdx) {
        if (scheCpuIdx < curDie0MaxCpuId_) {
            return DieId::DIE_0;
        }

        if (scheCpuIdx >= curDie1StartCpuId_) {
            return DieId::DIE_1;
        }

        return DieId::DIE_MIX;
    }

     inline void RemoveMixReadyCoreIdx(int coreIdx, int type) {
        uint32_t tail = --coreRunReadyCnt_[type];
        uint32_t pos = coreIdxPosition_[coreIdx];
        if (pos != tail) {
            runReadyCoreIdx_[type][pos] = runReadyCoreIdx_[type][tail];
            coreIdxPosition_[runReadyCoreIdx_[type][pos]] = pos;
        }
        coreIdxPosition_[coreIdx] = INVALID_COREIDX_POSITION;
        corePendReadyCnt_[type]--;
     }

    inline void Init(DeviceTask* curDevTask, uint32_t* coreRunReadyCnt, uint32_t* runReadyCoreIdxZero,
        uint32_t* runReadyCoreIdxOne, uint32_t* corePendReadyCnt, uint32_t* pendingIds, uint32_t* runningIds,
        int aicValidNum, uint32_t* coreIdxPosition, bool* wrapCoreAvail,
        SendTaskToAiCoreFunc func, AddReadyCoreIdxFunc addReadyCoreIdxFunc) {

        if (archInfo != ArchInfo::DAV_3510) {
            return;
        }
        isOpenMixSche = curDevTask->mixTaskData.wrapIdNum > 0;
        curDevTask_ = curDevTask;
        coreRunReadyCnt_ = coreRunReadyCnt;
        runReadyCoreIdx_[CORE_IDX_AIV] = runReadyCoreIdxZero;
        runReadyCoreIdx_[CORE_IDX_AIC] = runReadyCoreIdxOne;
        corePendReadyCnt_ = corePendReadyCnt;
        pendingIds_ = pendingIds;
        runningIds_ = runningIds;

        aicValidNum_ = aicValidNum;
        coreIdxPosition_ = coreIdxPosition;
        wrapCoreAvail_ = wrapCoreAvail;
        SendTaskToAiCore = func;
        AddReadyCoreIdx = addReadyCoreIdxFunc;
        readyWrapCoreFunctionQue_ = reinterpret_cast<WrapInfoQueue *>(curDevTask_->mixTaskData.readyWrapCoreFunctionQue);

        wrapQueueForThread_.head = 0;
        wrapQueueForThread_.tail = 0;
        wrapQueueForThread_.elem = curDevTask_->mixTaskData.wrapIdNum == 0 ? nullptr :
            static_cast<uint64_t *>(malloc(curDevTask_->mixTaskData.wrapIdNum * sizeof(uint64_t)));
        SetDieReadyQueue(curDevTask->dieReadyFunctionQue);
    }

    inline void Deinit() {
        RETURN_NULL_IF_NOT(isOpenMixSche);
        if (wrapQueueForThread_.elem != nullptr) {
            free(wrapQueueForThread_.elem);
            wrapQueueForThread_.elem = nullptr;
        }
    }

    inline bool GetIsMixarch() {
        return archInfo == ArchInfo::DAV_3510;
    }

    // Get available wrap core and remove it from ready lists
    // For MIX_1C1V: Find 1 AIC + 1 AIV available combination
    // For MIX_1C2V: Find 1 AIC + 2 AIV available combination
    // Returns: coreIdx if found, INVALID_CORE_IDX if no available core
    inline uint32_t RemoveAndGetAvailableCoreIdx(uint32_t mixType) {
        uint32_t coreIdx = INVALID_CORE_IDX;
        switch (mixType) {
            case static_cast<uint32_t>(MixResourceType::MIX_1C1V):
                for (uint32_t i = 0; i < coreRunReadyCnt_[CORE_IDX_AIC]; i++) {
                    uint32_t aivIdx0 = runReadyCoreIdx_[CORE_IDX_AIC][i] * AIV_NUM_PER_AI_CORE + aicValidNum_;
                    if (coreIdxPosition_[aivIdx0] != INVALID_COREIDX_POSITION) {
                        coreIdx = runReadyCoreIdx_[CORE_IDX_AIC][i];
                        CheckCoreIdxInitStatus(coreIdx);
                        CheckCoreIdxInitStatus(aivIdx0);
                        RemoveMixReadyCoreIdx(coreIdx, static_cast<int>(CoreType::AIC));
                        RemoveMixReadyCoreIdx(aivIdx0, static_cast<int>(CoreType::AIV));
                        DEV_VERBOSE_DEBUG("remove coreIdx %u  %u", coreIdx, aivIdx0);
                        break;
                    }
                }
                break;
            case static_cast<uint32_t>(MixResourceType::MIX_1C2V):
                for (uint32_t i = 0; i < coreRunReadyCnt_[CORE_IDX_AIC]; i++) {
                    uint32_t aivIdx0 = runReadyCoreIdx_[CORE_IDX_AIC][i] * AIV_NUM_PER_AI_CORE + aicValidNum_;
                    uint32_t aivIdx1 = aivIdx0 + 1;
                    if (coreIdxPosition_[aivIdx0] != INVALID_COREIDX_POSITION &&
                        coreIdxPosition_[aivIdx1] != INVALID_COREIDX_POSITION) {
                        coreIdx = runReadyCoreIdx_[CORE_IDX_AIC][i];
                        CheckCoreIdxInitStatus(coreIdx);
                        CheckCoreIdxInitStatus(aivIdx0);
                        CheckCoreIdxInitStatus(aivIdx1);
                        RemoveMixReadyCoreIdx(coreIdx, static_cast<int>(CoreType::AIC));
                        RemoveMixReadyCoreIdx(aivIdx0, static_cast<int>(CoreType::AIV));
                        RemoveMixReadyCoreIdx(aivIdx1, static_cast<int>(CoreType::AIV));
                        DEV_VERBOSE_DEBUG("remove coreIdx %u  %u  %u", coreIdx, aivIdx0, aivIdx1);
                        break;
                    }
                }
                break;
            default:
                DEV_ERROR("illegal mixType %d", static_cast<int>(mixType));
                break;
        }
        return coreIdx;
    }

    inline void CheckCoreIdxInitStatus(uint32_t coreIdx) {
        DEV_IF_VERBOSE_DEBUG {
            if (pendingIds_[coreIdx] != AICORE_TASK_INIT || runningIds_[coreIdx] != AICORE_TASK_INIT) {
                DEV_ERROR("core[%u]: pendingId=%x, runningId=%x, is illegal!", coreIdx, pendingIds_[coreIdx], runningIds_[coreIdx]);
            }
        }
    }

    inline void AddRunReadyCoreIdxForWrap(uint32_t coreIdx, MixResourceType mixType = MixResourceType::MIX_UNKNOWN) {
        uint32_t aivIdx0 = coreIdx * AIV_NUM_PER_AI_CORE + aicValidNum_;

        // Add coreIdx to AIC ready list using AddReadyCoreIdx
        // Add aivIdx0 to AIV ready list using AddReadyCoreIdx
        CheckCoreIdxInitStatus(coreIdx);
        CheckCoreIdxInitStatus(aivIdx0);
        AddReadyCoreIdx(coreIdx, static_cast<int>(CoreType::AIC));
        AddReadyCoreIdx(aivIdx0, static_cast<int>(CoreType::AIV));
        corePendReadyCnt_[CORE_IDX_AIC]++;
        corePendReadyCnt_[CORE_IDX_AIV]++;

        if (mixType != MixResourceType::MIX_1C1V) {
            // Add aivIdx1 to AIV ready list using AddReadyCoreIdx
            uint32_t aivIdx1 = coreIdx * AIV_NUM_PER_AI_CORE + aicValidNum_ + 1;
            CheckCoreIdxInitStatus(aivIdx1);
            AddReadyCoreIdx(aivIdx1, static_cast<int>(CoreType::AIV));
            corePendReadyCnt_[CORE_IDX_AIV]++;
            DEV_VERBOSE_DEBUG("add coreIdx %u  %u  %u", coreIdx, coreIdx * AIV_NUM_PER_AI_CORE + aicValidNum_,
                coreIdx * AIV_NUM_PER_AI_CORE + aicValidNum_ + 1);
        } else {
            DEV_VERBOSE_DEBUG("add coreIdx %u  %u", coreIdx, coreIdx * AIV_NUM_PER_AI_CORE + aicValidNum_);
        }
    }

    inline void UpdateWrapQueueForThread() {
        // when readyWrapCoreFunctionQueue has valid value and has available wrapCore
        // move wrapId from readyWrapCoreFunctionQueue to wrapQueueForThread, and occpy wrapCore
        
        uint32_t head = __atomic_load_n(&readyWrapCoreFunctionQue_->head, __ATOMIC_RELAXED);
        uint32_t tail = __atomic_load_n(&readyWrapCoreFunctionQue_->tail, __ATOMIC_RELAXED);
        if (tail - head == 0 || coreRunReadyCnt_[CORE_IDX_AIC] == 0) {
            return;
        }

        WrapInfoQueueLock(readyWrapCoreFunctionQue_);
        head = __atomic_load_n(&readyWrapCoreFunctionQue_->head, __ATOMIC_RELAXED);
        tail = __atomic_load_n(&readyWrapCoreFunctionQue_->tail, __ATOMIC_RELAXED);
        uint32_t taskCount = tail - head;
        if (taskCount == 0) {
            DEV_VERBOSE_DEBUG("mixcore taskCount is zero.");
            WrapInfoQueueUnLock(readyWrapCoreFunctionQue_);
            return;
        }

        while (taskCount-- > 0) {
            WrapInfo *wrapInfo = &readyWrapCoreFunctionQue_->elem[readyWrapCoreFunctionQue_->head];
            uint32_t wrapId = wrapInfo->wrapId;
            uint32_t mixType = wrapInfo->mixResourceType;

            uint32_t avaiCoreIdx = RemoveAndGetAvailableCoreIdx(mixType);
            if (avaiCoreIdx == INVALID_CORE_IDX) {
                DEV_VERBOSE_DEBUG("no available wrap core.");
                WrapInfoQueueUnLock(readyWrapCoreFunctionQue_);
                return;
            }

            DEV_VERBOSE_DEBUG("move wrapId[%u] to wrapQueueForThread. occupy coreIdx[%u]", wrapId, avaiCoreIdx);
            wrapQueueForThread_.elem[wrapQueueForThread_.tail++] = reinterpret_cast<uint64_t>(wrapInfo);
            __atomic_fetch_add(&readyWrapCoreFunctionQue_->head, 1, std::memory_order_release);

            wrapInfo->aicoreIdxList[WRAP_IDX_AIC] = avaiCoreIdx;
            wrapInfo->aicoreIdxList[WRAP_IDX_AIV0] = avaiCoreIdx * AIV_NUM_PER_AI_CORE + aicValidNum_;
            wrapInfo->aicoreIdxList[WRAP_IDX_AIV1] = wrapInfo->aicoreIdxList[WRAP_IDX_AIV0] + (mixType != static_cast<uint32_t>(MixResourceType::MIX_1C1V) ? 1 : 0);
            wrapCoreAvail_[wrapInfo->aicoreIdxList[WRAP_IDX_AIC]] = false;
            wrapCoreAvail_[wrapInfo->aicoreIdxList[WRAP_IDX_AIV0]] = false;
            wrapCoreAvail_[wrapInfo->aicoreIdxList[WRAP_IDX_AIV1]] = false;
            DEV_VERBOSE_DEBUG("add wrapInfo, aicCoreIdx = %u, aivCoreIdxZero = %u, aivCoreIdxOne = %u, mixResourceType = %u",
                wrapInfo->aicoreIdxList[WRAP_IDX_AIC], wrapInfo->aicoreIdxList[WRAP_IDX_AIV0], wrapInfo->aicoreIdxList[WRAP_IDX_AIV1], mixType);
        }
        WrapInfoQueueUnLock(readyWrapCoreFunctionQue_);
    }

    inline void DispatchMixCoreTask() {
        RETURN_NULL_IF_NOT(isOpenMixSche);
        UpdateWrapQueueForThread();
        for (uint32_t idx = wrapQueueForThread_.head; idx < wrapQueueForThread_.tail; idx++) {
            WrapInfo *wrapInfo = reinterpret_cast<WrapInfo *>(wrapQueueForThread_.elem[idx]);
            uint32_t taskNum = GetTaskNumByMixResType(wrapInfo->mixResourceType);
            for (uint32_t taskIdx = 0; taskIdx < taskNum; taskIdx++) {
                uint32_t taskId = wrapInfo->tasklist[taskIdx];
                // 此处可能一个Task准备下发，另一个还没初始化。另一个准备下发时，前面一个已经结束
                if (invalidTaskIds.find(taskId) != invalidTaskIds.end()) {
                    continue;
                }
                CoreType coreType = taskIdx == WRAP_IDX_AIC ? CoreType::AIC : CoreType::AIV;
                DEV_VERBOSE_DEBUG("try to send wrapId[%u]'s taskIdx[%u] taskId[%u]", wrapInfo->wrapId, taskIdx, taskId);
                SendTaskToAiCore(coreType, wrapInfo->aicoreIdxList[taskIdx], taskId);
                wrapInfo->tasklist[taskIdx] = AICORE_SAY_ACK;
            }
        }
    }

    int32_t GetWrapId(uint32_t taskId) {
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto opWrapList = reinterpret_cast<int32_t*>(dyntask->devTask.mixTaskData.opWrapList[funcId]);
        if (opWrapList[opIndex] != -1) {
            return MakeMixWrapID(funcId, opWrapList[opIndex]);
        } else {
            return -1;
        }
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

    inline int32_t GetMixTaskIdx(uint32_t taskId) {
        auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto cceBinary = dyntask->cceBinary;
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
        auto coreType = static_cast<CoreType>(cceBinary[callList[opIndex]].coreType);
        auto wrapVecId = cceBinary[callList[opIndex]].wrapVecId;
        if (coreType == CoreType::AIC) {
            return WRAP_IDX_AIC;
        } else {
            return wrapVecId == 1 ? WRAP_IDX_AIV1 : WRAP_IDX_AIV0;
        }
    }

    bool IsBindedWrapId(uint32_t taskId) {
        RETURN_RET_IF_NOT(isOpenMixSche, false);
        if (GetWrapId(taskId) == -1) {
            return false;
        }
        return true;
    }

    inline void PushTaskToTasklist(uint32_t wrapId, uint32_t taskId, uint32_t taskIdx) {
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
            wrapInfo->mixResourceType = GetMixResourceType(taskId);
            for (uint32_t i = 0; i < MAX_WRAP_TASK_NUM; i++) {
                wrapInfo->tasklist[i] = AICORE_TASK_INIT;
                wrapInfo->aicoreIdxList[i] = 0;
            }
            __atomic_fetch_add(&readyWrapCoreFunctionQue_->tail, 1, std::memory_order_release);
        }
        WrapInfoQueueUnLock(readyWrapCoreFunctionQue_);

        wrapInfo->tasklist[taskIdx] = taskId;
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

        int32_t taskIdx = GetMixTaskIdx(taskId);

        if (wrapInfo == nullptr) { // the wrap is not in this thread
            DEV_VERBOSE_DEBUG("the wrapId %u is not in this thread, push taskId %u to tasklist", wrapId, taskId);
            PushTaskToTasklist(wrapId, taskId, taskIdx);
            return;
        }

        CoreType coreType = taskIdx == WRAP_IDX_AIC ? CoreType::AIC : CoreType::AIV;
        DEV_VERBOSE_DEBUG("directly send taskId %u to core, core type idx: %d", taskId, taskIdx);
        SendTaskToAiCore(coreType, wrapInfo->aicoreIdxList[taskIdx], taskId);
    }

    inline void UpdateFinishIdForMixCore(uint32_t finishId) {
        RETURN_NULL_IF_NOT(isOpenMixSche);
        int32_t id = GetWrapId(finishId);
        if (id == -1) {
            return;
        }
        uint32_t wrapId = id;
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

        uint32_t taskIdx = GetMixTaskIdx(finishId);
        wrapInfo->tasklist[taskIdx] = AICORE_TASK_STOP;

        if (IsMixTaskFinish(wrapInfo)) { // all tasks for this wrap finish
            DEV_VERBOSE_DEBUG("wrapId %u 's all tasks finish, release wrapcore", wrapId);
            AddRunReadyCoreIdxForWrap(wrapInfo->aicoreIdxList[WRAP_IDX_AIC], static_cast<MixResourceType>(wrapInfo->mixResourceType)); // free wrap core
            wrapCoreAvail_[wrapInfo->aicoreIdxList[WRAP_IDX_AIC]] = true;
            wrapCoreAvail_[wrapInfo->aicoreIdxList[WRAP_IDX_AIV0]] = true;
            wrapCoreAvail_[wrapInfo->aicoreIdxList[WRAP_IDX_AIV1]] = true;
            std::swap(wrapQueueForThread_.elem[wrapIdx], wrapQueueForThread_.elem[--wrapQueueForThread_.tail]);
        }
    }

    // for die-to-die schedule
    inline void SetDieReadyQueue(const struct DieReadyQueueData dieReadyFunctionQue) {
        for (size_t i = 0 ; i < DIE_NUM ; i++) {
           readyDieAivFunctionQue_[i] =  reinterpret_cast<ReadyCoreFunctionQueue *>(dieReadyFunctionQue.readyDieAivCoreFunctionQue[i]);
           readyDieAicFunctionQue_[i] =  reinterpret_cast<ReadyCoreFunctionQueue *>(dieReadyFunctionQue.readyDieAicCoreFunctionQue[i]);
        }
    }

    inline ReadyCoreFunctionQueue* GetDieReadyQueue(CoreType type, ReadyCoreFunctionQueue* defaultReadyQue) {
        if (!GetIsMixarch() || dieId_ == DieId::DIE_MIX || dieId_ == DieId::DIE_UNKNOW) {
            return defaultReadyQue;
        }

#ifdef SUPPORT_DIE_TO_DIE_SCHE
        size_t dieIndex = static_cast<size_t>(dieId_);
        ReadyCoreFunctionQueue* dieReadyQueue = nullptr;
        switch(type) {
            case CoreType::AIC:
                dieReadyQueue = readyDieAicFunctionQue_[dieIndex];
                break;
            case CoreType::AIV:
                dieReadyQueue = readyDieAivFunctionQue_[dieIndex];
                break;
            default:
                break;
        }
        return (dieReadyQueue != nullptr) ? dieReadyQueue : defaultReadyQue;
#else
        (void)type;
        return defaultReadyQue;
#endif
    }
};
}