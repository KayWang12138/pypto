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
#include "machine/device/tilefwk/core_func_data.h"

namespace npu::tile_fwk::dynamic {

struct SchDeviceTaskContext;
using SendTaskToAiCoreFunc =
    std::function<void(struct SchDeviceTaskContext* devCtx, CoreType type, int coreIdx, uint64_t newTask)>;
using AddReadyCoreIdxFunc = std::function<void(int coreIdx, int type)>;

enum class MixResourceType { MIX_UNKNOWN = 0, MIX_1C1V = 1, MIX_1C2V = 2 };

enum class DieId { DIE_0 = 0, DIE_1 = 1, DIE_MIX = 2, DIE_UNKNOWN };

inline void WrapInfoQueueLock(WrapInfoQueue* rq)
{
    while (!__sync_bool_compare_and_swap(&rq->lock, 0, 1)) {
    }
}

inline void WrapInfoQueueUnLock(WrapInfoQueue* rq)
{
    while (!__sync_bool_compare_and_swap(&rq->lock, 1, 0)) {
    }
}

inline uint32_t GetTaskNumByMixResType(uint8_t mixType)
{
    switch (mixType) {
        case static_cast<uint8_t>(MixResourceType::MIX_1C1V):
            return 2;
        case static_cast<uint8_t>(MixResourceType::MIX_1C2V):
            return 3;
        default:
            return 0;
    }
}

inline bool IsMixTaskFinish(WrapInfo* wrapInfo)
{
    switch (wrapInfo->mixResourceType) {
        case static_cast<uint8_t>(MixResourceType::MIX_1C1V):
            return wrapInfo->tasklist[0] == AICORE_TASK_STOP && wrapInfo->tasklist[1] == AICORE_TASK_STOP;
        case static_cast<uint8_t>(MixResourceType::MIX_1C2V):
            return wrapInfo->tasklist[0] == AICORE_TASK_STOP && wrapInfo->tasklist[1] == AICORE_TASK_STOP &&
                   wrapInfo->tasklist[2] == AICORE_TASK_STOP;
        default:
            DEV_ERROR(
                DevCommonErr::PARAM_INVALID, "#sche.wrap.invalid_mode: illegal mixType: %hhu\n",
                wrapInfo->mixResourceType);
            return false;
    }
}

#define RETURN_NULL_IF_NOT(val) \
    if (!val) {                 \
        return;                 \
    }

#define RETURN_RET_IF_NOT(val, ret) \
    if (!val) {                     \
        return ret;                 \
    }

class WrapManager {
public:
    ~WrapManager() {}
    WrapManager() {}

    SchDeviceTaskContext* schDevTaskCtx{nullptr};
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

    uint64_t* aicoreUsageMask_{nullptr};
    AddReadyCoreIdxFunc AddReadyCoreIdx{nullptr};

    WrapInfoQueue* readyWrapCoreFunctionQue_{nullptr};
    // Queue managed by each thread, elem is wrapInfo's addr
    StaticReadyCoreFunctionQueue* wrapQueueForThread_;
    SendTaskToAiCoreFunc SendTaskToAiCore;
    bool isOpenMixSche{false};
    ArchInfo archInfo;
    int schedIdx_;

    // for die-to-die shchedule
    ReadyCoreFunctionQueue* readyDieAicFunctionQue_[DIE_NUM] = {nullptr};
    ReadyCoreFunctionQueue* readyDieAivFunctionQue_[DIE_NUM] = {nullptr};
    ReadyCoreFunctionQueue* selectReadyDieAicFunctionQue_{nullptr};
    ReadyCoreFunctionQueue* selectReadyDieAivFunctionQue_{nullptr};

    struct WrapIdTaskGroup {
        uint32_t wrapId;
        uint8_t taskMask;
        uint8_t mixResourceType;
        uint32_t taskIds[MAX_WRAP_TASK_NUM];
    };

    static constexpr uint32_t MAX_WRAPID_GROUP_NUM = 32;

    struct MixCoreDepBatchContext {
        WrapIdTaskGroup groups[MAX_WRAPID_GROUP_NUM];
        uint32_t groupCount = 0;

        bool AddTask(uint32_t taskId, uint32_t wrapId, const DevCceBinary* cceBinary)
        {
            WrapIdTaskGroup* targetGroup = nullptr;
            for (uint32_t i = 0; i < groupCount; i++) {
                if (groups[i].wrapId == wrapId) {
                    targetGroup = &groups[i];
                    break;
                }
            }
            if (targetGroup == nullptr) {
                if (groupCount >= MAX_WRAPID_GROUP_NUM) {
                    return false;
                }
                targetGroup = &groups[groupCount++];
                targetGroup->wrapId = wrapId;
                targetGroup->taskMask = 0;
                targetGroup->mixResourceType = cceBinary->mixResourceType;
            }
            uint32_t wrapAicoreIdx = WrapManager::GetWrapAicoreIdx(cceBinary->coreType, cceBinary->wrapVecId);
            targetGroup->taskIds[wrapAicoreIdx] = taskId;
            targetGroup->taskMask |= 1 << wrapAicoreIdx;
            return true;
        }
    };

    inline void InitDeviceInfo(DeviceArgs* deviceArgs, int schedIdx)
    {
        archInfo = deviceArgs->archInfo;
        InitDieMaxCpuId(static_cast<int>(deviceArgs->scheCpuNum));
        InitDieId(schedIdx);
        schedIdx_ = schedIdx;
    }

    inline void InitDieMaxCpuId(int scheCpuNum)
    {
        curDie0MaxCpuId_ = scheCpuNum >> 1;
        // In odd scenes, scheCpuIdx = curDie0MaxCpuId_ is DIE_MIX, else is DIE_1
        curDie1StartCpuId_ = (scheCpuNum & 1) ? curDie0MaxCpuId_ + 1 : curDie0MaxCpuId_;
    }

    inline void InitDieId(int schedIdx)
    {
        if (schedIdx < curDie0MaxCpuId_) {
            dieId_ = DieId::DIE_0;
        } else if (schedIdx >= curDie1StartCpuId_) {
            dieId_ = DieId::DIE_1;
        } else {
            dieId_ = DieId::DIE_MIX;
        }
    }

    inline void GetDieSchedIdRange(int& schedStart, int& schedEnd, int scheCpuNum)
    {
        if (dieId_ == DieId::DIE_0) {
            schedStart = 0;
            schedEnd = curDie0MaxCpuId_;
        } else if (dieId_ == DieId::DIE_1) {
            schedStart = curDie1StartCpuId_;
            schedEnd = scheCpuNum;
        }
    }

    inline DieId GetDieId() { return dieId_; }

    inline void RemoveMixReadyCoreIdx(int coreIdx, int type)
    {
        uint32_t tail = --coreRunReadyCnt_[type];
        uint32_t pos = (coreIdx < aicValidNum_) ? 0 : ((coreIdx - aicValidNum_) / AIV_NUM_PER_AI_CORE + 1);
        if (pos != tail) {
            runReadyCoreIdx_[type][pos] = runReadyCoreIdx_[type][tail];
        }
        corePendReadyCnt_[type]--;
    }

    inline void Init(
        DeviceTask* curDevTask, uint32_t* coreRunReadyCnt, uint32_t* runReadyCoreIdxZero, uint32_t* runReadyCoreIdxOne,
        uint32_t* corePendReadyCnt, uint32_t* pendingIds, uint32_t* runningIds, int aicValidNum,
        uint64_t* aicoreUsageMask, SendTaskToAiCoreFunc func, AddReadyCoreIdxFunc addReadyCoreIdxFunc)
    {
        if (archInfo != ArchInfo::DAV_3510) {
            return;
        }
        schDevTaskCtx = devTaskctx;
        isOpenMixSche = curDevTask->mixTaskData.wrapIdNum > 0;
        curDevTask_ = curDevTask;
        coreRunReadyCnt_ = coreRunReadyCnt;
        runReadyCoreIdx_[CORE_IDX_AIV] = runReadyCoreIdxZero;
        runReadyCoreIdx_[CORE_IDX_AIC] = runReadyCoreIdxOne;
        corePendReadyCnt_ = corePendReadyCnt;
        pendingIds_ = pendingIds;
        runningIds_ = runningIds;

        aicValidNum_ = aicValidNum;
        aicoreUsageMask_ = aicoreUsageMask;
        SendTaskToAiCore = func;
        AddReadyCoreIdx = addReadyCoreIdxFunc;
        readyWrapCoreFunctionQue_ = reinterpret_cast<WrapInfoQueue*>(curDevTask_->mixTaskData.readyWrapCoreFunctionQue);
        wrapQueueForThread_ =
            reinterpret_cast<StaticReadyCoreFunctionQueue*>(curDevTask_->mixTaskData.wrapQueueForThread[schedIdx_]);

        SetDieReadyQueue(curDevTask->dieReadyFunctionQue);

        selectReadyDieAicFunctionQue_ = GetDieReadyQueue(
            CoreType::AIC, reinterpret_cast<ReadyCoreFunctionQueue*>(curDevTask->readyAicCoreFunctionQue));
        selectReadyDieAivFunctionQue_ = GetDieReadyQueue(
            CoreType::AIV, reinterpret_cast<ReadyCoreFunctionQueue*>(curDevTask->readyAivCoreFunctionQue));
    }

    inline bool GetIsMixarch() { return archInfo == ArchInfo::DAV_3510; }

    inline bool GetAvailableWrapCoreIdx(
        uint8_t mixResourceType, uint32_t aicReadyCnt, uint32_t& startIdx, uint32_t& coreIdx, uint32_t& v0Idx)
    {
        for (uint32_t idx = startIdx; idx < aicReadyCnt; idx++) {
            uint32_t aicIdx = runReadyCoreIdx_[CORE_IDX_AIC][idx];
            uint32_t aivIdx0 = aicIdx * AIV_NUM_PER_AI_CORE + aicValidNum_;
            switch (mixResourceType) {
                case static_cast<uint8_t>(MixResourceType::MIX_1C1V):
                    if (coreIdxPosition_[aivIdx0] != INVALID_COREIDX_POSITION) {
                        startIdx = idx + 1;
                        coreIdx = aicIdx;
                        v0Idx = aivIdx0;
                        return true;
                    }
                    break;
                case static_cast<uint8_t>(MixResourceType::MIX_1C2V):
                    if (coreIdxPosition_[aivIdx0] != INVALID_COREIDX_POSITION &&
                        coreIdxPosition_[aivIdx0 + 1] != INVALID_COREIDX_POSITION) {
                        startIdx = idx + 1;
                        coreIdx = aicIdx;
                        v0Idx = aivIdx0;
                        return true;
                    }
                    break;
                default:
                    break;
            }
        }
        startIdx = aicReadyCnt - 1;
        return false;
    }

    inline uint32_t GetAvailableWrapCoreCnt(uint32_t& core1c1vCnt, uint32_t& core1c2vCnt, uint32_t maxCoreCnt)
    {
        uint32_t aicReadyCnt = coreRunReadyCnt_[CORE_IDX_AIC];
        for (uint32_t idx = 0; idx < aicReadyCnt && core1c2vCnt < maxCoreCnt; idx++) {
            uint32_t aicIdx = runReadyCoreIdx_[CORE_IDX_AIC][idx];
            uint32_t aivIdx0 = aicIdx * AIV_NUM_PER_AI_CORE + aicValidNum_;
            uint32_t aivIdx1 = aivIdx0 + 1;
            int aivIdx = (aivIdx0 - aicValidNum_) / AIV_NUM_PER_AI_CORE;
            if (!(aicoreUsageMask_[1] & (1ULL << aivIdx))) {
                CheckCoreIdxInitStatus(aicIdx);
                CheckCoreIdxInitStatus(aivIdx0);
                int aivIdx1Val = (aivIdx1 - aicValidNum_) / AIV_NUM_PER_AI_CORE;
                if (!(aicoreUsageMask_[2] & (1ULL << aivIdx1Val))) {
                    CheckCoreIdxInitStatus(aivIdx1);
                    core1c2vCnt++;
                } else {
                    core1c1vCnt++;
                }
            }
        }
        return core1c2vCnt + core1c1vCnt;
    }

    inline void RemoveCoreIdx(uint32_t* aicoreIdxList, uint32_t wrapAicoreIdx, uint32_t coreIdx, CoreType coreType)
    {
        CheckCoreIdxInitStatus(coreIdx);
        aicoreIdxList[wrapAicoreIdx] = coreIdx;
        RemoveMixReadyCoreIdx(coreIdx, static_cast<int>(coreType));
        wrapCoreAvail_[coreIdx] = false;
    }

    inline void UpdateWrapQueueAndRmvCoreIdx(
        WrapInfo* wrap1c2vTasks[], uint32_t task1c2vCnt, WrapInfo* wrap1c1vTasks[], uint32_t task1c1vCnt)
    {
        uint32_t idx = 0;
        for (uint32_t taskIdx = 0; taskIdx < task1c2vCnt; taskIdx++) {
            WrapInfo* wrapInfo = wrap1c2vTasks[taskIdx];
            wrapQueueForThread_->elem[wrapQueueForThread_->tail++] = reinterpret_cast<uint64_t>(wrapInfo);
            uint32_t* aicoreIdxList = wrapInfo->aicoreIdxList;
            while (idx < coreRunReadyCnt_[CORE_IDX_AIC]) {
                uint32_t aicIdx = runReadyCoreIdx_[CORE_IDX_AIC][idx];
                uint32_t aivIdx0 = aicIdx * AIV_NUM_PER_AI_CORE + aicValidNum_;
                uint32_t aivIdx1 = aivIdx0 + 1;

                int aivIdx = (aivIdx0 - aicValidNum_) / AIV_NUM_PER_AI_CORE;
                int aivIdx1Val = (aivIdx1 - aicValidNum_) / AIV_NUM_PER_AI_CORE;
                if (!(aicoreUsageMask_[1] & (1ULL << aivIdx)) && !(aicoreUsageMask_[2] & (1ULL << aivIdx1Val))) {
                    CheckCoreIdxInitStatus(aicIdx);
                    CheckCoreIdxInitStatus(aivIdx0);
                    CheckCoreIdxInitStatus(aivIdx1);
                    aicoreIdxList[WRAP_IDX_AIC] = aicIdx;
                    aicoreIdxList[WRAP_IDX_AIV0] = aivIdx0;
                    aicoreIdxList[WRAP_IDX_AIV1] = aivIdx1;
                    RemoveMixReadyCoreIdx(aicIdx, static_cast<int>(CoreType::AIC));
                    RemoveMixReadyCoreIdx(aivIdx0, static_cast<int>(CoreType::AIV));
                    RemoveMixReadyCoreIdx(aivIdx1, static_cast<int>(CoreType::AIV));
                    aicoreUsageMask_[0] |= (1ULL << aicIdx);
                    aicoreUsageMask_[1] |= (1ULL << aivIdx);
                    aicoreUsageMask_[2] |= (1ULL << aivIdx1Val);
                    break;
                } else {
                    idx++;
                }
            }
        }

        idx = 0;
        for (uint32_t taskIdx = 0; taskIdx < task1c1vCnt; taskIdx++) {
            WrapInfo* wrapInfo = wrap1c1vTasks[taskIdx];
            wrapQueueForThread_->elem[wrapQueueForThread_->tail++] = reinterpret_cast<uint64_t>(wrapInfo);
            uint32_t* aicoreIdxList = wrapInfo->aicoreIdxList;
            while (idx < coreRunReadyCnt_[CORE_IDX_AIC]) {
                uint32_t aicIdx = runReadyCoreIdx_[CORE_IDX_AIC][idx];
                uint32_t aivIdx0 = aicIdx * AIV_NUM_PER_AI_CORE + aicValidNum_;

                int aivIdx = (aivIdx0 - aicValidNum_) / AIV_NUM_PER_AI_CORE;
                if (!(aicoreUsageMask_[1] & (1ULL << aivIdx))) {
                    CheckCoreIdxInitStatus(aicIdx);
                    CheckCoreIdxInitStatus(aivIdx0);
                    aicoreIdxList[WRAP_IDX_AIC] = aicIdx;
                    aicoreIdxList[WRAP_IDX_AIV0] = aivIdx0;
                    RemoveMixReadyCoreIdx(aicIdx, static_cast<int>(CoreType::AIC));
                    RemoveMixReadyCoreIdx(aivIdx0, static_cast<int>(CoreType::AIV));
                    aicoreUsageMask_[0] |= (1ULL << aicIdx);
                    aicoreUsageMask_[1] |= (1ULL << aivIdx);
                    break;
                } else {
                    idx++;
                }
            }
        }
    }

    inline void CheckCoreIdxInitStatus(uint32_t coreIdx)
    {
        DEV_IF_VERBOSE_DEBUG
        {
            if (pendingIds_[coreIdx] != AICORE_TASK_INIT || runningIds_[coreIdx] != AICORE_TASK_INIT) {
                DEV_ERROR(
                    CtrlErr::TASK_STATS_ABNORMAL,
                    "#sche.task.run.wrap.stats: core[%u]: pendingId=%x, runningId=%x, is illegal!", coreIdx,
                    pendingIds_[coreIdx], runningIds_[coreIdx]);
            }
        }
    }

    inline void AddRunReadyCoreIdxForWrap(uint32_t coreIdx, CoreType coreType)
    {
        CheckCoreIdxInitStatus(coreIdx);
        AddReadyCoreIdx(coreIdx, static_cast<int>(coreType));
        uint32_t idx = (coreType == CoreType::AIC) ? CORE_IDX_AIC : CORE_IDX_AIV;
        corePendReadyCnt_[idx]++;
    }

    inline uint32_t CalculateTaskCountInSync()
    {
        uint32_t taskCount = 0;
        uint32_t head = readyWrapCoreFunctionQue_->head;
        for (uint32_t i = head; i < readyWrapCoreFunctionQue_->tail; i++) {
            WrapInfo* info = &readyWrapCoreFunctionQue_->elem[i];
            bool isC1V1Ready =
                (info->mixResourceType == static_cast<uint8_t>(MixResourceType::MIX_1C1V) &&
                 info->tasklist[0] != AICORE_TASK_INIT && info->tasklist[1] != AICORE_TASK_INIT);
            bool isC1V2Ready =
                (info->mixResourceType == static_cast<uint8_t>(MixResourceType::MIX_1C2V) &&
                 info->tasklist[0] != AICORE_TASK_INIT && info->tasklist[1] != AICORE_TASK_INIT &&
                 info->tasklist[2] != AICORE_TASK_INIT); // 2:v1 index
            if (isC1V1Ready || isC1V2Ready) {
                std::swap(readyWrapCoreFunctionQue_->elem[i], readyWrapCoreFunctionQue_->elem[head + taskCount]);
                taskCount++;
            }
        }
        return taskCount;
    }

    inline void UpdateWrapQueueForThread()
    {
        // when readyWrapCoreFunctionQueue has valid value and has available wrapCore
        // move wrapId from readyWrapCoreFunctionQueue to wrapQueueForThread, and occpy wrapCore

        uint32_t head = __atomic_load_n(&readyWrapCoreFunctionQue_->head, __ATOMIC_RELAXED);
        uint32_t tail = __atomic_load_n(&readyWrapCoreFunctionQue_->tail, __ATOMIC_RELAXED);
        uint32_t core1c1vCnt = 0;
        uint32_t core1c2vCnt = 0;
        constexpr uint32_t maxTaskCnt = 8u;
        uint32_t wrapCoreCnt = GetAvailableWrapCoreCnt(core1c1vCnt, core1c2vCnt, maxTaskCnt);
        if (tail - head == 0 || wrapCoreCnt == 0) {
            return;
        }

        WrapInfoQueueLock(readyWrapCoreFunctionQue_);
        head = readyWrapCoreFunctionQue_->head;
#ifdef NO_EARLY_SEND_TASK
        uint32_t taskCount = CalculateTaskCountInSync();
#else
        uint32_t taskCount = readyWrapCoreFunctionQue_->tail - head;
#endif
        if (taskCount == 0) {
            DEV_VERBOSE_DEBUG("mixcore taskCount is zero.");
            WrapInfoQueueUnLock(readyWrapCoreFunctionQue_);
            return;
        }
        WrapInfo* localTasks[maxTaskCnt];
        uint32_t maxReadyCnt = taskCount > maxTaskCnt ? maxTaskCnt : taskCount;
        maxReadyCnt = maxReadyCnt > wrapCoreCnt ? wrapCoreCnt : maxReadyCnt;
        uint32_t taskHead = 0, taskTail = maxReadyCnt;
        while (taskHead < taskTail) {
            WrapInfo* info = &readyWrapCoreFunctionQue_->elem[head++];
            if (info->mixResourceType == static_cast<uint32_t>(MixResourceType::MIX_1C2V)) {
                if (core1c2vCnt == 0) {
                    break;
                }
                localTasks[taskHead++] = info;
                core1c2vCnt--;
            } else {
                localTasks[--taskTail] = info;
            }
        }
        uint32_t valid1c1vCnt = maxReadyCnt - taskTail;
        uint32_t validReadyCnt = taskHead + valid1c1vCnt;
        readyWrapCoreFunctionQue_->head += validReadyCnt;
        WrapInfoQueueUnLock(readyWrapCoreFunctionQue_);

        WrapInfo** task1c1vPtr = localTasks + taskTail;
        UpdateWrapQueueAndRmvCoreIdx(localTasks, taskHead, task1c1vPtr, valid1c1vCnt);
    }

    inline void DispatchMixCoreTask()
    {
        RETURN_NULL_IF_NOT(isOpenMixSche);
        UpdateWrapQueueForThread();
        for (uint32_t idx = wrapQueueForThread_->head; idx < wrapQueueForThread_->tail; idx++) {
            WrapInfo* wrapInfo = reinterpret_cast<WrapInfo*>(wrapQueueForThread_->elem[idx]);
            uint32_t taskNum = GetTaskNumByMixResType(wrapInfo->mixResourceType);
            for (uint32_t wrapAicoreIdx = 0; wrapAicoreIdx < taskNum; wrapAicoreIdx++) {
                uint32_t taskId = wrapInfo->tasklist[wrapAicoreIdx];
                // 此处可能一个Task准备下发，另一个还没初始化。另一个准备下发时，前面一个已经结束
                if (taskId == AICORE_TASK_DISTRIBUTED || taskId == AICORE_TASK_INIT || taskId == AICORE_TASK_STOP) {
                    continue;
                }
                CoreType coreType = wrapAicoreIdx == WRAP_IDX_AIC ? CoreType::AIC : CoreType::AIV;
                DEV_VERBOSE_DEBUG(
                    "try to send wrapId[%u]'s wrapAicoreIdx[%u] taskId[%u]", wrapInfo->wrapId, wrapAicoreIdx, taskId);
                SendTaskToAiCore(schDevTaskCtx, coreType, wrapInfo->aicoreIdxList[wrapAicoreIdx], taskId);
                wrapInfo->tasklist[wrapAicoreIdx] = AICORE_TASK_DISTRIBUTED;
            }
        }
    }

    int32_t GetWrapId(uint32_t taskId)
    {
        auto dyntask = reinterpret_cast<DynDeviceTask*>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto opWrapList = reinterpret_cast<int32_t*>(dyntask->devTask.mixTaskData.opWrapList[funcId]);
        if (opWrapList[opIndex] != -1) {
            return MakeMixWrapID(funcId, opWrapList[opIndex]);
        } else {
            return -1;
        }
    }

    int32_t GetWrapVecId(uint32_t taskId)
    {
        auto dyntask = reinterpret_cast<DynDeviceTask*>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto cceBinary = dyntask->cceBinary;
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
        return cceBinary[callList[opIndex]].wrapVecId;
    }

    CoreType GetCoreType(uint32_t taskId)
    {
        auto dyntask = reinterpret_cast<DynDeviceTask*>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto cceBinary = dyntask->cceBinary;
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
        return static_cast<CoreType>(cceBinary[callList[opIndex]].coreType);
    }

    uint8_t GetMixResourceType(uint32_t taskId)
    {
        auto dyntask = reinterpret_cast<DynDeviceTask*>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto cceBinary = dyntask->cceBinary;
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
        return cceBinary[callList[opIndex]].mixResourceType;
    }

    inline static int32_t GetWrapAicoreIdx(uint32_t coreType, int32_t wrapVecId)
    {
        if (coreType == static_cast<uint32_t>(CoreType::AIC)) {
            return WRAP_IDX_AIC;
        } else {
            return wrapVecId == 1 ? WRAP_IDX_AIV1 : WRAP_IDX_AIV0;
        }
    }

    inline int32_t GetWrapAicoreIdx(uint32_t taskId)
    {
        auto dyntask = reinterpret_cast<DynDeviceTask*>(curDevTask_);
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto cceBinary = dyntask->cceBinary;
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
        auto coreType = cceBinary[callList[opIndex]].coreType;
        auto wrapVecId = cceBinary[callList[opIndex]].wrapVecId;
        return GetWrapAicoreIdx(coreType, wrapVecId);
    }

    bool IsBindedWrapId(uint32_t taskId, uint32_t& wrapId)
    {
        RETURN_RET_IF_NOT(isOpenMixSche, false);
        int id = GetWrapId(taskId);
        if (id == -1) {
            return false;
        }
        wrapId = id;
        return true;
    }

    inline WrapInfo* findWrapInfo(uint32_t start, uint32_t end, uint32_t wrapId)
    {
        for (uint32_t idx = start; idx < end; idx++) {
            if (readyWrapCoreFunctionQue_->elem[idx].wrapId == wrapId) {
                return &readyWrapCoreFunctionQue_->elem[idx];
            }
        }
        return nullptr;
    }

    inline void PushTaskToTasklist(uint32_t wrapId, uint32_t taskId, uint32_t wrapAicoreIdx, uint8_t mixResourceType)
    {
        auto funcId = FuncID(taskId);
        auto wrapIndex = TaskID(wrapId);
        auto dyntask = reinterpret_cast<DynDeviceTask*>(curDevTask_);
        auto opWrapPtrList = reinterpret_cast<WrapInfo**>(dyntask->devTask.mixTaskData.opWrapPtrList[funcId]);
        WrapInfo* wrapInfo = opWrapPtrList[wrapIndex];
        if (wrapInfo != nullptr) {
            wrapInfo->tasklist[wrapAicoreIdx] = taskId;
            return;
        }

        WrapInfoQueueLock(readyWrapCoreFunctionQue_);
        wrapInfo = opWrapPtrList[wrapIndex];
        if (unlikely(wrapInfo != nullptr)) {
            WrapInfoQueueUnLock(readyWrapCoreFunctionQue_);
            wrapInfo->tasklist[wrapAicoreIdx] = taskId;
            return;
        }

        // add a new wrapinfo
        wrapInfo = &readyWrapCoreFunctionQue_->elem[readyWrapCoreFunctionQue_->tail];
        readyWrapCoreFunctionQue_->tail += 1;
        opWrapPtrList[wrapIndex] = wrapInfo;
        WrapInfoQueueUnLock(readyWrapCoreFunctionQue_);

        wrapInfo->wrapId = wrapId;
        wrapInfo->mixResourceType = mixResourceType;
        for (uint32_t i = 0; i < MAX_WRAP_TASK_NUM; i++) {
            wrapInfo->tasklist[i] = AICORE_TASK_INIT;
            wrapInfo->aicoreIdxList[i] = 0;
        }
        wrapInfo->tasklist[wrapAicoreIdx] = taskId;
    }

    inline void ResolveDepForMixCore(uint32_t taskId, uint32_t wrapId, const DevCceBinary* cceBinary)
    {
        // resolve dep, if has available core, send task directly, else call PushTaskToTasklist, try to send task in
        // next loop
        DEV_VERBOSE_DEBUG("taskId = %u, wrapId = %u", taskId, wrapId);
        int32_t wrapAicoreIdx = GetWrapAicoreIdx(cceBinary->coreType, cceBinary->wrapVecId);
        PushTaskToTasklist(wrapId, taskId, wrapAicoreIdx, cceBinary->mixResourceType);
        return;
    }

    inline void BatchResolveDepForMixCore(MixCoreDepBatchContext& batchCtx)
    {
        uint32_t startIdx = 0u;
        uint32_t aicReadyCnt = coreRunReadyCnt_[CORE_IDX_AIC];
        uint32_t coreIdx = 0;
        uint32_t aivIdx0 = 0;
        for (uint32_t g = 0; g < batchCtx.groupCount; g++) {
            auto& group = batchCtx.groups[g];
            uint8_t taskCount = __builtin_popcount(group.taskMask);
            if (taskCount == 0) {
                continue;
            }

            if (likely(taskCount == GetTaskNumByMixResType(group.mixResourceType))) {
                if (GetAvailableWrapCoreIdx(group.mixResourceType, aicReadyCnt, startIdx, coreIdx, aivIdx0)) {
                    auto taskIds = group.taskIds;
                    if (group.mixResourceType == static_cast<uint8_t>(MixResourceType::MIX_1C1V)) {
                        SendTaskToAiCore(schDevTaskCtx, CoreType::AIC, coreIdx, taskIds[WRAP_IDX_AIC]);
                        SendTaskToAiCore(schDevTaskCtx, CoreType::AIV, aivIdx0, taskIds[WRAP_IDX_AIV0]);
                        RemoveMixReadyCoreIdx(coreIdx, static_cast<int>(CoreType::AIC));
                        wrapCoreAvail_[coreIdx] = false;
                        RemoveMixReadyCoreIdx(aivIdx0, static_cast<int>(CoreType::AIV));
                        wrapCoreAvail_[aivIdx0] = false;
                    } else {
                        SendTaskToAiCore(schDevTaskCtx, CoreType::AIC, coreIdx, taskIds[WRAP_IDX_AIC]);
                        SendTaskToAiCore(schDevTaskCtx, CoreType::AIV, aivIdx0, taskIds[WRAP_IDX_AIV0]);
                        uint32_t aivIdx1 = aivIdx0 + 1;
                        SendTaskToAiCore(schDevTaskCtx, CoreType::AIV, aivIdx1, taskIds[WRAP_IDX_AIV1]);
                        RemoveMixReadyCoreIdx(coreIdx, static_cast<int>(CoreType::AIC));
                        wrapCoreAvail_[coreIdx] = false;
                        RemoveMixReadyCoreIdx(aivIdx0, static_cast<int>(CoreType::AIV));
                        wrapCoreAvail_[aivIdx0] = false;
                        RemoveMixReadyCoreIdx(aivIdx1, static_cast<int>(CoreType::AIV));
                        wrapCoreAvail_[aivIdx1] = false;
                    }
                } else {
                    WrapInfoQueueLock(readyWrapCoreFunctionQue_);
                    WrapInfo* wrapInfo = &readyWrapCoreFunctionQue_->elem[readyWrapCoreFunctionQue_->tail++];
                    WrapInfoQueueUnLock(readyWrapCoreFunctionQue_);
                    wrapInfo->mixResourceType = group.mixResourceType;
                    constexpr uint32_t copySize = MAX_WRAPID_GROUP_NUM * sizeof(uint32_t);
                    memcpy_s(wrapInfo->tasklist, copySize, group.taskIds, copySize);
                }
            } else {
                PushTaskByTaskMask(group);
            }
        }
    }

    void PushTaskByTaskMask(const WrapIdTaskGroup& group)
    {
        switch (group.taskMask) { // 0-7
            case 1:
                PushTaskToTasklist(group.wrapId, group.taskIds[WRAP_IDX_AIC], WRAP_IDX_AIC, group.mixResourceType);
                break;
            case 2:
                PushTaskToTasklist(group.wrapId, group.taskIds[WRAP_IDX_AIV0], WRAP_IDX_AIV0, group.mixResourceType);
                break;
            case 3:
                PushTaskToTasklist(group.wrapId, group.taskIds[WRAP_IDX_AIC], WRAP_IDX_AIC, group.mixResourceType);
                PushTaskToTasklist(group.wrapId, group.taskIds[WRAP_IDX_AIV0], WRAP_IDX_AIV0, group.mixResourceType);
                break;
            case 4:
                PushTaskToTasklist(group.wrapId, group.taskIds[WRAP_IDX_AIV1], WRAP_IDX_AIV1, group.mixResourceType);
                break;
            case 5:
                PushTaskToTasklist(group.wrapId, group.taskIds[WRAP_IDX_AIC], WRAP_IDX_AIC, group.mixResourceType);
                PushTaskToTasklist(group.wrapId, group.taskIds[WRAP_IDX_AIV1], WRAP_IDX_AIV1, group.mixResourceType);
                break;
            case 6:
                PushTaskToTasklist(group.wrapId, group.taskIds[WRAP_IDX_AIV0], WRAP_IDX_AIV0, group.mixResourceType);
                PushTaskToTasklist(group.wrapId, group.taskIds[WRAP_IDX_AIV1], WRAP_IDX_AIV1, group.mixResourceType);
                break;
            default:
                break;
        }
    }

    inline void UpdateFinishIdForMixCore(uint32_t finishId, uint32_t coreIdx, CoreType coreType)
    {
        RETURN_NULL_IF_NOT(isOpenMixSche);
        int32_t id = GetWrapId(finishId);
        if (id == -1) {
            return;
        }

        AddRunReadyCoreIdxForWrap(coreIdx, coreType); // free wrap core
        wrapCoreAvail_[coreIdx] = true;

        uint32_t wrapId = id;
        WrapInfo* wrapInfo = nullptr;
        uint32_t wrapIdx = 0;
        for (uint32_t idx = wrapQueueForThread_->head; idx < wrapQueueForThread_->tail; idx++) {
            auto tmpInfo = reinterpret_cast<WrapInfo*>(wrapQueueForThread_->elem[idx]);
            if (tmpInfo->wrapId == wrapId) {
                wrapInfo = tmpInfo;
                wrapIdx = idx;
                break;
            }
        }

        if (unlikely(wrapInfo == nullptr)) {
            // 没有走全局WrapQueue，走的本地直接下发
            return;
        }

        int32_t taskIdx = GetMixTaskIdx(finishId);
        wrapInfo->tasklist[taskIdx] = AICORE_TASK_STOP;

        CoreType coreType = (taskIdx == WRAP_IDX_AIC) ? CoreType::AIC : CoreType::AIV;
        AddRunReadyCoreIdxForWrap(wrapInfo->aicoreIdxList[taskIdx], coreType); // free wrap core
        uint32_t coreIdx = wrapInfo->aicoreIdxList[taskIdx];
        if (coreIdx < static_cast<uint32_t>(aicValidNum_)) {
            aicoreUsageMask_[0] &= ~(1ULL << coreIdx);
        } else {
            int aivIdx = (coreIdx - aicValidNum_) / AIV_NUM_PER_AI_CORE;
            int maskIdx = (coreIdx - aicValidNum_) % AIV_NUM_PER_AI_CORE + 1;
            aicoreUsageMask_[maskIdx] &= ~(1ULL << aivIdx);
        }

        if (IsMixTaskFinish(wrapInfo)) { // all tasks for this wrap finish
            DEV_VERBOSE_DEBUG("wrapId %u 's all tasks finish, release wrapcore", wrapId);
            std::swap(wrapQueueForThread_->elem[wrapIdx], wrapQueueForThread_->elem[--wrapQueueForThread_->tail]);
        }
    }

    // for die-to-die schedule
    inline void SetDieReadyQueue(const struct DieReadyQueueData dieReadyFunctionQue)
    {
        for (size_t i = 0; i < DIE_NUM; i++) {
            readyDieAivFunctionQue_[i] =
                reinterpret_cast<ReadyCoreFunctionQueue*>(dieReadyFunctionQue.readyDieAivCoreFunctionQue[i]);
            readyDieAicFunctionQue_[i] =
                reinterpret_cast<ReadyCoreFunctionQueue*>(dieReadyFunctionQue.readyDieAicCoreFunctionQue[i]);
        }
    }

    inline ReadyCoreFunctionQueue* GetDieReadyQueue(CoreType type, ReadyCoreFunctionQueue* defaultReadyQue)
    {
        if (!GetIsMixarch() || dieId_ == DieId::DIE_MIX || dieId_ == DieId::DIE_UNKNOWN) {
            return defaultReadyQue;
        }
        size_t dieIndex = static_cast<size_t>(dieId_);
        ReadyCoreFunctionQueue* dieReadyQueue = nullptr;
        switch (type) {
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
    }

    ReadyCoreFunctionQueue* GetDieReadyAicQue() { return selectReadyDieAicFunctionQue_; }
    ReadyCoreFunctionQueue* GetDieReadyAivQue() { return selectReadyDieAivFunctionQue_; }
};
} // namespace npu::tile_fwk::dynamic
