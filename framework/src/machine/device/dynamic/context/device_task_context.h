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
 * \file device_task_context.h
 * \brief
 */

#pragma once

#include "machine/device/dynamic/context/device_stitch_context.h"
#include "machine/utils/dynamic/dev_workspace.h"

namespace npu::tile_fwk::dynamic {
struct DeviceTaskContext {
    void InitAllocator(DevAscendProgram *devProg, DeviceWorkspaceAllocator &workspace,
                       npu::tile_fwk::DevStartArgsBase *startArgs);

    DynDeviceTask *BuildDeviceTaskData(DeviceStitchContext &stitchContext, uint32_t taskId, DevAscendProgram *devProg,
                                       bool withoutTail);

    void ReleaseFinishedTasks(int perfEvtReleaseFinishTask, int perfEvtDeallocateTask);

    void AppendFinishTask(DynDeviceTask *dynTask);

    void ShowStats();

    void UpdateReadyTaskNum(uint64_t cnt) { readyTaskNum += cnt; }
private:
    uint64_t stitchedFuncNum{0};
    uint64_t rootFuncNum{0};
    uint64_t leafFuncNum{0};
    uint64_t readyTaskNum {0};
    uint64_t dynFuncDataSize {0};
    uint64_t leafFuncDataSize {0};
private:
    DevAscendProgram *devProg_{nullptr};
    DeviceWorkspaceAllocator *workspace_{nullptr};
    npu::tile_fwk::DevStartArgsBase *startArgs_{nullptr};
private:
    void BuildReadyQueue(DynDeviceTask *dyntask, DevAscendProgram *devProg);

#ifdef SUPPORT_WRAP
    uint32_t* AllocWrapTasklist(DynDeviceTask *dyntask) {
        uint32_t size = dyntask->devTask.coreFunctionCnt; // can be optimized by wrapTaskNum
        WsAllocation qalloc = ControlFlowAllocateSlab(devProg_, size, workspace_->SlabAlloc(size, WsAicpuSlabMemType::WRAP_TASKLIST));
        uint32_t *wrapTasklistAddr = qalloc.As<uint32_t>();
        return wrapTasklistAddr;
    }

    WrapInfoQueue* AllocWrapQueue(DynDeviceTask *dyntask) {
        uint32_t size = sizeof(WrapInfoQueue) + dyntask->devTask.wrapIdNum * sizeof(WrapInfo);
        WsAllocation qalloc = ControlFlowAllocateSlab(devProg_, size, workspace_->SlabAlloc(size, WsAicpuSlabMemType::WRAP_QUEUE));
        WrapInfoQueue *q = qalloc.As<WrapInfoQueue>();
        q->head = 0;
        q->tail = 0;
        q->lock = 0;
        q->capacity = dyntask->devTask.wrapIdNum;
        q->elem = reinterpret_cast<WrapInfo *>(q + 1);
        return q;
    }

    void ProcessWrapQueue(DynDeviceTask *dyntask, uint32_t wrapId, int funcIndex, size_t opIndex,
        WrapInfoQueue *wrapQueue, uint32_t *wrapTasklistAddr);
#endif

    void BuildDynFuncData(DynDeviceTask *dyntask, uint32_t taskId, DevAscendProgram *devProg,
        DevAscendFunctionDupped *stitchedList, uint64_t stitchedSize);

    inline void doResolve(DynDeviceTask *dyntask, int coreType, size_t funcIdx, size_t succIdx, predcount_t *predList) {
        predList[succIdx] -= 1;
        if (predList[succIdx] != 0)
            return;

        if (coreType == static_cast<int>(CoreType::HUB)) {
            ResolveEarlyDepends(dyntask, funcIdx, succIdx);
        } else {
#ifdef SUPPORT_WRAP
            auto opWrapList = dyntask->dynFuncDataCacheList[funcIdx].opWrapList;
            if (dyntask->devTask.wrapIdNum>  0 && opWrapList[succIdx] != -1) {
                ProcessWrapQueue(dyntask, MakeWrapID(funcIdx, static_cast<uint32_t>(opWrapList[succIdx])), funcIdx, succIdx,
                    reinterpret_cast<WrapInfoQueue *>(dyntask->devTask.readyWrapCoreFunctionQue),
                    reinterpret_cast<uint32_t *>(dyntask->devTask.wrapTasklist));
            } else {
#endif
                auto q = dyntask->readyQueue[dyntask->GetReadyQueueIndexByCoreType(static_cast<CoreType>(coreType))];
                q->elem[q->tail++] = MakeTaskID(funcIdx, succIdx);
#ifdef SUPPORT_WRAP
            }
#endif
            readyTaskNum++;
        }
    }

    void ResolveEarlyDepends(DynDeviceTask *dyntask, size_t funcIdx, size_t opIdx);

    void ResolveEarlyDepends(DynDeviceTask *dyntask);

public:
    static void DumpReadyQueue(DynDeviceTask *dynTask, const char *prefix);

    static void DumpDepend(DynDeviceTask *dyntask, DevAscendProgram *devProg, DevStartArgs *startArgs, const char *prefix);

    void BuildDeviceTaskDataAndReadyQueue(DynDeviceTask *dyntask, uint32_t taskId, DevAscendProgram *devProg);
};
}
