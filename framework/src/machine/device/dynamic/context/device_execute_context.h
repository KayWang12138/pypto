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
 * \file device_execute_context.h
 * \brief
 */

#pragma once

#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/device/dynamic/aot_binary.h"
#include "machine/device/dynamic/context/device_slot_context.h"
#include "machine/device/dynamic/context/device_stitch_context.h"
#include "machine/device/dynamic/context/device_task_context.h"
#include "machine/device/dynamic/costmodel_utils.h"

namespace npu::tile_fwk::dynamic {

using DeviceTaskInspectorEntry = void (*)(void *inspector_, DeviceExecuteContext *execCtx, DynDeviceTask *task);

struct DeviceExecuteContext {
    using PushTaskEntry = std::function<void(DynDeviceTask *, DeviceExecuteContext *)>;
    PushTaskEntry pushTask;

    DevStartArgs *args{nullptr};
    uint64_t taskId{0};
    bool isFirstTaskSend{true};

    DevAscendProgram *devProg{nullptr};
    DeviceExecuteProgram execProg;
    uint16_t stitchTaskLoopNumThreshold{MAX_CACHED_FUNC_NUM};

    DeviceWorkspaceAllocator workspace;

    DeviceSlotContext slotContext;

    DeviceStitchContext stitchContext;

    DeviceTaskContext taskContext;

    Vector<uint64_t, WsMemCategory::VECTOR_SYMBOL_TABLE> symbolTable;

    DevAscendFunctionDupped currDevRootDup;

    CostModel::ModelData *costModelData{nullptr};

    void *aicoreModel{nullptr};

    SPSCQueue<DynDeviceTask *, SUBMMIT_TASK_QUE_SIZE> submmitTaskQueue_;

    uint64_t duppedRootCount{0};
    bool controlFlowCacheActivated{false};

    bool DuppedRootCached();

    bool DuppedRootUpdateAndCachedAllSubmitted();

    static uint64_t GetInputShapeDimSize(DeviceExecuteContext *ctx, uint64_t inputIndex);
    static uint64_t GetInputShapeDim(DeviceExecuteContext *ctx, uint64_t inputIndex, uint64_t n);
    static int64_t GetInputDataInt32Dim1(DeviceExecuteContext *ctx, uint64_t inputIndex, uint64_t off0);
    static int64_t GetInputDataInt32Dim2(DeviceExecuteContext *ctx, uint64_t inputIndex, uint64_t off0, uint64_t off1);
    static int64_t GetInputDataInt32Dim3(DeviceExecuteContext *ctx, uint64_t inputIndex, uint64_t off0, uint64_t off1,
        uint64_t off2);
    static int64_t GetInputDataInt32Dim4(DeviceExecuteContext *ctx, uint64_t inputIndex, uint64_t off0, uint64_t off1,
        uint64_t off2, uint64_t off3);

    static void *SymbolHandlerIdToHandler(SymbolHandlerId id);

    DeviceExecuteContext(DevStartArgs *startArgs);

    void ShowStats();

    void RunInit(DevStartArgs *startArgs, PushTaskEntry tPushTask);

    void PushTask(DynDeviceTask *dynTask);

    void GELaunchRunCached(DevStartArgs *startArgs, PushTaskEntry tPushTask);

    void RunControlFlow(DevStartArgs *startArgs);

    void GELaunchFullCacheRunControlFlow(DevStartArgs *startArgs, PushTaskEntry tPushTask);

    void GELaunchFullCache(DevStartArgs *startArgs, PushTaskEntry tPushTask);

    void GELaunchPartialCache(DevStartArgs *startArgs, PushTaskEntry tPushTask);

    void GELaunch(DevStartArgs *startArgs, PushTaskEntry tPushTask);

    bool AiCoreFree();

    static void DumpDeviceTask(uint64_t taskId, DynDeviceTask *deviceTask);

    void SubmitToAicoreAndRecycleMemory(bool withoutTail, bool isLastTask = false);

    schema::RUid GetRuid(uint64_t rootKey, bool afterAppend = false);

    void ControlFlowCacheStopCache(uint64_t rootKey);

    void *CallRootFunctionAlloc(uint64_t rootKey);

    void *CallRootFunctionStitch(uint64_t rootKey);

private:
    static void *DeviceExecuteCallAlloc(void *ctx_, uint64_t rootKey);

    static void *DeviceExecuteCallStitch(void *ctx_, uint64_t rootKey);

    static void *DeviceExecuteRuntimerLog(void *ctx_, uint64_t value);

    static void *DeviceExecuteShmemAlloctor(void *ctx_, uint64_t value);
};
}
