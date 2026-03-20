/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_device_task_context.cpp
 * \brief Unit tests for DeviceTaskContext
 */

#include <gtest/gtest.h>
#include <array>
#include <memory>
#define private public
#include "machine/device/dynamic/context/device_task_context.h"
#include "machine/device/dynamic/context/device_execute_context.h"
#include "machine/utils/dynamic/dev_start_args.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "interface/inner/tilefwk.h"
#include "interface/machine/device/tilefwk/aikernel_data.h"
#include "tilefwk/platform.h"
#include "tilefwk/tilefwk.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class TestDeviceTaskContext : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override { Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510); }

    void TearDown() override { Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN); }

protected:
    void CreateMockDynDeviceTask(DynDeviceTask *dyntask, uint32_t coreFunctionCnt = 100) {
        if (dyntask == nullptr) {
            return;
        }
        dyntask->devTask.coreFunctionCnt = coreFunctionCnt;
        dyntask->dynFuncDataCacheListSize = 0;
        dyntask->devTask.mixTaskData.wrapIdNum = 1;
    }

    void CreateMockDevAscendProgram(DevAscendProgram *devProg, ArchInfo archInfo) {
        if (devProg == nullptr) {
            return;
        }
        devProg->devArgs.archInfo = archInfo;
        devProg->ctrlFlowCacheAnchor = &devProg->controlFlowCache;
        devProg->controlFlowCache.isRecording = false;
        devProg->controlFlowCache.isRecordingStopped = false;
        devProg->controlFlowCache.cacheDataOffset = 0;
        devProg->stitchMaxFunctionNum = 10;
        devProg->stitchFunctionsize = 100;
    }
};

TEST_F(TestDeviceTaskContext, test_build_ready_queue_calls_wrap_functions) {
    DeviceTaskContext taskContext;
    DevStartArgsBase startArgs;
    constexpr size_t kControlFlowCacheSize = 64 * 1024;
    auto controlFlowCacheBuf = std::make_unique<uint8_t[]>(kControlFlowCacheSize);

    DevAscendProgram devProg;
    CreateMockDevAscendProgram(&devProg, ArchInfo::DAV_3510);
    devProg.stitchFunctionsize = 100;
    devProg.controlFlowCache.cacheData =
        DevRelocVector<uint8_t>(kControlFlowCacheSize, controlFlowCacheBuf.get());
    devProg.controlFlowCache.isRecording = true;

    DeviceWorkspaceAllocator workspace(&devProg);
    taskContext.InitAllocator(&devProg, workspace, &startArgs);

    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    CreateMockDynDeviceTask(dyntask.get(), 100);

    DevAscendFunction devFunc;
    devFunc.wrapIdNum_ = 1;

    dyntask->dynFuncDataCacheList[0].devFunc = &devFunc;
    dyntask->dynFuncDataCacheListSize = 1;

    bool isNeedWrap = taskContext.IsNeedWrapProcess(dyntask.get(), &devProg);
    EXPECT_TRUE(isNeedWrap);

    uint32_t *wrapTasklist = taskContext.AllocWrapTasklist(dyntask.get());
    EXPECT_NE(wrapTasklist, nullptr);

    WrapInfoQueue *wrapQueue = taskContext.AllocWrapQueue(dyntask.get());
    EXPECT_NE(wrapQueue, nullptr);
    EXPECT_EQ(wrapQueue->head, 0);
    EXPECT_EQ(wrapQueue->tail, 0);
    EXPECT_GT(wrapQueue->capacity, 0);
}

TEST_F(TestDeviceTaskContext, ShowStats_HitsDevErrorMacroLines) {
    DeviceTaskContext taskContext;
    taskContext.ShowStats();
}

TEST_F(TestDeviceTaskContext, InitReadyQueues_ExceedsStitchSize_ReturnsError) {
    DeviceTaskContext taskContext;
    DevStartArgsBase startArgs;
    DevAscendProgram devProg;
    CreateMockDevAscendProgram(&devProg, ArchInfo::DAV_3510);
    devProg.stitchFunctionsize = 10;
    DeviceWorkspaceAllocator workspace(&devProg);
    taskContext.InitAllocator(&devProg, workspace, &startArgs);
    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    CreateMockDynDeviceTask(dyntask.get(), 100U);
    ReadyCoreFunctionQueue *queues[READY_QUEUE_SIZE] = {};
    EXPECT_EQ(taskContext.InitReadyQueues(dyntask.get(), &devProg, queues), DEVICE_MACHINE_ERROR);
}

namespace {

void InitReadyQueueSlot(ReadyCoreFunctionQueue &q, std::array<taskid_t, 4> &elemBuf, uint32_t head,
    uint32_t tail, taskid_t firstId)
{
    q.lock = 0;
    q.head = head;
    q.tail = tail;
    q.capacity = static_cast<uint32_t>(elemBuf.size());
    q.elem = elemBuf.data();
    if (tail > head) {
        elemBuf[0] = firstId;
    }
}
} // namespace

TEST_F(TestDeviceTaskContext, DumpReadyQueue_CoversLoggingLines) {
    DeviceWorkspaceAllocator workspace;
    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    dyntask->devTask.coreFunctionCnt = 3;
    std::array<taskid_t, 4> bufAiv{};
    std::array<taskid_t, 4> bufAic{};
    std::array<taskid_t, 4> bufAicpu{};
    ReadyCoreFunctionQueue qslot[READY_QUEUE_SIZE];
    InitReadyQueueSlot(qslot[0], bufAiv, 0, 1, MakeTaskID(0, 1));
    InitReadyQueueSlot(qslot[1], bufAic, 0, 1, MakeTaskID(0, 2));
    InitReadyQueueSlot(qslot[2], bufAicpu, 0, 1, MakeTaskID(0, 3));
    for (size_t i = 0; i < READY_QUEUE_SIZE; ++i) {
        dyntask->readyQueue[i] = &qslot[i];
    }
    DeviceTaskContext::DumpReadyQueue(dyntask.get(), "ut_cov");
}

TEST_F(TestDeviceTaskContext, DumpDepend_CoversHeadLoggingWithoutDupData) {
    DeviceWorkspaceAllocator workspace;
    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    dyntask->devTask.coreFunctionCnt = 1;
    DynFuncHeader header{};
    header.seqNo = 42;
    header.funcNum = 0;
    header.funcSize = sizeof(DynFuncHeader);
    dyntask->dynFuncDataList = &header;

    std::array<taskid_t, 4> bufAiv{};
    std::array<taskid_t, 4> bufAic{};
    std::array<taskid_t, 4> bufAicpu{};
    ReadyCoreFunctionQueue qslot[READY_QUEUE_SIZE];
    InitReadyQueueSlot(qslot[0], bufAiv, 0, 1, MakeTaskID(0, 0));
    InitReadyQueueSlot(qslot[1], bufAic, 0, 0, 0);
    InitReadyQueueSlot(qslot[2], bufAicpu, 0, 0, 0);
    for (size_t i = 0; i < READY_QUEUE_SIZE; ++i) {
        dyntask->readyQueue[i] = &qslot[i];
    }

    std::array<DevTensorData, 2> tensors{};
    tensors[0].address = 0x1000ULL;
    tensors[1].address = 0x2000ULL;
    DevStartArgs startArgs{};
    startArgs.contextWorkspaceAddr = 0x3000ULL;
    startArgs.inputTensorSize = 1;
    startArgs.outputTensorSize = 1;
    startArgs.devTensorList = tensors.data();

    DevAscendProgram devProg{};
    DeviceTaskContext::DumpDepend(dyntask.get(), &devProg, &startArgs, "ut_cov");
}

TEST_F(TestDeviceTaskContext, DeviceExecute_InvalidCtx_ReturnsNull) {
    EXPECT_EQ(DeviceExecuteContext::DeviceExecuteRuntimeCallRootAlloc(nullptr, 0), nullptr);
    EXPECT_EQ(DeviceExecuteContext::DeviceExecuteRuntimeCallRootStitch(nullptr, 0), nullptr);
}

TEST_F(TestDeviceTaskContext, DeviceExecuteRuntimeCallLog_IsNullSafe) {
    EXPECT_EQ(DeviceExecuteContext::DeviceExecuteRuntimeCallLog(nullptr, 7ULL), nullptr);
}