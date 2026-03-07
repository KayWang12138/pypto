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
#include <memory>
#include "machine/device/dynamic/context/device_task_context.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "interface/inner/tilefwk.h"
#include "tilefwk/platform.h"
#include "tilefwk/tilefwk.h"

#define private public
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
        dyntask->devTask.mixTaskData.wrapIdNum = 0;
        for (size_t i = 0; i < DIE_NUM; i++) {
            dyntask->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[i] = 0;
            dyntask->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[i] = 0;
        }
    }

    void CreateMockDevAscendProgram(DevAscendProgram *devProg, ArchInfo archInfo) {
        if (devProg == nullptr) {
            return;
        }
        devProg->devArgs.archInfo = archInfo;
        devProg->stitchMaxFunctionNum = 10;
        devProg->stitchFunctionsize = 100;
    }
};

TEST_F(TestDeviceTaskContext, test_init_die_ready_queues_mix_arch) {
    DeviceTaskContext taskContext;
    DevStartArgsBase startArgs;

    DevAscendProgram devProg;
    CreateMockDevAscendProgram(&devProg, ArchInfo::DAV_3510);

    DeviceWorkspaceAllocator workspace(&devProg);

    taskContext.InitAllocator(&devProg, workspace, &startArgs);

    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    CreateMockDynDeviceTask(dyntask.get(), 100);

    taskContext.InitDieReadyQueues(dyntask.get(), &devProg);

    for (size_t i = 0; i < DIE_NUM; i++) {
        EXPECT_NE(dyntask->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[i], 0UL);
        EXPECT_NE(dyntask->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[i], 0UL);

        auto aivQueue = reinterpret_cast<ReadyCoreFunctionQueue *>(
            dyntask->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[i]);
        auto aicQueue = reinterpret_cast<ReadyCoreFunctionQueue *>(
            dyntask->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[i]);

        EXPECT_NE(aivQueue, nullptr);
        EXPECT_NE(aicQueue, nullptr);
        EXPECT_EQ(aivQueue->head, 0U);
        EXPECT_EQ(aivQueue->tail, 0U);
        EXPECT_EQ(aicQueue->head, 0U);
        EXPECT_EQ(aicQueue->tail, 0U);
    }
}

TEST_F(TestDeviceTaskContext, test_build_ready_queue_for_func_mix_arch_no_die_id) {
    DeviceTaskContext taskContext;
    DevStartArgsBase startArgs;

    DevAscendProgram devProg;
    CreateMockDevAscendProgram(&devProg, ArchInfo::DAV_3510);

    DeviceWorkspaceAllocator workspace(&devProg);

    taskContext.InitAllocator(&devProg, workspace, &startArgs);

    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    CreateMockDynDeviceTask(dyntask.get(), 100);
    taskContext.InitDieReadyQueues(dyntask.get(), &devProg);

    auto aivQueue = reinterpret_cast<ReadyCoreFunctionQueue *>(
        workspace.SlabAlloc(sizeof(ReadyCoreFunctionQueue) + 100 * sizeof(taskid_t), WsAicpuSlabMemType::READY_QUE)
            .As<void *>());
    auto aicQueue = reinterpret_cast<ReadyCoreFunctionQueue *>(
        workspace.SlabAlloc(sizeof(ReadyCoreFunctionQueue) + 100 * sizeof(taskid_t), WsAicpuSlabMemType::READY_QUE)
            .As<void *>());
    auto aicpuQueue = reinterpret_cast<ReadyCoreFunctionQueue *>(
        workspace.SlabAlloc(sizeof(ReadyCoreFunctionQueue) + 100 * sizeof(taskid_t), WsAicpuSlabMemType::READY_QUE)
            .As<void *>());

    dyntask->readyQueue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIV)] = aivQueue;
    dyntask->readyQueue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIC)] = aicQueue;
    dyntask->readyQueue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AICPU)] = aicpuQueue;

    taskContext.dieIdList_ = {0, 0, 0};

    int wrapTaskNum = 0;
    taskContext.BuildReadyQueueForFunc(dyntask.get(), 0, false, nullptr, nullptr, nullptr, wrapTaskNum);

    EXPECT_EQ(wrapTaskNum, 0);
}


TEST_F(TestDeviceTaskContext, test_build_ready_queue_for_func_func_index_out_of_range) {
    DeviceTaskContext taskContext;
    DevStartArgsBase startArgs;

    DevAscendProgram devProg;
    CreateMockDevAscendProgram(&devProg, ArchInfo::DAV_3510);

    DeviceWorkspaceAllocator workspace(&devProg);

    taskContext.InitAllocator(&devProg, workspace, &startArgs);

    auto dyntask = std::make_unique<DynDeviceTask>(workspace);
    CreateMockDynDeviceTask(dyntask.get(), 100);
    taskContext.InitDieReadyQueues(dyntask.get(), &devProg);

    auto aivQueue = reinterpret_cast<ReadyCoreFunctionQueue *>(
        workspace.SlabAlloc(sizeof(ReadyCoreFunctionQueue) + 100 * sizeof(taskid_t), WsAicpuSlabMemType::READY_QUE)
            .As<void *>());
    auto aicQueue = reinterpret_cast<ReadyCoreFunctionQueue *>(
        workspace.SlabAlloc(sizeof(ReadyCoreFunctionQueue) + 100 * sizeof(taskid_t), WsAicpuSlabMemType::READY_QUE)
            .As<void *>());
    auto aicpuQueue = reinterpret_cast<ReadyCoreFunctionQueue *>(
        workspace.SlabAlloc(sizeof(ReadyCoreFunctionQueue) + 100 * sizeof(taskid_t), WsAicpuSlabMemType::READY_QUE)
            .As<void *>());

    dyntask->readyQueue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIV)] = aivQueue;
    dyntask->readyQueue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIC)] = aicQueue;
    dyntask->readyQueue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AICPU)] = aicpuQueue;

    taskContext.dieIdList_ = {0, 0};

    int wrapTaskNum = 0;
    taskContext.BuildReadyQueueForFunc(dyntask.get(), 5, false, nullptr, nullptr, nullptr, wrapTaskNum);

    EXPECT_EQ(wrapTaskNum, 0);
}