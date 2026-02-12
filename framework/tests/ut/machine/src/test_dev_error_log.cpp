/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <memory>
#include <cstring>

#define private public
#include "machine/device/aicore_manager.h"
#include "machine/device/dynamic/context/device_execute_context.h"
#undef private
#include "machine/device/distributed/shmem_wait_until.h"
#include "interface/machine/device/tilefwk/core_func_data.h"
#include "machine/device/dynamic/context/device_stitch_context.h"
#include "machine/device/dynamic/context/device_task_context.h"
#include "machine/device/machine_interface/pypto_aicpu_interface.h"
#include "machine/utils/dynamic/dev_encode_function_dupped_data.h"

extern "C" uint32_t DynPyptoKernelServerNull(void *args);

namespace npu::tile_fwk {

class TestDevErrorLog : public testing::Test {
public:
    void SetUp() override {
        unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
        g_isLogEnableError = true;
    }

    void TearDown() override {
        unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
    }
};

TEST_F(TestDevErrorLog, SdmaPrefetch_InvalidPrefetchNum) {
    DeviceTask devTask;
    devTask.l2Info.prefetchNum = MAX_PREFETCH_NUM + 1;
    SdmaPrefetch(&devTask);
}

TEST_F(TestDevErrorLog, ResolveByCoreType_InvalidMixCoreType_LogsError) {
    AicpuTaskManager aicpuTaskManager;
    AiCoreManager manager(aicpuTaskManager);
    CoreFunctionReadyState readyState{};
    manager.ResolveByCoreType(static_cast<int>(MachineType::MIX), 0, &readyState);
}

TEST_F(TestDevErrorLog, HandkShake_Timeout_LogsError) {
    AicpuTaskManager aicpuTaskManager;
    AiCoreManager manager(aicpuTaskManager);
    KernelSharedBuffer sharedBuf{};
    manager.sharedBuffer_ = reinterpret_cast<int64_t>(&sharedBuf);
    manager.aicEnd_ = 1;
    manager.HandkShake();
}

TEST_F(TestDevErrorLog, DeviceExecuteRuntimeCallRootAlloc_NullContext_LogsError) {
    dynamic::DeviceExecuteContext::DeviceExecuteRuntimeCallRootAlloc(nullptr, 0);
}

TEST_F(TestDevErrorLog, DeviceExecuteRuntimeCallRootStitch_NullContext_LogsError) {
    dynamic::DeviceExecuteContext::DeviceExecuteRuntimeCallRootStitch(nullptr, 0);
}

TEST_F(TestDevErrorLog, MoveTo_StitchedListExceedsMax_LogsError) {
    dynamic::DeviceStitchContext ctx;

    constexpr uint32_t fakeSize = MAX_CACHED_FUNC_NUM + 1;
    alignas(16) uint8_t fakeElems[sizeof(dynamic::DevAscendFunctionDupped) * fakeSize] = {};
    ctx.stitchedList_.dataAllocation_.ptr = reinterpret_cast<uint64_t>(fakeElems);
    ctx.stitchedList_.capacity_ = fakeSize;
    ctx.stitchedList_.size_ = fakeSize;

    std::vector<uint8_t> taskBuf(sizeof(dynamic::DynDeviceTask) + 1024, 0);
    auto *dynTask = reinterpret_cast<dynamic::DynDeviceTask *>(taskBuf.data());

    int ret = ctx.MoveTo(dynTask);
    EXPECT_EQ(ret, dynamic::DEVICE_MACHINE_ERROR);
}

TEST_F(TestDevErrorLog, FastStitch_SlotIdxExceedsSize_LogsError) {
    alignas(64) uint8_t funcBuf[8192] = {};
    auto *func = reinterpret_cast<dynamic::DevAscendFunction *>(funcBuf);

    size_t incastOff = (sizeof(dynamic::DevAscendFunction) + alignof(dynamic::DevAscendFunctionIncast) - 1)
                       & ~(alignof(dynamic::DevAscendFunctionIncast) - 1);
    func->incastList.AssignOffsetSize(incastOff, 1);

    auto *incast = reinterpret_cast<dynamic::DevAscendFunctionIncast *>(funcBuf + incastOff);
    size_t slotDataOff = (incastOff + sizeof(dynamic::DevAscendFunctionIncast) + alignof(int) - 1)
                         & ~(alignof(int) - 1);
    incast->fromSlotList.AssignOffsetSize(slotDataOff, 1);

    alignas(64) uint8_t dupBuf[512] = {};
    auto *dupData = reinterpret_cast<dynamic::DevAscendFunctionDuppedData *>(dupBuf);
    dupData->source_ = func;
    dynamic::DevAscendFunctionDupped nextDup;
    nextDup.dupTiny_.ptr = reinterpret_cast<uint64_t>(dupData);

    dynamic::DeviceStitchContext ctx;
    ctx.FastStitch(nullptr, 0, nextDup, 0, 1);
}

TEST_F(TestDevErrorLog, ShowStats_LogsStats) {
    dynamic::DeviceTaskContext ctx;
    ctx.ShowStats();
}

TEST_F(TestDevErrorLog, InitReadyQueues_CoreFuncCntExceeds_LogsError) {
    dynamic::DeviceTaskContext ctx;

    std::vector<uint8_t> taskBuf(sizeof(dynamic::DynDeviceTask) + 1024, 0);
    auto *dynTask = reinterpret_cast<dynamic::DynDeviceTask *>(taskBuf.data());
    dynTask->devTask.coreFunctionCnt = 10;

    std::vector<uint8_t> progBuf(sizeof(dynamic::DevAscendProgram) + 1024, 0);
    auto *devProg = reinterpret_cast<dynamic::DevAscendProgram *>(progBuf.data());

    int ret = ctx.InitReadyQueues(dynTask, devProg, nullptr);
    EXPECT_EQ(ret, dynamic::DEVICE_MACHINE_ERROR);
}

TEST_F(TestDevErrorLog, BuildDynFuncData_CceBinaryNotAligned_LogsError) {
    dynamic::DeviceTaskContext ctx;

    alignas(64) uint8_t cacheBuf[4096] = {};

    alignas(64) uint8_t cacheMeta[sizeof(dynamic::DevControlFlowCache) + 256] = {};
    auto *cache = reinterpret_cast<dynamic::DevControlFlowCache *>(cacheMeta);
    cache->isRecording = true;
    cache->cacheDataOffset = 0;
    cache->cacheData.size_ = 4096;
    cache->cacheData.data_ = cacheBuf;

    std::vector<uint8_t> progBuf(sizeof(dynamic::DevAscendProgram) + 1024, 0);
    auto *devProg = reinterpret_cast<dynamic::DevAscendProgram *>(progBuf.data());
    devProg->ctrlFlowCacheAnchor = cache;

    dynamic::DeviceWorkspaceAllocator workspace(devProg);
    ctx.workspace_ = &workspace;

    std::vector<uint8_t> taskBuf(sizeof(dynamic::DynDeviceTask) + 1024, 0);
    auto *dynTask = reinterpret_cast<dynamic::DynDeviceTask *>(taskBuf.data());
    dynTask->cceBinary = reinterpret_cast<const dynamic::DevCceBinary *>(0x1);

    int ret = ctx.BuildDynFuncData(dynTask, 0, devProg, nullptr, 0);
    EXPECT_EQ(ret, dynamic::DEVICE_MACHINE_ERROR);
}

TEST_F(TestDevErrorLog, ProcessSingleDynFuncData_AlignmentChecks_LogsError) {
    dynamic::DeviceTaskContext ctx;

    std::vector<uint8_t> progBuf(sizeof(dynamic::DevAscendProgram) + 1024, 0);
    auto *devProg = reinterpret_cast<dynamic::DevAscendProgram *>(progBuf.data());

    dynamic::DeviceWorkspaceAllocator workspace(devProg);
    ctx.workspace_ = &workspace;

    alignas(64) uint8_t funcBuf[4096] = {};
    auto *func = reinterpret_cast<dynamic::DevAscendFunction *>(funcBuf);

    alignas(64) uint8_t dupBuf[1024] = {};
    auto *dupData = reinterpret_cast<dynamic::DevAscendFunctionDuppedData *>(dupBuf);
    dupData->source_ = func;

    dynamic::DevAscendFunctionDupped dupFunc;
    dupFunc.dupTiny_.ptr = reinterpret_cast<uint64_t>(dupData);

    DynFuncData dyndata{};
    int ret;

    func->operationAttrList_.offset_ = 1;
    ret = ctx.ProcessSingleDynFuncData(&dyndata, dupFunc, devProg);
    EXPECT_EQ(ret, dynamic::DEVICE_MACHINE_ERROR);

    func->operationAttrList_.offset_ = 0;
    func->opAttrOffsetList_.offset_ = 1;
    ret = ctx.ProcessSingleDynFuncData(&dyndata, dupFunc, devProg);
    EXPECT_EQ(ret, dynamic::DEVICE_MACHINE_ERROR);

    func->opAttrOffsetList_.offset_ = 0;
    dupData->expressionList_.base = 1;
    ret = ctx.ProcessSingleDynFuncData(&dyndata, dupFunc, devProg);
    EXPECT_EQ(ret, dynamic::DEVICE_MACHINE_ERROR);

    dupData->expressionList_.base = 0;
    dupData->incastList_.base = 1;
    ret = ctx.ProcessSingleDynFuncData(&dyndata, dupFunc, devProg);
    EXPECT_EQ(ret, dynamic::DEVICE_MACHINE_ERROR);
}

TEST_F(TestDevErrorLog, DumpReadyQueue_LogsAllQueues) {
    uint32_t elem0 = 0xA0;
    uint32_t elem1 = 0xB0;
    uint32_t elem2 = 0xC0;

    ReadyCoreFunctionQueue q0{0, 1, 1, &elem0, 0};
    ReadyCoreFunctionQueue q1{0, 1, 1, &elem1, 0};
    ReadyCoreFunctionQueue q2{0, 1, 1, &elem2, 0};

    std::vector<uint8_t> taskBuf(sizeof(dynamic::DynDeviceTask) + 1024, 0);
    auto *dynTask = reinterpret_cast<dynamic::DynDeviceTask *>(taskBuf.data());
    dynTask->devTask.coreFunctionCnt = 3;
    dynTask->readyQueue[0] = &q0;
    dynTask->readyQueue[1] = &q1;
    dynTask->readyQueue[2] = &q2;

    dynamic::DeviceTaskContext::DumpReadyQueue(dynTask, "test");
}

TEST_F(TestDevErrorLog, DumpDepend_LogsAllDependInfo) {
    uint32_t elems[3] = {0xA0, 0xB0, 0xC0};
    ReadyCoreFunctionQueue q0{0, 1, 1, &elems[0], 0};
    ReadyCoreFunctionQueue q1{0, 1, 1, &elems[1], 0};
    ReadyCoreFunctionQueue q2{0, 1, 1, &elems[2], 0};

    std::vector<uint8_t> taskBuf(sizeof(dynamic::DynDeviceTask) + 1024, 0);
    auto *dynTask = reinterpret_cast<dynamic::DynDeviceTask *>(taskBuf.data());
    dynTask->devTask.coreFunctionCnt = 1;
    dynTask->readyQueue[0] = &q0;
    dynTask->readyQueue[1] = &q1;
    dynTask->readyQueue[2] = &q2;

    alignas(16) uint8_t headerBuf[sizeof(DynFuncHeader) + sizeof(DynFuncData) + 64] = {};
    auto *funcHeader = reinterpret_cast<DynFuncHeader *>(headerBuf);
    funcHeader->funcNum = 1;
    funcHeader->seqNo = 0;
    dynTask->dynFuncDataList = funcHeader;

    alignas(64) uint8_t funcBuf[4096] = {};
    auto *func = reinterpret_cast<dynamic::DevAscendFunction *>(funcBuf);

    alignas(64) uint8_t dupBuf[2048] = {};
    auto *dupData = reinterpret_cast<dynamic::DevAscendFunctionDuppedData *>(dupBuf);
    dupData->source_ = func;
    dupData->operationList_.size = 1;
    dupData->operationList_.predCountBase = 0;
    dupData->operationList_.stitchCount = 0;
    dupData->expressionList_.size = 1;
    dupData->expressionList_.base = 16;
    dupData->incastList_.size = 1;
    dupData->incastList_.base = 32;
    dupData->outcastList_.size = 1;
    dupData->outcastList_.base = 64;

    dynTask->dynFuncDataCacheList[0].duppedData = dupData;

    DevTensorData tensorData[2] = {};
    std::vector<uint8_t> argsBuf(sizeof(dynamic::DevStartArgs) + 256, 0);
    auto *startArgs = reinterpret_cast<dynamic::DevStartArgs *>(argsBuf.data());
    startArgs->devTensorList = tensorData;
    startArgs->inputTensorSize = 1;
    startArgs->outputTensorSize = 1;
    startArgs->contextWorkspaceAddr = 0;

    std::vector<uint8_t> progBuf(sizeof(dynamic::DevAscendProgram) + 1024, 0);
    auto *devProg = reinterpret_cast<dynamic::DevAscendProgram *>(progBuf.data());

    dynamic::DeviceTaskContext::DumpDepend(dynTask, devProg, startArgs, "test");
}

TEST_F(TestDevErrorLog, StaticPyptoKernelServer_NullArgs_LogsError) {
    uint32_t ret = StaticPyptoKernelServer(nullptr);
    EXPECT_EQ(ret, 1u);
}

TEST_F(TestDevErrorLog, StaticPyptoKernelServer_SaveSoFileFails_LogsError) {
    DeviceArgs devArgs{};
    devArgs.aicpuSoLen = 1;
    char fakeSo = 'x';
    devArgs.aicpuSoBin = reinterpret_cast<uint64_t>(&fakeSo);
    uint32_t ret = StaticPyptoKernelServer(&devArgs);
    EXPECT_EQ(ret, 1u);
}

TEST_F(TestDevErrorLog, DynPyptoKernelServerNull_NullArgs_LogsError) {
    uint32_t ret = DynPyptoKernelServerNull(nullptr);
    EXPECT_EQ(ret, 1u);
}

TEST_F(TestDevErrorLog, DynPyptoKernelServerNull_SaveSoFileFails_LogsError) {
    DeviceArgs devArgs{};
    devArgs.aicpuSoLen = 1;
    char fakeSo = 'x';
    devArgs.aicpuSoBin = reinterpret_cast<uint64_t>(&fakeSo);
    DeviceKernelArgs kargs{};
    kargs.cfgdata = reinterpret_cast<int64_t *>(&devArgs);
    uint32_t ret = DynPyptoKernelServerNull(&kargs);
    EXPECT_EQ(ret, 1u);
}

TEST_F(TestDevErrorLog, DynPyptoKernelServer_ExecuteFuncFails_LogsError) {
    DeviceArgs devArgs{};
    uint32_t ret = DynPyptoKernelServer(&devArgs);
    EXPECT_EQ(ret, 1u);
}

TEST_F(TestDevErrorLog, DynPyptoKernelServerInit_ExecuteFuncFails_LogsError) {
    DeviceArgs devArgs{};
    uint32_t ret = DynPyptoKernelServerInit(&devArgs);
    EXPECT_EQ(ret, 1u);
}

} // namespace npu::tile_fwk
