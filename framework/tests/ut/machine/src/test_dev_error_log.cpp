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

//aicore_manager.cpp
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

// device_execute_context.cpp line 83:
// 传入枚举外的值（0xFF）命中 switch default → DEV_ERROR
TEST_F(TestDevErrorLog, SymbolHandlerIdToHandler_InvalidId_LogsError) {
    dynamic::DeviceExecuteContext::SymbolHandlerIdToHandler(
        static_cast<SymbolHandlerId>(0xFF));
}

// device_execute_context.cpp line 486:
// DeviceExecuteRuntimeCallRootAlloc 是私有静态函数，传 nullptr 直接触发 DEV_ERROR
TEST_F(TestDevErrorLog, DeviceExecuteRuntimeCallRootAlloc_NullContext_LogsError) {
    dynamic::DeviceExecuteContext::DeviceExecuteRuntimeCallRootAlloc(nullptr, 0);
}

// device_execute_context.cpp line 517:
// DeviceExecuteRuntimeCallRootStitch 是私有静态函数，传 nullptr 直接触发 DEV_ERROR
TEST_F(TestDevErrorLog, DeviceExecuteRuntimeCallRootStitch_NullContext_LogsError) {
    dynamic::DeviceExecuteContext::DeviceExecuteRuntimeCallRootStitch(nullptr, 0);
}

// device_stitch_context.cpp line 63-65:
// CheckStitch 发现 dynPredCount != dynSuccCount → DEV_ERROR，紧随 DEV_ASSERT abort
TEST_F(TestDevErrorLog, CheckStitch_PredCountMismatch_LogsError) {
    // 构造 DevAscendFunction：1 个 operation（depGraphPredCount=0, outcastStitchIndex=0）
    alignas(64) uint8_t funcBuf[4096] = {};
    auto *func = reinterpret_cast<dynamic::DevAscendFunction *>(funcBuf);
    size_t opOffset = (sizeof(dynamic::DevAscendFunction) + alignof(dynamic::DevAscendOperation) - 1)
                      & ~(alignof(dynamic::DevAscendOperation) - 1);
    func->operationList_.AssignOffsetSize(opOffset, 1);
    // funcBuf[opOffset] 零初始化 → depGraphPredCount=0, outcastStitchIndex=0

    // 构造 DevAscendFunctionDuppedData：
    //   predcount[0]=1 → dynPredCount += (1 - 0) = 1
    //   stitchList[0].head_=nullptr → dynSuccCount 不增长
    alignas(64) uint8_t dupDataBuf[512] = {};
    auto *dupData = reinterpret_cast<dynamic::DevAscendFunctionDuppedData *>(dupDataBuf);
    dupData->source_ = func;
    dupData->operationList_.size = 1;
    dupData->operationList_.predCountBase = 0;
    dupData->operationList_.stitchBase = 8;
    *reinterpret_cast<predcount_t *>(dupData->data_) = 1;
    // data_[8..] 零初始化 → DevAscendFunctionDuppedStitchList.head_ = nullptr

    dynamic::DevAscendFunctionDupped dup;
    dup.dupTiny_.ptr = reinterpret_cast<uint64_t>(dupData);

    // dynPredCount=1, dynSuccCount=0 → 不匹配 → DEV_ERROR
    dynamic::DeviceStitchContext::CheckStitch(nullptr, 0, &dup);
}

// device_stitch_context.cpp line 172-175:
// MoveTo 检测 stitchedList.size() > MAX_CACHED_FUNC_NUM → DEV_ERROR + return error（无 abort）
TEST_F(TestDevErrorLog, MoveTo_StitchedListExceedsMax_LogsError) {
    dynamic::DeviceStitchContext ctx;

    // 伪造 stitchedList_ 内部字段：size=129 > MAX_CACHED_FUNC_NUM(128)
    constexpr uint32_t fakeSize = MAX_CACHED_FUNC_NUM + 1;
    alignas(16) uint8_t fakeElems[sizeof(dynamic::DevAscendFunctionDupped) * fakeSize] = {};
    ctx.stitchedList_.dataAllocation_.ptr = reinterpret_cast<uint64_t>(fakeElems);
    ctx.stitchedList_.capacity_ = fakeSize;
    ctx.stitchedList_.size_ = fakeSize;

    // 用零初始化缓冲区模拟 DynDeviceTask，避免构造函数依赖
    std::vector<uint8_t> taskBuf(sizeof(dynamic::DynDeviceTask) + 1024, 0);
    auto *dynTask = reinterpret_cast<dynamic::DynDeviceTask *>(taskBuf.data());

    int ret = ctx.MoveTo(dynTask);
    EXPECT_EQ(ret, dynamic::DEVICE_MACHINE_ERROR);
}

// device_stitch_context.cpp line 202-207:
// HandleOneStitch → producerOperationIdx/consumerOperationIdx >= GetOperationSize() → DEV_ERROR + DEV_ASSERT → abort
TEST_F(TestDevErrorLog, HandleOneStitch_OpIdxExceedsSize_LogsError) {
    alignas(64) uint8_t funcBufP[4096] = {};
    alignas(64) uint8_t dupBufP[512] = {};
    auto *dupDataP = reinterpret_cast<dynamic::DevAscendFunctionDuppedData *>(dupBufP);
    dupDataP->source_ = reinterpret_cast<dynamic::DevAscendFunction *>(funcBufP);
    dynamic::DevAscendFunctionDupped producerDup;
    producerDup.dupTiny_.ptr = reinterpret_cast<uint64_t>(dupDataP);

    alignas(64) uint8_t funcBufC[4096] = {};
    alignas(64) uint8_t dupBufC[512] = {};
    auto *dupDataC = reinterpret_cast<dynamic::DevAscendFunctionDuppedData *>(dupBufC);
    dupDataC->source_ = reinterpret_cast<dynamic::DevAscendFunction *>(funcBufC);
    dynamic::DevAscendFunctionDupped consumerDup;
    consumerDup.dupTiny_.ptr = reinterpret_cast<uint64_t>(dupDataC);

    dynamic::DevAscendFunctionDuppedStitch stitchNode;
    stitchNode.next_ = nullptr;
    stitchNode.size_ = 0;
    dynamic::DevAscendFunctionDuppedStitchList stitchList;
    stitchList.head_ = &stitchNode;

    dynamic::DeviceStitchContext::HandleOneStitch(
        producerDup, consumerDup, stitchList,
        0, 0, 0, nullptr,
        dynamic::DeviceStitchContext::StitchKind::StitchDefault, 0);
}

// device_stitch_context.cpp line 447-450:
// FastStitch → slotIdx >= slotSize → DEV_ERROR + continue（无 abort）
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

} // namespace npu::tile_fwk
