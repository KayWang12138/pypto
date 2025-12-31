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
 * \file test_shmem_wait_until.cpp
 * \brief
 */

#include <gtest/gtest.h>

#include "machine/device/distributed/common.h"
#include "machine/device/distributed/shmem_wait_until.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "tileop/distributed/hccl_context.h"

namespace {

auto InitializeTestEnvironment() {
    npu::tile_fwk::Distributed::TensorInfo info;
    info.offset = {0, 1, 0, 0};
    constexpr uint32_t rankSize = 4;
    constexpr uint32_t rawShape0 = rankSize;
    constexpr uint32_t rawShape1 = rankSize;
    constexpr uint32_t rawShape2 = 4;
    constexpr uint32_t rawShape3 = 8;
    std::vector<int32_t> rawAddr(rawShape1 * rawShape2 * rawShape3, 0);
    info.rawAddr = reinterpret_cast<uint64_t>(rawAddr.data());
    constexpr int32_t value = 1;
    int32_t* addr = rawAddr.data() + info.offset[1] * rawShape2 * rawShape3 + info.offset[2] * rawShape3 + info.offset[3];
    addr[0] = value;
    constexpr size_t codeSize = 17;
    auto data = std::make_unique<int32_t[]>(codeSize);
    int32_t initData[codeSize] = {153, 2, 2, 44, 4, 2, 18, 4, 0, 2, 1, 0, 4, rankSize, rankSize, 4, 8};
    std::copy(initData, initData + codeSize, data.get());
    npu::tile_fwk::dynamic::DevRelocVector<int32_t> aicpuCode(codeSize, data.get());
    auto allocator = std::make_unique<npu::tile_fwk::dynamic::DeviceWorkspaceAllocator>();
    auto task = std::make_unique<npu::tile_fwk::dynamic::DynDeviceTask>(*allocator);
    auto shmemWaitUntil = std::make_unique<npu::tile_fwk::Distributed::ShmemWaitUntil>();

    size_t headerSize = sizeof(npu::tile_fwk::DynFuncHeader);
    size_t dataSize = sizeof(npu::tile_fwk::DynFuncData);
    std::unique_ptr<void, decltype(&free)> buffer(malloc(headerSize + dataSize), free);

    auto* header = new(buffer.get())npu::tile_fwk::DynFuncHeader();
    auto* funcData = new(header + 1)npu::tile_fwk::DynFuncData();
    task->dynFuncDataList = header;
    task->dynFuncDataList[0].seqNo = 1;
    task->dynFuncDataList[0].funcNum = 1;
    task->dynFuncDataList[0].funcSize = 1u;
    task->dynFuncDataList[0].cceBinary = nullptr;

    constexpr size_t exprTblSize = 50;
    auto exprTbl = std::make_unique<uint64_t[]>(exprTblSize);
    funcData->exprTbl = exprTbl.get();

    auto hcclParam = std::make_unique<TileOp::HcclCombinOpParam>();
    hcclParam->rankNum = 0;
    hcclParam->windowsIn[0] = reinterpret_cast<uint64_t>(rawAddr.data());
    funcData->hcclContext[0] = reinterpret_cast<uint64_t>(hcclParam.get());

    auto rawTensorAddrHolder = std::make_unique<uint64_t[]>(1);
    auto rawTensorDescHolder = std::make_unique<npu::tile_fwk::DevRawTensorDesc[]>(1);
    rawTensorAddrHolder[0] = 0;
    rawTensorDescHolder[0] = {0, 0};
    funcData->rawTensorAddr = rawTensorAddrHolder.get();
    funcData->rawTensorDesc = rawTensorDescHolder.get();

    constexpr size_t opAttrsLength = 17;
    auto opAttrs = std::make_unique<uint64_t[]>(opAttrsLength);
    uint64_t initAttrs[opAttrsLength] = {0, 0, 1, 0, 0, 1, 1, 1, 8, rawShape0, rawShape1, rawShape2, rawShape3, 0, 0, 0, 0};
    std::copy(initAttrs, initAttrs + opAttrsLength, opAttrs.get());
    shmemWaitUntil->Init(task.get());

    return std::make_tuple(
        std::move(rawAddr),       // 保持 rawAddr 生命周期
        std::move(data),          // 保持 data 生命周期
        std::move(allocator),     // 保持 allocator 生命周期
        std::move(task),          // 保持 task 生命周期
        std::move(shmemWaitUntil), // 主要测试对象
        std::move(buffer),        // 保持 funcData 内存
        std::move(exprTbl),       // 保持 exprTbl 生命周期
        std::move(hcclParam),     // 保持 hcclParam 生命周期
        std::move(rawTensorAddrHolder), // 保持 rawTensorAddrHolder 生命周期
        std::move(rawTensorDescHolder), // 保持 rawTensorDescHolder 生命周期
        std::move(opAttrs),       // opAttrs 用于任务准备
        std::move(aicpuCode),     // aicpuCode 用于任务准备
        funcData                 // funcData 指针，用于配置任务
    );
}

void PrepareTasks(uint32_t tileOpCount, npu::tile_fwk::Distributed::ShmemWaitUntil* shmemWaitUntil,
    const npu::tile_fwk::dynamic::DevRelocVector<int32_t>& aicpuCode, npu::tile_fwk::DynFuncData* funcData,
    uint64_t* opAttrsPtr)
{
    constexpr size_t opAttrsLength = 17;
    for (uint32_t taskId = 0; taskId < tileOpCount; ++taskId) {
        auto opAtrrOffsets = std::make_unique<int32_t[]>(taskId + 1);
        opAtrrOffsets[taskId] = 0;

        int opAttrsSize = 1 + opAtrrOffsets[taskId] + opAttrsLength;
        auto opAttrsCopy = std::make_unique<uint64_t[]>(opAttrsSize);
        std::copy(opAttrsPtr, opAttrsPtr + opAttrsLength, opAttrsCopy.get() + opAtrrOffsets[taskId]);

        funcData->opAtrrOffsets = opAtrrOffsets.get();
        funcData->opAttrs = opAttrsCopy.get();

        shmemWaitUntil->PrepareTask(taskId, aicpuCode);
    }
}

void RunTests(uint32_t tileOpCount, npu::tile_fwk::Distributed::ShmemWaitUntil* shmemWaitUntil)
{
    for (uint32_t taskId = 0; taskId < tileOpCount; ++taskId) {
        shmemWaitUntil->EnqueueOp(taskId);

        std::vector<uint64_t> completed;
        shmemWaitUntil->PollCompleted(completed);
        ASSERT_EQ(completed.size(), 0);
    }
}

void TestShmemWaitUntil(const uint32_t tileOpCount) {
    auto [rawAddr, data, allocator, task, shmemWaitUntil, buffer, exprTbl, hcclParam, 
          rawTensorAddrHolder, rawTensorDescHolder, opAttrs, aicpuCode, funcData] = InitializeTestEnvironment();

    PrepareTasks(tileOpCount, shmemWaitUntil.get(), aicpuCode, funcData, opAttrs.get());
    
    RunTests(tileOpCount, shmemWaitUntil.get());
}

TEST(ShmemWaitUntilTest, BasicFunctionality) {
    constexpr int32_t tileOpCount = 1;
    TestShmemWaitUntil(tileOpCount);
}

TEST(ShmemWaitUntilTest, VectorResize) {
    constexpr int32_t tileOpCount = npu::tile_fwk::Distributed::AICPU_TASK_ARRAY_SIZE - 1;
    TestShmemWaitUntil(tileOpCount);
}

} // namespace