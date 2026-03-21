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
#include <fstream>
#include <cstdio>
#define private public
#include "machine/device/dynamic/context/device_task_context.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/utils/dynamic/dev_encode_function_dupped_data.h"
#include "interface/inner/tilefwk.h"
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

    DevAscendFunction* CreateDevAscendFunctionBuffer(std::unique_ptr<uint8_t[]>& funcBuffer,
                                                     uint8_t*& funcDataPtr,
                                                     size_t kOpCount,
                                                     size_t kFuncBufferSize) {
        (void)kOpCount;
        funcBuffer = std::make_unique<uint8_t[]>(kFuncBufferSize);
        memset_s(funcBuffer.get(), kFuncBufferSize, 0, kFuncBufferSize);
        funcDataPtr = funcBuffer.get();

        DevAscendFunction *devFunc = reinterpret_cast<DevAscendFunction *>(funcDataPtr);
        funcDataPtr += sizeof(DevAscendFunction);

        devFunc->rootHash = 0x12345678;
        devFunc->funcKey = 100;
        devFunc->sourceFunc = nullptr;

        return devFunc;
    }

    void SetupDevAscendFunctionData(DevAscendFunction *devFunc,
                                     uint8_t* funcDataPtr,
                                     uint8_t* funcBuffer,
                                     size_t kOpCount) {
        size_t currentOffset = sizeof(DevAscendFunction);
        auto alignUp = [&currentOffset](size_t alignment) {
            currentOffset = (currentOffset + alignment - 1) & ~(alignment - 1);
        };

        alignUp(alignof(SymInt));
        devFunc->operationAttrList_.AssignOffsetSize(currentOffset, kOpCount);
        SymInt *attrData = reinterpret_cast<SymInt *>(funcDataPtr);
        for (size_t i = 0; i < kOpCount; i++) {
            attrData[i] = SymInt(static_cast<uint64_t>(0));
        }
        currentOffset += kOpCount * sizeof(SymInt);
        funcDataPtr += kOpCount * sizeof(SymInt);

        alignUp(alignof(int32_t));
        devFunc->opAttrOffsetList_.AssignOffsetSize(currentOffset, kOpCount);
        int32_t *attrOffsets = reinterpret_cast<int32_t *>(funcDataPtr);
        for (size_t i = 0; i < kOpCount; i++) {
            attrOffsets[i] = static_cast<int32_t>(i);
        }
        currentOffset += kOpCount * sizeof(int32_t);
        funcDataPtr += kOpCount * sizeof(int32_t);

        alignUp(alignof(DevAscendOperation));
        devFunc->operationList_.AssignOffsetSize(currentOffset, kOpCount);
        DevAscendOperation *ops = reinterpret_cast<DevAscendOperation *>(funcDataPtr);
        for (size_t i = 0; i < kOpCount; i++) {
            new (&ops[i]) DevAscendOperation();
            ops[i].debugOpmagic = static_cast<uint64_t>(i + 1);
            size_t attrOffset = reinterpret_cast<uint8_t*>(attrData + i) - funcBuffer;
            ops[i].attrList.AssignOffsetSize(attrOffset, 1);
            ops[i].depGraphSuccList.AssignOffsetSize(0, 0);
            ops[i].depGraphPredCount = 0;
            ops[i].outcastStitchIndex = 0;
        }
    }

    DevAscendFunctionDuppedData* CreateDevAscendFunctionDuppedData(
            std::unique_ptr<uint8_t[]>& duppedDataBuffer,
            uint8_t*& duppedDataPtr,
            DevAscendFunction *devFunc,
            size_t kOpCount,
            size_t kDuppedDataBufferSize) {
        duppedDataBuffer = std::make_unique<uint8_t[]>(kDuppedDataBufferSize);
        memset_s(duppedDataBuffer.get(), kDuppedDataBufferSize, 0, kDuppedDataBufferSize);
        duppedDataPtr = duppedDataBuffer.get();

        DevAscendFunctionDuppedData *duppedData = reinterpret_cast<DevAscendFunctionDuppedData *>(duppedDataPtr);
        duppedDataPtr += sizeof(DevAscendFunctionDuppedData);

        duppedData->source_ = devFunc;
        duppedData->operationList_.size = kOpCount;
        duppedData->operationList_.predCountBase = static_cast<uint32_t>(duppedDataPtr - duppedDataBuffer.get());
        duppedData->operationList_.stitchBase = duppedData->operationList_.predCountBase + kOpCount * sizeof(predcount_t);
        duppedData->operationList_.stitchCount = 1;

        predcount_t *predCounts = reinterpret_cast<predcount_t *>(duppedDataPtr);
        for (size_t i = 0; i < kOpCount; i++) {
            predCounts[i] = 0;
        }
        duppedDataPtr += kOpCount * sizeof(predcount_t);

        for (size_t i = 0; i <= kOpCount; i++) {
            new (duppedDataPtr + i * sizeof(DevAscendFunctionDuppedStitchList)) DevAscendFunctionDuppedStitchList();
        }

        duppedData->incastList_.size = 0;
        duppedData->incastList_.base = 0;
        duppedData->outcastList_.size = 0;
        duppedData->outcastList_.base = 0;
        duppedData->expressionList_.size = 0;
        duppedData->expressionList_.base = 0;

        return duppedData;
    }

    void SetupTestEnvironment(DeviceTask& devTask,
                               std::unique_ptr<int32_t[]>& opWrapListData,
                               DevCceBinary* cceBinary,
                               size_t kOpCount) {
        opWrapListData = std::make_unique<int32_t[]>(kOpCount);
        for (size_t i = 0; i < kOpCount; i++) {
            opWrapListData[i] = static_cast<int32_t>(i);
        }

        devTask.mixTaskData.wrapIdNum = 1;
        devTask.mixTaskData.opWrapList[0] = reinterpret_cast<uint64_t>(opWrapListData.get());

        cceBinary[0].coreType = 0;
        cceBinary[0].psgId = 0;
        cceBinary[0].funcHash = 0xABCDEF00;
    }

    void VerifyDumpTopoOutput(const std::string& testFilePath, size_t expectedLineCount) {
        std::ifstream inFile(testFilePath);
        ASSERT_TRUE(inFile.is_open());
        std::string line;
        size_t lineCount = 0;
        while (std::getline(inFile, line)) {
            lineCount++;
            EXPECT_FALSE(line.empty());
        }
        inFile.close();

        EXPECT_EQ(lineCount, expectedLineCount);
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

TEST_F(TestDeviceTaskContext, test_dev_ascend_function_dupped_dump_topo) {
    constexpr size_t kOpCount = 4;
    constexpr size_t kFuncBufferSize = 4096;
    constexpr size_t kDuppedDataBufferSize = 2048;

    std::unique_ptr<uint8_t[]> funcBuffer;
    uint8_t *funcDataPtr;
    DevAscendFunction *devFunc = CreateDevAscendFunctionBuffer(funcBuffer, funcDataPtr, kOpCount, kFuncBufferSize);

    SetupDevAscendFunctionData(devFunc, funcDataPtr, funcBuffer.get(), kOpCount);

    std::unique_ptr<uint8_t[]> duppedDataBuffer;
    uint8_t *duppedDataPtr;
    DevAscendFunctionDuppedData *duppedData = CreateDevAscendFunctionDuppedData(
        duppedDataBuffer, duppedDataPtr, devFunc, kOpCount, kDuppedDataBufferSize);

    DevAscendFunctionDupped funcDupped;
    WsAllocation tinyAlloc;
    tinyAlloc.ptr = reinterpret_cast<uint64_t>(duppedData);
    funcDupped = DevAscendFunctionDupped(tinyAlloc);

    auto devTaskPtr = std::make_unique<DeviceTask>();
    DeviceTask &devTask = *devTaskPtr;
    std::unique_ptr<int32_t[]> opWrapListData;
    DevCceBinary cceBinary[1];
    SetupTestEnvironment(devTask, opWrapListData, cceBinary, kOpCount);

    std::string testFilePath = "./tmp/test_dump_topo_direct_output.txt";
    {
        std::ofstream outFile(testFilePath);
        ASSERT_TRUE(outFile.is_open());

        int seqNo = 0;
        int funcIdx = 0;
        bool enableVFFusion = false;

        funcDupped.DumpTopo(outFile, seqNo, funcIdx, cceBinary, enableVFFusion, &devTask);

        outFile.close();

        VerifyDumpTopoOutput(testFilePath, kOpCount);
    }
    std::remove(testFilePath.c_str());
}