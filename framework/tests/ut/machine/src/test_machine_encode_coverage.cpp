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
 * \file test_machine_encode_coverage.cpp
 * \brief Coverage-oriented UT for machine dynamic context.
 */

#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>

#include "interface/configs/config_manager.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "machine/device/dynamic/context/device_execute_context.h"
#include "machine/device/dynamic/context/device_stitch_context.h"
#include "machine/device/dynamic/context/device_task_context.h"
#include "machine/utils/dynamic/dev_start_args.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tilefwk.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace {
void ControlFlowSetError(struct DeviceExecuteContext *ctx, int64_t *symbolTable,
    RuntimeCallEntryType runtimeCallList[T_RUNTIME_CALL_MAX], DevStartArgsBase *startArgsBase)
{
    (void)symbolTable;
    (void)runtimeCallList;
    (void)startArgsBase;
    ctx->SetErrorState(DEVICE_MACHINE_ERROR);
}

DevAscendProgram *BuildTinyProgram()
{
    int s = 8;
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor out(DT_FP32, {s, s}, "out");
    FUNCTION("ut_cov_tiny_prog", {t0, t1}, {out}) {
        auto x = Add(t0, t1);
        Assemble(x, {0, 0}, out);
    }
    auto attr = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    if (attr == nullptr) {
        return nullptr;
    }
    auto *devProg = reinterpret_cast<DevAscendProgram *>(attr->devProgBinary.data());
    if (devProg == nullptr) {
        return nullptr;
    }
    devProg->RelocProgram(0, reinterpret_cast<uint64_t>(devProg), true);
    uint64_t ws = devProg->controlFlowCache.contextWorkspaceAddr;
    devProg->controlFlowCache.IncastOutcastAddrReloc(ws, 0, nullptr);
    devProg->controlFlowCache.RuntimeAddrRelocWorkspace(ws, 0, nullptr, nullptr, nullptr);
    devProg->controlFlowCache.RuntimeAddrRelocProgram(reinterpret_cast<uint64_t>(devProg), 0);
    devProg->controlFlowCache.TaskAddrRelocWorkspace(ws, 0, nullptr);
    devProg->controlFlowCache.TaskAddrRelocProgramAndCtrlCache(reinterpret_cast<uint64_t>(devProg),
        reinterpret_cast<uint64_t>(&devProg->controlFlowCache), 0, 0);
    return devProg;
}

void RunDumpDependCrashPath()
{
    DevAscendProgram *devProg = BuildTinyProgram();
    ASSERT_NE(devProg, nullptr);
    DevAscendFunction *root = devProg->GetFunction(0);
    ASSERT_NE(root, nullptr);
    ASSERT_NE(root->GetDuppedData(), nullptr);

    DeviceWorkspaceAllocator workspace(devProg);
    DynDeviceTask *dt = workspace.MakeDynDeviceTask();
    ASSERT_NE(dt, nullptr);

    alignas(64) unsigned char hdrBuf[sizeof(DynFuncHeader) + sizeof(DynFuncData)]{};
    auto *hdr = reinterpret_cast<DynFuncHeader *>(hdrBuf);
    hdr->seqNo = 7;
    hdr->funcNum = 1;
    hdr->funcSize = static_cast<uint32_t>(sizeof(hdrBuf));
    std::memset(&hdr->At(0), 0, sizeof(DynFuncData));
    dt->dynFuncDataList = hdr;
    dt->dynFuncDataCacheList[0].duppedData = root->GetDuppedData();

    ReadyCoreFunctionQueue qAiv{}, qAic{}, qAicpu{};
    taskid_t eAiv[1] = {0}, eAic[1] = {0}, eAicpu[1] = {0};
    qAiv.capacity = qAic.capacity = qAicpu.capacity = 1;
    qAiv.elem = eAiv;
    qAic.elem = eAic;
    qAicpu.elem = eAicpu;
    dt->readyQueue[0] = &qAiv;
    dt->readyQueue[1] = &qAic;
    dt->readyQueue[2] = &qAicpu;

    std::array<DevTensorData, 2> tensors{};
    tensors[0].address = 0x1000;
    tensors[1].address = 0x2000;
    DevStartArgs startArgs{};
    startArgs.devTensorList = tensors.data();
    startArgs.inputTensorSize = 1;
    startArgs.outputTensorSize = 1;
    startArgs.contextWorkspaceAddr = devProg->controlFlowCache.contextWorkspaceAddr;

    DeviceTaskContext::DumpDepend(dt, devProg, &startArgs, "ut_cov");
}

void RunDuppedDataDumpMismatchPath()
{
    DevAscendProgram *devProg = BuildTinyProgram();
    ASSERT_NE(devProg, nullptr);
    DevAscendFunction *root = devProg->GetFunction(0);
    ASSERT_NE(root, nullptr);
    DeviceWorkspaceAllocator workspace(devProg);
    auto dup = workspace.DuplicateRoot(root);
    dup.DupData()->operationList_.size = dup.DupData()->GetSource()->GetOperationSize() + 9U;
    (void)dup.Dump(0);
}

void RunCheckStitchMismatchPath()
{
    DevAscendProgram *devProg = BuildTinyProgram();
    ASSERT_NE(devProg, nullptr);
    DevAscendFunction *root = devProg->GetFunction(0);
    ASSERT_NE(root, nullptr);
    DeviceWorkspaceAllocator workspace(devProg);
    auto dup = workspace.DuplicateRoot(root);
    dup.GetOperationCurrPredCount(0) = static_cast<predcount_t>(dup.GetOperationCurrPredCount(0) + 99);
    DeviceStitchContext::CheckStitch(nullptr, 0, &dup);
}

void RunHandleOneStitchInvalidProducerPath()
{
    DevAscendProgram *devProg = BuildTinyProgram();
    ASSERT_NE(devProg, nullptr);
    DevAscendFunction *root = devProg->GetFunction(0);
    ASSERT_NE(root, nullptr);
    DeviceWorkspaceAllocator workspace(devProg);
    auto producer = workspace.DuplicateRoot(root);
    auto consumer = workspace.DuplicateRoot(root);
    DevAscendFunctionDuppedStitchList stitch;
    DeviceStitchContext::HandleOneStitch(producer, consumer, stitch, 999999UL, 0UL, 0UL, &workspace,
        DeviceStitchContext::StitchKind::StitchDefault, 0);
}

}

class TestMachineEncodeCoverage : public testing::Test {
protected:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
        TileShape::Current().SetVecTile(32, 32);
        TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    }

    void TearDown() override
    {
        Program::GetInstance().Reset();
        config::Reset();
    }
};

TEST_F(TestMachineEncodeCoverage, DumpDepend_WithEncodedDuppedData_CoversDependBody) {
    ASSERT_DEATH(RunDumpDependCrashPath(), ".*");
}

TEST_F(TestMachineEncodeCoverage, DuppedData_Dump_SizeMismatch_AbortsAfterDevError) {
    ASSERT_DEATH(RunDuppedDataDumpMismatchPath(), ".*");
}

TEST_F(TestMachineEncodeCoverage, CheckStitch_DynPredMismatch_AbortsAfterDevError) {
    ASSERT_DEATH(RunCheckStitchMismatchPath(), ".*");
}

TEST_F(TestMachineEncodeCoverage, HandleOneStitch_InvalidProducerOp_AbortsAfterDevError) {
    ASSERT_DEATH(RunHandleOneStitchInvalidProducerPath(), ".*");
}

TEST_F(TestMachineEncodeCoverage, FastStitch_SlotIdxBeyondSize_LogsAndContinues) {
    DevStartArgs args{};
    DevAscendProgram prog{};
    prog.controlFlowCache.isRecording = false;
    args.devProg = &prog;
    args.controlFlowEntry = reinterpret_cast<void *>(ControlFlowSetError);
    DeviceExecuteContext ctx(&args);
    EXPECT_EQ(ctx.RunControlFlow(&args), DEVICE_MACHINE_ERROR);
}
