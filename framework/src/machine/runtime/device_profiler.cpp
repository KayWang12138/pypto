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
 * \file device_profiler.cpp
 * \brief Implementation of device profiling and host profiling coordination.
 */

#include "machine/runtime/device_profiler.h"
#include "tilefwk/pypto_fwk_log.h"
#include "adapter/api/msprof_api.h"
#include "adapter/api/runtime_api.h"
#include "machine/runtime/runtime.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/device/dynamic/device_common.h"
#include "dump_device_perf.h"

namespace npu::tile_fwk {

void DeviceProfiler::ReportProfilingInfo(
    RtStream stream, uint64_t startTime, uint32_t blockDim, uint16_t taskType, bool isCore)
{
    if (hostProf_.GetProfType() == MSPF_COMMANDHANDLE_TYPE_START) {
        uint64_t endTime = MspfSysCycleTime();
        if (isCore) {
            uint32_t mixBlockDim = MIX_BLOCK_DIM;
            blockDim = (mixBlockDim << HIGHT_BIT) | blockDim;
            hostProf_.HostProfReportContextInfo(endTime);
        }
        if ((hostProf_.GetProfSwitch() & MSPF_TASK_TIME_L1_MASK) != 0) {
            hostProf_.HostProfReportNodeInfo(endTime, blockDim, taskType);
        }
        endTime = MspfSysCycleTime();
        hostProf_.HostProfReportApi(startTime, endTime);
    }
    if (taskType == MSPF_GE_TASK_TYPE_MIX_AIC) {
        hostProf_.HostProfReportCacheTaskInfo(stream, blockDim, taskType);
    }
}

void DeviceProfiler::SetDebugEnable(const DeviceArgs& args, std::vector<void*>& perfData)
{
    for (uint32_t i = 0; i < args.nrAic + args.nrAiv + AICPU_NUM_OF_RUN_AICPU_TASKS; i++) {
        ResetMetrics(i, perfData);
        RuntimeMemcpy(
            (reinterpret_cast<uint8_t*>(args.sharedBuffer + sizeof(uint64_t) * SHAK_BUF_DFX_DATA_INDEX)) +
                i * SHARED_BUFFER_SIZE,
            sizeof(uint64_t), reinterpret_cast<uint8_t*>(&perfData[i]), sizeof(uint64_t),
            RtMemcpyKind::HOST_TO_DEVICE);
    }
    MACHINE_LOGD("Set debug enable aicore 0 devPtr: %p", perfData[0]);
}

void DeviceProfiler::ResetMetrics(const uint32_t& coreId, std::vector<void*>& perfData)
{
    if (perfData.empty()) {
        return;
    }
    if (enableDumpMachinePerfTrace_) {
        if (!g_is_machine_trace_addr_inited) {
            RuntimeMemset(perfData[coreId], sizeof(Metrics), 0, sizeof(Metrics));
            g_is_machine_trace_addr_inited = true;
        }
    } else {
        RuntimeMemset(perfData[coreId], sizeof(Metrics), 0, sizeof(Metrics));
    }
}

void DeviceProfiler::ResetPerData(const DeviceArgs& args, std::vector<void*>& perfData)
{
    auto size = MAX_DFX_TASK_NUM_PER_CORE * sizeof(TaskStat) + sizeof(Metrics);
    for (uint64_t i = 0; i < args.nrAic + args.nrAiv + AICPU_NUM_OF_RUN_AICPU_TASKS; i++) {
        int rc = RuntimeMemset(perfData[i], size, 0, size);
        if (rc != 0) {
            MACHINE_LOGW("CoreId %lu, rtMemSet failed, rc: %d", i, rc);
        }
    }
}

void DeviceProfiler::DumpAiCoreExecutionTimeData(DeviceArgs& args, const std::vector<void*>& perfData)
{
    args.nrValidAic = dynamic::GetCfgBlockdim();
    args.scheCpuNum = dynamic::CalcSchAicpuNumByBlockDim(args.nrValidAic, args.nrAicpu, args.archInfo);
    npu::tile_fwk::dynamic::DumpAicoreTaskExectInfo(args, perfData);
}

void DeviceProfiler::DumpAiCorePmuData() { MACHINE_LOGI("TODO: DumpAiCorePmuData"); }

void DeviceProfiler::SynchronizeDeviceToHostProfData(DeviceArgs& args, const std::vector<void*>& perfData)
{
    if (config::GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE) == CFG_DEBUG_ALL) {
        DumpAiCoreExecutionTimeData(args, perfData);
    }
}

int DeviceProfiler::RunPrepare(const DeviceArgs& args, const std::vector<void*>& perfData)
{
    int ret = 0;
    if (config::GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE) == CFG_DEBUG_ALL || ENABLE_PERF_TRACE == 1 ||
        PMU_COLLECT == 1) {
        for (uint32_t i = 0; i < args.nrAic + args.nrAiv + AICPU_NUM_OF_RUN_AICPU_TASKS; i++) {
            auto preCoreShareadBufferAddr =
                (reinterpret_cast<uint8_t*>(args.sharedBuffer + sizeof(uint64_t) * SHAK_BUF_DFX_DATA_INDEX)) +
                i * SHARED_BUFFER_SIZE;
            ret = RuntimeMemcpy(
                preCoreShareadBufferAddr, sizeof(uint64_t), reinterpret_cast<uint8_t*>(&perfData[i]), sizeof(uint64_t),
                RtMemcpyKind::HOST_TO_DEVICE);
        }
    }
    return ret;
}

void DeviceProfiler::AllocDfxMetricMemory(const DeviceArgs& args, std::vector<void*>& perfData)
{
    for (uint32_t i = 0; i < args.nrAic + args.nrAiv + AICPU_NUM_OF_RUN_AICPU_TASKS; i++) {
        KernelArgs kernelArgs;
        memset_s(&kernelArgs, sizeof(kernelArgs), 0, sizeof(kernelArgs));
        uint8_t* dfxAddr = nullptr;
        machine::GetRA()->AllocDevAddr(&dfxAddr, MAX_DFX_TASK_NUM_PER_CORE * sizeof(TaskStat) + sizeof(Metrics));
        kernelArgs.shakeBuffer[SHAK_BUF_DFX_DATA_INDEX] = reinterpret_cast<int64_t>(dfxAddr);
        perfData.push_back(dfxAddr);
        RuntimeMemcpy(
            (reinterpret_cast<uint8_t*>(args.sharedBuffer)) + i * SHARED_BUFFER_SIZE, sizeof(kernelArgs),
            reinterpret_cast<uint8_t*>(&kernelArgs), sizeof(kernelArgs), RtMemcpyKind::HOST_TO_DEVICE);
        MACHINE_LOGI("aicore %u , dfxaddr 0x%ld \n", i, kernelArgs.shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
    }
}

void DeviceProfiler::Dump(const DeviceArgs& args, const std::vector<void*>& perfData)
{
    (void)perfData;
    MACHINE_LOGI("======== aicore status ========");

    int coreNum = args.nrAic + args.nrAiv + AICPU_NUM_OF_RUN_AICPU_TASKS;
    uint64_t size = coreNum * SHARED_BUFFER_SIZE;
    std::vector<uint64_t> buffer(size / sizeof(uint64_t));
    int rc = RuntimeMemcpy(buffer.data(), size, reinterpret_cast<void*>(args.sharedBuffer), size,
                           RtMemcpyKind::DEVICE_TO_HOST);
    if (rc != 0) {
        MACHINE_LOGI("rtmemcpy failed");
        return;
    }

    uint64_t buffAddr = reinterpret_cast<uint64_t>(buffer.data());
    for (int i = 0; i < coreNum; i++) {
        KernelArgs* arg = reinterpret_cast<KernelArgs*>(buffAddr + i * SHARED_BUFFER_SIZE);
        MACHINE_LOGI("aicore %d hello status %ld", i, arg->shakeBuffer[0]);
        MACHINE_LOGI("last_taskId %ld", arg->shakeBuffer[1]);
        MACHINE_LOGI("task status %ld", arg->shakeBuffer[2]);
    }
}

void DeviceProfiler::StartMachinePerfTraceDumpThread(const DeviceArgs& args,
                                                     const std::vector<void*>& perfData)
{
    if (!enableDumpMachinePerfTrace_) {
        return;
    }
    if (dumpThread_.joinable()) {
        return;
    }
    dumpThreadStopFlag_.store(false);
    dumpThread_ = std::thread(&DeviceProfiler::MachinePerfTraceDumpThread, this, std::cref(args), std::cref(perfData));
    MACHINE_LOGI("Dump thread started");
}

void DeviceProfiler::StopMachinePerfTraceDumpThread()
{
    if (!dumpThread_.joinable()) {
        return;
    }
    dumpThreadStopFlag_.store(true);
    if (dumpThread_.joinable()) {
        dumpThread_.join();
    }
    MACHINE_LOGD("Dump thread stopped");
}

void DeviceProfiler::MachinePerfTraceDumpThread(const DeviceArgs& args,
                                                const std::vector<void*>& perfData)
{
    MACHINE_LOGD("Dump thread start to machine perf trace data");
    while (!dumpThreadStopFlag_.load()) {
        usleep(10000);
        npu::tile_fwk::dynamic::DumpDevTaskPerfData(args, perfData, false);
    }
    MACHINE_LOGD("Dump thread final dump");
    npu::tile_fwk::dynamic::DumpDevTaskPerfData(args, perfData, true);
}

} // namespace npu::tile_fwk
