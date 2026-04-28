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
 * \file device_profiler.h
 * \brief Manages device-side profiling, host profiling coordination, and perf data lifecycle.
 */

#pragma once

#include <vector>
#include <thread>
#include <atomic>
#include "machine/runtime/host_prof.h"
#include "interface/machine/device/tilefwk/aicpu_common.h"
#include "adapter/api/runtime_define.h"

namespace npu::tile_fwk {

class DeviceProfiler {
public:
    DeviceProfiler() = default;

    // HostProf access (for backward compatibility)
    HostProf& GetHostProf() { return hostProf_; }

    // Profiling report coordinator (replaces DeviceRunner::ReportHostProfInfo)
    void ReportProfilingInfo(RtStream stream, uint64_t startTime,
                             uint32_t blockDim, uint16_t taskType, bool isCore);

    // Debug / DFX control
    void SetDebugEnable(const DeviceArgs& args, std::vector<void*>& perfData);
    void ResetMetrics(const uint32_t& coreId, std::vector<void*>& perfData);
    void ResetPerData(const DeviceArgs& args, std::vector<void*>& perfData);

    // Data dump
    void DumpAiCoreExecutionTimeData(DeviceArgs& args, const std::vector<void*>& perfData);
    void DumpAiCorePmuData();
    void SynchronizeDeviceToHostProfData(DeviceArgs& args, const std::vector<void*>& perfData);

    // Threaded perf trace dump
    bool GetEnableDumpDevPref() const { return enableDumpMachinePerfTrace_; }
    void SetEnableDump(bool enable) { enableDumpMachinePerfTrace_ = enable; }
    void StartMachinePerfTraceDumpThread(const DeviceArgs& args,
                                         const std::vector<void*>& perfData);
    void StopMachinePerfTraceDumpThread();

    // Launch preparation
    int RunPrepare(const DeviceArgs& args, const std::vector<void*>& perfData);

    // Initialization helpers
    void AllocDfxMetricMemory(const DeviceArgs& args, std::vector<void*>& perfData);
    void Dump(const DeviceArgs& args, const std::vector<void*>& perfData);

private:
    void MachinePerfTraceDumpThread(const DeviceArgs& args,
                                    const std::vector<void*>& perfData);

    HostProf hostProf_;
    bool enableDumpMachinePerfTrace_{false};

    std::thread dumpThread_;
    std::atomic<bool> dumpThreadStopFlag_{false};
};

} // namespace npu::tile_fwk
