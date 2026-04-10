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
 * \file e2e_host_sim_launcher.cpp
 * \brief DebugMode E2E host simulation launcher
 */

#include "machine/runtime/e2e_host_sim_launcher.h"

#include "machine/runtime/device_launcher.h"
#include "machine/utils/machine_error.h"
#include "tilefwk/platform.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk::dynamic {

namespace {
constexpr int64_t kAivPerAic = 2;
} // namespace

E2EExecutionProfile E2EHostSimLauncher::BuildExecutionProfileByBlockDim(int64_t blockdim, int64_t scheCpuNum)
{
    E2EExecutionProfile profile;
    profile.blockdim = blockdim;
    profile.scheCpuNum = scheCpuNum;
    profile.aicoreLogicalNum = blockdim * (kAivPerAic + 1);
    profile.aicpuThreadNum = scheCpuNum + 1; // 1 ctrl + sche
    return profile;
}

int E2EHostSimLauncher::E2EHostSimRunOnce(
    Function* function, DevControlFlowCache* ctrlCache, const DeviceLauncherConfig& config)
{
    if (function == nullptr) {
        MACHINE_LOGE(DevCommonErr::NULLPTR, "E2EHostSimRunOnce invalid function.");
        return -1;
    }

    DeviceLauncherConfig launchConfig = config;
    DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(launchConfig);
    auto archInfo = static_cast<ArchInfo>(Platform::Instance().GetSoc().GetNPUArch());
    auto scheCpuNum =
        static_cast<int64_t>(CalcSchAicpuNumByBlockDim(launchConfig.blockdim, launchConfig.aicpuNum, archInfo));
    auto profile = BuildExecutionProfileByBlockDim(launchConfig.blockdim, scheCpuNum);
    MACHINE_LOGI(
        "[E2EHostSim] blockdim=%ld scheCpuNum=%ld aicoreLogicalNum=%ld aicpuThreadNum=%ld", profile.blockdim,
        profile.scheCpuNum, profile.aicoreLogicalNum, profile.aicpuThreadNum);

    // Phase-1 implementation: reuse host emulation execution path while preserving E2E launch routing.
    return EmulationLauncher::EmulationRunOnce(function, ctrlCache, launchConfig);
}

int E2EHostSimLauncher::E2EHostSimLaunchDeviceTensorData(
    Function* function, const std::vector<DeviceTensorData>& inputList, const std::vector<DeviceTensorData>& outputList,
    const DeviceLauncherConfig& config)
{
    if (function == nullptr) {
        MACHINE_LOGE(DevCommonErr::NULLPTR, "E2EHostSimLaunchDeviceTensorData invalid function.");
        return -1;
    }

    DeviceLauncherConfig launchConfig = config;
    DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(launchConfig);
    auto archInfo = static_cast<ArchInfo>(Platform::Instance().GetSoc().GetNPUArch());
    auto scheCpuNum =
        static_cast<int64_t>(CalcSchAicpuNumByBlockDim(launchConfig.blockdim, launchConfig.aicpuNum, archInfo));
    auto profile = BuildExecutionProfileByBlockDim(launchConfig.blockdim, scheCpuNum);
    MACHINE_LOGI(
        "[E2EHostSim] DeviceTensorData blockdim=%ld scheCpuNum=%ld aicoreLogicalNum=%ld aicpuThreadNum=%ld",
        profile.blockdim, profile.scheCpuNum, profile.aicoreLogicalNum, profile.aicpuThreadNum);

    return EmulationLauncher::EmulationLaunchDeviceTensorData(function, inputList, outputList, launchConfig);
}

} // namespace npu::tile_fwk::dynamic
