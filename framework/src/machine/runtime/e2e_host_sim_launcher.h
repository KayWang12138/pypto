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
 * \file e2e_host_sim_launcher.h
 * \brief DebugMode E2E host simulation launcher
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "machine/runtime/device_launcher_binding.h"

namespace npu::tile_fwk::dynamic {

struct E2EExecutionProfile {
    int64_t blockdim{0};
    int64_t scheCpuNum{0};
    int64_t aicoreLogicalNum{0};
    int64_t aicpuThreadNum{0};
};

struct E2EProtocolSnapshot {
    int64_t bindCoreNum{0};
    uint64_t pgmask{0};
    int64_t interceptedBlockdim{0};
    int64_t realRtCallCount{0};
    uint64_t logicalClockNs{0};
    bool forcedRecycle{false};
    std::vector<int32_t> logicalToPhysical;
    std::vector<std::string> events;
};

class E2EHostSimLauncher {
public:
    static E2EExecutionProfile BuildExecutionProfileByBlockDim(int64_t blockdim, int64_t scheCpuNum);

    static int64_t ResolveBindCoreNum(int64_t aicoreLogicalNum);
    static uint64_t BuildDefaultPgMask(int64_t blockdim);
    static std::vector<int32_t> BuildLogicalToPhysicalMap(int64_t aicoreLogicalNum, int64_t bindCoreNum);
    static E2EProtocolSnapshot SimulateProtocolOnce(
        const E2EExecutionProfile& profile, uint64_t pgmask, int64_t bindCoreNum);

    static int E2EHostSimRunOnce(
        Function* function, DevControlFlowCache* ctrlCache,
        const DeviceLauncherConfig& config = DeviceLauncherConfig());

    static int E2EHostSimLaunchDeviceTensorData(
        Function* function, const std::vector<DeviceTensorData>& inputList,
        const std::vector<DeviceTensorData>& outputList, const DeviceLauncherConfig& config = DeviceLauncherConfig());
};

} // namespace npu::tile_fwk::dynamic
