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

class E2EHostSimLauncher {
public:
    static int E2EHostSimRunOnce(
        Function* function, DevControlFlowCache* inputCtrlCache, const DeviceLauncherConfig& config = DeviceLauncherConfig());

    static int E2EHostSimLaunchDeviceTensorData(
            Function* function, const std::vector<DeviceTensorData>& inDevList, const std::vector<DeviceTensorData>& outDevList,
    const DeviceLauncherConfig& config = DeviceLauncherConfig());
};  

}
