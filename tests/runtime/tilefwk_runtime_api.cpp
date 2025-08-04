/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tilefwk_runtime_api.h"
#include "machine/host/device_agent_task.h"
#include "machine_agent.h"

namespace npu::tile_fwk {
void RunAsync(const void *stream, const void *workSpaceGmAddr, void *handle,
              const std::vector<void *> &opOriginArgs, const std::vector<size_t> &argsSize) {
    ALOG_INFO("Program::Run stream = ", stream);
    DeviceAgentTask *deviceAgentTask = reinterpret_cast<DeviceAgentTask *>(handle);
    int ret = Run(stream, workSpaceGmAddr, deviceAgentTask, opOriginArgs, argsSize, true);
    ALOG_INFO("End run: func name = ", deviceAgentTask->compileTask->GetFunction()->GetRawName(), " ret = ", ret);
}

int32_t TileFwkRunAsync(void *handle, const void *workspace, const void *stream, const std::vector<void *> &opArgs,
                        const std::vector<size_t> &prefetchSizes) {
    RunAsync(stream, workspace, handle, opArgs, prefetchSizes);
    return 0;
}
}
