/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "interface/machine/host/machine_task.h"

namespace npu::tile_fwk {
class DeviceAgentTask {
public:
    explicit DeviceAgentTask(std::shared_ptr<MachineTask> task) : compileTask(task) {}

    ~DeviceAgentTask() {}

    uint64_t GetTaskId() const { return this->compileTask->GetTaskId(); }

    Function *GetFunction() const { return this->compileTask->GetFunction(); }

public:
    std::shared_ptr<MachineTask> compileTask{nullptr};
private:
};
using DeviceAgentTaskPtr = std::shared_ptr<DeviceAgentTask>;
extern DeviceAgentTaskPtr gDeviceAgentTaskPtr;
}
