/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file task_dump_utils.h
 * \brief dump and recover for DeviceAgentTask
 */

#pragma once

#include "interface/machine/host/machine_task.h"
#include "runtime/host/machine_agent.h"

namespace npu::tile_fwk {
class TaskDumpUtils {
 public:
    static bool DumpTaskToBinFile(const DeviceAgentTask *deviceAgentTask, const std::string &dumpFileName);
    static bool RecoverTaskFromBinFile(const std::string &binFilePath, DeviceAgentTask *deviceAgentTask);
};
}
