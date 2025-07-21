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
 * \file machine_dump.h
 * \brief
 */

#ifndef MACHINE_DUMP_H
#define MACHINE_DUMP_H

#include <iostream>
#include "machine/utils/machine_ws_intf.h"
#include "interface/machine/host/machine_task.h"
#include "machine/host/machine_agent.h"

namespace npu::tile_fwk {
class MachineDump {
 public:
    static bool DumpASTBinData(const DeviceAgentTask *deviceAgentTask,
                               const std::string &dumpFileName, const std::string &dumpPath);
    static void GetDumpBinData(const DeviceAgentTask *deviceAgentTask, std::vector<uint8_t> &binData);
 private:
    static std::string PrepareBinPath();
    static void InitTaskBinData(const DeviceAgentTask *deviceAgentTask, DeviceTaskBin &taskBin);
    static void DumpJsonFile(const DeviceAgentTask *deviceAgentTask, const std::string &dumpFileName,
                             const std::string &dumpPath);
    static bool DumpBinAndJsonToFile(const DeviceAgentTask *deviceAgentTask,
                                     const DeviceTaskBin* binData, const size_t allSize,
                                     const std::string &dumpFileName, const std::string &dumpPath);
};
}
#endif // MACHINE_DUMP_H
