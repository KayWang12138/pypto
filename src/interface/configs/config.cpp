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
 * \file config.cpp
 * \brief
 */

#include "tilefwk/tilefwk.h"
#include "interface/program/program.h"

namespace npu::tile_fwk {
Config &Config::GetInstance() {
    static Config Config;
    return Config;
}

Config& Config::SetCycleUpperBound(int sgCycleUpperBound) {
    Program::GetInstance().GetConfig().Set<int>(SG_CYCLE_UPPER_BOUND, sgCycleUpperBound);
    return *this;
}
 
Config& Config::SetCycleLowerBound(int sgCycleLowerBound) {
    Program::GetInstance().GetConfig().Set<int>(SG_CYCLE_LOWER_BOUND, sgCycleLowerBound);
    return *this;
}
 
Config& Config::SetParallelNum(int sgParallelNum) {
    Program::GetInstance().GetConfig().Set<int>(SG_PARALLEL_NUM, sgParallelNum);
    return *this;
}

Config& Config::SetMachineSchMode(const std::vector<MachineScheduleConfig> &config) {
    uint8_t machineConfig = 0;
    for (size_t i = 0; i < config.size(); i++) {
        machineConfig |= static_cast<uint8_t>(config[i]);
    }

    Program::GetInstance().GetConfig().Set<uint8_t>(MACHINE_CONFIG, machineConfig);
    return *this;
}

} // end ascend