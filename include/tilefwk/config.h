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
 * \file config.h
 * \brief
 */

#pragma once

namespace npu::tile_fwk {

enum class MachineScheduleConfig {
    DEFAULT_SCH = 0x0, // default sch mode:L2CACHE_AFFINITY_SCH(disable) MULTI_CORE_FAIR_SCH(disable)
    L2CACHE_AFFINITY_SCH = 0x1, // Dispatch the most recently ready task to maximize cache reuse
    MULTI_CORE_FAIR_SCH = 0x2 // Fair scheduling refers to maintaining as balanced a distribution of tasks across cores as possible,
                              // Enabling this configuration will introduce some additional public scheduling overhead.
};

class Config {
public:
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;

    static Config &GetInstance();

    /**
     * @brief Set the Cycle Upper Bound object
     *
     * @param sgCycleUpperBound
     * @return Config&
     */
    Config& SetCycleUpperBound(int sgCycleUpperBound);

    Config& SetCycleLowerBound(int sgCycleLowerBound);

    Config& SetParallelNum(int sgParallelNum);

    /**
     * @brief Set the Machine Sch Mode object
     *
     * @param config
     * @return Config&
     */
    Config& SetMachineSchMode(const std::vector<MachineScheduleConfig> &config);
private:
    Config() = default;
    ~Config() = default;
};

} // end npu::tile_fwk
