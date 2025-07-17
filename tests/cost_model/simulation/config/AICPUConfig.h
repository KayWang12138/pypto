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
 * \file AICPUConfig.h
 * \brief
 */

// generated from config.toml
#pragma once

#include <cstdint>
#include <string>
#include "simulation/base/Config.h"

namespace CostModel {
struct AICPUConfig : public Config {
    AICPUConfig();
    uint64_t completionCycles = 240;
    uint64_t schedulerCycles = 200;
    uint64_t resolveCycles = 390;
    uint64_t threadsNum = 1;
};
}