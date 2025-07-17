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
 * \file TraceConfig.h
 * \brief
 */

// generated from config.toml
#pragma once

#include <cstdint>
#include <string>
#include "simulation/base/Config.h"

namespace CostModel {
struct TraceConfig : public Config {
    TraceConfig();
    uint64_t countEventPrintInterval = 100;
    uint64_t sysClockTicks = 1800000;
};
}