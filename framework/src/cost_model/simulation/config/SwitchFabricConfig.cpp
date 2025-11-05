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
 * \file SwitchFabricConfig.cpp
 * \brief
 */

#include "cost_model/simulation/config/SwitchFabricConfig.h"

using namespace std;

namespace CostModel {
SwitchFabricConfig::SwitchFabricConfig()
{
    Config::prefix = "SwitchFabric";
    Config::dispatcher = {
        {"simCommLatency", [&](string v){ simCommLatency = ParseBoolean(v); }},
    };

    Config::recorder = {
        {"simCommLatency", [&](){ return "simCommLatency = " + ParameterToStr(simCommLatency); }},
    };
}
}