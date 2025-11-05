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
 * \file TraceConfig.cpp
 * \brief
 */

#include "cost_model/simulation/config/TraceConfig.h"

using namespace std;

namespace CostModel {
TraceConfig::TraceConfig()
{
    Config::prefix = "Trace";
    Config::dispatcher = {
        {"countEventPrintInterval", [&](string v){ countEventPrintInterval = ParseInteger(v); }},
        {"sysClockTicks", [&](string v){ sysClockTicks = ParseInteger(v); }},
    };

    Config::recorder = {
        {"countEventPrintInterval", [&](){ return "countEventPrintInterval = " + ParameterToStr(countEventPrintInterval); }},
        {"sysClockTicks", [&](){ return "sysClockTicks = " + ParameterToStr(sysClockTicks); }},
    };
}
}