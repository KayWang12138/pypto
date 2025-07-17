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
 * \file AICPUConfig.cpp
 * \brief
 */

#include "simulation/config/AICPUConfig.h"

using namespace std;

namespace CostModel {
AICPUConfig::AICPUConfig()
{
    Config::prefix = "AICPU";
    Config::dispatcher = {
        {"completionCycles", [&](string v){ completionCycles = ParseInteger(v); }},
        {"schedulerCycles", [&](string v){ schedulerCycles = ParseInteger(v); }},
        {"resolveCycles", [&](string v){ resolveCycles = ParseInteger(v); }},
        {"threadsNum", [&](string v){ threadsNum = ParseInteger(v); }},
    };

    Config::recorder = {
        {"completionCycles", [&](){ return "completionCycles = " + ParameterToStr(completionCycles); }},
        {"schedulerCycles", [&](){ return "schedulerCycles = " + ParameterToStr(schedulerCycles); }},
        {"resolveCycles", [&](){ return "resolveCycles = " + ParameterToStr(resolveCycles); }},
        {"threadsNum", [&](){ return "threadsNum = " + ParameterToStr(threadsNum); }},
    };
}
}