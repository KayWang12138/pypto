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
 * \file AICPUStats.cpp
 * \brief
 */

#include "simulation/statistics/AICPUStats.h"

#include <climits>
#include <iostream>

void AICPUStats::Report(std::string &name)
{
    if (threadSubmitNum.size() > 1) {
        rpt->ReportTitle("AICPUMachine " + name + " Thread Details");
        rpt->ReportMap("Thread Submit Detail:", threadSubmitNum);
        rpt->ReportMap("Thread Resolve Detail:", threadResolveNum);
        rpt->ReportMap("Thread Batch Detail:", threadBatchNum);
    }
}

void AICPUStats::Reset()
{
    totalSubmitNum = 0;
    cubeSubmitNum = 0;
    vectorSubmitNum = 0;
    totalTaskExecuteCycles = 0;
    maxTaskExecuteCycles = 0;
    resolveNum = 0;
    pollingNum = 0;

    minTaskExecuteCycles = INT_MAX;
    threadResolveNum = std::map<uint64_t, uint64_t>();
    threadBatchNum = std::map<uint64_t, uint64_t>();
    threadSubmitNum = std::map<uint64_t, uint64_t>();
}