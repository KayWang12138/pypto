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
 * \file DeviceStats.cpp
 * \brief
 */

#include "simulation/statistics/DeviceStats.h"

#include <climits>
#include <iostream>

void DeviceStats::Report(std::string &name)
{
    rpt->ReportTitle("DeviceMachine " + name + " Statistics");
    rpt->ReportVal("Total Submit Tasks", totalSubmitNum);
    rpt->ReportValAndPct("  |--Cube Tasks", cubeSubmitNum, totalSubmitNum);
    rpt->ReportValAndPct("  |--Vector Tasks", vectorSubmitNum, totalSubmitNum);

    float avgTaskExecuteCycle = ((totalSubmitNum != 0) ?
                                (float(totalTaskExecuteCycles) / totalSubmitNum) : float(0.0));
    rpt->ReportVal("Average Task Cycles", avgTaskExecuteCycle);
    rpt->ReportVal("  |--Max Task Cycles", maxTaskExecuteCycles);
    rpt->ReportVal("  |--Min Task Cycles", minTaskExecuteCycles);
    rpt->ReportVal("Total Resolve Times", resolveNum);
    rpt->ReportVal("Total Batch", pollingNum);
}

void DeviceStats::Reset()
{
    totalSubmitNum = 0;
    cubeSubmitNum = 0;
    vectorSubmitNum = 0;
    totalTaskExecuteCycles = 0;
    maxTaskExecuteCycles = 0;
    resolveNum = 0;
    pollingNum = 0;
    minTaskExecuteCycles = INT_MAX;
}