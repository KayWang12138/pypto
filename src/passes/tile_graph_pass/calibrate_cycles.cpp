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
 * \file calibrate_cycles.cpp
 * \brief
 */

#include <fstream>
#include <cstring>
#include <unistd.h>
#include <sys/stat.h>

#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_utils/reschedule_utils.h"
#include "passes/pass_interface/pass.h"
#include "calibrate_cycles.h"

namespace npu::tile_fwk {

bool CalibratedCyclesPass::LoadCalibratedCyclesCSV(const std::string csvFilePath,
    std::unordered_map<unsigned long, int>& cyclesMap) {
    cyclesMap.clear();
    std::ifstream inFile(csvFilePath);
    std::string str;
    if (!inFile) {
        ALOG_WARN << "Error: Open Calibrated Cycles CSV file failed";
        return false;
    }
    getline(inFile, str);
    while (getline(inFile, str)) {
        size_t pos = str.find(',', 1);
        if (pos != std::string::npos) {
            unsigned long hash = std::stoul(str.substr(0, pos).c_str());
            int cycles = std::atoi(str.substr(pos + 1).c_str());
            if (cycles > 0)
                cyclesMap[hash] = cycles;
        }
    }
    return true;
}

Status CalibratedCyclesPass::RunOnFunction(Function &function) {
    ASLOGI("===> Start CalibratedCyclesPass.");
    std::unordered_map<unsigned long, int> cyclesMap;
    LoadCalibratedCyclesCSV("./calibrated_cycles.csv", cyclesMap);

    for (auto& op : function.Operations()) {
        unsigned long hash = RescheduleUtils::ComputeOperationHash(&op);
        if (hash != 0) {
            if (cyclesMap.find(hash) == cyclesMap.end()) {
                ALOG_WARN << "Op hash " << hash << " [op : " << op.GetOpcodeStr() << "] not in calibrated_cycles.csv";
                continue;
            }
            op.UpdateLatency(cyclesMap[hash]);
        }
    }
    ASLOGI("===> Finish CalibratedCyclesPass.");
    return SUCCESS;
}
}  // namespace npu::tile_fwk