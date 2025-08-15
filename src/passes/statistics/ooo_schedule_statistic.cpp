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
 * \file ooo_schedule_statistic.cpp
 * \brief
 */

#include "ooo_schedule_statistic.h"
namespace npu {
namespace tile_fwk {

constexpr int32_t percent = 100;

Json OoOSchedulerCheck::HealthCheckOoOSchedule() {
    Json report;
    int64_t maxL0ASize = PassConfigManager::Instance().GetPlatformConfig().GetMemoryLimit(MemoryType::MEM_L0A);
    int64_t maxL0BSize = PassConfigManager::Instance().GetPlatformConfig().GetMemoryLimit(MemoryType::MEM_L0B);
    int64_t maxL0CSize = PassConfigManager::Instance().GetPlatformConfig().GetMemoryLimit(MemoryType::MEM_L0C);
    int64_t maxUBSize = PassConfigManager::Instance().GetPlatformConfig().GetMemoryLimit(MemoryType::MEM_UB);
    int64_t maxL1Size = PassConfigManager::Instance().GetPlatformConfig().GetMemoryLimit(MemoryType::MEM_L1);
    // Workspace Info
    report["Workspace Offset (bytes)"] = workspaceOffset;
    // Execution Info
    report["Total Cycles"] = clock;
    // Pipe Usage Rate
    report["PIPE_S Usage Rate (%)"] = static_cast<float>(pipeUsageCount.at(PipeType::PIPE_S)) / clock * percent;
    report["PIPE_V Usage Rate (%)"] = static_cast<float>(pipeUsageCount.at(PipeType::PIPE_V)) / clock * percent;
    report["PIPE_M Usage Rate (%)"] = static_cast<float>(pipeUsageCount.at(PipeType::PIPE_M)) / clock * percent;
    report["PIPE_MTE1 Usage Rate (%)"] = static_cast<float>(pipeUsageCount.at(PipeType::PIPE_MTE1)) / clock * percent;
    report["PIPE_MTE2 Usage Rate (%)"] = static_cast<float>(pipeUsageCount.at(PipeType::PIPE_MTE2)) / clock * percent;
    report["PIPE_MTE3 Usage Rate (%)"] = static_cast<float>(pipeUsageCount.at(PipeType::PIPE_MTE3)) / clock * percent;
    report["PIPE_FIX Usage Rate (%)"] = static_cast<float>(pipeUsageCount.at(PipeType::PIPE_FIX)) / clock * percent;
    
    uint64_t maxUsage = 0;
    for (const auto& entry : pipeUsageCount) {
        if (entry.second > maxUsage) {
            maxUsage = entry.second;
        }
    }
    report["Theoretical Minimum Cycles"] = maxUsage;
    // Memory Usage
    report["MEM_UB Peak Usage (%)"] = static_cast<float>(bufferMaxUsage.at(MemoryType::MEM_UB)) / maxUBSize * percent;
    report["MEM_UB Average Usage (%)"] = static_cast<float>(bufferTotalUsage.at(MemoryType::MEM_UB)) / clock / maxUBSize * percent;
    report["MEM_L1 Peak Usage (%)"] = static_cast<float>(bufferMaxUsage.at(MemoryType::MEM_L1)) / maxL1Size * percent;
    report["MEM_L1 Average Usage (%)"] = static_cast<float>(bufferTotalUsage.at(MemoryType::MEM_L1)) / clock / maxL1Size * percent;
    report["MEM_L0A Peak Usage (%)"] = static_cast<float>(bufferMaxUsage.at(MemoryType::MEM_L0A)) / maxL0ASize * percent;
    report["MEM_L0A Average Usage (%)"] = static_cast<float>(bufferTotalUsage.at(MemoryType::MEM_L0A)) / clock / maxL0ASize * percent;
    report["MEM_L0B Peak Usage (%)"] = static_cast<float>(bufferMaxUsage.at(MemoryType::MEM_L0B)) / maxL0BSize * percent;
    report["MEM_L0B Average Usage (%)"] = static_cast<float>(bufferTotalUsage.at(MemoryType::MEM_L0B)) / clock / maxL0BSize * percent;
    report["MEM_L0C Peak Usage (%)"] = static_cast<float>(bufferMaxUsage.at(MemoryType::MEM_L0C)) / maxL0CSize * percent;
    report["MEM_L0C Average Usage (%)"] = static_cast<float>(bufferTotalUsage.at(MemoryType::MEM_L0C)) / clock / maxL0CSize * percent;
    // Spill Info
    report["Spill Count"] = spillInfoVec.size();
    // Detailed spill information
    int spillIdx = 0;
    Json spill = Json::array();
    for (auto spillInfo : spillInfoVec) {
        Json spillDetails;
        spillDetails["Spill Event Idx"] = spillIdx++;
        spillDetails["Spill Buffer Type"] = MemoryTypeToString(spillInfo.spillType);
        spillDetails["Buffer Current Usage"] = spillInfo.bufferCurrUsage;
        spillDetails["Buffer Current Usage Rate (%)"] = static_cast<float>(spillInfo.bufferCurrUsage) / PassConfigManager::Instance().GetPlatformConfig().GetMemoryLimit(spillInfo.spillType) * percent;
        spillDetails["Buffer Occupied By Alloc Size"] = spillInfo.allocOccupiedSize;
        spillDetails["Spill Tensor Size"] = spillInfo.spillTensorSize;
        spillDetails["Trigger Tensor Size"] = spillInfo.triggerTensorSize;
        spillDetails["Spill Copyout Size"] = spillInfo.spillCopyoutSize;
        spill.emplace_back(spillDetails);
    }
    report["Spill Info"] = spill;
    return report;
}

} // namespace tile_fwk
} // namespace npu