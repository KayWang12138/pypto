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
 * \file schedule_ooo.cpp
 * \brief
 */

#include "schedule_ooo.h"

namespace npu::tile_fwk {

bool OoOSchedule::IsAicpuProgram(std::vector<Operation *> opList) {
    for (auto &op : opList) {
        if (op->GetCoreType() == CoreType::AICPU) {
            return true;
        }
    }
    return false;
}

Status OoOSchedule::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(GetName().c_str(), "Operation", "=============== START OoOSchedule ===============");
    int maxWorkeSpaceSize = 0;
    for (auto &program : function.rootFunc_->programs_) {
        auto opList = program.second->Operations().DuplicatedOpList();
        oriFunctions.emplace_back(program.second);
        // ooo不处理aicpu子图
        if (IsAicpuProgram(opList)) {
            continue;
        }
        OoOScheduler oooSchedule(*program.second);
        oooSchedule.oooCheck.doHealthCheck = passDfxconfigs_.healthCheck;
        APASS_LOG_INFO_F(GetName().c_str(), "Operation", "Subgraph[%d] OOOSchedule start.", program.first);
        if (oooSchedule.Schedule(opList) != SUCCESS) { 
            APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "Subgraph[%d] OoO Schedule failed.", program.first); 
            return FAILED;
        }
        APASS_LOG_INFO_F(GetName().c_str(), "Operation", "Subgraph[%d] OOOSchedule end.", program.first);
        program.second->ScheduleBy(oooSchedule.GetNewOperations());
        program.second->RecordOOOSeq();
        RescheduleUtils::UpdateTensorConsProd(program.second);
        maxWorkeSpaceSize = std::max(maxWorkeSpaceSize, (*program.second).GetStackWorkespaceSize());
        function.SetStackWorkespaceSize(maxWorkeSpaceSize);
        if (oooSchedule.oooCheck.doHealthCheck) {
            oooSchedule.oooCheck.workspaceOffset = oooSchedule.workspaceOffset;
            oooSchedule.oooCheck.clock = oooSchedule.clock;
            oooSchedule.oooCheck.jsonFileName = GetDumpFilePrefix(function, false, program.second, program.first);
            schedulerMap.insert({program.first, oooSchedule});
        }
    }
    APASS_LOG_INFO_F(GetName().c_str(), "Operation", "=============== END OoOSchedule =================");
    return SUCCESS;
}

void OoOSchedule::DoHealthCheckAfter(Function &function, const std::string &folderPath) {
    for (auto &scheduler : schedulerMap) {
        auto fileName = folderPath + '/' + scheduler.second.oooCheck.jsonFileName + "_Block_Graph_Health_Report.json";
        auto it = function.rootFunc_->programs_.find(scheduler.first);
        if (it != function.rootFunc_->programs_.end()) {
            auto subFunc = it->second;
            scheduler.second.oooCheck.DoHealthCheck(subFunc, fileName);
        }
    }
}

Status OoOSchedule::PreCheck(Function &function) {
    return checker.DoPreCheck(function);
}

Status OoOSchedule::PostCheck(Function &function) {
    checker.SetOriFunctions(oriFunctions);
    return checker.DoPostCheck(function);
}
} // namespace npu::tile_fwk