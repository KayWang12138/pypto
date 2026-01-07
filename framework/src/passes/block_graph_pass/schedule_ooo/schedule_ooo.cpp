/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
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
#include "passes/pass_log/pass_log.h"
#include "core_assign.h"

#ifndef MODULE_NAME
#define MODULE_NAME "OoOSchedule"
#endif

namespace npu::tile_fwk {

bool OoOSchedule::IsAicpuProgram(std::vector<Operation *> opList) {
    for (auto &op : opList) {
        if (op->GetCoreType() == CoreType::AICPU) {
            return true;
        }
    }
    return false;
}

inline bool IsMixGraph(const std::vector<Operation*> &opList) {
    bool hasAIC = false;
    bool hasAIV = false;
    for (auto opPtr : opList) {
        if (OpcodeManager::Inst().GetCoreType(opPtr->GetOpcode()) == OpCoreType::AIC) {
            hasAIC = true;
        } else if (OpcodeManager::Inst().GetCoreType(opPtr->GetOpcode()) == OpCoreType::AIV) {
            hasAIV = true;
        }
        if (hasAIC && hasAIV) {
            return true;
        }
    }
    return false;
}

void OoOSchedule::SortTaskList(std::vector<Operation*> &opList, std::vector<Operation*> &taskList) {
    std::vector<Operation*> newTaskList;
    for (auto op : opList) {
        if (std::find(taskList.begin(), taskList.end(), op) != taskList.end()) {
            newTaskList.push_back(op);
        }
    }
    taskList = newTaskList;
}

Status OoOSchedule::SortAndLatencyEstimate(std::vector<Operation*> &opList, std::vector<Operation*> &taskOpList,
    int &latency) {
    APASS_LOG_INFO_F(Elements::Operation, "=======>start SortAndLatencyEstimate");
    SortTaskList(opList, taskOpList);
    LatencyEstimator latencyEstimator(taskOpList);
    if (latencyEstimator.LatencyEstimatorMainLoop() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "SortAndLatencyEstimate LatencyEstimatorMainLoop failed.");
        return FAILED;
    }
    latency = latencyEstimator.clock;
    APASS_LOG_INFO_F(Elements::Operation, "=======>end SortAndLatencyEstimate");
    return SUCCESS;
}

Status OoOSchedule::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Operation, "=============== START 2CoreSplit ===============");
    std::unordered_map<TargetCoreType, std::string>  targetToString{{TargetCoreType::AIC, "AIC"}, {TargetCoreType::AIV0, "AIV0"}, {TargetCoreType::AIV1, "AIV1"}, {TargetCoreType::UNKNOWN, "UNKNOWN"}};
    for (auto &program : function.rootFunc_->programs_) {
        auto opList = program.second->Operations(false).DuplicatedOpList();
        oriFunctions.emplace_back(program.second);
        OptimizeSort optimizeSort(opList, *program.second);
        if (optimizeSort.SortOps() != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Global sortOps failed.");
            return FAILED;
        }
        // 全局排序的序列
        opList = optimizeSort.operations;
        if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510 || !IsMixGraph(opList)) {
            // 直接对oplist进行GenSpill和mainLoop
            OoOScheduler ooOSchedule(*program.second);
            if (ooOSchedule.Schedule(opList) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "Non-mixGraph schedule failed.");
                return FAILED;
            }
            continue;
        }
        TaskSpliter spliter;
        spliter.SplitGraph(opList);
        for (auto &taskNode : spliter.GetTaskGraph().tasks) {
            // 对taskNode.opList_进行排序，并返回预估的latency
            if (SortAndLatencyEstimate(opList, taskNode.opList_, taskNode.latency) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "SortAndLatencyEstimate failed, taskNode[%d].", taskNode.idx);
                return FAILED;
            }
        }
        CoreScheduler coreScheduler;
        coreScheduler.Schedule(spliter.GetTaskGraph(), 10); // BruteForce threshold is 10
        for (auto &taskNode : spliter.GetTaskGraph().tasks) {
            APASS_LOG_INFO_F(Elements::Operation,  "eval task %d on %s: %d - %d.", taskNode.idx, targetToString[taskNode.targetCoreType].c_str(), taskNode.startTime, taskNode.endTime);
        }
        spliter.MergeTaskByTargetCoreType();
        for (auto &taskNode : spliter.GetTaskGraph().tasks) {
            SortTaskList(opList, taskNode.opList_);
            OoOScheduler ooOSchedule(*program.second);
            if (ooOSchedule.Schedule(taskNode.opList_) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "TaskNode[%d] schedule failed.", taskNode.idx);
                return FAILED;
            }
        }
        spliter.MarkInternalSubgraphID();
        program.second->ScheduleBy(spliter.GetMergedOperations());
    }
    APASS_LOG_INFO_F(Elements::Operation, "=============== END 2CoreSplit ===============");
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