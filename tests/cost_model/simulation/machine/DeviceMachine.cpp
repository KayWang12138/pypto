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
 * \file DeviceMachine.cpp
 * \brief
 */

#include "simulation/machine/DeviceMachine.h"

#include <sstream>
#include <utility>
#include <fstream>
#include <mutex>

#include "nlohmann/json.hpp"
#include "simulation/base/ModelLogger.h"
#include "simulation/base/ModelTop.h"
#include "simulation/common/ISA.h"
#include "simulation/value/TileCalculator.h"
#include "interface/function/function.h"

using Json = nlohmann::json;
using namespace std::string_literals;
using namespace std::chrono_literals;

#define EXPVAL_LOG MLOG_DEBUG

#define TOPO_LOG MLOG_DEBUG

namespace CostModel {

void DeviceMachine::Step()
{
    // device machine is useless in calendar mode
    if (GetSim()->config.calendarMode != static_cast<uint64_t>(CalendarMode::DEVICE)) {
        return;
    }
    RunAtBegin();
    BuildDeviceTask();
    SubmitDeviceTask();
    RunAtEnd();
}

void DeviceMachine::RunAtBegin()
{
    if (!taskBuilded) {
        InitFunctions();
        PrintTopo();
        CalculateTileGolden();
        taskBuilded = true;
    }

    for (auto &submachine : subMachines) {
        if (submachine->machineType == MachineType::CPU) {
            auto core = std::dynamic_pointer_cast<AICPUMachine>(submachine);
            if (!core->localReadyQueues.Empty()) {
                uint64_t taskId;
                core->localReadyQueues.Dequeue(taskId);
                MLOG_INFO("Dequeued task ID:", taskId);
                PushReadyQueue(taskMap.at(taskId)->machineType, taskId);
            }
        }
    }

    // init
    if (currentEnd == 0 && !functionQueue.empty()) {
        LoggerRecordTaskStart("Build Device Task");
        currentEnd = GetSim()->GetCycles() + config.stitchLatency;
    }
}

void DeviceMachine::RunAtEnd()
{
    // need to build device task
    if (GetSim()->GetCycles() >= currentEnd && !functionQueue.empty()) {
        LoggerRecordTaskStart("Build Device Task");
        currentEnd = GetSim()->GetCycles() + config.stitchLatency;
    }
    nextCycles = GetSim()->GetCycles() < currentEnd ? currentEnd : INT_MAX;
    GetSim()->UpdateNextCycles(nextCycles);
    needTerminate = IsTerminate();
}

void DeviceMachine::RunPVModelDeviceTask()
{
    for (const auto& [taskId, task] : taskMap) {
        auto function = GetSim()->functionCache.GetFunction(task->functionHash);
        GetSim()->pv->Run(taskId, function->pSgId);
    }
    taskMap.clear();
    MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][Device ", machineId, "] run pvmodel execute tasks", taskMap.size());
}

void DeviceMachine::SubmitDeviceTask()
{
    if (!taskMap.empty()) {
        return;
    }
    if (taskMapQueue.empty()) {
        return;
    }
    taskMap = std::move(taskMapQueue.front()), taskMapQueue.pop_front();
    if (GetSim()->pvLevel != PVModelLevel::PV_NON) {
        RunPVModelDeviceTask();
        return;
    }
    for (const auto& [taskId, task] : taskMap) {
        if (task->remainingPredecessors == 0) {
            PushReadyQueue(task->machineType, taskId);
        }
    }
    MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][Device ", machineId, "] submit a new device task to AICPUs, size = ",
              taskMap.size());
}

// Device Init
void DeviceMachine::Build()
{
    config.OverrideDefaultConfig(&sim->cfgs);
    globalSubtaskId = 0;
    currentEnd = 0;
    readyQueuePid = GetSim()->RegisterQueuePid("DeviceReadyQ");
    GetSim()->GetLogger()->SetProcessName("DeviceReadyQ", readyQueuePid, readyQueuePid);
    readyQueueTotalTid = queueSeq + coreTid;
    GetSim()->GetLogger()->SetThreadName("Total_ReadyQ", readyQueuePid, readyQueueTotalTid);
    queueSeq++;
    for (const auto &machineTypeStr : config.submachineTypes) {
        MachineType mType = ToMachineType(machineTypeStr);
        if (mType != MachineType::UNKNOWN) {
            readyQueues.try_emplace(mType);
            readyQueueTid[mType] = queueSeq + coreTid;
            GetSim()->GetLogger()->SetThreadName((MachineName(mType) + "_ReadyQ"), readyQueuePid, readyQueueTid[mType]);
            queueSeq++;
        }
        if (mType == MachineType::MIXAICORE) {
            cubeVecMix = true;
        }
    }

    stats = std::make_shared<DeviceStats>(GetSim()->GetReporter());
    tileStateGolden = std::make_shared<TileState>();
    tileState = std::make_shared<TileState>();
}

std::shared_ptr<SimSys> DeviceMachine::GetSim()
{
    return sim;
}

void DeviceMachine::Xfer()
{
    StepQueue();
    lastCycles = GetSim()->GetCycles();
    currentHeartModulo = GetSim()->GetCycles() % (GetSim()->config.heartInterval);
    if (currentHeartModulo < lastHeartModulo) {
        MLOG_WARN("@CostModel Heart Cycle:", GetSim()->GetCycles(), ", submit tasks: ", stats->totalSubmitNum);
    }
    lastHeartModulo = currentHeartModulo;
}

void DeviceMachine::Report()
{
    int machineSeq = GetMachineSeq(machineId);
    std::string name = std::to_string(machineSeq);
    stats->Report(name);
}

bool DeviceMachine::IsTerminate()
{
    if (sim->config.replayAllMode == 1) {
        if (taskBuilded) {
            return true;
        }
    }
    if (sim->config.calendarMode != static_cast<uint64_t>(CalendarMode::DEVICE)) {
        return true;
    }
    bool readyQueueIsEmpty = std::all_of(
        readyQueues.begin(), readyQueues.end(),
        [](const auto& pair) { return pair.second.empty(); }
    );
    return readyQueueIsEmpty && taskMap.empty() && taskMapQueue.empty() && functionQueue.empty();
}

// build functionQueue
void DeviceMachine::InitFunctions()
{
    functionQueue.clear();

    if (GetSim()->dynamicWorkflow) {
        BuildLeafFunctionTasks();
        return;
    }

    auto functionCache = GetSim()->functionCache.cache;
    auto startFuncHash = GetSim()->startFuncHash;

    if (GetSim()->testSingleFunc) {
        BuildSingleFuncTask();
        return;
    }

    if (functionCache[startFuncHash]->useInputTopo) {
        BuildSubtasksFromTopo();
        return;
    }
    // root is a static function
    functionQueue.push_back(startFuncHash);
}

void DeviceMachine::BuildLeafFunctionTasks() {
    sim->enableExpectValue = false;
    TaskMap taskM;
    auto functionCache = GetSim()->functionCache.cache;
    for (auto &[hash, func] : functionCache) {
        if (func->funcName.find("leaf") == std::string::npos) continue;
        auto subtask = std::make_shared<Task>();
        subtask->status = false;
        subtask->functionHash = hash;
        subtask->functionName = func->funcName;
        subtask->taskId = taskM.size();
        subtask->machineType = func->machineType;
        subtask->remainingPredecessors = 0;
        GetSim()->taskToHash[subtask->taskId] = subtask->functionHash;
        taskM.insert({subtask->taskId, subtask});
    }
    taskMapQueue.push_back(taskM);
    GetSim()->ProcessTaskMap(taskM);
    MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][DeviceMachine][BuildLeafFunctionTasks] ", "Machine ", machineId,
    " build subtasks done");
}

void DeviceMachine::BuildSubtasksFromTopo()
{
    TaskMap taskM;
    auto functionCache = GetSim()->functionCache.cache;
    auto startFuncHash = GetSim()->startFuncHash;
    auto startFunc = functionCache[startFuncHash];
    for (const auto &topoEntry : startFunc->inputTopo) {
        auto subtask = std::make_shared<Task>();
        subtask->status = false;
        subtask->functionHash = topoEntry.calleeHash;
        subtask->functionName = functionCache[subtask->functionHash]->funcName;
        subtask->taskId = topoEntry.eSgId;
        subtask->machineType = functionCache[subtask->functionHash]->machineType;
        subtask->remainingPredecessors = -topoEntry.readyState;
        subtask->fixedLatency = topoEntry.fixedLatency;
        subtask->fixedLatencyVal = topoEntry.fixedLatencyVal;
        if (uint64_t(startFunc->tileOps.size()) > topoEntry.eSgId) {
            subtask->semanticLabels = startFunc->tileOps[topoEntry.eSgId]->semanticLabels;
        }
        GetSim()->taskToHash[subtask->taskId] = subtask->functionHash;
        for (auto &out : topoEntry.outGraph) {
            subtask->successors.push_back(out);
        }
        taskM.insert({subtask->taskId, subtask});
    }
    for (const auto &it : taskM) {
        for (auto &successor : it.second->successors) {
            taskM.at(successor)->predecessors.push_back(it.first);
        }
    }

    for (const auto &it : taskM) {
        MLOG_INFO("Task ID: ", it.second->taskId);
        MLOG_INFO("  Remaing task num: ", it.second->remainingPredecessors);
        for (auto &pre : it.second->predecessors) {
            MLOG_INFO("  Predecessor:", pre);
        }
        for (auto &suc : it.second->successors) {
            MLOG_INFO("  Predecessor:", suc);
        }
    }
    taskMapQueue.push_back(taskM);
    GetSim()->ProcessTaskMap(taskM);

    if (GetSim()->config.replayDispatchMode == 1 || GetSim()->config.replayAllMode == 1) {
        staticSimData = DeviceMachine::ParseSimulateJson(GetSim()->config.replayFile);
        std::unordered_set<uint64_t> taskMIds;
        for (const auto& pair : taskM) {
            taskMIds.insert(pair.first);
        }

        std::unordered_set<uint64_t> simulatedTaskIds;
        for (const auto& aicpuMachine : staticSimData) {
            for (const auto& coreMachine : aicpuMachine.coreMachines) {
                for (const auto& taskPair : coreMachine.taskIds) {
                    simulatedTaskIds.insert(taskPair.first);
                }
            }
        }

        for (const auto& [taskId, task] : taskM) {
            if (simulatedTaskIds.find(taskId) == simulatedTaskIds.end()) {
                MLOG_INFO("[Check][DeviceMachine] Task ID ", taskId, " (Function: ", task->functionName,
                          ") exists in taskM but not in simulation data");
            }
        }
        
        for (const auto& taskId : simulatedTaskIds) {
            if (taskMIds.find(taskId) == taskMIds.end()) {
                MLOG_INFO("[Check][DeviceMachine] Task ID ", taskId, " exists in simulation data but not in taskM");
            }
        }

        MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][DeviceMachine][build_subtasks_from_replay_file] ",
                  "Machine ", machineId, " build subtasks done");
        return;
    }
    
    MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][DeviceMachine][build_subtasks_from_topo] ", "Machine ",
              machineId, " build subtasks done");
}

void DeviceMachine::BuildSingleFuncTask()
{
    auto functionCache = GetSim()->functionCache.cache;
    TaskMap taskM;
    auto subtask = std::make_shared<Task>();
    subtask->status = false;
    subtask->functionHash = GetSim()->singleFuncHash;
    subtask->functionName = functionCache[subtask->functionHash]->funcName;
    subtask->taskId = 1;
    subtask->machineType = functionCache[subtask->functionHash]->machineType;
    subtask->remainingPredecessors = 0;
    GetSim()->taskToHash[subtask->taskId] = subtask->functionHash;
    taskM.insert({subtask->taskId, subtask});
    GetSim()->GetCalendarGenerator()->InitTaskTopoInfo(taskM);
    taskMapQueue.push_back(taskM);
}

void DeviceMachine::BuildDeviceTask()
{
    if (GetSim()->GetCycles() < currentEnd) {
        return;
    }
    if (functionQueue.empty()) {
        return;
    }
    // build a task map from functionQueue
    auto taskM = BuildATaskMap();
    GetSim()->GetCalendarGenerator()->InitTaskTopoInfo(taskM);
    taskMapQueue.push_back(taskM);
    LoggerRecordTaskEnd();
    MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][Device ", machineId, "] build a new device task, size = ",
              taskM.size());
}

// build one device task
TaskMap DeviceMachine::BuildATaskMap()
{
    std::unordered_map<int, std::vector<int>> outcastToSubtask;
    std::unordered_map<int, std::vector<int>> incastToSubtask;
    std::unordered_map<int, int> tensorMap; // RESHAPE's outcast to incast
    TaskMap taskM;
    auto functionCache = GetSim()->functionCache.cache;
    auto functionHash = functionQueue.front();
    functionQueue.pop_front();
    auto rootFunction = functionCache[functionHash];

    // reshape
    for (const auto &op : rootFunction->tileOps) {
        if (op->opcode == "RESHAPE") {
            tensorMap[op->oOperand[0]->magic] = op->iOperand[0]->magic;
        }
    }
    // create subtasks into taskM
    for (const auto &op : rootFunction->tileOps) {
        // Every subtask is a call operation
        if (!op->IsCall()) {
            continue;
        }
        uint64_t subtaskId = globalSubtaskId++;
        auto subtask = std::make_shared<Task>();
        subtask->status = false;
        subtask->functionHash = op->calleeHash;
        subtask->functionName = functionCache[subtask->functionHash]->funcName;
        subtask->taskId = subtaskId;
        subtask->machineType = functionCache[subtask->functionHash]->machineType;

        for (auto incast : op->iOperand) {
            int magic = incast->magic;
            if (tensorMap.find(magic) != tensorMap.end()) {
                magic = tensorMap[magic];
            }
            subtask->incasts.push_back(magic);
            incastToSubtask[magic].push_back(subtaskId);
        }
        for (auto outcast : op->oOperand) {
            int magic = outcast->magic;
            subtask->outcasts.push_back(magic);
            outcastToSubtask[magic].push_back(subtaskId);
        }
        taskM[subtaskId] = subtask;
    }
    // connect subtasks
    for (const auto& [iop, itasks] : incastToSubtask) {
        auto oit = outcastToSubtask.find(iop);
        if (oit == outcastToSubtask.end()) {
            continue;
        }
        auto otasks = oit->second;
        for (const auto& idB : itasks) {
            for (const auto& idA : otasks) {
                auto it = std::find(taskM[idA]->successors.begin(), taskM[idA]->successors.end(), idB);
                if (it == taskM[idA]->successors.end()) {
                    taskM[idA]->successors.push_back(idB);
                }
                it = std::find(taskM[idB]->predecessors.begin(), taskM[idB]->predecessors.end(), idA);
                if (it == taskM[idB]->predecessors.end()) {
                    taskM[idB]->predecessors.push_back(idA);
                }
            }
        }
    }
    for (auto it : taskM) {
        it.second->remainingPredecessors = it.second->predecessors.size();
    }
    // build complete
    return taskM;
}

void DeviceMachine::PrintFunctionTopo(FunctionPtr func) {
    auto cache = GetSim()->functionCache.cache;
    TOPO_LOG("Function -> " + func->funcName);
    TOPO_LOG("incast:");
    for (const auto &incast : func->incastMagic) {
        TOPO_LOG(func->tileMap[incast]->Dump());
    }

    TOPO_LOG("outcast:");
    for (const auto &outcast : func->outcastMagic) {
        TOPO_LOG(func->tileMap[outcast]->Dump());
    }

    for (const auto &op: func->tileOps) {
        TOPO_LOG(op->opcode);
        TOPO_LOG("incast:");
        for (auto &incast : op->iOperand) {
            TOPO_LOG(incast->Dump());
        }

        TOPO_LOG("outcast:");
        for (auto &outcast : op->oOperand) {
            TOPO_LOG(outcast->Dump());
        }

        if (op->IsCall()) {
            auto invoke = op->operation->GetSubFuncInvokeInfo();
            invoke.PrintInvokeInfo("");
            PrintFunctionTopo(cache[op->calleeHash]);
        }
    }
}

void DeviceMachine::PrintTopo() {
    auto cache = GetSim()->functionCache.cache;
    auto startFuncHash = GetSim()->startFuncHash;
    if (startFuncHash == 0) {
        return;
    }
    auto func = cache[startFuncHash];

    if (func->parentFunction) {
        auto topo = func->parentFunction->topoInfo_;
        for (auto &e : topo.topology_) {
            TOPO_LOG("[TOPO] " + std::to_string(e.esgId) + ", " + std::to_string(e.readyState));
            for (auto &o : e.outGraph) {
                TOPO_LOG("[TOPO] out -> " + std::to_string(o));
            }
        }
    }

    PrintFunctionTopo(func);
}

void DeviceMachine::CalculateFunctionArgTile(FunctionPtr func, std::shared_ptr<TileState> state)
{
    for (auto &incast : func->incastMagic) {
        TileCalculator::Self().CalculateInput(func->tileMap[incast], state);
    }
}

void DeviceMachine::PrintFunctionOutputTile(FunctionPtr func, std::shared_ptr<TileState> state)
{
    for (auto &outcast : func->outcastMagic) {
        auto tile = func->tileMap[outcast];
        auto k = TileState::TileKey(tile->rawMagic, tile->bufType,
                            tile->shape, tile->offset);
        auto value = state->Load(k);
        EXPVAL_LOG("[EXPVAL] outcast -> " + std::to_string(tile->magic) + "-" + k.Dump() + " : " + std::to_string(value));
    }
}

void DeviceMachine::CalculateFunctionTileGolden(FunctionPtr func, std::shared_ptr<TileState> local,
                                                std::shared_ptr<TileState> global, int esgId) {
    auto cache = GetSim()->functionCache.cache;
    for (const auto &op: func->tileOps) {
        if (op->IsCall()) {
            auto callee = cache[op->calleeHash];
            EXPVAL_LOG(std::string("[EXPVAL] function") + " -> [" + callee->funcName + ", " + std::to_string(op->magic) + "] incast");
            for (auto &incast : op->iOperand) {
                auto k = TileState::TileKey(incast->rawMagic, incast->bufType, incast->shape, incast->offset);
                auto value = global->Load(k);
                EXPVAL_LOG("[EXPVAL] incast -> " + std::to_string(incast->magic) + "-" + k.Dump() + " : " + std::to_string(value));
            }
            EXPVAL_LOG("{");

            std::shared_ptr<TileState> l = std::make_shared<TileState>();
            CalculateFunctionTileGolden(callee, l, global, esgId);
            esgId++;

            EXPVAL_LOG("}");

            EXPVAL_LOG(std::string("[EXPVAL] function") + " -> [" + callee->funcName + ", " + std::to_string(op->magic) + "] outcast");
            for (auto &outcast : op->oOperand) {
                auto k = TileState::TileKey(outcast->rawMagic, outcast->bufType, outcast->shape, outcast->offset);
                auto value = global->Load(k);
                EXPVAL_LOG("[EXPVAL] outcast -> " + std::to_string(outcast->magic) + "-" + k.Dump() + " : " + std::to_string(value));
            }

            EXPVAL_LOG("[EXPVAL] ###########################");
        }
        else {
            TileCalculator::Self().Calculate(op, func->invoke[esgId], local, global);
        }
    }
}

void DeviceMachine::CalculateTileGolden() {
    if (!sim->enableExpectValue) {
        return;
    }

    auto cache = sim->functionCache.cache;
    auto startFuncHash = GetSim()->startFuncHash;

    EXPVAL_LOG("[EXPVAL] Function Root Golden");
    TileCalculator::Self().Reset();
    CalculateFunctionArgTile(cache[startFuncHash], tileStateGolden);
    CalculateFunctionTileGolden(cache[startFuncHash], nullptr, tileStateGolden, 0);
    EXPVAL_LOG("[EXPVAL] Function Root Golden Output:");
    PrintFunctionOutputTile(cache[startFuncHash], tileStateGolden);
    EXPVAL_LOG("[EXPVAL] ###########################");

    EXPVAL_LOG("[EXPVAL] Function Root");
    TileCalculator::Self().Reset();
    CalculateFunctionArgTile(cache[startFuncHash], tileState);
}

std::vector<DeviceMachine::AICPUMachineGroup> DeviceMachine::ParseSimulateJson(const std::string& filename)
{
    std::vector<AICPUMachineGroup> aicpuMachines(GetSim()->config.aicpuMachineNumber);
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        Json j;
        file >> j;
        for (const auto& item : j) {
            uint64_t blockIdx = item["blockIdx"];
            uint64_t aicpuIdx;
            MachineType coreType = ToMachineType(std::string(item["coreType"]).substr(0, 3));
            if (coreType == MachineType::AIC) {
                aicpuIdx = blockIdx / GetSim()->config.cubeMachineNumberPerAICPU;
            } else if (coreType == MachineType::AIV) {
                uint64_t relativeAivIdx = blockIdx -
                    (GetSim()->config.aicpuMachineNumber * GetSim()->config.cubeMachineNumberPerAICPU);
                aicpuIdx = relativeAivIdx / GetSim()->config.vecMachineNumberPerAICPU;
            } else {
                continue;
            }

            if (aicpuIdx >= GetSim()->config.aicpuMachineNumber) {
                MLOG_INFO("BlockIdx ", blockIdx, " exceeds expected AICPU count range");
                continue;
            }

            CoreMachineQueue cmq;
            cmq.coreType = coreType;
            cmq.blockIdx = blockIdx;

            const auto& tasks = item["tasks"];
            for (const auto& task : tasks) {
                uint64_t taskId = task["taskId"];
                uint64_t beginCycle = task["execStart"];
                uint64_t endCycle = task["execEnd"];
                cmq.taskIds.push_back({taskId, {beginCycle, endCycle}});
            }
            aicpuMachines[aicpuIdx].coreMachines.push_back(cmq);
        }

        // Print parsing results
        for (size_t i = 0; i < aicpuMachines.size(); i++) {
            MLOG_INFO("AICPU Machine ", i, " Configuration:");
            for (const auto& cm : aicpuMachines[i].coreMachines) {
                std::string taskList;
                std::deque<std::pair<uint64_t, std::pair<uint64_t, uint64_t>>> tempQueue = cm.taskIds;
                while (!tempQueue.empty()) {
                    taskList += "Task ID: " + std::to_string(tempQueue.front().first) +
                                " [Begin: " + std::to_string(tempQueue.front().second.first) +
                                ", End: " + std::to_string(tempQueue.front().second.second) + "] ";
                    tempQueue.pop_front();
                }
                MLOG_INFO("    Core BlockIdx: ", cm.blockIdx, " Task IDs: ", taskList);
            }
        }
    } catch (const std::exception& e) {
        MLOG_INFO("JSON parsing error: ", e.what());
    }
    return aicpuMachines;
}

void DeviceMachine::PushReadyQueue(MachineType mType, uint64_t taskId)
{
    if (cubeVecMix) {
        mType = MachineType::MIXAICORE;
    }
    readyQueues[mType].push_back(taskId);
    GetSim()->GetLogger()->AddCounterEvent(readyQueuePid, readyQueueTotalTid, CounterType::QUEUE_PUSH);
    GetSim()->GetLogger()->AddCounterEvent(readyQueuePid, readyQueueTid[mType], CounterType::QUEUE_PUSH);
}

uint64_t DeviceMachine::PopReadyQueue(MachineType mType)
{
    if (cubeVecMix) {
        mType = MachineType::MIXAICORE;
    }
    uint64_t taskId = readyQueues[mType].front();
    readyQueues[mType].pop_front();
    GetSim()->GetLogger()->AddCounterEvent(readyQueuePid, readyQueueTotalTid, CounterType::QUEUE_POP);
    GetSim()->GetLogger()->AddCounterEvent(readyQueuePid, readyQueueTid[mType], CounterType::QUEUE_POP);
    return taskId;
}

bool DeviceMachine::EraseReadyQueue(MachineType mType, uint64_t taskId)
{
    if (cubeVecMix) {
        mType = MachineType::MIXAICORE;
    }
    auto it = std::find(readyQueues[mType].begin(), readyQueues[mType].end(), taskId);
    if (it != readyQueues[mType].end()) {
        readyQueues[mType].erase(it);
        GetSim()->GetLogger()->AddCounterEvent(readyQueuePid, readyQueueTotalTid, CounterType::QUEUE_POP);
        GetSim()->GetLogger()->AddCounterEvent(readyQueuePid, readyQueueTid[mType], CounterType::QUEUE_POP);
        return true;
    } else {
        return false;
    }
}

void DeviceMachine::Reset() {}
void DeviceMachine::InitQueueDelay() {}
void DeviceMachine::StepQueue() {}
}
