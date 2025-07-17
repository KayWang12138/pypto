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
 * \file DeviceMachine.h
 * \brief
 */

#pragma once

#include <map>
#include <vector>
#include <random>
#include <deque>
#include <mutex>
#include <unordered_set>

#include "simulation/base/Machine.h"
#include "simulation/machine/CoreMachine.h"
#include "simulation/common/ISA.h"
#include "simulation/config/ModelConfig.h"
#include "simulation/config/DeviceConfig.h"
#include "simulation/statistics/DeviceStats.h"
#include "simulation/statistics/TraceLogger.h"
#include "simulation/value/TileState.h"

namespace CostModel {
class DeviceMachine : public Machine {
public:
    bool cubeVecMix = false;
    std::map<MachineType, std::deque<uint64_t>> readyQueues;
    std::size_t readyQueuePid;
    std::size_t readyQueueTotalTid;
    std::map<MachineType, std::size_t> readyQueueTid;
    TaskMap taskMap;

    uint64_t globalSubtaskId;
    uint64_t currentEnd;
    std::deque<TaskMap> taskMapQueue;
    std::deque<uint64_t> functionQueue;
    std::map<uint64_t, uint64_t> executingTaskMap;

    DeviceConfig config;
    std::shared_ptr<DeviceStats> stats;
    std::shared_ptr<TileState> tileStateGolden;
    std::shared_ptr<TileState> tileState;

    uint64_t lastHeartModulo = 0;
    uint64_t currentHeartModulo = 0;

    bool taskBuilded = false;

    void RunAtBegin();
    void RunAtEnd();
    void RunPVModelDeviceTask();
    void SubmitDeviceTask();
    void BuildDeviceTask();
    TaskMap BuildATaskMap();
    void InitFunctions();
    void BuildLeafFunctionTasks();
    void BuildSubtasksFromTopo();
    void BuildSingleFuncTask();
    void PushReadyQueue(MachineType mType, uint64_t taskId);
    uint64_t PopReadyQueue(MachineType mType);
    bool EraseReadyQueue(MachineType mType, uint64_t taskId);

    void Step() override;
    void Xfer() override;
    void Build() override;
    void Reset() override;
    std::shared_ptr<SimSys> GetSim() override;
    void Report() override;
    void InitQueueDelay() override;
    void StepQueue() override;
    bool IsTerminate() override;

    // Static Sim
    struct CoreMachineQueue {
        CostModel::MachineType coreType;
        int blockIdx;
        // queue <int, <begin_cycle, end_cycle>>, <int, <begin_cycle, end_cycle>>
        std::deque<std::pair<uint64_t, std::pair<uint64_t, uint64_t>>> taskIds;
    };

    struct AICPUMachineGroup {
        std::vector<CoreMachineQueue> coreMachines;
    };

    std::vector<AICPUMachineGroup> ParseSimulateJson(const std::string& filename);
    std::vector<AICPUMachineGroup> staticSimData;

private:
    void CalculateTileGolden();
    void CalculateFunctionArgTile(FunctionPtr func, std::shared_ptr<TileState> state);
    void CalculateFunctionTileGolden(FunctionPtr func, std::shared_ptr<TileState> local,
                                     std::shared_ptr<TileState> global, int esgId);
    void PrintFunctionOutputTile(FunctionPtr func, std::shared_ptr<TileState> state);
    void PrintTopo();
    void PrintFunctionTopo(FunctionPtr func);
};

}  // namespace CostModel