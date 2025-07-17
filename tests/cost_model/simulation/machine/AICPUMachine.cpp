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
 * \file AICPUMachine.cpp
 * \brief
 */

#include "simulation/machine/AICPUMachine.h"

#include <queue>
#include <mutex>

#include "simulation/base/ModelLogger.h"
#include "simulation/base/ModelTop.h"
#include "simulation/statistics/DeviceStats.h"

namespace CostModel {
void AICPUMachine::Step()
{
    if (GetSim()->config.replayAllMode == 1) {
        if (isTerminate) {
            MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][AICPUMachine: ", machineId, " is shutdown");
            return;
        }
        MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][AICPUMachine: ", machineId, " is executing");
        AICPUMachine::ReplayAll();
        return;
    }
    if (top->taskMap.empty()) {
        return;
    }
    RunAtBegin();
    PollingMachine();
    DispatchPacket();
    RunAtEnd();
}

void AICPUMachine::RunAtBegin()
{
    nextCycles = GetSim()->GetCycles();

    for (uint64_t threadId = 0; threadId < threadsNum; threadId++) {
        if (!threadState->batchProcessing[threadId]) {
            threadState->batchProcessing[threadId] = true;
            threadState->completionDone[threadId] = false;
            threadState->schedulerDone[threadId] = false;
        }
    }
}

void AICPUMachine::RunAtEnd()
{
    if (!top->taskMap.empty()) {
        CheckDeadlock();
    }

    for (uint64_t threadId = 0; threadId < threadsNum; threadId++) {
        if (threadState->batchProcessing[threadId] &&
            GetSim()->GetCycles() >= threadState->currentSchedulerEnd[threadId]) {
            threadState->batchProcessing[threadId] = false;
        }
    }
    needTerminate = IsTerminate();
}

std::shared_ptr<SimSys> AICPUMachine::GetSim()
{
    return sim;
}

void AICPUMachine::Build()
{
    config.OverrideDefaultConfig(&sim->cfgs);
    threadsNum = config.threadsNum;
    globalDelay = 0;
    threadState = std::make_unique<AICPUThreadState>(threadsNum);
    stats = std::make_shared<AICPUStats>(GetSim()->GetReporter());
    LoggerRecordTaskStart("AICPU Init");
    GetSim()->AddCycles();
    LoggerRecordTaskEnd();
    GetSim()->AddCycles();
    InitQueueDelay();
    for (uint64_t threadId = 0; threadId < threadsNum; threadId++) {
        std::string threadName = "AICPU_Thread_" + std::to_string(threadId);
        GetSim()->GetLogger()->SetThreadName(threadName, machineId, threadId + reversedTidNum);
    }
}

AICPUMachine::AICPUMachine()
{
    machineType = MachineType::CPU;
}

AICPUMachine::AICPUMachine(CostModel::MachineType type) : AICPUMachine()
{
    machineType = type;
}

void AICPUMachine::Reset() {}

void AICPUMachine::Xfer()
{
    StepQueue();
    lastCycles = GetSim()->GetCycles();
    nextCycles = INT_MAX;
    nextCycles = std::min(nextCycles, GetQueueNextCycles());
    MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][AICPUMachine:", machineId, "][Xfer] ", nextCycles);
    GetSim()->UpdateNextCycles(nextCycles);
}

void AICPUMachine::Report()
{
    int machineSeq = GetMachineSeq(machineId);
    std::string name = std::to_string(machineSeq);
    stats->Report(name);
}

void AICPUMachine::InitQueueDelay()
{
    completionQueue.SetWriteDelay(0);
    completionQueue.SetReadDelay(0);
    outcastReferenceQueue.SetWriteDelay(0);
    outcastReferenceQueue.SetReadDelay(0);
    incastReferenceQueue.SetWriteDelay(0);
    incastReferenceQueue.SetReadDelay(0);
    releaseQueue.SetWriteDelay(0);
    releaseQueue.SetReadDelay(0);
    cacheRespQueue.SetWriteDelay(0);
    cacheRespQueue.SetReadDelay(0);
}

void AICPUMachine::StepQueue()
{
    uint64_t intervalCycles = GetSim()->GetCycles() - lastCycles;
    completionQueue.UpdateIntervalCycles(intervalCycles);
    outcastReferenceQueue.UpdateIntervalCycles(intervalCycles);
    incastReferenceQueue.UpdateIntervalCycles(intervalCycles);
    releaseQueue.UpdateIntervalCycles(intervalCycles);
    cacheRespQueue.UpdateIntervalCycles(intervalCycles);
    localReadyQueues.UpdateIntervalCycles(intervalCycles);
    completionQueue.Step();
    outcastReferenceQueue.Step();
    incastReferenceQueue.Step();
    releaseQueue.Step();
    cacheRespQueue.Step();
    localReadyQueues.Step();
}

uint64_t AICPUMachine::GetQueueNextCycles()
{
    uint64_t res = INT_MAX;
    uint64_t gCycles = GetSim()->GetCycles();
    res = std::min(res, gCycles + completionQueue.GetMinWaitCycles());
    res = std::min(res, gCycles + outcastReferenceQueue.GetMinWaitCycles());
    res = std::min(res, gCycles + incastReferenceQueue.GetMinWaitCycles());
    res = std::min(res, gCycles + releaseQueue.GetMinWaitCycles());
    res = std::min(res, gCycles + cacheRespQueue.GetMinWaitCycles());
    return res;
}

bool AICPUMachine::IsTerminate()
{
    if (GetSim()->config.replayAllMode == 1) {
        return isTerminate;
    }

    for (auto &readyQ : top->readyQueues) {
        if (!readyQ.second.empty()) {
            return false;
        }
    }
    return (top->taskMap.empty() && completionQueue.IsTerminate() && outcastReferenceQueue.IsTerminate() &&
            incastReferenceQueue.IsTerminate() && releaseQueue.IsTerminate() && cacheRespQueue.IsTerminate());
}

void AICPUMachine::UpdatePollingStates(std::vector<bool> &threadGroupActive)
{
    for (uint64_t threadId = 0; threadId < threadsNum; threadId++) {
        if (GetSim()->GetCycles() < threadState->currentCompletionEnd[threadId]) {
            MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][AICPUMachine:", machineId, "][updatepolling] ",
                      pollingTimeAxe);
            GetSim()->UpdateNextCycles(std::min<uint64_t>(INT_MAX, pollingTimeAxe));
        } else {
            pollingTimeAxe = INT_MAX;
            MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][AICPUMachine:", machineId, "][updatepolling] ",
                      pollingTimeAxe);
            GetSim()->UpdateNextCycles(std::min<uint64_t>(INT_MAX, pollingTimeAxe));
        }
        // 检查submission和scheduler状态
        bool threadBusy = (GetSim()->GetCycles() < threadState->currentSchedulerEnd[threadId]) ||
                           (GetSim()->GetCycles() <= threadState->currentCompletionEnd[threadId]);
        bool canProcess = !threadBusy && !top->taskMap.empty();
        if (canProcess && !threadState->completionDone[threadId]) {
            threadGroupActive[threadId] = true;
            threadState->completionDone[threadId] = true;
        }
    }
}

void AICPUMachine::UpdateDispatchStates(std::vector<bool> &threadGroupActive, std::vector<uint64_t> &currentCycle)
{
    for (uint64_t threadId = 0; threadId < threadsNum; threadId++) {
        if (GetSim()->GetCycles() < threadState->currentSchedulerEnd[threadId]) {
            MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][AICPUMachine:", machineId, "][updatedispatch] ",
                      dispatchTimeAxe);
            GetSim()->UpdateNextCycles(std::min<uint64_t>(INT_MAX, dispatchTimeAxe));
        } else {
            dispatchTimeAxe = INT_MAX;
            MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][AICPUMachine:", machineId, "][updatedispatch] ",
                      dispatchTimeAxe);
            GetSim()->UpdateNextCycles(std::min<uint64_t>(INT_MAX, dispatchTimeAxe));
        }
        currentCycle.at(threadId) = std::max<uint64_t>(
            std::max<uint64_t>(GetSim()->GetCycles(), threadState->currentCompletionEnd[threadId]),
            threadState->currentSchedulerEnd.at(threadId));
        // 检查submission和completion状态
        bool threadBusy = ((GetSim()->GetCycles() < threadState->currentCompletionEnd[threadId]) ||
                           (GetSim()->GetCycles() <= threadState->currentSchedulerEnd[threadId]));

        bool machineBusy = true;
        uint64_t startIdx = threadId * ((subMachines.size() + threadsNum - 1) / threadsNum);
        uint64_t endIdx =
            std::min((threadId + 1) * ((subMachines.size() + threadsNum - 1) / threadsNum), subMachines.size());
        
        for (uint64_t i = startIdx; i < endIdx; i++) {
            auto &submachine = subMachines[i];
            if (uint64_t(top->executingTaskMap[submachine->machineId] < submachine->maxRunningTasks)) {
                machineBusy = false;
                break;
            }
        }

        if (!threadBusy && !machineBusy && !top->taskMap.empty()) {
            if (!threadState->schedulerDone[threadId]) {
                threadGroupActive[threadId] = true;
                threadState->schedulerDone[threadId] = true;
            }
        }
    }
}

void AICPUMachine::RecordDependency(std::shared_ptr<Task> task)
{
    for (auto &pre : task->predecessors) {
        GetSim()->GetLogger()->AddFlow(pre, task->taskId);
    }
}

void AICPUMachine::ResolveDependence(const std::shared_ptr<CoreMachine> &core, uint64_t threadId,
                                     std::vector<uint64_t> &threadCompletionCycles)
{
    auto resCycles = GetSim()->GetCycles() + threadCompletionCycles[threadId];
    CompletedPacket packet;
    core->completionQueue.Dequeue(packet);
    top->executingTaskMap[core->machineId] --;

    uint64_t taskId = packet.taskId;
    uint64_t taskExeCycle = packet.cycleInfo.taskExecuteEndCycle - packet.cycleInfo.taskExecuteStartCycle;
    stats->totalTaskExecuteCycles += taskExeCycle;
    stats->minTaskExecuteCycles = std::min(stats->minTaskExecuteCycles, taskExeCycle);
    stats->maxTaskExecuteCycles = std::max(stats->maxTaskExecuteCycles, taskExeCycle);

    top->stats->totalTaskExecuteCycles += taskExeCycle;
    top->stats->maxTaskExecuteCycles = std::max(top->stats->maxTaskExecuteCycles, taskExeCycle);
    top->stats->minTaskExecuteCycles = std::min(top->stats->minTaskExecuteCycles, taskExeCycle);

    auto funcHash = top->taskMap[taskId]->functionHash;
    GetSim()->leafFunctionTime[funcHash] = taskExeCycle;

    MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][AICPUMachine][AnalysisPacket] ",
              "  Processing completed taskId: ", taskId, " from core ", core->machineId);

    std::shared_ptr<Task> task;
    std::vector<uint64_t> successors;

    auto taskIt = top->taskMap.find(taskId);
    if (taskIt == top->taskMap.end()) {
        return;
    }
    task = taskIt->second;
    successors = task->successors;  // 复制successors以减少持锁时间
    ASSERT(task->status == true);
    RecordDependency(task);

    // 如果没有successor，则不需要解依赖耗时
    if (!successors.empty()) {
        GetSim()->GetLogger()->AddEventBegin("Resolving_" + std::to_string(taskId), machineId,
                                             threadId + reversedTidNum,
                                             std::max<uint64_t>(GetSim()->GetCycles(), resCycles),
                                             "Device Machine Thread: " + std::to_string(threadId));
    }
    // Process successors
    MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][AICPUMachine][AnalysisPacket] ", "Found ", successors.size(),
              " successors");
    for (uint64_t successorTaskId : successors) {
        MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][AICPUMachine][AnalysisPacket] ",
                  "Processing successor taskId: ", successorTaskId);
        auto &successor = top->taskMap.at(successorTaskId);
        if (successor->remainingPredecessors == 0) {
            continue;
        }
        MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][AICPUMachine][AnalysisPacket] ",
                  "Remaining before: ", successor->remainingPredecessors);
        successor->remainingPredecessors--;
        resCycles += config.resolveCycles;
        stats->threadResolveNum[threadId]++;
        stats->resolveNum++;
        top->stats->resolveNum++;
        threadCompletionCycles[threadId] += config.resolveCycles;
        MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][AICPUMachine][AnalysisPacket] ",
                  "Remaining after: ", successor->remainingPredecessors);

        if (successor->remainingPredecessors == 0) {
            MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][AICPUMachine][AnalysisPacket] ",
                      "Successor is now READY, pushing to ready queue");
            lastCycles = GetSim()->GetCycles();
            localReadyQueues.Enqueue(successorTaskId, resCycles - GetSim()->GetCycles());
        }
    }

    top->taskMap.erase(taskId);

    resolveTimeAxe = std::min<uint64_t>(INT_MAX, resCycles);
    MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][AICPUMachine:", machineId, "][resolve] ", resolveTimeAxe);
    GetSim()->UpdateNextCycles(resolveTimeAxe);

    resolveTimeAxe = std::min<uint64_t>(INT_MAX, resCycles);
    MLOG_INFO("[AICPU:", machineId, "][resolve] ", resolveTimeAxe);
    GetSim()->UpdateNextCycles(resolveTimeAxe);

    // 如果没有successor，则不需要解依赖耗时
    if (!successors.empty()) {
        GetSim()->GetLogger()->AddEventEnd(machineId, threadId + reversedTidNum,
                                           std::max<uint64_t>(GetSim()->GetCycles(), resCycles));
    }
}

void AICPUMachine::PollingMachine()
{
    std::vector<uint64_t> threadCompletionCycles(threadsNum, 0);
    std::vector<bool> threadGroupActive(threadsNum, false);

    // 更新每个线程组的状态
    UpdatePollingStates(threadGroupActive);

    // 按照所有submachine轮询
    for (auto &submachine : subMachines) {
        auto core = std::dynamic_pointer_cast<CoreMachine>(submachine);
        if (!core) {
            continue;
        }

        for (uint64_t threadId = 0; threadId < threadsNum; threadId++) {
            if (!threadGroupActive[threadId]) {
                continue;
            }

            GetSim()->GetLogger()->AddEventBegin(
                "PollingCore_" + std::to_string(core->machineId), machineId, threadId + reversedTidNum,
                GetSim()->GetCycles() + threadCompletionCycles[threadId],
                "AICPU Machine Thread: " + std::to_string(threadId));

            threadCompletionCycles[threadId] += config.completionCycles;
            stats->pollingNum++;
            stats->threadBatchNum[threadId]++;
            top->stats->pollingNum++;

            while (!core->completionQueue.Empty()) {
                ResolveDependence(core, threadId, threadCompletionCycles);
            }

            GetSim()->GetLogger()->AddEventEnd(
                machineId, threadId + reversedTidNum,
                GetSim()->GetCycles() + threadCompletionCycles[threadId]);
        }
    }

    for (uint64_t threadId = 0; threadId < threadsNum; threadId++) {
        if (!threadGroupActive[threadId]) {
            continue;
        }
        threadState->currentCompletionEnd[threadId] =
            GetSim()->GetCycles() + threadCompletionCycles[threadId];

        pollingTimeAxe  = std::min<uint64_t>(INT_MAX, threadState->currentCompletionEnd[threadId]);
        MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][AICPUMachine:", machineId, "][polling] ", pollingTimeAxe);
        GetSim()->UpdateNextCycles(pollingTimeAxe);
    }
}

void AICPUMachine::StatTaskType(const MachineType &type, uint64_t &threadId)
{
    stats->totalSubmitNum++;
    top->stats->totalSubmitNum++;
    stats->threadSubmitNum[threadId]++;
    if (type == MachineType::AIC) {
        stats->cubeSubmitNum++;
        top->stats->cubeSubmitNum++;
    } else if (type == MachineType::AIV) {
        stats->vectorSubmitNum++;
        top->stats->vectorSubmitNum++;
    }
}

void AICPUMachine::SendTask(uint64_t taskId, std::shared_ptr<Machine> subMachine, uint64_t delay, uint64_t threadId)
{
    auto& task = top->taskMap[taskId];
    TaskPack packet;
    packet.taskId = taskId;
    packet.task.taskPtr = task;
    packet.task.coreMachineType = subMachine->machineType;
    packet.task.functionHash = top->taskMap[taskId]->functionHash;
    packet.cycleInfo.taskGenCycle = GetSim()->GetCycles();
    packet.semanticLabels = top->taskMap[taskId]->semanticLabels;
    if (GetSim()->config.replayAllMode == 1) {
        subMachine->SubmitTask(packet, delay);
        MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][AICPU][DispatchPacket] submit task ", taskId,
                  " to Machine ", subMachine->machineId);
        StatTaskType(subMachine->machineType, threadId);
    } else {
        if (!task->status) {
            task->status = true;
            top->executingTaskMap[subMachine->machineId] ++;
            subMachine->SubmitTask(packet, delay);
            MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][AICPU][DispatchPacket] submit task ", taskId,
                      " to Machine ", subMachine->machineId);
            StatTaskType(subMachine->machineType, threadId);
        }
    }
}

void AICPUMachine::DispatchTasksInReplayMode(uint64_t threadId, std::vector<uint64_t>& threadSchedulerCycles,
                                             std::vector<uint64_t>& currentCycle, size_t& lastSubmachineIdx,
                                             uint64_t delayCycle)
{
    size_t count = 0;
    size_t currentIdx = lastSubmachineIdx;
    uint64_t taskId;
    while (count < subMachines.size()) {
        auto& submachine = subMachines[currentIdx];
        
        if (top->readyQueues[submachine->machineType].empty()) {
            currentIdx = (currentIdx + 1) % subMachines.size();
            count++;
            continue;
        }

        auto& aicpuMachines = top->staticSimData;
        uint64_t absaicoreIdx = GetMachineSeq(submachine->machineId);
        uint64_t aicpuIdx;
        uint64_t aicoreIdx;

        if (submachine->machineType == MachineType::AIC) {
            aicpuIdx = absaicoreIdx / GetSim()->config.cubeMachineNumberPerAICPU;
            aicoreIdx = absaicoreIdx % GetSim()->config.cubeMachineNumberPerAICPU;
        } else if (submachine->machineType == MachineType::AIV) {
            uint64_t relativeAivIdx = absaicoreIdx -
                (GetSim()->config.aicpuMachineNumber * GetSim()->config.cubeMachineNumberPerAICPU);
            aicpuIdx = relativeAivIdx / GetSim()->config.vecMachineNumberPerAICPU;
            aicoreIdx = GetSim()->config.cubeMachineNumberPerAICPU +
                (relativeAivIdx % GetSim()->config.vecMachineNumberPerAICPU);
        }
        auto& cm = aicpuMachines[aicpuIdx].coreMachines[aicoreIdx];
        auto& taskPair = cm.taskIds.front();
        taskId = taskPair.first;
        bool res = top->EraseReadyQueue(submachine->machineType, taskId);
        if (res) {
            cm.taskIds.pop_front();
            GetSim()->GetLogger()->AddEventBegin(
                "Dispatch_Task_" + std::to_string(taskId),
                machineId,
                threadId + reversedTidNum,
                currentCycle.at(threadId) + threadSchedulerCycles[threadId],
                "Dispatching Machine Thread: " + std::to_string(threadId)
            );
            threadSchedulerCycles[threadId] += config.schedulerCycles;
            delayCycle += config.schedulerCycles;
            SendTask(taskId, submachine, delayCycle, threadId);
            GetSim()->GetLogger()->AddEventEnd(
                machineId,
                threadId + reversedTidNum,
                currentCycle.at(threadId) + threadSchedulerCycles[threadId]
            );
            currentIdx = (currentIdx + 1) % subMachines.size();
            count++;
        } else {
            currentIdx = (currentIdx + 1) % subMachines.size();
            count++;
            continue;
        }
    }
    lastSubmachineIdx = currentIdx;
}

uint64_t AICPUMachine::GetTaskLoad(MachineType type)
{
    uint64_t minTaskNum = UINT64_MAX;
    if (type == MachineType::AIC) {
        for (const auto& pair : top->executingTaskMap) {
            if (GetMachineType(pair.first) == static_cast<int>(CostModel::MachineType::AIC)) {
                minTaskNum = std::min(minTaskNum, pair.second);
            }
        }
    } else if (type == MachineType::AIV) {
        for (const auto& pair : top->executingTaskMap) {
            if (GetMachineType(pair.first) == static_cast<int>(CostModel::MachineType::AIV)) {
                minTaskNum = std::min(minTaskNum, pair.second);
            }
        }
    }

    if (minTaskNum == UINT64_MAX) {
        minTaskNum = 0;
    }

    return minTaskNum;
}

void AICPUMachine::DispatchTasksInNormalMode(uint64_t threadId, std::vector<uint64_t>& threadSchedulerCycles,
                                             std::vector<uint64_t>& currentCycle, uint64_t delayCycle)
{
    for (size_t level = 0; level < wlStatus.maxLevel; level++) {
        for (size_t group = 0; group < wlStatus.groups; group++) {
            size_t subMachineIndex = wlStatus.smtGroupIndexs[group][level];
            auto &submachine = subMachines[subMachineIndex];
            // No task exists in the readQueue of the current machine type.
            if (top->readyQueues[submachine->machineType].empty()) {
                continue;
            }

            // Submit Task To submachine.
            uint64_t minTaskNum = GetTaskLoad(submachine->machineType);
            if (top->executingTaskMap[submachine->machineId] < submachine->maxRunningTasks && 
                top->executingTaskMap[submachine->machineId] <= minTaskNum) {
                uint64_t taskId = top->PopReadyQueue(submachine->machineType);
                GetSim()->GetLogger()->AddEventBegin("Dispatch_Task_" + std::to_string(taskId), machineId,
                threadId + reversedTidNum, currentCycle.at(threadId) + threadSchedulerCycles[threadId],
                "Dispatching Machine Thread: " + std::to_string(threadId));
                threadSchedulerCycles[threadId] += config.schedulerCycles;
                delayCycle += config.schedulerCycles;
                SendTask(taskId, submachine, delayCycle, threadId);
                GetSim()->GetLogger()->AddEventEnd(machineId, threadId + reversedTidNum,
                    currentCycle.at(threadId) + threadSchedulerCycles[threadId]);
            }
        }
    }
}

void AICPUMachine::DispatchTasksForThread(uint64_t threadId, std::vector<uint64_t>& threadSchedulerCycles,
                                          std::vector<uint64_t>& currentCycle, size_t& lastSubmachineIdx)
{
    uint64_t delayCycle = 0;
    if (GetSim()->config.replayDispatchMode == 1) {
        DispatchTasksInReplayMode(threadId, threadSchedulerCycles, currentCycle, lastSubmachineIdx, delayCycle);
    } else {
        DispatchTasksInNormalMode(threadId, threadSchedulerCycles, currentCycle, delayCycle);
    }
}

void AICPUMachine::DispatchPacket()
{
    std::vector<uint64_t > threadSchedulerCycles(threadsNum, 0);
    std::vector<uint64_t> currentCycle(threadsNum, 0);
    std::vector<bool> threadGroupActive(threadsNum, false);
    static size_t lastSubmachineIdx = 0;

    bool readyQueueEmpty = true;
    for (auto &readyQ : top->readyQueues) {
        if (!readyQ.second.empty()) {
            readyQueueEmpty = false;
            break;
        }
    }
    // 更新每个线程组的状态
    UpdateDispatchStates(threadGroupActive, currentCycle);
    // 按照所有submachine轮询
    for (uint64_t threadId = 0; threadId < threadsNum; threadId++) {
        if (!threadGroupActive[threadId] || readyQueueEmpty) {
            continue;
        }
        DispatchTasksForThread(threadId, threadSchedulerCycles, currentCycle, lastSubmachineIdx);
    }
    
    for (uint64_t threadId = 0; threadId < threadsNum; threadId++) {
        if (!threadGroupActive[threadId]) {
            continue;
        }
        threadState->currentSchedulerEnd[threadId] =
            std::max<uint64_t>(GetSim()->GetCycles(), currentCycle.at(threadId)) +
            threadSchedulerCycles[threadId];

        dispatchTimeAxe = std::min<uint64_t>(INT_MAX, threadState->currentSchedulerEnd[threadId]);
        if (threadSchedulerCycles[threadId] != 0) {
            MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][AICPUMachine:", machineId, "][dispatch] ", dispatchTimeAxe);
            GetSim()->UpdateNextCycles(dispatchTimeAxe);
        } else {
            MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][AICPUMachine:", machineId, "][dispatch] ",
                      GetSim()->GetCycles() + 1);
            GetSim()->UpdateNextCycles(GetSim()->GetCycles() + 1);
        }
    }
}

void AICPUMachine::ProcessCompletionPacket(std::shared_ptr<Machine> &submachine, DeviceMachine::CoreMachineQueue &cm,
		                           uint64_t &taskId, uint64_t &currentRecordCycles, uint64_t &currentGlobalDelay)
{
    CompletedPacket packet;
    submachine->completionQueue.Front(packet);
    if (packet.taskId == taskId) {
        submachine->completionQueue.Dequeue(packet);
        cm.taskIds.pop_front();
        MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][AICPU][ProcessPacket] Completed task ", packet.taskId, " from Core ", submachine->machineId);
        if (!cm.taskIds.empty()) {
            auto nextTaskPair = cm.taskIds.front();
            MLOG_INFO("NextTaskID-", nextTaskPair.first, " RecordCycles-", currentRecordCycles,
                      " NextTaskCycle-", nextTaskPair.second.first + currentGlobalDelay);
            currentRecordCycles = std::min(currentRecordCycles, nextTaskPair.second.first + currentGlobalDelay);
            if (currentRecordCycles == GetSim()->GetCycles()) {
                currentRecordCycles ++;
                nextTaskPair.second.first++;
                cm.taskIds.front() = nextTaskPair;
            }
        }
    else {
            currentRecordCycles = std::min(currentRecordCycles, GetSim()->GetCycles() + 1);
        }
    }
}

void AICPUMachine::ReplayWithTimeStamp(std::shared_ptr<CostModel::Machine> &submachine,
                                       DeviceMachine::CoreMachineQueue &cm, uint64_t aicpuIdx,
                                       uint64_t &delayCycle)
{
    uint64_t threadId = 1; // Json info has no thread info now
    auto taskPair = cm.taskIds.front();
    uint64_t taskId = taskPair.first;
    uint64_t beginCycle = taskPair.second.first;
    uint64_t endCycle = taskPair.second.second;
    static std::unordered_map<uint64_t, bool> taskSubmitted;

    MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][AICPU][ReplayWithTimeStamp] Begin replay for task ", taskId);
    MLOG_INFO("  Machine ID: ", submachine->machineId);
    MLOG_INFO("  AICPU Index: ", aicpuIdx); 
    MLOG_INFO("  Record Begin Cycle: ", beginCycle);
    MLOG_INFO("  Record End Cycle: ", endCycle);
    MLOG_INFO("  Current Global Delay: ", globalDelay);
    MLOG_INFO("  Task Submitted Status: ", taskSubmitted[taskId]);

    if (GetSim()->GetCycles() < beginCycle + globalDelay && !taskSubmitted[taskId]) {
        recordCycles = std::min(recordCycles, beginCycle + globalDelay);
        MLOG_INFO("  Waiting to reach begin cycle. Next record cycle updated to: ", recordCycles);
    } else if (GetSim()->GetCycles() == beginCycle + globalDelay && !taskSubmitted[taskId]) {
        recordCycles = std::min(recordCycles, endCycle + globalDelay);
        MLOG_INFO("  Reached begin cycle, submitting task. Next record cycle: ", recordCycles);
        MLOG_INFO("  Adding dispatch event begin at cycle: ", beginCycle + globalDelay);
        
        GetSim()->GetLogger()->AddEventBegin(
            "Dispatch_Task_" + std::to_string(taskId),
            machineId,
            reversedTidNum,
            beginCycle + globalDelay,
            "Dispatching AICPUMachine: " + std::to_string(aicpuIdx)
        );

        MLOG_INFO("  Sending task to submachine...");
        SendTask(taskId, submachine, delayCycle, threadId);
        
        taskSubmitted[taskId] = true;
        MLOG_INFO("  Task submitted successfully");
        MLOG_INFO("  Adding dispatch event end at cycle: ", beginCycle + globalDelay + config.schedulerCycles);
        GetSim()->GetLogger()->AddEventEnd(
            machineId,
            reversedTidNum,
            (beginCycle + globalDelay + config.schedulerCycles)
        );
    } else if (!submachine->completionQueue.Empty()) {
        MLOG_INFO("  Processing completion packet");
        MLOG_INFO("  Old global delay: ", globalDelay);
    if (GetSim()->GetCycles() >= endCycle) {
            globalDelay = std::max(globalDelay, GetSim()->GetCycles() - endCycle);
        }
        MLOG_INFO("  New global delay: ", globalDelay);
        
        ProcessCompletionPacket(submachine, cm, taskId, recordCycles, globalDelay);
        
        taskSubmitted.erase(taskId);
        MLOG_INFO("  Removed task ", taskId, " from submitted map");
    }

    MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][AICPU][ReplayWithTimeStamp] End replay for task ", taskId);
}

void AICPUMachine::ReplayAll()
{
    auto& aicpuMachines = top->staticSimData;
    uint64_t delayCycle = 0;
    recordCycles = INT_MAX;
    uint64_t aicpuIdx;
    for (auto &submachine : subMachines) {
        uint64_t absaicoreIdx = GetMachineSeq(submachine->machineId);
        uint64_t aicoreIdx;
        if (submachine->machineType == MachineType::AIC) {
            aicpuIdx = absaicoreIdx / GetSim()->config.cubeMachineNumberPerAICPU;
            aicoreIdx = absaicoreIdx % GetSim()->config.cubeMachineNumberPerAICPU;
        } else if (submachine->machineType == MachineType::AIV) {
            uint64_t relativeAivIdx = absaicoreIdx -
                (GetSim()->config.aicpuMachineNumber * GetSim()->config.cubeMachineNumberPerAICPU);
            aicpuIdx = relativeAivIdx / GetSim()->config.vecMachineNumberPerAICPU;
            aicoreIdx = GetSim()->config.cubeMachineNumberPerAICPU +
                (relativeAivIdx % GetSim()->config.vecMachineNumberPerAICPU);
        } else {
            continue;
        }

        auto& cm = aicpuMachines[aicpuIdx].coreMachines[aicoreIdx];
        if (cm.taskIds.empty()) {
            continue;
        }
        ReplayWithTimeStamp(submachine, cm, aicpuIdx, delayCycle);
    }
    
    bool aicpuEmpty = true;
    for (const auto& coremap : aicpuMachines[aicpuIdx].coreMachines) {
        if (!coremap.taskIds.empty()) {
            aicpuEmpty = false;
            break;
        }
    }
    isTerminate = aicpuEmpty;
    
    GetSim()->UpdateNextCycles(recordCycles);
}

void AICPUMachine::CheckDeadlock()
{
    // shutdown check
    bool execute = false;

    for (auto &submachine : subMachines) {
        if (submachine &&
            (submachine->machineType == MachineType::AIC || submachine->machineType == MachineType::AIV)) {
            if (!submachine->completionQueue.Empty()) {
                execute = true;
                break;
            }
        }
    }

    if (!top->taskMap.empty() && dispatchTimeAxe == INT_MAX && pollingTimeAxe == INT_MAX && execute) {
        GetSim()->UpdateNextCycles(GetSim()->GetCycles() + 1);
    }

    for (const auto &pair : top->readyQueues) {
        if (!pair.second.empty()) {
            execute = true;
        }
    }

    // 检查每个线程的状态
    for (uint64_t threadId = 0; threadId < threadsNum; threadId++) {
        if (threadState->batchProcessing[threadId] || !threadState->completionDone[threadId] ||
            !threadState->schedulerDone[threadId]) {
            threadState->threadDone[threadId] = true;
        }
    }

    execute |= std::any_of(threadState->threadDone.begin(), threadState->threadDone.end(),
                           [](bool done) { return !done; });

    SetMachineExecuting(execute);
}
}
