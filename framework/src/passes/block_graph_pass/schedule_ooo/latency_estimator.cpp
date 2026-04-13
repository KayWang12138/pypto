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
 * \file latency_estimator.cpp
 * \brief
 */

#include "passes/block_graph_pass/schedule_ooo/latency_estimator.h"

namespace npu::tile_fwk {

LatencyEstimator::LatencyEstimator(std::vector<Operation*>& newTaskList,
    std::vector<Operation*>& newOperations, CoreLocationType coreLocation)
    : OoOScheduler(), taskList(newTaskList), operations(newOperations), coreLocation_(coreLocation)
{
    InitMemWithoutAlloc();
    InitLatencyEstimator();
}

void LatencyEstimator::InitLatencyEstimator()
{
    // 初始化芯片各buffer大小
    localMemSize = CommonUtils::GetLocalMemorySize();
    localMemoryCurrentSize = localMemSize;

    // 校验并初始化 Operation，直接使用 taskList 作为调度对象
    for (const auto& op : taskList) {
        if (CheckOpBufferSize(op) != SUCCESS) {
            APASS_LOG_ERROR_F(
                Elements::Operation, "%s[%d] CheckOpBufferSize failed!",
                op->GetOpcodeStr().c_str(), op->GetOpMagic());
            continue;
        }
    }

    for (const auto& op : taskList) {
        opIsRetiredMap[op] = false;
    }

    if (InitBufRefCount(taskList) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitBufRefCount failed!");
        return;
    }

    if (depManager_.InitDependencies(taskList, true) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitDependencies failed!");
        return;
    }

    depManager_.InitDependencies(taskList);
    // 单独初始化 latency 专用的 IssueQueue
    InitLatencyAllocIssueQueues();
}

void LatencyEstimator::InitLatencyIssueQueues()
{
    // 初始化 issueQueues
    for (size_t i = 0; i <= static_cast<int>(PipeType::PIPE_FIX); i++) {
        issueQueues[coreLocation_][static_cast<PipeType>(i)] = IssueQueue();
    }
    allocIssueQueue.clear();
    // 仅为当前 coreLocation 创建所需内存类型的 alloc 队列
    if (coreLocation_ == CoreLocationType::AIV0 || coreLocation_ == CoreLocationType::AIV1) {
        allocIssueQueue[coreLocation_][MemoryType::MEM_UB] = IssueQueue();
    } else {
        for (size_t i = 1; i < static_cast<int>(MemoryType::MEM_DEVICE_DDR); i++) {
            allocIssueQueue[coreLocation][static_cast<MemoryType>(i)] = IssueQueue();
        }
    }
}

void LatencyEstimator::LaunchReadyIssue()
{
    for (auto &op : taskList) {
        if (USE_LESS_OPS.find(op->GetOpcode()) != USE_LESS_OPS.end() && depManager_.GetPredecessors(op).empty()) {
            auto type = RescheduleUtils::GetOpPipeType(op);
            issueQueues[coreLocation_][type].Insert(op);
        }
        if (IsOpAlloc(op)) {
            auto tensor = op->GetOOperands()[0];
            auto memId = tensor->memoryrange.memId;
            allocIssueQueue[coreLocation_][localBufferMap_[memId]->memType].Insert(op);
        }
    }
}

Status LatencyEstimator::PreMainLoop()
{
    LaunchReadyIssue();
    numTotalIssues = taskList.size();
    return SUCCESS;
}

Status LatencyEstimator::FreeBuffer(Operation* op)
{
    for (auto tensor : GetInOutOperandCached(op)) {
        auto memId = tensor->memoryrange.memId;
        if (DelBufRefCount(memId) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "DelBufRefCount tensor [%d] failed.", memId);
            return FAILED;
        }
        if (bufRefCount_[memId] == 0) {
            auto freeMemSize = localBufferMap_[memId]->size;
            if (spillblockMemIds.find(memId) == spillblockMemIds.end()) {
                localMemoryCurrentSize[localBufferMap_[memId]->memType] += freeMemSize;
                APASS_LOG_DEBUG_F(
                    Elements::Operation, "FreeBuffer memType: %d, currentSize %ld, memId: %d, freeMemSize: %lu.",
                    localBufferMap_[memId]->memType,
                    static_cast<long>(localMemoryCurrentSize[localBufferMap_[memId]->memType]), memId,
                    static_cast<unsigned long>(freeMemSize));
            } else {
                APASS_LOG_DEBUG_F(
                    Elements::Operation, "FreeBuffer memType: %d, memId: %d free in spillblock",
                    localBufferMap_[memId]->memType, memId);
            }

            if (localMemoryCurrentSize[localBufferMap_[memId]->memType] >
                    localMemSize[localBufferMap_[memId]->memType] ||
                localMemoryCurrentSize[localBufferMap_[memId]->memType] < 0) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Free tensor [%d] failed.", memId);
                return FAILED;
            }
            APASS_LOG_DEBUG_F(Elements::Tensor, "Free tensor [%d] success.", memId);
        }
    }
    return SUCCESS;
}

Status LatencyEstimator::ExecuteAllocIssue(uint64_t& commitCnt, MemoryType memType, IssueQueue& pipe)
{
    bool canAlloc = true;
    while (canAlloc) {
        if (pipe.Empty()) {
            canAlloc = false;
            break;
        }
        Operation* op = pipe.Front();
        auto memId = GetInOutOperandCached(op)[0]->memoryrange.memId;
        auto needMemSize = localBufferMap_[memId]->size;
        if (localMemoryCurrentSize[memType] >= static_cast<long int>(needMemSize)) {
            APASS_LOG_DEBUG_F(Elements::Operation, "ALLOCATE: %s.", GetOpInfo(op).c_str());
            localMemoryCurrentSize[memType] -= needMemSize;
            APASS_LOG_DEBUG_F(
                Elements::Operation, "ExecuteAllocIssue memType: %d, currentSize %ld, memId: %d.", memType,
                static_cast<long>(localMemoryCurrentSize[memType]), memId);
            if (localMemoryCurrentSize[memType] > localMemSize[memType] || localMemoryCurrentSize[memType] < 0) {
                APASS_LOG_ERROR_F(
                    Elements::Tensor, "Allocate Tensor[%d] failed.", GetInOutOperandCached(op)[0]->GetMagic());
                return FAILED;
            }
            pipe.PopFront();
            if (RetireOpAndAwakeSucc(op, commitCnt) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "RetireOpAndAwakeSucc failed. %s", GetOpInfo(op).c_str());
                return FAILED;
            }
        } else {
            canAlloc = false;
            APASS_LOG_DEBUG_F(Elements::Tensor, "Cannot alloc Tensor[%d] ", GetInOutOperandCached(op)[0]->GetMagic());
            break;
        }
    }
    return SUCCESS;
}

Status LatencyEstimator::LaunchIssueStage(int& nextCycle)
{
    for (auto& [coreLocation, queue] : issueQueues)
        for (auto& [pipeType, pipe] : queue) {
            (void)pipeType;
            if (pipe.Empty() || pipe.busy) {
                continue;
            }
            Operation* op = pipe.PopFront();
            pipe.busy = true;
            pipe.curIssue = op;
            pipe.curOpRetireCycle = clock + op->GetLatency();
            if (nextCycle == -1 || nextCycle > pipe.curOpRetireCycle) {
                nextCycle = pipe.curOpRetireCycle;
            }

            APASS_LOG_DEBUG_F(Elements::Operation, "issueQueues Insert: %s.", GetOpInfo(op).c_str());
        }
    }
    return SUCCESS;
}

Status LatencyEstimator::SpillOnBlock()
{
    MemoryType spillMemType;
    if (!allocIssueQueue[coreLocation_][MemoryType::MEM_UB].Empty()) {
        spillMemType = MemoryType::MEM_UB;
    } else if (!allocIssueQueue[coreLocation_][MemoryType::MEM_L1].Empty()) {
        spillMemType = MemoryType::MEM_L1;
    } else {
        APASS_LOG_ERROR_F(
            Elements::Operation, "Buffer[L0A/B/C] is Full. Please check tile shape and OOO spill failed info.");
        return FAILED;
    }

    Operation* op = allocIssueQueue[coreLocation_][spillMemType].Front();
    size_t needMemSize = GetInOutOperandCached(op)[0]->MemorySize();
    spillblockMemIds.insert(GetInOutOperandCached(op)[0]->memoryrange.memId);
    localMemoryCurrentSize[spillMemType] += static_cast<long int>(needMemSize);
    if (localMemoryCurrentSize[spillMemType] < 0 || localMemoryCurrentSize[spillMemType] > localMemSize[spillMemType]) {
        APASS_LOG_ERROR_F(Elements::Operation, "Buffer[%d] is valid. Please check", spillMemType);
        return FAILED;
    }
    return SUCCESS;
}

Status LatencyEstimator::LatencyEstimatorMainLoop()
{
    if (RunMainLoop() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "RunMainLoop failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status LatencyEstimator::PostMainLoop()
{
    APASS_LOG_DEBUG_F(Elements::Operation, "\n Estimate Latency: %d", clock);
    return SUCCESS;
}

void LatencyEstimator::InitMemWithoutAlloc()
{
    std::unordered_set<int> memIds;
    std::unordered_map<int, Operation*> memIdAllocMap;
    bool needAddAlloc = false;
    for (const auto& op : taskList) {
        if (IsOpAlloc(op)) {
            memIdAllocMap[op->GetOutputOperand(0)->memoryrange.memId] = op;
        }
        for (auto& iOperand : op->GetIOperands()) {
            if (iOperand->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
                memIds.insert(iOperand->memoryrange.memId);
            }
        }
        for (auto& oOperand : op->GetOOperands()) {
            if (oOperand->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
                memIds.insert(oOperand->memoryrange.memId);
            }
        }
    }
    for (const auto& memId : memIds) {
        if (memIdAllocMap.find(memId) != memIdAllocMap.end()) {
            continue;
        }
        APASS_LOG_INFO_F(Elements::Operation, "The alloc op of memId[%d] in other graph", memId);
        needAddAlloc = true;
        for (const auto& op : operations) {
            if (IsOpAlloc(op) && op->GetOutputOperand(0)->memoryrange.memId == memId) {
                taskList.push_back(op);
                APASS_LOG_INFO_F(Elements::Operation, "Add alloc op %s for memId[%d]", GetOpInfo(op).c_str(), memId);
            }
        }
    }
    std::vector<Operation*> opList;
    if (needAddAlloc) {
        for (auto op : operations) {
            if (std::find(taskList.begin(), taskList.end(), op) != taskList.end()) {
                opList.push_back(op);
            }
        }
        taskList = opList;
    }
}

} // namespace npu::tile_fwk
