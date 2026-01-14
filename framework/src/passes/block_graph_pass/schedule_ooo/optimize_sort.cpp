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
 * \file optimize_sort.cpp
 * \brief
 */

#include "optimize_sort.h"
#include "passes/pass_log/pass_log.h"


namespace npu::tile_fwk {
void OptimizeSort::UpdatePreNodeQueue(std::unordered_set<Operation*> &curr,
    std::unordered_set<Operation*> &preNodeTotal, std::map<Operation*, bool>& visited) {
    std::unordered_set<Operation*> next;
    for (auto& curOp : curr) {
        for (auto& preOp : inGraph[curOp]) {
            if (!visited[preOp] && preNodeTotal.find(preOp) == preNodeTotal.end()) {
                next.insert(preOp);
            }
        }
    }
    for (auto& nextOp : next) {
        preNodeTotal.insert(nextOp);
    }
    curr.swap(next);
}

int OptimizeSort::GetNumUnvisitPreNode(Operation* op, std::map<Operation*, bool>& visited) {
    std::unordered_set<Operation*> preNodeTotal;
    std::unordered_set<Operation*> curr;
    for (auto& preOp : inGraph[op]) {
        if (!visited[preOp]) {
            curr.insert(preOp);
            preNodeTotal.insert(preOp);
        }
    }
    while (!curr.empty()) {
        UpdatePreNodeQueue(curr, preNodeTotal, visited);
    }
    return preNodeTotal.size();
}

Operation* OptimizeSort::FindNodeMinNumUnvisitedPreNode(
    std::map<Operation*, bool> visited, std::vector<Operation*> outNodeQueue) {
    Operation* res = nullptr;
    int minUnvisitedNode = INT_MAX;
    for (auto& outNode : outNodeQueue) {
        if (visited[outNode]) {
            continue;
        }
        int curUnvisitedNode = GetNumUnvisitPreNode(outNode, visited);
        if (curUnvisitedNode < minUnvisitedNode) {
            res = outNode;
            minUnvisitedNode = curUnvisitedNode;
        }
    }
    return res;
}

int OptimizeSort::GetNodePriority(std::unordered_map<Opcode, int> preNodePriority, Operation* op) {
    int prior = 10;
    if (preNodePriority.find(op->GetOpcode()) != preNodePriority.end()) {
        prior = preNodePriority[op->GetOpcode()];
    }
    return prior;
}

int OptimizeSort::GetMaxDepthSimple(Operation* op) {
    if (inGraph[op].empty()) {
        return 1;
    }

    int maxDepth = 0;
    for (const auto& pre : inGraph[op]) {
        int depth = GetMaxDepthSimple(pre);
        if (depth > maxDepth) {
            maxDepth = depth;
        }
    }
    return maxDepth + 1;
}

void OptimizeSort::QueueNotReadyPreNode(Operation* curOp, std::map<Operation*, bool>& visited,
    std::unordered_map<Opcode, int> preNodePriority, std::deque<Operation*> &queue) {
    std::vector<Operation*> notReadyPreNode;
    for (auto& preOp : inGraph[curOp]) {
        if (!visited[preOp]) {
            notReadyPreNode.push_back(preOp);
        }
    }
    std::sort(notReadyPreNode.begin(), notReadyPreNode.end(), [&](Operation* a, Operation* b) {
        int priorA = GetNodePriority(preNodePriority, a);
        int priorB = GetNodePriority(preNodePriority, b);
        if (priorA != priorB) {
            return priorA < priorB;
        } else {
            int depA = GetMaxDepthSimple(a);
            int depB = GetMaxDepthSimple(b);
            if (depA == depB) {
                int aIdx = std::find(operations.begin(), operations.end(), a) - operations.begin();
                int bIdx = std::find(operations.begin(), operations.end(), b) - operations.begin();
                return aIdx < bIdx;
            }
            return depA < depB;
        }
    });
    for (auto& preOp : notReadyPreNode) {
        queue.push_front(preOp);
    }
}

void OptimizeSort::ForwardDfs(Operation* curOp, std::vector<Operation*>& newOpList,
    std::map<Operation*, bool>& visited, std::unordered_map<Opcode, int> preNodePriority,
    std::deque<Operation*> &queue) {
    bool ready = true;
    for (auto& preOp : inGraph[curOp]) {
        if (!visited[preOp]) {
            ready = false;
            break;
        }
    }

    if (ready) {
        visited[curOp] = true;
        queue.pop_front();
        newOpList.push_back(curOp);
    } else {
        QueueNotReadyPreNode(curOp, visited, preNodePriority, queue);
    }
}

void OptimizeSort::DFSFromSingleNode(Operation* op, std::map<Operation*, bool>& visited,
    std::vector<Operation*>& newOpList, std::unordered_map<Opcode, int> preNodePriority) {
    if (visited[op]) {
        return;
    }

    std::deque<Operation*> queue = {op};
    while (!queue.empty()) {
        auto curOp = queue.front();
        if (visited[curOp]) {
            queue.pop_front();
            continue;
        }

        ForwardDfs(curOp, newOpList, visited, preNodePriority, queue);
    }
}

Status OptimizeSort::DFSFromOutNode(std::vector<Operation*> outNodeQueue,
    std::unordered_map<Opcode, int> preNodePriority, std::map<Operation*, bool> &visited) {
    std::vector<Operation*> newOpList;
    if (outNodeQueue.size() != 0) {
       DFSFromSingleNode(outNodeQueue[0], visited, newOpList, preNodePriority);
    } else {
        APASS_LOG_ERROR_F(Elements::Operation, "Subgraph must have operation with outdegree 0.");
        return FAILED;
    }

    for (size_t i = 1; i < outNodeQueue.size(); i++) {
        while (!visited[outNodeQueue[i]]) {
            auto node = FindNodeMinNumUnvisitedPreNode(visited, outNodeQueue);
            if (node == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation, "FindNodeMinNumUnvisitedPreNode failed.");
                return FAILED;
            }
            DFSFromSingleNode(node, visited, newOpList, preNodePriority);
        }
    }
    operations = newOpList;
    return SUCCESS;
}

Status OptimizeSort::PriorDFS(std::unordered_map<Opcode, int> preNodePriority) {
    std::map<Operation*, bool> visited;
    std::vector<Operation*> outNodeQueue;
    for (size_t i = 0; i < operations.size(); i++) {
        visited[operations[i]] = false;
        if (outGraph[operations[i]].empty()) {
            outNodeQueue.emplace_back(operations[i]);
        }
    }

    if (DFSFromOutNode(outNodeQueue, preNodePriority, visited) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "DFSFromOutNode failed.");
        return FAILED;
    }
    return SUCCESS;
}

// rollBackOp 和 backOp 是否存在前后序依赖
bool OptimizeSort::HasDependency(Operation* rollBackOp, Operation* backOp) {
    std::map<Operation*, bool> visited;
    for (auto op : operations) {
        visited[op] = false;
    }
    std::function<bool(Operation*)> dfs = [&](Operation* op) ->bool{
        if (op == backOp) return true;
        if (visited[op]) return false;

        visited[op] = true;
        for (auto succOp : outGraph[op]) {
            if (dfs(succOp)) {
                return true;
            }
        }
        return false;
    };
    return dfs(rollBackOp);
}

// 在 curOpList 中将 advanceIndexList 中的序列提前到 rollBackIndex 之前,更新 curOpList
void OptimizeSort::ReplaceIndex(std::vector<Operation*> &curOpList,
    std::set<size_t> advanceIndexList, size_t rollBackIndex) {
    std::vector<Operation*> moveOpList;
    for (auto i : advanceIndexList) {
        APASS_LOG_DEBUG_F(Elements::Operation, "advance index: %d, op: %s",
                i, GetOpInfo(curOpList[i]).c_str());
        moveOpList.push_back(curOpList[i]);
    }
    for (auto it = advanceIndexList.rbegin(); it != advanceIndexList.rend(); ++it) {
        curOpList.erase(curOpList.begin() + (*it));
    }
    curOpList.insert(curOpList.begin() + rollBackIndex, moveOpList.begin(), moveOpList.end());
}

void OptimizeSort::GetPreNode(size_t i, std::vector<Operation*> curOpList, size_t rollBackIndex,
    size_t backTraceIndex, std::set<size_t> &dependencyIndexList) {
    dependencyIndexList.insert(i);
    APASS_LOG_DEBUG_F(Elements::Operation, "dependencyIndexList push index: %d, Op: %s",
        i, GetOpInfo(curOpList[i]).c_str());
    for (auto preOp : inGraph[curOpList[i]]) {
        auto it = std::find(curOpList.begin() + rollBackIndex + 1, curOpList.begin() + backTraceIndex, preOp);
        if (it != curOpList.begin() + backTraceIndex) {
            auto index = std::distance(curOpList.begin(), it);
            GetPreNode(index, curOpList, rollBackIndex, backTraceIndex, dependencyIndexList);
        }
    }
}

// 记录 curOpList 中从 rollBackIndex 到 backTraceIndex 中所有和 rollBack 没有后继依赖的点
void OptimizeSort::GetListToAdvance(size_t rollBackIndex, size_t backTraceIndex,
    std::vector<Operation*> curOpList, std::set<size_t> &advanceIndexList) {
    std::set<size_t> dependencyIndexList;
    for (size_t i = rollBackIndex + 1; i <= backTraceIndex; i++) {
        if (HasDependency(curOpList[rollBackIndex], curOpList[i])) {
            GetPreNode(i, curOpList, rollBackIndex, backTraceIndex, dependencyIndexList);
        }
    }
    for (size_t i = rollBackIndex + 1; i <= backTraceIndex; i++) {
        if (dependencyIndexList.count(i) == 0) {
            advanceIndexList.insert(i);
            APASS_LOG_DEBUG_F(Elements::Operation, "advanceIndexList push index: %d, op: %s",
                i, GetOpInfo(curOpList[i]).c_str());
        }
    }
}

// rollBackIndex 位置回退
Status OptimizeSort::RollBack(size_t &startIndex,
    std::vector<Operation*> &curOpList, std::map<MemoryType, int64_t> &curMemoryMap) {
    APASS_LOG_DEBUG_F(Elements::Operation, "=====> Start RollBack.");
    RestoreCheckpoint(backTraceCheckpoint, startIndex, curOpList, curMemoryMap);
    size_t backTraceIndex = startIndex + 1;
    if (backTraceIndex >= curOpList.size()) {
        APASS_LOG_ERROR_F(Elements::Operation, "RollBack invalid backTraceIndex: %d", backTraceIndex);
        return FAILED;
    }
    backTraceOp = curOpList[backTraceIndex];
    MemoryType memType = backTraceOp->GetOutputOperand(0)->GetMemoryTypeOriginal();
    size_t rollBackIndex = backTraceIndex;
    APASS_LOG_DEBUG_F(Elements::Operation, "backTraceOp: %s, backTraceIndex: %d, memType: %d",
        GetOpInfo(backTraceOp).c_str(), backTraceIndex, memType);
    std::set<size_t> advanceIndexList;
    while (rollBackIndex < curOpList.size() && rollBackIndex > 0) {
        rollBackIndex--;
        Operation* rollBackOp = curOpList[rollBackIndex];
        if (rollBackOp->GetOutputOperand(0)->GetMemoryTypeOriginal() != memType || !(IsOpAlloc(rollBackOp)) ||
            HasDependency(rollBackOp, backTraceOp)) {
            continue;
        }
        rollBackNodeOp = rollBackOp;
        APASS_LOG_DEBUG_F(Elements::Operation, "Select rollBackOp: %s, rollBackIndex: %d",
            GetOpInfo(rollBackOp).c_str(), rollBackIndex);
        advanceIndexList.clear();
        GetListToAdvance(rollBackIndex, backTraceIndex, curOpList, advanceIndexList);
        ReplaceIndex(curOpList, advanceIndexList, rollBackIndex);
        startIndex = rollBackIndex;
        APASS_LOG_DEBUG_F(Elements::Operation, "RollBack==>change startIndex: %d", startIndex);
        if (rollBackIndex != 0) {
            if (RebuildStateToIndex(curOpList, rollBackIndex - 1, curMemoryMap) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "RollBack rebuild failed.");
                return FAILED;
            }
            UpdateVisitedByIndex(curOpList, startIndex - 1, true);
            return SUCCESS;
        }
        curMemoryMap = {{MemoryType::MEM_L0A, 0}, {MemoryType::MEM_L0B, 0}, {MemoryType::MEM_L0C, 0}};
        operations = curOpList;
        for (auto op : curOpList) {
            visitedOp[op] = false;
        }
        InitBufRefCount();
        opDeltas.clear();
        return SUCCESS;
    }
    APASS_LOG_ERROR_F(Elements::Operation, "RollBack Failed");
    return FAILED;
}

// 在 curOpList 中将 preOpList 中的序列提前到 startIndex 之后，更新 curOpList
void OptimizeSort::ReorderOp(std::vector<size_t> &preIdx, std::vector<Operation*> &curOpList,
    size_t startIndex) {
    // 对 perOpList 排序，再进行插入
    std::sort(preIdx.begin(), preIdx.end());
    std::vector<Operation*> moveOpList;
    APASS_LOG_DEBUG_F(Elements::Operation, "current index: %d, preIdx size: %d", startIndex, preIdx.size());
    for (auto i : preIdx) {
        APASS_LOG_DEBUG_F(Elements::Operation, "preidx : %d, curOp: %s", i, GetOpInfo(curOpList[i]).c_str());
        moveOpList.push_back(curOpList[i]);
    }
    for (auto it = preIdx.rbegin(); it != preIdx.rend(); ++it) {
        curOpList.erase(curOpList.begin() + (*it));
    }
    curOpList.insert(curOpList.begin() + startIndex + 1, moveOpList.begin(), moveOpList.end());
}

void OptimizeSort::FindIndex(Operation* op, std::vector<Operation*> curOpList, size_t &index) {
    for (size_t i = 0; i < curOpList.size(); i++) {
        if (curOpList[i] == op) {
            index = i;
            return;
        }
    }
}

// 在curOpList中，向前遍历找到consumerIndex的前序未被访问的节点，并放入preOpList中
Status OptimizeSort::FindConsumerList(size_t consumerIndex, std::vector<size_t> &preOpList, std::vector<Operation*> &curOpList) {
    if (curOpList[consumerIndex] == backTraceOp) {
        APASS_LOG_WARN_F(Elements::Operation, "backTraceOp is one of the predecessor node.");
        return FAILED;
    }
    if (curOpList[consumerIndex] == rollBackNodeOp) {
        APASS_LOG_WARN_F(Elements::Operation, "rollBackNodeOp is one of the predecessor node.");
        return FAILED;
    }
    visitedOp[curOpList[consumerIndex]] = true;
    preOpList.push_back(consumerIndex);
    APASS_LOG_DEBUG_F(Elements::Operation, "unvisited consumer idx: %d, op: %s", consumerIndex, GetOpInfo(curOpList[consumerIndex]).c_str());
    for (auto op : inGraph[curOpList[consumerIndex]]) {
        if (visitedOp[op] == false) {
            size_t index;
            FindIndex(op, curOpList, index);
            APASS_LOG_DEBUG_F(Elements::Operation, "consumer preIdx: %d, op: %s", index, GetOpInfo(op).c_str());
            if (FindConsumerList(index, preOpList, curOpList) != SUCCESS) {
                APASS_LOG_WARN_F(Elements::Operation, "FindConsumerList failed");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

// 将 consumersGroup 和其前序依赖按原有顺序放入 preOpList
Status OptimizeSort::UpdateOOperandPreDependence(size_t startIndex, std::vector<Operation*> &curOpList,
    std::vector<Operation*> consumersGroup) {
    // curOpList 中向后找
    std::vector<size_t> preOpList;
    size_t index = startIndex;
    while (index < curOpList.size()) {
        if (std::find(consumersGroup.begin(), consumersGroup.end(), curOpList[index]) != consumersGroup.end()) {
            APASS_LOG_DEBUG_F(Elements::Operation, "consumer Idx: %d", index);
            if (FindConsumerList(index, preOpList, curOpList) != SUCCESS) {
                APASS_LOG_WARN_F(Elements::Operation, "FindConsumerList failed");
                return FAILED;
            }
        }
        index++;
    }
    ReorderOp(preOpList, curOpList, startIndex);
    return SUCCESS;
}

void OptimizeSort::UpdateVisitedByIndex(const std::vector<Operation*> &curOpList, size_t startIndex,
    bool startIndexExecuted) {
    for (size_t i = 0; i < curOpList.size(); i++) {
        visitedOp[curOpList[i]] = startIndexExecuted ? (i <= startIndex) : (i < startIndex);
    }
}

OptimizeSort::Checkpoint OptimizeSort::MakeCheckpoint(size_t startIndex, const std::vector<Operation*> &curOpList,
    const std::map<MemoryType, int64_t> &curMemoryMap, bool startIndexExecuted) {
    Checkpoint checkpoint;
    checkpoint.startIndex = startIndex;
    checkpoint.startIndexExecuted = startIndexExecuted;
    checkpoint.opList = curOpList;
    checkpoint.memoryMap = curMemoryMap;
    checkpoint.bufRefCount = bufRefCount;
    checkpoint.opDeltas = opDeltas;
    return checkpoint;
}

void OptimizeSort::RestoreCheckpoint(const Checkpoint &checkpoint, size_t &startIndex, std::vector<Operation*> &curOpList,
    std::map<MemoryType, int64_t> &curMemoryMap) {
    startIndex = checkpoint.startIndex;
    curOpList = checkpoint.opList;
    curMemoryMap = checkpoint.memoryMap;
    bufRefCount = checkpoint.bufRefCount;
    opDeltas = checkpoint.opDeltas;
    UpdateVisitedByIndex(curOpList, startIndex, checkpoint.startIndexExecuted);
}

Status OptimizeSort::UndoOpEffects(Operation* op, std::map<MemoryType, int64_t> &curMemoryMap) {
    if (!visitedOp[op]) {
        return SUCCESS;
    }
    auto it = opDeltas.find(op);
    if (it == opDeltas.end()) {
        APASS_LOG_ERROR_F(Elements::Operation, "UndoOpEffects cannot find delta for %s", GetOpInfo(op).c_str());
        return FAILED;
    }
    const OpDelta &delta = it->second;
    for (const auto &freed : delta.freedBuffers) {
        if (ModifyBuffer(curMemoryMap, freed.memType, freed.size, true) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "UndoOpEffects failed to restore memory for tensor[%d]", freed.memId);
            return FAILED;
        }
    }
    for (auto tensor : GetInOutOperand(op)) {
        int memId = tensor->memoryrange.memId;
        auto iter = bufRefCount.find(memId);
        if (iter == bufRefCount.end()) {
            APASS_LOG_ERROR_F(Elements::Tensor, "UndoOpEffects cannot find Tensor[%d] refcount.", memId);
            return FAILED;
        }
        iter->second++;
    }
    if (delta.allocApplied) {
        if (ModifyBuffer(curMemoryMap, delta.allocMemType, delta.allocSize, false) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "UndoOpEffects failed to revert alloc for %s", GetOpInfo(op).c_str());
            return FAILED;
        }
    }
    visitedOp[op] = false;
    return SUCCESS;
}

Status OptimizeSort::RebuildStateToIndex(const std::vector<Operation*> &curOpList, size_t endIndex,
    std::map<MemoryType, int64_t> &curMemoryMap) {
    ScheduleBase::operations = curOpList;
    InitBufRefCount();
    curMemoryMap = {{MemoryType::MEM_L0A, 0}, {MemoryType::MEM_L0B, 0}, {MemoryType::MEM_L0C, 0}};
    for (size_t i = 0; i <= endIndex; i++) {
        Operation* op = curOpList[i];
        OpDelta delta;
        if (IsOpAlloc(op)) {
            auto tensor = op->GetOutputOperand(0);
            delta.allocApplied = true;
            delta.allocMemType = tensor->GetMemoryTypeOriginal();
            delta.allocSize = ShapeCeilAlign(tensor->GetShape(), tensor->Datatype());
            if (ModifyBuffer(curMemoryMap, delta.allocMemType, delta.allocSize, true) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "RebuildStateToIndex alloc failed for %s", GetOpInfo(op).c_str());
                return FAILED;
            }
        }
        std::vector<FreedBuffer> freedBuffers;
        if (RetireOpBuffer(curMemoryMap, op, freedBuffers) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "RebuildStateToIndex retire failed for %s", GetOpInfo(op).c_str());
            return FAILED;
        }
        delta.freedBuffers = std::move(freedBuffers);
        opDeltas[op] = std::move(delta);
        visitedOp[op] = true;
    }
    if (endIndex + 1 < curOpList.size()) {
        for (size_t i = endIndex + 1; i < curOpList.size(); i++) {
            visitedOp[curOpList[i]] = false;
        }
    }
    return SUCCESS;
}

// 找未被执行的 consumer
void OptimizeSort::GetConsumerGroup(std::set<Operation*> consumers, std::vector<Operation*> &consumersGroup) {
    for (auto op : consumers) {
        APASS_LOG_DEBUG_F(Elements::Operation, "consumer: %s", GetOpInfo(op).c_str());
        if (!visitedOp[op]) {
            consumersGroup.push_back(op);
            APASS_LOG_DEBUG_F(Elements::Operation, "unvisited consumer: %s", GetOpInfo(op).c_str());
        }
    }
}

void OptimizeSort::GetStackTop(size_t &startIndex, std::vector<Operation*> &curOpList,
    std::map<MemoryType, int64_t> &curMemoryMap) {
    auto topFrame = backtraceStack.top();
    backtraceStack.pop();
    RestoreCheckpoint(topFrame.checkpoint, startIndex, curOpList, curMemoryMap);
}

Status OptimizeSort::BacktraceOnMemoryExceeded(size_t &startIndex,
    std::vector<Operation*> &curOpList, std::map<MemoryType, int64_t> &curMemoryMap) {
    APASS_LOG_DEBUG_F(Elements::Tensor, "=====> Start Backtrace.");
    MemoryType memType = curOpList[startIndex]->GetOutputOperand(0)->GetMemoryTypeOriginal();
    std::vector<Operation*> consumersGroup;
    while (startIndex < curOpList.size() && startIndex > 0) {
        Operation* prevOp = curOpList[startIndex];
        if (UndoOpEffects(prevOp, curMemoryMap) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Backtrace undo failed for %s", GetOpInfo(prevOp).c_str());
            return FAILED;
        }
        startIndex--;
        auto op = curOpList[startIndex];
        if (!backtraceStack.empty() && backtraceStack.top().op == op) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Having traversed %s, the stack needs to be popped", GetOpInfo(op).c_str());
            break;
        }
        if (op->GetOutputOperand(0)->GetMemoryTypeOriginal() != memType || IsOpAlloc(op)) {
            continue;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "===>start to find unvisited consumer");
        APASS_LOG_DEBUG_F(Elements::Operation, "current index： %d", startIndex);
        consumersGroup.clear();
        GetConsumerGroup(outGraph[op], consumersGroup);
        if (consumersGroup.empty()) {
            continue;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "push %s to stack", GetOpInfo(op).c_str());
        backtraceStack.push({op, memType, MakeCheckpoint(startIndex, curOpList, curMemoryMap)});
        if (UpdateOOperandPreDependence(startIndex, curOpList, consumersGroup) != SUCCESS) {
            backtraceStack.pop();
            APASS_LOG_DEBUG_F(Elements::Operation, "UpdateOOperandPreDependence failed.");
            continue;
        }
        startIndex++;
        APASS_LOG_DEBUG_F(Elements::Operation, "Backtrace==>change startIndex: %d", startIndex);
        return SUCCESS;
    }
    if (backtraceStack.empty()) {
        APASS_LOG_WARN_F(Elements::Tensor, "Stack is empty. Start to rollback.");
        return FAILED;
    }
    GetStackTop(startIndex, curOpList, curMemoryMap);
    APASS_LOG_DEBUG_F(Elements::Operation, "pop %s from stack", GetOpInfo(curOpList[startIndex]).c_str());
    if (BacktraceOnMemoryExceeded(startIndex, curOpList, curMemoryMap) != SUCCESS) {
        APASS_LOG_WARN_F(Elements::Tensor, "BacktraceOnMemoryExceeded Failed");
        return FAILED;
    }
    return SUCCESS;
}

// 计算 tensor 对应的 memType （只对 L0C L0A L0B 进行内存处理） 是否已满
bool OptimizeSort::IsBufferFull(std::map<MemoryType, int64_t> curMemoryMap, MemoryType memType, int64_t size) {
    if (memType != MemoryType::MEM_L0A && memType != MemoryType::MEM_L0B && memType != MemoryType::MEM_L0C) {
        APASS_LOG_DEBUG_F(Elements::Operation, "MemoryType is not L0A, L0B, or L0C.");
        return false;
    }
    if (curMemoryMap[memType] + size > localMemSize[memType]) {
        APASS_LOG_DEBUG_F(Elements::Operation, "The %d-memType memory is full, current memory: %d, memory to add: %d",
            memType, curMemoryMap[memType], size);
        return true;
    }
    return false;
}

// 修改内存
Status OptimizeSort::ModifyBuffer(std::map<MemoryType, int64_t> &curMemoryMap, MemoryType memType, int64_t size, bool isAdd) {
    if (memType != MemoryType::MEM_L0A && memType != MemoryType::MEM_L0B && memType != MemoryType::MEM_L0C) {
        APASS_LOG_DEBUG_F(Elements::Operation, "MemoryType is not L0A, L0B, or L0C.");
        return SUCCESS;
    }
    if (isAdd) {
        if (curMemoryMap[memType] + size > localMemSize[memType]) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Failed to increase memory");
            return FAILED;
        }
        curMemoryMap[memType] = curMemoryMap[memType] + size;
        APASS_LOG_DEBUG_F(Elements::Operation, "Increase %d-memType memory, size: %d, total memory %d", memType, size, curMemoryMap[memType]);
        return SUCCESS;
    }
    if (curMemoryMap[memType] - size < 0) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Failed to reduce memory");
        return FAILED;
    }
    curMemoryMap[memType] = curMemoryMap[memType] - size;
    APASS_LOG_DEBUG_F(Elements::Operation, "Reduce %d-memType memory, size: %d, total memory %d", memType, size, curMemoryMap[memType]);
    return SUCCESS;
}

// 释放内存 notTaskOp需要减去bufRefCount
Status OptimizeSort::RetireOpBuffer(std::map<MemoryType, int64_t> &curMemoryMap, Operation* op,
    std::vector<FreedBuffer> &freedBuffers) {
    freedBuffers.clear();
    for (auto tensor : GetInOutOperand(op)) {
        auto memId = tensor->memoryrange.memId;
        if (DelBufRefCount(memId) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "DelBufRefCount tensor[%d] failed.", memId);
            return FAILED;
        }
        if (bufRefCount[memId] == 0) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Start to free memory:");
            int64_t size = ShapeCeilAlign(tensor->GetShape(), tensor->Datatype());
            if (ModifyBuffer(curMemoryMap, tensor->GetMemoryTypeOriginal(), size, false) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Free tensor[%d] failed.", memId);
                return FAILED;
            }
            freedBuffers.push_back({tensor->GetMemoryTypeOriginal(), size, memId});
        }
    }
    return SUCCESS;
}

Status OptimizeSort::AllocExecute(Operation* op, std::vector<Operation*> &curOpList,
    std::map<MemoryType, int64_t> &curMemoryMap, size_t &startIndex, bool &isContinue) {
    APASS_LOG_DEBUG_F(Elements::Operation, "alloc op: %s", GetOpInfo(op).c_str());
    auto tensor = op->GetOutputOperand(0);
    if (IsBufferFull(curMemoryMap, tensor->GetMemoryTypeOriginal(), ShapeCeilAlign(tensor->GetShape(), tensor->Datatype()))) {
        APASS_LOG_DEBUG_F(Elements::Operation, "The memory of %s needs to be released", std::to_string(tensor->GetMemoryTypeOriginal()).c_str());
        backTraceOp = curOpList[startIndex];
        backTraceCheckpoint = MakeCheckpoint(startIndex, curOpList, curMemoryMap, false);
        APASS_LOG_DEBUG_F(Elements::Operation, "backTraceOp: %s, backTraceIndex: %d, memType: %d",
            GetOpInfo(backTraceOp).c_str(), backTraceCheckpoint.startIndex,
            backTraceOp->GetOutputOperand(0)->GetMemoryTypeOriginal());
        APASS_LOG_DEBUG_F(Elements::Operation, "=====> Need backtrace.");
        if (BacktraceOnMemoryExceeded(startIndex, curOpList, curMemoryMap) != SUCCESS) {
            if (RollBack(startIndex, curOpList, curMemoryMap) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "AllocExecute failed.");
                return FAILED;
            }
            isContinue = true;
            return SUCCESS;
        }
        isContinue = true;
        return SUCCESS;
    }
    return SUCCESS;
}

Status OptimizeSort::OpListExecute(std::vector<Operation*> &curOpList,
    std::map<MemoryType, int64_t> &curMemoryMap, size_t &startIndex) {
    APASS_LOG_DEBUG_F(Elements::Operation, "===>Start opListExecute, startIndex: %d", startIndex);
    if (curOpList.empty()) {
        curOpList = operations;
    }
    while (startIndex < curOpList.size()) {
        auto op = curOpList[startIndex];
        APASS_LOG_DEBUG_F(Elements::Operation, "execute op: %s, index: %d", GetOpInfo(op).c_str(), startIndex);
        OpDelta delta;
        if (IsOpAlloc(op)) {
            bool isContinue = false;
            if (AllocExecute(op, curOpList, curMemoryMap,  startIndex, isContinue) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "AllocExecute failed.");
                return FAILED;
            }
            if (isContinue) {
                return SUCCESS;
            }
            auto tensor = op->GetOutputOperand(0);
            delta.allocApplied = true;
            delta.allocMemType = tensor->GetMemoryTypeOriginal();
            delta.allocSize = ShapeCeilAlign(tensor->GetShape(), tensor->Datatype());
            if (ModifyBuffer(curMemoryMap, delta.allocMemType, delta.allocSize, true) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Allocate tensor[%u] failed.", tensor->GetMagic());
                return FAILED;
            }
        }
        visitedOp[op] = true;
        std::vector<FreedBuffer> freedBuffers;
        if (RetireOpBuffer(curMemoryMap, op, freedBuffers) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "RetireOp failed! %s", GetOpInfo(op).c_str());
            return FAILED;
        }
        delta.freedBuffers = std::move(freedBuffers);
        opDeltas[op] = std::move(delta);
        startIndex += 1;
    }
    opFinish = true;
    return SUCCESS;
}

Status OptimizeSort::ExecuteOp() {
    std::vector<Operation*> curOpList;
    std::map<MemoryType, int64_t> curMemoryMap = {{MemoryType::MEM_L0A, 0}, {MemoryType::MEM_L0B, 0},
        {MemoryType::MEM_L0C, 0}};
    size_t startIndex{0};
    opDeltas.clear();
    while (!backtraceStack.empty()) {
        backtraceStack.pop();
    }
    for (auto &op : operations) {
        visitedOp[op] = false;
    }
    while(!opFinish) {
        if (OpListExecute(curOpList, curMemoryMap, startIndex) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "OpListExecute failed.");
            return FAILED;
        }
    }
    operations = curOpList;
    return SUCCESS;
}

Status OptimizeSort::SortOps() {
    APASS_LOG_INFO_F(Elements::Operation, "====>start SortOps");
    Init(operations);
    if (operations.empty()) {
        return SUCCESS;
    }
    std::vector<Operation*> allocOps;
    std::vector<Operation*> normalOps;
    for (auto& op : operations) {
        if (op->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            allocOps.emplace_back(op);
            continue;
        }
        normalOps.emplace_back(op);
    }
    std::vector<Operation *> newOperations;
    std::reverse(allocOps.begin(), allocOps.end());
    newOperations.swap(allocOps);
    newOperations.insert(newOperations.end(), normalOps.begin(), normalOps.end());
    operations = newOperations;
    std::string sortMethodStr;
    std::string funcName = function_.GetMagicName();

    sortMethodStr = function_.paramConfigs_.OoOPreScheduleMethod;
    if (sortMethodStr == "PriorDFS") {
        std::unordered_map<Opcode, int> preNodePriority = {
            // ALLOC 节点优先级最高，因为一个节点的前序ALLOC节点要在最靠近该节点的地方访问。
            {Opcode::OP_UB_ALLOC, 0}, {Opcode::OP_L1_ALLOC, 0}, {Opcode::OP_L0A_ALLOC, 0}, {Opcode::OP_L0B_ALLOC, 0},
            {Opcode::OP_L0C_ALLOC, 0}, {Opcode::OP_BT_ALLOC, 0}, {Opcode::OP_FIX_ALLOC, 0},
            // 其次是L0级数据搬运Op。
            {Opcode::OP_L1_TO_L0A, 1}, {Opcode::OP_L1_TO_L0B, 1}, {Opcode::OP_L1_TO_L0_AT, 1},
            {Opcode::OP_L1_TO_L0_BT, 1}, {Opcode::OP_L1_TO_FIX, 1}, {Opcode::OP_L1_TO_FIX_QUANT_PRE, 1},
            {Opcode::OP_L1_TO_FIX_RELU_PRE, 1}, {Opcode::OP_L1_TO_FIX_RELU_POST, 1},
            {Opcode::OP_L1_TO_FIX_QUANT_POST, 1}, {Opcode::OP_L1_TO_FIX_ELT_ANTIQ, 1},
            {Opcode::OP_L1_TO_FIX_MTE2_ANTIQ, 1}, {Opcode::OP_L1_TO_BT, 1},
            // 再其次是L1级数据搬运Op。
            {Opcode::OP_COPY_IN, 2}, {Opcode::OP_UB_COPY_IN, 2}, {Opcode::OP_L1_COPY_IN, 2},
            {Opcode::OP_L1_COPY_IN_FRACTAL_Z, 2}, {Opcode::OP_L1_COPY_UB, 2},
            {Opcode::OP_L0C_COPY_UB, 2}, {Opcode::OP_UB_COPY_L1, 2},
            // 最后访问其它计算节点（其它节点默认的优先级为10）。
        };
        if (PriorDFS(preNodePriority) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PriorDFS failed.");
            return FAILED;
        }
        if (ExecuteOp() != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "ExecuteOp failed.");
            return FAILED;
        }
    } else {
        APASS_LOG_ERROR_F(Elements::Operation, "PreSchedule method not recognized.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Operation, "====>end SortOps");
    return SUCCESS;
}

} // namespace npu::tile_fwk
