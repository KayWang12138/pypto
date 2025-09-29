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
 * \file sort_ops.cpp
 * \brief
 */

#include "scheduler.h"

namespace npu::tile_fwk {

Status DFSVisit(std::unordered_set<int> &visited, std::vector<int32_t> &tasks, std::vector<int> &visitEntrySeq,
        std::vector<std::set<int32_t>> &entryInGraph)
{
    while (tasks.size() > 0) {
        int currTask = tasks.back();
        if (visited.count(currTask) > 0) {
            tasks.pop_back();
            continue;
        }
        bool allVisited = true;
        for (auto prevIdx : entryInGraph[currTask]) {
            if (visited.count(prevIdx) == 0) {
                allVisited = false;
                tasks.push_back(prevIdx);
            }
        }
        if (allVisited) {
            visited.insert(currTask);
            visitEntrySeq.push_back(currTask);
            tasks.pop_back();
        }
    }
    return SUCCESS;
}

std::vector<int32_t> GetLayerTasks(std::map<int, std::set<int>> &depthToEntries,
                                   std::vector<std::set<int>> &entryOutGraph, int lastDepth, int currDepth)
{
    std::vector<int32_t> tasks;
    for (int dp = lastDepth; dp < currDepth; dp++) {
        for (int entryIdx : depthToEntries[dp]) {
            if (entryOutGraph[dp].size() == 0) {
                tasks.push_back(entryIdx);
            }
        }
    }
    for (int entryIdx : depthToEntries[currDepth]) {
        tasks.push_back(entryIdx);
    }
    return tasks;
}

Status EntryInOutGraph(std::vector<IssueEntryPtr> &issueEntries, std::vector<std::set<int>> &entryInGraph,
    std::vector<std::set<int>> &entryOutGraph, std::unordered_map<int, IssueEntryPtr> issueEntryMap) {
    entryInGraph.clear();
    entryOutGraph.clear();
    entryInGraph.resize(issueEntries.size());
    entryOutGraph.resize(issueEntries.size());
    std::unordered_map<IssueEntry*, int> entryPtr2Idx;
    for (int ptrIdx = 0; ptrIdx < static_cast<int>(issueEntries.size()); ptrIdx++) {
        entryPtr2Idx[issueEntries[ptrIdx].get()] = ptrIdx;
    }
    for (int ptrIdx = 0; ptrIdx < static_cast<int>(issueEntries.size()); ptrIdx++) {
        for (auto outPtrId : issueEntries[ptrIdx]->successors) {
            auto outPtr = issueEntryMap[outPtrId];
            entryOutGraph[ptrIdx].insert(entryPtr2Idx[outPtr.get()]);
            entryInGraph[entryPtr2Idx[outPtr.get()]].insert(ptrIdx);
        }
    }
    return SUCCESS;
}

Status EntryTopoSort(std::vector<std::set<int32_t>> &entryInGraph, std::vector<std::set<int32_t>> &entryOutGraph,
    std::vector<int32_t> &seqToColor, std::vector<int32_t> &colorToSeq)
{
    seqToColor.clear();
    colorToSeq.resize(entryInGraph.size());
    std::vector<int> inLinkNum(entryInGraph.size());
    std::deque<int> zeroInLinkColor;
    for (size_t i = 0; i < entryInGraph.size(); i++) {
        inLinkNum[i] = entryInGraph[i].size();
        if (inLinkNum[i] == 0) {
            zeroInLinkColor.push_back(i);
        }
    }
    std::vector<int32_t> visitOrder;
    while (zeroInLinkColor.size() > 0) {
        int currColor = zeroInLinkColor.front();
        zeroInLinkColor.pop_front();
        colorToSeq[currColor] = seqToColor.size();
        seqToColor.push_back(currColor);
        for (int consumerColor : entryOutGraph[currColor]) {
            inLinkNum[consumerColor] -= 1;
            if (inLinkNum[consumerColor] == 0) {
                zeroInLinkColor.push_back(consumerColor);
            }
        }
    }
    return SUCCESS;
}

Status OutputFixBasedDepth(std::vector<int32_t> &depth, std::vector<std::set<int32_t>> &entryInGraph,
        std::vector<std::set<int32_t>> &entryOutGraph, std::vector<int32_t> &seqToColor)
{
    for (int idx = static_cast<int>(entryInGraph.size())-1; idx >= 0; idx--) {
        int currEntryIdx = seqToColor[idx];
        if (entryOutGraph[currEntryIdx].size() == 0) {
            continue;
        }
        int minDepth = static_cast<int>(entryInGraph.size()) + 1;
        for (auto succIdx : entryOutGraph[currEntryIdx]) {
            minDepth = minDepth < depth[succIdx] ? minDepth : depth[succIdx];
        }
        depth[currEntryIdx] = minDepth - 1;
    }
    return SUCCESS;
}

Status InputFixBasedDepth(std::vector<int32_t> &depth, std::vector<std::set<int32_t>> &entryInGraph,
        std::vector<std::set<int32_t>> &entryOutGraph, std::vector<int32_t> &seqToColor)
{
    (void)entryOutGraph;
    for (int idx = 0; idx < static_cast<int>(entryInGraph.size()); idx++) {
        int currEntryIdx = seqToColor[idx];
        if (entryInGraph[currEntryIdx].size() == 0) {
            continue;
        }
        int maxDepth = -static_cast<int>(entryInGraph.size()) - 1;
        for (auto predIdx : entryInGraph[currEntryIdx]) {
            maxDepth = maxDepth > depth[predIdx] ? maxDepth : depth[predIdx];
        }
        depth[currEntryIdx] = maxDepth + 1;
    }
    return SUCCESS;
}

Status OoOScheduler::LayerBasedDFS(int layerDepth)
{
    std::vector<std::set<int>> entryInGraph;
    std::vector<std::set<int>> entryOutGraph;
    EntryInOutGraph(issueEntries, entryInGraph, entryOutGraph, issueEntryMap);
    std::vector<int32_t> seqToColor;
    std::vector<int32_t> colorToSeq;
    EntryTopoSort(entryInGraph, entryOutGraph, seqToColor, colorToSeq);
    std::vector<int32_t> depth(entryInGraph.size(), 0);
    OutputFixBasedDepth(depth, entryInGraph, entryOutGraph, seqToColor);
    InputFixBasedDepth(depth, entryInGraph, entryOutGraph, seqToColor);
    std::map<int, std::set<int>> depthToEntries;
    int lowerDepth = static_cast<int>(entryInGraph.size()) + 1;
    int upperDepth = -static_cast<int>(entryInGraph.size()) - 1;
    for (int idx = 0; idx < static_cast<int>(depth.size()); idx++) {
        lowerDepth = lowerDepth < depth[idx] ? lowerDepth : depth[idx];
        upperDepth = upperDepth > depth[idx] ? upperDepth : depth[idx];
        depthToEntries[depth[idx]].insert(idx);
    }
    std::vector<IssueEntryPtr> newIssueEntries;
    std::unordered_set<int> visited;
    std::vector<int> visitEntrySeq;
    int lastDepth = lowerDepth;
    int currDepth = lowerDepth + layerDepth - 1;
    currDepth = currDepth <= upperDepth ? currDepth : upperDepth;
    bool keepVisit = true;
    while (keepVisit) {
        std::vector<int32_t> tasks = GetLayerTasks(depthToEntries, entryOutGraph, lastDepth, currDepth);
        DFSVisit(visited, tasks, visitEntrySeq, entryInGraph);
        if (currDepth == upperDepth) {
            keepVisit = false;
        }
        lastDepth = currDepth;
        currDepth += layerDepth;
        currDepth = currDepth <= upperDepth ? currDepth : upperDepth;
    }
    for (auto idx : visitEntrySeq) {
        newIssueEntries.push_back(issueEntries[idx]);
    }
    issueEntries = newIssueEntries;
    return SUCCESS;
}

void OoOScheduler::UpdatePreNodeQueue(std::unordered_set<IssueEntryPtr> &curr,
    std::unordered_set<IssueEntryPtr> &preNodeTotal, std::map<IssueEntryPtr, bool>& visited) {
    std::unordered_set<IssueEntryPtr> next;
    for (auto& curIssue : curr) {
        for (auto& preIssueId : curIssue->predecessors) {
            auto preIssue = issueEntryMap[preIssueId];
            if (!visited[preIssue] && preNodeTotal.find(preIssue) == preNodeTotal.end()) {
                next.insert(preIssue);
            }
        }
    }
    for (auto& nextIssue : next) {
        preNodeTotal.insert(nextIssue);
    }
    curr.swap(next);
}

int OoOScheduler::GetNumUnvisitPreNode(IssueEntryPtr issue, std::map<IssueEntryPtr, bool>& visited) {
    std::unordered_set<IssueEntryPtr> preNodeTotal;
    std::unordered_set<IssueEntryPtr> curr;
    for (auto& preIssueId : issue->predecessors) {
        auto preIssue = issueEntryMap[preIssueId];
        if (!visited[preIssue]) {
            curr.insert(preIssue);
            preNodeTotal.insert(preIssue);
        }
    }
    while (!curr.empty()) {
        UpdatePreNodeQueue(curr, preNodeTotal, visited);
    }
    return preNodeTotal.size();
}

IssueEntryPtr OoOScheduler::FindNodeMinNumUnvisitedPreNode(
    std::map<IssueEntryPtr, bool> visited, std::vector<IssueEntryPtr> outNodeQueue) {
    IssueEntryPtr res = nullptr;
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

int OoOScheduler::GetNodePriority(std::unordered_map<Opcode, int> preNodePriority, IssueEntryPtr issue) {
    int prior = 10;
    if (preNodePriority.find(issue->tileOp.GetOpcode()) != preNodePriority.end()) {
        prior = preNodePriority[issue->tileOp.GetOpcode()];
    }
    return prior;
}

void OoOScheduler::QueueNotReadyPreNode(IssueEntryPtr curIssue, std::map<IssueEntryPtr, bool>& visited,
    std::unordered_map<Opcode, int> preNodePriority, std::deque<IssueEntryPtr> &queue) {
    std::vector<IssueEntryPtr> notReadyPreNode;
    for (auto& preIssueId : curIssue->predecessors) {
        auto preIssue = issueEntryMap[preIssueId];
        if (!visited[preIssue]) {
            notReadyPreNode.push_back(preIssue);
        }
    }
    std::sort(notReadyPreNode.begin(), notReadyPreNode.end(), [&](IssueEntryPtr a, IssueEntryPtr b) {
        int priorA = GetNodePriority(preNodePriority, a);
        int priorB = GetNodePriority(preNodePriority, b);
        if (priorA != priorB) {
            return priorA < priorB;
        } else {
            return a->execOrder < b->execOrder;
        }
    });
    for (auto& preIssue : notReadyPreNode) {
        queue.push_front(preIssue);
    }
}

void OoOScheduler::ForwardDfs(IssueEntryPtr curIssue, std::vector<IssueEntryPtr>& newIssueEntries,
    std::map<IssueEntryPtr, bool>& visited, std::unordered_map<Opcode, int> preNodePriority,
    std::deque<IssueEntryPtr> &queue) {
    bool ready = true;
    for (auto& preIssueId : curIssue->predecessors) {
        auto preIssue = issueEntryMap[preIssueId];
        if (!visited[preIssue]) {
            ready = false;
            break;
        }
    }

    if (ready) {
        visited[curIssue] = true;
        queue.pop_front();
        newIssueEntries.push_back(curIssue);
    } else {
        QueueNotReadyPreNode(curIssue, visited, preNodePriority, queue);
    }
}

void OoOScheduler::DFSFromSingleNode(IssueEntryPtr issue, std::map<IssueEntryPtr, bool>& visited,
    std::vector<IssueEntryPtr>& newIssueEntries, std::unordered_map<Opcode, int> preNodePriority) {
    if (visited[issue]) {
        return;
    }

    std::deque<IssueEntryPtr> queue = {issue};
    while (!queue.empty()) {
        auto curIssue = queue.front();
        if (visited[curIssue]) {
            queue.pop_front();
            continue;
        }

        ForwardDfs(curIssue, newIssueEntries, visited, preNodePriority, queue);
    }
}

Status OoOScheduler::DFSFromOutNode(std::vector<IssueEntryPtr> outNodeQueue,
    std::unordered_map<Opcode, int> preNodePriority, std::map<IssueEntryPtr, bool> &visited) {
    std::vector<IssueEntryPtr> newIssueEntries;
    if (outNodeQueue.size() != 0) {
       DFSFromSingleNode(outNodeQueue[0], visited, newIssueEntries, preNodePriority);
    } else {
        ALOG_ERROR_F("Subgraph must have operation with outdegree 0.");
        return FAILED;
    }

    for (size_t i = 1; i < outNodeQueue.size(); i++) {
        while (!visited[outNodeQueue[i]]) {
            auto curNode = outNodeQueue[i];
            auto node = FindNodeMinNumUnvisitedPreNode(visited, outNodeQueue);
            if (node == nullptr) {
                ALOG_ERROR_F("FindNodeMinNumUnvisitedPreNode failed.");
                return FAILED;
            }
            DFSFromSingleNode(node, visited, newIssueEntries, preNodePriority);
        }
    }
    issueEntries = newIssueEntries;
    return SUCCESS;
}

Status OoOScheduler::PriorDFS(std::unordered_map<Opcode, int> preNodePriority) {
    std::map<IssueEntryPtr, bool> visited;
    std::vector<IssueEntryPtr> outNodeQueue;
    for (auto &issue : issueEntries) {
        visited[issue] = false;
        if (issue->successors.empty()) {
            outNodeQueue.push_back(issue);
        }
    }

    if (DFSFromOutNode(outNodeQueue, preNodePriority, visited) != SUCCESS) {
        ALOG_ERROR_F("DFSFromOutNode failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::SortOps() {
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
            {Opcode::OP_L1_TO_L0_BT, 1}, {Opcode::OP_FIX_COPY_IN, 1}, {Opcode::OP_FIX_COPY_IN_QUANT_PRE, 1},
            {Opcode::OP_FIX_COPY_IN_RELU_PRE, 1}, {Opcode::OP_FIX_COPY_IN_RELU_POST, 1},
            {Opcode::OP_FIX_COPY_IN_QUANT_POST, 1}, {Opcode::OP_FIX_COPY_IN_ELT_ANTIQ, 1},
            {Opcode::OP_FIX_COPY_IN_MTE2_ANTIQ, 1}, {Opcode::OP_BT_COPY_IN, 1},
            // 再其次是L1级数据搬运Op。
            {Opcode::OP_COPY_IN, 2}, {Opcode::OP_UB_COPY_IN, 2}, {Opcode::OP_L1_COPY_IN, 2},
            {Opcode::OP_L1_COPY_IN_FRACTAL_Z, 2}, {Opcode::OP_L1_COPY_UB, 2},
            {Opcode::OP_L0C_COPY_UB, 2}, {Opcode::OP_UB_COPY_L1, 2},
            // 最后访问其它计算节点（其它节点默认的优先级为10）。
        };
        if (PriorDFS(preNodePriority) != SUCCESS) {
            ALOG_ERROR_F("PriorDFS failed.");
            return FAILED;
        }
    } else if (sortMethodStr == "LayerBasedDFS") {
        const int layerDepth = 10;
        if (LayerBasedDFS(layerDepth) != SUCCESS) {
            ALOG_ERROR_F("LayerBasedDFS failed.");
            return FAILED;
        }
    } else {
        ALOG_ERROR_F("PreSchedule method not recognized.");
        return FAILED;
    }
    return SUCCESS;
}

} // namespace npu::tile_fwk