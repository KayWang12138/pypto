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
 * \file scheduler.h
 * \brief
 */

#ifndef PASS_SCHEDULER_H
#define PASS_SCHEDULER_H

#include <climits>
#include "passes/pass_interface/pass.h"
#include "passes/pass_utils/reschedule_utils.h"
#include "passes/pass_check/schedule_ooo_checker.h"
#include "passes/block_graph_pass/schedule_ooo/buffer_pool.h"
#include "passes/statistics/ooo_schedule_statistic.h"

namespace npu::tile_fwk {

inline int BytesPerElement(DataType dataType) {
    return BytesOf(dataType);
}

inline int FloorAlign(int a, int b) {
    return (a / b) * b;
}

inline uint64_t CeilAlign(uint64_t a, int b) {
    return ((a + b - 1) / b) * b;
}

using LocalBufferPtr = std::shared_ptr<LocalBuffer>;

const std::unordered_set<Opcode> USE_LESS_OPS = {
    Opcode::OP_RESHAPE, Opcode::OP_VIEW, Opcode::OP_ASSEMBLE, Opcode::OP_COMM_WAIT_FLAG, Opcode::OP_SHMEM_WAIT_UNTIL};

struct IssueEntry {
    Operation &tileOp;
    int id{-1};
    int execOrder{-1};
    PipeType type{PipeType::PIPE_ALL};
    bool isAlloc{false};
    bool isRetired{false};

    // 当前op的前序op
    std::unordered_set<int> predecessors;

    // 当前op的后序op
    std::unordered_set<int> successors;

    // op计算所需的memId
    std::vector<int> reqMemIds;

    IssueEntry(Operation &op, uint64_t issueId);
    void Clear();
    int GetOOperandIdx(int curMemId);
    void UpdateTensorInput(std::shared_ptr<IssueEntry> &spillSrcIssue, LogicalTensorPtr tensor) const;
    const char* GetOpInfo();
};

using IssueEntryPtr = std::shared_ptr<IssueEntry>;

struct IssueQueue {
    bool busy{false};
    IssueEntryPtr curIssue = nullptr;
    int curOpRetireCycle{-1};
    std::vector<std::pair<IssueEntryPtr, int>> queue;

    IssueQueue() {}
    ~IssueQueue() {}

    void Insert(IssueEntryPtr op, int priority) {
        // priority越小，表示优先级越高。
        queue.push_back(std::make_pair(op, priority));
        std::push_heap(queue.begin(), queue.end(),
            [](std::pair<IssueEntryPtr, int> &a, std::pair<IssueEntryPtr, int> &b) { return a.second > b.second; });
    }

    void InsertReloadAlloc(IssueEntryPtr op, IssueEntryPtr spillIssue,
        std::unordered_map<int, IssueEntryPtr> issueEntryMap) {
        IssueEntryPtr firstSuccIssue = nullptr;
        for (auto& succId : spillIssue->successors) {
            auto succ = issueEntryMap[succId];
            if (succ->isRetired) {
                continue;
            }
            if (firstSuccIssue == nullptr) {
                firstSuccIssue = succ;
                continue;
            }
            firstSuccIssue = firstSuccIssue->execOrder < succ->execOrder ? firstSuccIssue : succ;
        }
        Insert(op, firstSuccIssue->execOrder);
        op->execOrder = firstSuccIssue->execOrder;
    }

    bool Empty() {
        return queue.size() == 0;
    }

    IssueEntryPtr Front() {
        return queue[0].first;
    }

    IssueEntryPtr PopFront() {
        std::pop_heap(queue.begin(), queue.end(), [](std::pair<IssueEntryPtr, int>& a,
        std::pair<IssueEntryPtr, int>& b){
            return a.second > b.second;
        });

        IssueEntryPtr op = queue.back().first;
        queue.pop_back();
        return op;
    }
};

struct SpillInfo {
    int spillMemId_;
    IssueEntryPtr spillIssue_;
    LogicalTensorPtr spillTensor_;
    LogicalTensorPtr ddrTensor_;
    SpillInfo(int id, IssueEntryPtr issue, LogicalTensorPtr tensor, LogicalTensorPtr ddrTensor) 
        : spillMemId_(id), spillIssue_(issue), spillTensor_(tensor), ddrTensor_(ddrTensor) {}
};

class OoOScheduler {
private:
    std::vector<IssueEntryPtr> issueEntries;
    std::unordered_map<int, IssueEntryPtr> issueEntryMap;

    std::unordered_map<int, LocalBufferPtr> localBufferMap;
    std::unordered_map<npu::tile_fwk::MemoryType, BufferPool> bufferManagerMap;
    std::unordered_map<int, int> bufRefCount;
    std::unordered_map<MemoryType, std::map<int, IssueEntryPtr>> tensorOccupyMap;

    std::map<MemoryType, IssueQueue> allocIssueQueue;
    std::map<PipeType, IssueQueue> issueQueues;
    std::unordered_map<MemoryType, int64_t> inChipMemorySize;

    int subGraphID;
    Function &function_;
    int issueId{0};
    uint64_t spillIssueCnt{0};
    int workspaceMemId{SYMBOL_STACK_BASE};
    int maxTensorMagic{-1};
    int maxOpMagic{-1};
    uint64_t numTotalIssues{0};
    std::vector<Operation *> newOperations_;

    // scheduler
    Status Init(const std::vector<Operation *> &operations);
    Status CheckOpBufferSize(Operation *op);
    void CalcBufferSize(LogicalTensors tensors, std::map<MemoryType, int64_t> &bufferSize, std::set<int> &memIdMap);
    Status InitDependencies();
    void AddDependencies(IssueEntryPtr issue, std::map<int, IssueEntryPtr> lastWriteOpMap, LogicalTensors tensors);
    Status InitLocalBuffer(LogicalTensorPtr oOperand, int memId);
    Status CheckAllocIssue();
    void UpdateAllocMap(IssueEntryPtr issue, std::map<int, IssueEntryPtr> &tensorAllocMap);
    void InitIssueQueuesAndBufferManager();

    Status GenSpillSchedule();
    Status ExecuteAllocIssue(IssueEntryPtr issue, size_t &pcIdx);
    Status RetireIssue(IssueEntryPtr issue);

    Status ScheduleMainLoop();
    void LaunchReadyIssue();
    Status RetireIssueStage(uint64_t& commitCnt, int& nextCycle);
    Status RetireOpAndAwakeSucc(IssueEntryPtr issue, uint64_t& commitCnt);
    Status FreeBuffer(IssueEntryPtr issue);
    Status BufferAllocStage(uint64_t& commitCnt);
    Status ExecuteAllocIssue(uint64_t &commitCnt, MemoryType memType, 
        IssueQueue &pipe);
    Status LaunchIssueStage(int& nextCycle);
    Status AllocTensorMemRange(IssueEntryPtr issue);
    Status SpillOnBlock();
    Status CheckAndUpdateLifecycle();
    
    size_t ShapeCeilAlign(std::vector<int64_t> shape, DataType dtype);
    void PrintOpList(std::vector<Operation *> operations);
    Status DelBufRefCount(const int memId);
    void UpdateBufferUsage(MemoryType bufferType, int memId, bool isFree);
    void PrintDependenciesAndRelations();
    bool GetBufNextUseTime(int curMemId, size_t& nextUseTime);
    bool GetBufLastUseTime(int curMemId, size_t& lastUseTime);
    bool GetBufLastWriteTime(int curMemId, size_t& lastWriteTime);
    void PrintSpillFailedInfo(IssueEntryPtr allocIssue, MemoryType bufferType);
    Status PrintSpillFailedInfo(int currPc);

    // sort ops
    Status SortOps();
    Status PriorDFS(std::unordered_map<Opcode, int> preNodePriority);
    Status DFSFromOutNode(std::vector<IssueEntryPtr> outNodeQueue, std::unordered_map<Opcode, int> preNodePriority,
        std::map<IssueEntryPtr, bool> &visited);
    void DFSFromSingleNode(IssueEntryPtr issue, std::map<IssueEntryPtr, bool>& visited,
        std::vector<IssueEntryPtr>& newIssueEntries, std::unordered_map<Opcode, int> preNodePriority);
    void ForwardDfs(IssueEntryPtr curIssue, std::vector<IssueEntryPtr>& newIssueEntries,
        std::map<IssueEntryPtr, bool>& visited, std::unordered_map<Opcode, int> preNodePriority,
        std::deque<IssueEntryPtr> &queue);
    void QueueNotReadyPreNode(IssueEntryPtr curIssue, std::map<IssueEntryPtr, bool>& visited,
    std::unordered_map<Opcode, int> preNodePriority, std::deque<IssueEntryPtr> &queue);
    int GetNodePriority(std::unordered_map<Opcode, int> preNodePriority, IssueEntryPtr issue);
    IssueEntryPtr FindNodeMinNumUnvisitedPreNode(
        std::map<IssueEntryPtr, bool> visited, std::vector<IssueEntryPtr> outNodeQueue);
    int GetNumUnvisitPreNode(IssueEntryPtr issue, std::map<IssueEntryPtr, bool>& visited);
    void UpdatePreNodeQueue(std::unordered_set<IssueEntryPtr> &curr, std::unordered_set<IssueEntryPtr> &preNodeTotal,
        std::map<IssueEntryPtr, bool>& visited);
    
    Status LayerBasedDFS(int layerDepth);

    // gen spill    
    Status GenSpillOp(LocalBufferPtr allocBuffer, size_t &pcIdx);
    Status SelectSpillBufferGroup(std::vector<std::vector<int>>& groups, int currPc, std::vector<int> &spillGroup);
    Status GetGroupNextUseTime(std::vector<int> group, int currPc, std::vector<int> &groupNextUseTime, 
        std::unordered_map<int, size_t> &nextUseTimeCache);
    bool IsBelongSpillBlackList(size_t bufLastWriteTime, IssueEntryPtr issue);
    void FindFilterLtags(IssueEntryPtr allocIssue, std::set<IssueEntryPtr> &filterLtags);
    Status SpillBuffer(int spillMemId, size_t &pcIdx, LocalBufferPtr allocBuffer);
    Status GetSpillTensor(IssueEntryPtr spillIssue, int spillMemId, LogicalTensorPtr &spillTensor);
    Status SpillOutBuffer(SpillInfo &spillInfo, size_t &bufNextUseTime, size_t &pcIdx, bool isGenSpill);
    Status SpillInBuffer(SpillInfo &spillInfo, MemoryType bufferType, size_t bufNextUseTime, bool isGenSpill);

    Status CreateSpillCopyout(IssueEntryPtr spillIssue, LogicalTensorPtr spillTensor, int spillMemId,
        IssueEntryPtr &spillCopyout);
    Status CreateSpillReloadIssue(LogicalTensorPtr spillOutTensor, LogicalTensorPtr spillTensor,
        IssueEntryPtr &spillIssue, std::pair<IssueEntryPtr, IssueEntryPtr> &reloadIssues);
    Status GetOldestBuffer(IssueEntryPtr allocIssue, MemoryType bufferType, int& memId, IssueEntryPtr &spillIssue);
    Status GenBufferSpill(IssueEntryPtr allocIssue, MemoryType bufferType);
    
    Status UpdateReloadIssueDepend(IssueEntryPtr reloadCopyin, IssueEntryPtr spillIssue, int spillMemId);
    void ReplaceTensorMemId(IssueEntryPtr &issue, int oldMemId, int newMemId);
    Status UpdateRemainOpBufId(int oldMemId, int newMemId);
    void UpdateOpAttr(Operation &op, int opLatency, LogicalTensorPtr spillTensor, std::vector<int64_t> offset,
        IssueEntryPtr spillIssue);
    Status UpdateTensorAttr(LogicalTensorPtr tensor, MemoryType memType, LogicalTensorPtr spillTensor, int spillMemId);
    Status UpdateReloadIssueInfo(IssueEntryPtr reloadAlloc, IssueEntryPtr reloadCopyin, IssueEntryPtr spillIssue,
        int spillMemId, int bufNextUseTime);
    
    OoOSchedulerCheck::SpillInfo RecordSpillInfo(MemoryType bufferType, int memId, LocalBufferPtr allocIssue, LogicalTensorPtr spillOutTensor, bool needCopyOut);

public:
    Status Schedule(const std::vector<Operation *> &operations);
    OoOScheduler(Function &function) : function_(function) {}

    std::vector<Operation *> GetNewOperations() { return newOperations_; }
    int GetSubgraphID() { return subGraphID; }
    int workspaceOffset{0};
    int clock{0};
    OoOSchedulerCheck oooCheck;
};
} // namespace npu::tile_fwk
#endif // PASS_SCHEDULER_H