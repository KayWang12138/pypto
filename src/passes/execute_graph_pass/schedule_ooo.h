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
 * \file schedule_ooo.h
 * \brief
 */

#ifndef PASS_SCHEDULE_OOO_H
#define PASS_SCHEDULE_OOO_H

#include <climits>
#include <numeric>
#include <deque>
#include "interface/tensor/logical_tensor.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "passes/pass_interface/pass.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_utils/reschedule_utils.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/execute_graph_pass/buffer_pool.h"

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
    Opcode::OP_RESHAPE, Opcode::OP_VIEW, Opcode::OP_ASSEMBLE, Opcode::OP_COMM_WAIT_FLAG};

struct IssueEntry {
    Operation *tileOp; 
    int execOrder{-1};
    PipeType type{PipeType::PIPE_ALL};
    bool isAlloc{false};
    bool isRetired{false};

    // 当前op的前序op
    std::unordered_set<std::shared_ptr<IssueEntry>> predecessors;

    // 当前op的后序op
    std::unordered_set<std::shared_ptr<IssueEntry>> successors;

    // op计算所需的memId
    std::vector<int> reqMemIds;

    IssueEntry(Operation *op, uint64_t issueId);
    void Clear();
    int GetOOperandIdx(int curMemId);
    void UpdateTensorInput(std::shared_ptr<IssueEntry> &spillSrcIssue, LogicalTensorPtr tensor) const;
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
        std::push_heap(queue.begin(), queue.end(), [] (std::pair<IssueEntryPtr, int>& a, 
            std::pair<IssueEntryPtr, int>& b) {
            return a.second > b.second;
        });
    }
    
    void InsertReloadAlloc(IssueEntryPtr op, IssueEntryPtr spillIssue) {
        IssueEntryPtr firstSuccIssue = nullptr;
        for (auto& succ : spillIssue->successors) {
            if (!succ->isRetired) {
                if (firstSuccIssue == nullptr) {
                    firstSuccIssue = succ;
                } else {
                    firstSuccIssue = firstSuccIssue->execOrder < succ->execOrder ? firstSuccIssue : succ;
                }
            }
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

class OoOScheduler {
private:
    std::vector<IssueEntryPtr> issueEntries;

    std::unordered_map<int, LocalBufferPtr> localBufferMap;
    std::unordered_map<npu::tile_fwk::MemoryType, BufferPool> bufferManagerMap;
    std::unordered_map<int, int> bufRefCount;
    std::unordered_map<MemoryType, std::map<int, IssueEntryPtr>> tensorOccupyMap;
    
    std::map<MemoryType, IssueQueue> allocIssueQueue;
    std::map<PipeType, IssueQueue> issueQueues;
    std::unordered_map<MemoryType, int64_t> inChipMemorySize;

    int subGraphID;
    uint64_t spillIssueCnt{0};
    int workspaceMemId{SYMBOL_STACK_BASE};
    int maxTensorMagic{-1};
    int maxOpMagic{-1};
    int workspaceOffset{0}; 
    uint64_t numTotalIssues{0};
    int clock{0};

    Status Init(const std::vector<Operation *> &operations);
    Status InitDependencies();
    Status InitLocalBuffer(LogicalTensorPtr oOperand, int memId);
    Status CheckAllocIssue();
    void InitIssueQueuesAndBufferManager();
    Status CheckOpBufferSize(Operation *op);
    void AddDependencies(IssueEntryPtr issue, std::map<int, IssueEntryPtr> lastWriteOpMap, 
        LogicalTensorPtr tensor);

    Status SortOps();
    Status PriorDFS(std::unordered_map<Opcode, int> preNodePriority);
    void DFSFromSingleNode(IssueEntryPtr issue, std::map<IssueEntryPtr, bool>& visited,
        std::vector<IssueEntryPtr>& newIssueEntries, std::unordered_map<Opcode, int> preNodePriority);
    IssueEntryPtr FindNodeMinNumUnvisitedPreNode(std::map<IssueEntryPtr, bool> visited, 
        std::vector<IssueEntryPtr> outNodeQueue);
    int GetNumUnvisitPreNode(IssueEntryPtr issue, std::map<IssueEntryPtr, bool>& visited);

    Status SelectSpillBufferGroup(std::vector<std::vector<int>>& groups, int currPc, std::vector<int> &spillGroup);
    Status GenSpillOp(Function &function, LocalBufferPtr allocBuffer, size_t &pcIdx);
    Status GetBufTimes(int spillMemId, size_t &bufNextUseTime, size_t &bufLastUseTime, size_t &bufLastWriteTime);
    bool GetBufNextUseTime(int curMemId, size_t& nextUseTime);
    bool GetBufLastUseTime(int curMemId, size_t& lastUseTime);
    bool GetBufLastWriteTime(int curMemId, size_t& lastWriteTime);
    Status SpillBuffer(Function &function, int spillMemId, size_t &pcIdx);
    Status CreateSpillCopyout(Function &func, IssueEntryPtr spillIssue, LogicalTensorPtr spillTensor, 
        int spillMemId, IssueEntryPtr &spillCopyout);
    Status CreateSpillReloadIssue(Function &func, LogicalTensorPtr spillOutTensor, 
        LogicalTensorPtr spillTensor, IssueEntryPtr &spillIssue, std::pair<IssueEntryPtr, IssueEntryPtr> &reloadIssues);
    Status GetOldestBuffer(IssueEntryPtr allocIssue, MemoryType bufferType, int& memId, IssueEntryPtr &spillIssue);
    Status GenBufferSpill(Function &function, IssueEntryPtr allocIssue, MemoryType bufferType, 
        std::vector<Operation *> &newOperations);
    Status GenSpillSchedule(Function &function);
    void FindFilterLtags(IssueEntryPtr allocIssue, std::set<IssueEntryPtr> &filterLtags);

    Status UpdateRemainOpBufId(int oldMemId, int newMemId);
    void UpdateOpAttr(Operation &op, int opLatency, LogicalTensorPtr spillTensor, std::vector<int> offset, 
    IssueEntryPtr spillIssue);
    Status UpdateTensorAttr(LogicalTensorPtr tensor, MemoryType memType, LogicalTensorPtr spillTensor, int spillMemId);
    Status GetSpillTensor(IssueEntryPtr spillIssue, int spillMemId, LogicalTensorPtr &spillTensor);
    Status UpdateReloadIssueInfo(IssueEntryPtr reloadAlloc, IssueEntryPtr reloadCopyin, IssueEntryPtr spillIssue, 
        int spillMemId, int bufNextUseTime);

    Status ScheduleMainLoop(Function &func, std::vector<Operation *> &newOperations);
    void LaunchReadyIssue();
    Status BufferAllocStage(std::vector<Operation *>& newOperations, uint64_t& commitCnt);
    Status LaunchIssueStage(int& nextCycle, std::vector<Operation *> &newOperations);
    Status RetireIssueStage(uint64_t& commitCnt, int& nextCycle);
    Status RetireOpAndAwakeSucc(IssueEntryPtr issue, uint64_t& commitCnt);

    size_t ShapeCeilAlign(std::vector<int> shape, DataType dtype);
    Status DelBufRefCount(const int memId);
    void PrintDependenciesAndRelations();

public:
    Status Schedule(Function &function, const std::vector<Operation *> &operations, 
        std::vector<Operation *> &newOperations);

    int GetSubgraphID() { 
        return subGraphID; 
    }
};

class OoOSchedule : public Pass {
public:
    OoOSchedule() : Pass("OoOSchedule") {}
    ~OoOSchedule() override {}

private:
    Status RunOnFunction(Function &function) override;
    bool IsAicpuProgram(std::vector<Operation *> opList);
    Status PreCheck(Function &function) override;
    Status PostCheck(Function &function) override;
    bool PreCheckTensorInfo(const int subGraphId, const LogicalTensorPtr tensor);
    bool PreCheckOpInfo(const int subGraphId, const Operation *op);
    bool PostCheckOpMagic(std::set<int> opSet, const Operation *op, const int programIdx);
    bool PostCheckNewOpConnection(const std::vector<Operation *> opListBeforePass,
        const std::vector<int> opMagicListBeforePass, const Operation *op, const int programIdx);
    bool PostCheckSpecialOp(const Operation *op, const int subGraphId);
    bool PostCheckTensorMagic(std::set<int> tensorSet, const LogicalTensorPtr tensor, const int programIdx);
    bool PostCheckLocalTensor(const LogicalTensorPtr tensor, const int subGraphId, const int programIdx);
    bool PostCheckGlobalTensor(const LogicalTensorPtr tensor, const int subGraphId, const int programIdx);
    bool PostCheckDynValidShape(const LogicalTensorPtr tensor, const int programIdx);
    bool PostCheckNewTensor(const int subGraphId, std::pair<const int, Function*> program, const int programIdx);
    std::vector<std::unordered_set<LogicalTensorPtr>> tensorListBeforePass;
    std::vector<std::unordered_set<LogicalTensorPtr>> tensorListAfterPass;
    std::vector<Function *> oriFunctions;
};
} // namespace npu::tile_fwk
#endif // PASS_SCHEDULE_OOO_H