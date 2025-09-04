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
#include "passes/pass_check/schedule_ooo_checker.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/execute_graph_pass/buffer_pool.h"
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
        std::push_heap(queue.begin(), queue.end(),
            [](std::pair<IssueEntryPtr, int> &a, std::pair<IssueEntryPtr, int> &b) { return a.second > b.second; });
    }

    void InsertReloadAlloc(IssueEntryPtr op, IssueEntryPtr spillIssue) {
        IssueEntryPtr firstSuccIssue = nullptr;
        for (auto& succ : spillIssue->successors) {
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

enum class SortOpMethod : int
{
    PriorDFS =  0,
    LayerBasedDFS
};

struct SpillInfo {
    int spillMemId;
    MemoryType bufferType;
    IssueEntryPtr spillIssue;
    IssueEntryPtr allocIssue;
    LogicalTensorPtr spillTensor;
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
    uint64_t numTotalIssues{0};

    Status Init(const std::vector<Operation *> &operations);
    Status InitDependencies();
    Status InitLocalBuffer(LogicalTensorPtr oOperand, int memId);
    Status UpdateIssueIOTensor(const IssueEntryPtr &issue, std::map<int, IssueEntryPtr> &tensorAllocMap);
    Status CheckAllocIssue();
    void InitIssueQueuesAndBufferManager();
    Status CheckOpBufferSize(Operation *op);
    void AddDependencies(IssueEntryPtr issue, std::map<int, IssueEntryPtr> lastWriteOpMap, LogicalTensorPtr tensor);

    Status SortOps(SortOpMethod sortMethod = SortOpMethod::PriorDFS);
    Status PriorDFS(std::unordered_map<Opcode, int> preNodePriority);
    Status LayerBasedDFS(int layerDepth);
    bool CheckIsReady(const IssueEntryPtr &curIssue, std::map<IssueEntryPtr, bool>& visited);
    void UpdateIssueEntriesWithUnvisitedPredecessors(const IssueEntryPtr &curIssue, std::map<IssueEntryPtr, bool>& visited, std::unordered_map<Opcode, int> preNodePriority, std::deque<IssueEntryPtr> &queue);
    void DFSFromSingleNode(IssueEntryPtr issue, std::map<IssueEntryPtr, bool>& visited,
        std::vector<IssueEntryPtr>& newIssueEntries, std::unordered_map<Opcode, int> preNodePriority);
    IssueEntryPtr FindNodeMinNumUnvisitedPreNode(
        std::map<IssueEntryPtr, bool> visited, std::vector<IssueEntryPtr> outNodeQueue);
    int GetNumUnvisitPreNode(IssueEntryPtr issue, std::map<IssueEntryPtr, bool>& visited);

    Status UpdateBufNextUseTime(int currPc, int memId, std::unordered_map<int, size_t> &nextUseTimeCache, std::vector<size_t> &bufNextUseTime);
    Status UpdateGroupNextUseTime(int currPc, const std::vector<int> &group, std::unordered_map<int, size_t> &nextUseTimeCache, std::vector<int> &groupNextUseTime);
    Status SelectSpillBufferGroup(std::vector<std::vector<int>>& groups, int currPc, std::vector<int> &spillGroup);
    Status SpillBufferGroup(Function &function, size_t &pcIdx, LocalBufferPtr allocBuffer, std::vector<std::vector<int>> &canSpillGroups);
    Status GenSpillOp(Function &function, LocalBufferPtr allocBuffer, size_t &pcIdx);
    Status GetBufTimes(int spillMemId, size_t &bufNextUseTime, size_t &bufLastUseTime, size_t &bufLastWriteTime);
    bool GetBufNextUseTime(int curMemId, size_t& nextUseTime);
    bool GetBufLastUseTime(int curMemId, size_t& lastUseTime);
    bool GetBufLastWriteTime(int curMemId, size_t& lastWriteTime);
    Status SpillBuffer(Function &function, int spillMemId, size_t &pcIdx, LocalBufferPtr allocBuffer);
    Status CreateSpillCopyout(Function &func, IssueEntryPtr spillIssue, LogicalTensorPtr spillTensor, int spillMemId,
        IssueEntryPtr &spillCopyout);
    Status CreateSpillReloadIssue(Function &func, LogicalTensorPtr spillOutTensor, LogicalTensorPtr spillTensor,
        IssueEntryPtr &spillIssue, std::pair<IssueEntryPtr, IssueEntryPtr> &reloadIssues);
    Status UpdateOldestBuffer(IssueEntryPtr allocIssue, MemoryType bufferType, int& memId, IssueEntryPtr &spillIssue);
    Status GetOldestBuffer(IssueEntryPtr allocIssue, MemoryType bufferType, int& memId, IssueEntryPtr &spillIssue);
    Status GetddrTensor(Function &function, const SpillInfo &spillInfo, LogicalTensorPtr &ddrTensor, std::vector<Operation *> &newOperations);
    Status GenBufferSpill(
        Function &function, IssueEntryPtr allocIssue, MemoryType bufferType, std::vector<Operation *> &newOperations);
    Status ProcessAlloc(Function &function, IssueEntryPtr &issue, size_t &pcIdx);
    Status ProcessRetire(IssueEntryPtr &issue);
    Status ProcessAllocAndRetire(Function &function);
    Status GenSpillSchedule(Function &function);
    void FindFilterLtags(IssueEntryPtr allocIssue, std::set<IssueEntryPtr> &filterLtags);

    Status UpdateRemainOpBufId(int oldMemId, int newMemId);
    void UpdateOpAttr(Operation &op, int opLatency, LogicalTensorPtr spillTensor, std::vector<int64_t> offset,
        IssueEntryPtr spillIssue);
    Status UpdateTensorAttr(LogicalTensorPtr tensor, MemoryType memType, LogicalTensorPtr spillTensor, int spillMemId);
    Status GetSpillTensor(IssueEntryPtr spillIssue, int spillMemId, LogicalTensorPtr &spillTensor);
    Status UpdateReloadIssueInfo(IssueEntryPtr reloadAlloc, IssueEntryPtr reloadCopyin, IssueEntryPtr spillIssue,
        int spillMemId, int bufNextUseTime);

    Status ExecuteSortOps(Function &function);
    Status Initiate(const std::vector<Operation *> &operations);
    Status AdjustMemType(Function &func, int nextCycle, std::vector<Operation *> &newOperations);
    Status ProcessScheduleStage(uint64_t &commitCnt, int &nextCycle, std::vector<Operation *> &newOperations);
    Status ProcessTillRetired(Function &func, std::vector<Operation *> &newOperations);
    Status UpdateIssues();
    Status ScheduleMainLoop(Function &func, std::vector<Operation *> &newOperations);
    void LaunchReadyIssue();
    Status AllocateTensor(MemoryType memType, IssueEntryPtr &issue, std::vector<Operation *>& newOperations);
    Status ExecuteAllocOp(MemoryType memType, IssueQueue &pipe, std::vector<Operation *>& newOperations, uint64_t& commitCnt);
    Status BufferAllocStage(std::vector<Operation *>& newOperations, uint64_t& commitCnt);
    Status ReallocOutTensor(IssueEntryPtr &issue);
    Status LaunchIssueStage(int& nextCycle, std::vector<Operation *> &newOperations);
    Status RetireIssueStage(uint64_t& commitCnt, int& nextCycle);
    Status RetireOp(IssueEntryPtr &issue);
    Status AwakeSucc(IssueEntryPtr &issue);
    Status RetireOpAndAwakeSucc(IssueEntryPtr issue, uint64_t& commitCnt);

    size_t ShapeCeilAlign(std::vector<int64_t> shape, DataType dtype);
    Status DelBufRefCount(const int memId);
    void PrintDependenciesAndRelations();
    void UpdateBufferUsage(MemoryType bufferType, int memId, bool isFree);
    OoOSchedulerCheck::SpillInfo RecordSpillInfo(MemoryType bufferType, int memId, LocalBufferPtr allocIssue, LogicalTensorPtr spillOutTensor, bool needCopyOut);

public:
    Status Schedule(
        Function &function, const std::vector<Operation *> &operations, std::vector<Operation *> &newOperations);

    int GetSubgraphID() { return subGraphID; }
    int workspaceOffset{0};
    int clock{0};
    OoOSchedulerCheck oooCheck;
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
    void DoHealthCheckAfter(Function &function, const std::string &folderPath) override;
    std::vector<Function *> oriFunctions;
    std::map<uint64_t, OoOScheduler> schedulerMap;
    OoOScheduleChecker checker;
};
} // namespace npu::tile_fwk
#endif // PASS_SCHEDULE_OOO_H