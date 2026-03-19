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
 * \file scheduler.h
 * \brief
 */

#ifndef PASS_SCHEDULER_H
#define PASS_SCHEDULER_H

#include <climits>
#include "passes/pass_interface/pass.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_utils/reschedule_utils.h"
#include "passes/pass_check/schedule_ooo_checker.h"
#include "passes/block_graph_pass/schedule_ooo/buffer_pool.h"
#include "passes/statistics/ooo_schedule_statistic.h"

namespace npu::tile_fwk {

inline int BytesPerElement(DataType dataType) {
    return BytesOf(dataType);
}

inline uint64_t CeilAlign(uint64_t a, int b) {
    return ((a + b - 1) / b) * b;
}

inline bool IsViewOp(const Operation& op) {
    const auto opc = op.GetOpcode();
    return opc == Opcode::OP_VIEW || opc == Opcode::OP_VIEW_TYPE;
}

using LocalBufferPtr = std::shared_ptr<LocalBuffer>;

const std::unordered_set<Opcode> USE_LESS_OPS = {
    Opcode::OP_NOP,
    Opcode::OP_RESHAPE, 
    Opcode::OP_VIEW, 
    Opcode::OP_ASSEMBLE, 
    Opcode::OP_SHMEM_WAIT_UNTIL,
    Opcode::OP_BIND_TENSOR,
    Opcode::OP_VIEW_TYPE,
    Opcode::OP_HUB
};

const std::unordered_set<Opcode> COPY_IN_OPS = {
    Opcode::OP_COPY_IN,
    Opcode::OP_UB_COPY_IN,
    Opcode::OP_L1_COPY_IN,
    Opcode::OP_L1_COPY_IN_FRACTAL_Z,
    Opcode::OP_L1_COPY_IN_DMA,
    Opcode::OP_L1_COPY_UB,
    Opcode::OP_L0C_COPY_UB,
    Opcode::OP_UB_COPY_L1,
    Opcode::OP_UB_COPY_L1_ND
};

const std::unordered_map<OpCoreType, std::vector<int>> CORE_INIT_CONFIGS_HARDWARE_TWO = {
    {OpCoreType::AIV, {0, 1}},
    {OpCoreType::AIC, {0}}
};

const std::unordered_map<OpCoreType, std::vector<int>> CORE_INIT_CONFIGS_HARDWARE_ONE = {
    {OpCoreType::AIV, {0}},
    {OpCoreType::AIC, {0}}
};

const std::unordered_map<OpCoreType, std::pair<OpCoreType, int>> opCoreTypeMap {
    {OpCoreType::AIV, std::make_pair(OpCoreType::AIV, 0)},
    {OpCoreType::AIC, std::make_pair(OpCoreType::AIC, 0)}
};

// ============================================================================
// 新的调度字段结构体（不包含 Operation 引用和 id）
// 用于替代 IssueEntry，通过全局 map 存储
// ============================================================================
struct ScheduleFields;
using ScheduleFieldsPtr = std::shared_ptr<ScheduleFields>;

struct ScheduleFields {
    int execOrder{-1};
    PipeType type{PipeType::PIPE_ALL};
    bool isAlloc{false};
    bool isRetired{false};
    std::vector<Operation*> viewOps;
    std::pair<OpCoreType, int> coreLocation;

    // 当前op的前序op（直接存储 Operation*）
    std::unordered_set<Operation*> predecessors;

    // 当前op的后序op（直接存储 Operation*）
    std::unordered_set<Operation*> successors;

    // op计算所需的memId
    std::vector<int> reqMemIds;

    ScheduleFields() = default;
    ~ScheduleFields() = default;

    void Clear();
};


// ============================================================================
// 调度队列结构体
// 存储 Operation*，通过外部传入的 ScheduleFields 映射进行排序
// ============================================================================
struct ScheduleQueue {
    bool busy{false};
    Operation* curOp = nullptr;
    int curOpRetireCycle{-1};
    std::vector<Operation*> queue;

    ScheduleQueue() = default;
    ~ScheduleQueue() = default;

    void Insert(Operation* op, const std::unordered_map<Operation*, ScheduleFieldsPtr>& fieldsMap) {
        queue.push_back(op);
        std::push_heap(queue.begin(), queue.end(),
            [&fieldsMap](Operation* a, Operation* b) {
                auto itA = fieldsMap.find(a);
                auto itB = fieldsMap.find(b);
                int orderA = (itA != fieldsMap.end()) ? itA->second->execOrder : INT_MAX;
                int orderB = (itB != fieldsMap.end()) ? itB->second->execOrder : INT_MAX;
                return orderA > orderB;
            });
    }

    bool Empty() const {
        return queue.empty();
    }

    Operation* Front() const {
        return queue.empty() ? nullptr : queue[0];
    }

    Operation* PopFront(const std::unordered_map<Operation*, ScheduleFieldsPtr>& fieldsMap) {
        if (queue.empty()) {
            return nullptr;
        }
        std::pop_heap(queue.begin(), queue.end(),
            [&fieldsMap](Operation* a, Operation* b) {
                auto itA = fieldsMap.find(a);
                auto itB = fieldsMap.find(b);
                int orderA = (itA != fieldsMap.end()) ? itA->second->execOrder : INT_MAX;
                int orderB = (itB != fieldsMap.end()) ? itB->second->execOrder : INT_MAX;
                return orderA > orderB;
            });
        Operation* op = queue.back();
        queue.pop_back();
        return op;
    }
};


// ============================================================================
// SpillInfo 结构体
// ============================================================================
struct SpillInfo {
    int spillMemId_;
    Operation* spillOp_;
    LogicalTensorPtr spillTensor_;
    LogicalTensorPtr ddrTensor_;
    // A5 中 L1-spill 且 前序 op 不为 COPY_IN 时 为 true
    bool isSpecialL1_{false};
};


class OoOScheduler {
private:
    // ========================================================================
    // 调度数据结构
    // ========================================================================
    // 有序存储 Operation*，保持执行顺序
    std::vector<Operation*> scheduledOps;
    // Operation -> ScheduleFields 的映射（核心数据结构）
    std::unordered_map<Operation*, ScheduleFieldsPtr> opScheduleFields;
    // 辅助：用于快速判断 Operation 是否在调度列表中
    std::unordered_set<int> scheduledOpMagics;

    // 调度队列
    std::unordered_map<OpCoreType, std::map<int, std::map<PipeType, ScheduleQueue>>> scheduleQueues;
    std::unordered_map<OpCoreType, std::map<int, std::map<MemoryType, ScheduleQueue>>> allocScheduleQueues;

    // ========================================================================
    // 其他成员变量
    // ========================================================================
    std::unordered_map<OpCoreType, std::vector<int>> CORE_INIT_CONFIGS;

    std::unordered_map<int, LocalBufferPtr> localBufferMap;
    // 分核数据结构
    std::unordered_map<OpCoreType, std::map<int, std::map<npu::tile_fwk::MemoryType, BufferPool>>> bufferManagerMap;

    std::unordered_map<int, int> bufRefCount_;
    std::unordered_map<MemoryType, std::map<int, Operation*>> tensorOccupyMap;
    // tensor和其初始化时对应的alloc的core类型 memId-core类型
    std::unordered_map<int, std::pair<OpCoreType, int>> tensorAllocCoreMap;

    std::unordered_map<MemoryType, int64_t> localMemorySize;
    std::unordered_map<LogicalTensorPtr, LogicalTensorPtr> l02L0MXMap_;

    Function &function_;
    uint64_t spillIssueCnt{0};
    int workspaceMemId{SYMBOL_STACK_BASE};
    uint64_t numTotalIssues{0};
    std::vector<Operation *> newOperations_;
    std::vector<Operation *> operations_;

    // 深度缓存（用于排序算法）
    std::unordered_map<Operation*, int> depthCache_;

    // ========================================================================
    // ScheduleFields 辅助函数（用于操作调度字段）
    // ========================================================================
    // 基础访问函数
    ScheduleFieldsPtr GetFields(Operation* op);
    bool IsRetired(Operation* op);
    bool IsAlloc(Operation* op);
    int GetExecOrder(Operation* op);
    std::pair<OpCoreType, int> GetCoreLocation(Operation* op);
    bool IsInScheduledOps(Operation* op);

    // 操作函数
    int GetOOperandIdx(Operation* op, int curMemId);
    void UpdateTensorInput(Operation* op, ScheduleFieldsPtr spillSrcField, LogicalTensorPtr tensor);
    void UpdateTensorInputForOperand(Operation* op, size_t index, ScheduleFieldsPtr spillSrcField,
        LogicalTensorPtr tensor);
    void UpdateTensorInputForView(Operation& op, ScheduleFieldsPtr spillSrcField, LogicalTensorPtr tensor);
    std::string GetOpInfo(Operation& op);

    // ========================================================================
    // 初始化函数
    // ========================================================================
    Status InitScheduleFields(Operation* op, const std::unordered_map<Operation*, std::pair<OpCoreType, int>> &opCoreMap);
    Status InitScheduleCoreType(ScheduleFieldsPtr field, Operation* op, const std::unordered_map<Operation*, std::pair<OpCoreType, int>> &opCoreMap);
    void InitTensorCoreMap();

    // ========================================================================
    // 依赖相关函数
    // ========================================================================
    Status InitBufRefCount();
    void UpdateBufRefCount(Operation* op, ScheduleFieldsPtr field, LogicalTensorPtr tensor);
    Status InitDependencies();
    void FindDependencies(Operation* op, ScheduleFieldsPtr field);
    void AddDependency(Operation* preOp, ScheduleFieldsPtr preField, Operation* postOp, ScheduleFieldsPtr postField, bool isAlloc);
    Status InitAllocDependencies(Operation* op, ScheduleFieldsPtr field, std::unordered_map<int, Operation*> &tensor2AllocOpMap);
    Status CheckAllocIssue();
    void UpdateAllocMap(Operation* op, ScheduleFieldsPtr field, std::map<int, Operation*> &tensorAllocOpMap);
    void PrintDependencies();

    // ========================================================================
    // 调度主循环函数
    // ========================================================================
    void UpdateExecOrder();
    Status ScheduleMainLoop();
    void LaunchReadyIssue();
    Status RetireIssue(Operation* op, ScheduleFieldsPtr field);
    Status RetireIssueStage(uint64_t& commitCnt, int& nextCycle);
    Status RetireOpAndAwakeSucc(Operation* op, ScheduleFieldsPtr field, uint64_t& commitCnt);
    Status FreeBuffer(Operation* op, ScheduleFieldsPtr field);
    Status BufferAllocStage(uint64_t& commitCnt);
    Status LaunchIssueStage(int& nextCycle);
    Status CheckAndUpdateLifecycle();
    bool IsInScheduledOpsInternal(Operation* op);

    // ========================================================================
    // Spill相关函数
    // ========================================================================
    int GetBufNextUseOrder(Operation* op, int curMemId);
    int GetBufLastUseOrder(Operation* op, int curMemId);
    Operation* GetBufLastWriteOp(Operation* op, int curMemId);
    Status GenSpillSchedule();
    Status GenSpillOp(size_t &pcIdx);
    void InsertScheduledOps(Operation* insertOp, int insertOrder);
    void ReplaceTensorMemId(Operation* op, ScheduleFieldsPtr field, int oldMemId, int newMemId);
    Status UpdateRemainOpBufId(int oldMemId, int newMemId);
    Status SpillOnBlock();
    void InitScheduleQueuesAndBufferManager();
    Status ExecuteAllocIssue(Operation* op, ScheduleFieldsPtr field, size_t &pcIdx);
    Status CreateSpillCopyout(Operation* spillOp, ScheduleFieldsPtr spillField,
        LogicalTensorPtr spillTensor, int spillMemId, Operation* &spillCopyout);
    Status SpillOutBuffer(SpillInfo &spillInfo, Operation* allocOp, size_t &pcIdx, bool isGenSpill);
    Status SpillInBuffer(SpillInfo &spillInfo, Operation* allocOp, MemoryType bufferType, bool isGenSpill);
    Status GetSpillInfo(Operation* allocOp, int spillMemId, bool isGenSpill, SpillInfo &spillInfo);
    Operation* GetSpillOp(Operation* allocOp, int memId, bool isGenSpill);
    Status SpillBuffer(SpillInfo &spillInfo, Operation* allocOp, size_t &pcIdx,
        LocalBufferPtr allocBuffer, bool isGenSpill);

    // Buffer选择和多重Spill函数
    Operation* GetSpillOpFromTensorOccupy(MemoryType memType, int memId);
    bool CheckMachineAndL1(Operation* spillOp, Operation* allocOp);
    bool CheckParallelL0C2L1(Operation* spillOp);
    bool IsBelongSpillBlackList(Operation* spillOp, Operation* allocOp);
    Status GetGroupNextUseOrder(std::vector<int> group, Operation* allocOp,
        std::vector<int> &groupNextUseTime, std::unordered_map<int, size_t> &nextUseTimeCache, bool isGenSpill);
    int GetMemidAllocPriority(int memId);
    bool CanAllocateAll(std::vector<LocalBufferPtr> tensors, MemoryType memType);
    bool HasEnoughBuffer(Operation* allocOp, MemoryType memType);
    Status SelectSpillBuffers(LocalBufferPtr allocBuffer, Operation* allocOp,
        std::vector<int> &spillGroup, bool isGenSpill);
    Status RearrangeBuffer(Operation* allocOp, MemoryType memType, std::pair<OpCoreType, int> corePair, bool isGenSpill);
    Status SpillAllBuffer(Operation* allocOp, size_t &pcIdx, bool isGenSpill, LocalBufferPtr allocBuffer);
    Status SpillMultiBuffer(Operation* allocOp, std::vector<int> spillGroup, size_t &pcIdx,
        LocalBufferPtr allocBuffer, bool isGenSpill);
    Status GenBufferSpill(Operation* allocOp);
    Status GenSpillOpImpl(size_t &pcIdx);

    // ========================================================================
    // 其他辅助函数
    // ========================================================================
    Status Init(const std::vector<Operation *> &operations,
        const std::unordered_map<Operation*, std::pair<OpCoreType, int>> &opCoreMap = std::unordered_map<Operation*, std::pair<OpCoreType, int>>(),
        const std::unordered_map<OpCoreType, std::vector<int>> fixCoreConfig = CORE_INIT_CONFIGS_HARDWARE_ONE);
    void InitCoreConfig(const std::vector<Operation *> &operations);
    Status CheckOpBufferSize(Operation *op);
    std::string dumpOpInfo(Operation &op);
    Status CalcBufferSize(LogicalTensors tensors, std::map<MemoryType, int64_t> &bufferSize, std::set<int> &memIdMap);
    Status InitLocalBuffer(LogicalTensorPtr oOperand, int memId);
    Status DelBufRefCount(const int memId);
    void UpdateBufferUsage(MemoryType bufferType, int memId, bool isFree);
    void PrintOpList(std::vector<Operation *> operations);
    size_t ShapeCeilAlign(std::vector<int64_t> shape, DataType dtype);
    Status AllocViewTensorMemRange(Operation &operation);

    int64_t CalcWorkspaceOffset(std::vector<int64_t> shape, std::vector<int64_t> offset);
    void GetWorkspaceBaseOffset(LogicalTensorPtr ddrTensor, int64_t &base);
    Status UpdateCopyOutMode(Operation &copyOutOp);
    Status UpdateCopyInMode(Operation &copyInOp);
    Status UpdateTensorAttr(LogicalTensorPtr tensor, MemoryType memType, LogicalTensorPtr spillTensor, int spillMemId);
    OoOSchedulerCheck::SpillInfo RecordSpillInfo(MemoryType bufferType, int memId, LocalBufferPtr allocBuffer,
        LogicalTensorPtr spillOutTensor, bool needCopyOut);

    // buffer rearrange
    Status UpdateMemId(int oldMemId, int newMemId);
    void UpdateMoveOpAttr(Operation &moveOp, Operation &occupyOp);
    Status UpdateRange(int newMemId, size_t offset, MemoryType memType, BufferPool &bufferManager);
    Status FindMoveFromTensor(Operation &occupyOp, int oldMemId, MemoryType memType, bool &rearrangeUBBF16, LogicalTensorPtr &moveFromTensor);
    Status GetMoveOpInTensor(Opcode moveOpcode, Operation &occupyOp, LogicalTensorPtr &inTensor, LogicalTensorPtr &moveFromTensor);

public:
    Status Schedule(const std::vector<Operation *> &operations,
        const std::unordered_map<Operation*, std::pair<OpCoreType, int>> &opCoreMap = std::unordered_map<Operation*, std::pair<OpCoreType, int>>(),
        const std::unordered_map<OpCoreType, std::vector<int>> fixCoreConfig = CORE_INIT_CONFIGS_HARDWARE_ONE);
    OoOScheduler(Function &function) : function_(function) {}

    std::vector<Operation *> GetNewOperations() { return newOperations_; }
    int64_t workspaceOffset{0};
    int clock{0};
    OoOSchedulerCheck oooCheck;
    std::unordered_map<PipeType, int> pipeEndTime;
};
} // namespace npu::tile_fwk
#endif // PASS_SCHEDULER_H