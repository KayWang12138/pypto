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
 * \file schedule_ooo.h
 * \brief
 */

#ifndef PASS_SCHEDULE_OOO_H
#define PASS_SCHEDULE_OOO_H

#include "passes/block_graph_pass/schedule_ooo/buffer_pool.h"
#include "passes/block_graph_pass/schedule_ooo/ooo_scheduler.h"
#include "passes/statistics/ooo_schedule_statistic.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/block_graph_pass/schedule_ooo/optimize_sort.h"
#include "passes/block_graph_pass/schedule_ooo/latency_estimator.h"
#include "passes/block_graph_pass/schedule_ooo/core_assign.h"

namespace npu::tile_fwk {

const std::unordered_map<TargetCoreType, CoreLocationType> targetCoreTypeMap{
    {TargetCoreType::AIC, CoreLocationType::AIC},
    {TargetCoreType::AIV0, CoreLocationType::AIV0},
    {TargetCoreType::AIV1, CoreLocationType::AIV1}};

const std::unordered_set<CoreLocationType> CORE_INIT_CONFIGS_HARDWARE_TWO_AIV = {
    CoreLocationType::AIC, CoreLocationType::AIV0, CoreLocationType::AIV1};

// DualDst pair structure for tracking pairs of OP_L0C_COPY_UB operations
struct DualDstPair {
    Operation* copyOp1 = nullptr;       // 第一个OP_L0C_COPY_UB
    Operation* copyOp2 = nullptr;       // 第二个OP_L0C_COPY_UB
    LogicalTensorPtr l0cTensor = nullptr; // 共享的L0C输入tensor
    bool isSplitM = true;               // true=SplitM, false=SplitN
    int firstOpIdx = -1;                // 第一个操作在opList中的位置
};

class OoOSchedule : public Pass {
public:
    OoOSchedule() : Pass("OoOSchedule") {}
    ~OoOSchedule() override {}

private:
    Status RunOnFunction(Function& function) override;
    bool IsAicpuProgram(std::vector<Operation*> opList);
    Status PreCheck(Function& function) override;
    Status PostCheck(Function& function) override;
    void DoHealthCheckAfter(Function& function, const std::string& folderPath) override;
    void SortTaskList(std::vector<Operation*>& operations, std::vector<Operation*>& taskList);
    Status SortAndLatencyEstimate(std::vector<Operation*>& opList, std::vector<Operation*>& taskOpList, int& latency);
    void CollectStatistic(OoOScheduleStatistic& oooHealthCheck,
        Function& function, std::pair<uint64_t, Function*>& program);
    Status RecordLastUseMemory(Function& function);
    Status NonMixSchedule(
        std::vector<Operation*>& opList, Function& function, std::pair<uint64_t, Function*>& program,
        int64_t& maxWorkeSpaceSize);
    Status MixSchedule(
        std::vector<Operation*>& opList, Function& function, std::pair<uint64_t, Function*>& program,
        int64_t& maxWorkeSpaceSize);
    Status EstimateTaskLatencyAndSchedule(TaskSpliter& spliter, std::vector<Operation*>& opList);
    Status BuildMixedScheduleOps(TaskSpliter& spliter, std::vector<Operation*>& opList,
        std::unordered_map<Operation*, CoreLocationType>& opCoreMap);
    Status AdvanceAlloc(std::vector<Operation*>& opList, Operation* op, size_t& index);
    Status ModifyBoundaryOrder(std::vector<Operation*>& opList);
    bool IsBoundary(Operation* op);
    Status UpdateOpCoreMap(
        const TaskNode& taskNode, std::unordered_map<Operation*, CoreLocationType>& opCoreMap);

    // DualDst related methods
    bool CheckDualDstEnabled();
    Status DetectDualDstPairs(std::vector<Operation*>& opList);
    void DetectSplitPairs(LogicalTensorPtr l0cTensor, std::vector<Operation*>& copyOps);
    Status CreateDualDstOps(std::vector<Operation*>& opList, Function& function);
    void SetDualDstOpAttribute(Operation& dualDstOp, const DualDstPair& pair,
        LogicalTensorPtr ubTensor1, LogicalTensorPtr ubTensor2);
    Operation* GetAllocOpForTensor(LogicalTensorPtr tensor, std::vector<Operation*>& opList);
    void MarkDualDstAlloc(Operation* allocOp1, Operation* allocOp2);
    Status UpdateDualDstDependencies(Operation& dualDstOp, Operation* copyOp1, Operation* copyOp2);

    std::vector<Function*> oriFunctions;
    std::map<uint64_t, OoOScheduleStatistic> statisticMap_;
    std::unordered_map<LogicalTensorPtr, Operation*> lastUseMap_;
    OoOScheduleChecker checker;

    // DualDst configuration and tracking
    bool enableDualDst_ = false;        // DualDst开关
    std::vector<DualDstPair> dualDstPairs_; // 记录所有检测到的dual_dst pairs
};
} // namespace npu::tile_fwk
#endif // PASS_SCHEDULE_OOO_H
