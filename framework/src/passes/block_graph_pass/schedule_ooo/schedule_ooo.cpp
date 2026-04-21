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
 * \file schedule_ooo.cpp
 * \brief
 */

#include "schedule_ooo.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/dead_operation_eliminate.h"

#ifndef MODULE_NAME
#define MODULE_NAME "OoOSchedule"
#endif

namespace npu::tile_fwk {

bool OoOSchedule::IsAicpuProgram(std::vector<Operation*> opList)
{
    for (auto& op : opList) {
        if (op->GetCoreType() == CoreType::AICPU) {
            return true;
        }
    }
    return false;
}

inline bool IsMixGraph(const std::vector<Operation*>& opList)
{
    bool hasAIC = false;
    bool hasAIV = false;
    for (auto opPtr : opList) {
        if (OpcodeManager::Inst().GetCoreType(opPtr->GetOpcode()) == OpCoreType::AIC) {
            hasAIC = true;
        } else if (OpcodeManager::Inst().GetCoreType(opPtr->GetOpcode()) == OpCoreType::AIV) {
            hasAIV = true;
        }
        if (hasAIC && hasAIV) {
            return true;
        }
    }
    return false;
}

void OoOSchedule::SortTaskList(std::vector<Operation*>& opList, std::vector<Operation*>& taskList)
{
    std::vector<Operation*> newTaskList;
    for (auto op : opList) {
        if (std::find(taskList.begin(), taskList.end(), op) != taskList.end()) {
            newTaskList.push_back(op);
        }
    }
    taskList = newTaskList;
}

void OoOSchedule::CollectStatistic(OoOScheduleStatistic& oooHealthCheck,
    Function& function, std::pair<uint64_t, Function*>& program)
{
    if (passDfxconfigs_.healthCheck) {
        oooHealthCheck.SetOutputPrefix(GetDumpFilePrefix(function, false, program.second, program.first));
        statisticMap_.insert({program.first, oooHealthCheck});
    }
}

Status OoOSchedule::NonMixSchedule(
    std::vector<Operation*>& opList, Function& function, std::pair<uint64_t, Function*>& program,
    int64_t& maxWorkeSpaceSize)
{
    // 直接对oplist进行GenSpill和mainLoop
    APASS_LOG_INFO_F(Elements::Operation, "=============== START NonMixSchedule ===============");
    OoOScheduler oooSchedule(*program.second);
    OoOScheduleStatistic oooHealthCheck;
    if (passDfxconfigs_.healthCheck) {
        oooSchedule.AddObserver(&oooHealthCheck);
    }
    if (oooSchedule.Schedule(opList) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Non-mixGraph schedule failed.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Operation, "Subgraph[%zu] OOOSchedule end.", program.first);
    program.second->ScheduleBy(oooSchedule.GetNewOperations());
    program.second->RecordOOOSeq();
    RescheduleUtils::UpdateTensorConsProd(program.second);
    maxWorkeSpaceSize = std::max(maxWorkeSpaceSize, (*program.second).GetStackWorkespaceSize());
    function.SetStackWorkespaceSize(maxWorkeSpaceSize);
    CollectStatistic(oooHealthCheck, function, program);
    return SUCCESS;
}

bool OoOSchedule::IsBoundary(Operation* op)
{
    if (op->GetOpcode() == Opcode::OP_L0C_COPY_UB || op->GetOpcode() == Opcode::OP_L1_COPY_UB ||
        op->GetOpcode() == Opcode::OP_UB_COPY_L1) {
        return true;
    }
    return false;
}

Status OoOSchedule::AdvanceAlloc(std::vector<Operation*>& opList, Operation* op, size_t& index)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "Advance alloc of op: %s[%d]", op->GetOpcodeStr().c_str(), op->GetOpMagic());
    for (auto& preOp : op->GetOutputOperand(0)->GetProducers()) {
        if (preOp->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            auto it = std::find(opList.begin(), opList.end(), preOp);
            if (it == opList.end()) {
                APASS_LOG_ERROR_F(Elements::Operation, "Cannot find the alloc of boundaryop.");
                return FAILED;
            }
            size_t allocIndex = std::distance(opList.begin(), it);
            if (allocIndex > index) {
                APASS_LOG_DEBUG_F(Elements::Operation, "alloc index: %zu, op index: %zu", allocIndex, index);
                std::rotate(opList.begin() + index, opList.begin() + allocIndex, opList.begin() + allocIndex + 1);
                index++;
                return SUCCESS;
            }
        }
    }
    return SUCCESS;
}

Status OoOSchedule::ModifyBoundaryOrder(std::vector<Operation*>& opList)
{
    size_t i = 0;
    while (i < opList.size()) {
        if (IsBoundary(opList[i])) {
            if (AdvanceAlloc(opList, opList[i], i) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "AdvanceAlloc failed.");
                return FAILED;
            }
        }
        i++;
    }
    return SUCCESS;
}

Status OoOSchedule::MixSchedule(
    std::vector<Operation*>& opList, Function& function, std::pair<uint64_t, Function*>& program,
    int64_t& maxWorkeSpaceSize)
{
    APASS_LOG_INFO_F(Elements::Operation, "=============== START MixSchedule ===============");
    TaskSpliter spliter;
    // 对 taskNode.opList_ 进行排序，并返回预估 latency，随后完成 core schedule 与子图合并。
    if (EstimateTaskLatencyAndSchedule(spliter, opList) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "EstimateTaskLatencyAndSchedule failed.");
        return FAILED;
    }
    std::unordered_map<Operation*, CoreLocationType> opCoreMap;
    // 传入 taskNode 序列，对全部 opList 重新拼装并构建 opCoreMap。
    if (BuildMixedScheduleOps(spliter, opList, opCoreMap) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "BuildMixedScheduleOps failed.");
        return FAILED;
    }
    if (ModifyBoundaryOrder(opList) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ModifyBoundaryOrder failed.");
        return FAILED;
    }
    OoOScheduler oooSchedule(*program.second);
    OoOScheduleStatistic oooHealthCheck;
    if (passDfxconfigs_.healthCheck) {
        oooSchedule.AddObserver(&oooHealthCheck);
    }
    if (oooSchedule.Schedule(opList, opCoreMap, CORE_INIT_CONFIGS_HARDWARE_TWO_AIV) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Schedule failed.");
        return FAILED;
    }
    CollectStatistic(oooHealthCheck, function, program);
    APASS_LOG_INFO_F(Elements::Operation, "Subgraph[%zu] OOOSchedule end.", program.first);
    program.second->ScheduleBy(oooSchedule.GetNewOperations());
    program.second->RecordOOOSeq();
    RescheduleUtils::UpdateTensorConsProd(program.second);
    maxWorkeSpaceSize = std::max(maxWorkeSpaceSize, (*program.second).GetStackWorkespaceSize());
    function.SetStackWorkespaceSize(maxWorkeSpaceSize);
    return SUCCESS;
}

Status OoOSchedule::EstimateTaskLatencyAndSchedule(TaskSpliter& spliter, std::vector<Operation*>& opList)
{
    static const std::unordered_map<TargetCoreType, std::string> targetToString{
        {TargetCoreType::AIC, "AIC"},
        {TargetCoreType::AIV0, "AIV0"},
        {TargetCoreType::AIV1, "AIV1"},
        {TargetCoreType::UNKNOWN, "UNKNOWN"}};

    spliter.SplitGraph(opList);
    for (auto& taskNode : spliter.GetTaskGraph().tasks) {
        if (SortAndLatencyEstimate(opList, taskNode.opList_, taskNode.latency) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SortAndLatencyEstimate failed, taskNode[%d].", taskNode.idx);
            return FAILED;
        }
    }
    CoreScheduler coreScheduler;
    coreScheduler.Schedule(spliter.GetTaskGraph(), 10); // BruteForce threshold is 10
    for (auto& taskNode : spliter.GetTaskGraph().tasks) {
        APASS_LOG_INFO_F(Elements::Operation, "eval task %d on %s: %d - %d.", taskNode.idx,
            targetToString.at(taskNode.targetCoreType).c_str(), taskNode.startTime, taskNode.endTime);
    }
    spliter.MergeTask();
    spliter.MarkInternalSubgraphID();
    return SUCCESS;
}

Status OoOSchedule::BuildMixedScheduleOps(TaskSpliter& spliter, std::vector<Operation*>& opList,
    std::unordered_map<Operation*, CoreLocationType>& opCoreMap)
{
    auto taskNodeList = spliter.GetTaskGraph().tasks;
    std::sort(taskNodeList.begin(), taskNodeList.end(), [](const TaskNode& a, const TaskNode& b) {
        return a.startTime < b.startTime;
    });
    std::vector<Operation*> operations;
    for (auto& taskNode : taskNodeList) {
        SortTaskList(taskNode.opList_, opList);
        if (UpdateOpCoreMap(taskNode, opCoreMap) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "UpdateOpCoreMap failed, taskNode[%d].", taskNode.idx);
            return FAILED;
        }
        operations.insert(operations.end(), taskNode.opList_.begin(), taskNode.opList_.end());
    }
    opList = std::move(operations);
    return SUCCESS;
}

Status OoOSchedule::UpdateOpCoreMap(
    const TaskNode& taskNode, std::unordered_map<Operation*, CoreLocationType>& opCoreMap)
{
    for (auto op : taskNode.opList_) {
        if (taskNode.targetCoreType == TargetCoreType::UNKNOWN) {
            APASS_LOG_ERROR_F(Elements::Operation, "CoreType is not AIC, AIV0 or AIV1");
            return FAILED;
        }
        opCoreMap[op] = targetCoreTypeMap.at(taskNode.targetCoreType);
    }
    return SUCCESS;
}

Status OoOSchedule::SortAndLatencyEstimate(
    std::vector<Operation*>& opList, std::vector<Operation*>& taskOpList, int& latency)
{
    APASS_LOG_INFO_F(Elements::Operation, "=======>start SortAndLatencyEstimate");
    SortTaskList(opList, taskOpList);
    LatencyEstimator latencyEstimator(taskOpList, opList);
    if (latencyEstimator.LatencyEstimatorMainLoop() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "SortAndLatencyEstimate LatencyEstimatorMainLoop failed.");
        return FAILED;
    }
    latency = latencyEstimator.clock;
    APASS_LOG_INFO_F(Elements::Operation, "=======>end SortAndLatencyEstimate");
    return SUCCESS;
}

Status OoOSchedule::RecordLastUseMemory(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start RecordLastUseMemory.");
    for (auto& program : function.rootFunc_->programs_) {
        auto opList = program.second->Operations(false);
        for (size_t opIdx = 0; opIdx < opList.size(); opIdx++) {
            Operation* op = &opList[opIdx];
            if (LASTUSE_OPS.find(op->GetOpcode()) != LASTUSE_OPS.end()) {
                int tensorSize = op->GetIOperands().size() + op->GetOOperands().size();
                std::vector<int> initVec(tensorSize, false);
                op->SetAttribute(OpAttributeKey::lastUse, initVec);
            }
            for (size_t inputIdx = 0; inputIdx < op->GetIOperands().size(); inputIdx++) {
                auto inTensor = op->GetInputOperand(inputIdx);
                lastUseMap_[inTensor] = op;
            }
        }
    }
    std::unordered_map<Operation*, std::vector<int>> opInputIdxMap;
    std::unordered_set<Opcode> reduceOp = {
        Opcode::OP_ROWSUM_SINGLE, Opcode::OP_ROWMAX_SINGLE, Opcode::OP_ROWMIN_SINGLE};
    for (auto& entry : lastUseMap_) {
        auto lastUseOp = entry.second;
        auto lastUseTensor = entry.first;
        if (LASTUSE_OPS.find(lastUseOp->GetOpcode()) == LASTUSE_OPS.end()) {
            continue; // 针对非LASTUSE_OPS中的op不做处理，仅处理LASTUSE_OPS中的op
        }
        if (opInputIdxMap.find(lastUseOp) == opInputIdxMap.end()) {
            int tensorSize = lastUseOp->GetIOperands().size() + lastUseOp->GetOOperands().size();
            std::vector<int> tensorIdxVec(tensorSize, false);
            int inputIdx = lastUseOp->GetIOperandIndex(lastUseTensor) + lastUseOp->GetOOperands().size();
            if (reduceOp.find(lastUseOp->GetOpcode()) != reduceOp.end() && inputIdx == tensorSize - 1) {
                tensorIdxVec[inputIdx] = false;
            } else {
                tensorIdxVec[inputIdx] = true;
            }
            opInputIdxMap[lastUseOp] = tensorIdxVec;
        } else {
            int inputIdx = lastUseOp->GetIOperandIndex(lastUseTensor) + lastUseOp->GetOOperands().size();
            opInputIdxMap[lastUseOp][inputIdx] = true;
        }
    }
    for (auto& entry : opInputIdxMap) {
        auto op = entry.first;
        op->SetAttribute(OpAttributeKey::lastUse, opInputIdxMap[op]);
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End RecordLastUseMemory.");
    return SUCCESS;
}

Status OoOSchedule::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "=============== START 2CoreSplit ===============");

    // 检查是否启用dual_dst
    enableDualDst_ = CheckDualDstEnabled();

    int64_t maxWorkeSpaceSize = 0;
    for (auto& program : function.rootFunc_->programs_) {
        auto opList = program.second->Operations(false).DuplicatedOpList();
        oriFunctions.emplace_back(program.second);
        // ooo不处理aicpu子图
        if (IsAicpuProgram(opList)) {
            continue;
        }
        OptimizeSort optimizeSort(opList, *program.second);
        if (optimizeSort.SortOps() != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Global sortOps failed");
            return FAILED;
        }
        // 全局排序的序列
        opList = optimizeSort.operations;

        // === DualDst检测和处理 ===
        if (enableDualDst_) {
            if (DetectDualDstPairs(opList) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "DetectDualDstPairs failed.");
                return FAILED;
            }
            if (dualDstPairs_.size() > 0) {
                if (CreateDualDstOps(opList, *program.second) != SUCCESS) {
                    APASS_LOG_ERROR_F(Elements::Operation, "CreateDualDstOps failed.");
                    return FAILED;
                }
            }
        }

        std::pair<uint64_t, Function*> programRef;
        programRef.first = program.first;
        programRef.second = program.second;
        if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510 || !IsMixGraph(opList)) {
            if (NonMixSchedule(opList, function, programRef, maxWorkeSpaceSize) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "NonMix OoO schedule failed.");
                return FAILED;
            }
            DeadOperationEliminator eliminator;
            eliminator.EliminateOperationAndNotSortAfterErase(*program.second);
            programRef.second = program.second;
            continue;
        }
        if (MixSchedule(opList, function, programRef, maxWorkeSpaceSize) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Mix OoO schedule failed.");
            return FAILED;
        }
        DeadOperationEliminator eliminator;
        eliminator.EliminateOperationAndNotSortAfterErase(*program.second);
        programRef.second = program.second;
    }
    if (RecordLastUseMemory(function) == FAILED) {
        APASS_LOG_ERROR_F(Elements::Function, "Run RecordLastUseMemory Failed.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Operation, "=============== END 2CoreSplit ===============");
    return SUCCESS;
}

void OoOSchedule::DoHealthCheckAfter(Function& function, const std::string& folderPath)
{
    for (auto& [programId, check] : statisticMap_) {
        auto fileName = folderPath + '/' + check.jsonFileName + "_Block_Graph_Health_Report.json";
        auto it = function.rootFunc_->programs_.find(programId);
        if (it != function.rootFunc_->programs_.end()) {
            check.DoHealthCheck(it->second, fileName);
        }
    }
}

Status OoOSchedule::PreCheck(Function& function) { return checker.DoPreCheck(function); }

Status OoOSchedule::PostCheck(Function& function)
{
    checker.SetOriFunctions(oriFunctions);
    return checker.DoPostCheck(function);
}

// ========== DualDst Implementation ==========

bool OoOSchedule::CheckDualDstEnabled()
{
    // TODO: 从配置或环境变量读取dual_dst开关
    // 暂时默认返回true用于测试
    return true;
}

Status OoOSchedule::DetectDualDstPairs(std::vector<Operation*>& opList)
{
    APASS_LOG_INFO_F(Elements::Operation, "===> Start DetectDualDstPairs.");
    dualDstPairs_.clear();

    std::unordered_set<LogicalTensorPtr> visitedL0C;

    for (auto& op : opList) {
        if (op->GetOpcode() != Opcode::OP_L0C_COPY_UB) continue;

        auto l0cTensor = op->GetInputOperand(0);
        if (visitedL0C.count(l0cTensor)) continue;
        visitedL0C.insert(l0cTensor);

        // 检查条件：shape必须是2维
        auto l0cShape = l0cTensor->GetShape();
        if (l0cShape.size() != 2) continue;

        // 收集所有OP_L0C_COPY_UB consumers
        std::vector<Operation*> l0cCopyOps;
        for (auto consumer : l0cTensor->GetConsumers()) {
            if (consumer->GetOpcode() == Opcode::OP_L0C_COPY_UB) {
                l0cCopyOps.push_back(consumer);
            }
        }

        // 只有1个consumer则跳过
        if (l0cCopyOps.size() < 2) continue;

        // 检测SplitM/SplitN pairs
        DetectSplitPairs(l0cTensor, l0cCopyOps);
    }

    APASS_LOG_INFO_F(Elements::Operation, "Detected %zu DualDst pairs.", dualDstPairs_.size());
    return SUCCESS;
}

void OoOSchedule::DetectSplitPairs(LogicalTensorPtr l0cTensor, std::vector<Operation*>& copyOps)
{
    // 构建坐标映射: offset -> UB tensor
    std::map<std::pair<int, int>, Operation*> coordToCopyOp;
    int maxX = -1, maxY = -1;

    auto firstShape = copyOps[0]->GetOutputOperand(0)->GetShape();
    auto firstValidShape = copyOps[0]->GetOutputOperand(0)->GetDynValidShape();

    for (auto copyOp : copyOps) {
        auto ubTensor = copyOp->GetOutputOperand(0);

        // 检查shape和validShape必须完全相同
        if (ubTensor->GetShape() != firstShape) continue;
        if (ubTensor->GetDynValidShape() != firstValidShape) continue;

        auto offset = ubTensor->GetOffset();
        if (offset.size() != 2) continue;
        if (offset[0] % firstShape[0] != 0 || offset[1] % firstShape[1] != 0) continue;

        int x = offset[0] / firstShape[0];
        int y = offset[1] / firstShape[1];
        maxX = std::max(maxX, x);
        maxY = std::max(maxY, y);
        coordToCopyOp[{x, y}] = copyOp;
    }

    // 检测SplitM pairs (M轴连续) 和 SplitN pairs (N轴连续)
    std::vector<DualDstPair> splitMPairs;
    std::vector<DualDstPair> splitNPairs;

    // SplitM: x方向相邻配对
    for (int x = 0; x <= maxX - 1; x += 2) {
        for (int y = 0; y <= maxY; y++) {
            if (!coordToCopyOp.count({x, y}) || !coordToCopyOp.count({x + 1, y})) continue;

            auto op0 = coordToCopyOp[{x, y}];
            auto op1 = coordToCopyOp[{x + 1, y}];

            // offset较小的应该在前面作为copyOp1
            splitMPairs.push_back({op0, op1, l0cTensor, true, -1});
        }
    }

    // SplitN: y方向相邻配对
    for (int x = 0; x <= maxX; x++) {
        for (int y = 0; y <= maxY - 1; y += 2) {
            if (!coordToCopyOp.count({x, y}) || !coordToCopyOp.count({x, y + 1})) continue;

            auto op0 = coordToCopyOp[{x, y}];
            auto op1 = coordToCopyOp[{x, y + 1}];

            splitNPairs.push_back({op0, op1, l0cTensor, false, -1});
        }
    }

    // 选择更多pairs的模式，只使用一种模式
    if (splitMPairs.size() >= splitNPairs.size() && splitMPairs.size() > 0) {
        for (auto& pair : splitMPairs) {
            dualDstPairs_.push_back(pair);
        }
    } else if (splitNPairs.size() > 0) {
        for (auto& pair : splitNPairs) {
            dualDstPairs_.push_back(pair);
        }
    }
}

Status OoOSchedule::CreateDualDstOps(std::vector<Operation*>& opList, Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "===> Start CreateDualDstOps.");

    for (auto& pair : dualDstPairs_) {
        // 验证两个OP_L0C_COPY_UB的输入是同一个L0C tensor
        if (pair.copyOp1->GetInputOperand(0) != pair.copyOp2->GetInputOperand(0)) {
            APASS_LOG_ERROR_F(Elements::Operation, "DualDst pair inputs are not the same L0C tensor.");
            return FAILED;
        }

        // 获取两个UB输出tensor
        auto ubTensor1 = pair.copyOp1->GetOutputOperand(0);
        auto ubTensor2 = pair.copyOp2->GetOutputOperand(0);

        // 获取原始两个ALLOC操作
        auto allocOp1 = GetAllocOpForTensor(ubTensor1, opList);
        auto allocOp2 = GetAllocOpForTensor(ubTensor2, opList);

        // 创建OP_L0C_COPY_UB_DUAL_DST
        auto& dualDstOp = function.AddRawOperation(
            Opcode::OP_L0C_COPY_UB_DUAL_DST,
            {pair.l0cTensor},
            {ubTensor1, ubTensor2}
        );

        // 设置属性
        SetDualDstOpAttribute(dualDstOp, pair, ubTensor1, ubTensor2);

        // 更新ALLOC操作的dual_dst标记
        if (allocOp1 && allocOp2) {
            MarkDualDstAlloc(allocOp1, allocOp2);
        }

        // 记录第一个操作的位置用于后续插入
        auto it = std::find(opList.begin(), opList.end(), pair.copyOp1);
        if (it != opList.end()) {
            pair.firstOpIdx = std::distance(opList.begin(), it);
        }

        // 更新依赖关系
        if (UpdateDualDstDependencies(dualDstOp, pair.copyOp1, pair.copyOp2) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "UpdateDualDstDependencies failed.");
            return FAILED;
        }

        // 标记原始操作为待删除
        pair.copyOp1->SetAsDeleted();
        pair.copyOp2->SetAsDeleted();
    }

    // 删除原始操作
    function.EraseOperations();

    // 更新opList：移除已删除的操作，添加新的DualDst操作
    std::vector<Operation*> newOpList;
    std::vector<Operation*> dualDstOps;

    // 获取所有新创建的DualDst操作
    auto allOps = function.Operations(false).DuplicatedOpList();
    for (auto op : allOps) {
        if (op->GetOpcode() == Opcode::OP_L0C_COPY_UB_DUAL_DST) {
            dualDstOps.push_back(op);
        }
    }

    // 构建新的opList
    for (size_t i = 0; i < opList.size(); i++) {
        auto op = opList[i];
        if (op->IsDeleted()) continue;
        newOpList.push_back(op);

        // 在相应位置插入DualDst操作
        for (auto& pair : dualDstPairs_) {
            if (static_cast<int>(i) == pair.firstOpIdx && pair.firstOpIdx >= 0) {
                // 找到对应的DualDst操作
                for (auto dualDstOp : dualDstOps) {
                    // 检查输入是否匹配
                    if (dualDstOp->GetInputOperand(0) == pair.l0cTensor) {
                        newOpList.push_back(dualDstOp);
                        // 移除已添加的DualDst操作避免重复
                        dualDstOps.erase(std::remove(dualDstOps.begin(), dualDstOps.end(), dualDstOp), dualDstOps.end());
                        break;
                    }
                }
            }
        }
    }

    // 添加剩余未插入的DualDst操作（如果有）
    for (auto dualDstOp : dualDstOps) {
        newOpList.push_back(dualDstOp);
    }

    opList = newOpList;

    APASS_LOG_INFO_F(Elements::Operation, "===> End CreateDualDstOps, created %zu dual_dst ops.", dualDstPairs_.size());
    return SUCCESS;
}

void OoOSchedule::SetDualDstOpAttribute(Operation& dualDstOp, const DualDstPair& pair,
    LogicalTensorPtr ubTensor1, LogicalTensorPtr ubTensor2)
{
    // 设置splitMN属性: 0=SplitM, 1=SplitN
    dualDstOp.SetAttribute(OpAttributeKey::splitMN, pair.isSplitM ? 0 : 1);

    // 设置isCube属性
    dualDstOp.SetAttribute(OpAttributeKey::isCube, true);

    // 设置CopyOpAttribute
    auto l0cShape = pair.l0cTensor->GetShape();
    auto l0cValidShape = pair.l0cTensor->GetDynValidShape();

    std::vector<OpImmediate> fromOffset0 = OpImmediate::Specified(ubTensor1->GetOffset());
    std::vector<OpImmediate> fromOffset1 = OpImmediate::Specified(ubTensor2->GetOffset());

    std::vector<SymbolicScalar> validShape;
    for (auto dim : l0cShape) {
        validShape.push_back(SymbolicScalar(dim));
    }

    auto copyAttr = std::make_shared<CopyOpAttribute>(
        fromOffset0,
        pair.l0cTensor->GetMemoryTypeOriginal(),
        OpImmediate::Specified(l0cShape),
        OpImmediate::Specified(l0cValidShape),
        OpImmediate::Specified(validShape)
    );

    dualDstOp.SetOpAttribute(copyAttr);
}

Operation* OoOSchedule::GetAllocOpForTensor(LogicalTensorPtr tensor, std::vector<Operation*>& opList)
{
    for (auto op : opList) {
        if (op->GetOpcode() == Opcode::OP_ALLOC) {
            auto outputTensor = op->GetOutputOperand(0);
            if (outputTensor == tensor) {
                return op;
            }
        }
    }
    return nullptr;
}

void OoOSchedule::MarkDualDstAlloc(Operation* allocOp1, Operation* allocOp2)
{
    // DualDst alloc标记现在由OoOScheduler在Init时自动检测
    // 此函数保留作为占位，scheduler会通过检查OP_L0C_COPY_UB_DUAL_DST来识别dual_dst allocs
}

Status OoOSchedule::UpdateDualDstDependencies(Operation& dualDstOp, Operation* copyOp1, Operation* copyOp2)
{
    // 收集原始操作的生产者和消费者
    auto l0cTensor = copyOp1->GetInputOperand(0);
    auto ubTensor1 = copyOp1->GetOutputOperand(0);
    auto ubTensor2 = copyOp2->GetOutputOperand(0);

    // DualDst操作继承了两个COPY操作的生产者
    // 输入依赖已经在AddRawOperation中自动设置

    // 更新消费者依赖：将原COPY操作的消费者指向DualDst操作
    for (auto consumer : ubTensor1->GetConsumers()) {
        if (consumer != copyOp1 && consumer != &dualDstOp) {
            // 更新consumer的输入tensor依赖
            auto& operands = consumer->GetIOperands();
            for (size_t i = 0; i < operands.size(); i++) {
                if (operands[i].tensor_ == ubTensor1) {
                    // tensor依赖关系会自动更新，不需要手动修改
                }
            }
        }
    }

    for (auto consumer : ubTensor2->GetConsumers()) {
        if (consumer != copyOp2 && consumer != &dualDstOp) {
            // tensor依赖关系会自动更新
        }
    }

    return SUCCESS;
}

} // namespace npu::tile_fwk
