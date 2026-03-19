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
 * \file scheduler.cpp
 * \brief
 */

#include "scheduler.h"
#include "buffer_rearrange.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "OoOSchedule"

namespace npu::tile_fwk {

constexpr int32_t DIM_FIVE = 5;
constexpr int32_t LAST_TWO_DIM = 2;
constexpr int32_t UB_BLOCK_SIZE = 32;

inline std::string coreTypeToString(OpCoreType coreType) {
    switch (coreType) {
        case OpCoreType::AIV: return "AIV";
        case OpCoreType::AIC: return "AIC";
        default: return "MEM_UNKNOWN";
    }
}

inline bool IsMixGraph(const std::vector<Operation*> &operations) {
    bool hasAIC = false;
    bool hasAIV = false;
    for (auto opPtr : operations) {
        if (OpcodeManager::Inst().GetCoreType(opPtr->GetOpcode()) == OpCoreType::AIV) {
            hasAIV = true;
        } else if (OpcodeManager::Inst().GetCoreType(opPtr->GetOpcode()) == OpCoreType::AIC) {
            hasAIC = true;
        }
        if (hasAIC && hasAIV) {
            return true;
        }
    }
    return false;
}

inline Operation* SkipViewChain(Operation* start, bool followProducers) {
    if (start == nullptr) return nullptr;
    Operation* op = start;
    Operation* lastView = nullptr;
    while (op != nullptr && IsViewOp(*op)) {
        lastView = op;
        if (followProducers) {
            const auto& nextOps = op->GetInputOperand(0)->GetProducers();
            if (nextOps.size() != 1) break;
            op = *nextOps.begin();
        } else {
            const auto& nextOps = op->GetOutputOperand(0)->GetConsumers();
            if (nextOps.size() != 1) break;
            op = *nextOps.begin();
        }
    }
    return lastView;
}

// ============================================================================
// ScheduleFields 结构体方法实现
// ============================================================================
void ScheduleFields::Clear() {
    isRetired = false;
    predecessors.clear();
    successors.clear();
    reqMemIds.clear();
}

// ============================================================================
// OoOScheduler 辅助函数实现（用于操作 ScheduleFields）
// ============================================================================
ScheduleFieldsPtr OoOScheduler::GetFields(Operation* op) {
    if (op == nullptr) {
        return nullptr;
    }
    auto it = opScheduleFields.find(op);
    if (it == opScheduleFields.end()) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Cannot find schedule fields for op[%d]", op->GetOpMagic());
        return nullptr;
    }
    return it->second;
}

bool OoOScheduler::IsRetired(Operation* op) {
    auto fields = GetFields(op);
    return fields ? fields->isRetired : false;
}

bool OoOScheduler::IsAlloc(Operation* op) {
    auto fields = GetFields(op);
    return fields ? fields->isAlloc : false;
}

int OoOScheduler::GetExecOrder(Operation* op) {
    auto fields = GetFields(op);
    return fields ? fields->execOrder : -1;
}

std::pair<OpCoreType, int> OoOScheduler::GetCoreLocation(Operation* op) {
    auto fields = GetFields(op);
    static const std::pair<OpCoreType, int> defaultLocation = {OpCoreType::AIV, 0};
    return fields ? fields->coreLocation : defaultLocation;
}

bool OoOScheduler::IsInScheduledOps(Operation* op) {
    if (op == nullptr) {
        return false;
    }
    return scheduledOpMagics.count(op->GetOpMagic()) > 0;
}

int OoOScheduler::GetOOperandIdx(Operation* op, int curMemId) {
    if (op == nullptr) {
        return -1;
    }
    for (size_t i = 0; i < op->GetOOperands().size(); i++) {
        if (op->GetOOperands()[i]->memoryrange.memId == curMemId) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

void OoOScheduler::UpdateTensorInput(Operation* op, ScheduleFieldsPtr spillSrcField, LogicalTensorPtr tensor) {
    if (op == nullptr || spillSrcField == nullptr) {
        return;
    }
    for (size_t index = 0; index < op->GetIOperands().size(); index++) {
        UpdateTensorInputForOperand(op, index, spillSrcField, tensor);
    }
}

void OoOScheduler::UpdateTensorInputForOperand(Operation* op, size_t index,
    ScheduleFieldsPtr spillSrcField, LogicalTensorPtr tensor) {
    if (op == nullptr || spillSrcField == nullptr) {
        return;
    }
    // 获取 spillSrcField 对应的 Operation（需要从 opScheduleFields 反向查找）
    Operation* spillSrcOp = nullptr;
    for (const auto& pair : opScheduleFields) {
        if (pair.second == spillSrcField) {
            spillSrcOp = pair.first;
            break;
        }
    }
    if (spillSrcOp == nullptr) {
        return;
    }

    for (auto& inOp : op->GetIOperands()[index]->GetProducers()) {
        if (IsViewOp(*inOp)) {
            Operation* lastView = SkipViewChain(inOp, true);
            if (lastView != nullptr) {
                UpdateTensorInputForView(*lastView, spillSrcField, tensor);
            }
        } else if (inOp == spillSrcOp) {
            op->UpdateInputOperand(index, tensor);
        }
    }
}

void OoOScheduler::UpdateTensorInputForView(Operation& op,
    ScheduleFieldsPtr spillSrcField, LogicalTensorPtr tensor) {
    if (spillSrcField == nullptr) {
        return;
    }
    // 获取 spillSrcField 对应的 Operation
    Operation* spillSrcOp = nullptr;
    for (const auto& pair : opScheduleFields) {
        if (pair.second == spillSrcField) {
            spillSrcOp = pair.first;
            break;
        }
    }
    if (spillSrcOp == nullptr) {
        return;
    }

    bool hit = false;
    for (auto it : op.GetInputOperand(0)->GetProducers()) {
        if (it == spillSrcOp) {
            hit = true;
            op.UpdateInputOperand(0, tensor);
            break;
        }
    }
    if (!hit) return;
    // 向后刷该View链路上的MemId
    for (Operation* p = &op; p != nullptr && IsViewOp(*p); ) {
        p->GetOutputOperand(0)->memoryrange.memId = tensor->memoryrange.memId;
        auto consumers = p->GetOutputOperand(0)->GetConsumers();
        if (consumers.empty()) break;
        p = *consumers.begin();
    }
}

std::string OoOScheduler::GetOpInfo(Operation& op) {
    return op.GetOpcodeStr() + "[" + std::to_string(op.GetOpMagic()) + "]";
}

// ============================================================================
// 新的初始化函数实现（用于替代 IssueEntry 相关初始化）
// ============================================================================

Status OoOScheduler::InitScheduleCoreType(ScheduleFieldsPtr field, Operation* op,
    const std::unordered_map<Operation*, std::pair<OpCoreType, int>> &opCoreMap) {
    if (!opCoreMap.empty()) {
        field->coreLocation = opCoreMap.at(op);
        return SUCCESS;
    }
    if (op->GetCoreType() == CoreType::AIC) {
        field->coreLocation = opCoreTypeMap.at(OpCoreType::AIC);
        return SUCCESS;
    }
    if (op->GetCoreType() == CoreType::AIV) {
        field->coreLocation = opCoreTypeMap.at(OpCoreType::AIV);
        return SUCCESS;
    }
    // 对 ANY 类型进行处理
    if (op->GetOutputOperand(0)->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
        field->coreLocation = opCoreTypeMap.at(OpCoreType::AIV);
        return SUCCESS;
    }
    if (op->GetOutputOperand(0)->GetMemoryTypeOriginal() <= MemoryType::MEM_BT) {
        field->coreLocation = opCoreTypeMap.at(OpCoreType::AIC);
        return SUCCESS;
    }
    if (op->GetOutputOperand(0)->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
        if (op->GetIOperands().size() == 0 || op->GetInputOperand(0)->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
            field->coreLocation = opCoreTypeMap.at(OpCoreType::AIC);
            return SUCCESS;
        }
        if (op->GetInputOperand(0)->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
            field->coreLocation = opCoreTypeMap.at(OpCoreType::AIV);
            return SUCCESS;
        }
        if (op->GetInputOperand(0)->GetMemoryTypeOriginal() <= MemoryType::MEM_BT) {
            field->coreLocation = opCoreTypeMap.at(OpCoreType::AIC);
            return SUCCESS;
        }
        APASS_LOG_ERROR_F(Elements::Operation, "%s init coreLocation failed. IOperand memoryType is %s",
            GetOpInfo(*op).c_str(), MemoryTypeToString(op->GetInputOperand(0)->GetMemoryTypeOriginal()).c_str());
    }
    APASS_LOG_ERROR_F(Elements::Operation, "%s init coreLocation failed. OOperand memoryType is %s",
        GetOpInfo(*op).c_str(), MemoryTypeToString(op->GetOutputOperand(0)->GetMemoryTypeOriginal()).c_str());
    return FAILED;
}

void OoOScheduler::InitTensorCoreMap() {
    // 使用新的数据结构：scheduledOps 和 opScheduleFields
    for (auto op : scheduledOps) {
        auto field = GetFields(op);
        if (field && field->isAlloc) {
            auto memId = op->GetOutputOperand(0)->memoryrange.memId;
            tensorAllocCoreMap[memId] = field->coreLocation;
        }
    }
}

Status OoOScheduler::InitScheduleFields(Operation* op,
    const std::unordered_map<Operation*, std::pair<OpCoreType, int>> &opCoreMap) {
    // 处理 ViewOp 特殊情况
    if (IsViewOp(*op)) {
        if (op->GetOutputOperand(0)->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
            newOperations_.push_back(op);
        }
        return SUCCESS;
    }

    // 检查 buffer 大小
    if (CheckOpBufferSize(op) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] CheckOpBufferSize failed! %s",
            op->GetOpcodeStr().c_str(), op->GetOpMagic(), GetFormatBacktrace(*op).c_str());
        return FAILED;
    }

    // 创建 ScheduleFields 并初始化
    auto field = std::make_shared<ScheduleFields>();
    if (field == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "ScheduleFields %s, %d init failed! %s",
            op->GetOpcodeStr().c_str(), op->GetOpMagic(), GetFormatBacktrace(*op).c_str());
        return FAILED;
    }

    // 设置 type（管道类型）
    field->type = RescheduleUtils::GetOpPipeType(op);

    // 设置 isAlloc
    if (op->GetOpcodeStr().find("ALLOC") != std::string::npos) {
        field->isAlloc = true;
    }

    // 设置 viewOps
    for (auto iOperand : op->GetIOperands()) {
        for (auto pre : iOperand->GetProducers()) {
            while (IsViewOp(*pre) && pre->GetOutputOperand(0)->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
                field->viewOps.push_back(pre);
                pre = *(pre->GetInputOperand(0)->GetProducers().begin());
            }
        }
    }

    // 存储到全局 map
    opScheduleFields[op] = field;
    scheduledOps.push_back(op);
    scheduledOpMagics.insert(op->GetOpMagic());

    // 设置初始 execOrder（使用 scheduledOps 的索引）
    field->execOrder = static_cast<int>(scheduledOps.size() - 1);

    // 初始化核属性
    if (InitScheduleCoreType(field, op, opCoreMap) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ScheduleFields %s init coreType failed!", GetOpInfo(*op).c_str());
        return FAILED;
    }

    APASS_LOG_DEBUG_F(Elements::Operation, "field: %s, coreType: %s, idx: %d",
            GetOpInfo(*op).c_str(), coreTypeToString(field->coreLocation.first).c_str(), field->coreLocation.second);
    return SUCCESS;
}

// ============================================================================
// 新的依赖相关函数实现
// ============================================================================

void OoOScheduler::UpdateBufRefCount(Operation* op, ScheduleFieldsPtr field, LogicalTensorPtr tensor) {
    int memId = tensor->memoryrange.memId;
    if (tensor->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
        bufRefCount_[memId]++;
        field->reqMemIds.push_back(memId);
    }
}

Status OoOScheduler::InitBufRefCount() {
    bufRefCount_.clear();
    for (auto op : scheduledOps) {
        auto field = GetFields(op);
        if (!field) continue;
        field->Clear();
        for (auto &tensor : op->GetIOperands()) {
            UpdateBufRefCount(op, field, tensor);
            int memId = tensor->memoryrange.memId;
            if (InitLocalBuffer(tensor, memId) == FAILED) {
                APASS_LOG_ERROR_F(Elements::Operation, "InitLocalBuffer failed at InitBufRefCount!");
                return FAILED;
            }
        }
        for (auto &tensor : op->GetOOperands()) {
            UpdateBufRefCount(op, field, tensor);
            int memId = tensor->memoryrange.memId;
            if (InitLocalBuffer(tensor, memId) == FAILED) {
                APASS_LOG_ERROR_F(Elements::Operation, "InitLocalBuffer failed at InitBufRefCount!");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

void OoOScheduler::AddDependency(Operation* preOp, ScheduleFieldsPtr preField,
    Operation* postOp, ScheduleFieldsPtr postField, bool isAlloc) {
    if (preOp == nullptr || postOp == nullptr || !preField || !postField) {
        return;
    }
    if (isAlloc || (!preField->isAlloc && !postField->isAlloc)) {
        preField->successors.insert(preOp);
        postField->predecessors.insert(postOp);
    }
}

void OoOScheduler::FindDependencies(Operation* op, ScheduleFieldsPtr field) {
    // 处理 OP_L1_TO_L0A_SCALE 特殊情况
    if (op->GetOpcode() == Opcode::OP_L1_TO_L0A_SCALE) {
        auto matmulOp = *(op->GetOOperands()[0])->GetConsumers().begin();
        for (auto &input : matmulOp->GetIOperands()) {
            if (input->GetMemoryTypeOriginal() == MemoryType::MEM_L0A) {
                auto prodOp = *input->GetProducers().begin();
                auto prodField = GetFields(prodOp);
                if (prodField) {
                    AddDependency(prodOp, prodField, op, field, false);
                }
            }
        }
    }
    // 处理 OP_L1_TO_L0B_SCALE 特殊情况
    if (op->GetOpcode() == Opcode::OP_L1_TO_L0B_SCALE) {
        auto matmulOp = *(op->GetOOperands()[0])->GetConsumers().begin();
        for (auto &input : matmulOp->GetIOperands()) {
            if (input->GetMemoryTypeOriginal() == MemoryType::MEM_L0B) {
                auto prodOp = *input->GetProducers().begin();
                auto prodField = GetFields(prodOp);
                if (prodField) {
                    AddDependency(prodOp, prodField, op, field, false);
                }
            }
        }
    }
    // 处理生产者依赖
    for (auto &producer : op->ProducerOps()) {
        if (IsViewOp(*producer)) {
            for (auto viewProducer : producer->ProducerOps()) {
                Operation* lastView = SkipViewChain(viewProducer, true);
                Operation* realProd = (lastView != nullptr) ? *lastView->ProducerOps().begin() : viewProducer;
                auto viewProdField = GetFields(realProd);
                if (viewProdField) {
                    AddDependency(realProd, viewProdField, op, field, false);
                }
            }
        } else {
            auto prodField = GetFields(producer);
            if (prodField) {
                AddDependency(producer, prodField, op, field, false);
            }
        }
    }
    // 处理消费者依赖（针对 ViewOp）
    for (auto &consumer : op->ConsumerOps()) {
        if (IsViewOp(*consumer)) {
            for (auto viewConsumer : consumer->ConsumerOps()) {
                Operation* lastView = SkipViewChain(viewConsumer, false);
                Operation* realCon = (lastView != nullptr) ? *lastView->ConsumerOps().begin() : viewConsumer;
                auto viewConField = GetFields(realCon);
                if (viewConField) {
                    AddDependency(op, field, realCon, viewConField, false);
                }
            }
        }
    }
}

Status OoOScheduler::InitAllocDependencies(Operation* op, ScheduleFieldsPtr field,
    std::unordered_map<int, Operation*> &tensor2AllocOpMap) {
    for (auto &tensor : op->GetOOperands()) {
        int memId = tensor->memoryrange.memId;
        if (tensor->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
            if (tensor2AllocOpMap.find(memId) == tensor2AllocOpMap.end()) {
                APASS_LOG_ERROR_F(Elements::Operation, "Tensor[%d] must have alloc.", memId);
                return FAILED;
            }
            auto allocOp = tensor2AllocOpMap[memId];
            auto allocField = GetFields(allocOp);
            if (allocField) {
                AddDependency(allocOp, allocField, op, field, true);
            }
        }
    }
    return SUCCESS;
}

void OoOScheduler::UpdateAllocMap(Operation* op, ScheduleFieldsPtr field, std::map<int, Operation*> &tensorAllocOpMap) {
    for (auto outTensor : op->GetOOperands()) {
        if (outTensor->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        int memId = outTensor->memoryrange.memId;
        if (tensorAllocOpMap.find(memId) == tensorAllocOpMap.end()) {
            tensorAllocOpMap[memId] = op;
        }
    }
    for (auto inTensor : op->GetIOperands()) {
        if (inTensor->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        int memId = inTensor->memoryrange.memId;
        if (tensorAllocOpMap.find(memId) == tensorAllocOpMap.end()) {
            tensorAllocOpMap[memId] = op;
        }
    }
}

Status OoOScheduler::CheckAllocIssue() {
    std::map<int, Operation*> tensorAllocOpMap;
    // 先处理 Alloc 操作
    for (auto op : scheduledOps) {
        auto field = GetFields(op);
        if (!field) continue;
        if (field->isAlloc) {
            if (field->reqMemIds.size() != 1) {
                APASS_LOG_ERROR_F(Elements::Operation, "ALLOC[%d] reqMemIds size not equal to 0. %s",
                    op->GetOpMagic(), GetFormatBacktrace(*op).c_str());
                return FAILED;
            }
            UpdateAllocMap(op, field, tensorAllocOpMap);
        }
    }
    // 再处理非 Alloc 操作
    for (auto op : scheduledOps) {
        auto field = GetFields(op);
        if (!field) continue;
        if (!field->isAlloc) {
            UpdateAllocMap(op, field, tensorAllocOpMap);
        }
    }
    // 检查是否所有 tensor 都有对应的 alloc
    for (auto tensorAlloc : tensorAllocOpMap) {
        auto allocOp = tensorAlloc.second;
        auto allocField = GetFields(allocOp);
        if (allocField && !allocField->isAlloc) {
            APASS_LOG_ERROR_F(Elements::Tensor, "%s Tensor[%d] is missing Alloc.",
                GetOpInfo(*allocOp).c_str(), tensorAlloc.first);
            return FAILED;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::InitDependencies() {
    std::unordered_map<int, Operation*> tensor2AllocOpMap;
    // 第一遍：构建 op2field 映射，收集 alloc 操作
    for (auto op : scheduledOps) {
        auto field = GetFields(op);
        if (!field) continue;
        field->predecessors.clear();
        field->successors.clear();
        if (field->isAlloc) {
            if (op->GetOOperands().size() != 1) {
                APASS_LOG_ERROR_F(Elements::Operation, "Alloc[%d] oOperand must be 1.", op->GetOpMagic());
                return FAILED;
            }
            int memId = op->GetOutputOperand(0)->memoryrange.memId;
            tensor2AllocOpMap[memId] = op;
            continue;
        }
    }
    // 第二遍：构建依赖关系
    for (auto op : scheduledOps) {
        auto field = GetFields(op);
        if (!field) continue;
        if (!field->isAlloc) {
            FindDependencies(op, field);
            if (InitAllocDependencies(op, field, tensor2AllocOpMap) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "InitAllocDependencies failed.");
                return FAILED;
            }
        }
    }
    PrintDependencies();
    return SUCCESS;
}

void OoOScheduler::PrintDependencies() {
    if (static_cast<int>(LoggerManager::GetManager().level) > static_cast<int>(LoggerLevel::DEBUG)) {
        return;
    }
    for (auto op : scheduledOps) {
        auto field = GetFields(op);
        if (!field) continue;
        APASS_LOG_DEBUG_F(Elements::Operation, "%s, latency: %d.", GetOpInfo(*op).c_str(), op->GetLatency());
        for (const auto &preOp : field->predecessors) {
            APASS_LOG_DEBUG_F(Elements::Operation, "    |--- Predecessors:");
            APASS_LOG_DEBUG_F(Elements::Operation, "        |--- %s", GetOpInfo(*preOp).c_str());
        }
        for (const auto &succOp : field->successors) {
            APASS_LOG_DEBUG_F(Elements::Operation, "    |--- Successors:");
            APASS_LOG_DEBUG_F(Elements::Operation, "        |--- %s", GetOpInfo(*succOp).c_str());
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "\n");
    }
}

// ============================================================================
// 新的调度主循环函数实现
// ============================================================================

void OoOScheduler::UpdateExecOrder() {
    for (size_t idx = 0; idx < scheduledOps.size(); idx++) {
        auto field = GetFields(scheduledOps[idx]);
        if (field) {
            field->execOrder = static_cast<int>(idx);
        }
    }
}

bool OoOScheduler::IsInIssueEntries(Operation* op) {
    return IsInScheduledOps(op);
}

Status OoOScheduler::RetireIssue(Operation* op, ScheduleFieldsPtr field) {
    field->isRetired = true;
    for (auto memId : field->reqMemIds) {
        if (DelBufRefCount(memId) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "DelBufRefCount tensor[%d] failed.", memId);
            return FAILED;
        }
        if (bufRefCount_[memId] == 0) {
            auto corePair = tensorAllocCoreMap[memId];
            if (bufferManagerMap[corePair.first][corePair.second][localBufferMap[memId]->memType].Free(localBufferMap[memId]->id) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Free tensor[%d] failed.", memId);
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status OoOScheduler::FreeBuffer(Operation* op, ScheduleFieldsPtr field) {
    for (auto memId : field->reqMemIds) {
        if (DelBufRefCount(memId) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "DelBufRefCount tensor[%d] failed.", memId);
            return FAILED;
        }
        if (bufRefCount_[memId] == 0) {
            auto corePair = tensorAllocCoreMap[memId];
            UpdateBufferUsage(localBufferMap[memId]->memType, memId, true);
            if (bufferManagerMap[corePair.first][corePair.second][localBufferMap[memId]->memType].Free(localBufferMap[memId]->id) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Free tensor[%d] failed.", memId);
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status OoOScheduler::CheckAndUpdateLifecycle() {
    for (auto op : scheduledOps) {
        auto field = GetFields(op);
        if (field && !field->isRetired) {
            APASS_LOG_ERROR_F(Elements::Operation, "Unexecuted op: %s. %s", GetOpInfo(*op).c_str(), GetFormatBacktrace(*op).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

void OoOScheduler::InitScheduleQueuesAndBufferManager() {
    // 初始化新的调度队列
    for (auto [coreType, idxVec] : CORE_INIT_CONFIGS) {
        for (auto idx : idxVec) {
            for (size_t i = 0; i <= static_cast<int>(PipeType::PIPE_FIX); i++) {
                scheduleQueues[coreType][idx][static_cast<PipeType>(i)] = ScheduleQueue();
            }
        }
    }

    // 初始化 bufferManagerMap
    for (auto [coreType, idxVec] : CORE_INIT_CONFIGS) {
        for (auto idx : idxVec) {
            if (coreType == OpCoreType::AIV) {
                allocScheduleQueues[coreType][idx][MemoryType::MEM_UB] = ScheduleQueue();
                if (bufferManagerMap[coreType][idx].find(MemoryType::MEM_UB) == bufferManagerMap[coreType][idx].end()) {
                    bufferManagerMap[coreType][idx].insert({MemoryType::MEM_UB,
                        BufferPool(MemoryType::MEM_UB, localMemorySize[MemoryType::MEM_UB])});
                }
                continue;
            }
            for (size_t i = 1; i < static_cast<int>(MemoryType::MEM_DEVICE_DDR); i++) {
                allocScheduleQueues[coreType][idx][static_cast<MemoryType>(i)] = ScheduleQueue();
                if (localMemorySize.find(static_cast<MemoryType>(i)) != localMemorySize.end() &&
                    bufferManagerMap[coreType][idx].find(static_cast<MemoryType>(i)) == bufferManagerMap[coreType][idx].end()) {
                    bufferManagerMap[coreType][idx].insert({static_cast<MemoryType>(i),
                        BufferPool(static_cast<MemoryType>(i), localMemorySize[static_cast<MemoryType>(i)])});
                }
            }
        }
    }
}

Status OoOScheduler::SpillOnBlock() {
    bool didSpill = false;
    for (const auto &[coreType, idxVec] : CORE_INIT_CONFIGS) {
        for (auto idx : idxVec) {
            bool anyNotEmpty = false;
            for (auto &kv : allocScheduleQueues[coreType][idx]) {
                if (!kv.second.Empty()) {
                    anyNotEmpty = true;
                    break;
                }
            }
            if (!anyNotEmpty) {
                continue;
            }

            MemoryType spillMemType;
            if (!allocScheduleQueues[coreType][idx][MemoryType::MEM_UB].Empty()) {
                spillMemType = MemoryType::MEM_UB;
            } else if (!allocScheduleQueues[coreType][idx][MemoryType::MEM_L1].Empty()) {
                spillMemType = MemoryType::MEM_L1;
            } else {
                APASS_LOG_ERROR_F(Elements::Operation, "Buffer[L0A/B/C] is Full in SpillOnBlock.");
                continue;
            }

            // 获取需要 spill 的 op
            Operation* spillOp = allocScheduleQueues[coreType][idx][spillMemType].Front();
            if (spillOp == nullptr) {
                continue;
            }

            // 调用新的 spill 逻辑
            if (GenBufferSpill(spillOp) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "SpillOnBlock failed at GenBufferSpill for op %s", GetOpInfo(*spillOp).c_str());
                continue;
            }
            didSpill = true;
        }
    }
    if (!didSpill) {
        APASS_LOG_ERROR_F(Elements::Operation, "SpillOnBlock failed at all coreType.");
        return FAILED;
    }
    return SUCCESS;
}

void OoOScheduler::LaunchReadyIssue() {
    // 将所有就绪的 op 加入到相应的 issueQueue 中
    for (auto op : scheduledOps) {
        auto field = GetFields(op);
        if (!field) continue;
        // 跳过已退休的 op
        if (field->isRetired) continue;
        // 检查是否所有前驱都已完成
        bool allPredRetired = true;
        for (auto predOp : field->predecessors) {
            auto predField = GetFields(predOp);
            if (predField && !predField->isRetired) {
                allPredRetired = false;
                break;
            }
        }
        if (!allPredRetired) continue;

        // 将 op 加入到相应的队列中
        auto coreType = field->coreLocation.first;
        auto coreIdx = field->coreLocation.second;
        auto pipeType = field->type;

        // 插入到新的调度队列
        if (USE_LESS_OPS.find(op->GetOpcode()) != USE_LESS_OPS.end() && field->predecessors.empty()) {
            scheduleQueues[coreType][coreIdx][pipeType].Insert(op, opScheduleFields);
        }
        if (field->isAlloc) {
            auto memType = localBufferMap[field->reqMemIds[0]]->memType;
            allocScheduleQueues[coreType][coreIdx][memType].Insert(op, opScheduleFields);
        }
    }
}

Status OoOScheduler::RetireOpAndAwakeSucc(Operation* op, ScheduleFieldsPtr field, uint64_t& commitCnt) {
    commitCnt++;
    field->isRetired = true;
    if (FreeBuffer(op, field) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "FreeBuffer failed. %s", GetFormatBacktrace(*op).c_str());
        return FAILED;
    }

    // 唤醒后继 op
    for (auto succOp : field->successors) {
        auto succField = GetFields(succOp);
        if (!succField || succField->isRetired) {
            continue;
        }
        bool ready = true;
        for (auto predOp : succField->predecessors) {
            auto predField = GetFields(predOp);
            if (predField && !predField->isRetired) {
                ready = false;
                break;
            }
        }
        if (ready) {
            auto corePair = field->coreLocation;
            scheduleQueues[corePair.first][corePair.second][succField->type].Insert(succOp, opScheduleFields);
            APASS_LOG_DEBUG_F(Elements::Operation, "    Wakeup: %s, execOrder: %d", GetOpInfo(*succOp).c_str(), succField->execOrder);
        }
    }
    return SUCCESS;
}

Status OoOScheduler::RetireIssueStage(uint64_t& commitCnt, int& nextCycle) {
    // 遍历所有核，检查是否有已完成的 op
    for (auto& [coreType, idxMap] : scheduleQueues) {
        for (auto& [coreIdx, pipeMap] : idxMap) {
            for (auto& [pipeType, queue] : pipeMap) {
                if (!queue.busy) {
                    continue;
                }
                if (!pipeEndTime.count(pipeType)) {
                    pipeEndTime.emplace(pipeType, queue.curOpRetireCycle);
                } else {
                    auto curEndTime = pipeEndTime[pipeType];
                    pipeEndTime[pipeType] = std::max(curEndTime, queue.curOpRetireCycle);
                }
                if (queue.curOpRetireCycle <= clock) {
                    Operation* retiredOp = queue.curOp;
                    auto retiredField = GetFields(retiredOp);
                    queue.busy = false;
                    queue.curOp = nullptr;
                    APASS_LOG_DEBUG_F(Elements::Operation, "EXECUTE END: %s", GetOpInfo(*retiredOp).c_str());
                    if (RetireOpAndAwakeSucc(retiredOp, retiredField, commitCnt) != SUCCESS) {
                        APASS_LOG_ERROR_F(Elements::Operation, "RetireOpAndAwakeSucc failed!");
                        return FAILED;
                    }
                    continue;
                }
                APASS_LOG_DEBUG_F(Elements::Operation, "EXECUTING[%d]: %s", queue.curOpRetireCycle, GetOpInfo(*queue.curOp).c_str());
                if (nextCycle == -1 || nextCycle > queue.curOpRetireCycle) {
                    nextCycle = queue.curOpRetireCycle;
                }
            }
        }
    }
    return SUCCESS;
}

Status OoOScheduler::BufferAllocStage(uint64_t& commitCnt) {
    for (auto& [coreType, idxMap] : allocScheduleQueues) {
        for (auto& [coreIdx, memTypeMap] : idxMap) {
            for (auto& [memType, queue] : memTypeMap) {
                if (queue.Empty()) {
                    continue;
                }
                // 执行 alloc 指令直到 buffer 被占满
                bool canAlloc = true;
                while (canAlloc && !queue.Empty()) {
                    Operation* op = queue.Front();
                    auto field = GetFields(op);
                    if (!field || field->reqMemIds.empty()) {
                        queue.PopFront(opScheduleFields);
                        continue;
                    }
                    auto corePair = field->coreLocation;
                    if (!bufferManagerMap[corePair.first][corePair.second][memType].IsFull(localBufferMap[field->reqMemIds[0]])) {
                        APASS_LOG_DEBUG_F(Elements::Operation, "ALLOCATE: %s.", GetOpInfo(*op).c_str());
                        if (bufferManagerMap[corePair.first][corePair.second][memType].Allocate(localBufferMap[field->reqMemIds[0]]) != SUCCESS) {
                            APASS_LOG_ERROR_F(Elements::Tensor, "Allocate Tensor[%d] failed.", field->reqMemIds[0]);
                            return FAILED;
                        }
                        if (oooCheck.doHealthCheck) {
                            UpdateBufferUsage(memType, field->reqMemIds[0], false);
                        }
                        localBufferMap[field->reqMemIds[0]]->allocCycle = clock;
                        tensorOccupyMap[memType][field->reqMemIds[0]] = nullptr; // 使用新结构后不再需要存储IssueEntryPtr
                        queue.PopFront(opScheduleFields);
                        commitCnt++;
                        field->isRetired = true;
                        // 唤醒后继
                        for (auto succOp : field->successors) {
                            auto succField = GetFields(succOp);
                            if (!succField || succField->isRetired) continue;
                            bool ready = true;
                            for (auto predOp : succField->predecessors) {
                                auto predField = GetFields(predOp);
                                if (predField && !predField->isRetired) {
                                    ready = false;
                                    break;
                                }
                            }
                            if (ready) {
                                auto succCorePair = field->coreLocation;
                                scheduleQueues[succCorePair.first][succCorePair.second][succField->type].Insert(succOp, opScheduleFields);
                            }
                        }
                    } else {
                        canAlloc = false;
                    }
                }
            }
        }
    }
    return SUCCESS;
}

Status OoOScheduler::LaunchIssueStage(int& nextCycle) {
    // 从所有队列发射 op
    for (auto& [coreType, idxMap] : scheduleQueues) {
        for (auto& [coreIdx, pipeMap] : idxMap) {
            for (auto& [pipeType, queue] : pipeMap) {
                if (queue.Empty() || queue.busy) {
                    continue;
                }
                Operation* op = queue.PopFront(opScheduleFields);
                auto field = GetFields(op);
                if (!field) continue;

                // 标注 op 的生命周期
                op->cycleStart = clock;
                op->cycleEnd = clock + op->GetLatency();
                queue.busy = true;
                queue.curOp = op;
                queue.curOpRetireCycle = clock + op->GetLatency();
                oooCheck.pipeUsageCount[pipeType] += op->GetLatency();

                // 处理 view ops
                for (auto& viewOp : field->viewOps) {
                    if (std::find(newOperations_.begin(), newOperations_.end(), viewOp) != newOperations_.end()) {
                        continue;
                    }
                    newOperations_.emplace_back(viewOp);
                }
                newOperations_.emplace_back(op);

                if (nextCycle == -1 || nextCycle > queue.curOpRetireCycle) {
                    nextCycle = queue.curOpRetireCycle;
                }

                // 分配 tensor 内存范围
                for (auto& viewOp : field->viewOps) {
                    if (!IsViewOp(*viewOp)) {
                        APASS_LOG_ERROR_F(Elements::Operation, "op[%d] is not OP_VIEW.", viewOp->GetOpMagic());
                        return FAILED;
                    }
                    auto outTensor = viewOp->GetOOperands()[0];
                    int memId = outTensor->memoryrange.memId;
                    if (localBufferMap.find(memId) == localBufferMap.end()) {
                        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in localBufferMap.", memId);
                        return FAILED;
                    }
                    outTensor->memoryrange = TileRange(localBufferMap[memId]->start, localBufferMap[memId]->end, memId);
                }

                APASS_LOG_DEBUG_F(Elements::Operation, "Insert: %s.", GetOpInfo(*op).c_str());
            }
        }
    }
    return SUCCESS;
}

Status OoOScheduler::ScheduleMainLoop() {
    UpdateExecOrder();
    LaunchReadyIssue();
    LOG_SCOPE_BEGIN(tScheduleMainLoop, Elements::Function, "ScheduleMainLoop");
    uint64_t commitCnt = 0;
    bool isAllRetired = false;
    while (!isAllRetired) {
        int nextCycle = -1;
        APASS_LOG_DEBUG_F(Elements::Operation, "     clock: %d", clock);

        // Retire Stage
        if (RetireIssueStage(commitCnt, nextCycle) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "RetireIssueStage failed.");
            return FAILED;
        }

        // Buffer Allocation Stage
        if (BufferAllocStage(commitCnt) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "BufferAllocStage failed.");
            return FAILED;
        }

        // Launch Stage
        if (LaunchIssueStage(nextCycle) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "LaunchIssueStage failed.");
            return FAILED;
        }

        if (numTotalIssues == commitCnt && nextCycle == -1) {
            isAllRetired = true;
            break;
        }

        if (nextCycle == -1) {
            // 处理阻塞情况，执行 spill
            if (SpillOnBlock() != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "SpillOnBlock failed.");
                return FAILED;
            }
        } else {
            clock = nextCycle;
        }
    }
    LOG_SCOPE_END(tScheduleMainLoop);
    return SUCCESS;
}

// ============================================================================
// 辅助函数实现
// ============================================================================
void OoOScheduler::UpdateBufferUsage(MemoryType bufferType, int memId, bool isFree) {
    if (isFree) {
        int freeBufferSize = localBufferMap[memId]->size;
        oooCheck.bufferTotalUsage[bufferType] +=
            oooCheck.bufferLastUsage[bufferType] * (clock - oooCheck.lastClock[bufferType]);
        oooCheck.bufferLastUsage[bufferType] -= freeBufferSize;
        oooCheck.lastClock[bufferType] = clock;
    } else {
        oooCheck.bufferTotalUsage[bufferType] +=
            oooCheck.bufferLastUsage[bufferType] * (clock - oooCheck.lastClock[bufferType]);
        oooCheck.bufferLastUsage[bufferType] += localBufferMap[memId]->size;
        oooCheck.lastClock[bufferType] = clock;
        oooCheck.bufferMaxUsage[bufferType] =
            std::max(oooCheck.bufferMaxUsage[bufferType], oooCheck.bufferLastUsage[bufferType]);
    }
}

Status OoOScheduler::DelBufRefCount(const int memId) {
    if (bufRefCount_.find(memId) == bufRefCount_.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "bufRefCount cannot find Tensor[%d].", memId);
        return FAILED;
    }
    bufRefCount_[memId]--;
    if (bufRefCount_[memId] < 0) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] bufRefCount cannot less than 0.", memId);
        return FAILED;
    }
    return SUCCESS;
}

void OoOScheduler::PrintOpList(std::vector<Operation *> operations) {
    APASS_LOG_INFO_F(Elements::Operation, "==================== OP_LIST =====================");
    bool needMark = false;
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510 || !IsMixGraph(operations)) {
        needMark = true;
    }
    for (auto &op : operations) {
        if (needMark) {
            bool isCubeComponent = op->HasAttribute(OpAttributeKey::isCube) && op->GetBoolAttribute(OpAttributeKey::isCube);
            if (!isCubeComponent) {
                op->SetAIVCore(AIVCore::AIV0);
            }
        }
        if (!op->oOperand.empty()) {
            APASS_LOG_INFO_F(Elements::Operation, "%s[%d], range[%zu, %zu]", op->GetOpcodeStr().c_str(),
                op->GetOpMagic(), op->oOperand[0]->memoryrange.start, op->oOperand[0]->memoryrange.end);
        } else {
            APASS_LOG_INFO_F(Elements::Operation, "%s[%d]", op->GetOpcodeStr().c_str(), op->GetOpMagic());
        }
    }
}

uint64_t OoOScheduler::ShapeCeilAlign(std::vector<int64_t> shape, DataType dtype) {
    uint64_t bytes = 0;
    if (shape.size() == DIM_FIVE) {
        bytes = BytesPerElement(dtype) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        bytes = CeilAlign(bytes, UB_BLOCK_SIZE);
    } else {
        uint64_t preDimSize = 1;
        uint64_t last2DimSize = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            if ((shape.size() != 1) && (i < (shape.size() - LAST_TWO_DIM))) {
                preDimSize *= shape[i];
            } else {
                last2DimSize *= shape[i];
            }
        }
        bytes = preDimSize * CeilAlign(last2DimSize * BytesPerElement(dtype), UB_BLOCK_SIZE);
    }
    return bytes;
}

Status OoOScheduler::AllocViewTensorMemRange(Operation &operation) {
    auto outTensor = operation.GetOOperands()[0];
    int memId = outTensor->memoryrange.memId;
    if (localBufferMap.find(memId) == localBufferMap.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in localBufferMap.", memId);
        return FAILED;
    }
    outTensor->memoryrange =
            TileRange(localBufferMap[memId]->start, localBufferMap[memId]->end, memId);
    return SUCCESS;
}

Status OoOScheduler::InitLocalBuffer(LogicalTensorPtr oOperand, int memId) {
    if (oOperand->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
        return SUCCESS;
    }
    if (static_cast<uint64_t>(oOperand->tensor->GetRawDataSize()) != ShapeCeilAlign(oOperand->tensor->rawshape, oOperand->tensor->datatype)) {
        APASS_LOG_WARN_F(Elements::Tensor, "InitLocalBuffer Failed at ShapeCeilAlign! "
            "Please ensure that the rawTensor[%d] shapes are aligned.", oOperand->GetRawMagic());
    }
    if (localBufferMap.find(memId) == localBufferMap.end()) {
        localBufferMap[memId] = std::make_shared<LocalBuffer>(
            memId, oOperand->tensor->GetRawDataSize(), oOperand->GetMemoryTypeOriginal());
    } else {
        localBufferMap[memId]->size =
            std::max(localBufferMap[memId]->size, static_cast<uint64_t>(oOperand->tensor->GetRawDataSize()));
    }
    return SUCCESS;
}

Status OoOScheduler::CalcBufferSize(LogicalTensors tensors, std::map<MemoryType, int64_t> &bufferSize, std::set<int> &memIdMap) {
    for (auto tensor : tensors) {
        if (tensor->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        const auto &shape = tensor->tensor->GetRawShape();
        if (std::any_of(shape.begin(), shape.end(), [](int64_t d) {return d <= 0;})) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Dynamic axis detected in %s, "
                "OoOSchedule requires static rawShape.", tensor->Dump().c_str());
            return FAILED;
        }
        if (memIdMap.find(tensor->memoryrange.memId) == memIdMap.end()) {
            bufferSize[tensor->GetMemoryTypeOriginal()] += tensor->tensor->GetRawDataSize();
            memIdMap.insert(tensor->memoryrange.memId);
        }
    }
    return SUCCESS;
}

std::string OoOScheduler::dumpOpInfo(Operation &op) {
    std::ostringstream oss;
    oss << "OP: " << op.GetOpcodeStr().c_str() << "[" << op.GetOpMagic() << "] | ";
    oss << "Inputs: {";
    for (size_t i = 0; i < op.iOperand.size(); i++) {
        oss << "RawTensor[" << op.GetInputOperand(i)->tensor->GetRawMagic() << "] ";
        oss << op.iOperand[i]->tensor->DumpSSA(true, true);
        if (i != op.iOperand.size() - 1) {
            oss << ", ";
        }
    }
    oss << "}" << " | ";
    oss << "Outputs: {";
    for (size_t i = 0; i < op.oOperand.size(); i++) {
        oss << "RawTensor[" << op.GetOutputOperand(i)->tensor->GetRawMagic() << "] ";
        oss << op.oOperand[i]->tensor->DumpSSA(true, true);
        if (i != op.oOperand.size() - 1) {
            oss << ", ";
        }
    }
    oss << "}";
    return oss.str();
}

Status OoOScheduler::CheckOpBufferSize(Operation *op) {
    std::map<MemoryType, int64_t> bufferSize;
    std::set<int> memIdMap;
    if (CalcBufferSize(op->GetIOperands(), bufferSize, memIdMap) != SUCCESS || CalcBufferSize(op->GetOOperands(), bufferSize, memIdMap) != SUCCESS) {
        return FAILED;
    }
    for (auto &buffer : bufferSize) {
        if (localMemorySize.find(buffer.first) == localMemorySize.end()) {
            continue;
        }
        if (buffer.second <= localMemorySize[buffer.first]) {
            continue;
        }
        if (op->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            APASS_LOG_ERROR_F(Elements::Operation, "Alloc tensor[%d] size[%ld] exceeds %s size[%ld]! %s",
                op->GetOutputOperand(0)->GetMagic(), buffer.second, MemoryTypeToString(buffer.first).c_str(),
                localMemorySize[buffer.first], GetFormatBacktrace(*op).c_str());
            APASS_LOG_ERROR_F(Elements::Operation, "Tensor[%d] producer info:", op->GetOutputOperand(0)->GetMagic());
            for (auto producer : op->GetOutputOperand(0)->GetProducers()) {
                if (producer == op) {
                    continue;
                }
                APASS_LOG_ERROR_F(Elements::Operation, "    %s.", dumpOpInfo(*producer).c_str());
            }
        } else {
            APASS_LOG_ERROR_F(Elements::Operation, "OP %s[%d] in/output total size[%ld] exceeds %s size[%ld]!",
                op->GetOpcodeStr().c_str(), op->GetOpMagic(), buffer.second, MemoryTypeToString(buffer.first).c_str(),
                localMemorySize[buffer.first]);
            APASS_LOG_ERROR_F(Elements::Operation, "%s.", dumpOpInfo(*op).c_str());
        }
        return FAILED;
    }
    return SUCCESS;
}

void OoOScheduler::InitCoreConfig(const std::vector<Operation *> &operations) {
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510 || !IsMixGraph(operations)) {
        CORE_INIT_CONFIGS = CORE_INIT_CONFIGS_HARDWARE_ONE;
    } else {
        CORE_INIT_CONFIGS = CORE_INIT_CONFIGS_HARDWARE_TWO;
    }
}

// ============================================================================
// 主入口函数
// ============================================================================
Status OoOScheduler::Init(const std::vector<Operation *> &operations, const std::unordered_map<Operation*, std::pair<OpCoreType, int>> &opCoreMap,
    const std::unordered_map<OpCoreType, std::vector<int>> fixCoreConfig) {
    // 清理数据结构
    scheduledOps.clear();
    opScheduleFields.clear();
    scheduledOpMagics.clear();
    localBufferMap.clear();
    depthCache_.clear();
    LOG_SCOPE_BEGIN(tInit, Elements::Function, "Init");
    // 初始化芯片各buffer大小
    localMemorySize = CommonUtils::GetLocalMemorySize();
    if (fixCoreConfig.empty()) {
        InitCoreConfig(operations);
    } else {
        CORE_INIT_CONFIGS = fixCoreConfig;
    }
    // 校验并初始化 ScheduleFields
    for (const auto &op : operations) {
        if (InitScheduleFields(op, opCoreMap) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Operation %s[%d] init schedule fields failed!", op->GetOpcodeStr().c_str(), op->GetOpMagic());
            return FAILED;
        }
    }
    numTotalIssues = scheduledOps.size();

    // 初始化 buffer 引用计数
    if (InitBufRefCount() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitBufRefCount failed!");
        return FAILED;
    }
    // 初始化依赖关系
    if (InitDependencies() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitDependencies failed!");
        return FAILED;
    }
    if (CheckAllocIssue() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "CheckAllocIssue failed!");
        return FAILED;
    }
    InitTensorCoreMap();
    // 初始化内存管理器
    InitScheduleQueuesAndBufferManager();
    LOG_SCOPE_END(tInit);
    return SUCCESS;
}

Status OoOScheduler::Schedule(const std::vector<Operation *> &operations, const std::unordered_map<Operation*, std::pair<OpCoreType, int>> &opCoreMap,
    const std::unordered_map<OpCoreType, std::vector<int>> fixCoreConfig) {
    if (operations.empty()) {
        return SUCCESS;
    }
    PrintOpList(operations);
    if (Init(operations, opCoreMap, fixCoreConfig) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Init failed!");
        return FAILED;
    }
    // 生成spill指令
    if (GenSpillSchedule() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "GenSpillSchedule failed!");
        return FAILED;
    }
    // 模拟调度
    if (ScheduleMainLoop() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ScheduleMainLoop failed!");
        return FAILED;
    }
    if (CheckAndUpdateLifecycle() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "CheckAndUpdateLifecycle failed!");
        return FAILED;
    }
    for (size_t i = 0; i < operations.size(); i++) {
        if (operations[i]->GetOpcode() == Opcode::OP_L1_TO_L0B_SCALE) {
            auto l0MxOut = operations[i]->GetOOperands()[0];
            auto consOp = *l0MxOut->GetConsumers().begin();
            LogicalTensorPtr l0ATensor, l0BTensor, l0AMXTensor, l0BMXTensor;
            for (auto &l0Tensor : consOp->GetIOperands()) {
                if (l0Tensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0A) {
                    l0ATensor = l0Tensor;
                } else if (l0Tensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0B) {
                    l0BTensor = l0Tensor;
                } else if (l0Tensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0AMX) {
                    l0AMXTensor = l0Tensor;
                } else if (l0Tensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0BMX) {
                    l0BMXTensor = l0Tensor;
                }
            }
            l02L0MXMap_[l0ATensor] = l0AMXTensor;
            l02L0MXMap_[l0BTensor] = l0BMXTensor;
        }
    }
    for (auto &entry : l02L0MXMap_) {
        auto l0Tensor = entry.first;
        auto l0MXTensor = entry.second;
        int l0MemID = l0Tensor->memoryrange.memId;
        int l0MemMXID = l0MXTensor->memoryrange.memId;
        l0MXTensor->memoryrange = TileRange(localBufferMap[l0MemID]->start >> 4, localBufferMap[l0MemID]->end >> 4, l0MemMXID);
    }
    PrintOpList(newOperations_);
    function_.SetStackWorkespaceSize(workspaceOffset);
    function_.pipeEndTime = pipeEndTime;
    return SUCCESS;
}

// ============================================================================
// Buffer rearrange 相关函数（使用旧数据结构，待后续迁移）
// ============================================================================
void OoOScheduler::UpdateMoveOpAttr(Operation &moveOp, Operation &occupyOp) {
    if (moveOp.GetOpcode() == Opcode::OP_COPY_IN && occupyOp.GetOpcode() == Opcode::OP_COPY_IN) {
        moveOp.SetOpAttribute(occupyOp.GetOpAttribute()->Clone());
        moveOp.inParamLocation_ = occupyOp.inParamLocation_;
        moveOp.SetIOpAttrOffset(0, occupyOp.GetIOpAttrOffset(0));
    } else if (moveOp.GetOpcode() == Opcode::OP_ADDS) {
        moveOp.SetAttr(OpAttributeKey::scalar, Element(DataType::DT_UINT64, 0));
        if (moveOp.GetIOperands()[0]->tensor->rawshape.back() == 1) {
            std::vector<bool> attrIn{true};
            moveOp.SetAttr(OpAttributeKey::inputCombineAxis, attrIn);
        }
        if (moveOp.GetOOperands()[0]->tensor->rawshape.back() == 1) {
            std::vector<bool> attrOut{true};
            moveOp.SetAttr(OpAttributeKey::outputCombineAxis, attrOut);
        }
    }
    if (occupyOp.GetInternalSubgraphID() != NOT_IN_SUBGRAPH) {
        moveOp.UpdateInternalSubgraphID(occupyOp.GetInternalSubgraphID());
        moveOp.SetAIVCore(occupyOp.GetAIVCore());
    }
}

Status OoOScheduler::FindMoveFromTensor(Operation &occupyOp, int oldMemId, MemoryType memType, bool &rearrangeUBBF16, LogicalTensorPtr &moveFromTensor) {
    for (auto outTensorPtr : occupyOp.GetOOperands()) {
        if (outTensorPtr->memoryrange.memId == oldMemId) {
            moveFromTensor = outTensorPtr;
            break;
        }
    }
    if (moveFromTensor == nullptr) {
        APASS_LOG_WARN_F(Elements::Tensor, "Cannot find tensor(memId: %d) according to tensorOccupyMap, GenRearrangeCopyOp failed", oldMemId);
        return FAILED;
    }
    // 如果moveFrom Tensor是UB且数据类型为bf16, rearrange失败
    if (memType == MemoryType::MEM_UB && moveFromTensor->Datatype() == DataType::DT_BF16) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Cannot rearrange UB tensor with datatype bf16, do schedulemainloop spill.");
        rearrangeUBBF16 = true;
    }
    return SUCCESS;
}

Status OoOScheduler::GetMoveOpInTensor(Opcode moveOpcode, Operation &occupyOp, LogicalTensorPtr &inTensor, LogicalTensorPtr &moveFromTensor) {
    if (moveOpcode == Opcode::OP_COPY_IN) {
        if (occupyOp.GetOpcode() != Opcode::OP_COPY_IN) {
            APASS_LOG_WARN_F(Elements::Operation, "Occupy op is not COPY_IN, GetMoveOpInTensor failed.");
            return FAILED;
        }
        inTensor = occupyOp.GetIOperands()[0];
        if (inTensor == nullptr || inTensor->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
            APASS_LOG_WARN_F(Elements::Tensor, "inTensor is illegal, GetMoveOpInTensor failed.");
            return FAILED;
        }
    } else {
        inTensor = moveFromTensor;
    }
    return SUCCESS;
}

Status OoOScheduler::UpdateRange(int newMemId, size_t offset, MemoryType memType, BufferPool &bufferManager) {
    auto moveToBufferPtr = localBufferMap[newMemId];
    if (bufferManager.ModifyBufferRange(moveToBufferPtr, offset) != SUCCESS) {
        APASS_LOG_WARN_F(Elements::Tensor, "UpdateRange failed at ModifyBufferRange.");
        return FAILED;
    }
    if (oooCheck.doHealthCheck) {
        UpdateBufferUsage(memType, newMemId, false);
    }
    // Note: tensorOccupyMap access needs update for new structure
    // tensorOccupyMap[memType][newMemId]->tileOp.GetOOperands()[0]->memoryrange =
    //     TileRange(offset, offset + moveToBufferPtr->size, newMemId);
    localBufferMap[newMemId]->startCycle = clock;
    return SUCCESS;
}

} // namespace npu::tile_fwk
