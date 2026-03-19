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
 * \file spill_buffer.cpp
 * \brief
 */

#include "scheduler.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "OoOSchedule"

namespace npu::tile_fwk {

constexpr int32_t TWO_ISSUE = 2;
constexpr int32_t DEFAULT_LATENCY = 511;

OoOSchedulerCheck::SpillInfo OoOScheduler::RecordSpillInfo(MemoryType bufferType, int memId,
    LocalBufferPtr allocBuffer, LogicalTensorPtr spillOutTensor, bool needCopyOut) {
    OoOSchedulerCheck::SpillInfo spillInfo;
    spillInfo.spillType = bufferType;
    spillInfo.bufferCurrUsage = oooCheck.bufferLastUsage[bufferType];
    spillInfo.spillTensorSize = localBufferMap[memId]->size;
    spillInfo.spillTensorMagic = spillOutTensor->GetMagic();
    spillInfo.triggerTensorSize = allocBuffer->size;
    int allocOccupied = 0;
    for (const auto &pair : tensorOccupyMap[bufferType]) {
        if (pair.second->isAlloc) {
            allocOccupied += localBufferMap[pair.first]->size;
        }
    }
    spillInfo.allocOccupiedSize = allocOccupied;
    if (needCopyOut) {
        auto dtype = spillOutTensor->tensor->datatype;
        spillInfo.spillCopyoutSize = std::accumulate(spillOutTensor->shape.begin(), spillOutTensor->shape.end(),
            1, std::multiplies<int64_t>()) * BytesOf(dtype);
    } else {
        spillInfo.spillCopyoutSize = 0;
    }
    return spillInfo;
}

// ============================================================================
// 新的Spill函数实现（用于替代 IssueEntry 相关spill函数）
// ============================================================================

int OoOScheduler::GetBufNextUseOrder(Operation* op, int curMemId) {
    auto field = GetFields(op);
    if (!field) return -1;

    auto it = std::find_if(scheduledOps.begin(), scheduledOps.end(), [op, curMemId, this](Operation* a) {
        auto aField = GetFields(a);
        return aField && aField->execOrder > GetExecOrder(op) &&
            std::find(aField->reqMemIds.begin(), aField->reqMemIds.end(), curMemId) != aField->reqMemIds.end();
    });
    return (it != scheduledOps.end()) ? GetExecOrder(*it) : -1;
}

int OoOScheduler::GetBufLastUseOrder(Operation* op, int curMemId) {
    auto targetIt = std::find(scheduledOps.begin(), scheduledOps.end(), op);
    if (targetIt == scheduledOps.end()) {
        return -1;
    }
    auto field = GetFields(op);
    if (!field) return -1;

    for (auto it = std::make_reverse_iterator(targetIt); it != scheduledOps.rend(); it++) {
        Operation* curOp = *it;
        auto curField = GetFields(curOp);
        if (curField && curField->execOrder < field->execOrder &&
            std::find(curField->reqMemIds.begin(), curField->reqMemIds.end(), curMemId) != curField->reqMemIds.end()) {
            return curField->execOrder;
        }
    }
    return -1;
}

Operation* OoOScheduler::GetBufLastWriteOp(Operation* op, int curMemId) {
    auto targetIt = std::find(scheduledOps.begin(), scheduledOps.end(), op);
    if (targetIt == scheduledOps.end()) {
        return nullptr;
    }
    auto field = GetFields(op);
    if (!field) return nullptr;

    for (auto it = std::make_reverse_iterator(targetIt); it != scheduledOps.rend(); it++) {
        Operation* curOp = *it;
        auto curField = GetFields(curOp);
        if (curField == nullptr || curField->execOrder >= field->execOrder) {
            continue;
        }
        for (auto& outTensor : curOp->GetOOperands()) {
            if (outTensor->memoryrange.memId == curMemId) {
                return curOp;
            }
        }
    }
    return nullptr;
}

void OoOScheduler::InsertScheduledOps(Operation* insertOp, int insertOrder) {
    auto it = scheduledOps.begin();
    for (; it != scheduledOps.end(); it++) {
        auto field = GetFields(*it);
        if (field && field->execOrder >= insertOrder) {
            break;
        }
    }
    // 记录插入位置的索引
    size_t insertIdx = static_cast<size_t>(std::distance(scheduledOps.begin(), it));
    scheduledOps.insert(it, insertOp);
    scheduledOpMagics.insert(insertOp->GetOpMagic());

    // 只更新插入位置及其之后的 execOrder
    for (size_t idx = insertIdx; idx < scheduledOps.size(); idx++) {
        auto field = GetFields(scheduledOps[idx]);
        if (field) {
            field->execOrder = static_cast<int>(idx);
        }
    }
}

void OoOScheduler::ReplaceTensorMemId(Operation* op, ScheduleFieldsPtr field, int oldMemId, int newMemId) {
    for (auto& memId : field->reqMemIds) {
        if (memId == oldMemId) {
            memId = newMemId;
        }
    }
    for (auto& outTensor : op->GetOOperands()) {
        if (outTensor->memoryrange.memId == oldMemId) {
            outTensor->memoryrange.memId = newMemId;
        }
    }
}

Status OoOScheduler::UpdateRemainOpBufId(int oldMemId, int newMemId) {
    if (bufRefCount_.find(oldMemId) == bufRefCount_.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "bufRefCount cannot find Tensor[%d].", oldMemId);
        return FAILED;
    }
    bufRefCount_[newMemId] = bufRefCount_[oldMemId] + TWO_ISSUE;
    bufRefCount_[oldMemId] = 0;
    for (auto op : scheduledOps) {
        auto field = GetFields(op);
        if (!field || field->isRetired) {
            continue;
        }
        ReplaceTensorMemId(op, field, oldMemId, newMemId);
    }
    return SUCCESS;
}

Status OoOScheduler::GenSpillSchedule() {
    UpdateExecOrder();
    size_t pcIdx = 0;
    LOG_SCOPE_BEGIN(tGenSpillSchedule, Elements::Function, "GenSpillSchedule");
    while (pcIdx < scheduledOps.size()) {
        auto op = scheduledOps[pcIdx];
        auto field = GetFields(op);
        if (!field) {
            pcIdx++;
            continue;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "Launch %s", GetOpInfo(*op).c_str());
        if (field->isAlloc) {
            if (ExecuteAllocIssue(op, field, pcIdx) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "ExecuteAllocIssue failed! %s", GetFormatBacktrace(*op).c_str());
                return FAILED;
            }
        }
        if (RetireIssue(op, field) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "RetireIssue failed! %s", GetFormatBacktrace(*op).c_str());
            return FAILED;
        }
        pcIdx += 1;
    }
    for (auto bufRef : bufRefCount_) {
        if (bufRef.second != 0) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] bufRefCount not equal to 0!", bufRef.first);
            return FAILED;
        }
    }
    LOG_SCOPE_END(tGenSpillSchedule);
    if (InitBufRefCount() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitBufRefCount failed!");
        return FAILED;
    }
    // 更新依赖关系
    if (InitDependencies() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitDependencies failed!");
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::ExecuteAllocIssue(Operation* op, ScheduleFieldsPtr field, size_t &pcIdx) {
    if (field->reqMemIds.empty()) {
        return SUCCESS;
    }
    if (localBufferMap.find(field->reqMemIds[0]) == localBufferMap.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in localBufferMap!", field->reqMemIds[0]);
        return FAILED;
    }
    LocalBufferPtr allocBuffer = localBufferMap[field->reqMemIds[0]];
    auto corePair = field->coreLocation;
    if (bufferManagerMap[corePair.first][corePair.second][allocBuffer->memType].IsFull(allocBuffer)) {
        if (GenSpillOp(pcIdx) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "GenSpillOp failed at ExecuteAllocIssue. %s", GetFormatBacktrace(*op).c_str());
            return FAILED;
        }
    }
    if (bufferManagerMap[corePair.first][corePair.second][allocBuffer->memType].Allocate(allocBuffer) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Allocate tensor[%d] failed.", allocBuffer->id);
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::GenSpillOp(size_t &pcIdx) {
    return GenSpillOpImpl(pcIdx);
}

Status OoOScheduler::CreateSpillCopyout(Operation* spillOp, ScheduleFieldsPtr spillField,
    LogicalTensorPtr spillTensor, int spillMemId, Operation* &spillCopyout) {
    // 创建 DDR tensor
    std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(
        spillTensor->Datatype(), spillTensor->tensor->rawshape,
        TileOpFormat::TILEOP_ND, "WorkspaceGm", SYMBOL_STACK_BASE);
    if (ddrRawTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Create DDR raw tensor failed!");
        return FAILED;
    }

    std::vector<int64_t> offset = spillTensor->GetOffset();
    LogicalTensorPtr ddrTensor = std::make_shared<LogicalTensor>(function_, ddrRawTensor, offset, spillTensor->GetShape());
    if (ddrTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Create DDR tensor failed!");
        return FAILED;
    }

    // 保存当前 workspaceOffset，用于 UpdateOpAttr
    int64_t workspaceOffsetTemp = workspaceOffset;

    // 更新 tensor 属性
    ddrTensor->SetMemoryTypeToBe(MEM_DEVICE_DDR);
    ddrTensor->SetMemoryTypeOriginal(MEM_DEVICE_DDR);
    ddrTensor->oriShape = spillTensor->oriShape;
    ddrTensor->UpdateDynValidShape(spillTensor->GetDynValidShape());
    ddrTensor->tensor->rawshape = spillTensor->tensor->rawshape;
    ddrTensor->memoryrange = TileRange(workspaceOffset, workspaceOffset + localBufferMap[spillMemId]->size, workspaceMemId++);
    workspaceOffset += localBufferMap[spillMemId]->size;

    // 创建 COPY_OUT 操作
    Operation &spillOutOp = function_.AddRawOperation(Opcode::OP_COPY_OUT, {spillTensor}, {ddrTensor});
    spillOutOp.UpdateLatency(DEFAULT_LATENCY);

    // 设置 CopyOpAttribute（与旧代码 UpdateOpAttr 逻辑一致）
    spillOutOp.SetAttr(OpAttributeKey::workspaceBaseOffset, workspaceOffsetTemp);
    spillOutOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(spillTensor->GetMemoryTypeOriginal(),
        OpImmediate::Specified(offset), OpImmediate::Specified(spillTensor->GetShape()),
        OpImmediate::Specified(spillTensor->GetRawTensor()->GetDynRawShape())));

    // 创建新的 ScheduleFields
    auto copyoutField = std::make_shared<ScheduleFields>();
    copyoutField->type = PipeType::PIPE_M;
    copyoutField->reqMemIds = {spillMemId};
    copyoutField->coreLocation = spillField->coreLocation;
    copyoutField->isRetired = true;  // 标记为已退休

    // 设置依赖关系
    copyoutField->predecessors.insert(spillOp);
    spillField->successors.insert(&spillOutOp);

    // 存储到全局 map
    opScheduleFields[&spillOutOp] = copyoutField;
    spillCopyout = &spillOutOp;

    APASS_LOG_DEBUG_F(Elements::Operation, "Add SPILL_OUT: %s.", GetOpInfo(spillOutOp).c_str());
    return SUCCESS;
}

Status OoOScheduler::SpillOutBuffer(SpillInfo &spillInfo, Operation* allocOp, size_t &pcIdx, bool isGenSpill) {
    // 如果是 COPY_IN，直接使用其输入 tensor
    if (spillInfo.spillOp_->GetOpcodeStr().find("COPY_IN") != std::string::npos) {
        spillInfo.ddrTensor_ = spillInfo.spillOp_->GetInputOperand(0);
        return SUCCESS;
    }

    Operation* spillCopyout = nullptr;
    int bufLastUseOrder = -1;

    // 处理 A5 芯片上的特殊 L1 spill
    if (spillInfo.isSpecialL1_) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Start to spill-out special L1 in A5.");
        auto spillOp = spillInfo.spillOp_;
        auto preTensor = spillOp->GetInputOperand(0);
        if (spillOp->GetOpcode() != Opcode::OP_RESHAPE &&
            preTensor->GetMemoryTypeOriginal() != MemoryType::MEM_UB &&
            preTensor->GetMemoryTypeOriginal() != MemoryType::MEM_L0C) {
            APASS_LOG_ERROR_F(Elements::Operation, "spillOp %s is not RESHAPE/UB/L0C in A5 L1 spill",
                GetOpInfo(*spillOp).c_str());
            return FAILED;
        }

        // 找到实际的 spill issue（非 ALLOC 的前驱）
        Operation* actualSpillOp = nullptr;
        auto spillField = GetFields(spillOp);
        if (spillField) {
            for (auto* pred : spillField->predecessors) {
                auto predField = GetFields(pred);
                if (predField && !predField->isAlloc) {
                    actualSpillOp = pred;
                    break;
                }
            }
        }

        // RESHAPE 特殊处理：检查是否可以直接使用 DDR tensor
        if (spillOp->GetOpcode() == Opcode::OP_RESHAPE && actualSpillOp) {
            if (actualSpillOp->GetOpcodeStr().find("COPY_IN") != std::string::npos) {
                // DDR->copy_in->L1->reshape->L1 场景
                spillInfo.ddrTensor_ = actualSpillOp->GetInputOperand(0);
                APASS_LOG_DEBUG_F(Elements::Operation, "Spill out finish in A5: DDR->copy_in->L1->reshape->L1");
                return SUCCESS;
            }
        }

        // 对于其他特殊情况，使用实际的 spill tensor
        if (actualSpillOp) {
            LogicalTensorPtr actualSpillTensor = preTensor;
            if (spillOp->GetOpcode() == Opcode::OP_RESHAPE) {
                actualSpillTensor = actualSpillOp->GetInputOperand(0);
                // 找到 actualSpillOp 的非 ALLOC 前驱
                auto actualSpillField = GetFields(actualSpillOp);
                if (actualSpillField) {
                    for (auto* pred : actualSpillField->predecessors) {
                        auto predField = GetFields(pred);
                        if (predField && !predField->isAlloc) {
                            actualSpillOp = pred;
                            break;
                        }
                    }
                }
            }

            if (actualSpillOp->GetOpcodeStr().find("COPY_IN") != std::string::npos) {
                APASS_LOG_ERROR_F(Elements::Operation, "A5 does not support the COPY_IN-actualSpillOp.");
                return FAILED;
            }

            if (CreateSpillCopyout(actualSpillOp, GetFields(actualSpillOp),
                actualSpillTensor, actualSpillTensor->memoryrange.memId, spillCopyout) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "CreateSpillCopyout failed for specialL1 spill!");
                return FAILED;
            }
            bufLastUseOrder = GetBufLastUseOrder(allocOp, actualSpillTensor->memoryrange.memId);
        } else {
            APASS_LOG_ERROR_F(Elements::Operation, "Cannot find actual spill op for special L1.");
            return FAILED;
        }
    } else {
        // 普通情况
        if (CreateSpillCopyout(spillInfo.spillOp_, GetFields(spillInfo.spillOp_),
            spillInfo.spillTensor_, spillInfo.spillMemId_, spillCopyout) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "CreateSpillCopyout failed!");
            return FAILED;
        }
        bufLastUseOrder = GetBufLastUseOrder(allocOp, spillInfo.spillMemId_);
    }

    if (spillCopyout == nullptr) {
        return SUCCESS;
    }

    if (bufLastUseOrder == -1) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] last used order.", spillInfo.spillMemId_);
        return FAILED;
    }

    // 设置 COPY_OUT 模式
    if (UpdateCopyOutMode(*spillCopyout) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateCopyOutMode failed!");
        return FAILED;
    }

    // 插入到 scheduledOps
    auto allocField = GetFields(allocOp);
    auto copyoutField = GetFields(spillCopyout);
    copyoutField->execOrder = bufLastUseOrder + 1;
    InsertScheduledOps(spillCopyout, copyoutField->execOrder);

    if (isGenSpill) {
        pcIdx++;
        numTotalIssues++;
    } else {
        newOperations_.push_back(spillCopyout);
    }

    spillInfo.ddrTensor_ = spillCopyout->GetOutputOperand(0);
    return SUCCESS;
}

Status OoOScheduler::SpillInBuffer(SpillInfo &spillInfo, Operation* allocOp, MemoryType bufferType,
    bool isGenSpill) {
    // A5 中 L1->reshape->L1 时第二个 L1 为 spill tensor 的特殊情况
    if (spillInfo.isSpecialL1_ && spillInfo.spillOp_->GetOpcodeStr().find("RESHAPE") != std::string::npos) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Start to spill-reshape special L1 in A5.");
        // 创建 reshape tensor
        LogicalTensorPtr reshapeTensor = std::make_shared<LogicalTensor>(function_,
            spillInfo.spillTensor_->Datatype(), spillInfo.spillTensor_->shape, spillInfo.spillTensor_->Format());
        if (reshapeTensor == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Create reshape tensor failed!");
            return FAILED;
        }

        // 更新 reshape tensor 属性
        reshapeTensor->SetMemoryTypeToBe(spillInfo.spillTensor_->GetMemoryTypeOriginal());
        reshapeTensor->SetMemoryTypeOriginal(spillInfo.spillTensor_->GetMemoryTypeOriginal());
        reshapeTensor->oriShape = spillInfo.spillTensor_->oriShape;
        reshapeTensor->UpdateDynValidShape(spillInfo.spillTensor_->GetDynValidShape());
        reshapeTensor->tensor->rawshape = spillInfo.spillTensor_->tensor->rawshape;
        int reshapeMagic = reshapeTensor->GetRawTensor()->GetRawMagic();
        reshapeTensor->memoryrange.memId = reshapeMagic;
        localBufferMap[reshapeMagic] = std::make_shared<LocalBuffer>(
            reshapeMagic, reshapeTensor->tensor->GetRawDataSize(), spillInfo.spillTensor_->GetMemoryTypeOriginal());
        if (localBufferMap[reshapeMagic] == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Init Tensor[%d] localBuffer failed.", reshapeMagic);
            return FAILED;
        }
        tensorAllocCoreMap[reshapeMagic] = tensorAllocCoreMap[spillInfo.spillTensor_->memoryrange.memId];

        // 更新后继 op 的 tensor 输入
        auto spillField = GetFields(spillInfo.spillOp_);
        if (spillField) {
            for (auto* succOp : spillField->successors) {
                auto succField = GetFields(succOp);
                if (!succField || succField->isRetired) continue;
                if (std::count(succField->reqMemIds.begin(), succField->reqMemIds.end(), spillInfo.spillMemId_) > 0) {
                    UpdateTensorInput(succOp, spillField, reshapeTensor);
                }
            }
        }

        auto allocField = GetFields(allocOp);
        auto corePair = allocField->coreLocation;
        int bufNextUseOrder = GetBufNextUseOrder(allocOp, spillInfo.spillMemId_);
        if (bufNextUseOrder == -1) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Get Tensor[%d] next use order failed.", spillInfo.spillMemId_);
            return FAILED;
        }

        // 获取 spillOp 的前驱 COPY_IN（用于获取属性）
        Operation* preCopyInOp = nullptr;
        if (spillField) {
            for (auto* pred : spillField->predecessors) {
                auto predField = GetFields(pred);
                if (predField && !predField->isAlloc) {
                    preCopyInOp = pred;
                    break;
                }
            }
        }

        // 创建 ALLOC
        auto &spillAllocOp = function_.AddRawOperation(Opcode::OP_L1_ALLOC, {}, {reshapeTensor});
        spillAllocOp.UpdateLatency(1);
        auto spillAllocField = std::make_shared<ScheduleFields>();
        spillAllocField->type = PipeType::PIPE_M;
        spillAllocField->isAlloc = true;
        spillAllocField->reqMemIds = {reshapeMagic};
        spillAllocField->coreLocation = corePair;
        spillAllocField->execOrder = bufNextUseOrder++;
        opScheduleFields[&spillAllocOp] = spillAllocField;
        InsertScheduledOps(&spillAllocOp, spillAllocField->execOrder);

        // 创建 COPY_IN
        auto iOperand = spillInfo.spillOp_->GetInputOperand(0);
        auto &spillCopyInOp = (preCopyInOp && preCopyInOp->GetOpcode() == Opcode::OP_COPY_IN) ?
            preCopyInOp->CloneOperation(function_, {spillInfo.ddrTensor_}, {reshapeTensor}) :
            function_.AddRawOperation(Opcode::OP_COPY_IN, {spillInfo.ddrTensor_}, {reshapeTensor});
        if (preCopyInOp && preCopyInOp->GetOpcode() == Opcode::OP_COPY_IN) {
            spillCopyInOp.SetIOpAttrOffset(0, preCopyInOp->GetIOpAttrOffset(0));
        }
        int64_t base = 0;
        GetWorkspaceBaseOffset(spillInfo.ddrTensor_, base);
        spillCopyInOp.SetAttr(OpAttributeKey::workspaceBaseOffset, base);
        spillCopyInOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
            OpImmediate::Specified(iOperand->GetOffset()),
            iOperand->GetMemoryTypeOriginal(),
            OpImmediate::Specified(iOperand->GetShape()),
            OpImmediate::Specified(iOperand->tensor->GetDynRawShape())));
        spillCopyInOp.UpdateLatency(DEFAULT_LATENCY);
        if ((!preCopyInOp || preCopyInOp->GetOpcode() != Opcode::OP_COPY_IN) &&
            UpdateCopyInMode(spillCopyInOp) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "UpdateCopyInMode failed.");
            return FAILED;
        }
        auto spillCopyInField = std::make_shared<ScheduleFields>();
        spillCopyInField->type = PipeType::PIPE_M;
        spillCopyInField->reqMemIds = {reshapeMagic};
        spillCopyInField->coreLocation = corePair;
        spillCopyInField->execOrder = bufNextUseOrder++;
        spillCopyInField->predecessors.insert(&spillAllocOp);
        spillAllocField->successors.insert(&spillCopyInOp);
        opScheduleFields[&spillCopyInOp] = spillCopyInField;
        InsertScheduledOps(&spillCopyInOp, spillCopyInField->execOrder);

        // 创建 RESHAPE
        auto &reshapeOp = function_.AddRawOperation(Opcode::OP_RESHAPE, {reshapeTensor}, {reshapeTensor});
        reshapeOp.UpdateLatency(1);
        auto reshapeField = std::make_shared<ScheduleFields>();
        reshapeField->type = PipeType::PIPE_M;
        reshapeField->reqMemIds = {reshapeMagic, reshapeMagic};
        reshapeField->coreLocation = corePair;
        reshapeField->execOrder = bufNextUseOrder;
        reshapeField->predecessors.insert(&spillCopyInOp);
        spillCopyInField->successors.insert(&reshapeOp);
        opScheduleFields[&reshapeOp] = reshapeField;
        InsertScheduledOps(&reshapeOp, reshapeField->execOrder);

        APASS_LOG_DEBUG_F(Elements::Operation, "Add SPILL_ALLOC: %s.", GetOpInfo(spillAllocOp).c_str());
        APASS_LOG_DEBUG_F(Elements::Operation, "Add SPILL_IN: %s.", GetOpInfo(spillCopyInOp).c_str());
        APASS_LOG_DEBUG_F(Elements::Operation, "Add SPILL_RESHAPE: %s.", GetOpInfo(reshapeOp).c_str());

        // 释放被 spill 的 buffer 并更新 bufRefCount
        if (bufferManagerMap[corePair.first][corePair.second][bufferType].Free(spillInfo.spillMemId_) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Free spill tensor[%d] failed!", spillInfo.spillMemId_);
            return FAILED;
        }
        if (UpdateRemainOpBufId(spillInfo.spillMemId_, reshapeMagic) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "UpdateRemainOpBufId failed.");
            return FAILED;
        }
        bufRefCount_[reshapeMagic] = 0;
        for (auto op : scheduledOps) {
            auto field = GetFields(op);
            if (field && !field->isRetired) {
                for (auto memId : field->reqMemIds) {
                    if (memId == reshapeMagic) {
                        bufRefCount_[reshapeMagic]++;
                    }
                }
            }
        }
        if (InitDependencies() != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "InitDependencies failed.");
            return FAILED;
        }

        numTotalIssues += 3;  // ALLOC + COPY_IN + RESHAPE
        return SUCCESS;
    }

    auto allocField = GetFields(allocOp);
    auto corePair = allocField->coreLocation;

    // 释放被spill的buffer
    if (bufferManagerMap[corePair.first][corePair.second][bufferType].Free(spillInfo.spillMemId_) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Free spill tensor[%d] failed!", spillInfo.spillMemId_);
        return FAILED;
    }

    // 创建reload所需的tensor和操作
    MemoryType memType = spillInfo.spillTensor_->GetMemoryTypeOriginal();

    // 创建新的本地tensor
    LogicalTensorPtr localTensor = std::make_shared<LogicalTensor>(
            function_, spillInfo.spillTensor_->Datatype(), spillInfo.spillTensor_->shape, spillInfo.spillTensor_->Format());
    if (localTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Create local tensor failed!");
        return FAILED;
    }

    // 更新tensor属性
    localTensor->SetMemoryTypeToBe(memType);
    localTensor->SetMemoryTypeOriginal(memType);
    localTensor->oriShape = spillInfo.spillTensor_->oriShape;
    localTensor->UpdateDynValidShape(spillInfo.spillTensor_->GetDynValidShape());
    localTensor->tensor->rawshape = spillInfo.spillTensor_->tensor->rawshape;
    int rawMagic = localTensor->GetRawTensor()->GetRawMagic();
    localTensor->memoryrange.memId = rawMagic;
    localBufferMap[rawMagic] = std::make_shared<LocalBuffer>(
        rawMagic, localTensor->tensor->GetRawDataSize(), memType);
    if (localBufferMap[rawMagic] == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Init Tensor[%d] localBuffer failed.", rawMagic);
        return FAILED;
    }
    tensorAllocCoreMap[rawMagic] = tensorAllocCoreMap[spillInfo.spillTensor_->memoryrange.memId];
    localTensor->offset = std::vector<int64_t>(localTensor->GetShape().size(), 0);

    // 创建 ALLOC 操作
    Opcode allocOpCode = memType == MemoryType::MEM_UB ? Opcode::OP_UB_ALLOC : Opcode::OP_L1_ALLOC;
    auto &reloadAllocOp = function_.AddRawOperation(allocOpCode, {}, {localTensor});
    reloadAllocOp.UpdateLatency(1);

    // 创建 COPY_IN 操作
    Operation* reloadCopyinOp = nullptr;
    if (spillInfo.spillOp_->GetOpcode() == Opcode::OP_COPY_IN) {
        // 如果原来是 COPY_IN，直接克隆
        auto &clonedOp = spillInfo.spillOp_->CloneOperation(function_, {spillInfo.ddrTensor_}, {localTensor});
        reloadCopyinOp = &clonedOp;
        reloadCopyinOp->SetIOpAttrOffset(0, spillInfo.spillOp_->GetIOpAttrOffset(0));
    } else {
        auto &newCopyInOp = function_.AddRawOperation(Opcode::OP_COPY_IN, {spillInfo.ddrTensor_}, {localTensor});
        reloadCopyinOp = &newCopyInOp;
    }

    int64_t base = 0;
    GetWorkspaceBaseOffset(spillInfo.ddrTensor_, base);
    reloadCopyinOp->SetAttr(OpAttributeKey::workspaceBaseOffset, base);
    reloadCopyinOp->SetOpAttribute(std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(spillInfo.ddrTensor_->GetOffset()),
        spillInfo.spillTensor_->GetMemoryTypeOriginal(),
        OpImmediate::Specified(spillInfo.spillTensor_->GetShape()),
        OpImmediate::Specified(spillInfo.spillTensor_->tensor->GetDynRawShape())));
    reloadCopyinOp->UpdateLatency(DEFAULT_LATENCY);

    // 设置 copy_in_mode
    if (spillInfo.spillOp_->GetOpcode() != Opcode::OP_COPY_IN && UpdateCopyInMode(*reloadCopyinOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateCopyInMode failed!");
        return FAILED;
    }

    // 创建 ALLOC 的 ScheduleFields
    auto reloadAllocField = std::make_shared<ScheduleFields>();
    reloadAllocField->type = PipeType::PIPE_M;
    reloadAllocField->isAlloc = true;
    reloadAllocField->reqMemIds = {rawMagic};
    reloadAllocField->coreLocation = corePair;

    // 创建 COPY_IN 的 ScheduleFields
    auto reloadCopyinField = std::make_shared<ScheduleFields>();
    reloadCopyinField->type = PipeType::PIPE_M;
    reloadCopyinField->reqMemIds = {rawMagic};
    reloadCopyinField->coreLocation = corePair;
    reloadCopyinField->predecessors.insert(&reloadAllocOp);
    reloadAllocField->successors.insert(&reloadCopyinOp);

    // 存储到全局 map
    opScheduleFields[&reloadAllocOp] = reloadAllocField;
    opScheduleFields[reloadCopyinOp] = reloadCopyinField;

    // 计算 execOrder
    int bufNextUseOrder = GetBufNextUseOrder(allocOp, spillInfo.spillMemId_);
    if (bufNextUseOrder == -1) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Get Tensor[%d] next use order failed.", spillInfo.spillMemId_);
        return FAILED;
    }
    reloadAllocField->execOrder = bufNextUseOrder++;
    reloadCopyinField->execOrder = bufNextUseOrder;

    // 插入到 scheduledOps
    InsertScheduledOps(&reloadAllocOp, reloadAllocField->execOrder);
    InsertScheduledOps(reloadCopyinOp, reloadCopyinField->execOrder);

    // 更新依赖关系
    auto spillField = GetFields(spillInfo.spillOp_);
    for (auto* succOp : spillField->successors) {
        auto succField = GetFields(succOp);
        if (!succField || succField->isRetired) {
            continue;
        }
        if (std::count(succField->reqMemIds.begin(), succField->reqMemIds.end(), spillInfo.spillMemId_) > 0) {
            // 更新依赖
            succField->predecessors.erase(spillInfo.spillOp_);
            succField->predecessors.insert(reloadCopyinOp);
            reloadCopyinField->successors.insert(succOp);

            // 更新 tensor 输入
            UpdateTensorInput(succOp, spillField, localTensor);
        }
    }

    // 更新后续op的bufId
    if (UpdateRemainOpBufId(spillInfo.spillMemId_, rawMagic) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateRemainOpBufId failed.");
        return FAILED;
    }

    numTotalIssues += TWO_ISSUE;

    APASS_LOG_DEBUG_F(Elements::Operation, "Add SPILL_ALLOC: %s.", GetOpInfo(reloadAllocOp).c_str());
    APASS_LOG_DEBUG_F(Elements::Operation, "Add SPILL_IN: %s.", GetOpInfo(*reloadCopyinOp).c_str());

    return SUCCESS;
}

Operation* OoOScheduler::GetSpillOp(Operation* allocOp, int memId, bool isGenSpill) {
    if (isGenSpill) {
        return GetBufLastWriteOp(allocOp, memId);
    }
    auto allocField = GetFields(allocOp);
    if (!allocField || allocField->reqMemIds.empty()) {
        return nullptr;
    }
    auto memType = localBufferMap[allocField->reqMemIds[0]]->memType;
    return GetSpillOpFromTensorOccupy(memType, memId);
}

Status OoOScheduler::GetSpillInfo(Operation* allocOp, int spillMemId, bool isGenSpill,
    SpillInfo &spillInfo) {
    auto spillOp = GetSpillOp(allocOp, spillMemId, isGenSpill);
    if (spillOp == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] last write op.", spillMemId);
        return FAILED;
    }

    // 获取 spill tensor
    auto spillField = GetFields(spillOp);
    if (!spillField) {
        return FAILED;
    }

    int spillTensorIdx = GetOOperandIdx(spillOp, spillMemId);
    if (spillTensorIdx == -1) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in op's oOperand.", spillMemId);
        return FAILED;
    }

    LogicalTensorPtr spillTensor = spillOp->GetOutputOperand(spillTensorIdx);
    if (spillTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Op cannot find oOperand[%d]. %s",
            spillTensorIdx, GetFormatBacktrace(*spillOp).c_str());
        return FAILED;
    }

    APASS_LOG_DEBUG_F(Elements::Operation, "Begin spill op %s tensor[%d]!",
        GetOpInfo(*spillOp).c_str(), spillMemId);

    spillInfo.spillOp_ = spillOp;
    spillInfo.spillTensor_ = spillTensor;
    spillInfo.spillMemId_ = spillMemId;
    spillInfo.ddrTensor_ = nullptr;

    // 检查是否是特殊 L1 spill
    auto allocField = GetFields(allocOp);
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 &&
        allocOp->GetOpcodeStr().find("L1_ALLOC") != std::string::npos &&
        spillOp->GetOpcodeStr().find("COPY_IN") == std::string::npos) {
        spillInfo.isSpecialL1_ = true;
    }

    return SUCCESS;
}

Status OoOScheduler::SpillBuffer(SpillInfo &spillInfo, Operation* allocOp, size_t &pcIdx,
    LocalBufferPtr allocBuffer, bool isGenSpill) {
    if (SpillOutBuffer(spillInfo, allocOp, pcIdx, isGenSpill) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "SpillOutBuffer failed. %s",
            GetFormatBacktrace(*spillInfo.spillOp_).c_str());
        return FAILED;
    }

    // Healthcheck record
    if (oooCheck.doHealthCheck) {
        // Record spill info
    }

    if (SpillInBuffer(spillInfo, allocOp, allocBuffer->memType, isGenSpill) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "SpillInBuffer failed. %s",
            GetFormatBacktrace(*spillInfo.spillOp_).c_str());
        return FAILED;
    }

    if (!isGenSpill) {
        if (oooCheck.doHealthCheck) {
            UpdateBufferUsage(allocBuffer->memType, spillInfo.spillMemId_, true);
        }
        localBufferMap[spillInfo.spillMemId_]->retireCycle = clock;
        if (tensorOccupyMap[allocBuffer->memType].erase(spillInfo.spillMemId_) == 0) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Erase tensor[%d] failed.", spillInfo.spillMemId_);
            return FAILED;
        }
    }

    return SUCCESS;
}

// ============================================================================
// 新的Buffer选择和多重Spill函数
// ============================================================================

Operation* OoOScheduler::GetSpillOpFromTensorOccupy(MemoryType memType, int memId) {
    if (tensorOccupyMap.find(memType) != tensorOccupyMap.end() &&
        tensorOccupyMap[memType].find(memId) != tensorOccupyMap[memType].end()) {
        auto issue = tensorOccupyMap[memType][memId];
        if (issue && issue->tileOp.GetOpcode() == Opcode::OP_ASSEMBLE) {
            // 对于ASSEMBLE，需要找到实际的生产者
            for (auto* producer : issue->tileOp.ProducerOps()) {
                if (producer->GetOpcodeStr().find("ALLOC") == std::string::npos) {
                    // 查找对应的 Operation*
                    for (auto op : scheduledOps) {
                        if (op == producer) {
                            return op;
                        }
                    }
                }
            }
        }
        // 需要找到对应的 Operation*
        for (auto op : scheduledOps) {
            if (&(issue->tileOp) == op) {
                return op;
            }
        }
    }
    return nullptr;
}

bool OoOScheduler::CheckMachineAndL1(Operation* spillOp, Operation* allocOp) {
    auto allocField = GetFields(allocOp);
    auto spillField = GetFields(spillOp);
    if (!allocField || !spillField) {
        return true;
    }

    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 &&
        allocOp->GetOpcodeStr().find("L1_ALLOC") != std::string::npos &&
        spillOp->GetOpcodeStr().find("COPY_IN") == std::string::npos &&
        spillOp->GetOpcodeStr().find("RESHAPE") == std::string::npos &&
        spillOp->GetInputOperand(0)->GetMemoryTypeOriginal() != MemoryType::MEM_UB &&
        spillOp->GetInputOperand(0)->GetMemoryTypeOriginal() != MemoryType::MEM_L0C) {
        return false;
    }
    return true;
}

bool OoOScheduler::CheckParallelL0C2L1(Operation* spillOp) {
    if (spillOp->GetOpcode() != Opcode::OP_L0C_TO_L1) {
        return true;
    }
    auto tensor = spillOp->GetOutputOperand(0);
    if (tensor == nullptr) {
        return true;
    }

    for (auto* producer : tensor->GetProducers()) {
        if (producer != spillOp && producer->GetOpcode() == Opcode::OP_L0C_TO_L1) {
            return false;
        }
    }
    return true;
}

bool OoOScheduler::IsBelongSpillBlackList(Operation* spillOp, Operation* allocOp) {
    auto spillField = GetFields(spillOp);
    auto allocField = GetFields(allocOp);
    if (!spillField || !allocField) {
        return true;
    }

    // 检查是否是 ALLOC 操作
    if (spillField->isAlloc) {
        return true;
    }

    // 检查是否在过滤列表中
    // 简化版：检查 COPY_IN 的后继
    auto dstOps = allocField->successors;
    std::set<Operation*> filterOps;
    for (auto* dstOp : dstOps) {
        auto dstField = GetFields(dstOp);
        if (!dstField) continue;
        if (COPY_IN_OPS.find(dstOp->GetOpcode()) != COPY_IN_OPS.end()) {
            for (auto* succ : dstField->successors) {
                auto succField = GetFields(succ);
                if (!succField) continue;
                for (auto* pred : succField->predecessors) {
                    filterOps.insert(pred);
                }
            }
        }
        for (auto* pred : dstField->predecessors) {
            filterOps.insert(pred);
        }
    }

    if (filterOps.count(spillOp) != 0 || !CheckMachineAndL1(spillOp, allocOp) || !CheckParallelL0C2L1(spillOp)) {
        return true;
    }
    return false;
}

Status OoOScheduler::GetGroupNextUseOrder(std::vector<int> group, Operation* allocOp,
    std::vector<int> &groupNextUseTime, std::unordered_map<int, size_t> &nextUseTimeCache, bool isGenSpill) {
    std::vector<size_t> bufNextUseTime;
    for (auto& memId : group) {
        Operation* spillOp = isGenSpill ? GetBufLastWriteOp(allocOp, memId) : GetSpillOpFromTensorOccupy(
            localBufferMap[GetFields(allocOp)->reqMemIds[0]]->memType, memId);
        if (spillOp == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] last write op.", memId);
            return FAILED;
        }
        if (IsBelongSpillBlackList(spillOp, allocOp)) {
            groupNextUseTime.push_back(-1);
            return SUCCESS;
        }
        if (nextUseTimeCache.find(memId) != nextUseTimeCache.end()) {
            bufNextUseTime.push_back(nextUseTimeCache[memId]);
        } else {
            int nextUseOrder = GetBufNextUseOrder(allocOp, memId);
            if (nextUseOrder == -1) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find Tensor[%d] next used time.", memId);
                return FAILED;
            }
            nextUseTimeCache[memId] = static_cast<size_t>(nextUseOrder);
            bufNextUseTime.push_back(static_cast<size_t>(nextUseOrder));
        }
    }
    if (bufNextUseTime.empty()) {
        groupNextUseTime.push_back(-1);
    } else {
        groupNextUseTime.push_back(*std::min_element(bufNextUseTime.begin(), bufNextUseTime.end()));
    }
    return SUCCESS;
}

int OoOScheduler::GetMemidAllocPriority(int memId) {
    for (auto op : scheduledOps) {
        auto field = GetFields(op);
        if (!field || !field->isAlloc) {
            continue;
        }
        if (!field->reqMemIds.empty() && field->reqMemIds[0] == memId) {
            return field->execOrder;
        }
    }
    return -1;
}

bool OoOScheduler::CanAllocateAll(std::vector<LocalBufferPtr> tensors, MemoryType memType) {
    if (tensors.empty()) {
        APASS_LOG_INFO_F(Elements::Operation, "CanAllocateAll input tensors is empty.");
        return true;
    }
    auto corePair = tensorAllocCoreMap[tensors[0]->id];
    std::map<uint64_t, std::map<uint64_t, uint64_t>> freeIntervals = bufferManagerMap[corePair.first][corePair.second][memType].FindFreeIntervals();
    for (auto tensor : tensors) {
        bool canAlloc = false;
        std::pair<uint64_t, uint64_t> newInterval;
        uint64_t allocInterval;
        uint64_t allocAddrStart;
        for (auto &interval : freeIntervals) {
            if (interval.first < tensor->size) {
                continue;
            }
            uint64_t addrStart = interval.second.begin()->first;
            uint64_t addrEnd = interval.second.begin()->second;
            interval.second.erase(addrStart);
            newInterval = {addrStart + tensor->size, addrEnd};
            allocInterval = interval.first;
            allocAddrStart = addrStart;
            canAlloc = true;
            break;
        }
        if (!canAlloc) {
            return false;
        }
        freeIntervals[newInterval.second - newInterval.first].insert(newInterval);
        freeIntervals[allocInterval].erase(allocAddrStart);
        if (freeIntervals[allocInterval].empty()) {
            freeIntervals.erase(allocInterval);
        }
    }
    return true;
}

bool OoOScheduler::HasEnoughBuffer(Operation* allocOp, MemoryType memType) {
    std::vector<LocalBufferPtr> tensors;
    std::vector<int> memIds;
    auto allocField = GetFields(allocOp);
    if (!allocField || allocOp->GetOOperands().size() != 1) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s must only have one ooperand.", GetOpInfo(*allocOp).c_str());
        return false;
    }

    for (auto* dstOp : allocField->successors) {
        auto dstField = GetFields(dstOp);
        if (!dstField) continue;
        if (dstOp != *(allocOp->GetOutputOperand(0)->GetProducers().begin())) {
            continue;
        }
        for (auto& memId : dstField->reqMemIds) {
            if (localBufferMap[memId]->memType != memType) {
                continue;
            }
            auto corePair = tensorAllocCoreMap[memId];
            if (bufferManagerMap[corePair.first][corePair.second][memType].isAllocate(memId)) {
                continue;
            }
            if (std::count(memIds.begin(), memIds.end(), memId) == 0) {
                memIds.push_back(memId);
            }
        }
    }
    std::sort(memIds.begin(), memIds.end(), [&](int a, int b) {
        int priorA = GetMemidAllocPriority(a);
        int priorB = GetMemidAllocPriority(b);
        return priorA < priorB;
    });
    for (auto memId : memIds) {
        tensors.push_back(localBufferMap[memId]);
    }
    return CanAllocateAll(tensors, memType);
}

Status OoOScheduler::SelectSpillBuffers(LocalBufferPtr allocBuffer, Operation* allocOp,
    std::vector<int> &spillGroup, bool isGenSpill) {
    std::vector<std::vector<int>> canSpillGroups;
    auto allocField = GetFields(allocOp);
    auto corePair = allocField->coreLocation;

    if (bufferManagerMap[corePair.first][corePair.second][allocBuffer->memType].GetSpillGroup(allocBuffer->size, canSpillGroups) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "GetSpillGroup failed.");
        return FAILED;
    }
    if (canSpillGroups.empty()) {
        APASS_LOG_WARN_F(Elements::Tensor, "Cannot find tensor to spill.");
        return FAILED;
    }

    std::unordered_map<int, size_t> nextUseTimeCache;
    std::vector<int> groupNextUseTime;
    for (auto& group : canSpillGroups) {
        if (GetGroupNextUseOrder(group, allocOp, groupNextUseTime, nextUseTimeCache, isGenSpill) != SUCCESS) {
            APASS_LOG_WARN_F(Elements::Operation, "GetGroupNextUseOrder failed.");
            return FAILED;
        }
    }

    size_t groupSel = std::max_element(groupNextUseTime.begin(), groupNextUseTime.end()) - groupNextUseTime.begin();
    if (groupNextUseTime[groupSel] == -1) {
        APASS_LOG_WARN_F(Elements::Tensor, "Cannot find tensor to spill.");
        return FAILED;
    }
    spillGroup = canSpillGroups[groupSel];
    return SUCCESS;
}

Status OoOScheduler::RearrangeBuffer(Operation* allocOp, MemoryType memType, std::pair<OpCoreType, int> corePair, bool isGenSpill) {
    std::vector<int> memIds = bufferManagerMap[corePair.first][corePair.second][memType].GetAddrSortedBufs();
    for (auto memId : memIds) {
        auto spillOp = GetBufLastWriteOp(allocOp, memId);
        if (spillOp == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] last write op.", memId);
            return FAILED;
        }
        if (spillOp->GetOpcodeStr().find("ALLOC") == std::string::npos) {
            return FAILED;
        }
    }
    return bufferManagerMap[corePair.first][corePair.second][memType].CompactBufferSlices(localBufferMap);
}

Status OoOScheduler::SpillAllBuffer(Operation* allocOp, size_t &pcIdx, bool isGenSpill, LocalBufferPtr allocBuffer) {
    MemoryType memType = allocBuffer->memType;
    auto allocField = GetFields(allocOp);
    auto corePair = allocField->coreLocation;
    std::vector<int> memIds = bufferManagerMap[corePair.first][corePair.second][memType].GetAddrSortedBufs();

    for (auto memId : memIds) {
        Operation* spillOp = isGenSpill ? GetBufLastWriteOp(allocOp, memId) : GetSpillOpFromTensorOccupy(memType, memId);
        if (spillOp == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] last write op.", memId);
            return FAILED;
        }

        auto spillField = GetFields(spillOp);
        if (!spillField) continue;

        if (!CheckMachineAndL1(spillOp, allocOp) || !CheckParallelL0C2L1(spillOp) ||
            IsViewOp(*spillOp) || spillOp->GetOpcode() == Opcode::OP_ASSEMBLE ||
            spillOp->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            continue;
        }

        SpillInfo spillInfo;
        if (GetSpillInfo(allocOp, memId, isGenSpill, spillInfo) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "GetSpillInfo failed. %s",
                GetOpInfo(*spillOp).c_str());
            return FAILED;
        }

        if (SpillBuffer(spillInfo, allocOp, pcIdx, allocBuffer, isGenSpill) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SpillBuffer[%d] failed. %s", memId, GetOpInfo(*spillOp).c_str());
            return FAILED;
        }
    }

    // Alloc内存整理
    if (RearrangeBuffer(allocOp, memType, corePair, isGenSpill) != SUCCESS) {
        APASS_LOG_WARN_F(Elements::Operation, "RearrangeBuffer failed at SpillAllBuffer. %s", GetOpInfo(*allocOp).c_str());
    }

    if (!HasEnoughBuffer(allocOp, memType)) {
        APASS_LOG_ERROR_F(Elements::Operation, "Spill all buffer failed! %s", GetOpInfo(*allocOp).c_str());
        APASS_LOG_ERROR_F(Elements::Operation, "Possible causes: incorrect memory reuse, memory fragmentation, or spill not supported."
            "Please check tile shape.");
        return FAILED;
    }

    return SUCCESS;
}

Status OoOScheduler::SpillMultiBuffer(Operation* allocOp, std::vector<int> spillGroup, size_t &pcIdx,
    LocalBufferPtr allocBuffer, bool isGenSpill) {
    for (auto& spillMemId : spillGroup) {
        SpillInfo spillInfo;
        if (GetSpillInfo(allocOp, spillMemId, isGenSpill, spillInfo) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "GetSpillInfo failed. %s", GetOpInfo(*spillInfo.spillOp_).c_str());
            return FAILED;
        }

        if (spillInfo.spillOp_->GetOpcode() == Opcode::OP_ASSEMBLE) {
            if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 &&
                allocOp->GetOpcodeStr().find("L1_ALLOC") != std::string::npos) {
                APASS_LOG_ERROR_F(Elements::Operation, "Failed to spill %d in L1 spill. SpillOp is assemble op.", spillMemId);
                return FAILED;
            }
            // 暂时不支持 ASSEMBLE spill，使用旧逻辑
            APASS_LOG_ERROR_F(Elements::Operation, "ASSEMBLE spill not yet migrated.");
            return FAILED;
        } else {
            if (SpillBuffer(spillInfo, allocOp, pcIdx, allocBuffer, isGenSpill) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "SpillBuffer[%d] failed. %s", spillMemId, GetOpInfo(*spillInfo.spillOp_).c_str());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status OoOScheduler::GenBufferSpill(Operation* allocOp) {
    std::vector<int> spillGroup;
    bool spillFailed = false;
    auto allocField = GetFields(allocOp);
    if (!allocField || allocField->reqMemIds.empty()) {
        return FAILED;
    }

    if (SelectSpillBuffers(localBufferMap[allocField->reqMemIds[0]], allocOp, spillGroup, false) != SUCCESS) {
        spillFailed = true;
    }
    size_t temp = 1;
    if (spillFailed) {
        return SpillAllBuffer(allocOp, temp, false, localBufferMap[allocField->reqMemIds[0]]);
    } else {
        if (SpillMultiBuffer(allocOp, spillGroup, temp, localBufferMap[allocField->reqMemIds[0]], false) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SpillMultiBuffer failed! %s", GetOpInfo(*allocOp).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

// ============================================================================
// 更新 GenSpillOp 实现以使用新函数
// ============================================================================

Status OoOScheduler::GenSpillOpImpl(size_t &pcIdx) {
    APASS_LOG_DEBUG_F(Elements::Operation, "START: SPILL tensor (new)!");
    auto op = scheduledOps[pcIdx];
    auto field = GetFields(op);
    if (!field || field->reqMemIds.empty()) {
        return FAILED;
    }

    LocalBufferPtr allocBuffer = localBufferMap[field->reqMemIds[0]];
    if (allocBuffer->memType != MemoryType::MEM_L1 && allocBuffer->memType != MemoryType::MEM_UB) {
        APASS_LOG_ERROR_F(Elements::Operation, "Buffer[L0A/B/C] is Full (new). Please check tile shape.");
        return FAILED;
    }

    // 选择需要spill的tensor组
    std::vector<int> spillGroup;
    if (SelectSpillBuffers(allocBuffer, op, spillGroup, true) != SUCCESS) {
        return SpillAllBuffer(op, pcIdx, true, allocBuffer);
    } else {
        if (SpillMultiBuffer(op, spillGroup, pcIdx, allocBuffer, true) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SpillMultiBuffer failed!");
            return FAILED;
        }
    }

    APASS_LOG_DEBUG_F(Elements::Operation, "END: SPILL tensor (new)!");
    return SUCCESS;
}

} // namespace npu::tile_fwk
