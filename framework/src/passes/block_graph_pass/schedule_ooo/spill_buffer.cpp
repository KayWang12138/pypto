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

#include "ooo_scheduler.h"
#include "passes/pass_log/pass_log.h"

namespace npu::tile_fwk {

constexpr int32_t TWO_ISSUE = 2;
constexpr int32_t DEFAULT_LATENCY = 511;

Status OoOScheduler::GenBufferSpill(Operation* allocOp, SpillContext &ctx) {
    std::vector<int> spillGroup = SelectSpillBuffers(allocOp);
    if (spillGroup.empty()) {
        // 选不出可spill的，报错
        APASS_LOG_ERROR_F(Elements::Operation, "Select buffer to spill failed.");
        return FAILED;
    }
    ctx.spillMemIds = spillGroup;
    for (auto &memId : spillGroup) {
        SpillBuffer(memId, allocOp, ctx);
    }

    if (RearrangeBuffer(allocOp, localBufferMap_[opReqMemIdsMap[allocOp][0]]->memType) != SUCCESS) {
        APASS_LOG_WARN_F(Elements::Operation, "RearrangeBuffer failed at SpillAllBuffer. %s", GetFormatBacktrace(*allocOp).c_str());
    }
    MemoryType memType = localBufferMap_[opReqMemIdsMap[allocOp][0]]->memType;
    if (!HasEnoughBuffer(allocOp, memType)) {
        APASS_LOG_ERROR_F(Elements::Operation, "Spill all buffer failed! %s", GetFormatBacktrace(*allocOp).c_str());
        if (PrintSpillFailedInfo(allocOp) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PrintSpillFailedInfo failed; Please check the PrintSpillFailedInfo method.");
            return FAILED;
        }
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "Possible causes: incorrect memory reuse, memory fragmentation, or spill not supported for L0C_COPY_TO_L1."
            "Please check tile shape and OOO spill failed info. Consider avoiding cube-aligned matrix sizes.");
        return FAILED;
    }
    return SUCCESS;
}

std::vector<int> OoOScheduler::SelectSpillBuffers(Operation* allocOp) {
    LocalBufferPtr allocBuffer = localBufferMap_[opReqMemIdsMap[allocOp][0]];
    auto coreType = opCoreLocationMap[allocOp];
    std::vector<int> spillGroup = bufferManagerMap[coreType][allocBuffer->memType].GetAddrSortedBufs();
    // 查找出可以spill 单个或多个tensor的集合
    std::vector<std::vector<int>> canSpillGroups = 
        bufferManagerMap[coreType][allocBuffer->memType].GetSpillGroup(allocBuffer->size);
    if (canSpillGroups.empty()) {
        APASS_LOG_WARN_F(Elements::Tensor, "Cannot find tensor to spill.");
        return spillGroup;
    }
    std::unordered_map<int, size_t> nextUseTimeCache;
    std::vector<int> groupNextUseTime;
    for (auto &group : canSpillGroups) {
        if (GetGroupNextUseTime(group, allocOp, groupNextUseTime, nextUseTimeCache) != SUCCESS) {
            APASS_LOG_WARN_F(Elements::Operation, "Get group next use time failed.");
            return spillGroup;
        }
    }
    size_t groupSel = std::max_element(groupNextUseTime.begin(), groupNextUseTime.end()) - groupNextUseTime.begin();
    if (groupNextUseTime[groupSel] == -1) {
        APASS_LOG_WARN_F(Elements::Tensor, "Cannot find tensor to spill.");
        return spillGroup;
    }
    spillGroup = canSpillGroups[groupSel];
    return spillGroup;
}

Status OoOScheduler::GetGroupNextUseTime(std::vector<int> group, Operation* allocOp,
    std::vector<int> &groupNextUseTime, std::unordered_map<int, size_t> &nextUseTimeCache) {
    size_t minNextUseTime = INT_MAX;
    for (auto& memId : group) {
        Operation* spillOp = GetSpillOp(memId);
        if (spillOp == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d]'s op.", memId);
            return FAILED;
        }
        if (IsBelongSpillBlackList(spillOp, allocOp)) {
            // 存在非法memId时将该group排除
            groupNextUseTime.push_back(-1);
            return SUCCESS;
        }
        if (nextUseTimeCache.find(memId) != nextUseTimeCache.end()) {
            minNextUseTime = std::min(minNextUseTime, nextUseTimeCache[memId]);
        } else {
            int nextUseTime = GetBufNextUseTime(allocOp, memId);
            if (nextUseTime == -1) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find Tensor[%d] next used time.", memId);
                return FAILED;
            }
            nextUseTimeCache[memId] = static_cast<size_t>(nextUseTime);
            minNextUseTime = std::min(minNextUseTime, static_cast<size_t>(nextUseTime));
        }
    }
    groupNextUseTime.push_back(minNextUseTime);
    return SUCCESS;
}

bool OoOScheduler::IsBelongSpillBlackList(Operation* spillOp, Operation* op) {
    std::set<Operation*> filterLtags;
    FindFilterLtags(op, filterLtags);
    if (opIsAllocMap[spillOp] || filterLtags.count(spillOp) != 0 || !CheckMachineAndL1(spillOp, op) ||
        !CheckParallelL0C2L1(spillOp)) {
        return true;
    }
    return false;
}

void OoOScheduler::FindFilterLtags(Operation* allocOp, std::set<Operation*> &filterLtags) {
    auto dstOpList = depManager_.GetSuccessors(allocOp);
    auto dstOp = *dstOpList.begin();
    if (COPY_IN_OPS.find(dstOp->GetOpcode()) != COPY_IN_OPS.end()) {
        for (auto &dstOpId : depManager_.GetSuccessors(dstOp)) {
            auto dstOp_level0 = dstOpId;
            for (auto &inOp : depManager_.GetPredecessors(dstOp_level0)) {
                filterLtags.insert(inOp);
            }
        }
    }
    for (auto &dstOp_level1 : dstOpList) {
        for (auto &inOp : depManager_.GetPredecessors(dstOp_level1)) {
            filterLtags.insert(inOp);
        }
    }
}

bool OoOScheduler::CheckMachineAndL1(Operation* spillOp, Operation* allocOp) {
    if (!spillOp->GetInputOperand(0)) {
        APASS_LOG_WARN_F(Elements::Tensor, "CheckMachineAndL1: spillOp %s has no inputOperand.", GetOpInfo(spillOp).c_str());
        return false;
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

    for (auto *producer : tensor->GetProducers()) {
        if (producer != spillOp && producer->GetOpcode() == Opcode::OP_L0C_TO_L1) {
            return false;
        }
    }
    return true;
}

Status OoOScheduler::SpillBuffer(int memId, Operation* spillAllocOp, SpillContext &ctx) {
    Operation* spillOp = GetSpillOp(memId);
    if (spillOp == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d]'s op.", memId);
        return FAILED;
    }
    if (opIsAllocMap[spillOp] || !CheckMachineAndL1(spillOp, spillAllocOp) ||
        !CheckParallelL0C2L1(spillOp)) {
        return SUCCESS;
    }
    LogicalTensorPtr spillTensor = GetSpillTensor(spillOp, memId);
    if (spillTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d].", memId);
        return FAILED;
    }
    if (oooCheck.doHealthCheck) {
        oooCheck.spillInfoVec.emplace_back(
            RecordSpillInfo(spillTensor, memId, spillAllocOp, spillOp->GetOpcodeStr().find("COPY_IN") == std::string::npos));
    }
    if (spillOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
        // spill的tensor存在多个生产者
        if (SpillMultiProducerBuffer(memId, spillOp, spillTensor, spillAllocOp, ctx) != SUCCESS) {
            return FAILED;
        }
    } else if (spillOp->GetOpcodeStr().find("COPY_IN") != std::string::npos) {
        // spill的tensor来自DDR
        if (SpillBufferFromDDR(memId, spillOp, spillTensor, spillAllocOp, ctx) != SUCCESS) {
            return FAILED;
        }
    } else if (localBufferMap_[memId]->memType == MemoryType::MEM_L1 && 
        Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
        // A5平台, spill的tensor位于L1, 且不来自于DDR
        if (SpillL1BufferFor3510(memId, spillOp, spillTensor, spillAllocOp, ctx) != SUCCESS) {
            return FAILED;
        }
    } else {
        // 通用spill: Copyout + Copyin
        if (SpillGeneralBuffer(memId, spillOp, spillTensor, spillAllocOp, ctx) != SUCCESS) {
            return FAILED;
        }
    }
    if (bufferManagerMap[opCoreLocationMap[spillAllocOp]][localBufferMap_[memId]->memType].Free(memId) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Free spill tensor[%d] failed!", memId);
        return FAILED;
    }
    tensorOccupyMap.erase(memId);
    return SUCCESS;
}

// GMTensor --> spillOp --> spillTensor(UB/L1)
Status OoOScheduler::SpillBufferFromDDR(int memId, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx) {
    LogicalTensorPtr gmTensor = spillOp->GetInputOperand(0);
    LogicalTensorPtr localTensor = CreateLocalTensor(spillTensor);
    Operation* allocOp = CreateAllocOp(localTensor);
    Operation* copyinOp = CloneCopyinOp(spillOp, gmTensor, localTensor);

    UpdateOpScheduleInfo(allocOp, {localTensor->memoryrange.memId}, spillAllocOp);
    UpdateOpScheduleInfo(copyinOp, {localTensor->memoryrange.memId}, spillAllocOp);

    if (InsertOps({allocOp, copyinOp}, spillAllocOp, memId) != SUCCESS) {
        return FAILED;
    }
    if (UpdateSpillOpDepend(spillOp, localTensor, memId) != SUCCESS) {
        return FAILED;
    }
    if (UpdateRemainMemid(memId, opReqMemIdsMap[allocOp][0]) != SUCCESS) {
        return FAILED;
    }
    depManager_.InitDependencies(orderedOps, false);
    ctx.newAllocOps.push_back(allocOp);
    return SUCCESS;
}

// spillOp --> spillTensor(UB/L1)
Status OoOScheduler::SpillGeneralBuffer(int spillMemId, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx) {
    int64_t workspaceOffsetTemp = workspaceOffset;
    LogicalTensorPtr gmTensor = CreateGMTensor(spillTensor, spillMemId);
    LogicalTensorPtr localTensor = CreateLocalTensor(spillTensor);

    Operation *copyoutOp = CreateCopyoutOp(spillOp, spillTensor, gmTensor, workspaceOffsetTemp);
    int64_t base = 0;
    GetWorkspaceBaseOffset(gmTensor, base);
    Operation *allocOp = CreateAllocOp(localTensor);
    Operation *copyinOp = CreateCopyinOp(gmTensor, localTensor, gmTensor->GetOffset(), base);

    if (UpdateCopyoutScheduleInfo(copyoutOp, spillTensor, spillMemId, spillAllocOp) != SUCCESS) {
        return FAILED;
    }

    UpdateOpScheduleInfo(allocOp, {localTensor->memoryrange.memId}, spillAllocOp);
    UpdateOpScheduleInfo(copyinOp, {localTensor->memoryrange.memId}, spillAllocOp);

    if (InsertOps({allocOp, copyinOp}, spillAllocOp, spillMemId) != SUCCESS) {
        return FAILED;
    }
    if (UpdateSpillOpDepend(spillOp, localTensor, spillMemId) != SUCCESS) {
        return FAILED;
    }
    if (UpdateRemainMemid(spillMemId, opReqMemIdsMap[allocOp][0]) != SUCCESS) {
        return FAILED;
    }
    depManager_.InitDependencies(orderedOps, false);
    ctx.newCopyoutOps.push_back(copyoutOp);
    ctx.newAllocOps.push_back(allocOp);
    return SUCCESS;
}

Status OoOScheduler::SpillL1BufferFor3510(int memId, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx) {
    if (spillOp->GetOpcode() != Opcode::OP_RESHAPE) {
        if (spillOp->GetIOperands().size() == 1) {
            SpillGeneralL1BufferFor3510(memId, spillOp, spillTensor, spillAllocOp, ctx);
        } else {
            return FAILED;
        }
    } else {
        Operation* actualSpillOp = nullptr;
        for (auto &preOp : depManager_.GetPredecessors(spillOp)) {
            if (!opIsAllocMap[preOp]) {
                actualSpillOp = preOp;
            }
        }
        if (actualSpillOp == nullptr || actualSpillOp->GetIOperands().size() != 1) {
            return FAILED;
        }
        if (actualSpillOp->GetOpcode() == Opcode::OP_COPY_IN) {
            SpillReshapeFromDDRFor3510(memId, actualSpillOp, spillOp, spillTensor, spillAllocOp, ctx);
        } else {
            SpillReshapeL1BufferFor3510(memId, actualSpillOp, spillOp, spillTensor, spillAllocOp, ctx);
        }
    }
    return SUCCESS;
}

// actualSpillTensor(L0C/UB) --> spillOp --> spillTensor(L1)
Status OoOScheduler::SpillGeneralL1BufferFor3510(int memId, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx) {
    LogicalTensorPtr actualSpillTensor = spillOp->GetInputOperand(0);
    // workspaceOffset 在 gmtensor 创建时被更新，copyout 属性需要使用当前的 workspaceOffset
    int64_t workspaceOffsetTemp = workspaceOffset;
    // TODO 1 L0C_COPY_L1 dataType设置为 L1 的
    LogicalTensorPtr gmTensor = CreateGMTensor(actualSpillTensor, memId);
    LogicalTensorPtr localTensor = CreateLocalTensor(spillTensor);

    Operation *copyoutOp = CreateCopyoutOp(spillOp, actualSpillTensor, gmTensor, workspaceOffsetTemp);
    // WorkspaceBaseOffset
    int64_t base = 0;
    GetWorkspaceBaseOffset(gmTensor, base);
    Operation* allocOp = CreateAllocOp(localTensor);
    // TODO 2 copyin属性与actualspillTensor一致
    Operation* copyinOp = CreateCopyinOp(gmTensor, localTensor, gmTensor->GetOffset(), base);

    if (UpdateCopyoutScheduleInfo(
            copyoutOp, actualSpillTensor, actualSpillTensor->memoryrange.memId, spillAllocOp) != SUCCESS) {
        return FAILED;
    }
    UpdateOpScheduleInfo(allocOp, {localTensor->memoryrange.memId}, spillAllocOp);
    UpdateOpScheduleInfo(copyinOp, {localTensor->memoryrange.memId}, spillAllocOp);
    
    if (InsertOps({allocOp, copyinOp}, spillAllocOp, memId) != SUCCESS) {
        return FAILED;
    }
    if (UpdateSpillOpDepend(spillOp, localTensor, memId) != SUCCESS) {
        return FAILED;
    }
    if (UpdateRemainMemid(memId, opReqMemIdsMap[allocOp][0]) != SUCCESS) {
        return FAILED;
    }
    depManager_.InitDependencies(orderedOps, false);
    ctx.newCopyoutOps.push_back(copyoutOp);
    ctx.newAllocOps.push_back(allocOp);
    return SUCCESS;
}

// actualSpillTensor(DDR) --> actualSpillOp(copyin) --> preSpillTensor --> spillOp(reshape) --> spillTensor(L1)
Status OoOScheduler::SpillReshapeFromDDRFor3510(int memId, Operation* actualSpillOp, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx) {
    LogicalTensorPtr preSpillTensor = spillOp->GetInputOperand(0);
    LogicalTensorPtr ddrTensor = actualSpillOp->GetInputOperand(0);
    LogicalTensorPtr reshapeTensor = CreateLocalTensor(spillTensor);
    LogicalTensorPtr copyinTensor = CreateParticalTensor(preSpillTensor, reshapeTensor, preSpillTensor, preSpillTensor->GetOffset());

    Operation* allocOp = CreateAllocOp(copyinTensor);
    Operation* copyinOp = CloneCopyinOp(actualSpillOp, ddrTensor, copyinTensor);
    Operation* reshapeOp = CreateReshapeOp(copyinTensor, reshapeTensor);

    UpdateOpScheduleInfo(allocOp, {reshapeTensor->memoryrange.memId}, spillAllocOp);
    UpdateOpScheduleInfo(copyinOp, {reshapeTensor->memoryrange.memId}, spillAllocOp);
    UpdateOpScheduleInfo(reshapeOp, {reshapeTensor->memoryrange.memId, reshapeTensor->memoryrange.memId}, spillAllocOp); 

    if (InsertOps({allocOp, copyinOp, reshapeOp}, spillAllocOp, memId) != SUCCESS) {
        return FAILED;
    }
    if (UpdateSpillOpDepend(spillOp, reshapeTensor, memId) != SUCCESS) {
        return FAILED;
    }
    if (UpdateRemainMemid(memId, reshapeTensor->memoryrange.memId) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateRemainMemid failed.");
        return FAILED;
    }
    depManager_.InitDependencies(orderedOps, false);
    ctx.newAllocOps.push_back(allocOp);
    return SUCCESS;
}

// actualSpillTensor(L0C/UB) --> actualSpillOp --> preSpillTensor(L1) --> spillOp(reshape) --> spillTensor(L1)
Status OoOScheduler::SpillReshapeL1BufferFor3510(int spillMemId, Operation* actualSpillOp, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx) {
    LogicalTensorPtr preSpillTensor = spillOp->GetInputOperand(0);
    // workspaceOffset 在 gmtensor 创建时被更新，copyout 属性需要使用当前的 workspaceOffset
    int64_t workspaceOffsetTemp = workspaceOffset;
    LogicalTensorPtr actualSpillTensor = actualSpillOp->GetInputOperand(0);
    LogicalTensorPtr gmTensor = CreateGMTensor(actualSpillOp->GetInputOperand(0), spillMemId);
    LogicalTensorPtr reshapeTensor = CreateLocalTensor(spillTensor);
    LogicalTensorPtr l1Tensor = CreateParticalTensor(preSpillTensor, reshapeTensor, spillTensor, preSpillTensor->GetOffset());

    Operation* copyoutOp = CreateCopyoutOp(actualSpillOp, actualSpillOp->GetInputOperand(0), gmTensor, workspaceOffsetTemp);
    // WorkspaceBaseOffset
    int64_t base = 0;
    GetWorkspaceBaseOffset(gmTensor, base);

    Operation* allocOp = CreateAllocOp(l1Tensor);
    // TODO 4 workspaceBaseOffset属性
    // copyin shape属性来源于哪个Tensor
    Operation* copyinOp = CreateCopyinOp(gmTensor, l1Tensor, gmTensor->GetOffset(), base);
    Operation* reshapeOp = CreateReshapeOp(l1Tensor, reshapeTensor);

    if (UpdateCopyoutScheduleInfo(
            copyoutOp, actualSpillTensor, actualSpillTensor->memoryrange.memId, spillAllocOp) != SUCCESS) {
        return FAILED;
    }
    UpdateOpScheduleInfo(allocOp, {l1Tensor->memoryrange.memId}, spillAllocOp);
    UpdateOpScheduleInfo(copyinOp, {l1Tensor->memoryrange.memId}, spillAllocOp);
    UpdateOpScheduleInfo(reshapeOp, {l1Tensor->memoryrange.memId, l1Tensor->memoryrange.memId}, spillAllocOp); 

    if (InsertOps({allocOp, copyinOp, reshapeOp}, spillAllocOp, spillMemId) != SUCCESS) {
        return FAILED;
    }
    if (UpdateSpillOpDepend(spillOp, reshapeOp->GetOutputOperand(0), spillMemId) != SUCCESS) {
        return FAILED;
    }
    if (UpdateRemainMemid(spillMemId, opReqMemIdsMap[allocOp][0]) != SUCCESS) {
        return FAILED;
    }

    depManager_.InitDependencies(orderedOps, false);
    ctx.newCopyoutOps.push_back(copyoutOp);
    ctx.newAllocOps.push_back(allocOp);
    return SUCCESS;
}

// tensor(UB/L1)*n--> spillOp(Assemble/L0C_COPY_L1)*n --> spillTensor(UB/L1)
Status OoOScheduler::SpillMultiProducerBuffer(int spillMemid, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp, SpillContext &ctx) {
    int64_t workspaceOffsetTemp = workspaceOffset;
    LogicalTensorPtr gmTensor = CreateGMTensor(spillTensor, spillMemid);
    LogicalTensorPtr assembleOOperand = CreateLocalTensor(spillTensor);

    Operation *copyoutOp = CreateCopyoutOp(spillOp, spillTensor, gmTensor, workspaceOffsetTemp);

    if (UpdateCopyoutScheduleInfo(
            copyoutOp, spillTensor, spillMemid, spillAllocOp) != SUCCESS) {
        return FAILED;
    }
    if (UpdateSpillOpDepend(spillOp, assembleOOperand, spillMemid) != SUCCESS) {
        return FAILED;
    }

    for (auto &op : spillTensor->GetProducers()) {
        for (auto &producer : op->ProducerOps()) {
            if (opIsAllocMap[producer]) {
                producer->UpdateOutputOperand(0, spillTensor);
            }
        }
    }
    Operation* allocOp = CreateAllocOp(assembleOOperand);
    UpdateOpScheduleInfo(allocOp, {assembleOOperand->memoryrange.memId}, spillAllocOp);
    if (InsertOps({allocOp}, spillAllocOp, spillMemid) != SUCCESS) {
        return FAILED;
    }

    for (auto &op : spillTensor->GetProducers()) {
        if (opIsAllocMap[op]) {
            continue;
        }
        if (opIsRetiredMap[op]) {
            CreateParticalBuffer(spillMemid, op, assembleOOperand, copyoutOp, spillAllocOp);
        } else {
            op->ReplaceOutput(assembleOOperand, spillTensor);
        }
    }

    if (UpdateRemainMemid(spillMemid, assembleOOperand->memoryrange.memId) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateRemainMemid failed.");
        return FAILED;
    }
    depManager_.InitDependencies(orderedOps, false);
    ctx.newCopyoutOps.push_back(copyoutOp);
    ctx.newAllocOps.push_back(allocOp);
    return SUCCESS;
}

Status OoOScheduler::CreateParticalBuffer(int spillMemid, Operation* producerOp, LogicalTensorPtr assembleOOperand,
    Operation* copyoutOp, Operation* spillAllocOp) {
    LogicalTensorPtr gmTensor = copyoutOp->GetOutputOperand(0);
    LogicalTensorPtr spillTensor = copyoutOp->GetInputOperand(0);
    auto assembleAttr = std::static_pointer_cast<AssembleOpAttribute>(producerOp->GetOpAttribute());
    std::vector<int64_t> toOffset = assembleAttr->GetToOffset();
    LogicalTensorPtr assembleIOperand = CreateParticalTensor(gmTensor, assembleOOperand, spillTensor, toOffset);

    int64_t gmRelatOffset = CalcWorkspaceOffset(assembleOOperand->GetShape(), toOffset, assembleOOperand->Datatype());
    if (gmRelatOffset == -1) {
        APASS_LOG_ERROR_F(Elements::Operation, "CalcWorkspaceOffset failed.");
        return FAILED;
    }
    int64_t base = 0;
    GetWorkspaceBaseOffset(gmTensor, base);
    Operation* copyinOp = CreateCopyinOp(gmTensor, assembleIOperand, toOffset, gmRelatOffset + base);
    Operation* assembleOp = CreateAssembleOp(assembleIOperand, assembleOOperand, assembleAttr);

    UpdateOpScheduleInfo(copyinOp, {assembleOOperand->memoryrange.memId}, spillAllocOp);
    UpdateOpScheduleInfo(assembleOp, {assembleOOperand->memoryrange.memId, assembleOOperand->memoryrange.memId}, spillAllocOp);
    if (InsertOps({copyinOp, assembleOp}, spillAllocOp, spillMemid) != SUCCESS) {
        return FAILED;
    }
    return SUCCESS;
}

LogicalTensorPtr OoOScheduler::CreateLocalTensor(LogicalTensorPtr spillTensor) {
    LogicalTensorPtr localTensor = 
        std::make_shared<LogicalTensor>(function_, spillTensor->Datatype(), spillTensor->GetShape(), spillTensor->Format());
    localTensor->SetMemoryTypeToBe(spillTensor->GetMemoryTypeOriginal());
    localTensor->SetMemoryTypeOriginal(spillTensor->GetMemoryTypeOriginal());
    localTensor->oriShape = spillTensor->oriShape;
    localTensor->UpdateDynValidShape(spillTensor->GetDynValidShape());
    localTensor->tensor->rawshape = spillTensor->tensor->rawshape;
    int rawMagic = localTensor->GetRawTensor()->GetRawMagic();
    localTensor->memoryrange.memId = rawMagic;
    localBufferMap_[rawMagic] =
        std::make_shared<LocalBuffer>(rawMagic, localTensor->tensor->GetRawDataSize(), localTensor->GetMemoryTypeOriginal());
    tensorAllocCoreMap[rawMagic] = tensorAllocCoreMap[spillTensor->memoryrange.memId];
    localTensor->offset = std::vector<int64_t>(localTensor->GetShape().size(), 0);
    return localTensor;
}

LogicalTensorPtr OoOScheduler::CreateGMTensor(LogicalTensorPtr spillTensor, int spillMemId) {
    // TODO A5 中 L0C_L1 时,dataType 应该为 L1 的 dataType(因为会做随路量化)
    std::shared_ptr<RawTensor> gmRawTensor =
        std::make_shared<RawTensor>(spillTensor->Datatype(), spillTensor->tensor->rawshape,
        TileOpFormat::TILEOP_ND, "WorkspaceGm", SYMBOL_STACK_BASE);
    LogicalTensorPtr gmTensor =
        std::make_shared<LogicalTensor>(function_, gmRawTensor, spillTensor->GetOffset(), spillTensor->GetShape());
    gmTensor->SetMemoryTypeToBe(MemoryType::MEM_DEVICE_DDR);
    gmTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    gmTensor->oriShape = spillTensor->oriShape;
    gmTensor->UpdateDynValidShape(spillTensor->GetDynValidShape());
    gmTensor->tensor->rawshape = spillTensor->tensor->rawshape;
    if (localBufferMap_.find(spillMemId) == localBufferMap_.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find Tensor[%d] in localBufferMap_.", spillMemId);
        return nullptr;
    }
    gmTensor->memoryrange =
        TileRange(workspaceOffset, workspaceOffset + localBufferMap_[spillMemId]->size, workspaceMemId++);
    workspaceOffset += localBufferMap_[spillMemId]->size;
    return gmTensor;
}

LogicalTensorPtr OoOScheduler::CreateParticalTensor(
    LogicalTensorPtr iOperand, LogicalTensorPtr oriOperand, LogicalTensorPtr spillTensor,
    std::vector<int64_t> toOffset)
{
    LogicalTensorPtr localTensor =
        std::make_shared<LogicalTensor>(function_, iOperand->Datatype(), iOperand->GetShape(), iOperand->Format());
    localTensor->SetMemoryTypeToBe(oriOperand->GetMemoryTypeToBe());
    localTensor->SetMemoryTypeOriginal(oriOperand->GetMemoryTypeOriginal());
    localTensor->oriShape = iOperand->shape;
    localTensor->tensor = oriOperand->tensor;
    localTensor->memoryrange.memId = oriOperand->memoryrange.memId;
    localTensor->UpdateDynValidShape(spillTensor->GetDynValidShape());
    localTensor->offset = toOffset;
    tensorAllocCoreMap[localTensor->memoryrange.memId] = tensorAllocCoreMap[oriOperand->memoryrange.memId];
    return localTensor;
}

Operation* OoOScheduler::CreateAllocOp(LogicalTensorPtr oOperand) {
    Opcode opcode = 
        oOperand->GetMemoryTypeOriginal() == MemoryType::MEM_UB ? Opcode::OP_UB_ALLOC : Opcode::OP_L1_ALLOC;
    Operation& allocOp = function_.AddRawOperation(opcode, {}, {oOperand});
    allocOp.UpdateLatency(1);
    return &allocOp;
}

Operation* OoOScheduler::CloneCopyinOp(Operation* spillOp, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand) {
    Operation& copyinOp = spillOp->CloneOperation(function_, {iOperand}, {oOperand});
    copyinOp.SetIOpAttrOffset(0, spillOp->GetIOpAttrOffset(0));
    copyinOp.SetOpAttribute(spillOp->GetOpAttribute()->Clone());
    copyinOp.inParamLocation_ = spillOp->inParamLocation_;
    copyinOp.UpdateLatency(DEFAULT_LATENCY);
    return &copyinOp;
}

Operation* OoOScheduler::CreateCopyinOp(LogicalTensorPtr iOperand, LogicalTensorPtr oOperand, 
    std::vector<int64_t> offset, int64_t workspaceBaseOffset) 
{
    Operation& copyinOp = function_.AddRawOperation(Opcode::OP_COPY_IN, {iOperand}, {oOperand});
    copyinOp.SetAttr(OpAttributeKey::workspaceBaseOffset, workspaceBaseOffset);
    copyinOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(offset), // 搬运GM上的偏移
        oOperand->GetMemoryTypeOriginal(),
        OpImmediate::Specified(oOperand->GetShape()), // 搬运数据量
        OpImmediate::Specified(oOperand->tensor->GetDynRawShape()))); // 暂未使用
    copyinOp.UpdateLatency(DEFAULT_LATENCY);
    if (oOperand->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
        if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
            copyinOp.SetAttribute(OpAttributeKey::copyInMode, static_cast<int64_t>(Matrix::CopyInMode::ND2NZ));
        } else {
            copyinOp.SetAttribute(OpAttributeKey::copyInMode, static_cast<int64_t>(Matrix::CopyInMode::ND2ND));
        }
    }
    return &copyinOp;
}

// 入参缺少 workspaceBaseOffset
Operation* OoOScheduler::CreateCopyoutOp(Operation* spillOp, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand, int64_t workspaceBaseOffset) {
    Operation &copyoutOp = function_.AddRawOperation(Opcode::OP_COPY_OUT, {iOperand}, {oOperand});
    copyoutOp.SetAttr(OpAttributeKey::workspaceBaseOffset, workspaceBaseOffset);
    copyoutOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        iOperand->GetMemoryTypeOriginal(),
        OpImmediate::Specified(oOperand->GetOffset()),
        OpImmediate::Specified(iOperand->GetShape()),
        OpImmediate::Specified(iOperand->GetRawTensor()->GetDynRawShape())));
    if (spillOp->HasAttribute(OpAttributeKey::scaleValue)) {
        Element scaleValue = Element(DataType::DT_UINT64, 0);
        spillOp->GetAttr(OpAttributeKey::scaleValue, scaleValue);
        copyoutOp.SetAttribute(OpAttributeKey::scaleValue, scaleValue);
    }
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
        if (iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_L0C) {
            copyoutOp.SetAttribute(OpAttributeKey::copyIsNZ, 0);
        }
    } else {
        if (iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
            copyoutOp.SetAttribute(OpAttributeKey::copyOutMode, static_cast<int64_t>(Matrix::CopyOutMode::ND2ND));
        }
    }
    copyoutOp.UpdateLatency(DEFAULT_LATENCY);
    return &copyoutOp;
}

Operation* OoOScheduler::CreateReshapeOp(LogicalTensorPtr iOperand, LogicalTensorPtr oOperand) {
    Operation& reshapeOp = function_.AddRawOperation(Opcode::OP_RESHAPE, {iOperand}, {oOperand});
    reshapeOp.UpdateLatency(0);
    return &reshapeOp;
}

Operation* OoOScheduler::CreateAssembleOp(LogicalTensorPtr iOperand, LogicalTensorPtr oOperand,
    std::shared_ptr<AssembleOpAttribute> assembleAttr) {
    Operation& assembleOp = function_.AddRawOperation(Opcode::OP_ASSEMBLE, {iOperand}, {oOperand});
    assembleOp.UpdateLatency(1);
    assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(assembleAttr->GetFrom(),
        assembleAttr->GetToOffset(), assembleAttr->GetToDynOffset(), assembleAttr->GetFromDynValidShape()));
    return &assembleOp;
}

LogicalTensorPtr OoOScheduler::GetSpillTensor(Operation* spillOp, int spillMemId) {
    int spillTensorIdx = GetOOperandIdx(spillOp, spillMemId);
    return spillOp->GetOutputOperand(spillTensorIdx);
}

Status OoOScheduler::UpdateCopyoutScheduleInfo(Operation* op, LogicalTensorPtr spillTensor, int spillMemId, Operation* spillAllocOp) {
    opReqMemIdsMap[op] = {spillMemId};
    opIsRetiredMap[op] = true;
    opIsAllocMap[op] = false;
    opPipeTypeMap[op] = RescheduleUtils::GetOpPipeType(op);
    depManager_.RegisterOp(op);
    for (auto preOp : spillTensor->GetProducers()) {
        opCoreLocationMap[op] = opCoreLocationMap[preOp];
        UpdateOpInternalSubgraphID(*op, preOp);
        break;
    }
    int bufLastUseTime = GetBufLastUseTime(spillAllocOp, spillMemId);
    if (bufLastUseTime == -1) {
        return FAILED;
    }
    opExecOrderMap[op] = bufLastUseTime + 1;
    InsertOrdered(op);
    return SUCCESS;
}

void OoOScheduler::UpdateOpScheduleInfo(Operation* op, std::vector<int> memIds, Operation* AllocOp) {
    opPipeTypeMap[op] = RescheduleUtils::GetOpPipeType(op);
    opIsAllocMap[op] = op->GetOpcodeStr().find("ALLOC") != std::string::npos;
    opIsRetiredMap[op] = false;
    opReqMemIdsMap[op] = memIds;
    depManager_.RegisterOp(op);
    opCoreLocationMap[op] = opCoreLocationMap[AllocOp];
    UpdateOpInternalSubgraphID(*op, AllocOp);
    numTotalIssues++;
}

Status OoOScheduler::InsertOps(std::vector<Operation*> ops, Operation* spillAllocOp, int memId) {
    int bufNextUseTime = GetBufNextUseTime(spillAllocOp, memId);
    if (bufNextUseTime == -1) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Get Tensor[%d] next use time failed.", memId);
        return FAILED;
    }
    for (auto &op : ops) {
        opExecOrderMap[op] = bufNextUseTime++;
        InsertOrdered(op);
    }
    return SUCCESS;
}

Status OoOScheduler::UpdateSpillOpDepend(Operation* spillOp, LogicalTensorPtr newTensor, int spillMemId) 
{
    auto& successors = depManager_.GetSuccessors(spillOp);
    for (auto succOp : successors) {
        if (!opIsRetiredMap[succOp]) {
            auto& reqMemIds = opReqMemIdsMap[succOp];
            if (std::count(reqMemIds.begin(), reqMemIds.end(), spillMemId) > 0) {
                UpdateOperationInput(succOp, spillOp, newTensor);
            }
        }
    }
    return SUCCESS;
}

void OoOScheduler::UpdateOperationInput(Operation* targetOp, Operation* spillOp, LogicalTensorPtr newTensor) {
    for (size_t index = 0; index < targetOp->GetIOperands().size(); index++) {
        for (auto &inOp : targetOp->GetIOperands()[index]->GetProducers()) {
            if (IsViewOp(*inOp)) {
                Operation* op = SkipViewChain(inOp, true);
                UpdateTensorInputForView(*op, spillOp, newTensor);
            } else if (inOp == spillOp) {
                targetOp->UpdateInputOperand(index, newTensor);
            }
        }
    }
}

void OoOScheduler::UpdateTensorInputForView(Operation& op, Operation* spillOp, LogicalTensorPtr tensor) {
    bool hit = false;
    for (auto it : op.GetInputOperand(0)->GetProducers()) {
        if (it == spillOp) {
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

OoOSchedulerCheck::SpillInfo OoOScheduler::RecordSpillInfo(
    LogicalTensorPtr spillOutTensor, int memId, Operation* spillAllocOp, bool needCopyOut)
{
    OoOSchedulerCheck::SpillInfo spillInfo;
    MemoryType bufferType = spillOutTensor->GetMemoryTypeOriginal();
    spillInfo.spillType = bufferType;
    spillInfo.bufferCurrUsage = oooCheck.bufferLastUsage[bufferType];
    spillInfo.spillTensorSize = localBufferMap_[memId]->size;
    spillInfo.spillTensorMagic = spillOutTensor->GetMagic();
    spillInfo.triggerTensorSize = spillAllocOp->GetOutputOperand(0)->tensor->GetRawDataSize();
    int allocOccupied = 0;
    for (const auto &pair : tensorOccupyMap) {
        if (opIsAllocMap[pair.second]) {
            allocOccupied += localBufferMap_[pair.first]->size;
        }
    }
    spillInfo.allocOccupiedSize = allocOccupied;
    if (needCopyOut) {
        spillInfo.spillCopyoutSize = spillOutTensor->tensor->GetRawDataSize();
    } else {
        spillInfo.spillCopyoutSize = 0;
    }
    return spillInfo;
}

int OoOScheduler::GetBufLastUseTime(Operation* op, int curMemId) {
    auto targetIt = std::find(orderedOps.begin(), orderedOps.end(), op);
    if (targetIt == orderedOps.end()) {
        return -1;
    }
    int execOrder = opExecOrderMap[op];
    for (auto it = std::make_reverse_iterator(targetIt); it != orderedOps.rend(); it++) {
        Operation* curOp = *it;
        if (curOp && opExecOrderMap[curOp] < execOrder) {
            auto& reqMemIds = opReqMemIdsMap[curOp];
            if (std::find(reqMemIds.begin(), reqMemIds.end(), curMemId) != reqMemIds.end()) {
                return opExecOrderMap[curOp];
            }
        }
    }
    return -1;
}

void OoOScheduler::UpdateOpInternalSubgraphID(Operation &op, Operation* srcOp) {
    if (srcOp->GetInternalSubgraphID() != NOT_IN_SUBGRAPH) {
        op.UpdateInternalSubgraphID(srcOp->GetInternalSubgraphID());
        op.SetAIVCore(srcOp->GetAIVCore());
    }
}

void OoOScheduler::ReplaceViewOpChainMemId(LogicalTensorPtr startTensor, int oldMemId, int newMemId)
{
    std::vector<Operation*> viewConsumers;
    for (auto* consumer : startTensor->GetConsumers()) {
        if (IsViewOp(*consumer)) {
            viewConsumers.push_back(consumer);
        }
    }

    while (!viewConsumers.empty()) {
        Operation* viewOp = viewConsumers.back();
        viewConsumers.pop_back();
        auto viewOutTensor = viewOp->GetOutputOperand(0);
        if (viewOutTensor == nullptr) {
            continue;
        }
        if (viewOutTensor->memoryrange.memId == oldMemId) {
            viewOutTensor->memoryrange.memId = newMemId;
        }
        for (auto* consumer : viewOutTensor->GetConsumers()) {
            if (IsViewOp(*consumer)) {
                viewConsumers.push_back(consumer);
            }
        }
    }
}

void OoOScheduler::ReplaceTensorMemId(Operation* op, int oldMemId, int newMemId) {
    auto& reqMemIds = opReqMemIdsMap[op];
    for (auto memId : reqMemIds) {
        if (memId == oldMemId || memId == newMemId) {
            bufRefCount_[newMemId]++;
        }
        if (memId == oldMemId) {
            std::replace(reqMemIds.begin(), reqMemIds.end(), oldMemId, newMemId);
        }
    }
    for (auto &outTensor : op->GetOOperands()) {
        if (outTensor->memoryrange.memId == oldMemId) {
            outTensor->memoryrange.memId = newMemId;
            ReplaceViewOpChainMemId(outTensor, oldMemId, newMemId);
        }
    }
}

Status OoOScheduler::UpdateRemainMemid(int oldMemId, int newMemId) {
    if (bufRefCount_.find(oldMemId) == bufRefCount_.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "bufRefCount cannot find Tensor[%d].", oldMemId);
        return FAILED;
    }
    bufRefCount_[newMemId] = 0;
    bufRefCount_[oldMemId] = 0;
    for (auto& op : orderedOps) {
        if (opIsRetiredMap[op]) {
            continue;
        }
        ReplaceTensorMemId(op, oldMemId, newMemId);
    }
    return SUCCESS;
}

void OoOScheduler::InsertOrdered(Operation* insertOp) {
    int execOrder = opExecOrderMap[insertOp];
    auto it = orderedOps.begin();
    for (; it != orderedOps.end(); it++) {
        if (opExecOrderMap[*it] >= execOrder) {
            break;
        }
    }
    auto insertPos = orderedOps.insert(it, insertOp);
    // 更新后续元素的execOrder
    for (auto adjustIt = insertPos + 1; adjustIt != orderedOps.end(); adjustIt++) {
        if (opExecOrderMap[*adjustIt] >= execOrder) {
            opExecOrderMap[*adjustIt]++;
        }
    }
}

int64_t OoOScheduler::CalcWorkspaceOffset(std::vector<int64_t> shape, std::vector<int64_t> offset, DataType dataType)
{
    if (shape.size() != offset.size()) {
        return -1;
    }
    if (shape.size() == 0) {
        return 0;
    }

    int64_t linearOffset = 0;
    int64_t stride = 1;
    // 从最低维到最高维计算
    for (size_t i = shape.size(); i > 0; --i) {
        linearOffset += offset[i - 1] * stride;
        if (i > 0) {
            stride *= shape[i - 1];
        }
    }
    return linearOffset * BytesOf(dataType);
}

bool OoOScheduler::HasEnoughBuffer(Operation* allocOp, MemoryType memType) {
    return !bufferManagerMap[opCoreLocationMap[allocOp]][memType].IsFull(localBufferMap_[opReqMemIdsMap[allocOp][0]]);
}

Status OoOScheduler::RearrangeBuffer(Operation* allocOp, MemoryType memType) {
    std::vector<int> memIds = bufferManagerMap[opCoreLocationMap[allocOp]][memType].GetAddrSortedBufs();
    for (auto memId : memIds) {
        auto op = GetSpillOp(memId);
        if (op == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] last write issue.", memId);
            return FAILED;
        }
        if (op->GetOpcodeStr().find("ALLOC") == std::string::npos) {
            return SUCCESS;
        }
    }
    return bufferManagerMap[opCoreLocationMap[allocOp]][memType].CompactBufferSlices(localBufferMap_);
}

Operation* OoOScheduler::GetSpillOp(int memId) {
    if (tensorOccupyMap.count(memId)) {
        return tensorOccupyMap[memId];
    }
    return nullptr;
}

int OoOScheduler::GetBufNextUseTime(Operation* op, int curMemId) {
    int execOrder = opExecOrderMap[op];
    auto it = std::find_if(
        orderedOps.begin(),
        orderedOps.end(),
        [this, execOrder, curMemId](Operation* a) {
            // 条件1: 操作指针有效
            if (!a) return false;
            // 条件2: 操作的执行顺序必须大于当前操作（向后查找）
            if (opExecOrderMap[a] <= execOrder) return false;
            // 条件3: 该操作需要使用 curMemId 对应的 Buffer
            auto& reqMemIds = opReqMemIdsMap[a];
            return std::find(reqMemIds.begin(), reqMemIds.end(), curMemId) != reqMemIds.end();
        }
    );
    return (it != orderedOps.end()) ? opExecOrderMap[*it] : -1;
}

void OoOScheduler::GetWorkspaceBaseOffset(LogicalTensorPtr ddrTensor, int64_t& base)
{
    for (auto* producer : ddrTensor->GetProducers()) {
        if (producer->GetOpcode() == Opcode::OP_COPY_OUT) {
            producer->GetAttr(OpAttributeKey::workspaceBaseOffset, base);
        }
    }
}

} // namespace npu::tile_fwk
