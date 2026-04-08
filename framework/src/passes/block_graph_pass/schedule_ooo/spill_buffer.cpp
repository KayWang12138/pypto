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

Status OoOScheduler::GenBufferSpill(Operation* allocOp) {
    std::vector<int> spillGroup = SelectSpillBuffers(allocOp);
    if (spillGroup.empty()) {
        // 选不出可spill的，报错
        APASS_LOG_ERROR_F(Elements::Operation, "Select buffer to spill failed.");
        return FAILED;
    }
    for (auto &memId : spillGroup) {
        SpillBuffer(memId, allocOp);
    }
}

std::vector<int> OoOScheduler::SelectSpillBuffers(Operation* allocOp) {
    LocalBufferPtr allocBuffer = localBufferMap[opReqMemIdsMap[allocOp][0]];
    auto corePair = opCoreLocationMap[allocOp];
    std::vector<int> spillGroup = bufferManagerMap[corePair.first][corePair.second][memType].GetAddrSortedBufs();
    // 查找出可以spill 单个或多个tensor的集合
    std::vector<std::vector<int>> canSpillGroups = 
        bufferManagerMap[corePair.first][corePair.second][allocBuffer->memType].GetSpillGroup(allocBuffer->size);
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
        Operation* spillOp = GetSpillOp(localBufferMap[memId]->memType, memId);
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
            minNextUseTime = std::min(minNextUseTime, nextUseTime);
        }
    }
    groupNextUseTime.push_back(minNextUseTime);
    return SUCCESS;
}

Operation* OoOScheduler::GetSpillOp(MemoryType memType, int memId) {
    if (tensorOccupyMap.count(memType) && tensorOccupyMap[memType].count(memId)) {
        return tensorOccupyMap[memType][memId];
    }
    return nullptr;
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
    auto dstOpList = GetSuccessors(allocOp);
    auto dstOp = *dstOpList.begin();
    if (COPY_IN_OPS.find(dstOp->GetOpcode()) != COPY_IN_OPS.end()) {
        for (auto &dstOpId : GetSuccessors(dstOp)) {
            auto dstOp_level0 = dstOpId;
            for (auto &inOp : GetPredecessors(dstOp_level0)) {
                filterLtags.insert(inOp);
            }
        }
    }
    for (auto &dstOp_level1 : dstOpList) {
        for (auto &inOp : GetPredecessors(dstOp_level1)) {
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

Status OoOScheduler::SpillBuffer(int memId, Operation* spillAllocOp) {
    Operation* spillOp = GetSpillOp(localBufferMap[memId]->memType, memId);
    if (spillOp == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d]'s op.", memId);
        return FAILED;
    }
    LogicalTensorPtr spillTensor = GetSpillTensor(spillOp, memId);
    if (spillTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d].", memId);
        return FAILED;
    }
    if (spillOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
        // spill的tensor存在多个生产者
        if (SpillMultiProducerBuffer(spillInfo, allocOp, allocBuffer, ctx) != SUCCESS) {
            return FAILED;
        }
    } else if (spillOp->GetOpcodeStr().find("COPY_IN") != std::string::npos) {
        // spill的tensor来自DDR
        if (SpillBufferFromDDR(spillInfo, allocOp, allocBuffer, ctx) != SUCCESS) {
            return FAILED;
        }
    } else if (localBufferMap[memId]->memType == MemoryType::MEM_L1 && 
        Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
        // A5平台, spill的tensor位于L1, 且不来自于DDR
        if (SpillL1BufferFor3510(spillInfo, allocOp, allocBuffer, ctx) != SUCCESS) {
            return FAILED;
        }
    } else {
        // 通用spill: Copyout + Copyin
        if (SpillGeneralBuffer(spillInfo, allocOp, allocBuffer, ctx) != SUCCESS) {
            return FAILED;
        }
    }
}

// GMTensor --> spillOp --> spillTensor(UB/L1)
Status OoOScheduler::SpillBufferFromDDR(int memId, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp) {
    LogicalTensorPtr ddrTensor = spillOp->GetInputOperand(0);
    LogicalTensorPtr localTensor = CreateLocalTensor(spillTensor, spillTensor->GetMemoryTypeOriginal());
    Operation* allocOp = CreateAllocOp(localTensor);
    Operation* copyinOp = CloneCopyinOp(spillOp, ddrTensor, localTensor);
    
    opPipeTypeMap[allocOp] = RescheduleUtils::GetOpPipeType(allocOp);
    opIsAllocMap[allocOp] = true;
    opIsRetiredMap[allocOp] = false;
    opReqMemIdsMap[allocOp] = {localTensor->memoryrange.memId};

    depManager_.RegisterOp(allocOp);

    opPipeTypeMap[copyinOp] = RescheduleUtils::GetOpPipeType(copyinOp);
    opIsAllocMap[copyinOp] = false;
    opIsRetiredMap[copyinOp] = false;
    opReqMemIdsMap[copyinOp] = {localTensor->memoryrange.memId};

    depManager_.RegisterOp(copyinOp);

    depManager_.AddAllocDependency(allocOp, copyinOp);

    int bufNextUseOrder = GetBufNextUseTime(spillAllocOp, spillMemId);
    if (bufNextUseOrder == -1) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Get Tensor[%d] next use time failed.", spillMemId);
        return FAILED;
    }

    opExecOrderMap[allocOp] = bufNextUseOrder++;
    InsertOrdered(allocOp);
    opExecOrderMap[copyinOp] = bufNextUseOrder;
    InsertOrdered(copyinOp);
    opCoreLocationMap[allocOp] = opCoreLocationMap[allocOp];
    opCoreLocationMap[copyinOp] = opCoreLocationMap[allocOp];
    UpdateOpInternalSubgraphID(*allocOp, allocOp);
    UpdateOpInternalSubgraphID(*copyinOp, allocOp);
    if (UpdateSpillOpDepend(copyinOp, allocOp, spillOp, spillMemId) != SUCCESS) {
        return FAILED;
    }
    if (UpdateRemainMemid(spillMemId, opReqMemIdsMap[reloadAlloc][0])) {
        return FAILED;
    }
}

// spillOp --> spillTensor(UB/L1)
Status OoOScheduler::SpillGeneralBuffer(int memId, Operation* spillOp, LogicalTensorPtr spillTensor) {
    LogicalTensorPtr gmTensor = CreateGMTensor(spillTensor, memId);
    LogicalTensorPtr localTensor = CreateLocalTensor(spillTensor);
    
    Operation *copyoutOp = CreateCopyoutOp(spillOp, spillTensor, gmTensor);
    Operation* allocOp = CreateAllocOp(localTensor);
    Operation* copyinOp = CreateCopyinOp(gmTensor, localTensor, gmTensor->GetOffset());
}

Status OoOScheduler::SpillL1BufferFor3510(int memId, Operation* spillOp, LogicalTensorPtr spillTensor) {
    if (spillOp->GetOpcode() != Opcode::OP_RESHAPE) {
        if (spillOp->GetIOperands().size() == 1) {
            SpillGeneralL1BufferFor3510(memId, spillOp, spillTensor);
        } else {
            return FAILED;
        }
    } else {
        Operation* actualSpillOp = nullptr;
        for (auto &preOp : GetPredecessors(spillOp)) {
            if (!opIsAllocMap[preOp]) {
                actualSpillOp = preOp;
            }
        }
        if (actualSpillOp == nullptr || actualSpillOp->GetIOperands().size() != 1) {
            return FAILED;
        }
        LogicalTensorPtr preSpillTensor = spillOp->GetInputOperand(0);
        if (actualSpillOp->GetOpcode() == Opcode::OP_COPY_IN) {
            SpillReshapeFromDDRFor3510(memId, actualSpillOp, spillOp, preSpillTensor, spillTensor);
        } else {
            SpillReshapeL1BufferFor3510(memId, actualSpillOp, spillOp, preSpillTensor, spillTensor);
        }
    }
}

// actualSpillTensor(L0C/UB) --> spillOp --> spillTensor(L1)
Status OoOScheduler::SpillGeneralL1BufferFor3510(int memId, Operation* spillOp, LogicalTensorPtr spillTensor) {
    LogicalTensorPtr actualSpillTensor = spillOp->GetInputOperand(0);
    LogicalTensorPtr gmTensor = CreateGMTensor(actualSpillTensor, memId);
    LogicalTensorPtr localTensor = CreateLocalTensor(spillTensor);
    
    Operation *copyoutOp = CreateCopyoutOp(spillOp, actualSpillTensor, gmTensor);
    Operation* allocOp = CreateAllocOp(localTensor);
    Operation* copyinOp = CreateCopyinOp(gmTensor, localTensor, gmTensor->GetOffset());
}

// actualSpillTensor(DDR) --> actualSpillOp(copyin) --> preSpillTensor --> spillOp(reshape) --> spillTensor(L1)
Status OoOScheduler::SpillReshapeFromDDRFor3510(int memId, Operation* actualSpillOp, Operation* spillOp, LogicalTensorPtr preSpillTensor, LogicalTensorPtr spillTensor) {
    LogicalTensorPtr ddrTensor = actualSpillOp->GetInputOperand(0);
    LogicalTensorPtr l1Tensor = CreateLocalTensor(preSpillTensor);
    LogicalTensorPtr reshapeTensor = CreateLocalTensor(spillTensor);

    Operation* allocOp = CreateAllocOp(l1Tensor);
    Operation* copyinOp = CloneCopyinOp(actualSpillOp, ddrTensor, l1Tensor);
    Operation* reshapeOp = CreateReshapeOp(l1Tensor, reshapeTensor);

}

// actualSpillTensor(L0C/UB) --> actualSpillOp --> preSpillTensor --> spillOp(reshape) --> spillTensor(L1)
Status OoOScheduler::SpillReshapeL1BufferFor3510(int memId, Operation* actualSpillOp, Operation* spillOp, LogicalTensorPtr preSpillTensor, LogicalTensorPtr spillTensor) {
    LogicalTensorPtr gmTensor = CreateGMTensor(actualSpillOp->GetInputOperand(0), memId);
    LogicalTensorPtr l1Tensor = CreateLocalTensor(preSpillTensor);
    LogicalTensorPtr reshapeTensor = CreateLocalTensor(spillTensor);

    Operation *copyoutOp = CreateCopyoutOp(actualSpillOp, actualSpillOp->GetInputOperand(0), gmTensor);
    Operation* allocOp = CreateAllocOp(l1Tensor);
    Operation* copyinOp = CreateCopyinOp(gmTensor, l1Tensor, gmTensor->GetOffset());
    Operation* reshapeOp = CreateReshapeOp(l1Tensor, reshapeTensor);
}

// tensor(UB/L1)*n--> spillOp(Assemble/L0C_COPY_L1)*n --> spillTensor(UB/L1)
Status OoOScheduler::SpillMultiProducerBuffer(int spillMemid, Operation* spillOp, LogicalTensorPtr spillTensor, Operation* spillAllocOp) {
    LogicalTensorPtr gmTensor = CreateGMTensor(spillTensor, spillMemid);
    LogicalTensorPtr assembleOOperand = CreateLocalTensor(spillTensor);

    Operation *copyoutOp = CreateCopyoutOp(spillOp, spillTensor, gmTensor);
    for (auto &op : spillTensor->GetProducers()) {
        for (auto &producer : op->ProducerOps()) {
            if (opIsAllocMap[producer]) {
                producer->UpdateOutputOperand(0, spillTensor);
            }
        }
    }
    Operation* allocOp = CreateAllocOp(assembleOOperand);
    for (auto &op : spillTensor->GetProducers()) {
        if (opIsRetiredMap[op]) {
            CreateParticalBuffer(spillMemid, op, assembleOOperand, copyoutOp, spillAllocOp);
        } else {
            op->ReplaceOutput(assembleOOperand, spillTensor);
        }
    }

    auto corePair = tensorAllocCoreMap[opReqMemIdsMap[spillAllocOp][0]];
    if (bufferManagerMap[corePair.first][corePair.second][assembleOOperand->GetMemoryTypeOriginal()].Free(spillMemid) !=
        SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Free spill tensor[%d] failed!", spillMemid);
        return FAILED;
    }
    if (UpdateRemainMemid(spillMemid, assembleTensor->memoryrange.memId)) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateRemainMemid failed.");
        return FAILED;
    }
    bufRefCount_[assembleTensor->memoryrange.memId] = 0;
    for (auto op: orderedOps) {
        if (opIsRetiredMap[op]) {
            continue;
        }
        auto& reqMemIds = opReqMemIdsMap[op];
        for (auto memId : reqMemIds) {
            if (memId == assembleTensor->memoryrange.memId) {
                bufRefCount_[assembleTensor->memoryrange.memId]++;
            }
        }
    }
    depManager_.InitDependencies(orderedOps, false);
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
    localBufferMap[rawMagic] =
        std::make_shared<LocalBuffer>(rawMagic, localTensor->tensor->GetRawDataSize(), localTensor->GetMemoryTypeOriginal());
    tensorAllocCoreMap[rawMagic] = tensorAllocCoreMap[spillTensor->memoryrange.memId];
    localTensor->offset = std::vector<int64_t>(localTensor->GetShape().size(), 0);
    return localTensor;
}

LogicalTensorPtr OoOScheduler::CreateGMTensor(LogicalTensorPtr spillTensor, int spillMemId) {
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
    if (localBufferMap.find(spillMemId) == localBufferMap.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find Tensor[%d] in localBufferMap.", spillMemId);
        return nullptr;
    }
    tensor->memoryrange =
        TileRange(workspaceOffset, workspaceOffset + localBufferMap[spillMemId]->size, workspaceMemId++);
    workspaceOffset += localBufferMap[spillMemId]->size;
}

LogicalTensorPtr OoOScheduler::CreateParticalTensor(
    LogicalTensorPtr iOperand, LogicalTensorPtr assembleOOperand, LogicalTensorPtr spillTensor,
    std::vector<int64_t> toOffset)
{
    LogicalTensorPtr localTensor =
        std::make_shared<LogicalTensor>(function_, iOperand->Datatype(), iOperand->GetShape(), iOperand->Format());
    localTensor->SetMemoryTypeToBe(assembleOOperand->GetMemoryTypeToBe());
    localTensor->SetMemoryTypeOriginal(assembleOOperand->GetMemoryTypeOriginal());
    localTensor->oriShape = iOperand->shape;
    localTensor->tensor = assembleOOperand->tensor;
    localTensor->memoryrange.memId = assembleOOperand->memoryrange.memId;
    localTensor->UpdateDynValidShape(spillTensor->GetDynValidShape());
    localTensor->offset = toOffset;
    tensorAllocCoreMap[localTensor->memoryrange.memId] = tensorAllocCoreMap[iOperand->memoryrange.memId];
    return localTensor;
}

Operation* OoOScheduler::CreateAllocOp(LogicalTensorPtr oOperand) {
    Opcode opcode = 
        oOperand->GetMemoryTypeOriginal() == MemoryType::MEM_UB ? Opcode::OP_UB_ALLOC : Opcode::OP_L1_ALLOC;
    Operation& allocOp = function_.AddRawOperation(opcode, {}, {oOperand});
    allocOp.UpdateLatency(0);
    return &allocOp;
}

Operation* OoOScheduler::CloneCopyinOp(Operation* spillOp, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand) {
    Operation& copyinOp = spillOp->CloneOperation(function_, {iOperand}, {oOperand});
    copyinOp.SetIOpAttrOffset(0, spillOp->GetIOpAttrOffset(0));
    copyinOp.SetOpAttribute(spillOp->GetOpAttribute()->Clone());
    copyinOp.inParamLocation_ = spillOp->inParamLocation_;
    copyinOp.UpdateLatency(DEFINE_LATENCY);
    return &copyinOp;
}

Operation* OoOScheduler::CreateCopyinOp(LogicalTensorPtr iOperand, LogicalTensorPtr oOperand, std::vector<int64_t> offset) {
    Operation& copyinOp = function_.AddRawOperation(Opcode::OP_COPY_IN, {iOperand}, {oOperand});
    copyinOp.SetAttr(OpAttributeKey::workspaceBaseOffset, workspaceOffset);
    copyinOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(offset), // 搬运GM上的偏移
        oOperand->GetMemoryTypeOriginal(),
        OpImmediate::Specified(oOperand->GetShape()), // 搬运数据量
        OpImmediate::Specified(oOperand->tensor->GetDynRawShape()))); // 暂未使用
    copyinOp.UpdateLatency(DEFINE_LATENCY);
    if (oOperand->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
        if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
            copyinOp.SetAttribute(OpAttributeKey::copyInMode, static_cast<int64_t>(Matrix::CopyInMode::ND2NZ));
        } else {
            copyinOp.SetAttribute(OpAttributeKey::copyInMode, static_cast<int64_t>(Matrix::CopyInMode::ND2NZ));
        }
    }
    return &copyinOp;
}

Operation* OoOScheduler::CreateCopyoutOp(Operation* spillOp, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand) {
    Operation &copyoutOp = function_.AddRawOperation(Opcode::OP_COPY_OUT, {iOperand}, {oOperand});
    copyoutOp.SetAttr(OpAttributeKey::workspaceBaseOffset, workspaceBaseOffset - localBufferMap[iOperand->memoryrange.memId]->size);
    copyoutOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        iOperand->GetMemoryTypeOriginal(), 
        OpImmediate::Specified(oOperand->GetOffset()),
        OpImmediate::Specified(iOperand->GetShape()),
        OpImmediate::Specified(iOperand->GetRawTensor()->GetDynRawShape())));
    if (spillOp->HasAttr(OpAttributeKey::scaleValue, scaleValue)) {
        Element scaleValue = Element(DataType::DT_UINT64, 0);
        spillOp->GetAttr(OpAttributeKey::scaleValue, scaleValue);
        copyoutOp.SetAttribute(OpAttributeKey::scaleValue, scaleValue);
    }
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
        if (iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_L0C) {
            copyOutOp.SetAttribute(OpAttributeKey::copyIsNZ, 0);
        }
    } else {
        if (iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
            copyOutOp.SetAttribute(OpAttributeKey::copyOutMode, static_cast<int64_t>(Matrix::CopyOutMode::ND2ND));
        }
    }
    return &copyoutOp;
}

Operation* OoOScheduler::CreateReshapeOp(LogicalTensorPtr iOperand, LogicalTensorPtr oOperand) {
    Operation& reshapeOp = function_.AddRawOperation(Opcode::OP_COPY_IN, {iOperand}, {oOperand});
    reshapeOp.UpdateLatency(0);
    return &reshapeOp;
}

Operation* OoOScheduler::CreateAssembleOp(LogicalTensorPtr iOperand, LogicalTensorPtr oOperand,
    AssembleOpAttribute assembleAttr) {
    Operation& assembleOp = function_.AddRawOperation(Opcode::OP_ASSEMBLE, {iOperand}, {oOperand});
    assembleOp.UpdateLatency(1);
    assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(assembleAttr->GetFrom(),
        assembleAttr->GetToOffset(), assembleAttr->GetToDynOffset(), assembleAttr->GetFromDynValidShape()));
    
}

Status OoOScheduler::CreateParticalBuffer(int spillMemid, Operation* producerOp, LogicalTensorPtr assembleOOperand, 
    Operation* copyoutOp, Operation* spillAllocOp) {
    LogicalTensorPtr gmTensor = copyoutOp->GetOutputOperand(0);
    LogicalTensorPtr spillTensor = copyoutOp->GetInputOperand(0);
    auto assembleAttr = std::static_pointer_cast<AssembleOpAttribute>(producerOp->GetOpAttribute());
    std::vector<int64_t> toOffset = assembleAttr->GetToOffset();
    LogicalTensorPtr assembleIOperand = CreateParticalTensor(gmTensor, assembleOOperand, spillTensor, toOffset);

    Operation* copyinOp = CreateCopyinOp(gmTensor, assembleIOperand, toOffset);
    int64_t gmRelatOffset = CalcWorkspaceOffset(assembleOOperand->GetShape(), toOffset);
    if (gmRelatOffset == -1) {
        APASS_LOG_ERROR_F(Elements::Operation, "CalcWorkspaceOffset failed.");
        return FAILED;
    }
    copyinOp->SetAttr(OpAttributeKey::workspaceBaseOffset, gmRelatOffset + workspaceBaseOffset - localBufferMap[gmTensor->memoryrange.memId]->size);

    UpdateOpInternalSubgraphID(copyinOp, spillAllocOp);
    auto& corePair = opCoreLocationMap[spillAllocOp];

    // 初始化Operation属性到map
    int bufNextUseTime = GetBufNextUseTime(spillAllocOp, spillMemId);
    opExecOrderMap[copyinOp] = bufNextUseTime++;
    opPipeTypeMap[copyinOp] = RescheduleUtils::GetOpPipeType(newOpPtr);
    opIsAllocMap[copyinOp] = false;
    opIsRetiredMap[copyinOp] = false;
    opReqMemIdsMap[copyinOp] = {assembleOOperand->memoryrange.memId};

    depManager_.RegisterOp(copyinOp);
    opCoreLocationMap[copyinOp] = corePair;
    // 插入orderedOps
    InsertOrdered(copyinOp);

    Operation* assebmleOp = CreateAssembleOp(assembleIOperand, assembleOOperand, assembleAttr);
    UpdateOpInternalSubgraphID(assebmleOp, spillAllocOp);
    auto& corePair = opCoreLocationMap[spillAllocOp];

    opExecOrderMap[assebmleOp] = bufNextUseOrder;
    opPipeTypeMap[assebmleOp] = RescheduleUtils::GetOpPipeType(newOpPtr);
    opIsAllocMap[assebmleOp] = false;
    opIsRetiredMap[assebmleOp] = false;
    opReqMemIdsMap[assebmleOp] = {assembleOOperand->memoryrange.memId, assembleOOperand->memoryrange.memId};

    depManager_.RegisterOp(assebmleOp);
    opCoreLocationMap[assebmleOp] = corePair;
    // 插入orderedOps
    InsertOrdered(assebmleOp);
}

LogicalTensorPtr OoOScheduler::GetSpillTensor(Operation* spillOp, int spillMemId, Operation* spillAllocOp) {
    int spillTensorIdx = GetOOperandIdx(spillOp, spillMemId);
    return spillOp->GetOutputOperand(spillTensorIdx);
}

// 新增：基于Operation*的版本
Status OoOScheduler::UpdateSpillOpDepend(Operation* copyinOp, Operation* allocOp, Operation* spillOp, int spillMemId) {
    auto& successors = GetSuccessors(spillOp);
    for (auto succOp : successors) {
        if (!opIsRetiredMap[succOp]) {
            auto& reqMemIds = opReqMemIdsMap[succOp];
            if (std::count(reqMemIds.begin(), reqMemIds.end(), spillMemId) > 0) {
                depManager_.InsertSuccessor(copyinOp, succOp);
                depManager_.RemovePredecessor(succOp, spillOp) == 0;
                depManager_.InsertPredecessor(succOp, copyinOp);
                if (copyinOp->GetOutputOperand(0) == nullptr) {
                    APASS_LOG_ERROR_F(Elements::Operation, "%s cannot find oOperand[0]. %s", GetOpInfo(copyinOp).c_str(), GetFormatBacktrace(*reloadCopyin).c_str());
                    return FAILED;
                }
                UpdateOperationInput(succOp, spillOp, copyinOp->GetOutputOperand(0));
            }
        }
    }
    for (auto& op : orderedOps) {
        if (opIsRetiredMap[op] || opIsAllocMap[op]) {
            continue;
        }
        auto predecessors = GetSuccessors(op);
        for (auto predOp : predecessors) {
            if (opIsAllocMap[predOp]) {
                auto& predReqMemIds = opReqMemIdsMap[predOp];
                if (std::find(predReqMemIds.begin(), predReqMemIds.end(), spillMemId) != predReqMemIds.end()) {
                    depManager_.RemovePredecessor(op, predOp);
                    depManager_.InsertPredecessor(op, allocOp);
                }
            }
        }
    }
    return SUCCESS;
}

void OoOScheduler::UpdateOperationInput(Operation* targetOp, Operation* spillOp, LogicalTensorPtr tensor) {
    for (size_t index = 0; index < targetOp->GetIOperands().size(); index++) {
        for (auto &inOp : targetOp->GetIOperands()[index]->GetProducers()) {
            if (IsViewOp(*inOp)) {
                Operation* op = SkipViewChain(inOp, true);
                UpdateTensorInputForView(*op, spillOp, tensor);
            } else if (inOp == spillOp) {
                targetOp->UpdateInputOperand(index, tensor);
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
    MemoryType bufferType, int memId, LocalBufferPtr allocBuffer, LogicalTensorPtr spillOutTensor, bool needCopyOut)
{
    OoOSchedulerCheck::SpillInfo spillInfo;
    spillInfo.spillType = bufferType;
    spillInfo.bufferCurrUsage = oooCheck.bufferLastUsage[bufferType];
    spillInfo.spillTensorSize = localBufferMap[memId]->size;
    spillInfo.spillTensorMagic = spillOutTensor->GetMagic();
    spillInfo.triggerTensorSize = allocBuffer->size;
    int allocOccupied = 0;
    for (const auto &pair : tensorOccupyMap[bufferType]) {
        if (opIsAllocMap[pair.second]) {
            allocOccupied += localBufferMap[pair.first]->size;
        }
    }
    spillInfo.allocOccupiedSize = allocOccupied;
    if (needCopyOut) {
        auto dtype = spillOutTensor->tensor->datatype;
        spillInfo.spillCopyoutSize =
            std::accumulate(spillOutTensor->shape.begin(), spillOutTensor->shape.end(), 1, std::multiplies<int64_t>()) *
            BytesOf(dtype);
    } else {
        spillInfo.spillCopyoutSize = 0;
    }
    return spillInfo;
}

// 新增：基于Operation*的版本
int OoOScheduler::GetBufLastUseOrder(Operation* op, int curMemId) {
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

// 新增：基于Operation*的版本
Operation* OoOScheduler::GetBufLastWriteOp(Operation* op, int curMemId) {
    auto targetIt = std::find(orderedOps.begin(), orderedOps.end(), op);
    if (targetIt == orderedOps.end()) {
        return nullptr;
    }
    int execOrder = opExecOrderMap[op];
    for (auto it = std::make_reverse_iterator(targetIt); it != orderedOps.rend(); it++) {
        Operation* curOp = *it;
        if (curOp == nullptr || opExecOrderMap[curOp] >= execOrder) {
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


// 新增：基于Operation*的版本
void OoOScheduler::UpdateOpInternalSubgraphID(Operation &op, Operation* srcOp) {
    if (srcOp->GetInternalSubgraphID() != NOT_IN_SUBGRAPH) {
        op.UpdateInternalSubgraphID(srcOp->GetInternalSubgraphID());
        op.SetAIVCore(srcOp->GetAIVCore());
    }
}

// 新增：基于Operation*的版本
void OoOScheduler::ReplaceTensorMemId(Operation* op, int oldMemId, int newMemId) {
    auto& reqMemIds = opReqMemIdsMap[op];
    for (auto memId : reqMemIds) {
        if (memId == oldMemId) {
            std::replace(reqMemIds.begin(), reqMemIds.end(), oldMemId, newMemId);
        }
    }
    for (auto &outTensor : op->GetOOperands()) {
        if (outTensor->memoryrange.memId == oldMemId) {
            outTensor->memoryrange.memId = newMemId;
        }
    }
}

// 新增：基于Operation*的版本
Status OoOScheduler::UpdateRemainMemid(int oldMemId, int newMemId) {
    if (bufRefCount_.find(oldMemId) == bufRefCount_.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "bufRefCount cannot find Tensor[%d].", oldMemId);
        return FAILED;
    }
    bufRefCount_[newMemId] = bufRefCount_[oldMemId] + TWO_ISSUE;
    bufRefCount_[oldMemId] = 0;
    for (auto& op : orderedOps) {
        if (opIsRetiredMap[op]) {
            continue;
        }
        ReplaceTensorMemId(op, oldMemId, newMemId);
    }
    return SUCCESS;
}

// 新增：插入Operation到orderedOps
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

Status OoOScheduler::UpdateReshapeDependAndBuf(Operation* allocOp, SpillInfo &spillInfo, LogicalTensorPtr reshapeTensor) {
    auto& corePair = opCoreLocationMap[allocOp];
    // 依赖 reqmemId
    if (bufferManagerMap[corePair.first][corePair.second][spillInfo.spillTensor_->GetMemoryTypeOriginal()].Free(
            spillInfo.spillMemId_) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Free spill tensor[%d] failed!", spillInfo.spillMemId_);
        return FAILED;
    }
    if (UpdateRemainMemid(spillInfo.spillMemId_, reshapeTensor->memoryrange.memId)) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateRemainMemid failed.");
        return FAILED;
    }
    bufRefCount_[reshapeTensor->memoryrange.memId] = 0;
    for (auto op: orderedOps) {
        if (opIsRetiredMap[op]) {
            continue;
        }
        auto& reqMemIds = opReqMemIdsMap[op];
        for (auto memId : reqMemIds) {
            if (memId == reshapeTensor->memoryrange.memId) {
                bufRefCount_[reshapeTensor->memoryrange.memId]++;
            }
        }
    }
    depManager_.InitDependencies(orderedOps, false);
    return SUCCESS;
}

LogicalTensorPtr OoOScheduler::CreateReshapeL1Tensor(LogicalTensorPtr iOperand, LogicalTensorPtr reshapeTensor)
{
    LogicalTensorPtr newTensor =
        std::make_shared<LogicalTensor>(function_, iOperand->Datatype(), iOperand->shape, iOperand->Format());
    newTensor->SetMemoryTypeToBe(iOperand->GetMemoryTypeToBe());
    newTensor->SetMemoryTypeOriginal(iOperand->GetMemoryTypeOriginal());
    newTensor->oriShape = iOperand->shape;
    newTensor->tensor = reshapeTensor->tensor;
    newTensor->memoryrange.memId = reshapeTensor->memoryrange.memId;
    newTensor->UpdateDynValidShape(iOperand->GetDynValidShape());
    newTensor->offset = iOperand->GetOffset();
    tensorAllocCoreMap[newTensor->memoryrange.memId] = tensorAllocCoreMap[iOperand->memoryrange.memId];
    return newTensor;
}

// 新增：基于Operation*的版本
Status OoOScheduler::SpillReshapeParticalBuffer(SpillInfo &spillInfo, Operation* allocOp, LogicalTensorPtr reshapeTensor) {
    auto iOperand = spillInfo.spillOp_->GetInputOperand(0);
    LogicalTensorPtr newTensor = CreateReshapeL1Tensor(iOperand, reshapeTensor);
    int bufNextUseOrder = GetBufNextUseTime(allocOp, spillInfo.spillMemId_);
    if (bufNextUseOrder == -1) {
        APASS_LOG_ERROR_F(Elements::Operation, "Get Tensor[%d] next use time failed.", spillInfo.spillMemId_);
        return FAILED;
    }
    // 创建 alloc
    auto& spillAllocOp = function_.AddRawOperation(Opcode::OP_L1_ALLOC, {}, {newTensor});
    spillAllocOp.UpdateLatency(1);
    auto spillAllocOpPtr = UpdateIssueAttr(spillAllocOp, {reshapeTensor->memoryrange.memId}, allocOp, bufNextUseOrder);
    // 创建 copyin
    Operation* preOp = nullptr;
    auto& predecessors = GetPredecessors(spillInfo.spillOp_);
    for (auto predOp : predecessors) {
        if (!opIsAllocMap[predOp]) {
            preOp = predOp;
        }
    }
    if (preOp == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "preOp is nullptr");
        return FAILED;
    }
    if (preOp->GetOpcode() != Opcode::OP_COPY_IN && preOp->GetInputOperand(0)->GetMemoryTypeOriginal() != MemoryType::MEM_UB &&
            preOp->GetInputOperand(0)->GetMemoryTypeOriginal() != MemoryType::MEM_L0C) {
        APASS_LOG_ERROR_F(Elements::Operation, "The preOp of reshape is not COPY_IN/UB_COPY_L1/L0C_COPY_L1");
        return FAILED;
    }
    auto &spillCopyInOp = (preOp->GetOpcode() == Opcode::OP_COPY_IN) ?
        preOp->CloneOperation(function_, {spillInfo.ddrTensor_}, {newTensor}) :
        function_.AddRawOperation(Opcode::OP_COPY_IN, {spillInfo.ddrTensor_}, {newTensor});
    if (preOp->GetOpcode() == Opcode::OP_COPY_IN) {
        spillCopyInOp.SetIOpAttrOffset(0, preOp->GetIOpAttrOffset(0));
    }
    int64_t base = 0;
    GetWorkspaceBaseOffset(spillInfo.ddrTensor_, base);
    UpdateOpAttr(spillCopyInOp, DEFAULT_LATENCY, newTensor, spillInfo.ddrTensor_->GetOffset(), preOp, base);
    auto spillCopyInOpPtr = UpdateIssueAttr(spillCopyInOp, {reshapeTensor->memoryrange.memId}, allocOp, bufNextUseOrder);
    // A5 下 DDR->COPY_IN->L1->RESHAPE->L1 场景不标记 copy_in_mode
    if (preOp->GetOpcode() != Opcode::OP_COPY_IN && UpdateCopyInMode(spillCopyInOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateCopyInMode failed");
        return FAILED;
    }
    // 创建 reshape
    auto& reshapeOp = function_.AddRawOperation(Opcode::OP_RESHAPE, {newTensor}, {reshapeTensor});
    reshapeOp.UpdateLatency(1);
    auto reshapeOpPtr = UpdateIssueAttr(reshapeOp, {reshapeTensor->memoryrange.memId, reshapeTensor->memoryrange.memId}, allocOp, bufNextUseOrder);
    APASS_LOG_DEBUG_F(Elements::Operation, "Add SPILL_ALLOC: %s. ", GetOpInfo(spillAllocOpPtr).c_str());
    APASS_LOG_DEBUG_F(Elements::Operation, "Add SPILL_IN: %s. ", GetOpInfo(spillCopyInOpPtr).c_str());
    APASS_LOG_DEBUG_F(Elements::Operation, "Add SPILL_RESHAPE: %s. ", GetOpInfo(reshapeOpPtr).c_str());
    return SUCCESS;
}

// 新增：基于Operation*的版本
Status OoOScheduler::SpillInReshapeBuffer(SpillInfo &spillInfo, Operation* allocOp) {
    LogicalTensorPtr reshapeTensor = std::make_shared<LogicalTensor>(function_,
        spillInfo.spillTensor_->Datatype(), spillInfo.spillTensor_->shape, spillInfo.spillTensor_->Format());
    if (reshapeTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Create reshape tensor failed!");
        return FAILED;
    }
    if (UpdateTensorAttr(reshapeTensor, spillInfo.spillTensor_->GetMemoryTypeOriginal(), spillInfo.spillTensor_, -1) !=
        SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateTensorAttr reshape tensor failed!");
        return FAILED;
    }
    auto &successors = GetSuccessors(spillInfo.spillOp_);
    for (auto succOp : successors) {
        if (!opIsRetiredMap[succOp]) {
            auto& reqMemIds = opReqMemIdsMap[succOp];
            if (std::count(reqMemIds.begin(), reqMemIds.end(), spillInfo.spillMemId_) > 0) {
                UpdateOperationInput(succOp, spillInfo.spillOp_, reshapeTensor);
            }
        }
    }
    if (SpillReshapeParticalBuffer(spillInfo, allocOp, reshapeTensor) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "SpillReshapeParticalBufferOp failed!");
        return FAILED;
    }
    // 依赖关系 memId
    if (UpdateReshapeDependAndBuf(allocOp, spillInfo, reshapeTensor) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateReshapeDependAndBufOp failed!");
        return FAILED;
    }

    return SUCCESS;
}


int64_t OoOScheduler::CalcWorkspaceOffset(std::vector<int64_t> shape, std::vector<int64_t> offset)
{
    if (shape.size() != offset.size()) {
        return -1;
    }
    if (shape.size() == 0) {
        return 0;
    }

    int64_t result = 0;
    int64_t stride = 1;
    // 从最低维到最高维计算
    for (size_t i = shape.size(); i > 0; --i) {
        result += offset[i - 1] * stride;
        if (i > 0) {
            stride *= shape[i - 1];
        }
    }
    return result;
}

bool OoOScheduler::CanAllocateAll(std::vector<LocalBufferPtr> tensors, MemoryType memType)
{
    if (tensors.empty()) {
        APASS_LOG_INFO_F(Elements::Operation, "CanAllocateAll input tensors is empty.");
        return true;
    }
    auto corePair = tensorAllocCoreMap[tensors[0]->id];
    std::map<uint64_t, std::map<uint64_t, uint64_t>> freeIntervals =
        bufferManagerMap[corePair.first][corePair.second][memType].FindFreeIntervals();
    for (auto tensor : tensors) {
        bool canAlloc = false;
        std::pair<uint64_t, uint64_t> newInterval;
        uint64_t allocInterval;
        uint64_t allocAddrStart;
        for (auto& interval : freeIntervals) {
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

// 新增：基于Operation*的版本
int OoOScheduler::GetMemidAllocPriority(int memId) {
    for (auto op : orderedOps) {
        if (!opIsAllocMap[op]) {
            continue;
        }
        auto& reqMemIds = opReqMemIdsMap[op];
        if (!reqMemIds.empty() && reqMemIds[0] == memId) {
            return opExecOrderMap[op];
        }
    }
    return -1;
}

bool OoOScheduler::HasEnoughBuffer(Operation* allocOp, MemoryType memType) {
    std::vector<LocalBufferPtr> tensors;
    std::vector<int> memIds;
    if (allocOp->GetOOperands().size() != 1) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s must only have one ooperand.", GetFormatBacktrace(*allocOp).c_str());
        return false;
    }
    for (auto &succOp : GetSuccessors(allocOp)) {
        if (succOp != *(allocOp->GetOutputOperand(0)->GetProducers().begin())) {
            continue;
        }
        for (auto &memId : opReqMemIdsMap[succOp]) {
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

Status OoOScheduler::RearrangeBuffer(Operation* allocOp, MemoryType memType, std::pair<OpCoreType, int> corePair) {
    std::vector<int> memIds = bufferManagerMap[corePair.first][corePair.second][memType].GetAddrSortedBufs();
    for (auto memId : memIds) {
        auto op = GetSpillIssue(allocOp, memId);
        if (op == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] last write issue.", memId);
            return FAILED;
        }
        if (op->GetOpcodeStr().find("ALLOC") == std::string::npos) {
            return FAILED;
        }
    }
    return bufferManagerMap[corePair.first][corePair.second][memType].CompactBufferSlices(localBufferMap);
}


} // namespace npu::tile_fwk
