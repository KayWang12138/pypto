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
 * \file spill_buffer.cpp
 * \brief
 */

#include "scheduler.h"

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

int OoOScheduler::GetBufNextUseOrder(IssueEntryPtr issue, int curMemId) {
    auto it = std::find_if(issueEntries.begin(), issueEntries.end(), [issue, curMemId](const IssueEntryPtr a) {
        return a && a->execOrder > issue->execOrder && 
            std::find(a->reqMemIds.begin(), a->reqMemIds.end(), curMemId) != a->reqMemIds.end();
    });
    return (it != issueEntries.end()) ? (*it)->execOrder : -1;
}

int OoOScheduler::GetBufLastUseOrder(IssueEntryPtr issue, int curMemId) {
    auto targetIt = std::find(issueEntries.begin(), issueEntries.end(), issue);
    if (targetIt == issueEntries.end()) {
        return -1;
    }
    for (auto it = std::make_reverse_iterator(targetIt); it != issueEntries.rend(); it++) {
        IssueEntryPtr curIssue = *it;
        if (curIssue && curIssue->execOrder < issue->execOrder && std::find(curIssue->reqMemIds.begin(), 
            curIssue->reqMemIds.end(), curMemId) != curIssue->reqMemIds.end()) {
            return curIssue->execOrder;
        }
    }
    return -1;
}

IssueEntryPtr OoOScheduler::GetBufLastWriteIssue(IssueEntryPtr issue, int curMemId) {
    auto targetIt = std::find(issueEntries.begin(), issueEntries.end(), issue);
    if (targetIt == issueEntries.end()) {
        return nullptr;
    }
    for (auto it = std::make_reverse_iterator(targetIt); it != issueEntries.rend(); it++) {
        IssueEntryPtr curIssue = *it;
        if (curIssue == nullptr || curIssue->execOrder >= issue->execOrder) {
            continue;
        }
        for (auto& outTensor : curIssue->tileOp.GetOOperands()) {
            if (outTensor->memoryrange.memId == curMemId) {
                return curIssue;
            }
        }
    }
    return nullptr;
}

Status OoOScheduler::UpdateTensorAttr(
    LogicalTensorPtr tensor, MemoryType memType, LogicalTensorPtr spillTensor, int spillMemId) {
    tensor->SetMemoryTypeToBe(memType);
    tensor->SetMemoryTypeOriginal(memType);
    tensor->oriShape = spillTensor->oriShape;
    tensor->SetMagic(++maxTensorMagic);
    tensor->UpdateDynValidShape(spillTensor->GetDynValidShape());
    if (memType == MEM_DEVICE_DDR) {
        if (localBufferMap.find(spillMemId) == localBufferMap.end()) {
            ALOG_ERROR_F("Cannot find Tensor[%d] in localBufferMap.", spillMemId);
            return FAILED;
        }
        tensor->memoryrange =
            TileRange(workspaceOffset, workspaceOffset + localBufferMap[spillMemId]->size, workspaceMemId++);
        workspaceOffset += localBufferMap[spillMemId]->size;
    } else {
        tensor->memoryrange.memId = maxTensorMagic;
        localBufferMap[maxTensorMagic] = std::make_shared<LocalBuffer>(
            maxTensorMagic, ShapeCeilAlign(tensor->GetShape(), tensor->Datatype()), tensor->GetMemoryTypeOriginal());
        if (localBufferMap[maxTensorMagic] == nullptr) {
            ALOG_ERROR_F("Init Tensor[%d] localBuffer failed.", maxTensorMagic);
            return FAILED;
        }
    }
    return SUCCESS;
}

void OoOScheduler::UpdateOpAttr(
    Operation &op, int opLatency, LogicalTensorPtr spillTensor, std::vector<int64_t> offset, IssueEntryPtr spillIssue) {
    if (op.GetOpcode() == Opcode::OP_COPY_OUT) {
        op.SetOpAttribute(std::make_shared<CopyOpAttribute>(spillTensor->GetMemoryTypeOriginal(),
            OpImmediate::Specified(offset), OpImmediate::Specified(spillTensor->GetShape()),
            OpImmediate::Specified(spillTensor->GetRawTensor()->GetDynRawShape())));
    } else if (op.GetOpcodeStr().find("ALLOC") == std::string::npos) {
        if (spillIssue->tileOp.GetOpcode() == Opcode::OP_COPY_IN) {
            op.SetOpAttribute(spillIssue->tileOp.GetOpAttribute());
            op.inParamLocation_ = spillIssue->tileOp.inParamLocation_;
        } else {
            op.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified(offset),
                spillTensor->GetMemoryTypeOriginal(), OpImmediate::Specified(spillTensor->GetShape()),
                OpImmediate::Specified(spillTensor->tensor->GetDynRawShape())));
        }
    }
    op.UpdateLatency(opLatency);
    op.opmagic = ++maxOpMagic;
}

void OoOScheduler::ReplaceTensorMemId(IssueEntryPtr &issue, int oldMemId, int newMemId) {
    for (auto memId : issue->reqMemIds) {
        if (memId == oldMemId) {
            std::replace(issue->reqMemIds.begin(), issue->reqMemIds.end(), oldMemId, newMemId);
        }
    }
    for (auto &outTensor : issue->tileOp.GetOOperands()) {
        if (outTensor->memoryrange.memId == oldMemId) {
            outTensor->memoryrange.memId = newMemId;
        }
    }
}

Status OoOScheduler::UpdateRemainOpBufId(int oldMemId, int newMemId) {
    if (bufRefCount.find(oldMemId) == bufRefCount.end()) {
        ALOG_ERROR_F("bufRefCount cannot find Tensor[%d]", oldMemId);
        return FAILED;
    }
    bufRefCount[newMemId] = bufRefCount[oldMemId] + TWO_ISSUE;
    bufRefCount[oldMemId] = 0;
    for (auto& issue : issueEntries) {
        if (issue->isRetired) {
            continue;
        }
        ReplaceTensorMemId(issue, oldMemId, newMemId);
    }
    return SUCCESS;
}

Status OoOScheduler::UpdateReloadIssueDepend(IssueEntryPtr reloadCopyin, IssueEntryPtr spillIssue, int spillMemId) {
    for (auto& succId : spillIssue->successors) {
        auto succ = issueEntryMap[succId];
        if (!succ->isRetired && (std::count(succ->reqMemIds.begin(), succ->reqMemIds.end(), spillMemId) > 0)) {
            reloadCopyin->successors.insert(succ->id);
            if (succ->predecessors.erase(spillIssue->id) == 0) {
                ALOG_ERROR_F("Erase issueEntry %s failed", spillIssue->GetOpInfo());
                return FAILED;
            }
            succ->predecessors.insert(reloadCopyin->id);
            if (reloadCopyin->tileOp.GetOutputOperand(0) == nullptr) {
                ALOG_ERROR_F("%s cannot find oOperand[0]", reloadCopyin->GetOpInfo());
                return FAILED;
            }
            succ->UpdateTensorInput(spillIssue, reloadCopyin->tileOp.GetOutputOperand(0));
        }
    }
    return SUCCESS;
}

Status OoOScheduler::UpdateReloadIssueInfo(IssueEntryPtr reloadAlloc, IssueEntryPtr reloadCopyin,
    IssueEntryPtr spillIssue, int spillMemId, IssueEntryPtr allocIssue) {
    reloadAlloc->reqMemIds = {maxTensorMagic};
    reloadAlloc->successors.insert(reloadCopyin->id);
    reloadCopyin->reqMemIds = {maxTensorMagic};
    reloadCopyin->predecessors.insert(reloadAlloc->id);
    
    int bufNextUseOrder = GetBufNextUseOrder(allocIssue, spillMemId);
    if (bufNextUseOrder == -1) {
        ALOG_ERROR_F("Get Tensor[%d] next use order failed.", spillMemId);
        return FAILED;
    }
    reloadAlloc->execOrder = bufNextUseOrder++;
    InsertIssueEntries(reloadAlloc);
    reloadCopyin->execOrder = bufNextUseOrder;
    InsertIssueEntries(reloadCopyin);

    if (UpdateReloadIssueDepend(reloadCopyin, spillIssue, spillMemId) != SUCCESS) {
        ALOG_ERROR_F("UpdateReloadIssueDepend failed.");
        return FAILED;
    }
    if (UpdateRemainOpBufId(spillMemId, reloadAlloc->reqMemIds[0])) {
        ALOG_ERROR_F("UpdateRemainOpBufId failed.");
        return FAILED;
    }
    numTotalIssues += TWO_ISSUE;
    return SUCCESS;
}

Status OoOScheduler::CreateSpillReloadIssue(LogicalTensorPtr spillOutTensor,
    LogicalTensorPtr spillTensor, IssueEntryPtr &spillIssue, std::pair<IssueEntryPtr, IssueEntryPtr> &reloadIssues) {
    MemoryType memType = spillTensor->GetMemoryTypeOriginal();
    // 创建将spill搬出数据搬回OP_COPY_IN的tensor
    LogicalTensorPtr localTensor = std::make_shared<LogicalTensor>(
            function_, spillTensor->Datatype(), spillTensor->shape);
    if (localTensor == nullptr) {
        ALOG_ERROR_F("Create local tensor failed!");
        return FAILED;
    }
    if (UpdateTensorAttr(localTensor, memType, spillTensor, -1) != SUCCESS) {
        ALOG_ERROR_F("UpdateTensorAttr local tensor failed!");
        return FAILED;
    }

    // 创建spill搬出数据搬回OP_COPY_IN/OP_ALLOC
    Opcode allocOp = memType == MemoryType::MEM_UB ? Opcode::OP_UB_ALLOC : Opcode::OP_L1_ALLOC;
    auto &spillAllocOp = function_.AddRawOperation(allocOp, {}, {localTensor});
    auto &spillCopyInOp = function_.AddRawOperation(Opcode::OP_COPY_IN, {spillOutTensor}, {localTensor});

    UpdateOpAttr(spillAllocOp, 1, localTensor, {}, spillIssue);
    UpdateOpAttr(spillCopyInOp, DEFAULT_LATENCY, localTensor, spillOutTensor->GetOffset(), spillIssue);

    // 初始化OP_COPY_IN/OP_ALLOC的issueEntry
    IssueEntryPtr spillAllocInst = std::make_shared<IssueEntry>(spillAllocOp, issueId);
    issueEntryMap[issueId++] = spillAllocInst;
    IssueEntryPtr spillInInst = std::make_shared<IssueEntry>(spillCopyInOp, issueId);
    issueEntryMap[issueId++] = spillInInst;
    if (spillAllocInst == nullptr || spillInInst == nullptr) {
        ALOG_ERROR_F("Create OP_COPY_IN/OP_ALLOC issueEntry failed!");
        return FAILED;
    }
    reloadIssues.first = spillAllocInst;
    reloadIssues.second = spillInInst;
    ALOG_DEBUG_F("Add SPILL_ALLOC: %s.", spillAllocInst->GetOpInfo());
    ALOG_DEBUG_F("Add SPILL_IN: %s.", spillInInst->GetOpInfo());
    return SUCCESS;
}

Status OoOScheduler::SpillInBuffer(SpillInfo &spillInfo, IssueEntryPtr allocIssue, MemoryType bufferType, 
    bool isGenSpill) {
    IssueEntryPtr reloadCopyin = nullptr;
    IssueEntryPtr reloadAlloc = nullptr;
    std::pair<IssueEntryPtr, IssueEntryPtr> reloadIssues = {reloadAlloc, reloadCopyin};
    if (CreateSpillReloadIssue(spillInfo.ddrTensor_, spillInfo.spillTensor_, spillInfo.spillIssue_,
        reloadIssues) != SUCCESS) { 
        ALOG_ERROR_F("CreateSpillReloadIssue failed!"); 
        return FAILED; 
    }
    reloadAlloc = reloadIssues.first;
    reloadCopyin = reloadIssues.second;
    if (UpdateReloadIssueInfo(reloadAlloc, reloadCopyin, spillInfo.spillIssue_, spillInfo.spillMemId_, 
        allocIssue) != SUCCESS) { 
        ALOG_ERROR_F("UpdateReloadIssueInfo failed!"); 
        return FAILED; 
    }
    if (!isGenSpill) {
        allocIssueQueue[bufferType].Insert(reloadAlloc, reloadAlloc->execOrder);
    }
    if (bufferManagerMap[bufferType].Free(spillInfo.spillMemId_) != SUCCESS) {
        ALOG_ERROR_F("Free spill tensor[%d] failed!", spillInfo.spillMemId_);
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::CreateSpillCopyout(IssueEntryPtr spillIssue, LogicalTensorPtr spillTensor,
    int spillMemId, IssueEntryPtr &spillCopyout) {
    // 创建spill搬出所需的DDR rawtensor/tensor
    std::shared_ptr<RawTensor> ddrRawTensor =
        std::make_shared<RawTensor>(spillTensor->Datatype(), spillTensor->GetShape(), "WorkspaceGm", SYMBOL_STACK_BASE);
    if (ddrRawTensor == nullptr) {
        ALOG_ERROR_F("Create DDR raw tensor failed!");
        return FAILED;
    }
    std::vector<int64_t> offset(spillTensor->GetShape().size(), 0);
    offset.front() = workspaceOffset;

    LogicalTensorPtr ddrTensor = std::make_shared<LogicalTensor>(function_, ddrRawTensor, offset, spillTensor->GetShape());
    if (ddrTensor == nullptr) {
        ALOG_ERROR_F("Create DDR tensor failed!");
        return FAILED;
    }
    if (UpdateTensorAttr(ddrTensor, MEM_DEVICE_DDR, spillTensor, spillMemId) != SUCCESS) {
        ALOG_ERROR_F("UpdateTensorAttr DDR tensor failed!");
        return FAILED;
    }

    // 创建spill搬出所需的DDR OP_COPY_OUT
    Operation &spillOutOp = function_.AddRawOperation(Opcode::OP_COPY_OUT, {spillTensor}, {ddrTensor});
    UpdateOpAttr(spillOutOp, DEFAULT_LATENCY, spillTensor, offset, spillIssue);

    // 创建spill搬出数据OP_COPY_OUT的issueEntry
    spillCopyout = std::make_shared<IssueEntry>(spillOutOp, issueId);
    issueEntryMap[issueId++] = spillCopyout;
    if (spillCopyout == nullptr) {
        ALOG_ERROR_F("Create OP_COPY_OUT issueEntry failed!");
        return FAILED;
    }
    spillCopyout->reqMemIds = {spillMemId};
    spillCopyout->predecessors.insert(spillIssue->id);
    spillIssue->successors.insert(spillCopyout->id);
    spillCopyout->isRetired = true;
    ALOG_DEBUG_F("Add SPILL_OUT: %s.", spillCopyout->GetOpInfo());
    return SUCCESS;
}

Status OoOScheduler::SpillOutBuffer(SpillInfo &spillInfo, IssueEntryPtr issue, size_t &pcIdx, bool isGenSpill) {
    if (spillInfo.spillIssue_->tileOp.GetOpcodeStr().find("COPY_IN") == std::string::npos) {
        IssueEntryPtr spillCopyout = nullptr;
        if (CreateSpillCopyout(spillInfo.spillIssue_, spillInfo.spillTensor_, spillInfo.spillMemId_, 
            spillCopyout) != SUCCESS) { 
            ALOG_ERROR_F("CreateSpillCopyout failed!"); 
            return FAILED; 
        }
        int bufLastUseOrder = GetBufLastUseOrder(issue, spillInfo.spillMemId_);
        if (bufLastUseOrder == -1) {
            ALOG_ERROR_F("Cannot find spill Tensor[%d] last used order.", spillInfo.spillMemId_);
            return FAILED;
        }
        spillCopyout->execOrder = bufLastUseOrder + 1;
        InsertIssueEntries(spillCopyout);
        if (isGenSpill) {
            pcIdx++;
            numTotalIssues++;
        } else {
            newOperations_.push_back(&(spillCopyout->tileOp));
            ALOG_DEBUG_F("Insert: %s", spillCopyout->GetOpInfo());
        }
        spillInfo.ddrTensor_ = spillCopyout->tileOp.GetOutputOperand(0);
    } else {
        spillInfo.ddrTensor_ = spillInfo.spillIssue_->tileOp.GetInputOperand(0);
    }
    return SUCCESS;
}

Status OoOScheduler::GetSpillTensor(IssueEntryPtr spillIssue, int spillMemId, LogicalTensorPtr &spillTensor) {
    int spillTensorIdx = spillIssue->GetOOperandIdx(spillMemId);
    if (spillTensorIdx == -1) {
        ALOG_ERROR_F("Tensor[%d] cannot find in op's oOperand", spillMemId);
        return FAILED;
    }
    spillTensor = spillIssue->tileOp.GetOutputOperand(spillTensorIdx);
    if (spillTensor == nullptr) {
        ALOG_ERROR_F("Op cannot find oOperand[%d]", spillTensorIdx);
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::SpillBuffer(SpillInfo &spillInfo, IssueEntryPtr allocIssue, size_t &pcIdx, 
    LocalBufferPtr allocBuffer, bool isGenSpill) {
    if (SpillOutBuffer(spillInfo, allocIssue, pcIdx, isGenSpill) != SUCCESS) {
        ALOG_ERROR_F("SpillOutBuffer failed.");
        return FAILED;
    }
    // Healthcheck record - spill info
    if (oooCheck.doHealthCheck) {
        oooCheck.spillInfoVec.emplace_back(
            RecordSpillInfo(allocBuffer->memType, spillInfo.spillMemId_, allocBuffer, spillInfo.ddrTensor_,
                spillInfo.spillIssue_->tileOp.GetOpcodeStr().find("COPY_IN") == std::string::npos));
    }
    if (SpillInBuffer(spillInfo, allocIssue, allocBuffer->memType, isGenSpill) != SUCCESS) {
        ALOG_ERROR_F("SpillInBuffer failed.");
        return FAILED;
    }
    if (!isGenSpill) {
        // Healthcheck record - update buffer usage statistics
        if (oooCheck.doHealthCheck) { 
            UpdateBufferUsage(allocBuffer->memType, spillInfo.spillMemId_, true); 
        }
        localBufferMap[spillInfo.spillMemId_]->retireCycle = clock;
        if (tensorOccupyMap[allocBuffer->memType].erase(spillInfo.spillMemId_) == 0) {
            ALOG_ERROR_F("Erase tensor[%d] failed", spillInfo.spillMemId_);
            return FAILED;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::GetSpillInfo(IssueEntryPtr allocIssue, int spillMemId, bool isGenSpill, 
    SpillInfo &spillInfo) {
    auto spillIssue = GetSpillIssue(allocIssue, spillMemId, isGenSpill);
    if (spillIssue == nullptr) {
        ALOG_ERROR_F("Cannot find spill Tensor[%d] last write issue.", spillMemId);
        return FAILED;
    }
    LogicalTensorPtr spillTensor = nullptr;
    if (GetSpillTensor(spillIssue, spillMemId, spillTensor) != SUCCESS) {
        ALOG_ERROR_F("%s GetSpillTensor failed!", spillIssue->GetOpInfo());
        return FAILED;
    }
    ALOG_DEBUG_F("Begin spill %s tensor[%d].", spillIssue->GetOpInfo(), spillMemId);
    LogicalTensorPtr ddrTensor = nullptr;
    spillInfo.ddrTensor_ = ddrTensor;
    spillInfo.spillTensor_ = spillTensor;
    spillInfo.spillIssue_ = spillIssue;
    spillInfo.spillMemId_ = spillMemId;
    return SUCCESS;
}

Status OoOScheduler::SpillMultiBuffer(IssueEntryPtr allocIssue, std::vector<int> spillGroup, size_t &pcIdx, 
    LocalBufferPtr allocBuffer, bool isGenSpill) {
    for (auto &spillMemId : spillGroup) {
        SpillInfo spillInfo;
        if (GetSpillInfo(allocIssue, spillMemId, isGenSpill, spillInfo) != SUCCESS) {
            ALOG_ERROR_F("GetSpillInfo failed.");
            return FAILED;
        }
        if (SpillBuffer(spillInfo, allocIssue, pcIdx, allocBuffer, isGenSpill) != SUCCESS) {
            ALOG_ERROR_F("SpillBuffer[%d] failed.", spillMemId);
            return FAILED;
        }
    }
    return SUCCESS;
}

void OoOScheduler::FindFilterLtags(IssueEntryPtr allocIssue, std::set<IssueEntryPtr> &filterLtags) {
    for (auto &dstIssueId : allocIssue->successors) {
        auto dstIssue = issueEntryMap[dstIssueId];
        for (auto &inIssueId : dstIssue->predecessors) {
            auto inIssue = issueEntryMap[inIssueId];
            filterLtags.insert(inIssue);
        }
    }
}

bool OoOScheduler::IsBelongSpillBlackList(IssueEntryPtr spillIssue, IssueEntryPtr issue) {
    std::set<IssueEntryPtr> filterLtags;
    FindFilterLtags(issue, filterLtags);
    if (spillIssue->isAlloc || spillIssue->tileOp.GetOpcode() == Opcode::OP_VIEW ||
        spillIssue->tileOp.GetOpcode() == Opcode::OP_ASSEMBLE || filterLtags.count(spillIssue) != 0) {
        return true;
    }
    return false;
}

IssueEntryPtr OoOScheduler::GetSpillIssue(IssueEntryPtr allocIssue, int memId, bool isGenSpill) {
    if (isGenSpill) {
        return GetBufLastWriteIssue(allocIssue, memId);
    } 
    return tensorOccupyMap[localBufferMap[allocIssue->reqMemIds[0]]->memType][memId];
}

Status OoOScheduler::GetGroupNextUseOrder(std::vector<int> group, IssueEntryPtr allocIssue, 
    std::vector<int> &groupNextUseTime, std::unordered_map<int, size_t> &nextUseTimeCache, bool isGenSpill) {
    std::vector<size_t> bufNextUseTime;
    for (auto& memId : group) {
        IssueEntryPtr spillIssue = GetSpillIssue(allocIssue, memId, isGenSpill);
        if (spillIssue == nullptr) {
            ALOG_ERROR_F("Cannot find spill Tensor[%d] last write issue.", memId);
            return FAILED;
        }
        if (IsBelongSpillBlackList(spillIssue, allocIssue)) {
            break;
        }
        if (nextUseTimeCache.find(memId) != nextUseTimeCache.end()) {
            bufNextUseTime.push_back(nextUseTimeCache[memId]);
        } else {
            int nextUseOrder = GetBufNextUseOrder(allocIssue, memId);
            if (nextUseOrder == -1) {
                ALOG_ERROR_F("Cannot find Tensor[%d] next used time.", memId);
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

Status OoOScheduler::SelectSpillBuffers(LocalBufferPtr allocBuffer, IssueEntryPtr allocIssue, 
    std::vector<int> &spillGroup, bool isGenSpill) {
    // 查找出可以spill 单个或多个tensor的集合
    std::vector<std::vector<int>> canSpillGroups;
    if (bufferManagerMap[allocBuffer->memType].GetSpillGroup(allocBuffer->size, canSpillGroups) != SUCCESS) {
        ALOG_ERROR_F("GetSpillGroup failed.");
        return FAILED;
    }
    if (canSpillGroups.empty()) { 
        ALOG_ERROR_F("Cannot find tensor to spill."); 
        return FAILED; 
    }
    std::unordered_map<int, size_t> nextUseTimeCache;
    std::vector<int> groupNextUseTime;
    for (auto &group : canSpillGroups) {
        if (GetGroupNextUseOrder(group, allocIssue, groupNextUseTime, nextUseTimeCache, isGenSpill) != SUCCESS) {
            ALOG_ERROR_F("GetGroupNextUseOrder failed.");
            return FAILED;
        }
    }
    size_t groupSel = std::max_element(groupNextUseTime.begin(), groupNextUseTime.end()) - groupNextUseTime.begin();
    if (groupNextUseTime[groupSel] == -1) {
        if (PrintSpillFailedInfo(allocIssue) != SUCCESS) {
            ALOG_ERROR_F("PrintSpillFailedInfo failed.");
            return FAILED;
        }
        ALOG_ERROR_F("Cannot find tensor to spill.");
        return FAILED;
    }
    spillGroup = canSpillGroups[groupSel];
    return SUCCESS;
}

Status OoOScheduler::GenBufferSpill(IssueEntryPtr allocIssue) {
    std::vector<int> spillGroup;
    if (SelectSpillBuffers(localBufferMap[allocIssue->reqMemIds[0]], allocIssue, spillGroup, false) != SUCCESS) {
        ALOG_ERROR_F("SelectSpillBuffers failed!");
        return FAILED;
    }
    size_t temp = 1;
    if (SpillMultiBuffer(allocIssue, spillGroup, temp, localBufferMap[allocIssue->reqMemIds[0]], false) != SUCCESS) { 
        ALOG_ERROR_F("SpillMultiBuffer failed!");
        return FAILED; 
    }
    return SUCCESS;
}

Status OoOScheduler::GenSpillOp(LocalBufferPtr allocBuffer, size_t &pcIdx) {
    ALOG_DEBUG_F("---> START: SPILL tensor.");
    if (allocBuffer->memType != MemoryType::MEM_L1 && allocBuffer->memType != MemoryType::MEM_UB) {
        if (PrintSpillFailedInfo(issueEntries[pcIdx]) != SUCCESS) {
            ALOG_ERROR_F("PrintSpillFailedInfo failed.");
            return FAILED;
        }
        ALOG_ERROR_F("Buffer[L0A/B/C] is Full. Please check tile shape and OOO spill failed info.");
        return FAILED;
    }
    // 选择最晚被使用的spill 单个或多个tensor
    std::vector<int> spillGroup;
    if (SelectSpillBuffers(allocBuffer, issueEntries[pcIdx], spillGroup, true) != SUCCESS) {
        ALOG_ERROR_F("SelectSpillBuffers failed!");
        return FAILED;
    }
    if (SpillMultiBuffer(issueEntries[pcIdx], spillGroup, pcIdx, allocBuffer, true) != SUCCESS) { 
        ALOG_ERROR_F("SpillMultiBuffer failed!");
        return FAILED; 
    }
    ALOG_DEBUG_F("---> END: SPILL tensor.");
    return SUCCESS;
}

} // namespace npu::tile_fwk