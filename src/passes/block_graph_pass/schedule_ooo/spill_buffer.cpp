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

Status OoOScheduler::GetOldestBuffer(IssueEntryPtr allocIssue, MemoryType bufferType, int& memId,
    IssueEntryPtr &spillIssue) {
    std::set<IssueEntryPtr> filterLtags;
    FindFilterLtags(allocIssue, filterLtags);
    uint64_t maxIdx = 0;
    for (auto &occupyIssue : tensorOccupyMap[bufferType]) {
        if (occupyIssue.second->isAlloc || filterLtags.count(occupyIssue.second) != 0 ||
            USE_LESS_OPS.find(occupyIssue.second->tileOp.GetOpcode()) != USE_LESS_OPS.end()) {
            continue;
        }
        auto occupyTensor = localBufferMap[occupyIssue.first];
        if ((occupyTensor->end - occupyTensor->start) >= localBufferMap[allocIssue->reqMemIds[0]]->size) {
            size_t nextUseTime = occupyIssue.second->execOrder;
            if (!GetBufNextUseTime(occupyIssue.first, nextUseTime)) {
                ALOG_ERROR_F("Cannot find spill Tensor[%d] next used time.", occupyIssue.first);
                return FAILED;
            }
            if (maxIdx < nextUseTime) {
                maxIdx = nextUseTime;
                spillIssue = occupyIssue.second;
                memId = occupyIssue.first;
            }
        }
    }
    if (spillIssue == nullptr || memId == -1) {
        PrintSpillFailedInfo(allocIssue, bufferType);
        ALOG_ERROR_F("Could not find availalble buffer to spill!"); 
        return FAILED; 
    }
    ALOG_DEBUG_F("  Spill op: %s.", spillIssue->GetOpInfo());
    return SUCCESS;
}

Status OoOScheduler::UpdateTensorAttr(
    LogicalTensorPtr tensor, MemoryType memType, LogicalTensorPtr spillTensor, int spillMemId) {
    tensor->SetMemoryTypeToBe(memType);
    tensor->SetMemoryTypeOriginal(memType);
    tensor->subGraphID = subGraphID;
    tensor->oriShape = spillTensor->oriShape;
    tensor->SetMagic(++maxTensorMagic);
    tensor->UpdateDynValidShape(spillTensor->GetDynValidShape());
    if (memType == MEM_DEVICE_DDR) {
        if (localBufferMap.find(spillMemId) == localBufferMap.end()) {
            ALOG_ERROR_F("Cannot find Tensor[%d] in localBufferMap.", spillMemId);
            return FAILED;
        }
        tensor->memorymap[subGraphID] =
            TileRange(workspaceOffset, workspaceOffset + localBufferMap[spillMemId]->size, workspaceMemId++);
        workspaceOffset += localBufferMap[spillMemId]->size;
    } else {
        tensor->memorymap[subGraphID].memId = maxTensorMagic;
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
    op.UpdateSubgraphID(subGraphID);
    op.opmagic = ++maxOpMagic;
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
        for (auto memId : issue->reqMemIds) {
            if (memId == oldMemId) {
                std::replace(issue->reqMemIds.begin(), issue->reqMemIds.end(), oldMemId, newMemId);
            }
        }
        for (auto &outTensor : issue->tileOp.GetOOperands()) {
            if (outTensor->memorymap[subGraphID].memId == oldMemId) {
                outTensor->memorymap[subGraphID].memId = newMemId;
            }
        }
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

Status OoOScheduler::UpdateReloadIssueInfo(IssueEntryPtr reloadAlloc, IssueEntryPtr reloadCopyin,
    IssueEntryPtr spillIssue, int spillMemId, int bufNextUseTime) {
    reloadAlloc->reqMemIds = {maxTensorMagic};
    reloadAlloc->successors.insert(reloadCopyin->id);
    reloadCopyin->reqMemIds = {maxTensorMagic};
    reloadCopyin->predecessors.insert(reloadAlloc->id);

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
    if (UpdateRemainOpBufId(spillMemId, reloadAlloc->reqMemIds[0])) {
        ALOG_ERROR_F("UpdateRemainOpBufId failed.");
        return FAILED;
    }
    issueEntries.insert(issueEntries.begin() + bufNextUseTime, reloadCopyin);
    issueEntries.insert(issueEntries.begin() + bufNextUseTime, reloadAlloc);
    numTotalIssues += TWO_ISSUE;
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

Status OoOScheduler::CreateSpillReloadIssue(LogicalTensorPtr spillOutTensor,
    LogicalTensorPtr spillTensor, IssueEntryPtr &spillIssue, std::pair<IssueEntryPtr, IssueEntryPtr> &reloadIssues) {
    MemoryType memType = spillTensor->GetMemoryTypeOriginal();
    // 创建将spill搬出数据搬回OP_COPY_IN的tensor
    LogicalTensorPtr localTensor = std::make_shared<LogicalTensor>(function_, spillTensor->Datatype(), spillTensor->shape);
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

OoOSchedulerCheck::SpillInfo OoOScheduler::RecordSpillInfo(MemoryType bufferType, int memId, LocalBufferPtr allocBuffer, LogicalTensorPtr spillOutTensor, bool needCopyOut) {
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
        spillInfo.spillCopyoutSize = std::accumulate(spillOutTensor->shape.begin(), spillOutTensor->shape.end(), 1, std::multiplies<int64_t>()) * BytesOf(dtype);
    } else {
        spillInfo.spillCopyoutSize = 0;
    }
    return spillInfo;
}

Status OoOScheduler::GenBufferSpill(IssueEntryPtr allocIssue, MemoryType bufferType, std::vector<Operation *> &newOperations) {
    int spillMemId = -1;
    IssueEntryPtr spillIssue = nullptr;
    if (GetOldestBuffer(allocIssue, bufferType, spillMemId, spillIssue) != SUCCESS) {
        ALOG_ERROR_F("GetOldestBuffer failed.");
        return FAILED;
    }
    LogicalTensorPtr spillTensor = nullptr;
    if (GetSpillTensor(spillIssue, spillMemId, spillTensor) != SUCCESS) {
        ALOG_ERROR_F("%s GetSpillTensor failed!", spillIssue->GetOpInfo()); 
        return FAILED;
    }
    LogicalTensorPtr ddrTensor = nullptr;
    bool needCopyOut = false;
    if (spillIssue->tileOp.GetOpcode() != Opcode::OP_COPY_IN) { // 若spill的tensor不来自OP_COPY_IN，则将tensor搬出，在需要的时候再搬入
        needCopyOut = true;
        IssueEntryPtr spillCopyout = nullptr;
        if (CreateSpillCopyout(spillIssue, spillTensor, spillMemId, spillCopyout) != SUCCESS) {
            ALOG_ERROR_F("CreateSpillCopyout failed!");
            return FAILED;
        }
        issueEntries.emplace_back(spillCopyout);
        ddrTensor = spillCopyout->tileOp.GetOutputOperand(0);
        newOperations.push_back(&(spillCopyout->tileOp));
        ALOG_DEBUG_F("Insert: %s", spillCopyout->GetOpInfo());
    } else { // 若spill的tensor来自OP_COPY_IN，则数据无需搬出到DDR
        ddrTensor = spillIssue->tileOp.GetInputOperand(0);
    }
    // Healthcheck record - spill info
    if (oooCheck.doHealthCheck) {
        oooCheck.spillInfoVec.emplace_back(RecordSpillInfo(bufferType, spillMemId, localBufferMap[allocIssue->reqMemIds[0]], ddrTensor, needCopyOut));
    }
    IssueEntryPtr reloadCopyin = nullptr;
    IssueEntryPtr reloadAlloc = nullptr;
    std::pair<IssueEntryPtr, IssueEntryPtr> reloadIssues = {reloadAlloc, reloadCopyin};
    if (CreateSpillReloadIssue(ddrTensor, spillTensor, spillIssue, reloadIssues) != SUCCESS) {
        ALOG_ERROR_F("CreateSpillReloadIssue failed!"); 
        return FAILED;
    }
    reloadAlloc = reloadIssues.first;
    reloadCopyin = reloadIssues.second;
    if (UpdateReloadIssueInfo(reloadAlloc, reloadCopyin, spillIssue, spillMemId, issueEntries.size()) != SUCCESS) {
        ALOG_ERROR_F("UpdateReloadIssueInfo failed!");
        return FAILED;
    }
    if (bufferManagerMap[bufferType].Free(spillMemId) != SUCCESS) {
        ALOG_ERROR_F("Free spill tensor[%d] failed!", spillMemId);
        return FAILED;
    }
    // Healthcheck record - update buffer usage statistics
    if (oooCheck.doHealthCheck) { UpdateBufferUsage(bufferType, spillMemId, true); }
    localBufferMap[spillMemId]->retireCycle = clock;
    if (tensorOccupyMap[bufferType].erase(spillMemId) == 0) {
        ALOG_ERROR_F("Erase tensor[%d] failed", spillMemId);
        return FAILED;
    }
    allocIssueQueue[localBufferMap[spillMemId]->memType].InsertReloadAlloc(reloadAlloc, spillIssue, issueEntryMap);
    return SUCCESS;
}

Status OoOScheduler::SpillBuffer(int spillMemId, size_t &pcIdx, LocalBufferPtr allocBuffer) {
    size_t bufNextUseTime = pcIdx;
    size_t bufLastUseTime = pcIdx;
    size_t bufLastWriteTime = pcIdx;
    if (GetBufTimes(spillMemId, bufNextUseTime, bufLastUseTime, bufLastWriteTime) != SUCCESS) { 
        ALOG_ERROR_F("GetBufTimes failed!"); 
        return FAILED; 
    }
    auto spillIssue = issueEntries[bufLastWriteTime];
    LogicalTensorPtr spillTensor = nullptr;
    if (GetSpillTensor(spillIssue, spillMemId, spillTensor) != SUCCESS) {
        ALOG_ERROR_F("%s GetSpillTensor failed!", spillIssue->GetOpInfo());
        return FAILED;
    }
    ALOG_DEBUG_F("Begin spill %s tensor[%d].", spillIssue->GetOpInfo(), spillMemId);
    LogicalTensorPtr ddrTensor = nullptr;
    bool needCopyOut = false;
    if (spillIssue->tileOp.GetOpcodeStr().find("COPY_IN") == std::string::npos) {
        needCopyOut = true;
        IssueEntryPtr spillOutIssue = nullptr;
        if (CreateSpillCopyout(spillIssue, spillTensor, spillMemId, spillOutIssue) != SUCCESS) { 
            ALOG_ERROR_F("CreateSpillCopyout failed!"); 
            return FAILED; 
        }
        issueEntries.insert(issueEntries.begin() + bufLastUseTime + 1, spillOutIssue);
        bufNextUseTime++;
        ddrTensor = spillOutIssue->tileOp.GetOutputOperand(0);
        pcIdx++;
        numTotalIssues++;
    } else {
        ddrTensor = spillIssue->tileOp.GetInputOperand(0);
    }
    // Healthcheck record - spill info
    if (oooCheck.doHealthCheck) {
        oooCheck.spillInfoVec.emplace_back(RecordSpillInfo(allocBuffer->memType, spillMemId, allocBuffer, ddrTensor, needCopyOut));
    }
    IssueEntryPtr reloadCopyin = nullptr;
    IssueEntryPtr reloadAlloc = nullptr;
    std::pair<IssueEntryPtr, IssueEntryPtr> reloadIssues = {reloadAlloc, reloadCopyin};
    if (CreateSpillReloadIssue(ddrTensor, spillTensor, spillIssue, reloadIssues) != SUCCESS) { 
        ALOG_ERROR_F("CreateSpillReloadIssue failed!"); 
        return FAILED; 
    }
    reloadAlloc = reloadIssues.first;
    reloadCopyin = reloadIssues.second;
    if (UpdateReloadIssueInfo(reloadAlloc, reloadCopyin, spillIssue, spillMemId, bufNextUseTime) != SUCCESS) { 
        ALOG_ERROR_F("UpdateReloadIssueInfo failed!"); 
        return FAILED; 
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

Status OoOScheduler::SelectSpillBufferGroup(
    std::vector<std::vector<int>> &groups, int currPc, std::vector<int> &spillGroup) {
    if (groups.empty()) { ALOG_ERROR_F("Cannot find tensor to spill."); return FAILED; }
    std::unordered_map<int, size_t> nextUseTimeCache;
    std::vector<int> groupNextUseTime;
    for (auto& group : groups) {
        std::vector<size_t> bufNextUseTime;
        bool cannotSpill = false;
        for (auto& memId : group) {
            std::set<IssueEntryPtr> filterLtags;
            FindFilterLtags(issueEntries[currPc], filterLtags);
            size_t bufLastWriteTime = currPc;
            if (!GetBufLastWriteTime(memId, bufLastWriteTime)) {
                ALOG_ERROR_F("Cannot find spill Tensor[%d] last write time.", memId);
                return FAILED;
            }
            auto spillIssue = issueEntries[bufLastWriteTime];
            if (spillIssue->tileOp.GetOpcode() == Opcode::OP_VIEW ||
                spillIssue->tileOp.GetOpcode() == Opcode::OP_ASSEMBLE || filterLtags.count(spillIssue) != 0) {
                cannotSpill = true;
                break;
            }
            if (nextUseTimeCache.find(memId) != nextUseTimeCache.end()) {
                bufNextUseTime.push_back(nextUseTimeCache[memId]);
            } else {
                size_t nextUseTime = currPc;
                if (!GetBufNextUseTime(memId, nextUseTime)) {
                    ALOG_ERROR_F("Cannot find Tensor[%d] next used time.", memId);
                    return FAILED;
                }
                nextUseTimeCache[memId] = nextUseTime;
                bufNextUseTime.push_back(nextUseTime);
            }
        }
        if (cannotSpill) {
            groupNextUseTime.push_back(-1);
        } else {
            groupNextUseTime.push_back(*std::min_element(bufNextUseTime.begin(), bufNextUseTime.end()));
        }
    }
    size_t groupSel = std::max_element(groupNextUseTime.begin(), groupNextUseTime.end()) - groupNextUseTime.begin();
    if (groupNextUseTime[groupSel] == -1) {
        if (PrintSpillFailedInfo(currPc) != SUCCESS) {
            ALOG_ERROR_F("PrintSpillFailedInfo failed.");
            return FAILED;
        }
        ALOG_ERROR_F("Cannot find tensor to spill.");
        return FAILED;
    }
    spillGroup = groups[groupSel];
    return SUCCESS;
}

Status OoOScheduler::GenSpillOp(LocalBufferPtr allocBuffer, size_t &pcIdx) {
    if (bufferManagerMap[allocBuffer->memType].IsFull(allocBuffer)) {
        ALOG_DEBUG_F("---> START: SPILL tensor.");
        if (allocBuffer->memType != MemoryType::MEM_L1 && allocBuffer->memType != MemoryType::MEM_UB) {
            if (PrintSpillFailedInfo(pcIdx) != SUCCESS) {
                ALOG_ERROR_F("PrintSpillFailedInfo failed.");
                return FAILED;
            }
            ALOG_ERROR_F("Buffer[L0A/B/C] is Full. Please check tile shape and OOO spill failed info.");
            return FAILED;
        }
        // 查找出可以spill 单个或多个tensor的集合
        std::vector<std::vector<int>> canSpillGroups;
        if (bufferManagerMap[allocBuffer->memType].GetSpillGroup(allocBuffer->size, canSpillGroups) != SUCCESS) {
            ALOG_ERROR_F("GetSpillGroup failed.");
            return FAILED;  
        }
        // 选择最晚被使用的spill 单个或多个tensor
        std::vector<int> spillGroup;
        if (SelectSpillBufferGroup(canSpillGroups, pcIdx, spillGroup) != SUCCESS) { 
            ALOG_ERROR_F("SelectSpillBufferGroup failed!"); 
            return FAILED;
        }
        for (auto spillMemId : spillGroup) {
            // 插入spill搬出重载数据的COPY_OUT/COPY_IN
            if (SpillBuffer(spillMemId, pcIdx, allocBuffer) != SUCCESS) { 
                ALOG_ERROR_F("Tensor[%d] SpillBuffer failed!", spillMemId); 
                return FAILED; 
            }
            if (bufferManagerMap[allocBuffer->memType].Free(spillMemId) != SUCCESS) { 
                ALOG_ERROR_F("Free spill tensor[%d] failed!", spillMemId); 
                return FAILED; 
            }
        }
        ALOG_DEBUG_F("---> END: SPILL tensor.");
    }
    return SUCCESS;
}

Status OoOScheduler::GenSpillSchedule() {
    size_t pcIdx = 0;
    ALOG_DEBUG_F("=========> Begin GenSpillSchedule.");
    while (pcIdx < issueEntries.size()) {
        auto issue = issueEntries[pcIdx];
        ALOG_DEBUG_F("Launch %s", issue->GetOpInfo());
        if (issue->isAlloc) {
            if (localBufferMap.find(issue->reqMemIds[0]) == localBufferMap.end()) {
                ALOG_ERROR_F("Tensor[%d] cannot find in localBufferMap!", issue->reqMemIds[0]);
                return FAILED;
            }
            LocalBufferPtr allocBuffer = localBufferMap[issue->reqMemIds[0]];
            if (GenSpillOp(allocBuffer, pcIdx) != SUCCESS) { 
                ALOG_ERROR_F("GenSpillOp failed!"); 
                return FAILED; 
            }
            if (bufferManagerMap[allocBuffer->memType].Allocate(allocBuffer) != SUCCESS) { 
                ALOG_ERROR_F("Allocate tensor[%u] failed", allocBuffer->id); 
                return FAILED; 
            }
        }
        issue->isRetired = true;
        for (auto memId : issue->reqMemIds) {
            if (DelBufRefCount(memId) != SUCCESS) { 
                ALOG_ERROR_F("DelBufRefCount tensor[%d] failed", memId); 
                return FAILED; 
            }
            if (bufRefCount[memId] == 0) {
                if (bufferManagerMap[localBufferMap[memId]->memType].Free(localBufferMap[memId]->id) != SUCCESS) { 
                    ALOG_ERROR_F("Free tensor[%d] failed", memId); 
                    return FAILED; 
                }
            }
        }
        pcIdx += 1;
    }
    for (auto bufRef : bufRefCount) {
        if (bufRef.second != 0) { 
            ALOG_ERROR_F("Tensor[%d] bufRefCount not equal to 0!", bufRef.first); 
            return FAILED; 
        }
    }
    ALOG_DEBUG_F("=========> End GenSpillSchedule.");
    // 更新依赖关系
    if (InitDependencies() != SUCCESS) { 
        ALOG_ERROR_F("InitDependencies failed!"); 
        return FAILED; 
    }
    return SUCCESS;
}


} // namespace npu::tile_fwk