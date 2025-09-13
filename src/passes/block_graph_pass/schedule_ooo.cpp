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
 * \file schedule_ooo.cpp
 * \brief
 */

#include "schedule_ooo.h"
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "passes/pass_config/pass_config_manager.h"

namespace npu::tile_fwk {
namespace {
int getEntryPriority(const IssueEntryPtr& entry, const std::unordered_map<Opcode, int> &preNodePriority) {
    const int defaultVal = 10;
    auto it = preNodePriority.find(entry->tileOp.GetOpcode());
    if (it != preNodePriority.end()) {
        return it->second;
    }
    return defaultVal;
}

bool compareIssueEntries(const IssueEntryPtr &a, const IssueEntryPtr &b, 
                         const std::unordered_map<Opcode, int> &preNodePriority) {
    int priorA = getEntryPriority(a, preNodePriority);
    int priorB = getEntryPriority(b, preNodePriority);
    if (priorA != priorB) {
        return priorA < priorB;
    }
    return a->execOrder < b->execOrder;
}
}

constexpr int64_t MAX_L0A_SIZE = 64 * 1024;
constexpr int64_t MAX_L0C_SIZE = 128 * 1024;
constexpr int64_t MAX_BT_SIZE = 1 * 1024;
constexpr int64_t MAX_FIX_SIZE = 1 * 1024;
constexpr int32_t TWO_ISSUE = 2;
constexpr int32_t DEFAULT_LATENCY = 511;
constexpr int32_t DIM_FIVE = 5;
constexpr int32_t LAST_TWO_DIM = 2;
constexpr int32_t UB_BLOCK_SIZE = 32;

IssueEntry::IssueEntry(Operation &op, uint64_t issueId)
    : tileOp(op), id(issueId), execOrder(issueId), type(RescheduleUtils::GetOpPipeType(&op)) {
    if (tileOp.GetOpcodeStr().find("ALLOC") != std::string::npos) {
        isAlloc = true;
    }
}

int IssueEntry::GetOOperandIdx(int curMemId) {
    for (size_t i = 0; i < tileOp.GetOOperands().size(); i++) {
        if (tileOp.GetOOperands()[i]->memorymap[tileOp.GetSubgraphID()].memId == curMemId) {
            return i;
        }
    }
    return -1;
}

void IssueEntry::Clear() {
    isRetired = false;
    predecessors.clear();
    successors.clear();
    reqMemIds.clear();
}

void IssueEntry::UpdateTensorInput(std::shared_ptr<IssueEntry> &spillSrcIssue, LogicalTensorPtr tensor) const {
    for (size_t index = 0; index < tileOp.GetIOperands().size(); index++) {
        for (auto &inOp : tileOp.GetIOperands()[index]->GetProducers()) {
            if (inOp == &(spillSrcIssue->tileOp)) {
                tileOp.UpdateInputOperand(index, tensor);
            }
        }
    }
}

uint64_t OoOScheduler::ShapeCeilAlign(std::vector<int64_t> shape, DataType dtype) {
    if (shape.size() == DIM_FIVE) {
        uint64_t bytes = BytesPerElement(dtype) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        return CeilAlign(bytes, UB_BLOCK_SIZE);
    }
    uint64_t preDimSize = 1;
    uint64_t last2DimSize = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        if ((i < (shape.size() - LAST_TWO_DIM)) && (shape.size() != 1)) {
            preDimSize *= shape[i];
            continue;
        }
        last2DimSize *= shape[i];
    }
    return preDimSize * CeilAlign(last2DimSize * BytesPerElement(dtype), UB_BLOCK_SIZE);
}

Status OoOScheduler::DelBufRefCount(const int memId) {
    if (bufRefCount.find(memId) == bufRefCount.end()) {
        ALOG_ERROR_F("bufRefCount cannot find Tensor[%d].", memId);
        return FAILED;
    }
    bufRefCount[memId] -= 1;
    if (bufRefCount[memId] < 0) {
        ALOG_ERROR_F("Tensor[%d] bufRefCount cannot less than 0.", memId);
        return FAILED;
    }
    return SUCCESS;
};

Status OoOScheduler::UpdateOldestBuffer(IssueEntryPtr allocIssue, MemoryType bufferType, int& memId, IssueEntryPtr &spillIssue) {   
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
    return SUCCESS;
}

Status OoOScheduler::GetOldestBuffer(IssueEntryPtr allocIssue, MemoryType bufferType, int& memId,
    IssueEntryPtr &spillIssue) {
    if (UpdateOldestBuffer(allocIssue, bufferType, memId, spillIssue) != SUCCESS) {
        ALOG_ERROR_F("======== OoO Spill failed info ===========");
        return FAILED;
    }
    if (spillIssue == nullptr || memId == -1) {
        ALOG_ERROR_F("======== OoO Spill failed info ===========");
        ALOG_ERROR_F("Spill failed memoryType: %s.", MemoryTypeToString(bufferType).c_str());
        if (localBufferMap.find(allocIssue->reqMemIds[0]) != localBufferMap.end()) {
            ALOG_ERROR_F("%s[%d] alloc buffer size: %lu.", allocIssue->tileOp.GetOpcodeStr().c_str(), 
                allocIssue->tileOp.GetOpMagic(), localBufferMap[allocIssue->reqMemIds[0]]->size);
        }
        if (tensorOccupyMap.find(bufferType) != tensorOccupyMap.end()) {
            for (auto occupyIssue : tensorOccupyMap[bufferType]) {
                ALOG_ERROR_F("%s[%d], range[%lu, %lu], Tensor[%d] size: %lu", 
                    occupyIssue.second->tileOp.GetOpcodeStr().c_str(), occupyIssue.second->tileOp.GetOpMagic(),
                    localBufferMap[occupyIssue.first]->start, localBufferMap[occupyIssue.first]->end, occupyIssue.first,
                    localBufferMap[occupyIssue.first]->size);
            }
        }
        ALOG_ERROR_F("Could not find availalble buffer to spill!"); 
        return FAILED; 
    }
    ALOG_DEBUG_F("  Spill op: %s %d.", spillIssue->tileOp.GetOpcodeStr().c_str(), spillIssue->tileOp.GetOpMagic());
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
        return SUCCESS;
    }
    tensor->memorymap[subGraphID].memId = maxTensorMagic;
    localBufferMap[maxTensorMagic] = std::make_shared<LocalBuffer>(
        maxTensorMagic, ShapeCeilAlign(tensor->GetShape(), tensor->Datatype()), tensor->GetMemoryTypeOriginal());
    if (localBufferMap[maxTensorMagic] == nullptr) {
        ALOG_ERROR_F("Init Tensor[%d] localBuffer failed.", maxTensorMagic);
        return FAILED;
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
                ALOG_ERROR_F("Erase issueEntry %s, %d failed", spillIssue->tileOp.GetOpcodeStr().c_str(),
                    spillIssue->tileOp.GetOpMagic());
                return FAILED;
            }
            succ->predecessors.insert(reloadCopyin->id);
            if (reloadCopyin->tileOp.GetOutputOperand(0) == nullptr) {
                ALOG_ERROR_F("%s %d cannot find oOperand[0]", reloadCopyin->tileOp.GetOpcodeStr().c_str(),
                    reloadCopyin->tileOp.GetOpMagic());
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

Status OoOScheduler::CreateSpillCopyout(Function &func, IssueEntryPtr spillIssue, LogicalTensorPtr spillTensor,
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

    LogicalTensorPtr ddrTensor = std::make_shared<LogicalTensor>(func, ddrRawTensor, offset, spillTensor->GetShape());
    if (ddrTensor == nullptr) {
        ALOG_ERROR_F("Create DDR tensor failed!");
        return FAILED;
    }
    if (UpdateTensorAttr(ddrTensor, MEM_DEVICE_DDR, spillTensor, spillMemId) != SUCCESS) {
        ALOG_ERROR_F("UpdateTensorAttr DDR tensor failed!");
        return FAILED;
    }

    // 创建spill搬出所需的DDR OP_COPY_OUT
    Operation &spillOutOp = func.AddRawOperation(Opcode::OP_COPY_OUT, {spillTensor}, {ddrTensor});
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
    ALOG_DEBUG_F("Add SPILL_OUT: %s[%d].", spillCopyout->tileOp.GetOpcodeStr().c_str(), spillCopyout->tileOp.GetOpMagic());
    return SUCCESS;
}

Status OoOScheduler::CreateSpillReloadIssue(Function &func, LogicalTensorPtr spillOutTensor,
    LogicalTensorPtr spillTensor, IssueEntryPtr &spillIssue, std::pair<IssueEntryPtr, IssueEntryPtr> &reloadIssues) {
    MemoryType memType = spillTensor->GetMemoryTypeOriginal();
    // 创建将spill搬出数据搬回OP_COPY_IN的tensor
    LogicalTensorPtr localTensor = std::make_shared<LogicalTensor>(func, spillTensor->Datatype(), spillTensor->shape);
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
    auto &spillAllocOp = func.AddRawOperation(allocOp, {}, {localTensor});
    auto &spillCopyInOp = func.AddRawOperation(Opcode::OP_COPY_IN, {spillOutTensor}, {localTensor});

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
    ALOG_DEBUG_F("Add SPILL_ALLOC: %s[%d].", spillAllocInst->tileOp.GetOpcodeStr().c_str(), spillAllocInst->tileOp.GetOpMagic());
    ALOG_DEBUG_F("Add SPILL_IN: %s[%d].", spillInInst->tileOp.GetOpcodeStr().c_str(), spillInInst->tileOp.GetOpMagic());
    return SUCCESS;
}

void OoOScheduler::UpdateBufferUsage(MemoryType bufferType, int memId, bool isFree) {
    if (isFree) {
        int freeBufferSize = localBufferMap[memId]->size;
        oooCheck.bufferTotalUsage[bufferType] += oooCheck.bufferLastUsage[bufferType] * (clock - oooCheck.lastClock[bufferType]);
        oooCheck.bufferLastUsage[bufferType] -= freeBufferSize;
        oooCheck.lastClock[bufferType] = clock;
        return;
    }
    oooCheck.bufferTotalUsage[bufferType] += oooCheck.bufferLastUsage[bufferType] * (clock - oooCheck.lastClock[bufferType]);
    oooCheck.bufferLastUsage[bufferType] += localBufferMap[memId]->size;
    oooCheck.lastClock[bufferType] = clock;
    oooCheck.bufferMaxUsage[bufferType] = std::max(oooCheck.bufferMaxUsage[bufferType], oooCheck.bufferLastUsage[bufferType]);
}

OoOSchedulerCheck::SpillInfo OoOScheduler::RecordSpillInfo(
    MemoryType bufferType, int memId, LocalBufferPtr allocBuffer, LogicalTensorPtr spillOutTensor, bool needCopyOut) {
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
        spillInfo.spillCopyoutSize =
            std::accumulate(spillOutTensor->shape.begin(), spillOutTensor->shape.end(), 1, std::multiplies<int64_t>()) *
            BytesOf(dtype);
        return spillInfo;
    }
    spillInfo.spillCopyoutSize = 0;
    return spillInfo;
}

Status OoOScheduler::GetddrTensor(Function &function, const SpillInfo &spillInfo, LogicalTensorPtr &ddrTensor, std::vector<Operation *> &newOperations) {
    int spillMemId = spillInfo.spillMemId;
    MemoryType bufferType = spillInfo.bufferType;
    IssueEntryPtr spillIssue = spillInfo.spillIssue;
    IssueEntryPtr allocIssue = spillInfo.allocIssue;
    LogicalTensorPtr spillTensor = spillInfo.spillTensor;
    bool needCopyOut = false;
    if (spillIssue->tileOp.GetOpcode() != Opcode::OP_COPY_IN) { // 若spill的tensor不来自OP_COPY_IN，则将tensor搬出，在需要的时候再搬入
        needCopyOut = true;
        IssueEntryPtr spillCopyout = nullptr;
        if (CreateSpillCopyout(function, spillIssue, spillTensor, spillMemId, spillCopyout) != SUCCESS) {
            ALOG_ERROR_F("CreateSpillCopyout failed!");
            return FAILED;
        }
        issueEntries.emplace_back(spillCopyout);
        ddrTensor = spillCopyout->tileOp.GetOutputOperand(0);
        newOperations.push_back(&(spillCopyout->tileOp));
        ALOG_DEBUG_F("Insert: %s[%d]", spillCopyout->tileOp.GetOpcodeStr().c_str(), spillCopyout->tileOp.GetOpMagic());
    } else { // 若spill的tensor来自OP_COPY_IN，则数据无需搬出到DDR
        ddrTensor = spillIssue->tileOp.GetInputOperand(0);
    }
    // Healthcheck record - spill info
    if (oooCheck.doHealthCheck) {
        oooCheck.spillInfoVec.emplace_back(RecordSpillInfo(bufferType, spillMemId, localBufferMap[allocIssue->reqMemIds[0]], ddrTensor, needCopyOut));
    }
    return SUCCESS;
}

Status OoOScheduler::GenBufferSpill(
    Function &function, IssueEntryPtr allocIssue, MemoryType bufferType, std::vector<Operation *> &newOperations) {
    int spillMemId = -1;
    IssueEntryPtr spillIssue = nullptr;
    if (GetOldestBuffer(allocIssue, bufferType, spillMemId, spillIssue) != SUCCESS) {
        ALOG_ERROR_F("GetOldestBuffer failed.");
        return FAILED;
    }
    LogicalTensorPtr spillTensor = nullptr;
    if (GetSpillTensor(spillIssue, spillMemId, spillTensor) != SUCCESS) {
        ALOG_ERROR_F("%d %s GetSpillTensor failed!", spillIssue->tileOp.GetOpMagic(), spillIssue->tileOp.GetOpcodeStr().c_str());
        return FAILED;
    }
    LogicalTensorPtr ddrTensor = nullptr;
    SpillInfo spillInfo = {spillMemId, bufferType, spillIssue, allocIssue, spillTensor};
    if (GetddrTensor(function, spillInfo, ddrTensor, newOperations) != SUCCESS) {
        ALOG_ERROR_F("%s GetddrTensor failed!", spillIssue->tileOp.GetOpMagic());
        return FAILED;
    }
    IssueEntryPtr reloadCopyin = nullptr;
    IssueEntryPtr reloadAlloc = nullptr;
    std::pair<IssueEntryPtr, IssueEntryPtr> reloadIssues = {reloadAlloc, reloadCopyin};
    if (CreateSpillReloadIssue(function, ddrTensor, spillTensor, spillIssue, reloadIssues) != SUCCESS) {
        ALOG_ERROR_F("CreateSpillReloadIssue failed!"); return FAILED;
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
    if (oooCheck.doHealthCheck) {
        UpdateBufferUsage(bufferType, spillMemId, true);
    }
    localBufferMap[spillMemId]->retireCycle = clock;
    if (tensorOccupyMap[bufferType].erase(spillMemId) == 0) {
        ALOG_ERROR_F("Erase tensor[%d] failed", spillMemId);
        return FAILED;
    }
    allocIssueQueue[localBufferMap[spillMemId]->memType].InsertReloadAlloc(reloadAlloc, spillIssue, issueEntryMap);
    return SUCCESS;
}

void OoOScheduler::PrintDependenciesAndRelations() {
    for (const auto &issue : issueEntries) {
        if (issue->tileOp.GetBoolAttribute(OpAttributeKey::dontTouch)) {
            continue;
        }
        ALOG_DEBUG_F("%d %s, latency: %d.", issue->tileOp.GetOpMagic(), issue->tileOp.GetOpcodeStr().c_str(),
            issue->tileOp.GetLatency());
        for (const auto &preId : issue->predecessors) {
            auto pre = issueEntryMap[preId];
            ALOG_DEBUG_F("    |--- Predecessors:");
            ALOG_DEBUG_F("        |--- %s[%d]", pre->tileOp.GetOpcodeStr().c_str(), pre->tileOp.GetOpMagic());
        }
        for (const auto &successorId : issue->successors) {
            auto successor = issueEntryMap[successorId];
            ALOG_DEBUG_F("    |--- Successors:");
            ALOG_DEBUG_F("        |--- %s[%d]", successor->tileOp.GetOpcodeStr().c_str(), successor->tileOp.GetOpMagic());
        }
        ALOG_DEBUG_F("\n");
    }
}

Status OoOScheduler::CheckOpBufferSize(Operation *op) {
    std::map<MemoryType, int64_t> bufferSize;
    std::set<int> memIdMap;
    for (auto &inTensor : op->GetIOperands()) {
        if (memIdMap.find(inTensor->memorymap[subGraphID].memId) == memIdMap.end()) {
            bufferSize[inTensor->GetMemoryTypeOriginal()] += inTensor->GetDataSize();
            memIdMap.insert(inTensor->memorymap[subGraphID].memId);
        }
    }
    for (auto &outTensor : op->GetOOperands()) {
        if (memIdMap.find(outTensor->memorymap[subGraphID].memId) == memIdMap.end()) {
            bufferSize[outTensor->GetMemoryTypeOriginal()] += outTensor->GetDataSize();
            memIdMap.insert(outTensor->memorymap[subGraphID].memId);
        }
    }
    for (auto &buffer : bufferSize) {
        if (inChipMemorySize.find(buffer.first) != inChipMemorySize.end()) {
            if (buffer.second > inChipMemorySize[buffer.first]) {
                ALOG_ERROR_F("OP %s[%d] in/output total size[%d] exceeds %s size[%d]!", op->GetOpcodeStr().c_str(),
                    op->GetOpMagic(), buffer.second, MemoryTypeToString(buffer.first).c_str(),
                    inChipMemorySize[buffer.first]);
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

void OoOScheduler::AddDependencies(
    IssueEntryPtr issue, std::map<int, IssueEntryPtr> lastWriteOpMap, LogicalTensorPtr tensor) {
    int memId = tensor->memorymap[subGraphID].memId;
    if (tensor->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
        bufRefCount[memId]++;
        issue->reqMemIds.push_back(memId);
    }
    if (lastWriteOpMap.find(memId) != lastWriteOpMap.end()) {
        issue->predecessors.insert(lastWriteOpMap[memId]->id);
        lastWriteOpMap[memId]->successors.insert(issue->id);
    }
}

Status OoOScheduler::InitLocalBuffer(LogicalTensorPtr oOperand, int memId) {
    if (oOperand->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
        return SUCCESS;
    }
    if (localBufferMap.find(memId) == localBufferMap.end()) {
        localBufferMap[memId] = std::make_shared<LocalBuffer>(
            memId, ShapeCeilAlign(oOperand->GetShape(), oOperand->Datatype()), oOperand->GetMemoryTypeOriginal());
        if (localBufferMap[memId] == nullptr) {
            ALOG_ERROR_F("Init tensor[%d] localBuffer failed!", memId);
            return FAILED;
        }
        return SUCCESS;
    }
    localBufferMap[memId]->size =
        std::max(localBufferMap[memId]->size, ShapeCeilAlign(oOperand->GetShape(), oOperand->Datatype()));
    return SUCCESS;
}

Status OoOScheduler::InitDependencies() {
    bufRefCount.clear();
    std::map<int, IssueEntryPtr> lastWriteOpMap;
    for (const auto &issue : issueEntries) {
        issue->Clear();
        // 仅检测 RAW 和 WAW，不检测 WAR -->SSA
        // RAW
        for (auto &inTensor : issue->tileOp.GetIOperands()) {
            AddDependencies(issue, lastWriteOpMap, inTensor);
        }

        // WAW
        for (auto &outTensor : issue->tileOp.GetOOperands()) {
            AddDependencies(issue, lastWriteOpMap, outTensor);
        }

        for (auto &oOperand : issue->tileOp.GetOOperands()) {
            int memId = oOperand->memorymap[subGraphID].memId;
            maxTensorMagic = std::max(maxTensorMagic, memId);
            lastWriteOpMap[memId] = issue;
            if (InitLocalBuffer(oOperand, memId) != SUCCESS) {
                ALOG_ERROR_F("InInitLocalBuffer failed.");
                return FAILED;
            }
        }
    }
    PrintDependenciesAndRelations();
    return SUCCESS;
}

Status OoOScheduler::UpdateIssueIOTensor(const IssueEntryPtr &issue, std::map<int, IssueEntryPtr> &tensorAllocMap) {
    for (auto outTensor : issue->tileOp.GetOOperands()) {
        if (outTensor->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        int memId = outTensor->memorymap[subGraphID].memId;
        if (tensorAllocMap.find(memId) == tensorAllocMap.end()) {
            tensorAllocMap[memId] = issue;
        }
    }
    for (auto inTensor : issue->tileOp.GetIOperands()) {
        if (inTensor->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        int memId = inTensor->memorymap[subGraphID].memId;
        if (tensorAllocMap.find(memId) == tensorAllocMap.end()) {
            tensorAllocMap[memId] = issue;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::CheckAllocIssue() {
    std::map<int, IssueEntryPtr> tensorAllocMap;
    for (const auto &issue : issueEntries) {
        if (issue->isAlloc) {
            if (issue->reqMemIds.size() != 1) {
                ALOG_ERROR_F("ALLOC[%d] reqMemIds size not equal to 0.", issue->tileOp.GetOpMagic());
                return FAILED;
            }
        }
        if (UpdateIssueIOTensor(issue, tensorAllocMap) != SUCCESS) {
            ALOG_ERROR_F("UpdateIssueIOTensor failed.");
            return FAILED;
        }
    }
    for (auto tensorAlloc : tensorAllocMap) {
        if (!tensorAlloc.second->isAlloc) {
            ALOG_ERROR_F("%s[%d] Tensor[%d] is missing Alloc.", tensorAlloc.second->tileOp.GetOpcodeStr().c_str(),
                tensorAlloc.second->tileOp.GetOpMagic(), tensorAlloc.first);
            return FAILED;
        }
    }
    return SUCCESS;
}

void OoOScheduler::InitIssueQueuesAndBufferManager() {
    for (size_t i = 0; i <= static_cast<int>(PipeType::PIPE_FIX); i++) {
        issueQueues[static_cast<PipeType>(i)] = IssueQueue();
    }

    bufferManagerMap.clear();
    for (size_t i = 0; i < static_cast<int>(MemoryType::MEM_DEVICE_DDR); i++) {
        allocIssueQueue[static_cast<MemoryType>(i)] = IssueQueue();
        if (inChipMemorySize.find(static_cast<MemoryType>(i)) != inChipMemorySize.end()) {
            bufferManagerMap.insert({static_cast<MemoryType>(i),
                BufferPool(static_cast<MemoryType>(i), inChipMemorySize[static_cast<MemoryType>(i)])});
        }
    }
}

Status OoOScheduler::Init(const std::vector<Operation *> &operations) {
    issueEntries.clear();
    localBufferMap.clear();

    // 初始化芯片各buffer大小
    inChipMemorySize = {
        {MemoryType::MEM_L0A, MAX_L0A_SIZE},
        {MemoryType::MEM_L0C, MAX_L0C_SIZE},
        {MemoryType::MEM_BT, MAX_BT_SIZE},
        {MemoryType::MEM_FIX, MAX_FIX_SIZE},
    };
    inChipMemorySize.insert({MemoryType::MEM_UB,  PassConfigManager::Instance().GetPlatformConfig().GetMemoryLimit(MemoryType::MEM_UB)});
    inChipMemorySize.insert({MemoryType::MEM_L1,  PassConfigManager::Instance().GetPlatformConfig().GetMemoryLimit(MemoryType::MEM_L1)});
    inChipMemorySize.insert({MemoryType::MEM_L0B,  PassConfigManager::Instance().GetPlatformConfig().GetMemoryLimit(MemoryType::MEM_L0B)});

    std::vector<Operation *> newOperations;
    for (auto& op : operations) {
        if (op->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            newOperations.insert(newOperations.begin(), op);
            continue;
        }
        newOperations.push_back(op);
    }

    // 校验并初始化issueEntry
    for (const auto &op : newOperations) {
        maxOpMagic = std::max(maxOpMagic, op->GetOpMagic());
        if (CheckOpBufferSize(op) != SUCCESS) {
            ALOG_ERROR_F("%s[%d] CheckOpBufferSize failed!", op->GetOpcodeStr().c_str(), op->GetOpMagic());
            return FAILED;
        }
        auto issue = std::make_shared<IssueEntry>(*op, issueId);
        issueEntryMap[issueId] = issue;
        issueId++;
        if (issue == nullptr) {
            ALOG_ERROR_F("IssueEntry %s, %d init failed!", op->GetOpcodeStr().c_str(), op->GetOpMagic());
            return FAILED;
        }
        issueEntries.emplace_back(issue);
    }
    numTotalIssues = issueEntries.size();

    // 初始化issueEntry，构建依赖关系
    if (InitDependencies() != SUCCESS) {
        ALOG_ERROR_F("InitDependencies failed!");
        return FAILED;
    }

    if (CheckAllocIssue() != SUCCESS) {
        ALOG_ERROR_F("CheckAllocIssue failed!");
        return FAILED;
    }

    // 初始化内存管理器
    InitIssueQueuesAndBufferManager();
    return SUCCESS;
}

int OoOScheduler::GetNumUnvisitPreNode(IssueEntryPtr issue, std::map<IssueEntryPtr, bool>& visited) {
    std::unordered_set<IssueEntryPtr> preNodeTotal;
    std::unordered_set<IssueEntryPtr> curr;
    for (auto& preIssueId : issue->predecessors) {
        auto preIssue = issueEntryMap[preIssueId];
        if (!visited[preIssue]) {
            curr.insert(preIssue);
            preNodeTotal.insert(preIssue);
        }
    }
    while (!curr.empty()) {
        std::unordered_set<IssueEntryPtr> next;
        for (auto& curIssue : curr) {
            for (auto& preIssueId : curIssue->predecessors) {
                auto preIssue = issueEntryMap[preIssueId];
                if (!visited[preIssue] && preNodeTotal.find(preIssue) == preNodeTotal.end()) {
                    next.insert(preIssue);
                }
            }
        }
        for (auto& nextIssue : next) {
            preNodeTotal.insert(nextIssue);
        }
        curr.swap(next);
    }
    return preNodeTotal.size();
}

IssueEntryPtr OoOScheduler::FindNodeMinNumUnvisitedPreNode(
    std::map<IssueEntryPtr, bool> visited, std::vector<IssueEntryPtr> outNodeQueue) {
    IssueEntryPtr res = nullptr;
    int minUnvisitedNode = INT_MAX;
    for (auto& outNode : outNodeQueue) {
        if (visited[outNode]) {
            continue;
        }
        int curUnvisitedNode = GetNumUnvisitPreNode(outNode, visited);
        if (curUnvisitedNode < minUnvisitedNode) {
            res = outNode;
            minUnvisitedNode = curUnvisitedNode;
        }
    }
    return res;
}

void OoOScheduler::UpdateIssueEntriesWithUnvisitedPredecessors(const IssueEntryPtr &curIssue, std::map<IssueEntryPtr, bool>& visited, std::unordered_map<Opcode, int> preNodePriority, std::deque<IssueEntryPtr> &queue) {
    std::vector<IssueEntryPtr> notReadyPreNode;
    for (auto& preIssueId : curIssue->predecessors) {
        auto preIssue = issueEntryMap[preIssueId];
        if (!visited[preIssue]) {
            notReadyPreNode.push_back(preIssue);
        }
    }
    std::sort(notReadyPreNode.begin(), notReadyPreNode.end(), [&](IssueEntryPtr a, IssueEntryPtr b) {
        return compareIssueEntries(a, b, preNodePriority);
    });
    for (auto& preIssue : notReadyPreNode) {
        queue.push_front(preIssue);
    }
}

bool OoOScheduler::CheckIsReady(const IssueEntryPtr &curIssue, std::map<IssueEntryPtr, bool>& visited) {
    for (auto& preIssueId : curIssue->predecessors) {
        auto preIssue = issueEntryMap[preIssueId];
        if (!visited[preIssue]) {
            return false;
        }
    }
    return true;
}

void OoOScheduler::DFSFromSingleNode(IssueEntryPtr issue, std::map<IssueEntryPtr, bool>& visited,
    std::vector<IssueEntryPtr>& newIssueEntries, std::unordered_map<Opcode, int> preNodePriority) {
    if (visited[issue]) {
        return;
    }
    std::deque<IssueEntryPtr> queue = {issue};
    while (!queue.empty()) {
        auto curIssue = queue.front();
        if (visited[curIssue]) {
            queue.pop_front();
            continue;
        }
        if (CheckIsReady(curIssue, visited)) {
            visited[curIssue] = true;
            queue.pop_front();
            newIssueEntries.push_back(curIssue);
            continue;
        }
        UpdateIssueEntriesWithUnvisitedPredecessors(curIssue, visited, preNodePriority, queue);
    }
}

Status EntryInOutGraph(std::vector<IssueEntryPtr> &issueEntries, std::vector<std::set<int>> &entryInGraph, 
    std::vector<std::set<int>> &entryOutGraph, std::unordered_map<int, IssueEntryPtr> issueEntryMap)
{
    entryInGraph.clear();
    entryOutGraph.clear();
    entryInGraph.resize(issueEntries.size());
    entryOutGraph.resize(issueEntries.size());
    std::unordered_map<IssueEntry*, int> entryPtr2Idx;
    for (int ptrIdx = 0; ptrIdx < static_cast<int>(issueEntries.size()); ptrIdx++) {
        entryPtr2Idx[issueEntries[ptrIdx].get()] = ptrIdx;
    }
    for (int ptrIdx = 0; ptrIdx < static_cast<int>(issueEntries.size()); ptrIdx++) {
        for (auto outPtrId : issueEntries[ptrIdx]->successors) {
            auto outPtr = issueEntryMap[outPtrId];
            entryOutGraph[ptrIdx].insert(entryPtr2Idx[outPtr.get()]);
            entryInGraph[entryPtr2Idx[outPtr.get()]].insert(ptrIdx);
        }
    }
    return SUCCESS;
}

Status EntryTopoSort(std::vector<std::set<int32_t>> &entryInGraph, std::vector<std::set<int32_t>> &entryOutGraph,
    std::vector<int32_t> &seqToColor, std::vector<int32_t> &colorToSeq)
{
    seqToColor.clear();
    colorToSeq.resize(entryInGraph.size());
    std::vector<int> inLinkNum(entryInGraph.size());
    std::deque<int> zeroInLinkColor;
    for (size_t i = 0; i < entryInGraph.size(); i++) {
        inLinkNum[i] = entryInGraph[i].size();
        if (inLinkNum[i] == 0) {
            zeroInLinkColor.push_back(i);
        }
    }
    std::vector<int32_t> visitOrder;
    while (zeroInLinkColor.size() > 0) {
        int currColor = zeroInLinkColor.front();
        zeroInLinkColor.pop_front();
        colorToSeq[currColor] = seqToColor.size();
        seqToColor.push_back(currColor);
        for (int consumerColor : entryOutGraph[currColor]) {
            inLinkNum[consumerColor] -= 1;
            if (inLinkNum[consumerColor] == 0) {
                zeroInLinkColor.push_back(consumerColor);
            }
        }
    }
    return SUCCESS;
}

Status OutputFixBasedDepth(std::vector<int32_t> &depth, std::vector<std::set<int32_t>> &entryInGraph,
        std::vector<std::set<int32_t>> &entryOutGraph, std::vector<int32_t> &seqToColor)
{
    for (int idx = static_cast<int>(entryInGraph.size())-1; idx >= 0; idx--) {
        int currEntryIdx = seqToColor[idx];
        if (entryOutGraph[currEntryIdx].size() == 0) {
            continue;
        }
        int minDepth = static_cast<int>(entryInGraph.size()) + 1;
        for (auto succIdx : entryOutGraph[currEntryIdx]) {
            minDepth = minDepth < depth[succIdx] ? minDepth : depth[succIdx];
        }
        depth[currEntryIdx] = minDepth - 1;
    }
    return SUCCESS;
}

Status InputFixBasedDepth(std::vector<int32_t> &depth, std::vector<std::set<int32_t>> &entryInGraph,
        std::vector<std::set<int32_t>> &entryOutGraph, std::vector<int32_t> &seqToColor)
{
    (void)entryOutGraph;
    for (int idx = 0; idx < static_cast<int>(entryInGraph.size()); idx++) {
        int currEntryIdx = seqToColor[idx];
        if (entryInGraph[currEntryIdx].size() == 0) {
            continue;
        }
        int maxDepth = -static_cast<int>(entryInGraph.size()) - 1;
        for (auto predIdx : entryInGraph[currEntryIdx]) {
            maxDepth = maxDepth > depth[predIdx] ? maxDepth : depth[predIdx];
        }
        depth[currEntryIdx] = maxDepth + 1;
    }
    return SUCCESS;
}

Status DFSVisit(std::unordered_set<int> &visited, std::vector<int32_t> &tasks, std::vector<int> &visitEntrySeq,
        std::vector<std::set<int32_t>> &entryInGraph)
{
    while (tasks.size() > 0) {
        int currTask = tasks.back();
        if (visited.count(currTask) > 0) {
            tasks.pop_back();
            continue;
        }
        bool allVisited = true;
        for (auto prevIdx : entryInGraph[currTask]) {
            if (visited.count(prevIdx) == 0) {
                allVisited = false;
                tasks.push_back(prevIdx);
            }
        }
        if (allVisited) {
            visited.insert(currTask);
            visitEntrySeq.push_back(currTask);
            tasks.pop_back();
        }
    }
    return SUCCESS;
}

std::vector<int32_t> GetLayerTasks(std::map<int, std::set<int>> &depthToEntries,
                                   std::vector<std::set<int>> &entryOutGraph, int lastDepth, int currDepth)
{
    std::vector<int32_t> tasks;
    for (int dp = lastDepth; dp < currDepth; dp++) {
        for (int entryIdx : depthToEntries[dp]) {
            if (entryOutGraph[dp].size() == 0) {
                tasks.push_back(entryIdx);
            }
        }
    }
    for (int entryIdx : depthToEntries[currDepth]) {
        tasks.push_back(entryIdx);
    }
    return tasks;
}

Status OoOScheduler::LayerBasedDFS(int layerDepth)
{
    std::vector<std::set<int>> entryInGraph;
    std::vector<std::set<int>> entryOutGraph;
    EntryInOutGraph(issueEntries, entryInGraph, entryOutGraph, issueEntryMap);
    std::vector<int32_t> seqToColor;
    std::vector<int32_t> colorToSeq;
    EntryTopoSort(entryInGraph, entryOutGraph, seqToColor, colorToSeq);
    std::vector<int32_t> depth(entryInGraph.size(), 0);
    OutputFixBasedDepth(depth, entryInGraph, entryOutGraph, seqToColor);
    InputFixBasedDepth(depth, entryInGraph, entryOutGraph, seqToColor);
    std::map<int, std::set<int>> depthToEntries;
    int lowerDepth = static_cast<int>(entryInGraph.size()) + 1;
    int upperDepth = -static_cast<int>(entryInGraph.size()) - 1;
    for (int idx = 0; idx < static_cast<int>(depth.size()); idx++) {
        lowerDepth = lowerDepth < depth[idx] ? lowerDepth : depth[idx];
        upperDepth = upperDepth > depth[idx] ? upperDepth : depth[idx];
        depthToEntries[depth[idx]].insert(idx);
    }
    std::vector<IssueEntryPtr> newIssueEntries;
    std::unordered_set<int> visited;
    std::vector<int> visitEntrySeq;
    int lastDepth = lowerDepth;
    int currDepth = lowerDepth + layerDepth - 1;
    currDepth = currDepth <= upperDepth ? currDepth : upperDepth;
    bool keepVisit = true;
    while (keepVisit) {
        std::vector<int32_t> tasks = GetLayerTasks(depthToEntries, entryOutGraph, lastDepth, currDepth);
        DFSVisit(visited, tasks, visitEntrySeq, entryInGraph);
        if (currDepth == upperDepth) {
            keepVisit = false;
        }
        lastDepth = currDepth;
        currDepth += layerDepth;
        currDepth = currDepth <= upperDepth ? currDepth : upperDepth;
    }
    for (auto idx : visitEntrySeq) {
        newIssueEntries.push_back(issueEntries[idx]);
    }
    issueEntries = newIssueEntries;
    return SUCCESS;
}

Status OoOScheduler::PriorDFS(std::unordered_map<Opcode, int> preNodePriority) {
    std::vector<IssueEntryPtr> newIssueEntries;
    std::map<IssueEntryPtr, bool> visited;
    std::vector<IssueEntryPtr> outNodeQueue;
    for (auto &issue : issueEntries) {
        visited[issue] = false;
        if (issue->successors.empty()) {
            outNodeQueue.push_back(issue);
        }
    }

    if (outNodeQueue.size() == 0) {
        ALOG_ERROR_F("Subgraph must have operation with outdegree 0.");
        return FAILED;
    }
    DFSFromSingleNode(outNodeQueue[0], visited, newIssueEntries, preNodePriority);

    for (size_t i = 1; i < outNodeQueue.size(); i++) {
        while (!visited[outNodeQueue[i]]) {
            auto curNode = outNodeQueue[i];
            auto node = FindNodeMinNumUnvisitedPreNode(visited, outNodeQueue);
            if (node == nullptr) {
                ALOG_ERROR_F("FindNodeMinNumUnvisitedPreNode failed.");
                return FAILED;
            }
            DFSFromSingleNode(node, visited, newIssueEntries, preNodePriority);
        }
    }
    issueEntries = newIssueEntries;
    return SUCCESS;
}

Status OoOScheduler::SortOps(SortOpMethod sortMethod) {
    if (sortMethod == SortOpMethod::PriorDFS) {
        std::unordered_map<Opcode, int> preNodePriority = {
            // ALLOC 节点优先级最高，因为一个节点的前序ALLOC节点要在最靠近该节点的地方访问。
            {Opcode::OP_UB_ALLOC, 0},
            {Opcode::OP_L1_ALLOC, 0},
            {Opcode::OP_L0A_ALLOC, 0},
            {Opcode::OP_L0B_ALLOC, 0},
            {Opcode::OP_L0C_ALLOC, 0},
            {Opcode::OP_BT_ALLOC, 0},
            {Opcode::OP_FIX_ALLOC, 0},
            // 其次是L0级数据搬运Op。
            {Opcode::OP_L1_TO_L0A, 1},
            {Opcode::OP_L1_TO_L0B, 1},
            {Opcode::OP_L1_TO_L0_AT, 1},
            {Opcode::OP_L1_TO_L0_BT, 1},
            {Opcode::OP_FIX_COPY_IN, 1},
            {Opcode::OP_FIX_COPY_IN_QUANT_PRE, 1},
            {Opcode::OP_FIX_COPY_IN_RELU_PRE, 1},
            {Opcode::OP_FIX_COPY_IN_RELU_POST, 1},
            {Opcode::OP_FIX_COPY_IN_QUANT_POST, 1},
            {Opcode::OP_FIX_COPY_IN_ELT_ANTIQ, 1},
            {Opcode::OP_FIX_COPY_IN_MTE2_ANTIQ, 1},
            {Opcode::OP_BT_COPY_IN, 1},
            // 再其次是L1级数据搬运Op。
            {Opcode::OP_COPY_IN, 2},
            {Opcode::OP_UB_COPY_IN, 2},
            {Opcode::OP_L1_COPY_IN, 2},
            {Opcode::OP_L1_COPY_IN_FRACTAL_Z, 2},
            {Opcode::OP_L1_COPY_UB, 2},
            {Opcode::OP_L0C_COPY_UB, 2},
            {Opcode::OP_UB_COPY_L1, 2},
            // 最后访问其它计算节点（其它节点默认的优先级为10）。
        };
        if (PriorDFS(preNodePriority) != SUCCESS) {
            ALOG_ERROR_F("PriorDFS failed.");
            return FAILED;
        }
        return SUCCESS;
    }
    if (sortMethod != SortOpMethod::LayerBasedDFS) {
         ALOG_ERROR_F("Op sort method not recognized.");
         return FAILED;
    }
    const int layerDepth = 10;
    if (LayerBasedDFS(layerDepth) != SUCCESS) {
        ALOG_ERROR_F("LayerBasedDFS failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::GetBufTimes(
    int spillMemId, size_t &bufNextUseTime, size_t &bufLastUseTime, size_t &bufLastWriteTime) {
    if (!GetBufNextUseTime(spillMemId, bufNextUseTime)) {
        ALOG_ERROR_F("Cannot find spill Tensor[%d] next used time.", spillMemId);
        return FAILED;
    }
    if (!GetBufLastUseTime(spillMemId, bufLastUseTime)) {
        ALOG_ERROR_F("Cannot find spill Tensor[%d] last used time.", spillMemId);
        return FAILED;
    }
    if (!GetBufLastWriteTime(spillMemId, bufLastWriteTime)) {
        ALOG_ERROR_F("Cannot find spill Tensor[%d] last write time.", spillMemId);
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::SpillBuffer(Function &function, int spillMemId, size_t &pcIdx, LocalBufferPtr allocBuffer) {
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
        ALOG_ERROR_F("%d %s GetSpillTensor failed!", spillIssue->tileOp.GetOpMagic(), spillIssue->tileOp.GetOpcodeStr().c_str());
        return FAILED;
    }
    ALOG_DEBUG_F("Begin spill %s[%d] tensor[%d].", spillIssue->tileOp.GetOpcodeStr().c_str(), spillIssue->tileOp.GetOpMagic(), spillMemId);
    LogicalTensorPtr ddrTensor = nullptr;
    bool needCopyOut = false;
    if (spillIssue->tileOp.GetOpcodeStr().find("COPY_IN") == std::string::npos) {
        needCopyOut = true;
        IssueEntryPtr spillOutIssue = nullptr;
        if (CreateSpillCopyout(function, spillIssue, spillTensor, spillMemId, spillOutIssue) != SUCCESS) {
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
    if (CreateSpillReloadIssue(function, ddrTensor, spillTensor, spillIssue, reloadIssues) != SUCCESS) {
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

bool OoOScheduler::GetBufLastWriteTime(int curMemId, size_t& lastWriteTime) {
    lastWriteTime -= 1;
    while (lastWriteTime > 0) {
        for (auto& outTensor : issueEntries[lastWriteTime]->tileOp.GetOOperands()) {
            if (outTensor->memorymap[subGraphID].memId == curMemId) {
                return true;
            }
        }
        lastWriteTime -= 1U;
    }
    return false;
}

bool OoOScheduler::GetBufLastUseTime(int curMemId, size_t& lastUseTime) {
    lastUseTime -= 1;
    while (lastUseTime > 0) {
        for (auto& memId : issueEntries[lastUseTime]->reqMemIds) {
            if (memId == curMemId) {
                return true;
            }
        }
        lastUseTime -= 1U;
    }
    return false;
}

bool OoOScheduler::GetBufNextUseTime(int curMemId, size_t& nextUseTime) {
    while (++nextUseTime < issueEntries.size()) {
        if (issueEntries[nextUseTime]->isRetired) {
            continue;
        }
        for (auto& memId : issueEntries[nextUseTime]->reqMemIds) {
            if (memId == curMemId) {
                return true;
            }
        }
    }
    return false;
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

Status OoOScheduler::UpdateBufNextUseTime(int currPc, int memId, std::unordered_map<int, size_t> &nextUseTimeCache, std::vector<size_t> &bufNextUseTime) {
    if (nextUseTimeCache.find(memId) != nextUseTimeCache.end()) {
        bufNextUseTime.push_back(nextUseTimeCache[memId]);
        return SUCCESS;
    }
    size_t nextUseTime = currPc;
    if (!GetBufNextUseTime(memId, nextUseTime)) {
        ALOG_ERROR_F("Cannot find Tensor[%d] next used time.", memId);
        return FAILED;
    }
    nextUseTimeCache[memId] = nextUseTime;
    bufNextUseTime.push_back(nextUseTime);
    return SUCCESS;
}

Status OoOScheduler::UpdateGroupNextUseTime(int currPc, const std::vector<int> &group, std::unordered_map<int, size_t> &nextUseTimeCache, std::vector<int> &groupNextUseTime) {
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
        if (UpdateBufNextUseTime(currPc, memId, nextUseTimeCache, bufNextUseTime) != SUCCESS) {
            ALOG_ERROR_F("UpdateBufNextUseTime failed.");
            return FAILED;
        }
    }
    if (cannotSpill) {
        groupNextUseTime.push_back(-1);
        return SUCCESS;
    }
    groupNextUseTime.push_back(*std::min_element(bufNextUseTime.begin(), bufNextUseTime.end()));
    return SUCCESS;
}

Status OoOScheduler::SelectSpillBufferGroup(
    std::vector<std::vector<int>> &groups, int currPc, std::vector<int> &spillGroup) {
    if (groups.empty()) { ALOG_ERROR_F("Cannot find tensor to spill."); return FAILED; }
    std::unordered_map<int, size_t> nextUseTimeCache;
    std::vector<int> groupNextUseTime;
    for (auto& group : groups) {
        if (UpdateGroupNextUseTime(currPc, group, nextUseTimeCache, groupNextUseTime) != SUCCESS) {
            ALOG_ERROR_F("UpdateGroupNextUseTime failed.");
            return FAILED;
        }
    }
    size_t groupSel = std::max_element(groupNextUseTime.begin(), groupNextUseTime.end()) - groupNextUseTime.begin();
    if (groupNextUseTime[groupSel] == -1) {
        ALOG_ERROR_F("Cannot find tensor to spill.");
        return FAILED;
    }
    spillGroup = groups[groupSel];
    return SUCCESS;
}

Status OoOScheduler::SpillBufferGroup(Function &function, size_t &pcIdx, LocalBufferPtr allocBuffer, std::vector<std::vector<int>> &canSpillGroups) {
    std::vector<int> spillGroup;
    if (SelectSpillBufferGroup(canSpillGroups, pcIdx, spillGroup) != SUCCESS) {
        ALOG_ERROR_F("SelectSpillBufferGroup failed!");
        return FAILED;
    }
    for (auto spillMemId : spillGroup) {
        // 插入spill搬出重载数据的COPY_OUT/COPY_IN
        if (SpillBuffer(function, spillMemId, pcIdx, allocBuffer) != SUCCESS) {
            ALOG_ERROR_F("Tensor[%d] SpillBuffer failed!", spillMemId);
            return FAILED;
        }
        if (bufferManagerMap[allocBuffer->memType].Free(spillMemId) != SUCCESS) {
            ALOG_ERROR_F("Free spill tensor[%d] failed!", spillMemId);
            return FAILED;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::GenSpillOp(Function &function, LocalBufferPtr allocBuffer, size_t &pcIdx) {
    if (bufferManagerMap[allocBuffer->memType].IsFull(allocBuffer)) {
        ALOG_DEBUG_F("---> START: SPILL tensor.");
        if (allocBuffer->memType != MemoryType::MEM_L1 && allocBuffer->memType != MemoryType::MEM_UB) {
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
        if (SpillBufferGroup(function, pcIdx, allocBuffer, canSpillGroups) != SUCCESS) {
            ALOG_ERROR_F("SpillBufferGroup failed.");
            return FAILED;  
        }
        ALOG_DEBUG_F("---> END: SPILL tensor.");
    }
    return SUCCESS;
}

Status OoOScheduler::ProcessAlloc(Function &function, IssueEntryPtr &issue, size_t &pcIdx) {
    if (issue->isAlloc) {
        if (localBufferMap.find(issue->reqMemIds[0]) == localBufferMap.end()) {
            ALOG_ERROR_F("Tensor[%d] cannot find in localBufferMap!", issue->reqMemIds[0]);
            return FAILED;
        }
        LocalBufferPtr allocBuffer = localBufferMap[issue->reqMemIds[0]];
        if (GenSpillOp(function, allocBuffer, pcIdx) != SUCCESS) {
            ALOG_ERROR_F("GenSpillOp failed!");
            return FAILED;
        }
        if (bufferManagerMap[allocBuffer->memType].Allocate(allocBuffer) != SUCCESS) {
            ALOG_ERROR_F("Allocate tensor[%u] failed", allocBuffer->id);
            return FAILED;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::ProcessRetire(IssueEntryPtr &issue) {
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
    return SUCCESS;
}

Status OoOScheduler::ProcessAllocAndRetire(Function &function) {
    size_t pcIdx = 0;
    while (pcIdx < issueEntries.size()) {
        auto issue = issueEntries[pcIdx];
        ALOG_DEBUG_F("Launch %s[%d]", issue->tileOp.GetOpcodeStr().c_str(), issue->tileOp.GetOpMagic());
        if (ProcessAlloc(function, issue, pcIdx) != SUCCESS) {
            ALOG_ERROR_F("ProcessAlloc failed");
            return FAILED;
        }
        if (ProcessRetire(issue) != SUCCESS) {
            ALOG_ERROR_F("ProcessRetire failed");
            return FAILED;
        }
        pcIdx += 1;
    }
    return SUCCESS;
}

Status OoOScheduler::GenSpillSchedule(Function &function) {
    ALOG_DEBUG_F("=========> Begin GenSpillSchedule.");
    if (ProcessAllocAndRetire(function) != SUCCESS) {
        ALOG_ERROR_F("ProcessAllocAndRetire failed.");
        return FAILED;
    }
    for (auto bufRef : bufRefCount) {
        if (bufRef.second != 0) {
            ALOG_ERROR_F("Tensor[%d] bufRefCount not equal to 0!", bufRef.first);
            return FAILED;
        }
    }
    ALOG_DEBUG_F("=========> End GenSpillSchedule.");
    // 更新依赖关系
    if (InitDependencies() != SUCCESS) { ALOG_ERROR_F("InitDependencies failed!"); return FAILED; }
    return SUCCESS;
}

Status OoOScheduler::RetireOp(IssueEntryPtr &issue) {
    for (auto memId : issue->reqMemIds) {
        if (DelBufRefCount(memId) != SUCCESS) {
            ALOG_ERROR_F("DelBufRefCount tensor[%d] failed", memId);
            return FAILED;
        }
        if (bufRefCount[memId] == 0) {
            if (bufferManagerMap[localBufferMap[memId]->memType].Free(localBufferMap[memId]->id) != SUCCESS) { ALOG_ERROR_F("Free tensor[%d] failed", memId); return FAILED; }
            // Healthcheck record - update buffer usage statistics
            if (oooCheck.doHealthCheck) {
                UpdateBufferUsage(localBufferMap[memId]->memType, memId, true);
            }
            localBufferMap[memId]->retireCycle = clock;
            if (tensorOccupyMap[localBufferMap[memId]->memType].erase(localBufferMap[memId]->id) == 0) {
                ALOG_ERROR_F("Erase tensor[%d] failed", memId);
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status OoOScheduler::AwakeSucc(IssueEntryPtr &issue) {
    for (auto succId : issue->successors) {
        auto succ = issueEntryMap[succId];
        if (succ->isRetired) {
            continue;
        }
        bool ready = true;
        for (auto predId : succ->predecessors) {
            auto pred = issueEntryMap[predId];
            if (!pred->isRetired) {
                ready = false;
            }
        }
        if (ready) {
            issueQueues[succ->type].Insert(succ, succ->execOrder);
            ALOG_DEBUG_F("    Wakeup: %s[%d], execOrder: %d", succ->tileOp.GetOpcodeStr().c_str(), succ->tileOp.GetOpMagic(), succ->execOrder);
        }
    }
    return SUCCESS;
}

Status OoOScheduler::RetireOpAndAwakeSucc(IssueEntryPtr issue, uint64_t& commitCnt) {
    commitCnt++;
    issue->isRetired = true;
    if (RetireOp(issue) != SUCCESS) {
        ALOG_ERROR_F("RetireOp failed");
        return FAILED;
    }
    if (AwakeSucc(issue) != SUCCESS) {
        ALOG_ERROR_F("AwakeSucc failed");
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::RetireIssueStage(uint64_t& commitCnt, int& nextCycle) {
    for (auto& [pipeType, pipe] : issueQueues) {
        (void)pipeType;
        if (!pipe.busy) {
            continue;
        }
        if (pipe.curOpRetireCycle > clock) {   // 如果该pipe内当前正在执行op，在clock的时刻已经执行完毕。
            ALOG_DEBUG_F("EXECUTING[%ld]: %s[%d]", pipe.curOpRetireCycle,
                pipe.curIssue->tileOp.GetOpcodeStr().c_str(), pipe.curIssue->tileOp.GetOpMagic());
            if (nextCycle == -1 || nextCycle > pipe.curOpRetireCycle) {
                nextCycle = pipe.curOpRetireCycle;
            }
            continue;
        }
        IssueEntryPtr issue = pipe.curIssue;
        pipe.busy = false;
        pipe.curIssue = nullptr;
        ALOG_DEBUG_F("EXECUTE END: %s[%d]", issue->tileOp.GetOpcodeStr().c_str(), issue->tileOp.GetOpMagic());
        if (RetireOpAndAwakeSucc(issue, commitCnt) != SUCCESS) {
            ALOG_ERROR_F("RetireOpAndAwakeSucc failed!");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::ReallocOutTensor(IssueEntryPtr &issue) {
    for (auto& outTensor : issue->tileOp.GetOOperands()) {
        MemoryType memType = outTensor->GetMemoryTypeOriginal();
        if (memType == MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        int memId = outTensor->memorymap[subGraphID].memId;
        if (tensorOccupyMap.find(memType) == tensorOccupyMap.end()) {
            ALOG_ERROR_F("%s cannot find in tensorOccupyMap.", MemoryTypeToString(memType).c_str());
            return FAILED;
        }
        if (tensorOccupyMap[memType].find(memId) == tensorOccupyMap[memType].end()) {
            ALOG_ERROR_F("Tensor[%d] cannot find in tensorOccupyMap.", memId);
            return FAILED;
        }
        if (localBufferMap.find(memId) == localBufferMap.end()) {
            ALOG_ERROR_F("Tensor[%d] cannot find in localBufferMap.", memId);
            return FAILED;
        }
        ALOG_DEBUG_F("REALLOC Tensor[%u] %s %d --> %s, %d. ", memId,
            tensorOccupyMap[memType][memId]->tileOp.GetOpcodeStr().c_str(),
            tensorOccupyMap[memType][memId]->tileOp.GetOpMagic(),
            issue->tileOp.GetOpcodeStr().c_str(), issue->tileOp.GetOpMagic());
        tensorOccupyMap[memType][memId] = issue;
        outTensor->memorymap[subGraphID] =
            TileRange(localBufferMap[memId]->start, localBufferMap[memId]->end, memId);
    }
    return SUCCESS;
}

Status OoOScheduler::LaunchIssueStage(int& nextCycle, std::vector<Operation *> &newOperations) {
    // issue from all pipes
    for (auto &[pipeType, pipe] : issueQueues) {
        if (pipe.Empty() || pipe.busy) {
            continue;
        }
        IssueEntryPtr issue = pipe.PopFront();
        pipe.busy = true;
        pipe.curIssue = issue;
        pipe.curOpRetireCycle = clock + issue->tileOp.GetLatency();
        oooCheck.pipeUsageCount[pipeType] += issue->tileOp.GetLatency();
        if (ReallocOutTensor(issue) != SUCCESS) {
            ALOG_ERROR_F("ReallocOutTensor failed.");
            return FAILED;
        }
        newOperations.emplace_back(&(issue->tileOp));
        if (nextCycle == -1 || nextCycle > pipe.curOpRetireCycle) {
            nextCycle = pipe.curOpRetireCycle;
        }
        ALOG_DEBUG_F("Insert: %s[%d]", issue->tileOp.GetOpcodeStr().c_str(), issue->tileOp.GetOpMagic());
    }
    return SUCCESS;
}

Status OoOScheduler::AllocateTensor(MemoryType memType, IssueEntryPtr &issue, std::vector<Operation *>& newOperations) {
    ALOG_DEBUG_F("ALLOCATE: %s[%d]", issue->tileOp.GetOpcodeStr().c_str(), issue->tileOp.GetOpMagic());
    if (bufferManagerMap[memType].Allocate(localBufferMap[issue->reqMemIds[0]]) != SUCCESS) {
        ALOG_ERROR_F("Allocate Tensor[%d] failed.", issue->reqMemIds[0]);
        return FAILED;
    }
    // Healthcheck record - update buffer usage statistics
    if (oooCheck.doHealthCheck) {
        UpdateBufferUsage(memType, issue->reqMemIds[0], false);
    }
    tensorOccupyMap[memType][issue->reqMemIds[0]] = issue;
    localBufferMap[issue->reqMemIds[0]]->startCycle = clock;
    if (issue->tileOp.GetOutputOperand(0) == nullptr) {
        ALOG_ERROR_F("Alloc[%d] cannot find oOperand[0].", issue->tileOp.GetOpMagic());
        return FAILED;
    }
    issue->tileOp.GetOutputOperand(0)->SetAttr(OpAttributeKey::needAlloc, true);
    newOperations.push_back(&(issue->tileOp));
    ALOG_DEBUG_F("Insert: %s[%d]", issue->tileOp.GetOpcodeStr().c_str(), issue->tileOp.GetOpMagic());
    return SUCCESS;
}

Status OoOScheduler::ExecuteAllocOp(MemoryType memType, IssueQueue &pipe, std::vector<Operation *>& newOperations, uint64_t& commitCnt) {
    bool canAlloc = true;
    while (canAlloc) {
        if (pipe.Empty()) {
            canAlloc = false;
            break;
        }
        IssueEntryPtr issue = pipe.Front();
        if (bufferManagerMap[memType].IsFull(localBufferMap[issue->reqMemIds[0]])) {
            canAlloc = false;
            break;
        }
        if (AllocateTensor(memType, issue, newOperations) != SUCCESS) {
            ALOG_ERROR_F("AllocateTensor failed.");
            return FAILED;
        }
        pipe.PopFront();
        if (RetireOpAndAwakeSucc(issue, commitCnt) != SUCCESS) {
            ALOG_ERROR_F("RetireOpAndAwakeSucc failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::BufferAllocStage(std::vector<Operation *>& newOperations, uint64_t& commitCnt) {
    for (auto& [memType, pipe] : allocIssueQueue) {
        if (pipe.Empty()) {
            continue;
        }
        // 不断按顺序执行alloc指令，直到buffer被占满为止。
        if (ExecuteAllocOp(memType, pipe, newOperations, commitCnt) != SUCCESS) {
            ALOG_ERROR_F("ExecuteAllocOp failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

void OoOScheduler::LaunchReadyIssue() {
    for (size_t i = 0; i < issueEntries.size(); i++) {
        issueEntries[i]->execOrder = i;
        if (USE_LESS_OPS.find(issueEntries[i]->tileOp.GetOpcode()) != USE_LESS_OPS.end() &&
            issueEntries[i]->predecessors.empty()) {
            issueQueues[issueEntries[i]->type].Insert(issueEntries[i], i);
        }
        if (issueEntries[i]->isAlloc) {
            allocIssueQueue[localBufferMap[issueEntries[i]->reqMemIds[0]]->memType].Insert(issueEntries[i], i);
        }
    }
}

Status OoOScheduler::ProcessScheduleStage(uint64_t &commitCnt, int &nextCycle, std::vector<Operation *> &newOperations) {
    // Retire Stage : 检查现有pipe中的op是否执行完。如果op执行完，则将op标记为retired状态，将可以被释放的buffer释放掉，并唤醒后续已经就绪的op。
    // 完毕后更新整个pipe的状态。
    if (RetireIssueStage(commitCnt, nextCycle) != SUCCESS) {
        ALOG_ERROR_F("RetireIssueStage failed.");
        return FAILED;
    }
    // Buffer Allocation Stage : 分配buffer。对于所有类型的buffer，按顺序执行alloc指令，并激活后续已经就绪的op。不断执行alloc直到buffer被占满为止。
    if (BufferAllocStage(newOperations, commitCnt) != SUCCESS) {
        ALOG_ERROR_F("BufferAllocStage failed.");
        return FAILED;
    }
    // Launch Stage ：检查idle的pipe中是否有已经就绪的指令。如果有，则执行该指令，并更新pipe的状态为busy。
    if (LaunchIssueStage(nextCycle, newOperations) != SUCCESS) {
        ALOG_ERROR_F("LaunchIssueStage failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::AdjustMemType(Function &func, int nextCycle, std::vector<Operation *> &newOperations) {
    if (nextCycle != -1) {
        clock = nextCycle;
        return SUCCESS;
    }
    // 如果nextCycle为-1，说明每个pipe都处于idle的状态，判断出现阻塞。需要spill调整内存
    if (allocIssueQueue[MemoryType::MEM_UB].Empty() && allocIssueQueue[MemoryType::MEM_L1].Empty()) {
        ALOG_ERROR_F("Buffer[L0A/B/C] is Full. Please check tile shape and OOO spill failed info.");
        return FAILED;
    }
    MemoryType spillMemType = allocIssueQueue[MemoryType::MEM_UB].Empty() ? MemoryType::MEM_L1 : MemoryType::MEM_UB;
    if (GenBufferSpill(func, allocIssueQueue[spillMemType].Front(), spillMemType, newOperations) != SUCCESS) {
        ALOG_ERROR_F("GenBufferSpill failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::ProcessTillRetired(Function &func, std::vector<Operation *> &newOperations) {
    uint64_t commitCnt = 0; // 当前已提交的issue数量
    bool isAllRetired = false;
    while (!isAllRetired) {
        int nextCycle = -1;
        ALOG_DEBUG_F("\n clock: %d", clock);
        if (ProcessScheduleStage(commitCnt, nextCycle, newOperations) != SUCCESS) {
            ALOG_ERROR_F("ProcessScheduleStage failed.");
            return FAILED;
        }
        if (numTotalIssues == commitCnt && nextCycle == -1) {
            isAllRetired = true;
            break;
        }
        if (AdjustMemType(func, nextCycle, newOperations) != SUCCESS) {
            ALOG_ERROR_F("AdjustMemType failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::UpdateIssues() {
    for (const auto &issue : issueEntries) {
        if (!issue->isRetired) {
            ALOG_ERROR_F("Unexecuted op: %s %d", issue->tileOp.GetOpcodeStr().c_str(), issue->tileOp.GetOpMagic());
            return FAILED;
        }
        if (issue->isAlloc) {
            issue->tileOp.GetOutputOperand(0)->memorymap[subGraphID].lifeStart =
                localBufferMap[issue->reqMemIds[0]]->startCycle;
            issue->tileOp.GetOutputOperand(0)->memorymap[subGraphID].lifeEnd =
                localBufferMap[issue->reqMemIds[0]]->retireCycle;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::ScheduleMainLoop(Function &func, std::vector<Operation *> &newOperations) {
    LaunchReadyIssue();
    if (ProcessTillRetired(func, newOperations) != SUCCESS) {
        ALOG_ERROR_F("ProcessTillRetired failed.");
        return FAILED;
    }
    if (UpdateIssues() != SUCCESS) {
        ALOG_ERROR_F("UpdateIssues failed.");
        return FAILED;
    }
    ALOG_DEBUG_F("====================== NEW OPS =====================");
    for (auto op : newOperations) {
        if (op->oOperand.empty()) {
            ALOG_INFO(op->GetOpcodeStr(), ", ", op->GetOpMagic());
            continue;
        }
        bool needAlloc = false;
        op->oOperand[0]->GetAttr(OpAttributeKey::needAlloc, needAlloc);
        ALOG_DEBUG_F("%s, %d, range[%zu, %zu], needAlloc: %d", op->GetOpcodeStr().c_str(), op->GetOpMagic(),
            op->oOperand[0]->memorymap[op->GetSubgraphID()].start,
            op->oOperand[0]->memorymap[op->GetSubgraphID()].end, static_cast<int>(needAlloc));
    }
    return SUCCESS;
}

Status OoOScheduler::ExecuteSortOps(Function &function) {
    std::string funcName = function.GetMagicName();
    auto funcNameToSortMethod = function.paramConfigs_.OoOPreScheduleMethodMap;
    std::string sortMethodStr = (funcNameToSortMethod.count(funcName) > 0) ?
                                    funcNameToSortMethod[funcName] :
                                    function.paramConfigs_.OoOPreScheduleMethodDefault;
    if (sortMethodStr != "PriorDFS" && sortMethodStr != "LayerBasedDFS") {
        ALOG_ERROR_F("PreSchedule method not recognized.");
        return FAILED;
    }
    SortOpMethod sortMethod = (sortMethodStr == "PriorDFS") ? SortOpMethod::PriorDFS : SortOpMethod::LayerBasedDFS;
    return SortOps(sortMethod);
}

Status OoOScheduler::Initiate(const std::vector<Operation *> &operations) {
    for (auto &op : operations) {
        if (op == nullptr) {
            ALOG_ERROR_F("Operation is nullptr!");
            return FAILED;
        }
        if (op->GetBoolAttribute(OpAttributeKey::dontTouch)) {
            continue;
        }
        ALOG_DEBUG_F("%s, %d, memId: %d", op->GetOpcodeStr().c_str(), op->GetOpMagic(),
            op->oOperand[0]->memorymap[op->GetSubgraphID()].memId);
    }
    subGraphID = operations.front()->GetSubgraphID();
    if (Init(operations) != SUCCESS) {
        ALOG_ERROR_F("Init failed!");
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::Schedule(
    Function &function, const std::vector<Operation *> &operations, std::vector<Operation *> &newOperations) {
    if (operations.empty()) {
        return SUCCESS;
    }
    ALOG_DEBUG_F("==================== ORIGIN OPS =====================");
    if (Initiate(operations) != SUCCESS) {
        ALOG_ERROR_F("Initiate failed!");
        return FAILED;
    }
    // op执行排序
    if (ExecuteSortOps(function) != SUCCESS) {
        ALOG_ERROR_F("ExecuteSortOps failed!");
        return FAILED;
    }
    // 生成spill指令
    if (GenSpillSchedule(function) != SUCCESS) {
        ALOG_ERROR_F("GenSpillSchedule failed!");
        return FAILED;
    }
    // 模拟调度
    if (ScheduleMainLoop(function, newOperations) != SUCCESS) {
        ALOG_ERROR_F("ScheduleMainLoop failed");
        return FAILED;
    }
    function.SetStackWorkespaceSize(workspaceOffset);
    return SUCCESS;
}

bool OoOSchedule::IsAicpuProgram(std::vector<Operation *> opList) {
    for (auto &op : opList) {
        if (op->GetCoreType() == CoreType::AICPU) {
            return true;
        }
    }
    return false;
}

Status OoOSchedule::RunOnFunction(Function &function) {
    ALOG_INFO_F("=============== START OoOSchedule ===============");
    int maxWorkeSpaceSize = 0;

    for (auto &program : function.rootFunc_->programs_) {
        auto opList = program.second->Operations().DuplicatedOpList();
        oriFunctions.emplace_back(program.second);
        if (IsAicpuProgram(opList)) {
            continue;
        }
        std::vector<Operation *> newOperations;
        OoOScheduler oooSchedule;
        ALOG_INFO_F("Subgraph[%d] OOOSchedule start.", program.first);
        if (oooSchedule.Schedule(*program.second, opList, newOperations) != SUCCESS) { ALOG_ERROR_F("Subgraph[%d] OoO Schedule failed.", program.first); return FAILED;}
        ALOG_INFO_F("Subgraph[%d] OOOSchedule end.", program.first);
        program.second->ScheduleBy(newOperations);
        program.second->RecordOOOSeq();
        RescheduleUtils::UpdateTensorConsProd(program.second);
        maxWorkeSpaceSize = std::max(maxWorkeSpaceSize, (*program.second).GetStackWorkespaceSize());
        function.SetStackWorkespaceSize(maxWorkeSpaceSize);
        oooSchedule.oooCheck.doHealthCheck = passDfxconfigs_.healthCheck;
        if (oooSchedule.oooCheck.doHealthCheck) {
            oooSchedule.oooCheck.workspaceOffset = oooSchedule.workspaceOffset;
            oooSchedule.oooCheck.clock = oooSchedule.clock;
            oooSchedule.oooCheck.jsonFileName = GetDumpFilePrefix(function, false, program.second, program.first);
            schedulerMap.insert({program.first, oooSchedule});
        }
    }
    ALOG_INFO_F("=============== END OoOSchedule =================");
    return SUCCESS;
}

void OoOSchedule::DoHealthCheckAfter(Function &function, const std::string &folderPath) {
    for (auto &scheduler : schedulerMap) {
        auto fileName = folderPath + '/' + scheduler.second.oooCheck.jsonFileName + "_Kernel_Graph_Health_Report.json";
        auto it = function.rootFunc_->programs_.find(scheduler.first);
        if (it != function.rootFunc_->programs_.end()) {
            auto subFunc = it->second;
            scheduler.second.oooCheck.DoHealthCheck(subFunc, fileName);
        }
    }
}

Status OoOSchedule::PreCheck(Function &function) {
    return checker.DoPreCheck(function);
}

Status OoOSchedule::PostCheck(Function &function) {
    checker.SetOriFunctions(oriFunctions);
    return checker.DoPostCheck(function);
}
} // namespace npu::tile_fwk