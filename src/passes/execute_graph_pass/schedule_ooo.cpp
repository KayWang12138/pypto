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

constexpr int64_t MAX_L0A_SIZE = 64 * 1024; 
constexpr int64_t MAX_L0C_SIZE = 128 * 1024;
constexpr int64_t MAX_BT_SIZE = 1 * 1024;
constexpr int64_t MAX_FIX_SIZE = 1 * 1024;
constexpr int32_t TWO_ISSUE = 2;
constexpr int32_t DEFAULT_LATENCY = 511;
constexpr int32_t DIM_FIVE = 5;
constexpr int32_t LAST_TWO_DIM = 2;
constexpr int32_t UB_BLOCK_SIZE = 32;

IssueEntry::IssueEntry(Operation *op, uint64_t issueId) : tileOp(op), execOrder(issueId), 
    type(RescheduleUtils::GetOpPipeType(op)) {
    if (tileOp->GetOpcodeStr().find("ALLOC") != std::string::npos) {
        isAlloc = true;
    }
}

int IssueEntry::GetOOperandIdx(int curMemId) {
    for (size_t i = 0; i < tileOp->GetOOperands().size(); i++) {
        if (tileOp->GetOOperands()[i]->memorymap[tileOp->GetSubgraphID()].memId == curMemId) {
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
    for (size_t index = 0; index < tileOp->GetIOperands().size(); index++) {
        for (auto &inOp : tileOp->GetIOperands()[index]->GetProducers()) {
            if (inOp == spillSrcIssue->tileOp) {
                tileOp->UpdateInputOperand(index, tensor);
            }
        }
    }
}

uint64_t OoOScheduler::ShapeCeilAlign(std::vector<int> shape, DataType dtype) {
    uint64_t bytes = 0;
    if (shape.size() == DIM_FIVE) {
        bytes = BytesPerElement(dtype) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        bytes = CeilAlign(bytes, UB_BLOCK_SIZE);
    } else {
        uint64_t preDimSize = 1;
        uint64_t last2DimSize = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            if ((i < (shape.size() - LAST_TWO_DIM)) && (shape.size() != 1)) {
                preDimSize *= shape[i];
            } else {
                last2DimSize *= shape[i];
            }
        }
        bytes = preDimSize * CeilAlign(last2DimSize * BytesPerElement(dtype), UB_BLOCK_SIZE);
    }
    return bytes;
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

Status OoOScheduler::GetOldestBuffer(IssueEntryPtr allocIssue, MemoryType bufferType, int& memId,
    IssueEntryPtr &spillIssue) {
    std::set<IssueEntryPtr> filterLtags;
    FindFilterLtags(allocIssue, filterLtags);
    uint64_t maxIdx = 0;
    for (auto &occupyIssue : tensorOccupyMap[bufferType]) {
        if (occupyIssue.second->isAlloc || filterLtags.count(occupyIssue.second) != 0 ||
            USE_LESS_OPS.find(occupyIssue.second->tileOp->GetOpcode()) != USE_LESS_OPS.end()) {
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
        ALOG_ERROR_F("Could not find avaialble buffer to spill!"); 
        return FAILED; 
    }
    ALOG_DEBUG_F("  Spill op: %s %d.", spillIssue->tileOp->GetOpcodeStr().c_str(), spillIssue->tileOp->GetOpMagic());
    return SUCCESS;
}

Status OoOScheduler::UpdateTensorAttr(LogicalTensorPtr tensor, MemoryType memType, LogicalTensorPtr spillTensor, 
    int spillMemId) {
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
        tensor->memorymap[subGraphID] = TileRange(workspaceOffset, 
            workspaceOffset + localBufferMap[spillMemId]->size, workspaceMemId++);
        workspaceOffset += localBufferMap[spillMemId]->size;
    } else {
        tensor->memorymap[subGraphID].memId = maxTensorMagic;
        localBufferMap[maxTensorMagic] = std::make_shared<LocalBuffer>(maxTensorMagic,
            ShapeCeilAlign(tensor->GetShape(), tensor->Datatype()), tensor->GetMemoryTypeOriginal()); 
        if (localBufferMap[maxTensorMagic] == nullptr) {
            ALOG_ERROR_F("Init Tensor[%d] localBuffer failed.", maxTensorMagic);
            return FAILED;
        }
    }
    return SUCCESS;
}

void OoOScheduler::UpdateOpAttr(Operation &op, int opLatency, LogicalTensorPtr spillTensor, std::vector<int> offset, 
    IssueEntryPtr spillIssue) {
    if (op.GetOpcode() == Opcode::OP_COPY_OUT) {
        op.SetOpAttribute(std::make_shared<CopyOpAttribute>(spillTensor->GetMemoryTypeOriginal(), 
        OpImmediate::Specified(offset), OpImmediate::Specified(spillTensor->GetShape()), 
        OpImmediate::Specified(spillTensor->GetRawTensor()->GetDynRawShape())));
    } else if (op.GetOpcodeStr().find("ALLOC") == std::string::npos) {
        if (spillIssue->tileOp->GetOpcode() == Opcode::OP_COPY_IN) {
            op.SetOpAttribute(spillIssue->tileOp->GetOpAttribute());
            op.inParamLocation_ = spillIssue->tileOp->inParamLocation_;
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
        for (auto &outTensor : issue->tileOp->GetOOperands()) {
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
    spillTensor = spillIssue->tileOp->GetOutputOperand(spillTensorIdx);
    if (spillTensor == nullptr) { 
        ALOG_ERROR_F("Op cannot find oOperand[%d]", spillTensorIdx); 
        return FAILED; 
    }
    return SUCCESS;
}

Status OoOScheduler::UpdateReloadIssueInfo(IssueEntryPtr reloadAlloc, IssueEntryPtr reloadCopyin, 
    IssueEntryPtr spillIssue, int spillMemId, int bufNextUseTime) {
    reloadAlloc->reqMemIds = {maxTensorMagic};
    reloadAlloc->successors.insert(reloadCopyin);
    reloadCopyin->reqMemIds = {maxTensorMagic};
    reloadCopyin->predecessors.insert(reloadAlloc);

    for (auto& succ : spillIssue->successors) {
        if (!succ->isRetired && (std::count(succ->reqMemIds.begin(), succ->reqMemIds.end(), spillMemId) > 0)) {
            reloadCopyin->successors.insert(succ);
            if (succ->predecessors.erase(spillIssue) == 0) {
                ALOG_ERROR_F("Erase issueEntry %s, %d failed", spillIssue->tileOp->GetOpcodeStr().c_str(),
                    spillIssue->tileOp->GetOpMagic());
                return FAILED;
            }
            succ->predecessors.insert(reloadCopyin);
            if (reloadCopyin->tileOp->GetOutputOperand(0) == nullptr) {
                ALOG_ERROR_F("%s %d cannot find oOperand[0]", reloadCopyin->tileOp->GetOpcodeStr().c_str(), 
                    reloadCopyin->tileOp->GetOpMagic());
                return FAILED;
            }
            succ->UpdateTensorInput(spillIssue, reloadCopyin->tileOp->GetOutputOperand(0));
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
    std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(spillTensor->Datatype(), 
        spillTensor->GetShape(), "WorkspaceGm", SYMBOL_STACK_BASE);
    if (ddrRawTensor == nullptr) { 
        ALOG_ERROR_F("Create DDR raw tensor failed!"); 
        return FAILED; 
    }
    std::vector<int> offset(spillTensor->GetShape().size(), 0);
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
    spillCopyout = std::make_shared<IssueEntry>(&spillOutOp, issueEntries.size());
    if (spillCopyout == nullptr) { 
        ALOG_ERROR_F("Create OP_COPY_OUT issueEntry failed!"); 
        return FAILED; 
    }
    spillCopyout->reqMemIds = {spillMemId};
    spillCopyout->predecessors.insert(spillIssue);
    spillIssue->successors.insert(spillCopyout);
    spillCopyout->isRetired = true;
    ALOG_DEBUG_F("Add SPILL_OUT: %s[%d].", spillCopyout->tileOp->GetOpcodeStr().c_str(), spillCopyout->tileOp->GetOpMagic());
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
    FunctionUtils::AddControlEdge(spillAllocOp, spillCopyInOp);

    // 初始化OP_COPY_IN/OP_ALLOC的issueEntry
    IssueEntryPtr spillAllocInst = std::make_shared<IssueEntry>(&spillAllocOp, issueEntries.size());
    IssueEntryPtr spillInInst = std::make_shared<IssueEntry>(&spillCopyInOp, issueEntries.size() + 1);
    if (spillAllocInst == nullptr || spillInInst == nullptr) { 
        ALOG_ERROR_F("Create OP_COPY_IN/OP_ALLOC issueEntry failed!"); 
        return FAILED; 
    }
    reloadIssues.first = spillAllocInst;
    reloadIssues.second = spillInInst;
    ALOG_DEBUG_F("Add SPILL_ALLOC: %s[%d].", spillAllocInst->tileOp->GetOpcodeStr().c_str(), spillAllocInst->tileOp->GetOpMagic());
    ALOG_DEBUG_F("Add SPILL_IN: %s[%d].", spillInInst->tileOp->GetOpcodeStr().c_str(), spillInInst->tileOp->GetOpMagic());
    return SUCCESS;
}

Status OoOScheduler::GenBufferSpill(Function &function, IssueEntryPtr allocIssue, MemoryType bufferType, 
    std::vector<Operation *> &newOperations) {
    int spillMemId = -1;
    IssueEntryPtr spillIssue = nullptr;
    if (GetOldestBuffer(allocIssue, bufferType, spillMemId, spillIssue) != SUCCESS) { 
        ALOG_ERROR_F("GetOldestBuffer failed."); return FAILED; 
    }
    LogicalTensorPtr spillTensor = nullptr;
    if (GetSpillTensor(spillIssue, spillMemId, spillTensor) != SUCCESS) {
        ALOG_ERROR_F("%d %s GetSpillTensor failed!", spillIssue->tileOp->GetOpMagic(), 
            spillIssue->tileOp->GetOpcodeStr().c_str());
        return FAILED;
    }
    LogicalTensorPtr ddrTensor = nullptr;
    if (spillIssue->tileOp->GetOpcode() != Opcode::OP_COPY_IN) { // 若spill的tensor来自OP_COPY_IN，则数据无需搬出到DDR
        IssueEntryPtr spillCopyout = nullptr;
        if (CreateSpillCopyout(function, spillIssue, spillTensor, spillMemId, spillCopyout) != SUCCESS) { 
            ALOG_ERROR_F("CreateSpillCopyout failed!");  return FAILED; 
        }
        issueEntries.emplace_back(spillCopyout);
        ddrTensor = spillCopyout->tileOp->GetOutputOperand(0);
        newOperations.push_back(spillCopyout->tileOp);
        ALOG_DEBUG_F("Insert: %s[%d]", spillCopyout->tileOp->GetOpcodeStr().c_str(), spillCopyout->tileOp->GetOpMagic());
    } else { // 若spill的tensor不来自OP_COPY_IN，则将tensor搬出，在需要的时候再搬入
        ddrTensor = spillIssue->tileOp->GetInputOperand(0);
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
        ALOG_ERROR_F("UpdateReloadIssueInfo failed!"); return FAILED; 
    }
    if (bufferManagerMap[bufferType].Free(spillMemId) != SUCCESS) { 
        ALOG_ERROR_F("Free spill tensor[%d] failed!", spillMemId); return FAILED; 
    }
    localBufferMap[spillMemId]->retireCycle = clock;
    if (tensorOccupyMap[bufferType].erase(spillMemId) == 0) { 
        ALOG_ERROR_F("Erase tensor[%d] failed", spillMemId);  return FAILED; 
    }
    allocIssueQueue[localBufferMap[spillMemId]->memType].InsertReloadAlloc(reloadAlloc, spillIssue);
    return SUCCESS;
}

void OoOScheduler::PrintDependenciesAndRelations() {
    for (const auto &issue : issueEntries) {
        if (issue->tileOp->GetBoolAttribute(OpAttributeKey::dontTouch)) {
            continue;
        }
        ALOG_DEBUG_F("%d %s, latency: %d.", issue->tileOp->GetOpMagic(), 
            issue->tileOp->GetOpcodeStr().c_str(), issue->tileOp->GetLatency());
        for (const auto &pre : issue->predecessors) {
            ALOG_DEBUG_F("    |--- Predecessors:");
            ALOG_DEBUG_F("        |--- %s[%d]", pre->tileOp->GetOpcodeStr().c_str(), pre->tileOp->GetOpMagic());
        }
        for (const auto &successor : issue->successors) {
            ALOG_DEBUG_F("    |--- Successors:");
            ALOG_DEBUG_F("        |--- %s[%d]", successor->tileOp->GetOpcodeStr().c_str(), successor->tileOp->GetOpMagic());
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

void OoOScheduler::AddDependencies(IssueEntryPtr issue, 
    std::map<int, IssueEntryPtr> lastWriteOpMap, LogicalTensorPtr tensor) {
    int memId = tensor->memorymap[subGraphID].memId;
    if (tensor->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
        bufRefCount[memId]++;
        issue->reqMemIds.push_back(memId);
    }
    if (lastWriteOpMap.find(memId) != lastWriteOpMap.end()) {
        issue->predecessors.insert(lastWriteOpMap[memId]);
        lastWriteOpMap[memId]->successors.insert(issue);
    }
}

Status OoOScheduler::InitLocalBuffer(LogicalTensorPtr oOperand, int memId) {
    if (oOperand->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
        return SUCCESS;
    }
    if (localBufferMap.find(memId) == localBufferMap.end()) {
        localBufferMap[memId] = std::make_shared<LocalBuffer>(memId, 
            ShapeCeilAlign(oOperand->GetShape(), oOperand->Datatype()), oOperand->GetMemoryTypeOriginal());
        if (localBufferMap[memId] == nullptr) { 
            ALOG_ERROR_F("Init tensor[%d] localBuffer failed!", memId); 
            return FAILED; 
        }
    } else {
        localBufferMap[memId]->size = 
            std::max(localBufferMap[memId]->size, ShapeCeilAlign(oOperand->GetShape(), oOperand->Datatype()));
    }
    return SUCCESS;
}

Status OoOScheduler::InitDependencies() {
    bufRefCount.clear();
    std::map<int, IssueEntryPtr> lastWriteOpMap;
    for (const auto &issue : issueEntries) {
        issue->Clear();
        // 仅检测 RAW 和 WAW，不检测 WAR -->SSA
        // RAW
        for (auto &inTensor : issue->tileOp->GetIOperands()) {
            AddDependencies(issue, lastWriteOpMap, inTensor);
        }

        // WAW
        for (auto &outTensor : issue->tileOp->GetOOperands()) {
            AddDependencies(issue, lastWriteOpMap, outTensor);
        }
        
        for (auto &oOperand : issue->tileOp->GetOOperands()) {
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

Status OoOScheduler::CheckAllocIssue() {
    std::map<int, IssueEntryPtr> tensorAllocMap;
    for (const auto &issue : issueEntries) {
        if (issue->isAlloc) {
            if (issue->reqMemIds.size() != 1) {
                ALOG_ERROR_F("ALLOC[%d] reqMemIds size not equal to 0.", issue->tileOp->GetOpMagic());
                return FAILED;
            }
        }
        for (auto outTensor : issue->tileOp->GetOOperands()) {
            if (outTensor->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            int memId = outTensor->memorymap[subGraphID].memId;
            if (tensorAllocMap.find(memId) == tensorAllocMap.end()) {
                tensorAllocMap[memId] = issue;
            }
        }
        for (auto inTensor : issue->tileOp->GetIOperands()) {
            if (inTensor->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            int memId = inTensor->memorymap[subGraphID].memId;
            if (tensorAllocMap.find(memId) == tensorAllocMap.end()) {
                tensorAllocMap[memId] = issue;
            }
        }
    }
    for (auto tensorAlloc : tensorAllocMap) {
        if (!tensorAlloc.second->isAlloc) {
            ALOG_ERROR_F("%s[%d] Tensor[%d] is missing Alloc.", tensorAlloc.second->tileOp->GetOpcodeStr().c_str(), 
                tensorAlloc.second->tileOp->GetOpMagic(), tensorAlloc.first);
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
        } else {
            newOperations.push_back(op);
        }
    }

    // 校验并初始化issueEntry
    uint64_t issueId = 0;
    for (const auto &op : newOperations) {
        maxOpMagic = std::max(maxOpMagic, op->GetOpMagic());
        if (CheckOpBufferSize(op) != SUCCESS) {
            ALOG_ERROR_F("%s[%d] CheckOpBufferSize failed!", op->GetOpcodeStr().c_str(), op->GetOpMagic()); 
            return FAILED; 
        }
        auto issue = std::make_shared<IssueEntry>(op, issueId++);
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
    for (auto& preIssue : issue->predecessors) {
        if (!visited[preIssue]) {
            curr.insert(preIssue);
            preNodeTotal.insert(preIssue);
        }
    }
    while (!curr.empty()) {
        std::unordered_set<IssueEntryPtr> next;
        for (auto& curIssue : curr) {
            for (auto& preIssue : curIssue->predecessors) {
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

IssueEntryPtr OoOScheduler::FindNodeMinNumUnvisitedPreNode(std::map<IssueEntryPtr, bool> visited, 
    std::vector<IssueEntryPtr> outNodeQueue) {
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
        
        bool ready = true;
        for (auto& preIssue : curIssue->predecessors) {
            if (!visited[preIssue]) {
                ready = false;
                break;
            }
        }

        if (ready) {
            visited[curIssue] = true;
            queue.pop_front();
            newIssueEntries.push_back(curIssue);
        } else {
            std::vector<IssueEntryPtr> notReadyPreNode;
            for (auto& preIssue : curIssue->predecessors) {
                if (!visited[preIssue]) {
                    notReadyPreNode.push_back(preIssue);
                }
            }
            std::sort(notReadyPreNode.begin(), notReadyPreNode.end(), [&](IssueEntryPtr a, IssueEntryPtr b) {
                int priorA = 10;
                int priorB = 10;
                if (preNodePriority.find(a->tileOp->GetOpcode()) != preNodePriority.end()) {
                    priorA = preNodePriority[a->tileOp->GetOpcode()];
                }
                if (preNodePriority.find(b->tileOp->GetOpcode()) != preNodePriority.end()) {
                    priorB = preNodePriority[b->tileOp->GetOpcode()];
                }
                if (priorA != priorB) {
                    return priorA < priorB;
                } else {
                    return a->execOrder < b->execOrder;
                }
            });
            for (auto& preIssue : notReadyPreNode) {
                queue.push_front(preIssue);
            }
        }
    }
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

    if (outNodeQueue.size() != 0) {
       DFSFromSingleNode(outNodeQueue[0], visited, newIssueEntries, preNodePriority);
    } else {
        ALOG_ERROR_F("Subgraph must have operation with outdegree 0.");
        return FAILED;
    }

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

Status OoOScheduler::SortOps() {
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
    if (PriorDFS(preNodePriority) != SUCCESS) { ALOG_ERROR_F("PriorDFS failed."); return FAILED; }
    return SUCCESS;
}

Status OoOScheduler::GetBufTimes(int spillMemId, size_t &bufNextUseTime, size_t &bufLastUseTime, 
    size_t &bufLastWriteTime) {
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

Status OoOScheduler::SpillBuffer(Function &function, int spillMemId, size_t &pcIdx) {
    size_t bufNextUseTime = pcIdx;
    size_t bufLastUseTime = pcIdx;
    size_t bufLastWriteTime = pcIdx;
    if (GetBufTimes(spillMemId, bufNextUseTime, bufLastUseTime, bufLastWriteTime) != SUCCESS) { ALOG_ERROR_F("GetBufTimes failed!"); return FAILED; }
    auto spillIssue = issueEntries[bufLastWriteTime];
    LogicalTensorPtr spillTensor = nullptr;
    if (GetSpillTensor(spillIssue, spillMemId, spillTensor) != SUCCESS) {
        ALOG_ERROR_F("%d %s GetSpillTensor failed!", spillIssue->tileOp->GetOpMagic(), 
            spillIssue->tileOp->GetOpcodeStr().c_str());
        return FAILED;
    }
    ALOG_DEBUG_F("Begin spill %s[%d] tensor[%d].", spillIssue->tileOp->GetOpcodeStr().c_str(), 
        spillIssue->tileOp->GetOpMagic(), spillMemId);
    LogicalTensorPtr ddrTensor = nullptr;
    if (spillIssue->tileOp->GetOpcodeStr().find("COPY_IN") == std::string::npos) {
        IssueEntryPtr spillOutIssue = nullptr;
        if (CreateSpillCopyout(function, spillIssue, spillTensor, spillMemId, spillOutIssue) != SUCCESS) { ALOG_ERROR_F("CreateSpillCopyout failed!"); return FAILED; }
        issueEntries.insert(issueEntries.begin() + bufLastUseTime + 1, spillOutIssue);
        bufNextUseTime++;
        ddrTensor = spillOutIssue->tileOp->GetOutputOperand(0);
        pcIdx++;
        numTotalIssues++;
    } else {
        ddrTensor = spillIssue->tileOp->GetInputOperand(0);
    }
    IssueEntryPtr reloadCopyin = nullptr;
    IssueEntryPtr reloadAlloc = nullptr;
    std::pair<IssueEntryPtr, IssueEntryPtr> reloadIssues = {reloadAlloc, reloadCopyin};
    if (CreateSpillReloadIssue(function, ddrTensor, spillTensor, spillIssue, reloadIssues) != SUCCESS) { ALOG_ERROR_F("CreateSpillReloadIssue failed!"); return FAILED; }
    reloadAlloc = reloadIssues.first;
    reloadCopyin = reloadIssues.second;
    if (UpdateReloadIssueInfo(reloadAlloc, reloadCopyin, spillIssue, spillMemId, bufNextUseTime) != SUCCESS) { ALOG_ERROR_F("UpdateReloadIssueInfo failed!"); return FAILED; }
    return SUCCESS;
}

bool OoOScheduler::GetBufLastWriteTime(int curMemId, size_t& lastWriteTime) {
    lastWriteTime -= 1;
    while (lastWriteTime > 0) {
        for (auto& outTensor : issueEntries[lastWriteTime]->tileOp->GetOOperands()) {
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
    for (auto &dstIssue : allocIssue->successors) {
        for (auto &inIssue : dstIssue->predecessors) {
            filterLtags.insert(inIssue);
        }
    }
}

Status OoOScheduler::SelectSpillBufferGroup(std::vector<std::vector<int>>& groups, int currPc, 
    std::vector<int> &spillGroup) {
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
            if (spillIssue->tileOp->GetOpcode() == Opcode::OP_VIEW || 
                spillIssue->tileOp->GetOpcode() == Opcode::OP_ASSEMBLE || filterLtags.count(spillIssue) != 0) {
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
        ALOG_ERROR_F("Cannot find tensor to spill."); 
        return FAILED;
    }
    spillGroup = groups[groupSel];
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
        std::vector<std::vector<int>> canSpillGroups = 
            bufferManagerMap[allocBuffer->memType].GetSpillGroup(allocBuffer->size);
        // 选择最晚被使用的spill 单个或多个tensor
        std::vector<int> spillGroup;
        if (SelectSpillBufferGroup(canSpillGroups, pcIdx, spillGroup) != SUCCESS) { ALOG_ERROR_F("SelectSpillBufferGroup failed!"); return FAILED;}
        for (auto spillMemId : spillGroup) {
            // 插入spill搬出重载数据的COPY_OUT/COPY_IN
            if (SpillBuffer(function, spillMemId, pcIdx) != SUCCESS) { ALOG_ERROR_F("Tensor[%d] SpillBuffer failed!", spillMemId); return FAILED; }
            if (bufferManagerMap[allocBuffer->memType].Free(spillMemId) != SUCCESS) { ALOG_ERROR_F("Free spill tensor[%d] failed!", spillMemId); return FAILED; }
        }
        ALOG_DEBUG_F("---> END: SPILL tensor.");
    }
    return SUCCESS;
}

Status OoOScheduler::GenSpillSchedule(Function &function) {
    size_t pcIdx = 0;
    ALOG_DEBUG_F("=========> Begin GenSpillSchedule.");
    while (pcIdx < issueEntries.size()) {
        auto issue = issueEntries[pcIdx];
        ALOG_DEBUG_F("Launch %s[%d]", issue->tileOp->GetOpcodeStr().c_str(), issue->tileOp->GetOpMagic());
        if (issue->isAlloc) {
            if (localBufferMap.find(issue->reqMemIds[0]) == localBufferMap.end()) { 
                ALOG_ERROR_F("Tensor[%d] cannot find in localBufferMap!", issue->reqMemIds[0]); 
                return FAILED; 
            }
            LocalBufferPtr allocBuffer = localBufferMap[issue->reqMemIds[0]];
            if (GenSpillOp(function, allocBuffer, pcIdx) != SUCCESS) { ALOG_ERROR_F("GenSpillOp failed!"); return FAILED; }
            if (bufferManagerMap[allocBuffer->memType].Allocate(allocBuffer) != SUCCESS) { ALOG_ERROR_F("Allocate tensor[%u] failed", allocBuffer->id); return FAILED; }
        }
        issue->isRetired = true;
        for (auto memId : issue->reqMemIds) {
            if (DelBufRefCount(memId) != SUCCESS) { ALOG_ERROR_F("DelBufRefCount tensor[%d] failed", memId); return FAILED; }
            if (bufRefCount[memId] == 0) {
                if (bufferManagerMap[localBufferMap[memId]->memType].Free(localBufferMap[memId]->id) != SUCCESS) { ALOG_ERROR_F("Free tensor[%d] failed", memId); return FAILED; }
            }
        }
        pcIdx += 1;
    }
    for (auto bufRef : bufRefCount) {
        if (bufRef.second != 0) { ALOG_ERROR_F("Tensor[%d] bufRefCount not equal to 0!", bufRef.first); return FAILED; }
    }
    ALOG_DEBUG_F("=========> End GenSpillSchedule.");
    // 更新依赖关系
    if (InitDependencies() != SUCCESS) { ALOG_ERROR_F("InitDependencies failed!"); return FAILED; }
    return SUCCESS;
}

Status OoOScheduler::RetireOpAndAwakeSucc(IssueEntryPtr issue, uint64_t& commitCnt) {
    commitCnt++;
    issue->isRetired = true;
    for (auto memId : issue->reqMemIds) {
        if (DelBufRefCount(memId) != SUCCESS) { ALOG_ERROR_F("DelBufRefCount tensor[%d] failed", memId); return FAILED; }
        if (bufRefCount[memId] == 0) {
            if (bufferManagerMap[localBufferMap[memId]->memType].Free(localBufferMap[memId]->id) != SUCCESS) { ALOG_ERROR_F("Free tensor[%d] failed", memId); return FAILED; }
            localBufferMap[memId]->retireCycle = clock;
            if (tensorOccupyMap[localBufferMap[memId]->memType].erase(localBufferMap[memId]->id) == 0) {
                ALOG_ERROR_F("Erase tensor[%d] failed", memId);
                return FAILED;
            }
        }
    }

    for (auto succ : issue->successors) {
        if (succ->isRetired) {
            continue;
        }
        bool ready = true;
        for (auto pred : succ->predecessors) {
            if (!pred->isRetired) {
                ready = false;
            }
        }
        if (ready) {
            issueQueues[succ->type].Insert(succ, succ->execOrder);
            ALOG_DEBUG_F("    Wakeup: %s[%d], execOrder: %d", succ->tileOp->GetOpcodeStr().c_str(), succ->tileOp->GetOpMagic(), succ->execOrder);
        }
    }
    return SUCCESS;
}

Status OoOScheduler::RetireIssueStage(uint64_t& commitCnt, int& nextCycle) {
    for (auto& [pipeType, pipe] : issueQueues) {
        (void)pipeType;
        if (!pipe.busy) {
            continue;
        }
        if (pipe.curOpRetireCycle <= clock) {   // 如果该pipe内当前正在执行op，在clock的时刻已经执行完毕。
            IssueEntryPtr issue = pipe.curIssue;
            pipe.busy = false;
            pipe.curIssue = nullptr;
            ALOG_DEBUG_F("EXECUTE END: %s[%d]", issue->tileOp->GetOpcodeStr().c_str(), issue->tileOp->GetOpMagic());
            if (RetireOpAndAwakeSucc(issue, commitCnt) != SUCCESS) { ALOG_ERROR_F("RetireOpAndAwakeSucc failed!"); return FAILED; }
        } else {
            ALOG_DEBUG_F("EXECUTING[%ld]: %s[%d]", pipe.curOpRetireCycle,
                pipe.curIssue->tileOp->GetOpcodeStr().c_str(), pipe.curIssue->tileOp->GetOpMagic());
            if (nextCycle == -1 || nextCycle > pipe.curOpRetireCycle) {
                nextCycle = pipe.curOpRetireCycle;
            }
        }
    }
    return SUCCESS;
}

Status OoOScheduler::LaunchIssueStage(int& nextCycle, std::vector<Operation *> &newOperations) {
    // issue from all pipes
    for (auto &[pipeType, pipe] : issueQueues) {
        (void)pipeType;
        if (pipe.Empty() || pipe.busy) {
            continue;
        }
        IssueEntryPtr issue = pipe.PopFront();
        pipe.busy = true;
        pipe.curIssue = issue;
        pipe.curOpRetireCycle = clock + issue->tileOp->GetLatency();

        newOperations.emplace_back(issue->tileOp);
        if (nextCycle == -1 || nextCycle > pipe.curOpRetireCycle) {
            nextCycle = pipe.curOpRetireCycle;
        }
        for (auto& outTensor : issue->tileOp->GetOOperands()) {
            MemoryType memType = outTensor->GetMemoryTypeOriginal();
            if (memType == MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            int memId = outTensor->memorymap[subGraphID].memId;
            if (tensorOccupyMap.find(memType) != tensorOccupyMap.end()) {
                if (tensorOccupyMap[memType].find(memId) == tensorOccupyMap[memType].end()) {
                    ALOG_ERROR_F("Tensor[%d] cannot find in tensorOccupyMap.", memId);
                    return FAILED;  
                }
            } else {
                ALOG_ERROR_F("%s cannot find in tensorOccupyMap.", MemoryTypeToString(memType).c_str());
                return FAILED;
            }
            if (localBufferMap.find(memId) == localBufferMap.end()) {
                ALOG_ERROR_F("Tensor[%d] cannot find in localBufferMap.", memId);
                return FAILED;
            }
            ALOG_DEBUG_F("REALLOC Tensor[%u] %s %d --> %s, %d. ", memId,
                tensorOccupyMap[memType][memId]->tileOp->GetOpcodeStr().c_str(),
                tensorOccupyMap[memType][memId]->tileOp->GetOpMagic(),
                issue->tileOp->GetOpcodeStr().c_str(), issue->tileOp->GetOpMagic());
            tensorOccupyMap[memType][memId] = issue;
            outTensor->memorymap[subGraphID] = 
                TileRange(localBufferMap[memId]->start, localBufferMap[memId]->end, memId);
        }
        ALOG_DEBUG_F("Insert: %s[%d]", issue->tileOp->GetOpcodeStr().c_str(), issue->tileOp->GetOpMagic());
    }
    return SUCCESS;
}

Status OoOScheduler::BufferAllocStage(std::vector<Operation *>& newOperations, uint64_t& commitCnt) {
    for (auto& [memType, pipe] : allocIssueQueue) {
        if (pipe.Empty()) {
            continue;
        }
        // 不断按顺序执行alloc指令，直到buffer被占满为止。
        bool canAlloc = true;
        while (canAlloc) {
            if (pipe.Empty()) {
                canAlloc = false;
                break;
            }
            IssueEntryPtr issue = pipe.Front();
            if (!bufferManagerMap[memType].IsFull(localBufferMap[issue->reqMemIds[0]])) {
                ALOG_DEBUG_F("ALLOCATE: %s[%d]", issue->tileOp->GetOpcodeStr().c_str(), issue->tileOp->GetOpMagic());
                if (bufferManagerMap[memType].Allocate(localBufferMap[issue->reqMemIds[0]]) != SUCCESS) { ALOG_ERROR_F("Allocate Tensor[%d] failed.", issue->reqMemIds[0]); return FAILED; }
                tensorOccupyMap[memType][issue->reqMemIds[0]] = issue;
                localBufferMap[issue->reqMemIds[0]]->startCycle = clock;
                if (issue->tileOp->GetOutputOperand(0) == nullptr) { 
                    ALOG_ERROR_F("Alloc[%d] cannot find oOperand[0].", issue->tileOp->GetOpMagic()); 
                    return FAILED; 
                }
                issue->tileOp->GetOutputOperand(0)->SetAttr(OpAttributeKey::needAlloc, true);
                newOperations.push_back(issue->tileOp);
                ALOG_DEBUG_F("Insert: %s[%d]", issue->tileOp->GetOpcodeStr().c_str(), issue->tileOp->GetOpMagic());
                pipe.PopFront();
                if (RetireOpAndAwakeSucc(issue, commitCnt) != SUCCESS) { ALOG_ERROR_F("RetireOpAndAwakeSucc failed."); return FAILED; }
            } else {
                canAlloc = false;
                break;
            }
        }
    }
    return SUCCESS;
}

void OoOScheduler::LaunchReadyIssue() {
    for (size_t i = 0; i < issueEntries.size(); i++) {
        issueEntries[i]->execOrder = i;
        if (USE_LESS_OPS.find(issueEntries[i]->tileOp->GetOpcode()) != USE_LESS_OPS.end() && 
            issueEntries[i]->predecessors.empty()) {
            issueQueues[issueEntries[i]->type].Insert(issueEntries[i], i);
        }
        if (issueEntries[i]->isAlloc) {
            allocIssueQueue[localBufferMap[issueEntries[i]->reqMemIds[0]]->memType].Insert(issueEntries[i], i);
        }
    }
}

Status OoOScheduler::ScheduleMainLoop(Function &func, std::vector<Operation *> &newOperations) {
    LaunchReadyIssue();
    uint64_t commitCnt = 0; // 当前已提交的issue数量
    bool isAllRetired = false;
    while (!isAllRetired) {
        int nextCycle = -1;
        ALOG_DEBUG_F("\n clock: %d", clock);
        // Retire Stage : 检查现有pipe中的op是否执行完。如果op执行完，则将op标记为retired状态，将可以被释放的buffer释放掉，并唤醒后续已经就绪的op。
        // 完毕后更新整个pipe的状态。
        if (RetireIssueStage(commitCnt, nextCycle) != SUCCESS) { ALOG_ERROR_F("RetireIssueStage failed."); return FAILED;}
        // Buffer Allocation Stage : 分配buffer。对于所有类型的buffer，按顺序执行alloc指令，并激活后续已经就绪的op。不断执行alloc直到buffer被占满为止。
        if (BufferAllocStage(newOperations, commitCnt) != SUCCESS) { ALOG_ERROR_F("BufferAllocStage failed."); return FAILED;}
        // Launch Stage ：检查idle的pipe中是否有已经就绪的指令。如果有，则执行该指令，并更新pipe的状态为busy。
        if (LaunchIssueStage(nextCycle, newOperations) != SUCCESS) { ALOG_ERROR_F("LaunchIssueStage failed."); return FAILED; }
        if (numTotalIssues == commitCnt && nextCycle == -1) { isAllRetired = true; break; }
        // 如果nextCycle为-1，说明每个pipe都处于idle的状态，判断出现阻塞。需要spill调整内存
        if (nextCycle == -1) {
            MemoryType spillMemType;
            if (!allocIssueQueue[MemoryType::MEM_UB].Empty()) {
                spillMemType = MemoryType::MEM_UB;
            } else if (!allocIssueQueue[MemoryType::MEM_L1].Empty()) {
                spillMemType = MemoryType::MEM_L1;
            } else { ALOG_ERROR_F("Buffer[L0A/B/C] is Full. Please check tile shape and OOO spill failed info."); return FAILED; }
            if (GenBufferSpill(func, allocIssueQueue[spillMemType].Front(), spillMemType, newOperations) != SUCCESS) {
                ALOG_ERROR_F("GenBufferSpill failed."); return FAILED; }
        } else { clock = nextCycle; }
    }
    for (const auto &issue : issueEntries) {
        if (!issue->isRetired) { ALOG_ERROR_F("Unexecuted op: %s %d", issue->tileOp->GetOpcodeStr().c_str(), issue->tileOp->GetOpMagic()); return FAILED; }
        if (issue->isAlloc) {
            issue->tileOp->GetOutputOperand(0)->memorymap[subGraphID].lifeStart = 
                localBufferMap[issue->reqMemIds[0]]->startCycle;
            issue->tileOp->GetOutputOperand(0)->memorymap[subGraphID].lifeEnd = 
                localBufferMap[issue->reqMemIds[0]]->retireCycle;
        }
    }
    ALOG_DEBUG_F("====================== NEW OPS =====================");
    for (auto op : newOperations) {
        if (!op->oOperand.empty()) {
            bool needAlloc = false;
            op->oOperand[0]->GetAttr(OpAttributeKey::needAlloc, needAlloc);
            ALOG_DEBUG_F("%s, %d, range[%zu, %zu], needAlloc: %d", op->GetOpcodeStr().c_str(), op->GetOpMagic(),
                op->oOperand[0]->memorymap[op->GetSubgraphID()].start,
                op->oOperand[0]->memorymap[op->GetSubgraphID()].end, static_cast<int>(needAlloc));
        } else { ALOG_INFO(op->GetOpcodeStr(), ", ", op->GetOpMagic()); }
    }
    return SUCCESS;
}

Status OoOScheduler::Schedule(Function &function, const std::vector<Operation *> &operations, 
    std::vector<Operation *> &newOperations) {
    if (operations.empty()) {
        return SUCCESS;
    }
    ALOG_DEBUG_F("==================== ORIGIN OPS =====================");
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

    if (Init(operations) != SUCCESS) { ALOG_ERROR_F("Init failed!"); return FAILED; }

    // op执行排序
    if (SortOps() != SUCCESS) { ALOG_ERROR_F("SortOps failed!"); return FAILED; }

    // 生成spill指令
    if (GenSpillSchedule(function) != SUCCESS) { ALOG_ERROR_F("GenSpillSchedule failed!"); return FAILED; }
    
    // 模拟调度
    if (ScheduleMainLoop(function, newOperations) != SUCCESS) { ALOG_ERROR_F("ScheduleMainLoop failed"); return FAILED; }
    function.SetStackWorkespaceSize(workspaceOffset);
    return SUCCESS;
}

bool OoOSchedulePass::IsAicpuProgram(std::vector<Operation *> opList) {
    for (auto &op : opList) {
        if (op->GetCoreType() == CoreType::AICPU) {
            return true;
        }
    }
    return false;
}

Status OoOSchedulePass::RunOnFunction(Function &function) {
    ALOG_INFO_F("=============== START OoOSchedulePass ===============");
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
    }
    ALOG_INFO_F("=============== END OoOSchedulePass =================");
    return SUCCESS;
}

} // namespace npu::tile_fwk