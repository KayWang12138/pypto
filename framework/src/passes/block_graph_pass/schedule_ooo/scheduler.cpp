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
 * \file scheduler.cpp
 * \brief
 */

#include "scheduler.h"
#include "buffer_rearrange.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "OoOSchedule"

namespace npu::tile_fwk {

constexpr int64_t MAX_L0A_SIZE = 64 * 1024;
constexpr int64_t MAX_L0C_SIZE = 128 * 1024;
constexpr int64_t MAX_BT_SIZE = 1 * 1024;
constexpr int64_t MAX_FIX_SIZE = 1 * 1024;
constexpr int64_t MAX_FIX_QUANT_PRE_SIZE = 1 * 2048;
constexpr int32_t DIM_FIVE = 5;
constexpr int32_t LAST_TWO_DIM = 2;
constexpr int32_t UB_BLOCK_SIZE = 32;

IssueEntry::IssueEntry(Operation &op, uint64_t issueId)
    : tileOp(op), id(issueId), execOrder(issueId), type(RescheduleUtils::GetOpPipeType(&op)) {
    if (tileOp.GetOpcodeStr().find("ALLOC") != std::string::npos) {
        isAlloc = true;
    }
    for (auto iOperand : op.GetIOperands()) {
        for (auto pre : iOperand->GetProducers()) {
            if (pre->GetOpcode() == Opcode::OP_VIEW &&
                pre->GetOutputOperand(0)->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
                viewOps.push_back(pre);
            }
        }
    }
}

int IssueEntry::GetOOperandIdx(int curMemId) {
    for (size_t i = 0; i < tileOp.GetOOperands().size(); i++) {
        if (tileOp.GetOOperands()[i]->memoryrange.memId == curMemId) {
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
        UpdateTensorInputForOperand(index, spillSrcIssue, tensor);
    }
}

void IssueEntry::UpdateTensorInputForOperand(size_t index, std::shared_ptr<IssueEntry> &spillSrcIssue,
    LogicalTensorPtr tensor) const {
    for (auto &inOp : tileOp.GetIOperands()[index]->GetProducers()) {
        if (inOp->GetOpcode() == Opcode::OP_VIEW) {
            Operation* op = inOp;
            UpdateTensorInputForView(op, spillSrcIssue, tensor);
        } else if (inOp == &(spillSrcIssue->tileOp)) {
            tileOp.UpdateInputOperand(index, tensor);
        }
    }
}

void IssueEntry::UpdateTensorInputForView(Operation *op,
    std::shared_ptr<IssueEntry> &spillSrcIssue, LogicalTensorPtr tensor) const {
    for (auto it : op->GetInputOperand(0)->GetProducers()) {
        if (it == &(spillSrcIssue->tileOp)) {
            op->UpdateInputOperand(0, tensor);
            op->GetOutputOperand(0)->memoryrange.memId = tensor->memoryrange.memId;
            break;
        }
    }
}

const char* IssueEntry::GetOpInfo() {
    return (tileOp.GetOpcodeStr() + "[" + std::to_string(tileOp.GetOpMagic()) + "]").c_str();
}

Status OoOScheduler::PrintSpillFailedInfo(IssueEntryPtr allocIssue) {
    APASS_LOG_ERROR_F(Elements::Operation, "======== OoO Spill failed info ===========");
    APASS_LOG_ERROR_F(Elements::Operation, "Spill failed memoryType: %s.", 
        MemoryTypeToString(localBufferMap[allocIssue->reqMemIds[0]]->memType).c_str());
    if (localBufferMap.find(allocIssue->reqMemIds[0]) != localBufferMap.end()) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s alloc buffer size: %lu.", allocIssue->GetOpInfo(), 
            localBufferMap[allocIssue->reqMemIds[0]]->size);
    }
    auto bufferSlices = bufferManagerMap[localBufferMap[allocIssue->reqMemIds[0]]->memType].GetBufferSlices();
    for (auto memId : bufferSlices) {
        auto occupyIssue = GetBufLastWriteIssue(allocIssue, memId);
        if (occupyIssue == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] last write time.", memId);
            return FAILED;
        }
        APASS_LOG_ERROR_F(Elements::Operation, "%s, range[%lu, %lu], Tensor[%d] size: %lu.", occupyIssue->GetOpInfo(), 
            localBufferMap[memId]->start, localBufferMap[memId]->end, memId, localBufferMap[memId]->size);
    }
    return SUCCESS;
}

void OoOScheduler::PrintSpillFailedInfo(IssueEntryPtr allocIssue, MemoryType bufferType) {
    APASS_LOG_ERROR_F(Elements::Operation, "======== OoO Spill failed info ===========");
    APASS_LOG_ERROR_F(Elements::Operation, "Spill failed memoryType: %s.", MemoryTypeToString(bufferType).c_str());
    if (localBufferMap.find(allocIssue->reqMemIds[0]) != localBufferMap.end()) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s alloc buffer size: %lu.", allocIssue->GetOpInfo(), 
            localBufferMap[allocIssue->reqMemIds[0]]->size);
    }
    if (tensorOccupyMap.find(bufferType) != tensorOccupyMap.end()) {
        for (auto occupyIssue : tensorOccupyMap[bufferType]) {
            APASS_LOG_ERROR_F(Elements::Operation, "%s, range[%lu, %lu], Tensor[%d] size: %lu.", occupyIssue.second->GetOpInfo(),
                localBufferMap[occupyIssue.first]->start, localBufferMap[occupyIssue.first]->end, 
                occupyIssue.first, localBufferMap[occupyIssue.first]->size);
        }
    }
}

void OoOScheduler::PrintDependencies() {
    for (const auto &issue : issueEntries) {
        APASS_LOG_DEBUG_F(Elements::Operation, "%s, latency: %d.", issue->GetOpInfo(), issue->tileOp.GetLatency());
        for (const auto &preId : issue->predecessors) {
            auto pre = issueEntryMap[preId];
            APASS_LOG_DEBUG_F(Elements::Operation, "    |--- Predecessors:");
            APASS_LOG_DEBUG_F(Elements::Operation, "        |--- %s", pre->GetOpInfo());
        }
        for (const auto &succId : issue->successors) {
            auto successor = issueEntryMap[succId];
            APASS_LOG_DEBUG_F(Elements::Operation, "    |--- Successors:");
            APASS_LOG_DEBUG_F(Elements::Operation, "        |--- %s", successor->GetOpInfo());
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "\n");
    }
}

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
    if (bufRefCount.find(memId) == bufRefCount.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "bufRefCount cannot find Tensor[%d].", memId);
        return FAILED;
    }
    bufRefCount[memId]--;
    if (bufRefCount[memId] < 0) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] bufRefCount cannot less than 0.", memId);
        return FAILED;
    }
    return SUCCESS;
}

void OoOScheduler::PrintOpList(std::vector<Operation *> operations) {
    APASS_LOG_DEBUG_F(Elements::Operation, "==================== OP_LIST =====================");
    for (auto &op : operations) {
        if (!op->oOperand.empty()) {
            bool needAlloc = false;
            op->oOperand[0]->GetAttr(OpAttributeKey::needAlloc, needAlloc);
            APASS_LOG_DEBUG_F(Elements::Operation, "%s[%d], range[%zu, %zu], needAlloc: %d", 
                op->GetOpcodeStr().c_str(), op->GetOpMagic(), op->oOperand[0]->memoryrange.start,
                op->oOperand[0]->memoryrange.end, static_cast<int>(needAlloc));
        } else {
            APASS_LOG_INFO_F(Elements::Operation, "%s[%d]", op->GetOpcodeStr(), op->GetOpMagic()); 
        }
    }
}

void OoOScheduler::InsertIssueEntries(IssueEntryPtr insertIssue) {
    auto it = issueEntries.begin();
    for (; it != issueEntries.end(); it++) {
        if ((*it)->execOrder >= insertIssue->execOrder) {
            break;
        }
    }
    auto insertPos = issueEntries.insert(it++, insertIssue);
    for (auto adjustIt = insertPos + 1; adjustIt != issueEntries.end(); adjustIt++) {
        if ((*adjustIt)->execOrder >= insertIssue->execOrder) {
            (*adjustIt)->execOrder++;
        }
    }
}

void OoOScheduler::UpdateIssueExecOrder() {
    for (size_t idx = 0; idx < issueEntries.size(); idx++) {
        issueEntries[idx]->execOrder = idx;
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

Status OoOScheduler::CheckAndUpdateLifecycle() {
    for (const auto &issue : issueEntries) {
        if (!issue->isRetired) { 
            APASS_LOG_ERROR_F(Elements::Operation, "Unexecuted op: %s.", issue->GetOpInfo()); 
            return FAILED; 
        }
        if (issue->isAlloc) {
            issue->tileOp.GetOutputOperand(0)->memoryrange.lifeStart =
                localBufferMap[issue->reqMemIds[0]]->startCycle;
            issue->tileOp.GetOutputOperand(0)->memoryrange.lifeEnd =
                localBufferMap[issue->reqMemIds[0]]->retireCycle;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::SpillOnBlock() {
    MemoryType spillMemType;
    if (!allocIssueQueue[MemoryType::MEM_UB].Empty()) {
        spillMemType = MemoryType::MEM_UB;
    } else if (!allocIssueQueue[MemoryType::MEM_L1].Empty()) {
        spillMemType = MemoryType::MEM_L1;
    } else {
        for (auto memType: allocIssueQueue) {
            if (memType.second.Empty()) {
                continue;
            }
            PrintSpillFailedInfo(memType.second.Front(), memType.first);
        }
        APASS_LOG_ERROR_F(Elements::Operation, "Buffer[L0A/B/C] is Full. Please check tile shape and OOO spill failed info."); 
        return FAILED; 
    }
    if (RearrangeBuffers(allocIssueQueue[spillMemType].Front(), false) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "SpillOnBlock failed at RearrangeBuffers.");
        return FAILED;
    }
    return SUCCESS;
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

Status OoOScheduler::AllocTensorMemRange(IssueEntryPtr issue) {
    for (auto& op : issue->viewOps) {
        if (op->GetOpcode() != Opcode::OP_VIEW) {
            APASS_LOG_ERROR_F(Elements::Operation, "op[%s] is not OP_VIEW.", op->GetOpMagic());
            return FAILED;
        }
        if (AllocViewTensorMemRange(*op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "AllocViewTensorMemRange failed.");
            return FAILED;
        }
    }
    for (auto& outTensor : issue->tileOp.GetOOperands()) {
        MemoryType memType = outTensor->GetMemoryTypeOriginal();
        if (memType == MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        int memId = outTensor->memoryrange.memId;
        if (tensorOccupyMap.find(memType) != tensorOccupyMap.end()) {
            if (tensorOccupyMap[memType].find(memId) == tensorOccupyMap[memType].end()) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in tensorOccupyMap.", memId);
                return FAILED;
            }
        } else {
            APASS_LOG_ERROR_F(Elements::Operation, "%s cannot find in tensorOccupyMap.", MemoryTypeToString(memType).c_str());
            return FAILED;
        }
        if (localBufferMap.find(memId) == localBufferMap.end()) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in localBufferMap.", memId);
            return FAILED;
        }
        APASS_LOG_DEBUG_F(Elements::Tensor, "REALLOC Tensor[%u] %s --> %s.", 
            memId, tensorOccupyMap[memType][memId]->GetOpInfo(), issue->GetOpInfo());
        if (tensorOccupyMap[memType][memId]->isAlloc) {
            outTensor->SetAttr(OpAttributeKey::needAlloc, true);
        }
        tensorOccupyMap[memType][memId] = issue;
        outTensor->memoryrange =
            TileRange(localBufferMap[memId]->start, localBufferMap[memId]->end, memId);
    }
    return SUCCESS;
}

Status OoOScheduler::LaunchIssueStage(int& nextCycle) {
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
        for (auto& op : issue->viewOps) {
            if (std::find(newOperations_.begin(), newOperations_.end(), op) != newOperations_.end()) {
                continue;
            }
            newOperations_.emplace_back(op);
        }
        newOperations_.emplace_back(&(issue->tileOp));
        if (nextCycle == -1 || nextCycle > pipe.curOpRetireCycle) {
            nextCycle = pipe.curOpRetireCycle;
        }
        if (AllocTensorMemRange(issue) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "AllocTensorMemRange failed.");
            return FAILED;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "Insert: %s.", issue->GetOpInfo());
    }
    return SUCCESS;
}

Status OoOScheduler::ExecuteAllocIssue(uint64_t &commitCnt, MemoryType memType, IssueQueue &pipe) {
    bool canAlloc = true;
    while (canAlloc) {
        if (pipe.Empty()) {
            canAlloc = false;
            break;
        }
        IssueEntryPtr issue = pipe.Front();
        if (!bufferManagerMap[memType].IsFull(localBufferMap[issue->reqMemIds[0]])) {
            APASS_LOG_DEBUG_F(Elements::Operation, "ALLOCATE: %s.", issue->GetOpInfo());
            if (bufferManagerMap[memType].Allocate(localBufferMap[issue->reqMemIds[0]]) != SUCCESS) { 
                APASS_LOG_ERROR_F(Elements::Tensor, "Allocate Tensor[%d] failed.", issue->reqMemIds[0]); 
                return FAILED; 
            }
            // Healthcheck record - update buffer usage statistics
            if (oooCheck.doHealthCheck) {
                UpdateBufferUsage(memType, issue->reqMemIds[0], false);
            }
            tensorOccupyMap[memType][issue->reqMemIds[0]] = issue;
            localBufferMap[issue->reqMemIds[0]]->startCycle = clock;
            if (issue->tileOp.GetOutputOperand(0) == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation, "Alloc[%d] cannot find oOperand[0].", issue->tileOp.GetOpMagic());
                return FAILED;
            }
            newOperations_.push_back(&(issue->tileOp));
            APASS_LOG_DEBUG_F(Elements::Operation, "Insert: %s.", issue->GetOpInfo());
            pipe.PopFront();
            if (RetireOpAndAwakeSucc(issue, commitCnt) != SUCCESS) { 
                APASS_LOG_ERROR_F(Elements::Operation, "RetireOpAndAwakeSucc failed."); 
                return FAILED; 
            }
        } else {
            canAlloc = false;
            break;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::BufferAllocStage(uint64_t &commitCnt) {
    for (auto& [memType, pipe] : allocIssueQueue) {
        if (pipe.Empty()) {
            continue;
        }
        // 不断按顺序执行alloc指令，直到buffer被占满为止。
        if (ExecuteAllocIssue(commitCnt, memType, pipe) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "ExecuteAllocIssue failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::FreeBuffer(IssueEntryPtr issue) {
    for (auto memId : issue->reqMemIds) {
        if (DelBufRefCount(memId) != SUCCESS) { 
            APASS_LOG_ERROR_F(Elements::Tensor, "DelBufRefCount tensor [%d] failed.", memId); 
            return FAILED; 
        }
        if (bufRefCount[memId] == 0) {
            if (bufferManagerMap[localBufferMap[memId]->memType].Free(localBufferMap[memId]->id) != SUCCESS) { 
                APASS_LOG_ERROR_F(Elements::Tensor, "Free tensor [%d] failed.", memId); 
                return FAILED; 
            }
            // Healthcheck record - update buffer usage statistics
            if (oooCheck.doHealthCheck) {
                UpdateBufferUsage(localBufferMap[memId]->memType, memId, true);
            }
            localBufferMap[memId]->retireCycle = clock;
            if (tensorOccupyMap[localBufferMap[memId]->memType].erase(localBufferMap[memId]->id) == 0) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Erase tensor[%d] failed.", memId);
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status OoOScheduler::RetireOpAndAwakeSucc(IssueEntryPtr issue, uint64_t& commitCnt) {
    commitCnt++;
    issue->isRetired = true;
    if (FreeBuffer(issue) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "FreeBuffer failed.");
        return FAILED;
    }

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
            issueQueues[succ->type].Insert(succ);
            APASS_LOG_DEBUG_F(Elements::Operation, "    Wakeup: %s, execOrder: %d", succ->GetOpInfo(), succ->execOrder);
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
            APASS_LOG_DEBUG_F(Elements::Operation, "EXECUTE END: %s", issue->GetOpInfo());
            if (RetireOpAndAwakeSucc(issue, commitCnt) != SUCCESS) { 
                APASS_LOG_ERROR_F(Elements::Operation, "RetireOpAndAwakeSucc failed!"); 
                return FAILED; 
            }
        } else {
            APASS_LOG_DEBUG_F(Elements::Operation, "EXECUTING[%ld]: %s", pipe.curOpRetireCycle, pipe.curIssue->GetOpInfo());
            if (nextCycle == -1 || nextCycle > pipe.curOpRetireCycle) {
                nextCycle = pipe.curOpRetireCycle;
            }
        }
    }
    return SUCCESS;
}

void OoOScheduler::LaunchReadyIssue() {
    for (auto &issue : issueEntries) {
        if (USE_LESS_OPS.find(issue->tileOp.GetOpcode()) != USE_LESS_OPS.end() && issue->predecessors.empty()) {
            issueQueues[issue->type].Insert(issue);
        }
        if (issue->isAlloc) {
            allocIssueQueue[localBufferMap[issue->reqMemIds[0]]->memType].Insert(issue);
        }
    }
}

Status OoOScheduler::ScheduleMainLoop() {
    UpdateIssueExecOrder();
    LaunchReadyIssue();
    uint64_t commitCnt = 0; // 当前已提交的issue数量
    bool isAllRetired = false;
    while (!isAllRetired) {
        int nextCycle = -1;
        APASS_LOG_DEBUG_F(Elements::Operation, "\n clock: %d", clock);
        // Retire Stage : 检查现有pipe中的op是否执行完。如果op执行完，则将op标记为retired状态，将可以被释放的buffer释放掉，并唤醒后续已经就绪的op。
        // 完毕后更新整个pipe的状态。
        if (RetireIssueStage(commitCnt, nextCycle) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "RetireIssueStage failed.");
            return FAILED;
        }
        // Buffer Allocation Stage : 分配buffer。对于所有类型的buffer，按顺序执行alloc指令，并激活后续已经就绪的op。不断执行alloc直到buffer被占满为止。
        if (BufferAllocStage(commitCnt) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "BufferAllocStage failed.");
            return FAILED;
        }
        // Launch Stage ：检查idle的pipe中是否有已经就绪的指令。如果有，则执行该指令，并更新pipe的状态为busy。
        if (LaunchIssueStage(nextCycle) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "LaunchIssueStage failed.");
            return FAILED;
        }
        if (numTotalIssues == commitCnt && nextCycle == -1) {
            isAllRetired = true;
            break;
        }
        // 如果nextCycle为-1，说明每个pipe都处于idle的状态，判断出现阻塞。需要spill调整内存
        if (nextCycle == -1) {
            if (SpillOnBlock() != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "SpillOnBlock failed.");
                return FAILED;
            }
        } else {
            clock = nextCycle;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::RetireIssue(IssueEntryPtr issue) {
    issue->isRetired = true;
    for (auto memId : issue->reqMemIds) {
        if (DelBufRefCount(memId) != SUCCESS) { 
            APASS_LOG_ERROR_F(Elements::Tensor, "DelBufRefCount tensor[%d] failed.", memId); 
            return FAILED; 
        }
        if (bufRefCount[memId] == 0) {
            if (bufferManagerMap[localBufferMap[memId]->memType].Free(localBufferMap[memId]->id) != SUCCESS) { 
                APASS_LOG_ERROR_F(Elements::Tensor, "Free tensor[%d] failed.", memId); 
                return FAILED; 
            }
        }
    }
    return SUCCESS;
}

Status OoOScheduler::ExecuteAllocIssue(IssueEntryPtr issue, size_t &pcIdx) {
    if (localBufferMap.find(issue->reqMemIds[0]) == localBufferMap.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] cannot find in localBufferMap!", issue->reqMemIds[0]);
        return FAILED;
    }
    LocalBufferPtr allocBuffer = localBufferMap[issue->reqMemIds[0]];

    if (bufferManagerMap[allocBuffer->memType].IsFull(allocBuffer)) {
        if (GenSpillOp(allocBuffer, pcIdx) != SUCCESS) {
            APASS_LOG_WARN_F(Elements::Operation, "GenSpillOp failed, start trying buffer rearrangement.");
            if (bufferManagerMap[allocBuffer->memType].IsFullWithoutRearrange(allocBuffer->size)) {
                APASS_LOG_ERROR_F(Elements::Operation, "GenSpillOp failed and there is no enough buffer space for rearrangement.");
                return FAILED;
            }
            // 如果内存剩余空间 > 需要alloc空间, 进行内存重排
            if (RearrangeBuffers(issue, true) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "ExecuteAllocIssue failed at RearrangeBuffers!");
                return FAILED;
            }
        }
    }

    if (bufferManagerMap[allocBuffer->memType].Allocate(allocBuffer) != SUCCESS) { 
        APASS_LOG_ERROR_F(Elements::Tensor, "Allocate tensor[%u] failed.", allocBuffer->id); 
        return FAILED; 
    }
    return SUCCESS;
}

Status OoOScheduler::GenSpillSchedule() {
    UpdateIssueExecOrder();
    size_t pcIdx = 0;
    APASS_LOG_DEBUG_F(Elements::Operation, "=========> Begin GenSpillSchedule.");
    while (pcIdx < issueEntries.size()) {
        auto issue = issueEntries[pcIdx];
        APASS_LOG_DEBUG_F(Elements::Operation, "Launch %s", issue->GetOpInfo());
        if (issue->isAlloc) {
            if (ExecuteAllocIssue(issue, pcIdx) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "ExecuteAllocIssue failed!"); 
                return FAILED;
            }
        }
        if (RetireIssue(issue) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "RetireIssue failed!"); 
            return FAILED;
        }
        pcIdx += 1;
    }
    for (auto bufRef : bufRefCount) {
        if (bufRef.second != 0) { 
            APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] bufRefCount not equal to 0!", bufRef.first); 
            return FAILED; 
        }
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "=========> End GenSpillSchedule.");
    // 更新依赖关系
    if (InitDependencies() != SUCCESS) { 
        APASS_LOG_ERROR_F(Elements::Operation, "InitDependencies failed!"); 
        return FAILED; 
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
        if (localMemorySize.find(static_cast<MemoryType>(i)) != localMemorySize.end()) {
            bufferManagerMap.insert({static_cast<MemoryType>(i),
                BufferPool(static_cast<MemoryType>(i), localMemorySize[static_cast<MemoryType>(i)])});
        }
    }
}

void OoOScheduler::UpdateAllocMap(IssueEntryPtr issue, std::map<int, IssueEntryPtr> &tensorAllocMap) {
    for (auto outTensor : issue->tileOp.GetOOperands()) {
        if (outTensor->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        int memId = outTensor->memoryrange.memId;
        if (tensorAllocMap.find(memId) == tensorAllocMap.end()) {
            tensorAllocMap[memId] = issue;
        }
    }
    for (auto inTensor : issue->tileOp.GetIOperands()) {
        if (inTensor->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        int memId = inTensor->memoryrange.memId;
        if (tensorAllocMap.find(memId) == tensorAllocMap.end()) {
            tensorAllocMap[memId] = issue;
        }
    }
}

Status OoOScheduler::CheckAllocIssue() {
    std::map<int, IssueEntryPtr> tensorAllocMap;
    for (const auto &issue : issueEntries) {
        if (issue->isAlloc) {
            if (issue->reqMemIds.size() != 1) {
                APASS_LOG_ERROR_F(Elements::Operation, "ALLOC[%d] reqMemIds size not equal to 0.", issue->tileOp.GetOpMagic());
                return FAILED;
            }
        }
        UpdateAllocMap(issue, tensorAllocMap);
    }
    for (auto tensorAlloc : tensorAllocMap) {
        if (!tensorAlloc.second->isAlloc) {
            APASS_LOG_ERROR_F(Elements::Tensor, "%s Tensor[%d] is missing Alloc.", 
                tensorAlloc.second->GetOpInfo(), tensorAlloc.first);
            return FAILED;
        }
    }
    return SUCCESS;
}

void OoOScheduler::InitLocalBuffer(LogicalTensorPtr oOperand, int memId) {
    if (oOperand->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
        return;
    }
    if (localBufferMap.find(memId) == localBufferMap.end()) {
        localBufferMap[memId] = std::make_shared<LocalBuffer>(
            memId, ShapeCeilAlign(oOperand->GetShape(), oOperand->Datatype()), oOperand->GetMemoryTypeOriginal());
    } else {
        localBufferMap[memId]->size =
            std::max(localBufferMap[memId]->size, ShapeCeilAlign(oOperand->GetShape(), oOperand->Datatype()));
    }
}

void OoOScheduler::UpdateBufRefCount(IssueEntryPtr issue, LogicalTensorPtr tensor) {
    int memId = tensor->memoryrange.memId;
    if (tensor->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
        bufRefCount[memId]++;
        issue->reqMemIds.push_back(memId);
    }
}

void OoOScheduler::InitBufRefCount() {
    for (const auto &issue : issueEntries) {
        for (auto &tensor : issue->tileOp.GetIOperands()) {
            UpdateBufRefCount(issue, tensor);
        }
        for (auto &tensor : issue->tileOp.GetOOperands()) {
            UpdateBufRefCount(issue, tensor);
            int memId = tensor->memoryrange.memId;
            maxTensorMagic = std::max(maxTensorMagic, std::max(tensor->GetMagic(), memId));
            InitLocalBuffer(tensor, memId);
        }
    }
}

Status OoOScheduler::InitAllocDependencies(IssueEntryPtr issue, std::map<int, IssueEntryPtr> tensor2AllocMap) {
    for (auto &tensor : issue->tileOp.GetOOperands()) {
        int memId = tensor->memoryrange.memId;
        if (tensor->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
            if (tensor2AllocMap.find(memId) == tensor2AllocMap.end()) {
                APASS_LOG_ERROR_F(Elements::Operation, "Tensor[%d] must have alloc.", memId);
                return FAILED;
            }
            AddDependency(tensor2AllocMap[memId], issue, true);
        }
    }
    return SUCCESS;
}

void OoOScheduler::AddDependency(IssueEntryPtr preIssue, IssueEntryPtr postIssue, bool isAlloc) {
    if (isAlloc || (!preIssue->isAlloc && !postIssue->isAlloc)) {
        preIssue->successors.insert(postIssue->id);
        postIssue->predecessors.insert(preIssue->id);
    }
}

Status OoOScheduler::InitDependencies() {
    bufRefCount.clear();
    std::map<Operation*, IssueEntryPtr> op2IssueEntryMap;
    for (const auto &issue : issueEntries) {
        issue->Clear();
        op2IssueEntryMap[&(issue->tileOp)] = issue;
    }
    InitBufRefCount();
    std::map<int, IssueEntryPtr> tensor2AllocMap;
    for (const auto &issue : issueEntries) {
        if (issue->isAlloc) {
            if (issue->tileOp.GetOOperands().size() != 1) {
                APASS_LOG_ERROR_F(Elements::Operation, "Alloc[%d] oOperand must be 1.", issue->tileOp.GetOpMagic());
                return FAILED;
            }
            int memId = issue->tileOp.GetOutputOperand(0)->memoryrange.memId;
            tensor2AllocMap[memId] = issue;
            continue;
        }
        for (auto &producer : issue->tileOp.ProducerOps()) {
            if (producer->GetOpcode() == Opcode::OP_VIEW) {
                for (auto viewProducer : producer->ProducerOps()) {
                    auto viewProdIsue = op2IssueEntryMap[viewProducer];
                    AddDependency(viewProdIsue, issue, false);
                }
            } else {
                auto prodIssue = op2IssueEntryMap[producer];
                AddDependency(prodIssue, issue, false);
            }
        }
        for (auto &consumer : issue->tileOp.ConsumerOps()) {
            if (consumer->GetOpcode() == Opcode::OP_VIEW) {
                for (auto viewConsumer : consumer->ConsumerOps()) {
                    auto viewConIssue = op2IssueEntryMap[viewConsumer];
                    AddDependency(issue, viewConIssue, false);
                }
            }
        }
        if (InitAllocDependencies(issue, tensor2AllocMap) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "InitAllocDependencies failed.");
            return FAILED;
        }
    }
    PrintDependencies();
    return SUCCESS;
}

void OoOScheduler::CalcBufferSize(LogicalTensors tensors, std::map<MemoryType, int64_t> &bufferSize, std::set<int> &memIdMap) {
    for (auto tensor : tensors) {
        if (memIdMap.find(tensor->memoryrange.memId) == memIdMap.end()) {
            bufferSize[tensor->GetMemoryTypeOriginal()] += tensor->GetDataSize();
            memIdMap.insert(tensor->memoryrange.memId);
        }
    }
}

Status OoOScheduler::CheckOpBufferSize(Operation *op) {
    std::map<MemoryType, int64_t> bufferSize;
    std::set<int> memIdMap;
    CalcBufferSize(op->GetIOperands(), bufferSize, memIdMap);
    CalcBufferSize(op->GetOOperands(), bufferSize, memIdMap);
    for (auto &buffer : bufferSize) {
        if (localMemorySize.find(buffer.first) != localMemorySize.end()) {
            if (buffer.second > localMemorySize[buffer.first]) {
                APASS_LOG_ERROR_F(Elements::Operation, "OP %s[%d] in/output total size[%d] exceeds %s size[%d]!", 
                    op->GetOpcodeStr().c_str(), op->GetOpMagic(), buffer.second, MemoryTypeToString(buffer.first).c_str(),
                    localMemorySize[buffer.first]);
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status OoOScheduler::Init(const std::vector<Operation *> &operations) {
    issueEntries.clear();
    localBufferMap.clear();

    // 初始化芯片各buffer大小
    InitMemorySize();

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
        if (op->GetOpcode() == Opcode::OP_VIEW) {
            if (op->GetOutputOperand(0)->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
                newOperations_.push_back(op);
            }
            continue;
        }
        maxOpMagic = std::max(maxOpMagic, op->GetOpMagic());
        if (CheckOpBufferSize(op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] CheckOpBufferSize failed!", op->GetOpcodeStr().c_str(), op->GetOpMagic());
            return FAILED;
        }
        auto issue = std::make_shared<IssueEntry>(*op, issueId);
        issueEntryMap[issueId++] = issue;
        if (issue == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "IssueEntry %s, %d init failed!", op->GetOpcodeStr().c_str(), op->GetOpMagic());
            return FAILED;
        }
        issueEntries.emplace_back(issue);
    }
    numTotalIssues = issueEntries.size();

    // 初始化issueEntry，构建依赖关系
    if (InitDependencies() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InitDependencies failed!");
        return FAILED;
    }

    if (CheckAllocIssue() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "CheckAllocIssue failed!");
        return FAILED;
    }

    // 初始化内存管理器
    InitIssueQueuesAndBufferManager();
    return SUCCESS;
}

void OoOScheduler::InitMemorySize() {
    localMemorySize = {
        {MemoryType::MEM_L0A, MAX_L0A_SIZE},
        {MemoryType::MEM_L0C, MAX_L0C_SIZE},
        {MemoryType::MEM_BT, MAX_BT_SIZE},
        {MemoryType::MEM_FIX, MAX_FIX_SIZE},
        {MemoryType::MEM_FIX_QUANT_PRE, MAX_FIX_QUANT_PRE_SIZE},
    };
    localMemorySize.insert({MemoryType::MEM_UB,
        PassConfigManager::Instance().GetPlatformConfig().GetMemoryLimit(MemoryType::MEM_UB)});
    localMemorySize.insert({MemoryType::MEM_L1,
        PassConfigManager::Instance().GetPlatformConfig().GetMemoryLimit(MemoryType::MEM_L1)});
    localMemorySize.insert({MemoryType::MEM_L0B,
        PassConfigManager::Instance().GetPlatformConfig().GetMemoryLimit(MemoryType::MEM_L0B)});
    localMemorySize.insert({MemoryType::MEM_FIX_QUANT_PRE,
        PassConfigManager::Instance().GetPlatformConfig().GetMemoryLimit(MemoryType::MEM_FIX_QUANT_PRE)});
}

Status OoOScheduler::Schedule(const std::vector<Operation *> &operations) {
    if (operations.empty()) {
        return SUCCESS;
    }
    PrintOpList(operations);
    if (Init(operations) != SUCCESS) { 
        APASS_LOG_ERROR_F(Elements::Operation, "Init failed!"); 
        return FAILED; 
    }
    // op执行排序
    if (SortOps() != SUCCESS) { 
        APASS_LOG_ERROR_F(Elements::Operation, "SortOps failed!"); 
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
    PrintOpList(newOperations_);
    function_.SetStackWorkespaceSize(workspaceOffset);
    return SUCCESS;
}

// UpdateRemainOpBufId函数不能直接用
Status OoOScheduler::UpdateMemId(int oldMemId, int newMemId) {
    if (bufRefCount.find(oldMemId) == bufRefCount.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "bufRefCount cannot find Tensor[%d]", oldMemId);
        return FAILED;
    }
    bufRefCount[newMemId] = 0;
    for (auto &issue : issueEntries) {
        if (issue->isRetired) {
            continue;
        }
        for (auto &memId : issue->reqMemIds) {
            if (memId == oldMemId) {
                memId = newMemId;
                bufRefCount[oldMemId] -= 1;
                bufRefCount[newMemId] += 1;
            }
        }
        for (auto &outTensor : issue->tileOp.GetOOperands()) {
            if (outTensor->memoryrange.memId == oldMemId) {
                outTensor->memoryrange.memId = newMemId;
            }
        }
    }
    if (bufRefCount[oldMemId] != 0) {
        APASS_LOG_ERROR_F(Elements::Tensor, "oldMemId %d bufRefCount is not 0, UpdateMemId failed.", oldMemId);
        return FAILED;
    }
    return SUCCESS;
}

void OoOScheduler::UpdateMoveOpAttr(Operation &moveOp, Operation &occupyOp) {
    if (moveOp.GetOpcode() == Opcode::OP_COPY_IN && occupyOp.GetOpcode() == Opcode::OP_COPY_IN) {
        moveOp.SetOpAttribute(occupyOp.GetOpAttribute());
        moveOp.inParamLocation_ = occupyOp.inParamLocation_;
    } else if (moveOp.GetOpcode() == Opcode::OP_ADDS) {
        moveOp.SetAttr(OpAttributeKey::scalar, Element(DataType::DT_UINT64, 0));
    }
}

IssueEntryPtr OoOScheduler::ProcessMoveOp(Operation &moveOp, Operation &occupyOp, int oldMemId, int newMemId) {
    UpdateMoveOpAttr(moveOp, occupyOp);
    IssueEntryPtr moveIssue = std::make_shared<IssueEntry>(moveOp, issueId);
    issueEntryMap[issueId++] = moveIssue;
    moveIssue->reqMemIds = {oldMemId, newMemId};
    moveIssue->isRetired = true;
    APASS_LOG_DEBUG_F(Elements::Operation, "Add MOVEOP: %s.", moveIssue->GetOpInfo());
    return moveIssue;
}

Status OoOScheduler::GenRearrangeCopyOp(MemoryType memType, int oldMemId, int &newMemId) {
    if (memType != MemoryType::MEM_L1 && memType != MemoryType::MEM_UB) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Unexpected rearrange tensor memory type found, GenRearrangeCopyOp failed.");
        return FAILED;
    }
    Opcode moveOpcode = memType == MemoryType::MEM_L1 ? Opcode::OP_COPY_IN : Opcode::OP_ADDS;
    auto &occupyOp = tensorOccupyMap[memType][oldMemId]->tileOp;
    LogicalTensorPtr moveFromTensor{nullptr};
    for (auto outTensorPtr : occupyOp.GetOOperands()) {
        if (outTensorPtr->memoryrange.memId == oldMemId) {
            moveFromTensor = outTensorPtr;
            break;
        }
    }
    if (moveFromTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find tensor(memId: %d) according to tensorOccupyMap, GenRearrangeCopyOp failed", oldMemId);
        return FAILED;
    }
    LogicalTensorPtr moveToTensor = std::make_shared<LogicalTensor>(function_, moveFromTensor->Datatype(), moveFromTensor->shape);
    // 给moveToTensor分配memId和创建新的localbuffer
    moveToTensor->SetAttr(OpAttributeKey::needAlloc, true);
    if (UpdateTensorAttr(moveToTensor, memType, moveFromTensor, oldMemId) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "GenRearrangeCopyOp failed at UpdateTensorAttr.");
        return FAILED;
    }
    newMemId = moveToTensor->memoryrange.memId;
    auto &moveOp = function_.AddRawOperation(moveOpcode, {moveFromTensor}, {moveToTensor});
    newOperations_.push_back(&moveOp);
    // UpdateMoveOpAttr & 创建moveop的issueEntry
    auto moveIssuePtr = ProcessMoveOp(moveOp, occupyOp, oldMemId, newMemId);
    tensorOccupyMap[memType][newMemId] = moveIssuePtr;
    // 更新moveIssue的相关信息
    auto occupyIssuePtr = tensorOccupyMap[memType][oldMemId];
    moveIssuePtr->predecessors.insert(occupyIssuePtr->id);
    occupyIssuePtr->successors.insert(moveIssuePtr->id);
    // 找出moveFromTensor的所有consumer中未执行的, 并改变图的连接
    UpdateReloadIssueDepend(moveIssuePtr, occupyIssuePtr, oldMemId);
    // 更新memId
    if (UpdateMemId(oldMemId, newMemId) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "GenRearrangeCopyOp failed at UpdateMemId.");
        return FAILED;
    }
    // Free oldMemId
    bufferManagerMap[memType].Free(oldMemId);
    if (oooCheck.doHealthCheck) {
        UpdateBufferUsage(memType, oldMemId, true);
    }
    localBufferMap[oldMemId]->retireCycle = clock;
    if (tensorOccupyMap[memType].erase(oldMemId) == 0) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Erase tensor[%d] failed", oldMemId);
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::RearrangeBuffers(IssueEntryPtr issue, bool isGenSpillStage) {
    LocalBufferPtr allocBuffer = localBufferMap[issue->reqMemIds[0]];
    BufferPool &bufferManager = bufferManagerMap[allocBuffer->memType];
    auto rearrangeScheme = GetRearrangeScheme(bufferManager, allocBuffer->size);
    if (rearrangeScheme.cost == INT_MAX) {
        APASS_LOG_ERROR_F(Elements::Operation, "RearrangeBuffers failed at GetRearrangeScheme.");
        return FAILED;
    }
    // 修改tensor对应的localbuffer
    for (auto &[memId, offset] : rearrangeScheme.orderedMoveTo) {
        auto targetBufferPtr = localBufferMap[memId];
        if (rearrangeScheme.moveFrom[memId] != targetBufferPtr->start ||
            rearrangeScheme.memSizeMap[memId] != targetBufferPtr->size) {
            APASS_LOG_ERROR_F(Elements::Tensor, "MemId %d localBuffer and rearrangeScheme range donot match, RearrangeBuffers failed.", memId);
            return FAILED;
        }
        IssueEntryPtr occupyIssuePtr = GetSpillIssue(issue, memId, isGenSpillStage);
        if (occupyIssuePtr == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "OccupyIssue is nullptr, RearrangeBuffers failed");
            return FAILED;
        }
        if (occupyIssuePtr->tileOp.GetOpcode() == Opcode::OP_VIEW || occupyIssuePtr->tileOp.GetOpcode() == Opcode::OP_ASSEMBLE) {
            APASS_LOG_ERROR_F(Elements::Operation, "Target rearrange tensor(memId: %d)'s occupy op is %d %s, RearrangeBuffers failed.",
                memId, issue->tileOp.GetOpMagic(), issue->tileOp.GetOpcodeStr().c_str());
            return FAILED;
        }
        // GenSpillStage阶段的内存整理不需要插入搬运节点
        // ScheduleMainLoop阶段如果是alloc占有的tensor不需要插入搬运节点
        if (isGenSpillStage || occupyIssuePtr->tileOp.GetOpcodeStr().find("ALLOC") != std::string::npos) {
            if (bufferManager.ModifyBufferRange(targetBufferPtr, offset) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "RearrangeBuffers failed at ModifyBufferRange.");
                return FAILED;
            }
        } else {
            int newMemId = INT_MAX;
            if (GenRearrangeCopyOp(allocBuffer->memType, memId, newMemId) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "RearrangeBuffers failed at GenRearrangeCopyOp.");
                return FAILED;
            }
            // 更新moveToTensor的localbuffer和bufferslice range
            auto moveToBufferPtr = localBufferMap[newMemId];
            if (bufferManager.ModifyBufferRange(moveToBufferPtr, offset) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Tensor, "RearrangeBuffers failed at ModifyBufferRange.");
                return FAILED;
            }
            if (oooCheck.doHealthCheck) {
                UpdateBufferUsage(allocBuffer->memType, newMemId, false);
            }
            tensorOccupyMap[allocBuffer->memType][newMemId]->tileOp.GetOOperands()[0]->memoryrange =
                TileRange(offset, offset + moveToBufferPtr->size, newMemId);
            localBufferMap[newMemId]->startCycle = clock;
        }
    }
    return SUCCESS;
}
} // namespace npu::tile_fwk