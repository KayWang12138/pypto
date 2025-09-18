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

namespace npu::tile_fwk {

constexpr int64_t MAX_L0A_SIZE = 64 * 1024;
constexpr int64_t MAX_L0C_SIZE = 128 * 1024;
constexpr int64_t MAX_BT_SIZE = 1 * 1024;
constexpr int64_t MAX_FIX_SIZE = 1 * 1024;
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

const char* IssueEntry::GetOpInfo() {
    return (tileOp.GetOpcodeStr() + "[" + std::to_string(tileOp.GetOpMagic()) + "]").c_str();
}

Status OoOScheduler::PrintSpillFailedInfo(int currPc) {
    auto allocIssue = issueEntries[currPc];
    ALOG_ERROR_F("======== OoO Spill failed info ===========");
    ALOG_ERROR_F("Spill failed memoryType: %s.", 
        MemoryTypeToString(localBufferMap[allocIssue->reqMemIds[0]]->memType).c_str());
    if (localBufferMap.find(allocIssue->reqMemIds[0]) != localBufferMap.end()) {
        ALOG_ERROR_F("%s alloc buffer size: %lu.", allocIssue->GetOpInfo(), 
            localBufferMap[allocIssue->reqMemIds[0]]->size);
    }
    auto bufferSlices = bufferManagerMap[localBufferMap[allocIssue->reqMemIds[0]]->memType].GetBufferSlices();
    for (auto memId : bufferSlices) {
        size_t bufLastWriteTime = currPc;
        if (!GetBufLastWriteTime(memId, bufLastWriteTime)) {
            ALOG_ERROR_F("Cannot find spill Tensor[%d] last write time.", memId);
            return FAILED;
        }
        auto occupyIssue = issueEntries[bufLastWriteTime];
        ALOG_ERROR_F("%s, range[%lu, %lu], Tensor[%d] size: %lu", occupyIssue->GetOpInfo(), 
            localBufferMap[memId]->start, localBufferMap[memId]->end, memId, localBufferMap[memId]->size);
    }
    return SUCCESS;
}

void OoOScheduler::PrintSpillFailedInfo(IssueEntryPtr allocIssue, MemoryType bufferType) {
    ALOG_ERROR_F("======== OoO Spill failed info ===========");
    ALOG_ERROR_F("Spill failed memoryType: %s.", MemoryTypeToString(bufferType).c_str());
    if (localBufferMap.find(allocIssue->reqMemIds[0]) != localBufferMap.end()) {
        ALOG_ERROR_F("%s alloc buffer size: %lu.", allocIssue->GetOpInfo(), 
            localBufferMap[allocIssue->reqMemIds[0]]->size);
    }
    if (tensorOccupyMap.find(bufferType) != tensorOccupyMap.end()) {
        for (auto occupyIssue : tensorOccupyMap[bufferType]) {
            ALOG_ERROR_F("%s, range[%lu, %lu], Tensor[%d] size: %lu", occupyIssue.second->GetOpInfo(),
                localBufferMap[occupyIssue.first]->start, localBufferMap[occupyIssue.first]->end, 
                occupyIssue.first, localBufferMap[occupyIssue.first]->size);
        }
    }
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

void OoOScheduler::PrintDependenciesAndRelations() {
    for (const auto &issue : issueEntries) {
        if (issue->tileOp.GetBoolAttribute(OpAttributeKey::dontTouch)) {
            continue;
        }
        ALOG_DEBUG_F("%s, latency: %d.", issue->GetOpInfo(), issue->tileOp.GetLatency());
        for (const auto &preId : issue->predecessors) {
            auto pre = issueEntryMap[preId];
            ALOG_DEBUG_F("    |--- Predecessors:");
            ALOG_DEBUG_F("        |--- %s", pre->GetOpInfo());
        }
        for (const auto &succId : issue->successors) {
            auto successor = issueEntryMap[succId];
            ALOG_DEBUG_F("    |--- Successors:");
            ALOG_DEBUG_F("        |--- %s", successor->GetOpInfo());
        }
        ALOG_DEBUG_F("\n");
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
        ALOG_ERROR_F("bufRefCount cannot find Tensor[%d].", memId);
        return FAILED;
    }
    bufRefCount[memId]--;
    if (bufRefCount[memId] < 0) {
        ALOG_ERROR_F("Tensor[%d] bufRefCount cannot less than 0.", memId);
        return FAILED;
    }
    return SUCCESS;
}

void OoOScheduler::PrintOpList(std::vector<Operation *> operations) {
    ALOG_DEBUG_F("==================== OP_LIST =====================");
    for (auto &op : operations) {
        if (!op->oOperand.empty()) {
            bool needAlloc = false;
            op->oOperand[0]->GetAttr(OpAttributeKey::needAlloc, needAlloc);
            ALOG_DEBUG_F("%s[%d], range[%zu, %zu], needAlloc: %d", op->GetOpcodeStr().c_str(), op->GetOpMagic(),
                op->oOperand[0]->memorymap[op->GetSubgraphID()].start,
                op->oOperand[0]->memorymap[op->GetSubgraphID()].end, static_cast<int>(needAlloc));
        } else {
            ALOG_INFO_F("%s[%d]", op->GetOpcodeStr(), op->GetOpMagic()); 
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
            ALOG_ERROR_F("Unexecuted op: %s", issue->GetOpInfo()); 
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

Status OoOScheduler::SpillOnBlock(std::vector<Operation *> &newOperations) {
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
        ALOG_ERROR_F("Buffer[L0A/B/C] is Full. Please check tile shape and OOO spill failed info."); 
        return FAILED; 
    }
    if (GenBufferSpill(allocIssueQueue[spillMemType].Front(), spillMemType, newOperations) != SUCCESS) {
        ALOG_ERROR_F("GenBufferSpill failed."); 
        return FAILED; 
    }
    return SUCCESS;
}

Status OoOScheduler::AllocTensorMemRange(IssueEntryPtr issue) {
    for (auto& outTensor : issue->tileOp.GetOOperands()) {
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
        ALOG_DEBUG_F("REALLOC Tensor[%u] %s --> %s. ", memId, tensorOccupyMap[memType][memId]->GetOpInfo(), 
            issue->GetOpInfo());
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

        newOperations.emplace_back(&(issue->tileOp));
        if (nextCycle == -1 || nextCycle > pipe.curOpRetireCycle) {
            nextCycle = pipe.curOpRetireCycle;
        }
        if (AllocTensorMemRange(issue) != SUCCESS) {
            ALOG_ERROR_F("AllocTensorMemRange failed.");
            return FAILED;
        }
        ALOG_DEBUG_F("Insert: %s", issue->GetOpInfo());
    }
    return SUCCESS;
}

Status OoOScheduler::ExecuteAllocIssue(std::vector<Operation *>& newOperations, uint64_t &commitCnt,
    MemoryType memType, IssueQueue &pipe) {
    bool canAlloc = true;
    while (canAlloc) {
        if (pipe.Empty()) {
            canAlloc = false;
            break;
        }
        IssueEntryPtr issue = pipe.Front();
        if (!bufferManagerMap[memType].IsFull(localBufferMap[issue->reqMemIds[0]])) {
            ALOG_DEBUG_F("ALLOCATE: %s", issue->GetOpInfo());
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
            ALOG_DEBUG_F("Insert: %s", issue->GetOpInfo());
            pipe.PopFront();
            if (RetireOpAndAwakeSucc(issue, commitCnt) != SUCCESS) { 
                ALOG_ERROR_F("RetireOpAndAwakeSucc failed."); 
                return FAILED; 
            }
        } else {
            canAlloc = false;
            break;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::BufferAllocStage(std::vector<Operation *>& newOperations, uint64_t &commitCnt) {
    for (auto& [memType, pipe] : allocIssueQueue) {
        if (pipe.Empty()) {
            continue;
        }
        // 不断按顺序执行alloc指令，直到buffer被占满为止。
        if (ExecuteAllocIssue(newOperations, commitCnt, memType, pipe) != SUCCESS) {
            ALOG_ERROR_F("ExecuteAllocIssue failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::FreeBuffer(IssueEntryPtr issue) {
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

Status OoOScheduler::RetireOpAndAwakeSucc(IssueEntryPtr issue, uint64_t& commitCnt) {
    commitCnt++;
    issue->isRetired = true;
    if (FreeBuffer(issue) != SUCCESS) {
        ALOG_ERROR_F("FreeBuffer failed.");
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
            issueQueues[succ->type].Insert(succ, succ->execOrder);
            ALOG_DEBUG_F("    Wakeup: %s, execOrder: %d", succ->GetOpInfo(), succ->execOrder);
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
            ALOG_DEBUG_F("EXECUTE END: %s", issue->GetOpInfo());
            if (RetireOpAndAwakeSucc(issue, commitCnt) != SUCCESS) { 
                ALOG_ERROR_F("RetireOpAndAwakeSucc failed!"); 
                return FAILED; 
            }
        } else {
            ALOG_DEBUG_F("EXECUTING[%ld]: %s", pipe.curOpRetireCycle, pipe.curIssue->GetOpInfo());
            if (nextCycle == -1 || nextCycle > pipe.curOpRetireCycle) {
                nextCycle = pipe.curOpRetireCycle;
            }
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

Status OoOScheduler::ScheduleMainLoop(std::vector<Operation *> &newOperations) {
    LaunchReadyIssue();
    uint64_t commitCnt = 0; // 当前已提交的issue数量
    bool isAllRetired = false;
    while (!isAllRetired) {
        int nextCycle = -1;
        ALOG_DEBUG_F("\n clock: %d", clock);
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
        if (numTotalIssues == commitCnt && nextCycle == -1) { 
            isAllRetired = true; 
            break; 
        }
        // 如果nextCycle为-1，说明每个pipe都处于idle的状态，判断出现阻塞。需要spill调整内存
        if (nextCycle == -1) {
            if (SpillOnBlock(newOperations) != SUCCESS) {
                ALOG_ERROR_F("SpillOnBlock failed.");
                return FAILED;
            }
        } else { 
            clock = nextCycle; 
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

void OoOScheduler::UpdateAllocMap(IssueEntryPtr issue, std::map<int, IssueEntryPtr> &tensorAllocMap) {
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
        UpdateAllocMap(issue, tensorAllocMap);
    }
    for (auto tensorAlloc : tensorAllocMap) {
        if (!tensorAlloc.second->isAlloc) {
            ALOG_ERROR_F("%s Tensor[%d] is missing Alloc.", tensorAlloc.second->GetOpInfo(), tensorAlloc.first);
            return FAILED;
        }
    }
    return SUCCESS;
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
    } else {
        localBufferMap[memId]->size =
            std::max(localBufferMap[memId]->size, ShapeCeilAlign(oOperand->GetShape(), oOperand->Datatype()));
    }
    return SUCCESS;
}

void OoOScheduler::AddDependencies(
    IssueEntryPtr issue, std::map<int, IssueEntryPtr> lastWriteOpMap, LogicalTensors tensors) {
    for (auto &tensor : tensors) {
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
}

Status OoOScheduler::InitDependencies() {
    bufRefCount.clear();
    std::map<int, IssueEntryPtr> lastWriteOpMap;
    for (const auto &issue : issueEntries) {
        issue->Clear();
        // 仅检测 RAW 和 WAW，不检测 WAR -->SSA
        // RAW
        AddDependencies(issue, lastWriteOpMap, issue->tileOp.GetIOperands());

        // WAW
        AddDependencies(issue, lastWriteOpMap, issue->tileOp.GetOOperands());

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

void OoOScheduler::CalcBufferSize(LogicalTensors tensors, std::map<MemoryType, int64_t> &bufferSize, std::set<int> &memIdMap) {
    for (auto tensor : tensors) {
        if (memIdMap.find(tensor->memorymap[subGraphID].memId) == memIdMap.end()) {
            bufferSize[tensor->GetMemoryTypeOriginal()] += tensor->GetDataSize();
            memIdMap.insert(tensor->memorymap[subGraphID].memId);
        }
    }
}

Status OoOScheduler::CheckOpBufferSize(Operation *op) {
    std::map<MemoryType, int64_t> bufferSize;
    std::set<int> memIdMap;
    CalcBufferSize(op->GetIOperands(), bufferSize, memIdMap);
    CalcBufferSize(op->GetOOperands(), bufferSize, memIdMap);
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
    inChipMemorySize.insert({MemoryType::MEM_UB, 
        PassConfigManager::Instance().GetPlatformConfig().GetMemoryLimit(MemoryType::MEM_UB)});
    inChipMemorySize.insert({MemoryType::MEM_L1, 
        PassConfigManager::Instance().GetPlatformConfig().GetMemoryLimit(MemoryType::MEM_L1)});
    inChipMemorySize.insert({MemoryType::MEM_L0B, 
        PassConfigManager::Instance().GetPlatformConfig().GetMemoryLimit(MemoryType::MEM_L0B)});

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
        issueEntryMap[issueId++] = issue;
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

Status OoOScheduler::Schedule(const std::vector<Operation *> &operations, std::vector<Operation *> &newOperations) {
    if (operations.empty()) {
        return SUCCESS;
    }
    PrintOpList(operations);
    subGraphID = operations.front()->GetSubgraphID();
    if (Init(operations) != SUCCESS) { 
        ALOG_ERROR_F("Init failed!"); 
        return FAILED; 
    }
    // op执行排序
    if (SortOps() != SUCCESS) { 
        ALOG_ERROR_F("SortOps failed!"); 
        return FAILED; 
    }
    // 生成spill指令
    if (GenSpillSchedule() != SUCCESS) { 
        ALOG_ERROR_F("GenSpillSchedule failed!"); 
        return FAILED; 
    }
    // 模拟调度
    if (ScheduleMainLoop(newOperations) != SUCCESS) {
        ALOG_ERROR_F("ScheduleMainLoop failed"); 
        return FAILED;
    }
    if (CheckAndUpdateLifecycle() != SUCCESS) {
        ALOG_ERROR_F("CheckAndUpdateLifecycle failed.");
        return FAILED;
    }
    PrintOpList(newOperations);
    function_.SetStackWorkespaceSize(workspaceOffset);
    return SUCCESS;
}
} // namespace npu::tile_fwk