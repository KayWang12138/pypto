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
 * \file insert_sync.cpp
 * \brief
 */

#include "passes/block_graph_pass/insert_sync.h"
#include <thread>

namespace npu {
namespace tile_fwk {
Status RangeSearchTree::ProcessTreeNode(
    const Interval &interval, IntervalTreeNode *currPtr, std::vector<IntervalTreeNode *> &intervalStack) {
    int start = currPtr->interval.start;
    if (interval.start < start) {
        if (currPtr->left != nullptr) {
            intervalStack.push_back(currPtr->left);
            return SUCCESS;
        }
        currPtr->left = new IntervalTreeNode(interval);
        if (currPtr->left == nullptr) {
            ALOG_ERROR_F("New created left tree node is nullptr, ProcessTreeNode failed!");
            return FAILED;
        }
        return SUCCESS;
    }
    if (currPtr->right != nullptr) {
        intervalStack.push_back(currPtr->right);
        return SUCCESS;
    }
    currPtr->right = new IntervalTreeNode(interval);
    if (currPtr->right == nullptr) {
        ALOG_ERROR_F("New created right tree node is nullptr, ProcessTreeNode failed!");
        return FAILED;
    }
    return SUCCESS;
}

Status RangeSearchTree::InsertInterval(const Interval &interval) {
    std::vector<IntervalTreeNode*> intervalStack;
    if (treeRoot == nullptr) {
        treeRoot = new IntervalTreeNode(interval);
        if (treeRoot == nullptr) { ALOG_ERROR_F("TreeRoot is nullptr, InsertInterval failed!"); return FAILED; }
        return SUCCESS;
    }
    intervalStack.push_back(treeRoot);
    while (intervalStack.size() > 0) {
        IntervalTreeNode *currPtr = intervalStack.back();
        intervalStack.pop_back();
        if (ProcessTreeNode(interval, currPtr, intervalStack) != SUCCESS) { ALOG_ERROR_F("InsertInterval failed at function ProcessTreeNode!"); return FAILED; }
        if (currPtr->max < interval.end) {
            currPtr->max = interval.end;
        }
    }
    return SUCCESS;
}

void RangeSearchTree::OverlapSearch(const Interval &interval, std::set<int> &result) {
    std::vector<IntervalTreeNode*> intervalStack;
    intervalStack.push_back(treeRoot);

    while (intervalStack.size() > 0) {
        IntervalTreeNode *currPtr = intervalStack.back();
        intervalStack.pop_back();
        if (currPtr == nullptr) {
            continue;
        }
        if (interval.start <= currPtr->interval.end && interval.end >= currPtr->interval.start) {
            result.insert(currPtr->interval.idx);
        }
        if (currPtr->left != nullptr && currPtr->left->max >= interval.start) {
            intervalStack.push_back(currPtr->left);
        }
        intervalStack.push_back(currPtr->right);
    }
}

void RangeSearchTree::FreeTree() {
    std::vector<IntervalTreeNode *> intervalStack;
    intervalStack.push_back(treeRoot);
    while (intervalStack.size() > 0) {
        IntervalTreeNode *currPtr = intervalStack.back();
        intervalStack.pop_back();
        if (currPtr == nullptr) {
            continue;
        }
        intervalStack.push_back(currPtr->left);
        intervalStack.push_back(currPtr->right);
        delete currPtr;
    }
}

void RangeSearchTree::Insert(int left, int right, int idx) {
    Interval interval(left, right, idx);
    InsertInterval(interval);
}

std::set<int> RangeSearchTree::GetCovered(int left, int right) {
    Interval givenInterval(left, right, 0);
    std::set<int> overlappingIdx;
    OverlapSearch(givenInterval, overlappingIdx);
    return overlappingIdx;
}

void DataDependencySearcher::CheckWAWSearchTree(Operation *opWait, std::set<int> &res) {
    for (size_t outIdx = 0; outIdx < opWait->GetOOperands().size(); outIdx++) {
        MemoryType currMemoryType = opWait->GetOOperands()[outIdx]->GetMemoryTypeOriginal();
        if (wawSearchTree_.count(currMemoryType) > 0) {
            TileRange rg = opWait->GetOOperands()[outIdx]->memorymap[BLOCK_GRAPH_DEFAULT_COLOR];
            std::set<int> found = wawSearchTree_[currMemoryType].GetCovered(rg.start, rg.end);
            res.insert(found.begin(), found.end());
        }
    }
}

void DataDependencySearcher::CheckRAWSearchTree(Operation *opWait, std::set<int> &res) {
    for (size_t inIdx = 0; inIdx < opWait->GetIOperands().size(); inIdx++) {
        MemoryType readMemoryType = opWait->GetIOperands()[inIdx]->GetMemoryTypeOriginal();
        int readDDRmemId = opWait->GetIOperands()[inIdx]->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId;
        if (readDDRmemId != -1 && writeDdrMemMap.count(readDDRmemId) > 0) {
            std::set<int> found = writeDdrMemMap[readDDRmemId];
            res.insert(found.begin(), found.end());
        }
        if (rawSearchTree_.count(readMemoryType) > 0) {
            TileRange rg = opWait->GetIOperands()[inIdx]->memorymap[BLOCK_GRAPH_DEFAULT_COLOR];
            std::set<int> found = rawSearchTree_[readMemoryType].GetCovered(rg.start, rg.end);
            res.insert(found.begin(), found.end());
        }
    }
}

void DataDependencySearcher::CheckWARSearchTree(Operation *opWait, std::set<int> &res) {
    for (size_t outIdx = 0; outIdx < opWait->GetOOperands().size(); outIdx++) {
        MemoryType writeMemoryType = opWait->GetOOperands()[outIdx]->GetMemoryTypeOriginal();
        int writeDDRmemId = opWait->GetOOperands()[outIdx]->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId;
        if (writeDDRmemId != -1 && readDdrMemMap.count(writeDDRmemId) > 0) {
            std::set<int> found = readDdrMemMap[writeDDRmemId];
            res.insert(found.begin(), found.end());
        }
        if (warSearchTree_.count(writeMemoryType) > 0) {
            TileRange rg = opWait->GetOOperands()[outIdx]->memorymap[BLOCK_GRAPH_DEFAULT_COLOR];
            std::set<int> found = warSearchTree_[writeMemoryType].GetCovered(rg.start, rg.end);
            res.insert(found.begin(), found.end());
        }
    }
}

std::set<int> DataDependencySearcher::Find(Operation *opWait) {
    std::set<int> res;

    std::string opStr = opWait->GetOpcodeStr();
    // check WAW
    CheckWAWSearchTree(opWait, res);
    // check RAW
    CheckRAWSearchTree(opWait, res);
    // check WAR
    CheckWARSearchTree(opWait, res);
    return res;
}

void DataDependencySearcher::InsertWAWSearchTree(const Operation *opSet, int idx) {
    for (size_t outIdx = 0; outIdx < opSet->GetOOperands().size(); outIdx++) {
        MemoryType prevMemoryType = opSet->GetOOperands()[outIdx]->GetMemoryTypeOriginal();
        if (wawSearchTree_.count(prevMemoryType) == 0) {
            wawSearchTree_[prevMemoryType] = RangeSearchTree();
        }
        TileRange rg = opSet->GetOOperands()[outIdx]->memorymap[BLOCK_GRAPH_DEFAULT_COLOR];
        wawSearchTree_[prevMemoryType].Insert(rg.start, rg.end, idx);
    }
}

void DataDependencySearcher::InsertRAWSearchTree(const Operation *opSet, int idx) {
    for (size_t outIdx = 0; outIdx < opSet->GetOOperands().size(); outIdx++) {
        MemoryType writeMemoryType = opSet->GetOOperands()[outIdx]->GetMemoryTypeOriginal();
        int writeDDRmemId = opSet->GetOOperands()[outIdx]->GetMemoryTypeOriginal();
        if (writeDDRmemId != -1) {
            if (writeDdrMemMap.count(writeDDRmemId) == 0) {
                writeDdrMemMap[writeDDRmemId] = std::set<int>{};
            }
            writeDdrMemMap[writeDDRmemId].insert(idx);
        }
        if (rawSearchTree_.count(writeMemoryType) == 0) {
            rawSearchTree_[writeMemoryType] = RangeSearchTree();
        }
        TileRange rg = opSet->GetOOperands()[outIdx]->memorymap[BLOCK_GRAPH_DEFAULT_COLOR];
        rawSearchTree_[writeMemoryType].Insert(rg.start, rg.end,idx);
    }
}

void DataDependencySearcher::InsertWARSearchTree(const Operation *opSet, int idx) {
    for (size_t inIdx = 0; inIdx < opSet->GetIOperands().size(); inIdx++) {
        MemoryType readMemoryType = opSet->GetIOperands()[inIdx]->GetMemoryTypeOriginal();
        int readDDRmemId = opSet->GetIOperands()[inIdx]->GetMemoryTypeOriginal();
        if (readDDRmemId != -1) {
            if (readDdrMemMap.count(readDDRmemId) == 0) {
                readDdrMemMap[readDDRmemId] = std::set<int>{};
            }
            readDdrMemMap[readDDRmemId].insert(idx);
        }
        if (warSearchTree_.count(readMemoryType) == 0) {
            warSearchTree_[readMemoryType] = RangeSearchTree();
        }
        TileRange rg = opSet->GetIOperands()[inIdx]->memorymap[BLOCK_GRAPH_DEFAULT_COLOR];
        warSearchTree_[readMemoryType].Insert(rg.start, rg.end,idx);
    }
}

void DataDependencySearcher::Insert(const Operation *opSet, int idx) {
    InsertWAWSearchTree(opSet, idx);
    InsertRAWSearchTree(opSet, idx);
    InsertWARSearchTree(opSet, idx);
}

Status PipeSync::InsertSync(Function &function, std::vector<Operation *> &syncedOpLog) {
    std::vector<IndexOp> synced;
    std::vector<Operation *> opLogPtr(function.Operations().DuplicatedOpList());
    uint64_t idxInput = 0;
    for (const auto &op : opLogPtr) {
        ALOG_DEBUG_F("Input operation %d %d: %s", idxInput, op->GetOpMagic(), op->GetOpcodeStr().c_str());
        idxInput++;
    }
    if (PipeDispatch(opLogPtr, synced) != SUCCESS) { ALOG_ERROR_F("InsertSync failed at function PipeDispatch!"); return FAILED; }

    if (IssueOp(function, opLogPtr, synced) != SUCCESS) { ALOG_ERROR_F("InsertSync failed at function IssueOp!"); return FAILED; }

    for (auto &log : synced) {
        if (log.second->GetOpcode() != Opcode::OP_NOP) {
            syncedOpLog.push_back(log.second);
        }
    }
    return SUCCESS;
}

std::string PipeSync::DepOp::DumpDepOp(std::vector<Operation *> opLog) {
    std::stringstream ss;
    ss << "idx: " << idx << " opmagic: " << opLog[idx]->GetOpMagic() << ", ";
    if (!opLog.empty()) {
        ss << opLog[idx]->GetOpcodeStr() << ", ";
    }
    ss << "setPipe: {";
    for (auto i : setPipe) {
        if (opLog.empty()) {
            ss << opLog[i]->GetOpMagic() << ", ";
            continue;
        }
        ss << opLog[i]->GetOpMagic() << " " << opLog[i]->GetOpcodeStr() << ", ";
    }
    ss << "}, waitPipe: {";
    for (auto i : waitPipe) {
        if (opLog.empty()) {
            ss << opLog[i]->GetOpMagic() << ", ";
            continue;
        }
        ss << opLog[i]->GetOpMagic() << " " << opLog[i]->GetOpcodeStr() << ", ";
    }
    ss << "}";
    return ss.str();
}

std::string PipeSync::IssueQueue::DumpIssueQueue(std::vector<Operation *> opLogPtr) {
    std::stringstream ss;
    ss << "pipe type: " << PipeTypeName(selfPipeCore.pipe) << ", Op in this pipe: {";
    for (auto op : ops) {
        ss << opLogPtr[op]->GetOpMagic() << " " << opLogPtr[op]->GetOpcodeStr() << ", ";
    }
    ss << "}";
    return ss.str();
}

std::string PipeSync::PipeDepInfo::DumpPipeDepInfo() {
    std::stringstream ss;
    ss << "    wait idx: " << waitIdx << "\n";
    ss << "    setPipes:" << "\n";
    for (auto pair : setPipes) {
        ss << "        pipetype: " << PipeTypeName(pair.first.pipe) << "  opidx: " << pair.second << "\n";
    }
    return ss.str();
}

std::string PipeSync::DumpLatestPipeDepMap() {
    std::stringstream ss;
    for (auto pair : latestPipeDep_) {
        ss << "current pipe type: " << PipeTypeName(pair.first.pipe) << "\n";
        ss << pair.second.DumpPipeDepInfo() << "\n";
    }
    return ss.str();
}

std::string PipeSync::PipeSeqName(PipeSeq seq) const{
    switch (seq) {
        case PipeSeq::AIC_MTE2: return "AIC_MTE2";
        case PipeSeq::AIC_MTE1: return "AIC_MTE1";
        case PipeSeq::AIC_M: return "AIC_M";
        case PipeSeq::AIC_FIX: return "AIC_FIX";
        case PipeSeq::AIV_MTE2: return "AIV_MTE2";
        case PipeSeq::AIV_V: return "AIV_V";
        case PipeSeq::AIV_MTE3: return "AIV_MTE3";
        case PipeSeq::AIC_MTE3: return "AIC_MTE3";
        case PipeSeq::AIV_S: return "AIV_S";
        case PipeSeq::AIC_S: return "AIC_S";
        case PipeSeq::PIPE_END: return "PIPE_END";
        default: return "ILLEGAL";
    }
}

std::map<PipeSync::PipeCoreReal, PipeSeq, PipeSync::PipeCoreRealCompare> PipeSync::pipe2Seq = {
    {{PIPE_MTE2, CoreType::AIC}, PipeSeq::AIC_MTE2},
    {{PIPE_MTE1, CoreType::AIC}, PipeSeq::AIC_MTE1},
    {   {PIPE_M, CoreType::AIC},    PipeSeq::AIC_M},
    { {PIPE_FIX, CoreType::AIC},  PipeSeq::AIC_FIX},
    {{PIPE_MTE2, CoreType::AIV}, PipeSeq::AIV_MTE2},
    {   {PIPE_V, CoreType::AIV},    PipeSeq::AIV_V},
    {{PIPE_MTE3, CoreType::AIV}, PipeSeq::AIV_MTE3},
    {{PIPE_MTE3, CoreType::AIC}, PipeSeq::AIC_MTE3},
    {   {PIPE_S, CoreType::AIV},    PipeSeq::AIV_S},
    {   {PIPE_S, CoreType::AIC},    PipeSeq::AIC_S},
};

std::map<PipeSeq, PipeSync::PipeCoreReal> PipeSync::seq2pipe = {
    {PipeSeq::AIC_MTE2, {PIPE_MTE2, CoreType::AIC}},
    {PipeSeq::AIC_MTE1, {PIPE_MTE1, CoreType::AIC}},
    {   PipeSeq::AIC_M,    {PIPE_M, CoreType::AIC}},
    { PipeSeq::AIC_FIX,  {PIPE_FIX, CoreType::AIC}},
    {PipeSeq::AIV_MTE2, {PIPE_MTE2, CoreType::AIV}},
    {   PipeSeq::AIV_V,    {PIPE_V, CoreType::AIV}},
    {PipeSeq::AIV_MTE3, {PIPE_MTE3, CoreType::AIV}},
    {PipeSeq::AIC_MTE3, {PIPE_MTE3, CoreType::AIC}},
    {   PipeSeq::AIV_S,    {PIPE_S, CoreType::AIV}},
    {   PipeSeq::AIC_S,    {PIPE_S, CoreType::AIC}},
};

PipeSeq PipeSync::GetPipeSeq(PipeSync::PipeCoreReal pipe) {
    return pipe2Seq.at(pipe);
}

PipeSync::PipeCoreReal PipeSync::GetPipeFromSeq(PipeSeq seq) {
    return seq2pipe.at(seq);
}

Status PipeSync::AdjustReshapeCfg(TileOpCfg &opcfg, Operation *opptr) {
    if (opptr->GetIOperands().size() < 1 || opptr->GetOOperands().size() < 1) {
        ALOG_ERROR_F("%d RESHAPE op operands size is 0, AdjustOpCfg failed!", opptr->GetOpMagic());
        return FAILED;
    }
    if (opptr->GetIOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
        opptr->GetOOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
        opcfg.pipeIdStart_ = PipeType::PIPE_MTE3;
        opcfg.pipeIdEnd_ = PipeType::PIPE_MTE3;
    }
    return SUCCESS;
}

Status PipeSync::AdjustCopyInCfg(TileOpCfg &opcfg, Operation *opptr) {
    if (opptr->GetOpAttribute() == nullptr) {
        ALOG_ERROR_F("%d COPYIN op attr is nullptr, AdjustOpCfg failed!", opptr->GetOpMagic());
        return FAILED;
    }
    std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(opptr->GetOpAttribute());
    auto dstMemType = attr->GetCopyInAttr().second;
    if (dstMemType == MemoryType::MEM_L1) {
        opcfg.pipeIdStart_ = PipeType::PIPE_MTE2;
        opcfg.pipeIdEnd_ = PipeType::PIPE_MTE2;
        opcfg.coreType_ = CoreType::AIC;
        return SUCCESS;
    }
    if (dstMemType == MemoryType::MEM_UB) {
        opcfg.pipeIdStart_ = PipeType::PIPE_MTE2;
        opcfg.pipeIdEnd_ = PipeType::PIPE_MTE2;
        opcfg.coreType_ = CoreType::AIV;
    }
    return SUCCESS;
}

Status PipeSync::AdjustCopyOutCfg(TileOpCfg &opcfg, Operation *opptr) {
    if (opptr->GetOpAttribute() == nullptr) {
        ALOG_ERROR_F("%d COPYOUT op attr is nullptr, AdjustOpCfg failed!", opptr->GetOpMagic());
        return FAILED;
    }
    std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(opptr->GetOpAttribute());
    auto srcMemType = attr->GetCopyOutAttr().first;
    if (srcMemType == MemoryType::MEM_L0C) {
        opcfg.pipeIdStart_ = PipeType::PIPE_FIX;
        opcfg.pipeIdEnd_ = PipeType::PIPE_FIX;
        opcfg.coreType_ = CoreType::AIC;
        return SUCCESS;
    }
    if (srcMemType == MemoryType::MEM_UB) {
        opcfg.pipeIdStart_ = PipeType::PIPE_MTE3;
        opcfg.pipeIdEnd_ = PipeType::PIPE_MTE3;
        opcfg.coreType_ = CoreType::AIV;
        return SUCCESS;
    }
    if (srcMemType == MemoryType::MEM_L1) {
        opcfg.pipeIdStart_ = PipeType::PIPE_MTE3;
        opcfg.pipeIdEnd_ = PipeType::PIPE_MTE3;
        opcfg.coreType_ = CoreType::AIC;
    }
    return SUCCESS;
}

Status PipeSync::AdjustOpCfg(TileOpCfg &opcfg, Operation *opptr) {
    if (opptr->GetOpcode() == Opcode::OP_RESHAPE) {
        if (AdjustReshapeCfg(opcfg, opptr) != SUCCESS) {
            ALOG_ERROR_F("AdjustReshapeCfg failed!");
            return FAILED;
        }
    }
    if (opptr->GetOpcode() == Opcode::OP_COPY_IN) {
        if (AdjustCopyInCfg(opcfg, opptr) != SUCCESS) {
            ALOG_ERROR_F("AdjustCopyInCfg failed!");
            return FAILED;
        }
    }
    if (opptr->GetOpcode() == Opcode::OP_COPY_OUT) {
        if (AdjustCopyOutCfg(opcfg, opptr) != SUCCESS) {
            ALOG_ERROR_F("AdjustCopyOutCfg failed!");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status PipeSync::PipeDispatch(const std::vector<Operation *> opLogPtr, std::vector<IndexOp> &syncedOpLog) {
    DataDependencySearcher dataDependencySearcher;
    for (size_t i = 0; i < opLogPtr.size(); i++) {
        if (opLogPtr[i]->GetOpcodeStr().find("ALLOC") != std::string::npos) { ALOG_ERROR_F("%d ALLOC op should not appear in InsertSync, PipeDispatch failed!", opLogPtr[i]->GetOpMagic()); return FAILED; }
        maxOpMagic = std::max(maxOpMagic, opLogPtr[i]->GetOpMagic());
        auto opcfg = OpcodeManager::Inst().GetTileOpCfg(opLogPtr[i]->GetOpcode());
        if (AdjustOpCfg(opcfg, opLogPtr[i]) != SUCCESS) { ALOG_ERROR_F("PipeDispatch failed at function AdjustOpCfg!"); return FAILED; }
        DepOp op(i, {opcfg.pipeIdStart_, opcfg.pipeIdEnd_, opcfg.coreType_});
        DepOp &opRef = depOps_.emplace_back(op);
        FindDep(opRef, opLogPtr, i, dataDependencySearcher);
        EnqueueOp(opRef, opLogPtr, syncedOpLog);
    }
    return SUCCESS;
}

void PipeSync::InitIssueQueue() {
    for (int i = 0; i < static_cast<int>(PipeSeq::PIPE_END); i++) {
        issueState_.emplace_back(GetPipeFromSeq(static_cast<PipeSeq>(i)));
    }
}

void PipeSync::EnqueueOp(DepOp &op, const std::vector<Operation *> opLogPtr, std::vector<IndexOp> &syncedOpLog) {
   if (opLogPtr[op.idx]->GetOpcode() == Opcode::OP_ASSEMBLE || opLogPtr[op.idx]->GetOpcode() == Opcode::OP_VIEW) {
        syncedOpLog.emplace_back(std::make_pair(op.idx, opLogPtr[op.idx]));
        return;
    }
    PipeCoreReal opPipeCore(op.selfPipeCore.pipeEnd, op.selfPipeCore.core);
    auto &issueQ = issueState_[static_cast<int>(GetPipeSeq(opPipeCore))];
    issueQ.ops.emplace_back(op.idx);
    op.idxInPipe = issueQ.ops.size() - 1;
    // 若op的pipeStart和pipeEnd不同, 进行记录
    if (op.selfPipeCore.pipeStart != op.selfPipeCore.pipeEnd) {
        PipeCoreReal opPipeCoreEnd(op.selfPipeCore.pipeEnd, op.selfPipeCore.core);
        PipePair pp{opPipeCore, opPipeCoreEnd};
        int opMagic = opLogPtr[op.idx]->GetOpMagic();
        doublePipeOp[pp].emplace_back(opMagic);
    }
}

void PipeSync::RemoveOpDep(DepOp &setOp, DepOp &waitOp) const {
    size_t setOpIdx = setOp.idx;
    size_t waitOpIdx = waitOp.idx;
    std::vector<size_t> newSetDep;
    for (auto ele : setOp.setPipe) {
        if (ele == waitOpIdx) {
            continue;
        }
        newSetDep.emplace_back(ele);
    }
    setOp.setPipe = newSetDep;

    std::vector<size_t> newWaitDep;
    for (auto ele : waitOp.waitPipe) {
        if (ele == setOpIdx) {
            continue;
        }
        newWaitDep.emplace_back(ele);
    }
    waitOp.waitPipe = newWaitDep;
}

Status PipeSync::AddOpDep(DepOp &setOp, DepOp &waitOp) {
    size_t setOpIdx = setOp.idx;
    size_t waitOpIdx = waitOp.idx;

    size_t depWaitIdx = static_cast<size_t>(-1);
    for (auto ele : setOp.setPipe) {
        if (ele == waitOpIdx) { ALOG_ERROR_F("This dependency should not exist, AddOpDep failed!"); return FAILED; }
        PipeCoreReal elePipeCore(depOps_[ele].selfPipeCore.pipeStart, depOps_[ele].selfPipeCore.core);
        PipeCoreReal waitOpPipeCore(depOps_[waitOpIdx].selfPipeCore.pipeStart, depOps_[waitOpIdx].selfPipeCore.core);
        if (elePipeCore == waitOpPipeCore) {
            if (ele  <= waitOpIdx) { ALOG_ERROR_F("New waitidx should less than old, AddOpDep failed!"); return FAILED; }
            depWaitIdx = ele;
            break;
        }
    }
    //同一种pipecore，不需要存记录依赖关系
    if (depWaitIdx != static_cast<size_t>(-1)) {
        RemoveOpDep(setOp, depOps_[depWaitIdx]);
    }
    setOp.setPipe.emplace_back(waitOpIdx);
    waitOp.waitPipe.emplace_back(setOpIdx);
    return SUCCESS;
}

Status PipeSync::AdjustOpDep(DepOp &op, size_t waitOpIdx, IssueQueue &issueQ) {
    //op为靠前的， waitOp为靠后的
    auto &waitOp = depOps_[waitOpIdx];

    if (issueQ.currOp + 1 == issueQ.ops.size()) {
        return SUCCESS;
    }

    auto &nextOpIdx = issueQ.ops[issueQ.currOp + 1];
    auto &nextOp = depOps_[nextOpIdx];
    RemoveOpDep(op, waitOp);
    if (AddOpDep(nextOp, waitOp) != SUCCESS) { ALOG_ERROR_F("AdjustOpDep failed at function AddOpDep!"); return FAILED; }
    return SUCCESS;
}

Status PipeSync::HandleEventID(DepOp &op, IssueQueue &issueQ, IssueNum &issuenum, bool &deadlock, bool &res) {
    bool eventIdOk = true;
    for (auto ele : op.setPipe) {
        if (op.selfPipeCore.pipeEnd == depOps_[ele].selfPipeCore.pipeStart) {
            continue;
        }
        PipeCoreReal currPipeCore(op.selfPipeCore.pipeEnd, op.selfPipeCore.core);
        PipeCoreReal elePipeCore(depOps_[ele].selfPipeCore.pipeStart, depOps_[ele].selfPipeCore.core);
        PipePair pp{currPipeCore, elePipeCore};
        issuenum.maxIssueNum.emplace(pp, GetFreeEventIdQueue(pp).size());
        issuenum.currIssueNum.emplace(pp, 0);

        if (issuenum.currIssueNum[pp] >= issuenum.maxIssueNum[pp]) {
            if (!deadlock) {
                eventIdOk = false;
                break;
            }
            // eventID deadlock, adjust op dependency to release eventID.
            if (AdjustOpDep(op, ele, issueQ) != SUCCESS) {
                ALOG_ERROR_F("HandleEventID failed at function AdjustOpDep!");
                return FAILED;
            }
        }
    }

    deadlock = false;

    if (!eventIdOk) {
        res = false;
        return SUCCESS;
    }
    res = true;
    return SUCCESS;
}

bool PipeSync::CheckIssuedOp(const DepOp &op) {
    // current op will be issued only when all of the waitop are issued
    for (const auto &waitOp : op.waitPipe) {
        if (!depOps_[waitOp].issued) {
            return false;
        }
    }
    return true;
}

Status PipeSync::PopFromQueue(IssueQueue &issueQ, std::vector<size_t> &poped, bool &deadlock) {
    IssueNum issuenum;

    for (uint64_t i = 0; i < MAX_POP; i++) {
        if (issueQ.currOp >= issueQ.ops.size()) {
            break;
        }
        auto &op = depOps_[issueQ.ops[issueQ.currOp]];
        if (op.issued) {
            ALOG_ERROR_F("Try to issue a op which is already issued, PopFromQueue failed!");
            return FAILED;
        }
        bool ready = CheckIssuedOp(op);
        bool res = false;
        if (HandleEventID(op, issueQ, issuenum, deadlock, res) != SUCCESS) {
            ALOG_ERROR_F("PopFromQueue failed at function HandleEventID!");
            return FAILED;
        }
        if (!ready || !res) {
            break;
        }
        op.issued = true;
        poped.emplace_back(op.idx);
        for (auto ele : op.setPipe) {
            PipeCoreReal currPipeCore(op.selfPipeCore.pipeEnd, op.selfPipeCore.core);
            PipeCoreReal elePipeCore(depOps_[ele].selfPipeCore.pipeStart, depOps_[ele].selfPipeCore.core);
            auto pp = PipePair{currPipeCore, elePipeCore};
            issuenum.currIssueNum[pp] = issuenum.currIssueNum[pp] + 1;
        }
        issueQ.currOp++;
    }
    return SUCCESS;
}

Status PipeSync::InjectSync(Function &function, std::vector<Operation *> opLogPtr, size_t idx, std::vector<IndexOp> &syncedOpLog) {
    PipeCore currPipe = depOps_[idx].selfPipeCore;

    // serch the waitpipe of current op
    for (const auto &ele : depOps_[idx].waitPipe) {
        PipeCore setPipe = depOps_[ele].selfPipeCore;
        PipeCoreReal setPipeReal(setPipe.pipeEnd, setPipe.core);
        PipeCoreReal currPipeReal(currPipe.pipeStart, currPipe.core);
        int eventId = setWaitPairMap_[{ele, idx}];
        std::vector<std::shared_ptr<LogicalTensor>> input;
        std::vector<std::shared_ptr<LogicalTensor>> output;
        Operation &syncOp = function.AddRawOperation(npu::tile_fwk::Opcode::OP_SYNC_DST, {input}, {output});
        syncOp.opmagic = ++maxOpMagic;
        Operation *syncOpPtr = &syncOp;
        bool res = GenSyncOp(setPipeReal, currPipeReal, eventId, false, syncOpPtr);
        if (!res) {
            syncOpPtr->SetAsDeleted();
            continue;
        }
        // insert wait_flag
        syncedOpLog.emplace_back(std::make_pair(-1, syncOpPtr));
        GetFreeEventIdQueue({setPipeReal, currPipeReal}).push_back(eventId);
    }

    syncedOpLog.emplace_back(std::make_pair(idx, opLogPtr[idx]));
    depOps_[idx].issued = true;

    // insert set_flag
    for (const auto &ele : depOps_[idx].setPipe) {
        PipeCore waitPipe = depOps_[ele].selfPipeCore;
        PipeCoreReal waitPipeReal(waitPipe.pipeStart, waitPipe.core);
        PipeCoreReal currPipeReal(currPipe.pipeEnd, currPipe.core);
        int eventId{0};
        if (GetEventId({currPipeReal, waitPipeReal}, eventId) != SUCCESS) { ALOG_ERROR_F("InjectSync failed at function GetEventId!"); return FAILED; }
        std::vector<std::shared_ptr<LogicalTensor>> input;
        std::vector<std::shared_ptr<LogicalTensor>> output;
        Operation &syncOp = function.AddRawOperation(npu::tile_fwk::Opcode::OP_SYNC_SRC, {input}, {output});
        syncOp.opmagic = ++maxOpMagic;
        Operation *syncOpPtr = &syncOp;
        bool res = GenSyncOp(currPipeReal, waitPipeReal, eventId, true, syncOpPtr);
        if (res) {
            // insert set_flag
            syncedOpLog.emplace_back(std::make_pair(-1, syncOpPtr));
            setWaitPairMap_[{idx, ele}] = eventId;
            continue;
        }
        syncOpPtr->SetAsDeleted();
        setWaitPairMap_[{idx, ele}] = eventId;
    }
    return SUCCESS;
}

int PipeSync::GetMaxEventId(const PipePair &pp) {
    PipePair ppReverse = {pp.second, pp.first};
    auto it1 = doublePipeOp.find(pp);
    auto it2 = doublePipeOp.find(ppReverse);
    if (it1 == doublePipeOp.end() && it2 == doublePipeOp.end()) {
        return EVENT_NUM;
    }
    return EVENT_ID7;
}

Status PipeSync::ProcessDeadLock(uint64_t &eventIdDeadlockEnterTimes, bool &eventIdDeadlock, std::vector<IndexOp> &syncedOpLog) {
    eventIdDeadlockEnterTimes++;
    // eventID deadlock
    if (eventIdDeadlockEnterTimes > DEADLOCK_TIME_THRESHOLD) {
        eventIdDeadlock = true;
    }
    if (RelaxFakeDataDep(syncedOpLog) != SUCCESS) { ALOG_ERROR_F("ProcessDeadLock failed at function RelaxFakeDataDep!"); return FAILED; }
    if (eventIdDeadlockEnterTimes >= EVENTID_DEADLOCK_ENTER_TIME) { ALOG_ERROR_F("Unbreakable deadlock detected, ProcessDeadLock failed!"); return FAILED; }
    return SUCCESS;
}

Status PipeSync::IssueOpPipeSeq(Function &function, std::vector<Operation *> opLogPtr, std::vector<IndexOp> &syncedOpLog, bool &eventIdDeadlock, size_t &issued) {
    for (int i = 0; i < static_cast<int>(PipeSeq::PIPE_END); i++) {
        std::vector<size_t> issuedOps;
        if (PopFromQueue(issueState_[i], issuedOps, eventIdDeadlock) != SUCCESS) {
            ALOG_ERROR_F("IssueOp failed at function PopFromQueue!");
            return FAILED;
        }
        issued += issuedOps.size();
        for (auto idx : issuedOps) {
            if (InjectSync(function, opLogPtr, idx, syncedOpLog) != SUCCESS) {
                ALOG_ERROR_F("IssueOp failed at function InjectSync!");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status PipeSync::IssueSyncOp(Function &function, std::vector<Operation *> opLogPtr, std::vector<IndexOp> &syncedOpLog, size_t &totalIssued, size_t &allIssued) {
    bool eventIdDeadlock = false;
    uint64_t eventIdDeadlockEnterTimes = 0;
    while (totalIssued < allIssued) {
        size_t issued = 0;
        if (IssueOpPipeSeq(function, opLogPtr, syncedOpLog, eventIdDeadlock, issued) != SUCCESS) {
            ALOG_ERROR_F("IssueOp failed at function IssueOpPipeSeq!");
            return FAILED;
        }
        totalIssued += issued;
        // eventIdDeadlockEnterTimes eventIdDeadlock syncedOpLog
        if (issued == 0) {
            if (ProcessDeadLock(eventIdDeadlockEnterTimes, eventIdDeadlock, syncedOpLog) != SUCCESS) {
                ALOG_ERROR_F("IssueOp failed at function ProcessDeadLock!");
                return FAILED;
            }
            continue;
        }
        eventIdDeadlock = false;
        eventIdDeadlockEnterTimes = 0;
    }
    return SUCCESS;
}

Status PipeSync::IssueOp(Function &function, std::vector<Operation *> opLogPtr, std::vector<IndexOp> &syncedOpLog) {
    size_t totalIssued = 0;
    size_t allIssued = 0;
    for (int i = 0; i < static_cast<int>(PipeSeq::PIPE_END); i++) {
        allIssued += issueState_[i].ops.size();
        ALOG_DEBUG_F("Pipe seq %d: %s %s", i, PipeSeqName(static_cast<PipeSeq>(i)).c_str(), issueState_[i].DumpIssueQueue(opLogPtr).c_str());
    }
    if (IssueSyncOp(function, opLogPtr, syncedOpLog, totalIssued, allIssued) != SUCCESS) {
        ALOG_ERROR_F("IssueOp failed with IssueSyncOp!");
        return FAILED;
    }
    if (totalIssued != allIssued) {
        ALOG_ERROR_F("Issue error, IssueOp failed!");
        return FAILED;
    }
    ALOG_DEBUG_F("ALL op issued: %zu", totalIssued);
    return SUCCESS;
}

std::vector<PipeSync::PipePair> PipeSync::dataDepPair = {
    // PIPE_MTE1只有AIC PIPE_V只有AIV PIPE_M只有AIC PIPE_FIX只有AIC
    // PIPE_MTE3->PIPE_MTE2
    {{PIPE_MTE3, CoreType::AIV}, {PIPE_MTE2, CoreType::AIV}},
    {{PIPE_MTE3, CoreType::AIC}, {PIPE_MTE2, CoreType::AIC}},
    // PIPE_MTE2->PIPE_MTE3
    {{PIPE_MTE2, CoreType::AIV}, {PIPE_MTE3, CoreType::AIV}},
    {{PIPE_MTE2, CoreType::AIC}, {PIPE_MTE3, CoreType::AIC}},
    // PIPE_MTE2->PIPE_V
    {{PIPE_MTE2, CoreType::AIV}, {PIPE_V, CoreType::AIV}},
    // PIPE_V->PIPE_MTE2
    {{PIPE_V, CoreType::AIV}, {PIPE_MTE2, CoreType::AIV}},
    // PIPE_MTE3->PIPE_V
    {{PIPE_MTE3, CoreType::AIV}, {PIPE_V, CoreType::AIV}},
    // PIPE_V->PIPE_MTE3
    {{PIPE_V, CoreType::AIV}, {PIPE_MTE3, CoreType::AIV}},
    // PIPE_S->PIPE_V
    {{PIPE_S, CoreType::AIV}, {PIPE_V, CoreType::AIV}},
    // PIPE_V->PIPE_S
    {{PIPE_V, CoreType::AIV}, {PIPE_S, CoreType::AIV}},
    // PIPE_S->PIPE_M
    {{PIPE_S, CoreType::AIC}, {PIPE_M, CoreType::AIC}},
    // PIPE_M->PIPE_S
    {{PIPE_M, CoreType::AIC}, {PIPE_S, CoreType::AIC}},
    // PIPE_S->PIPE_MTE1
    {{PIPE_S, CoreType::AIC}, {PIPE_MTE1, CoreType::AIC}},
    // PIPE_MTE1->PIPE_S
    {{PIPE_MTE1, CoreType::AIC}, {PIPE_S, CoreType::AIC}},
    // PIPE_S->PIPE_MTE2
    {{PIPE_S, CoreType::AIC}, {PIPE_MTE2, CoreType::AIC}},
    {{PIPE_S, CoreType::AIV}, {PIPE_MTE2, CoreType::AIV}},
    // PIPE_MTE2->PIPE_S
    {{PIPE_MTE2, CoreType::AIC}, {PIPE_S, CoreType::AIC}},
    {{PIPE_MTE2, CoreType::AIV}, {PIPE_S, CoreType::AIV}},
    // PIPE_S->PIPE_MTE3
    {{PIPE_S, CoreType::AIC}, {PIPE_MTE3, CoreType::AIC}},
    {{PIPE_S, CoreType::AIV}, {PIPE_MTE3, CoreType::AIV}},
    // PIPE_MTE3->PIPE_S
    {{PIPE_MTE3, CoreType::AIC}, {PIPE_S, CoreType::AIC}},
    {{PIPE_MTE3, CoreType::AIV}, {PIPE_S, CoreType::AIV}},
    // PIPE_S->PIPE_FIX
    {{PIPE_S, CoreType::AIC}, {PIPE_FIX, CoreType::AIC}},
    // PIPE_FIX->PIPE_S
    {{PIPE_FIX, CoreType::AIC}, {PIPE_S, CoreType::AIC}},
    // PIPE_M->PIPE_MTE1
    {{PIPE_M, CoreType::AIC}, {PIPE_MTE1, CoreType::AIC}},
    // PIPE_MTE1->PIPE_M
    {{PIPE_MTE1, CoreType::AIC}, {PIPE_M, CoreType::AIC}},
    // PIPE_M->PIPE_MTE2
    {{PIPE_M, CoreType::AIC}, {PIPE_MTE2, CoreType::AIC}},
    // PIPE_MTE2->PIPE_M
    {{PIPE_MTE2, CoreType::AIC}, {PIPE_M, CoreType::AIC}},
    // PIPE_M->PIPE_MTE3
    {{PIPE_M, CoreType::AIC}, {PIPE_MTE3, CoreType::AIC}},
    // PIPE_MTE3->PIPE_M
    {{PIPE_MTE3, CoreType::AIC}, {PIPE_M, CoreType::AIC}},
    // PIPE_M->PIPE_FIX
    {{PIPE_M, CoreType::AIC}, {PIPE_FIX, CoreType::AIC}},
    // PIPE_FIX->PIPE_M
    {{PIPE_FIX, CoreType::AIC}, {PIPE_M, CoreType::AIC}},
    // PIPE_MTE1->PIPE_MTE2
    {{PIPE_MTE1, CoreType::AIC}, {PIPE_MTE2, CoreType::AIC}},
    // PIPE_MTE2->PIPE_MTE1
    {{PIPE_MTE2, CoreType::AIC}, {PIPE_MTE1, CoreType::AIC}},
    // PIPE_MTE1->PIPE_MTE3
    {{PIPE_MTE1, CoreType::AIC}, {PIPE_MTE3, CoreType::AIC}},
    // PIPE_MTE3->PIPE_MTE1
    {{PIPE_MTE3, CoreType::AIC}, {PIPE_MTE1, CoreType::AIC}},
    // PIPE_MTE1->PIPE_FIX
    {{PIPE_MTE1, CoreType::AIC}, {PIPE_FIX, CoreType::AIC}},
    // PIPE_FIX->PIPE_MTE1
    {{PIPE_FIX, CoreType::AIC}, {PIPE_MTE1, CoreType::AIC}},
    // PIPE_MTE2->PIPE_FIX
    {{PIPE_MTE2, CoreType::AIC}, {PIPE_FIX, CoreType::AIC}},
    // PIPE_FIX->PIPE_MTE2
    {{PIPE_FIX, CoreType::AIC}, {PIPE_MTE2, CoreType::AIC}},
    // PIPE_MTE3->PIPE_FIX
    {{PIPE_MTE3, CoreType::AIC}, {PIPE_FIX, CoreType::AIC}},
    // PIPE_FIX->PIPE_MTE3
    {{PIPE_FIX, CoreType::AIC}, {PIPE_MTE3, CoreType::AIC}},
};

bool PipeSync::ConstructDepInfo(DataDepInfo &depInfo, std::vector<IndexOp> &syncedOpLog, int i) {
    auto &log = syncedOpLog[i].second;
    if (log->GetOpcodeStr() != "SYNC_SRC") {
        return false;
    }
    auto setPipe = log->syncQueue_.pipeId_;
    auto waitPipe = log->syncQueue_.trigPipeId_;
    auto setCore = log->syncQueue_.coreType_;
    auto waitCore = log->syncQueue_.trigCoreType_;
    auto eventId = log->syncQueue_.eventId_;
    if (!(setPipe == depInfo.setp && setCore == depInfo.setc && waitPipe == depInfo.waitp && waitCore == depInfo.waitc)) {
        return false;
    }
    if (std::find(depInfo.setOpEventIdList.begin(), depInfo.setOpEventIdList.end(), eventId) !=
        depInfo.setOpEventIdList.end()) {
        return false;
    }
    depInfo.setOpIdList.push_back(i);
    depInfo.setOpEventIdList.push_back(eventId);
    return true;
}

int PipeSync::GetSyncSrcLogIdx(std::vector<IndexOp> &syncedOpLog, int i) {
    int j = i - 1;
    for (; j >= 0; j--) {
        if (syncedOpLog[j].first != -1) {
            break;
        }
    }
    return syncedOpLog[j].first;
}

bool PipeSync::FindDataDep(DataDepInfo &depInfo, std::vector<IndexOp> &syncedOpLog, int i) {
    if (!ConstructDepInfo(depInfo, syncedOpLog, i)) {
        return false;
    }
    int syncSrcLogIdx = GetSyncSrcLogIdx(syncedOpLog, i);
    DepOp &depOpSrc = depOps_[syncSrcLogIdx];
    for (auto syncDstLogIdx : depOpSrc.setPipe) { //setpipe中的op为该op之后的，依赖于该op的op id
        DepOp &depOpDst = depOps_[syncDstLogIdx];
        if (depOpDst.selfPipeCore.core == depInfo.waitc && depOpDst.selfPipeCore.pipeStart == depInfo.waitp) {
            depInfo.opDepList.push_back(std::make_pair(syncSrcLogIdx, syncDstLogIdx));
        }
    }
    return true;
}

bool PipeSync::FindMaxOverlap(DataDepInfo &depInfo, int &maxOverlapDepIdx) {
    int maxOverlap = minimalMergeOverlap - 1;
    for (int idx = 0; idx < static_cast<int>(depInfo.opDepList.size() - 1); idx++) {
        if (depInfo.opDepList[idx].second < depInfo.opDepList[idx + 1].first) { //相邻的两个依赖之间没有重叠
            continue;
        }
        //max_overlap初始值为阈值，相邻两个依赖之间的重叠超过阈值，可以合并
        //max_overlap用来记录遍历到的最大overlap
        if ((depInfo.opDepList[idx].second - depInfo.opDepList[idx + 1].first) > maxOverlap) {
            maxOverlapDepIdx = idx;
            maxOverlap = depInfo.opDepList[idx].second - depInfo.opDepList[idx + 1].first;
        }
    }
    if (maxOverlapDepIdx == -1) { // 8个依赖中，前后依赖都没有重叠，或者重叠全部小于阈值。
        return false;
    }
    return true;
}

Status PipeSync::SynDependency(int maxOverlapDepIdx, const DataDepInfo &depInfo, const PipePair &pipePair, std::vector<IndexOp> &syncedOpLog) {
    int set1 = depInfo.opDepList[maxOverlapDepIdx].first;
    int wait1 = depInfo.opDepList[maxOverlapDepIdx].second;
    int set2 = depInfo.opDepList[maxOverlapDepIdx + 1].first;
    int wait2 = depInfo.opDepList[maxOverlapDepIdx + 1].second;
    int eventId1 = depInfo.setOpEventIdList[maxOverlapDepIdx];
    int eventId2 = depInfo.setOpEventIdList[maxOverlapDepIdx + 1];
    int syncOpIdx1 = depInfo.setOpIdList[maxOverlapDepIdx];
    if (set1 >= set2 || wait1 >= wait2) {
        ALOG_ERROR_F("Dependency error, RelaxFakeDataDep failed!");
        return FAILED;
    }
    RemoveOpDep(depOps_[set1], depOps_[wait1]);
    RemoveOpDep(depOps_[set2], depOps_[wait2]);
    if (AddOpDep(depOps_[set2], depOps_[wait1]) != SUCCESS) {
        ALOG_ERROR_F("RelaxFakeDataDep failed at function AddOpDep!");
        return FAILED;
    }
    GetFreeEventIdQueue(pipePair).push_back(eventId1);
    setWaitPairMap_[{set2, wait1}] = eventId2;
    //将靠前的一对有依赖关系op中插入的SYNC_SRC op删除
    syncedOpLog[syncOpIdx1].second->SetAsDeleted();
    syncedOpLog.erase(syncedOpLog.begin() + syncOpIdx1);
    return SUCCESS;
}

Status PipeSync::GetDepInfo(std::vector<IndexOp> &syncedOpLog, const PipePair &pipePair, DataDepInfo &depInfo) {
    depInfo.setp = pipePair.first.pipe;
    depInfo.setc = pipePair.first.core;
    depInfo.waitp = pipePair.second.pipe;
    depInfo.waitc = pipePair.second.core;
    // 8个eventid全部被占用
    // 说明该set pipe内肯定有8个set op已经发射，而wait pipe内对应的8个op一个都没发射。
    // 找到这8个setop的idx，对应的event id，以及对应的waitop idx
    auto eventNum = static_cast<std::vector<std::pair<int, int>>::size_type>(GetMaxEventId(pipePair));
    for (int i = syncedOpLog.size() - 1; i >= 0; i--) {
        if (!(FindDataDep(depInfo, syncedOpLog, i))) {
            continue;
        }
        if (depInfo.setOpIdList.size() == eventNum) {
            break;
        }
    }
    // 由于从后向前寻找，得到的结果列表进行反转。
    std::reverse(depInfo.opDepList.begin(), depInfo.opDepList.end());
    std::reverse(depInfo.setOpIdList.begin(), depInfo.setOpIdList.end());
    std::reverse(depInfo.setOpEventIdList.begin(), depInfo.setOpEventIdList.end());
    if (depInfo.opDepList.size() != eventNum || depInfo.setOpIdList.size() != eventNum || depInfo.setOpEventIdList.size() != eventNum) {
        ALOG_ERROR_F("dep size should be %d, RelaxFakeDataDep failed!", eventNum);
        return FAILED;
    }
    return SUCCESS;
}

Status PipeSync::RelaxFakeDataDep(std::vector<IndexOp> &syncedOpLog) {
    // 合并阈值。只有前后两个set-wait对之间的重叠（以op数量度量）超过该阈值时，才对这两个set-wait对进行合并。
    for (const auto &pipePair : dataDepPair) {
        if (HasFreeEventId(pipePair)) {
            continue;
        }
        // 找到free id exhausted的pipe pair
        DataDepInfo depInfo;
        if (GetDepInfo(syncedOpLog, pipePair, depInfo) != SUCCESS) {
            ALOG_ERROR_F("GetDepInfo failed!");
            return FAILED;
        }
        // 找到前一个依赖和后一个依赖间重叠最大的地方
        int maxOverlapDepIdx{-1};
        if (!(FindMaxOverlap(depInfo, maxOverlapDepIdx))) {
            continue;
        }
        // 合并依赖
        if (SynDependency(maxOverlapDepIdx, depInfo, pipePair, syncedOpLog) != SUCCESS) {
            ALOG_ERROR_F("SynDependency failed!");
            return FAILED;
        }
    }
    return SUCCESS;
}

bool PipeSync::GenSyncOp(PipeCoreReal set, PipeCoreReal wait, int eventId, bool isSet, Operation *op) {
    if (set.core != wait.core && !config::GetPassGlobalConfig("enable_cv_fuse", false)) {
        return true;
    }
    if (set.pipe != wait.pipe) {
        op->SetOpCode(isSet ? Opcode::OP_SYNC_SRC : Opcode::OP_SYNC_DST);
        op->syncQueue_ = {set.pipe, wait.pipe, set.core, wait.core, eventId};
        return true;
    }
    if (isSet || set.pipe == PipeType::PIPE_S) {
        return false;
    }
    //同步相关的信息放在operation属性里
    op->syncQueue_ = {set.pipe, wait.pipe, set.core, wait.core, eventId};
    if (set.core == CoreType::AIV) {
        op->SetOpCode(Opcode::OP_BAR_V);
        return true;
    }
    op->SetOpCode(Opcode::OP_BAR_M);
    return true;
}

Status PipeSync::GetEventId(const PipePair &pp, int &eventId) {
    if (pp.first.core != pp.second.core && !config::GetPassGlobalConfig("enable_cv_fuse", false)) {
        // CV sync
        eventId = cvSyncEventId++;
        return SUCCESS;
    }
    if (pp.first.pipe == pp.second.pipe) {
        // Barrier.V
        eventId = -1;
        return SUCCESS;
    }

    auto &eventQ = GetFreeEventIdQueue(pp);
    if (eventQ.empty()) { ALOG_ERROR_F("Eventid exhausted, GetEventId failed!"); return FAILED; }

    eventId = eventQ.front();
    eventQ.pop_front();
    return SUCCESS;
}

bool PipeSync::HasFreeEventId(const PipePair &pp) {
    std::deque<int> &eventQ = GetFreeEventIdQueue(pp);
    return !eventQ.empty();
}

bool PipeSync::BufOverlap(const TileRange &range1, int magic1, const TileRange &range2, int magic2) const {
    ALOG_DEBUG_F("        Range 1 [%zu ~ %zu], range 2 [%zu ~ %zu].", range1.start, range1.end, range2.start, range2.end);
    if (range1.end > range2.start && range2.end > range1.start) {
        ALOG_DEBUG_F("        Tensor %d and tensor %d have overlap ", magic1, magic2);
        return true;
    }
    ALOG_DEBUG_F("        Tensor %d and tensor %d don't have overlap ", magic1, magic2);
    return false;
}

bool PipeSync::CheckWawDependency(const Operation *opSet, const Operation *opWait, size_t k, size_t idx) const {
    for (size_t setIdx = 0; setIdx < opSet->GetOOperands().size(); setIdx++) {
        for (size_t waitIdx = 0; waitIdx < opWait->GetOOperands().size(); waitIdx++) {
            if (opSet->GetOOperands()[setIdx]->GetMemoryTypeOriginal() == opWait->GetOOperands()[waitIdx]->GetMemoryTypeOriginal() &&
                BufOverlap(opSet->GetOOperands()[setIdx]->memorymap[BLOCK_GRAPH_DEFAULT_COLOR], opSet->GetOOperands()[setIdx]->GetMagic(),
                    opWait->GetOOperands()[waitIdx]->memorymap[BLOCK_GRAPH_DEFAULT_COLOR], opWait->GetOOperands()[waitIdx]->GetMagic())) {
                ALOG_DEBUG_F("        %d %zu %s and %d %zu %s has WAW data dependency", opSet->GetOpMagic(), k, opSet->GetOpcodeStr().c_str(),
                    opWait->GetOpMagic(), idx, opWait->GetOpcodeStr().c_str());
                return true;
            }
        }
    }
    return false;
}

bool PipeSync::CheckRawDependency(const Operation *opSet, const Operation *opWait, size_t k, size_t idx) const {
    for (size_t outIdx = 0; outIdx < opSet->GetOOperands().size(); outIdx++) {
        for (size_t inIdx = 0; inIdx < opWait->GetIOperands().size(); inIdx++) {
            auto memTypeSame = opWait->GetIOperands()[inIdx]->GetMemoryTypeOriginal() == opSet->GetOOperands()[outIdx]->GetMemoryTypeOriginal();
            auto ddrTensorSame = opSet->GetOOperands()[outIdx]->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
                opWait->GetIOperands()[inIdx]->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId ==
                opSet->GetOOperands()[outIdx]->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId;
            auto overlap = BufOverlap(opWait->GetIOperands()[inIdx]->memorymap[BLOCK_GRAPH_DEFAULT_COLOR], opWait->GetIOperands()[inIdx]->GetMagic(),
                opSet->GetOOperands()[outIdx]->memorymap[BLOCK_GRAPH_DEFAULT_COLOR], opSet->GetOOperands()[outIdx]->GetMagic());
            if (memTypeSame && (overlap || ddrTensorSame)) {
                ALOG_DEBUG_F("        %d %zu %s and %d %zu %s has RAW data dependency", opSet->GetOpMagic(), k, opSet->GetOpcodeStr().c_str(),
                       opWait->GetOpMagic(), idx, opWait->GetOpcodeStr().c_str());
                return true;
            }
        }
    }
    return false;
}

bool PipeSync::CheckWarDependency(const Operation *opSet, const Operation *opWait, size_t k, size_t idx) const {
    for (size_t outIdx = 0; outIdx < opWait->GetOOperands().size(); outIdx++) {
        for (size_t inIdx = 0; inIdx < opSet->GetIOperands().size(); inIdx++) {
            auto memTypeSame = opSet->GetIOperands()[inIdx]->GetMemoryTypeOriginal() == opWait->GetOOperands()[outIdx]->GetMemoryTypeOriginal();
            auto ddrTensorSame = opSet->GetIOperands()[inIdx]->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
                opWait->GetOOperands()[outIdx]->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId ==
                opSet->GetIOperands()[inIdx]->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId;
            auto overlap = BufOverlap(opSet->GetIOperands()[inIdx]->memorymap[BLOCK_GRAPH_DEFAULT_COLOR], opSet->GetIOperands()[inIdx]->GetMagic(),
                opWait->GetOOperands()[outIdx]->memorymap[BLOCK_GRAPH_DEFAULT_COLOR], opWait->GetOOperands()[outIdx]->GetMagic());
            if (memTypeSame && (overlap || ddrTensorSame)) {
                ALOG_DEBUG_F("        %d %zu %s and %d %zu %s has WAR data dependency", opSet->GetOpMagic(), k, opSet->GetOpcodeStr().c_str(),
                       opWait->GetOpMagic(), idx, opWait->GetOpcodeStr().c_str());
                return true;
            }
        }
    }
    return false;
}

bool PipeSync::HasDataDependency(const Operation *opSet, const Operation *opWait, size_t k, size_t idx) const {
    std::string opSetStr = opSet->GetOpcodeStr();
    std::string opWaitStr = opWait->GetOpcodeStr();

    // check WAW
    bool checkWaw = true;
    auto setCfg = OpcodeManager::Inst().GetTileOpCfg(opSet->GetOpcode());
    auto waitCfg = OpcodeManager::Inst().GetTileOpCfg(opWait->GetOpcode());
    if (waitCfg.pipeIdStart_ == setCfg.pipeIdStart_ && (opSetStr.find("CUBE_A_MUL") == std::string::npos || opWaitStr.find("CUBE_A_MUL") == std::string::npos)) {
        checkWaw = false;
    }
    if (checkWaw) {
        if (CheckWawDependency(opSet, opWait, k, idx)) {
            return true;
        }
    }

    // check RAW
    if (CheckRawDependency(opSet, opWait, k, idx)) {
        return true;
    }

    // check WAR
    if (CheckWarDependency(opSet, opWait, k, idx)) {
        return true;
    }

    return false;
}

void PipeSync::UpdateDep(DepOp &currOp, DepOp &prevOp) {
    PipeCoreReal currPipe(currOp.selfPipeCore.pipeStart, currOp.selfPipeCore.core);
    PipeCoreReal prevPipe(prevOp.selfPipeCore.pipeEnd, prevOp.selfPipeCore.core);
    auto &currPipeDep = latestPipeDep_[currPipe];
    currPipeDep.waitIdx = currOp.idx;

    auto currSetPipeIter = currPipeDep.setPipes.find(prevPipe);
    if (currSetPipeIter == currPipeDep.setPipes.end() || currSetPipeIter->second < prevOp.idx) {
        // no indirect dependency exist, save current dependency
        currOp.waitPipe.emplace_back(prevOp.idx);
        prevOp.setPipe.emplace_back(currOp.idx);
        currPipeDep.setPipes[prevPipe] = prevOp.idx;
        auto prevPipeDepIter = latestPipeDep_.find(prevPipe);
        auto prevWaitPipeIdx = prevPipeDepIter->second.waitIdx;
        if (prevPipeDepIter != latestPipeDep_.end() && prevWaitPipeIdx <= prevOp.idx) {
            // merge dependency
            std::map<PipeCoreReal, size_t, PipeCoreRealCompare> prevSetPipes = prevPipeDepIter->second.setPipes;
            for (auto ele : prevSetPipes) {
                PipeCoreReal prevSetPipeType = ele.first;
                size_t prevSetPipeIdx = ele.second;
                auto res = currPipeDep.setPipes.emplace(prevSetPipeType, prevSetPipeIdx);
                //isExist == isPrevSetPipeTypeExist
                bool isExist = !res.second;
                size_t &existIdx = res.first->second;
                if (isExist && existIdx < prevSetPipeIdx) {
                    // overwrite dependency
                    existIdx = prevSetPipeIdx;
                }
            }
        }
    }
}

bool PipeSync::CheckNotIgnorableCase(size_t prev, size_t curr, const std::vector<Operation *> opLogPtr) {
    auto &prevOp = opLogPtr[prev];
    auto &depPrevOp = depOps_[prev];
    auto &depCurrOp = depOps_[curr];
    // different pipe core type datandependency cannot be ignored
    if (depPrevOp.selfPipeCore.pipeEnd != depCurrOp.selfPipeCore.pipeStart || depPrevOp.selfPipeCore.core != depCurrOp.selfPipeCore.core) {
        return false;
    }
    // only check pipe_barrier(PIPE_V) and pipe_barrier(PIPE_M)
    PipeCoreReal prevOpPipeCore(depPrevOp.selfPipeCore.pipeEnd, depPrevOp.selfPipeCore.core);
    if (prevOpPipeCore != PipeCoreReal{PIPE_V, CoreType::AIV} && prevOpPipeCore != PipeCoreReal{PIPE_M, CoreType::AIC}) {
        return false;
    }
    // only consider single output
    if (prevOp->GetOOperands().size() != 1) {
        return false;
    }
    return true;
}

bool PipeSync::IgnorableIntraPipeDep(size_t prev, size_t curr, const std::vector<Operation *> opLogPtr) {
    // true表示依赖关系可忽略，false表示依赖关系不可忽略
    auto &prevOp = opLogPtr[prev];
    // VIEW or ASSEMBLE data dependency can be ignored
    if (opLogPtr[prev]->GetOpcode() == Opcode::OP_VIEW || opLogPtr[curr]->GetOpcode() == Opcode::OP_VIEW ||
        opLogPtr[prev]->GetOpcode() == Opcode::OP_ASSEMBLE || opLogPtr[curr]->GetOpcode() == Opcode::OP_ASSEMBLE) {
        ALOG_DEBUG_F("        %d %s and %d %s dependency is ignorable because op is VIEW or ASSEMBLE", opLogPtr[prev]->GetOpMagic(), opLogPtr[prev]->GetOpcodeStr().c_str(),
            opLogPtr[curr]->GetOpMagic(), opLogPtr[curr]->GetOpcodeStr().c_str());
        return true;
    }
    if (!CheckNotIgnorableCase(prev, curr, opLogPtr)) {
        return false;
    }
    auto outputShape = prevOp->GetOOperands()[0]->shape;
    auto dtype = prevOp->GetOOperands()[0]->tensor->datatype;
    int repeatsize = std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<int64_t>()) * BytesOf(dtype) / 256;
    // pipe_barrier can be safely ignored when intrins REPEAT > 16
    if (repeatsize > IGNORABLE_REPEAT_SIZE) {
        ALOG_DEBUG_F("        %d %s and %d %s dependency is ignorable because pipeBarrier can be safely ignored when intrins REPEAT > 16, now REPEAT is %d",
            opLogPtr[prev]->GetOpMagic(), opLogPtr[prev]->GetOpcodeStr().c_str(), opLogPtr[curr]->GetOpMagic(), opLogPtr[curr]->GetOpcodeStr().c_str(), repeatsize);
        return true;
    }
    return false;
}

// find depend op in opLog for 0 to idx
void PipeSync::FindDep(DepOp &op, const std::vector<Operation *> opLogPtr, size_t idx, DataDependencySearcher& dataDependencySearcher) {
    const auto currOp = opLogPtr[idx];
    ALOG_DEBUG_F("=== OP: %d %zu %s ===", currOp->GetOpMagic(), idx, currOp->GetOpcodeStr().c_str());
    // check dependency from latest op to oldest
    auto dataDependencySet = dataDependencySearcher.Find(currOp);
    for (auto it = dataDependencySet.rbegin(); it != dataDependencySet.rend(); it++) {
        size_t k = *it;
        const Operation *prevAOp = opLogPtr[k];
        DepOp &prevOp = depOps_[k];
        ALOG_DEBUG_F("    Current process ops: %d %zu %s and %d %zu %s", prevAOp->GetOpMagic(), k, prevAOp->GetOpcodeStr().c_str(),
            currOp->GetOpMagic(), idx, currOp->GetOpcodeStr().c_str());

        if (HasDataDependency(prevAOp, currOp, k, idx)) {
            bool ignorable = false;
            if (IgnorableIntraPipeDep(k, idx, opLogPtr)) {
                ignorable = true;
            }
            if (!ignorable) {
                UpdateDep(op, prevOp);
            }
        }
    }
    dataDependencySearcher.Insert(currOp, idx);
}

std::deque<int> &PipeSync::GetFreeEventIdQueue(const PipePair &pp) {
    if (freeEventId_.count(pp) == 0) {
        for (int i = 0; i < GetMaxEventId(pp); i++) {
            freeEventId_[pp].push_back(i);
        }
    }
    return freeEventId_[pp];
}

void PipeSync::SetTileOpCfg(Function &function, std::vector<Operation *> srcLog, std::vector<Operation *> &dstLog, size_t &i, size_t &prerun) {
    constexpr size_t prerunNum = 2;
    for (; i < srcLog.size(); i++) {
        auto opcfg = OpcodeManager::Inst().GetTileOpCfg(srcLog[i]->GetOpcode());
        if (srcLog[i]->GetOpcode() == Opcode::OP_COPY_IN) {
            opcfg.pipeIdStart_ = PipeType::PIPE_MTE2;
        }
        if (srcLog[i]->GetOpcode() == Opcode::OP_COPY_OUT) {
            opcfg.pipeIdStart_ = PipeType::PIPE_MTE3;
        }
        if ((opcfg.pipeIdStart_ != PIPE_S && opcfg.pipeIdStart_ != PIPE_MTE2 && srcLog[i]->GetOpcode() != Opcode::OP_RESHAPE) || prerun == prerunNum) {
            break;
        }
        if (opcfg.pipeIdStart_ == PIPE_MTE2) {
            if (prerun == 0) {
                std::vector<std::shared_ptr<LogicalTensor>> input;
                std::vector<std::shared_ptr<LogicalTensor>> output;
                Operation &phaseOp = function.AddRawOperation(npu::tile_fwk::Opcode::OP_PHASE1, {input}, {output});
                phaseOp.opmagic = ++maxOpMagic;
                Operation *phaseOpPtr = &phaseOp;
                dstLog.emplace_back(phaseOpPtr);
            }
            prerun++;
        }
        dstLog.emplace_back(srcLog[i]);
    }
}

void PipeSync::AddPhaseOp(Function &function, std::vector<Operation *> &dstLog, size_t &prerun) {
    if (prerun > 0) {
        std::vector<std::shared_ptr<LogicalTensor>> input;
        std::vector<std::shared_ptr<LogicalTensor>> output;
        Operation &phaseOp = function.AddRawOperation(npu::tile_fwk::Opcode::OP_PHASE2, {input}, {output});
        phaseOp.opmagic = ++maxOpMagic;
        Operation *phaseOpPtr = &phaseOp;
        dstLog.emplace_back(phaseOpPtr);
    }
}

void PipeSync::PhaseKernelProcess(Function &function, std::vector<Operation *> srcLog, std::vector<Operation *> &dstLog) {
    size_t prerun = 0;
    size_t i = 0;
    SetTileOpCfg(function, srcLog, dstLog, i, prerun);
    AddPhaseOp(function, dstLog, prerun);
    for (; i < srcLog.size(); i++) {
        dstLog.emplace_back(srcLog[i]);
    }
}

Status PipeSync::ProcessView(std::vector<Operation *> &opLogNew, std::pair<Operation *, Operation *> pair) {
    auto it1 = std::find(opLogNew.begin(), opLogNew.end(), pair.second);
    auto it2 = std::find(opLogNew.begin(), opLogNew.end(), pair.first);
    if (it1 == opLogNew.end()) {
        if (it2 == opLogNew.end()) {
            opLogNew.emplace_back(pair.first);
            opLogNew.emplace_back(pair.second);
            return SUCCESS;
        }
        opLogNew.insert(it2+1, pair.second);
        return SUCCESS;
    }
    if (it2 == opLogNew.end()) {
        opLogNew.insert(it1, pair.first);
    }
    return SUCCESS;
}

Status PipeSync::ProcessAssemble(std::vector<Operation *> &opLogNew, std::pair<Operation *, Operation *> pair) {
    auto it1 = std::find(opLogNew.begin(), opLogNew.end(), pair.first);
    auto it2 = std::find(opLogNew.begin(), opLogNew.end(), pair.second);
    if (it1 == opLogNew.end()) {
        if (it2 == opLogNew.end()) {
            opLogNew.emplace_back(pair.second);
            opLogNew.emplace_back(pair.first);
            return SUCCESS;
        }
        opLogNew.insert(it2+1, pair.first);
        return SUCCESS;
    }
    if (it2 == opLogNew.end()) {
        opLogNew.insert(it1, pair.second);
    }
    return SUCCESS;
}

Status PipeSync::ProcessViewAssemble(std::vector<Operation *> &opLogNew, std::pair<Operation *, Operation *> pair) {
    if (pair.first->GetOpcode() == Opcode::OP_VIEW) {
        if (ProcessView(opLogNew, pair) != SUCCESS) {
            ALOG_ERROR_F("ProcessView failed.");
            return FAILED;
        }
        return SUCCESS;
    }
    if (pair.first->GetOpcode() != Opcode::OP_ASSEMBLE) {
        ALOG_ERROR_F("ProcessViewAssemble failed, this op should be ASSEMBLE!");
        return FAILED;
    }
    if (ProcessAssemble(opLogNew, pair) != SUCCESS) {
        ALOG_ERROR_F("ProcessAssemble failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status PipeSync::ReorderViewAssemble(std::vector<Operation *> &opLog, std::vector<Operation *> &opListNew, const std::unordered_map<Operation *, Operation *> &changeMap) {
    std::unordered_set<Operation *> toBeInsert;
    for (auto pair : changeMap) {
        toBeInsert.insert(pair.first);
        toBeInsert.insert(pair.second);
    }
    for (auto opPtr : opLog) {
        auto it = toBeInsert.find(opPtr);
        if (it == toBeInsert.end()) {
            opListNew.emplace_back(opPtr);
            continue;
        }
        for (auto pair : changeMap) {
            if (pair.second == opPtr && (ProcessViewAssemble(opListNew, pair) != SUCCESS)) {
                ALOG_ERROR_F("ReorderViewAssemble failed at function ProcessViewAssemble!");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status PipeSync::ProcessViewOrder(Operation *opPtr, std::vector<Operation *> &opLog, std::unordered_map<Operation *, Operation *> &changeMap) {
    auto consumers = opPtr->ConsumerOps();
    if (consumers.empty()) {
        ALOG_ERROR_F("%d VIEW op doesn't have consumer, ProcessViewAssembleOrder failed!", opPtr->GetOpMagic());
        return FAILED;
    }
    auto minIt = opLog.end();
    for (auto &consumer : consumers) {
        auto it = std::find(opLog.begin(), opLog.end(), consumer);
        if (it != opLog.end() && it < minIt) {
            minIt = it;
        }
    }
    changeMap[opPtr] = *minIt;
    ALOG_DEBUG_F("%d VIEW consumer: %d", opPtr->GetOpMagic(), (*minIt)->GetOpMagic());
    return SUCCESS;
}

Status PipeSync::ProcessAssembleOrder(Operation *opPtr, std::vector<Operation *> &opLog, std::unordered_map<Operation *, Operation *> &changeMap) {
    auto producers = opPtr->ProducerOps();
    if (producers.empty()) {
        ALOG_ERROR_F("%d ASSEMBLE op doesn't have producer, ProcessViewAssembleOrder failed!", opPtr->GetOpMagic());
        return FAILED;
    }
    auto maxIt = opLog.begin();
    for (auto &producer : producers) {
        auto it = std::find(opLog.begin(), opLog.end(), producer);
        if (it != opLog.begin() && it > maxIt) {
            maxIt = it;
        }
    }
    changeMap[opPtr] = *maxIt;
    ALOG_DEBUG_F("%d ASSEMBLE producer: %d", opPtr->GetOpMagic(), (*maxIt)->GetOpMagic());
    return SUCCESS;
}

Status PipeSync::ProcessViewAssembleOrder(std::vector<Operation *> &opLog, std::vector<Operation *> &opListNew) {
    std::unordered_map<Operation *, Operation *> changeMap;
    for (auto &opPtr : opLog) {
        if (opPtr->GetOpcode() == Opcode::OP_VIEW) {
            if (ProcessViewOrder(opPtr, opLog, changeMap)) {
                ALOG_ERROR_F("ProcessViewOrder failed!");
                return FAILED;
            }
            continue;
        }
        if (opPtr->GetOpcode() != Opcode::OP_ASSEMBLE) {
            break;
        }
        if (ProcessAssembleOrder(opPtr, opLog, changeMap)) {
            ALOG_ERROR_F("ProcessAssembleOrder failed!");
            return FAILED;
        }
    }
    if (ReorderViewAssemble(opLog, opListNew, changeMap) != SUCCESS) {
        ALOG_ERROR_F("ProcessViewAssembleOrder failed at function ReorderViewAssemble!");
        return FAILED;
    }
    return SUCCESS;
}

void InsertSync::InsertPipeAll(Function *subGraphFunc) {
    std::vector<Operation*> oriOpList(subGraphFunc->Operations().DuplicatedOpList());
    std::vector<Operation*> newOpList;
    for (auto op : oriOpList) {
        newOpList.push_back(op);
        if (op->GetOpcode() == Opcode::OP_RESHAPE || op->GetOpcode() == Opcode::OP_VIEW || op->GetOpcode() == Opcode::OP_ASSEMBLE) {
            continue;
        }
        std::vector<std::shared_ptr<LogicalTensor>> input;
        std::vector<std::shared_ptr<LogicalTensor>> output;
        Operation &syncOp = subGraphFunc->AddRawOperation(npu::tile_fwk::Opcode::OP_BAR_ALL, {input}, {output});
        syncOp.syncQueue_ = {PipeType::PIPE_ALL, PipeType::PIPE_ALL, CoreType::AIV, CoreType::AIV, -1};
        newOpList.push_back(&syncOp);
    }
    subGraphFunc->ScheduleBy(newOpList);
}

Status InsertSync::GenNewOpList(Function *subGraphFunc, std::vector<Operation *> &opListNew) {
    PipeSync ps;
    std::vector<Operation *> syncedOpLogPtr;
    std::vector<Operation *> operationLogWithSync;
    if (ps.InsertSync(*subGraphFunc, syncedOpLogPtr) != SUCCESS) {
        ALOG_ERROR_F("InsertSyncMainLoop failed at function InsertSync!");
        return FAILED;
    }
    ps.PhaseKernelProcess(*subGraphFunc, syncedOpLogPtr, operationLogWithSync);
    subGraphFunc->EraseOperations(true, false);
    if (ps.ProcessViewAssembleOrder(operationLogWithSync, opListNew) != SUCCESS) {
        ALOG_ERROR_F("InsertSyncMainLoop failed at function ProcessViewAssembleOrder!");
        return FAILED;
    }
    return SUCCESS;
}

Status InsertSync::InsertSyncMainLoop(Function *subGraphFunc) {
    if (enableDebug_) {
        InsertPipeAll(subGraphFunc);
        return SUCCESS;
    }
    std::vector<Operation *> opListNew;
    if (GenNewOpList(subGraphFunc, opListNew) != SUCCESS) {
        ALOG_ERROR_F("GenNewOpList failed.");
        return FAILED;
    }
    subGraphFunc->ScheduleBy(opListNew, true);
    ALOG_DEBUG_F("==========================================================================================");
    for (const auto &op : subGraphFunc->Operations().DuplicatedOpList()) {
        if (op->GetOpcodeStr() == "SYNC_SRC" || op->GetOpcodeStr() == "SYNC_DST" || op->GetOpcodeStr() == "BAR.V" ||
            op->GetOpcodeStr() == "BAR.M") {
            ALOG_DEBUG_F("Output operation %d: %s, setpipe type: %s, waitpipe type: %s, eventid: %d", op->GetOpMagic(),
                op->GetOpcodeStr().c_str(), PipeTypeName(op->syncQueue_.pipeId_).c_str(),
                PipeTypeName(op->syncQueue_.trigPipeId_).c_str(), op->syncQueue_.eventId_);
            continue;
        }
        ALOG_DEBUG_F("Output operation %d: %s", op->GetOpMagic(), op->GetOpcodeStr().c_str());
    }
    return SUCCESS;
}

// regist pass
Status InsertSync::RunOnFunction(Function &function) {
    ALOG_INFO_F("===============================================================> Start InsertSync.");
    const unsigned hardwareConcurrency = config::GetPassGlobalConfig("pass_thread_num", 1);
    uint64_t index = 0;
    std::vector<std::pair<uint64_t, Function*>> subPrograms;
    for (auto &subProgram : function.rootFunc_->programs_) {
        subPrograms.push_back(subProgram);
    }
    std::atomic<size_t> nextIdx(0);
    size_t leafFuncSize = function.rootFunc_->programs_.size();
    // The max thread number by std::thread::hardware_concurrency()

    const unsigned threadNum = std::min(
        static_cast<unsigned>(leafFuncSize),
        hardwareConcurrency
    );
    std::vector<std::thread> workers;

    std::atomic<bool> multiThreadsStatus(true);
    for (unsigned i = 0; i < threadNum; ++i) {
        workers.emplace_back([&subPrograms, &nextIdx, leafFuncSize, &index, this, &multiThreadsStatus] {
            for (size_t idx = nextIdx.fetch_add(1, std::memory_order_relaxed); idx < leafFuncSize; idx = nextIdx.fetch_add(1, std::memory_order_relaxed)) {
                auto program = subPrograms[idx];
                ALOG_DEBUG_F("====================================Program %d ===========================================", index);
                if (InsertSyncMainLoop(program.second) != SUCCESS) {
                    multiThreadsStatus.store(false);
                }
                index++;
            }
        });
    }
    if (!multiThreadsStatus.load()) {
        if (threadNum == 1) {
            ALOG_ERROR_F("InsertSync RunOnFunction failed at function InsertSyncMainLoop in Single Thread scenario!");
            return FAILED;
        }
        ALOG_ERROR_F("InsertSync RunOnFunction failed at function InsertSyncMainLoop in Multiple Threads scenario!");
        return FAILED;
    }
    // Wait for all threads to finish
    for (auto& t : workers) {
        if (t.joinable()) {
            t.join();
        }
    }
    ALOG_INFO_F("===============================================================> Finish InsertSync.");
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
