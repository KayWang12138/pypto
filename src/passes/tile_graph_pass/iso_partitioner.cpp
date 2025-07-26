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
 * \file iso_partitioner.cpp
 * \brief
 */

#include "passes/tile_graph_pass/iso_partitioner.h"
#include <iostream>
#include <deque>
#include <algorithm>
#include "interface/function/function.h"
#include "interface/utils/log.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_utils/parallel_tool.h"

namespace npu::tile_fwk {

Status GraphPartitionPass::RunOnFunction(Function &function)
{
    ALOG_INFO_F("===> Start GraphPartitionPass.");
    IsoPartitioner partitioner;
    if (partitioner.SetParameter(function.paramConfigs_.sgCycleUpperBound,
                                 function.paramConfigs_.sgParallelNum,
                                 function.paramConfigs_.sgCycleLowerBound,
                                 function.paramConfigs_.useNodeHash) != SUCCESS) {
        ALOG_ERROR_F("Set parameters of GraphPartitionPass failed.");
        return FAILED;
    }
    if (partitioner.PartitionGraph(function) != SUCCESS) {
        ALOG_ERROR_F("GraphPartitionPass failed.");
        return FAILED;
    }
    ALOG_INFO_F("===> End GraphPartitionPass.");
    return SUCCESS;
}

Status GraphPartitionPass::PreCheck(Function &function)
{
    ALOG_INFO_F("PreCheck for pass: GraphPartitionPass.");
    for (auto &op : function.Operations().DuplicatedOpList()) {
        if (op == nullptr) {
            ALOG_ERROR_F("Null pointer in Operations.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status GraphPartitionPass::PostOperationCheck(Function &function)
{
    for (auto &op : function.Operations()) {
        int32_t curSubgraphID = op.GetSubgraphID();
        bool isStartNodeInSubgraph = true;
        for (auto iTensor : op.GetIOperands()) {
            for (auto &producer : iTensor->GetProducers()) {
                if (curSubgraphID == producer->GetSubgraphID()) {
                    isStartNodeInSubgraph = false;
                    break;
                }
            }
            if (!isStartNodeInSubgraph) {
                break;
            }
        }
        if (isStartNodeInSubgraph) {
            ALOG_DEBUG_F("Operation (opmagic: %d) is start node in subgraph.", op.GetOpMagic());
            if (op.GetOpcode() == Opcode::OP_ASSEMBLE || op.GetOpcode() == Opcode::OP_COPY_OUT) {
                ALOG_ERROR_F("Operation (opmagic: %d) is the start node of the subgraph, opcode should not be %s.",
                             op.GetOpMagic(), op.GetOpcodeStr().c_str());
                return FAILED;
            }
            continue;
        }
        bool isEndNodeInSubgraph = true;
        for (auto oTensor : op.GetOOperands()) {
            for (auto &consumer : oTensor->GetConsumers()) {
                if (curSubgraphID == consumer->GetSubgraphID()) {
                    isEndNodeInSubgraph = false;
                    break;
                }
            }
            if (!isEndNodeInSubgraph) {
                break;
            }
        }
        if (isEndNodeInSubgraph) {
            ALOG_DEBUG_F("Operation (opmagic: %d) is end node in subgraph.", op.GetOpMagic());
            if (op.GetOpcode() == Opcode::OP_VIEW || op.GetOpcode() == Opcode::OP_COPY_IN) {
                ALOG_ERROR_F("Operation (opmagic: %d) is the end node of the subgraph, opcode should not be %s.",
                             op.GetOpMagic(), op.GetOpcodeStr().c_str());
                return FAILED;
            }
            continue;
        }
    }
    return SUCCESS;
}

Status GraphPartitionPass::PostSubgraphCheck(const std::vector<std::vector<Operation*>> &subgraphs)
{
    for (auto subgraph : subgraphs) {
        if (subgraph.empty()) {
            continue;
        }
        int32_t aicCount = 0;
        int32_t aivCount = 0;
        int32_t aicMemoryCount = 0;
        int32_t aivMemoryCount = 0;
        std::unordered_set<std::shared_ptr<LogicalTensor>> tensorList;
        int32_t subgraphId = subgraph[0]->GetSubgraphID();
        for (auto &op : subgraph) {
            for (auto iTensor : op->GetIOperands()) {
                if (tensorList.find(iTensor) != tensorList.end()) {
                    continue;
                }
                tensorList.insert(iTensor);
                if (iTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L1 ||
                    iTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0A ||
                    iTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0B ||
                    iTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0C) {
                    aicMemoryCount++;
                } else if (iTensor->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
                    aivMemoryCount++;
                }
            }
            if (op->HasAttr(OpAttributeKey::isCube) && op->GetBoolAttribute(OpAttributeKey::isCube)) {
                aicCount++;
            } else {
                aivCount++;
            }
        }
        if (aicCount > 0 && aivCount > 0) {
            ALOG_ERROR_F("Subgraph %d has both AIV and AIC operation.", subgraphId);
            return FAILED;
        }
        if (aicMemoryCount > 0 && aivMemoryCount > 0) {
            ALOG_ERROR_F("Subgraph %d has both ub and l0/l1 memory type tensor.", subgraphId);
            return FAILED;
        }
    }
    return SUCCESS;
}

Status GraphPartitionPass::PostCheck(Function &function)
{
    ALOG_INFO("PostCheck for pass: GraphPartitionPass.");
    std::vector<std::vector<Operation*>> subgraphs(function.GetTotalSubGraphCount());
    for (auto &op : function.Operations()) {
        int32_t curSubgraphID = op.GetSubgraphID();
        if (curSubgraphID == -1) {
            ALOG_ERROR_F("Operation (opmagic: %d) is not in any subgraph.", op.GetOpMagic());
            return FAILED;
        }
        if (curSubgraphID < 0 || curSubgraphID >= static_cast<int32_t>(function.GetTotalSubGraphCount())) {
            ALOG_ERROR_F("Operation (opmagic: %d) has illegal SubgraphID.", op.GetOpMagic());
            return FAILED;
        }
        subgraphs[curSubgraphID].push_back(&op);
    }
    for (int graphID = 0; graphID < static_cast<int>(subgraphs.size()); graphID++) {
        if (subgraphs[graphID].size() == 0) {
            ALOG_ERROR_F("Subgraph %d includes no Operation.", graphID);
            return FAILED;
        }
    }
    if (!function.LoopCheck().empty()) {
        ALOG_ERROR("Loopcheck failed after pass: GraphPartitionPass.");
        return FAILED;
    }
    if (PostOperationCheck(function) != SUCCESS) {
        ALOG_ERROR_F("Operation post check failed.");
        return FAILED;
    }
    if (PostSubgraphCheck(subgraphs) != SUCCESS) {
        ALOG_ERROR_F("Subgraph post check failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status IsoPartitioner::PartitionGraph(Function &function)
{
    if (cycleThreshold_ == -1 || parallelThreshold_ == -1 || smallGraphThreshold_ == -1) {
        ALOG_ERROR_F("Partition parameters not initialized.");
        return FAILED;
    }
    if (BuildOpGraph(function.Operations().DuplicatedOpList()) != SUCCESS) {
        ALOG_ERROR_F("Partition the computational graph failed in building operation graph.");
        return FAILED;
    }
    if (BuildSuperNodeGraph() != SUCCESS) {
        ALOG_ERROR_F("Partition the computational graph failed in building SuperNode graph.");
        return FAILED;
    }
    if (BuildHashValues() != SUCCESS) {
        ALOG_ERROR_F("Partition the computational graph failed in building SuperNode hash values.");
        return FAILED;
    }
    if (BuildIsomorphismGroups() != SUCCESS) {
        ALOG_ERROR_F("Partition the computational graph failed in partitioning the graph.");
        return FAILED;
    }
    if (IsomorphismGroupMergeProcess(true) != SUCCESS) {
        ALOG_ERROR_F("Partition the computational graph failed in merging non-isomorphism groups.");
        return FAILED;
    }
    if (IsomorphismGroupMergeProcess(false) != SUCCESS) {
        ALOG_ERROR_F("Partition the computational graph failed in merging isomorphism groups.");
        return FAILED;
    }
    if (UpdatePartitionResult(function) != SUCCESS) {
        ALOG_ERROR_F("Partition the computational graph failed in updating the Function.");
        return FAILED;
    }
    return SUCCESS;
}

uint64_t OperationGraphInfo::GetHash(const Operation *op) const
{
    std::string hashString;
    hashString.append(op->GetOpcodeStr());
    for (auto tensor : op->GetIOperands()) {
        hashString.append("IOperand-");
        hashString.append(std::to_string(tensor->GetMemoryTypeOriginal()));
        hashString.append(std::to_string(tensor->tensor->datatype));
    }
    for (auto tensor : op->GetOOperands()) {
        hashString.append("OOperand-");
        hashString.append(std::to_string(tensor->GetMemoryTypeOriginal()));
        hashString.append(std::to_string(tensor->tensor->datatype));
    }
    return std::hash<std::string>{}(hashString);
}

std::vector<int32_t> OperationGraphInfo::GetSameLevelOpIdx(int32_t opIdx, Opcode opLabel) const
{
    if (opIdx < 0 || opIdx >= static_cast<int32_t>(opList_.size()) || opList_[opIdx]->GetOOperands().size() == 0) {
        return {};
    }
    std::vector<int32_t> res;
    std::shared_ptr<LogicalTensor> output = opList_[opIdx]->GetOOperands()[0];
    for (auto &parentOpPtr : output->GetProducers()) {
        if (parentOpPtr->GetOpcode() == opLabel) {
            int32_t parentOpMagic = parentOpPtr->GetOpMagic();
            if (magic2Idx_.count(parentOpMagic) > 0) {
                int32_t targetIdx = magic2Idx_.at(parentOpPtr->GetOpMagic());
                res.push_back(targetIdx);
            }
        }
    }
    return res;
}

Status IsoPartitioner::BuildOpGraph(const std::vector<Operation*> &opList)
{
    operationInfo_ = std::make_shared<OperationGraphInfo>();
    if (operationInfo_ == nullptr) {
        ALOG_ERROR_F("Create OperationInfo failed.");
        return FAILED;
    }
    operationInfo_->opList_ = opList;
    operationInfo_->inGraph_.resize(opList.size());
    operationInfo_->outGraph_.resize(opList.size());
    operationInfo_->opHashList_.resize(opList.size());
    operationInfo_->opCoreType_.resize(opList.size());
    operationInfo_->useCVMixPartition_ = useCVMixPartition_;
    for (size_t i = 0; i < opList.size(); i++) {
        operationInfo_->magic2Idx_[opList[i]->GetOpMagic()] = i;
    }
    for (size_t i = 0; i < opList.size(); i++) {
        for (auto &input : opList[i]->GetIOperands()) {
            for (auto &parentOpPtr : input->GetProducers()) {
                if (operationInfo_->magic2Idx_.count(parentOpPtr->GetOpMagic()) == 0) {
                    continue;
                }
                int32_t operationInIdx = operationInfo_->magic2Idx_[parentOpPtr->GetOpMagic()];
                operationInfo_->inGraph_[i].insert(operationInIdx);
                operationInfo_->outGraph_[operationInIdx].insert(i);
            }
        }
    }
    for (size_t i = 0; i < opList.size(); i++) {
        operationInfo_->opHashList_[i] = operationInfo_->GetHash(opList[i]);
        operationInfo_->opCoreType_[i] = OpcodeManager::Inst().GetCoreType(opList[i]->GetOpcode());
    }
    return SUCCESS;
}

inline bool L1CopyInCombine(const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*> &opList,
                            int32_t i, std::vector<std::pair<int32_t, int32_t>> &mergePair)
{
    if (i < 0 || i > static_cast<int32_t>(opList.size())) {
        return false;
    }
    if (opList[i]->GetOOperands().size() == 1U &&
        opList[i]->GetOOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
        for (auto outNode : operationInfo->outGraph_[i]) {
            mergePair.emplace_back(outNode, i);
        }
        return true;
    }
    return false;
}

inline bool AssembleCombine(const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*> &opList,
                            int32_t i, std::vector<std::pair<int32_t, int32_t>> &mergePair)
{
    if (i < 0 || i > static_cast<int32_t>(opList.size())) {
        return false;
    }
    // assemble 特殊处理, assemble到local tensor，需要将这些assemble统一island
    if (opList[i]->GetOpcode() == Opcode::OP_ASSEMBLE) {
        if (opList[i]->GetOOperands().size() == 0) {
            return false;
        }
        // assmemble和其输入绑定
        if (operationInfo->inGraph_[i].size() > 0) {
            mergePair.emplace_back(i, *(operationInfo->inGraph_[i].begin()));
            ALOG_DEBUG_F("Combine %d and %d for Assemble in building SuperNode.",
                         opList[i]->GetOpMagic(), opList[*(operationInfo->inGraph_[i].begin())]->GetOpMagic());
        }
        return true;
    }
    return false;
}

inline bool CopyOutCombine(const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*> &opList,
                           int32_t i, std::vector<std::pair<int32_t, int32_t>> &mergePair)
{
    if (i < 0 || i > static_cast<int32_t>(opList.size())) {
        return false;
    }
    std::vector<int32_t> candidateOpMagic;
    // 所有的copyout操作与其输入绑定
    if (OpcodeManager::Inst().GetOpCalcType(opList[i]->GetOpcode()) == OpCalcType::MOVE_OUT &&
        opList[i]->ProducerOps().size() == 1U) {
        for (auto inputTensor : opList[i]->GetIOperands()) {
            if (inputTensor->GetMemoryTypeOriginal() != MemoryType::MEM_HOST1 &&
                inputTensor->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
                for (auto &producer : inputTensor->GetProducers()) {
                    candidateOpMagic.push_back(producer->GetOpMagic());
                }
            }
        }
        for(auto candidate : candidateOpMagic) {
            if (operationInfo->magic2Idx_.count(candidate) > 0) {
                mergePair.emplace_back(operationInfo->magic2Idx_[candidate], i);
                ALOG_DEBUG_F("Combine %d and %d for CopyOut in building SuperNode.",
                             opList[operationInfo->magic2Idx_[candidate]]->GetOpMagic(), opList[i]->GetOpMagic());
            }
        }
        return true;
    }
    return false;
}

inline bool CopyInCombine(const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*> &opList,
                          int32_t i, std::vector<std::pair<int32_t, int32_t>> &mergePair)
{
    if (i < 0 || i > static_cast<int32_t>(opList.size())) {
        return false;
    }
    // 所有的copyin操作与其输出绑定
    if ((OpcodeManager::Inst().GetOpCalcType(opList[i]->GetOpcode()) == OpCalcType::MOVE_IN ||
         OpcodeManager::Inst().GetOpCalcType(opList[i]->GetOpcode()) == OpCalcType::MOVE_LOCAL) &&
        operationInfo->outGraph_[i].size() == 1) {
        mergePair.emplace_back(i, *(operationInfo->outGraph_[i].begin()));
        ALOG_DEBUG_F("Combine %d and %d for CopyIn in building SuperNode.",
                     opList[i]->GetOpMagic(), opList[*(operationInfo->outGraph_[i].begin())]->GetOpMagic());
        return true;
    }
    return false;
}

inline bool MulAccCombine(const std::shared_ptr<OperationGraphInfo> operationInfo, std::vector<Operation*> &opList,
                          int32_t i, std::vector<std::pair<int32_t, int32_t>> &mergePair)
{
    if (i < 0 || i > static_cast<int32_t>(opList.size())) {
        return false;
    }
    // MulAcc需要与其输入mul绑定
    if (OpcodeManager::Inst().GetOpCalcType(opList[i]->GetOpcode()) == OpCalcType::MATMUL) {
        for (auto inOp : operationInfo->inGraph_[i]) {
            if (OpcodeManager::Inst().GetOpCalcType(opList[inOp]->GetOpcode()) == OpCalcType::MATMUL) {
                mergePair.emplace_back(i, inOp);
                ALOG_DEBUG_F("Combine %d and %d for MulAcc in building SuperNode.",
                             opList[i]->GetOpMagic(), opList[inOp]->GetOpMagic());
            }
        }
        return true;
    }
    return false;
}

Status IsoPartitioner::BuildSuperNodeGraph()
{
    std::vector<Operation*> &opList = operationInfo_->opList_;
    if (opList.size() != operationInfo_->inGraph_.size() || opList.size() != operationInfo_->outGraph_.size()) {
        ALOG_ERROR_F("Operation inGraph and outGraph have not been initialized.");
        return FAILED;
    }
    std::vector<std::pair<int32_t, int32_t>> mergePair;
    for (size_t i = 0; i < opList.size(); i++) {
        if (L1CopyInCombine(operationInfo_, opList, i, mergePair)) {
            continue;
        }
        if (AssembleCombine(operationInfo_, opList, i, mergePair)) {
            continue;
        }
        if (CopyOutCombine(operationInfo_, opList, i, mergePair)) {
            continue;
        }
        if (CopyInCombine(operationInfo_, opList, i, mergePair)) {
            continue;
        }
        if (MulAccCombine(operationInfo_, opList, i, mergePair)) {
            continue;
        }
    }
    superNodeInfo_ = std::make_shared<NodeGraphInfo>();
    if (superNodeInfo_ == nullptr) {
        ALOG_ERROR_F("Create SuperNodeInfo failed.");
        return FAILED;
    }
    if (superNodeInfo_->Build(operationInfo_, mergePair, !useCVMixPartition_) != SUCCESS) {
        ALOG_ERROR_F("Build SuperNodeInfo Failed.");
        return FAILED;
    }
    return SUCCESS;
}

int32_t NodeGraphInfo::FindParent(std::vector<int32_t> &parent, int32_t i)
{
    if (i < 0 || i >= static_cast<int32_t>(parent.size())) {
        ALOG_ERROR_F("Call FindParent with illegal parameter %d.", i);
        return -1;
    }
    if (parent[i] == i) {
        return i;
    }
    std::vector<int32_t> searchPath;
    int32_t currIdx = i;
    while(parent[currIdx] != currIdx) {
        searchPath.push_back(currIdx);
        currIdx = parent[currIdx];
        if (currIdx < 0 || currIdx >= static_cast<int32_t>(parent.size())) {
            ALOG_ERROR_F("Find illegal parameter %d in FindParent.", currIdx);
            return -1;
        }
        if (searchPath.size() > (parent.size() + 1)) { ALOG_ERROR_F("Find loop in FindParent."); return -1; }
    }
    for (auto parentIdx : searchPath) {
        parent[parentIdx] = currIdx;
    }
    return currIdx;
}

Status NodeGraphInfo::MergeSrcToDstIsland(const std::shared_ptr<OperationGraphInfo> operationGraphInfo,
                                          std::vector<int32_t> &parent, int32_t src, int32_t dst)
{
    int32_t srcParent = FindParent(parent, src);
    int32_t dstParent = FindParent(parent, dst);
    if (srcParent == -1 || dstParent == -1) {
        ALOG_ERROR_F("Merge node in the disjoint set failed.");
        return FAILED;
    }
    std::set<OpCoreType> coreTypes{operationGraphInfo->opCoreType_[src], operationGraphInfo->opCoreType_[dst],
                                   operationGraphInfo->opCoreType_[srcParent],
                                   operationGraphInfo->opCoreType_[dstParent]};
    if (operationGraphInfo->CoreTypeMergeable(coreTypes)) {
        if (srcParent > dstParent) {
            parent[srcParent] = dstParent;
        } else {
            parent[dstParent] = srcParent;
        }
    } else {
        ALOG_ERROR_F("Try to merge not mergeable operation pair.");
        return FAILED;
    }
    return SUCCESS;
}

std::vector<int32_t> NodeInnerExpand(const std::shared_ptr<OperationGraphInfo> operationGraphInfo, std::vector<int32_t> &nodeOps)
{
    std::vector<int32_t> frontBackVisitedOp;
    int32_t minOpIdx = static_cast<int32_t>(operationGraphInfo->opList_.size());
    int32_t maxOpIdx = -1;
    for (int32_t opIdx : nodeOps) {
        minOpIdx = opIdx < minOpIdx ? opIdx : minOpIdx;
        maxOpIdx = opIdx > maxOpIdx ? opIdx : maxOpIdx;
    }
    std::unordered_set<int32_t> frontVisitedOp;
    std::vector<int32_t> frontVisitStack(nodeOps);
    while (frontVisitStack.size() > 0) {
        int32_t opIdx = frontVisitStack.back();
        frontVisitStack.pop_back();
        if (frontVisitedOp.count(opIdx) > 0) {
            continue;
        }
        frontVisitedOp.insert(opIdx);
        for (int32_t nextOpIdx : operationGraphInfo->outGraph_[opIdx]) {
            if (nextOpIdx <= maxOpIdx) {
                frontVisitStack.push_back(nextOpIdx);
            }
        }
    }
    std::unordered_set<int32_t> backVisitedOp;
    std::vector<int32_t> backVisitStack(nodeOps);
    while (backVisitStack.size() > 0) {
        int32_t opIdx = backVisitStack.back();
        backVisitStack.pop_back();
        if (backVisitedOp.count(opIdx) > 0) {
            continue;
        }
        if (frontVisitedOp.count(opIdx) > 0) {
            frontBackVisitedOp.push_back(opIdx);
        }
        backVisitedOp.insert(opIdx);
        for (int32_t prevOpIdx : operationGraphInfo->inGraph_[opIdx]) {
            if (prevOpIdx >= minOpIdx) {
                backVisitStack.push_back(prevOpIdx);
            }
        }
    }
    return frontBackVisitedOp;
}

Status NodeGraphInfo::AvoidLoop(const std::shared_ptr<OperationGraphInfo> operationGraphInfo,
                                std::vector<int32_t> &parent, std::vector<std::vector<int32_t>> &node2Op, bool &updated)
{
    std::vector<Operation*> &opList = operationGraphInfo->opList_;
    std::vector<int32_t> parentToNodes(opList.size(), -1);
    updated = false;
    node2Op.clear();
    for (int32_t i = 0; i < static_cast<int32_t>(opList.size()); i++) {
        int32_t currParent = FindParent(parent, i);
        if (currParent == -1) { 
            ALOG_ERROR_F("Find parent in the union set failed.");
            return FAILED; 
        }
        if (currParent == i) {
            parentToNodes[i] = node2Op.size();
            node2Op.push_back(std::vector<int32_t>());
        }
    }
    for (int32_t i = 0; i < static_cast<int32_t>(operationGraphInfo->opList_.size()); i++) {
        int32_t currParent = FindParent(parent, i);
        if (currParent == -1) {
            ALOG_ERROR_F("Find parent in the union set failed.");
            return FAILED;
        }
        int32_t nodeIdx = parentToNodes[currParent];
        node2Op[nodeIdx].push_back(i);
    }
    for (size_t nodeIdx = 0; nodeIdx < node2Op.size(); nodeIdx++) {
        std::vector<int32_t> expandNode = NodeInnerExpand(operationGraphInfo, node2Op[nodeIdx]);
        if (expandNode.size() == node2Op[nodeIdx].size() || expandNode.size() == 0) {
            continue;
        }
        updated = true;
        for (size_t opIdx = 1; opIdx < expandNode.size(); opIdx++) {
            if (MergeSrcToDstIsland(operationGraphInfo, parent, expandNode[0], expandNode[opIdx]) != SUCCESS) {
                ALOG_ERROR_F("Build the disjoint set failed.");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status NodeGraphInfo::Build(const std::shared_ptr<OperationGraphInfo> operationGraphInfo,
                            const std::vector<std::pair<int32_t, int32_t>> &mergePair, bool markIsCube)
{
    std::vector<Operation*> &opList = operationGraphInfo->opList_;
    std::vector<int32_t> parent(opList.size());
    for (size_t i = 0; i < opList.size(); i++) {
        parent[i] = i;
    }
    for (auto &pr : mergePair) {
        if (MergeSrcToDstIsland(operationGraphInfo, parent, pr.first, pr.second) != SUCCESS) {
            ALOG_ERROR_F("Build the disjoint set failed.");
            return FAILED;
        }
    }
    bool updated = true;
    while (updated) {
        updated = false;
        if (AvoidLoop(operationGraphInfo, parent, node2Op_, updated) != SUCCESS) {
            ALOG_ERROR_F("Avoid loop in building node failed");
            return FAILED;
        }
    }
    op2Node_.resize(opList.size());
    nodeCycles_.resize(opList.size());
    for (size_t nodeIdx = 0; nodeIdx < node2Op_.size(); nodeIdx++) {
        nodeCycles_[nodeIdx] = 0;
        for (size_t opNodeIdx = 0; opNodeIdx < node2Op_[nodeIdx].size(); opNodeIdx++) {
            int32_t opIdx = node2Op_[nodeIdx][opNodeIdx];
            op2Node_[opIdx] = nodeIdx;
            nodeCycles_[nodeIdx] += operationGraphInfo->opList_[opIdx]->GetLatency();
        }
    } 
    BuildInOutGraph(operationGraphInfo, markIsCube);
    return SUCCESS;
}

Status NodeGraphInfo::BuildInOutGraph(const std::shared_ptr<OperationGraphInfo> operationGraphInfo, bool markIsCube)
{
    nodeInGraph_.resize(node2Op_.size());
    nodeOutGraph_.resize(node2Op_.size());
    nodeInGraphList_.resize(node2Op_.size());
    nodeOutGraphList_.resize(node2Op_.size());
    for (size_t i = 0; i < node2Op_.size(); i++) {
        std::vector<int32_t> &currNode = node2Op_[i];
        for (int32_t opIdx : currNode) {
            for (int32_t publisherOpIdx : operationGraphInfo->inGraph_[opIdx]) {
                int32_t publisherNodeIdx = op2Node_[publisherOpIdx];
                if (publisherNodeIdx != static_cast<int32_t>(i)) {
                    nodeInGraph_[i].insert(publisherNodeIdx);
                    nodeOutGraph_[publisherNodeIdx].insert(i);
                }
            }
        }
    }
    for (size_t i = 0; i < node2Op_.size(); i++) {
        nodeInGraphList_[i].insert(nodeInGraphList_[i].begin(), nodeInGraph_[i].begin(), nodeInGraph_[i].end());
        nodeOutGraphList_[i].insert(nodeOutGraphList_[i].begin(), nodeOutGraph_[i].begin(), nodeOutGraph_[i].end());
    }
    nodeCoreType_.resize(node2Op_.size());
    nodeMergeable_.resize(node2Op_.size());
    for (size_t i = 0; i < node2Op_.size(); i++) {
        nodeCoreType_[i] = OpCoreType::AIV;
        for (int32_t opIdx : node2Op_[i]) {
            if (operationGraphInfo->opCoreType_[opIdx] != OpCoreType::ANY) {
                nodeCoreType_[i] = operationGraphInfo->opCoreType_[opIdx];
                break;
            }
        }
        nodeMergeable_[i] = !(node2Op_[i].size() == 1 &&
                              operationGraphInfo->opList_[node2Op_[i][0]]->GetOpcode() == Opcode::OP_RESHAPE &&
                              nodeInGraph_[i].size() > 1 && nodeOutGraph_[i].size() > 1);
        if (!markIsCube) {
            continue;
        }
        bool isCube = false;
        for (auto j : node2Op_[i]) {
            if (operationGraphInfo->opCoreType_[j] == OpCoreType::AIC) {
                isCube = true;
                break;
            }
        }
        for (auto j : node2Op_[i]) {
            operationGraphInfo->opList_[j]->SetAttribute(OpAttributeKey::isCube, isCube);
        }
    }
    return SUCCESS;
}

uint64_t IsoPartitioner::CombineHash(const uint64_t h1, const uint64_t h2) const
{
    const uint64_t mask52 = 0xFFFFFFFFFFFFF;
    const uint64_t maskXor = 0x12345678;
    const uint64_t prime = 881;
    uint64_t h1Trunc = h1 & mask52;
    uint64_t h2Trunc = h2 & mask52;
    uint64_t h3 = (h1Trunc * prime) + (h2Trunc ^ maskXor);
    return h3;
}

bool OperationGraphInfo::CoreTypeMergeable(const std::set<OpCoreType> &coreTypes) const
{
    if (useCVMixPartition_ || coreTypes.size() == 1) {
        return true;
    }
    const size_t maxSeperateCoreNum = 2;
    if (coreTypes.size() > maxSeperateCoreNum) {
        return false;
    }
    if (coreTypes.size() == maxSeperateCoreNum) {
        auto firstType = *coreTypes.begin();
        auto secondType = *(++coreTypes.begin());
        if (firstType == OpCoreType::AICPU || secondType == OpCoreType::AICPU) {
            return false;
        }
        if (firstType == OpCoreType::ANY || secondType == OpCoreType::ANY) {
            return true;
        }
    }
    return false;
}

std::vector<std::pair<int32_t, int32_t>> IsoPartitioner::GetReduceNodeMergePair() const
{
    std::unordered_set<Opcode> reduceType{Opcode::OP_PAIRMAX, Opcode::OP_PAIRSUM};
    std::vector<Operation*> &opList = operationInfo_->opList_;
    std::vector<std::pair<int32_t, int32_t>> mergePair;
    for (size_t i = 0; i < opList.size(); i++) {
        if (OpcodeManager::Inst().GetOpCalcType(opList[i]->GetOpcode()) == OpCalcType::MATMUL) {
            for (auto inOp : operationInfo_->inGraph_[i]) {
                if (OpcodeManager::Inst().GetOpCalcType(opList[inOp]->GetOpcode()) == OpCalcType::MATMUL) {
                    mergePair.emplace_back(i, inOp);
                    ALOG_DEBUG_F("Combine %d and %d for MulAcc in building ReduceNode.",
                                 opList[i]->GetOpMagic(), opList[inOp]->GetOpMagic());
                }
            }
            continue;
        }
        if (reduceType.count(opList[i]->GetOpcode()) > 0 && operationInfo_->outGraph_[i].size() == 1 &&
            opList[i]->GetOpcode() == opList[*(operationInfo_->outGraph_[i].begin())]->GetOpcode()) {
            mergePair.emplace_back(i, *(operationInfo_->outGraph_[i].begin()));
            ALOG_DEBUG_F("Combine %d and %d for Reduce AIV Operation in building ReduceNode.",
                         opList[i]->GetOpMagic(), opList[*(operationInfo_->outGraph_[i].begin())]->GetOpMagic());
        }
    }
    return mergePair;
}

Status IsoPartitioner::BuildReduceNodeHash(std::shared_ptr<NodeGraphInfo> reduceNodeInfo)
{
    std::vector<uint64_t> reduceNodeHashListFront(reduceNodeInfo->node2Op_.size(), 0);
    std::vector<uint64_t> reduceNodeHashListBack(reduceNodeInfo->node2Op_.size(), 0);
    std::vector<uint64_t> reduceNodeHashList(reduceNodeInfo->node2Op_.size(), 0);
    if (operationInfo_->opHashList_.size() != reduceNodeInfo->op2Node_.size()) {
        ALOG_ERROR_F("Operation number mismatch in OperationInfo and ReduceNodeInfo.");
        return FAILED;
    }
    for (size_t i = 0; i < reduceNodeInfo->node2Op_.size(); i++) {
        reduceNodeHashList[i] = 0;
        for (int32_t opInNode : reduceNodeInfo->node2Op_[i]) {
            reduceNodeHashList[i] = CombineHash(reduceNodeHashList[i], operationInfo_->opHashList_[opInNode]);
        }
    }

    for (size_t i = 0; i < operationInfo_->opList_.size(); i++) {
        int32_t nodeIdx = reduceNodeInfo->op2Node_[i];
        if (reduceNodeHashListFront[nodeIdx] != 0) {
            continue;
        }
        reduceNodeHashListFront[nodeIdx] = reduceNodeHashList[nodeIdx];
        for (int32_t j : reduceNodeInfo->nodeInGraph_[nodeIdx]) {
            reduceNodeHashListFront[nodeIdx] =
                CombineHash(reduceNodeHashListFront[nodeIdx], reduceNodeHashListFront[j]);
        }
    }

    for (int32_t i = static_cast<int32_t>(operationInfo_->opList_.size()) - 1; i >= 0; i--) {
        int32_t nodeIdx = reduceNodeInfo->op2Node_[i];
        if (reduceNodeHashListBack[nodeIdx] != 0) {
            continue;
        }
        reduceNodeHashListBack[nodeIdx] = reduceNodeHashList[nodeIdx];
        for (int32_t j : reduceNodeInfo->nodeOutGraph_[nodeIdx]) {
            reduceNodeHashListBack[nodeIdx] = CombineHash(reduceNodeHashListBack[nodeIdx], reduceNodeHashListBack[j]);
        }
    }

    for (size_t i = 0; i < reduceNodeInfo->node2Op_.size(); i++) {
        reduceNodeHashList[i] = CombineHash(reduceNodeHashListFront[i], reduceNodeHashListBack[i]);
    }
    reduceNodeInfo->nodeHashList_ = reduceNodeHashList;
    return SUCCESS;
}

Status IsoPartitioner::BuildBalanceOpHash(std::vector<uint64_t> &opHashList)
{
    std::vector<std::pair<int32_t, int32_t>> mergePair = GetReduceNodeMergePair();
    std::shared_ptr<NodeGraphInfo> reduceNodeInfo = std::make_shared<NodeGraphInfo>();
    if (reduceNodeInfo == nullptr) {
        ALOG_ERROR_F("Create ReduceNodeInfo failed.");
        return FAILED;
    }
    reduceNodeInfo->Build(operationInfo_, mergePair, false);
    BuildReduceNodeHash(reduceNodeInfo);
    std::vector<uint64_t> opHashListFrontBack(operationInfo_->opList_.size(), 0);
    for (size_t i = 0; i < reduceNodeInfo->node2Op_.size(); i++) {
        if (reduceNodeInfo->node2Op_[i].size() == 1) {
            opHashListFrontBack[reduceNodeInfo->node2Op_[i][0]] = reduceNodeInfo->nodeHashList_[i];
            continue;
        }
        std::vector<int32_t> &localOps = reduceNodeInfo->node2Op_[i];
        std::unordered_map<int32_t, uint64_t> localFront;
        std::unordered_map<int32_t, uint64_t> localBack;
        for (size_t localIdx = 0; localIdx < localOps.size(); localIdx++) {
            int32_t localOpIdx = localOps[localIdx];
            localFront[localOpIdx] = operationInfo_->opHashList_[localOpIdx];
            for (int32_t publisherOpIdx : operationInfo_->inGraph_[localOpIdx]) {
                if (localFront.count(publisherOpIdx) > 0) {
                    localFront[localOpIdx] = CombineHash(localFront[localOpIdx], localFront[publisherOpIdx]);
                }
            }
        }
        for (int32_t localIdx = static_cast<int32_t>(localOps.size()) - 1; localIdx >= 0; localIdx--) {
            int32_t localOpIdx = localOps[localIdx];
            localBack[localOpIdx] = operationInfo_->opHashList_[localOpIdx];
            for (int32_t consumerOpIdx : operationInfo_->outGraph_[localOpIdx]) {
                if (localBack.count(consumerOpIdx) > 0) {
                    localBack[localOpIdx] = CombineHash(localBack[localOpIdx], localBack[consumerOpIdx]);
                }
            }
        }
        for (size_t localIdx = 0; localIdx < localOps.size(); localIdx++) {
            int32_t localOpIdx = localOps[localIdx];
            uint64_t localHash = CombineHash(localFront[localOpIdx], localBack[localOpIdx]);
            opHashListFrontBack[localOpIdx] = CombineHash(reduceNodeInfo->nodeHashList_[i], localHash);
        }
    }
    opHashList = opHashListFrontBack;
    return SUCCESS;
}

Status IsoPartitioner::BuildHashValues()
{
    std::vector<uint64_t> opHashList;
    if (useReduceBalanceHash_) {
        if (BuildBalanceOpHash(opHashList) != SUCCESS) {
            ALOG_ERROR_F("BuildBalanceOpHash failed.");
            return FAILED;
        }
    } else {
        std::vector<uint64_t> opHashListFront(operationInfo_->opList_.size(), 0);
        std::vector<uint64_t> opHashListBack(operationInfo_->opList_.size(), 0);
        std::vector<uint64_t> opHashListFrontBack(operationInfo_->opList_.size(), 0);
        for (size_t i = 0; i < operationInfo_->opList_.size(); i++) {
            opHashListFront[i] = operationInfo_->opHashList_[i];
            for (int32_t j : operationInfo_->inGraph_[i]) {
                opHashListFront[i] = CombineHash(opHashListFront[i], opHashListFront[j]);
            }
        }
        for (int32_t i = static_cast<int32_t>(operationInfo_->opList_.size() - 1); i >= 0; i--) {
            opHashListBack[i] = operationInfo_->opHashList_[i];
            for (int32_t j : operationInfo_->outGraph_[i]) {
                std::set<OpCoreType> coreTypes{operationInfo_->opCoreType_[i], operationInfo_->opCoreType_[j]};
                if (!operationInfo_->CoreTypeMergeable(coreTypes)) {
                    continue;
                }
                opHashListBack[i] = CombineHash(opHashListBack[i], opHashListBack[j]);
            }
        }
        for (size_t i = 0; i < operationInfo_->opList_.size(); i++) {
            opHashListFrontBack[i] = CombineHash(opHashListFront[i], opHashListBack[i]);
        }
        opHashList.swap(opHashListFrontBack);
    }
    if (superNodeInfo_->op2Node_.size() != operationInfo_->opList_.size()) {
        ALOG_ERROR_F("Operation number mismatch in SuperNodeInfo and OperationInfo.");
        return FAILED;
    }
    int32_t numNode = superNodeInfo_->node2Op_.size();
    superNodeInfo_->nodeHashList_.resize(numNode);
    for (int32_t i = 0; i < numNode; i++) {
        superNodeInfo_->nodeHashList_[i] = 0;
        for (int32_t opIdx : superNodeInfo_->node2Op_[i]) {
            superNodeInfo_->nodeHashList_[i] = CombineHash(superNodeInfo_->nodeHashList_[i], opHashList[opIdx]);
        }
    }
    for (int32_t i = 0; i < numNode; i++) {
        superNodeInfo_->hash2NodeMap_[superNodeInfo_->nodeHashList_[i]].push_back(i);
    }
    return SUCCESS;
}

Status IsoPartitioner::BuildIsomorphismGroups()
{
    std::vector<int32_t> idxInLinkNum;
    std::deque<int32_t> zeroInQueue;
    std::unordered_set<int32_t> currentNodeSet;
    for (size_t i = 0; i < superNodeInfo_->nodeInGraph_.size(); i++) {
        idxInLinkNum.push_back(superNodeInfo_->nodeInGraph_[i].size());
        if (superNodeInfo_->nodeInGraph_[i].size() == 0) {
            zeroInQueue.push_front(i);
        }
    }
    currentNodeSet.clear();
    while (zeroInQueue.size() > 0) {
        int32_t currIdx = zeroInQueue[0];
        zeroInQueue.pop_front();
        uint64_t hs = superNodeInfo_->nodeHashList_[currIdx];
        std::vector<int32_t> &expandCandidate = superNodeInfo_->hash2NodeMap_[hs];
        bool isLegalStart = true;
        for (size_t i = 0; i < expandCandidate.size(); i++) {
            if (idxInLinkNum[expandCandidate[i]] != 0 || currentNodeSet.count(expandCandidate[i]) > 0) {
                isLegalStart = false;
                break;
            }
        }
        if (!isLegalStart) {
            continue;
        }
        std::shared_ptr<IsomorphismGraphGroup> currentGraphGroup = std::make_shared<IsomorphismGraphGroup>();
        if (currentGraphGroup == nullptr) {
            ALOG_ERROR_F("Create current IsomorphismGraphGroup failed.");
            return FAILED;
        }
        if (currentGraphGroup->BuildGraphGroup(operationInfo_, superNodeInfo_, expandCandidate, currentNodeSet,
                                               idxInLinkNum, zeroInQueue) != SUCCESS) {
            ALOG_ERROR_F("Build initial IsomorphismGraphGroup failed.");
            return FAILED;
        }
        if (currentGraphGroup->GetMergeable()) {
            if (currentGraphGroup->ExpandIsoGraphs(currentNodeSet, idxInLinkNum,
                                                   zeroInQueue, cycleThreshold_) != SUCCESS) {
                ALOG_ERROR_F("Expand the isomorphism group failed.");
                return FAILED;
            }
        }
        isoSubGroups_.push_back(currentGraphGroup);
    }
    return SUCCESS;
}

Status IsomorphismGraphGroup::BuildGraphGroup(std::shared_ptr<OperationGraphInfo> operationInfo,
                                              std::shared_ptr<NodeGraphInfo> superNodeInfo,
                                              std::vector<int32_t> &expandCandidate,
                                              std::unordered_set<int32_t> &currentNodeSet,
                                              std::vector<int32_t> &idxInLinkNum, std::deque<int32_t> &zeroInQueue)
{
    operationInfo_ = operationInfo;
    superNodeInfo_ = superNodeInfo;
    subVisitedNodeSet_.clear();
    subVisitedNodeSet_.insert(expandCandidate.begin(), expandCandidate.end());
    currentNodeSet.insert(expandCandidate.begin(), expandCandidate.end());
    for (int32_t nodeIdx : expandCandidate) {
        if (InLinkCountDelete(nodeIdx, idxInLinkNum, zeroInQueue) != SUCCESS) {
            ALOG_ERROR_F("In-link count delete failed.");
            return FAILED;
        }
        std::shared_ptr<SubGraph> sgPtr = std::make_shared<SubGraph>(operationInfo, superNodeInfo);
        if (sgPtr == nullptr) {
            ALOG_ERROR_F("Create SubGraph failed.");
            return FAILED;
        }
        sgPtr->AddNode(nodeIdx);
        isoGraphs_.push_back(sgPtr);
    }
    mergeable_ = superNodeInfo_->nodeMergeable_[expandCandidate[0]];
    return SUCCESS;
}

Status IsomorphismGraphGroup::InLinkCountDelete(int32_t nodeIdx, std::vector<int32_t> &idxInLinkNum,
                                                std::deque<int32_t> &zeroInQueue)
{
    for (int32_t consumer : superNodeInfo_->nodeOutGraph_[nodeIdx]) {
        if (consumer < 0 || consumer >= static_cast<int32_t>(idxInLinkNum.size())) {
            ALOG_ERROR_F("Consumer index illegal in InLinkCountDelete.");
            return FAILED;
        }
        idxInLinkNum[consumer] -= 1;
        if (idxInLinkNum[consumer] == 0) {
            zeroInQueue.push_back(consumer);
        }
        if (idxInLinkNum[consumer] < 0) {
            ALOG_ERROR_F("Negative in-link count in InLinkCountDelete.");
            return FAILED;
        }
    }
    return SUCCESS;
}

size_t IsomorphismGraphGroup::Size() const
{
    return isoGraphs_.size();
}

bool IsomorphismGraphGroup::GetMergeable()
{
    return mergeable_ != 0;
}

void SubGraph::AddNode(int32_t nodeIdx)
{
    if (coreType_ == OpCoreType::ANY) {
        coreType_ = superNodeInfo_->nodeCoreType_[nodeIdx];
    }
    nodeList_.push_back(nodeIdx);
    nodeSet_.insert(nodeIdx);
    cycle_ += superNodeInfo_->GetNodeCycle(nodeIdx);
}

Status IsomorphismGraphGroup::ExpandIsoGraphs(std::unordered_set<int32_t> &currentNodeSet,
                                              std::vector<int32_t> &idxInLinkNum, std::deque<int32_t> &zeroInQueue,
                                              int32_t cycleThreshold)
{
    size_t expandNodeIdx = 0;
    size_t expandLinkIdx = 0;
    GraphExtendResult extendStatus = GraphExtendResult::EXTEND_SUCCESS;
    while (extendStatus != GraphExtendResult::EXTEND_NODE_EXHAUST) {
        std::vector<int32_t> expandCandidate;
        for (size_t i = 0; i < isoGraphs_.size(); i++) {
            int32_t newLeaf = isoGraphs_[i]->GetExpandCandidate(expandNodeIdx, expandLinkIdx, extendStatus);
            expandCandidate.push_back(newLeaf);
        }
        if (extendStatus == GraphExtendResult::EXTEND_LINK_EXHAUST) {
            expandNodeIdx += 1;
            expandLinkIdx = 0;
            continue;
        } else if (extendStatus == GraphExtendResult::EXTEND_NODE_EXHAUST) {
            break;
        } else if (!IsLegalIsoGraphExtender(expandCandidate, currentNodeSet, idxInLinkNum, cycleThreshold)) {
            expandLinkIdx += 1;
            continue;
        }
        expandLinkIdx += 1;
        for (size_t i = 0; i < expandCandidate.size(); i++) {
            isoGraphs_[i]->AddNode(expandCandidate[i]);
            currentNodeSet.insert(expandCandidate[i]);
            subVisitedNodeSet_.insert(expandCandidate[i]);
            if (InLinkCountDelete(expandCandidate[i], idxInLinkNum, zeroInQueue) != SUCCESS) {
                ALOG_ERROR_F("In-link count delete failed.");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

bool SubGraph::HasNode(int32_t nodeIdx) const
{
    if (nodeSet_.count(nodeIdx) > 0) {
        return true;
    }
    return false;
}

bool IsomorphismGraphGroup::IsLegalIsoGraphExtender(std::vector<int32_t> &expandCandidate,
                                                    std::unordered_set<int32_t> &currentNodeSet,
                                                    std::vector<int32_t> &idxInLinkNum, int32_t cycleThreshold)
{
    if (!superNodeInfo_->nodeMergeable_[expandCandidate[0]]) {
        return false;
    }
    if (superNodeInfo_->hash2NodeMap_[superNodeInfo_->nodeHashList_[expandCandidate[0]]].size() != isoGraphs_.size()) {
        return false;
    }
    for (size_t i = 0; i < expandCandidate.size(); i++) {
        int32_t candidate = expandCandidate[i];
        if (idxInLinkNum[candidate] != 0) {
            return false;
        }
        if (superNodeInfo_->nodeHashList_[candidate] != superNodeInfo_->nodeHashList_[expandCandidate[0]]) {
            return false;
        }
        if (currentNodeSet.count(candidate) > 0) {
            return false;
        }
        for (int32_t toNode : superNodeInfo_->nodeOutGraph_[candidate]) {
            if (subVisitedNodeSet_.count(toNode) > 0 && !isoGraphs_[i]->HasNode(toNode)) {
                return false;
            }
        }
    }
    int32_t newLatency = isoGraphs_[0]->GetLatency() + superNodeInfo_->GetNodeCycle(expandCandidate[0]);
    if (newLatency > cycleThreshold) {
        return false;
    }
    std::set<int32_t> candSet(expandCandidate.begin(), expandCandidate.end());
    if (candSet.size() != expandCandidate.size()) {
        return false;
    }
    OpCoreType graphIsCube = superNodeInfo_->nodeCoreType_[isoGraphs_[0]->GetNodeList()[0]];
    OpCoreType candidateIsCube = superNodeInfo_->nodeCoreType_[expandCandidate[0]];
    std::set<OpCoreType> coreTypes{graphIsCube, candidateIsCube};
    return operationInfo_->CoreTypeMergeable(coreTypes);
}

int32_t SubGraph::GetExpandCandidate(size_t expandNodeIdx, size_t expandLinkIdx, GraphExtendResult &res)
{
    if (expandNodeIdx >= nodeList_.size()) {
        res = GraphExtendResult::EXTEND_NODE_EXHAUST;
        return 0;
    }
    if (expandLinkIdx >= superNodeInfo_->nodeOutGraph_[nodeList_[expandNodeIdx]].size()) {
        res = GraphExtendResult::EXTEND_LINK_EXHAUST;
        return 0;
    }
    res = GraphExtendResult::EXTEND_SUCCESS;
    int32_t value = superNodeInfo_->nodeOutGraphList_[nodeList_[expandNodeIdx]][expandLinkIdx];
    return value;
}

const std::vector<int32_t> &SubGraph::GetNodeList()
{
    return nodeList_;
}

void SubGraph::BuildInOutSet()
{
    inNodes_.clear();
    outNodes_.clear();
    for (int32_t nodeIdx : nodeList_) {
        for (int32_t inIdx : superNodeInfo_->nodeInGraph_[nodeIdx]) {
            if (nodeSet_.count(inIdx) == 0) {
                inNodes_.insert(inIdx);
            }
        }
        for (int32_t outIdx : superNodeInfo_->nodeOutGraph_[nodeIdx]) {
            if (nodeSet_.count(outIdx) == 0) {
                outNodes_.insert(outIdx);
            }
        }
    }
}

std::shared_ptr<SubGraph> IsomorphismGraphGroup::GetSubGraph(int32_t idx)
{
    if (idx < 0 || idx >= static_cast<int32_t>(isoGraphs_.size())) {
        return nullptr;
    }
    return isoGraphs_[idx];
}

int32_t SubGraph::GetLatency() const
{
    return cycle_;
}

int32_t IsomorphismGraphGroup::GetLatency() const
{
    if (isoGraphs_.size() == 0) {
        return 0;
    }
    return isoGraphs_[0]->GetLatency();
}

void IsomorphismGraphGroup::Clear()
{
    isoGraphs_.clear();
    mergeable_ = true;
    operationInfo_ = nullptr;
    superNodeInfo_ = nullptr;
}

Status IsoPartitioner::IsomorphismGroupMergePrepare(std::vector<std::pair<int32_t, int32_t>> &isoSubIdxs,
                                                    std::vector<std::set<int32_t>> &isoInGraph,
                                                    std::vector<std::set<int32_t>> &isoOutGraph,
                                                    std::vector<std::vector<int32_t>> &isoNodeList,
                                                    std::vector<int32_t> &isoIdx2color)
{
    isoSubIdxs.resize(superNodeInfo_->nodeInGraph_.size());
    isoInGraph.resize(isoSubGroups_.size());
    isoOutGraph.resize(isoSubGroups_.size());
    for (size_t i = 0; i < isoSubGroups_.size(); i++) {
        for (size_t j = 0; j < isoSubGroups_[i]->Size(); j++) {
            for (int32_t nodeIdx : isoSubGroups_[i]->GetSubGraph(j)->GetNodeList()) {
                if (nodeIdx < 0 || nodeIdx >= static_cast<int32_t>(superNodeInfo_->nodeInGraph_.size())) {
                    ALOG_ERROR_F("NodeIdx illegal in IsomorphismGroupMergePrepare.");
                    return FAILED;
                }
                isoSubIdxs[nodeIdx] = std::pair<int32_t, int32_t>{i, j};
            }
            isoSubGroups_[i]->GetSubGraph(j)->BuildInOutSet();
        }
    }
    for (size_t i = 0; i < isoSubGroups_.size(); i++) {
        for (size_t j = 0; j < isoSubGroups_[i]->Size(); j++) {
            isoSubGroups_[i]->GetSubGraph(j)->mergeHistoryIsoSub_.insert(std::pair<int32_t, int32_t>{i, j});
            for (int32_t nodeIdx : isoSubGroups_[i]->GetSubGraph(j)->inNodes_) {
                if (nodeIdx < 0 || nodeIdx >= static_cast<int32_t>(superNodeInfo_->nodeInGraph_.size())) {
                    ALOG_ERROR_F("NodeIdx illegal in IsomorphismGroupMergePrepare.");
                    return FAILED;
                }
                if (isoSubIdxs[nodeIdx].first != static_cast<int32_t>(i)) {
                    isoInGraph[i].insert(isoSubIdxs[nodeIdx].first);
                }
            }
            for (int32_t nodeIdx : isoSubGroups_[i]->GetSubGraph(j)->outNodes_) {
                if (nodeIdx < 0 || nodeIdx >= static_cast<int32_t>(superNodeInfo_->nodeInGraph_.size())) {
                    ALOG_ERROR_F("NodeIdx illegal in IsomorphismGroupMergePrepare.");
                    return FAILED;
                }
                if (isoSubIdxs[nodeIdx].first != static_cast<int32_t>(i)) {
                    isoOutGraph[i].insert(isoSubIdxs[nodeIdx].first);
                }
            }
        }
    }
    isoNodeList.resize(isoSubGroups_.size());
    isoIdx2color.resize(isoSubGroups_.size());
    for (int32_t i = 0; i < static_cast<int32_t>(isoSubGroups_.size()); i++) {
        isoIdx2color[i] = i;
        isoNodeList[i].push_back(i);
    }
    return SUCCESS;
}

std::vector<int32_t> IsoPartitioner::GetCandidateMergeColors(int32_t currColor,
                                                             std::vector<std::set<int32_t>> &isoInGraph,
                                                             std::vector<std::set<int32_t>> &isoOutGraph,
                                                             std::vector<std::vector<int32_t>> &isoNodeList,
                                                             std::vector<int32_t> &isoIdx2color, bool nonIsoGraphsMerge)
{
    std::set<int32_t> inputColors;
    std::set<int32_t> selfNodes;
    std::set<int32_t> outputColors;
    if (!isoSubGroups_[currColor]->mergeable_) {
        return {};
    }
    if (nonIsoGraphsMerge && isoSubGroups_[currColor]->Size() != 1) {
        return {};
    }
    if (!nonIsoGraphsMerge && isoSubGroups_[currColor]->Size() <= 1) {
        return {};
    }
    selfNodes.insert(isoNodeList[currColor].begin(), isoNodeList[currColor].end());
    for (size_t idx : isoNodeList[currColor]) {
        for (size_t inIdx : isoInGraph[idx]) {
            inputColors.insert(isoIdx2color[inIdx]);
        }
        for (size_t outIdx : isoOutGraph[idx]) {
            outputColors.insert(isoIdx2color[outIdx]);
        }
    }
    inputColors.erase(currColor);
    outputColors.erase(currColor);
    std::vector<int32_t> candidateMergeColors;
    if (inputColors.size() == 1 && isoSubGroups_[*inputColors.begin()]->mergeable_) {
        candidateMergeColors.push_back(*inputColors.begin());
    }
    if (outputColors.size() == 1 && isoSubGroups_[*outputColors.begin()]->mergeable_) {
        candidateMergeColors.push_back(*outputColors.begin());
    }
    std::vector<int32_t> mergeColors;
    for (int32_t candidate : candidateMergeColors) {
        if (nonIsoGraphsMerge && isoSubGroups_[candidate]->Size() == 1) {
            mergeColors.push_back(candidate);
        } else if (!nonIsoGraphsMerge && isoSubGroups_[candidate]->Size() > 1) {
            mergeColors.push_back(candidate);
        }
    }
    return mergeColors;
}

bool IsoPartitioner::SuitableForMergeCheck(int32_t currColor, int32_t mergeColor, bool nonIsoGraphsMerge) const
{
    std::set<OpCoreType> opcoreTypes{isoSubGroups_[currColor]->GetSubGraph(0)->coreType_,
                                     isoSubGroups_[mergeColor]->GetSubGraph(0)->coreType_};
    bool coreTypeMergable = operationInfo_->CoreTypeMergeable(opcoreTypes);
    int32_t latencyMerged = 0;
    int32_t currColorSize = static_cast<int32_t>(isoSubGroups_[currColor]->Size());
    int32_t mergeColorSize = static_cast<int32_t>(isoSubGroups_[mergeColor]->Size());
    if (currColorSize == 0 || mergeColorSize == 0) {
        return false;
    }
    if (currColorSize <= mergeColorSize) {
        latencyMerged = isoSubGroups_[currColor]->GetLatency() +
                        isoSubGroups_[mergeColor]->GetLatency() * (mergeColorSize / currColorSize);
    } else {
        latencyMerged = isoSubGroups_[currColor]->GetLatency() * (currColorSize / mergeColorSize) +
                        isoSubGroups_[mergeColor]->GetLatency();
    }
    bool cycleMergable = latencyMerged <= cycleThreshold_;
    if (nonIsoGraphsMerge) {
        bool shouldMerge = coreTypeMergable && cycleMergable;
        ALOG_DEBUG_F("Try merge current group: %d [%s]\n\t with: %d [%s], is suitable for merge: %d.",
                     currColor, isoSubGroups_[currColor]->GetSubGraph(0)->DumpStr().c_str(),
                     mergeColor, isoSubGroups_[mergeColor]->GetSubGraph(0)->DumpStr().c_str(), shouldMerge);
        return shouldMerge;
    } 
    bool isSuitableForMerge = (currColorSize == mergeColorSize);
    isSuitableForMerge = isSuitableForMerge || (std::min(currColorSize, mergeColorSize) > parallelThreshold_);
    isSuitableForMerge = isSuitableForMerge ||
                         (std::min(isoSubGroups_[currColor]->GetLatency(), isoSubGroups_[mergeColor]->GetLatency()) <
                          smallGraphThreshold_);
    isSuitableForMerge = coreTypeMergable && isSuitableForMerge && cycleMergable;
    ALOG_DEBUG_F("Try merge current group: %d [%s]\n\t with: %d [%s], is suitable for merge: %d.",
                 currColor, isoSubGroups_[currColor]->GetSubGraph(0)->DumpStr().c_str(),
                 mergeColor, isoSubGroups_[mergeColor]->GetSubGraph(0)->DumpStr().c_str(), coreTypeMergable);
    return isSuitableForMerge;
}

Status IsoPartitioner::IsomorphismGroupMergeStep(bool nonIsoGraphsMerge)
{
    std::vector<std::pair<int32_t, int32_t>> isoSubIdxs;
    std::vector<std::set<int32_t>> isoInGraph;
    std::vector<std::set<int32_t>> isoOutGraph;
    std::vector<std::vector<int32_t>> isoNodeList;
    std::vector<int32_t> isoIdx2color;
    if (IsomorphismGroupMergePrepare(isoSubIdxs, isoInGraph, isoOutGraph, isoNodeList, isoIdx2color) != SUCCESS) {
        ALOG_ERROR_F("IsomorphismGroupMergePrepare failed.");
        return FAILED;
    }
    size_t currColor = 0;
    while (currColor < isoSubGroups_.size()) {
        std::vector<int32_t> mergeColors =
            GetCandidateMergeColors(currColor, isoInGraph, isoOutGraph, isoNodeList, isoIdx2color, nonIsoGraphsMerge);
        bool updated = false;
        for (int32_t mergeColor : mergeColors) {
            if (SuitableForMergeCheck(currColor, mergeColor, nonIsoGraphsMerge) &&
                IsomorphismGraphGroup::IsoGraphMerge(isoSubGroups_[currColor], isoSubGroups_[mergeColor], isoSubIdxs)) {
                ALOG_DEBUG_F("Merge current group %d with %d succeed.", currColor, mergeColor);
                for (int32_t mergeNodeIdx : isoNodeList[mergeColor]) {
                    isoIdx2color[mergeNodeIdx] = currColor;
                }
                isoNodeList[currColor].insert(isoNodeList[currColor].end(), isoNodeList[mergeColor].begin(),
                                              isoNodeList[mergeColor].end());
                isoNodeList[mergeColor].clear();
                updated = true;
                break;
            }
        }
        if (!updated) {
            currColor++;
        }
    }
    return SUCCESS;
}

Status IsoPartitioner::IsomorphismGroupMergeProcess(bool nonIsoGraphsMerge)
{
    for (int32_t loopCount = 0; loopCount < tryMergeLoopNum_; loopCount++) {
        if (IsomorphismGroupMergeStep(nonIsoGraphsMerge) != SUCCESS) {
            ALOG_ERROR_F("IsomorphismGroupMergeStep failed.");
            return FAILED;
        }
        size_t originalColor = isoSubGroups_.size();
        isoSubGroups_.erase(std::remove_if(isoSubGroups_.begin(), isoSubGroups_.end(),
                                           [](auto &groupPtr) { return groupPtr->Size() == 0; }),
                            isoSubGroups_.end());
        if (originalColor == isoSubGroups_.size()) {
            break;
        }
    }
    return SUCCESS;
}

std::string SubGraph::DumpStr()
{
    std::stringstream ss;
    ss << "  NodeList: {";
    for (auto op : GetOpList()) {
        ss << op->GetOpcodeStr() << "(" << op->GetOpMagic() << "), ";
    }
    ss << "}; cycles: {" << cycle_ << "};";
    ss << " core type: {" << static_cast<int32_t>(coreType_) << "};" << std::endl;
    return ss.str();
}

std::vector<Operation*> SubGraph::GetOpList()
{
    std::vector<Operation*> subOpList;
    if (cycle_ == 0) {
        return subOpList;
    }
    for (size_t i = 0; i < nodeList_.size(); i++) {
        for (int32_t opIdx : superNodeInfo_->node2Op_[nodeList_[i]]) {
            subOpList.push_back(operationInfo_->opList_[opIdx]);
        }
    }
    return subOpList;
}

bool IsomorphismGraphGroup::IsoGraphMerge(std::shared_ptr<IsomorphismGraphGroup> &currGraph,
                                          std::shared_ptr<IsomorphismGraphGroup> &mergeGraph,
                                          std::vector<std::pair<int32_t, int32_t>> &isoSubIdxs)
{
    bool swapped = currGraph->Size() > mergeGraph->Size();
    if (swapped) {
        currGraph.swap(mergeGraph);
    }
    size_t currSize = currGraph->Size();
    size_t mergeSize = mergeGraph->Size();
    std::vector<std::set<int32_t>> connection(currSize, std::set<int32_t>{});
    std::map<std::pair<int32_t, int32_t>, int32_t> mergeHistory2MergeIdx;
    for (size_t j = 0; j < mergeSize; j++) {
        for (const std::pair<int32_t, int32_t> &mHist : mergeGraph->GetSubGraph(j)->mergeHistoryIsoSub_) {
            mergeHistory2MergeIdx[mHist] = j;
        }
    }
    for (size_t i = 0; i < currSize; i++) {
        for (int32_t nodeIdx : currGraph->GetSubGraph(i)->inNodes_) {
            std::pair<int32_t, int32_t> &nodeBelong = isoSubIdxs[nodeIdx];
            if (mergeHistory2MergeIdx.count(nodeBelong) > 0) {
                connection[i].insert(mergeHistory2MergeIdx[nodeBelong]);
            }
        }
        for (int32_t nodeIdx : currGraph->GetSubGraph(i)->outNodes_) {
            std::pair<int32_t, int32_t> &nodeBelong = isoSubIdxs[nodeIdx];
            if (mergeHistory2MergeIdx.count(nodeBelong) > 0) {
                connection[i].insert(mergeHistory2MergeIdx[nodeBelong]);
            }
        }
    }
    size_t mergeTimes = 0;
    std::set<int32_t> mergeSet;
    for (auto &conn : connection) {
        mergeTimes += conn.size();
        mergeSet.insert(conn.begin(), conn.end());
    }
    if (mergeTimes != mergeSize || mergeSet.size() != mergeSize) {
        if (swapped) {
            currGraph.swap(mergeGraph);
        }
        return false;
    }
    for (int32_t i = 0; i < static_cast<int32_t>(currSize); i++) {
        for (int32_t j : connection[i]) {
            currGraph->GetSubGraph(i)->Merge(mergeGraph->GetSubGraph(j).get());
        }
    }
    mergeGraph->Clear();
    return true;
}

int32_t NodeGraphInfo::GetNodeCycle(int32_t nodeIdx) const
{
    if (nodeIdx < 0 || nodeIdx >= static_cast<int32_t>(nodeCycles_.size())) {
        return 0;
    }
    return nodeCycles_[nodeIdx];
}

void SubGraph::Merge(SubGraph *sg)
{
    nodeList_.insert(nodeList_.end(), sg->nodeList_.begin(), sg->nodeList_.end());
    if (nodeSet_.size() < sg->nodeSet_.size()) {
        nodeSet_.swap(sg->nodeSet_);
    }
    nodeSet_.insert(sg->nodeSet_.begin(), sg->nodeSet_.end());
    cycle_ += sg->cycle_;
    mergeHistoryIsoSub_.insert(sg->mergeHistoryIsoSub_.begin(), sg->mergeHistoryIsoSub_.end());

    std::unordered_set<int32_t> inNodesTmp;
    std::unordered_set<int32_t> outNodesTmp;
    for (int32_t nodeIdx : inNodes_) {
        if (nodeSet_.count(nodeIdx) == 0) {
            inNodesTmp.insert(nodeIdx);
        }
    }
    for (int32_t nodeIdx : sg->inNodes_) {
        if (nodeSet_.count(nodeIdx) == 0) {
            inNodesTmp.insert(nodeIdx);
        }
    }
    for (int32_t nodeIdx : outNodes_) {
        if (nodeSet_.count(nodeIdx) == 0) {
            outNodesTmp.insert(nodeIdx);
        }
    }
    for (int32_t nodeIdx : sg->outNodes_) {
        if (nodeSet_.count(nodeIdx) == 0) {
            outNodesTmp.insert(nodeIdx);
        }
    }
    inNodes_.swap(inNodesTmp);
    outNodes_.swap(outNodesTmp);
}

Status IsoPartitioner::UpdatePartitionResult(Function &function)
{
    int32_t colorIdx = 0;
    for (size_t i = 0; i < isoSubGroups_.size(); i++) {
        for (size_t j = 0; j < isoSubGroups_[i]->Size(); j++) {
            for (auto op : isoSubGroups_[i]->GetSubGraph(j)->GetOpList()) {
                op->UpdateSubgraphID(colorIdx);
            }
            colorIdx++;
        }
    }
    function.SetTotalSubGraphCount(colorIdx);
    return SUCCESS;
}

Status IsoPartitioner::SetParameter(int32_t cycleThreshold, int32_t parallelThreshold, int32_t smallGraphThreshold, 
                                    bool useReduceBalanceHash)
{
    if (cycleThreshold < 0) {
        ALOG_ERROR_F("Illegal cycle threshold : %d.", cycleThreshold);
        return FAILED;
    }
    if (parallelThreshold < 0) {
        ALOG_ERROR_F("Illegal parallel threshold : %d.", parallelThreshold);
        return FAILED;
    }
    if (smallGraphThreshold < 0) {
        ALOG_ERROR_F("Illegal small graph threshold : %d.", smallGraphThreshold);
        return FAILED;
    }
    cycleThreshold_ = cycleThreshold;
    parallelThreshold_ = parallelThreshold;
    smallGraphThreshold_ = smallGraphThreshold;
    useReduceBalanceHash_ = useReduceBalanceHash;
    return SUCCESS;
}
}  // namespace npu::tile_fwk