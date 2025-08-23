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
 * \file subgraph_to_function_check.cpp
 * \brief
 */

#include "passes/execute_graph_pass/subgraph_to_function.h"
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_utils/pass_utils.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_utils/parallel_tool.h"

namespace npu {
namespace tile_fwk {

Status SubgraphToFunction::NOPCheck(const Operation &op) const {
    if (!op.IsNOP()) {
        ALOG_ERROR_F("op is not an NOP");
        return FAILED;
    }
    if (op.GetIOperands().size() > 0) {
        ALOG_ERROR_F("NOP has IOperands size %zu", op.GetIOperands().size());
        return FAILED;
    }
    if (op.GetInCtrlOperations().size() > 0) {
        ALOG_ERROR_F("NOP has InCtrlOperations size %zu", op.GetInCtrlOperations().size());
        return FAILED;
    }
    if (op.GetOOperands().size() > 0) {
        ALOG_ERROR_F("NOP has OOperands size %zu", op.GetOOperands().size());
        return FAILED;
    }
    if (op.GetOutCtrlOperations().size() > 0) {
        ALOG_ERROR_F("NOP has OutCtrlOperations size %zu", op.GetOutCtrlOperations().size());
        return FAILED;
    }
    return SUCCESS;
}

Status SubgraphToFunction::CheckSubGraphTopo(Function &function) const {
    auto operations = function.Operations();
    int totalSubGraphNum = function.GetTotalSubGraphCount();
    if (operations.size() > 0 && totalSubGraphNum <= 0) {
        ALOG_ERROR_F("input totalSubGraphNum %d is invalid", totalSubGraphNum);
        return FAILED;
    }
    std::vector<bool> hitSubgraph = std::vector<bool>(totalSubGraphNum, false);
    for (size_t i = 0; i < operations.size(); i++) {
        auto &op = operations[i];
        int subGraphId = op.GetSubgraphID();
        if (subGraphId < 0 && NOPCheck(op) != SUCCESS) {
            ALOG_ERROR_F("operation %d has negative subGraphID %d and failed NOP check", i, subGraphId);
            return FAILED;
        }else if (subGraphId >= totalSubGraphNum) {
            ALOG_ERROR_F("operation %d has subGraphID %d that exceeds totalSubGraphNum %d", i, subGraphId, totalSubGraphNum);
            return FAILED;
        }
        hitSubgraph[subGraphId] = true;

        for (auto inOperand : op.GetIOperands()) {
            for (auto parentOp : inOperand->GetProducers()) {
                int parentSubGraphId = parentOp->GetSubgraphID();
                if (parentSubGraphId > subGraphId) {
                    ALOG_ERROR_F("operation %d has subGraphId %d and parent subGraphId %d, parent subGraphId should be less than or equal to subGraphId", i, subGraphId, parentSubGraphId);
                    return FAILED;
                }
            }
        }
    }

    for (int i = 0; i < totalSubGraphNum; i++) {
        if (hitSubgraph[i] == false) {
            ALOG_ERROR_F("Subgraph %d is empty", i);
            return FAILED;
        }
    }

    return SUCCESS;
}

template <typename eType>
Status SubgraphToFunction::InAndOutGraphConsistencyCheck(
    const std::vector<std::vector<eType>> &inEdgeGraph,
    const std::vector<std::vector<eType>> &outEdgeGraph)
{
    if (inEdgeGraph.size() != outEdgeGraph.size()) {
        ALOG_ERROR_F("inEdgeGraph size %zu, outEdgeGraph size %zu", inEdgeGraph.size(), outEdgeGraph.size());
        return FAILED;
    }
    
    std::vector<size_t> nodeColIdx = std::vector<size_t>(inEdgeGraph.size(), 0);
    for (size_t i = 0; i < inEdgeGraph.size(); i++) {
        for (size_t j = 0; j < inEdgeGraph[i].size(); j++) {
            size_t parentSeqNo = static_cast<size_t>(inEdgeGraph[i][j]);
            if (nodeColIdx[parentSeqNo] >= outEdgeGraph[parentSeqNo].size()) {
                ALOG_ERROR_F("node %zu, %zu th parentSeqNo %d exceeds outgraph[%d] size %zu",
                    i, j, parentSeqNo, parentSeqNo, outEdgeGraph[parentSeqNo].size());
                return FAILED;
            }
            // inEdgeGraph和outEdgeGraph都是按顺序排列的
            if (static_cast<size_t>(outEdgeGraph[parentSeqNo][nodeColIdx[parentSeqNo]++]) != i) {
                ALOG_ERROR_F("node %zu, %zu th parentSeqNo %d is not found in outgraph[%d]",
                    i, j, parentSeqNo, parentSeqNo);
                return FAILED;
            }
        }
    }

    // check outEdgeGraph has been fully traversed
    for (size_t i = 0; i < outEdgeGraph.size(); i++) {
        if (outEdgeGraph[i].size() != nodeColIdx[i]) {
            ALOG_ERROR_F("outEdgeGraph[%zu] has size %zu, but only %zu of them have been traversed",
                i, outEdgeGraph[i].size(), nodeColIdx[i]);
            return FAILED;
        }
    }
    
    return SUCCESS;
}

Status SubgraphToFunction::EdgeIndexCheck(const bool found, const int newIndex, const size_t graphSize) const {
    if (!found) {
        ALOG_ERROR_F("op magic not found");
        return FAILED;
    }
    if (static_cast<size_t>(newIndex) >= graphSize) {
        ALOG_ERROR_F("parent index %d is larger than operations_ size %zu", newIndex, graphSize);
        return FAILED;
    }
    return SUCCESS;
}

Status SubgraphToFunction::BuildOutGraph(Function &function) {
    auto operationViewer = function.Operations();
    for (size_t i = 0; i < operationViewer.size(); i++) {
        for (auto &outOperand : operationViewer[i].GetOOperands()) {
            for (auto &childOp : outOperand->GetConsumers()) {
                auto [childSeqNo, found] = operationViewer.FindOpPosition(*childOp);
                if (EdgeIndexCheck(found, childSeqNo, inGraph.size()) != SUCCESS) {
                    ALOG_ERROR_F("error inserting op magic %d in function %d %s to outGraph", childOp->GetOpMagic(), function.GetFuncMagic(),
                        function.GetRawName().c_str());
                    return FAILED;
                }
                outGraph[i].push_back(childSeqNo);
            }
        }

        for (const auto &outControlOp : operationViewer[i].GetOutCtrlOperations()) {
            auto [childSeqNo, found] = operationViewer.FindOpPosition(*outControlOp);
            if (EdgeIndexCheck(found, childSeqNo, inGraph.size()) != SUCCESS) {
                ALOG_ERROR_F("error inserting op magic %d in function %d %s to outGraph", outControlOp->GetOpMagic(), function.GetFuncMagic(),
                    function.GetRawName().c_str());
                return FAILED;
            }
            outGraph[i].push_back(childSeqNo);
        }
        std::sort(outGraph[i].begin(), outGraph[i].end());
    }
    return SUCCESS;
}

Status SubgraphToFunction::CheckInAndOutGraphMatch(Function &function) {
    auto operationViewer = function.Operations();
    inGraph.resize(operationViewer.size());
    outGraph.resize(operationViewer.size());
    // 1. Build inGraph and outGraph
    if (BuildInGraph(function) != SUCCESS) {
        ALOG_ERROR_F("Build inGraph failed");
        return FAILED;
    }
    for (size_t i = 0; i < operationViewer.size(); i++) {
        std::sort(inGraph[i].begin(), inGraph[i].end());   
    }

    if (BuildOutGraph(function) != SUCCESS) {
        ALOG_ERROR_F("Build outGraph failed");
        return FAILED;
    }

    // 2. Check inGraph and outGraph
    if (InAndOutGraphConsistencyCheck(inGraph, outGraph) != SUCCESS) {
        ALOG_ERROR_F("Consistency check for input inGraph and outGraph failed");
        return FAILED;
    }

    // 3. Clear inGraph and outGraph
    for (size_t i = 0; i < inGraph.size(); i++) {
        inGraph[i].clear();
        outGraph[i].clear();
    }
    inGraph.clear();
    outGraph.clear();

    return SUCCESS;
}

Status SubgraphToFunction::CheckSubGraphBoundary(Function &function) {
    auto operations = function.Operations();
    for (size_t i = 0; i < operations.size(); i++) {
        auto &op = operations[i];
        int subGraphId = op.GetSubgraphID();
        for (size_t k = 0; k < op.iOperand.size(); k++) {
            auto iOperand = op.GetInputOperand(k);
            // Rule 1: Operands from DDR memory must be marked as subgraph boundary
            if (iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR && !iOperand->isSubGraphBoundary) { ALOG_ERROR_F("Input operand %zu of operation %zu (opdump: %s) is from DDR but not marked as subgraph boundary!", k, i, op.Dump().c_str()); return FAILED; }
            // Rule 2: Operands with consumer in a different subgraph must be marked as subgraph boundary
            if (subGraphId != iOperand->subGraphID && !iOperand->isSubGraphBoundary) { ALOG_ERROR_F("Input operand %zu of operation %zu (opdump: %s) has a consumer in a different subgraph but not marked as subgraph boundary!", k, i, op.Dump().c_str()); return FAILED; }
            // Rule 3: Input operands of special ops (e.g., OP_UB_COPY_IN) must be marked as subgraph boundary
            if (IsCopyIn(op.GetOpcode()) && !iOperand->isSubGraphBoundary) { ALOG_ERROR_F("Input operand %zu of IsCopyIn operation %zu (opdump: %s) is not marked as subgraph boundary!",k, i, op.Dump().c_str()); return FAILED; }
        }
        for (size_t k = 0; k < op.oOperand.size(); k++) {
            auto oOperand = op.GetOutputOperand(k);
            // Rule 1: Operands from DDR memory must be marked as subgraph boundary
            if (oOperand->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR && !oOperand->isSubGraphBoundary) { ALOG_ERROR_F("Output operand %zu of operation %zu (opdump: %s) is from DDR but not marked as subgraph boundary!", k, i, op.Dump().c_str()); return FAILED; }
            // Rule 2: Operands with producer in a different subgraph must be marked as subgraph boundary
            if (subGraphId != oOperand->subGraphID && !oOperand->isSubGraphBoundary) { ALOG_ERROR_F("Output operand %zu of operation %zu (opdump: %s) has a producer in a different subgraph but not marked as subgraph boundary!", k, i, op.Dump().c_str()); return FAILED; }
            // Rule 3: Output operands of special ops (e.g., OP_UB_COPY_OUT, OP_TRANSPOSE_DATA_MOVE, OP_INDEX_OUTCAST) must be marked as subgraph boundary
            if (IsCopyOut(op.GetOpcode()) && !oOperand->isSubGraphBoundary) { ALOG_ERROR_F("Output operand %zu of IsCopyOut operation %zu (opdump: %s) is not marked as subgraph boundary!",k, i, op.Dump().c_str()); return FAILED; }
        }
    }

    return SUCCESS;
}

Status SubgraphToFunction::PreCheck(Function &function) {
    Status baseStatus = Pass::PreCheck(function);
    if (baseStatus != SUCCESS) { ALOG_ERROR_F("PreCheck failed in base Pass class"); return baseStatus; }

    // Check subgraph topology
    if (CheckSubGraphTopo(function) != SUCCESS) {
        ALOG_ERROR_F("CheckSubGraphTopo failed");
        return FAILED;
    }
    
    // Check subgraph boundary
    if (CheckSubGraphBoundary(function) != SUCCESS) {
        ALOG_ERROR_F("CheckSubGraphBoundary failed");
        return FAILED;
    }

    // Check iOperands and oOperands are matched
    if (CheckInAndOutGraphMatch(function) != SUCCESS) {
        ALOG_ERROR_F("check input ioperands and ooperands relation failed");
        return FAILED;
    }
    
    ALOG_INFO_F("SubgraphToFunction PreCheck completed successfully!");
    return SUCCESS;
}

bool SubgraphToFunction::foundNodeInNeighbor(const int dstNode, const std::vector<int> &searchGraph) const {
    auto it = std::find(searchGraph.begin(), searchGraph.end(), dstNode);
    if (it != searchGraph.end()) {
        return true;
    }
    return false;
}

Status SubgraphToFunction::VerifyRedundantEdge(const int srcNode, const int dstNode) const {
    // dstNode需要在srcNode的三跳之内
    if (foundNodeInNeighbor(dstNode, colorOutGraph[srcNode])) {
        return SUCCESS;
    }
    for (int firstNbr : colorOutGraph[srcNode]) {
        if (foundNodeInNeighbor(dstNode, colorOutGraph[firstNbr])) {
            return SUCCESS;
        }
        for (int secondNbr : colorOutGraph[firstNbr]) {
            if (foundNodeInNeighbor(dstNode, colorOutGraph[secondNbr])) {
                return SUCCESS;
            }
        }
    }
    ALOG_ERROR_F("source node %d and destination node %d are not related in colorOutGraph within three jumps", srcNode, dstNode);
    return FAILED;
}

Status SubgraphToFunction::ColorOutGraphCheck(Function &function) const {
    std::vector<std::vector<bool>> hitEdgeMark = std::vector<std::vector<bool>>(colorOutGraph.size());
    for (size_t i = 0; i < colorOutGraph.size(); i++) {
        hitEdgeMark[i] = std::vector<bool>(colorOutGraph[i].size(), false);
    }

    auto list = function.Operations();
    for (size_t i = 0; i < list.size(); i++) {
        int iSubGraphId = list[i].GetSubgraphID();
        if (iSubGraphId < 0) {
            continue;
        }
        for (int j : outGraph[i]) {
            int jSubGraphId = list[j].GetSubgraphID();
            if (iSubGraphId == jSubGraphId || jSubGraphId < 0) {
                continue;
            }
            
            auto it = std::find(colorOutGraph[iSubGraphId].begin(), colorOutGraph[iSubGraphId].end(), jSubGraphId);
            if (it != colorOutGraph[iSubGraphId].end()) { // found edge
                int index = std::distance(colorOutGraph[iSubGraphId].begin(), it);
                hitEdgeMark[iSubGraphId][index] = true;
            } else if (VerifyRedundantEdge(iSubGraphId, jSubGraphId) != SUCCESS) { // check whether is redundant edge
                ALOG_ERROR_F("edge between original operator %d with subgraph ID %d and operator %d with subgraph ID %d is missed in colorOutGraph", i, iSubGraphId, j, jSubGraphId);
                return FAILED;
            }
        }
    }

    // check all edges have been hit
    for (size_t i = 0; i < hitEdgeMark.size(); i++) {
        for (size_t j = 0; j < hitEdgeMark[i].size(); j++) {
            if (hitEdgeMark[i][j] == false) {
                ALOG_ERROR_F("edge between %d and %d on colorOutGraph has no correspondent edge in outGraph", i, colorOutGraph[i][j]);
                return FAILED;
            }
        }
    }

    return SUCCESS;
}

Status SubgraphToFunction::PostCheck(Function &function) {
    Status baseStatus = Pass::PostCheck(function);
    if (baseStatus != SUCCESS) { ALOG_ERROR_F("PostCheck failed in base Pass class"); return baseStatus; }

    // Check colorInGraph and colorOutGraph consistency
    if (InAndOutGraphConsistencyCheck(colorInGraph, colorOutGraph) != SUCCESS) {
        ALOG_ERROR_F("Consistency check for input colorInGraph and colorOutGraph failed");
        return FAILED;
    }

    // Check colorOutGraph matches outGraph
    if (ColorOutGraphCheck(function) != SUCCESS) {
        ALOG_ERROR_F("Consistency check for colorOutGraph and input failed");
        return FAILED;
    }

    // Check the mapping relationships in psgToESgMap
    for (auto [psgId, esgId] : psgToESgMap) {
        if (CheckSinglePsgEsgMapping(function, psgId, esgId) != SUCCESS) { ALOG_ERROR_F("Failed to check mapping between psg %d and esg %d", psgId, esgId); return FAILED; }
    }

    for (size_t i = 0; i < function.rootFunc_->Operations().size(); ++i) {
        if (VerifySingleOpTopology(function, i) != SUCCESS) { ALOG_ERROR_F("Failed to verify topology for operation %zu", i); return FAILED; }
    }

    // Verify readyState matches negative predecessor count
    for (size_t i = 0; i < function.rootFunc_->topoInfo_.topology_.size(); i++) {
        if (CheckReadyStateConsistency(function, i) != SUCCESS) { ALOG_ERROR_F("Ready state inconsistency found for topology entry %zu", i); return FAILED; }
    }

    ALOG_INFO_F("SubgraphToFunction PostCheck completed successfully!");
    return SUCCESS;
}
    
Status SubgraphToFunction::CheckSinglePsgEsgMapping(Function &function, uint32_t psgId, uint32_t esgId) {
    auto iter = function.rootFunc_->programs_.find(psgId);
    if (iter == function.rootFunc_->programs_.end()) { ALOG_ERROR_F("Psg %d not found in program", psgId); return FAILED; }
    auto operations = function.rootFunc_->Operations();
    const Operation* targetCallOp = nullptr;
    for (size_t i = 0; i < operations.size(); ++i) {
        const auto& op = operations[i];
        uint32_t currentSubgraphId = op.GetSubgraphID();
        if (currentSubgraphId == esgId) {
            targetCallOp = &op;
            break;
        }
    }
    if (!targetCallOp) {
        ALOG_ERROR_F("No callOp found with subgraphId %u", esgId);
        return FAILED;
    }
    Operation* nonConstTargetCallOp = const_cast<Operation*>(targetCallOp);
    auto &esg = nonConstTargetCallOp->GetSubFuncInvokeInfo();
    auto &psg = iter->second->GetParameter();
    ALOG_DEBUG_F("start match psg %d - esg %d\n", psgId, esgId);
    if (!CompareParamLists(esg.GetIncastTensorParamList(), psg.inCastArgs_, "Incast", psgId, esgId)) { ALOG_ERROR_F("Incast parameter lists mismatch between psg %d and esg %d", psgId, esgId); return FAILED; }
    if (!CompareParamLists(esg.GetOutcastTensorParamList(), psg.outCastArgs_, "Outcast", psgId, esgId)) { ALOG_ERROR_F("Outcast parameter lists mismatch between psg %d and esg %d", psgId, esgId); return FAILED; }
    if (!CompareParamLists(esg.GetTensorParamList(), psg.tensorsArgs_, "Tensor", psgId, esgId)) { ALOG_ERROR_F("Tensor parameter lists mismatch between psg %d and esg %d", psgId, esgId); return FAILED; }
    return SUCCESS;
}

template <typename ESGParamType, typename PSGParamContainer>
bool SubgraphToFunction::CompareParamListsImpl(
    const std::vector<ESGParamType>& esgParams, 
    const PSGParamContainer& psgParams, 
    const std::string &paramType, uint32_t psgId, uint32_t esgId) const
{
    if (esgParams.size() != psgParams.size()) {
        ALOG_ERROR_F("Psg %d esg %d %s size mismatch[%zu : %zu]", psgId, esgId, paramType.c_str(), esgParams.size(), psgParams.size());
        return false;
    }
    for (size_t i = 0; i < esgParams.size(); i++) {
        const auto& e = esgParams[i];
        const auto& p = psgParams[i];
        // 动态shape豁免检查
        if (p.shape[0] == kShapePlaceholderForParameterized) {
            ALOG_DEBUG_F("Skip dynamic shape check");
            continue;
        }
        if (!(p.CompareParam(e))) {
            ALOG_ERROR_F("Psg %d esg %d %s shape mismatch at %zu", psgId, esgId, paramType.c_str(), i);
            return false;
        }
    }
    return true;
}

bool SubgraphToFunction::CompareParamLists(
    const std::vector<SubfuncInvokeInfoTy::IncastParamPackTy>& esgParams, 
    const SubfuncParam::InCastParamListTy& psgParams, 
    const std::string &paramType, uint32_t psgId, uint32_t esgId) const
{
    return CompareParamListsImpl(esgParams, psgParams, paramType, psgId, esgId);
}
        
bool SubgraphToFunction::CompareParamLists(
    const std::vector<SubfuncInvokeInfoTy::OutcastParamPackTy>& esgParams, 
    const SubfuncParam::OutCastParamListTy& psgParams, 
    const std::string &paramType, uint32_t psgId, uint32_t esgId) const   
{
    return CompareParamListsImpl(esgParams, psgParams, paramType, psgId, esgId);
}

bool SubgraphToFunction::CompareParamLists(
    const std::vector<SubfuncInvokeInfoTy::TensorParamPackTy>& esgParams, 
    const SubfuncParam::TensorParamListTy& psgParams, 
    const std::string& paramType, uint32_t psgId, uint32_t esgId) const
{
    return CompareParamListsImpl(esgParams, psgParams, paramType, psgId, esgId);  
}
        
Status SubgraphToFunction::VerifySingleOpTopology(Function &function, size_t opIndex) {
    const auto &callOps = function.rootFunc_->Operations();
    // 通过 subgraphId 查找对应的 currentOp
    Operation* currentOp = nullptr;
    for (const auto& op : callOps) {
        if (static_cast<size_t>(op.GetSubgraphID()) == opIndex) {
            currentOp = const_cast<Operation*>(&op);
            break;
        }
    }
    
    if (currentOp == nullptr) {
        ALOG_ERROR_F("Cannot find operation with subgraphId %zu", opIndex);
        return FAILED;
    }
    
    auto consumers = currentOp->ConsumerOps();
    auto producers = currentOp->ProducerOps();
    ALOG_DEBUG_F("=================Call ===============%zu", opIndex);
    for (auto &prod : producers) {
        ALOG_DEBUG_F("Producer %s %d", prod->GetOpcodeStr().c_str(), prod->opmagic);
    }
    auto &topoInfo = function.rootFunc_->topoInfo_.topology_[opIndex];
    std::unordered_set<Operation *> consumersNoSelf;
    for (auto *cons : consumers) {
        if (cons->opmagic != currentOp->opmagic) {
            consumersNoSelf.insert(cons);
        }
    }
    if (consumersNoSelf.size() < topoInfo.outGraph.size()) { ALOG_ERROR_F("Call %zu %d consumers size are %zu and %zu", opIndex, currentOp->opmagic, consumersNoSelf.size(), topoInfo.outGraph.size()); return FAILED; }
    for (auto succ : topoInfo.outGraph) {
        const int consumerSubgraphId = static_cast<int>(succ); 
        
        // 通过 subgraphId 查找预期的消费者op
        Operation* expectedConsumer = nullptr;
        for (const auto& op : callOps) {
            if (op.GetSubgraphID() == consumerSubgraphId) {
                expectedConsumer = const_cast<Operation*>(&op);
                break;
            }
        }
        
        if (expectedConsumer == nullptr) {
            ALOG_ERROR_F("Cannot find expected consumer with subgraphId %d", consumerSubgraphId);
            return FAILED;
        }
        if (consumers.count(expectedConsumer) == 0) { ALOG_ERROR_F("Cannot find consumer %d for call %zu", succ, opIndex); return FAILED; }
    }
    return SUCCESS;
}        
        
Status SubgraphToFunction::CheckReadyStateConsistency(Function &function, size_t opIndex) {
    auto &topology = function.rootFunc_->topoInfo_.topology_;
    // Calculate actual predecessor count
    int actualPredCount = 0;

    for (size_t j = 0; j < topology.size(); j++) {
        if (opIndex == j) {
            continue;
        }    
        auto &otherEntry = topology[j];
        if (std::find(otherEntry.outGraph.begin(), otherEntry.outGraph.end(), opIndex) != otherEntry.outGraph.end()) {
            actualPredCount++;
        }
    }
    // readyState should equal negative predecessor counts
    if (topology[opIndex].readyState != -actualPredCount) { ALOG_ERROR_F("Subgraph %zu has inconsistent readyState: actual=%d, expected=%d", opIndex, topology[opIndex].readyState, -actualPredCount); return FAILED; }
    return SUCCESS;
}
    
} // namespace tile_fwk
} // namespace npu
