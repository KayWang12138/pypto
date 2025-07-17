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
 * \file subgraph_to_function.cpp
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

namespace npu::tile_fwk {
// Add string name for codegen
std::string SubgraphToFunction::FindSymbolName(std::shared_ptr<LogicalTensor> op, int magic) const {
    if (magic < 0){
        return std::to_string(magic);
    }
    if (magic == 0){
        return "NULL_OPERAND";
    }
    // DDR variable
    MemoryType originalType = op->GetMemoryTypeOriginal();
    if (originalType == MemoryType::MEM_DEVICE_DDR){
        return "Var$" + std::to_string(magic);
    }

    auto name = "$" + std::to_string(magic);
    return name;
}

Status SubgraphToFunction::PreCheck(Function &function) {
    Status baseStatus = Pass::PreCheck(function);
    if (baseStatus != SUCCESS) { ALOG_ERROR_F("PreCheck failed in base Pass class"); return baseStatus; }
    auto operations = function.Operations();
    for (size_t i = 0; i < operations.size(); i++) {
        auto &op = operations[i];
        int subGraphId = op.GetSubgraphID();
        // Check input operands
        for (size_t k = 0; k < op.iOperand.size(); k++) {
            auto iOperand = op.GetInputOperand(k);
            // Rule 1: Operands from DDR memory must be marked as subgraph boundary
            if (iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR && !iOperand->isSubGraphBoundary) { ALOG_ERROR_F("Input operand %zu of operation %zu (opdump: %s) is from DDR but not marked as subgraph boundary!", k, i, op.Dump().c_str()); return FAILED; }
            // Rule 2: Operands with consumer in a different subgraph must be marked as subgraph boundary
            if (subGraphId != iOperand->subGraphID && !iOperand->isSubGraphBoundary) { ALOG_ERROR_F("Input operand %zu of operation %zu (opdump: %s) has a consumer in a different subgraph but not marked as subgraph boundary!", k, i, op.Dump().c_str()); return FAILED; }
            // Rule 3: Input operands of special ops (e.g., OP_UB_COPY_IN) must be marked as subgraph boundary
            if (IsCopyIn(op.GetOpcode()) && !iOperand->isSubGraphBoundary) { ALOG_ERROR_F("Input operand %zu of IsCopyIn operation %zu (opdump: %s) is not marked as subgraph boundary!",k, i, op.Dump().c_str()); return FAILED; }
        }
        // Check output operands
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
    ALOG_INFO_F("SubgraphToFunction PreCheck completed successfully!");
    return SUCCESS;
}

Status SubgraphToFunction::PostCheck(Function &function) {
    Status baseStatus = Pass::PostCheck(function);
    if (baseStatus != SUCCESS) { ALOG_ERROR_F("PostCheck failed in base Pass class"); return baseStatus; }
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
    auto &esg = function.rootFunc_->Operations()[esgId].GetSubFuncInvokeInfo();
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
        if (e.shape != p.shape) {
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
    auto consumers = callOps[opIndex].ConsumerOps();
    auto producers = callOps.at(opIndex).ProducerOps();
    ALOG_DEBUG_F("=================Call ===============%zu", opIndex);
    for (auto &prod : producers) {
        ALOG_DEBUG_F("Producer %s %d", prod->GetOpcodeStr().c_str(), prod->opmagic);
    }
    auto &topoInfo = function.rootFunc_->topoInfo_.topology_[opIndex];
    std::unordered_set<Operation *> consumersNoSelf;
    for (auto *cons : consumers) {
        if (cons->opmagic != callOps[opIndex].opmagic) {
            consumersNoSelf.insert(cons);
        }
    }
    if (consumersNoSelf.size() < topoInfo.outGraph.size()) { ALOG_ERROR_F("Call %zu %d consumers size are %zu and %zu", opIndex, callOps.at(opIndex).opmagic, consumersNoSelf.size(), topoInfo.outGraph.size()); return FAILED; }
    for (auto succ : topoInfo.outGraph) {
        if (consumers.count(&(callOps.at(succ))) == 0) { ALOG_ERROR_F("Cannot find consumer %d for call %zu", succ, opIndex); return FAILED; }
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
    
void SubgraphToFunction::RecordEsgIncast(Function &function, size_t i, size_t j, size_t k) {
    auto &iter = subFuncInvokeInfos[i];
    auto iOperand = nLIST[i][j]->GetInputOperand(k);
    auto offset = iOperand->offset;
    auto shape = iOperand->shape;
    if (IsCopyIn(nLIST[i][j]->GetOpcode())){
        offset.clear();
        std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(nLIST[i][j]->GetOpAttribute());
        std::vector<OpImmediate> opImmList = attr->GetCopyInAttr().first;
        for (auto &opImm : opImmList){
            offset.push_back(opImm.GetSpecifiedValue());
        }
        shape = attr->GetSpecifiedShape(1);
    }

    // 1. IndexOutCast存在外部输入也是子图内的输出和输入的情况
    // 2. 对于Assemble的输入如果是来自于OutCast，那么一定会在某个CopyOut的输出上被记录
    if (function.IsFromInCast(iOperand) ||
        function.IsFromOutCast(iOperand)) {
        iter.RecordTensorArg(nLIST[i][j]->GetIntAttribute(OpAttributeKey::seqNo), k,
            iOperand->GetRawMagic(),
            offset, shape,
            iOperand->tensor->rawshape,
            iOperand->Datatype(), false, iOperand, nLIST[i][j]->opmagic);
    } else if (iOperand->isSubGraphBoundary && iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
        auto producers = iOperand ->GetProducers();
        if (producers.size() == 0){
            if (IsCopyIn(nLIST[i][j]->GetOpcode())) {
                iter.RecordConnection(i, i, nLIST[i][j]->GetIntAttribute(OpAttributeKey::seqNo), k,
                    iOperand->GetRawMagic(),  offset,
                    shape, iOperand->tensor->rawshape,
                    iOperand->Datatype(), iOperand, nLIST[i][j]->opmagic);
            }
        }
        else {
            std::vector<int> assembleRawMagic;
            for (auto &producer : producers) {
                auto eSgId = producer->GetSubgraphID();
                std::vector<int>::iterator it = find(assembleRawMagic.begin(), assembleRawMagic.end(), iOperand->GetRawMagic());
                if (it != assembleRawMagic.end()) {
                    continue;
                }
                else {
                    assembleRawMagic.push_back(iOperand->GetRawMagic());
                }
                iter.RecordConnection(eSgId, i, nLIST[i][j]->GetIntAttribute(OpAttributeKey::seqNo), k,
                    iOperand->GetRawMagic() /*placeHolder*/, offset,
                    shape, iOperand->tensor->rawshape,
                    iOperand->Datatype(), iOperand, nLIST[i][j]->opmagic);
            }
        }
    }
}

void SubgraphToFunction::RecordEsgOutcast(Function &function, size_t i, size_t j, size_t k){
    auto &iter = subFuncInvokeInfos[i];
    // 4.2 Record oOperand info, global tensor and outCasts_
    auto oOperand = nLIST[i][j]->GetOutputOperand(k);
    auto offset = oOperand->offset;
    auto shape = oOperand->shape;
    if (IsCopyOut(nLIST[i][j]->GetOpcode())){
        offset.clear();
        std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(nLIST[i][j]->GetOpAttribute());
        std::vector<OpImmediate> opImmList = attr->GetCopyOutAttr().second;
        for (auto &opImm : opImmList){
            offset.push_back(opImm.GetSpecifiedValue());
        }
        shape = attr->GetSpecifiedShape(1);
    }

    if (function.IsFromOutCast(oOperand) ||
        function.IsFromInCast(oOperand)) {
        iter.RecordTensorArg(nLIST[i][j]->GetIntAttribute(OpAttributeKey::seqNo), k,
            oOperand->GetRawMagic(),
            offset, shape, oOperand->tensor->rawshape,
            oOperand->Datatype(), true, oOperand, nLIST[i][j]->opmagic);
    }
    // boundary outCasts_
    else {
        int refCount = 0;
        typename SubfuncInvokeInfoTy::SuccessorIncastInfoTy relatedIncastList;
        if (oOperand->isSubGraphBoundary && oOperand->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            auto consumers = oOperand->GetConsumers();
            for (auto &consumer : consumers) {
                auto eSgId = consumer->GetSubgraphID();
                if (eSgId != static_cast<int>(i)) {
                    refCount++;
                    int connectedTgtOperandIdx = oOperand->magic;
                    relatedIncastList.push_back(typename SubfuncInvokeInfoTy::SuccessorIncastRecTy(
                        eSgId, connectedTgtOperandIdx, nullptr, consumer->GetOpMagic()));
                }
            }
            iter.RecordOutcast(i, nLIST[i][j]->GetIntAttribute(OpAttributeKey::seqNo),
                refCount, oOperand->GetRawMagic(),
                relatedIncastList, offset, shape,
                oOperand->tensor->rawshape, oOperand->Datatype(), oOperand,
                nLIST[i][j]->opmagic);
            nLIST[i][j]->outcastRefcount = refCount;
        }
    }
}

void SubgraphToFunction::RecordIncastOutcast(Function &function) {
    // 1. Get function->operations_, construct 2-dimension nLIST（subgraphID, operations)
    auto list = function.Operations();
    nLIST.resize(function.GetTotalSubGraphCount());
    for (size_t i = 0; i < list.size(); i++) {
        if (list[i].IsNOP() || list[i].GetSubgraphID() < 0) {
            continue;
        } else {
            nLIST[list[i].GetSubgraphID()].push_back(list.operations_[i]);
        }
    }
    // 2. Init InvokeInfo, {ESGID, SubfuncInvokeInfo}, which contains all the invoke info of each subgrahs
    for (size_t i = 0; i < nLIST.size(); i++) {
        subFuncInvokeInfos.push_back(SubfuncInvokeInfoTy());
    }
    // 3. Construct subgraph with seqNo, and label boundary.
    for(int i = 0; i < static_cast<int>(nLIST.size()); i++){
        for (size_t j = 0; j < nLIST[i].size(); j++) {
            nLIST[i][j]->SetAttribute(OpAttributeKey::seqNo, static_cast<int>(j));
            // 4.1 Record iOperand info, global tensor and incast
            for (size_t k = 0; k < nLIST[i][j]->iOperand.size(); k++) {
                RecordEsgIncast(function, i, j, k);
            }
            for (size_t k = 0; k < nLIST[i][j]->GetOOperands().size(); k++) {
                RecordEsgOutcast(function, i, j, k);
            }
        }
    }
    // 4. Incast， Outcast，Construct connection using connetion and outcast
    for (auto &item : subFuncInvokeInfos) {
        item.DoFinishRecord();
    }
}

void SubgraphToFunction::BuildColorGraph(Function &function) {
    colorInGraph = std::vector<std::vector<int>>(function.GetTotalSubGraphCount());
    colorOutGraph = std::vector<std::vector<int>>(function.GetTotalSubGraphCount());
    isReshape = std::vector<bool>(function.GetTotalSubGraphCount(), false);
    auto list = function.Operations();
    for (size_t i = 0; i < list.size(); i++) {
        if (list[i].GetSubgraphID() < 0) {
            continue;
        }
        for (int j : outGraph[i]) {
            if (list[i].GetSubgraphID() != list[j].GetSubgraphID()) {
                if (list[j].GetSubgraphID() < 0) {
                    continue;
                }
                colorOutGraph[list[i].GetSubgraphID()].push_back(list[j].GetSubgraphID());
                colorInGraph[list[j].GetSubgraphID()].push_back(list[i].GetSubgraphID());
            }
        }
    }
    for (size_t i = 0; i < function.GetTotalSubGraphCount(); i++) {
        if (nLIST[i].size() == 1UL && colorInGraph[i].size() == 0 && nLIST[i][0]->GetOpcode() == Opcode::OP_RESHAPE){
            isReshape[i] = true;
        }
        std::sort(colorInGraph[i].begin(), colorInGraph[i].end());
        colorInGraph[i].resize(std::unique(colorInGraph[i].begin(), colorInGraph[i].end()) -
                            colorInGraph[i].begin());

        std::sort(colorOutGraph[i].begin(), colorOutGraph[i].end());
        colorOutGraph[i].resize(std::unique(colorOutGraph[i].begin(), colorOutGraph[i].end()) -
                            colorOutGraph[i].begin());
    }
}

void SubgraphToFunction::PrintColorGraph(const Function &function) {
    ALOG_INFO_F("********** Color Graph **********\n");
    for (size_t i = 0; i < function.GetTotalSubGraphCount(); i++) {
        ALOG_INFO_F("%zu: %zu, %zu", i, colorInGraph[i].size(), colorOutGraph[i].size());
        ALOG_INFO_F("%s", IntVecToStr(colorInGraph[i]).c_str());
        ALOG_INFO_F("%s", IntVecToStr(colorOutGraph[i]).c_str());
    }
    int inCount = 0, outCount = 0;
    for (size_t i = 0; i < function.GetTotalSubGraphCount(); i++) {
        inCount += colorInGraph[i].size();
        outCount += colorOutGraph[i].size();
    }
    ALOG_INFO_F("total in: %d, total out: %d\n", inCount, outCount);
}

void SubgraphToFunction::FindRedundantEdges(int color, std::vector<std::vector<int>>& redundantColorInGraph,
    std::vector<std::vector<int>>& redundantColorOutGraph) {
    std::vector<int> tag(color);
    int tagValue = 2;
    for (int i = 0; i < color; i++) {
        std::vector<int> queue0 = colorOutGraph[i];
        std::vector<int> queue1, queue2;
        for (int j : colorOutGraph[i]) {
            tag[j] = tagValue;
        }
        for (int j : queue0) {
            for (int k : colorOutGraph[j]) {
                if (tag[k] == tagValue) {
                tag[k] = 1;
                redundantColorOutGraph[i].push_back(k);
                redundantColorInGraph[k].push_back(i);
                } else if (tag[k] == 0) {
                tag[k] = 1;
                queue1.push_back(k);
                }
            }
        }
        for (int j : queue1) {
            for (int k : colorOutGraph[j]) {
                if (tag[k] == tagValue) {
                tag[k] = 1;
                redundantColorOutGraph[i].push_back(k);
                redundantColorInGraph[k].push_back(i);
                } else if (tag[k] == 0) {
                }
            }
        }
        for (int j : queue0) {
            tag[j] = 0;
        }
        for (int j : queue1) {
            tag[j] = 0;
        }
    }
}

void SubgraphToFunction::EraseRedundantColorEdges(const Function &function) {
    size_t color = function.GetTotalSubGraphCount();
    std::vector<std::vector<int>> redundantColorInGraph(color), redundantColorOutGraph(color);
    std::vector<int> tag(color);
    // Find redundant edges
    FindRedundantEdges(color, redundantColorInGraph, redundantColorOutGraph);
    // Erase redundant edges
    for (size_t i = 0; i < color; i++) {
        std::sort(redundantColorOutGraph[i].begin(), redundantColorOutGraph[i].end());
        std::vector<int> newGraph;
        // update color_in_graph
        size_t j = 0U;
        for (int k : redundantColorInGraph[i]) {
            while (colorInGraph[i][j] != k) {
                newGraph.push_back(colorInGraph[i][j]);
                j++;
            }
            j++;
        }
        while (j < colorInGraph[i].size()) {
            newGraph.push_back(colorInGraph[i][j]);
            j++;
        }
        colorInGraph[i] = newGraph;
        // update color_out_graph
        newGraph.clear();
        j = 0;
        for (int k : redundantColorOutGraph[i]) {
            while (colorOutGraph[i][j] != k) {
                newGraph.push_back(colorOutGraph[i][j]);
                j++;
            }
            j++;
        }
        while (j < colorOutGraph[i].size()) {
            newGraph.push_back(colorOutGraph[i][j]);
            j++;
        }
        colorOutGraph[i] = newGraph;
    }
}

SubfuncTopologyInfoTy SubgraphToFunction::ConstructSubgraphTopologyInfo(
    Function &function, std::vector<SubfuncInvokeInfoTy> &esgInvokeInfoMap) {
    int maxOutDegree = 0;
    SubfuncTopologyInfoTy topo;
    topo.SetTableSize(esgInvokeInfoMap.size());
    subgTopoParamOffsets.emplace_back(0);
    BuildColorGraph(function);
    PrintColorGraph(function);
    EraseRedundantColorEdges(function);
    PrintColorGraph(function);
    ALOG_INFO_F("colorOutGraph size %zu", colorOutGraph.size());
    for (size_t i = 0; i < colorOutGraph.size(); i++) {
        setType succESgs;
        int eSgId = i;
        bool skip = isReshape[i];
        succESgs.clear();
        if (!skip){
            for (size_t j = 0; j < colorOutGraph[i].size(); j++) {
                succESgs.insert(colorOutGraph[i][j]);
            }
        }
        maxOutDegree = static_cast<int>(succESgs.size()) > maxOutDegree ? static_cast<int>(succESgs.size()) : maxOutDegree;
        int realOutDegree = 0;
        for (auto &item : colorInGraph[i]) {
            if (!isReshape[item]){
                realOutDegree++;
            }
        }
        int readyOrNot = -1 * realOutDegree;
        topo.AddEntry(eSgId, readyOrNot, succESgs);
        if ((nLIST[i].size() == 1UL) && (nLIST[i][0]->GetCoreType() == CoreType::AICPU)){
            auto &op = nLIST[i][0];
            const std::string extParamKey = OP_ATTR_PREFIX + "distributed";
            if (op->HasAttr(extParamKey)) {
                std::vector<int> extParams = op->GetVectorIntAttribute(extParamKey);
                topo.UpdateEntry(static_cast<uint32_t>(op->GetOpcode()), extParams.size(), extParams);
                ALOG_DEBUG_F("######## UpdateEntry size=%lu ######", extParams.size());
            }
        }
        ALOG_DEBUG_F("######## AddEntry ESgId %d ReadyOrNot %d %zu %d ######", eSgId, readyOrNot, topo.readyIds_.size(),
            topo.readyIds_[0]);
        // Add subgTopoParamOffsets for simcpu
        int64_t offSize = sizeof(int64_t) +                             // ProgramSubgraph Id size
                          sizeof(int64_t) +                             // ReadyOrNot size
                          sizeof(int64_t) +                             // outDependEsgIdList.size
                          sizeof(int64_t) * (static_cast<int64_t>(succESgs.size())); // outDependEsgIdList
        subgTopoParamOffsets.emplace_back(subgTopoParamOffsets.back() + offSize);
    }
    topo.SetMaxM(maxOutDegree);
    return topo;
}

void SubgraphToFunction::ConstructParamMap(Function &function) {
    function.topoInfo_ = ConstructSubgraphTopologyInfo(function, subFuncInvokeInfos);
    for (size_t i = 0; i < subFuncInvokeInfos.size(); i++) {
        subFuncInvokeInfos[i].ConstructActualInvokeParam(i);
    }
}

bool IsCubeOp(Operation &op) {
    if (op.GetBoolAttribute(OpAttributeKey::isCube)) {
        return true;
    }
    if ((op.GetOpcode() == Opcode::OP_L0C_COPY_OUT) || (op.GetOpcode() == Opcode::OP_L1_COPY_IN)) {
        return true;
    }
    return false;
}

bool IsAICPUOp(Operation &op) {
    if ((op.GetCoreType() == CoreType::AICPU)) {
        return true;
    }
    return false;
}

void SubgraphToFunction::ProcessInputOperands(Function* rootFunc, Operation& tileOp, SubfuncParam& pSgParamInfo, int& tParamLoc, int& iParamLoc) const {
    for (size_t k = 0; k < tileOp.GetIOperands().size(); k++) {
        auto iOperand = tileOp.GetInputOperand(k);
        std::string name = FindSymbolName(iOperand, iOperand->GetRawMagic());
        auto offset = iOperand->offset;
        auto shape = iOperand->shape;

        if (IsCopyIn(tileOp.GetOpcode())){
            ProcessCopyInOperand(tileOp, offset, shape);
        }
        int seqNo = tileOp.GetIntAttribute(OpAttributeKey::seqNo);
        if (rootFunc->IsFromInCast(iOperand) ||
            rootFunc->IsFromOutCast(iOperand)) {
            // Offsets are a part of each parameter, but we need to keep their original value to
            // keep track of dependencies within a subgraph.
            pSgParamInfo.AppendTensorParam(
                seqNo, k, iOperand->GetRawMagic(), shape, offset, name, tParamLoc,
                iOperand->tensor->GetSymbol(), iOperand->tensor->GetDataType());
            tileOp.inParamLocation_.push_back(tParamLoc);
            tParamLoc++;
        } else if (iOperand->isSubGraphBoundary &&
            iOperand->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            pSgParamInfo.AppendIncastParam(seqNo, k, iOperand->GetRawMagic(), shape,
                offset, name, iParamLoc, iOperand->tensor->GetSymbol(), iOperand->tensor->GetDataType());
            tileOp.inParamLocation_.push_back(iParamLoc);
            iParamLoc++;
        }
    }
}

void SubgraphToFunction::ProcessOutputOperands(Function* rootFunc, Operation& tileOp, SubfuncParam& pSgParamInfo, int& tParamLoc, int& oParamLoc) const {
    for (size_t k = 0; k < tileOp.GetOOperands().size(); k++) {
        auto oOperand = tileOp.GetOutputOperand(k);
        std::string name = FindSymbolName(oOperand, oOperand->GetRawMagic());
        auto offset = oOperand->offset;
        auto shape = oOperand->shape;
        if (IsCopyOut(tileOp.GetOpcode())){
            ProcessCopyOutOperand(tileOp, offset, shape);
        }
        int seqNo = tileOp.GetIntAttribute(OpAttributeKey::seqNo);
        if (rootFunc->IsFromOutCast(oOperand) || rootFunc->IsFromInCast(oOperand)) {
            pSgParamInfo.AppendTensorParam(
                seqNo, k, oOperand->GetRawMagic(), shape,
                offset, name, tParamLoc, oOperand->tensor->GetSymbol(), oOperand->tensor->GetDataType());
            tileOp.outParamLocation_.push_back(tParamLoc);
            tParamLoc++;
        } else if (oOperand->isSubGraphBoundary &&
            oOperand->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR){
            pSgParamInfo.AppendOutcastParam(seqNo, k, oOperand->GetRawMagic(),
                tileOp.outcastRefcount, shape,
                offset, name, oParamLoc, oOperand->tensor->GetSymbol(), oOperand->tensor->GetDataType());
            tileOp.outParamLocation_.push_back(oParamLoc);
            oParamLoc++;
        }
    }
}

void SubgraphToFunction::ProcessCopyInOperand(Operation& tileOp, std::vector<int>& offset, std::vector<int>& shape) const{
    offset.clear();
    std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(tileOp.GetOpAttribute());
    std::vector<OpImmediate> opImmList = attr->GetCopyInAttr().first;
    for (auto &opImm : opImmList){
        offset.push_back(opImm.GetSpecifiedValue());
    }
    shape = attr->GetSpecifiedShape(1);
}

void SubgraphToFunction::ProcessCopyOutOperand(Operation& tileOp, std::vector<int>& offset, std::vector<int>& shape) const{
    offset.clear();
    std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(tileOp.GetOpAttribute());
    std::vector<OpImmediate> opImmList = attr->GetCopyOutAttr().second;
    for (auto &opImm : opImmList){
        offset.push_back(opImm.GetSpecifiedValue());
    }
    shape = attr->GetSpecifiedShape(1);
}

void SubgraphToFunction::SymbolizeEachFunction(Function *rootFunc, std::vector<Function *> &mergedFuncList1, size_t i) const{
    int pSgId = i;
    int tParamLoc = 0;
    int iParamLoc = 0 | 0x10000000;
    int oParamLoc = 0 | 0x20000000;
    SubfuncParam pSgParamInfo;
    // do symbolize only for real merged subgraph, others are constant program
    auto &leafFunc = mergedFuncList1[i];
    for (auto &tileOp : leafFunc->Operations()) {
        // symbolic
        ProcessInputOperands(rootFunc, tileOp, pSgParamInfo, tParamLoc, iParamLoc);
        ProcessOutputOperands(rootFunc, tileOp, pSgParamInfo, tParamLoc, oParamLoc);
    }        
          
    pSgParamInfo.Finalize();
    mergedFuncList1[pSgId]->SetParameter(pSgParamInfo);
    mergedFuncList1[pSgId]->SetProgramId(pSgId);
    rootFunc->programs_.insert({pSgId, mergedFuncList1[pSgId]});
}

void SubgraphToFunction::SymbolizeFunction(Function *rootFunc, std::vector<Function *> &mergedFuncList1) const{
    for (size_t i = 0; i < mergedFuncList1.size(); i++) {
        SymbolizeEachFunction(rootFunc, mergedFuncList1, i);
    }
}

void SubgraphToFunction::BuildGraph(Function &function) {
    auto operationViewer = function.Operations();
    inGraph.resize(operationViewer.size());
    outGraph.resize(operationViewer.size());
    for (size_t i = 0; i < operationViewer.size(); i++) {
        inGraph[i].clear();
        outGraph[i].clear();
        for (auto &inOperand : operationViewer[i].GetIOperands()) {
            for (auto &parentOp : inOperand->GetProducers()) {
                auto [parentSeqNo, found] = operationViewer.FindOpPosition(*parentOp);
                if (!found) {
                    ASLOGE("cannot find op magic %d in function %d %s", parentOp->GetOpMagic(), function.GetFuncMagic(),
                        function.GetRawName().c_str());
                    continue;
                }
                if (static_cast<size_t>(parentSeqNo) >= inGraph.size()) {
                    ASLOGE("parent index %d is larger than operations_ size %zu", parentSeqNo, operationViewer.size());
                    continue;
                }
                inGraph[i].push_back(parentSeqNo);
                outGraph[parentSeqNo].push_back(i);
            }
        }

        for (const auto &inControlOp : operationViewer[i].GetInCtrlOperations()) {
            auto [parentSeqNo, found] = operationViewer.FindOpPosition(*inControlOp);
            if (!found) {
                ASLOGE("cannot find op magic %d in function %d %s", inControlOp->GetOpMagic(), function.GetFuncMagic(),
                    function.GetRawName().c_str());
                continue;
            }
            if (static_cast<size_t>(parentSeqNo) >= inGraph.size()) {
                ASLOGE("control parent index %d is larger than operations_ size %zu", parentSeqNo, operationViewer.size());
                continue;
            }
            inGraph[i].push_back(parentSeqNo);
            outGraph[parentSeqNo].push_back(i);
        }
    }
}

void SubgraphToFunction::InsertParameter(size_t i, Function* leafFunc) {
    for (auto &in : subFuncInvokeInfos[i].GetIncastTensorParamList()) {
        leafFunc->inCasts_.emplace_back(in.tensor);
        leafFunc->InsertOpmagicToIncastIdx(in.opMagic, leafFunc->inCasts_.size() - 1);
    }
    for (auto &out : subFuncInvokeInfos[i].GetOutcastTensorParamList()) {
        leafFunc->outCasts_.emplace_back(out.tensor);
        leafFunc->InsertOpmagicToOutcastIdx(out.opMagic, leafFunc->outCasts_.size() - 1);
    }
    for (auto &tensor : subFuncInvokeInfos[i].GetTensorParamList()) {
        leafFunc->AddGlobalTensor(tensor.tensor);
        if (tensor.isOutputToGM) {
            leafFunc->outCasts_.emplace_back(tensor.tensor);
            leafFunc->InsertOpmagicToOutcastIdx(tensor.opMagic, leafFunc->outCasts_.size() - 1);
        } else {
            leafFunc->inCasts_.emplace_back(tensor.tensor);
            leafFunc->InsertOpmagicToIncastIdx(tensor.opMagic, leafFunc->inCasts_.size() - 1);
        }
    }
}

Status SubgraphToFunction::ProcessSubgraph(Function& function, size_t i, 
                                         size_t& programIdx, 
                                         std::vector<Function*>& outputFuncList) {
    auto subgraph = nLIST[i];
    auto leafName = function.GetRawName() + "_leaf" + std::to_string(i);
    ALOG_DEBUG_F("Add leafFunction %s", leafName.c_str());

    Program::GetInstance().BeginFunction(leafName, FunctionType::STATIC, GraphType::LEAF_GRAPH);
    auto leafFunc = Program::GetInstance().GetCurrentFunction();
    leafFunc->SetProgramOp(subgraph);
    InsertParameter(i, leafFunc);

    //In EndFunction to calculate cache hash
    auto result = Program::GetInstance().EndFunction(leafName);
    auto callOp = std::get<1>(result);
    callOp->UpdateSubgraphID(i);
    callOp->SetSubFuncInvokeInfo(subFuncInvokeInfos[i]);

    SetSemanticLabel(subgraph, callOp);

    return ProcessCacheResult(result, i, programIdx, outputFuncList, callOp);
}

Status SubgraphToFunction::ProcessCacheResult(const std::tuple<Function*, Operation*, bool>& result, 
                                            size_t i, size_t& programIdx, 
                                            std::vector<Function*>& outputFuncList, 
                                            Operation* callOp) {
    const int getValue = 2;
    // 3.1 Hit subgraph
    if (std::get<getValue>(result)) {
        ALOG_DEBUG_F("####### leafFunc %zu Hit Current hashValue is %lu, ######", 
                     i, std::get<0>(result)->ComputeHash().GetHash());
        psgToESgMap.insert({std::get<0>(result)->GetProgramId(), i});
        auto callAttr = dynamic_cast<CallOpAttribute *>(callOp->GetOpAttribute().get());
        if (callAttr == nullptr) { ALOG_ERROR_F("Failed to get CallOpAttribute for operation %zu", i); return FAILED; }
        auto cacheValue = Program::GetInstance().GetHostMachine().TryHitCahce(callAttr->GetCalleeHash());
        if (!cacheValue) { ALOG_ERROR_F("Cache miss for callee hash %lu", callAttr->GetCalleeHash()); return FAILED; }
        callAttr->SetCalleeMagicName(cacheValue->cacheFunction->GetMagicName());
        callAttr->invokeInfo_->UpdateProgramSubgraphId(std::get<0>(result)->GetProgramId());
    } else {
        // 3.2 not hit subgraph
        ALOG_DEBUG_F("######## leafFunc %zu Not Hit. hashValue is %lu, ######", 
                    i, std::get<0>(result)->ComputeHash().GetHash());
        psgToESgMap.insert({programIdx, i});
        std::get<0>(result)->SetProgramId(programIdx);
        auto callAttr = dynamic_cast<CallOpAttribute *>(callOp->GetOpAttribute().get());
        if (callAttr == nullptr) { ALOG_ERROR_F("Failed to get CallOpAttribute for operation %zu", i); return FAILED; }
        callAttr->invokeInfo_->UpdateProgramSubgraphId(programIdx);
        programIdx++;
        outputFuncList.push_back(std::get<0>(result));
        std::get<0>(result)->UpdateBelongToThis();
    }
    return SUCCESS;
}

void SubgraphToFunction::SetSemanticLabel(const std::vector<std::shared_ptr<Operation>>& subgraph, Operation* callOp) {
    std::string tag;   
    if (GetConfig("USE_MAX_FREQ_LABEL", false)) {
        std::unordered_map<std::string, int> frequencyMap;
        // Count the occurrences of each string
        for (const auto& op : subgraph) {
            auto str = op->GetSemanticLabel();
            frequencyMap[str]++;
        }
        // Find the string with the maximum occurrence
        std::string maxOccurrenceString;
        int maxCount = 0;
        for (const auto& pair : frequencyMap) {
            if (pair.second > maxCount) {
                maxCount = pair.second;
                maxOccurrenceString = pair.first;
            }
        }
        tag = maxOccurrenceString;
    } else {
        if (subgraph.size() > 0) {
            tag = subgraph[0]->GetSemanticLabel();
        }
    }
    callOp->SetSemanticLabel(tag);     
}

CoreType SubgraphToFunction::DetermineGraphType(size_t i) {
    int32_t cubeOpCnt = 0;
    int32_t vecOpCnt = 0;
    int32_t aicpuOpCnt = 0;
    for (size_t j = 0; j < nLIST[i].size(); j++) {
        if (IsCubeOp(*nLIST[i][j])) {
            cubeOpCnt += 1;
        } else if(IsAICPUOp(*nLIST[i][j])){
            aicpuOpCnt += 1;
        } else {
            vecOpCnt += 1;
        }
    }
    CoreType esgGraphType = CoreType::AIV;
    if (aicpuOpCnt > 0){
        esgGraphType = CoreType::AICPU;
    } else if (cubeOpCnt == 0 && vecOpCnt > 0) {
        esgGraphType = CoreType::AIV;
    } else if (cubeOpCnt > 0 && vecOpCnt == 0) {
        esgGraphType = CoreType::AIC;
    } else if (cubeOpCnt > 0 && vecOpCnt > 0) {
        esgGraphType = CoreType::MIX;
    }
    if(nLIST[i].size()==1 && nLIST[i][0]->GetOpcode() == Opcode::OP_RESHAPE && colorInGraph[i].size() != 0){
        esgGraphType = CoreType::HUB;
    }

    return esgGraphType;
}

Status SubgraphToFunction::HandleReadyStates(Function* rootFunc) {
    if (rootFunc == nullptr) { ALOG_ERROR("Root function is nullptr"); return FAILED; }
    for (size_t i = 0; i < nLIST.size(); i++) {
        CoreType esgGraphType = DetermineGraphType(i);
        // Get the operation and verify it exists
        if (i >= rootFunc->Operations().size()) { ALOG_ERROR_F("Operation index %zu out of bounds (total operations: %zu)", i, rootFunc->Operations().size()); return FAILED; }
        auto& op = rootFunc->Operations()[i];
        auto callAttr = dynamic_cast<CallOpAttribute *>(op.GetOpAttribute().get());
        if (callAttr == nullptr) { ALOG_ERROR_F("Failed to get CallOpAttribute for operation %zu (opcode: %s)", i, op.GetOpcodeStr().c_str()); return FAILED; }
        callAttr->invokeInfo_->SetGraphType(esgGraphType);
        // Verify topology index is valid
        if (i >= rootFunc->topoInfo_.topology_.size()) { ALOG_ERROR_F("Topology index %zu out of bounds (total topology entries: %zu)", i, rootFunc->topoInfo_.topology_.size()); return FAILED; }
        if (rootFunc->topoInfo_.topology_[i].readyState == 0) {
            if (esgGraphType == CoreType::AIC) {
                rootFunc->EmplaceReadySubGraphIds(CoreType::AIC, i);
                ALOG_DEBUG_F("!!!!!Esg %zu is ready aic sub graph.", i);
            } else if (esgGraphType == CoreType::AIV) {
                rootFunc->EmplaceReadySubGraphIds(CoreType::AIV, i);
                ALOG_DEBUG_F("!!!!!Esg %zu is ready aiv sub graph.", i);
            } else if (esgGraphType == CoreType::AICPU) {
                rootFunc->EmplaceReadySubGraphIds(CoreType::AICPU, i);
                ALOG_DEBUG_F("!!!!!Esg %zu is ready aicpu sub graph.", i);
            } else if (esgGraphType == CoreType::MIX) {
                ALOG_DEBUG_F("!!!!!Esg %zu is ready mix sub graph.", i);
            }
        }
    }
    return SUCCESS;
}

Status SubgraphToFunction::IslandToFunction(Function &function) {
    // 1. Create root function
    auto rootName = Function::CreateRootRawName(function.GetRawName());
    Program::GetInstance().BeginFunction(rootName, function.GetFunctionType(), GraphType::ROOT_GRAPH);
    auto rootFunc = Program::GetInstance().GetCurrentFunction();
    if (rootFunc == nullptr) { ALOG_ERROR_F("Failed to create root function"); return FAILED; }
    InitializeRootFunction(function, rootFunc);
    
    // 2. Call HashInterface to compute hash value to determine isomorphsim of each subgraph.
    size_t programIdx = 0;    
    for (size_t i = 0; i < nLIST.size(); i++) {
        Status status = ProcessSubgraph(function, i, programIdx, mergedFuncList);
        if (status != SUCCESS) { ALOG_ERROR_F("Failed to process subgraph %zu", i); return status; }
    }        
        
    // 3. Finalize root function       
    auto rootEndResult = Program::GetInstance().EndFunction(rootName, false);
    auto resultFunc = std::get<0>(rootEndResult);
    if (resultFunc != rootFunc) { ALOG_ERROR_F("Root function mismatch after finalization"); return FAILED; }
    rootFunc->topoInfo_ = function.topoInfo_;
    function.rootFunc_ = rootFunc;
    
    // 4. Add graphType of ESGInvokeInfoMap
    Status readyStateStatus = HandleReadyStates(rootFunc);
    if (readyStateStatus != SUCCESS) { ALOG_ERROR("Failed to handle ready states"); return readyStateStatus; } 
    
    // 5. symbolize esg to program subgraph
    if (rootFunc->GetFunctionType() != FunctionType::DYNAMIC_LOOP_PATH) {
        SymbolizeFunction(rootFunc, mergedFuncList);
    } else {
        for (size_t i = 0; i < mergedFuncList.size(); i++) {
            rootFunc->programs_.emplace(std::make_pair(i, mergedFuncList[i]));
        }
    }

    auto graphNum = nLIST.size();
    ALOG_INFO_F("#### Compressed Graph #### %zu, total_Graph %zu", programIdx, graphNum);
    for (auto &p : psgToESgMap) {
        ALOG_INFO_F("#### after SymbolizeFunction PSgToESgMap[%d] = %d ####", p.first, p.second);
    }
    
    return SUCCESS;
}

void SubgraphToFunction::InitializeRootFunction(Function& function, Function* rootFunc) {
    rootFunc->SetParent(nullptr);
    if (function.IsFunctionTypeAndGraphType(FunctionType::DYNAMIC_LOOP_PATH, {GraphType::TENSOR_GRAPH, GraphType::TILE_GRAPH})) {
        rootFunc->SetDynloopAttribute(function.GetDynloopAttribute());
    }
    for (auto &tensor: function.inCasts_) {
        rootFunc->inCasts_.push_back(tensor->Clone(*rootFunc));
    }
    for (auto &tensor: function.outCasts_) {
        rootFunc->outCasts_.push_back(tensor->Clone(*rootFunc));
    }
    ALOG_DEBUG_F("Root function tensor map size is %zu %zu",
        rootFunc->GetTensorMap().inverseMap_.size(), rootFunc->GetTensorMap().tensorMap_.size());
}
} // namespace npu::tile_fwk

