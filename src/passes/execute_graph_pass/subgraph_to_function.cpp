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

Status SubgraphToFunction::RunOnFunction(Function &function) {
    /* 需要将所有缓存在类成员的信息清零 */
    subFuncInvokeInfos.clear();
    // build in-graph and out-graph at first
    // GetTensorData: Add dependency
    GetTensorDataDependencyInsert(function);    
    // 1. Construct in-graph & out-graph
    if (BuildGraph(function) != SUCCESS) {
        ASLOGE("failed to build graph from input function");
        return FAILED;
    }
    // reconnect in-graph and out-graph by Incast and Outcast
    RecordIncastOutcast(function);
    // Construct funtion.subFunctionInvokeMap
    ConstructParamMap(function);
    // Determine the isomorphism of subgraphs and record ProgramInfoMap
    Function::EnableMagicLookupRecord(true, &function);
    IslandToFunction(function);
    Function::EnableMagicLookupRecord(false, &function);
    // GetTensorData: Remove dependency
    GetTensorDataDependencyClear(function);
    return SUCCESS;
}
    
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
            offset.push_back(opImm.GetSpecifiedValue().ConcreteValid() ?
                static_cast<int>(opImm.GetSpecifiedValue()) : -1);
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
            offset.push_back(opImm.GetSpecifiedValue().ConcreteValid() ?
                static_cast<int>(opImm.GetSpecifiedValue()) : -1);
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
            iter.RecordOutcast(i, nLIST[i][j]->GetIntAttribute(OpAttributeKey::seqNo), k,
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
    std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(tileOp.GetOpAttribute());
    if (!attr) {
        ALOG_DEBUG_F("CopyinOperand: Invalid op attribute for op %d", tileOp.GetOpMagic());
        return;
    }
    std::vector<OpImmediate> opImmList = attr->GetCopyInAttr().first;
    shape = attr->GetSpecifiedShape(1);
    if (!opImmList.empty() && opImmList[0].IsParameter()) {
        ALOG_DEBUG_F("CopyinOperand: First operand is paramter, skip offset processiong");
        return;
    }
    offset.clear();
    for (auto &opImm : opImmList){
        offset.push_back(opImm.GetSpecifiedValue().ConcreteValid() ? static_cast<int>(opImm.GetSpecifiedValue()) : -1);
    }   
}

void SubgraphToFunction::ProcessCopyOutOperand(Operation& tileOp, std::vector<int>& offset, std::vector<int>& shape) const{    
    std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(tileOp.GetOpAttribute());
    if (!attr) {
        ALOG_DEBUG_F("CopyOutOperand: Invalid op attribute for op %d", tileOp.GetOpMagic());
        return;
    }
    std::vector<OpImmediate> opImmList = attr->GetCopyOutAttr().second;
    shape = attr->GetSpecifiedShape(1);
    if (!opImmList.empty() && opImmList[0].IsParameter()) {
        ALOG_DEBUG_F("CopyOutOperand: First operand is paramter, skip offset processiong");
        return;
    }
    offset.clear();
    for (auto &opImm : opImmList){
        offset.push_back(opImm.GetSpecifiedValue().ConcreteValid() ? static_cast<int>(opImm.GetSpecifiedValue()) : -1);
    }   
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

Status SubgraphToFunction::BuildInGraph(Function &function) {
    auto operationViewer = function.Operations();
    for (size_t i = 0; i < operationViewer.size(); i++) {
        inGraph[i].clear();
        outGraph[i].clear();
        // inGraph
        for (auto &inOperand : operationViewer[i].GetIOperands()) {
            for (auto &parentOp : inOperand->GetProducers()) {
                auto [parentSeqNo, found] = operationViewer.FindOpPosition(*parentOp);
                if (EdgeIndexCheck(found, parentSeqNo, inGraph.size()) != SUCCESS) {
                    ALOG_ERROR_F("error inserting op magic %d in function %d %s to inGraph", parentOp->GetOpMagic(), function.GetFuncMagic(),
                        function.GetRawName().c_str());
                    return FAILED;
                }
                inGraph[i].push_back(parentSeqNo);
            }
        }

        for (const auto &inControlOp : operationViewer[i].GetInCtrlOperations()) {
            auto [parentSeqNo, found] = operationViewer.FindOpPosition(*inControlOp);
            if (EdgeIndexCheck(found, parentSeqNo, inGraph.size()) != SUCCESS) {
                ALOG_ERROR_F("error inserting op magic %d in function %d %s to inGraph", inControlOp->GetOpMagic(), function.GetFuncMagic(),
                    function.GetRawName().c_str());
                return FAILED;
            }
            inGraph[i].push_back(parentSeqNo);
        }
    }
    return SUCCESS;
}

Status SubgraphToFunction::BuildGraph(Function &function) {
    auto operationViewer = function.Operations();
    inGraph.resize(operationViewer.size());
    outGraph.resize(operationViewer.size());
    if (BuildInGraph(function) != SUCCESS) {
        ALOG_ERROR_F("Build failed");
        return FAILED;
    }
    for (size_t i = 0; i < operationViewer.size(); i++) {
        for (auto parentSeqNo : inGraph[i]) {
            outGraph[parentSeqNo].push_back(i);
        }
    }
    return SUCCESS;
}

void SubgraphToFunction::InsertParameter(size_t i, Function* leafFunc) {
    for (auto &in : subFuncInvokeInfos[i].GetIncastTensorParamList()) {
        leafFunc->AppendIncast(in.tensor, in.opMagic, in.operandIdx);
    }
    for (auto &out : subFuncInvokeInfos[i].GetOutcastTensorParamList()) {
        leafFunc->AppendOutcast(out.tensor, out.opMagic, out.operandIdx);
    }
    for (auto &tensor : subFuncInvokeInfos[i].GetTensorParamList()) {
        leafFunc->AddGlobalTensor(tensor.tensor);
        if (tensor.isOutputToGM) {
            leafFunc->AppendOutcast(tensor.tensor, tensor.opMagic, tensor.operandIdx);
        } else {
            leafFunc->AppendIncast(tensor.tensor, tensor.opMagic, tensor.operandIdx);
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
    if (callOp == nullptr) {
        ALOG_ERROR_F("leafname %s, program returned nullptr");
        return FAILED;
    }
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

bool SubgraphToFunction::IsCVSeparatePlatform() {
    auto socVersion = config::GetDevicePlatform();
    if (socVersion == DPlatform::ASCEND_910B1 || socVersion == DPlatform::ASCEND_910B2 || socVersion == DPlatform::ASCEND_910B3 || socVersion == DPlatform::ASCEND_910B4) {
        return true;
    }
    return false;
}

Status SubgraphToFunction::DetermineGraphType(size_t i, CoreType &esgGraphType) {
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

    if (aicpuOpCnt > 0){
        esgGraphType = CoreType::AICPU;
    } else if (cubeOpCnt == 0 && vecOpCnt > 0) {
        esgGraphType = CoreType::AIV;
    } else if (cubeOpCnt > 0 && vecOpCnt == 0) {
        esgGraphType = CoreType::AIC;
    } else if (cubeOpCnt > 0 && vecOpCnt > 0) {
        esgGraphType = CoreType::MIX;
        if (IsCVSeparatePlatform() == true) {
            ALOG_ERROR_F("Get CoreType::MIX in C-V separate platform");
            return FAILED;
        }        
    }
    if(nLIST[i].size()==1 && nLIST[i][0]->GetOpcode() == Opcode::OP_RESHAPE && colorInGraph[i].size() != 0){
        esgGraphType = CoreType::HUB;
    }

    return SUCCESS;
}

Status SubgraphToFunction::HandleReadyStates(Function* rootFunc) {
    if (rootFunc == nullptr) { ALOG_ERROR("Root function is nullptr"); return FAILED; }
    for (size_t i = 0; i < nLIST.size(); i++) {
        CoreType esgGraphType = CoreType::AIV;
        if (DetermineGraphType(i, esgGraphType) != SUCCESS ) {
            ALOG_ERROR_F("DetermineGraphType failed");
            return FAILED;
        }
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
    
    // 2. Call HashInterface to compute hash value to determine isomorphism of each subgraph.
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
    
    // 5. symbolize esg to program subgraph for both static and dynamic paths
    SymbolizeFunction(rootFunc, mergedFuncList);

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

void SubgraphToFunction::GetTensorDataDependencyInsert(Function &function) {
    auto operationViewer = function.Operations(false);
    struct GetTensorDataOutcastDesc {
        std::unordered_map<Opcode, std::vector<Operation *>> opListDict;
        Operation *mark;
        Operation *copyout;
        std::shared_ptr<LogicalTensor> outcast;
    };
    std::unordered_map<int, GetTensorDataOutcastDesc> getTensorDataOutcastDescDict;
    for (size_t i = 0; i < operationViewer.size(); i++) {
        auto &op = operationViewer[i];
        if (op.HasAttr(OP_EMUOP_PREFIX + "GetTensorData_index")) {
            int index = *op.GetAttr<int>(OP_EMUOP_PREFIX + "GetTensorData_index");
            getTensorDataOutcastDescDict[index].opListDict[op.GetOpcode()].push_back(&op);
        }
    }
    for (auto &[index, desc] : getTensorDataOutcastDescDict) {
        (void)index;
        ASSERT(desc.opListDict[Opcode::OP_ADDS].size() == 1);
        auto mark = desc.opListDict[Opcode::OP_ADDS][0];

        std::shared_ptr<LogicalTensor> addsOpOut = mark->GetOOperands()[0];
        auto copyout = *addsOpOut->GetConsumers().begin();
        ASSERT(copyout->GetOpcode() == Opcode::OP_COPY_OUT);

        auto outcast = copyout->GetOOperands()[0];

        desc.mark = mark;
        desc.copyout = copyout;
        desc.outcast = outcast;
    }

    struct GetTensorDataDesc {
        Operation *refOp;
        std::vector<int> indexList;
        MemoryType subgraphMemoryType;
        int subgraphID;

        GetTensorDataDesc(Operation *refOp_, std::vector<int> indexList_, MemoryType subgraphMemoryType_, int subgraphID_)
            : refOp(refOp_), indexList(indexList_), subgraphMemoryType(subgraphMemoryType_), subgraphID(subgraphID_) {}
    };
    std::vector<GetTensorDataDesc> getTensorDataDescList;
    for (size_t i = 0; i < operationViewer.size(); i++) {
        auto &refOp = operationViewer[i];
        auto attr = std::static_pointer_cast<CopyOpAttribute>(refOp.GetOpAttribute());

        std::vector<OpImmediate> dynAttrList;
        std::shared_ptr<LogicalTensor> subgraphTensor;
        switch (refOp.GetOpcode()) {            
            case Opcode::OP_COPY_IN:
                dynAttrList = attr->GetFromOffset();
                subgraphTensor = refOp.GetOOperands()[0];
                break;
            case Opcode::OP_COPY_OUT:
                dynAttrList = attr->GetToOffset();
                subgraphTensor = refOp.GetIOperands()[0];
                break;
            default:
                break;
        }
        if (dynAttrList.size() == 0) {
            continue;
        }
        std::vector<SymbolicScalar> dynAttrScalarList;
        for (auto &dynAttr : dynAttrList) {
            if (dynAttr.IsSpecified()) {
                dynAttrScalarList.push_back(dynAttr.GetSpecifiedValue());
            }
        }
        std::map<int, RawSymbolicScalarPtr> outcastDict = GetTensorDataDict(dynAttrScalarList);
        if (outcastDict.size() == 0) {
            continue;
        }
        MemoryType subgraphMemoryType = subgraphTensor->GetMemoryTypeToBe();
        int subgraphID = subgraphTensor->GetSubgraphID();

        std::vector<int> indexList;
        for (auto [index, _] : outcastDict) {
            (void)_;
            indexList.push_back(index);
        }
        getTensorDataDescList.emplace_back(&refOp, indexList, subgraphMemoryType, subgraphID);
    }
    for (auto &[refOp, indexList, subgraphMemoryType, subgraphID] : getTensorDataDescList) {
        for (int index : indexList) {
            ASSERT(getTensorDataOutcastDescDict.count(index)) << "Index: " << index << " not found!\n";
            auto &outcastDesc = getTensorDataOutcastDescDict[index];
            auto outcastAttr = std::static_pointer_cast<CopyOpAttribute>(outcastDesc.copyout->GetOpAttribute());

            std::shared_ptr<LogicalTensor> copyInTensor = std::make_shared<LogicalTensor>(function, outcastDesc.outcast->Datatype(), outcastDesc.outcast->GetShape());
            copyInTensor->UpdateSubgraphID(subgraphID);
            copyInTensor->SetMemoryTypeBoth(subgraphMemoryType);

            auto &copyInOp = function.AddOperation(Opcode::OP_COPY_IN, {outcastDesc.outcast}, {copyInTensor}, false);
            auto copyInAttr = std::make_shared<CopyOpAttribute>(outcastAttr->GetToOffset(), MemoryType::MEM_UB, outcastAttr->GetShape(), outcastAttr->GetRawShape());
            copyInOp.UpdateSubgraphID(subgraphID);
            copyInOp.SetOpAttribute(copyInAttr);
            copyInOp.SetAttr<int>(OP_EMUOP_PREFIX + "opc", EMUOP_TENSOR_GETDATA);
            copyInOp.SetAttr<int>(OP_EMUOP_PREFIX + "GetTensorData_index", index);

            refOp->GetIOperands().push_back(copyInTensor);
            copyInTensor->AddConsumer(refOp);
        }
    }
}

void SubgraphToFunction::GetTensorDataDependencyClear(Function &function) {
    auto root = function.GetRootFunction();
    std::vector<Function *> leafList = root->GetCalleeFunctionList();
    std::unordered_set<Function *> leafSet(leafList.begin(), leafList.end());

    SymbolicScalar getAddr = SymbolicScalar(AddRuntimeCoaPrefix("GET_PARAM_ADDR"));
    for (auto &leaf : leafSet) {
        auto iodescDict = leaf->GetTensorDataForLeafGraph();

        for (auto &op : leaf->Operations()) {
            if (!op.HasAttr(OP_EMUOP_PREFIX + "opc")) {
                continue;
            }
            if (*op.GetAttr<int>(OP_EMUOP_PREFIX + "opc")  != EMUOP_TENSOR_GETDATA) {
                continue;
            }
            auto &copyInOp = op;
            copyInOp.SetAsDeleted();

            int tensorIndex = *op.GetAttr<int>(OP_EMUOP_PREFIX + "GetTensorData_index");
            int addrIndex = *op.GetAttr<int>(OP_EMUOP_PREFIX + "GetTensorData_coaIndex");
            iodescDict[tensorIndex].address = getAddr(-1, addrIndex);
        }
        leaf->EraseOperations(true, true);
        leaf->GetTensorDataRefreshIO(iodescDict);
    }
}

} // namespace npu::tile_fwk
