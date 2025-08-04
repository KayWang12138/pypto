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
 * \file subgraph_to_function.h
 * \brief
 */

#ifndef PASS_SUGGRAPH_TO_FUNCTION_H_
#define PASS_SUGGRAPH_TO_FUNCTION_H_

#include <vector>
#include "passes/pass_interface/pass.h"
#include "interface/operation/opcode.h"
#include "tilefwk/data_type.h"
#include "passes/pass_utils/pass_utils.h"

namespace npu::tile_fwk {
class SubgraphToFunction : public Pass {
public:
    SubgraphToFunction() : Pass("SubgraphToFunction") {}
    ~SubgraphToFunction() override = default;

private:
    Status RunOnFunction(Function &function) override;
    void GetTensorDataDependencyInsert(Function &function);
    void GetTensorDataDependencyClear(Function &function);
    Status BuildGraph(Function &function);
    void InsertParameter(size_t i, Function* leafFunc);
    Status NOPCheck(const Operation &op) const;
    Status CheckSubGraphTopo(Function &function) const;
    template <typename eType>
    Status InAndOutGraphConsistencyCheck(
        const std::vector<std::vector<eType>> &inEdgeGraph,
        const std::vector<std::vector<eType>> &outEdgeGraph);
    Status EdgeIndexCheck(const bool found, const int newIndex, const size_t graphSize) const;
    Status BuildInGraph(Function &function);
    Status BuildOutGraph(Function &function);
    Status CheckInAndOutGraphMatch(Function &function);
    Status CheckSubGraphBoundary(Function &function);
    bool foundNodeInNeighbor(const int dstNode, const std::vector<int> &searchGraph) const;
    Status VerifyRedundantEdge(const int srcNode, const int dstNode) const;
    Status ColorOutGraphCheck(Function &function) const;
    Status PreCheck(Function &function) override;
    Status PostCheck(Function &function) override;
    Status CheckSinglePsgEsgMapping(Function &function, uint32_t psgId, uint32_t esgId);
    Status VerifySingleOpTopology(Function &function, size_t opIndex);
    Status CheckReadyStateConsistency(Function &function, size_t opIndex);
    template <typename ESGParamType, typename PSGParamContainer>
    bool CompareParamListsImpl(const std::vector<ESGParamType>& esgParams, const PSGParamContainer& psgParams, const std::string &paramType, uint32_t psgId, uint32_t esgId) const;
    bool CompareParamLists(const std::vector<SubfuncInvokeInfoTy::IncastParamPackTy>& esgParams, const SubfuncParam::InCastParamListTy& psgParams, const std::string &paramType, uint32_t psgId, uint32_t esgId) const;
    bool CompareParamLists(const std::vector<SubfuncInvokeInfoTy::OutcastParamPackTy>& esgParams, const SubfuncParam::OutCastParamListTy& psgParams, const std::string &paramType, uint32_t psgId, uint32_t esgId) const;
    bool CompareParamLists(const std::vector<SubfuncInvokeInfoTy::TensorParamPackTy>& esgParams, const SubfuncParam::TensorParamListTy& psgParams, const std::string& paramType, uint32_t psgId, uint32_t esgId) const;
    void ConstructParamMap(Function &function);
    Status ProcessSubgraph(Function& function, size_t i, size_t& programIdx, std::vector<Function*>& outputFuncList);
    Status ProcessCacheResult(const std::tuple<Function*, Operation*, bool>& result, size_t i, size_t& programIdx, std::vector<Function*>& outputFuncList, Operation* callOp);
    void SetSemanticLabel(const std::vector<std::shared_ptr<Operation>>& subgraph, Operation* callOp);
    bool IsCVSeparatePlatform();
    Status DetermineGraphType(size_t i, CoreType &esgGraphType);
    Status HandleReadyStates(Function* rootFunc);
    void InitializeRootFunction(Function& function, Function* rootFunc);
    Status IslandToFunction(Function &function);
    void RecordIncastOutcast(Function &function);
    void BuildLocalGraph(std::vector<std::vector<std::shared_ptr<Operation>>> &nLIST,
        std::vector<std::vector<std::vector<size_t>>> &localInGraph,
        std::vector<std::vector<std::vector<size_t>>> &localOutGraph);
    SubfuncTopologyInfoTy ConstructSubgraphTopologyInfo(
        Function &function, std::vector<SubfuncInvokeInfoTy> &esgInvokeInfoMap);
    void SymbolizeFunction(Function *rootFunc, std::vector<Function*> &mergedFuncList1) const;
    void BuildColorGraph(Function &function);
    void PrintColorGraph(const Function &function);
    void FindRedundantEdges(int color, std::vector<std::vector<int>>& redundantColorInGraph,
        std::vector<std::vector<int>>& redundantColorOutGraph);
    void EraseRedundantColorEdges(const Function &function);
    std::string FindSymbolName(std::shared_ptr<LogicalTensor> op, int magic) const;
    void RecordEsgIncast(Function &function, size_t i, size_t j, size_t k);
    void RecordEsgOutcast(Function &function, size_t i, size_t j, size_t k);
    void ProcessInputOperands(Function* rootFunc, Operation& tileOp, SubfuncParam& pSgParamInfo, int& tParamLoc, int& iParamLoc) const;
    void ProcessOutputOperands(Function* rootFunc, Operation& tileOp, SubfuncParam& pSgParamInfo, int& tParamLoc, int& oParamLoc) const;
    void ProcessCopyInOperand(Operation& tileOp, std::vector<int>& offset, std::vector<int>& shape) const;
    void ProcessCopyOutOperand(Operation& tileOp, std::vector<int>& offset, std::vector<int>& shape) const;
    void SymbolizeEachFunction(Function *rootFunc, std::vector<Function *> &mergedFuncList1, size_t i) const;

    std::vector<std::vector<OperationPtr>> nLIST;
    std::vector<std::vector<size_t>> inGraph;
    std::vector<std::vector<size_t>> outGraph;
    std::vector<bool> isReshape;
    std::vector<std::vector<int>> colorInGraph;
    std::vector<std::vector<int>> colorOutGraph;
    std::vector<Function *> mergedFuncList;
    std::vector<std::vector<OperationPtr>> mergedSubgraphList;
    std::multimap<int, int> psgToESgMap;
    std::vector<int64_t> subgTopoParamOffsets;
    std::vector<SubfuncInvokeInfoTy> subFuncInvokeInfos;
    bool printDetails = false;
};
} // namespace npu::tile_fwk
#endif // PASS_SUGGRAPH_TO_FUNCTION_H_
