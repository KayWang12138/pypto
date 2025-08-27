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
#include "passes/statistics/execute_graph_statistic.h"
#include "passes/pass_check/subgraph_to_function_checker.h"

namespace npu::tile_fwk {
class SubgraphToFunction : public Pass {
public:
    SubgraphToFunction() : Pass("SubgraphToFunction") {}
    ~SubgraphToFunction() override = default;

private:
    Status PreCheck(Function &function) override;
    Status PostCheck(Function &function) override;
    Status RunOnFunction(Function &function) override;
    
    void GetTensorDataDependencyInsert(Function &function);
    void GetTensorDataDependencyClear(Function &function);
    Status BuildGraph(Function &function);
    void InsertParameter(size_t i, Function* leafFunc);
    Status BuildInGraph(Function &function);
    Status EdgeIndexCheck(const bool found, const int newIndex, const size_t graphSize) const;
    void DoHealthCheckAfter(Function &function, const std::string &folderPath) override;
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
    void GenerateAndExportCombinedReport(Function& func,
    const std::multimap<int, int>& psgToESgMapParam,
    const std::vector<std::vector<OperationPtr>>& subgraphGroups,
    const std::string& filename="ExecuteGraph_Health_Report.json");

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
    static constexpr int kShapePlaceholderForParameterized = -2;
};
} // namespace npu::tile_fwk
#endif // PASS_SUGGRAPH_TO_FUNCTION_H_