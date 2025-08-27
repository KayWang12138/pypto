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
 * \file subgraph_to_function_checker.h
 * \brief
 */

#ifndef SUBGRAPH_TO_FUNCTION_CHECKER_H
#define SUBGRAPH_TO_FUNCTION_CHECKER_H

#include "checker.h"
#include "interface/utils/common.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
namespace npu {
namespace  tile_fwk {
class SubGraphToFuncChecker : Checker {
  public:
    Status DoPreCheck(Function &function) override;
    Status DoPostCheck(Function &function) override;
    void SetInOutGraph(const std::vector<std::vector<size_t>> &inGraph,
                       const std::vector<std::vector<size_t>> &outGraph);
    void SetColorGraph(const std::vector<std::vector<int>> &colorInGraph_,
                       const std::vector<std::vector<int>> &colorOutGraph_);
    void SetPsgToESgMap(const std::multimap<int, int> &psgToESgMap);
  private:
    Status NOPCheck(const Operation &op) const;
    Status CheckSubGraphTopo(Function &function) const;
    template <typename eType>
    Status InAndOutGraphConsistencyCheck(
        const std::vector<std::vector<eType>> &inEdgeGraph,
        const std::vector<std::vector<eType>> &outEdgeGraph);
    bool foundNodeInNeighbor(const int dstNode, const std::vector<int> &searchGraph) const;
    Status BuildInGraph(Function &function);
    Status BuildOutGraph(Function &function);
    Status EdgeIndexCheck(const bool found, const int newIndex, const size_t graphSize) const;
    Status CheckInAndOutGraphMatch(Function &function);
    Status CheckSubGraphBoundary(Function &function);
    Status VerifyRedundantEdge(const int srcNode, const int dstNode) const;
    Status ColorOutGraphCheck(Function &function) const;
     Status CheckSinglePsgEsgMapping(Function &function, uint32_t psgId, uint32_t esgId);  
    Status VerifySingleOpTopology(Function &function, size_t opIndex);
    Status CheckReadyStateConsistency(Function &function, size_t opIndex);
    template <typename ESGParamType, typename PSGParamContainer>
    bool CompareParamListsImpl(const std::vector<ESGParamType>& esgParams, const PSGParamContainer& psgParams,
                               const std::string &paramType, uint32_t psgId, uint32_t esgId) const;
    bool CompareParamLists(const std::vector<SubfuncInvokeInfoTy::IncastParamPackTy>& esgParams,
                           const SubfuncParam::InCastParamListTy& psgParams, const std::string &paramType,
                           uint32_t psgId, uint32_t esgId) const;
    bool CompareParamLists(const std::vector<SubfuncInvokeInfoTy::OutcastParamPackTy>& esgParams,
                           const SubfuncParam::OutCastParamListTy& psgParams, const std::string &paramType,
                           uint32_t psgId, uint32_t esgId) const;
    bool CompareParamLists(const std::vector<SubfuncInvokeInfoTy::TensorParamPackTy>& esgParams,
                           const SubfuncParam::TensorParamListTy& psgParams, const std::string& paramType,
                           uint32_t psgId, uint32_t esgId) const;
  private:
    std::vector<std::vector<size_t>> inGraph_;
    std::vector<std::vector<size_t>> outGraph_;
    std::vector<std::vector<int>> colorInGraph_;
    std::vector<std::vector<int>> colorOutGraph_;
    std::multimap<int, int> psgToESgMap_;
    const int kShapePlaceholderForParameterized = -2;
};
}  // namespace tile_fwk
}  // namespace npu
#endif  // SUBGRAPH_TO_FUNCTION_H