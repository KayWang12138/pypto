/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mix_subgraph_split_utils.h
 * \brief 将Mix子图拆分为多个独立的Cube和Vector子图，并重新分配subgraphID
 */

#ifndef MIX_SUBGRAPH_SPLIT_UTILS_H
#define MIX_SUBGRAPH_SPLIT_UTILS_H

#include "passes/pass_interface/pass.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/program/program.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "passes/tile_graph_pass/subgraph_to_function.h"
#include <unordered_map>
#include <set>
#include <vector>

namespace npu {
namespace tile_fwk {
class MixSubgraphSplitUtils {
public:
    static void BroadcastDependencyClosure(std::set<int> &deps_i, std::set<int> &newDeps, std::unordered_map<int, std::set<int>> &closure, bool &changed, int i);
    static void InitiateClosure(const std::unordered_map<int, std::vector<int>>& directDeps, std::unordered_map<int, std::set<int>> &closure);
    static void CalculateClosure(std::unordered_map<int, std::set<int>> &closure);
    static void UpdateOperandsForIncast(const std::vector<SubfuncInvokeInfoTy::IncastParamPackTy> &incastParamList,
                                        const LogicalTensors &originalTensors, 
                                        const LogicalTensors &originalOperands, 
                                        LogicalTensors &newOperands, 
                                        std::set<LogicalTensorPtr> &processedTensors);
    static void UpdateOperandsForOutcast(const std::vector<SubfuncInvokeInfoTy::OutcastParamPackTy> &outcastParamList,
                                        const LogicalTensors &originalTensors, 
                                        const LogicalTensors &originalOperands, 
                                        LogicalTensors &newOperands, 
                                        std::set<LogicalTensorPtr> &processedTensors);
    static void UpdateOperandsForGlobalTensor(const std::vector<SubfuncInvokeInfoTy::TensorParamPackTy> &paramList,
                                            const LogicalTensors &originalTensors, 
                                            const LogicalTensors &originalOperands, 
                                            LogicalTensors &newOperands, 
                                            std::set<LogicalTensorPtr> &processedTensors);
    static void UpdateBroadcastForInOutCast(const LogicalTensors &actualTensors, 
                                            const LogicalTensors &originalTensors, 
                                            const LogicalTensors &originalOperands, 
                                            LogicalTensors &newOperands, 
                                            std::set<LogicalTensorPtr> &processedTensors);
                                            
    // 搜索函数
    static Operation* FindFirstOpForward(Operation* startOp,
                                        Function& mixSubgraphFunc,
                                        std::function<bool(Operation*)> predicate);
    static Operation* FindFirstOpBackward(Operation* startOp,
                                        Function& mixSubgraphFunc,
                                        std::function<bool(Operation*)> predicate);
    // 判断是否为同步算子
    static bool IsSyncOperation(Operation* op);
    // 检查是否为需要拆分的Mix子图
    static bool IsMixSubgraph(Function& leafFunc);

    static bool FindOriginalOffsetInProducers(LogicalTensorPtr tensor, int &offset);
    static bool FindOriginalOffsetInConsumers(LogicalTensorPtr tensor, int &offset);
};
} // namespace tile_fwk
} // namespace npu

#endif // MIX_SUBGRAPH_SPLIT_UTILS_H