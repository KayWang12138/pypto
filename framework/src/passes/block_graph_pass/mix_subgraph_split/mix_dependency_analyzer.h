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
 * \file mix_dependency_analyzer.h
 * \brief 用于提供所需的接口
 */

#ifndef MIX_DEPENDENCY_ANALYZER_H
#define MIX_DEPENDENCY_ANALYZER_H

#include <set>
#include <vector>
#include <unordered_map>
#include "passes/pass_interface/pass.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/program/program.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "passes/tile_graph_pass/subgraph_to_function.h"
#include "passes/block_graph_pass/mix_subgraph_split/mix_subgraph_split_utils.h"

namespace npu {
namespace tile_fwk {
struct SimpleTensorParam {
    LogicalTensorPtr tensor;
    int opMagic;
    int operandIdx;

    SimpleIncastParam(LogicalTensorPtr t, int magic, int idx)
        : tensor(t), opMagic(magic), operandIdx(idx) {}
};

class MixDependencyAnalyzer {
public:
    // 1.分析组件间直接依赖
    std::unordered_map<int, std::set<int>> AnalyzeComponentDependencies(Function &mixFunc);
    // 2.计算依赖传递闭包
    void ComputeDependencyClosure(std::unordered_map<int, std::set<int>> &dependencies);
    // 3.提取外部依赖
    void ExtractExternalDependencies(const SubgraphToFunction &subgraphToFunction, 
                                    std::unordered_map<int, std::vector<SimpleTensorParam>> &allIncasts,
                                    std::unordered_map<int, std::vector<SimpleTensorParam>> &allOutcasts);
    // 4.传播外部依赖(基于传递闭包)
    // 看SRS-2依赖重建
    // 传递结果记录在allIncast上
    // key->component， value: 哪些incast
    void PropagateExternalDependenciesWithClosure(const std::unorderd_map<int, std::set<int>> &dependencyClosure, 
                                                std::unordered_map<int, std::vector<SimpleTensorParam>> &allIncasts,
                                                std::unordered_map<int, std::vector<SimpleTensorParam>> &allOutcasts);
    // 5.收集内部依赖(C-C, V-V)
    // 只用同类型的依赖需要记录，其他的需要消除
    // 需要分析同类型的依赖有哪些
    // 最后转成控制边的依赖internalDeps
    // 先识别cube/vector, component先标上
    void CollectInternalDependencies(const std::unorderd_map<int, std::set<int>> &dependencyClosure,
                                    const std::vector<InternalComponentInfo> &components,
                                    std::vector<InternalDependencyInfo> &internalDeps);
    // 6.消除冗余依赖
    // 将多余的依赖转换成普通的依赖
    // 可以优化一下，只消除了外部的
    // 本质上就是看是否存在冗余依赖
    void EliminateRedundantDependencies(std::unordered_map<int, std::vector<SimpleTensorParam>> &allIncasts,
                                        std::unordered_map<int, std::vector<SimpleTensorParam>> &allOutcasts,
                                        std::vector<InternalDependencyInfo> &internalDeps);
    // 7.应用最终依赖到leaf functions
    // 将依赖重建回去
    // apply就是掉了AppendIncast上
    void ApplyFinalDependencies(const std::vector<Function*> &newFunctions,
                                std::unordered_map<int, std::vector<SimpleTensorParam>> &allIncasts,
                                std::unordered_map<int, std::vector<SimpleTensorParam>> &allOutcasts);

    // 基于可达性移除冗余的外部依赖
    void EliminateRedundantOuterDeps(const std::vector<std::vector<bool>> innerDeps, 
                                    std::unordered_map<int, std::vector<SimpleTensorParam>> &allTensors);
    // 基于可达性移除冗余的内部依赖
    void EliminateRedundantInnerDeps(std::vector<std::vector<bool>> &innerDeps,
                                    std::vector<InternalDependencyInfo> &internalDeps);

    void ApplyIncastDependencies(Function* leafFunc,
                                int componentId,
                                const std::vector<SimpleTensorParam>& incastParams);
    void ApplyOutcastDependencies(Function* leafFunc,
                                int componentId,
                                const std::vector<SimpleTensorParam>& outcastParams);
private:
    // 完成闭包信息的初始化处理
    void InitDependencies(std::unordered_map<int, std::set<int>> &dependencies);
    // 使用Warshall算法
    void WarshallAlgorithm(std::vector<std::vector<int>> &matrix);
    // 使用Warshall算法更新依赖关系
    void UpdateDependencies(std::unordered_map<int, std::set<int>> &dependencies);
    // 将可达阵退化为最小邻接阵
    void ObtainMinAdjMatrix(std::vector<std::vector<bool>> &matrix);
    // 判断是否包含对应tensor
    bool ContainsTensor(const std::vector<SimpleTensorParam> &tensors, const LogicalTensorPtr &tensor) const;
    // 构建可达阵的转置（即反向的可达阵）
    std::vector<std::vector<int>> Transpose(const std::vector<std::vector<bool>> &matrix);

    int maxComponent;
}
} // namespace tile_fwk
} // namespace npu

#endif // MIX_DEPENDENCY_ANALYZER_H