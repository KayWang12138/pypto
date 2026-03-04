/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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

#include "passes/block_graph_pass/mix_subgraph_split/mix_subgraph_split_utils.h"

namespace npu {
namespace tile_fwk {
struct AnalyzerInput {
    std::vector<InternalComponentInfo> components;
    Function* originalMixFunc;
    AnalyzerInput(const std::vector<InternalComponentInfo> &comp, Function* func)
        : components(comp), originalMixFunc(func) {}
};

struct AnalyzerOutput {
    std::vector<InternalDependencyInfo> internalDeps;
    std::unordered_map<int, std::vector<LogicalTensorPtr>> allIncasts;
    std::unordered_map<int, std::vector<LogicalTensorPtr>> allOutcasts;
    AnalyzerOutput(const std::vector<InternalDependencyInfo> &deps,
                const std::unordered_map<int, std::vector<LogicalTensorPtr>> incasts,
                const std::unordered_map<int, std::vector<LogicalTensorPtr>> outcasts)
        : internalDeps(deps), allIncasts(incasts), allOutcasts(outcasts) {}
};

class MixDependencyAnalyzer {
public:   
    // 1.分析组件间直接依赖
    std::unordered_map<int, std::set<int>> AnalyzeComponentDependencies(Function &mixFunc,
        std::map<std::pair<int, int>, std::vector<LogicalTensorPtr>>& crossComponentTensors);
    Status ValidateCrossComponentDependencies(
        const AnalyzerInput &input,
        const std::unordered_map<int, std::set<int>>& directDeps,
        const std::map<std::pair<int, int>, std::vector<LogicalTensorPtr>>& crossComponentTensors);
    // 2.计算依赖传递闭包
    void ComputeDependencyClosure(std::unordered_map<int, std::set<int>> &dependencies);
    // 3.提取外部依赖
    void ExtractExternalDependencies();
    // 5.收集内部依赖(C-C, V-V)
    // 只用同类型的依赖需要记录，其他的需要消除
    // 需要分析同类型的依赖有哪些
    // 最后转成控制边的依赖internalDeps
    // 先识别cube/vector, component先标上
    void CollectInternalDependencies(const std::unordered_map<int, std::set<int>> &dependencyClosure,
                                    const std::vector<InternalComponentInfo> &components);

    // 外部接口
    Status ProcessDependencyAnalyzer(const AnalyzerInput &input, AnalyzerOutput &output);

    void SetOriginalMixFunc(Function* func) { originalMixFunc_ = func; }
private:
    // 完成闭包信息的初始化处理
    void InitDependencies(std::unordered_map<int, std::set<int>> &dependencies);
    // 使用Warshall算法
    void WarshallAlgorithm(std::vector<std::vector<bool>> &matrix);
    // 使用Warshall算法更新依赖关系
    void UpdateDependencies(std::unordered_map<int, std::set<int>> &dependencies);
    // 将可达阵退化为最小邻接阵
    void ObtainMinAdjMatrix(std::vector<std::vector<bool>> &matrix);
    void Reset();
    std::vector<std::vector<bool>> Transpose(const std::vector<std::vector<bool>> &matrix);
    bool CheckDirectionAndCollectValid(const std::vector<LogicalTensorPtr>& tensors, int src, int dst, bool& hasValid) const;
    void LogIllegalBidirectionalDependency(int comp1, int comp2, const AnalyzerInput& input) const;
    int maxComponent;
    std::vector<InternalDependencyInfo> internalDeps;
    std::unordered_map<int, std::vector<LogicalTensorPtr>> allIncasts;
    std::unordered_map<int, std::vector<LogicalTensorPtr>> allOutcasts;
    Function* originalMixFunc_ = nullptr;
};
} // namespace tile_fwk
} // namespace npu

#endif // MIX_DEPENDENCY_ANALYZER_H