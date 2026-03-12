/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mixsubgraph_merger.h
 * \brief Mix subgraph merge algorithm for CV mixed graph optimization.
 */

#ifndef PASS_MIXSUBGRAPH_MERGER_H_
#define PASS_MIXSUBGRAPH_MERGER_H_

#include <vector>
#include <unordered_map>
#include <set>
#include <utility>

namespace npu::tile_fwk {

struct MixSubgraphMergerInput {
    int numOp;
    int numSubgraph;
    int maxLatency;
    std::pair<double, double> aivRatio;
    std::vector<int> opSubgraph;
    std::vector<int> opLatency;
    std::vector<bool> isCubeSubgraph;
    std::unordered_map<int, std::set<int>> opOutGraph;
    std::vector<std::pair<std::vector<int>, int>> mergeGroup;
};

struct MixSubgraphMergerOutput {
    int numSubgraphUpdated;
    std::vector<int> opSubgraphUpdated;
};

class MixSubgraphMerger {
public:
    MixSubgraphMerger() = default;
    ~MixSubgraphMerger() = default;

    MixSubgraphMergerOutput Merge(const MixSubgraphMergerInput& input);

private:
    struct SubgraphInfo {
        int aivLatency;
        int aicLatency;
        bool isMixed;
        
        SubgraphInfo() : aivLatency(0), aicLatency(0), isMixed(false) {}
    };

    bool ValidateInput(const MixSubgraphMergerInput& input);
    bool ValidateOutput(const MixSubgraphMergerOutput& output, int numOp);
    void CalcSubgraphLatency(const MixSubgraphMergerInput& input,
                             std::vector<SubgraphInfo>& subgraphInfos);
    void BuildSubgraphDeps(const MixSubgraphMergerInput& input,
                           std::unordered_map<int, std::set<int>>& subgraphDeps);
    bool CanMerge(int newAivLatency, int newAicLatency, int maxLatency,
                  const std::pair<double, double>& aivRatio);
    bool WouldCreateCycle(const std::set<int>& subgraphsToMerge,
                          const std::unordered_map<int, std::set<int>>& subgraphDeps,
                          int mergedId);
};

}

#endif