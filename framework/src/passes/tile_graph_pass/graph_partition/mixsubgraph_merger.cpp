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
 * \file mixsubgraph_merger.cpp
 * \brief Mix subgraph merge algorithm implementation.
 */

#include "passes/tile_graph_pass/graph_partition/mixsubgraph_merger.h"
#include "interface/utils/log.h"
#include "passes/pass_log/pass_log.h"
#include <algorithm>
#include <queue>
#include <map>
#include <sstream>

#define MODULE_NAME "MixSubgraphMerger"

namespace npu::tile_fwk {

namespace {

class UnionFind {
public:
    std::vector<int> parent;
    std::vector<int> rank_;

    UnionFind(int n) {
        parent.resize(n);
        rank_.resize(n, 0);
        for (int i = 0; i < n; ++i) {
            parent[i] = i;
        }
    }

    int Find(int x) {
        if (parent[x] != x) {
            parent[x] = Find(parent[x]);
        }
        return parent[x];
    }

    void Union(int x, int y) {
        int rootX = Find(x);
        int rootY = Find(y);
        if (rootX == rootY) return;
        if (rank_[rootX] < rank_[rootY]) {
            parent[rootX] = rootY;
        } else if (rank_[rootX] > rank_[rootY]) {
            parent[rootY] = rootX;
        } else {
            parent[rootY] = rootX;
            rank_[rootX]++;
        }
    }
};

std::string SetToString(const std::set<int>& s) {
    std::ostringstream oss;
    oss << "{";
    bool first = true;
    for (int v : s) {
        if (!first) oss << ", ";
        oss << v;
        first = false;
    }
    oss << "}";
    return oss.str();
}

std::string VecToString(const std::vector<int>& v) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << v[i];
    }
    oss << "]";
    return oss.str();
}

}

bool MixSubgraphMerger::ValidateInput(const MixSubgraphMergerInput& input) {
    if (input.numOp < 0 || input.numSubgraph < 0) {
        return false;
    }
    if (static_cast<int>(input.opSubgraph.size()) != input.numOp) {
        return false;
    }
    if (static_cast<int>(input.opLatency.size()) != input.numOp) {
        return false;
    }
    if (static_cast<int>(input.isCubeSubgraph.size()) != input.numSubgraph) {
        return false;
    }
    for (int i = 0; i < input.numOp; ++i) {
        if (input.opSubgraph[i] < 0 || 
            input.opSubgraph[i] >= input.numSubgraph) {
            return false;
        }
        if (input.opLatency[i] <= 0) {
            return false;
        }
    }
    for (const auto& pair : input.opOutGraph) {
        if (pair.first < 0 || pair.first >= input.numOp) {
            return false;
        }
        for (int consumer : pair.second) {
            if (consumer < 0 || consumer >= input.numOp) {
                return false;
            }
        }
    }
    for (const auto& mg : input.mergeGroup) {
        for (int sg : mg.first) {
            if (sg < 0 || sg >= input.numSubgraph) {
                return false;
            }
        }
    }
    return input.aivRatio.first > 0 && input.aivRatio.second > 0 &&
           input.aivRatio.first <= input.aivRatio.second;
}

bool MixSubgraphMerger::ValidateOutput(const MixSubgraphMergerOutput& output, int numOp) {
    if (output.numSubgraphUpdated < 0) {
        return false;
    }
    if (static_cast<int>(output.opSubgraphUpdated.size()) != numOp) {
        return false;
    }
    std::set<int> usedIds;
    for (int i = 0; i < numOp; ++i) {
        int sg = output.opSubgraphUpdated[i];
        if (sg < 0 || sg >= output.numSubgraphUpdated) {
            return false;
        }
        usedIds.insert(sg);
    }
    for (int i = 0; i < output.numSubgraphUpdated; ++i) {
        if (usedIds.find(i) == usedIds.end()) {
            return false;
        }
    }
    return true;
}

void MixSubgraphMerger::CalcSubgraphLatency(const MixSubgraphMergerInput& input,
                                            std::vector<SubgraphInfo>& subgraphInfos) {
    subgraphInfos.resize(input.numSubgraph);
    for (int i = 0; i < input.numOp; ++i) {
        int sg = input.opSubgraph[i];
        if (input.isCubeSubgraph[sg]) {
            subgraphInfos[sg].aicLatency += input.opLatency[i];
        } else {
            subgraphInfos[sg].aivLatency += input.opLatency[i];
        }
    }
    for (int i = 0; i < input.numSubgraph; ++i) {
        subgraphInfos[i].isMixed = false;
    }
}

void MixSubgraphMerger::BuildSubgraphDeps(const MixSubgraphMergerInput& input,
                                          std::unordered_map<int, std::set<int>>& subgraphDeps) {
    for (const auto& pair : input.opOutGraph) {
        int srcOp = pair.first;
        int srcSg = input.opSubgraph[srcOp];
        for (int dstOp : pair.second) {
            int dstSg = input.opSubgraph[dstOp];
            if (srcSg != dstSg) {
                subgraphDeps[srcSg].insert(dstSg);
            }
        }
    }
}

bool MixSubgraphMerger::CanMerge(int newAivLatency, int newAicLatency, int maxLatency,
                                 const std::pair<double, double>& aivRatio) {
    int totalLatency = newAivLatency + newAicLatency;
    if (totalLatency > maxLatency) {
        return false;
    }
    if (newAivLatency <= 0 || newAicLatency <= 0) {
        return false;
    }
    double ratio = static_cast<double>(newAivLatency) / 
                   static_cast<double>(newAicLatency);
    return ratio >= aivRatio.first && ratio <= aivRatio.second;
}

bool MixSubgraphMerger::WouldCreateCycle(const std::set<int>& subgraphsToMerge,
                                         const std::unordered_map<int, std::set<int>>& subgraphDeps,
                                         int mergedId) {
    std::unordered_map<int, std::set<int>> newDeps;
    std::set<int> toMergeSet = subgraphsToMerge;
    for (const auto& pair : subgraphDeps) {
        int src = pair.first;
        int newSrc = (toMergeSet.count(src) > 0) ? mergedId : src;
        for (int dst : pair.second) {
            int newDst = (toMergeSet.count(dst) > 0) ? mergedId : dst;
            if (newSrc != newDst) {
                newDeps[newSrc].insert(newDst);
            }
        }
    }
    std::unordered_map<int, int> inDegree;
    for (const auto& pair : newDeps) {
        if (inDegree.find(pair.first) == inDegree.end()) {
            inDegree[pair.first] = 0;
        }
        for (int dst : pair.second) {
            inDegree[dst]++;
        }
    }
    std::queue<int> q;
    int processedCount = 0;
    int totalNodes = 0;
    std::set<int> allNodes;
    for (const auto& pair : newDeps) {
        allNodes.insert(pair.first);
        for (int dst : pair.second) {
            allNodes.insert(dst);
        }
    }
    totalNodes = static_cast<int>(allNodes.size());
    if (totalNodes == 0) {
        return false;
    }
    for (int node : allNodes) {
        if (inDegree[node] == 0) {
            q.push(node);
        }
    }
    while (!q.empty()) {
        int curr = q.front();
        q.pop();
        processedCount++;
        if (newDeps.find(curr) != newDeps.end()) {
            for (int next : newDeps.at(curr)) {
                inDegree[next]--;
                if (inDegree[next] == 0) {
                    q.push(next);
                }
            }
        }
    }
    return processedCount < totalNodes;
}

MixSubgraphMergerOutput MixSubgraphMerger::Merge(const MixSubgraphMergerInput& input) {
    MixSubgraphMergerOutput output;
    output.numSubgraphUpdated = input.numSubgraph;
    output.opSubgraphUpdated = input.opSubgraph;
    
    if (!ValidateInput(input)) {
        APASS_LOG_INFO_F(Elements::Operation, "Input validation failed");
        return output;
    }
    if (input.numSubgraph <= 1 || input.mergeGroup.empty()) {
        APASS_LOG_INFO_F(Elements::Operation, 
            "No merge needed: numSubgraph=%d, mergeGroupSize=%zu",
            input.numSubgraph, input.mergeGroup.size());
        return output;
    }
    
    APASS_LOG_INFO_F(Elements::Operation, 
        "Start merge: initial numSubgraph=%d", input.numSubgraph);
    
    std::vector<SubgraphInfo> subgraphInfos;
    CalcSubgraphLatency(input, subgraphInfos);
    
    std::unordered_map<int, std::set<int>> subgraphDeps;
    BuildSubgraphDeps(input, subgraphDeps);
    
    UnionFind uf(input.numSubgraph);
    
    std::map<std::vector<int>, int> mergedGroups;
    for (const auto& mg : input.mergeGroup) {
        std::vector<int> sortedSgs = mg.first;
        std::sort(sortedSgs.begin(), sortedSgs.end());
        sortedSgs.erase(std::unique(sortedSgs.begin(), sortedSgs.end()), 
                        sortedSgs.end());
        if (static_cast<int>(sortedSgs.size()) <= 1) {
            continue;
        }
        mergedGroups[sortedSgs] += mg.second;
    }
    
    std::vector<std::pair<std::vector<int>, int>> sortedMergeGroups;
    for (const auto& pair : mergedGroups) {
        sortedMergeGroups.push_back({pair.first, pair.second});
    }
    std::sort(sortedMergeGroups.begin(), sortedMergeGroups.end(),
              [](const std::pair<std::vector<int>, int>& a,
                 const std::pair<std::vector<int>, int>& b) {
                  return a.second > b.second;
              });
    
    int currentSubgraphCount = input.numSubgraph;
    
    for (const auto& mg : sortedMergeGroups) {
        const std::vector<int>& origSgs = mg.first;
        std::set<int> actualSgs;
        for (int sg : origSgs) {
            int rootSg = uf.Find(sg);
            actualSgs.insert(rootSg);
        }
        
        APASS_LOG_INFO_F(Elements::Operation, 
            "Trying to merge subgraphs: original=%s, expanded=%s, priority=%d",
            VecToString(origSgs).c_str(), SetToString(actualSgs).c_str(), mg.second);
        
        if (static_cast<int>(actualSgs.size()) <= 1) {
            APASS_LOG_INFO_F(Elements::Operation, 
                "Merge skipped: subgraphs already merged into one");
            continue;
        }
        
        int newAivLatency = 0;
        int newAicLatency = 0;
        bool hasAiv = false;
        bool hasAic = false;
        for (int sg : actualSgs) {
            newAivLatency += subgraphInfos[sg].aivLatency;
            newAicLatency += subgraphInfos[sg].aicLatency;
            if (subgraphInfos[sg].aivLatency > 0 || 
                (!subgraphInfos[sg].isMixed && 
                 !input.isCubeSubgraph[sg] && subgraphInfos[sg].aicLatency == 0)) {
                hasAiv = true;
            }
            if (subgraphInfos[sg].aicLatency > 0 ||
                (!subgraphInfos[sg].isMixed && 
                 input.isCubeSubgraph[sg] && subgraphInfos[sg].aivLatency == 0)) {
                hasAic = true;
            }
            if (subgraphInfos[sg].isMixed) {
                hasAiv = true;
                hasAic = true;
            }
        }
        
        int totalLatency = newAivLatency + newAicLatency;
        double ratio = (newAicLatency > 0) ? 
            static_cast<double>(newAivLatency) / static_cast<double>(newAicLatency) : 0.0;
        
        if (!hasAiv || !hasAic) {
            APASS_LOG_INFO_F(Elements::Operation,
                "Merge failed: result would not be mixed subgraph (hasAiv=%d, hasAic=%d)",
                hasAiv ? 1 : 0, hasAic ? 1 : 0);
            continue;
        }
        if (totalLatency > input.maxLatency) {
            APASS_LOG_INFO_F(Elements::Operation,
                "Merge failed: totalLatency=%d exceeds maxLatency=%d",
                totalLatency, input.maxLatency);
            continue;
        }
        if (ratio < input.aivRatio.first || ratio > input.aivRatio.second) {
            APASS_LOG_INFO_F(Elements::Operation,
                "Merge failed: aiv/aic ratio=%.2f not in range [%.2f, %.2f]",
                ratio, input.aivRatio.first, input.aivRatio.second);
            continue;
        }
        
        int mergedId = *actualSgs.begin();
        if (WouldCreateCycle(actualSgs, subgraphDeps, mergedId)) {
            APASS_LOG_INFO_F(Elements::Operation,
                "Merge failed: would create cycle in dependency graph");
            continue;
        }
        
        for (int sg : actualSgs) {
            if (sg != mergedId) {
                uf.Union(mergedId, sg);
            }
        }
        subgraphInfos[mergedId].aivLatency = newAivLatency;
        subgraphInfos[mergedId].aicLatency = newAicLatency;
        subgraphInfos[mergedId].isMixed = true;
        
        std::unordered_map<int, std::set<int>> newSubgraphDeps;
        for (const auto& pair : subgraphDeps) {
            int src = pair.first;
            int newSrc = (actualSgs.count(src) > 0) ? mergedId : src;
            for (int dst : pair.second) {
                int newDst = (actualSgs.count(dst) > 0) ? mergedId : dst;
                if (newSrc != newDst) {
                    newSubgraphDeps[newSrc].insert(newDst);
                }
            }
        }
        subgraphDeps = newSubgraphDeps;
        
        int mergedCount = static_cast<int>(actualSgs.size());
        currentSubgraphCount -= (mergedCount - 1);
        
        APASS_LOG_INFO_F(Elements::Operation, 
            "Merge succeeded: merged %d subgraphs into one, "
            "totalLatency=%d (aiv=%d, aic=%d), ratio=%.2f, "
            "numSubgraph: %d -> %d",
            mergedCount, totalLatency, newAivLatency, newAicLatency, ratio,
            currentSubgraphCount + mergedCount - 1, currentSubgraphCount);
    }
    
    std::map<int, int> oldToNew;
    int newId = 0;
    for (int i = 0; i < input.numOp; ++i) {
        int oldSg = input.opSubgraph[i];
        int rootSg = uf.Find(oldSg);
        if (oldToNew.find(rootSg) == oldToNew.end()) {
            oldToNew[rootSg] = newId++;
        }
        output.opSubgraphUpdated[i] = oldToNew[rootSg];
    }
    output.numSubgraphUpdated = newId;
    
    APASS_LOG_INFO_F(Elements::Operation, 
        "Merge completed: numSubgraph %d -> %d", 
        input.numSubgraph, output.numSubgraphUpdated);
    
    if (!ValidateOutput(output, input.numOp)) {
        APASS_LOG_INFO_F(Elements::Operation, "Output validation failed");
    }
    
    return output;
}

}