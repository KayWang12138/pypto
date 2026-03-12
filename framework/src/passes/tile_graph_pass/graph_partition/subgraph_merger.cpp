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
 * \file subgraph_merger.cpp
 * \brief Subgraph merge algorithm implementation.
 */

#include "passes/tile_graph_pass/graph_partition/subgraph_merger.h"
#include "interface/utils/log.h"
#include <algorithm>
#include <queue>

namespace {

class UnionFind {
public:
    std::vector<int> parent;
    std::vector<int> rank;

    UnionFind(int n) {
        parent.resize(n);
        rank.resize(n, 0);
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
        if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
        } else if (rank[rootX] > rank[rootY]) {
            parent[rootY] = rootX;
        } else {
            parent[rootY] = rootX;
            rank[rootX]++;
        }
    }
};

void BuildMergeGroups(int numOp, const std::vector<int>& opSubgraph,
                      const std::unordered_map<int, std::set<int>>& opOutGraph,
                      const std::unordered_map<int, std::set<int>>& opInGraph,
                      UnionFind& uf) {
    for (int opIdx = 0; opIdx < numOp; ++opIdx) {
        int srcSubgraph = opSubgraph[opIdx];
        
        auto outIt = opOutGraph.find(opIdx);
        if (outIt != opOutGraph.end()) {
            for (int consumerOp : outIt->second) {
                int dstSubgraph = opSubgraph[consumerOp];
                if (srcSubgraph != dstSubgraph) {
                    uf.Union(srcSubgraph, dstSubgraph);
                }
            }
        }
        
        auto inIt = opInGraph.find(opIdx);
        if (inIt != opInGraph.end()) {
            for (int producerOp : inIt->second) {
                int producerSubgraph = opSubgraph[producerOp];
                if (srcSubgraph != producerSubgraph) {
                    uf.Union(srcSubgraph, producerSubgraph);
                }
            }
        }
    }
}

void BuildSubgraphDag(int numOp, int, const std::vector<int>& opSubgraph,
                      const std::unordered_map<int, std::set<int>>& opOutGraph,
                      std::unordered_map<int, std::set<int>>& subgraphDag) {
    for (int opIdx = 0; opIdx < numOp; ++opIdx) {
        int srcSubgraph = opSubgraph[opIdx];
        auto outIt = opOutGraph.find(opIdx);
        if (outIt == opOutGraph.end()) continue;
        
        for (int consumerOp : outIt->second) {
            int dstSubgraph = opSubgraph[consumerOp];
            if (srcSubgraph != dstSubgraph) {
                subgraphDag[srcSubgraph].insert(dstSubgraph);
            }
        }
    }
}

void CalcSubgraphLatency(int numOp, int numSubgraph,
                         const std::vector<int>& opLatency,
                         const std::vector<int>& opSubgraph,
                         std::vector<int>& subgraphLatency) {
    subgraphLatency.assign(numSubgraph, 0);
    for (int opIdx = 0; opIdx < numOp; ++opIdx) {
        subgraphLatency[opSubgraph[opIdx]] += opLatency[opIdx];
    }
}

bool CheckDagAfterMerge(const std::set<int>& mergeGroup,
                        const std::unordered_map<int, std::set<int>>& subgraphDag,
                        int numSubgraph) {
    std::unordered_map<int, std::set<int>> newDag;
    std::set<int> mergedSet = mergeGroup;
    int mergedId = *mergeGroup.begin();
    
    for (int sg = 0; sg < numSubgraph; ++sg) {
        if (mergedSet.count(sg)) continue;
        auto it = subgraphDag.find(sg);
        if (it == subgraphDag.end()) continue;
        
        for (int dst : it->second) {
            if (mergedSet.count(dst)) {
                newDag[sg].insert(mergedId);
            } else {
                newDag[sg].insert(dst);
            }
        }
    }
    
    auto mergedIt = subgraphDag.find(mergedId);
    if (mergedIt != subgraphDag.end()) {
        for (int dst : mergedIt->second) {
            if (!mergedSet.count(dst)) {
                newDag[mergedId].insert(dst);
            }
        }
    }
    
    for (int sg : mergedSet) {
        if (sg == mergedId) continue;
        auto it = subgraphDag.find(sg);
        if (it == subgraphDag.end()) continue;
        
        for (int dst : it->second) {
            if (!mergedSet.count(dst)) {
                newDag[mergedId].insert(dst);
            }
        }
    }
    
    for (int sg : mergedSet) {
        for (const auto& pair : subgraphDag) {
            if (mergedSet.count(pair.first)) continue;
            if (pair.second.count(sg)) {
                newDag[pair.first].insert(mergedId);
            }
        }
    }
    
    std::unordered_map<int, int> inDegree;
    for (int sg = 0; sg < numSubgraph; ++sg) {
        if (mergedSet.count(sg) && sg != mergedId) continue;
        inDegree[sg] = 0;
    }
    
    for (const auto& pair : newDag) {
        for (int dst : pair.second) {
            inDegree[dst]++;
        }
    }
    
    std::queue<int> q;
    int visitedCount = 0;
    int totalCount = numSubgraph - static_cast<int>(mergeGroup.size()) + 1;
    
    for (const auto& pair : inDegree) {
        if (pair.second == 0) {
            q.push(pair.first);
        }
    }
    
    while (!q.empty()) {
        int curr = q.front();
        q.pop();
        visitedCount++;
        
        auto it = newDag.find(curr);
        if (it == newDag.end()) continue;
        
        for (int dst : it->second) {
            inDegree[dst]--;
            if (inDegree[dst] == 0) {
                q.push(dst);
            }
        }
    }
    
    return visitedCount == totalCount;
}

void LogMergeGroup(const std::set<int>& mergeGroup) {
    char buffer[512];
    int pos = 0;
    pos += snprintf(buffer + pos, sizeof(buffer) - pos, "Processing merge group: {");
    bool first = true;
    for (int sg : mergeGroup) {
        if (!first) {
            pos += snprintf(buffer + pos, sizeof(buffer) - pos, ", ");
        }
        pos += snprintf(buffer + pos, sizeof(buffer) - pos, "%d", sg);
        first = false;
    }
    snprintf(buffer + pos, sizeof(buffer) - pos, "}");
    APASS_LOG_INFO_F(Elements::Operation, "%s", buffer);
}

void LogMergeDecision(const std::set<int>& mergeGroup,
                      int totalLatency, bool dagOk) {
    LogMergeGroup(mergeGroup);
    APASS_LOG_INFO_F(Elements::Operation, "  Total latency: %d, Limit: %d", totalLatency, MAX_LATENCY);
    
    if (totalLatency > MAX_LATENCY) {
        APASS_LOG_INFO_F(Elements::Operation, "  Decision: SKIP (latency exceeded)");
    } else if (!dagOk) {
        APASS_LOG_INFO_F(Elements::Operation, "  DAG check: FAILED (would create cycle)");
        APASS_LOG_INFO_F(Elements::Operation, "  Decision: SKIP (DAG constraint violated)");
    } else {
        APASS_LOG_INFO_F(Elements::Operation, "  DAG check: PASSED");
        APASS_LOG_INFO_F(Elements::Operation, "  Decision: MERGE (latency OK, DAG OK)");
    }
}

void LogMergeGroups(const std::unordered_map<int, std::set<int>>& mergeGroups) {
    APASS_LOG_INFO_F(Elements::Operation, "Found %zu merge group(s)", mergeGroups.size());
    for (const auto& pair : mergeGroups) {
        char buffer[512];
        int pos = snprintf(buffer, sizeof(buffer), "  Group %d: {", pair.first);
        bool first = true;
        for (int sg : pair.second) {
            if (!first) {
                pos += snprintf(buffer + pos, sizeof(buffer) - pos, ", ");
            }
            pos += snprintf(buffer + pos, sizeof(buffer) - pos, "%d", sg);
            first = false;
        }
        snprintf(buffer + pos, sizeof(buffer) - pos, "}");
        APASS_LOG_INFO_F(Elements::Operation, "%s", buffer);
    }
}

} // anonymous namespace

void MergeSubgraphs(int numOp, int numSubgraph,
                    const std::vector<int>& opLatency,
                    std::vector<int>& opSubgraph,
                    const std::unordered_map<int, std::set<int>>& opOutGraph,
                    const std::unordered_map<int, std::set<int>>& opInGraph) {
    APASS_LOG_INFO_F(Elements::Operation, "=== Subgraph Merge Algorithm Started ===");
    APASS_LOG_INFO_F(Elements::Operation, "numOp: %d, numSubgraph: %d", numOp, numSubgraph);
    
    UnionFind uf(numSubgraph);
    BuildMergeGroups(numOp, opSubgraph, opOutGraph, opInGraph, uf);
    
    std::unordered_map<int, std::set<int>> subgraphDag;
    BuildSubgraphDag(numOp, numSubgraph, opSubgraph, opOutGraph, subgraphDag);
    
    std::vector<int> subgraphLatency;
    CalcSubgraphLatency(numOp, numSubgraph, opLatency, opSubgraph, subgraphLatency);
    
    APASS_LOG_INFO_F(Elements::Operation, "Subgraph latencies:");
    for (int i = 0; i < numSubgraph; ++i) {
        APASS_LOG_INFO_F(Elements::Operation, "  Subgraph %d: latency = %d", i, subgraphLatency[i]);
    }
    
    std::unordered_map<int, std::set<int>> mergeGroups;
    for (int sg = 0; sg < numSubgraph; ++sg) {
        int root = uf.Find(sg);
        mergeGroups[root].insert(sg);
    }
    
    LogMergeGroups(mergeGroups);
    
    std::vector<std::pair<int, int>> sortedSubgraphs;
    for (int sg = 0; sg < numSubgraph; ++sg) {
        sortedSubgraphs.push_back({subgraphLatency[sg], sg});
    }
    std::sort(sortedSubgraphs.begin(), sortedSubgraphs.end());
    
    std::vector<bool> processed(numSubgraph, false);
    int mergeCount = 0;
    
    for (const auto& pair : sortedSubgraphs) {
        int sg = pair.second;
        if (processed[sg]) continue;
        
        int root = uf.Find(sg);
        const std::set<int>& group = mergeGroups[root];
        
        int totalLatency = 0;
        for (int memberSg : group) {
            totalLatency += subgraphLatency[memberSg];
        }
        
        bool dagOk = CheckDagAfterMerge(group, subgraphDag, numSubgraph);
        bool willMerge = (totalLatency <= MAX_LATENCY) && dagOk;
        
        LogMergeDecision(group, totalLatency, dagOk);
        
        if (willMerge) {
            int targetSg = *group.begin();
            for (int memberSg : group) {
                if (memberSg == targetSg) continue;
                for (int opIdx = 0; opIdx < numOp; ++opIdx) {
                    if (opSubgraph[opIdx] == memberSg) {
                        opSubgraph[opIdx] = targetSg;
                    }
                }
            }
            mergeCount++;
            APASS_LOG_INFO_F(Elements::Operation, "  Merged %zu subgraphs into subgraph %d", group.size(), targetSg);
        }
        
        for (int memberSg : group) {
            processed[memberSg] = true;
        }
    }
    
    APASS_LOG_INFO_F(Elements::Operation, "Renumbering subgraphs...");
    std::unordered_map<int, int> oldToNew;
    int newId = 0;
    for (int opIdx = 0; opIdx < numOp; ++opIdx) {
        int oldId = opSubgraph[opIdx];
        if (oldToNew.find(oldId) == oldToNew.end()) {
            oldToNew[oldId] = newId++;
        }
        opSubgraph[opIdx] = oldToNew[oldId];
    }
    
    int finalNumSubgraph = newId;
    APASS_LOG_INFO_F(Elements::Operation, "Final number of subgraphs: %d", finalNumSubgraph);
    APASS_LOG_INFO_F(Elements::Operation, "Merge operations performed: %d", mergeCount);
    
    APASS_LOG_INFO_F(Elements::Operation, "Final opSubgraph mapping:");
    for (int i = 0; i < numOp; ++i) {
        APASS_LOG_INFO_F(Elements::Operation, "  op%d -> subgraph %d", i, opSubgraph[i]);
    }
    
    APASS_LOG_INFO_F(Elements::Operation, "=== Subgraph Merge Algorithm Completed ===");
}