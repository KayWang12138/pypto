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
 * \file orbit_graph_processor.h
 * \brief
 */

#ifndef OSP_ORBIT_GRAPH_PROCESSOR_H
#define OSP_ORBIT_GRAPH_PROCESSOR_H

#include <algorithm>
#include <map>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "passes/algorithms/osp/coarser/coarser_util.h"
#include "passes/algorithms/osp/dag_divider/isomorphism_divider/hash_computer.h"
#include "passes/algorithms/osp/dag_divider/isomorphism_divider/merkle_hash_computer.h"
#include "passes/algorithms/osp/graph_algorithms/directed_graph_path_util.h"
#include "passes/algorithms/osp/graph_algorithms/directed_graph_util.h"
#include "passes/algorithms/osp/graph_algorithms/subgraph_algorithms.h"

#include "interface/utils/common.h"

namespace npu::tile_fwk {
namespace osp {

/**
 * @class OrbitGraphProcessor
 * @brief A simple processor that groups nodes of a DAG based on their Merkle hash.
 *
 * This class uses a MerkleHashComputer to assign a structural hash to each node.
 * It then partitions the DAG by grouping all nodes with the same hash into an "orbit".
 * A coarse graph is constructed where each node represents one such orbit.
 */
template <typename GraphT, typename ConstrGraphT>
class OrbitGraphProcessor {
  public:
    using VertexType = VertexIdxT<GraphT>;

    // Represents a group of isomorphic subgraphs, corresponding to a single node in a coarse graph.
    struct Group {
        // Each vector of vertices represents one of the isomorphic subgraphs in this group.
        std::vector<std::vector<VertexType>> subgraphs_;

        inline size_t size() const { return subgraphs_.size(); }
    };

  private:
    // Results from the first (orbit) coarsening step
    ConstrGraphT coarseGraph_;
    std::vector<VertexType> contractionMap_;

    // Results from the second (custom) coarsening step
    ConstrGraphT finalCoarseGraph_;
    std::vector<VertexType> finalContractionMap_;
    std::vector<Group> finalGroups_;
    size_t currentSymmetry_;

    size_t minSymmetry_ = 2;    // min symmetry threshold
    VWorkwT<ConstrGraphT> workThreshold_ = 0;
    VWorkwT<ConstrGraphT> criticalPathThreshold_ = 0;
    bool mergeDifferentNodeTypes_ = true;
    double lockOrbitRatio_ = 0.5;

    double naturalBreaksCountPercentage_ = 0.2;

    struct PairHasher {
        template <class T1, class T2>
        std::size_t operator()(const std::pair<T1, T2> &p) const {
            auto h1 = std::hash<T1>{}(p.first);
            auto h2 = std::hash<T2>{}(p.second);
            HashCombine(h1, h2);
            return h1;
        }
    };

    std::unordered_set<std::pair<VertexType, VertexType>, PairHasher> nonViableEdgesCache_;
    std::unordered_set<std::pair<VertexType, VertexType>, PairHasher> nonViableCritPathEdgesCache_;

    std::pair<ConstrGraphT, std::vector<VertexType>> SimulateMerge(VertexType u, VertexType v,
                                                                   const ConstrGraphT &currentCoarseGraph) const {
        std::vector<VertexType> tempContractionMap(currentCoarseGraph.NumVertices());
        VertexType newIdx = 0;
        for (VertexType i = 0; i < static_cast<VertexType>(tempContractionMap.size()); ++i) {
            if (i != v) {
                tempContractionMap[i] = newIdx++;
            }
        }
        tempContractionMap[v] = tempContractionMap[u];

        ConstrGraphT tempCoarseGraph;
        coarser_util::ConstructCoarseDag(currentCoarseGraph, tempCoarseGraph, tempContractionMap);

        return {std::move(tempCoarseGraph), std::move(tempContractionMap)};
    }

    void CommitMerge(VertexType u, VertexType v, ConstrGraphT &&nextCoarseGraph,
                     const std::vector<VertexType> &groupRemap,
                     std::vector<std::vector<VertexType>> &&newSubgraphs,
                     ConstrGraphT &currentCoarseGraph,
                     std::vector<Group> &currentGroups) {
        currentCoarseGraph = std::move(nextCoarseGraph);

        // Update caches for new vertex indices
        auto UpdateCache = [&](auto &cache) {
            std::unordered_set<std::pair<VertexType, VertexType>, PairHasher> nextCache;
            for (const auto &[oldU, oldV] : cache) {
                const VertexType newU = groupRemap[oldU];
                const VertexType newV = groupRemap[oldV];
                if (oldU != v && oldV != v && newU != newV) {
                    nextCache.insert({newU, newV});
                }
            }
            cache = std::move(nextCache);
        };
        UpdateCache(nonViableEdgesCache_);
        UpdateCache(nonViableCritPathEdgesCache_);

        // Update groups
        std::vector<Group> nextGroups(currentCoarseGraph.NumVertices());
        for (VertexType i = 0; i < static_cast<VertexType>(currentGroups.size()); ++i) {
            if (i != u && i != v) {
                nextGroups[groupRemap[i]] = std::move(currentGroups[i]);
            }
        }
        nextGroups[groupRemap[u]].subgraphs_ = std::move(newSubgraphs);
        currentGroups = std::move(nextGroups);
    }

    bool ShouldSkipEdge(VertexType u, VertexType v, const ConstrGraphT &currentCoarseGraph,
                       const std::vector<Group> &currentGroups,
                       const std::vector<VertexIdxT<ConstrGraphT>> &vertexPoset,
                       const std::vector<VertexIdxT<ConstrGraphT>> &vertexBotPoset,
                       const VWorkwT<ConstrGraphT> workThreshold) const {
        // Check node type compatibility
        if (not mergeDifferentNodeTypes_) {
            if (currentCoarseGraph.VertexType(u) != currentCoarseGraph.VertexType(v)) {
                return true;
            }
        }

        // Check if edge is in non-viable cache
        if (nonViableEdgesCache_.count({u, v}) || nonViableCritPathEdgesCache_.count({u, v})) {
            return true;
        }

        // Check work thresholds
        const VWorkwT<ConstrGraphT> uWorkWeight = currentCoarseGraph.VertexWorkWeight(u);
        const VWorkwT<ConstrGraphT> vWorkWeight = currentCoarseGraph.VertexWorkWeight(v);
        const VWorkwT<ConstrGraphT> vThreshold
            = workThreshold * static_cast<VWorkwT<ConstrGraphT>>(currentGroups[v].size());
        const VWorkwT<ConstrGraphT> uThreshold
            = workThreshold * static_cast<VWorkwT<ConstrGraphT>>(currentGroups[u].size());

        if (uWorkWeight > uThreshold && vWorkWeight > vThreshold) {
            return true;
        }

        // Check poset constraints
        if ((vertexPoset[u] + 1 != vertexPoset[v]) && (vertexBotPoset[u] != 1 + vertexBotPoset[v])) {
            return true;
        }

        return false;
    }

    bool TryMergeEdge(VertexType u, VertexType v, const GraphT &originalDag, const ConstrGraphT &currentCoarseGraph,
                     const std::vector<Group> &currentGroups, const VWorkwT<ConstrGraphT> pathThreshold,
                     std::vector<std::vector<VertexType>> &outNewSubgraphs, ConstrGraphT &outTempGraph,
                     std::vector<VertexType> &outTempContractionMap) {
        // Check merge structural viability
        const bool mergeIsValid = IsMergeViable(originalDag, currentGroups[u], currentGroups[v], outNewSubgraphs);
        if (!mergeIsValid) {
            nonViableEdgesCache_.insert({u, v});
            return false;
        }

        // Simulate merge and check critical path
        auto [tempCoarseGraph, tempContractionMap] = SimulateMerge(u, v, currentCoarseGraph);

        if (CriticalPathWeight(tempCoarseGraph)
            > (pathThreshold * static_cast<VWorkwT<ConstrGraphT>>(outNewSubgraphs.size())
               + CriticalPathWeight(currentCoarseGraph))) {
            nonViableCritPathEdgesCache_.insert({u, v});
            return false;
        }

        outTempGraph = std::move(tempCoarseGraph);
        outTempContractionMap = std::move(tempContractionMap);
        return true;
    }

    void MergeSmallOrbits(const GraphT &originalDag,
                          ConstrGraphT &currentCoarseGraph,
                          std::vector<Group> &currentGroups,
                          const VWorkwT<ConstrGraphT> workThreshold,
                          const VWorkwT<ConstrGraphT> pathThreshold = 0) {
        bool changed = true;
        while (changed) {
            const std::vector<VertexIdxT<ConstrGraphT>> vertexPoset
                = GetTopNodeDistance<ConstrGraphT, VertexIdxT<ConstrGraphT>>(currentCoarseGraph);
            const std::vector<VertexIdxT<ConstrGraphT>> vertexBotPoset
                = GetBottomNodeDistance<ConstrGraphT, VertexIdxT<ConstrGraphT>>(currentCoarseGraph);

            changed = false;
            for (const auto u : currentCoarseGraph.Vertices()) {
                for (const auto v : currentCoarseGraph.Children(u)) {
                    if (ShouldSkipEdge(u, v, currentCoarseGraph, currentGroups, vertexPoset, vertexBotPoset, workThreshold)) continue;

                    std::vector<std::vector<VertexType>> newSubgraphs;
                    ConstrGraphT tempCoarseGraph;
                    std::vector<VertexType> tempContractionMap;

                    if (!TryMergeEdge(u, v, originalDag, currentCoarseGraph, currentGroups, pathThreshold,
                                      newSubgraphs, tempCoarseGraph, tempContractionMap)) continue;

                    CommitMerge(u, v, std::move(tempCoarseGraph), tempContractionMap, std::move(newSubgraphs),
                                currentCoarseGraph, currentGroups);

                    changed = true;
                    break;
                }
                if (changed) break;                
            }
        }
    }

    bool IsEdgeMergeCandidate(VertexType u, VertexType v,
                              const ConstrGraphT &currentCoarseGraph,
                              const std::vector<VertexIdxT<ConstrGraphT>> &vertexPoset,
                              const std::vector<VertexIdxT<ConstrGraphT>> &vertexBotPoset,
                              const bool mergeDifferentNodeTypes) {
        if (nonViableEdgesCache_.count({u, v}) || nonViableCritPathEdgesCache_.count({u, v})) {
            return false;
        }
        if (not mergeDifferentNodeTypes && currentCoarseGraph.VertexType(u) != currentCoarseGraph.VertexType(v)) {
            return false;
        }
        if ((vertexPoset[u] + 1 != vertexPoset[v]) && (vertexBotPoset[u] != 1 + vertexBotPoset[v])) {
            return false;
        }
        return true;
    }

    bool IsSignificanceMergeBlocked(VertexType u, VertexType v,
                                    const ConstrGraphT &currentCoarseGraph,
                                    const std::vector<Group> &currentGroups,
                                    const std::vector<VWorkwT<GraphT>> &lockThresholdPerType,
                                    const bool mergeDifferentNodeTypes,
                                    std::size_t newSize) {
        VTypeT<GraphT> uType = 0;
        VTypeT<GraphT> vType = 0;
        if (not mergeDifferentNodeTypes) {
            uType = currentCoarseGraph.VertexType(u);
            vType = currentCoarseGraph.VertexType(v);
        }

        const std::size_t uSize = currentGroups[u].size();
        const std::size_t vSize = currentGroups[v].size();
        const bool uSig = (uSize >= minSymmetry_) && (currentCoarseGraph.VertexWorkWeight(u) > lockThresholdPerType[uType]);
        const bool vSig = (vSize >= minSymmetry_) && (currentCoarseGraph.VertexWorkWeight(v) > lockThresholdPerType[vType]);

        return (uSig && vSig && newSize < std::min(uSize, vSize)) ||
               ((uSig ^ vSig) && newSize < (uSig ? uSize : vSize));
    }

    void ContractEdgesAdpativeSym(const GraphT &originalDag,
                                  ConstrGraphT &currentCoarseGraph,
                                  std::vector<Group> &currentGroups,
                                  const bool mergeDifferentNodeTypes,
                                  const bool mergeBelowThreshold,
                                  const std::vector<VWorkwT<GraphT>> &lockThresholdPerType,
                                  const VWorkwT<ConstrGraphT> pathThreshold = 0) {
        bool changed = true;
        while (changed) {
            const std::vector<VertexIdxT<ConstrGraphT>> vertexPoset
                = GetTopNodeDistance<ConstrGraphT, VertexIdxT<ConstrGraphT>>(currentCoarseGraph);
            const std::vector<VertexIdxT<ConstrGraphT>> vertexBotPoset
                = GetBottomNodeDistance<ConstrGraphT, VertexIdxT<ConstrGraphT>>(currentCoarseGraph);

            changed = false;
            for (const auto &edge : Edges(currentCoarseGraph)) {
                VertexType u = Source(edge, currentCoarseGraph);
                VertexType v = Target(edge, currentCoarseGraph);

                if (!IsEdgeMergeCandidate(u, v, currentCoarseGraph, vertexPoset, vertexBotPoset, mergeDifferentNodeTypes)) {
                    continue;
                }

                std::vector<std::vector<VertexType>> newSubgraphs;
                const bool mergeIsValid = IsMergeViable(originalDag, currentGroups[u], currentGroups[v], newSubgraphs);
                const std::size_t newSize = newSubgraphs.size();

                if (!mergeIsValid) {
                    nonViableEdgesCache_.insert({u, v});
                    continue;
                }

                const bool mergeViable = (newSize >= currentSymmetry_);
                const bool bothBelowMinimalThreshold = mergeBelowThreshold
                    && (currentGroups[u].size() < minSymmetry_) && (currentGroups[v].size() < minSymmetry_);

                if (!mergeViable && !bothBelowMinimalThreshold) {
                    nonViableEdgesCache_.insert({u, v});
                    continue;
                }

                if (IsSignificanceMergeBlocked(u, v, currentCoarseGraph, currentGroups,
                                               lockThresholdPerType, mergeDifferentNodeTypes, newSize)) {
                    nonViableEdgesCache_.insert({u, v});
                    continue;
                }

                auto [tempCoarseGraph, tempContractionMap] = SimulateMerge(u, v, currentCoarseGraph);

                if (CriticalPathWeight(tempCoarseGraph) > (pathThreshold * static_cast<VWorkwT<ConstrGraphT>>(newSubgraphs.size())
                                                           + CriticalPathWeight(currentCoarseGraph))) {
                    nonViableCritPathEdgesCache_.insert({u, v});
                    continue;
                }

                CommitMerge(u, v, std::move(tempCoarseGraph), tempContractionMap,
                            std::move(newSubgraphs), currentCoarseGraph, currentGroups);

                changed = true;
                break;
            }
        }
    }

  public:
    explicit OrbitGraphProcessor() {}
    void SetMergeDifferentNodeTypes(bool flag) { mergeDifferentNodeTypes_ = flag; }
    void SetWorkThreshold(VWorkwT<ConstrGraphT> workThreshold) { workThreshold_ = workThreshold; }
    void SetCriticalPathThreshold(VWorkwT<ConstrGraphT> criticalPathThreshold) { criticalPathThreshold_ = criticalPathThreshold; }
    void SetLockRatio(double lockRatio) { lockOrbitRatio_ = lockRatio; }
    void SetNaturalBreaksCountPercentage(double percentage) { naturalBreaksCountPercentage_ = percentage; }

    /**
     * @brief Discovers isomorphic groups (orbits) in the DAG and constructs an initial coarse graph.
     *
     * Uses a HashComputer to identify symmetric nodes (orbits) and groups them.
     * Then performs coarsening (either adaptive or static) to merge these groups further.
     *
     * @param dag The input computational DAG.
     * @param hasher The hash computer providing orbit information.
     */
    void DiscoverIsomorphicGroups(const GraphT &dag, const HashComputer<VertexType> &hasher) {
        coarseGraph_ = ConstrGraphT();
        contractionMap_.clear();
        finalCoarseGraph_ = ConstrGraphT();
        finalContractionMap_.clear();
        finalGroups_.clear();
        nonViableEdgesCache_.clear();
        nonViableCritPathEdgesCache_.clear();

        if (dag.NumVertices() == 0) return;
        const auto &orbits = hasher.GetOrbits();

        contractionMap_.assign(dag.NumVertices(), 0);
        VertexType coarseNodeIdx = 0;

        for (const auto &hashVerticesPair : orbits) {
            const auto &vertices = hashVerticesPair.second;
            for (const auto v : vertices) {
                contractionMap_[v] = coarseNodeIdx;
            }
            coarseNodeIdx++;
        }

        std::vector<VWorkwT<GraphT>> workPerVertexType;
        workPerVertexType.resize(mergeDifferentNodeTypes_ ? 1U : dag.NumVertexTypes(), 0);

        std::map<size_t, size_t> orbitSizeCounts;
        for (const auto &orbit : orbits) {
            const auto &vertices = orbit.second;
            const size_t orbitSize = vertices.size();
            if (orbitSize == 1U) continue;

            orbitSizeCounts[orbitSize]++;

            VWorkwT<GraphT> orbitWork = 0;
            for (const auto v : vertices) {
                orbitWork += dag.VertexWorkWeight(v);
            }

            if (not mergeDifferentNodeTypes_) {
                workPerVertexType[dag.VertexType(vertices[0])] += orbitWork;
            } else {
                workPerVertexType[0] += orbitWork;
            }
        }

        std::vector<VWorkwT<GraphT>> lockThresholdPerType(workPerVertexType.size());
        for (size_t i = 0; i < workPerVertexType.size(); ++i) {
            lockThresholdPerType[i] = static_cast<VWorkwT<GraphT>>(lockOrbitRatio_ * workPerVertexType[i]);
        }

        std::vector<size_t> symmetryLevelsToTest = ComputeSymmetryLevels(orbitSizeCounts);
        coarser_util::ConstructCoarseDag(dag, coarseGraph_, contractionMap_);
        PerformCoarseningAdaptiveSymmetry(dag, coarseGraph_, lockThresholdPerType, symmetryLevelsToTest);
    }

  private:
    std::vector<size_t> FindSignificantSymmetryLevels(const std::map<size_t, size_t> &orbitSizeCounts,
                                                      size_t countThreshold) {
        std::vector<size_t> sortedSizes;
        sortedSizes.reserve(orbitSizeCounts.size());
        for (const auto &pair: orbitSizeCounts) {
            sortedSizes.push_back(pair.first);
        }
        std::sort(sortedSizes.rbegin(), sortedSizes.rend());

        std::vector<size_t> levels;
        for (const size_t currentSize : sortedSizes) {
            if (currentSize >= minSymmetry_ && orbitSizeCounts.at(currentSize) >= countThreshold) {
                levels.push_back(currentSize);
            }
        }
        return levels;
    }

    size_t FindFallbackSymmetryLevel(const std::map<size_t, size_t> &orbitSizeCounts) {
        size_t maxCount = 0;
        size_t sizeWithMaxCount = 0;
        for (const auto &[size, count] : orbitSizeCounts) {
            if (count > maxCount) {
                maxCount = count;
                sizeWithMaxCount = size;
            }
        }
        return sizeWithMaxCount;
    }

    std::vector<size_t> ComputeSymmetryLevels(const std::map<size_t, size_t> orbitSizeCounts) {
        minSymmetry_ = 2;

        size_t totalOrbitGroups = 0;
        for (const auto &pair: orbitSizeCounts) {
            totalOrbitGroups += pair.second;
        }
        size_t countThreshold = static_cast<size_t>(static_cast<double>(totalOrbitGroups) * naturalBreaksCountPercentage_);
        if (countThreshold == 0 && totalOrbitGroups > 0) {
            countThreshold = 1;
        }

        std::vector<size_t> symmetryLevelsToTest = FindSignificantSymmetryLevels(orbitSizeCounts, countThreshold);

        if (symmetryLevelsToTest.empty()) {
            const size_t fallback = FindFallbackSymmetryLevel(orbitSizeCounts);
            if (fallback > 0) {
                symmetryLevelsToTest.push_back(fallback);
            }
        }
        if (symmetryLevelsToTest.empty()) {
            symmetryLevelsToTest.push_back(2);
        }

        minSymmetry_ = symmetryLevelsToTest.back();

        std::sort(symmetryLevelsToTest.rbegin(), symmetryLevelsToTest.rend());
        auto last = std::unique(symmetryLevelsToTest.begin(), symmetryLevelsToTest.end());
        symmetryLevelsToTest.erase(last, symmetryLevelsToTest.end());

        return symmetryLevelsToTest;
    }

    void PerformCoarseningAdaptiveSymmetry(const GraphT &originalDag,
                                           const ConstrGraphT &initialCoarseGraph,
                                           const std::vector<VWorkwT<GraphT>> &lockThresholdPerType,
                                           const std::vector<size_t> &symmetryLevelsToTest) {
        finalCoarseGraph_ = ConstrGraphT();
        finalContractionMap_.clear();

        if (initialCoarseGraph.NumVertices() == 0) return;

        ConstrGraphT currentCoarseGraph = initialCoarseGraph;
        std::vector<Group> currentGroups(initialCoarseGraph.NumVertices());
        std::vector<VertexType> currentContractionMap = contractionMap_;

        for (VertexType i = 0; i < originalDag.NumVertices(); ++i) {
            const VertexType coarseNode = contractionMap_[i];
            currentGroups[coarseNode].subgraphs_.push_back({i});
        }

        for (const auto sym : symmetryLevelsToTest) {
            currentSymmetry_ = sym;
            const bool isLastLoop = (sym == symmetryLevelsToTest.back());

            nonViableEdgesCache_.clear();
            ContractEdgesAdpativeSym(originalDag, currentCoarseGraph, currentGroups, false, isLastLoop, lockThresholdPerType);

            if (mergeDifferentNodeTypes_) {
                ContractEdgesAdpativeSym(originalDag, currentCoarseGraph, currentGroups, mergeDifferentNodeTypes_, isLastLoop, lockThresholdPerType);
            }

            nonViableCritPathEdgesCache_.clear();
            ContractEdgesAdpativeSym(originalDag, currentCoarseGraph, currentGroups, mergeDifferentNodeTypes_, isLastLoop, lockThresholdPerType, criticalPathThreshold_);
        }

        nonViableEdgesCache_.clear();
        MergeSmallOrbits(originalDag, currentCoarseGraph, currentGroups, workThreshold_);

        // Rebuild contraction map from currentGroups
        currentContractionMap.assign(originalDag.NumVertices(), 0);
        for (VertexType coarseIdx = 0; coarseIdx < static_cast<VertexType>(currentGroups.size()); ++coarseIdx) {
            for (const auto &subgraph : currentGroups[coarseIdx].subgraphs_) {
                for (const auto v : subgraph) {
                    currentContractionMap[v] = coarseIdx;
                }
            }
        }

        finalCoarseGraph_ = std::move(currentCoarseGraph);
        finalContractionMap_ = std::move(currentContractionMap);
        finalGroups_ = std::move(currentGroups);
    }

    bool IsMergeViable(const GraphT &originalDag,
                       const Group &groupU,
                       const Group &groupV,
                       std::vector<std::vector<VertexType>> &outNewSubgraphs) const {
        std::vector<VertexType> allNodes;
        const size_t uNodes = groupU.subgraphs_.empty() ? 0 : groupU.subgraphs_.size() * groupU.subgraphs_[0].size();
        const size_t vNodes = groupV.subgraphs_.empty() ? 0 : groupV.subgraphs_.size() * groupV.subgraphs_[0].size();
        allNodes.reserve(uNodes + vNodes);
        for (const auto &sg : groupU.subgraphs_) {
            allNodes.insert(allNodes.end(), sg.begin(), sg.end());
        }
        for (const auto &sg : groupV.subgraphs_) {
            allNodes.insert(allNodes.end(), sg.begin(), sg.end());
        }

        std::sort(allNodes.begin(), allNodes.end());
        ConstrGraphT inducedSubgraph;

        auto map = CreateInducedSubgraphMap(originalDag, inducedSubgraph, allNodes);
        std::vector<VertexType> components;    // local -> component_id
        size_t numComponents = ComputeWeaklyConnectedComponents(inducedSubgraph, components);
        outNewSubgraphs.assign(numComponents, std::vector<VertexType>());
        if (allNodes.empty()) return true;

        for (const auto &node : allNodes) {
            outNewSubgraphs[components[map[node]]].push_back(node);
        }

        if (numComponents > 1) {
            const size_t firstSgSize = outNewSubgraphs[0].size();
            ConstrGraphT repSg;
            CreateInducedSubgraphMap(originalDag, repSg, outNewSubgraphs[0]);

            for (size_t i = 1; i < numComponents; ++i) {
                if (outNewSubgraphs[i].size() != firstSgSize) return false;

                ConstrGraphT currentSg;
                CreateInducedSubgraphMap(originalDag, currentSg, outNewSubgraphs[i]);
                if (!AreIsomorphicByMerkleHash(repSg, currentSg)) return false;
            }
        }
        return true;
    }

  public:
    const ConstrGraphT &GetCoarseGraph() const { return coarseGraph_; }

    const std::vector<VertexType> &GetContractionMap() const { return contractionMap_; }

    const ConstrGraphT &GetFinalCoarseGraph() const { return finalCoarseGraph_; }

    const std::vector<VertexType> &GetFinalContractionMap() const { return finalContractionMap_; }

    const std::vector<Group> &GetFinalGroups() const { return finalGroups_; }
};

}    // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_ORBIT_GRAPH_PROCESSOR_HPP
