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
 * \file IsomorphicSubgraphScheduler.hpp
 * \brief
 */

#ifndef OSP_ISOMORPHIC_SUBGRAPH_SCHEDULER_HPP
#define OSP_ISOMORPHIC_SUBGRAPH_SCHEDULER_HPP

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>

#include "EftSubgraphScheduler.hpp"
#include "HashComputer.hpp"
#include "MerkleHashComputer.hpp"
#include "OrbitGraphProcessor.hpp"
#include "TrimmedGroupScheduler.hpp"
#include "passes/algorithms/osp/bsp/scheduler/Scheduler.hpp"
#include "passes/algorithms/osp/graph_algorithms/subgraph_algorithms.hpp"

namespace npu::tile_fwk {
namespace osp {

/**
 * @brief A scheduler that leverages isomorphic subgraphs to partition a DAG.
 *
 * @class IsomorphicSubgraphScheduler
 *
 * This scheduler first identifies isomorphic subgraphs within the input DAG using a hash-based approach.
 * It then groups these isomorphic subgraphs into "orbits". Each orbit is treated as a single node in a
 * coarser graph. The scheduler then uses an ETF-like approach to schedule these coarse nodes (orbits)
 * onto available processors. Finally, the schedule for each orbit is "unrolled" back to the original
 * DAG, assigning a partition ID to each original vertex.
 *
 * The scheduler supports trimming of isomorphic groups to better fit processor counts, and can
 * dynamically switch between a standard BSP scheduler and a specialized TrimmedGroupScheduler
 * for these trimmed groups.
 *
 * @tparam GraphT The type of the input computational DAG.
 * @tparam ConstrGraphT The type of the constructable computational DAG used for internal representations.
 */
template <typename GraphT, typename ConstrGraphT>
class IsomorphicSubgraphScheduler {
    static_assert(isComputationalDagV<GraphT>, "Graph must be a computational DAG");
    static_assert(isComputationalDagV<ConstrGraphT>, "ConstrGraphT must be a computational DAG");
    static_assert(isConstructableCdagV<ConstrGraphT>, "ConstrGraphT must satisfy the constructable_cdag_vertex concept");
    static_assert(std::is_same_v<VertexIdxT<GraphT>, VertexIdxT<ConstrGraphT>>,
                  "GraphT and ConstrGraphT must have the same VertexIdx types");

  private:
    const HashComputer<VertexIdxT<GraphT>> *hashComputer_;

    Scheduler<ConstrGraphT> *bspScheduler_;
    bool useMaxGroupSize_ = false;
    unsigned maxGroupSize_ = 0;
    VWorkwT<ConstrGraphT> workThreshold_ = 10;
    VWorkwT<ConstrGraphT> criticalPathThreshold_ = 10;
    double orbitLockRatio_ = 0.4;
    double naturalBreaksCountPercentage_ = 0.1;
    bool mergeDifferentNodeTypes_ = true;
    bool allowUseTrimmedScheduler_ = true;
    bool useMaxBsp_ = false;


  public:
    /**
     * @brief Constructs the scheduler with a reference to a base BSP scheduler.
     * @param bspScheduler The underlying scheduler to use for scheduling individual subgraphs.
     */
    IsomorphicSubgraphScheduler(Scheduler<ConstrGraphT> &bspScheduler)
        : hashComputer_(nullptr), bspScheduler_(&bspScheduler) {}

    /**
     * @brief Constructs the scheduler with a base scheduler and an existing hash computer.
     * @param bspScheduler The underlying scheduler.
     * @param hashComputer The pre-computed hash computer for the graph.
     */
    IsomorphicSubgraphScheduler(Scheduler<ConstrGraphT> &bspScheduler, const HashComputer<VertexIdxT<GraphT>> &hashComputer)
        : hashComputer_(&hashComputer), bspScheduler_(&bspScheduler) {}

    virtual ~IsomorphicSubgraphScheduler() {}

    /**
     * @brief Sets whether to merge nodes of different types during coarsening.
     * @param flag True to allow merging different types, false otherwise.
     */
    void SetMergeDifferentTypes(bool flag) { mergeDifferentNodeTypes_ = flag; }

    /**
     * @brief Sets the work weight threshold for merging orbits.
     * @param workThreshold The threshold value.
     */
    void SetWorkThreshold(VWorkwT<ConstrGraphT> workThreshold) { workThreshold_ = workThreshold; }

    /**
     * @brief Sets the critical path threshold for merging orbits.
     * @param criticalPathThreshold The threshold value.
     */
    void SetCriticalPathThreshold(VWorkwT<ConstrGraphT> criticalPathThreshold) { criticalPathThreshold_ = criticalPathThreshold; }

    /**
     * @brief Sets the ratio of work-weight that locks an orbit from being merged.
     * @param orbitLockRatio The ratio (0.0 to 1.0).
     */
    void SetOrbitLockRatio(double orbitLockRatio) { orbitLockRatio_ = orbitLockRatio; }

    /**
     * @brief Sets the percentage count for the natural breaks heuristic.
     * @param naturalBreaksCountPercentage The percentage (0.0 to 1.0).
     */
    void SetNaturalBreaksCountPercentage(double naturalBreaksCountPercentage) {
        naturalBreaksCountPercentage_ = naturalBreaksCountPercentage;
    }

    /**
     * @brief Sets whether to allow using the specialized trimmed scheduler.
     * @param flag True to allow, false otherwise.
     */
    void SetAllowTrimmedScheduler(bool flag) { allowUseTrimmedScheduler_ = flag; }

    /**
     * @brief Disables the use of a fixed maximum group size for trimming.
     */
    void DisableUseMaxGroupSize() { useMaxGroupSize_ = false; }

    /**
     * @brief Sets whether to use the MaxBSP strategy for the representative subgraph.
     * @param flag True to use MaxBSP.
     */
    void SetUseMaxBsp(bool flag) { useMaxBsp_ = flag; }

    /**
     * @brief Enables the use of a fixed maximum group size for trimming.
     * @param maxGroupSize The maximum group size.
     */
    void EnableUseMaxGroupSize(const unsigned maxGroupSize) {
        useMaxGroupSize_ = true;
        maxGroupSize_ = maxGroupSize;
    }

    /**
     * @brief Enables the adaptive symmetry threshold heuristic.
     */


    /**
     * @brief Sets a static symmetry level, disabling adaptive threshold logic.
     * @param staticSymmetryLevel The static symmetry level to use.
     */


    /**
     * @brief Computes the partition of the graph.
     *
     * This is the main entry point. It discovers isomorphic groups, potentially trims them,
     * schedules the coarse groups, and then expands the schedule to the original graph elements.
     *
     * @param instance The BSP instance containing the graph and architecture.
     * @return A vector mapping each vertex index to a processor/partition ID.
     */
    std::vector<VertexIdxT<GraphT>> ComputePartition(const BspInstance<GraphT> &instance) {
        OrbitGraphProcessor<GraphT, ConstrGraphT> orbitProcessor;
        orbitProcessor.SetWorkThreshold(workThreshold_);
        orbitProcessor.SetMergeDifferentNodeTypes(mergeDifferentNodeTypes_);
        orbitProcessor.SetCriticalPathThreshold(criticalPathThreshold_);
        orbitProcessor.SetLockRatio(orbitLockRatio_);
        orbitProcessor.SetNaturalBreaksCountPercentage(naturalBreaksCountPercentage_);


        std::unique_ptr<HashComputer<VertexIdxT<GraphT>>> localHasher;
        if (!hashComputer_) {
            localHasher = std::make_unique<MerkleHashComputer<GraphT, BwdMerkleNodeHashFunc<GraphT>, true>>(
                instance.GetComputationalDag(), instance.GetComputationalDag());
            hashComputer_ = localHasher.get();
        }

        orbitProcessor.DiscoverIsomorphicGroups(instance.GetComputationalDag(), *hashComputer_);

        auto isomorphicGroups = orbitProcessor.GetFinalGroups();

        std::vector<bool> wasTrimmed(isomorphicGroups.size(), false);
        TrimSubgraphGroups(isomorphicGroups, instance, wasTrimmed);    // Apply trimming and record which groups were affected

        auto input = PrepareSubgraphSchedulingInput(instance, isomorphicGroups, wasTrimmed);

        EftSubgraphScheduler<ConstrGraphT> etfScheduler;
        SubgraphSchedule subgraphSchedule
            = etfScheduler.Run(input.instance_, input.multiplicities_, input.requiredProcTypes_, input.maxNumProcessors_);
        subgraphSchedule.wasTrimmed_ = std::move(wasTrimmed);    // Pass through trimming info

        std::vector<VertexIdxT<GraphT>> partition(instance.NumberOfVertices(), 0);
        ScheduleIsomorphicGroup(instance, isomorphicGroups, subgraphSchedule, partition);

        return partition;
    }

  protected:
    template <typename GT, typename CGT>
    struct SubgraphSchedulerInput {
        BspInstance<CGT> instance_;
        std::vector<unsigned> multiplicities_;
        std::vector<unsigned> maxNumProcessors_;
        std::vector<std::vector<VWorkwT<GT>>> requiredProcTypes_;
    };

    /**
     * @brief Trims isomorphic subgraph groups to better fit processor availability.
     *
     * Splits large groups into smaller chunks if their size shares a common divisor with
     * the available processor count (or max group size), effectively increasing the number
     * of schedulable tasks ("trimming").
     *
     * @param isomorphicGroups The groups to potentially trim.
     * @param instance The BSP instance.
     * @param wasTrimmed Output vector indicating which groups were trimmed.
     */
    void TrimSubgraphGroups(std::vector<typename OrbitGraphProcessor<GraphT, ConstrGraphT>::Group> &isomorphicGroups,
                            const BspInstance<GraphT> &instance,
                            std::vector<bool> &wasTrimmed) {
        for (size_t groupIdx = 0; groupIdx < isomorphicGroups.size(); ++groupIdx) {
            auto &group = isomorphicGroups[groupIdx];
            const unsigned groupSize = static_cast<unsigned>(group.size());
            if (groupSize <= 1) {
                continue;
            }

            unsigned effectiveMinProcTypeCount = 0;

            if (useMaxGroupSize_) {
                effectiveMinProcTypeCount = maxGroupSize_;
            } else {
                // Determine if the group consists of a single node type
                bool isSingleTypeGroup = true;
                VTypeT<GraphT> commonNodeType = 0;

                if constexpr (hasTypedVerticesV<GraphT>) {
                    if (!group.subgraphs_.empty() && !group.subgraphs_[0].empty()) {
                        commonNodeType = instance.GetComputationalDag().VertexType(group.subgraphs_[0][0]);
                        const auto &repSubgraph = group.subgraphs_[0];
                        for (const auto &vertex : repSubgraph) {
                            if (instance.GetComputationalDag().VertexType(vertex) != commonNodeType) {
                                isSingleTypeGroup = false;
                                break;
                            }
                        }
                    } else {
                        isSingleTypeGroup = false;
                    }
                } else {
                    isSingleTypeGroup = false;
                }

                if (isSingleTypeGroup) {
                    // Dynamically determine min_proc_type_count based on compatible processors for this type
                    unsigned minCompatibleProcessors = std::numeric_limits<unsigned>::max();
                    const auto &procTypeCounts = instance.GetArchitecture().GetProcessorTypeCount();

                    bool foundCompatibleProcessor = false;
                    for (unsigned procTypeIdx = 0; procTypeIdx < procTypeCounts.size(); ++procTypeIdx) {
                        if (instance.IsCompatibleType(commonNodeType, procTypeIdx)) {
                            minCompatibleProcessors = std::min(minCompatibleProcessors, procTypeCounts[procTypeIdx]);
                            foundCompatibleProcessor = true;
                        }
                    }
                    if (foundCompatibleProcessor) {
                        effectiveMinProcTypeCount = minCompatibleProcessors;
                    } else {
                        effectiveMinProcTypeCount = 1;
                    }
                } else {
                    // Fallback to a default min_proc_type_count if not a single-type group or no typed vertices.
                    const auto &typeCount = instance.GetArchitecture().GetProcessorTypeCount();
                    if (typeCount.empty()) {
                        effectiveMinProcTypeCount = 0;
                    }
                    effectiveMinProcTypeCount = *std::min_element(typeCount.begin(), typeCount.end());
                }
            }

            // Ensure effective_min_proc_type_count is at least 1 for valid GCD calculation.
            if (effectiveMinProcTypeCount == 0) {
                effectiveMinProcTypeCount = 1;
            }

            // If effective_min_proc_type_count is 1, no trimming is needed as gcd(X, 1) = 1.
            if (effectiveMinProcTypeCount <= 1) {
                continue;
            }

            unsigned gcd = std::gcd(groupSize, effectiveMinProcTypeCount);

            if (gcd < groupSize) {

                if (allowUseTrimmedScheduler_) {
                    gcd = 1;
                }

                wasTrimmed[groupIdx] = true;
                const unsigned mergeSize = groupSize / gcd;
                std::vector<std::vector<VertexIdxT<GraphT>>> newSubgraphs;
                newSubgraphs.reserve(gcd);

                size_t originalSgCursor = 0;

                for (unsigned j = 0; j < gcd; ++j) {
                    std::vector<VertexIdxT<GraphT>> mergedSgVertices;
                    // Estimate capacity for efficiency. Assuming subgraphs have similar sizes.
                    if (!group.subgraphs_.empty()) {
                        mergedSgVertices.reserve(group.subgraphs_[0].size() * mergeSize);
                    }

                    for (unsigned k = 0; k < mergeSize; ++k) {
                        const auto &sgToMergeVertices = group.subgraphs_[originalSgCursor];
                        originalSgCursor++;
                        mergedSgVertices.insert(mergedSgVertices.end(), sgToMergeVertices.begin(), sgToMergeVertices.end());
                    }
                    newSubgraphs.push_back(std::move(mergedSgVertices));
                }
                group.subgraphs_ = std::move(newSubgraphs);
            } else {
                wasTrimmed[groupIdx] = false;
            }
        }
    }

    /**
     * @brief Prepares the input for the coarse-level ETF scheduler.
     *
     * Constructs a coarse graph where each node represents an isomorphic group (or a trimmed chunk).
     * Calculates the aggregated work and required processor types for each coarse node.
     *
     * @param originalInstance The original BSP instance.
     * @param isomorphicGroups The groups of isomorphic subgraphs.
     * @param wasTrimmed Indicator if a group was trimmed.
     * @return The input structure for the EftSubgraphScheduler.
     */
    SubgraphSchedulerInput<GraphT, ConstrGraphT> PrepareSubgraphSchedulingInput(
        const BspInstance<GraphT> &originalInstance,
        const std::vector<typename OrbitGraphProcessor<GraphT, ConstrGraphT>::Group> &isomorphicGroups,
        const std::vector<bool> &wasTrimmed) {
        SubgraphSchedulerInput<GraphT, ConstrGraphT> result;
        result.instance_.GetArchitecture() = originalInstance.GetArchitecture();
        const unsigned numProcTypes = originalInstance.GetArchitecture().GetNumberOfProcessorTypes();

        result.multiplicities_.resize(isomorphicGroups.size());
        result.maxNumProcessors_.resize(isomorphicGroups.size());
        result.requiredProcTypes_.resize(isomorphicGroups.size());
        std::vector<VertexIdxT<ConstrGraphT>> contractionMap(originalInstance.NumberOfVertices());

        size_t coarseNodeIdx = 0;
        for (const auto &group : isomorphicGroups) {
            result.maxNumProcessors_[coarseNodeIdx] = static_cast<unsigned>(group.size() * group.subgraphs_[0].size());
            result.multiplicities_[coarseNodeIdx]
                = (wasTrimmed[coarseNodeIdx] && allowUseTrimmedScheduler_) ? 1 : static_cast<unsigned>(group.subgraphs_.size());
            result.requiredProcTypes_[coarseNodeIdx].assign(numProcTypes, 0);

            for (const auto &subgraph : group.subgraphs_) {
                for (const auto &vertex : subgraph) {
                    contractionMap[vertex] = static_cast<VertexIdxT<ConstrGraphT>>(coarseNodeIdx);
                    const auto vertexWork = originalInstance.GetComputationalDag().VertexWorkWeight(vertex);
                    const auto vertexType = originalInstance.GetComputationalDag().VertexType(vertex);
                    for (unsigned j = 0; j < numProcTypes; ++j) {
                        if (originalInstance.IsCompatibleType(vertexType, j)) {
                            result.requiredProcTypes_[coarseNodeIdx][j] += vertexWork;
                        }
                    }
                }
            }

            ++coarseNodeIdx;
        }
        coarser_util::ConstructCoarseDag(
            originalInstance.GetComputationalDag(), result.instance_.GetComputationalDag(), contractionMap);
        return result;
    }

    /**
     * @brief Schedules internal nodes of an isomorphic group by replicating the representative's schedule.
     *
     * Solves the scheduling problem for one "representative" subgraph using the base scheduler
     * (e.g., standard BSP). Then, maps this schedule to all other subgraphs in the same group
     * using isomorphism mapping.
     *
     * @param instance The BSP instance.
     * @param isomorphicGroups The vector of isomorphic groups.
     * @param subSched The coarse-level schedule.
     * @param partition Output partition vector to be filled.
     */
    void ScheduleIsomorphicGroup(const BspInstance<GraphT> &instance,
                                 const std::vector<typename OrbitGraphProcessor<GraphT, ConstrGraphT>::Group> &isomorphicGroups,
                                 const SubgraphSchedule &subSched,
                                 std::vector<VertexIdxT<GraphT>> &partition) {
        VertexIdxT<GraphT> currentPartitionIdx = 0;

        for (size_t groupIdx = 0; groupIdx < isomorphicGroups.size(); ++groupIdx) {
            const auto &group = isomorphicGroups[groupIdx];
            if (group.subgraphs_.empty()) {
                continue;
            }

            // Schedule the Representative Subgraph to get a BSP schedule pattern ---
            auto repSubgraphVertices = group.subgraphs_[0];

            BspInstance<ConstrGraphT> representativeInstance;
            auto repGlobalToLocalMap = CreateInducedSubgraphMap(
                instance.GetComputationalDag(), representativeInstance.GetComputationalDag(), repSubgraphVertices);

            representativeInstance.GetArchitecture() = instance.GetArchitecture();
            const auto &procsForGroup = subSched.nodeAssignedWorkerPerType_[groupIdx];
            std::vector<VMemwT<ConstrGraphT>> memWeights(procsForGroup.size(), 0);
            for (unsigned procType = 0; procType < procsForGroup.size(); ++procType) {
                memWeights[procType]
                    = static_cast<VMemwT<ConstrGraphT>>(instance.GetArchitecture().MaxMemoryBoundProcType(procType));
            }
            representativeInstance.GetArchitecture().SetProcessorsConsequTypes(procsForGroup, memWeights);
            representativeInstance.SetNodeProcessorCompatibility(instance.GetProcessorCompatibilityMatrix());

            // --- Decide which scheduler to use ---
            unsigned minNonZeroProcs = std::numeric_limits<unsigned>::max();
            for (const auto &procCount : procsForGroup) {
                if (procCount > 0) {
                    minNonZeroProcs = std::min(minNonZeroProcs, procCount);
                }
            }

            bool useTrimmedScheduler = subSched.wasTrimmed_[groupIdx] && minNonZeroProcs > 1 && allowUseTrimmedScheduler_;

            Scheduler<ConstrGraphT> *schedulerForGroupPtr;
            std::unique_ptr<Scheduler<ConstrGraphT>> trimmedSchedulerOwner;
            if (useTrimmedScheduler) {
                trimmedSchedulerOwner = std::make_unique<TrimmedGroupScheduler<ConstrGraphT>>(*bspScheduler_, minNonZeroProcs);
                schedulerForGroupPtr = trimmedSchedulerOwner.get();
            } else {
                schedulerForGroupPtr = bspScheduler_;
            }

            // --- Schedule the representative to get the pattern ---
            BspSchedule<ConstrGraphT> bspSchedule(representativeInstance);
            schedulerForGroupPtr->ComputeSchedule(bspSchedule);
            const bool maxBsp = useMaxBsp_ && (representativeInstance.GetComputationalDag().NumEdges() == 0)
                                && (representativeInstance.GetComputationalDag().VertexType(0) == 0);

            // Build data structures for applying the pattern ---
            // Map (superstep, processor) -> relative partition ID
            std::map<std::pair<unsigned, unsigned>, VertexIdxT<GraphT>> spProcToRelativePartition;
            VertexIdxT<GraphT> numPartitionsPerSubgraph = 0;
            for (VertexIdxT<GraphT> j = 0; j < static_cast<VertexIdxT<GraphT>>(repSubgraphVertices.size()); ++j) {
                auto spPair = std::make_pair(bspSchedule.AssignedSuperstep(j), bspSchedule.AssignedProcessor(j));

                if (maxBsp) {
                    spPair = std::make_pair(j, 0);
                }

                if (spProcToRelativePartition.find(spPair) == spProcToRelativePartition.end()) {
                    spProcToRelativePartition[spPair] = numPartitionsPerSubgraph++;
                }
            }

            // Pre-compute hashes for the representative to use for mapping
            MerkleHashComputer<ConstrGraphT> repHasher(representativeInstance.GetComputationalDag());

            // Replicate the schedule pattern for ALL subgraphs in the group ---
            for (VertexIdxT<GraphT> i = 0; i < static_cast<VertexIdxT<GraphT>>(group.subgraphs_.size()); ++i) {
                auto currentSubgraphVertices = group.subgraphs_[i];

                // Map from a vertex in the current subgraph to its corresponding local index (0, 1, ...) in the representative's schedule
                std::unordered_map<VertexIdxT<GraphT>, VertexIdxT<ConstrGraphT>> currentVertexToRepLocalIdx;

                if (i == 0) {    // The first subgraph is the representative itself
                    currentVertexToRepLocalIdx = std::move(repGlobalToLocalMap);
                } else {    // For other subgraphs, build the isomorphic mapping
                    ConstrGraphT currentSubgraphGraph;
                    auto currentGlobalToLocalMap = CreateInducedSubgraphMap(
                        instance.GetComputationalDag(), currentSubgraphGraph, currentSubgraphVertices);

                    std::vector<VertexIdxT<GraphT>> currentLocalToGlobalMap(currentGlobalToLocalMap.size());
                    for (const auto &[globalVertex, localVertex] : currentGlobalToLocalMap) {
                        currentLocalToGlobalMap[localVertex] = globalVertex;
                    }

                    MerkleHashComputer<ConstrGraphT> currentHasher(currentSubgraphGraph);

                    for (const auto &[hash, repOrbitNodes] : repHasher.GetOrbits()) {
                        const auto &currentOrbitNodes = currentHasher.GetOrbitFromHash(hash);
                        for (size_t k = 0; k < repOrbitNodes.size(); ++k) {
                            // Map: current_subgraph_vertex -> representative_subgraph_local_idx
                            currentVertexToRepLocalIdx[currentLocalToGlobalMap[currentOrbitNodes[k]]]
                                = static_cast<VertexIdxT<ConstrGraphT>>(repOrbitNodes[k]);
                        }
                    }
                }

                // Apply the partition pattern
                for (const auto &currentVertex : currentSubgraphVertices) {
                    const auto repLocalIdx = currentVertexToRepLocalIdx.at(currentVertex);
                    auto spPair
                        = std::make_pair(bspSchedule.AssignedSuperstep(repLocalIdx), bspSchedule.AssignedProcessor(repLocalIdx));

                    if (maxBsp) {
                        spPair = std::make_pair(repLocalIdx, 0);
                    }

                    partition[currentVertex] = currentPartitionIdx + spProcToRelativePartition.at(spPair);
                }
                currentPartitionIdx += numPartitionsPerSubgraph;
            }
        }
    }
};

}    // namespace osp
} // namespace npu::tile_fwk

#endif // OSP_ISOMORPHIC_SUBGRAPH_SCHEDULER_HPP
