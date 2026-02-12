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
 * \file kl_improver.hpp
 * \brief
 */

#ifndef OSP_KL_IMPROVER_HPP
#define OSP_KL_IMPROVER_HPP

#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "kl_active_schedule.hpp"
#include "kl_util.hpp"
#include "passes/algorithms/osp/auxiliary/datastructures/heaps/PairingHeap.hpp"
#include "passes/algorithms/osp/bsp/model/util/CompatibleProcessorRange.hpp"
#include "passes/algorithms/osp/bsp/scheduler/ImprovementScheduler.hpp"
#include "passes/algorithms/osp/graph_algorithms/directed_graph_util.hpp"

namespace npu::tile_fwk {
namespace osp {

struct KlParameter {
    double timeQuality_ = 0.8;
    double superstepRemoveStrength_ = 0.5;
    unsigned numParallelLoops_ = 4;

    unsigned maxInnerIterationsReset_ = 500;
    unsigned maxNoImprovementIterations_ = 50;

    constexpr static unsigned abortScatterNodesViolationThreshold_ = 500;
    constexpr static unsigned initialViolationThreshold_ = 250;

    unsigned maxNoVioaltionsRemovedBacktrackReset_;
    unsigned removeStepEpocs_;
    unsigned nodeMaxStepSelectionEpochs_;
    unsigned maxNoVioaltionsRemovedBacktrackForRemoveStepReset_;
    unsigned maxOuterIterations_;
    unsigned tryRemoveStepAfterNumOuterIterations_;
    unsigned minInnerIterReset_;

    unsigned threadMinRange_ = 8;
    unsigned threadRangeGap_ = 0;
};

template <typename VertexType>
struct KlUpdateInfo {
    VertexType node_ = 0;

    bool fullUpdate_ = false;
    bool updateFromStep_ = false;
    bool updateToStep_ = false;
    bool updateEntireToStep_ = false;
    bool updateEntireFromStep_ = false;

    KlUpdateInfo() = default;

    KlUpdateInfo(VertexType n) : node_(n), fullUpdate_(false), updateEntireToStep_(false), updateEntireFromStep_(false) {}

    KlUpdateInfo(VertexType n, bool full)
        : node_(n), fullUpdate_(full), updateEntireToStep_(false), updateEntireFromStep_(false) {}
};

template <typename GraphT,
          typename CommCostFunctionT,
          unsigned windowSize = 1,
          typename CostT = double>
class KlImprover : public ImprovementScheduler<GraphT> {
  protected:
    constexpr static unsigned windowRange_ = 2 * windowSize + 1;
    constexpr static bool enableQuickMoves_ = true;
    constexpr static double epsilon_ = 1e-9;

    using VertexMemWeightT = osp::VMemwT<GraphT>;
    using VertexCommWeightT = osp::VCommwT<GraphT>;
    using VertexWorkWeightT = osp::VWorkwT<GraphT>;
    using VertexType = VertexIdxT<GraphT>;
    using EdgeType = EdgeDescT<GraphT>;

    using KlMove = KlMoveStruct<CostT, VertexType>;
    using HeapDatastructure = MaxPairingHeap<VertexType, KlMove>;
    using ActiveScheduleT = KlActiveSchedule<GraphT, CostT>;
    using NodeSelectionContainerT = AdaptiveAffinityTable<GraphT, CostT, ActiveScheduleT, windowSize>;
    using KlGainUpdateInfo = KlUpdateInfo<VertexType>;

    struct ThreadSearchContext {
        unsigned threadId_ = 0;
        unsigned startStep_ = 0;
        unsigned endStep_ = 0;
        unsigned originalEndStep_ = 0;

        VectorVertexLockManager<VertexType> lockManager_;
        HeapDatastructure maxGainHeap_;
        NodeSelectionContainerT affinityTable_;
        std::vector<std::vector<CostT>> localAffinityTable_;
        RewardPenaltyStrategy<CostT, CommCostFunctionT, ActiveScheduleT> rewardPenaltyStrat_;
        VertexSelectionStrategy<GraphT, NodeSelectionContainerT, ActiveScheduleT> selectionStrategy_;
        ThreadLocalActiveScheduleData<GraphT, CostT> activeScheduleData_;

        double averageGain_ = 0.0;
        unsigned maxInnerIterations_ = 0;
        unsigned noImprovementIterationsReducePenalty_ = 0;
        unsigned minInnerIter_ = 0;
        unsigned noImprovementIterationsIncreaseInnerIter_ = 0;
        unsigned stepSelectionEpochCounter_ = 0;
        unsigned stepSelectionCounter_ = 0;
        unsigned stepToRemove_ = 0;
        unsigned localSearchStartStep_ = 0;
        unsigned unlockEdgeBacktrackCounter_ = 0;
        unsigned unlockEdgeBacktrackCounterReset_ = 0;
        unsigned maxNoVioaltionsRemovedBacktrack_ = 0;

        inline unsigned NumSteps() const { return endStep_ - startStep_ + 1; }

        inline unsigned StartIdx(const unsigned nodeStep) const {
            return nodeStep < startStep_ + windowSize ? windowSize - (nodeStep - startStep_) : 0;
        }

        inline unsigned EndIdx(unsigned nodeStep) const {
            return nodeStep + windowSize <= endStep_ ? windowRange_ : windowRange_ - (nodeStep + windowSize - endStep_);
        }
    };

    bool computeWithTimeLimit_ = false;

    BspSchedule<GraphT> *inputSchedule_;
    const GraphT *graph_;
    const BspInstance<GraphT> *instance_;

    CompatibleProcessorRange<GraphT> procRange_;

    KlParameter parameters_;
    std::mt19937 gen_;

    ActiveScheduleT activeSchedule_;
    CommCostFunctionT commCostF_;
    std::vector<ThreadSearchContext> threadDataVec_;
    std::vector<bool> threadFinishedVec_;

    inline unsigned RelStepIdx(const unsigned nodeStep, const unsigned moveStep) const {
        return (moveStep >= nodeStep) ? ((moveStep - nodeStep) + windowSize) : (windowSize - (nodeStep - moveStep));
    }

    inline bool IsCompatible(VertexType node, unsigned proc) const {
        return activeSchedule_.GetInstance().IsCompatible(node, proc);
    }

    void SetStartStep(const unsigned step, ThreadSearchContext &threadData) {
        threadData.startStep_ = step;
        threadData.stepToRemove_ = step;
        threadData.stepSelectionCounter_ = step;

        threadData.averageGain_ = 0.0;
        threadData.maxInnerIterations_ = parameters_.maxInnerIterationsReset_;
        threadData.noImprovementIterationsReducePenalty_ = parameters_.maxNoImprovementIterations_ / 5;
        threadData.minInnerIter_ = parameters_.minInnerIterReset_;
        threadData.stepSelectionEpochCounter_ = 0;
        threadData.noImprovementIterationsIncreaseInnerIter_ = 10;
        threadData.unlockEdgeBacktrackCounterReset_ = 0;
        threadData.unlockEdgeBacktrackCounter_ = threadData.unlockEdgeBacktrackCounterReset_;
        threadData.maxNoVioaltionsRemovedBacktrack_ = parameters_.maxNoVioaltionsRemovedBacktrackReset_;
    }

    KlMove GetBestMove(NodeSelectionContainerT &affinityTable,
                       VectorVertexLockManager<VertexType> &lockManager,
                       HeapDatastructure &maxGainHeap) {
        // To introduce non-determinism and help escape local optima, if there are multiple moves with the same
        // top gain, we randomly select one. We check up to `local_max` ties.
        const unsigned localMax = 50;
        std::vector<VertexType> topGainNodes = maxGainHeap.GetTopKeys(localMax);

        if (topGainNodes.empty()) {
            // This case is guarded by the caller, but for safety:
            topGainNodes.push_back(maxGainHeap.Top());
        }

        std::uniform_int_distribution<size_t> dis(0, topGainNodes.size() - 1);
        const VertexType node = topGainNodes[dis(gen_)];

        KlMove bestMove = maxGainHeap.GetValue(node);
        maxGainHeap.Erase(node);
        lockManager.Lock(node);
        affinityTable.Remove(node);

        return bestMove;
    }

    inline void ProcessOtherStepsBestMove(const unsigned idx,
                                          const VertexType &node,
                                          const CostT affinityCurrentProcStep,
                                          CostT &maxGain,
                                          unsigned &maxProc,
                                          unsigned &maxStep,
                                          const std::vector<std::vector<CostT>> &affinityTableNode) const {
        for (const unsigned p : procRange_.CompatibleProcessorsVertex(node)) {
            const CostT gain = affinityCurrentProcStep - affinityTableNode[p][idx];
            if (gain > maxGain) {
                maxGain = gain;
                maxProc = p;
                maxStep = idx;
            }
        }
    }

    template <bool moveToSameSuperStep>
    KlMove ComputeBestMove(VertexType node,
                           const std::vector<std::vector<CostT>> &affinityTableNode,
                           ThreadSearchContext &threadData) {
        const unsigned nodeStep = activeSchedule_.AssignedSuperstep(node);
        const unsigned nodeProc = activeSchedule_.AssignedProcessor(node);

        CostT maxGain = std::numeric_limits<CostT>::lowest();

        unsigned maxProc = std::numeric_limits<unsigned>::max();
        unsigned maxStep = std::numeric_limits<unsigned>::max();

        const CostT affinityCurrentProcStep = affinityTableNode[nodeProc][windowSize];

        unsigned idx = threadData.StartIdx(nodeStep);
        for (; idx < windowSize; idx++) {
            ProcessOtherStepsBestMove(idx, node, affinityCurrentProcStep, maxGain, maxProc, maxStep, affinityTableNode);
        }

        if constexpr (moveToSameSuperStep) {
            for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
                if (proc == nodeProc) {
                    continue;
                }

                const CostT gain = affinityCurrentProcStep - affinityTableNode[proc][windowSize];
                if (gain > maxGain) {
                    maxGain = gain;
                    maxProc = proc;
                    maxStep = idx;
                }
            }
        }

        idx++;

        const unsigned bound = threadData.EndIdx(nodeStep);
        for (; idx < bound; idx++) {
            ProcessOtherStepsBestMove(idx, node, affinityCurrentProcStep, maxGain, maxProc, maxStep, affinityTableNode);
        }

        return KlMove(node, maxGain, nodeProc, nodeStep, maxProc, nodeStep + maxStep - windowSize);
    }

    CostT ComputeNodeProcAffinity(VertexWorkWeightT vertexWeight,
                                   VertexWorkWeightT maxWork,
                                   VertexWorkWeightT secondMaxWork,
                                   VertexWorkWeightT stepProcWork,
                                   unsigned maxWorkProcCount) {
        const bool isSoleMaxProcessor = (maxWorkProcCount == 1) && (maxWork == stepProcWork);
        return isSoleMaxProcessor ? std::min(vertexWeight, maxWork - secondMaxWork) : 0.0;
    }

    void UpdateMoveProcAffinity(VertexType node,
                                unsigned nodeStep,
                                unsigned moveProc,
                                VertexWorkWeightT weightAdjustment,
                                VertexWorkWeightT prevMaxWork,
                                VertexWorkWeightT newMaxWeight,
                                CostT prevNodeProcAffinity,
                                CostT newNodeProcAffinity,
                                std::vector<std::vector<CostT>> &affinityTableNode) {
        if (activeSchedule_.AssignedProcessor(node) == moveProc || !IsCompatible(node, moveProc)) {
            return;
        }
        const VertexWorkWeightT vertexWeight = graph_->VertexWorkWeight(node);
        const VertexWorkWeightT prevNewWeight
            = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, moveProc) + weightAdjustment;
        const CostT prevOtherAffinity = ComputeSameStepAffinity(prevMaxWork, prevNewWeight, prevNodeProcAffinity);
        const VertexWorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, moveProc);
        const CostT otherAffinity = ComputeSameStepAffinity(newMaxWeight, newWeight, newNodeProcAffinity);
        affinityTableNode[moveProc][windowSize] += (otherAffinity - prevOtherAffinity);
    }

    void HandleSameStepSameNode(VertexType node,
                                const KlMove &move,
                                const PreMoveWorkData<VertexWorkWeightT> &prevWorkData,
                                std::vector<std::vector<CostT>> &affinityTableNode,
                                KlGainUpdateInfo &updateInfo) {
        const unsigned nodeStep = activeSchedule_.AssignedSuperstep(node);
        const VertexWorkWeightT vertexWeight = graph_->VertexWorkWeight(node);
        const unsigned nodeProc = activeSchedule_.AssignedProcessor(node);

        const VertexWorkWeightT prevMaxWork = prevWorkData.fromStepMaxWork_;
        const VertexWorkWeightT newMaxWeight = activeSchedule_.GetStepMaxWork(move.fromStep_);
        const VertexWorkWeightT newStepProcWork = activeSchedule_.GetStepProcessorWork(nodeStep, nodeProc);
        const VertexWorkWeightT prevStepProcWork
            = (nodeProc == move.fromProc_) ? newStepProcWork + graph_->VertexWorkWeight(move.node_)
              : (nodeProc == move.toProc_) ? newStepProcWork - graph_->VertexWorkWeight(move.node_)
                                           : newStepProcWork;

        const CostT prevNodeProcAffinity = ComputeNodeProcAffinity(vertexWeight, prevMaxWork,
            prevWorkData.fromStepSecondMaxWork_, prevStepProcWork, prevWorkData.fromStepMaxWorkProcessorCount_);
        const CostT newNodeProcAffinity = ComputeNodeProcAffinity(vertexWeight, newMaxWeight,
            activeSchedule_.GetStepSecondMaxWork(move.fromStep_), newStepProcWork,
            activeSchedule_.GetStepMaxWorkProcessorCount()[nodeStep]);

        const CostT diff = newNodeProcAffinity - prevNodeProcAffinity;
        if (std::abs(diff) > epsilon_) {
            updateInfo.fullUpdate_ = true;
            affinityTableNode[nodeProc][windowSize] += diff;
        }

        if ((prevMaxWork != newMaxWeight) || updateInfo.fullUpdate_) {
            updateInfo.updateEntireFromStep_ = true;
            for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
                if ((proc == nodeProc) || (proc == move.fromProc_) || (proc == move.toProc_)) {
                    continue;
                }
                const VertexWorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, proc);
                const CostT prevOtherAffinity = ComputeSameStepAffinity(prevMaxWork, newWeight, prevNodeProcAffinity);
                const CostT otherAffinity = ComputeSameStepAffinity(newMaxWeight, newWeight, newNodeProcAffinity);
                affinityTableNode[proc][windowSize] += (otherAffinity - prevOtherAffinity);
            }
        }

        const VertexWorkWeightT moveNodeWeight = graph_->VertexWorkWeight(move.node_);
        UpdateMoveProcAffinity(node, nodeStep, move.fromProc_, moveNodeWeight,
                               prevMaxWork, newMaxWeight, prevNodeProcAffinity, newNodeProcAffinity, affinityTableNode);
        UpdateMoveProcAffinity(node, nodeStep, move.toProc_, -moveNodeWeight,
                               prevMaxWork, newMaxWeight, prevNodeProcAffinity, newNodeProcAffinity, affinityTableNode);
    }

    void HandleSameStepDifferentNodeMaxChanged(VertexType node,
                                               const KlMove &move,
                                               VertexWorkWeightT vertexWeight,
                                               VertexWorkWeightT prevMaxWork,
                                               VertexWorkWeightT newMaxWeight,
                                               unsigned idx,
                                               std::vector<std::vector<CostT>> &affinityTableNode) {
        for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
            const VertexWorkWeightT newWeight
                = vertexWeight + activeSchedule_.GetStepProcessorWork(move.fromStep_, proc);
            if (proc == move.fromProc_) {
                const VertexWorkWeightT prevNewWeight
                    = vertexWeight + activeSchedule_.GetStepProcessorWork(move.fromStep_, proc)
                      + graph_->VertexWorkWeight(move.node_);
                const CostT prevAffinity = prevMaxWork < prevNewWeight ? static_cast<CostT>(prevNewWeight)
                                                                             - static_cast<CostT>(prevMaxWork)
                                                                       : 0.0;
                const CostT newAffinity = newMaxWeight < newWeight
                                              ? static_cast<CostT>(newWeight) - static_cast<CostT>(newMaxWeight)
                                              : 0.0;
                affinityTableNode[proc][idx] += newAffinity - prevAffinity;
            } else if (proc == move.toProc_) {
                const VertexWorkWeightT prevNewWeight = vertexWeight
                                                        + activeSchedule_.GetStepProcessorWork(move.toStep_, proc)
                                                        - graph_->VertexWorkWeight(move.node_);
                const CostT prevAffinity = prevMaxWork < prevNewWeight ? static_cast<CostT>(prevNewWeight)
                                                                             - static_cast<CostT>(prevMaxWork)
                                                                       : 0.0;
                const CostT newAffinity = newMaxWeight < newWeight
                                              ? static_cast<CostT>(newWeight) - static_cast<CostT>(newMaxWeight)
                                              : 0.0;
                affinityTableNode[proc][idx] += newAffinity - prevAffinity;
            } else {
                const CostT prevAffinity = prevMaxWork < newWeight
                                               ? static_cast<CostT>(newWeight) - static_cast<CostT>(prevMaxWork)
                                               : 0.0;
                const CostT newAffinity = newMaxWeight < newWeight
                                              ? static_cast<CostT>(newWeight) - static_cast<CostT>(newMaxWeight)
                                              : 0.0;
                affinityTableNode[proc][idx] += newAffinity - prevAffinity;
            }
        }
    }

    void HandleSameStepDifferentNodeMaxUnchanged(VertexType node,
                                                 const KlMove &move,
                                                 VertexWorkWeightT vertexWeight,
                                                 VertexWorkWeightT prevMaxWork,
                                                 VertexWorkWeightT newMaxWeight,
                                                 unsigned idx,
                                                 std::vector<std::vector<CostT>> &affinityTableNode) {
        if (IsCompatible(node, move.fromProc_)) {
            const VertexWorkWeightT fromNewWeight
                = vertexWeight + activeSchedule_.GetStepProcessorWork(move.fromStep_, move.fromProc_);
            const VertexWorkWeightT fromPrevNewWeight = fromNewWeight + graph_->VertexWorkWeight(move.node_);
            const CostT fromPrevAffinity = prevMaxWork < fromPrevNewWeight ? static_cast<CostT>(fromPrevNewWeight)
                                                                                 - static_cast<CostT>(prevMaxWork)
                                                                           : 0.0;

            const CostT fromNewAffinity = newMaxWeight < fromNewWeight ? static_cast<CostT>(fromNewWeight)
                                                                             - static_cast<CostT>(newMaxWeight)
                                                                       : 0.0;
            affinityTableNode[move.fromProc_][idx] += fromNewAffinity - fromPrevAffinity;
        }

        if (IsCompatible(node, move.toProc_)) {
            const VertexWorkWeightT toNewWeight
                = vertexWeight + activeSchedule_.GetStepProcessorWork(move.toStep_, move.toProc_);
            const VertexWorkWeightT toPrevNewWeight = toNewWeight - graph_->VertexWorkWeight(move.node_);
            const CostT toPrevAffinity = prevMaxWork < toPrevNewWeight ? static_cast<CostT>(toPrevNewWeight)
                                                                             - static_cast<CostT>(prevMaxWork)
                                                                       : 0.0;

            const CostT toNewAffinity = newMaxWeight < toNewWeight
                                            ? static_cast<CostT>(toNewWeight) - static_cast<CostT>(newMaxWeight)
                                            : 0.0;
            affinityTableNode[move.toProc_][idx] += toNewAffinity - toPrevAffinity;
        }
    }

    void HandleSameStepMove(VertexType node,
                            const KlMove &move,
                            const PreMoveWorkData<VertexWorkWeightT> &prevWorkData,
                            std::vector<std::vector<CostT>> &affinityTableNode,
                            KlGainUpdateInfo &updateInfo) {
        const unsigned nodeStep = activeSchedule_.AssignedSuperstep(node);
        const VertexWorkWeightT vertexWeight = graph_->VertexWorkWeight(node);

        const unsigned lowerBound = move.fromStep_ > windowSize ? move.fromStep_ - windowSize : 0;
        if (!(lowerBound <= nodeStep && nodeStep <= move.fromStep_ + windowSize)) {
            return;
        }

        updateInfo.updateFromStep_ = true;
        updateInfo.updateToStep_ = true;

        if (nodeStep == move.fromStep_) {
            HandleSameStepSameNode(node, move, prevWorkData, affinityTableNode, updateInfo);
        } else {
            const VertexWorkWeightT prevMaxWork = prevWorkData.fromStepMaxWork_;
            const VertexWorkWeightT newMaxWeight = activeSchedule_.GetStepMaxWork(move.fromStep_);
            const unsigned idx = RelStepIdx(nodeStep, move.fromStep_);
            if (prevMaxWork != newMaxWeight) {
                updateInfo.updateEntireFromStep_ = true;
                HandleSameStepDifferentNodeMaxChanged(node, move, vertexWeight, prevMaxWork, newMaxWeight, idx, affinityTableNode);
            } else {
                HandleSameStepDifferentNodeMaxUnchanged(node, move, vertexWeight, prevMaxWork, newMaxWeight, idx, affinityTableNode);
            }
        }
    }

    KlGainUpdateInfo UpdateNodeWorkAffinityAfterMove(VertexType node,
                                                     KlMove move,
                                                     const PreMoveWorkData<VertexWorkWeightT> &prevWorkData,
                                                     std::vector<std::vector<CostT>> &affinityTableNode) {
        KlGainUpdateInfo updateInfo(node);

        if (move.fromStep_ == move.toStep_) {
            HandleSameStepMove(node, move, prevWorkData, affinityTableNode, updateInfo);
        } else {
            const unsigned nodeStep = activeSchedule_.AssignedSuperstep(node);
            const unsigned nodeProc = activeSchedule_.AssignedProcessor(node);
            const VertexWorkWeightT vertexWeight = graph_->VertexWorkWeight(node);
            ProcessWorkUpdateStep(node, nodeStep, nodeProc, vertexWeight, move.fromStep_, move.fromProc_, graph_->VertexWorkWeight(move.node_),
                                  prevWorkData.fromStepMaxWork_, prevWorkData.fromStepSecondMaxWork_, prevWorkData.fromStepMaxWorkProcessorCount_,
                                  updateInfo.updateFromStep_, updateInfo.updateEntireFromStep_, updateInfo.fullUpdate_, affinityTableNode);
            ProcessWorkUpdateStep(node, nodeStep, nodeProc, vertexWeight, move.toStep_, move.toProc_, -graph_->VertexWorkWeight(move.node_),
                                  prevWorkData.toStepMaxWork_, prevWorkData.toStepSecondMaxWork_, prevWorkData.toStepMaxWorkProcessorCount_,
                                  updateInfo.updateToStep_, updateInfo.updateEntireToStep_, updateInfo.fullUpdate_, affinityTableNode);
        }

        return updateInfo;
    }

    void ProcessWorkUpdateStep(VertexType node, unsigned nodeStep, unsigned nodeProc, VertexWorkWeightT vertexWeight, unsigned moveStep, unsigned moveProc, VertexWorkWeightT moveCorrectionNodeWeight,
                               const VertexWorkWeightT prevMoveStepMaxWork, const VertexWorkWeightT prevMoveStepSecondMaxWork, unsigned prevMoveStepMaxWorkProcessorCount,
                               bool &updateStep, bool &updateEntireStep, bool &fullUpdate, std::vector<std::vector<CostT>> &affinityTableNode);
    void UpdateNodeWorkAffinity(NodeSelectionContainerT &nodes, KlMove move, const PreMoveWorkData<VertexWorkWeightT> &prevWorkData, std::map<VertexType, KlGainUpdateInfo> &recomputeMaxGain);
    void UpdateBestMove(VertexType node, unsigned step, unsigned proc, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData);
    void UpdateBestMove(VertexType node, unsigned step, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData);
    void UpdateMaxGain(KlMove move, std::map<VertexType, KlGainUpdateInfo> &recomputeMaxGain, ThreadSearchContext &threadData);
    void ComputeWorkAffinity(VertexType node, std::vector<std::vector<CostT>> &affinityTableNode, ThreadSearchContext &threadData);

    inline void RecomputeNodeMaxGain(VertexType node, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData) {
        const auto bestMove = ComputeBestMove<true>(node, affinityTable[node], threadData);
        threadData.maxGainHeap_.Update(node, bestMove);
    }

    inline CostT ComputeSameStepAffinity(const VertexWorkWeightT &maxWorkForStep,
                                         const VertexWorkWeightT &newWeight,
                                         const CostT &nodeProcAffinity) {
        const CostT maxWorkAfterRemoval = static_cast<CostT>(maxWorkForStep) - nodeProcAffinity;
        if (newWeight > maxWorkAfterRemoval) {
            return newWeight - maxWorkAfterRemoval;
        }
        return 0.0;
    }

    inline CostT ApplyMove(KlMove move, ThreadSearchContext &threadData) {
        activeSchedule_.ApplyMove(move, threadData.activeScheduleData_);
        commCostF_.UpdateDatastructureAfterMove(move, threadData.startStep_, threadData.endStep_);
        CostT changeInCost = -move.gain_;
        changeInCost += static_cast<CostT>(threadData.activeScheduleData_.resolvedViolations_.size())
                        * threadData.rewardPenaltyStrat_.reward_;
        changeInCost
            -= static_cast<CostT>(threadData.activeScheduleData_.newViolations_.size()) * threadData.rewardPenaltyStrat_.penalty_;

        threadData.activeScheduleData_.UpdateCost(changeInCost);

        return changeInCost;
    }

    enum class QuickMoveResult { kContinue, kSkip, kAbort };

    QuickMoveResult ProcessQuickMoveCandidate(VertexType nextNodeToMove, unsigned &innerIter, ThreadSearchContext &threadData,
                                              std::unordered_set<VertexType> &localLock, std::vector<VertexType> &quickMovesStack) {
        threadData.rewardPenaltyStrat_.InitRewardPenalty(
            static_cast<double>(threadData.activeScheduleData_.currentViolations_.size()) + 1.0);
        ComputeNodeAffinities(nextNodeToMove, threadData.localAffinityTable_, threadData);
        KlMove bestQuickMove = ComputeBestMove<true>(nextNodeToMove, threadData.localAffinityTable_, threadData);

        localLock.insert(nextNodeToMove);
        if (bestQuickMove.gain_ <= std::numeric_limits<CostT>::lowest()) {
            return QuickMoveResult::kContinue;
        }

        ApplyMove(bestQuickMove, threadData);
        innerIter++;

        if (threadData.activeScheduleData_.newViolations_.size() > 0) {
            for (const auto &keyValuePair : threadData.activeScheduleData_.newViolations_) {
                const auto &key = keyValuePair.first;
                if (localLock.find(key) != localLock.end()) {
                    return QuickMoveResult::kAbort;
                }
                quickMovesStack.push_back(key);
            }
            return QuickMoveResult::kContinue;
        }

        if (threadData.activeScheduleData_.feasible_) {
            return QuickMoveResult::kAbort;
        }

        return QuickMoveResult::kContinue;
    }

    void RunQuickMoves(unsigned &innerIter,
                       ThreadSearchContext &threadData,
                       const CostT changeInCost,
                       const VertexType bestMoveNode) {
        innerIter++;

        const size_t numAppliedMoves = threadData.activeScheduleData_.appliedMoves_.size() - 1;
        const CostT savedCost = threadData.activeScheduleData_.cost_ - changeInCost;

        std::unordered_set<VertexType> localLock;
        localLock.insert(bestMoveNode);
        std::vector<VertexType> quickMovesStack;
        quickMovesStack.reserve(10 + threadData.activeScheduleData_.newViolations_.size() * 2);

        for (const auto &keyValuePair : threadData.activeScheduleData_.newViolations_) {
            const auto &key = keyValuePair.first;
            quickMovesStack.push_back(key);
        }

        while (quickMovesStack.size() > 0) {
            auto nextNodeToMove = quickMovesStack.back();
            quickMovesStack.pop_back();

            QuickMoveResult result = ProcessQuickMoveCandidate(nextNodeToMove, innerIter, threadData, localLock, quickMovesStack);
            if (result == QuickMoveResult::kAbort) {
                break;
            }
        }

        if (!threadData.activeScheduleData_.feasible_) {
            activeSchedule_.RevertScheduleToBound(numAppliedMoves, savedCost, true, commCostF_, threadData.activeScheduleData_, threadData.startStep_, threadData.endStep_);
        }

        threadData.affinityTable_.Trim();
        threadData.maxGainHeap_.Clear();
        threadData.rewardPenaltyStrat_.InitRewardPenalty(1.0);
        InsertGainHeap(threadData);    // Re-initialize the heap with the current state
    }

    enum class InnerIterResult { kContinue, kBreak, kSkip };

    InnerIterResult HandleViolationEscalation(unsigned &violationRemovedCount, unsigned &resetCounter, unsigned &innerIter,
                                              bool iterInitalFeasible, ThreadSearchContext &threadData) {
        violationRemovedCount++;
        if (violationRemovedCount <= 3) {
            return InnerIterResult::kContinue;
        }

        if (resetCounter >= threadData.maxNoVioaltionsRemovedBacktrack_
            || (iterInitalFeasible && threadData.activeScheduleData_.cost_ >= threadData.activeScheduleData_.bestCost_)) {
            return InnerIterResult::kBreak;
        }

        threadData.affinityTable_.ResetNodeSelection();
        threadData.maxGainHeap_.Clear();
        threadData.lockManager_.Clear();
        threadData.selectionStrategy_.SelectNodesViolations(
            threadData.affinityTable_,
            threadData.activeScheduleData_.currentViolations_,
            threadData.startStep_,
            threadData.endStep_);
        threadData.rewardPenaltyStrat_.InitRewardPenalty(
            static_cast<double>(threadData.activeScheduleData_.currentViolations_.size()));
        InsertGainHeap(threadData);
        resetCounter++;
        innerIter++;
        return InnerIterResult::kSkip;
    }

    InnerIterResult HandleViolations(unsigned &violationRemovedCount, unsigned &resetCounter, unsigned &innerIter,
                                     bool iterInitalFeasible, ThreadSearchContext &threadData) {
        if (threadData.activeScheduleData_.currentViolations_.size() == 0) {
            return InnerIterResult::kContinue;
        }

        if (threadData.activeScheduleData_.resolvedViolations_.size() > 0) {
            violationRemovedCount = 0;
            return InnerIterResult::kContinue;
        }
        return HandleViolationEscalation(violationRemovedCount, resetCounter, innerIter, iterInitalFeasible, threadData);
    }

    bool ProcessInnerIteration(const KlMove &bestMove,
                               std::vector<VertexType> &newNodes,
                               std::vector<VertexType> &unlockNodes,
                               std::map<VertexType, KlGainUpdateInfo> &recomputeMaxGain,
                               const PreMoveWorkData<VertexWorkWeightT> &prevWorkData,
                               ThreadSearchContext &threadData) {
        if (IsLocalSearchBlocked(threadData)) {
            if (not BlockedEdgeStrategy(bestMove.node_, unlockNodes, threadData)) {
                return false;
            }
        }

        threadData.affinityTable_.Trim();
        UpdateAffinities(bestMove, threadData, recomputeMaxGain, newNodes, prevWorkData);

        for (const auto v : unlockNodes) {
            threadData.lockManager_.Unlock(v);
        }
        newNodes.insert(newNodes.end(), unlockNodes.begin(), unlockNodes.end());
        unlockNodes.clear();

        UpdateMaxGain(bestMove, recomputeMaxGain, threadData);
        InsertNewNodesGainHeap(newNodes, threadData.affinityTable_, threadData);

        recomputeMaxGain.clear();
        newNodes.clear();
        return true;
    }

    void RunInnerLoop(ThreadSearchContext &threadData,
                      std::vector<VertexType> &newNodes,
                      std::vector<VertexType> &unlockNodes,
                      std::map<VertexType, KlGainUpdateInfo> &recomputeMaxGain) {
        unsigned innerIter = 0;
        unsigned violationRemovedCount = 0;
        unsigned resetCounter = 0;
        bool iterInitalFeasible = threadData.activeScheduleData_.feasible_;

        while (innerIter < threadData.maxInnerIterations_ && threadData.maxGainHeap_.size() > 0) {
            KlMove bestMove = GetBestMove(threadData.affinityTable_, threadData.lockManager_, threadData.maxGainHeap_);
            if (bestMove.gain_ <= std::numeric_limits<CostT>::lowest()) {
                break;
            }
            UpdateAvgGain(bestMove.gain_, innerIter, threadData.averageGain_);

            if (innerIter > threadData.minInnerIter_ && threadData.averageGain_ < 0.0) {
                break;
            }

            const auto prevWorkData = activeSchedule_.GetPreMoveWorkData(bestMove);
            const typename CommCostFunctionT::PreMoveCommDataT prevCommData = commCostF_.GetPreMoveCommData(bestMove);
            const CostT changeInCost = ApplyMove(bestMove, threadData);

            if constexpr (enableQuickMoves_) {
                if (iterInitalFeasible && threadData.activeScheduleData_.newViolations_.size() > 0) {
                    RunQuickMoves(innerIter, threadData, changeInCost, bestMove.node_);
                    continue;
                }
            }

            InnerIterResult violationResult = HandleViolations(violationRemovedCount, resetCounter, innerIter, iterInitalFeasible, threadData);
            if (violationResult == InnerIterResult::kBreak) {
                break;
            }
            if (violationResult == InnerIterResult::kSkip) {
                continue;
            }

            if (!ProcessInnerIteration(bestMove, newNodes, unlockNodes, recomputeMaxGain, prevWorkData, threadData)) {
                break;
            }

            innerIter++;
        }
    }

    bool ShouldTerminateOuterLoop(const std::chrono::time_point<std::chrono::high_resolution_clock> &startTime,
                                  unsigned &noImprovementIterCounter,
                                  CostT initialInnerIterCost,
                                  ThreadSearchContext &threadData) {
        if (computeWithTimeLimit_) {
            auto finishTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(finishTime - startTime).count();
            if (duration > ImprovementScheduler<GraphT>::timeLimitSeconds_) {
                return true;
            }
        }

        if (OtherThreadsFinished(threadData.threadId_)) {
            return true;
        }

        if (initialInnerIterCost <= threadData.activeScheduleData_.cost_) {
            noImprovementIterCounter++;
            if (noImprovementIterCounter >= parameters_.maxNoImprovementIterations_) {
                return true;
            }
        } else {
            noImprovementIterCounter = 0;
        }

        return false;
    }

    void RunLocalSearch(ThreadSearchContext &threadData) {
        std::vector<VertexType> newNodes;
        std::vector<VertexType> unlockNodes;
        std::map<VertexType, KlGainUpdateInfo> recomputeMaxGain;

        const auto startTime = std::chrono::high_resolution_clock::now();
        unsigned noImprovementIterCounter = 0;

        for (unsigned outerIter = 0; outerIter < parameters_.maxOuterIterations_; outerIter++) {
            CostT initialInnerIterCost = threadData.activeScheduleData_.cost_;

            ResetInnerSearchStructures(threadData);
            SelectActiveNodes(threadData);
            threadData.rewardPenaltyStrat_.InitRewardPenalty(
                static_cast<double>(threadData.activeScheduleData_.currentViolations_.size()) + 1.0);
            InsertGainHeap(threadData);

            RunInnerLoop(threadData, newNodes, unlockNodes, recomputeMaxGain);

            activeSchedule_.RevertToBestSchedule(threadData.localSearchStartStep_,
                                                 threadData.stepToRemove_,
                                                 commCostF_,
                                                 threadData.activeScheduleData_,
                                                 threadData.startStep_,
                                                 threadData.endStep_);

            if (ShouldTerminateOuterLoop(startTime, noImprovementIterCounter, initialInnerIterCost, threadData)) {
                break;
            }

            AdjustLocalSearchParameters(outerIter, noImprovementIterCounter, threadData);
        }

        threadFinishedVec_[threadData.threadId_] = true;
    }

    bool OtherThreadsFinished(const unsigned threadId) {
        const size_t numThreads = threadFinishedVec_.size();
        if (numThreads == 1) {
            return false;
        }

        for (size_t i = 0; i < numThreads; i++) {
            if (i != threadId && !threadFinishedVec_[i]) {
                return false;
            }
        }
        return true;
    }

    inline void UpdateAffinities(const KlMove &bestMove,
                                 ThreadSearchContext &threadData,
                                 std::map<VertexType, KlGainUpdateInfo> &recomputeMaxGain,
                                 std::vector<VertexType> &newNodes,
                                 const PreMoveWorkData<VertexWorkWeightT> &prevWorkData) {
        UpdateNodeWorkAffinity(threadData.affinityTable_, bestMove, prevWorkData, recomputeMaxGain);
        commCostF_.UpdateNodeCommAffinity(bestMove, threadData, threadData.rewardPenaltyStrat_.penalty_,
                                            threadData.rewardPenaltyStrat_.reward_, recomputeMaxGain, newNodes);
        
    }

    inline bool BlockedEdgeStrategy(VertexType node, std::vector<VertexType> &unlockNodes, ThreadSearchContext &threadData) {
        if (threadData.unlockEdgeBacktrackCounter_ > 1) {
            for (const auto vertexEdgePair : threadData.activeScheduleData_.newViolations_) {
                const auto &e = vertexEdgePair.second;
                const auto sourceV = Source(e, *graph_);
                const auto targetV = Target(e, *graph_);

                if (node == sourceV && threadData.lockManager_.IsLocked(targetV)) {
                    unlockNodes.push_back(targetV);
                } else if (node == targetV && threadData.lockManager_.IsLocked(sourceV)) {
                    unlockNodes.push_back(sourceV);
                }
            }

            threadData.unlockEdgeBacktrackCounter_--;
            return true;
        } else {
            return false;    // or reset local search and initalize with violating nodes
        }
    }

    inline void AdjustLocalSearchParameters(unsigned outerIter, unsigned noImpCounter, ThreadSearchContext &threadData) {
        if (noImpCounter >= threadData.noImprovementIterationsReducePenalty_
            && threadData.rewardPenaltyStrat_.initialPenalty_ > 1.0) {
            threadData.rewardPenaltyStrat_.initialPenalty_
                = static_cast<CostT>(std::floor(std::sqrt(threadData.rewardPenaltyStrat_.initialPenalty_)));
            threadData.unlockEdgeBacktrackCounterReset_ += 1;
            threadData.noImprovementIterationsReducePenalty_ += 15;
        }

        if (parameters_.tryRemoveStepAfterNumOuterIterations_ > 0
            && ((outerIter + 1) % parameters_.tryRemoveStepAfterNumOuterIterations_) == 0) {
            threadData.stepSelectionEpochCounter_ = 0;
        }

        if (noImpCounter >= threadData.noImprovementIterationsIncreaseInnerIter_) {
            threadData.minInnerIter_ = static_cast<unsigned>(std::ceil(threadData.minInnerIter_ * 2.2));
            threadData.noImprovementIterationsIncreaseInnerIter_ += 20;
        }
    }

    bool IsLocalSearchBlocked(ThreadSearchContext &threadData);
    void SetParameters(VertexIdxT<GraphT> numNodes);
    void ResetInnerSearchStructures(ThreadSearchContext &threadData) const;
    void InitializeDatastructures(BspSchedule<GraphT> &schedule);
    void PrintHeap(HeapDatastructure &maxGainHeap) const;
    void CleanupDatastructures();
    void UpdateAvgGain(const CostT gain, const unsigned numIter, double &averageGain);
    void InsertGainHeap(ThreadSearchContext &threadData);
    void InsertNewNodesGainHeap(std::vector<VertexType> &newNodes, NodeSelectionContainerT &nodes, ThreadSearchContext &threadData);

    inline void ComputeNodeAffinities(VertexType node,
                                      std::vector<std::vector<CostT>> &affinityTableNode,
                                      ThreadSearchContext &threadData) {
        ComputeWorkAffinity(node, affinityTableNode, threadData);
        commCostF_.ComputeCommAffinity(node,
                                       affinityTableNode,
                                       threadData.rewardPenaltyStrat_.penalty_,
                                       threadData.rewardPenaltyStrat_.reward_,
                                       threadData.startStep_,
                                       threadData.endStep_);
    }

    void SelectActiveNodes(ThreadSearchContext &threadData) {
        if (SelectNodesCheckRemoveSuperstep(threadData.stepToRemove_, threadData)) {
            activeSchedule_.SwapEmptyStepFwd(threadData.stepToRemove_, threadData.endStep_);
            threadData.endStep_--;
            threadData.localSearchStartStep_ = static_cast<unsigned>(threadData.activeScheduleData_.appliedMoves_.size());
            threadData.activeScheduleData_.UpdateCost(static_cast<CostT>(-1.0 * instance_->SynchronisationCosts()));

            if (threadData.activeScheduleData_.currentViolations_.size() > parameters_.initialViolationThreshold_) {
                activeSchedule_.RevertToBestSchedule(threadData.localSearchStartStep_,
                                                     threadData.stepToRemove_,
                                                     commCostF_,
                                                     threadData.activeScheduleData_,
                                                     threadData.startStep_,
                                                     threadData.endStep_);
            } else {
                threadData.unlockEdgeBacktrackCounter_
                    = static_cast<unsigned>(threadData.activeScheduleData_.currentViolations_.size());
                threadData.maxInnerIterations_
                    = std::max(threadData.unlockEdgeBacktrackCounter_ * 5u, parameters_.maxInnerIterationsReset_);
                threadData.maxNoVioaltionsRemovedBacktrack_ = parameters_.maxNoVioaltionsRemovedBacktrackForRemoveStepReset_;
                return;
            }
        }
        threadData.localSearchStartStep_ = 0;
        threadData.selectionStrategy_.SelectActiveNodes(threadData.affinityTable_, threadData.startStep_, threadData.endStep_);
    }

    bool CheckRemoveSuperstep(unsigned step);
    bool SelectNodesCheckRemoveSuperstep(unsigned &step, ThreadSearchContext &threadData);
    bool ScatterNodesSuperstep(unsigned step, ThreadSearchContext &threadData);
    void SynchronizeActiveSchedule(const unsigned numThreads);

  public:

    KlImprover(unsigned seed = 42) : ImprovementScheduler<GraphT>() { gen_ = std::mt19937(seed); }

    virtual ~KlImprover() = default;

    virtual ReturnStatus ImproveSchedule(BspSchedule<GraphT> &schedule) override {
        if (schedule.GetInstance().NumberOfProcessors() < 2) {
            return ReturnStatus::OSP_BEST_FOUND;
        }

        const unsigned numThreads = 1;

        threadDataVec_.resize(numThreads);
        threadFinishedVec_.assign(numThreads, true);

        SetParameters(schedule.GetInstance().NumberOfVertices());
        InitializeDatastructures(schedule);
        const CostT initialCost = activeSchedule_.GetCost();
        const unsigned numSteps = schedule.NumberOfSupersteps();

        SetStartStep(0, threadDataVec_[0]);
        threadDataVec_[0].endStep_ = (numSteps > 0) ? numSteps - 1 : 0;

        auto &threadData = this->threadDataVec_[0];
        threadData.activeScheduleData_.InitializeCost(activeSchedule_.GetCost());
        threadData.selectionStrategy_.Setup(threadData.startStep_, threadData.endStep_);
        RunLocalSearch(threadData);

        SynchronizeActiveSchedule(numThreads);

        if (initialCost > activeSchedule_.GetCost()) {
            activeSchedule_.WriteSchedule(schedule);
            CleanupDatastructures();
            return ReturnStatus::OSP_SUCCESS;
        } else {
            CleanupDatastructures();
            return ReturnStatus::OSP_BEST_FOUND;
        }
    }

    virtual ReturnStatus ImproveScheduleWithTimeLimit(BspSchedule<GraphT> &schedule) override {
        computeWithTimeLimit_ = true;
        return ImproveSchedule(schedule);
    }

    virtual void SetTimeQualityParameter(const double timeQuality) { this->parameters_.timeQuality_ = timeQuality; }

    virtual void SetSuperstepRemoveStrengthParameter(const double superstepRemoveStrength) {
        this->parameters_.superstepRemoveStrength_ = superstepRemoveStrength;
    }
};

}    // namespace osp
} // namespace npu::tile_fwk

#include "kl_improver.tpp"
#endif // OSP_KL_IMPROVER_HPP
