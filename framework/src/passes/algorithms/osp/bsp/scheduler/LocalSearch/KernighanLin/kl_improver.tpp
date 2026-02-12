/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OSP_KL_IMPROVER_TPP
#define OSP_KL_IMPROVER_TPP

namespace npu::tile_fwk {
namespace osp {

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
bool KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::ScatterNodesSuperstep(unsigned step, ThreadSearchContext &threadData) {
    bool abort = false;

    for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
        const std::vector<VertexType> stepProcNodeVec(
            activeSchedule_.GetSetSchedule().GetProcessorStepVertices()[step][proc].begin(),
            activeSchedule_.GetSetSchedule().GetProcessorStepVertices()[step][proc].end());
        for (const auto &node : stepProcNodeVec) {
            threadData.rewardPenaltyStrat_.InitRewardPenalty(
                static_cast<double>(threadData.activeScheduleData_.currentViolations_.size()) + 1.0);
            ComputeNodeAffinities(node, threadData.localAffinityTable_, threadData);
            KlMove bestMove = ComputeBestMove<false>(node, threadData.localAffinityTable_, threadData);

            if (bestMove.gain_ <= std::numeric_limits<double>::lowest()) {
                abort = true;
                break;
            }

            ApplyMove(bestMove, threadData);
            if (threadData.activeScheduleData_.currentViolations_.size() > parameters_.abortScatterNodesViolationThreshold_) {
                abort = true;
                break;
            }

            threadData.affinityTable_.Insert(node);
            if (threadData.activeScheduleData_.newViolations_.size() > 0) {
                for (const auto &vertexEdgePair : threadData.activeScheduleData_.newViolations_) {
                    const auto &vertex = vertexEdgePair.first;
                    threadData.affinityTable_.Insert(vertex);
                }
            }
        }

        if (abort) {
            break;
        }
    }

    if (abort) {
        activeSchedule_.RevertToBestSchedule(
            0, 0, commCostF_, threadData.activeScheduleData_, threadData.startStep_, threadData.endStep_);
        threadData.affinityTable_.ResetNodeSelection();
        return false;
    }
    return true;
}


template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::SynchronizeActiveSchedule(const unsigned numThreads) {
    if (numThreads == 1) {    // single thread case
        activeSchedule_.SetCost(threadDataVec_[0].activeScheduleData_.cost_);
        activeSchedule_.GetVectorSchedule().NumberOfSupersteps() = threadDataVec_[0].NumSteps();
        return;
    }

    unsigned writeCursor = threadDataVec_[0].endStep_ + 1;
    for (unsigned i = 1; i < numThreads; ++i) {
        auto &thread = threadDataVec_[i];
        if (thread.startStep_ <= thread.endStep_) {
            for (unsigned j = thread.startStep_; j <= thread.endStep_; ++j) {
                if (j != writeCursor) {
                    activeSchedule_.SwapSteps(j, writeCursor);
                }
                writeCursor++;
            }
        }
    }
    activeSchedule_.GetVectorSchedule().NumberOfSupersteps() = writeCursor;
    const CostT newCost = commCostF_.ComputeScheduleCost();
    activeSchedule_.SetCost(newCost);
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::SetParameters(VertexIdxT<GraphT> numNodes) {
    const unsigned logNumNodes = (numNodes > 1) ? static_cast<unsigned>(std::log(numNodes)) : 1;

    // Total number of outer iterations. Proportional to sqrt N.
    parameters_.maxOuterIterations_
        = static_cast<unsigned>(std::sqrt(numNodes) * (parameters_.timeQuality_ * 10.0) / parameters_.numParallelLoops_);

    // Number of times to reset the search for violations before giving up.
    parameters_.maxNoVioaltionsRemovedBacktrackReset_ = parameters_.timeQuality_ < 0.75  ? 1
                                                        : parameters_.timeQuality_ < 1.0 ? 2
                                                                                         : 3;

    // Parameters for the superstep removal heuristic.
    parameters_.maxNoVioaltionsRemovedBacktrackForRemoveStepReset_
        = 3 + static_cast<unsigned>(parameters_.superstepRemoveStrength_ * 7);
    parameters_.nodeMaxStepSelectionEpochs_ = parameters_.superstepRemoveStrength_ < 0.75  ? 1
                                              : parameters_.superstepRemoveStrength_ < 1.0 ? 2
                                                                                           : 3;
    parameters_.removeStepEpocs_ = static_cast<unsigned>(parameters_.superstepRemoveStrength_ * 4.0);

    parameters_.minInnerIterReset_ = static_cast<unsigned>(logNumNodes + logNumNodes * (1.0 + parameters_.timeQuality_));

    if (parameters_.removeStepEpocs_ > 0) {
        parameters_.tryRemoveStepAfterNumOuterIterations_ = parameters_.maxOuterIterations_ / parameters_.removeStepEpocs_;
    } else {
        // Effectively disable superstep removal if remove_step_epocs is 0.
        parameters_.tryRemoveStepAfterNumOuterIterations_ = parameters_.maxOuterIterations_ + 1;
    }

    unsigned i = 0;
    for (auto &thread : threadDataVec_) {
        thread.threadId_ = i++;
        // The number of nodes to consider in each inner iteration. Proportional to log(N).
        thread.selectionStrategy_.selectionThreshold_
            = static_cast<std::size_t>(std::ceil(parameters_.timeQuality_ * 10 * logNumNodes + logNumNodes));
    }
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::UpdateNodeWorkAffinity(
    NodeSelectionContainerT &nodes,
    KlMove move,
    const PreMoveWorkData<VertexWorkWeightT> &prevWorkData,
    std::map<VertexType, KlGainUpdateInfo> &recomputeMaxGain) {
    const size_t activeCount = nodes.size();

    for (size_t i = 0; i < activeCount; ++i) {
        const VertexType node = nodes.GetSelectedNodes()[i];

        KlGainUpdateInfo updateInfo = UpdateNodeWorkAffinityAfterMove(node, move, prevWorkData, nodes.At(node));
        if (updateInfo.updateFromStep_ || updateInfo.updateToStep_) {
            recomputeMaxGain[node] = updateInfo;
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::UpdateMaxGain(
    KlMove move, std::map<VertexType, KlGainUpdateInfo> &recomputeMaxGain, ThreadSearchContext &threadData) {
    for (auto &pair : recomputeMaxGain) {
        if (pair.second.fullUpdate_) {
            RecomputeNodeMaxGain(pair.first, threadData.affinityTable_, threadData);
        } else {
            if (pair.second.updateEntireFromStep_) {
                UpdateBestMove(pair.first, move.fromStep_, threadData.affinityTable_, threadData);
            } else if (pair.second.updateFromStep_ && IsCompatible(pair.first, move.fromProc_)) {
                UpdateBestMove(pair.first, move.fromStep_, move.fromProc_, threadData.affinityTable_, threadData);
            }

            if (move.fromStep_ != move.toStep_ || not pair.second.updateEntireFromStep_) {
                if (pair.second.updateEntireToStep_) {
                    UpdateBestMove(pair.first, move.toStep_, threadData.affinityTable_, threadData);
                } else if (pair.second.updateToStep_ && IsCompatible(pair.first, move.toProc_)) {
                    UpdateBestMove(pair.first, move.toStep_, move.toProc_, threadData.affinityTable_, threadData);
                }
            }
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::ComputeWorkAffinity(
    VertexType node, std::vector<std::vector<CostT>> &affinityTableNode, ThreadSearchContext &threadData) {
    const unsigned nodeStep = activeSchedule_.AssignedSuperstep(node);
    const VertexWorkWeightT vertexWeight = graph_->VertexWorkWeight(node);

    unsigned step = (nodeStep > windowSize) ? (nodeStep - windowSize) : 0;
    for (unsigned idx = threadData.StartIdx(nodeStep); idx < threadData.EndIdx(nodeStep); ++idx, ++step) {
        if (idx == windowSize) {
            continue;
        }

        const CostT maxWorkForStep = static_cast<CostT>(activeSchedule_.GetStepMaxWork(step));

        for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
            const VertexWorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(step, proc);
            const CostT workDiff = static_cast<CostT>(newWeight) - maxWorkForStep;
            affinityTableNode[proc][idx] = std::max(0.0, workDiff);
        }
    }

    const unsigned nodeProc = activeSchedule_.AssignedProcessor(node);
    const VertexWorkWeightT maxWorkForStep = activeSchedule_.GetStepMaxWork(nodeStep);
    const bool isSoleMaxProcessor = (activeSchedule_.GetStepMaxWorkProcessorCount()[nodeStep] == 1)
                                    && (maxWorkForStep == activeSchedule_.GetStepProcessorWork(nodeStep, nodeProc));

    const CostT nodeProcAffinity
        = isSoleMaxProcessor ? std::min(vertexWeight, maxWorkForStep - activeSchedule_.GetStepSecondMaxWork(nodeStep)) : 0.0;
    affinityTableNode[nodeProc][windowSize] = nodeProcAffinity;

    for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
        if (proc == nodeProc) {
            continue;
        }

        const VertexWorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, proc);
        affinityTableNode[proc][windowSize] = ComputeSameStepAffinity(maxWorkForStep, newWeight, nodeProcAffinity);
    }
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::ProcessWorkUpdateStep(VertexType node, unsigned nodeStep, unsigned nodeProc, VertexWorkWeightT vertexWeight, unsigned moveStep, 
    unsigned moveProc, VertexWorkWeightT moveCorrectionNodeWeight, const VertexWorkWeightT prevMoveStepMaxWork, const VertexWorkWeightT prevMoveStepSecondMaxWork, 
    unsigned prevMoveStepMaxWorkProcessorCount, bool &updateStep, bool &updateEntireStep, bool &fullUpdate, std::vector<std::vector<CostT>> &affinityTableNode) {
    const unsigned lowerBound = moveStep > windowSize ? moveStep - windowSize : 0;
    if (lowerBound <= nodeStep && nodeStep <= moveStep + windowSize) {
        updateStep = true;
        if (nodeStep == moveStep) {
            const VertexWorkWeightT newMaxWeight = activeSchedule_.GetStepMaxWork(moveStep);
            const VertexWorkWeightT newSecondMaxWeight = activeSchedule_.GetStepSecondMaxWork(moveStep);
            const VertexWorkWeightT newStepProcWork = activeSchedule_.GetStepProcessorWork(nodeStep, nodeProc);

            const VertexWorkWeightT prevStepProcWork = (nodeProc == moveProc) ? newStepProcWork + moveCorrectionNodeWeight
                                                                              : newStepProcWork;
            const bool prevIsSoleMaxProcessor = (prevMoveStepMaxWorkProcessorCount == 1)
                                                && (prevMoveStepMaxWork == prevStepProcWork);
            const CostT prevNodeProcAffinity
                = prevIsSoleMaxProcessor ? std::min(vertexWeight, prevMoveStepMaxWork - prevMoveStepSecondMaxWork) : 0.0;

            const bool newIsSoleMaxProcessor = (activeSchedule_.GetStepMaxWorkProcessorCount()[nodeStep] == 1)
                                               && (newMaxWeight == newStepProcWork);
            const CostT newNodeProcAffinity = newIsSoleMaxProcessor ? std::min(vertexWeight, newMaxWeight - newSecondMaxWeight)
                                                                    : 0.0;

            const CostT diff = newNodeProcAffinity - prevNodeProcAffinity;
            const bool updateNodeProcAffinity = std::abs(diff) > epsilon_;
            if (updateNodeProcAffinity) {
                fullUpdate = true;
                affinityTableNode[nodeProc][windowSize] += diff;
            }

            if ((prevMoveStepMaxWork != newMaxWeight) || updateNodeProcAffinity) {
                updateEntireStep = true;

                for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
                    if ((proc == nodeProc) || (proc == moveProc)) {
                        continue;
                    }

                    const VertexWorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, proc);
                    const CostT prevOtherAffinity = ComputeSameStepAffinity(prevMoveStepMaxWork, newWeight, prevNodeProcAffinity);
                    const CostT otherAffinity = ComputeSameStepAffinity(newMaxWeight, newWeight, newNodeProcAffinity);

                    affinityTableNode[proc][windowSize] += (otherAffinity - prevOtherAffinity);
                }
            }

            if (nodeProc != moveProc && IsCompatible(node, moveProc)) {
                const VertexWorkWeightT prevNewWeight
                    = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, moveProc) + moveCorrectionNodeWeight;
                const CostT prevOtherAffinity = ComputeSameStepAffinity(prevMoveStepMaxWork, prevNewWeight, prevNodeProcAffinity);
                const VertexWorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, moveProc);
                const CostT otherAffinity = ComputeSameStepAffinity(newMaxWeight, newWeight, newNodeProcAffinity);

                affinityTableNode[moveProc][windowSize] += (otherAffinity - prevOtherAffinity);
            }

        } else {
            const VertexWorkWeightT newMaxWeight = activeSchedule_.GetStepMaxWork(moveStep);
            const unsigned idx = RelStepIdx(nodeStep, moveStep);
            if (prevMoveStepMaxWork != newMaxWeight) {
                updateEntireStep = true;

                // update moving to all procs with special for moveProc
                for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
                    const VertexWorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(moveStep, proc);
                    if (proc != moveProc) {
                        const CostT prevAffinity = prevMoveStepMaxWork < newWeight
                                                       ? static_cast<CostT>(newWeight) - static_cast<CostT>(prevMoveStepMaxWork)
                                                       : 0.0;
                        const CostT newAffinity
                            = newMaxWeight < newWeight ? static_cast<CostT>(newWeight) - static_cast<CostT>(newMaxWeight) : 0.0;
                        affinityTableNode[proc][idx] += newAffinity - prevAffinity;

                    } else {
                        const VertexWorkWeightT prevNewWeight
                            = vertexWeight + activeSchedule_.GetStepProcessorWork(moveStep, proc) + moveCorrectionNodeWeight;
                        const CostT prevAffinity = prevMoveStepMaxWork < prevNewWeight
                                                       ? static_cast<CostT>(prevNewWeight) - static_cast<CostT>(prevMoveStepMaxWork)
                                                       : 0.0;

                        const CostT newAffinity
                            = newMaxWeight < newWeight ? static_cast<CostT>(newWeight) - static_cast<CostT>(newMaxWeight) : 0.0;
                        affinityTableNode[proc][idx] += newAffinity - prevAffinity;
                    }
                }
            } else {
                // update only moveProc
                if (IsCompatible(node, moveProc)) {
                    const VertexWorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(moveStep, moveProc);
                    const VertexWorkWeightT prevNewWeight = newWeight + moveCorrectionNodeWeight;
                    const CostT prevAffinity = prevMoveStepMaxWork < prevNewWeight
                                                   ? static_cast<CostT>(prevNewWeight) - static_cast<CostT>(prevMoveStepMaxWork)
                                                   : 0.0;

                    const CostT newAffinity
                        = newMaxWeight < newWeight ? static_cast<CostT>(newWeight) - static_cast<CostT>(newMaxWeight) : 0.0;
                    affinityTableNode[moveProc][idx] += newAffinity - prevAffinity;
                }
            }
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
bool KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::SelectNodesCheckRemoveSuperstep(
    unsigned &stepToRemove, ThreadSearchContext &threadData) {
    if (threadData.stepSelectionEpochCounter_ >= parameters_.nodeMaxStepSelectionEpochs_ || threadData.NumSteps() < 3) {
        return false;
    }

    for (stepToRemove = threadData.stepSelectionCounter_; stepToRemove <= threadData.endStep_; stepToRemove++) {
        if (CheckRemoveSuperstep(stepToRemove)) {
            if (ScatterNodesSuperstep(stepToRemove, threadData)) {
                threadData.stepSelectionCounter_ = stepToRemove + 1;

                if (threadData.stepSelectionCounter_ > threadData.endStep_) {
                    threadData.stepSelectionCounter_ = threadData.startStep_;
                    threadData.stepSelectionEpochCounter_++;
                }
                return true;
            }
        }
    }

    threadData.stepSelectionEpochCounter_++;
    threadData.stepSelectionCounter_ = threadData.startStep_;
    return false;
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
bool KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::CheckRemoveSuperstep(unsigned step) {
    if (activeSchedule_.NumSteps() < 2) {
        return false;
    }

    if (activeSchedule_.GetStepMaxWork(step) < instance_->SynchronisationCosts()) {
        return true;
    }

    return false;
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::ResetInnerSearchStructures(
    ThreadSearchContext &threadData) const {
    threadData.unlockEdgeBacktrackCounter_ = threadData.unlockEdgeBacktrackCounterReset_;
    threadData.maxInnerIterations_ = parameters_.maxInnerIterationsReset_;
    threadData.maxNoVioaltionsRemovedBacktrack_ = parameters_.maxNoVioaltionsRemovedBacktrackReset_;
    threadData.averageGain_ = 0.0;
    threadData.affinityTable_.ResetNodeSelection();
    threadData.maxGainHeap_.Clear();
    threadData.lockManager_.Clear();
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
bool KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::IsLocalSearchBlocked(
    ThreadSearchContext &threadData) {
    for (const auto &pair : threadData.activeScheduleData_.newViolations_) {
        if (threadData.lockManager_.IsLocked(pair.first)) {
            return true;
        }
    }
    return false;
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::InitializeDatastructures(
    BspSchedule<GraphT> &schedule) {
    inputSchedule_ = &schedule;
    instance_ = &schedule.GetInstance();
    graph_ = &instance_->GetComputationalDag();

    activeSchedule_.Initialize(schedule);

    procRange_.Initialize(*instance_);
    commCostF_.Initialize(activeSchedule_, procRange_);
    const CostT initialCost = commCostF_.ComputeScheduleCost();
    activeSchedule_.SetCost(initialCost);

    for (auto &tData : threadDataVec_) {
        tData.affinityTable_.Initialize(activeSchedule_, tData.selectionStrategy_.selectionThreshold_);
        tData.lockManager_.Initialize(graph_->NumVertices());
        tData.rewardPenaltyStrat_.Initialize(
            activeSchedule_, commCostF_.GetMaxCommWeightMultiplied(), activeSchedule_.GetMaxWorkWeight());
        tData.selectionStrategy_.Initialize(activeSchedule_, gen_, tData.startStep_, tData.endStep_);

        tData.localAffinityTable_.resize(instance_->NumberOfProcessors());
        for (unsigned i = 0; i < instance_->NumberOfProcessors(); ++i) {
            tData.localAffinityTable_[i].resize(windowRange_);
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::UpdateAvgGain(const CostT gain,
                                                                                                const unsigned numIter,
                                                                                                double &averageGain) {
    averageGain = static_cast<double>((averageGain * numIter + gain)) / (numIter + 1.0);
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::InsertGainHeap(ThreadSearchContext &threadData) {
    const size_t activeCount = threadData.affinityTable_.size();

    for (size_t i = 0; i < activeCount; ++i) {
        const VertexType node = threadData.affinityTable_.GetSelectedNodes()[i];
        ComputeNodeAffinities(node, threadData.affinityTable_.At(node), threadData);
        const auto bestMove = ComputeBestMove<true>(node, threadData.affinityTable_[node], threadData);
        threadData.maxGainHeap_.Push(node, bestMove);
    }
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::InsertNewNodesGainHeap(
    std::vector<VertexType> &newNodes, NodeSelectionContainerT &nodes, ThreadSearchContext &threadData) {
    for (const auto &node : newNodes) {
        nodes.Insert(node);
        ComputeNodeAffinities(node, threadData.affinityTable_.At(node), threadData);
        const auto bestMove = ComputeBestMove<true>(node, threadData.affinityTable_[node], threadData);
        threadData.maxGainHeap_.Push(node, bestMove);
    }
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::CleanupDatastructures() {
    threadDataVec_.clear();
    activeSchedule_.Clear();
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::UpdateBestMove(
    VertexType node, unsigned step, unsigned proc, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData) {
    const unsigned nodeProc = activeSchedule_.AssignedProcessor(node);
    const unsigned nodeStep = activeSchedule_.AssignedSuperstep(node);

    if ((nodeProc == proc) && (nodeStep == step)) {
        return;
    }

    KlMove nodeMove = threadData.maxGainHeap_.GetValue(node);
    CostT maxGain = nodeMove.gain_;

    unsigned maxProc = nodeMove.toProc_;
    unsigned maxStep = nodeMove.toStep_;

    if ((maxStep == step) && (maxProc == proc)) {
        RecomputeNodeMaxGain(node, affinityTable, threadData);
    } else {
        const unsigned idx = RelStepIdx(nodeStep, step);
        const CostT gain = affinityTable[node][nodeProc][windowSize] - affinityTable[node][proc][idx];
        if (gain > maxGain) {
            maxGain = gain;
            maxProc = proc;
            maxStep = step;
        }

        const CostT diff = maxGain - nodeMove.gain_;
        if ((std::abs(diff) > epsilon_) || (maxProc != nodeMove.toProc_) || (maxStep != nodeMove.toStep_)) {
            nodeMove.gain_ = maxGain;
            nodeMove.toStep_ = maxStep;
            nodeMove.toProc_ = maxProc;
            threadData.maxGainHeap_.Update(node, nodeMove);
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, windowSize, CostT>::UpdateBestMove(
    VertexType node, unsigned step, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData) {
    const unsigned nodeProc = activeSchedule_.AssignedProcessor(node);
    const unsigned nodeStep = activeSchedule_.AssignedSuperstep(node);

    KlMove nodeMove = threadData.maxGainHeap_.GetValue(node);
    CostT maxGain = nodeMove.gain_;

    unsigned maxProc = nodeMove.toProc_;
    unsigned maxStep = nodeMove.toStep_;

    if (maxStep == step) {
        RecomputeNodeMaxGain(node, affinityTable, threadData);
    } else {
        if (nodeStep != step) {
            const unsigned idx = RelStepIdx(nodeStep, step);
            for (const unsigned p : procRange_.CompatibleProcessorsVertex(node)) {
                const CostT gain = affinityTable[node][nodeProc][windowSize] - affinityTable[node][p][idx];
                if (gain > maxGain) {
                    maxGain = gain;
                    maxProc = p;
                    maxStep = step;
                }
            }
        } else {
            for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
                if (proc == nodeProc) {
                    continue;
                }

                const CostT gain = affinityTable[node][nodeProc][windowSize] - affinityTable[node][proc][windowSize];
                if (gain > maxGain) {
                    maxGain = gain;
                    maxProc = proc;
                    maxStep = step;
                }
            }
        }

        const CostT diff = maxGain - nodeMove.gain_;
        if ((std::abs(diff) > epsilon_) || (maxProc != nodeMove.toProc_) || (maxStep != nodeMove.toStep_)) {
            nodeMove.gain_ = maxGain;
            nodeMove.toProc_ = maxProc;
            nodeMove.toStep_ = maxStep;
            threadData.maxGainHeap_.Update(node, nodeMove);
        }
    }
}

}    // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_KL_IMPROVER_TPP
