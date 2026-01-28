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
 * \file GreedyChildren.hpp
 * \brief
 */

#ifndef OSP_GREEDYCHILDREN_HPP
#define OSP_GREEDYCHILDREN_HPP

#include <algorithm>
#include <iterator>
#include <unordered_set>
#include <vector>

#include "passes/algorithms/osp/bsp/scheduler/Scheduler.hpp"
#include "passes/algorithms/osp/graph_algorithms/directed_graph_util.hpp"

namespace npu::tile_fwk {
namespace osp {

template <typename GraphT>
class GreedyChildren : public Scheduler<GraphT> {
  private:
    bool ensureEnoughSources_;

  public:
    GreedyChildren(bool ensureEnoughSources = true) : Scheduler<GraphT>(), ensureEnoughSources_(ensureEnoughSources) {};

    ReturnStatus ComputeSchedule(BspSchedule<GraphT> &sched) override {
        using VertexType = VertexIdxT<GraphT>;
        const auto &instance = sched.GetInstance();

        for (const auto &v : instance.GetComputationalDag().Vertices()) {
            sched.SetAssignedProcessor(v, std::numeric_limits<unsigned>::max());
        }

        const auto &graph = instance.GetComputationalDag();

        unsigned superstepCounter = 0;

        std::vector<VertexType> predecessorsCount(instance.NumberOfVertices(), 0);
        std::multiset<std::pair<unsigned, VertexType>, std::greater<>> next;
        for (const VertexType &i : SourceVerticesView(graph)) {
            next.emplace(graph.OutDegree(i), i);
        }

        while (!next.empty()) {
            std::unordered_set<VertexType> nodesAssignedThisSuperstep;
            std::vector<VWorkwT<GraphT>> processorWeights(instance.NumberOfProcessors(), 0);

            bool fewSources = next.size() < instance.NumberOfProcessors() ? true : false;
            bool nodeAdded = true;
            while (!next.empty() && nodeAdded) {
                nodeAdded = false;
                for (auto iter = next.begin(); iter != next.cend(); iter++) {
                    const auto &node = iter->second;
                    bool processorSet = false;
                    bool failedToAllocate = false;
                    unsigned processorToBeAllocated = 0;

                    for (const auto &par : graph.Parents(node)) {
                        if (nodesAssignedThisSuperstep.count(par)) {
                            if (!processorSet) {
                                const unsigned parProc = sched.AssignedProcessor(par);
                                if (!instance.IsCompatible(node, parProc)) {
                                    failedToAllocate = true;
                                    break;
                                }
                                processorSet = true;
                                processorToBeAllocated = parProc;
                            } else if (sched.AssignedProcessor(par) != processorToBeAllocated) {
                                failedToAllocate = true;
                                break;
                            }
                        }
                    }

                    if (failedToAllocate) {
                        continue;
                    }

                    sched.SetAssignedSuperstep(node, superstepCounter);
                    if (processorSet) {
                        sched.SetAssignedProcessor(node, processorToBeAllocated);
                    } else {
                        VWorkwT<GraphT> minWeight = std::numeric_limits<VWorkwT<GraphT>>::max();
                        unsigned bestProc = std::numeric_limits<unsigned>::max();
                        for (unsigned p = 0; p < instance.NumberOfProcessors(); ++p) {
                            if (instance.IsCompatible(node, p)) {
                                if (processorWeights[p] < minWeight) {
                                    minWeight = processorWeights[p];
                                    bestProc = p;
                                }
                            }
                        }
                        sched.SetAssignedProcessor(node, bestProc);
                    }

                    nodesAssignedThisSuperstep.emplace(node);
                    processorWeights[sched.AssignedProcessor(node)] += graph.VertexWorkWeight(node);
                    std::vector<VertexType> newNodes;
                    for (const auto &chld : graph.Children(node)) {
                        predecessorsCount[chld]++;
                        if (predecessorsCount[chld] == graph.InDegree(chld)) {
                            newNodes.emplace_back(chld);
                        }
                    }
                    next.erase(iter);
                    for (const auto &vrt : newNodes) {
                        next.emplace(graph.OutDegree(vrt), vrt);
                    }
                    nodeAdded = true;
                    break;
                }
                if (ensureEnoughSources_ && fewSources && next.size() >= instance.NumberOfProcessors()) {
                    break;
                }
            }

            superstepCounter++;
        }

        return ReturnStatus::OSP_SUCCESS;
    }
};

}    // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_GREEDYCHILDREN_HPP
