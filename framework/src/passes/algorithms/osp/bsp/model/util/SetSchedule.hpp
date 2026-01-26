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
 * \file SetSchedule.hpp
 * \brief
 */

#ifndef OSP_SETSCHEDULE_HPP
#define OSP_SETSCHEDULE_HPP

#include <unordered_set>
#include <vector>

#include "passes/algorithms/osp/bsp/model/IBspSchedule.hpp"
#include "passes/algorithms/osp/concepts/computational_dag_concept.hpp"

namespace npu::tile_fwk {
namespace osp {

/**
 * @class SetSchedule
 *
 * The SetSchedule stores the assignment of nodes to processors and supersteps in a vector of unordered sets.
 * Each element in the vector represents a superstep and contains a set of nodes for each processor.
 * This class is useful for cases where all nodes of a superstep/processor pair need to be enumerated often.
 *
 * @tparam GraphT The type of the computational DAG.
 */
template <typename GraphT>
class SetSchedule {
    static_assert(isComputationalDagV<GraphT>, "SetSchedule can only be used with computational DAGs.");

  private:
    using VertexIdx = VertexIdxT<GraphT>;

    const BspInstance<GraphT> *instance_ = nullptr;

    unsigned numberOfSupersteps_ = 0;
    std::vector<std::vector<std::unordered_set<VertexIdx>>> stepProcessorVertices_;

  public:
    SetSchedule() = default;

    /**
     * @brief Constructs a SetSchedule from another IBspSchedule.
     * @param schedule The source schedule to copy from.
     */
    SetSchedule(const IBspSchedule<GraphT> &schedule)
        : instance_(&schedule.GetInstance()), numberOfSupersteps_(schedule.NumberOfSupersteps()) {
        stepProcessorVertices_.resize(schedule.NumberOfSupersteps(),
                                      std::vector<std::unordered_set<VertexIdx>>(schedule.GetInstance().NumberOfProcessors()));

        for (const auto v : schedule.GetInstance().Vertices()) {
            const unsigned step = schedule.AssignedSuperstep(v);
            const unsigned proc = schedule.AssignedProcessor(v);

            if (step < numberOfSupersteps_ && proc < instance_->NumberOfProcessors()) {
                stepProcessorVertices_[step][proc].insert(v);
            }
        }
    }

    ~SetSchedule() = default;

    /**
     * @brief Clears the schedule assignments and resets the number of supersteps to 0.
     */
    void Clear() {
        stepProcessorVertices_.clear();
        numberOfSupersteps_ = 0;
    }

    /**
     * @brief Get the BSP instance associated with this schedule.
     *
     * @return The BSP instance.
     */
    [[nodiscard]] const BspInstance<GraphT> &GetInstance() const { return *instance_; }

    [[nodiscard]] unsigned NumberOfSupersteps() const { return numberOfSupersteps_; }

    /**
     * @brief Merges a range of supersteps into a single superstep (the startStep).
     * @param startStep The start of the range (inclusive).
     * @param endStep The end of the range (inclusive).
     */
    void MergeSupersteps(unsigned startStep, unsigned endStep) {
        if (startStep >= endStep || endStep >= numberOfSupersteps_) {
            return;
        }

        unsigned step = startStep + 1;
        // Merge contents of [startStep+1, endStep] into startStep
        for (; step <= endStep; step++) {
            for (unsigned proc = 0; proc < GetInstance().NumberOfProcessors(); proc++) {
                stepProcessorVertices_[startStep][proc].merge(stepProcessorVertices_[step][proc]);
            }
        }

        // Shift remaining supersteps down
        // The original logic was: step is now endStep + 1
        unsigned shift = endStep - startStep;
        for (; step < numberOfSupersteps_; step++) {
            for (unsigned proc = 0; proc < GetInstance().NumberOfProcessors(); proc++) {
                stepProcessorVertices_[step - shift][proc] = std::move(stepProcessorVertices_[step][proc]);
            }
        }

        numberOfSupersteps_ -= shift;
        stepProcessorVertices_.resize(numberOfSupersteps_);
    }

    /**
     * @brief Get the internal node assignment structure.
     * @return Reference to the vector of vectors of unordered sets of vertices.
     */
    [[nodiscard]] const std::vector<std::vector<std::unordered_set<VertexIdx>>> &GetProcessorStepVertices() const {
        return stepProcessorVertices_;
    }

    /**
     * @brief Get the internal node assignment structure (mutable).
     * @return Reference to the vector of vectors of unordered sets of vertices.
     */
    [[nodiscard]] std::vector<std::vector<std::unordered_set<VertexIdx>>> &GetProcessorStepVertices() {
        return stepProcessorVertices_;
    }
};

}    // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_SETSCHEDULE_HPP
