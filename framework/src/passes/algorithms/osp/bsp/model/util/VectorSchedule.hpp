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
 * \file VectorSchedule.hpp
 * \brief
 */

#ifndef OSP_VECTORSCHEDULE_HPP
#define OSP_VECTORSCHEDULE_HPP

#include <vector>

#include "passes/algorithms/osp/bsp/model/IBspSchedule.hpp"
#include "passes/algorithms/osp/concepts/computational_dag_concept.hpp"

namespace npu::tile_fwk {
namespace osp {

template <typename GraphT>
class VectorSchedule : public IBspSchedule<GraphT> {
    static_assert(isComputationalDagV<GraphT>, "BspSchedule can only be used with computational DAGs.");

  private:
    const BspInstance<GraphT> *instance_;

  public:
    unsigned int numberOfSupersteps_;

    std::vector<unsigned> nodeToProcessorAssignment_;
    std::vector<unsigned> nodeToSuperstepAssignment_;

    /**
     * @brief Default constructor for VectorSchedule.
     */
    VectorSchedule() : instance_(nullptr), numberOfSupersteps_(0) {}

    VectorSchedule(const BspInstance<GraphT> &inst) : instance_(&inst), numberOfSupersteps_(0) {
        nodeToProcessorAssignment_ = std::vector<unsigned>(inst.NumberOfVertices(), instance_->NumberOfProcessors());
        nodeToSuperstepAssignment_ = std::vector<unsigned>(inst.NumberOfVertices(), 0);
    }

    VectorSchedule(const IBspSchedule<GraphT> &schedule)
        : instance_(&schedule.GetInstance()), numberOfSupersteps_(schedule.NumberOfSupersteps()) {
        nodeToProcessorAssignment_
            = std::vector<unsigned>(schedule.GetInstance().NumberOfVertices(), instance_->NumberOfProcessors());
        nodeToSuperstepAssignment_
            = std::vector<unsigned>(schedule.GetInstance().NumberOfVertices(), schedule.NumberOfSupersteps());

        for (VertexIdxT<GraphT> i = 0; i < schedule.GetInstance().NumberOfVertices(); i++) {
            nodeToProcessorAssignment_[i] = schedule.AssignedProcessor(i);
            nodeToSuperstepAssignment_[i] = schedule.AssignedSuperstep(i);
        }
    }

    VectorSchedule(const VectorSchedule &other)
        : instance_(other.instance_),
          numberOfSupersteps_(other.numberOfSupersteps_),
          nodeToProcessorAssignment_(other.nodeToProcessorAssignment_),
          nodeToSuperstepAssignment_(other.nodeToSuperstepAssignment_) {}

    VectorSchedule &operator=(const IBspSchedule<GraphT> &other) {
        if (this != &other) {
            instance_ = &other.GetInstance();
            numberOfSupersteps_ = other.NumberOfSupersteps();
            nodeToProcessorAssignment_ = std::vector<unsigned>(instance_->NumberOfVertices(), instance_->NumberOfProcessors());
            nodeToSuperstepAssignment_ = std::vector<unsigned>(instance_->NumberOfVertices(), numberOfSupersteps_);

            for (VertexIdxT<GraphT> i = 0; i < instance_->NumberOfVertices(); i++) {
                nodeToProcessorAssignment_[i] = other.AssignedProcessor(i);
                nodeToSuperstepAssignment_[i] = other.AssignedSuperstep(i);
            }
        }
        return *this;
    }

    VectorSchedule &operator=(const VectorSchedule &other) {
        if (this != &other) {
            instance_ = other.instance_;
            numberOfSupersteps_ = other.numberOfSupersteps_;
            nodeToProcessorAssignment_ = other.nodeToProcessorAssignment_;
            nodeToSuperstepAssignment_ = other.nodeToSuperstepAssignment_;
        }
        return *this;
    }

    VectorSchedule(VectorSchedule &&other) noexcept
        : instance_(other.instance_),
          numberOfSupersteps_(other.numberOfSupersteps_),
          nodeToProcessorAssignment_(std::move(other.nodeToProcessorAssignment_)),
          nodeToSuperstepAssignment_(std::move(other.nodeToSuperstepAssignment_)) {}

    virtual ~VectorSchedule() = default;

    void Clear() {
        nodeToProcessorAssignment_.clear();
        nodeToSuperstepAssignment_.clear();
        numberOfSupersteps_ = 0;
    }

    const BspInstance<GraphT> &GetInstance() const override { return *instance_; }

    void SetAssignedSuperstep(VertexIdxT<GraphT> vertex, unsigned superstep) override {
        nodeToSuperstepAssignment_[vertex] = superstep;
    };

    void SetAssignedProcessor(VertexIdxT<GraphT> vertex, unsigned processor) override {
        nodeToProcessorAssignment_[vertex] = processor;
    };

    unsigned NumberOfSupersteps() const override { return numberOfSupersteps_; }

    unsigned AssignedSuperstep(VertexIdxT<GraphT> vertex) const override { return nodeToSuperstepAssignment_[vertex]; }

    unsigned AssignedProcessor(VertexIdxT<GraphT> vertex) const override { return nodeToProcessorAssignment_[vertex]; }

    void MergeSupersteps(unsigned startStep, unsigned endStep) {
        numberOfSupersteps_ = 0;

        for (const auto &vertex : GetInstance().Vertices()) {
            if (nodeToSuperstepAssignment_[vertex] > startStep && nodeToSuperstepAssignment_[vertex] <= endStep) {
                nodeToSuperstepAssignment_[vertex] = startStep;
            } else if (nodeToSuperstepAssignment_[vertex] > endStep) {
                nodeToSuperstepAssignment_[vertex] -= endStep - startStep;
            }

            if (nodeToSuperstepAssignment_[vertex] >= numberOfSupersteps_) {
                numberOfSupersteps_ = nodeToSuperstepAssignment_[vertex] + 1;
            }
        }
    }

    void InsertSupersteps(const unsigned stepBefore, const unsigned numNewSteps) {
        numberOfSupersteps_ += numNewSteps;

        for (const auto &vertex : GetInstance().Vertices()) {
            if (nodeToSuperstepAssignment_[vertex] > stepBefore) {
                nodeToSuperstepAssignment_[vertex] += numNewSteps;
            }
        }
    }
};

}    // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_VECTORSCHEDULE_HPP
