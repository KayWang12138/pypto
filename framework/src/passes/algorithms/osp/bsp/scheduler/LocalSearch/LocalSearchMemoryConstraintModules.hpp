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
 * \file LocalSearchMemoryConstraintModules.hpp
 * \brief
 */

#ifndef OSP_LOCALSEARCHMEMORYCONSTRAINTMODULES_HPP
#define OSP_LOCALSEARCHMEMORYCONSTRAINTMODULES_HPP

#include "passes/algorithms/osp/bsp/model/BspSchedule.hpp"
#include "passes/algorithms/osp/bsp/model/util/SetSchedule.hpp"
#include "passes/algorithms/osp/bsp/model/util/VectorSchedule.hpp"
#include "passes/algorithms/osp/graph_algorithms/directed_graph_util.hpp"

namespace npu::tile_fwk {
namespace osp {

/**
 * @brief A trait to check if a type is a memory constraint.
 *
 * This trait checks if a type has the required methods for a memory constraint.
 *
 */
template <typename T, typename = void>
struct IsLocalSearchMemoryConstraint : std::false_type {};

template <typename T>
struct IsLocalSearchMemoryConstraint<
    T,
    std::void_t<decltype(std::declval<T>().Initialize(std::declval<SetSchedule<typename T::GraphImplT>>(),
                                                      std::declval<VectorSchedule<typename T::GraphImplT>>())),
                decltype(std::declval<T>().ApplyMove(std::declval<VertexIdxT<typename T::GraphImplT>>(),
                                                     std::declval<unsigned>(),
                                                     std::declval<unsigned>(),
                                                     std::declval<unsigned>(),
                                                     std::declval<unsigned>())),
                decltype(std::declval<T>().ComputeMemoryDatastructure(std::declval<unsigned>(), std::declval<unsigned>())),
                decltype(std::declval<T>().SwapSteps(std::declval<unsigned>(), std::declval<unsigned>())),
                decltype(std::declval<T>().ResetSuperstep(std::declval<unsigned>())),
                decltype(std::declval<T>().OverrideSuperstep(
                    std::declval<unsigned>(), std::declval<unsigned>(), std::declval<unsigned>(), std::declval<unsigned>())),
                decltype(std::declval<T>().CanMove(
                    std::declval<VertexIdxT<typename T::GraphImplT>>(), std::declval<unsigned>(), std::declval<unsigned>())),
                decltype(std::declval<T>().Clear()),
                decltype(T())>> : std::true_type {};

template <typename T>
inline constexpr bool isLocalSearchMemoryConstraintV = IsLocalSearchMemoryConstraint<T>::value;

/**
 * @brief The default memory constraint type, no memory constraints apply.
 *
 */
struct NoLocalSearchMemoryConstraint {
    using GraphImplT = void;
};

/**
 * @brief A memory constraint module for local memory constraints.
 *
 * @tparam GraphT The graph type.
 */
template <typename GraphT>
struct LsLocalMemoryConstraint {
    using GraphImplT = GraphT;

    const SetSchedule<GraphT> *setSchedule_;
    const GraphT *graph_;

    std::vector<std::vector<VMemwT<GraphT>>> stepProcessorMemory_;

    LsLocalMemoryConstraint() : setSchedule_(nullptr), graph_(nullptr) {}

    inline void Initialize(const SetSchedule<GraphT> &setSchedule, const VectorSchedule<GraphT> &) {
        if (setSchedule.GetInstance().GetArchitecture().GetMemoryConstraintType() != MemoryConstraintType::LOCAL) {
            throw std::invalid_argument("Memory constraint type is not LOCAL");
        }

        setSchedule_ = &setSchedule;
        graph_ = &setSchedule_->GetInstance().GetComputationalDag();
        stepProcessorMemory_ = std::vector<std::vector<VMemwT<GraphT>>>(
            setSchedule_->NumberOfSupersteps(), std::vector<VMemwT<GraphT>>(setSchedule_->GetInstance().NumberOfProcessors(), 0));
    }

    inline void ApplyMove(VertexIdxT<GraphT> vertex, unsigned fromProc, unsigned fromStep, unsigned toProc, unsigned toStep) {
        stepProcessorMemory_[toStep][toProc] += graph_->VertexMemWeight(vertex);
        stepProcessorMemory_[fromStep][fromProc] -= graph_->VertexMemWeight(vertex);
    }

    inline bool CanMove(VertexIdxT<GraphT> vertex, const unsigned proc, unsigned step) const {
        return stepProcessorMemory_[step][proc] + graph_->VertexMemWeight(vertex)
               <= setSchedule_->GetInstance().GetArchitecture().MemoryBound(proc);
    }

    void SwapSteps(const unsigned step1, const unsigned step2) {
        std::swap(stepProcessorMemory_[step1], stepProcessorMemory_[step2]);
    }

    void ComputeMemoryDatastructure(unsigned startStep, unsigned endStep) {
        for (unsigned step = startStep; step <= endStep; step++) {
            for (unsigned proc = 0; proc < setSchedule_->GetInstance().NumberOfProcessors(); proc++) {
                stepProcessorMemory_[step][proc] = 0;

                for (const auto &node : setSchedule_->GetProcessorStepVertices()[step][proc]) {
                    stepProcessorMemory_[step][proc] += graph_->VertexMemWeight(node);
                }
            }
        }
    }

    inline void Clear() { stepProcessorMemory_.clear(); }

    inline void ForwardMove(VertexIdxT<GraphT> vertex, unsigned, unsigned, unsigned toProc, unsigned toStep) {
        stepProcessorMemory_[toStep][toProc] += graph_->VertexMemWeight(vertex);
    }

    inline void ResetSuperstep(unsigned step) {
        for (unsigned proc = 0; proc < setSchedule_->GetInstance().GetArchitecture().NumberOfProcessors(); proc++) {
            stepProcessorMemory_[step][proc] = 0;
        }
    }

    void OverrideSuperstep(unsigned step, unsigned proc, unsigned withStep, unsigned withProc) {
        stepProcessorMemory_[step][proc] = stepProcessorMemory_[withStep][withProc];
    }

    bool SatisfiedMemoryConstraint() const {
        for (unsigned step = 0; step < setSchedule_->NumberOfSupersteps(); step++) {
            for (unsigned proc = 0; proc < setSchedule_->GetInstance().NumberOfProcessors(); proc++) {
                if (stepProcessorMemory_[step][proc] > setSchedule_->GetInstance().GetArchitecture().MemoryBound(proc)) {
                    return false;
                }
            }
        }
        return true;
    }
};

}    // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_LOCALSEARCHMEMORYCONSTRAINTMODULES_HPP
