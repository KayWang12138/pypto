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
 * \file MemoryConstraintModules.hpp
 * \brief
 */

#ifndef OSP_MEMORYCONSTRAINTMODULES_HPP
#define OSP_MEMORYCONSTRAINTMODULES_HPP

#include "passes/algorithms/osp/bsp/model/BspSchedule.hpp"
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
struct IsMemoryConstraint : std::false_type {};

template <typename T>
struct IsMemoryConstraint<
    T,
    std::void_t<decltype(std::declval<T>().Initialize(std::declval<BspInstance<typename T::GraphImplT>>())),
                decltype(std::declval<T>().CanAdd(std::declval<VertexIdxT<typename T::GraphImplT>>(), std::declval<unsigned>())),
                decltype(std::declval<T>().Add(std::declval<VertexIdxT<typename T::GraphImplT>>(), std::declval<unsigned>())),
                decltype(std::declval<T>().Reset(std::declval<unsigned>())),
                decltype(T())>> : std::true_type {};

template <typename T>
inline constexpr bool isMemoryConstraintV = IsMemoryConstraint<T>::value;

/**
 * @brief The default memory constraint type, no memory constraints apply.
 *
 */
struct NoMemoryConstraint {
    using GraphImplT = void;
};

/**
 * @brief A memory constraint module for local memory constraints.
 *
 * @tparam GraphT The graph type.
 */
template <typename GraphT>
struct LocalMemoryConstraint {
    using GraphImplT = GraphT;

    const BspInstance<GraphT> *instance_;

    std::vector<VMemwT<GraphT>> currentProcMemory_;

    LocalMemoryConstraint() : instance_(nullptr) {}

    inline void Initialize(const BspInstance<GraphT> &instance) {
        instance_ = &instance;
        currentProcMemory_ = std::vector<VMemwT<GraphT>>(instance.NumberOfProcessors(), 0);

        if (instance.GetArchitecture().GetMemoryConstraintType() != MemoryConstraintType::LOCAL) {
            throw std::invalid_argument("Memory constraint type is not LOCAL");
        }
    }

    inline bool CanAdd(const VertexIdxT<GraphT> &v, const unsigned proc) const {
        return currentProcMemory_[proc] + instance_->GetComputationalDag().VertexMemWeight(v)
               <= instance_->GetArchitecture().MemoryBound(proc);
    }

    inline void Add(const VertexIdxT<GraphT> &v, const unsigned proc) {
        currentProcMemory_[proc] += instance_->GetComputationalDag().VertexMemWeight(v);
    }

    inline bool CanAdd(const unsigned proc, const VMemwT<GraphT> &customMemWeight, const VMemwT<GraphT> &) const {
        return currentProcMemory_[proc] + customMemWeight <= instance_->GetArchitecture().MemoryBound(proc);
    }

    inline void Add(const unsigned proc, const VMemwT<GraphT> &customMemWeight, const VMemwT<GraphT> &) {
        currentProcMemory_[proc] += customMemWeight;
    }

    inline void Reset(const unsigned proc) { currentProcMemory_[proc] = 0; }
};

/**
 * @brief A memory constraint module for local memory constraints.
 *
 * @tparam GraphT The graph type.
 */

/**
 * @brief A memory constraint module for persistent and transient memory constraints.
 *
 * @tparam GraphT The graph type.
 */
template <typename GraphT>
struct PersistentTransientMemoryConstraint {
    static_assert(std::is_convertible_v<VCommwT<GraphT>, VMemwT<GraphT>>,
                  "PersistentTransientMemoryConstraint requires that memory and communication weights are convertible.");

    using GraphImplT = GraphT;

    const BspInstance<GraphT> *instance_;

    std::vector<VMemwT<GraphT>> currentProcPersistentMemory_;
    std::vector<VCommwT<GraphT>> currentProcTransientMemory_;

    PersistentTransientMemoryConstraint() : instance_(nullptr) {}

    inline void Initialize(const BspInstance<GraphT> &instance) {
        instance_ = &instance;

        currentProcPersistentMemory_.assign(instance.NumberOfProcessors(), 0);
        currentProcTransientMemory_.assign(instance.NumberOfProcessors(), 0);

        if (instance.GetArchitecture().GetMemoryConstraintType() != MemoryConstraintType::PERSISTENT_AND_TRANSIENT) {
            throw std::invalid_argument("Memory constraint type is not PERSISTENT_AND_TRANSIENT");
        }
    }

    inline bool CanAdd(const VertexIdxT<GraphT> &v, const unsigned proc) const {
        return (currentProcPersistentMemory_[proc] + instance_->GetComputationalDag().VertexMemWeight(v)
                    + std::max(currentProcTransientMemory_[proc], instance_->GetComputationalDag().VertexCommWeight(v))
                <= instance_->GetArchitecture().MemoryBound(proc));
    }

    inline void Add(const VertexIdxT<GraphT> &v, const unsigned proc) {
        currentProcPersistentMemory_[proc] += instance_->GetComputationalDag().VertexMemWeight(v);
        currentProcTransientMemory_[proc]
            = std::max(currentProcTransientMemory_[proc], instance_->GetComputationalDag().VertexCommWeight(v));
    }

    inline bool CanAdd(const unsigned proc, const VMemwT<GraphT> &customMemWeight, const VCommwT<GraphT> &customCommWeight) const {
        return (currentProcPersistentMemory_[proc] + customMemWeight + std::max(currentProcTransientMemory_[proc], customCommWeight)
                <= instance_->GetArchitecture().MemoryBound(proc));
    }

    inline void Add(const unsigned proc, const VMemwT<GraphT> &customMemWeight, const VCommwT<GraphT> &customCommWeight) {
        currentProcPersistentMemory_[proc] += customMemWeight;
        currentProcTransientMemory_[proc] = std::max(currentProcTransientMemory_[proc], customCommWeight);
    }

    inline void Reset(const unsigned) {}
};

template <typename GraphT>
struct GlobalMemoryConstraint {
    using GraphImplT = GraphT;

    const BspInstance<GraphT> *instance_;

    std::vector<VMemwT<GraphT>> currentProcMemory_;

    GlobalMemoryConstraint() : instance_(nullptr) {}

    inline void Initialize(const BspInstance<GraphT> &instance) {
        instance_ = &instance;
        currentProcMemory_ = std::vector<VMemwT<GraphT>>(instance.NumberOfProcessors(), 0);

        if (instance.GetArchitecture().GetMemoryConstraintType() != MemoryConstraintType::GLOBAL) {
            throw std::invalid_argument("Memory constraint type is not GLOBAL");
        }
    }

    inline bool CanAdd(const VertexIdxT<GraphT> &v, const unsigned proc) const {
        return currentProcMemory_[proc] + instance_->GetComputationalDag().VertexMemWeight(v)
               <= instance_->GetArchitecture().MemoryBound(proc);
    }

    inline void Add(const VertexIdxT<GraphT> &v, const unsigned proc) {
        currentProcMemory_[proc] += instance_->GetComputationalDag().VertexMemWeight(v);
    }

    inline bool CanAdd(const unsigned proc, const VMemwT<GraphT> &customMemWeight, const VCommwT<GraphT> &) const {
        return currentProcMemory_[proc] + customMemWeight <= instance_->GetArchitecture().MemoryBound(proc);
    }

    inline void Add(const unsigned proc, const VMemwT<GraphT> &customMemWeight, const VCommwT<GraphT> &) {
        currentProcMemory_[proc] += customMemWeight;
    }

    inline void Reset(const unsigned) {}
};

template <typename T, typename = void>
struct IsMemoryConstraintSchedule : std::false_type {};

template <typename T>
struct IsMemoryConstraintSchedule<
    T,
    std::void_t<decltype(std::declval<T>().Initialize(std::declval<BspSchedule<typename T::GraphImplT>>(), std::declval<unsigned>())),
                decltype(std::declval<T>().CanAdd(std::declval<VertexIdxT<typename T::GraphImplT>>(), std::declval<unsigned>())),
                decltype(std::declval<T>().Add(std::declval<VertexIdxT<typename T::GraphImplT>>(), std::declval<unsigned>())),
                decltype(std::declval<T>().Reset(std::declval<unsigned>())),
                decltype(T())>> : std::true_type {};

template <typename T>
inline constexpr bool isMemoryConstraintScheduleV = IsMemoryConstraintSchedule<T>::value;

}    // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_MEMORYCONSTRAINTMODULES_HPP
