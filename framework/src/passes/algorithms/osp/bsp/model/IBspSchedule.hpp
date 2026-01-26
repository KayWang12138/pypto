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
 * \file IBspSchedule.hpp
 * \brief
 */

#ifndef OSP_IBSPSCHEDULE_HPP
#define OSP_IBSPSCHEDULE_HPP

#include "BspInstance.hpp"

namespace npu::tile_fwk {
namespace osp {

/**
 * @class IBspSchedule
 * @brief Pure interface class to organize the interaction with a BSP schedule.
 *
 * A BSP schedule assigns nodes to processors and supersteps, is based on an instance, and has a number of supersteps.
 * It provides unified access for different data implementations.
 *
 * - The class `BspSchedule` implements the assignments as vectors.
 * - The class `SetBspSchedule` implements containers that contain all nodes assigned to a pair of processor and superstep.
 *
 * @tparam GraphT The type of the computational DAG, which must satisfy `is_computational_dag_v`.
 * @see BspInstance
 * @see BspSchedule
 * @see SetBspSchedule
 */
template <typename GraphT>
class IBspSchedule {
    using VertexIdx = VertexIdxT<GraphT>;

    static_assert(isComputationalDagV<GraphT>, "IBspSchedule can only be used with computational DAGs.");

  public:
    virtual ~IBspSchedule() = default;

    /**
     * @brief Get the BSP instance associated with this schedule.
     *
     * @return The BSP instance.
     */
    [[nodiscard]] virtual const BspInstance<GraphT> &GetInstance() const = 0;

    /**
     * @brief Set the assigned superstep for a node.
     *
     * @param node The node index.
     * @param superstep The assigned superstep.
     */
    virtual void SetAssignedSuperstep(VertexIdx node, unsigned int superstep) = 0;

    /**
     * @brief Set the assigned processor for a node.
     *
     * @param node The node index.
     * @param processor The assigned processor.
     */
    virtual void SetAssignedProcessor(VertexIdx node, unsigned int processor) = 0;

    /**
     * @brief Get the assigned superstep of a node.
     *
     * @param node The node index.
     * @return The assigned superstep of the node.
     *         If the node is not assigned to a superstep, this.NumberOfSupersteps() is returned.
     */
    [[nodiscard]] virtual unsigned AssignedSuperstep(VertexIdx node) const = 0;

    /**
     * @brief Get the assigned processor of a node.
     *
     * @param node The node index.
     * @return The assigned processor of the node.
     *         If the node is not assigned to a processor, this.GetInstance().NumberOfProcessors() is returned.
     */
    [[nodiscard]] virtual unsigned AssignedProcessor(VertexIdx node) const = 0;

    /**
     * @brief Get the number of supersteps in the schedule.
     *
     * @return The number of supersteps in the schedule.
     */
    [[nodiscard]] virtual unsigned NumberOfSupersteps() const = 0;
};

}    // namespace  osp
} // namespace npu::tile_fwk
#endif // OSP_IBSPSCHEDULE_HPP
