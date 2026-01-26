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
 * \file Scheduler.hpp
 * \brief
 */

#ifndef OSP_SCHEDULER_HPP
#define OSP_SCHEDULER_HPP

#include <string>

#include "passes/algorithms/osp/auxiliary/return_status.hpp"
#include "passes/algorithms/osp/bsp/model/BspInstance.hpp"
#include "passes/algorithms/osp/bsp/model/BspSchedule.hpp"
#include "passes/algorithms/osp/concepts/computational_dag_concept.hpp"

namespace npu::tile_fwk {
namespace osp {

/**
 * @class Scheduler
 * @brief Interface for BSP schedulers.
 *
 * The Scheduler class defines the common interface for all scheduling algorithms computing BSP schedules.
 * It specifies the contract for computing standard BSP schedules (BspSchedule) and communication-aware schedules
 * (BspScheduleCS).
 */
template <typename GraphT>
class Scheduler {
    static_assert(isComputationalDagV<GraphT>, "Scheduler can only be used with computational DAGs.");

  public:
    /**
     * @brief Constructor for the Scheduler class.
     */
    Scheduler() = default;

    /**
     * @brief Destructor for the Scheduler class.
     */
    virtual ~Scheduler() = default;

    /**
     * @brief Get the name of the scheduling algorithm.
     * @return The name of the scheduling algorithm.
     */
    virtual std::string GetScheduleName() const = 0;

    /**
     * @brief Computes a BSP schedule for the given BSP instance.
     *
     * This pure virtual function must be implemented by derived classes to provide
     * the specific scheduling logic. It modifies the passed BspSchedule object.
     *
     * @param schedule The BspSchedule object to be computed. It contains the BspInstance.
     * @return ReturnStatus::OSP_SUCCESS if a schedule was successfully computed,
     *         ReturnStatus::ERROR if an error occurred, or other status codes as appropriate.
     */
    virtual ReturnStatus ComputeSchedule(BspSchedule<GraphT> &schedule) = 0;

};

}    // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_SCHEDULER_HPP
