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
 * \file GreedyMetaScheduler.hpp
 * \brief
 */

#ifndef OSP_GREEDYMETASCHEDULER_HPP
#define OSP_GREEDYMETASCHEDULER_HPP

#include <string>
#include <vector>

#include "passes/algorithms/osp/bsp/model/cost/LazyCommunicationCost.hpp"
#include "passes/algorithms/osp/bsp/scheduler/Scheduler.hpp"
#include "passes/algorithms/osp/bsp/scheduler/Serial.hpp"

namespace npu::tile_fwk {
namespace osp {

/**
 * @class GreedyMetaScheduler
 * @brief The GreedyMetaScheduler class represents a meta-scheduler that selects the best schedule produced from a list of
 * added schedulers.
 *
 * This class inherits from the Scheduler class and implements the ComputeSchedule() and GetScheduleName() methods.
 * The ComputeSchedule() method iterates through a list of schedulers, computes a schedule using each one,
 * and returns the schedule with the minimum cost.
 *
 * @tparam GraphT The graph type representing the computational DAG.
 * @tparam CostModel The cost model functor to evaluate schedules. Defaults to LazyCommunicationCost.
 */
template <typename GraphT, typename CostModel = LazyCommunicationCost<GraphT>>
class GreedyMetaScheduler : public Scheduler<GraphT> {
    Serial<GraphT> serialScheduler_;
    std::vector<Scheduler<GraphT> *> schedulers_;

    static constexpr bool verbose_ = false;

  public:
    /**
     * @brief Default constructor for GreedyMetaScheduler.
     */
    GreedyMetaScheduler() : Scheduler<GraphT>() {}

    /**
     * @brief Default destructor for MetaScheduler.
     */
    ~GreedyMetaScheduler() override = default;

    void AddSerialScheduler() { schedulers_.push_back(&serialScheduler_); }

    void AddScheduler(Scheduler<GraphT> &s) { schedulers_.push_back(&s); }

    void ResetScheduler() { schedulers_.clear(); }

    ReturnStatus ComputeSchedule(BspSchedule<GraphT> &schedule) override {
        if (schedule.GetInstance().GetArchitecture().NumberOfProcessors() == 1) {
            if constexpr (verbose_) {
                std::cout << "Using serial scheduler for P=1." << std::endl;
            }
            serialScheduler_.ComputeSchedule(schedule);
            return ReturnStatus::OSP_SUCCESS;
        }

        VWorkwT<GraphT> bestScheduleCost = std::numeric_limits<VWorkwT<GraphT>>::max();
        BspSchedule<GraphT> currentSchedule(schedule.GetInstance());

        for (Scheduler<GraphT> *scheduler : schedulers_) {
            scheduler->ComputeSchedule(currentSchedule);
            const VWorkwT<GraphT> scheduleCost = CostModel()(currentSchedule);

            if constexpr (verbose_) {
                std::cout << "Executed scheduler " << scheduler->GetScheduleName() << ", costs: " << scheduleCost
                          << ", nr. supersteps: " << currentSchedule.NumberOfSupersteps() << std::endl;
            }

            if (scheduleCost < bestScheduleCost) {
                bestScheduleCost = scheduleCost;
                schedule = currentSchedule;
                if constexpr (verbose_) {
                    std::cout << "New best schedule!" << std::endl;
                }
            }
        }

        return ReturnStatus::OSP_SUCCESS;
    }

    std::string GetScheduleName() const override { return "GreedyMetaScheduler"; }
};

}    // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_GREEDYMETASCHEDULER_HPP
