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
 * \file IBspScheduleEval.hpp
 * \brief
 */

#ifndef OSP_IBSPSCHEDULEEVAL_HPP
#define OSP_IBSPSCHEDULEEVAL_HPP

#include "BspInstance.hpp"

namespace npu::tile_fwk {
namespace osp {

/// @class IBspSchedule
/// @brief Interface for a BSP (Bulk Synchronous Parallel) schedule.
template <typename GraphT>
class IBspScheduleEval {
    using VertexIdx = VertexIdxT<GraphT>;

  public:
    /// @brief Destructor.
    virtual ~IBspScheduleEval() = default;

    virtual VWorkwT<GraphT> ComputeCosts() const = 0;
    virtual VWorkwT<GraphT> ComputeWorkCosts() const = 0;
    virtual unsigned NumberOfSupersteps() const = 0;
    virtual const BspInstance<GraphT> &GetInstance() const = 0;
};

}    // namespace  osp
} // namespace npu::tile_fwk
#endif // OSP_IBSPSCHEDULEEVAL_HPP
