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
 * \file kl_include.hpp
 * \brief
 */

#ifndef OSP_KL_INCLUDE_HPP
#define OSP_KL_INCLUDE_HPP

#include "comm_cost_modules/kl_hyper_total_comm_cost.hpp"
#include "kl_improver.hpp"

namespace npu::tile_fwk {
namespace osp {

using DoubleCostT = double;

template <typename GraphT, unsigned windowSize = 1>
using KlTotalLambdaCommImprover = KlImprover<GraphT,
                                             KlHyperTotalCommCostFunction<GraphT, DoubleCostT, windowSize>,
                                             windowSize,
                                             DoubleCostT>;

}    // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_KL_INCLUDE_HPP
