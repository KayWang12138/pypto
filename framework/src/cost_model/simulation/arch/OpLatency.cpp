/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file OpLatency.cpp
 * \brief
 */

#include "OpLatency.h"
#include "cost_model/simulation/arch/A2A3/OpParams.h"
#include "interface/operation/operation.h"

namespace CostModel {

using npu::tile_fwk::DataType;

int OpLatency::GetLatency(const npu::tile_fwk::Operation *op) {
    if (op == nullptr) {
        return OpRegistry::GetInstance().GetDefault()->Calculate(static_cast<const npu::tile_fwk::Operation*>(nullptr));
    }

    auto calc = OpRegistry::GetInstance().Get(op->GetOpcode());
    return calc->Calculate(op);
}

} // namespace CostModel