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
 * \file arithmetic_reordering.h
 * \brief
 */

#ifndef ARITHMETIC_REORDERING_H
#define ARITHMETIC_REORDERING_H

#include <unordered_set>
#include <vector>
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_interface/pass.h"

namespace npu::tile_fwk {

class ArithmeticReordering : public Pass {
public:
    ArithmeticReordering() : Pass("ArithmeticReordering") {}
    ~ArithmeticReordering() override = default;

    Status RunOnFunction(Function &function) override;

    struct Term {
        LogicalTensorPtr tensor;
        int waitTime;
    };

private:
    static bool IsRebalanceable(Opcode opcode);

    // Collect flat terms for a given tensor at the given opcode level.
    // Follows same-opcode producers recursively to flatten chains.
    // Different-opcode rebalanceable producers are returned as opaque terms.
    bool CollectTerms(const LogicalTensorPtr &tensor,
                      Opcode opcode,
                      std::vector<Term> &terms,
                      std::unordered_set<Operation *> &opsToDelete,
                      std::unordered_set<uint64_t> &visitedOps,
                      std::unordered_set<uint64_t> &visitedTensors,
                      Function &function) const;

    // Compute the wait time of a tensor by recursing into its producers.
    // Stops at non-rebalanceable producers (returns 0 for raw inputs,
    // 1+ for ops).
    int ComputeWaitTime(const LogicalTensorPtr &tensor) const;

    Status ProcessRoot(Operation &rootOp, Function &function) const;

};

} // namespace npu::tile_fwk

#endif // ARITHMETIC_REORDERING_H