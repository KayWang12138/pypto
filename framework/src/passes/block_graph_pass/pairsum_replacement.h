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
 * \file pairsum_replacement.h
 * \brief Pass for replacing OP_PAIRSUM with OP_ADD when valid_shape conditions are met
 */

#ifndef PAIRSUM_REPLACEMENT_H
#define PAIRSUM_REPLACEMENT_H

#include "interface/operation/opcode.h"
#include "interface/function/function.h"
#include "passes/pass_interface/pass.h"
#include "tilefwk/platform.h"

namespace npu {
namespace tile_fwk {

class PairSumReplacementPass : public Pass {
public:
    PairSumReplacementPass() : Pass("PairSumReplacementPass") {
        SetSupportedArches({NPUArch::DAV_3510});
    }
    ~PairSumReplacementPass() override = default;

private:
    Status RunOnFunction(Function &function) override;
    
    /**
     * @brief Check if OP_PAIRSUM should be replaced with OP_ADD
     * @param op The operation to check
     * @return true if replacement should occur, false otherwise
     */
    bool ShouldReplacePairSum(const Operation &op) const;
    
    /**
     * @brief Check if two tensors have matching valid_shape
     * @param tensor1 First tensor to compare
     * @param tensor2 Second tensor to compare
     * @return true if valid_shape matches, false otherwise
     */
    bool CheckValidShapeMatch(const LogicalTensorPtr &tensor1, const LogicalTensorPtr &tensor2) const;
    
    /**
     * @brief Replace OP_PAIRSUM with OP_ADD
     * @param op The operation to replace
     */
    void ReplacePairSumWithAdd(Operation &op);
};

} // namespace tile_fwk
} // namespace npu

#endif // PAIRSUM_REPLACEMENT_H
