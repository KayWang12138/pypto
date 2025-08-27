/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file remove_redundant_cast_checker.h
 * \brief
 */

#ifndef REMOVE_REDUNDANT_CAST_CHECKER_H
#define REMOVE_REDUNDANT_CAST_CHECKER_H

#include "checker.h"
#include "interface/operation/opcode.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"

namespace npu {
namespace tile_fwk {
class RemoveRedundantCastChecker : Checker {
public:
    Status DoPreCheck(Function &function) override;
    Status DoPostCheck(Function &function) override;
private:
    Status PreCheckReshape(const LogicalTensorPtr &in);
    Status ProcessPreCheck(const Operation *op);
    bool SupportBF16(Operation *op);
};
} // namespace tile_fwk
} // namespace npu
#endif  // REMOVE_REDUNDANT_CAST_CHECKER_H