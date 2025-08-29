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
 * \file remove_redundant_reshape_checker.cpp
 * \brief
 */

#include "remove_redundant_reshape_checker.h"

namespace npu {
namespace tile_fwk {
Status RemoveRedundantReshapeChecker::DoPreCheck(Function &function) {
    ALOG_INFO_F("PreCheck for RemoveRedundantReshape");
    if (CheckValidOp(function) != SUCCESS) {
        ALOG_ERROR_F("Found invalid op from the function.");
        return FAILED;
    }
    if (CheckOpIOValid(function) != SUCCESS) {
        ALOG_ERROR_F("Found invalid input/output in the function.");
        return FAILED;
    }
    for (const auto &op : function.Operations().DuplicatedOpList()) {
        if (ProcessPreCheck(op)) {
            ALOG_ERROR_F("Precheck RemoveRedundantReshape failed");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status RemoveRedundantReshapeChecker::ProcessPreCheck(const Operation *op) {
    if (op->GetOpcode() == Opcode::OP_RESHAPE) {
        auto in = op->iOperand.front();
        if (PreCheckReshape(in) != SUCCESS) {
            ALOG_ERROR_F("Precheck of reshape op[%d] failed!", op->GetOpMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}

// PreCheck for reshape
// ..->reshape->out (will be removed regardless of its function)
Status RemoveRedundantReshapeChecker::PreCheckReshape(const LogicalTensorPtr &in) {
    for (auto &childOp : in->GetConsumers()) {
        if (childOp->GetOpcode() == Opcode::OP_RESHAPE) {
            if (childOp->ConsumerOps().empty()) {
                ALOG_ERROR_F("At least one reshape op without consumer!");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu