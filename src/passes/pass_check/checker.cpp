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
 * \file remove_redundant_op_checker.cpp
 * \brief
 */

#include "remove_redundant_op_checker.h"

namespace npu{
namespace tile_fwk {
Status Checker::DoPreCheck(Function &function) {
    (void)function;
    return SUCCESS;
}

Status Checker::DoPostCheck(Function &function) {
    (void)function;
    return SUCCESS;
}

Status Checker::CheckValidOp(Function &function) {
    for (const auto &op : function.Operations().DuplicatedOpList()) {
        if (op == nullptr) {
            ALOG_ERROR_F("Found null op in function.Operations().");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status Checker::CheckOpIOValid(Function &function) {
    for (const auto &op : function.Operations().DuplicatedOpList()) {
        for (const auto &input : op->iOperand) {
            if (input == nullptr) {
                ALOG_ERROR_F("The input of op[%d] is null", op->opmagic);
                return FAILED;
            }
        }
        for (const auto &output : op->oOperand) {
            if (output == nullptr) {
                ALOG_ERROR_F("The output of op[%d] is null", op->opmagic);
                return FAILED;
            }
        }
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu