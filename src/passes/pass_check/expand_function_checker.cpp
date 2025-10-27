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
 * \file expand_function_checker.cpp
 * \brief
 */

#include "expand_function_checker.h"
#include "passes/pass_utils/pass_utils.h"

namespace npu {
namespace tile_fwk {
Status ExpandFunctionChecker::DoPreCheck(Function &function) {
    APASS_LOG_INFO_F("ExpandFunctionChecker", "Operation", "PreCheck for ExpandFunction.");
    if (!function.OperationLoopCheck()) {
        APASS_LOG_ERROR_F("ExpandFunctionChecker", "Operation", "Operation Loop detected before expand function; Please validate the operation input specifications.");
        return FAILED;
    }
    std::unordered_set<OpCalcType> calTypes{OpCalcType::ELMWISE, OpCalcType::BROADCAST, OpCalcType::REDUCE,
                                            OpCalcType::CONV};
    for (auto &op : function.Operations().DuplicatedOpList()) {
        OpCalcType opCalType = OpcodeManager::Inst().GetOpCalcType(op->GetOpcode());
        if (calTypes.count(opCalType) > 0) {
            for (auto &itensor: op->GetIOperands()) {
                if (itensor->tensor->datatype == DT_BF16) {
                    APASS_LOG_ERROR_F("ExpandFunctionChecker", "Tensor", "Calculation Op %d has BF16 operand %d.", op->GetOpMagic(), itensor->GetMagic());
                    return FAILED;
                }
            }
            for (auto &otensor: op->GetOOperands()) {
                if (otensor->tensor->datatype == DT_BF16) {
                    APASS_LOG_ERROR_F("ExpandFunctionChecker", "Tensor", "Calculation Op %d has BF16 operand %d.", op->GetOpMagic(), otensor->GetMagic());
                    return FAILED;
                }
            }
        }
    }
    return SUCCESS;
}

Status ExpandFunctionChecker::DoPostCheck(Function &function) {
    APASS_LOG_INFO_F("ExpandFunctionChecker", "Operation", "PostCheck for ExpandFunction.");
    if (function.expandFunctionAccelerate != false) {
        APASS_LOG_ERROR_F("ExpandFunctionChecker", "Operation", "ExpandFunctionAccelerate should equal to false after ExpandFunction process.");
        return FAILED;
    }
    if (!function.OperationLoopCheck()) {
        APASS_LOG_ERROR_F("ExpandFunctionChecker", "Operation", "Operation Loop detected after expand function; Please review the error messages generated during the processing procedure.");
        return FAILED;
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu