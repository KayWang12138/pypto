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
 * \file remove_redundant_cast_checker.cpp
 * \brief
 */

#include "remove_redundant_cast_checker.h"

namespace npu {
namespace tile_fwk {
Status RemoveRedundantCastChecker::DoPreCheck(Function &function) {
    ALOG_INFO_F("PreCheck for RemoveRedundantCast");
    std::vector<Operation *> opList = function.Operations().DuplicatedOpList();
    for (size_t opIdx = 0; opIdx < opList.size(); opIdx++) {
        Operation *op = opList[opIdx];
        if (op->GetOpcode() != Opcode::OP_CAST) {
            continue;
        }
        if (op->GetIOperands().size() != 1) {
            ALOG_ERROR_F("CAST op %d has %d input tensor, which should be 1.",
                         op->GetOpMagic(), static_cast<int>(op->GetIOperands().size()));
            return FAILED;
        }
        if (op->GetOOperands().size() != 1) {
            ALOG_ERROR_F("CAST op %d has %d output tensor, which should be 1.",
                         op->GetOpMagic(), static_cast<int>(op->GetOOperands().size()));
            return FAILED;
        }
    }
    return SUCCESS;
}

Status RemoveRedundantCastChecker::DoPostCheck(Function &function) {
    ALOG_INFO_F("PostCheck for RemoveRedundantCast");
    std::vector<Operation *> opList = function.Operations().DuplicatedOpList();
    for (size_t opIdx = 0; opIdx < opList.size(); opIdx++) {
        Operation *op = opList[opIdx];
        if (SupportBF16(op)) {
            continue;
        }
        auto iOperands = op->GetIOperands();
        for (auto &iop : iOperands) {
            if (iop->Datatype() == DataType::DT_BF16) {
                ALOG_ERROR_F("Exist unsupported BF16 compute between op %d and tensor %d",
                             op->GetOpMagic(), iop->GetMagic());
                return FAILED;
            }
        }
        auto oOperands = op->GetOOperands();
        for (auto &oop : oOperands) {
            if (oop->Datatype() == DataType::DT_BF16) {
                ALOG_ERROR_F("Exist unsupported BF16 compute between op %d and tensor %d",
                             op->GetOpMagic(), oop->GetMagic());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

bool RemoveRedundantCastChecker::SupportBF16(Operation *op) {
    std::unordered_set<OpCalcType> calTypes{OpCalcType::ELMWISE, OpCalcType::BROADCAST, OpCalcType::REDUCE,
                                            OpCalcType::CONV};
    OpCalcType opCalType = OpcodeManager::Inst().GetOpCalcType(op->GetOpcode());
    if (calTypes.count(opCalType) > 0) {
        return false;
    }
    return true;
}
} // namespace tile_fwk
} // namespace npu