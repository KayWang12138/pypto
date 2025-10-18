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
 * \file duplicate_gather_in.cpp
 * \brief
 */

#include "duplicate_gather_in.h"

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {
Status DuplicateGatherIn::RunOnFunction(Function &function) {
    ALOG_INFO_F("===> Start DuplicateGatherIn for function [%s].", function.GetRawName().c_str());
    if (Process(function) != SUCCESS) {
        ALOG_ERROR_F("Process failed.");
        return FAILED;
    }
    ALOG_INFO_F("===> End DuplicateGatherIn for function [%s].", function.GetRawName().c_str());
    return SUCCESS;
}

Status DuplicateGatherIn::ProcessOp(Function &function, Operation &operation) const {
    auto opcode = operation.GetOpcode();
    if (opcode != Opcode::OP_GATHER_IN_L1) {
        return SUCCESS;
    }
    for (auto &oOperand : operation.GetOOperands()) {
        std::unordered_set<const Operation *> newViewOps;
        if (oOperand == nullptr) {
            ALOG_ERROR_F("OP_GATHER_IN_L1's oOperand cannot be nullptr.");
            return FAILED;
        }
        if (oOperand->GetConsumers().size() == 1) {
            continue;
        }
        bool isFirst = true;
        auto consumers = oOperand->GetConsumers(); // copy consumers to avoid erase while iteration
        for (auto &consumer : consumers) {
            if (consumer == nullptr) {
                ALOG_ERROR_F("OP_GATHER_IN_L1's consumer cannot be nullptr.");
                return FAILED;
            }
            if (consumer->GetOpcode() == Opcode::OP_GATHER_IN_L1) {
                ALOG_ERROR_F("OP_GATHER_IN_L1's consumer cannot be OP_GATHER_IN_L1.");
                return FAILED;
            }
            if (isFirst) {
                isFirst = false;
                continue;
            }
            auto dst = oOperand->Clone(function, true);
            if (dst == nullptr) {
                ALOG_ERROR_F("Clone OP_GATHER_IN_L1's oOperand failed.");
                return FAILED;
            }
            consumer->ReplaceInput(dst, oOperand);
            auto &newOp = function.AddRawOperation(Opcode::OP_GATHER_IN_L1, operation.GetIOperands(), {dst});
            newOp.SetAttribute(OpAttributeKey::startOffset, operation.GetIntAttribute(OpAttributeKey::startOffset));
        }
    }
    return SUCCESS;
}

Status DuplicateGatherIn::Process(Function &function) const {
    for (auto &op : function.Operations()) {
        if (ProcessOp(function, op) != SUCCESS) {
            ALOG_ERROR_F("ProcessOp failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}
} // namespace npu::tile_fwk
