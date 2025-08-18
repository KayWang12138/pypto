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
 * \file remove_redundant_reshape.cpp
 * \brief
 */

#include "remove_redundant_reshape.h"
#include "interface/tensor/logical_tensor.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {
namespace {
Status CheckIOOperands(const Operation &op, LogicalTensorPtr &in, LogicalTensorPtr &out) {
    if (op.GetIOperands().size() != 1) {return FAILED;}
    if (op.GetOOperands().size() != 1) {return FAILED;}
    in = op.GetIOperands().front();
    if (in == nullptr) {return FAILED;}
    out = op.GetOOperands().front();
    if (out == nullptr) {return FAILED;}
    return SUCCESS;
}

// PreCheck for reshape
// ..->reshape->out (will be removed regardless of its function)
Status PreCheckReshape(const LogicalTensorPtr &in) {
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

Status ProcessPreCheck(const Operation *op) {
    if (op->GetOpcode() == Opcode::OP_RESHAPE) {
        auto in = op->iOperand.front();
        if (in == nullptr) {return FAILED;}
        if (PreCheckReshape(in) != SUCCESS) {
            return FAILED;
        }
    }
    return SUCCESS;
}
}

Status RemoveRedundantReshape::RunOnFunction(Function &function) {
    ALOG_INFO_F("===> Start RemoveRedundantShapePass for function [%s].", function.GetRawName().c_str());
    if (RemoveReshape(function) != SUCCESS) {return FAILED;}
    ALOG_INFO_F("===> End RemoveRedundantShapePass for function [%s].", function.GetRawName().c_str());
    return SUCCESS;
}

Status RemoveRedundantReshape::RemoveReshape(Function &function) const {
    std::unordered_set<Operation *> redundantResapes;
    LogicalTensorPtr in;
    LogicalTensorPtr out;
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_RESHAPE) {
            continue;
        }
        if (CheckIOOperands(op, in, out) != SUCCESS) {return FAILED;}
        auto consumers = out->GetConsumers();
        bool allConsumersIsReshape = true;
        for (auto &consumerOp : consumers) {
            if (consumerOp == nullptr) {return FAILED;}
            if (in->shape == out->shape || consumerOp->GetOpcode() == Opcode::OP_RESHAPE) {
                consumerOp->ReplaceInput(in, out);
            } else {
                allConsumersIsReshape = false;
            }
        }
        if (allConsumersIsReshape == true) {
            ALOG_DEBUG_F("All consummers of op [%d] are reshape.", op.GetOpMagic());
            redundantResapes.insert(&op);
        }
    }
    if (!redundantResapes.empty()) {
        for (auto &ele : redundantResapes) {
            ALOG_DEBUG_F("Delete OP_RESHAPE, magic %d", ele->GetOpMagic());
            if (ele->IsDeleted()) {return FAILED;}
            ele->SetAsDeleted();
        }
        function.EraseOperations(false);
    }
    return SUCCESS;
}

Status RemoveRedundantReshape::PreCheck(Function &function) {
    ALOG_INFO_F("PreCheck for RemoveRedundantReshape");
    auto ops = function.Operations().DuplicatedOpList();
    for (const auto &op : ops) {
        if (op == nullptr) {return FAILED;}
    }
    if (!function.LoopCheck().empty()) {return FAILED;}
    for (const auto &op : ops) {
        if (ProcessPreCheck(op)) {
            ALOG_ERROR_F("Precheck RemoveRedundantReshape failed");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status RemoveRedundantReshape::PostCheck(Function &function) {
    ALOG_INFO_F("PostCheck for RemoveRedundantReshape");
    if (!function.LoopCheck().empty()) {return FAILED;}
    return SUCCESS;
}
} // namespace npu::tile_fwk
