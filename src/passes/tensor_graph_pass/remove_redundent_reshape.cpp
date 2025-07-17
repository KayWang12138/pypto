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
 * \file remove_redundent_reshape.cpp
 * \brief
 */

#include "remove_redundent_reshape.h"
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
}

Status RemoveRedundentReshape::RunOnFunction(Function &function) {
    ALOG_INFO_F("===> Start RemoveRedundentShapePass for function [%s].", function.GetRawName().c_str());
    if (RemoveReshape(function) != SUCCESS) {return FAILED;}
    ALOG_INFO_F("===> End RemoveRedundentShapePass for function [%s].", function.GetRawName().c_str());
    return SUCCESS;
}

Status RemoveRedundentReshape::RemoveReshape(Function &function) const {
    std::unordered_set<Operation *> redundentResapes;
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
            ALOG_DEBUG_F("All consummers of op [%s] are reshape.", op.GetOpMagic());
            redundentResapes.insert(&op);
        }
    }
    if (!redundentResapes.empty()) {
        for (auto &ele : redundentResapes) {
            ALOG_DEBUG_F("Delete OP_RESHAPE, magic %d", ele->GetOpMagic());
            if (ele->IsDeleted()) {return FAILED;}
            ele->SetAsDeleted();
        }
        function.EraseOperations(false);
    }
    return SUCCESS;
}
} // namespace npu::tile_fwk
