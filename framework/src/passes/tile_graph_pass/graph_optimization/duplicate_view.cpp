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
 * \file duplicate_view.cpp
 * \brief
 */

#include "duplicate_view.h"

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_utils/pass_utils.h"

namespace npu::tile_fwk {
Status DuplicateView::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(GetName().c_str(), "Operation", "===> Start DuplicateView for function [%s].", function.GetRawName().c_str());
    if (DuplicateViewPass(function) != SUCCESS) {
        APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "Run DuplicateView for function [%s] failed.", function.GetRawName().c_str());
        return FAILED;
    }
    APASS_LOG_INFO_F(GetName().c_str(), "Operation", "===> End DuplicateView for function [%s].", function.GetRawName().c_str());
    return SUCCESS;
}

Status DuplicateView::ViewWithoutL1(Function &function, Operation &operation) const {
    auto iOperand = operation.iOperand[0];
    for (auto &oOperand : operation.oOperand) {
        if (oOperand == nullptr) {
            APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "Null output operand detected while iterating over the output operands of the operation [%d].", operation.opmagic); 
            return FAILED;
        }
        if (oOperand->GetConsumers().size() == 1) {
            continue;
        }
        auto consumers = oOperand->GetConsumers();
        for (auto &consumer : consumers) {
            if (consumer == nullptr) {
                APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "Null consumer detected while iterating over the consumers of the output operand [%d].", oOperand->magic);
                return FAILED;
            }
            if (consumer->GetOpcode() == Opcode::OP_VIEW) {
                continue;
            }
            auto dst = oOperand->Clone(function, true);
            if (dst == nullptr) {
                APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "Clone failed for output operand [%d].", oOperand->magic);
                return FAILED;
            }
            consumer->ReplaceInput(dst, oOperand);
            auto &newOp = function.AddRawOperation(Opcode::OP_VIEW, {iOperand}, {dst});
            auto oriViewAttr = dynamic_cast<ViewOpAttribute *>(operation.GetOpAttribute().get());
            auto newOffset = oriViewAttr->GetFromOffset();
            auto newDynOffset = oriViewAttr->GetFromDynOffset();
            auto newDynValidShape = oriViewAttr->GetToDynValidShape();
            auto newViewAttr = std::make_shared<ViewOpAttribute>(newOffset, newDynOffset, newDynValidShape);
            newOp.SetOpAttribute(newViewAttr);
        }
    }
    return SUCCESS;
}

Status DuplicateView::RunOnOperation(Function &function, Operation &operation) const {
    auto opcode = operation.GetOpcode();
    if (opcode != Opcode::OP_VIEW) {
        return SUCCESS;
    }
    auto viewAttr = dynamic_cast<ViewOpAttribute *>(operation.GetOpAttribute().get());
    if (viewAttr->GetTo() == MEM_L1) {
        return SUCCESS;
    } else {
        return ViewWithoutL1(function, operation);
    }
}

Status DuplicateView::DuplicateViewPass(Function &function) const {
    for (auto &op : function.Operations()) {
        if (RunOnOperation(function, op) != SUCCESS) {
            APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "RunOperation failed for operation [%d].", op.opmagic);
            return FAILED;
        }
    }
    return SUCCESS;
}
} // namespace npu::tile_fwk
