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

namespace npu::tile_fwk {
Status DuplicateView::RunOnFunction(Function &function) {
    ALOG_INFO_F("===> Start DuplicateViewPass for function [%s].", function.GetRawName().c_str());
    if (DuplicateViewPass(function) != SUCCESS) {return FAILED;}
    ALOG_INFO_F("===> End DuplicateViewPass for function [%s].", function.GetRawName().c_str());
    return SUCCESS;
}

Status DuplicateView::RunOnOperation(Function &function, Operation &operation, std::vector<std::pair<LogicalTensorPtr, LogicalTensorPtr>> &viewResults) const {
    auto opcode = operation.GetOpcode();
    if (opcode != Opcode::OP_VIEW) {
        return SUCCESS;
    }
    for (auto &oOperand : operation.oOperand) {
        std::unordered_set<const Operation *> newViewOps;
        if (oOperand == nullptr) {return FAILED;}
        if (oOperand->GetConsumers().size() == 1) {
            continue;
        }
        auto consumers = oOperand->GetConsumers(); // copy consumers to avoid erase while iteration
        for (auto &consumer : consumers) {
            if (consumer == nullptr) {return FAILED;}
            if (consumer->GetOpcode() == Opcode::OP_VIEW) {
                continue;
            }
            auto viewResult = std::make_shared<LogicalTensor>(function, oOperand->Datatype(), oOperand->shape,
                "View_" + oOperand->tensor->symbol, oOperand->nodetype);
            if (viewResult == nullptr) {return FAILED;}
            viewResults.emplace_back(std::make_pair(oOperand, viewResult));
            consumer->ReplaceInput(viewResult, oOperand);
        }
    }
    return SUCCESS;
}

Status DuplicateView::DuplicateViewPass(Function &function) const {
    std::vector<std::pair<LogicalTensorPtr, LogicalTensorPtr>> viewResults;
    for (auto &op : function.Operations()) {
        if (RunOnOperation(function, op, viewResults) != SUCCESS) {return FAILED;}
    }
    for (auto &viewResult : viewResults) {
        auto &viewOp = function.AddOperation(Opcode::OP_VIEW, {viewResult.first}, {viewResult.second});
        viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int>(viewResult.second->GetOffset().size())));
    }
    return SUCCESS;
}
} // namespace npu::tile_fwk
