/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file remove_undriven_view.cpp
 * \brief
 */

#include "remove_undriven_view.h"
#include "interface/operation/operation.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "RemoveUndrivenView"

using namespace npu::tile_fwk;

Status Process(Function &function) {
    bool hasDeleted = false;
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE_SSA) {
            continue;
        }
        ASSERT(op.HasAttribute(OpAttributeKey::inplaceIdx));
        auto inplaceIdx = op.GetIntAttribute(OpAttributeKey::inplaceIdx);
        auto iOperand = op.GetInputOperand(inplaceIdx);
        ASSERT(iOperand->GetProducers().size() == 1);
        auto &producerOp = **iOperand->GetProducers().begin();
        ASSERT(producerOp.GetOpcode() == Opcode::OP_VIEW);
        if (!producerOp.GetInputOperand(0)->GetProducers().empty()) {
            continue;
        }
        // 降级为普通的Assemble
        op.SetOpCode(Opcode::OP_ASSEMBLE);
        op.RemoveAttr(OpAttributeKey::inplaceIdx);
        iOperand->RemoveConsumer(op);
        op.EraseInput(iOperand);
        // 删除undriven的View
        producerOp.SetAsDeleted();
        hasDeleted = true;
    }
    if (hasDeleted) {
        function.EraseOperations();
    }
    return SUCCESS;
}

Status RemoveUndrivenView::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Operation, "===> Start RemoveUndrivenView.");
    if (Process(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "===> Process failed.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Operation, "===> End RemoveUndrivenView.");
    return SUCCESS;
}