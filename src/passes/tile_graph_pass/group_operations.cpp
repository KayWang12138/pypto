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
 * \file group_operations.cpp
 * \brief
 */

#include "group_operations.h"

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/operation/attribute.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {
bool IsMatrixMul(Opcode opCode) {
    return opCode == Opcode::OP_A_MUL_B || opCode == Opcode::OP_A_MUL_BT;
}

Status GroupOperationsOp::RunOnFunction(Function &function) {
    for (auto &op : function.Operations()) {
        if (IsMatrixMul(op.GetOpcode())) {
            ASSERT(op.GetOOperands().size() == 1);
            RunOnOperation(function, op);
        }
    }
    function.CheckGroupValid();
    return SUCCESS;
}

void GroupOperationsOp::RunOnOperation(Function &function, Operation &op) const {
    ASSERT(IsMatrixMul(op.GetOpcode()));
    std::vector<Operation *> operationGroup;
    std::unordered_set<Operation *> used;
    Operation *curOp = &op;
    while (true) {
        for (auto &iOperand : curOp->iOperand) {
            if (iOperand->IsDummy()) {
                continue;
            }
            ASSERT(iOperand->GetProducers().size() == 1);
            auto &producer = *iOperand->GetProducers().begin();
            ASSERT(used.count(producer) == 0);
            ASSERT(producer->oOperand[0]->GetConsumers().size() == 1);
            ASSERT(producer->GroupID() == NON_GROUP);
            used.emplace(producer);
            operationGroup.emplace_back(producer);
        }
        ASSERT(curOp->GroupID() == NON_GROUP);
        operationGroup.emplace_back(curOp);
        ASSERT(curOp->GetOOperands().size() == 1);
        auto oOperand = curOp->GetOOperands()[0];
        if (!oOperand->IsDummy()) {
            break;
        }
        ASSERT(oOperand->GetConsumers().size() == 1);
        curOp = *oOperand->GetConsumers().begin();
        ASSERT(curOp != nullptr);
        ASSERT(curOp->GetOpcode() == Opcode::OP_A_MULACC_B || curOp->GetOpcode() == Opcode::OP_A_MULACC_BT);
    }
    function.AddOperationGroup(std::move(operationGroup));
}
} // namespace

