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
 * \file assign_conv_memory.cpp
 * \brief
 */

#include "assign_conv_memory.h"
namespace npu::tile_fwk {
Status AssignConvMemoryType::RunOnFunction(Function &function) {
    for (auto &op : function.Operations()) {
        RunOnOperation(op);
    }
    for (auto &op : function.Operations()) {
        auto opcode = op.GetOpcode();
        if (opcode == Opcode::OP_L1_COPY_IN) {
            auto curOp = *(op.oOperand[0]->GetConsumers().begin());
            if (curOp->GetOpcode() == Opcode::OP_BT_COPY_IN || curOp->GetOpcode() == Opcode::OP_FIX_COPY_IN ||
                curOp->GetOpcode() == Opcode::OP_FIX_COPY_IN_QUANT_PRE ||
                curOp->GetOpcode() == Opcode::OP_FIX_COPY_IN_RELU_PRE ||
                curOp->GetOpcode() == Opcode::OP_FIX_COPY_IN_RELU_POST ||
                curOp->GetOpcode() == Opcode::OP_FIX_COPY_IN_QUANT_POST ||
                curOp->GetOpcode() == Opcode::OP_FIX_COPY_IN_ELT_ANTIQ ||
                curOp->GetOpcode() == Opcode::OP_FIX_COPY_IN_MTE2_ANTIQ) {
                op.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                    OpImmediate::Specified(std::vector<int>(op.oOperand[0]->shape.size(), 0)),
                    op.oOperand[0]->GetMemoryTypeOriginal(), OpImmediate::Specified(op.oOperand[0]->shape),
                    OpImmediate::Specified(op.iOperand[0]->tensor->GetRawShape()),
                    OpImmediate::Specified(op.iOperand[0]->GetDynValidShape())));
            }
        }
    }
    return SUCCESS;
}

void AssignConvMemoryType::RunOnOperation(Operation &operation) const {
    auto opcode = operation.GetOpcode();
    if (opcode == Opcode::OP_CONV || opcode == Opcode::OP_CONV_ADD) {
        std::vector<MemoryType> memTypeList;
        auto hasAcc = operation.GetIntAttribute(ConvOpAttributeKey::hasAccFlag);
        if (hasAcc > 0) {
            memTypeList.push_back(MemoryType::MEM_L1);
        }
        auto hasElt = operation.GetIntAttribute(ConvOpAttributeKey::hasEltFlag);
        if (hasElt > 0) {
            memTypeList.push_back(MemoryType::MEM_L1);
        }
        auto hasQuantPreVector = operation.GetIntAttribute(FixpOpAttributeKey::hasQuantPreVector);
        if (hasQuantPreVector > 0) {
            memTypeList.push_back(MEM_FIX_QUANT_PRE);
        }
        auto hasQuantPostVector = operation.GetIntAttribute(FixpOpAttributeKey::hasQuantPostVector);
        if (hasQuantPostVector > 0) {
            memTypeList.push_back(MEM_FIX_QUANT_POST);
        }
        auto hasAntiqVector = operation.GetIntAttribute(FixpOpAttributeKey::hasAntiqVector);
        if (hasAntiqVector > 0) {
            memTypeList.push_back(MEM_FIX_ELT_ANTIQ);
        }
        // weight
        memTypeList.push_back(MemoryType::MEM_L0B);
        auto fmaSrcNum = operation.GetIntAttribute(ConvOpAttributeKey::fmapSrcNum);
        for (int i = 0; i < fmaSrcNum; i++) {
            memTypeList.push_back(MemoryType::MEM_L1);
        }
        auto hasBias = operation.GetIntAttribute(ConvOpAttributeKey::hasBiasFlag);
        if (hasBias > 0) {
            memTypeList.push_back(MemoryType::MEM_BT);
        }
        ASSERT((memTypeList.size() == operation.iOperand.size())) << "Conv input operations not match. ";
        for (size_t i = 0; i < operation.iOperand.size(); ++i) {
            auto &tensor = operation.iOperand[i];
            tensor->SetMemoryTypeBoth(memTypeList[i]);
        }
        for (size_t i = 0; i < operation.oOperand.size(); ++i) {
            auto &tensor = operation.oOperand[i];
            tensor->SetMemoryTypeBoth(MemoryType::MEM_L1);
        }
    }
}
} // namespace npu::tile_fwk
