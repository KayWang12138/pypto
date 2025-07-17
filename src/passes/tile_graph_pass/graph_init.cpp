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
 * \file graph_init.cpp
 * \brief
 */

#include "graph_init.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {

inline bool IsOut(const Opcode &op) {
    return (op == Opcode::OP_UB_COPY_OUT || op == Opcode::OP_L0C_COPY_OUT || op == Opcode::OP_TRANSPOSE_DATAMOVE ||
            op == Opcode::OP_COPY_OUT);
}

Status GraphInitPass::RunOnFunction(Function &function) {
    auto opList = function.Operations();
    for (size_t i = 0; i < opList.size(); i++) {
        if (opList[i].GetOpcode() == Opcode::OP_A_MUL_B || opList[i].GetOpcode() == Opcode::OP_A_MULACC_B ||
            opList[i].GetOpcode() == Opcode::OP_L1_TO_L0A || opList[i].GetOpcode() == Opcode::OP_L1_TO_L0B) {
            opList[i].SetAttribute(OpAttributeKey::isCube, true);
        }
        else {
            opList[i].SetAttribute(OpAttributeKey::isCube, false);
        }
    }
    return SUCCESS;
}

Status GraphInitPass::PreCheck(Function &function) {
    ALOG_INFO("PreCheck for pass: GraphInitPass");
    if (!function.LoopCheck().empty()) {
        ALOG_ERROR("Loopcheck failed before pass: GraphInitPass");
    }
    return SUCCESS;
}

Status GraphInitPass::PostCheck(Function &function) {
    ALOG_INFO("PostCheck for pass: GraphInitPass");
    if (!function.LoopCheck().empty()) {
        ALOG_ERROR("Loopcheck failed after pass: GraphInitPass");
    }
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_A_MUL_B || op.GetOpcode() == Opcode::OP_A_MULACC_B ||
            op.GetOpcode() == Opcode::OP_L1_TO_L0A || op.GetOpcode() == Opcode::OP_L1_TO_L0B) {
            if (!op.GetBoolAttribute(OpAttributeKey::isCube)) {
                ALOG_WARN_F("isCube attribute is not set for op: %s", op.GetOpcodeStr().c_str());
            }
        }
        else {
            if (op.GetBoolAttribute(OpAttributeKey::isCube)) {
                ALOG_WARN_F("isCube attribute is set for op: %s", op.GetOpcodeStr().c_str());
            }
        }
    }
    return SUCCESS;
}
} // namespace npu::tile_fwk