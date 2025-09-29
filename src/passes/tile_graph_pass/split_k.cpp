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
 * \file split_k.cpp
 * \brief
 */

#include "split_k.h"

namespace npu {
namespace tile_fwk {
Status SplitK::PreCheck(Function &function) {
    ALOG_INFO_F("PreCheck for SplitK.");
    if (!function.LoopCheck().empty()) {
        ALOG_ERROR_F("Loopcheck failed before PreGraph");
        return FAILED;
    }
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_A_MUL_B && op.GetOpcode() == Opcode::OP_A_MULACC_B) {
            // L0C tensor 有且只有一个非空consumer op
            if (op.GetOOperands().size() != 1) {
                ALOG_ERROR_F("[SplitK] invalid op: %s[%d] has output num not equal to ONE.",
                    op.GetOpcodeStr().c_str(), op.GetOpMagic());
                return FAILED;
            }
            auto output = op.GetOOperands().front();
            if ((output->GetMemoryTypeOriginal() != MemoryType::MEM_L0C) || (output->GetConsumers().size() != 1) 
                || (*output->GetConsumers().begin() == nullptr)) {
                ALOG_ERROR_F("[SplitK] %s[%d] has invalid output tenosr[%d].",
                    op.GetOpcodeStr().c_str(), op.GetOpMagic(), output->magic);
                return FAILED;
            }
        }
        if (op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
            // Reduce Acc 的输入和输出必须都是DDR类型
            for (auto &in : op.GetIOperands()) {
                if (in->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
                    ALOG_ERROR_F("[SplitK] %s[%d] has non-DDR input tenosr[%d]",
                        op.GetOpcodeStr().c_str(), op.GetOpMagic(), in->magic);
                    return FAILED;
                }
            }
            for (auto &out : op.GetOOperands()) {
                if (out->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
                    ALOG_ERROR_F("[SplitK] %s[%d] has non-DDR output tenosr[%d]",
                        op.GetOpcodeStr().c_str(), op.GetOpMagic(), out->magic);
                    return FAILED;
                }
            }
        }
    }
    ALOG_INFO_F("PreCheck for SplitK success.");
    return SUCCESS;
}

// verstion 2.0
Status SplitK::RunOnFunction(Function &function) {
    ALOG_INFO_F("===> start SplitK");
    if (EliminateReduceAcc(function) != SUCCESS) {
        ALOG_ERROR_F("Eliminate ReduceAcc failed.");
        return FAILED;
    }
    if (DeadOperationEliminator::EliminateDeadOperation(function) != SUCCESS) {
        ALOG_ERROR_F("Eliminate dead operation failed in CommonOperationEliminate.");
        return FAILED;
    }
    ALOG_INFO_F("===> End SplitK");
    return SUCCESS;
}

Status SplitK::EliminateReduceAcc(Function &function) {
    /*
    Before:
    A_MUL_B --> L0C --> Copy_Out --> \
                                      Gm  ---> \
    A_MUL_B --> L0C --> Copy_Out --> /          \
                                                  Reduce_Acc -----> Gm(Final)
    A_MUL_B --> L0C --> Copy_Out --> \          /
                                      Gm  ---> /
    A_MUL_B --> L0C --> Copy_Out --> /

    After:
    A_MUL_B --> L0C --> Copy_Out -------------------------------->   \
                                                                      \
    A_MUL_B --> L0C --> Copy_Out --------------------------------> \   \
                                                                    Gm(Final)
    A_MUL_B --> L0C --> Copy_Out --------------------------------> /   /
                                                                      /
    A_MUL_B --> L0C --> Copy_Out -------------------------------->  /
    */
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
            ALOG_INFO_F("ATOMIC_ADD, opmagic: %d", op.GetOpMagic());
            if (op.GetIOperands().size() <= 1) {
                ALOG_ERROR_F("%s[%d] has input num less than 1", op.GetOpcodeStr().c_str(), op.GetOpMagic());
                return FAILED;
            }
            if (op.GetOOperands().size() != 1) {
                ALOG_ERROR_F("%s[%d] has output num != 1", op.GetOpcodeStr().c_str(), op.GetOpMagic());
                return FAILED;
            }

            auto reduceOut = op.GetOOperands().front();
            reduceOut->GetProducers().clear();

            for (auto &input : op.GetIOperands()) {
                auto producersBackup = input->GetProducers();
                for (auto &produceCopyOutOp : producersBackup) {
                    produceCopyOutOp->ReplaceOOperand(0, reduceOut);
                    // Set the Copy Out's atomic_add as true
                    produceCopyOutOp->SetAttribute(ACC_A_MUL_B, 1);
                }
            }
            // delete the Reduce_Acc
            op.SetAsDeleted();
            ALOG_DEBUG_F("%s[%d] will be deleted.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
        }
    }
    function.EraseOperations(true);
    return SUCCESS;
}

} // namespace tile_fwk
} // namespace npu