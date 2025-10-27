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
#include "passes/pass_utils/dead_operation_eliminate.h"

namespace npu {
namespace tile_fwk {
Status SplitK::PreCheck(Function &function) {
    APASS_LOG_INFO_F(GetName().c_str(), "Operation", "PreCheck for SplitK.");
    if (!function.LoopCheck().empty()) {
        APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "Loopcheck failed before PreGraph; Please check if there is a Loop.");
        return FAILED;
    }
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_A_MUL_B && op.GetOpcode() == Opcode::OP_A_MULACC_B) {
            // L0C tensor 有且只有一个非空consumer op
            if (op.GetOOperands().size() != 1) {
                APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "Invalid op: [%d] has output num not equal to ONE; Please check if the output num is ONE.", op.GetOpMagic());
                return FAILED;
            }
            auto output = op.GetOOperands().front();
            if ((output->GetMemoryTypeOriginal() != MemoryType::MEM_L0C) || (output->GetConsumers().size() != 1) 
                || (*output->GetConsumers().begin() == nullptr)) {
                APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "Op[%d] has invalid output tenosr[%d]; Please check if the output tensor is vaild.", op.GetOpMagic(), output->magic);
                return FAILED;
            }
        }
        if (op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
            // 输入数量不能小于1
            if (op.GetIOperands().size() < 1) {
                APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "Op[%d] has input num less than 1; Please check the input num.", op.GetOpMagic());
                return FAILED;
            }
            // 输出数量必须等于1
            if (op.GetOOperands().size() != 1) {
                APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "Op[%d] has output num != 1; Please check if the output num for is ONE.", op.GetOpMagic());
                return FAILED;
            }
            // Reduce Acc 的输入和输出必须都是DDR类型
            for (auto &in : op.GetIOperands()) {
                if (in->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
                    APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "Op[%d] has non-DDR input tenosr[%d]; Please check the memory type of the input tensor.", op.GetOpMagic(), in->magic);
                    return FAILED;
                }
            }
            for (auto &out : op.GetOOperands()) {
                if (out->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
                    APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "Op[%d] has non-DDR output tenosr[%d]; Please check the memory type of the output tensor.", op.GetOpMagic(), out->magic);
                    return FAILED;
                }
            }
        }
    }
    APASS_LOG_INFO_F(GetName().c_str(), "Operation", "PreCheck for SplitK success.");
    return SUCCESS;
}

// verstion 2.0
Status SplitK::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(GetName().c_str(), "Operation", "===> start SplitK");
    if (EliminateReduceAcc(function) != SUCCESS) {
        APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "Eliminate ReduceAcc failed.");
        return FAILED;
    }
    if (DeadOperationEliminator::EliminateDeadOperation(function) != SUCCESS) {
        APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "Eliminate dead operation failed in CommonOperationEliminate.");
        return FAILED;
    }
    APASS_LOG_INFO_F(GetName().c_str(), "Operation", "===> End SplitK");
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
            APASS_LOG_INFO_F(GetName().c_str(), "Operation", "ATOMIC_ADD, opmagic: %d", op.GetOpMagic());
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
            APASS_LOG_DEBUG_F(GetName().c_str(), "Operation", "%s[%d] will be deleted.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
        }
    }
    function.EraseOperations(true);
    return SUCCESS;
}

} // namespace tile_fwk
} // namespace npu