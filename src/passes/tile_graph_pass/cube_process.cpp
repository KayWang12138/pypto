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
 * \file cube_process.cpp
 * \brief
 */

#include "cube_process.h"

namespace npu {
namespace tile_fwk {
Status CubeProcess::PreCheck(Function &function) {
    ALOG_INFO_F("PreCheck for CubeProcess.");
    if (!function.LoopCheck().empty()) {
        ALOG_ERROR_F("Loopcheck failed before PreGraph");
        return FAILED;
    }
    for (auto &op : function.Operations()) {
        if (op.GetSubgraphID() == NOT_IN_SUBGRAPH) {
            ALOG_ERROR_F("%s[%d] is not partitioned.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
            return FAILED;
        }
        if (op.GetOpcode() == Opcode::OP_A_MUL_B && op.GetOpcode() == Opcode::OP_A_MULACC_B) {
            // L0C tensor 有且只有一个非空consumer op
            if (op.GetOOperands().size() != 1) {
                ALOG_ERROR_F("[CubeProcess] invalid op: %s[%d] has output num not equal to ONE.",
                    op.GetOpcodeStr().c_str(), op.GetOpMagic());
                return FAILED;
            }
            auto output = op.GetOOperands().front();
            if ((output->GetMemoryTypeOriginal() != MemoryType::MEM_L0C) || (output->GetConsumers().size() != 1) 
                || (*output->GetConsumers().begin() == nullptr)) {
                ALOG_ERROR_F("[CubeProcess] %s[%d] has invalid output tenosr[%d].",
                    op.GetOpcodeStr().c_str(), op.GetOpMagic(), output->magic);
                return FAILED;
            }
        }
        if (op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
            // Reduce Acc 的输入和输出必须都是DDR类型
            for (auto &in : op.GetIOperands()) {
                if (in->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
                    ALOG_ERROR_F("[CubeProcess] %s[%d] has non-DDR input tenosr[%d]",
                        op.GetOpcodeStr().c_str(), op.GetOpMagic(), in->magic);
                    return FAILED;
                }
            }
            for (auto &out : op.GetOOperands()) {
                if (out->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
                    ALOG_ERROR_F("[CubeProcess] %s[%d] has non-DDR output tenosr[%d]",
                        op.GetOpcodeStr().c_str(), op.GetOpMagic(), out->magic);
                    return FAILED;
                }
            }
        }
    }
    ALOG_INFO_F("PreCheck for CubeProcess success.");
    return SUCCESS;
}

// verstion 2.0
Status CubeProcess::RunOnFunction(Function &function) {
    ALOG_INFO_F("===> start CubeProcess");
    if (EliminateReduceAcc(function) != SUCCESS) {
        ALOG_ERROR_F("Eliminate ReduceAcc failed.");
        return FAILED;
    }
    if (UpdateCubeOp(function) != SUCCESS) {
        ALOG_ERROR_F("Update Cube attr failed.");
        return FAILED;
    }
    if (DeadOperationEliminator::EliminateDeadOperation(function) != SUCCESS) {
        ALOG_ERROR_F("Eliminate dead operation failed in CommonOperationEliminate.");
        return FAILED;
    }
    ALOG_INFO_F("===> End CubeProcess");
    return SUCCESS;
}

bool CubeProcess::IsFloat(const std::shared_ptr<LogicalTensor> tensor) const {
    auto dataType = tensor->Datatype();
    if ((dataType == DT_FP16) || (dataType == DT_FP32) || (dataType == DT_BF16)) {
        return true;
    }
    return false;
}

bool CubeProcess::IsInt(const std::shared_ptr<LogicalTensor> tensor) const {
    auto dataType = tensor->Datatype();
    if ((dataType == DT_INT8) || (dataType == DT_INT16) || (dataType == DT_INT32)) {
        return true;
    }
    return false;
}

Status CubeProcess::EliminateReduceAcc(Function &function) {
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

Status CubeProcess::UpdateCubeOp(Function &function) {
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_A_MUL_B && op.GetOpcode() != Opcode::OP_A_MULACC_B) {
            continue;
        }
        for (auto &input : op.GetIOperands()) {
            if (input->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            // Align copy out GM with the reset GM
            if (op.GetOOperands().size() != 1) {
                ALOG_ERROR_F("%s[%d] has output num != 1.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
                return FAILED;
            }
            auto outputL0C = op.GetOOperands().front();
            auto chainEndCopyOut = *(outputL0C->GetConsumers().begin());
            // recursively find: MatMul -> L0C -> Copy_Out -> Gm
            while (chainEndCopyOut->GetOpcode() != Opcode::OP_COPY_OUT ) {
                outputL0C = chainEndCopyOut->GetOOperands().front();
                chainEndCopyOut = *(outputL0C->GetConsumers().begin());
            }
            if (chainEndCopyOut == nullptr || chainEndCopyOut->GetOOperands().size() != 1) {
                continue;
            }
            auto finalOutput = chainEndCopyOut->GetOOperands().front();
            if (finalOutput->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            if (function.IsFromInCast(input)) {
                ALOG_WARN_F("CubeProcess::UpdateCubeOp: OP_A_MUL_B iOperand tensor[%d] is incast.", input->GetMagic());
                continue;
            }
            input->tensor = finalOutput->tensor;
        }
        for (auto &output: op.GetOOperands()) {
            if (output->GetMemoryTypeOriginal() != MemoryType::MEM_L0C) {
                continue;
            }
            // force setting the  data type
            if (IsFloat(output)) {
                output->tensor->datatype = DataType::DT_FP32;
                continue;
            }
            if (IsInt(output)) {
                output->tensor->datatype = DataType::DT_INT32;
            }
        }
        if (UpdateCopyAttr(op) != SUCCESS) {
            ALOG_ERROR_F("Set Attr for %s[%d] failed.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status CubeProcess::AddL1CopyInAttr(
    const std::shared_ptr<LogicalTensor> input, int nzValue, int mValue, int kValue, int nValue) const {
    auto copyInOp = *(input->GetProducers().begin());
    auto tensorL0 = copyInOp->GetIOperands().front();
    auto L1CopyInOp = *(tensorL0->GetProducers().begin());
    if (L1CopyInOp->GetOpcode() == Opcode::OP_VIEW) {
        /*
        大包搬运场景
        gm -> L1_COPY_IN -> L1 ---> View ---> L1_partial ---> L1_TO_L0A ---> L0 ---> A_MUL_B
                              \ ---> View ---> L1_partial ---> L1_TO_L0A ---> L0 ---> A_MUL_B
        */
        tensorL0 = L1CopyInOp->GetIOperands().front();
        L1CopyInOp = *(tensorL0->GetProducers().begin());
    }
    if (L1CopyInOp->GetOpcode() != Opcode::OP_COPY_IN) {
        ALOG_DEBUG_F("L0 tesnor[%d] has invalid corresponding L1CopyInOp, please check.", input->magic);
        return FAILED;
    }
    L1CopyInOp->SetAttribute(COPY_IS_NZ, nzValue);
    ALOG_DEBUG_F("Update %s[%d] attr is_Nz: %d", L1CopyInOp->GetOpcodeStr().c_str(), L1CopyInOp->GetOpMagic(), nzValue);
    if (copyInOp->GetOpcode() == Opcode::OP_L1_TO_L0A) {
        L1CopyInOp->SetAttribute(L1_COPY_IN_OUTER, mValue);
        L1CopyInOp->SetAttribute(L1_COPY_IN_INNER, kValue);
        ALOG_DEBUG_F("OP_L1_TO_L0A: Outer: %d, Inner: %d", mValue, kValue);
        return SUCCESS;
    }
    if (copyInOp->GetOpcode() == Opcode::OP_L1_TO_L0B) {
        L1CopyInOp->SetAttribute(L1_COPY_IN_OUTER, kValue);
        L1CopyInOp->SetAttribute(L1_COPY_IN_INNER, nValue);
        ALOG_DEBUG_F("OP_L1_TO_L0B: Outer: %d, Inner: %d", kValue, nValue);
        return SUCCESS;
    }
    if (copyInOp->GetOpcode() == Opcode::OP_L1_TO_L0_AT) {
        L1CopyInOp->SetAttribute(L1_COPY_IN_OUTER, kValue);
        L1CopyInOp->SetAttribute(L1_COPY_IN_INNER, mValue);
        ALOG_DEBUG_F("OP_L1_TO_L0_AT: Outer: %d, Inner: %d", kValue, mValue);
        return SUCCESS;
    }
    if (copyInOp->GetOpcode() == Opcode::OP_L1_TO_L0_BT) {
        L1CopyInOp->SetAttribute(L1_COPY_IN_OUTER, nValue);
        L1CopyInOp->SetAttribute(L1_COPY_IN_INNER, kValue);
        ALOG_DEBUG_F("OP_L1_TO_L0_BT: Outer: %d, Inner: %d", nValue, kValue);
        return SUCCESS;
    }
    ALOG_DEBUG_F("invalid Cube input %d, produced by %s[%d]", input->GetMagic(), copyInOp->GetOpcodeStr().c_str(),
        copyInOp->GetOpMagic());
    return FAILED;
}

Status CubeProcess::AddL0cCopyOutAttr(const std::shared_ptr<LogicalTensor> output, int nzValue, int mValue, int nValue) const {
    for (auto &childOp : output->GetConsumers()) {
        if (childOp->GetOpcode() != Opcode::OP_COPY_OUT) {
            continue;
        }
        childOp->SetAttribute(COPY_IS_NZ, nzValue);
        childOp->SetAttribute(L0C_COPY_OUT_OUTER, mValue);
        childOp->SetAttribute(L0C_COPY_OUT_INNER, nValue);
        ALOG_DEBUG_F("Update %s[%d] attr is_Nz: %d, curH: %d, curW: %d",
            childOp->GetOpcodeStr().c_str(), childOp->GetOpMagic(), nzValue, mValue, nValue);
    }
    return SUCCESS;
}

Status CubeProcess::UpdateCopyAttr(Operation &op) const {
    int32_t nzAttr = op.GetIntAttribute(MATMUL_NZ_ATTR);
    auto mValue = (op.HasAttr(A_MUL_B_ACT_M)) ? op.GetIntAttribute(A_MUL_B_ACT_M) : 0;
    auto kValue = (op.HasAttr(A_MUL_B_ACT_K)) ? op.GetIntAttribute(A_MUL_B_ACT_K) : 0;
    auto nValue = (op.HasAttr(A_MUL_B_ACT_N)) ? op.GetIntAttribute(A_MUL_B_ACT_N) : 0;

    int aIsNz = nzAttr % 2;
    int bIsNz = (nzAttr >> 1) % 2;
    int cIsNz = (nzAttr >> 2) % 2;
    ALOG_DEBUG_F("Retrive %s[%d] attr done, aIsNz: %d, bIsNz: %d, cIsNz: %d, mValue: %d, kValue: %d, nValue: %d",
            op.GetOpcodeStr().c_str(), op.GetOpMagic(), aIsNz, bIsNz, cIsNz, mValue, kValue, nValue);
    for (auto &input : op.GetIOperands()) {
        if (input->GetMemoryTypeOriginal() == MemoryType::MEM_L0A) {
            if (AddL1CopyInAttr(input, aIsNz, mValue, kValue, nValue) != SUCCESS) {
                ALOG_ERROR_F("Set Attr for matrix A L1_COPY_IN of %s[%d] failed.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
                return FAILED;
            }
            continue;
        }
        if (input->GetMemoryTypeOriginal() == MemoryType::MEM_L0B) {
            if (AddL1CopyInAttr(input, bIsNz, mValue, kValue, nValue) != SUCCESS) {
                ALOG_ERROR_F("Set Attr for matrix B L1_COPY_IN of %s[%d] failed.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
                return FAILED;
            }
        }
    }
    for (auto &output : op.GetOOperands()) {
        if (AddL0cCopyOutAttr(output, cIsNz, mValue, nValue) != SUCCESS) {
            ALOG_ERROR_F("Set Attr for L0C_COPY_OUT of %s[%d] failed.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu