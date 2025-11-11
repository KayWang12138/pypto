/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

namespace npu::tile_fwk {
/*
resetDdr: A_MUL_B的清零后DDR输入，数据边表依赖
copyOutOp: 当前A_MUL_B链路最终搬出的L0C_Copy_Out
功能: 当前Matmul链路最终的CopyOut属性需要与清零时的CopyOut属性对齐
限制条件: 依赖前端使能切K场景下，Matmul的tile展开中显示对清零后的Gm按C矩阵的切分大小做切分
*/
void AlignCopyOutAttr(LogicalTensorPtr &resetDdr, Operation *copyOutOp) {
    if (resetDdr->GetProducers().size() == 1) {
        auto ddrResetCopyOut = *resetDdr->GetProducers().begin();
        if (ddrResetCopyOut->GetOpcode() != Opcode::OP_COPY_OUT) {
            APASS_LOG_ERROR_F("PreGraphProcess:CubeProcess", "Operation", "DDR reset Op requires to be OP_COPY_OUT, but %s[%d]; Please check the Opcode.", 
                ddrResetCopyOut->GetOpcodeStr().c_str(), ddrResetCopyOut->GetOpMagic());
            return;
        }
        auto ddrResetCopyOutAttr = std::static_pointer_cast<CopyOpAttribute>(ddrResetCopyOut->GetOpAttribute());
        auto L0CCopyOutAttr = std::static_pointer_cast<CopyOpAttribute>(copyOutOp->GetOpAttribute());
        ddrResetCopyOutAttr->SetRawShape(L0CCopyOutAttr->GetRawShape());
        ddrResetCopyOutAttr->SetToOffset(L0CCopyOutAttr->GetToOffset());
    }
}

Status CubeProcess::AddL1CopyInAttr(
    const std::shared_ptr<LogicalTensor> input, int nzValue, int mValue, int kValue, int nValue) const {
    auto copyInOp = *(input->GetProducers().begin());
    auto tensorL0 = copyInOp->GetIOperands().front();
    auto L1CopyInOp = *(tensorL0->GetProducers().begin());
    if (L1CopyInOp->GetOpcode() == Opcode::OP_VIEW || L1CopyInOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
        /*
        1. View 对应大包搬运场景
        gm -> L1_COPY_IN -> L1 ---> View ---> L1_partial ---> L1_TO_L0A ---> L0 ---> A_MUL_B
                              \ ---> View ---> L1_partial ---> L1_TO_L0A ---> L0 ---> A_MUL_B
        2. Assemble 对应 Gather On L1 场景
        */
        tensorL0 = L1CopyInOp->GetIOperands().front();
        L1CopyInOp = *(tensorL0->GetProducers().begin());
    }
    /*L0C copy L1*/
    if(L1CopyInOp->GetOpcode() == Opcode::OP_L0C_COPY_L1) {
        return SUCCESS;
    }
    if (L1CopyInOp->GetOpcode() != Opcode::OP_COPY_IN && L1CopyInOp->GetOpcode() != Opcode::OP_GATHER_IN_L1) {
        APASS_LOG_ERROR_F("PreGraphProcess:CubeProcess", "Operation", "L0 tesnor[%d] has invalid corresponding L1CopyInOp, please check.", input->magic);
        return FAILED;
    }
    L1CopyInOp->SetAttribute(COPY_IS_NZ, nzValue);
    APASS_LOG_DEBUG_F("PreGraphProcess:CubeProcess", "Operation", "Update %s[%d] attr is_Nz: %d", L1CopyInOp->GetOpcodeStr().c_str(), L1CopyInOp->GetOpMagic(), nzValue);
    if (copyInOp->GetOpcode() == Opcode::OP_L1_TO_L0A) {
        L1CopyInOp->SetAttribute(L1_COPY_IN_OUTER, mValue);
        L1CopyInOp->SetAttribute(L1_COPY_IN_INNER, kValue);
        APASS_LOG_DEBUG_F("PreGraphProcess:CubeProcess", "Operation", "OP_L1_TO_L0A: Outer: %d, Inner: %d.", mValue, kValue);
        return SUCCESS;
    }
    if (copyInOp->GetOpcode() == Opcode::OP_L1_TO_L0B) {
        L1CopyInOp->SetAttribute(L1_COPY_IN_OUTER, kValue);
        L1CopyInOp->SetAttribute(L1_COPY_IN_INNER, nValue);
        APASS_LOG_DEBUG_F("PreGraphProcess:CubeProcess", "Operation", "OP_L1_TO_L0B: Outer: %d, Inner: %d.", kValue, nValue);
        return SUCCESS;
    }
    if (copyInOp->GetOpcode() == Opcode::OP_L1_TO_L0_AT) {
        L1CopyInOp->SetAttribute(L1_COPY_IN_OUTER, kValue);
        L1CopyInOp->SetAttribute(L1_COPY_IN_INNER, mValue);
        APASS_LOG_DEBUG_F("PreGraphProcess:CubeProcess", "Operation", "OP_L1_TO_L0_AT: Outer: %d, Inner: %d.", kValue, mValue);
        return SUCCESS;
    }
    if (copyInOp->GetOpcode() == Opcode::OP_L1_TO_L0_BT) {
        L1CopyInOp->SetAttribute(L1_COPY_IN_OUTER, nValue);
        L1CopyInOp->SetAttribute(L1_COPY_IN_INNER, kValue);
        APASS_LOG_DEBUG_F("PreGraphProcess:CubeProcess", "Operation", "OP_L1_TO_L0_BT: Outer: %d, Inner: %d.", nValue, kValue);
        return SUCCESS;
    }
    APASS_LOG_ERROR_F("PreGraphProcess:CubeProcess", "Operation", "Invalid Cube input %d, produced by %s[%d].", input->GetMagic(), copyInOp->GetOpcodeStr().c_str(),
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
        APASS_LOG_DEBUG_F("PreGraphProcess:CubeProcess", "Operation", "Update %s[%d] attr is_Nz: %d, curH: %d, curW: %d.",
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
    APASS_LOG_DEBUG_F("PreGraphProcess:CubeProcess", "Operation", "Retrive %s[%d] attr done, aIsNz: %d, bIsNz: %d, cIsNz: %d, mValue: %d, kValue: %d, nValue: %d.",
            op.GetOpcodeStr().c_str(), op.GetOpMagic(), aIsNz, bIsNz, cIsNz, mValue, kValue, nValue);
    for (auto &input : op.GetIOperands()) {
        if (input->GetMemoryTypeOriginal() == MemoryType::MEM_L0A) {
            if (AddL1CopyInAttr(input, aIsNz, mValue, kValue, nValue) != SUCCESS) {
                APASS_LOG_ERROR_F("PreGraphProcess:CubeProcess", "Operation", "Set Attr for matrix A L1_COPY_IN of %s[%d] failed.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
                return FAILED;
            }
            continue;
        }
        if (input->GetMemoryTypeOriginal() == MemoryType::MEM_L0B) {
            if (AddL1CopyInAttr(input, bIsNz, mValue, kValue, nValue) != SUCCESS) {
                APASS_LOG_ERROR_F("PreGraphProcess:CubeProcess", "Operation", "Set Attr for matrix B L1_COPY_IN of %s[%d] failed.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
                return FAILED;
            }
        }
    }
    for (auto &output : op.GetOOperands()) {
        if (AddL0cCopyOutAttr(output, cIsNz, mValue, nValue) != SUCCESS) {
            APASS_LOG_ERROR_F("PreGraphProcess:CubeProcess", "Operation", "Set Attr for L0C_COPY_OUT of %s[%d] failed.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status CubeProcess::CheckValidCube(const Operation &op) {
    /* 校验有且只有一个输出 */
    if (op.GetOOperands().size() != 1) {
        APASS_LOG_ERROR_F("PreGraphProcess:CubeProcess", "Tensor", "%s[%d] has output num != 1; Please check ooperands.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
        return FAILED;
    }
    /* 校验输出: 1. 非空，2. mem类型为L0C, 3.有消费者 */
    auto outputL0C = op.GetOOperands().front();
    if (outputL0C == nullptr) {
        APASS_LOG_ERROR_F("PreGraphProcess:CubeProcess", "Tensor", "%s[%d] output is nullptr; Please check outputL0C.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
        return FAILED;
    }
    if (outputL0C->GetMemoryTypeOriginal() != MemoryType::MEM_L0C) {
        APASS_LOG_ERROR_F("PreGraphProcess:CubeProcess", "Tensor", "%s[%d] output is NOT L0C; Please check outputL0C MemoryType.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
        return FAILED;
    }
    if (outputL0C->GetConsumers().size() < 1) {
        APASS_LOG_ERROR_F("PreGraphProcess:CubeProcess", "Tensor", "%s[%d] output has EMPTY consumers; Please check outputL0C consumer size.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
        return FAILED;
    }
    return SUCCESS;
}

Status CubeProcess::UpdateL0cDtype(Operation &op) {
    std::pair<DataType, DataType> inputDtypes = std::make_pair(DataType::DT_FP16, DataType::DT_FP16);
    for (auto &input : op.GetIOperands()) {
        if (input->GetMemoryTypeOriginal() == MemoryType::MEM_L0A) {
            inputDtypes.first = input->Datatype();
        } else if (input->GetMemoryTypeOriginal() == MemoryType::MEM_L0B) {
            inputDtypes.second = input->Datatype();
        }
    }
    if (supportDtypeMap.count(inputDtypes)) {
        DataType outDtype = supportDtypeMap.at(inputDtypes);
        for (auto &output: op.GetOOperands()) {
            output->tensor->datatype = outDtype;
        }
        return SUCCESS;
    } else {
        APASS_LOG_ERROR_F("PreGraphProcess:CubeProcess", "Operation", "%s[%d] has unsupport input dtypes (L0A: %s, L0B: %s), update L0C dtype Failed.",
            op.GetOpcodeStr().c_str(), op.GetOpMagic(), 
            BriefDataType2String(inputDtypes.first).c_str(),
            BriefDataType2String(inputDtypes.second).c_str());
        return FAILED;
    }
}

std::pair<Operation *, Operation *> CubeProcess::GetLastMmCopyOut(Operation &op) {
    auto outputL0C = op.GetOOperands().front();
    auto chainEndCopyOut = *(outputL0C->GetConsumers().begin());
    // recursively find: MatMul -> L0C -> Copy_Out -> Gm
    while (chainEndCopyOut->GetOpcode() != Opcode::OP_COPY_OUT) {
        outputL0C = chainEndCopyOut->GetOOperands().front();
        chainEndCopyOut = *(outputL0C->GetConsumers().begin());
    }
    if (chainEndCopyOut == nullptr) {
        APASS_LOG_ERROR_F("PreGraphProcess:CubeProcess", "Tensor", "%s[%d] has nullptr L0C_Copy_Out; Please check chainEndCopyOut.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
        return {nullptr, nullptr};
    }
    if (chainEndCopyOut->GetOOperands().size() != 1) {
        APASS_LOG_ERROR_F("PreGraphProcess:CubeProcess", "Tensor", "%s[%d] has more than ONE outputs.", chainEndCopyOut->GetOpcodeStr().c_str(), chainEndCopyOut->GetOpMagic());
        return {nullptr, nullptr};
    }
    auto finalOutput = chainEndCopyOut->GetOOperands().front();
    if (finalOutput->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
        APASS_LOG_ERROR_F("PreGraphProcess:CubeProcess", "Tensor", "%s[%d] has invlid output memType: %s, expect: MEM_DEVICE_DDR.",
            chainEndCopyOut->GetOpcodeStr().c_str(), chainEndCopyOut->GetOpMagic(),
            MemoryTypeToString(finalOutput->GetMemoryTypeOriginal()).c_str());
        return {nullptr, nullptr};
    }
    // Copy_Out 的上游Op即为最后一个Matmul
    auto lastMm = *chainEndCopyOut->ProducerOps().begin();
    return {lastMm, chainEndCopyOut};
}

Status CubeProcess::ReconnectGraph(Operation &mulOp, Operation *copyOutOp) {
    for (auto &input : mulOp.GetIOperands()) {
        if (input->GetMemoryTypeOriginal() == MemoryType::MEM_FIX_QUANT_PRE) {
            mulOp.EraseInput(input);
            copyOutOp->iOperand.emplace_back(input);
            input->RemoveConsumer(mulOp);
            input->AddConsumer(copyOutOp);
        }
        TransferAttr(mulOp, copyOutOp);
    }
    return SUCCESS;
}

Status CubeProcess::TransferAttr(Operation &mulOp, Operation *copyOutOp) {
    auto scaleValue = (mulOp.HasAttr(A_MUL_B_SCALE_ATTR)) ? mulOp.GetElementAttribute(A_MUL_B_SCALE_ATTR) : Element(DataType::DT_UINT64, 0);
    auto reluType = (mulOp.HasAttr(A_MUL_B_RELU_ATTR)) ? mulOp.GetIntAttribute(A_MUL_B_RELU_ATTR) : 0;
    copyOutOp->SetAttribute(A_MUL_B_SCALE_ATTR, scaleValue);
    copyOutOp->SetAttribute(A_MUL_B_RELU_ATTR, reluType);
    return SUCCESS;
}

Status CubeProcess::UpdateCubeOp(Function &function) {
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_A_MUL_B && op.GetOpcode() != Opcode::OP_A_MULACC_B) {
            continue;
        }
        if (CheckValidCube(op) != SUCCESS) {
            APASS_LOG_ERROR_F("PreGraphProcess:CubeProcess", "Operation", "%s[%d] is invalid.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
            return FAILED;
        }
        auto lastMmCopyOut = GetLastMmCopyOut(op);
        auto lastMm = lastMmCopyOut.first;
        auto chainEndCopyOut = lastMmCopyOut.second;
        if (lastMm == nullptr || chainEndCopyOut == nullptr) {
            APASS_LOG_ERROR_F("PreGraphProcess:CubeProcess", "Operation", "Get the last MatMul and L0C_Copy_Out for %s[%d] failed.", 
                op.GetOpcodeStr().c_str(), op.GetOpMagic());
            return FAILED;
        }
        for (auto &input : op.GetIOperands()) {
            if (input->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            // Align copy out GM with the reset GM
            if (function.IsFromInCast(input)) {
                APASS_LOG_WARN_F("PreGraphProcess:CubeProcess", "Operation", "PreGraphProcess:CubeProcess::UpdateCubeOp: OP_A_MUL_B iOperand tensor[%d] is incast.", input->GetMagic());
                continue;
            }
            auto finalOutput = chainEndCopyOut->GetOOperands().front();
            input->tensor = finalOutput->tensor;
            /*
            强制要求当前Matmul链路输出的Gm仅存在一个清零的Op，暂时通过指定清零和ReduceAcc使用的vec tilesize与tileM x tileN相同
            后续通过前端提供使能切K的API保证
            */
            AlignCopyOutAttr(input, chainEndCopyOut);
        }

        if (UpdateL0cDtype(op) != SUCCESS) {
            APASS_LOG_ERROR_F("PreGraphProcess:CubeProcess", "Operation", "Update L0C dtype for %s[%d] failed.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
            return FAILED;
        }
        if (UpdateCopyAttr(op) != SUCCESS) {
            APASS_LOG_ERROR_F("PreGraphProcess:CubeProcess", "Operation", "Set Attr for %s[%d] failed.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
            return FAILED;
        }

        if (op.GetOpcode() == Opcode::OP_A_MUL_B) {
            // FixPipe支持随路量化图重连 & MUL -> L0C_COPY_OUT属性传递
            ReconnectGraph(op, chainEndCopyOut);
        }
    }
    return SUCCESS;
}
} // namespace npu::tile_fwk