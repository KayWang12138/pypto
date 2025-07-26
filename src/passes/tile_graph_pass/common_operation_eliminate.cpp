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
 * \file common_operation_eliminate.cpp
 * \brief
 */

#include "passes/tile_graph_pass/common_operation_eliminate.h"
#include <unordered_map>
#include "interface/tensor/logical_tensor.h"
#include "passes/tile_graph_pass/dead_operation_eliminate.h"

namespace npu::tile_fwk {
Status CommonOperationEliminate::RunOnFunction(Function &function)
{
    for (auto &op : function.Operations().DuplicatedOpList()) {
        if (OpAlreadyExist(op)) {
            op->SetAsDeleted();
        }
    }
    function.EraseOperations();
    if (DeadOperationEliminator::EliminateDeadOperation(function) != SUCCESS) {
        ALOG_ERROR_F("Eliminate dead operation failed in CommonOperationEliminate.");
        return FAILED;
    }
    return SUCCESS;
}

Status CommonOperationEliminate::PreCheck(Function &function)
{
    ALOG_INFO_F("PreCheck for CommonOperationEliminate.");
    operationCache_.clear();
    for (auto &op : function.Operations().DuplicatedOpList()) {
        if (op->GetOpAttribute() != nullptr) {
            size_t fromOffsetSize = -1;
            if (auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(op->GetOpAttribute().get())) {
                auto &fromOffset = viewOpAttribute->GetFromOffset();
                fromOffsetSize = fromOffset.size();
            } else if (auto copyOpAttribute = dynamic_cast<CopyOpAttribute*>(op->GetOpAttribute().get())) {
                if (copyOpAttribute->IsCopyOut()) {
                    continue;
                }
                auto [fromOffset, memType] = copyOpAttribute->GetCopyInAttr();
                (void)memType;
                fromOffsetSize = fromOffset.size();
            } else {
                continue;
            }
            auto& ioperands = op->GetIOperands();
            if (ioperands.size() != 1) {
                ALOG_ERROR_F("View or Copy_In Operation %d with not one input operand.", op->GetOpMagic());
                return FAILED;
            }
            if (ioperands.front()->offset.size() != fromOffsetSize) {
                ALOG_ERROR_F("View or Copy_In Operation %d with mismatch input offset shape.", op->GetOpMagic());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status CommonOperationEliminate::PostCheck(Function &function)
{
    ALOG_INFO_F("PostCheck for CommonOperationEliminate.");
    operationCache_.clear();
    for (auto &op : function.Operations().DuplicatedOpList()) {
        if (OpAlreadyExist(op)) {
            ALOG_ERROR_F("Redundant Operation %d still exist after CommonOperationEliminate.", op->GetOpMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}

Operation *CommonOperationEliminate::OperationExist(Operation *operation)
{
    auto &inputsMemType = OpcodeManager::Inst().GetInputsMemType(operation->GetOpcode());
    auto &outputsMemType = OpcodeManager::Inst().GetOutputsMemType(operation->GetOpcode());
    OpCalcType opCalcType = OpcodeManager::Inst().GetOpCalcType(operation->GetOpcode());
    bool inputCheck = inputsMemType.size() == 1 && inputsMemType[0] == MemoryType::MEM_L1;
    bool calcTypeCheck = opCalcType == OpCalcType::MOVE_LOCAL || opCalcType == OpCalcType::MOVE_IN;
    bool outputCheck = outputsMemType.size() == 1 && outputsMemType[0] != MemoryType::MEM_L1;
    if (inputCheck && calcTypeCheck && outputCheck) { // copy from L1 to L0
        return nullptr;
    }
    if (operation->GetOpcode() == Opcode::OP_VIEW) {
        return nullptr;
    }
    if (operationCache_.count(operation->ComputeHash()) != 0) {
        return operationCache_[operation->ComputeHash()];
    } else {
        operationCache_.insert({operation->ComputeHash(), operation});
        return nullptr;
    }
}

void CommonOperationEliminate::UpdateView(ViewOpAttribute *viewOpAttribute,
                                          const std::shared_ptr<LogicalTensor> oldtensor,
                                          const std::shared_ptr<LogicalTensor> newtensor) const
{
    auto &fromOffset = viewOpAttribute->GetFromOffset();
    for (size_t j = 0; j < fromOffset.size(); j++) {
        fromOffset[j] -= oldtensor->offset[j] - newtensor->offset[j];
    }
}

void CommonOperationEliminate::UpdateCopy(CopyOpAttribute *copyOpAttribute,
                                          const std::shared_ptr<LogicalTensor> oldtensor,
                                          const std::shared_ptr<LogicalTensor> newtensor) const
{
    if (!copyOpAttribute->IsCopyOut()) {
        auto [fromOffset, memType] = copyOpAttribute->GetCopyInAttr();
        (void)memType;
        for (size_t j = 0; j < fromOffset.size(); j++) {
            fromOffset[j] -= oldtensor->offset[j] - newtensor->offset[j];
        }
        copyOpAttribute->SetFromOffset(fromOffset);
    }
}

bool CommonOperationEliminate::OpAlreadyExist(Operation *op)
{
    auto existOp = OperationExist(op);
    if (existOp == nullptr || op->GetOOperands().size() == 0 || existOp->GetOOperands().size() == 0) {
        return false;
    }
    if (op->GetOOperands().front()->shape == existOp->GetOOperands().front()->shape) {
        auto oldtensor = op->GetOOperands().front();
        if (oldtensor->GetConsumers().size() == 0) {
            return false;
        }
        auto newtensor = existOp->GetOOperands().front();
        if (newtensor->GetMagic() == oldtensor->GetMagic()) {
            ALOG_DEBUG_F("In CommonOperationEliminate, Operation %d is marked as redundant.", op->GetOpMagic());
            return true;
        }
        auto consumers = oldtensor->GetConsumers();
        for (auto &cur : consumers) {
            if (cur->GetOpAttribute() != nullptr) {
                if (auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(cur->GetOpAttribute().get())) {
                    // VIEW操作的offset要相应被修改。
                    UpdateView(viewOpAttribute, oldtensor, newtensor);
                } else if (auto copyOpAttribute = dynamic_cast<CopyOpAttribute*>(cur->GetOpAttribute().get())) {
                    // CopyIn操作的offset要相应被修改。
                    UpdateCopy(copyOpAttribute, oldtensor, newtensor);
                }
            }
            cur->ReplaceInput(newtensor, oldtensor);
        }
        auto producers = oldtensor->GetProducers();
        for (auto &cur : producers) {
            cur->ReplaceOutput(newtensor, oldtensor);
        }
        ALOG_DEBUG_F("In CommonOperationEliminate, Operation %d is marked as redundant.", op->GetOpMagic());
        return true;
    }
    return false;
}
}  // namespace npu::tile_fwk