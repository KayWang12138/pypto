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
 * \file codegen_preproc.cpp
 * \brief
 */

#include "interface/function/function.h"
#include "interface/operation/opcode.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/utils/common.h"
#include "passes/pass_interface/pass.h"
#include "codegen_preproc.h"

namespace npu::tile_fwk {
// only save general gm input/output, not contain spill-out scene
bool CodegenPreprocPass::IsNeedSave(const Operation &op) const {
    return OpcodeManager::Inst().IsCopyInOrOut(op.GetOpcode()) && (!op.IsNeedStackGM());
}

// only used in DYNAMIC_LOOP_PATH scene
Status CodegenPreprocPass::SaveGmTensorParamIdxToOp(Function &func) const {
    if (!func.IsUnderDynamicFunction()) {
        return SUCCESS;
    }

    std::map<int, std::vector<Operation *>> gmParamInCallFunc;
    for (auto &subProgram : func.rootFunc_->programs_) {
        gmParamInCallFunc.clear();
        for (auto &op : subProgram.second->Operations()) {
            if (IsNeedSave(op)) {
                const std::shared_ptr<OpAttribute> &attr = op.GetOpAttribute();
                if (attr == nullptr) {
                    ALOG_ERROR_F("Copy In attr is null, SaveGmTensorParamIdxToOp failed!");
                    return FAILED;
                }
                std::shared_ptr<CopyOpAttribute> copyAttr = std::static_pointer_cast<CopyOpAttribute>(attr);
                int addrPos;
                if (IsCopyIn(op.GetOpcode()))
                    addrPos = op.GetIOpAttrOffset(0);
                else
                    addrPos = op.GetOOpAttrOffset(0);
                gmParamInCallFunc[addrPos].emplace_back(&op);
            }
        }
        ALOG_INFO_F("%d:%sgmParamInCallFunc size: %zu", __LINE__, __FUNCTION__, gmParamInCallFunc.size());
        int tensorParamIdx{0};
        for (auto param : gmParamInCallFunc) {
            for (auto op : param.second) {
                op->SetAttribute("GmTensorParamIdxInCallFunc", tensorParamIdx);
                ++tensorParamIdx;
            }
        }
    }
    return SUCCESS;
}

void CodegenPreprocPass::CombineTailAxis(std::vector<int> &shape, size_t shapeSize) const {
    shape[shapeSize - 1] = shape[shapeSize - 1] * shape[shapeSize - NUM2];
    shape[shapeSize - NUM2] = 1;
}

Status CodegenPreprocPass::ProcessAxis(Operation &op, std::vector<bool> attr, bool isInput) const {
    LogicalTensors operands{};
    if (isInput) {
        operands = op.GetIOperands();
    } else {
        operands = op.GetOOperands();
    }
    if (attr.size() < operands.size()) {
        for (size_t i = 0; i < operands.size() - attr.size(); ++i) {
            attr.emplace_back(false);
        }
    }
    if (attr.size() != operands.size()) {
        ALOG_ERROR_F("attr size is not equal to operands size, ProcessAxis failed!");
        return FAILED;
    }
    for (size_t i = 0; i < operands.size(); ++i) {
        if (attr[i]) {
            size_t shapeSize = operands[i]->shape.size();
            CombineTailAxis(operands[i]->shape, shapeSize);
            CombineTailAxis(operands[i]->oriShape, shapeSize);
            CombineTailAxis(operands[i]->tensor->rawshape, shapeSize);
        }
    }
    return SUCCESS;
}

Status CodegenPreprocPass::ForceCombineAxis(Function &func) const {
    for (auto &subProgram : func.rootFunc_->programs_) {
        for (auto &op : subProgram.second->Operations()) {
            if (op.HasAttr(OP_ATTR_PREFIX + "input_combine_axis")) {
                std::vector<bool> attrIn;
                op.GetAttr(OP_ATTR_PREFIX + "input_combine_axis", attrIn);
                op.SetAttribute(OpAttributeKey::inputCombineAxisDone, true);
                if (ProcessAxis(op, attrIn, true) != SUCCESS) { ALOG_ERROR_F("ForceCombineAxis failed at function ProcessAxis(input)!"); return FAILED; }
                if (op.GetOpcode() == Opcode::OP_COPY_OUT) {
                    op.SetAttribute(OpAttributeKey::outputCombineAxisDone, true);
                    auto output = op.GetOOperands()[0];
                    CombineTailAxis(output->tensor->rawshape, output->tensor->rawshape.size());
                }
            }
            if (op.HasAttr(OP_ATTR_PREFIX + "output_combine_axis")) {
                std::vector<bool> attrOut;
                op.GetAttr(OP_ATTR_PREFIX + "output_combine_axis", attrOut);
                op.SetAttribute(OpAttributeKey::outputCombineAxisDone, true);
                if (ProcessAxis(op, attrOut, false) !=SUCCESS) { ALOG_ERROR_F("ForceCombineAxis failed at function ProcessAxis(out)!"); return FAILED; }
                if (op.GetOpcode() == Opcode::OP_COPY_IN) {
                    op.SetAttribute(OpAttributeKey::inputCombineAxisDone, true);
                    auto input = op.GetIOperands()[0];
                    CombineTailAxis(input->tensor->rawshape, input->tensor->rawshape.size());
                }
            }
        }
    }
    return SUCCESS;
}

Status CodegenPreprocPass::RunOnFunction(Function &function) {
    if (SaveGmTensorParamIdxToOp(function) != SUCCESS) {
        ALOG_ERROR_F("CodegenPreprocPass RunOnFunction failed at function SaveGmTensorParamIdxToOp!");
        return FAILED;
    }
    if (ForceCombineAxis(function) != SUCCESS) {
        ALOG_ERROR_F("CodegenPreprocPass RunOnFunction failed at function ForceCombineAxis!");
        return FAILED;
    }
    return SUCCESS;
}

} // namespace npu::tile_fwk
