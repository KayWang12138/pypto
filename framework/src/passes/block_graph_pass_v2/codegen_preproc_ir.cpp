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
 * \file codegen_preproc.cpp
 * \brief
 */

#include "interface/function/function.h"
#include "interface/operation/opcode.h"
#include "interface/tensor/logical_tensor.h"
#include "ir/value.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/utils/common.h"
#include "passes/pass_interface/pass.h"
#include "block_utils/block_pass_utils.h"
#include "codegen_preproc_ir.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "CodegenPreprocIR"

namespace npu {
namespace tile_fwk {
// only save general gm input/output, not contain spill-out scene
bool CodegenPreprocIR::IsNeedSave(const pto::Operation &op) const {
    return BlockPassUtils::IsCopyInOrOut(op.GetOpcode()) && (!op.IsNeedStackGM());
}

// only used in DYNAMIC_LOOP_PATH scene
Status CodegenPreprocIR::SaveGmTensorParamIdxToOp(pto::BlockFunction &func) const {

    std::map<int, std::vector<pto::Operation *>> gmParamInCallFunc;
    
    auto operations = BlockPassUtils::GetBlockFunctionOperations(func);

    gmParamInCallFunc.clear();
    for (auto opPtr : operations) {
        auto &op = *opPtr;
        if (IsNeedSave(op)) {
            int coaIndex = BlockPassUtils::IsCopyIn(op.GetOpcode()) ? op.GetIOpAttrOffset(0) : op.GetOOpAttrOffset(0);
            gmParamInCallFunc[coaIndex].emplace_back(&op);
        }
        // if (op.GetOpcode() == pto::Opcode::OP_GATHER_IN_L1) {
        //     gmParamInCallFunc[op.GetIOpAttrOffset(0)].emplace_back(&op);
        //     gmParamInCallFunc[op.GetIOpAttrOffset(1)].emplace_back(&op);
        //     gmParamInCallFunc[op.GetIOpAttrOffset(2)].emplace_back(&op);
        // }
        // if (op.GetOpcode() == pto::Opcode::OP_GATHER_IN_UB) {
        //     gmParamInCallFunc[op.GetIOpAttrOffset(0)].emplace_back(&op);
        //     gmParamInCallFunc[op.GetIOpAttrOffset(1)].emplace_back(&op);
        //     gmParamInCallFunc[op.GetIOpAttrOffset(2)].emplace_back(&op);
        // }
        // if (op.GetOpcode() == pto::Opcode::OP_GATHER) {
        //     gmParamInCallFunc[op.GetIOpAttrOffset(0)].emplace_back(&op);
        //     gmParamInCallFunc[op.GetIOpAttrOffset(1)].emplace_back(&op);
        // }
        // if (op.GetOpcode() == pto::Opcode::OP_LOAD) {
        //     int addrPos = op.GetIOpAttrOffset(0);
        //     gmParamInCallFunc[addrPos].emplace_back(&op);
        // }
        // Q : 缺失 OP_GATHER_IN_L1, OP_GATHER_IN_UB, OP_GATHER, OP_LOAD 等 OP 还是不需要？
    }
    APASS_LOG_INFO_F(Elements::Operation, "%d:%sgmParamInCallFunc size: %zu", __LINE__, __FUNCTION__, gmParamInCallFunc.size());
    int tensorParamIdx{0};

    for (auto param : gmParamInCallFunc) {
        for (auto op : param.second) {
            (void)op;
            //op->SetAttribute("GmTensorParamIdxInCallFunc", tensorParamIdx);
            //Q : 缺失 SetAttribute 方法, 或者说现在的 ATTR 系统应该怎么使用?
            ++tensorParamIdx;
        }
    }
    
    return SUCCESS;
}

std::vector<int64_t> CodegenPreprocIR::CombineTailAxis(const std::vector<int64_t> &shape) const {
    std::vector<int64_t> newshape = shape;
    if (newshape.size() < NUM2) {
        return newshape;
    }
    const size_t shapeSize = newshape.size();
    newshape[shapeSize - 1] = newshape[shapeSize - 1] * newshape[shapeSize - NUM2];
    newshape[shapeSize - NUM2] = 1;
    return newshape;
}

std::vector<pto::ScalarValuePtr> CodegenPreprocIR::CombineLastAxis(const std::vector<pto::ScalarValuePtr> &shape) const {
    std::vector<pto::ScalarValuePtr> newshape = shape;
    if (newshape.size() < NUM2) {
        return newshape;
    }
    
    // 将 ScalarValuePtr 转换为 SymbolicScalar 进行运算
    std::vector<SymbolicScalar> symbolicShape;
    symbolicShape.reserve(shape.size());
    for (const auto &svPtr : shape) {
        if (!svPtr) {
            throw std::runtime_error("ScalarValuePtr is null");
        }
        const auto &sv = *svPtr;
        if (sv.GetScalarValueKind() == pto::ScalarValueKind::Immediate) {
            int64_t value = sv.GetInt64Value();
            symbolicShape.emplace_back(value);
        } else {
            std::string name = sv.GetSSAName();
            symbolicShape.emplace_back(name);
        }
    }
    
    // 执行合并操作
    const size_t shapeSize = symbolicShape.size();
    symbolicShape[shapeSize - 1] = symbolicShape[shapeSize - 1] * symbolicShape[shapeSize - NUM2];
    symbolicShape[shapeSize - NUM2] = SymbolicScalar(1);
    
    // 将 SymbolicScalar 转换回 ScalarValuePtr
    std::vector<pto::ScalarValuePtr> result;
    result.reserve(symbolicShape.size());
    for (const auto &ss : symbolicShape) {
        if (ss.IsImmediate() && ss.ConcreteValid()) {
            // 立即值：创建 Immediate 类型的 ScalarValuePtr
            result.emplace_back(std::make_shared<pto::ScalarValue>(ss.Concrete()));
        } else if (ss.IsSymbol()) {
            // 符号值：创建 Symbolic 类型的 ScalarValuePtr
            std::string symbolName = ss.Raw()->GetSymbolName();
            result.emplace_back(std::make_shared<pto::ScalarValue>(
                pto::DataType::INT64, symbolName, pto::ScalarValueKind::Symbolic));
        } else {
            // 表达式：使用 Dump() 作为符号名
            std::string exprName = ss.Dump();
            result.emplace_back(std::make_shared<pto::ScalarValue>(
                pto::DataType::INT64, exprName, pto::ScalarValueKind::Symbolic));
        }
    }
    
    return result;
}

Status CodegenPreprocIR::ProcessAxis(pto::Operation &op, std::vector<bool> attr, bool isInput) const {

    std::vector<pto::ValuePtr> &operands = isInput ? op.GetIOperands() : op.GetOOperands();
    if (attr.size() < operands.size()) {
        for (size_t i = 0; i < operands.size() - attr.size(); ++i) {
            attr.emplace_back(false);
        }
    }
    if (attr.size() != operands.size()) {
        APASS_LOG_ERROR_F(Elements::Operation, "%d %s attr size(%zu) is not equal to operands size(%zu), ProcessAxis failed.", op.GetID(), pto::GetOpcodeName(op.GetOpcode()).c_str(), attr.size(), operands.size());
        return FAILED;
    }
    for (size_t i = 0; i < operands.size(); ++i) {
        if (attr[i]) {
            auto tileValue = std::dynamic_pointer_cast<pto::TileValue>(operands[i]);
            if (!tileValue) {
                APASS_LOG_ERROR_F(Elements::Operation, "%d %s operands[%zu] is not a tile value, ProcessAxis failed.", op.GetID(), pto::GetOpcodeName(op.GetOpcode()).c_str(), i);
                continue;
            }
            tileValue->SetShape(CombineTailAxis(tileValue->GetShape()));
            // operands[i]->oriShape = CombineTailAxis(operands[i]->oriShape);
            // operands[i]->tensor->rawshape = CombineTailAxis(operands[i]->tensor->rawshape);
            // Q : 不需要 oriShape 和 tensor->rawshape 的 CombineTailAxis ?

            if (ConfigManager::Instance().GetOperationConfig(KEY_FORCE_COMBINE_AXIS, false)) {
                tileValue->SetValidShape(CombineLastAxis(tileValue->GetValidShape()));
            }
        }
    }
    return SUCCESS;
}

Status CodegenPreprocIR::ForceCombineAxis(pto::BlockFunction &func) const {
    
    auto operations = BlockPassUtils::GetBlockFunctionOperations(func);

    for (auto &opPtr : operations) {
        auto &op = *opPtr;
        if (op.HasAttr(OP_ATTR_PREFIX + "input_combine_axis")) {
            std::vector<bool> attrIn;
            op.GetAttr(OP_ATTR_PREFIX + "input_combine_axis", attrIn);
            op.SetAttribute(OpAttributeKey::inputCombineAxisDone, true);
            if (ProcessAxis(op, attrIn, true) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "ForceCombineAxis failed at function ProcessAxis(input) for subProgram(%lu).", subProgram.first);
                return FAILED;
            }
            if (op.GetOpcode() == pto::Opcode::OP_COPY_OUT) {
                op.SetAttribute(OpAttributeKey::outputCombineAxisDone, true);
                auto output = op.GetOOperands()[0];
                CombineTailAxis(output->tensor->rawshape, output->tensor->rawshape.size());
            }
        }
        if (op.HasAttr(OP_ATTR_PREFIX + "output_combine_axis")) {
            std::vector<bool> attrOut;
            op.GetAttr(OP_ATTR_PREFIX + "output_combine_axis", attrOut);
            op.SetAttribute(OpAttributeKey::outputCombineAxisDone, true);
            if (ProcessAxis(op, attrOut, false) !=SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "ForceCombineAxis failed at function ProcessAxis(out) for subProgram(%lu).", subProgram.first);
                return FAILED;
            }
            if (op.GetOpcode() == pto::Opcode::OP_COPY_IN) {
                op.SetAttribute(OpAttributeKey::inputCombineAxisDone, true);
                auto input = op.GetIOperands()[0];
                CombineTailAxis(input->tensor->rawshape, input->tensor->rawshape.size());
            }
        }
    }
    
    return SUCCESS;
}

inline bool IsUBCopy(pto::Operation& op) {
    if (BlockPassUtils::IsCopyIn(op.GetOpcode())) {
        auto outTile = std::dynamic_pointer_cast<pto::TileValue>(*(op.GetOOperands().begin()));
        if (outTile->GetMemory()->GetSpace() == pto::MemSpaceKind::UB) {
            return true;
        }
    }
    if (BlockPassUtils::IsCopyOut(op.GetOpcode())) {
        auto inTile = std::dynamic_pointer_cast<pto::TileValue>(*(op.GetIOperands().begin()));
        if (inTile->GetMemory()->GetSpace() == pto::MemSpaceKind::UB) {
            return true;
        }
    }
    return false;
}

Status CodegenPreprocIR::ForceCombineAxisForAxisCombine(pto::BlockFunction &func) const {
    const std::set<Opcode> skipInputCombineOps = {Opcode::OP_BRCB, Opcode::OP_EXPAND};
    for (auto &subProgram : func.rootFunc_->programs_) {
        for (auto &op : subProgram.second->Operations(false)) {
            if (OpcodeManager::Inst().GetCoreType(op.GetOpcode()) != OpCoreType::AIV && !IsUBCopy(op)) {
                continue;
            }
            std::vector<bool> inputCombineAxis;
            for (size_t i = 0; i < op.GetIOperands().size(); ++i) {
                LogicalTensors operands = op.GetIOperands();
                if (operands[i]->tensor->rawshape.back() == 1 && skipInputCombineOps.count(op.GetOpcode()) == 0) {
                    inputCombineAxis.push_back(true);
                } else {
                    inputCombineAxis.push_back(false);
                }
            }
            op.SetAttr(OpAttributeKey::inputCombineAxis, inputCombineAxis);
            std::vector<bool> outputCombineAxis;
            for (size_t i = 0; i < op.GetOOperands().size(); ++i) {
                LogicalTensors operands = op.GetOOperands();
                if (operands[i]->tensor->rawshape.back() == 1 && OpcodeManager::Inst().GetOpCalcType(op.GetOpcode()) != OpCalcType::REDUCE) {
                    outputCombineAxis.push_back(true);
                } else {
                    outputCombineAxis.push_back(false);
                }
            }
            op.SetAttr(OpAttributeKey::outputCombineAxis, outputCombineAxis);
        }
    }
    return SUCCESS;
}

std::string CodegenPreprocIR::DumpOpList(pto::BlockFunction &function) {
    std::stringstream ss;
    int idx = 0;
    for (auto &subProgram : function.rootFunc_->programs_) {
        ss << "==================== OP_LIST Codegen_Preproc " << idx << " =====================" << "\n";
        for (auto &op : subProgram.second->Operations(false)) {
            if (!op.oOperand.empty()) {
                bool needAlloc = false;
                op.oOperand[0]->GetAttr(OpAttributeKey::needAlloc, needAlloc);
                ss << op.GetOpcodeStr() << "[" << op.GetOpMagic() << "], needAlloc: " << static_cast<int>(needAlloc)
                    << ", memId: " << op.oOperand[0]->memoryrange.memId << "\n";
            } else {
                ss << op.GetOpcodeStr() << "[" << op.GetOpMagic() << "]" << "\n";
            }
        }
        idx++;
    }
    return ss.str();
}

void CodegenPreprocIR::SetNeedAllocAttr(pto::BlockFunction &function) {
    for (auto &subProgram : function.rootFunc_->programs_) {
        std::unordered_set<int> appearedMemId;
        for (auto &op : subProgram.second->Operations(false)) {
            for (auto &outTensor : op.GetOOperands()) {
                if (outTensor->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
                    continue;
                }
                auto it = appearedMemId.find(outTensor->memoryrange.memId);
                if (it == appearedMemId.end()) {
                    outTensor->SetAttr(OpAttributeKey::needAlloc, true);
                    appearedMemId.insert(outTensor->memoryrange.memId);
                }
            }
        }
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "%s", DumpOpList(function).c_str());
}

Status CodegenPreprocIR::RunOnFunction(pto::BlockFunction &function) {
    APASS_LOG_INFO_F(Elements::Operation, "===============================================================> Start CodegenPreprocIR.");
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW_TYPE) {
            op.SetOpCode(Opcode::OP_VIEW);
        }
    }
    if (SaveGmTensorParamIdxToOp(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "CodegenPreprocIR RunOnFunction failed at function SaveGmTensorParamIdxToOp.");
        return FAILED;
    }

    if (ConfigManager::Instance().GetOperationConfig(KEY_COMBINE_AXIS, false)) {
        if (ForceCombineAxisForAxisCombine(function) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "CodegenPreprocIR RunOnFunction failed at function ForceCombineAxisForAxisCombine.");
            return FAILED;
        }
    } else {
        if (ForceCombineAxis(function) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "CodegenPreprocIR RunOnFunction failed at function ForceCombineAxis.");
            return FAILED;
        }
    }

    SetNeedAllocAttr(function);
    APASS_LOG_INFO_F(Elements::Operation, "===============================================================> Finish CodegenPreprocIR.");
    return SUCCESS;
}

} // namespace tile_fwk
} // namespace npu
