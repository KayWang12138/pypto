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
 * \file generate_move_op_checker.cpp
 * \brief
 */

#include "generate_move_op_checker.h"

namespace npu {
namespace tile_fwk {
Status GenerateMoveOpChecker::DoPreCheck(Function &function) {
    ALOG_INFO_F("Start Precheck for GenerateMoveOp.");
    auto operations = function.Operations();
    // Check iOperand and oOperand of OP_CONVERT
    for (auto &operation : operations) {
        const auto opcode = operation.GetOpcode();
        bool isValid = true;
        switch(opcode) {
            case Opcode :: OP_CONVERT:
                isValid = ValidConvertOp(operation);
                break;
            case Opcode :: OP_VIEW:
                isValid = ValidViewOp(operation);
                break;
            case Opcode :: OP_ASSEMBLE:
                isValid = ValidAssembleOp(operation);
                break;
            default:
                continue;
        }
        if (!isValid) {
            ALOG_ERROR_F("Operation validation failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status GenerateMoveOpChecker::DoPostCheck(Function &function) {
    ALOG_INFO_F("Start Postcheck for GenerateMoveOp.");
    auto operations = function.Operations();
    for (auto &operation : operations) {
        auto op = operation.GetOpcode();
        bool isValid =true;
        if (((op == Opcode::OP_ASSEMBLE || op == Opcode::OP_VIEW) && 
            ((operation.GetIOperands().size() != 1) || (operation.GetOOperands().size() != 1) || 
            (operation.GetIOperands().front()->GetMemoryTypeOriginal() != operation.GetOOperands().front()->GetMemoryTypeOriginal()))) || 
            (op == Opcode::OP_DUPLICATE || op == Opcode::OP_CONVERT)) {
                isValid = false;
            }
        if (!isValid) {
            ALOG_ERROR_F("Operation validation failed : op [%d] check failed.",operation.GetOpMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}

bool GenerateMoveOpChecker::ValidViewOp(const Operation &op) const {
    //校验view单输入单输出，指针非空
    if ((op.GetOpAttribute().get() == nullptr) ||
       (op.GetIOperands().size() != 1) || (op.GetOOperands().size() != 1) ||
       (op.GetIOperands().front() == nullptr) || (op.GetOOperands().front() == nullptr) ||
       (*(op.oOperand[0]->GetConsumers().begin()) == nullptr)) {
        ALOG_ERROR_F("View op [%d] check failed.", op.GetOpMagic());
        return false;
    }
    //校验view输出tensor内存是否合理
    if (op.GetOOperands().front()->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
        auto consumerOps = op.oOperand[0]->GetConsumers(); 
        for (auto childOp : consumerOps) {
            if (childOp == nullptr) {
                ALOG_ERROR_F("View op [%d] output has null consumers.",op.GetOpMagic());
                return false;
            }
            auto opcode = childOp->GetOpcode();
            const auto &inputsMemType = OpcodeManager::Inst().GetInputsMemType(opcode);
            bool hasDDRinput = std::find(inputsMemType.begin(),inputsMemType.end(),MemoryType::MEM_DEVICE_DDR) != inputsMemType.end();
            if (opcode == Opcode::OP_RESHAPE || hasDDRinput) {
                continue;
            } else if (opcode == Opcode::OP_CONVERT) {
                auto convertOpAttribute = dynamic_cast<ConvertOpAttribute *>(op.GetOpAttribute().get());
                auto convertPath = convertOpAttribute->GetConvertPath();
                if (convertPath.first != MemoryType::MEM_DEVICE_DDR){
                    ALOG_ERROR_F("View op [%d] consumer %s[%d] has invalid convert path.", op.GetOpMagic(),childOp->GetOpcodeStr().c_str(),childOp->GetOpMagic());
                    return false;
                }
            } else {
                ALOG_ERROR_F("View op [%d] consumer %s[%d] does not support DDR input.", op.GetOpMagic(),childOp->GetOpcodeStr().c_str(),childOp->GetOpMagic());
                return false;
            }
        }    
    }
    return true;
}

bool GenerateMoveOpChecker::ValidAssembleOp(const Operation &op) const {
    //校验assemble单输入单输出，指针非空
    bool valid = true;
    if ((op.GetOpAttribute().get() == nullptr) ||
       (op.GetIOperands().size() != 1) || (op.GetOOperands().size() != 1) ||
       (op.GetIOperands().front() == nullptr) || (op.GetOOperands().front() == nullptr)) {
        ALOG_ERROR_F("Assemble op [%d] check failed.", op.GetOpMagic());
        valid = false;
    }
    return valid;
}

bool GenerateMoveOpChecker::ValidConvertOp(const Operation &op) const {
    //校验convert单输入单输出，指针非空，输入输出内存类型不同，且存在DDR类型
    bool valid = true;
    if ((op.GetOpAttribute().get() == nullptr) || (op.GetIOperands().size() != 1) || (op.GetOOperands().size() != 1) ||
       (op.GetIOperands().front() == nullptr) || (op.GetOOperands().front() == nullptr) ||
       (op.GetIOperands().front()->GetMemoryTypeOriginal() == op.GetOOperands().front()->GetMemoryTypeOriginal()) ||
       (op.GetIOperands().front()->GetShape() != op.GetOOperands().front()->GetShape())) {
        ALOG_ERROR_F("Convert op [%d] check failed.",op.GetOpMagic());
        valid = false;
    }
    return valid;
}
} // namespace tile_fwk
} // namespace npu