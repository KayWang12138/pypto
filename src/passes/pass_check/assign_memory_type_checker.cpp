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
 * \file assign_memory_type_checker.cpp
 * \brief
 */

#include "assign_memory_type_checker.h"

namespace npu {
namespace tile_fwk {
Status AssignMemoryTypeChecker::DoPreCheck(Function &function) {
    ALOG_INFO_F("Start Precheck for AssignMemoryType.");
    auto operations = function.Operations();
    for(auto &operation : operations){
        Operation *op_ptr = &operation;
        //A_MUL_B操作输入tensor生产者校验
        if(operation.GetOpcode() == Opcode::OP_A_MUL_B) {
            Status status = CheckAmulBInputProducers(operation);
            if(status != SUCCESS) {
                return status;
            }
        }
        //创建队列，包含当前操作和嵌套深度
        std::queue<std::pair<Operation *, int>> opQueue;
        std::unordered_set<Operation *> visited;

        opQueue.emplace(op_ptr, 1);
        visited.insert(op_ptr);

        while(!opQueue.empty()){
            auto[currentOp,depth] = opQueue.front();
            opQueue.pop();

            //嵌套深度达到3失败
            if(depth > 3){
                ALOG_WARN_F("MEMORY WARNING:View/Assemble/Reshape depth is over 3. Potential suboptimal allocation!");
                return SUCCESS;
            }
            CheckPattern(currentOp,opQueue,depth,visited);
        }
    }
    return SUCCESS;
}

Status AssignMemoryTypeChecker::CheckAmulBInputProducers(Operation &operation) {
    auto inputs = operation.GetIOperands();
    auto producerOps = operation.ProducerOps();
    for(auto &producerOp : producerOps) {
        auto producerOpcode = producerOp->GetOpcode();
        if(producerOpcode != Opcode::OP_L1_TO_L0A && producerOpcode !=Opcode::OP_L1_TO_L0B && 
           producerOpcode != Opcode::OP_VIEW && producerOpcode !=Opcode::OP_VEC_DUP) {
            ALOG_ERROR_F("MEMORY ERROR:%s[%d] has invalid input producer:%s[%d].",
                operation.GetOpcodeStr().c_str(),operation.GetOpMagic(),producerOp->GetOpcodeStr().c_str(),producerOp->GetOpMagic());
            return FAILED;
           }
        if(producerOpcode == Opcode::OP_VIEW) {
            auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(producerOp->GetOpAttribute().get());
            MemoryType attrToType = viewOpAttribute->GetTo();
            if(attrToType != MemoryType::MEM_BT && attrToType != MemoryType::MEM_FIX_QUANT_PRE) {
                ALOG_ERROR_F("VIEW Attribute ERROR:%s[%d] has invalid input OP_VIEW:%s[%d].",
                    operation.GetOpcodeStr().c_str(),operation.GetOpMagic(),producerOp->GetOpcodeStr().c_str(),producerOp->GetOpMagic());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

//检查view/assemble/reshape的嵌套深度是否大于等于3
void AssignMemoryTypeChecker::CheckPattern(Operation *operation, std::queue<std::pair<Operation *, int>> &opQueue,
                                            int depth, std::unordered_set<Operation *> &visited) {
    for(auto &tensor : operation->oOperand){
        for(auto &consumerOp : tensor-> GetConsumers()){
            if(consumerOp->GetOpcode() == Opcode::OP_VIEW ||
                consumerOp->GetOpcode() == Opcode::OP_ASSEMBLE ||
                consumerOp->GetOpcode() == Opcode::OP_RESHAPE){
                Operation* consumerOpPtr = consumerOp;
                if(visited.find(consumerOpPtr) == visited.end()){
                    opQueue.emplace(consumerOpPtr, depth + 1);
                    visited.insert(consumerOpPtr);
                    break;
                }
            }
        }
    }
}
} // namespace tile_fwk
} // namespace npu