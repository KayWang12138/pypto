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
 * \file add_alloc.cpp
 * \brief
 */

#include "add_alloc.h"

namespace npu::tile_fwk {
Status AddAlloc::GenTensorAllocMsgMap(Function &function, 
    std::unordered_map<int, TensorAllocMsg> &tensorAllocMsgMap) const {
    for (auto& op : function.Operations().DuplicatedOpList()) {
        if (FindTensorAllocMsg(op, tensorAllocMsgMap) != SUCCESS) {
            ALOG_ERROR_F("FindTensorAllocMsg failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status AddAlloc::GenAllocNode(Function &function) {
    std::unordered_map<int, TensorAllocMsg> tensorAllocMsgMap;
    if (GenTensorAllocMsgMap(function, tensorAllocMsgMap) != SUCCESS) {
        ALOG_ERROR_F("GenTensorAllocMsgMap failed.");
        return FAILED;
    }
    for (auto& tensorAllocMsg : tensorAllocMsgMap) {
        if (tensorAllocMsg.second.isAllocated == false) {
            ALOG_DEBUG_F("create alloc node for tensor [%d]", tensorAllocMsg.first);
            CreateAllocNode(tensorAllocMsg.second, function);
        }
    }
    return SUCCESS;
}

Status AddAlloc::AddAndCheckAlloc(Function &function) {
    if (GenAllocNode(function) != SUCCESS) {
        ALOG_ERROR_F("GenAllocNode failed.");
        return FAILED;
    }
    std::vector<Operation *> newOperations;
    for (auto& op : function.Operations().DuplicatedOpList()) {
        if (op->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            newOperations.insert(newOperations.begin(), op);
            continue;
        }
        newOperations.push_back(op);
    }
    function.ScheduleBy(newOperations);
    return SUCCESS;
}

TensorAllocMsg AddAlloc::ConstructTensorAllocMsg(Operation *op, size_t i, int memId, const std::vector<int> &allocMagic) const {
    TensorAllocMsg tensorAllocMsg;
    tensorAllocMsg.producer.push_back(op);
    tensorAllocMsg.memType = op->GetOutputOperand(i)->GetMemoryTypeOriginal();
    tensorAllocMsg.memId = memId;
    if (allocMagic.empty() || i >= allocMagic.size()) {
        ALOG_INFO_F("Tensor [%d] is not allocted.", memId);
        tensorAllocMsg.isAllocated = false;
    }
    return tensorAllocMsg;
}

Status AddAlloc::UpdateTensorAllocMsg(Operation *op, size_t i, const std::vector<int> &allocMagic,
                                      std::unordered_map<int, TensorAllocMsg> &tensorAllocMsgMap) const {
    auto memId = op->GetOutputOperand(i)->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId;
    if (memId == -1) {
        ALOG_ERROR_F("Get memId in memorymap failed.");
        return FAILED;
    }
    if (tensorAllocMsgMap.find(memId) == tensorAllocMsgMap.end()) {
        tensorAllocMsgMap.emplace(memId, ConstructTensorAllocMsg(op, i, memId, allocMagic));
        return SUCCESS;
    }
    tensorAllocMsgMap[memId].producer.push_back(op);
    if (i < allocMagic.size() && tensorAllocMsgMap[memId].isAllocated == false) {
        ALOG_DEBUG_F("tensor [%d] is allocaterd at the first time.", memId);
        tensorAllocMsgMap[memId].isAllocated = true;
    }
    return SUCCESS;
}

Status AddAlloc::SetTensorAllocMsg(Operation *op, 
    std::unordered_map<int, TensorAllocMsg> &tensorAllocMsgMap, const std::vector<int> &allocMagic) const {
    for (size_t i = 0; i < op->GetOOperands().size(); i++) {
        if (op->GetOutputOperand(i)->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        if (op->GetOutputOperand(i)->memorymap.find(BLOCK_GRAPH_DEFAULT_COLOR) == op->GetOutputOperand(i)->memorymap.end()) {
            ALOG_ERROR_F("Cannot find memorymap");
            return FAILED;
        }
        if (UpdateTensorAllocMsg(op, i, allocMagic, tensorAllocMsgMap) != SUCCESS) {
            ALOG_ERROR_F("UpdateTensorAllocMsg failed!");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status AddAlloc::FindTensorAllocMsg(Operation *op, 
    std::unordered_map<int, TensorAllocMsg> &tensorAllocMsgMap) const {
    // 遍历所有节点，找到需要分配Alloc的tensor以及其第一次出现时候的位置
    std::vector<int> allocMagic;
    for (const auto &inCtrlOp : op->GetInCtrlOperations()) {
        if (inCtrlOp->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            allocMagic.emplace_back(inCtrlOp->GetOpMagic());
        }
    }
    if (SetTensorAllocMsg(op, tensorAllocMsgMap, allocMagic) != SUCCESS) {
        ALOG_ERROR_F("SetTensorAllocMsg failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status AddAlloc::GenAllocOpcode(const Opcode &allocOpcode, const TensorAllocMsg& tensorAllocMsg, Function& function) {
    int maxOpMagic = -1;
    for (auto &op : function.Operations()) {
        maxOpMagic = std::max(maxOpMagic, op.GetOpMagic());
    }
    for (auto &oOperand : tensorAllocMsg.producer[0]->GetOOperands()) {
        if (oOperand->memorymap[BLOCK_GRAPH_DEFAULT_COLOR].memId != tensorAllocMsg.memId) {
            continue;
        }
        auto &allocOp = function.AddOperation(allocOpcode, {}, 
            std::vector<std::shared_ptr<LogicalTensor>>({oOperand}));
        if (tensorAllocMsg.producer[0]->HasAttr(OpAttributeKey::tag)) {
            allocOp.SetAttribute(OpAttributeKey::tag, tensorAllocMsg.producer[0]->GetStringAttribute(OpAttributeKey::tag));
        }
        allocOp.opmagic = maxOpMagic + 1;
    }
    return SUCCESS;
}

Status AddAlloc::CreateAllocNode(const TensorAllocMsg& tensorAllocMsg, Function& function) {
    auto iter = allocOpcodeMap.find(tensorAllocMsg.memType);
    if (iter != allocOpcodeMap.end()) {
        ALOG_DEBUG_F("create alloc node for memtype [%d]", static_cast<int>(tensorAllocMsg.memType));
        if (tensorAllocMsg.producer.size() == 0) { 
            ALOG_ERROR_F("tensorAllocMsg's producer size cannot be 0."); 
            return FAILED; 
        }
        Opcode allocOpcode = iter->second;
        if (GenAllocOpcode(allocOpcode, tensorAllocMsg, function)) {
            ALOG_ERROR_F("GenAllocOpcode failed."); 
            return FAILED; 
        }
    }
    return SUCCESS;
}
}