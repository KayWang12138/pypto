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
 * \file common_operation_eliminate.cpp
 * \brief
 */

#include "common_operation_eliminate.h"
#include <unordered_map>
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_utils/dead_operation_eliminate.h"
#include "passes/pass_check/common_operation_eliminate_checker.h"

namespace npu::tile_fwk {
unsigned long ComputeHashOrderless(const std::vector <Operation*>& producers) {
    std::vector<std::string> opStrList;
    std::stringstream ss;
    std::vector<Operation*> sortedProducers = producers;
    std::sort(sortedProducers.begin(), sortedProducers.end(),
        [](const Operation* op1, const Operation* op2) {
            if (op1 == nullptr || op2 == nullptr) {
                return op1 == nullptr;
            }
            const auto& iOp1 = op1->GetIOperands();
            const auto& iOp2 = op2->GetIOperands();
            size_t minLen = std::min(iOp1.size(), iOp2.size());
            for (size_t i = 0; i < minLen; ++i) {
                // uint64_t ptr1 = reinterpret_cast<uint64_t>(iOp1[i]->tensor);//这个语法该怎么写 问！
                // uint64_t ptr2 = reinterpret_cast<uint64_t>(iOp2[i]->tensor);
                if (iOp1[i] == nullptr || iOp2[i] == nullptr) {
                    return iOp1[i] == nullptr;
                }
                LogicalTensor* ptr1 = iOp1[i].get();//这个语法该怎么写 问！.get()?
                LogicalTensor* ptr2 = iOp2[i].get();
                if (ptr1 != ptr2) {
                    return ptr1 < ptr2;//怪
                }
            }
            if (iOp1.size() != iOp2.size()) {
                return iOp1.size() < iOp2.size();
            }
            std::stringstream ss1, ss2;
            for (const auto &attr : OpcodeManager::Inst().GetAttrs(op1->GetOpcode())) {
                ss1 << " attr: [" << attr << " : " << op1->DumpAttr(attr) << "]";
            }
            for (const auto &attr : OpcodeManager::Inst().GetAttrs(op2->GetOpcode())) {
                ss2 << " attr: [" << attr << " : " << op2->DumpAttr(attr) << "]";
            }
        return ss1.str() < ss2.str();
    });
    for (const auto& op: sortedProducers) {
        if (op == nullptr) {
            continue;
        }
        ss.str(""), ss.clear();
        ss << op->GetOpcodeStr(true);
        for (const auto& iOperand: op->GetIOperands()) {
            if (iOperand == nullptr || iOperand->tensor == nullptr) {
                continue;
            }
            ss << "[i";
            ss << "$" << iOperand->tensor->DumpSSA(false, false);
            ss << iOperand->DumpType();
            ss << "(";
            for (size_t i = 0; i < iOperand->offset.size(); ++i) {
                ss << iOperand->offset[i];
                if (i != iOperand->offset.size() - 1) {
                    ss << ", ";
                }
            }
            ss << ")";
            ss << "]";
        }
        if (op->GetOpAttribute() != nullptr) {
            ss << " " << op->GetOpAttribute()->Dump();
        }
        for (const auto &attr : OpcodeManager::Inst().GetAttrs(op->GetOpcode())) {
            ss << " attr: [" << attr << " : " << op->DumpAttr(attr) << "]";
        }
        ss << "id" << op->GetSubgraphID();
        opStrList.push_back(ss.str());
    }
    ss.str(""), ss.clear();
    for (const auto& str: opStrList) {
        ss << str;
    }
    std::string s = ss.str();
    std::hash<std::string> hasher;
    auto result = hasher(s);
    return result;
}

unsigned long ComputeHash(const std::vector <Operation*>& producers) {
    // compute has every time to avoid member changed
    unsigned long operationHash_ = ComputeHashOrderless(producers);//这个等下放到pass里看下怎么改
    return operationHash_;
}

Status CommonOperationEliminate::RunOnFunction(Function &function) {
    auto tensorProducerMap = GetProducers(function);
    for (auto& tensorProducerPair: tensorProducerMap) {
        auto& producerGroup = tensorProducerPair.second;
        if (producerGroup.empty()) {
            continue;
        }
        if (OpAlreadyExist(tensorProducerPair)) {
            for (auto op: producerGroup) {
                if (op == nullptr) continue;
                op->SetAsDeleted();
            }
        }
    }
    function.DumpJsonFile("1.json");
    function.EraseOperations(true);
    function.DumpJsonFile("2.json");
    if (DeadOperationEliminator::EliminateDeadOperation(function) != SUCCESS) {
        ALOG_ERROR_F("Eliminate dead operation failed in CommonOperationEliminate.");
        return FAILED;
    }
    return SUCCESS;
}

Status CommonOperationEliminate::PreCheck(Function &function) {
    CommonOperationEliminateChecker checker;
    return checker.DoPreCheck(function);
}

std::unordered_map<LogicalTensor*, std::vector<Operation*>> CommonOperationEliminate::GetProducers(Function &function) {
    std::unordered_map<LogicalTensor*, std::vector<Operation*>> tensorProducerMap;
    std::unordered_set<int> visitedTensors;
    auto allOps = function.Operations().DuplicatedOpList();
    for (const auto& op: allOps) {
        if (op == nullptr) {
            continue;
        }
         auto& outputTensors = op->GetOOperands();
        for (auto& tensor: outputTensors) {
            if (tensor == nullptr || visitedTensors.count(tensor->GetMagic())) {
                continue;
            }
            visitedTensors.insert(tensor->GetMagic());
            for (auto& pro: tensor->GetProducers()) {
                tensorProducerMap[tensor.get()].push_back(pro);
            }
        }
    }
    return tensorProducerMap;
}

std::pair<LogicalTensor*, std::vector<Operation*>>  CommonOperationEliminate::OperationExist(const std::pair<LogicalTensor*, std::vector<Operation*>>& tensorProducersPair) {
    static std::unordered_map<size_t, std::pair<LogicalTensor*, std::vector<Operation*>>> hashCache;
    const std::vector<Operation*>& producers = tensorProducersPair.second;
    for (auto operation: producers) {
        if (operation == nullptr) continue;
        auto &inputsMemType = OpcodeManager::Inst().GetInputsMemType(operation->GetOpcode());
        auto &outputsMemType = OpcodeManager::Inst().GetOutputsMemType(operation->GetOpcode());
        OpCalcType opCalcType = OpcodeManager::Inst().GetOpCalcType(operation->GetOpcode());
        bool inputCheck = inputsMemType.size() == 1 && inputsMemType[0] == MemoryType::MEM_L1;
        bool calcTypeCheck = opCalcType == OpCalcType::MOVE_LOCAL || opCalcType == OpCalcType::MOVE_IN;
        bool outputCheck = outputsMemType.size() == 1 && outputsMemType[0] != MemoryType::MEM_L1;
        if (inputCheck && calcTypeCheck && outputCheck) { // copy from L1 to L0
            return {nullptr, {}};
        }
        if (operation->GetOpcode() == Opcode::OP_VIEW) {
            return {nullptr, {}};
        }
        if (operation->GetBoolAttribute(OpAttributeKey::dontTouch)) {
            return {nullptr, {}};
        }
    }
    size_t groupHash = ComputeHash(producers);
    std::cout << "tensorid:"<<tensorProducersPair.first->GetMagic() <<"hash:"<< groupHash << std::endl;
    if (hashCache.count(groupHash) != 0){
        return hashCache[groupHash];
    }

    hashCache.emplace(groupHash, tensorProducersPair);//后续可能改回operationCache_类变量，只存hash
    return {nullptr, {}};
}

void CommonOperationEliminate::UpdateView(ViewOpAttribute *viewOpAttribute,
                                          const std::shared_ptr<LogicalTensor> oldtensors,
                                          const std::shared_ptr<LogicalTensor> newtensors) const {
    auto &fromOffset = viewOpAttribute->GetFromOffset();
    for (size_t j = 0; j < fromOffset.size(); j++) {
        fromOffset[j] -= oldtensors->offset[j] - newtensors->offset[j];
    }
}

void CommonOperationEliminate::UpdateCopy(CopyOpAttribute *copyOpAttribute,
                                          const std::shared_ptr<LogicalTensor> oldtensors,
                                          const std::shared_ptr<LogicalTensor> newtensors) const {
    if (!copyOpAttribute->IsCopyOut()) {
        auto [fromOffset, memType] = copyOpAttribute->GetCopyInAttr();
        (void)memType;
        for (size_t j = 0; j < fromOffset.size(); j++) {
            fromOffset[j] -= oldtensors->offset[j] - newtensors->offset[j];
        }
        copyOpAttribute->SetFromOffset(fromOffset);
    }
}

bool CommonOperationEliminate::OpAlreadyExist(const std::pair<LogicalTensor*, std::vector<Operation*>>& tensorProducerPair) {
    auto& producers = tensorProducerPair.second;
    if (producers.empty()) {
        return false;
    }
    auto existOp = OperationExist(tensorProducerPair);
    if (existOp.first == nullptr || existOp.second.empty()) {
        return false;
    }
    if (tensorProducerPair.first == nullptr) {
        return false;
    }
    if (tensorProducerPair.first->shape != existOp.first->shape) {
        return false;
    }
    //LogicalTensorPtr oldtensors(tensorProducerPair.first);
    //LogicalTensorPtr newtensors(existOp.first);

    //auto oldtensors = *tensorProducerPair.first;
    //auto newtensors = *existOp.first;
    //该到这里
    LogicalTensor* oldtensors = tensorProducerPair.first;
    LogicalTensor* newtensors = existOp.first;
    if (oldtensors->nodetype == NodeType::OUTCAST) {
        return false;
    }
    if (oldtensors->GetConsumers().size() == 0) {
        return false;
    }
    auto consumers = oldtensors->GetConsumers();
    for (auto &cur : consumers) {
        if (cur == nullptr) continue;
        std::shared_ptr<LogicalTensor> old_ptr(oldtensors, [](LogicalTensor*){});
        std::shared_ptr<LogicalTensor> new_ptr(newtensors, [](LogicalTensor*){});
        //cur->ReplaceInput(newtensors, oldtensors);
        cur->ReplaceInput(new_ptr, old_ptr);
        if (cur->GetOpAttribute() == nullptr) {
            continue;
        }
        // if (auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(cur->GetOpAttribute().get())) {
        //     // VIEW操作的offset要相应被修改。
        //     UpdateView(viewOpAttribute, oldtensors, newtensors);
        //     continue;
        // }
        // if (auto copyOpAttribute = dynamic_cast<CopyOpAttribute*>(cur->GetOpAttribute().get())) {
        //     // CopyIn操作的offset要相应被修改。
        //     UpdateCopy(copyOpAttribute, oldtensors, newtensors);
        //     continue;
        // }
        auto attrPtr = cur->GetOpAttribute().get();
        if (auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(attrPtr)) {
            // ========== 只改这行：把oldtensors/newtensors换成old_ptr/new_ptr ==========
            UpdateView(viewOpAttribute, old_ptr, new_ptr);
            continue;
        }
        if (auto copyOpAttribute = dynamic_cast<CopyOpAttribute*>(attrPtr)) {
            // ========== 只改这行：把oldtensors/newtensors换成old_ptr/new_ptr ==========
            UpdateCopy(copyOpAttribute, old_ptr, new_ptr);
            continue;
        }
    }
    oldtensors->GetConsumers().clear();
    ALOG_DEBUG_F("In CommonOperationEliminate, Tensor %d producergroup is marked as redundant.", oldtensors->GetMagic());
    return true;
}
}// namespace npu::tile_fwk