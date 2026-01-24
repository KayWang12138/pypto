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
static std::unordered_map<size_t, std::pair<LogicalTensor*, std::vector<Operation*>>> hashCache;
void SortedProducer(std::vector<Operation*>& sortedProducers) {
    std::sort(sortedProducers.begin(), sortedProducers.end(),
        [](const Operation* op1, const Operation* op2) {
            if (op1 == nullptr || op2 == nullptr) {
                return op1 == nullptr;
            }
            const auto& iOp1 = op1->GetIOperands();
            const auto& iOp2 = op2->GetIOperands();
            size_t minLen = std::min(iOp1.size(), iOp2.size());
            for (size_t i = 0; i < minLen; ++i) {
                if (iOp1[i] == nullptr || iOp2[i] == nullptr) {
                    return iOp1[i] == nullptr;
                }
                LogicalTensor* ptr1 = iOp1[i].get();
                LogicalTensor* ptr2 = iOp2[i].get();
                if (ptr1 != ptr2) {
                    return ptr1 < ptr2;
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
}

unsigned long ComputeHashOrderless(const std::vector <Operation*>& producers) {
    std::vector<std::string> opStrList;
    std::stringstream ss;
    std::vector<Operation*> sortedProducers = producers;
    SortedProducer(sortedProducers);
    for (const auto& op: sortedProducers) {
        if (op == nullptr) {
            continue;
        }
        ss.str(""), ss.clear();
        ss << op->GetOpcodeStr(true);
        for (const auto& iOperands: op->GetIOperands()) {
            if (iOperands == nullptr || iOperands->tensor == nullptr) {
                continue;
            }
            ss << "[i";
            ss << "$" << iOperands->tensor->DumpSSA(false, false);
            ss << iOperands->DumpType();
            ss << "(";
            for (size_t i = 0; i < iOperands->offset.size(); ++i) {
                ss << iOperands->offset[i];
                if (i != iOperands->offset.size() - 1) {
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
    unsigned long operationHash_ = ComputeHashOrderless(producers);
    return operationHash_;
}

Status CommonOperationEliminate::RunOnFunction(Function &function) {
    auto tensorProducerMap = GetProducers(function);
    std::unordered_set<Operation*> cacheProducers;
    for (auto& tensorProducerPair: tensorProducerMap) {
        auto& producerGroup = tensorProducerPair.second;
        if (producerGroup.empty() || !OpAlreadyExist(tensorProducerPair)) {
            continue;
        }

        for (auto& [hashKey, tensorOpPair]: hashCache) {
            if (tensorOpPair.first == nullptr) {
                continue;
            }
            for (auto producer: tensorOpPair.first->GetProducers()) {
                if (producer != nullptr) cacheProducers.insert(producer);
            }
        }
        for (auto op: producerGroup) {
            if (op == nullptr) continue;
            if (!cacheProducers.count(op)) {
                op->SetAsDeleted();
            }
        }
    }
    function.EraseOperations();
    if (DeadOperationEliminator::EliminateDeadOperation(function) != SUCCESS) {
        ALOG_ERROR_F("Eliminate dead operation failed in CommonOperationEliminate.");
        return FAILED;
    }
    hashCache.clear();
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
                if (pro != nullptr) {
                    tensorProducerMap[tensor.get()].push_back(pro);
                }
            }
        }
    }
    return tensorProducerMap;
}

std::pair<LogicalTensor*, std::vector<Operation*>>  CommonOperationEliminate::OperationExist(const std::pair<LogicalTensor*, std::vector<Operation*>>& tensorProducersPair) {
    const std::vector<Operation*>& producers = tensorProducersPair.second;
    for (const auto& op: producers) {
        if (op == nullptr) {
            continue;
        }
        if (op->GetOpcode() == Opcode::OP_COMPARE_SWAP) {
            return {nullptr, {}};
        }
    }
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
    if (hashCache.count(groupHash) != 0){
        return hashCache[groupHash];
    }
    hashCache.emplace(groupHash, tensorProducersPair);
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
    if (tensorProducerPair.first->tensor->GetDataType() != existOp.first->tensor->GetDataType()) {
        return false;
    }
    LogicalTensor* oldtensors = tensorProducerPair.first;
    LogicalTensor* newtensors = existOp.first;
    if (oldtensors->nodetype == NodeType::OUTCAST) {
        return false;
    }
    if (newtensors->GetConsumers().size() == 0 || oldtensors->GetConsumers().size() == 0) {
        return false;
    }
    auto consumers = oldtensors->GetConsumers();
    for (auto &cur : consumers) {
        if (cur == nullptr) continue;
        std::shared_ptr<LogicalTensor> old_ptr(oldtensors, [](LogicalTensor*){});
        std::shared_ptr<LogicalTensor> new_ptr(newtensors, [](LogicalTensor*){});
        cur->ReplaceInput(new_ptr, old_ptr);
        auto attptr = cur->GetOpAttribute().get();
        if (cur->GetOpcode() == Opcode::OP_VIEW) {
            if (auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(attptr)) {
            // VIEW操作的offset要相应被修改。
                UpdateView(viewOpAttribute, old_ptr, new_ptr);
                continue;
            }
        } else if (cur->GetOpcode() == Opcode::OP_COPY_IN) { 
            if (auto copyOpAttribute = dynamic_cast<CopyOpAttribute*>(attptr)) {
            // CopyIn操作的offset要相应被修改。
                UpdateCopy(copyOpAttribute, old_ptr, new_ptr);
                continue;
            }
        }
    }
    oldtensors->GetConsumers().clear();
    ALOG_DEBUG_F("In CommonOperationEliminate, Tensor %d producergroup is marked as redundant.", oldtensors->GetMagic());
    return true;
}  
}// namespace npu::tile_fwk