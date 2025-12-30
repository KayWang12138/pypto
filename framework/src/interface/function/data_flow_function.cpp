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
 * \file data_flow_function.cpp
 * \brief
 */

#include "interface/function/data_flow_function.h"

#include "interface/operation/attribute.h"
#include "interface/operation/operation.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/program/program.h"
#include "interface/utils/log.h"

namespace npu::tile_fwk {

DataFlowFunction::DataFlowFunction(const Program &belongTo, const std::string &funcMagicName,
    const std::string &funcRawName, Function *parentFunc)
    : ControlFlowFunction(belongTo, funcMagicName, funcRawName, parentFunc) {
}

void *DataFlowFunction::GetParamAddress(int index) {
    return explicitArgAddrs_[index];
}

const SubfuncInvokeInfoTy &DataFlowFunction::GetSubFuncInvokeInfo(const size_t i) const {
    auto callAttr = std::dynamic_pointer_cast<CallOpAttribute>(operations_[i]->GetOpAttribute());
    ASSERT(callAttr != nullptr)
        << "Operation at index " << i << " must have a CallOpAttribute";
    return *(callAttr->invokeInfo_);
}

auto DataFlowFunction::AnnotateOperation() {
    std::map<int, std::vector<Operation *>> subgraphs;
    std::map<int, int> opToSubgraph;
    for (auto &&op : Operations()) {
        // same op magic shall only appear once
        ASSERT(opToSubgraph.find(op.GetOpMagic()) == opToSubgraph.end())
                        << "Same op magic shall only appear once." << "\n"
                        << "Duplicate OpMagic found: " << op.GetOpMagic() << "\n" << "Operation: " << op.Dump();
        if (op.GetSubgraphID() < 0) {
            ALOG_DEBUG("Op magic: ", op.GetOpMagic(), "less than 0 graph: ", op.GetSubgraphID());
            continue;
        }
        subgraphs[op.GetSubgraphID()].emplace_back(&op);
        opToSubgraph[op.GetOpMagic()] = op.GetSubgraphID();
        ALOG_DEBUG("Operation: ", op.GetOpMagic(), "Belong To subgraph: ", op.GetSubgraphID());
    }

    for (const auto &pair : subgraphs) {
        ALOG_DEBUG("Subgraph ID: ", pair.first);
        for (const auto &op : pair.second) {
            ALOG_DEBUG("Operation: ", op->Dump());
        }
    }
    return std::make_pair(std::move(subgraphs), std::move(opToSubgraph));
}

std::unordered_set<int> DataFlowFunction::LoopCheck() {
    if (GetTotalSubGraphCount() == 0) {
        return {};
    }
    ALOG_INFO("LoopCheck begin.");

    auto [subgraphs, opToSubgraph] = AnnotateOperation();
    std::map<LogicalTensor *, std::vector<int>> producers;
    std::map<LogicalTensor *, std::vector<int>> consumers;

    std::map<int, std::vector<std::shared_ptr<LogicalTensor>>> iOperands;
    std::map<int, std::vector<std::shared_ptr<LogicalTensor>>> oOperands;

    for (auto &&[subgraphID, operations] : subgraphs) {
        if (subgraphID == NOT_IN_SUBGRAPH) {
            continue;
        }

        GetAnIslandIncastsOutcasts(opToSubgraph, subgraphID, operations,
                                   iOperands[subgraphID], oOperands[subgraphID]);

        for (auto &&iop : iOperands[subgraphID]) {
            consumers[iop.get()].push_back(subgraphID);
        }
        for (auto &&oop : oOperands[subgraphID]) {
            producers[oop.get()].push_back(subgraphID);
        }
    }

    enum class DfsState {
        TODO = 0,
        IN_STACK,
        DONE,
    };

    std::map<int, DfsState> states;
    std::unordered_set<int> subGraphInCycle;
    for (auto &&[subgraphID, operations] : subgraphs) {
        (void)operations;
        if (subgraphID == NOT_IN_SUBGRAPH) {
            continue;
        }

        int duplicatedSubgraphID = -2;
        auto cycleDetection = [&states, &duplicatedSubgraphID, &oOperands, &consumers,
                               &subGraphInCycle](int currSubgraph, auto self) -> bool {
            if (states[currSubgraph] == DfsState::DONE) {
                return false;
            }

            if (states[currSubgraph] == DfsState::IN_STACK) {
                duplicatedSubgraphID = currSubgraph;
                ALOG_ERROR("[Cycle Detection] Cycle detected: ");
                ALOG_ERROR("[Cycle Detection]     subgraph id: ", currSubgraph);
                subGraphInCycle.emplace(currSubgraph);
                return true;
            }

            states[currSubgraph] = DfsState::IN_STACK;

            for (auto &&oop : oOperands[currSubgraph]) {
                for (int consumer : consumers[oop.get()]) {
                    if (self(consumer, self)) {
                        if (duplicatedSubgraphID != -2) {
                            ALOG_ERROR("[Cycle Detection]     tensor:      ", oop->Dump());
                            ALOG_ERROR("[producer]=");
                            for (const auto &producer : oop->GetProducers()) {
                                ALOG_ERROR(producer->GetOpMagic());
                            }
                            ALOG_ERROR("[Cycle Detection]     subgraph id: ", currSubgraph);
                            subGraphInCycle.emplace(currSubgraph);
                            if (currSubgraph == duplicatedSubgraphID) {
                                duplicatedSubgraphID = -2; // stop dumpping
                            }
                        }
                        return true;
                    }
                }
            }

            states[currSubgraph] = DfsState::DONE;
            return false;
        };
        if (cycleDetection(subgraphID, cycleDetection)) {
            return subGraphInCycle;
        }
    }
    return std::unordered_set<int>{};
}


void DataFlowFunction::GetAnIslandIncastsOutcasts(const std::map<int, int> &opToSubgraph, const int subgraphID,
    const std::vector<Operation *> &operations, std::vector<std::shared_ptr<LogicalTensor>> &iOperands,
    std::vector<std::shared_ptr<LogicalTensor>> &oOperands) const {
    std::set<std::shared_ptr<LogicalTensor>> allLogicalTensors;
    std::set<std::shared_ptr<LogicalTensor>> notOutcasts;
    std::set<std::shared_ptr<LogicalTensor>> notIncasts;
    for (const auto &opPtr : operations) {
        const auto &op = *opPtr;
        for (auto &&operand : op.GetIOperands()) {
            allLogicalTensors.insert(operand);
            bool usedbyotherfunction = false;
            for (auto &consumer : operand->GetConsumers()) {
                auto magic = consumer->GetOpMagic();
                if (consumer->GetOpcode() == Opcode::OP_CALL) {
                    continue;
                }
                ASSERT(opToSubgraph.find(magic) != opToSubgraph.end())
                    << "Consumer magic " << magic << " not found in opToSubgraph. " << "\n"
                    << "Operation: " << op.Dump();

                if (opToSubgraph.at(magic) != subgraphID) {
                    usedbyotherfunction = true;
                    break;
                }
            }
            if (!usedbyotherfunction) {
                notOutcasts.insert(operand);
            }
        }

        for (auto &&operand : op.GetOOperands()) {
            allLogicalTensors.insert(operand);
            notIncasts.insert(operand);
        }
    }
    std::set_difference(allLogicalTensors.begin(), allLogicalTensors.end(), notIncasts.begin(), notIncasts.end(),
        std::inserter(iOperands, iOperands.begin()));
    std::set<std::shared_ptr<LogicalTensor>> tryOOperands;
    std::set_difference(allLogicalTensors.begin(), allLogicalTensors.end(), notOutcasts.begin(), notOutcasts.end(),
        std::inserter(tryOOperands, tryOOperands.begin()));
    std::set_difference(tryOOperands.begin(), tryOOperands.end(), iOperands.begin(), iOperands.end(),
        std::inserter(oOperands, oOperands.begin()));

    std::sort(iOperands.begin(), iOperands.end(), TensorPtrComparator());
    std::sort(oOperands.begin(), oOperands.end(), TensorPtrComparator());
}


} // namespace npu::tile_fwk

