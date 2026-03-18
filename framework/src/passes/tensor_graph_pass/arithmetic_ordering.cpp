/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arithmetic_reordering.h"
#include <queue>
#include <unordered_set>
#include "interface/function/function.h"
#include "interface/operation/opcode.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "Arithmetic Reordering"

namespace npu::tile_fwk {

/*
Algorithm
Using this graph as an example:

MUL [10000]
  in:  15 <- VIEW [10003]
  in:  18 <- VIEW [10004]
  out: 10 -> ADD [10001]
ADD [10001]
  in:  21 <- VIEW [10005]
  in:  10 <- MUL [10000]
  out: 11 -> ADD [10002]
ADD [10002]
  in:  11 <- ADD [10001]
  in:  24 <- VIEW [10006]
  out: 28 -> ASSEMBLE [10007]

1.  Recreate the equation by traversing the tree: 15 * 18 + 21 + 24
2.  Use operator precedence to construct an order of operations tree.
    For the above example, the tree should be like: the root vertex is the result.
    The root has 3 children: (15 * 18), 21, and 24 and (15 * 18) has two children: 15 and 18.
3.  w(v) of each leaf vertex is 0. For each non-leaf (non-root) node, we minimize w(v) by using a min-heap to combine the two children with smallest w(v).
    When combining, add that to the operations. Notice that all child vertices will have the same operation (either all ADD and SUB or all MUL).
    Using the above example, when we see that we can merge 21 and 24, we add into the operations ADD(21, 24).
4.  Replace all operations in the tree with these new operations.
*/

static const std::unordered_set<Opcode> REBALANCEABLE_OPCODES = {
    Opcode::OP_ADD,
    Opcode::OP_MUL,
    Opcode::OP_MAXIMUM,
    Opcode::OP_MINIMUM,
    Opcode::OP_S_ADD,
    Opcode::OP_S_MUL,
};

bool ArithmeticReordering::IsRebalanceable(Opcode opcode) {
    return REBALANCEABLE_OPCODES.count(opcode) > 0;
}

int ArithmeticReordering::ComputeWaitTime(const LogicalTensorPtr &tensor) const {
    for (auto *prod : tensor->GetProducers()) {
        if (prod->GetIOperands().empty()) {
            return 0;
        }
        int maxChild = 0;
        for (auto &in : prod->GetIOperands()) {
            maxChild = std::max(maxChild, ComputeWaitTime(in));
        }
        return maxChild + 1;
    }
    return 0;
}

// Returns false if the chain is not a safe linear tree (any shared input or
// multi-consumer intermediate). Caller must abort rebalancing in that case.
bool ArithmeticReordering::CollectTerms(const LogicalTensorPtr &tensor,
                                        Opcode opcode,
                                        std::vector<Term> &terms,
                                        std::unordered_set<Operation *> &opsToDelete,
                                        std::unordered_set<uint64_t> &visitedOps,
                                        std::unordered_set<uint64_t> &visitedTensors,
                                        Function &function) const {
    Operation *sameOp = nullptr;
    for (auto *prod : tensor->GetProducers()) {
        if (prod->GetOpcode() == opcode && visitedOps.count(prod->GetOpMagic()) == 0) {
            sameOp = prod;
            break;
        }
    }

    if (sameOp != nullptr && tensor->GetConsumers().size() == 1) {
        visitedOps.insert(sameOp->GetOpMagic());
        opsToDelete.insert(sameOp);
        for (auto &iOperand : sameOp->GetIOperands()) {
            if (!CollectTerms(iOperand, opcode, terms, opsToDelete, visitedOps, visitedTensors, function)) {
                return false;
            }
        }
        return true;
    }

    // Leaf term — fail if we've seen this tensor before (shared input = DAG).
    if (visitedTensors.count(tensor->GetMagic()) > 0) {
        return false;
    }
    visitedTensors.insert(tensor->GetMagic());
    terms.push_back({tensor, ComputeWaitTime(tensor)});
    return true;
}

Status ArithmeticReordering::ProcessRoot(Operation &rootOp, Function &function) const {
    if (rootOp.IsDeleted()) {
        return SUCCESS;
    }

    const Opcode opcode = rootOp.GetOpcode();
    const LogicalTensorPtr &outputTensor = rootOp.GetOOperands()[0];

    std::vector<Term> terms;
    std::unordered_set<Operation *> opsToDelete;
    std::unordered_set<uint64_t> visitedOps;
    std::unordered_set<uint64_t> visitedTensors;
    int tmpIdx = 0;

    visitedOps.insert(rootOp.GetOpMagic());
    opsToDelete.insert(&rootOp);
    for (auto &iOperand : rootOp.GetIOperands()) {
        if (!CollectTerms(iOperand, opcode, terms, opsToDelete, visitedOps, visitedTensors, function)) {
            return SUCCESS;
        }
    }

    // Abort if any collected op's output has multiple consumers.
    for (auto *op : opsToDelete) {
        if (op == &rootOp) { continue; }
        if (op->GetOOperands()[0]->GetConsumers().size() > 1) {
            return SUCCESS;
        }
    }

    if (terms.size() < 3) {
        return SUCCESS;
    }

    APASS_LOG_DEBUG_F(Elements::Operation,
        "Rebalancing %s chain with %zu terms.",
        OpcodeManager::Inst().GetOpcodeStr(opcode).c_str(), terms.size());

    for (auto *op : opsToDelete) {
        op->SetAsDeleted();
    }

    auto cmp = [](const Term &a, const Term &b) {
        return a.waitTime > b.waitTime;
    };
    std::priority_queue<Term, std::vector<Term>, decltype(cmp)> heap(cmp);
    for (auto &t : terms) {
        heap.push(t);
    }

    while (heap.size() > 1) {
        Term left = heap.top(); heap.pop();
        Term right = heap.top(); heap.pop();

        LogicalTensorPtr result;
        if (heap.empty()) {
            result = outputTensor;
        } else {
            result = std::make_shared<LogicalTensor>(
                function,
                left.tensor->Datatype(),
                left.tensor->shape,
                left.tensor->Format(),
                "thr_tmp_" + std::to_string(tmpIdx++),
                NodeType::LOCAL);
            result->CopyMemoryType(left.tensor);
            function.GetTensorMap().Insert(result);
        }

        function.AddRawOperation(opcode, {left.tensor, right.tensor}, {result});

        int newWait = std::max(left.waitTime, right.waitTime) + 1;
        heap.push({result, newWait});
    }

    return SUCCESS;
}

Status ArithmeticReordering::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Operation,
        "Start ArithmeticReordering for function [%s].", function.GetRawName().c_str());

    std::vector<Operation *> roots;
    for (auto &op : function.Operations()) {
        if (!IsRebalanceable(op.GetOpcode())) {
            continue;
        }
        if (op.GetIOperands().size() != 2 || op.GetOOperands().size() != 1) {
            continue;
        }
        auto &output = op.GetOOperands()[0];
        bool consumedBySameOpcode = false;
        for (auto *consumer : output->GetConsumers()) {
            if (consumer->GetOpcode() == op.GetOpcode()) {
                consumedBySameOpcode = true;
                break;
            }
        }
        if (consumedBySameOpcode) {
            continue;
        }
        bool hasChainInput = false;
        for (auto &iOperand : op.GetIOperands()) {
            for (auto *prod : iOperand->GetProducers()) {
                if (prod->GetOpcode() == op.GetOpcode() &&
                    iOperand->GetConsumers().size() == 1) {
                    hasChainInput = true;
                    break;
                }
            }
            if (hasChainInput) { break; }
        }
        if (!hasChainInput) {
            continue;
        }
        roots.push_back(&op);
    }

    for (Operation *root : roots) {
        if (root->IsDeleted()) {
            continue;
        }
        if (ProcessRoot(*root, function) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation,
                "ArithmeticReordering failed on root op [%d].", root->GetOpMagic());
            return FAILED;
        }
    }

    function.EraseOperations(false);

    APASS_LOG_INFO_F(Elements::Operation,
        "End ArithmeticReordering for function [%s].", function.GetRawName().c_str());
    return SUCCESS;
}

} // namespace npu::tile_fwk