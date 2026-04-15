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
 * \file boundary_utils.cpp
 * \brief Utility functions for computing subgraph boundary status of tensors.
 */

#include "boundary_utils.h"

#include <queue>
#include <unordered_set>

#include "interface/tensor/logical_tensor.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation.h"

namespace npu::tile_fwk {
namespace {

bool HasCrossSubgraphConnections(const LogicalTensor& tensor)
{
    int subgraphId = -1;
    for (const auto& op : tensor.GetProducers()) {
        int opSubgraphId = op->GetSubgraphID();
        if (subgraphId == -1) {
            subgraphId = opSubgraphId;
        }
        if (subgraphId != opSubgraphId) {
            return true;
        }
    }
    for (const auto& op : tensor.GetConsumers()) {
        int opSubgraphId = op->GetSubgraphID();
        if (subgraphId == -1) {
            subgraphId = opSubgraphId;
        }
        if (subgraphId != opSubgraphId) {
            return true;
        }
    }
    return false;
}

bool IsSubGraphBoundaryBase(const LogicalTensor& tensor)
{
    // 1. No producers (graph input)
    if (tensor.GetProducers().empty()) {
        return true;
    }

    // 2. Producers/consumers span different subgraph IDs
    if (HasCrossSubgraphConnections(tensor)) {
        return true;
    }

    // 3. First input of any CopyIn operation
    for (const auto& consumer : tensor.GetConsumers()) {
        if (IsCopyIn(consumer->GetOpcode()) &&
            !consumer->GetIOperands().empty() &&
            consumer->GetIOperands().front().get() == &tensor) {
            return true;
        }
    }

    // 4. First output of any CopyOut operation (OP_COPY_OUT with inplaceIdx excluded)
    for (const auto& producer : tensor.GetProducers()) {
        if (IsCopyOut(producer->GetOpcode())) {
            if (producer->GetOpcode() == Opcode::OP_COPY_OUT &&
                producer->HasAttribute(OpAttributeKey::inplaceIdx)) {
                continue;
            }
            if (!producer->GetOOperands().empty() &&
                producer->GetOOperands().front().get() == &tensor) {
                return true;
            }
        }
    }

    // 5. Connected to ASSEMBLE where output is on DDR
    for (const auto& producer : tensor.GetProducers()) {
        if (producer->GetOpcode() == Opcode::OP_ASSEMBLE &&
            tensor.GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            return true;
        }
    }
    for (const auto& consumer : tensor.GetConsumers()) {
        if (consumer->GetOpcode() == Opcode::OP_ASSEMBLE &&
            !consumer->GetOOperands().empty() &&
            consumer->GetOOperands().front()->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            return true;
        }
    }

    return false;
}

bool IsDdrTensor(const LogicalTensor& tensor)
{
    return tensor.GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR;
}

bool TryVisitReshapeNeighbor(
    const LogicalTensor& current, const LogicalTensorPtr& neighbor, std::unordered_set<const LogicalTensor*>& visited,
    std::queue<const LogicalTensor*>& pending)
{
    if (neighbor == nullptr) {
        return false;
    }
    if (IsDdrTensor(current) && IsDdrTensor(*neighbor)) {
        return true;
    }
    if (!visited.insert(neighbor.get()).second) {
        return false;
    }
    if (IsSubGraphBoundaryBase(*neighbor)) {
        return true;
    }
    pending.push(neighbor.get());
    return false;
}

bool EnqueueReshapeNeighbors(
    const LogicalTensor& current, std::unordered_set<const LogicalTensor*>& visited,
    std::queue<const LogicalTensor*>& pending)
{
    for (const auto& producer : current.GetProducers()) {
        if (producer->GetOpcode() == Opcode::OP_RESHAPE && !producer->GetIOperands().empty()) {
            if (TryVisitReshapeNeighbor(current, producer->GetIOperands().front(), visited, pending)) {
                return true;
            }
        }
    }
    for (const auto& consumer : current.GetConsumers()) {
        if (consumer->GetOpcode() == Opcode::OP_RESHAPE && !consumer->GetOOperands().empty()) {
            if (TryVisitReshapeNeighbor(current, consumer->GetOOperands().front(), visited, pending)) {
                return true;
            }
        }
    }
    return false;
}

bool IsSubGraphBoundaryImpl(const LogicalTensor& tensor)
{
    if (IsSubGraphBoundaryBase(tensor)) {
        return true;
    }

    // RESHAPE chain propagation: BFS through RESHAPE connections.
    std::unordered_set<const LogicalTensor*> visited;
    std::queue<const LogicalTensor*> pending;
    visited.insert(&tensor);

    if (EnqueueReshapeNeighbors(tensor, visited, pending)) {
        return true;
    }

    while (!pending.empty()) {
        const auto* current = pending.front();
        pending.pop();
        if (EnqueueReshapeNeighbors(*current, visited, pending)) {
            return true;
        }
    }

    return false;
}

} // anonymous namespace

bool IsSubGraphBoundary(const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr) {
        return false;
    }
    return IsSubGraphBoundaryImpl(*tensor);
}

bool IsSubGraphBoundary(const LogicalTensor& tensor)
{
    return IsSubGraphBoundaryImpl(tensor);
}

} // namespace npu::tile_fwk
