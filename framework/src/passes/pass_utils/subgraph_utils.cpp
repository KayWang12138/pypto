/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file subgraph_utils.cpp
 * \brief
 */

#include <unordered_set>
#include "subgraph_utils.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/operation/operation.h"

namespace npu::tile_fwk {
namespace {
    bool HasCrossSubgraphEdges(const LogicalTensor& tensor)
    {
        int expectedSubgraphId = -1;

        auto check_op = [&](Operation* op) {
            if (op == nullptr) return false;
            int opSubgraphId = op->GetSubgraphID();
            if (expectedSubgraphId == -1) {
                expectedSubgraphId = opSubgraphId;
            } else if (expectedSubgraphId != opSubgraphId) {
                return true;
            }
            return false;
        };

        for (const auto& op : tensor.GetProducers()) {
            if (check_op(op)) return true;
        }
        for (const auto& op : tensor.GetConsumers()) {
            if (check_op(op)) return true;
        }
        return false;
    }

    bool IsBaseBoundary(const LogicalTensor& tensor)
    {
        if (tensor.GetProducers().empty() && !tensor.GetConsumers().empty()) {
            return true;
        }

        // 跨子图边缘
        if (HasCrossSubgraphEdges(tensor)) {
            return true;
        }

        // 当Tensor作为输入时
        for (const auto& op : tensor.GetConsumers()) {
            // Tensor为Copy In的输入
            if (op->GetOpcode() == Opcode::OP_COPY_IN) {
                if (!op->GetIOperands().empty() && op->GetIOperands().front().get() == &tensor) return true;
            }
            // DDR上的Assemble
            if (op->GetOpcode() == Opcode::OP_ASSEMBLE) {
                if (!op->GetOOperands().empty() && 
                    op->GetOOperands().front()->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) return true;
            }
        }

        // 当Tensor作为输出时
        for (const auto& op : tensor.GetProducers()) {
            // Tensor为Copy Out的输出
            if (op->GetOpcode() == Opcode::OP_COPY_OUT) {
                if (!op->HasAttribute(OpAttributeKey::inplaceIdx) && 
                    !op->GetOOperands().empty() && op->GetOOperands().front().get() == &tensor) {
                    return true;
                }
            }
            // DDR上的Assemble
            if (op->GetOpcode() == Opcode::OP_ASSEMBLE) {
                if (tensor.GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) return true;
            }
            // Tensor为Reshape的输出
            if (op->GetOpcode() == Opcode::OP_RESHAPE) {
                if (!op->GetIOperands().empty() && 
                    op->GetIOperands().front()->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
                    tensor.GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
                    return true;
                }
            }
        }

        return false;
    }

    bool IsBoundaryImpl(const LogicalTensor& tensor, std::unordered_set<const LogicalTensor*>& visited)
    {
        if (!visited.insert(&tensor).second) {
            return false;
        }

        if (IsBaseBoundary(tensor)) {
            return true;
        }

        // Reshape边界属性传播
        for (const auto& op : tensor.GetProducers()) {
            if (op->GetOpcode() == Opcode::OP_RESHAPE && !op->GetIOperands().empty()) {
                if (IsBoundaryImpl(*(op->GetIOperands().front()), visited)) return true;
            }
        }
        for (const auto& op : tensor.GetConsumers()) {
            if (op->GetOpcode() == Opcode::OP_RESHAPE && !op->GetOOperands().empty()) {
                if (IsBoundaryImpl(*(op->GetOOperands().front()), visited)) return true;
            }
        }

        return false;
    }

    } // anonymous namespace

    bool SubgraphUtils::IsBoundary(const LogicalTensorPtr& tensor)
    {
        if (tensor == nullptr) {
            return false;
        }
        std::unordered_set<const LogicalTensor*> visited;
        return IsBoundaryImpl(*tensor, visited);
    }

    bool SubgraphUtils::IsBoundary(const LogicalTensor& tensor)
    {
        std::unordered_set<const LogicalTensor*> visited;
        return IsBoundaryImpl(tensor, visited);
    }

} // namespace npu::tile_fwk