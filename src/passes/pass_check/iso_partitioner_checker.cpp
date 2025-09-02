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
 * \file iso_paritioner_checker.cpp
 * \brief
 */

#include "iso_partitioner_checker.h"

namespace npu {
namespace tile_fwk {
Status GraphPartitionChecker::DoPreCheck(Function &function) {
    ALOG_INFO_F("PreCheck for pass: GraphPartition.");
    if (CheckValidOp(function) != SUCCESS) {
        return FAILED;
    }
    return SUCCESS;
}

Status GraphPartitionChecker::DoPostCheck(Function &function) {
    ALOG_INFO("PostCheck for pass: GraphPartition.");
    std::vector<std::vector<Operation*>> subgraphs(function.GetTotalSubGraphCount());
    for (auto &op : function.Operations()) {
        int32_t curSubgraphID = op.GetSubgraphID();
        if (curSubgraphID == -1) {
            ALOG_ERROR_F("Operation (opmagic: %d) is not in any subgraph.", op.GetOpMagic());
            return FAILED;
        }
        if (curSubgraphID < 0 || curSubgraphID >= static_cast<int32_t>(function.GetTotalSubGraphCount())) {
            ALOG_ERROR_F("Operation (opmagic: %d) has illegal SubgraphID.", op.GetOpMagic());
            return FAILED;
        }
        subgraphs[curSubgraphID].push_back(&op);
    }
    for (int graphID = 0; graphID < static_cast<int>(subgraphs.size()); graphID++) {
        if (subgraphs[graphID].size() == 0) {
            ALOG_ERROR_F("Subgraph %d includes no Operation.", graphID);
            return FAILED;
        }
    }
    if (!function.LoopCheck().empty()) {
        ALOG_ERROR("Loopcheck failed after pass: GraphPartition.");
        return FAILED;
    }
    if (PostOperationCheck(function) != SUCCESS) {
        ALOG_ERROR_F("Operation post check failed.");
        return FAILED;
    }
    if (PostSubgraphCheck(subgraphs) != SUCCESS) {
        ALOG_ERROR_F("Subgraph post check failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status GraphPartitionChecker::PostOperationCheck(Function &function) {
    for (auto &op : function.Operations()) {
        int32_t curSubgraphID = op.GetSubgraphID();
        bool isStartNodeInSubgraph = true;
        for (auto iTensor : op.GetIOperands()) {
            for (auto &producer : iTensor->GetProducers()) {
                if (curSubgraphID == producer->GetSubgraphID()) {
                    isStartNodeInSubgraph = false;
                    break;
                }
            }
            if (!isStartNodeInSubgraph) {
                break;
            }
        }
        if (isStartNodeInSubgraph) {
            ALOG_DEBUG_F("Operation (opmagic: %d) is start node in subgraph.", op.GetOpMagic());
            if (op.GetOpcode() == Opcode::OP_ASSEMBLE || op.GetOpcode() == Opcode::OP_COPY_OUT) {
                ALOG_ERROR_F("Operation (opmagic: %d) is the start node of the subgraph, opcode should not be %s.",
                             op.GetOpMagic(), op.GetOpcodeStr().c_str());
                return FAILED;
            }
            continue;
        }
        bool isEndNodeInSubgraph = true;
        for (auto oTensor : op.GetOOperands()) {
            for (auto &consumer : oTensor->GetConsumers()) {
                if (curSubgraphID == consumer->GetSubgraphID()) {
                    isEndNodeInSubgraph = false;
                    break;
                }
            }
            if (!isEndNodeInSubgraph) {
                break;
            }
        }
        if (isEndNodeInSubgraph) {
            ALOG_DEBUG_F("Operation (opmagic: %d) is end node in subgraph.", op.GetOpMagic());
            if (op.GetOpcode() == Opcode::OP_VIEW || op.GetOpcode() == Opcode::OP_COPY_IN) {
                ALOG_ERROR_F("Operation (opmagic: %d) is the end node of the subgraph, opcode should not be %s.",
                             op.GetOpMagic(), op.GetOpcodeStr().c_str());
                return FAILED;
            }
            continue;
        }
    }
    return SUCCESS;
}

Status GraphPartitionChecker::PostSubgraphCheck(const std::vector<std::vector<Operation*>> &subgraphs) {
    for (auto subgraph : subgraphs) {
        if (subgraph.empty()) {
            continue;
        }
        int32_t aicCount = 0;
        int32_t aivCount = 0;
        int32_t aicMemoryCount = 0;
        int32_t aivMemoryCount = 0;
        std::unordered_set<std::shared_ptr<LogicalTensor>> tensorList;
        int32_t subgraphId = subgraph[0]->GetSubgraphID();
        for (auto &op : subgraph) {
            for (auto iTensor : op->GetIOperands()) {
                if (tensorList.find(iTensor) != tensorList.end()) {
                    continue;
                }
                tensorList.insert(iTensor);
                if (iTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L1 ||
                    iTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0A ||
                    iTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0B ||
                    iTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0C) {
                    aicMemoryCount++;
                    continue;
                }
                if (iTensor->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
                    aivMemoryCount++;
                }
            }
            if (op->HasAttr(OpAttributeKey::isCube) && op->GetBoolAttribute(OpAttributeKey::isCube)) {
                aicCount++;
                continue;
            }
            aivCount++;
        }
        if (aicCount > 0 && aivCount > 0) {
            ALOG_ERROR_F("Subgraph %d has both AIV and AIC operation.", subgraphId);
            return FAILED;
        }
        if (aicMemoryCount > 0 && aivMemoryCount > 0) {
            ALOG_ERROR_F("Subgraph %d has both ub and l0/l1 memory type tensor.", subgraphId);
            return FAILED;
        }
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu