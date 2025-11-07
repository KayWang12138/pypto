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
 * \file infer_memory_conflict.cpp
 * \brief
 */

#include "infer_memory_conflict.h"
#include <queue>
#include "passes/pass_utils/graph_utils.h"

namespace npu {
namespace tile_fwk {
Status InferMemoryConflict::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(GetName().c_str(), "Operation", "Start InferMemoryConflict for function [%s].", function.GetRawName().c_str());
    Init(function);
    if (InferFromIncast(function) != SUCCESS) { 
        APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "Infer INCAST and OUTCAST address failed; Try to roll back changes.");
        return FAILED;
    }
    if (InsertTensorCopy(function) != SUCCESS) {
        APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "Insert copy op failed; Try to roll back changes.");
        return FAILED;
    }
    APASS_LOG_INFO_F(GetName().c_str(), "Operation", "End InferMemoryConflict for function [%s].", function.GetRawName().c_str());
    return SUCCESS;
}

bool InferMemoryConflict::IsValidTileShape(const Operation &op) const {
    auto input = op.GetIOperands().front();
    VecTile tileSize = op.GetTileShape().GetVecTile();
    if (input->GetShape().size() != tileSize.size()) {
        APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "%s[%d] has unequal input shape dims size and tile shape dims, input shape: %s, tile size: %s; Check the tile shape configuration.",
            op.GetOpcodeStr().c_str(), op.GetOpMagic(),
            input->DumpType().c_str(), op.GetTileShape().toString(TileType::VEC).c_str());
        return false;
    }
    return true;
}

std::vector<std::pair<LogicalTensorPtr, Operation *>> GetInplacedTensors(LogicalTensorPtr targetTensor) {
    std::set<Opcode> inplaceNodes{Opcode::OP_VIEW, Opcode::OP_ASSEMBLE, Opcode::OP_RESHAPE, Opcode::OP_INDEX_OUTCAST};
    std::vector<std::pair<LogicalTensorPtr, Operation *>> inplacedTensor;
    for (auto &producer : targetTensor->GetProducers()) {  
        if (inplaceNodes.count(producer->GetOpcode()) == 0) {
            continue;
        }
        if (producer->GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
            auto consumerOp = *(targetTensor->GetConsumers().begin());
            if (consumerOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
                continue;
            }
            const int index = 2;
            inplacedTensor.emplace_back(std::make_pair(producer->GetInputOperand(index), producer));
            continue;
        }
        for (auto &inputTensor : producer->GetIOperands()) {
            inplacedTensor.emplace_back(std::make_pair(inputTensor, producer));
        }
    }
    return inplacedTensor;
}

inline bool IsInOutConflict(Function &function, LogicalTensorPtr &inTensor, LogicalTensorPtr &outTensor) {
    if (!function.IsFromInCast(inTensor) || !function.IsFromOutCast(outTensor)) {
        APASS_LOG_ERROR_F("InferMemoryConflict", "Operation", "Input or output tensor is not from INCAST/OUTCAST.");
        return false;
    }
    if (inTensor->Symbol() == outTensor->Symbol()) {
        APASS_LOG_ERROR_F("InferMemoryConflict", "Operation", "Input or output tensor have the same symbol.");
        return false;
    }
    if (inTensor->GetRawTensor()->memoryId == outTensor->GetRawTensor()->memoryId) {
        APASS_LOG_ERROR_F("InferMemoryConflict", "Operation", "Input or output tensor have the same memoryID.");
        return false;
    }
    return true;
}

std::vector<std::pair<LogicalTensorPtr, Operation *>> InferMemoryConflict::FilterCopyScenes(Function &function,
    LogicalTensorPtr targetTensor,
    const std::vector<std::pair<LogicalTensorPtr, Operation*>> &inplaceTensors) {
    std::vector<std::pair<LogicalTensorPtr, Operation *>> needInsertCopys;
    if (inplaceTensors.empty()) {
        return needInsertCopys;
    }
    auto targetParentIter = parentRawTensor_.find(targetTensor);
    if (targetParentIter == parentRawTensor_.end()) {
        return needInsertCopys;
    }
    for (size_t i = 0; i < inplaceTensors.size(); ++i) {
        if (IsInOutConflict(function, parentRawTensor_[inplaceTensors[i].first], targetParentIter->second)) {
            APASS_LOG_DEBUG_F(GetName().c_str(), "Tensor", "Input tensor [%d] (parent tensor [%d]) is conflict with outcast [%d]; Need to insert a copy operation.",
                inplaceTensors[i].first->GetMagic(), parentRawTensor_[inplaceTensors[i].first]->GetMagic(), targetParentIter->second->GetMagic());
            needInsertCopys.emplace_back(inplaceTensors[i]);
        }
    }
    return needInsertCopys;
}

void InferMemoryConflict::Init(Function &function) {
    auto opList = function.Operations().DuplicatedOpList();
    for (size_t i = 0; i < opList.size(); ++i) {
        opInputDegree_.emplace(opList[i], opList[i]->ProducerOps().size());
    }
}

// 从INCAST出发，按DFS做前向推导
Status InferMemoryConflict::InferFromIncast(Function &function) {
    std::queue<Operation *> procOpQueue;
    for (auto &opInputDegree : opInputDegree_) {
        if (opInputDegree.second == 0) {
            procOpQueue.push(opInputDegree.first);
        }
    }
    for (auto &incast : function.GetIncast()) {
       parentRawTensor_[incast] = incast;
    }
    for (auto &outcast : function.GetOutcast()) {
        parentRawTensor_[outcast] = outcast;
    }
    std::unordered_set<Operation *> visitedOps;
    while (!procOpQueue.empty()) {
        auto currentOp = procOpQueue.front();
        procOpQueue.pop();
        visitedOps.insert(currentOp);
        for (auto outOp : currentOp->ConsumerOps()) {
            opInputDegree_[outOp]--;
            if (opInputDegree_[outOp] == 0) {
                procOpQueue.push(outOp);
            }
        }
        for (auto &outputTensor : currentOp->GetOOperands()) {
            bool allInputReady = std::all_of(outputTensor->GetProducers().begin(), outputTensor->GetProducers().end(),
                [&visitedOps](Operation *producerOp) { return visitedOps.count(producerOp) > 0U; });
            std::vector<std::pair<LogicalTensorPtr, Operation *>> filterdTensor;
            if (!allInputReady) {
                continue;
            }
            auto inplacedTensor = GetInplacedTensors(outputTensor);
            filterdTensor = FilterCopyScenes(function, outputTensor, inplacedTensor);
            insertCopys_.emplace(outputTensor, filterdTensor);
            if (!filterdTensor.empty()) {
                parentRawTensor_[outputTensor] = outputTensor;
                continue;
            }
            if (inplacedTensor.empty()) {
                parentRawTensor_[outputTensor] = outputTensor;
            } else {
                parentRawTensor_[outputTensor] = parentRawTensor_[inplacedTensor[0].first];
            }
        }
    }
    return SUCCESS;
}

Status InferMemoryConflict::InsertTensorCopy(Function &function) {
    std::map<LogicalTensorPtr, std::set<Operation *>> insertedNodes;
    for (auto &copyInserts : insertCopys_) {
        auto &inplaceNodes = copyInserts.second;
        for (auto &inplaceNode : inplaceNodes) {
            auto &inputTensor = inplaceNode.first;
            if (insertedNodes.find(inputTensor) != insertedNodes.end()) {
                if (insertedNodes[inputTensor].count(inplaceNode.second) != 0U) {
                    continue;
                }
            }
            insertedNodes[inputTensor].insert(inplaceNode.second);
            std::shared_ptr<RawTensor> newRawTensor = std::make_shared<RawTensor>(inputTensor->Datatype(),
                inputTensor->GetShape());
            Offset newOffset(inputTensor->GetShape().size(), 0);
            LogicalTensorPtr newTensor = std::make_shared<LogicalTensor>(function, newRawTensor, newOffset,
                inputTensor->GetShape(), inputTensor->GetDynValidShape());
            auto &tensorCopyOp = function.AddRawOperation(Opcode::OP_REGISTER_COPY, {inputTensor}, {newTensor});
            APASS_LOG_DEBUG_F(GetName().c_str(), "Operation", "Insert copy op [%d];", tensorCopyOp.GetOpMagic());
            auto producerParentOp = *(inplaceNode.second->ProducerOps().begin());
            auto tileShapeSize = producerParentOp->GetTileShape().GetVecTile().size();
            if (tileShapeSize == 0 || tileShapeSize != inputTensor->GetShape().size()) {
                APASS_LOG_WARN_F(GetName().c_str(), "Operation", "Inserted op's producerop [%d] has no tile shape.", producerParentOp->GetOpMagic());
                TileShape tile;
                std::vector<int64_t> defaultTile(inputTensor->GetShape().size(), 1);
                const int64_t defaultTileSize = 128;
                const size_t defaultShapeLen = 2;
                for (size_t i = 0; inputTensor->GetShape().size() >= (i + 1) && i < defaultShapeLen; ++i) {
                    defaultTile[inputTensor->GetShape().size() - i - 1] = defaultTileSize;
                }
                tile.SetVecTile(defaultTile);
                tensorCopyOp.UpdateTileShape(tile);
            } else {
                tensorCopyOp.UpdateTileShape(producerParentOp->GetTileShape());
            }
            if (!IsValidTileShape(tensorCopyOp)) {
                APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "Invalid tile size for [%d]; Check if the previous logs for Tensor shape and Tile Shape setting details.", tensorCopyOp.GetOpMagic());
                return FAILED;
            }
            inputTensor->RemoveConsumer(inplaceNode.second);
            inplaceNode.second->ReplaceInput(newTensor, inputTensor);
        }
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu