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

#include "infer_discontinuous_input.h"
#include <queue>
#include "passes/pass_utils/graph_utils.h"

namespace npu {
namespace tile_fwk {
Status InferDiscontinuousInput::RunOnFunction(Function &function) {
    ALOG_INFO_F("===> Start InferMemoryConflict for function [%s].", function.GetRawName().c_str());
    Init(function);
    if (InferFromIncast(function) != SUCCESS) {
        ALOG_ERROR_F("Infer INCAST and OUTCAST address failed.");
        return FAILED;
    }
    if (InsertTensorCopy(function) != SUCCESS) {
        ALOG_ERROR_F("Insert copy op failed.");
        return FAILED;
    }
    ALOG_INFO_F("===> End InferMemoryConflict for function [%s].", function.GetRawName().c_str());
    return SUCCESS;
}

std::vector<std::pair<LogicalTensorPtr, Operation *>> GetInplacedTileTensors(LogicalTensorPtr targetTensor) {
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

std::vector<size_t> GetInputTileConflict(Function &function, const std::vector<LogicalTensorPtr> &inputTensors) {
    size_t incastIdx = 0;
    for (; incastIdx < inputTensors.size(); incastIdx++) {
        if (function.IsFromInCast(inputTensors[incastIdx])) {
            break;
        }
    }
    std::vector<size_t> copyIdx;
    for (size_t i = incastIdx; i < inputTensors.size(); i++) {
        if (function.IsFromInCast(inputTensors[i])) {
            if (inputTensors[incastIdx]->Symbol() != inputTensors[i]->Symbol() &&
                inputTensors[incastIdx]->GetRawTensor()->memoryId != inputTensors[i]->GetRawTensor()->memoryId) {
                copyIdx.push_back(i);
            }
        }
    }
    if (!copyIdx.empty()) {
        copyIdx.push_back(incastIdx);
    }
    return copyIdx;
}

std::vector<std::pair<LogicalTensorPtr, Operation *>> InferDiscontinuousInput::FilterCopyScenes(Function &function,
    LogicalTensorPtr targetTensor,
    const std::vector<std::pair<LogicalTensorPtr, Operation*>> &inplaceTensors) {
    std::vector<std::pair<LogicalTensorPtr, Operation *>> needInsertCopys;
    if (inplaceTensors.empty()) {
        return needInsertCopys;
    }
    auto targetParentIter = parentRawTensor_.find(targetTensor);
    if (targetParentIter == parentRawTensor_.end()) {
        std::vector<LogicalTensorPtr> inputTensors;
        for (auto inplaceTensor : inplaceTensors) {
            inputTensors.push_back(parentRawTensor_[inplaceTensor.first]);
        }
        auto copyIdx = GetInputTileConflict(function, inputTensors);
        for (auto idx : copyIdx) {
            needInsertCopys.push_back(inplaceTensors[idx]);
            ALOG_DEBUG_F("[MemConflict] Input tensor [%d] (parent tensor [%d]) conflit.", inplaceTensors[idx].first->GetMagic(),
                parentRawTensor_[inplaceTensors[idx].first]->GetMagic());
        }
        return needInsertCopys;
    }
    return needInsertCopys;
}

void InferDiscontinuousInput::Init(Function &function) {
    auto opList = function.Operations().DuplicatedOpList();
    for (size_t i = 0; i < opList.size(); ++i) {
        opInputDegree_.emplace(opList[i], opList[i]->ProducerOps().size());
        for (auto outTensor : opList[i]->GetOOperands()) {
            tensorProducers_[outTensor] = outTensor->GetProducers().size();
        }
    }
}

// 从INCAST出发，按DFS做前向推导
Status InferDiscontinuousInput::InferFromIncast(Function &function) {
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
    std::set<Operation *> visitedOps;
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
            tensorProducers_[outputTensor]--;
            std::vector<std::pair<LogicalTensorPtr, Operation *>> filterdTensor;
            if (tensorProducers_[outputTensor] != 0) {
                continue;
            }
            auto inplacedTensor = GetInplacedTileTensors(outputTensor);
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

Status InferDiscontinuousInput::InsertTensorCopy(Function &function) {
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
            ALOG_DEBUG_F("[MemConflict] Insert copy op [%d].", tensorCopyOp.GetOpMagic());
            inputTensor->RemoveConsumer(inplaceNode.second);
            inplaceNode.second->ReplaceInput(newTensor, inputTensor);
        }
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu