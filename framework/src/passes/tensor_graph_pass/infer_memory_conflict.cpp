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
#include "passes/pass_utils/graph_utils.h"

namespace npu {
namespace tile_fwk {
namespace {
uint32_t GetPowerOfTwo(uint32_t cur) {
    uint32_t ret = 1;
    while (ret < cur) {
        ret <<= 1;
    }
    return ret;
}
}

Status InferMemoryConflict::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(GetName().c_str(), "Operation", "Start InferMemoryConflict for function [%s].", function.GetRawName().c_str());
    if (Init(function) != SUCCESS) {
        APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "Init failed.");
        return FAILED;
    }
    if (ForwardPropagation(function) != SUCCESS) {
        APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "ForwardPropagation failed.");
        return FAILED;
    }
    if (BackwardPropagation(function) != SUCCESS) {
        APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "BackwardPropagation failed.");
        return FAILED;
    }
    if (InsertCopys(function) != SUCCESS) {
        APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "InsertCopys failed.");
        return FAILED;
    }
    APASS_LOG_INFO_F(GetName().c_str(), "Operation", "End InferMemoryConflict for function [%s].", function.GetRawName().c_str());
    return SUCCESS;
}

bool InferMemoryConflict::CheckConflict(const LogicalTensorPtr &inTensor, const LogicalTensorPtr &outTensor) {
    if (inTensor->Symbol() == outTensor->Symbol()) {
        return false;
    }
    if (inTensor->GetRawTensor()->memoryId == outTensor->GetRawTensor()->memoryId) {
        return false;
    }
    return true;
}

bool InferMemoryConflict::CheckRawShapeConflict(const LogicalTensorPtr &inTensor, const LogicalTensorPtr &outTensor) {
    int64_t inRawSize = 1;
    int64_t outRawSize = 1;
    Shape inShape = inTensor->GetRawTensor()->GetRawShape();
    Shape outShape = outTensor->GetRawTensor()->GetRawShape();
    for (size_t i = 0; i < inShape.size(); ++i) {
        inRawSize *= inShape[i];
    }
    for (size_t i = 0; i < outShape.size(); ++i) {
        outRawSize *= outShape[i];
    }
    if (inRawSize > 0 && outRawSize > 0 && inRawSize != outRawSize) {
        APASS_LOG_DEBUG_F(GetName().c_str(), "Operation", "The raw size of input is %d, the raw size of output is %d", inRawSize, outRawSize);
        return true;
    }
    return false;
}

bool InferMemoryConflict::CheckTransmit(Operation* curOp) {
    LogicalTensorPtr curTensor;
    std::set<Opcode> NonCalcNode = {Opcode::OP_VIEW, Opcode::OP_ASSEMBLE, Opcode::OP_RESHAPE, Opcode::OP_INDEX_OUTCAST};
    bool transmit = (NonCalcNode.find(curOp->GetOpcode()) != NonCalcNode.end());
    if (curOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
        curTensor = *(curOp->GetIOperands().begin());
        for (const auto &producer : curTensor->GetProducers()) {
            if (producer->GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
                transmit = false;
            }
        }
    }
    return transmit;
}

bool InferMemoryConflict::IsValidTileShape(const Operation &op) const {
    auto input = op.GetIOperands().front();
    VecTile tileSize = op.GetTileShape().GetVecTile();
    if (input->GetShape().size() != tileSize.size()) {
        APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "%s[%d] has unequal input shape dims size and tile shape dims, input shape: %s, tile size: %s", 
                            op.GetOpcodeStr().c_str(), op.GetOpMagic(),
                            input->DumpType().c_str(), op.GetTileShape().toString(TileType::VEC).c_str());
        return false;
    }
    APASS_LOG_DEBUG_F(GetName().c_str(), "Operation", "The size info of %s[%d]: input shape: %s, tile size: %s", op.GetOpcodeStr().c_str(), op.GetOpMagic(),
            input->DumpType().c_str(), op.GetTileShape().toString(TileType::VEC).c_str());
    return true;
}

Status InferMemoryConflict::UpdateForwardTensor(Function &function, const LogicalTensorPtr &curTensor, Operation* consumer, std::queue<LogicalTensorPtr> &curTensors) {
    for (auto &outputTensor : consumer->GetOOperands()) {
        if (consumer->GetOpcode() == Opcode::OP_RESHAPE && CheckRawShapeConflict(memoryInfo[curTensor], outputTensor)) {
            preregcopys.insert(consumer);
        } else if (memoryInfo.find(outputTensor) != memoryInfo.end() && function.IsFromOutCast(memoryInfo[outputTensor])) {
            if (CheckConflict(memoryInfo[curTensor], memoryInfo[outputTensor])) {
                preregcopys.insert(consumer);
            }
        } else {
            if (consumer->GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
                int index = 2;
                memoryInfo[outputTensor] = memoryInfo[consumer->GetInputOperand(index)];
            } else {
                memoryInfo[outputTensor] = memoryInfo[curTensor];
            }
            curTensors.push(outputTensor);
        }
    }
    return SUCCESS;
}

Status InferMemoryConflict::UpdateBackwardTensor(const LogicalTensorPtr &curTensor, Operation* producer, std::queue<LogicalTensorPtr> &curTensors) {
    for (auto &inputTensor : producer->GetIOperands()) {
        int index = 2;
        if (producer->GetOpcode() == Opcode::OP_INDEX_OUTCAST && producer->GetIOperandIndex(inputTensor) != index) {
            continue;
        }
        if (producer->GetOpcode() == Opcode::OP_RESHAPE && CheckRawShapeConflict(inputTensor, memoryInfo[curTensor])) {
            postregcopys.insert(producer);
        } else if (memoryInfo.find(inputTensor) != memoryInfo.end()) {
            if (CheckConflict(memoryInfo[curTensor], memoryInfo[inputTensor])) {
                if (producer->GetOpcode() == Opcode::OP_RESHAPE) {
                    postregcopys.insert(producer);
                } else {
                    preregcopys.insert(producer);
                }
            }
        } else {
            memoryInfo[inputTensor] = memoryInfo[curTensor];
            curTensors.push(inputTensor);
        }
    }
    return SUCCESS;
}

Status InferMemoryConflict::ForwardPropagation(Function &function) {
    std::queue<LogicalTensorPtr> curTensors;
    for (auto &incast : function.GetIncast()) {
        curTensors.push(incast);
    }
    while (!curTensors.empty()) {
        auto curTensor = curTensors.front();
        curTensors.pop();
        for (const auto &consumer : curTensor->GetConsumers()) {
            if (!CheckTransmit(consumer)) {
                continue;
            }
            int index = 2;
            if (consumer->GetOpcode() == Opcode::OP_INDEX_OUTCAST && consumer->GetIOperandIndex(curTensor) != index) {
                continue;
            }
            if (UpdateForwardTensor(function, curTensor, consumer, curTensors) != SUCCESS) {
                APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "UpdateForwardTensor failed.");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status InferMemoryConflict::BackwardPropagation(Function &function) {
    std::queue<LogicalTensorPtr> curTensors;
    for (auto &outcast : function.GetOutcast()) {
        curTensors.push(outcast);
    }
    while (!curTensors.empty()) {
        auto curTensor = curTensors.front();
        curTensors.pop();
        for (const auto &producer : curTensor->GetProducers()) {
            if (!CheckTransmit(producer)) {
                continue;
            }
            if (UpdateBackwardTensor(curTensor, producer, curTensors) != SUCCESS) {
                APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "UpdateBackwardTensor failed.");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status InferMemoryConflict::SetDefaultShape(const LogicalTensorPtr &tensor, std::vector<int64_t> &defaultTile) {
    int64_t maximalTileSize = 16384;
    int64_t alignTailSize = 32;
    Shape shape = tensor->GetShape();
    size_t shapeDim = shape.size();
    int64_t curTile;
    defaultTile.clear();
    for (size_t i = 0; i < shape.size(); ++i) {
        defaultTile.emplace_back(1);
    }
    curTile = shape[shapeDim - 1] < alignTailSize ? alignTailSize : GetPowerOfTwo(shape[shapeDim - 1]);
    defaultTile[shapeDim - 1] = maximalTileSize < curTile ? maximalTileSize : curTile;
    for (int i = shapeDim - 2; i >= 0; --i) {
        maximalTileSize /= defaultTile[i + 1];
        curTile = GetPowerOfTwo(shape[i]);
        defaultTile[i] = maximalTileSize < curTile ? maximalTileSize : curTile;
        defaultTile[i] = defaultTile[i] == 0 ? 1 : defaultTile[i];
    }
    return SUCCESS;
}

Status InferMemoryConflict::InferTileShape(Operation &op, Operation *parentOp, const LogicalTensorPtr &tensor) {
    auto tileShapeSize = parentOp->GetTileShape().GetVecTile().size();
    if (tileShapeSize == 0 || tileShapeSize != tensor->GetShape().size()) {
        APASS_LOG_WARN_F(GetName().c_str(), "Operation", "Inserted op's producer/consumer op [%d] has no tile shape.", parentOp->GetOpMagic());
        TileShape tile;
        std::vector<int64_t> defaultTile;
        if (SetDefaultShape(tensor, defaultTile) != SUCCESS) {
            APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "SetDefaultShape failed.");
            return FAILED;
        }
        tile.SetVecTile(defaultTile);
        op.UpdateTileShape(tile);
    } else {
        op.UpdateTileShape(parentOp->GetTileShape());
    }
    if (!IsValidTileShape(op)) {
        APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "Invalid tile size for %s[%d].", op.GetOpcodeStr().c_str(), op.GetOpMagic());
        return FAILED;
    }
    return SUCCESS;
}

Status InferMemoryConflict::InsertPrecededCopys(Function &function) {
    for (const auto op : preregcopys) {
        LogicalTensorPtr inputTensor = op->GetIOperands().front();
        std::shared_ptr<RawTensor> newRawTensor = std::make_shared<RawTensor>(inputTensor->Datatype(), inputTensor->GetShape());
        Offset newOffset(inputTensor->GetShape().size(), 0);
        LogicalTensorPtr newTensor = std::make_shared<LogicalTensor>(function, newRawTensor, newOffset, inputTensor->GetShape(), inputTensor->GetDynValidShape());
        auto &copyOp = function.AddRawOperation(Opcode::OP_REGISTER_COPY, {inputTensor}, {newTensor});
        APASS_LOG_DEBUG_F(GetName().c_str(), "Operation", "Insert copy op [%d].", copyOp.GetOpMagic());
        if (InferTileShape(copyOp, *(copyOp.ProducerOps().begin()), inputTensor) != SUCCESS) {
            APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "InferTileShape failed.");
            return FAILED;
        }
        inputTensor->RemoveConsumer(op);
        op->ReplaceInput(newTensor, inputTensor);
    }
    return SUCCESS;
}

Status InferMemoryConflict::InsertPostCopys(Function &function) {
    for (const auto op : postregcopys) {
        LogicalTensorPtr outputTensor = op->GetOOperands().front();
        std::shared_ptr<RawTensor> newRawTensor = std::make_shared<RawTensor>(outputTensor->Datatype(), outputTensor->GetShape());
        Offset newOffset(outputTensor->GetShape().size(), 0);
        LogicalTensorPtr newTensor = std::make_shared<LogicalTensor>(function, newRawTensor, newOffset, outputTensor->GetShape(), outputTensor->GetDynValidShape());
        auto &copyOp = function.AddRawOperation(Opcode::OP_REGISTER_COPY, {newTensor}, {outputTensor});
        APASS_LOG_DEBUG_F(GetName().c_str(), "Operation", "Insert copy op [%d].", copyOp.GetOpMagic());
        if (InferTileShape(copyOp, *(copyOp.ConsumerOps().begin()), outputTensor) != SUCCESS) {
            APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "InferTileShape failed.");
            return FAILED;
        }
        outputTensor->RemoveConsumer(op);
        op->ReplaceOutput(newTensor, outputTensor);
    }
    return SUCCESS;
}

Status InferMemoryConflict::InsertCopys(Function &function) {
    if (InsertPrecededCopys(function) != SUCCESS) {
        APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "InsertPrecededCopys failed.");
        return FAILED;
    }
    if (InsertPostCopys(function) != SUCCESS) {
        APASS_LOG_ERROR_F(GetName().c_str(), "Operation", "InsertPostCopys failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status InferMemoryConflict::Init(Function &function) {
    for (auto &incast : function.GetIncast()) {
        memoryInfo[incast] = incast;
    }
    for (auto &outcast : function.GetOutcast()) {
        memoryInfo[outcast] = outcast;
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
