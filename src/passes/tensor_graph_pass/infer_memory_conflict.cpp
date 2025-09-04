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

using namespace npu::tile_fwk;

Status InferMemoryConflict::RunOnFunction(Function &function) {
    ALOG_INFO_F("===> Start InferMemoryConflict for function [%s].", function.GetRawName().c_str());
    if (InferFromIncast(function) != SUCCESS) {
        ALOG_ERROR_F("Infer INCAST and OUTCAST address failed.");
        return FAILED;
    }
    if (InsertTensorCopy(function) != SUCCESS) {
        ALOG_ERROR_F("Insert copy op for INCAST-OUTCAST-same-address failed.");
        return FAILED;
    }
    ALOG_INFO_F("===> End InferMemoryConflict for function [%s].", function.GetRawName().c_str());
    return SUCCESS;
}

std::pair<Status, bool> InferMemoryConflict::IsInplace(Operation &op, std::shared_ptr<LogicalTensor> in, std::shared_ptr<LogicalTensor> out) const {
    if (inplaceRelationshipMap.find(op.GetOpcode()) == inplaceRelationshipMap.end()) {
        return std::make_pair(SUCCESS, false);
    }
    /* 先找到传入tensor对应当前op的输入和输入index */
    int inputIdx = -1;
    int outputIdx = -1;
    for (size_t i = 0; i < op.GetIOperands().size(); i++) {
        if (op.GetIOperands()[i]->magic == in->magic) {
            inputIdx = i;
            break;
        }
    }
    for (size_t i = 0; i < op.GetOOperands().size(); i++) {
        if (op.GetOOperands()[i]->magic == out->magic) {
            outputIdx = i;
            break;
        }
    }
    /* 没在当前op中找到输入或输出 */
    if (inputIdx == -1 || outputIdx == -1) {
        ALOG_ERROR_F("tesnor[%d] or tesnor[%d] is not the input or output for %s[%d]", in->magic, out->magic,
            op.GetOpcodeStr().c_str(), op.GetOpMagic());
        return std::make_pair(FAILED, false);
    }
    for (auto &reusePair : inplaceRelationshipMap.at(op.GetOpcode())) {
        if (reusePair == std::make_pair(static_cast<size_t>(inputIdx), static_cast<size_t>(outputIdx))) {
            return std::make_pair(SUCCESS, true);
        }
    }
    return std::make_pair(SUCCESS, false);
}

// 从INCAST出发，按DFS做前向推导
Status InferMemoryConflict::InferFromIncast(Function &function) {
    std::queue<std::shared_ptr<LogicalTensor>> leftTensors;
    for (auto &x_: function.GetIncast()) {
        /*
        初始状态下
        1. leftTensors仅记录了所有的INCAST
        2. 所有INCAST的parent为自己
        */
        ALOG_DEBUG_F("incast tensor magic %d, raw magic %d", x_->magic, x_->GetRawMagic());
        leftTensors.push(x_);
        parentRawForard[x_] = x_;
    }
    while (!leftTensors.empty()) {
        std::shared_ptr<LogicalTensor> currentTensor = leftTensors.front();
        for (auto &childOp: currentTensor->GetConsumers()) {
            /*
            currentTensor --> childOp --> out
            判断对childOp而言，指定输入和输出之间是否存在inplace 关系
            */
            for (auto &out : childOp->GetOOperands()) {
                /*
                currentTensor 时 childOp 的某个输入，遍历childOp的所有输出
                当这个输入和这个输出间存在inplace关系时：
                1. 将这个输出加入leftTensors中
                2. 将这个输出的parent记录为这个输入
                */
                std::pair<Status, bool> inplaceInfo = IsInplace(*childOp, currentTensor, out);
                if (inplaceInfo.first != SUCCESS) {
                    ALOG_ERROR_F("Find inplace relationship for %s[%d] filed.", childOp->GetOpcodeStr().c_str(), childOp->GetOpMagic());
                    return FAILED;
                }
                if (!inplaceInfo.second) {
                    continue;
                }
                leftTensors.push(out);
                parentRawForard[out] = parentRawForard.at(currentTensor);
                if (function.IsFromOutCast(out)) {
                    conflictTensors.emplace(out);
                }
            }
        }
        leftTensors.pop();
    }
    ALOG_INFO_F("========== conflict tensor size: %d ==========", conflictTensors.size());
    for (auto &t_ : conflictTensors) {
        ALOG_INFO_F("conflict tensor magic %d, raw magic %d", t_->magic, t_->GetRawMagic());
        auto finalParent = parentRawForard.at(t_);
        ALOG_INFO_F("Incast %d(raw %d, symbol %s), share memory with Outcast %d(raw %d, symbol %s).",
            finalParent->magic, finalParent->GetRawMagic(), finalParent->tensor->symbol.c_str(), t_->magic,
            t_->GetRawMagic(), t_->tensor->symbol.c_str());
    }
    return SUCCESS;
}

Status InferMemoryConflict::InsertTensorCopy(Function &function) {
    for (auto &t_ : conflictTensors) {
        auto incastParent = parentRawForard.at(t_);
        if (incastParent->tensor->symbol == t_->tensor->symbol) {
            /*
            symbol 相同说明前端写法为 a = op(a, ...)
            用户本意为原地inplace
            */
            continue;
        }
        // 获取前端显示声明的输入和输出同地址信息，不插入拷贝
        if (incastParent->tensor->memoryId == t_->tensor->memoryId) {
            ALOG_INFO_F("Incast %d(raw %d, symbol %s, memoryId: %d) have same GM address with Outcast %d(raw %d, "
                        "symbol %s, memoryId: %d), will not insert copy.",
                incastParent->magic, incastParent->GetRawMagic(), incastParent->tensor->symbol.c_str(),
                incastParent->tensor->memoryId, t_->magic, t_->GetRawMagic(), t_->tensor->symbol.c_str(),
                t_->tensor->memoryId);
            continue;
        }

        /*
        before:
        op1 --> tesnor --> Assemble1 -->\
                                        OCAST
        op2 --> tesnor --> Assemble2 -->/
        after:
        op1 --> tensor --> Register_Copy1 --> tensor_new --> Assemble1 -->\
                                                                        OCAST
        op2 --> tensor --> Register_Copy2 --> tensor_new --> Assemble2 -->/
        */
        for (auto &parentAssembleOp : t_->GetProducers()) {
            if (parentAssembleOp->GetOpcode() != Opcode::OP_ASSEMBLE) {
                 ALOG_DEBUG_F("OCAST magic: %d, raw: %d, symbol: %s,  has non-Assemble producer %s[%d]",
                    t_->magic, t_->GetRawMagic(), t_->tensor->symbol.c_str(),
                    parentAssembleOp->GetOpcodeStr().c_str(), parentAssembleOp->GetOpMagic());
                return FAILED;
            }

            auto producerParentOp = *parentAssembleOp->ProducerOps().begin();
            if (producerParentOp->GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
                continue;
            }
            auto assembleInput = parentAssembleOp->GetIOperands().front();
            std::shared_ptr<RawTensor> newRawTensor = std::make_shared<RawTensor>(assembleInput->Datatype(), assembleInput->tensor->rawshape);
            std::shared_ptr<LogicalTensor> newTensor = std::make_shared<LogicalTensor>(function, newRawTensor, assembleInput->offset, assembleInput->shape);
            auto &tensorCopyOp = function.AddRawOperation(Opcode::OP_REGISTER_COPY, {assembleInput}, {newTensor});

            /* Assemble 的前置op 为Reshape/NOP 时，需要从该op上获取tile shape */
            if ((producerParentOp->GetOpcode() == Opcode::OP_RESHAPE) || (producerParentOp->GetOpcode() == Opcode::OP_NOP)) {
                tensorCopyOp.UpdateTileShape(producerParentOp->GetTileShape());
            }
            assembleInput->RemoveConsumer(parentAssembleOp);
            parentAssembleOp->ReplaceInput(newTensor, assembleInput);
            ALOG_ERROR_F("******** insert %s[%d], may deteriorate the performance ********", tensorCopyOp.GetOpcodeStr().c_str(), tensorCopyOp.GetOpMagic());
            ALOG_ERROR_F("******** %s[%d] will expand as tile shape : %s ********", tensorCopyOp.GetOpcodeStr().c_str(),
                tensorCopyOp.GetOpMagic(), tensorCopyOp.GetTileShape().toString(TileType::VEC).c_str());
        }
    }
    return SUCCESS;
}
