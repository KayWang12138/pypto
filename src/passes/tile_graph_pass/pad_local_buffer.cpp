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
 * \file pad_local_buffer.cpp
 * \brief
 */

#include "pad_local_buffer.h"

namespace npu::tile_fwk {
constexpr size_t MATMUL_MIN_SHAPE_SIZE = 2;
constexpr size_t VECTOR_MIN_SHAPE_SIZE = 1;
constexpr size_t AXIS_COMBINE_MIN_SHAPE_SIZE = 2;
constexpr size_t TRANSPOSE_MIN_SHAPE_SIZE = 2;
constexpr size_t BROADCAST_OP_INPUT_SIZE = 2;
/* 记录不同matmul的padding值的index */
constexpr size_t CUBE_INPUT_SIZE = 4;
constexpr size_t MATMUL_AXIS_NUM = 2;
constexpr size_t MATRIX_AND_INDEX_NUM = 2;
constexpr size_t HIGH_INDEX = 0;
constexpr size_t LOW_INDEX = 1;
constexpr uint32_t LEFT_SHIFT32 = 32;
constexpr int64_t CUBE_PAD_VALUE = 16;
const std::vector<bool> AXIS_COMBINED = {true};
const std::vector<bool> BROADCAST_AXIS_COMBINED = {true, true};

int64_t Pad(int64_t dim, int64_t padValue) {
    return (dim + padValue - 1) / padValue * padValue;
}

void PadLocalBuffer::PadMatmul(Operation &op, LogicalTensorPtr &in) {
    if (in->shape.size() < MATMUL_MIN_SHAPE_SIZE) {
        ASLOGE("Matmul Op %d %s input %d shape size is less than 2", op.opmagic, op.GetOpcodeStr().c_str(), in->magic);
        return;
    }

    auto highIndex = in->shape.size() - 2; // matmul高轴
    auto lowIndex = in->shape.size() - 1;  // matmul低轴

    in->shape[highIndex] = Pad(in->shape[highIndex], CUBE_PAD_VALUE);
    in->shape[lowIndex] = Pad(in->shape[lowIndex], CUBE_PAD_VALUE);

    ALOG_DEBUG_F("####### %d original shape is %s\n", in->magic, IntVecToStr(in->oriShape).c_str());
    ALOG_DEBUG_F("####### %d #current shape is %s\n", in->magic, IntVecToStr(in->shape).c_str());
    in->tensor->oriRawshape = in->tensor->rawshape;
    in->tensor->rawshape[highIndex] = Pad(in->tensor->oriRawshape[highIndex], CUBE_PAD_VALUE);
    in->tensor->rawshape[lowIndex] = Pad(in->tensor->oriRawshape[lowIndex], CUBE_PAD_VALUE);
    ALOG_DEBUG_F("####### %d %d set rawshape as %s\n", in->tensor->rawmagic, in->magic,
        IntVecToStr(in->tensor->rawshape).c_str());
}

size_t PadLocalBuffer::GetPaddingValue(Operation &op, LogicalTensorPtr &in, OpCalcType calcType) {
    bool needTilePadding = op.GetBoolAttribute(OpAttributeKey::tilePadding);
    if (!needTilePadding && calcType == OpCalcType::MOVE_OUT) {
        auto &producers = in->GetProducers();
        for (auto &prod : producers) {
            needTilePadding = needTilePadding || prod->GetBoolAttribute(OpAttributeKey::tilePadding);
        }
    }
    if (needTilePadding) {
        const auto &vTileShape = op.GetTileShape().GetVecTileShapes();
        if (vTileShape.size() != in->shape.size()) {
            ALOG_DEBUG_F("VTileShape [size %zu] dims %s and shape size %zu mismatch for of op %d %s.", vTileShape.size(),
                IntVecToStr(vTileShape).c_str(), in->shape.size(), op.opmagic, op.GetOpcodeStr().c_str());
            return 1;
        }
        return vTileShape[in->shape.size() - 1];
    } else {
        auto bytes = BytesOf(in->Datatype());
        auto paddingIter = BLOCK_PADDING_DIM.find(bytes);
        if (paddingIter == BLOCK_PADDING_DIM.end()) {
            return 1;
        }
        return paddingIter->second;
    }
    return 0;
}

/* 1. 对于非BroadcastOp，默认做到Block对齐；
   2. 如果已经对齐到Block粒度，不做对齐---这里存在一个问题就是f16和fp32混用场景，可能对齐到一个block是不够的
   3. 对于broadcast op，如果shape小于一个Block的大小对齐到Block，否则做到两个输入之间的较大者的Block对齐。 */
void PadLocalBuffer::PadVector(Operation &op, LogicalTensorPtr &in, std::unordered_set<std::shared_ptr<RawTensor>> &visitedRaw,
    bool noPadding) {
    if (in->shape.empty()) {
        ASLOGE("Vector Op %d %s input %d shape size is less than 2", op.opmagic, op.GetOpcodeStr().c_str(), in->magic);
        return;
    }
    OpCalcType calcType = OpcodeManager::Inst().GetOpCalcType(op.GetOpcode());
    if (noPadding) {
        in->oriShape = in->shape;
        in->tensor->UpdateRawShape(in->shape);
        in->tensor->oriRawshape = in->tensor->rawshape;
        ALOG_DEBUG_F("Vector Op %d %s input %d, not handle unalign.", op.opmagic, op.GetOpcodeStr().c_str(), in->magic);
        return;
    }
    size_t paddingValue =  GetPaddingValue(op, in, calcType);
    size_t lastIdx = in->shape.size() - 1;
    in->oriShape = in->shape;
    int64_t lastDim = static_cast<int64_t>(in->shape[lastIdx]);
    if (calcType == OpCalcType::BROADCAST && op.HasAttr(OpAttributeKey::broadcastLastAxis)) {
        lastDim = op.GetIntAttribute(OpAttributeKey::broadcastLastAxis);
    }
    int64_t shapeAfterPad = Pad(lastDim, paddingValue);
    in->shape[lastIdx] = shapeAfterPad;
    if (in->shape[lastIdx] != in->oriShape[lastIdx]) {
        ALOG_DEBUG_F("op %d %s input has been changed\n", op.opmagic, op.GetOpcodeStr().c_str());
    }

    if (visitedRaw.count(in->tensor) == 0) {
        in->tensor->oriRawshape = in->tensor->rawshape;
        // shape已经对齐过，直接将rawShape对齐到shape；如果broadcast的输入是来自于view，那么整个链路上的非对齐shape都要按照
        // BROADCAST_LAST_AXIS来对齐，当前这样处理是有问题的
        in->tensor->rawshape[lastIdx] = Pad(in->tensor->oriRawshape[lastIdx], in->shape[lastIdx]);
        visitedRaw.emplace(in->tensor);
    }
    if (in->tensor->oriRawshape[lastIdx] != in->tensor->rawshape[lastIdx]) {
        ALOG_DEBUG_F(
            "op %d %s input has been changed, setAttribute shapepadded\n", op.opmagic, op.GetOpcodeStr().c_str());
        op.SetAttribute(OpAttributeKey::shapePadded, true);
        for (auto &prod : in->GetProducers()) {
            prod->SetAttribute(OpAttributeKey::shapePadded, true);
            ALOG_DEBUG_F("op %d %s setAttribute shapepadded\n", prod->opmagic, prod->GetOpcodeStr().c_str());
        }
    }
}

bool PadLocalBuffer::IsExpandLastDim(const Operation &op) {
    int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "EXPANDDIM");
    if (axis == static_cast<int>(op.GetOOperands()[0]->shape.size() - 1)) {
        return true;
    }
    return false;
}

void PadLocalBuffer::TraverseCopyInConsumers(Function &function, Operation *consumer, std::unordered_set<LogicalTensorPtr> &visitedTensors) {
    bool allBrodOrElem = true;
    for (auto &nextConsumer : function.FindConsumers(*consumer)) {
        auto nextCalcType = OpcodeManager::Inst().GetOpCalcType(nextConsumer->GetOpcode());
        if (nextCalcType != OpCalcType::ELMWISE && nextCalcType != OpCalcType::BROADCAST) {
            allBrodOrElem = false;
            break;
        }
    }
    if (allBrodOrElem) {
        ALOG_DEBUG_F("op %d %s consumers are all broadcast or elmwise op, output's last dim should not be padded", consumer->opmagic, consumer->GetOpcodeStr().c_str());
        consumer->SetAttr(OpAttributeKey::outputCombineAxis, AXIS_COMBINED);
        TraverseAndSetAttr(consumer->GetOOperands()[0], function, visitedTensors);
    }
}

void PadLocalBuffer::TraverseBroadcast(Function &function, Operation *consumer, LogicalTensorPtr output, std::unordered_set<LogicalTensorPtr> &visitedTensors) {
    std::vector<bool> broadcastInputCombined(consumer->GetIOperands().size(), false);
    consumer->GetAttr(OpAttributeKey::inputCombineAxis, broadcastInputCombined);
    for (size_t index = 0; index < consumer->GetIOperands().size(); ++index) {
        if (consumer->GetIOperands()[index] == output) {
            broadcastInputCombined[index] = true;
        }
    }
    if (broadcastInputCombined.empty()) {
        ALOG_ERROR_F("cannot find tensor %d in input of op %d %s", output->magic, consumer->opmagic,
            consumer->GetOpcodeStr().c_str());
    } else {
        ALOG_DEBUG_F("op %d %s input's last dim should not be padded", consumer->opmagic, consumer->GetOpcodeStr().c_str());
        consumer->SetAttr(OpAttributeKey::inputCombineAxis, std::move(broadcastInputCombined));
        if (broadcastInputCombined == BROADCAST_AXIS_COMBINED) {
            ALOG_DEBUG_F("op %d %s output's last dim should not be padded", consumer->opmagic, consumer->GetOpcodeStr().c_str());
            consumer->SetAttr(OpAttributeKey::outputCombineAxis, AXIS_COMBINED);
            TraverseAndSetAttr(consumer->GetOOperands()[0], function, visitedTensors);
        }
    }
}

void PadLocalBuffer::TraverseAndSetAttr(LogicalTensorPtr &output, Function &function, std::unordered_set<LogicalTensorPtr> &visitedTensors) {
    if (visitedTensors.count(output) != 0) {
        return;
    }
    visitedTensors.emplace(output);
    for (auto &consumer : output->GetConsumers()) {
        auto consCalcType = OpcodeManager::Inst().GetOpCalcType(consumer->GetOpcode());
        auto opcode = consumer->GetOpcode();
        // expandop为不padding的终止节点
        if ((opcode == Opcode::OP_EXPAND) && IsExpandLastDim(*consumer)) {
            ALOG_DEBUG_F("expand op %d %s input's last dim should not be padded", consumer->opmagic, consumer->GetOpcodeStr().c_str());
            consumer->SetAttr(OpAttributeKey::inputCombineAxis, AXIS_COMBINED);
        } else if ((consCalcType == OpCalcType::ELMWISE) || (opcode == Opcode::OP_ASSEMBLE)) {
            ALOG_DEBUG_F("op %d %s is elmwise or assemble op, input's and output's last dim should not be padded", consumer->opmagic, consumer->GetOpcodeStr().c_str());
            consumer->SetAttr(OpAttributeKey::inputCombineAxis, AXIS_COMBINED);
            consumer->SetAttr(OpAttributeKey::outputCombineAxis, AXIS_COMBINED);
            TraverseAndSetAttr(consumer->GetOOperands()[0], function, visitedTensors);
        } else if (opcode == Opcode::OP_COPY_OUT) {
            ALOG_DEBUG_F("op %d %s is copy out, input's last dim should not be padded", consumer->opmagic, consumer->GetOpcodeStr().c_str());
            consumer->SetAttr(OpAttributeKey::inputCombineAxis, AXIS_COMBINED);
            TraverseAndSetAttr(consumer->GetOOperands()[0], function, visitedTensors);
        } else if (opcode == Opcode::OP_COPY_IN) {
            // CopyIn只有在CopyIn的后续算子仍然是VectorOp的情况下才做合轴
            TraverseCopyInConsumers(function, consumer, visitedTensors);
        }  else if (consCalcType == OpCalcType::BROADCAST) {
            TraverseBroadcast(function, consumer, output, visitedTensors);
        } else if (consCalcType ==
                    OpCalcType::MOVE_OUT) {
            // 剩下来的move out，直接中断，仅在输入的地方支持尾轴非对齐
            ALOG_DEBUG_F("op %d %s is move out, input's last dim should not be padded", consumer->opmagic, consumer->GetOpcodeStr().c_str());
            consumer->SetAttr(OpAttributeKey::inputCombineAxis, AXIS_COMBINED);
        }
    }
    return;
}

bool PadLocalBuffer::IsReduceLastDim(const Operation &op) {
    // 目前仅支持四类reduce，另外两个无reduceaxis属性，待确认
    if ((op.GetOpcode() == Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE) ||
        (op.GetOpcode() == Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE)) {
        int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
        if (axis == static_cast<int>(op.GetOOperands()[0]->shape.size() - 1)) {
            ALOG_DEBUG_F("op %d %s is reduce last dim\n", op.opmagic, op.GetOpcodeStr().c_str());
            return true;
        }
    }
    return false;
}

/*
z0必须32B对齐,尾轴reduce
      reduce
      [z0, 1]<-------->[z0, 1]  [z0, z1]
     /        \            \      /
  elementwise  copy_out    add_brc(break)
  [z0, 1]       [z0, 1]    [z0, pad(z1)]
    |              |
  expand(break)  copy_in
  [z0, pad(z1)]  [z0, 1]
                   |
                elementwise
                 [z0, 1]
*/
void PadLocalBuffer::ProcessReduce(Function &function, Operation &op) {
    // 轴的数量必须大于等于2， 并且倒数第二根轴为32B对齐， 否则无法命中优化pattern
    if ((op.GetOOperands()[0]->shape.size() >= AXIS_COMBINE_MIN_SHAPE_SIZE)) {
        auto out_bytes = BytesOf(op.oOperand[0]->Datatype());
        int paddingDim = 1;
        auto paddingIter = BLOCK_PADDING_DIM.find(out_bytes);
        if (paddingIter != BLOCK_PADDING_DIM.end()) {
            paddingDim = paddingIter->second;
        }
        if (paddingDim > 0 && op.oOperand[0]->shape[op.GetOOperands()[0]->shape.size() - AXIS_COMBINE_MIN_SHAPE_SIZE] % paddingDim != 0) {
            return;
        }
        ALOG_DEBUG_F("op %d %s is reduce, next to last dim is aligned\n", op.opmagic, op.GetOpcodeStr().c_str());
        std::vector<bool> reduceAxesCombined(op.GetOOperands().size(), false);
        reduceAxesCombined[0] = true;
        op.SetAttr(OpAttributeKey::outputCombineAxis, reduceAxesCombined);
        std::unordered_set<LogicalTensorPtr> visitedTensors;
        // dfs遍历打上input/output不做padding的相关属性
        TraverseAndSetAttr(op.GetOOperands()[0], function, visitedTensors);
    }
}

void PadLocalBuffer::ProcessBroadcast(Operation &op, size_t blockPadding) {
    int maxLastAxis = 0;
    bool existLessBlock = false;
    for (auto &in : op.iOperand) {
        if (in->shape.back() <= static_cast<int>(blockPadding)) {
            existLessBlock = true;
        }
        maxLastAxis = std::max(maxLastAxis, in->shape.back());
    }
    if (!existLessBlock) {
        op.SetAttribute(OpAttributeKey::broadcastLastAxis, maxLastAxis);
    }
}

bool PadLocalBuffer::IsMatmul(const LogicalTensorPtr &tensor) const {
    if ((tensor->GetMemoryTypeOriginal() == MemoryType::MEM_L1) ||
        (tensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0A) ||
        (tensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0B) ||
        (tensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0C)) {
        return true;
    }
    return false;
}

bool PadLocalBuffer::IsVector(const LogicalTensorPtr &tensor) {
    if (tensor->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
        return true;
    }
    return false;
}


void PadLocalBuffer::DoPadding(Function &function) {
    std::unordered_set<std::shared_ptr<LogicalTensor>> visited;
    std::unordered_set<std::shared_ptr<RawTensor>> visitedRaw;
    for (auto &op : function.Operations()) {
        std::vector<bool> inputAxis;
        op.GetAttr(OpAttributeKey::inputCombineAxis, inputAxis);
        for (size_t i = 0; i < op.iOperand.size(); i++) {
            auto &in = op.iOperand[i];
            if (visited.count(in) != 0) {
                continue;
            }
            visited.emplace(in);
            if (IsMatmul(in)) {
                PadMatmul(op, in);
            } else if (IsVector(in)) {
                if (in->tensor->GetRawDataSize() == 0) {
                    continue;
                }
                bool noPadding = false;
                if ((inputAxis.size() > i) && inputAxis[i]) {
                    noPadding = true;
                }
                PadVector(op, in, visitedRaw, noPadding);
            }
        }
    }
}

// 对ub上transpose的特殊处理,其他类型的transpose不做处理
Status PadLocalBuffer::ProcessTranspose(Function &function) {
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_TRANSPOSE_VNCHWCONV || op.GetIOperands()[0]->shape.size() < TRANSPOSE_MIN_SHAPE_SIZE) {
            continue;
        }
        std::vector<int32_t> transposeAxis = npu::tile_fwk::AnyCast<std::vector<int32_t>>(op.GetAttribute(OP_ATTR_PREFIX + "shape"));
        if (transposeAxis.size() != TRANSPOSE_MIN_SHAPE_SIZE) {
            ALOG_DEBUG_F("transpose op %d %s's shape size %d is not two, skip", op.opmagic, op.GetOpcodeStr().c_str(), transposeAxis.size());
            continue;
        }
        if (op.iOperand.size() <= 0 || op.oOperand.size() <= 0) {
            ALOG_ERROR_F("transpose op %d %s's input or output is empty", op.opmagic, op.GetOpcodeStr().c_str());
            return FAILED;
        }
        auto &inTensor = op.iOperand[0];
        auto &outTensor = op.oOperand[0];
        if (transposeAxis[0] == transposeAxis[1]) {
            ALOG_ERROR_F("transpose op has the same transpose dims, not supported");
            return FAILED;
        }
        if ((transposeAxis[0] != static_cast<int32_t>(inTensor->shape.size() - 1)) && (transposeAxis[1] != static_cast<int32_t>(inTensor->shape.size() - 1))) {
            ALOG_DEBUG_F("transpose op %d %s's transpose axis %d %d, not last dim transpose, skip", op.opmagic, op.GetOpcodeStr().c_str(), transposeAxis[0], transposeAxis[1]);
            continue;
        }
        int32_t nonLastDimIdx = -1;
        int32_t lastDimIdx = inTensor->shape.size() - 1;
        if (transposeAxis[0] != static_cast<int32_t>(inTensor->shape.size() - 1)) {
            nonLastDimIdx = transposeAxis[0];
        } else {
            nonLastDimIdx = transposeAxis[1];
        }
        auto &inLastDim = inTensor->shape[lastDimIdx];
        auto &outFirstDim = outTensor->shape[nonLastDimIdx];
        if (inLastDim != outFirstDim) {
            ALOG_DEBUG_F("tune transpose output dim %d to %d", inLastDim, outFirstDim);
            outTensor->shape[nonLastDimIdx] = inLastDim;
            outTensor->tensor->rawshape[nonLastDimIdx] = inLastDim;
        }
    }
    return SUCCESS;
}

Status PadLocalBuffer::RunOnFunction(Function &function) {
    for (auto &op : function.Operations()) {
        auto calcType = OpcodeManager::Inst().GetOpCalcType(op.GetOpcode());
        // 尾轴Reduce且倒数第二根轴32B对齐的op起始的链路上的op不做padding，以节省UB空间
        if (IsReduceLastDim(op)) {
            ProcessReduce(function, op);
        }
        // Broadcast op设置最后一根轴的padding值
        if (calcType == OpCalcType::BROADCAST) {
            auto bytes = BytesOf(op.iOperand[0]->Datatype());
            auto paddingIter = BLOCK_PADDING_DIM.find(bytes);
            if (paddingIter == BLOCK_PADDING_DIM.end()) {
                ALOG_DEBUG_F("broadcast op %d %s's datatype is not supported", op.opmagic, op.GetOpcodeStr().c_str());
                continue;
            }
            ProcessBroadcast(op, paddingIter->second);
        }
    }
    DoPadding(function);
    if (processTranspose_) {
        if (ProcessTranspose(function) != SUCCESS) {
            ALOG_ERROR_F("ProcessTranspose failed");
            return FAILED;
        }
    }
    return SUCCESS;
}
} // namespace
