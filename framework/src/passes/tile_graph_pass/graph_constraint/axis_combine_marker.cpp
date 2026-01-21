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
 * \file axis_combine.h
 * \brief
 */

#include "axis_combine_marker.h"
namespace npu {
namespace tile_fwk {
void AxisCombineMarker::Run(Function &function) {
    Init(function);
    ForwardVisit();
    BackwardVisit();
}

bool AxisCombineMarker::IsTensorEnableAxisCombine(LogicalTensorPtr tensor) {
    if (tensorStatus_.find(tensor) != tensorStatus_.end() && tensorStatus_[tensor] == AxisReorderStatus::ENABLE) {
        return true;
    }
    return false;
}

void AxisCombineMarker::Init(Function &function) {
    size_t i = 0U;
    std::map<int, size_t> opMagic2Idx;
    opList_ = function.Operations().DuplicatedOpList();
    for (const auto op : opList_) {
        opMagic2Idx[op->GetOpMagic()] = i;
        i++;
    }
    opInGraph_.resize(opList_.size());
    opOutGraph_.resize(opList_.size());
    for (size_t opIdx = 0; opIdx < opList_.size(); opIdx++) {
        const auto& op = opList_[opIdx];
        for (const auto producer : op->ProducerOpsOrdered()) {
            opInGraph_[opMagic2Idx[op->GetOpMagic()]].push_back(opMagic2Idx[producer->GetOpMagic()]);
        }
        for (const auto consumer : op->ConsumerOpsOrdered()) {
            opOutGraph_[opMagic2Idx[op->GetOpMagic()]].push_back(opMagic2Idx[consumer->GetOpMagic()]);
        }
    }
}

void AxisCombineMarker::UpdateOpACEnableForward(uint16_t opIdx) {
    auto op = opList_[opIdx];
    if (op->GetOpcode() == Opcode::OP_COPY_IN) {
        auto inputTensor = op->GetIOperands()[0];
        auto outputTensor = op->GetOOperands()[0];
        if (outputTensor->GetShape().back() != 1) {
            tensorStatus_[outputTensor] = AxisReorderStatus::UNKNOWN;
            return;
        }
        if (outputTensor->GetShape().back() == inputTensor->GetShape().back()) {
            tensorStatus_[outputTensor] = AxisReorderStatus::ENABLE;
            return;
        }
        tensorStatus_[outputTensor] = AxisReorderStatus::DISABLE;
        return;
    }
    if (op->GetOpcode() == Opcode::OP_VIEW) {
        auto inputTensor = op->GetIOperands()[0];
        auto outputTensor = op->GetOOperands()[0];
        if (tensorStatus_.find(inputTensor) != tensorStatus_.end()) {
            if (tensorStatus_[inputTensor] == AxisReorderStatus::ENABLE) {
                tensorStatus_[outputTensor] = AxisReorderStatus::ENABLE;
                return;
            }
            tensorStatus_[outputTensor] = AxisReorderStatus::DISABLE;
            return;
        }
        if (inputTensor->GetShape().back() == 1 && outputTensor->GetShape().back() == 1) {
            tensorStatus_[inputTensor] = AxisReorderStatus::ENABLE;
            tensorStatus_[outputTensor] = AxisReorderStatus::ENABLE;
            return;
        }
        tensorStatus_[inputTensor] = AxisReorderStatus::DISABLE;
        tensorStatus_[outputTensor] = AxisReorderStatus::DISABLE;  // DDR场景，不涉及。
        return;
    }
    if (op->GetOpcode() == Opcode::OP_ASSEMBLE) {
        auto inputTensor = op->GetIOperands()[0];
        auto outputTensor = op->GetOOperands()[0];
        if (tensorStatus_.find(inputTensor) != tensorStatus_.end()) {
            if (tensorStatus_[inputTensor] == AxisReorderStatus::ENABLE) {
                if (inputTensor->GetShape().back() != outputTensor->GetShape().back()) {
                    tensorStatus_[outputTensor] = AxisReorderStatus::DISABLE;
                    tensorStatus_[inputTensor] = AxisReorderStatus::DISABLE;  // 如果尾轴有assemble，那么不能支持合轴
                } else {
                    tensorStatus_[outputTensor] = AxisReorderStatus::ENABLE;
                }
                return;
            }
            tensorStatus_[outputTensor] = AxisReorderStatus::DISABLE;
            return;
        }
        // 正向推导不应该存在assemble输入没被访问过的场景
        return;
    }
    if (op->GetOpcode() == Opcode::OP_EXPAND) {
        auto inputTensor = op->GetIOperands()[0];
        auto outputTensor = op->GetOOperands()[0];
        if (tensorStatus_[inputTensor] == AxisReorderStatus::ENABLE) {
            auto dimSize = static_cast<int>(inputTensor->GetShape().size());
            int axis = op->GetIntAttribute(OP_ATTR_PREFIX + "EXPANDDIM");
            // 在尾轴为1的条件下，要求尾轴没有发生broadcast。[n, 1, 1]->expand->[n, 8, 1]??
            if (axis < dimSize - 1) {
                tensorStatus_[outputTensor] = AxisReorderStatus::ENABLE;
                return;
            } else {
                // 如果是尾轴broadcast，不支持交换轴
                tensorStatus_[inputTensor] = AxisReorderStatus::DISABLE;
            }
            return;
        }
        // 如果expand输出尾轴为1，并且输入就不支持合轴，那么输出也不支持合轴
        if (outputTensor->GetShape().back() == 1) {
            tensorStatus_[outputTensor] = AxisReorderStatus::DISABLE;
            return;
        }
        // 如果expand的输出尾轴不为1，那么不涉及到合轴优化。
        tensorStatus_[outputTensor] = AxisReorderStatus::UNKNOWN;
    }
    if (OpcodeManager::Inst().GetOpCalcType(op->GetOpcode()) == OpCalcType::REDUCE) {
        // 最后两根轴不发生reduce，并且尾轴为1。那么支持交换轴，如果倒数第二根轴发生reduce，不支持。尾轴reduce，需不需要交换轴要看后继节点
        auto inputTensor = op->GetIOperands()[0];
        auto outputTensor = op->GetOOperands()[0];
        auto dimSize = static_cast<int>(inputTensor->GetShape().size());
        int axis = op->GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
        if (dimSize > 1 && axis < dimSize - 2) {
            tensorStatus_[outputTensor] = tensorStatus_[inputTensor];
            return;
        }
        if (axis == dimSize - 2) {
            // reduce倒数第二轴，当前不支持合轴优化
            tensorStatus_[outputTensor] = AxisReorderStatus::DISABLE;
            return;
        }
        // Reduce尾轴，默认可以
        tensorStatus_[outputTensor] = AxisReorderStatus::ENABLE;
        return;
    }
    if (OpcodeManager::Inst().GetOpCalcType(op->GetOpcode()) == OpCalcType::ELMWISE ||
        OpcodeManager::Inst().GetOpCalcType(op->GetOpcode()) == OpCalcType::BROADCAST) {
        auto outputTensor = op->GetOOperands()[0];
        for (auto inputTensor : op->GetIOperands()) {
            if (tensorStatus_[inputTensor] == AxisReorderStatus::DISABLE) {
                tensorStatus_[outputTensor] = AxisReorderStatus::DISABLE;
                return;
            }
        }
        if (outputTensor->GetShape().back() == 1) {
            tensorStatus_[outputTensor] = AxisReorderStatus::ENABLE;
        } else {
            tensorStatus_[outputTensor] = AxisReorderStatus::UNKNOWN;
        }
        return;
    }
    auto outputTensor = op->GetOOperands()[0];
    tensorStatus_[outputTensor] = AxisReorderStatus::UNKNOWN;
}

void AxisCombineMarker::UpdateOpACEnableBackward(uint16_t opIdx) {
    auto op = opList_[opIdx];
    if (OpcodeManager::Inst().GetOpCalcType(op->GetOpcode()) == OpCalcType::ELMWISE ||
        OpcodeManager::Inst().GetOpCalcType(op->GetOpcode()) == OpCalcType::BROADCAST ||
        op->GetOpcode() == Opcode::OP_VIEW || op->GetOpcode() == Opcode::OP_ASSEMBLE) {
        auto outputTensor = op->GetOOperands()[0];
        for (auto inputTensor : op->GetIOperands()) {
            if (tensorStatus_[outputTensor] == AxisReorderStatus::DISABLE) {
                tensorStatus_[inputTensor] = tensorStatus_[outputTensor];
            }
        }
    }
}

void AxisCombineMarker::ForwardVisit()
{
    std::queue<size_t> procOpQueue;
    std::vector<size_t> inDegree(opList_.size(), 0);
    for (size_t j = 0; j < opInGraph_.size(); ++j) {
        if (opInGraph_[j].empty()) {
            procOpQueue.push(j);
            UpdateOpACEnableForward(j);
        }
        inDegree[j] = opInGraph_[j].size();
    }
    while (!procOpQueue.empty()) {
        auto opIdx = procOpQueue.front();
        procOpQueue.pop();
        for (auto outIdx : opOutGraph_[opIdx]) {
            inDegree[outIdx]--;
            if (inDegree[outIdx] == 0) {
                procOpQueue.push(outIdx);
                UpdateOpACEnableForward(outIdx);
            }
        }
    }
}

void AxisCombineMarker::BackwardVisit()
{
    std::queue<size_t> procOpQueue;
    std::vector<size_t> outDegree(opList_.size(), 0);
    for (size_t j = 0; j < opOutGraph_.size(); ++j) {
        if (opOutGraph_[j].empty()) {
            procOpQueue.push(j);
            UpdateOpACEnableBackward(j);
        }
        outDegree[j] = opOutGraph_[j].size();
    }
    while (!procOpQueue.empty()) {
        auto opIdx = procOpQueue.front();
        procOpQueue.pop();
        for (auto outIdx : opInGraph_[opIdx]) {
            outDegree[outIdx]--;
            if (outDegree[outIdx] == 0) {
                procOpQueue.push(outIdx);
                UpdateOpACEnableBackward(outIdx);
            }
        }
    }
}
}
}