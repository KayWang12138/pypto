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
 * \file assign_memtype_refactored.cpp
 * \brief Memory type assignment pass - refactored version
 */

#include "assign_memory_type.h"

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/checker_utils.h"

#define MODULE_NAME "AssignMemoryType"

namespace npu::tile_fwk {
constexpr int64_t UB_ALIGN_BYTES = 32;

// ============================================================
// Phase 1: 确定性 Memtype 设置
// ============================================================

void AssignMemoryType::SetIncastOutcastMemtype(Function& function)
{
    for (auto& incast : function.inCasts_) {
        incast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
        for (const auto& consumerOp : incast->GetConsumers()) {
            inserter.UpdateTensorTobeMap(incast, *consumerOp, MemoryType::MEM_DEVICE_DDR);
        }
    }

    for (auto& outcast : function.outCasts_) {
        outcast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    }
}

void AssignMemoryType::RunOnOperation(Operation& operation)
{
    auto opcode = operation.GetOpcode();
    const auto& inputsMemType = OpcodeManager::Inst().GetInputsMemType(opcode);
    const auto& outputsMemType = OpcodeManager::Inst().GetOutputsMemType(opcode);
    if (opcode == Opcode::OP_VIEW) {
        ProcessViewWithSpecificMem(operation);
        return;
    }
    if (opcode == Opcode::OP_ASSEMBLE) {
        ProcessAssembleWithSpecificMem(operation);
        return;
    }
    SetInputTensorsMemtype(operation, inputsMemType);
    SetOutputTensorsMemtype(operation, outputsMemType);
}

void AssignMemoryType::SetInputTensorsMemtype(Operation& operation, const std::vector<MemoryType>& inputsMemType)
{
    for (size_t i = 0; i < operation.iOperand.size(); ++i) {
        auto& tensor = operation.iOperand[i];
        if (i < inputsMemType.size()) {
            inserter.UpdateTensorTobeMap(tensor, operation, inputsMemType[i]);
        }
    }
}

void AssignMemoryType::SetOutputTensorsMemtype(Operation& operation, const std::vector<MemoryType>& outputsMemType)
{
    for (size_t i = 0; i < operation.oOperand.size(); ++i) {
        auto& tensor = operation.oOperand[i];
        if (outputsMemType.size() > 0 && i < outputsMemType.size()) {
            tensor->SetMemoryTypeOriginal(outputsMemType[i]);
            for (const auto& consumerOp : tensor->GetConsumers()) {
                inserter.UpdateTensorTobeMap(tensor, *consumerOp, outputsMemType[i]);
            }
        } else {
            tensor->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN);
            for (const auto& consumerOp : tensor->GetConsumers()) {
                inserter.UpdateTensorTobeMap(tensor, *consumerOp, MemoryType::MEM_UNKNOWN);
            }
        }
    }
}

void AssignMemoryType::ProcessViewWithSpecificMem(Operation& operation)
{
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(operation.GetOpAttribute().get());
    MemoryType attrToType = viewOpAttribute->GetTo();

    if (attrToType == MemoryType::MEM_UNKNOWN) {
        return;
    }

    auto out = operation.GetOOperands().front();
    out->SetMemoryTypeOriginal(attrToType, true);
    for (auto& consumerOp : out->GetConsumers()) {
        inserter.UpdateTensorTobeMap(out, *consumerOp, attrToType);
    }
}

void AssignMemoryType::ProcessAssembleWithSpecificMem(Operation& operation)
{
    SetInputTensorsMemtype(operation, OpcodeManager::Inst().GetInputsMemType(operation.GetOpcode()));
    SetOutputTensorsMemtype(operation, OpcodeManager::Inst().GetOutputsMemType(operation.GetOpcode()));
}

void AssignMemoryType::AssignOpShmemWaitUntilMemtype(Operation& op)
{
    if (op.GetOpcode() == Opcode::OP_SHMEM_WAIT_UNTIL) {
        for (auto& output : op.GetOOperands()) {
            output->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
        }
    }
}

// ============================================================
// Phase 2: 不确定性推导 - 连接Op类型传递
// ============================================================

void AssignMemoryType::AssignMoveOp(Operation& operation)
{
    auto opcode = operation.GetOpcode();
    if (opcode == Opcode::OP_ASSEMBLE) {
        AssignMoveOpForAssemble(operation);
    } else if (opcode == Opcode::OP_VIEW) {
        AssignMoveOpForView(operation);
    }
}

// --------------------
// Assemble拆分
// --------------------

bool AssignMemoryType::CheckAssembleProducerConsistency(
    const LogicalTensorPtr& outputTensor, MemoryType& fromType)
{
    for (const auto& outputProducer : outputTensor->GetProducers()) {
        if (fromType != MemoryType::MEM_DEVICE_DDR &&
            outputProducer->iOperand.front()->GetMemoryTypeOriginal() != fromType) {
            fromType = MemoryType::MEM_DEVICE_DDR;
            return false;
        }
    }
    return true;
}

bool AssignMemoryType::CheckAssembleAlignment(const LogicalTensorPtr& outputTensor)
{
    for (const auto& outputProducer : outputTensor->GetProducers()) {
        auto opAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(outputProducer->GetOpAttribute());
        if (opAttr == nullptr) {
            continue;
        }

        int64_t lineOffset = CalcLineOffset(outputTensor->GetRawTensor()->rawshape, opAttr->GetToOffset());
        if (lineOffset == -1) {
            continue;
        }

        int64_t tensorBytes = static_cast<int64_t>(BytesOf(outputTensor->Datatype()));
        int64_t byteOffset = tensorBytes * lineOffset;

        if (byteOffset % UB_ALIGN_BYTES != 0) {
            return false;
        }
    }
    return true;
}

void AssignMemoryType::DeriveAssembleOutputOriginal(Operation& operation, LogicalTensorPtr& outputTensor)
{
    auto& inputTensor = operation.iOperand.front();
    MemoryType fromType = inserter.GetMemoryTypeFromTensorTobeMap(inputTensor, operation);

    CheckAssembleProducerConsistency(outputTensor, fromType);

    if (!CheckAssembleAlignment(outputTensor)) {
        fromType = MemoryType::MEM_DEVICE_DDR;
    }
    outputTensor->SetMemoryTypeOriginal(fromType, true);
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute());
    assembleOpAttribute->SetFromType(fromType);
}

void AssignMemoryType::AssignMoveOpForAssemble(Operation& operation)
{
    for (size_t i = 0; i < operation.oOperand.size(); ++i) {
        auto& outputTensor = operation.oOperand[i];
        DeriveAssembleOutputOriginal(operation, outputTensor);
    }
}

// --------------------
// View拆分
// --------------------

MemoryType AssignMemoryType::DeriveViewOutputOriginalFromTobeMap(
    LogicalTensorPtr& outputTensor, ViewOpAttribute* viewOpAttribute)
{
    auto tobeMap = inserter.GetMemoryTypeFromTensorTobeMap(outputTensor);
    std::set<MemoryType> uniqueTypes;
    for (const auto& [consumerOp, memType] : tobeMap) {
        if (memType != MemoryType::MEM_UNKNOWN) {
            uniqueTypes.insert(memType);
        }
    }

    if (uniqueTypes.size() == 1) {
        MemoryType derivedType = *uniqueTypes.begin();
        outputTensor->SetMemoryTypeOriginal(derivedType);
        viewOpAttribute->SetToType(derivedType);
        return derivedType;
    }

    return MemoryType::MEM_UNKNOWN;
}

void AssignMemoryType::DeriveViewOutputOriginalFromInput(
    LogicalTensorPtr& outputTensor, LogicalTensorPtr& inputTensor, ViewOpAttribute* viewOpAttribute)
{
    MemoryType inputOriginal = inputTensor->GetMemoryTypeOriginal();
    if (inputOriginal != MemoryType::MEM_UNKNOWN) {
        outputTensor->SetMemoryTypeOriginal(inputOriginal);
        viewOpAttribute->SetToType(inputOriginal);
    }
}

void AssignMemoryType::DeriveViewOutputOriginal(
    Operation& operation, ViewOpAttribute* viewOpAttribute, bool unaligned)
{
    auto outputTensor = operation.GetOOperands().front();
    auto& inputTensor = operation.iOperand.front();

    if (outputTensor->GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN || unaligned) {
        return;
    }

    MemoryType derivedType = DeriveViewOutputOriginalFromTobeMap(outputTensor, viewOpAttribute);
    if (derivedType == MemoryType::MEM_UNKNOWN) {
        DeriveViewOutputOriginalFromInput(outputTensor, inputTensor, viewOpAttribute);
    }
}

void AssignMemoryType::DeriveViewInputTobe(
    Operation& operation, ViewOpAttribute* viewOpAttribute, bool unaligned)
{
    auto outputTensor = operation.GetOOperands().front();
    auto& inputTensor = operation.iOperand.front();

    MemoryType outputOriginal = outputTensor->GetMemoryTypeOriginal();
    MemoryType inputOriginal = inputTensor->GetMemoryTypeOriginal();

    if (TryMemoryReuse(outputTensor, inputTensor, outputOriginal, inputOriginal, unaligned, viewOpAttribute)) {
        return;
    }

    if (TryL0C2L1Pathway(inputTensor, outputOriginal, inputOriginal, operation, viewOpAttribute)) {
        return;
    }

    if (unaligned) {
        inserter.UpdateTensorTobeMap(inputTensor, operation, MemoryType::MEM_DEVICE_DDR);
        return;
    }

    inserter.UpdateTensorTobeMap(inputTensor, operation, outputOriginal);
    viewOpAttribute->SetToType(outputOriginal);
}

bool AssignMemoryType::TryMemoryReuse(
    LogicalTensorPtr& outputTensor, LogicalTensorPtr& inputTensor,
    MemoryType outputOriginal, MemoryType inputOriginal, bool unaligned, ViewOpAttribute* viewOpAttribute)
{
    (void)inputTensor;
    if (outputOriginal == MemoryType::MEM_UNKNOWN &&
        inputOriginal != MemoryType::MEM_UNKNOWN &&
        inputOriginal != MemoryType::MEM_L0C &&
        !unaligned) {
        outputTensor->SetMemoryTypeOriginal(inputOriginal);
        viewOpAttribute->SetToType(inputOriginal);
        return true;
    }
    return false;
}

bool AssignMemoryType::TryL0C2L1Pathway(
    LogicalTensorPtr& inputTensor, MemoryType outputOriginal, MemoryType inputOriginal,
    Operation& operation, ViewOpAttribute* viewOpAttribute)
{
    if (inputOriginal == MemoryType::MEM_L0C &&
        outputOriginal == MemoryType::MEM_L1 &&
        inserter.FitL0C2L1(operation)) {
        inserter.UpdateTensorTobeMap(inputTensor, operation, MemoryType::MEM_L0C);
        viewOpAttribute->SetToType(outputOriginal);
        return true;
    }
    return false;
}

void AssignMemoryType::AssignMoveOpForView(Operation& operation)
{
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(operation.GetOpAttribute().get());
    MemoryType attrToType = viewOpAttribute->GetTo();

    if (attrToType != MemoryType::MEM_UNKNOWN) {
        if (!operation.iOperand.empty() && attrToType == MemoryType::MEM_L1 &&
            inserter.CrossCore(operation.iOperand.front()->GetMemoryTypeOriginal(), attrToType)) {
            inserter.UpdateTensorTobeMap(operation.iOperand.front(), operation, attrToType);
        }
        return;
    }

    auto outputTensor = operation.GetOOperands().front();
    auto viewOffset = viewOpAttribute->GetFromOffset();
    bool unaligned = ((BytesOf(outputTensor->Datatype()) * viewOffset.back()) % UB_ALIGN_BYTES != 0);

    DeriveViewOutputOriginal(operation, viewOpAttribute, unaligned);
    DeriveViewInputTobe(operation, viewOpAttribute, unaligned);
}

// ============================================================
// Phase 3: 不确定性推导 - UNKNOWN张量处理
// ============================================================

MemoryType AssignMemoryType::DeriveUnknownTensorOriginal(LogicalTensorPtr& tensor)
{
    auto localTobeMap = inserter.GetRequiredTobe(tensor);
    if (localTobeMap.size() == 1 &&
        localTobeMap.begin()->first != MemoryType::MEM_UNKNOWN) {
        return localTobeMap.begin()->first;
    }
    return MemoryType::MEM_DEVICE_DDR;
}

void AssignMemoryType::ProcessUnknownInputTensor(
    Operation& op, LogicalTensorPtr& tensor, std::unordered_set<LogicalTensorPtr>& visited)
{
    if (visited.count(tensor) > 0) {
        return;
    }
    visited.insert(tensor);

    if (tensor->GetMemoryTypeOriginal() == MemoryType::MEM_UNKNOWN) {
        MemoryType derivedType = DeriveUnknownTensorOriginal(tensor);
        tensor->SetMemoryTypeOriginal(derivedType);
        inserter.UpdateTensorTobeMap(tensor, op, derivedType);
    }

    inserter.UpdateTensorTobeMapUnknown(tensor, tensor->GetMemoryTypeOriginal());
}

void AssignMemoryType::ProcessUnknownOutputTensor(
    LogicalTensorPtr& tensor, std::unordered_set<LogicalTensorPtr>& visited)
{
    if (visited.count(tensor) > 0) {
        return;
    }
    visited.insert(tensor);

    if (tensor->GetMemoryTypeOriginal() == MemoryType::MEM_UNKNOWN) {
        MemoryType derivedType = DeriveUnknownTensorOriginal(tensor);
        tensor->SetMemoryTypeOriginal(derivedType);
        for (const auto& consumerOp : tensor->GetConsumers()) {
            inserter.UpdateTensorTobeMap(tensor, *consumerOp, derivedType);
        }
    }

    inserter.UpdateTensorTobeMapUnknown(tensor, tensor->GetMemoryTypeOriginal());
}

void AssignMemoryType::AssignMemUnknown(Function& function)
{
    std::unordered_set<LogicalTensorPtr> visited;

    for (auto& op : function.Operations()) {
        for (auto& tensor : op.iOperand) {
            ProcessUnknownInputTensor(op, tensor, visited);
        }
        for (auto& tensor : op.oOperand) {
            ProcessUnknownOutputTensor(tensor, visited);
        }
    }
}

// ============================================================
// Phase 4: 不确定性推导 - 特殊Op修正
// ============================================================

void AssignMemoryType::AssignSpecialOpMemtype(Operation& op, bool& infoBufferSize)
{
    AssignOpReshapeMemtype(op);
    AssignOpViewTypeMemtype(op);
    AssignOpNopMemtype(op);
    AssignOpShmemWaitUntilMemtype(op);
    AssignOpAssembleFinalMemtype(op, infoBufferSize);
}

void AssignMemoryType::AssignOpReshapeMemtype(Operation& op)
{
    if (op.GetOpcode() != Opcode::OP_RESHAPE) {
        return;
    }

    auto& input = op.iOperand.front();
    auto& output = op.oOperand.front();

    MemoryType inputMemType = inserter.GetMemoryTypeFromTensorTobeMap(input, op);
    MemoryType outputOriginal = output->GetMemoryTypeOriginal();

    if (inputMemType != outputOriginal) {
        if (input->GetMemoryTypeOriginal() == MemoryType::MEM_UB ||
            output->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
            inserter.UpdateTensorTobeMap(input, op, MemoryType::MEM_UB);
            output->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
        } else {
            inserter.UpdateTensorTobeMap(input, op, MemoryType::MEM_DEVICE_DDR);
            output->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
        }
    }
}

void AssignMemoryType::AssignOpViewTypeMemtype(Operation& op)
{
    if (op.GetOpcode() != Opcode::OP_VIEW_TYPE) {
        return;
    }

    auto& input = op.iOperand.front();
    auto& output = op.oOperand.front();

    auto inputMemType = inserter.GetMemoryTypeFromTensorTobeMap(input, op);
    auto outputOriginal = output->GetMemoryTypeOriginal();

    auto prod = *(input->GetProducers().begin());
    if (prod->GetOpcode() == Opcode::OP_VIEW) {
        input->SetMemoryTypeOriginal(outputOriginal, true);
        inserter.UpdateTensorTobeMap(input, op, outputOriginal);
        return;
    }

    if (inputMemType != outputOriginal) {
        inserter.UpdateTensorTobeMap(input, op, MemoryType::MEM_DEVICE_DDR);
        output->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
    }
}

void AssignMemoryType::AssignOpNopMemtype(Operation& op)
{
    if (op.GetOpcode() != Opcode::OP_NOP) {
        return;
    }

    auto& input = op.iOperand.front();
    auto& output = op.oOperand.front();

    if (input->GetMemoryTypeToBe() != output->GetMemoryTypeOriginal()) {
        input->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
        output->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    }

    if (output->GetMemoryTypeOriginal() != output->GetMemoryTypeToBe()) {
        output->SetMemoryTypeBoth(output->GetMemoryTypeOriginal(), true);
    }
}

void AssignMemoryType::AssignOpAssembleFinalMemtype(Operation& op, bool& infoBufferSize)
{
    if (op.GetOpcode() != Opcode::OP_ASSEMBLE) {
        return;
    }

    UpdateOverSizedLocalBuffer(op);
    infoBufferSize = true;
}

void AssignMemoryType::UpdateOverSizedLocalBuffer(Operation& operation)
{
    const int64_t UB_SIZE_THRESHOLD =
        static_cast<int64_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) * UB_THRESHOLD);
    const int64_t L1_SIZE_THRESHOLD =
        static_cast<int64_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L1) * L1_THRESHOLD);

    auto output = operation.GetOOperands().front();
    auto memType = output->GetMemoryTypeOriginal();

    if ((memType == MemoryType::MEM_UB && output->GetDataSize() > UB_SIZE_THRESHOLD) ||
        (memType == MemoryType::MEM_L1 && output->GetDataSize() > L1_SIZE_THRESHOLD)) {
        output->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    }
}

// ============================================================
// Phase 5: 不确定性推导 - Tile维度约束修正
// ============================================================

bool AssignMemoryType::CheckTileTobeConsistency(LogicalTensorPtr& output)
{
    auto tobeMap = inserter.GetMemoryTypeFromTensorTobeMap(output);
    for (const auto& [consumerOp, memType] : tobeMap) {
        if (memType != MemoryType::MEM_L1 && memType != MemoryType::MEM_UB) {
            return false;
        }
    }
    return true;
}

void AssignMemoryType::ProcessSmallTileToLargeTile(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }

        auto output = op.GetOOperands().front();
        auto input = op.GetIOperands().front();

        if (input->GetMemoryTypeOriginal() != MemoryType::MEM_L0C) {
            continue;
        }

        bool validTobe = CheckTileTobeConsistency(output);
        bool validDim = IsDimMultiple(output->GetShape(), input->GetShape());

        if (!validTobe || !validDim) {
            output->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
            auto tobeMap = inserter.GetMemoryTypeFromTensorTobeMap(output);
            for (const auto& [consumerOp, memType] : tobeMap) {
                if (memType == MemoryType::MEM_L0C) {
                    inserter.UpdateTensorTobeMap(output, *consumerOp, MemoryType::MEM_DEVICE_DDR);
                }
            }
        }
    }
}

void AssignMemoryType::ProcessLargeTileToSmallTile(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_VIEW) {
            continue;
        }

        auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute().get());
        MemoryType attrToType = viewOpAttribute->GetTo();

        auto input = op.GetIOperands().front();
        auto output = op.GetOOperands().front();

        if (attrToType == MemoryType::MEM_L1) {
            if (input->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR &&
                !IsDimMultiple(input->GetShape(), output->GetShape())) {
                inserter.UpdateTensorTobeMap(input, op, MemoryType::MEM_DEVICE_DDR);
            }
            continue;
        }

        if (input->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR &&
            !IsDimMultiple(input->GetShape(), output->GetShape())) {
            inserter.UpdateTensorTobeMap(input, op, MemoryType::MEM_DEVICE_DDR);
        }
    }
}

bool AssignMemoryType::IsDimMultiple(const Shape& shape1, const Shape& shape2)
{
    if (shape1.size() != shape2.size()) {
        return false;
    }

    for (size_t i = 0; i < shape1.size(); ++i) {
        if (shape1[i] <= 0 || shape2[i] <= 0 || shape1[i] % shape2[i] != 0) {
            return false;
        }
    }

    return true;
}

// ============================================================
// 辅助函数
// ============================================================

int64_t AssignMemoryType::CalcLineOffset(const Shape& shape, const Offset& offset)
{
    if (shape.size() != offset.size()) {
        return -1;
    }
    if (shape.size() == 0) {
        return 0;
    }

    int64_t lineOffset = 0;
    int64_t stride = 1;
    for (size_t i = shape.size(); i > 0; --i) {
        lineOffset += offset[i - 1] * stride;
        stride *= shape[i - 1];
    }

    return lineOffset;
}

std::string AssignMemoryType::PrintTensorMem(std::shared_ptr<LogicalTensor>& tensor) const
{
    std::ostringstream oss;
    oss << "tensor magic: " << tensor->magic;
    oss << " original: " << BriefMemoryTypeToString(tensor->GetMemoryTypeOriginal());
    oss << ", tobe: " << BriefMemoryTypeToString(tensor->GetMemoryTypeToBe());
    return oss.str();
}

// ============================================================
// 主入口函数
// ============================================================

Status AssignMemoryType::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start AssignMemoryType.");

    // Phase 1: 确定性 Memtype 设置
    SetIncastOutcastMemtype(function);
    for (auto& op : function.Operations()) {
        RunOnOperation(op);
    }
    // Phase 2: 不确定性推导 - 连接Op类型传递
    for (auto& op : function.Operations()) {
        AssignMoveOp(op);
    }

    // Phase 3: 不确定性推导 - UNKNOWN张量处理
    AssignMemUnknown(function);

    // Phase 4: 不确定性推导 - 特殊Op修正
    bool infoBufferSize = false;
    for (auto& op : function.Operations()) {
        AssignSpecialOpMemtype(op, infoBufferSize);
    }

    if (infoBufferSize) {
        const size_t UB_THRESHOLD_SIZE =
            static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) * UB_THRESHOLD);
        const size_t L1_THRESHOLD_SIZE =
            static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L1) * L1_THRESHOLD);
        APASS_LOG_INFO_F(Elements::Operation, "UB threshold %zu, L1 threshold %zu.",
                         UB_THRESHOLD_SIZE, L1_THRESHOLD_SIZE);
    }

    // Phase 5: 不确定性推导 - Tile维度约束修正
    ProcessSmallTileToLargeTile(function);
    ProcessLargeTileToSmallTile(function);

    // Phase 6: Convert Op 插入
    Status insertionStatus = inserter.DoInsertion(function);
    if (insertionStatus != SUCCESS) {
        return insertionStatus;
    }

    APASS_LOG_INFO_F(Elements::Function, "===> End AssignMemoryType.");
    return SUCCESS;
}

Status AssignMemoryType::PreCheck(Function& function)
{
    return checker.DoPreCheck(function);
}

Status AssignMemoryType::PostCheck(Function& function)
{
    return checker.DoPostCheck(function);
}

} // namespace npu::tile_fwk