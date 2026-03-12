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
 * \file remove_unaligned_reshape_op.cpp
 * \brief
 */

#include "remove_unaligned_reshape_op.h"
#include "passes/pass_utils/graph_utils.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "RemoveUnalignedReshape"

namespace npu::tile_fwk {
/*
before:
    add->reshape(padded)->mul

after:
    add->copyout->reshape->copyin->mul
*/
Status RemoveUnalignedReshape::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Function, "===> Start RemoveUnalignedReshape.");
    ReplaceDynUnalignedReshapeOps(function);
    CollectReshapeOps(function);
    for (auto &a : copyOuts) {
        GraphUtils::CopyDynStatus(a.output, a.input);
        auto &newCopyOut = function.AddRawOperation(Opcode::OP_COPY_OUT, {a.input}, {a.output});
        newCopyOut.SetOpAttribute(std::make_shared<CopyOpAttribute>(a.from, OpImmediate::Specified(a.toOffset),
            OpImmediate::Specified(newCopyOut.iOperand.front()->oriShape),
            OpImmediate::Specified(newCopyOut.oOperand.front()->tensor->GetDynRawShape())));
        auto producerOp = *(a.input->GetProducers().begin());
        newCopyOut.UpdateSubgraphID(producerOp->GetSubgraphID());
        APASS_LOG_INFO_F(Elements::Operation,
            "ADD OP_COPY_OUT, magic %d ,IOperand tensor magic %d OOperand tensor magic %d.", newCopyOut.opmagic,
            a.input->magic, a.output->magic);
    }
    for (auto &b : copyIns) {
        GraphUtils::CopyDynStatus(b.input, b.output);
        auto &newCopyIn = function.AddRawOperation(Opcode::OP_COPY_IN, {b.input}, {b.output});
        newCopyIn.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified(b.fromOffset), b.to,
            OpImmediate::Specified(newCopyIn.oOperand.front()->oriShape),
            OpImmediate::Specified(newCopyIn.iOperand.front()->tensor->GetDynRawShape()),
            OpImmediate::Specified(newCopyIn.iOperand.front()->GetDynValidShape())));
        auto consumerOp = *(b.output->GetConsumers().begin());
        newCopyIn.UpdateSubgraphID(consumerOp->GetSubgraphID());
        APASS_LOG_INFO_F(Elements::Operation,
            "ADD OP_VIEW, magic %d ,IOperand tensor magic %d OOperand tensor magic %d.", newCopyIn.opmagic,
            b.input->magic, b.output->magic);
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End RemoveUnalignedReshape.");
    return SUCCESS;
}

LogicalTensorPtr RemoveUnalignedReshape::InsertIOTensor(Function &function, Operation &op,
    std::unordered_map<OverlaprawMagic, std::shared_ptr<RawTensor>> &rawIO, LogicalTensorPtr &ioTensor) {
    if (rawIO.count(ioTensor->tensor->rawmagic) == 0) {
        auto reshapeRawTensor =
            std::make_shared<RawTensor>(ioTensor->Datatype(), ioTensor->tensor->oriRawshape, ioTensor->Format());
        reshapeRawTensor->oriRawshape = reshapeRawTensor->rawshape;
        rawIO.insert({ioTensor->tensor->rawmagic, reshapeRawTensor});
    }
    auto newReshapeIO = std::make_shared<LogicalTensor>(
        function, rawIO[ioTensor->tensor->rawmagic], ioTensor->offset, ioTensor->oriShape);
    newReshapeIO->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    newReshapeIO->subGraphID = op.GetSubgraphID();
    newReshapeIO->isSubGraphBoundary = true;
    function.GetTensorMap().Insert(newReshapeIO);
    return newReshapeIO;
}

bool RemoveUnalignedReshape::CheckUnaligned(Operation &op) {
    int lastIdx;
    for (const auto &input : op.GetIOperands()) {
        if (input != nullptr && input->tensor != nullptr) {
            lastIdx = input->shape.size() - 1;
            if (input->shape.size() == input->tensor->oriRawshape.size() &&
                input->shape.size() == input->tensor->rawshape.size() &&
                input->tensor->oriRawshape[lastIdx] != input->tensor->rawshape[lastIdx]) {
                return true;
            }
        }
    }
    for (const auto &output : op.GetOOperands()) {
        if (output != nullptr && output->tensor != nullptr) {
            lastIdx = output->shape.size() - 1;
            if (output->shape.size() == output->tensor->oriRawshape.size() &&
                output->shape.size() == output->tensor->rawshape.size() &&
                output->tensor->oriRawshape[lastIdx] != output->tensor->rawshape[lastIdx]) {
                return true;
            }
        }
    }
    return false;
}

std::vector<int64_t> FindChangedDims(
    const std::vector<int64_t> &inputShapes, const std::vector<int64_t> &outputShapes) {
    int inputDimSize = inputShapes.size();
    int outputDimSize = outputShapes.size();

    int left = -1;
    int right = -1;
    std::vector<int64_t> changedInputAxes = {};

    for (int i = 0; i < std::min(inputDimSize, outputDimSize); ++i) {
        if (inputShapes[i] != outputShapes[i] && left == -1) {
            left = i; // left第一次shape不等的位置
        }

        if (inputShapes[inputDimSize - 1 - i] != outputShapes[outputDimSize - 1 - i] && right == -1) {
            right = inputDimSize - 1 - i; // right第一次shape不等的位置
        }
    }

    if (left <= right && left != -1 && right != -1) {
        for (int i = left; i <= right; ++i) {
            changedInputAxes.push_back(i);
        }
    }

    return changedInputAxes;
}

void RemoveUnalignedReshape::ReplaceDynUnalignedReshapeOps(Function &function) {
    APASS_LOG_INFO_F(Elements::Function, "===> Start ReplaceDynUnalignedReshapeOps.");
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_RESHAPE) {
            continue;
        }

        auto input = op.GetIOperands().front();
        auto output = op.GetOOperands().front();

        if ((input->GetMemoryTypeOriginal() == MemoryType::MEM_UB) &&
            (output->GetMemoryTypeOriginal() == MemoryType::MEM_UB)) {
            ReplaceReshapeWithCopyOps(function, op);
        } else if ((input->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) &&
                   (output->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR)) {
            if (ReplaceReshapeWithCopyOpsForDDR(op) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Function, "DDR reshape replace failed for op %d.", op.GetOpMagic());
            }
        }
    }

    function.EraseOperations(true, false);

    APASS_LOG_INFO_F(Elements::Function, "===> End ReplaceDynUnalignedReshapeOps.");
}

void RemoveUnalignedReshape::ReplaceReshapeWithCopyOps(Function &function, Operation &op) {
    auto input = op.GetIOperands().front();
    auto output = op.GetOOperands().front();

    auto inputShapes = input->shape;
    auto outputShapes = output->shape;
    auto changedDims = FindChangedDims(outputShapes, inputShapes);

    auto inDynValidShape = input->GetDynValidShape();
    auto outDynValidShape = output->GetDynValidShape();

    for (const auto &dim : changedDims) {
        if (dim + 1 > (int)outDynValidShape.size()) {
            APASS_LOG_WARN_F(Elements::Operation, "The dynValidShape of output[%d] of op[%d] has no [%d] index.",
                output->GetMagic(), op.GetOpMagic(), dim);
            break;
        } else if (!outDynValidShape[dim].IsImmediate()) {
            op.SetAsDeleted();
            auto tmpWorkSpaceIn =
                std::make_shared<LogicalTensor>(function, input->Datatype(), input->oriShape, input->Format());
            auto tmpWorkSpaceOut =
                std::make_shared<LogicalTensor>(function, input->Datatype(), output->oriShape, output->Format());

            tmpWorkSpaceIn->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
            tmpWorkSpaceIn->UpdateDynValidShape(inDynValidShape);
            tmpWorkSpaceOut->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
            tmpWorkSpaceOut->UpdateDynValidShape(outDynValidShape);

            auto &reshapeCopyOutOp = function.AddOperation(Opcode::OP_RESHAPE_COPY_OUT, {input}, {tmpWorkSpaceIn});
            auto &reshapeOp = function.AddOperation(Opcode::OP_RESHAPE, {tmpWorkSpaceIn}, {tmpWorkSpaceOut});
            auto &reshapeCopyInOp = function.AddOperation(Opcode::OP_RESHAPE_COPY_IN, {tmpWorkSpaceOut}, {output});

            reshapeCopyOutOp.UpdateSubgraphID(op.GetSubgraphID());
            reshapeCopyInOp.UpdateSubgraphID(op.GetSubgraphID());
            reshapeOp.UpdateSubgraphID(op.GetSubgraphID());

            reshapeCopyOutOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(MemoryType::MEM_UB,
                OpImmediate::Specified(std::vector<SymbolicScalar>(input->shape.size(), 0)),
                OpImmediate::Specified(input->shape), OpImmediate::Specified(input->tensor->GetDynRawShape()),
                OpImmediate::Specified(input->GetDynValidShape())));

            reshapeCopyInOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                OpImmediate::Specified(std::vector<SymbolicScalar>(output->shape.size(), 0)),
                MemoryType::MEM_DEVICE_DDR, OpImmediate::Specified(output->shape),
                OpImmediate::Specified(output->tensor->GetDynRawShape()),
                OpImmediate::Specified(output->GetDynValidShape())));

            APASS_LOG_INFO_F(Elements::Operation,
                "Reshape op %d is replaceed by reshapeCopyOutOp %d and reshapeCopyInOp %d.", op.opmagic,
                reshapeCopyOutOp.opmagic, reshapeCopyInOp.opmagic);
            break;
        }
    }
}

/**
 * @brief DDR场景下替换非对齐reshape操作
 *
 * 处理流程：
 * 1. 检查changedDims和dynValidShape条件（与UB场景相同）
 * 2. 追溯reshape输入前方的copy_out生产者
 * 3. 确认copy_out只有一个生产者，否则报错返回FAILED
 * 4. 查找reshape输出后方的所有copy_in消费者，如果中间有view/assemble则报错返回FAILED
 * 5. 确认copy_out只有一个消费者（reshape），否则报错返回FAILED
 * 6. 修改opcode：OP_COPY_OUT -> OP_RESHAPE_COPY_OUT, OP_COPY_IN -> OP_RESHAPE_COPY_IN
 *
 * @param op 待处理的reshape操作
 * @return Status 处理结果，成功返回SUCCESS，失败返回FAILED
 */
Status RemoveUnalignedReshape::ReplaceReshapeWithCopyOpsForDDR(Operation &op) {
    auto input = op.GetIOperands().front();
    auto output = op.GetOOperands().front();

    auto inDynValidShape = input->GetDynValidShape();
    auto outDynValidShape = output->GetDynValidShape();

    // 检查 inDynValidShape 或 outDynValidShape 是否有非立即数
    bool inputHasNonImmediate = false;
    for (const auto &dim : inDynValidShape) {
        if (!dim.IsImmediate()) {
            inputHasNonImmediate = true;
            break;
        }
    }
    bool outputHasNonImmediate = false;
    for (const auto &dim : outDynValidShape) {
        if (!dim.IsImmediate()) {
            outputHasNonImmediate = true;
            break;
        }
    }

    if (inputHasNonImmediate && outputHasNonImmediate) {
        // ----- 第6步：追溯前方的所有 copy_out -----
        // 递归查找 input tensor 的所有生产者中的 copy_out
        std::vector<Operation *> copyOutOps = FindAllProducerCopyOuts(input);

        // 确保恰好找到一个 copy_out
        if (copyOutOps.size() != 1) {
            APASS_LOG_ERROR_F(Elements::Operation, "Reshape op %d has %d copy_out producers, only support exactly 1.",
                op.GetOpMagic(), copyOutOps.size());
            return FAILED;
        }

        Operation *copyOutOp = copyOutOps.front();

        std::vector<Operation *> copyInOps;
        bool hasViewOrAssemble = false;
        FindConsumerCopyIns(output, copyInOps, hasViewOrAssemble);
        if (hasViewOrAssemble) {
            APASS_LOG_ERROR_F(Elements::Operation,
                "Reshape op %d has view or assemble between reshape and copy_in, not supported.", op.GetOpMagic());
            return FAILED;
        }

        if (copyInOps.empty()) {
            APASS_LOG_ERROR_F(Elements::Operation, "Cannot find copy_in consumers for reshape op %d.", op.GetOpMagic());
            return FAILED;
        }

        auto copyOutConsumers = copyOutOp->GetOOperands().front()->GetConsumers();
        if (copyOutConsumers.size() != 1 || *(copyOutConsumers.begin()) != &op) {
            APASS_LOG_ERROR_F(Elements::Operation, "Reshape op %d has branch consumers before reshape, not supported.",
                op.GetOpMagic());
            return FAILED;
        }

        copyOutOp->SetOpCode(Opcode::OP_RESHAPE_COPY_OUT);
        for (auto *copyInOp : copyInOps) {
            copyInOp->SetOpCode(Opcode::OP_RESHAPE_COPY_IN);
        }

        APASS_LOG_INFO_F(Elements::Operation,
            "DDR Reshape op %d is replaced: copy_out %d -> reshape_copy_out, copy_in count %d -> reshape_copy_in.",
            op.GetOpMagic(), copyOutOp->GetOpMagic(), copyInOps.size());
    }

return SUCCESS;
}

/**
 * @brief 递归查找tensor的所有copy_out生产者操作
 *
 * 从tensor的生产者列表中递归查找所有OP_COPY_OUT：
 * - 如果遇到OP_COPY_OUT，记录并继续查找（可能还有其他生产者）
 * - 如果遇到OP_VIEW、OP_ASSEMBLE或OP_ASSEMBLE_SSA，递归继续向前追溯
 * - 遇到其他op也继续递归追溯
 *
 * @param tensor 待查找的tensor
 * @return std::vector<Operation*> 找到的所有copy_out操作指针列表
 */
std::vector<Operation *> RemoveUnalignedReshape::FindAllProducerCopyOuts(LogicalTensorPtr tensor) {
    std::vector<Operation *> copyOutOps;

    auto producers = tensor->GetProducers();
    if (producers.empty()) {
        return copyOutOps;
    }

    for (auto *producerOp : producers) {
        auto opcode = producerOp->GetOpcode();
        if (opcode == Opcode::OP_COPY_OUT) {
            copyOutOps.push_back(producerOp);
            // 继续查找，可能还有其他生产者
            continue;
        }

        // 其他类型的op（包括view/assemble或其他op），继续向前追溯
        auto inputOperands = producerOp->GetIOperands();
        if (!inputOperands.empty()) {
            auto subCopyOuts = FindAllProducerCopyOuts(inputOperands.front());
            copyOutOps.insert(copyOutOps.end(), subCopyOuts.begin(), subCopyOuts.end());
        }
    }

    return copyOutOps;
}

/**
 * @brief 查找tensor的消费者中的copy_in操作
 *
 * 从tensor的消费者列表中查找OP_COPY_IN，如果遇到OP_VIEW、OP_ASSEMBLE或OP_ASSEMBLE_SSA，
 * 则标记hasViewOrAssemble为true，表示不支持此类场景。
 * 注意：这里不会递归继续查找，因为遇到view/assemble已经表示不支持
 *
 * @param tensor 待查找的tensor
 * @param copyInOps 输出参数，存储找到的copy_in操作列表
 * @param hasViewOrAssemble 输出参数，标记是否发现view或assemble操作
 */
void RemoveUnalignedReshape::FindConsumerCopyIns(
    LogicalTensorPtr tensor, std::vector<Operation *> &copyInOps, bool &hasViewOrAssemble) {
    auto consumers = tensor->GetConsumers();
    for (auto *consumerOp : consumers) {
        auto opcode = consumerOp->GetOpcode();
        if (opcode == Opcode::OP_COPY_IN) {
            copyInOps.push_back(consumerOp);
        } else if (opcode == Opcode::OP_VIEW || opcode == Opcode::OP_ASSEMBLE || opcode == Opcode::OP_ASSEMBLE_SSA) {
            hasViewOrAssemble = true;
            APASS_LOG_ERROR_F(Elements::Operation,
                "Found OP_VIEW or OP_ASSEMBLE between reshape and copy_in, op %d, not supported.",
                consumerOp->GetOpMagic());
        }
    }
}

void RemoveUnalignedReshape::CollectReshapeOps(Function &function) {
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_RESHAPE) {
            continue;
        }

        if (!CheckUnaligned(op)) {
            continue;
        }

        auto input = op.GetIOperands().front();
        auto output = op.GetOOperands().front();
        if ((input->GetMemoryTypeOriginal() != MemoryType::MEM_UB) ||
            (output->GetMemoryTypeOriginal() != MemoryType::MEM_UB)) {
            continue;
        }

        // 插入copyout
        LogicalTensorPtr newReshapeInput = InsertIOTensor(function, op, reshapeRawInputs, input);
        copyOuts.emplace_back(
            CopyOutOpMemUnalign{input->GetMemoryTypeOriginal(), input->offset, input, newReshapeInput});
        op.ReplaceInput(newReshapeInput, input);

        // 插入copyin
        LogicalTensorPtr newReshapeOutput = InsertIOTensor(function, op, reshapeRawOutputs, output);
        copyIns.emplace_back(
            CopyInOpMemUnalign{output->GetMemoryTypeOriginal(), output->offset, newReshapeOutput, output});
        op.ReplaceOutput(newReshapeOutput, output);
        output->tensor->actualRawmagic = -1;
        op.GetOOperands().front()->tensor->actualRawmagic = op.GetIOperands().front()->tensor->GetRawMagic();
    }
}

} // namespace npu::tile_fwk
