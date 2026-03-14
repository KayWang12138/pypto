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
    if (ReplaceDynUnalignedReshapeOps(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "ReplaceDynUnalignedReshapeOps failed.");
    }
    CollectReshapeOps(function);
    for (auto &a : copyOuts) {
        GraphUtils::CopyDynStatus(a.output, a.input);
        auto &newCopyOut = function.AddRawOperation(Opcode::OP_COPY_OUT, {a.input}, {a.output});
        newCopyOut.SetOpAttribute(std::make_shared<CopyOpAttribute>(a.from, OpImmediate::Specified(a.toOffset),
            OpImmediate::Specified(newCopyOut.iOperand.front()->oriShape),
            OpImmediate::Specified(newCopyOut.oOperand.front()->tensor->GetDynRawShape())));
        auto producerOp = *(a.input->GetProducers().begin());
        newCopyOut.UpdateSubgraphID(producerOp->GetSubgraphID());
        APASS_LOG_INFO_F(Elements::Operation, "ADD OP_COPY_OUT, magic %d ,IOperand tensor magic %d OOperand tensor magic %d.",
            newCopyOut.opmagic, a.input->magic, a.output->magic);
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
        APASS_LOG_INFO_F(Elements::Operation, "ADD OP_VIEW, magic %d ,IOperand tensor magic %d OOperand tensor magic %d.",
            newCopyIn.opmagic, b.input->magic, b.output->magic);
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End RemoveUnalignedReshape.");
    return SUCCESS;
}

LogicalTensorPtr RemoveUnalignedReshape::InsertIOTensor(Function &function, Operation &op, std::unordered_map<OverlaprawMagic, std::shared_ptr<RawTensor>> &rawIO, LogicalTensorPtr &ioTensor) {
    if (rawIO.count(ioTensor->tensor->rawmagic) == 0) {
        auto reshapeRawTensor = std::make_shared<RawTensor>(ioTensor->Datatype(),
            ioTensor->tensor->oriRawshape, ioTensor->Format());
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

std::vector<int64_t> FindChangedDims(const std::vector<int64_t>& inputShapes, const std::vector<int64_t>& outputShapes) {
    int inputDimSize = inputShapes.size();
    int outputDimSize = outputShapes.size();

    int left = -1;
    int right = -1;
    std::vector<int64_t> changedInputAxes = {};

    for (int i = 0; i < std::min(inputDimSize, outputDimSize); ++i) {
        if (inputShapes[i] != outputShapes[i] && left == -1) {
            left = i;  // left第一次shape不等的位置
        }

        if (inputShapes[inputDimSize - 1 - i] != outputShapes[outputDimSize - 1 - i] && right == -1) {
            right = inputDimSize - 1 - i;  // right第一次shape不等的位置
        }
    }

    if (left <= right && left != -1 && right != -1) {
        for (int i = left; i <= right; ++i) {
            changedInputAxes.push_back(i);
        }
    }

    return changedInputAxes;
}
Status RemoveUnalignedReshape::ReplaceDynUnalignedReshapeOps(Function &function) {
    APASS_LOG_INFO_F(Elements::Function, "===> Start ReplaceDynUnalignedReshapeOps.");
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_RESHAPE || processedReshapeOps.count(op.GetOpMagic())) {
            continue;
        }
        auto input = op.GetIOperands().front();
        auto output = op.GetOOperands().front();
        if (input->GetMemoryTypeOriginal() == MemoryType::MEM_UB && output->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
            ReplaceDynUnalignedReshapeOpsForUB(function, op);
        } else if (input->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR && output->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            if (ReplaceDynUnalignedReshapeOpsForDDR(function, op) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Function, "DDR reshape replace failed for op %d.", op.GetOpMagic());
                return FAILED;
            }
        }
    }   
    APASS_LOG_INFO_F(Elements::Function, "===> End ReplaceDynUnalignedReshapeOps.");
    return SUCCESS;
}

void RemoveUnalignedReshape::ReplaceDynUnalignedReshapeOpsForUB(Function &function, Operation &op) {
    auto input = op.GetIOperands().front();
    auto output = op.GetOOperands().front();

    auto inputShapes = input->shape;
    auto outputShapes = output->shape;
    auto changedDims = FindChangedDims(outputShapes, inputShapes);

    auto inDynValidShape = input->GetDynValidShape();
    auto outDynValidShape = output->GetDynValidShape();

    for (const auto &dim : changedDims) {
        if ((size_t)dim > outDynValidShape.size() - 1) {
            APASS_LOG_WARN_F(Elements::Operation, "The dynValidShape of output[%d] of op[%d] has no [%ld] index.",
            output->GetMagic(), op.GetOpMagic(), static_cast<long>(dim));
            break;
        } else if (!outDynValidShape[dim].IsImmediate()) {
            op.SetAsDeleted();
            auto tmpWorkSpaceIn = std::make_shared<LogicalTensor>(function, input->Datatype(), input->oriShape, input->Format());
            auto tmpWorkSpaceOut = std::make_shared<LogicalTensor>(function, input->Datatype(), output->oriShape, output->Format());

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

            reshapeCopyOutOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                MemoryType::MEM_UB, OpImmediate::Specified(std::vector<SymbolicScalar>(input->shape.size(), 0)), OpImmediate::Specified(input->shape),
                OpImmediate::Specified(input->tensor->GetDynRawShape()), OpImmediate::Specified(input->GetDynValidShape())
            ));

            reshapeCopyInOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                OpImmediate::Specified(std::vector<SymbolicScalar>(output->shape.size(), 0)), MemoryType::MEM_DEVICE_DDR, OpImmediate::Specified(output->shape),
                OpImmediate::Specified(output->tensor->GetDynRawShape()), OpImmediate::Specified(output->GetDynValidShape())
            ));

            APASS_LOG_INFO_F(Elements::Operation,"Reshape op %d is replaceed by reshapeCopyOutOp %d and reshapeCopyInOp %d.", 
                op.opmagic, reshapeCopyOutOp.opmagic, reshapeCopyInOp.opmagic);
            processedReshapeOps.insert(op.GetOpMagic());
            break;
        }
    }
}

Status RemoveUnalignedReshape::ReplaceDynUnalignedReshapeOpsForDDR(Function &function, Operation &op) {
    auto input = op.GetIOperands().front();
    auto output = op.GetOOperands().front();

    auto inDynValidShape = input->GetDynValidShape();
    auto outDynValidShape = output->GetDynValidShape();

    bool hasNonImmediate = false;
    size_t maxSize = std::max(inDynValidShape.size(), outDynValidShape.size());
    for (size_t i = 0; i < maxSize; i++) {
        if (i < inDynValidShape.size() && !inDynValidShape[i].IsImmediate()){
            hasNonImmediate = true;
            break;
        }
        if (i < outDynValidShape.size() && !outDynValidShape[i].IsImmediate()){
            hasNonImmediate = true;
            break;
        }
    }
    
    if (hasNonImmediate) {
        bool hasiOtherBranch = false;
        std::vector<Operation *> copyOutOps = FindAllProducerCopyOuts(input, hasiOtherBranch);
        if (hasiOtherBranch) {
            APASS_LOG_ERROR_F(Elements::Operation, "There are more one op-tensor branch between Reshape and CopyOut.");
            return FAILED;
        }
        if (copyOutOps.size() != 1) {
            APASS_LOG_ERROR_F(Elements::Operation, "Reshape op %d has %lu copy_out producers, only support exactly 1.", op.GetOpMagic(), copyOutOps.size());
            return FAILED;
        }
        Operation *copyOutOp = copyOutOps.front();
        
        std::vector<Operation *> copyInOps;
        bool hasViewOrAssemble = false;
        FindAllConsumerCopyIns(output, copyInOps, hasViewOrAssemble);
        if (hasViewOrAssemble) {
            APASS_LOG_ERROR_F(Elements::Operation, "Reshape op %d is has view or assemble between reshape and copy in, not supported now.", op.GetOpMagic());
            return FAILED;
        }
         if (copyInOps.empty()) {
            APASS_LOG_ERROR_F(Elements::Operation, "Cannot find copy_in consumers for reshape op %d.", op.GetOpMagic());
            return FAILED;
        }

        auto copyOutConsumers = copyOutOp->GetOOperands().front()->GetConsumers();
        if (copyOutConsumers.size() != 1 || *(copyOutConsumers.begin()) != &op) {
            APASS_LOG_ERROR_F(Elements::Operation, "Reshape op %d has branch consumers before reshape, not supported.", op.GetOpMagic());
            return FAILED;
        }

        ProcessCopyOutOfDDRReshape(function, op, copyOutOp);
        ProcessCopyInOfDDRReshape(function, op, copyInOps);
    }
    processedReshapeOps.insert(op.GetOpMagic());

    return SUCCESS;
}

void RemoveUnalignedReshape::ProcessCopyOutOfDDRReshape(Function &function, Operation &op, Operation * copyOutOp) {
    //当copyout的输入是ub输出为ddr可以直接转化为reshapecopyop
    //否则需要插copy
    auto copyOutInput = copyOutOp->GetIOperands().front();
    auto copyOutInputMemType = copyOutInput->GetMemoryTypeOriginal();
    auto copyOutOutput = copyOutOp->GetOOperands().front();
    auto copyOutOutputMemType = copyOutOutput->GetMemoryTypeOriginal();
    if (copyOutInputMemType == MemoryType::MEM_UB && copyOutOutputMemType == MemoryType::MEM_DEVICE_DDR) {
        copyOutOp->SetOpCode(Opcode::OP_RESHAPE_COPY_OUT);
    } else if (copyOutInputMemType == MemoryType::MEM_L1 && copyOutOutputMemType == MemoryType::MEM_DEVICE_DDR) {
        //copyOutInput(L1) -- COPYOUT -- copyOutOutput(ddr) -- reshape
        //copyOutInput(L1) -- COPYOUT -- copyOutOutput(ddr) -- COPYIN -- newTensor(UB) -- RESHAPECOPYOUT -- newTensor2(ddr) -- reshape
        LogicalTensor newTensor(function, copyOutOutput->Datatype(), copyOutOutput->GetShape());
        newTensor.SetMemoryTypeBoth(MemoryType::MEM_UB, true);
        auto newTensorPtr = std::make_shared<LogicalTensor>(std::move(newTensor));
        auto &reshapeCopyInOp = function.AddOperation(Opcode::OP_COPY_IN, {copyOutOutput}, {newTensorPtr});
        reshapeCopyInOp.UpdateSubgraphID(op.GetSubgraphID());
        reshapeCopyInOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
            OpImmediate::Specified(std::vector<SymbolicScalar>(copyOutOutput->GetShape().size(), 0)),
            MemoryType::MEM_UB, OpImmediate::Specified(copyOutOutput->GetShape()),
            OpImmediate::Specified(copyOutOutput->tensor->GetDynRawShape()),
            OpImmediate::Specified(copyOutOutput->GetDynValidShape())
        ));
        
        LogicalTensor newTensor2(function, copyOutOutput->Datatype(), copyOutOutput->GetShape());
        newTensor2.SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
        auto newTensor2Ptr = std::make_shared<LogicalTensor>(std::move(newTensor2));
        auto &newCopyOutOp = function.AddOperation(Opcode::OP_RESHAPE_COPY_OUT, {newTensorPtr}, {newTensor2Ptr});
        newCopyOutOp.UpdateSubgraphID(op.GetSubgraphID());
        newCopyOutOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
            MemoryType::MEM_UB, OpImmediate::Specified(std::vector<SymbolicScalar>(copyOutOutput->GetShape().size(), 0)),
            OpImmediate::Specified(copyOutOutput->GetShape()),
            OpImmediate::Specified(copyOutOutput->tensor->GetDynRawShape()),
            OpImmediate::Specified(copyOutOutput->GetDynValidShape())
        ));

        copyOutOutput->RemoveConsumer(&op);
        op.ReplaceInput(newTensor2Ptr, copyOutOutput);
    }
}

void RemoveUnalignedReshape::ProcessCopyInOfDDRReshape(Function &function, Operation &op, std::vector<Operation *> &copyInOps) {
    for (auto *copyInOp : copyInOps) {
        auto copyInInput = copyInOp->GetIOperands().front();
        auto copyInInputMemType = copyInInput->GetMemoryTypeOriginal();
        auto copyInOutput = copyInOp->GetOOperands().front();
        auto copyInOutputMemType = copyInOutput->GetMemoryTypeOriginal();
        if (copyInInputMemType == MemoryType::MEM_DEVICE_DDR && copyInOutputMemType == MemoryType::MEM_UB) {
            copyInOp->SetOpCode(Opcode::OP_RESHAPE_COPY_IN);
        } else if (copyInInputMemType == MemoryType::MEM_DEVICE_DDR && copyInOutputMemType == MemoryType::MEM_L1) {
            //reshape -- copyInInput(ddr) -- COPYIN -- copyInOutout(L1)
            //reshape -- copyInInput(ddr) -- RESHAPECOPYIN -- newTensor(ub) -- COPYOUT -- newTensor2(ddr) -- COPYIN --copyInOutout(L1)
            LogicalTensor newTensor(function, copyInInput->Datatype(), copyInInput->GetShape());
            newTensor.SetMemoryTypeBoth(MemoryType::MEM_UB, true);
            auto newTensorPtr = std::make_shared<LogicalTensor>(std::move(newTensor));
            auto &reshapeCopyInOp = function.AddOperation(Opcode::OP_RESHAPE_COPY_IN, {copyInInput}, {newTensorPtr});
            reshapeCopyInOp.UpdateSubgraphID(op.GetSubgraphID());
            reshapeCopyInOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                OpImmediate::Specified(std::vector<SymbolicScalar>(copyInInput->GetShape().size(), 0)),
                MemoryType::MEM_UB, OpImmediate::Specified(copyInInput->GetShape()),
                OpImmediate::Specified(copyInInput->tensor->GetDynRawShape()),
                OpImmediate::Specified(copyInInput->GetDynValidShape())
            ));
            
            LogicalTensor newTensor2(function, copyInInput->Datatype(), copyInInput->GetShape());
            newTensor2.SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
            auto newTensor2Ptr = std::make_shared<LogicalTensor>(std::move(newTensor2));
            auto &newCopyOutOp = function.AddOperation(Opcode::OP_COPY_OUT, {newTensorPtr}, {newTensor2Ptr});
            newCopyOutOp.UpdateSubgraphID(op.GetSubgraphID());
            newCopyOutOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                MemoryType::MEM_UB, OpImmediate::Specified(std::vector<SymbolicScalar>(copyInInput->GetShape().size(), 0)),
                OpImmediate::Specified(copyInInput->GetShape()),
                OpImmediate::Specified(copyInInput->tensor->GetDynRawShape()),
                OpImmediate::Specified(copyInInput->GetDynValidShape())
            ));

            copyInInput->RemoveConsumer(copyInOp);
            copyInOp->ReplaceInput(newTensor2Ptr, copyInInput);
        }
    }
}

/* 从tensor的生产者列表中递归查找所有OP_COPY_OUT：
 * - 如果遇到OP_COPY_OUT，记录并继续查找（可能还有其他生产者）
 * - 如果遇到OP_VIEW、OP_ASSEMBLE或OP_ASSEMBLE_SSA，递归继续向前追溯
 * - 遇到其他op也继续递归追溯
 */
std::vector<Operation *> RemoveUnalignedReshape::FindAllProducerCopyOuts(LogicalTensorPtr tensor, bool &hasOtherBranch) {
    std::vector<Operation *> copyOutOps;
    //后续实现
    if (tensor->GetConsumers().size() > 1) {
        hasOtherBranch = true;
        return copyOutOps;
    }
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
            auto subCopyOuts = FindAllProducerCopyOuts(inputOperands.front(), hasOtherBranch);
            if (hasOtherBranch) {
                return copyOutOps;
            }
            copyOutOps.insert(copyOutOps.end(), subCopyOuts.begin(), subCopyOuts.end());
        }
    }

    return copyOutOps;
}
/* 从tensor的消费者列表中查找OP_COPY_IN，如果遇到OP_VIEW、OP_ASSEMBLE或OP_ASSEMBLE_SSA，
 * 则标记hasViewOrAssemble为true，表示不支持此类场景。
 */
void RemoveUnalignedReshape::FindAllConsumerCopyIns(
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
        if (op.GetOpcode() != Opcode::OP_RESHAPE || processedReshapeOps.count(op.GetOpMagic())) {
            continue;
        }

        if (!CheckUnaligned(op)) {
            continue;
        }

        auto input = op.GetIOperands().front();
        auto output = op.GetOOperands().front();
        if ((input->GetMemoryTypeOriginal() != MemoryType::MEM_UB) || (output->GetMemoryTypeOriginal() != MemoryType::MEM_UB)) {
            continue;
        }

        // 插入copyout
        LogicalTensorPtr newReshapeInput = InsertIOTensor(function, op, reshapeRawInputs, input);
        copyOuts.emplace_back(CopyOutOpMemUnalign{input->GetMemoryTypeOriginal(), input->offset, input, newReshapeInput});
        op.ReplaceInput(newReshapeInput, input);

        // 插入copyin
        LogicalTensorPtr newReshapeOutput = InsertIOTensor(function, op, reshapeRawOutputs, output);
        copyIns.emplace_back(
            CopyInOpMemUnalign{output->GetMemoryTypeOriginal(), output->offset, newReshapeOutput, output});
        op.ReplaceOutput(newReshapeOutput, output);
        output->tensor->actualRawmagic = -1;
        op.GetOOperands().front()->tensor->actualRawmagic = op.GetIOperands().front()->tensor->GetRawMagic();
        processedReshapeOps.insert(op.GetOpMagic());
    }
}

} // namespace
