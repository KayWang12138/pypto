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
 * \file remove_unaligned_reshape_op.cpp
 * \brief
 */

#include "remove_unaligned_reshape_op.h"
#include "passes/pass_utils/graph_utils.h"

namespace npu::tile_fwk {
/*
before:
    add->reshape(padded)->mul

after:
    add->copyout->reshape->copyin->mul
*/
Status RemoveUnalignedReshape::RunOnFunction(Function &function) {
    ALOG_INFO_F("===> start RemoveUnalignedReshape");
    CollectReshapeOps(function);
    for (auto &a : copyOuts) {
        GraphUtils::CopyDynStatus(a.output, a.input);
        auto &newCopyOut = function.AddRawOperation(Opcode::OP_COPY_OUT, {a.input}, {a.output});
        newCopyOut.SetOpAttribute(std::make_shared<CopyOpAttribute>(a.from, OpImmediate::Specified(a.toOffset),
            OpImmediate::Specified(newCopyOut.iOperand.front()->oriShape),
            OpImmediate::Specified(newCopyOut.oOperand.front()->tensor->GetDynRawShape())));
        auto producerOp = *(a.input->GetProducers().begin());
        newCopyOut.UpdateSubgraphID(producerOp->GetSubgraphID());
        ALOG_INFO_F("ADD OP_COPY_OUT, magic %d ,IOperand tensor magic %d OOperand tensor magic %d", newCopyOut.opmagic,
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
        ALOG_INFO_F("ADD OP_VIEW, magic %d ,IOperand tensor magic %d OOperand tensor magic %d", newCopyIn.opmagic,
            b.input->magic, b.output->magic);
    }
    ALOG_INFO_F("===> end RemoveUnalignedReshape");
    return SUCCESS;
}

LogicalTensorPtr RemoveUnalignedReshape::InsertIOTensor(Function &function, Operation &op, std::unordered_map<OverlaprawMagic, std::shared_ptr<RawTensor>> &rawIO, LogicalTensorPtr &ioTensor) {
    if (rawIO.count(ioTensor->tensor->rawmagic) == 0) {
        auto reshapeRawTensor = std::make_shared<RawTensor>(ioTensor->Datatype(), ioTensor->tensor->oriRawshape);
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

void RemoveUnalignedReshape::CollectReshapeOps(Function &function) {
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_RESHAPE) {
            continue;
        }

        bool isUnalign = op.GetBoolAttribute(OpAttributeKey::shapePadded);
        if (!isUnalign) {
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
    }
}

} // namespace
