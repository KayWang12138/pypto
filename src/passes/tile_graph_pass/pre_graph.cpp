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
 * \file pre_graph.cpp
 * \brief
 */

#include "pre_graph.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {

void PreGraphPass::ResetMemoryMap(Function &function) const {
    /* 初始化所有tensor的memorymap */
    for (auto &op : function.Operations()) {
        for (auto &input : op.GetIOperands()) {
            input->memorymap.clear();
        }
        for (auto &output : op.GetOOperands()) {
            output->memorymap.clear();
        }
    }
}

/* 将某个op的输入是expected的替换为newTensor并刷新Producer、Consumer关系 */
void SubstituteInput(Operation *op, LogicalTensorPtr &expected, LogicalTensorPtr &newTensor) {
    for (auto &input : op->iOperand) {
        if (input == expected) {
            newTensor->AddConsumer(op);
            input->RemoveConsumer(op);
            input = newTensor;
        }
    }
}

bool CalculateNewRawShape(const std::vector<int> &oriShape, const std::vector<int> &newShape,
    const std::vector<int> &oriRawShape, std::vector<int> &newRawShape) {
    std::vector<int> oriScale;
    size_t oriSize = oriShape.size();
    oriScale.resize(oriSize);
    for (size_t i = 0; i < oriSize; i++) {
        oriScale[i] = oriRawShape[i] / oriShape[i];
        if (i != 0) {
            if (oriScale[i] != 1) { // 只有当最高轴存在DAssemble的行为时，才可以将数据直接拷贝到DAssemble之后的内存
                return false;
            }
        }
    }
    ALOG_DEBUG_F("oriScale is %s", IntVecToStr(oriScale).c_str());
    size_t newSize = newShape.size();
    newRawShape.resize(newSize);
    std::vector<int> newScale(newSize, 1);
    int64_t accumuOriScale = oriScale[oriSize - 1];
    int64_t accumuOriShape = oriShape[oriSize - 1];
    int64_t accumuNewShape = newShape[newSize - 1];
    for (int i = oriSize - 1, j = newSize - 1; i >= 0 && j >= 0;) {
        if (accumuOriShape < accumuNewShape) {
            i--;
            if (i >= 0) {
                accumuOriShape *= oriShape[i];
                accumuOriScale *= oriScale[i];
            }
        } else if (accumuOriShape == accumuNewShape) {
            newScale[j] *= accumuOriScale;
            i--;
            j--;
            if (i >= 0 && j >= 0) {
                accumuOriScale = oriScale[i];
                accumuOriShape = oriShape[i];
                accumuNewShape = newShape[j];
            }
        } else {
            j--;
            if (j >= 0) {
                accumuNewShape *= newShape[j];
            }
        }
    }

    ALOG_DEBUG_F("newScale is %s", IntVecToStr(newScale).c_str());
    for (size_t j = 0; j < newSize; j++) {
        newRawShape[j] = newShape[j] * newScale[j];
    }
    return true;
}

void GetDynOffsetBeforeReshape(const std::vector<SymbolicScalar> &oriOffset, const std::vector<int> &oriShape,
    const std::vector<int> &newShape, std::vector<SymbolicScalar> &newOffset) {
    // 计算原始shape的步长（stride）
    ASSERT(oriShape.size() == oriOffset.size());
    size_t oriSize = oriOffset.size();
    size_t newSize = newShape.size();
    std::vector<int> oriStride(oriShape.size());
    int currentStride = 1;
    for (int i = oriSize - 1; i >= 0; --i) {
        oriStride[i] = currentStride;
        currentStride *= oriShape[i];
    }
    // 计算原始偏移量对应的线性索引
    SymbolicScalar linearIndex = oriOffset[0] * SymbolicScalar(oriStride[0]);
    for (size_t i = 1; i < oriOffset.size(); ++i) {
        linearIndex = linearIndex + oriOffset[i] * SymbolicScalar(oriStride[i]);
    }

    // 计算新shape的步长
    std::vector<int> newStride(newSize);
    currentStride = 1;
    for (int i = newSize - 1; i >= 0; --i) {
        newStride[i] = currentStride;
        currentStride *= newShape[i];
    }

    // 根据线性索引计算新的偏移量
    newOffset.resize(newSize);
    for (size_t i = 0; i < newSize; ++i) {
        newOffset[i] = linearIndex / SymbolicScalar(newStride[i]);
        linearIndex = linearIndex % SymbolicScalar(newStride[i]);
    }
}

/*
生效场景:
DAssemble拆分了最高轴，认为可以透传，不需要拷贝，前序在ExpandFunction中做了判断，属性NeedCopy=false
Copy_Out --> tensor(GM) --> Reshape --> oriBackUp [16, 16] --> DAssemble(offset, dynOffset) --> OCAST(offset, dynOffset) [16, 64]
因此需要: 重新计算Reshape输入的RawShape, offset, dynOffset
*/
void HandleDynOffsetForReshape(const LogicalTensorPtr &oriBackUp, std::unordered_set<Operation *> &concurrentAssembles,
    const std::set<Operation *, LogicalTensor::CompareOp> &producers) {
    std::vector<SymbolicScalar> newDynOffset;
    std::vector<int> newRawShape;
    for (auto assemble : concurrentAssembles) {
        auto opAttr = dynamic_cast<AssembleOpAttribute *>(assemble->GetOpAttribute().get());
        if (opAttr == nullptr) {
            continue;
        }
        auto &dynOffset = opAttr->GetToDynOffset();
        if (dynOffset.empty()) {
            continue;
        }
        if (concurrentAssembles.size() != 1 || producers.size() != 1) {
            return;
        }
        auto producer = *(producers.begin());
        if (producer->GetOpcode() != Opcode::OP_RESHAPE) {
            return;
        }

        auto &assembleOutShape = assemble->GetOOperands()[0]->tensor->rawshape;
        bool ret =
            CalculateNewRawShape(oriBackUp->shape, producer->GetIOperands()[0]->shape, assembleOutShape, newRawShape);
        if (ret == false) {
            return;
        }
        GetDynOffsetBeforeReshape(dynOffset, assembleOutShape, newRawShape, newDynOffset);
        for (auto copyOut : producer->GetIOperands()[0]->GetProducers()) {
            if (!IsCopyOut(copyOut->GetOpcode())) {
                continue;
            }
            const std::shared_ptr<OpAttribute> &attr = copyOut->GetOpAttribute();
            if (attr == nullptr) {
                continue;
            }
            std::shared_ptr<CopyOpAttribute> copyAttr = std::static_pointer_cast<CopyOpAttribute>(attr);
            auto oriCopyOffset = copyAttr->GetToOffset();
            std::vector<OpImmediate> newOffset = OpImmediate::Specified(newDynOffset);
            for (size_t i = 0; i < oriCopyOffset.size(); i++) {
                newOffset[i] = newOffset[i] + oriCopyOffset[i];
            }
            copyAttr->SetRawShape(OpImmediate::Specified(newRawShape));
            copyAttr->SetToOffset(newOffset);
        }
        producer->GetIOperands()[0]->tensor->rawshape = newRawShape;
    }
}

void PreGraphPass::HandleForAssembleFromInOut(Function &function, std::unordered_set<Operation *> &concurrentAssembles,
    std::set<Operation *, LogicalTensor::CompareOp> &producersBackup) const {
    LogicalTensorPtr inOrOutTensor = nullptr;
    for (auto &assemble : concurrentAssembles) {
        if (function.IsFromInCast(assemble->iOperand[0]) || function.IsFromOutCast(assemble->iOperand[0])) {
            inOrOutTensor = assemble->iOperand[0];
            break;
        }
    }
    if (inOrOutTensor == nullptr) {
        return;
    }
    ALOG_DEBUG_F("find in or out, tensor magic: %d, raw magic: %d", inOrOutTensor->magic, inOrOutTensor->GetRawMagic());
    for (auto &producer : producersBackup) {
        producer->oOperand[0]->tensor = inOrOutTensor->tensor;
        for (auto &cons : producer->oOperand[0]->GetConsumers()) {
            if (cons->GetOpcode() == Opcode::OP_RESHAPE && cons->oOperand[0]->tensor->actualRawmagic != -1) {
                ALOG_DEBUG_F("consumer[%d] is OP_RESHAPE", cons->GetOpMagic());
                cons->oOperand[0]->tensor->actualRawmagic = inOrOutTensor->GetRawMagic();
            }
        }
    }
}


/*
                /--> Assemble1-1(self) --> Tensor
op1 --> tensor1 ---> Assemble1-2 --> OCAST
                \--> Reshape --> tensor2
*/
/*
                /--> Assemble1-1(self) --> Tensor
op1 --> tensor1 ---> Assemble1-2 --> OCAST

                /--> Assemble2-1 --> Tensor
op2 --> tensor2 ---> Assemble2-2 --> OCAST
*/
void PreGraphPass::HandleForAssembleToOutcast(Function &function, std::unordered_set<Operation *> &concurrentAssembles,
    std::set<Operation *, LogicalTensor::CompareOp> &producersBackup) const {
    int outCastMagic = -1;
    for (auto &assemble : concurrentAssembles) {
        if (function.IsFromOutCast(assemble->oOperand[0]) && assemble->oOperand[0]->nodetype == NodeType::OUTCAST) {
            outCastMagic = assemble->oOperand[0]->GetMagic();
            break;
        }
    }
    if (outCastMagic != -1) {
        ALOG_DEBUG_F("find outCastMagic: %d", outCastMagic);
        for (auto &producer : producersBackup) {
            producer->oOperand[0]->SetMagic(outCastMagic);
            producer->oOperand[0]->nodetype = NodeType::OUTCAST;
        }
    }
}

void PreGraphPass::HandleForReshapeToOutcast(Function &function) const {
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            if (function.IsFromOutCast(op.GetOOperands()[0])) {
                op.GetIOperands()[0]->tensor->actualRawmagic = op.GetOOperands()[0]->GetRawMagic();
            }
        }
    }
}

/*
    Producer1 -->
                 \
    Producer2 --> input --> consAssemble -----> Tensor1 --> Op1
                    \---> consumer ------> Tensor2 --> Op2
    will be modified to:
    Producer1 -->
                 \
    Producer2 --> Tensor1 --> Op1
                    \---> Consumer ------> Tensor2 --> Op2
*/
void PreGraphPass::DeleteRedundantAssemble(Function &function) const {
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        auto &output = op.GetOOperands().front();
        if (output->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR || op.IsDeleted()) {
            continue;
        }
        auto &input = op.GetIOperands().front();
        auto &consumers = input->GetConsumers();
        std::unordered_set<Operation *> concurrentAssembles;
        auto producersBackup = input->GetProducers();
        LogicalTensorPtr oriOutputBackUp = nullptr;
        for (auto &cons : consumers) {
            if (cons->GetOpcode() == Opcode::OP_ASSEMBLE) {
                cons->SetAsDeleted();
                if (concurrentAssembles.empty()) {
                    concurrentAssembles.emplace(cons);
                    for (auto &producer : producersBackup) {
                        oriOutputBackUp = producer->oOperand[0]; // producer --> oriOutputBackUp(input) --> op
                        producer->ReplaceOutput(output, oriOutputBackUp);
                        output->isSubGraphBoundary = true;
                        if (IsCopyOut(producer->GetOpcode())) {
                            auto opAttr = std::static_pointer_cast<CopyOpAttribute>(producer->GetOpAttribute());
                            auto consOpAttr = std::static_pointer_cast<AssembleOpAttribute>(cons->GetOpAttribute());
                            if (consOpAttr->GetToDynOffset().size() != 0) {
                                opAttr->SetToOffset(OpImmediate::Specified(consOpAttr->GetToDynOffset()));
                            }
                            opAttr->SetRawShape(OpImmediate::Specified(output->tensor->GetRawShape()));
                        }
                    }
                } else {
                    concurrentAssembles.emplace(cons);
                    for (auto &consOfCons : cons->oOperand[0]->GetConsumers()) {
                        SubstituteInput(consOfCons, cons->oOperand[0], output);
                    }
                }
            } else {
                cons->iOperand[0] = output;
                cons->iOperand[0]->AddConsumer(cons);
            }
        }
        HandleForAssembleFromInOut(function, concurrentAssembles, producersBackup);
        HandleForAssembleToOutcast(function, concurrentAssembles, producersBackup);
        HandleDynOffsetForReshape(oriOutputBackUp, concurrentAssembles, producersBackup); // op为DAssemble
    }
    function.EraseOperations(false);
    HandleForReshapeToOutcast(function);
}

void PreGraphPass::ProcessSpecialMTEOperation(Operation &op) const {
    ALOG_DEBUG_F("Process Special MTE Operation %d", op.opmagic);
    auto inputTensor = op.iOperand.front();
    auto outputTensor = op.oOperand.front();
    if ((inputTensor == nullptr) || (outputTensor == nullptr)) {
        return;
    }
    if (op.GetOpcode() == Opcode::OP_TRANSPOSE_DATAMOVE) {
        /* transpose datamove 输入和输出的shape不相同 */
        op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MemoryType::MEM_UB,
            OpImmediate::Specified(outputTensor->GetTensorOffset()), OpImmediate::Specified(outputTensor->GetShape()),
            OpImmediate::Specified(outputTensor->tensor->GetDynRawShape())));
        op.oOperand[0]->isSubGraphBoundary = true;
    } else {
        op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MemoryType::MEM_UB,
            OpImmediate::Specified(outputTensor->GetTensorOffset()), OpImmediate::Specified(outputTensor->GetShape()),
            OpImmediate::Specified(outputTensor->tensor->GetDynRawShape())));
        op.oOperand[0]->isSubGraphBoundary = true;
    }
}

void PreGraphPass::InsertTemporaryCopyIn(Function &function, Operation &op) const {
    if (op.GetOpcode() == Opcode::OP_REMOTE_REDUCE ||
        op.GetOpcode() == Opcode::OP_WRITE_REMOTE ||
        op.GetOpcode() == Opcode::OP_LOCAL_COPY_OUT ||
        op.GetOpcode() == Opcode::OP_REMOTE_GATHER ||
        op.GetOpcode() == Opcode::OP_MOE_FFN_TO_ATTN ||
        op.GetOpcode() == Opcode::OP_MOE_ATTN_COMBINE ||
        op.GetOpcode() == Opcode::OP_SEND_TO_ROUTING_EXPERT ||
        op.GetOpcode() == Opcode::OP_SEND_TO_SHARED_EXPERT ||
        op.GetOpcode() == Opcode::OP_FFN_SCHED ||
        op.GetOpcode() == Opcode::OP_FFN_BATCHING ||
        op.GetOpcode() == Opcode::OP_COPY_TO_LOCAL_EXPERT ||
        op.GetOpcode() == Opcode::OP_DISPATCH_SET_FLAG) {
        for (auto &input : op.GetIOperands()) {
            if (input->GetProducers().size() == 0 && input->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
                // insert Copy_In before the op
                input->isSubGraphBoundary = false;
                std::vector<std::shared_ptr<LogicalTensor>> operandGm;
                std::shared_ptr<LogicalTensor> tensorGM =
                    std::make_shared<LogicalTensor>(function, input->Datatype(), input->shape);
                tensorGM->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
                tensorGM->SetMemoryTypeToBe(MemoryType::MEM_DEVICE_DDR);
                tensorGM->isSubGraphBoundary = true;
                tensorGM->subGraphID = op.GetSubgraphID();
                operandGm.push_back(tensorGM);
                function.GetTensorMap().Insert(tensorGM);

                std::vector<std::shared_ptr<LogicalTensor>> operandUb;
                operandUb.push_back(input);

                // add UB_Alloc && UB_COPY_IN
                auto &ubCopyIn = function.AddRawOperation(Opcode::OP_COPY_IN, operandGm, operandUb);
                ubCopyIn.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                    OpImmediate::Specified(input->GetTensorOffset()), MemoryType::MEM_UB,
                    OpImmediate::Specified(input->GetShape()), OpImmediate::Specified(input->tensor->GetDynRawShape()),
                    OpImmediate::Specified(input->GetDynValidShape())));
                ubCopyIn.SetAttribute(OpAttributeKey::isCube, false);
                ubCopyIn.UpdateSubgraphID(op.GetSubgraphID());
            }
        }
    }
}

void DFSColor(int color, int &count, SubgraphColorInfo &info, std::vector<std::set<int>> &inColorGraph,
    std::vector<std::set<int>> &outColorGraph) {
    if (info.visited[color]) {
        return;
    }
    for (auto inputColor : inColorGraph[color]) {
        if (!info.visited[inputColor]) {
            DFSColor(inputColor, count, info, inColorGraph, outColorGraph);
        }
    }
    if (!info.visited[color]) {
        info.visited[color] = true;
        count--;
        info.newColor.push_back(color);
    }
    for (auto outputColor : outColorGraph[color]) {
        if (!info.visited[outputColor]) {
            DFSColor(outputColor, count, info, inColorGraph, outColorGraph);
        }
    }
}

void PreGraphPass::SortColor(Function &function) {
    int colorCount = function.GetTotalSubGraphCount();
    std::vector<std::set<int>> inColorGraph(colorCount);
    std::vector<std::set<int>> outColorGraph(colorCount);
    std::set<int> graphIds;
    auto opList = function.Operations();
    for (size_t i = 0; i < opList.size(); i++) {
        auto &op = opList[i];
        if (op.GetSubgraphID() > -1) {
            auto curColor = op.GetSubgraphID();
            graphIds.insert(curColor);
            for (auto outTensor : op.GetOOperands()) {
                for (auto consumer : outTensor->GetConsumers()) {
                    if (consumer->GetSubgraphID() == -1) {
                        continue;
                    }
                    auto outColor = consumer->GetSubgraphID();
                    if (curColor != outColor) {
                        inColorGraph[outColor].insert(curColor);
                        outColorGraph[curColor].insert(outColor);
                    }
                }
            }
        }
    }
    if (graphIds.size() != static_cast<size_t>(colorCount)) {
        ALOG_ERROR_F("graphIds size is %zu, but graphIds count is %d", graphIds.size(), colorCount);
    }
    std::vector<bool> visited(colorCount, false);
    std::vector<int> newColor;
    int preStart = 0;
    int leftVisitedColor = colorCount;
    SubgraphColorInfo colorInfo{visited, newColor};
    while (leftVisitedColor > 0) {
        int start = preStart;
        while (colorInfo.visited[start]) {
            start++;
        }
        preStart = start;
        DFSColor(start, leftVisitedColor, colorInfo, inColorGraph, outColorGraph);
    }

    for (int i = 0; i < colorCount; i++) {
        oldToNewColor[colorInfo.newColor[i]] = i;
    }
}

void PreGraphPass::ProcessInplaceOp(Function &function) const {
    for (auto &op : function.Operations()) {
        /*
        正式方案：需要inplace的op帶有attribute, 通过attribute判斷
        */
        if (op.GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
            auto outputTensor = op.oOperand.front();
            ASSERT(op.GetOOperands().size() == 1);
            for (auto &input : op.GetIOperands()) {
                if (input->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
                    /*
                    case0:
                    dst(ddr) --> index_outcast --> output

                    case1: index_outcast输出为OCAST，要将dst, partial_dst的raw tensor统一刷新为OCAST的raw tensor
                    dst(ddr) --> view --> partial_dst(ddr) --> index_outcast --> output (OCAST)

                    case2: index_outcast输出不是OCAST, 要将partial_dst, output的raw tensor统一刷新为dst的raw tensor
                    dst(ddr) --> view --> partial_dst(ddr) --> index_outcast --> output --> op
                    */
                    if (input->GetProducers().empty()) {
                        input->tensor = outputTensor->tensor;
                        break;
                    }
                    auto parentView = *(input->GetProducers().begin());
                    ASSERT(parentView->GetOpcode() == Opcode::OP_VIEW);
                    auto dstIncast = parentView->GetIOperands().front();
                    ASSERT(dstIncast->tensor->GetRawDataSize() == outputTensor->tensor->GetRawDataSize());
                    if (function.IsFromOutCast(outputTensor)) {
                        dstIncast->tensor = outputTensor->tensor;
                        input->tensor = outputTensor->tensor;
                    } else {
                        input->tensor = dstIncast->tensor;
                        outputTensor->tensor = dstIncast->tensor;
                    }
                    for (auto &child : outputTensor->GetConsumers()) {
                        if (child->GetOpcode() == Opcode::OP_RESHAPE) {
                            child->oOperand[0]->tensor->actualRawmagic = outputTensor->GetRawMagic();
                        }
                    }
                    break;
                }
            }
        }
    }
}

void PreGraphPass::ProcessSameInOutOp(Function &function) const {
    for (auto &op : function.Operations()) {
        Opcode prod;
        if (op.GetAttr(OpAttributeKey::sameInOut, prod)) {
            for (auto &input : op.GetIOperands()) {
                for (auto &producer : input->GetProducers()) {
                    if (producer->GetOpcode() == prod) {
                        ASSERT(op.GetOOperands().size() == 1) << "output tensor size is not 1";
                        auto output = op.GetOOperands().front();
                        ASSERT(input->shape == output->shape)
                            << "op input output tensor shape is not equal, cannot reuse buffer";
                        input->tensor = output->tensor;
                        input->offset = output->offset;
                        if (IsCopyOut(producer->GetOpcode())) {
                            std::shared_ptr<CopyOpAttribute> attr =
                                std::static_pointer_cast<CopyOpAttribute>(producer->GetOpAttribute());
                            attr->SetToOffset(OpImmediate::Specified(output->offset));
                        }
                    }
                }
            }
        }
    }
}

void PreGraphPass::UpdateCopyOpIsCube(Operation &op) const {
    /*
    后续考虑移到InsertCopyOp
    copy_out for producer
    op(copy_in) --> input --> consumerOp(isCube?)
    */
    if (IsCopyIn(op.GetOpcode())) {
        for (auto &consumerOps : op.ConsumerOps()) {
            if ((consumerOps->HasAttr(OpAttributeKey::isCube)) &&
                (consumerOps->GetSubgraphID() == op.GetSubgraphID())) {
                op.SetAttribute(OpAttributeKey::isCube, consumerOps->GetBoolAttribute(OpAttributeKey::isCube));
                break;
            }
        }
    }
    /*
    copy_in for consumer
    producerOp(isCube?) --> input --> op(copy_out)
    */
    if (IsCopyOut(op.GetOpcode())) {
        for (auto &producerOps : op.ProducerOps()) {
            if ((producerOps->HasAttr(OpAttributeKey::isCube)) &&
                (producerOps->GetSubgraphID() == op.GetSubgraphID())) {
                op.SetAttribute(OpAttributeKey::isCube, producerOps->GetBoolAttribute(OpAttributeKey::isCube));
                break;
            }
        }
    }
}

void PreGraphPass::InitializeTensorMemorymap(Operation &op) const {
    const int newColor = oldToNewColor.at(op.GetSubgraphID());
    for (auto &input : op.GetIOperands()) {
        /* 使用TIG时需要跳过local buffer 的初始化流程 */
        for (auto &rangePair : input->memorymap) {
            if (rangePair.second.memId == -1) {
                rangePair.second.memId = input->tensor->GetRawMagic();
            }
        }
        TileRange range;
        range.memId = input->tensor->GetRawMagic();
        input->memorymap.insert(std::make_pair(newColor, range));
        if (input->GetProducers().size() == 0) {
            input->subGraphID = newColor;
        }
    }
    for (auto &output : op.GetOOperands()) {
        /* 使用TIG时需要跳过local buffer 的初始化流程 */
        for (auto &rangePair : output->memorymap) {
            if (rangePair.second.memId == -1) {
                rangePair.second.memId = output->tensor->GetRawMagic();
            }
        }
        TileRange range;
        range.memId = output->tensor->GetRawMagic();
        output->memorymap.insert(std::make_pair(newColor, range));
        output->subGraphID = newColor;
    }
}

void PreGraphPass::SetTensorBoundary(Function &function) const {
    for (auto &op : function.Operations()) {
        /* memory map size > 1 代表该tensor被多个子图使用，那么标记为boundary*/
        for (auto &input : op.GetIOperands()) {
            if (input->memorymap.size() > 1) {
                input->isSubGraphBoundary = true;
            }
            if (input->GetProducers().size() == 0) {
                input->isSubGraphBoundary = true;
                InsertTemporaryCopyIn(function, op);
            }
        }
        for (auto &output : op.GetOOperands()) {
            if (output->memorymap.size() > 1) {
                output->isSubGraphBoundary = true;
            }
        }
        if (op.GetOpcode() == Opcode::OP_COPY_IN) {
            /* Copy In 的输入*/
            op.GetIOperands().front()->isSubGraphBoundary = true;
        } else if (op.GetOpcode() == Opcode::OP_COPY_OUT) {
            /* Copy Out 的输出*/
            op.GetOOperands().front()->isSubGraphBoundary = true;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            /* GM上的Assemble*/
            auto assembleIn = op.GetIOperands().front();
            auto assembleOut = op.GetOOperands().front();
            bool isBoundary = (assembleOut->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR);
            assembleOut->isSubGraphBoundary = isBoundary;
            assembleIn->isSubGraphBoundary = isBoundary;
        } else if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            /* reshape*/
            auto reshapeIn = op.GetIOperands().front();
            auto reshapeOut = op.GetOOperands().front();
            bool isBoundary = (reshapeOut->isSubGraphBoundary || reshapeIn->isSubGraphBoundary);
            reshapeIn->isSubGraphBoundary = isBoundary;
            reshapeOut->isSubGraphBoundary = isBoundary;
        }
    }
}

Status PreGraphPass::RunOnFunction(Function &function) {
    ALOG_INFO_F("===> start PreGraphPass");
    ResetMemoryMap(function);
    SortColor(function);
    auto opList = function.Operations();
    for (auto &op : opList) {
        if (op.GetSubgraphID() > -1) {
            auto curColor = op.GetSubgraphID();
            assert(oldToNewColor.find(curColor) != oldToNewColor.end());
            InitializeTensorMemorymap(op);
            op.UpdateSubgraphID(oldToNewColor[curColor]);
            UpdateCopyOpIsCube(op);
        }
    }

    SetTensorBoundary(function);

    // Processing Special Ops
    for (auto &op : opList) {
        if (op.GetOpcode() == Opcode::OP_TRANSPOSE_DATAMOVE || op.GetOpcode() == Opcode::OP_INDEX_OUTCAST ||
            op.GetOpcode() == Opcode::OP_REMOTE_GATHER || op.GetOpcode() == Opcode::OP_LOCAL_COPY_OUT ||
            op.GetOpcode() == Opcode::OP_REMOTE_REDUCE || op.GetOpcode() == Opcode::OP_FFN_SCHED ||
            op.GetOpcode() == Opcode::OP_FFN_BATCHING || op.GetOpcode() == Opcode::OP_COPY_TO_LOCAL_EXPERT) {
            ProcessSpecialMTEOperation(op);
        }
        if ((op.GetOpcode() == Opcode::OP_ASSEMBLE) || (op.GetOpcode() == Opcode::OP_RESHAPE)) {
            // 校验单输入单输出，且输入输出mem类型相同
            if ((op.GetIOperands().size() != 1) || (op.GetOOperands().size() != 1) ||
                (op.GetIOperands().front() == nullptr) || (op.GetOOperands().front() == nullptr) ||
                (op.GetIOperands().front()->GetMemoryTypeOriginal() != op.GetOOperands().front()->GetMemoryTypeOriginal())) {
                return FAILED;
            }
        }
    }
    ProcessSameInOutOp(function);
    DeleteRedundantAssemble(function);
    ALOG_INFO_F("===> End PreGraphPass");
    return SUCCESS;
}

Status PreGraphPass::PreCheck(Function &function) {
    ALOG_INFO("PreCheck for pass: PreGraphPass");
    Pass::PreCheck(function);
    if (!function.LoopCheck().empty()) {
        ALOG_ERROR("Loopcheck failed before pass: PreGraphPass");
    }
    return SUCCESS;
}

Status PreGraphPass::PostCheck(Function &function) {
    ALOG_INFO("PostCheck for pass: PreGraphPass");
    Pass::PostCheck(function);
    if (!function.LoopCheck().empty()) {
        ALOG_ERROR("Loopcheck failed after pass: PreGraphPass");
    }

    for (auto &op : function.Operations()) {
        if ((op.GetOpcode() == Opcode::OP_ASSEMBLE || op.GetOpcode() == Opcode::OP_VIEW) &&
            op.GetIOperands()[0]->GetRawMagic() != op.GetOOperands()[0]->GetRawMagic()) {
            ALOG_WARN_F("Operation magic: %d, assemble or view op raw magic should not changed.", op.GetOpMagic());
        }

        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            if (op.GetOOperands()[0]->tensor->GetRawMagic() != op.GetIOperands()[0]->GetRawMagic()) {
                ALOG_ERROR_F(
                    "Operation magic: %d, reshape op's output actual raw magic shoule be same with input raw magic.",
                    op.GetOpMagic());
            }
            if (op.GetIOperands()[0]->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
                ALOG_DEBUG_F(" reshape on local buffer, opmagic: %d", op.opmagic);
                if (op.GetOOperands()[0]->GetMemoryTypeOriginal() != op.GetIOperands()[0]->GetMemoryTypeOriginal()) {
                    ALOG_ERROR_F("unmatched input output memory type for reshape opmagic: %d", op.opmagic);
                    ALOG_ERROR_F("input mem type: %s",
                        MemoryTypeToString(op.GetIOperands()[0]->GetMemoryTypeOriginal()).c_str());
                    ALOG_ERROR_F("output mem type: %s",
                        MemoryTypeToString(op.GetOOperands()[0]->GetMemoryTypeOriginal()).c_str());
                }
                auto opSubgraphId = op.GetSubgraphID();
                auto inputSubgraphId = op.GetIOperands()[0]->GetSubgraphID();
                auto outSubgraphId = op.GetIOperands()[0]->GetSubgraphID();
                ASSERT(opSubgraphId == inputSubgraphId && opSubgraphId == outSubgraphId);
                ASSERT(op.GetIOperands()[0]->memorymap.find(opSubgraphId) != op.GetOOperands()[0]->memorymap.end());
                ASSERT(op.GetOOperands()[0]->memorymap.find(opSubgraphId) != op.GetOOperands()[0]->memorymap.end());
                auto inputRange = op.GetIOperands()[0]->memorymap.find(opSubgraphId);
                auto outputRange = op.GetOOperands()[0]->memorymap.find(opSubgraphId);
                ALOG_DEBUG_F("input memid: %d, output memid: %d", inputRange->second.memId, outputRange->second.memId);
                if (inputRange->second.memId != outputRange->second.memId) {
                    ALOG_ERROR_F("unmatched memid for OP_RESHAPE, opmagic: %d, input memid: %d, output memid: %d",
                        op.opmagic, inputRange->second.memId, outputRange->second.memId);
                }
                ALOG_DEBUG_F(" check passed, input magic %d (raw %d), output magic %d (raw %d)",
                    op.GetIOperands()[0]->magic, op.GetIOperands()[0]->GetRawMagic(), op.GetOOperands()[0]->magic,
                    op.GetOOperands()[0]->GetRawMagic());
                auto childOp = *(op.GetOOperands()[0]->GetConsumers().begin());
                ALOG_DEBUG_F(" child op: %s, opmagic: %d", childOp->GetOpcodeStr().c_str(), childOp->opmagic);
                ALOG_DEBUG_F(" child op output magic %d (raw %d)", childOp->GetOOperands()[0]->magic,
                    childOp->GetOOperands()[0]->GetRawMagic());
                auto childoutputRange = childOp->GetOOperands()[0]->memorymap.find(opSubgraphId);
                ALOG_DEBUG_F(" child output memid: %d", childoutputRange->second.memId);
            }
        }

        for (const std::shared_ptr<LogicalTensor> &inputTensor : op.GetIOperands()) {
            if (inputTensor->subGraphID == NOT_IN_SUBGRAPH) {
                ALOG_WARN_F(
                    "Tensor magic: %d, its subgraph id should not be %d.", inputTensor->GetMagic(), NOT_IN_SUBGRAPH);
            }
            if (inputTensor->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
                inputTensor->isSubGraphBoundary == false) {
                ALOG_WARN_F("Tensor magic: %d, when memory type is DDR, this tensor should be subgraph boundary.",
                    inputTensor->GetMagic());
            }
            if (inputTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0C &&
                (inputTensor->Datatype() != DataType::DT_FP32 && inputTensor->Datatype() != DataType::DT_INT32)) {
                ALOG_WARN_F("Tensor magic: %d, when memory type is L0C, this tensor should be fp32 or int32.",
                    inputTensor->GetMagic());
            }
            for (auto &rangePair : inputTensor->memorymap) {
                if (rangePair.second.memId != inputTensor->GetRawMagic()) {
                    ALOG_WARN_F("Tensor magic: %d, its memId %d should be same with its raw magic %d, but not.",
                        inputTensor->GetMagic(), rangePair.second.memId, inputTensor->GetRawMagic());
                }
            }
            if (inputTensor->MemorySize() < 1) {
                ALOG_WARN_F("Tensor magic: %d, its memory size %d should be over than 0, but not.",
                    inputTensor->GetMagic(), inputTensor->MemorySize());
            }
        }

        for (const std::shared_ptr<LogicalTensor> &outputTensor : op.GetOOperands()) {
            if (outputTensor->subGraphID == NOT_IN_SUBGRAPH) {
                ALOG_WARN_F(
                    "Tensor magic: %d, its subgraph id should not be %d.", outputTensor->GetMagic(), NOT_IN_SUBGRAPH);
            }
            if (outputTensor->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
                outputTensor->isSubGraphBoundary == false) {
                ALOG_WARN_F("Tensor magic: %d, when memory type is DDR, this tensor should be subgraph boundary.",
                    outputTensor->GetMagic());
            }
            if (outputTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0C &&
                (outputTensor->Datatype() != DataType::DT_FP32 && outputTensor->Datatype() != DataType::DT_INT32)) {
                ALOG_WARN_F("Tensor magic: %d, when memory type is L0C, this tensor should be fp32 or int32.",
                    outputTensor->GetMagic());
            }
            for (auto &rangePair : outputTensor->memorymap) {
                if (rangePair.second.memId != outputTensor->GetRawMagic()) {
                    ALOG_WARN_F("Tensor magic: %d, its memId %d should be same with its raw magic %d, but not.",
                        outputTensor->GetMagic(), rangePair.second.memId, outputTensor->GetRawMagic());
                }
            }
            if (outputTensor->MemorySize() < 1) {
                ALOG_WARN_F("Tensor magic: %d, its memory size %d should be over than 0, but not.",
                    outputTensor->GetMagic(), outputTensor->MemorySize());
            }
        }
    }
    return SUCCESS;
}

} // namespace npu::tile_fwk
