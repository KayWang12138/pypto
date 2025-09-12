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
#include "passes/pass_check/pre_graph_checker.h"

namespace npu::tile_fwk {

void PreGraphProcess::ResetMemoryMap(Function &function) const {
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

bool CalculateNewRawShape(const std::vector<int64_t> &oriShape, const std::vector<int64_t> &newShape,
    const std::vector<int64_t> &oriRawShape, std::vector<int64_t> &newRawShape) {
    std::vector<int64_t> oriScale;
    size_t oriSize = oriShape.size();
    oriScale.resize(oriSize);
    for (size_t i = 0; i < oriSize; i++) {
        oriScale[i] = oriRawShape[i] / oriShape[i];
        if ((i != 0) && (oriScale[i] != 1)) {
            // 只有当最高轴存在Assemble的行为时，才可以将数据直接拷贝到Assemble之后的内存
            return false;
        }
    }
    ALOG_DEBUG_F("oriScale is %s.", IntVecToStr(oriScale).c_str());
    size_t newSize = newShape.size();
    newRawShape.resize(newSize);
    std::vector<int64_t> newScale(newSize, 1);
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
            continue;
        }
        if (accumuOriShape == accumuNewShape) {
            newScale[j] *= accumuOriScale;
            i--;
            j--;
            if (i >= 0 && j >= 0) {
                accumuOriScale = oriScale[i];
                accumuOriShape = oriShape[i];
                accumuNewShape = newShape[j];
            }
            continue;
        }
        j--;
        if (j >= 0) {
            accumuNewShape *= newShape[j];
        }
    }

    ALOG_DEBUG_F("newScale is %s", IntVecToStr(newScale).c_str());
    for (size_t j = 0; j < newSize; j++) {
        newRawShape[j] = newShape[j] * newScale[j];
    }
    return true;
}

void GetDynOffsetBeforeReshape(const std::vector<SymbolicScalar> &oriOffset, const std::vector<int64_t> &oriShape,
    const std::vector<int64_t> &newShape, std::vector<SymbolicScalar> &newOffset) {
    // 计算原始shape的步长（stride）
    ASSERT(oriShape.size() == oriOffset.size());
    size_t oriSize = oriOffset.size();
    size_t newSize = newShape.size();
    std::vector<int64_t> oriStride(oriShape.size());
    int64_t currentStride = 1;
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
    std::vector<int64_t> newStride(newSize);
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
Assemble拆分了最高轴，认为可以透传，不需要拷贝，前序在ExpandFunction中做了判断，属性NeedCopy=false
Copy_Out --> tensor(GM) --> Reshape --> oriBackUp [16, 16] --> Assemble(offset, dynOffset) --> OCAST(offset, dynOffset) [16, 64]
因此需要: 重新计算Reshape输入的RawShape, offset, dynOffset
*/
void HandleDynOffsetForReshape(const LogicalTensorPtr &oriBackUp, std::unordered_set<Operation *> &concurrentAssembles,
    const std::set<Operation *, LogicalTensor::CompareOp> &producers) {
    std::vector<SymbolicScalar> newDynOffset;
    std::vector<int64_t> newRawShape;
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
        producer->GetIOperands()[0]->tensor->UpdateRawShape(newRawShape);
    }
}

void PreGraphProcess::HandleForAssembleFromInOut(Function &function, std::unordered_set<Operation *> &concurrentAssembles,
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
void PreGraphProcess::HandleForAssembleToOutcast(Function &function, std::unordered_set<Operation *> &concurrentAssembles,
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

void PreGraphProcess::HandleForReshapeToOutcast(Function &function) const {
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            if (function.IsFromOutCast(op.GetOOperands()[0])) {
                // input --> reshape --> OCAST
                if (op.GetIOperands()[0]->tensor->actualRawmagic != -1) {
                    // 说明输入也来自于reshape，需要找到指向的raw tensor，并更新其actual raw
                    int inputActualRawId = op.GetIOperands()[0]->tensor->actualRawmagic;
                    auto inputRaw = function.GetTensorMap().GetRawTensorByRawMagic(inputActualRawId);
                    inputRaw->actualRawmagic = op.GetOOperands()[0]->GetRawMagic();
                }
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
void PreGraphProcess::DeleteRedundantAssemble(Function &function) const {
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
            if (cons->GetOpcode() != Opcode::OP_ASSEMBLE) {
                cons->iOperand[0] = output;
                cons->iOperand[0]->AddConsumer(cons);
                continue;
            }
            cons->SetAsDeleted();
            if (!concurrentAssembles.empty()) {
                concurrentAssembles.emplace(cons);
                auto consConsumersBackup = cons->GetOOperands().front()->GetConsumers();
                /*
                      /--> op[Assemble]   --> output
                input --> cons[Assemble] --> consOutput --> consOfCons1
                                              \--> consOfCons2
                after:
                                                /--> consOfCons1
                    /--> op[Assemble] --> output --> consOfCons2
                input --> cons[Assemble] --> consOutput
                */
                for (auto &consOfCons : consConsumersBackup) {
                    SubstituteInput(consOfCons, cons->GetOOperands().front(), output);
                }
                continue;
            }
            concurrentAssembles.emplace(cons);
            for (auto &producer : producersBackup) {
                oriOutputBackUp = producer->oOperand[0]; // producer --> oriOutputBackUp(input) --> op
                producer->ReplaceOutput(output, oriOutputBackUp);
                output->isSubGraphBoundary = true;
                if (!IsCopyOut(producer->GetOpcode())) { continue;}
                auto opAttr = std::static_pointer_cast<CopyOpAttribute>(producer->GetOpAttribute());
                auto consOpAttr = std::static_pointer_cast<AssembleOpAttribute>(cons->GetOpAttribute());
                if (consOpAttr->GetToDynOffset().size() != 0) {
                    opAttr->SetToOffset(OpImmediate::Specified(consOpAttr->GetToDynOffset()));
                }
                opAttr->SetRawShape(OpImmediate::Specified(output->tensor->GetRawShape()));
            }
        }
        HandleForAssembleFromInOut(function, concurrentAssembles, producersBackup);
        HandleForAssembleToOutcast(function, concurrentAssembles, producersBackup);
        HandleDynOffsetForReshape(oriOutputBackUp, concurrentAssembles, producersBackup); // op为Assemble
    }
    function.EraseOperations(false);
    HandleForReshapeToOutcast(function);
}

void PreGraphProcess::ProcessSpecialMTEOperation(Operation &op) const {
    ALOG_DEBUG_F("Process Special MTE Operation %d", op.opmagic);
    auto inputTensor = op.iOperand.front();
    auto outputTensor = op.oOperand.front();
    if ((inputTensor == nullptr) || (outputTensor == nullptr)) {
        return;
    }
    /* transpose datamove 输入和输出的shape不相同 */
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MemoryType::MEM_UB,
        OpImmediate::Specified(outputTensor->GetTensorOffset()), OpImmediate::Specified(outputTensor->GetShape()),
        OpImmediate::Specified(outputTensor->tensor->GetDynRawShape())));
    op.oOperand[0]->isSubGraphBoundary = true;
}

void PreGraphProcess::ProcessMoveInOperation(Operation &op) const {
    ALOG_DEBUG_F("Process MoveIn Operation %d", op.opmagic);
    auto inputTensor = op.iOperand.front();
    if (inputTensor == nullptr) {
        return;
    }
    TensorOffset offset = inputTensor->GetTensorOffset();
    auto producers = inputTensor->GetProducers();
    if (!producers.empty()) {
        auto pre = *(producers.begin());
        if (pre != nullptr && pre->GetOpcode() == Opcode::OP_VIEW) {
            auto attr = dynamic_cast<ViewOpAttribute *>(pre->GetOpAttribute().get());
            if (attr != nullptr) {
                op.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                    OpImmediate::Specified(TensorOffset(attr->GetFromOffset(), attr->GetFromDynOffset())),
                    MemoryType::MEM_UB, OpImmediate::Specified(inputTensor->GetShape()),
                    OpImmediate::Specified(inputTensor->tensor->GetDynRawShape()),
                    OpImmediate::Specified(inputTensor->GetDynValidShape())));
                op.iOperand[0]->isSubGraphBoundary = true;
                return;
            }
        }
    }
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified(offset),
        MemoryType::MEM_UB, OpImmediate::Specified(inputTensor->GetShape()),
        OpImmediate::Specified(inputTensor->tensor->GetDynRawShape()),
        OpImmediate::Specified(inputTensor->GetDynValidShape())));
    op.iOperand[0]->isSubGraphBoundary = true;
}

void PreGraphProcess::InsertTemporaryCopyIn(Function &function, Operation &op) const {
    if (std::find(DISTRIBUTED_OPS.begin(), DISTRIBUTED_OPS.end(), op.GetOpcode()) != DISTRIBUTED_OPS.end()) {
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

void PreGraphProcess::ProcessInplaceOp(Function &function) const {
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

void PreGraphProcess::ProcessSameInOutOp(Function &function) const {
    for (auto &op : function.Operations()) {
        Opcode prod;
        if (!op.GetAttr(OpAttributeKey::sameInOut, prod)) {
            continue;
        }
        for (auto &input : op.GetIOperands()) {
            for (auto &producer : input->GetProducers()) {
                if (producer->GetOpcode() != prod) {
                    continue;
                }
                auto output = op.GetOOperands().front();
                if (input->shape != output->shape) {
                    ALOG_INFO_F("Op[%d] input output tensor shape is not equal, cannot reuse buffer.", op.GetOpMagic());
                    continue;
                }
                if (function.IsFromInCast(input)) {
                    ALOG_WARN_F(
                        "PreGraphProcess::ProcessSameInOutOp: OP iOperand tensor[%d] is inCast.", input->GetMagic());
                    continue;
                }
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

void PreGraphProcess::UpdateCopyOpIsCube(Operation &op) const {
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

void PreGraphProcess::InitializeTensorMemorymap(Operation &op) const {
    const int newColor = op.GetSubgraphID();
    for (auto &input : op.GetIOperands()) {
        TileRange range;
        range.memId = input->tensor->GetRawMagic();
        input->memorymap.insert(std::make_pair(newColor, range));
        if (input->GetProducers().size() == 0) {
            input->subGraphID = newColor;
        }
    }
    for (auto &output : op.GetOOperands()) {
        TileRange range;
        range.memId = output->tensor->GetRawMagic();
        output->memorymap.insert(std::make_pair(newColor, range));
        output->subGraphID = newColor;
    }
}

void PreGraphProcess::SetTensorBoundary(Function &function) const {
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
            continue;
        }
        if (op.GetOpcode() == Opcode::OP_COPY_OUT) {
            /* Copy Out 的输出*/
            op.GetOOperands().front()->isSubGraphBoundary = true;
            continue;
        }
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            /* GM上的Assemble*/
            auto assembleIn = op.GetIOperands().front();
            auto assembleOut = op.GetOOperands().front();
            bool isBoundary = (assembleOut->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR);
            assembleOut->isSubGraphBoundary = isBoundary;
            assembleIn->isSubGraphBoundary = isBoundary;
            continue;
        }
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            /* reshape*/
            auto reshapeIn = op.GetIOperands().front();
            auto reshapeOut = op.GetOOperands().front();
            bool isBoundary = (reshapeOut->isSubGraphBoundary || reshapeIn->isSubGraphBoundary);
            reshapeIn->isSubGraphBoundary = isBoundary;
            reshapeOut->isSubGraphBoundary = isBoundary;
        }
    }
}

Status DFSVisit(std::unordered_set<int> &visited, int preColor, std::unordered_map<int, int> &newColorMap,
                std::vector<std::set<int>> &colorInGraph, std::vector<std::set<int>> &colorOutGraph)
{
    std::vector<int> visitStack{preColor};
    std::unordered_set<int> inStack;
    while (visitStack.size() > 0) {
        int currColor = visitStack.back();
        if (visited.count(currColor) > 0) {
            visitStack.pop_back();
            continue;
        }
        bool allVisited = true;
        for (int pred : colorInGraph[currColor]) {
            if (visited.count(pred) == 0) {
                visitStack.push_back(pred);
                inStack.insert(pred);
                allVisited = false;
            }
        }
        if (!allVisited) {
            continue;
        }
        int currColorNum = newColorMap.size();
        newColorMap[currColor] = currColorNum;
        visited.insert(currColor);
        visitStack.pop_back();
        for (int succ : colorOutGraph[currColor]) {
            if (visited.count(succ) == 0 && inStack.count(succ) == 0) {
                visitStack.push_back(succ);
                inStack.insert(succ);
            }
        }
    }
    return SUCCESS;
}

Status PreGraphProcess::PreColorSort(Function &function)
{
    int colorNum = function.GetTotalSubGraphCount();
    std::vector<std::set<int>> colorInGraph(colorNum);
    std::vector<std::set<int>> colorOutGraph(colorNum);
    for (auto &op : function.Operations()) {
        int opColor = op.GetSubgraphID();
        for (auto &consumer : op.ConsumerOps()) {
            int consumerColor = consumer->GetSubgraphID();
            if (opColor != consumerColor) {
                colorInGraph[consumerColor].insert(opColor);
                colorOutGraph[opColor].insert(consumerColor);
            }
        }
    }
    std::unordered_map<int, int> newColorMap;
    std::unordered_set<int> visited;
    for (int preColor = 0; preColor < colorNum; preColor++) {
        if (visited.count(preColor) > 0) {
            continue;
        }
        DFSVisit(visited, preColor, newColorMap, colorInGraph, colorOutGraph);
    }
    for (auto &op : function.Operations()) {
        int opColor = op.GetSubgraphID();
        op.UpdateSubgraphID(newColorMap[opColor]);
    }
    return SUCCESS;
}

Status PreGraphProcess::RunOnFunction(Function &function) {
    ALOG_INFO_F("===> start PreGraph");
    PreColorSort(function);
    ResetMemoryMap(function);
    auto opList = function.Operations();
    for (auto &op : opList) {
        if (op.GetSubgraphID() <= -1) {
            continue;
        }
        auto curColor = op.GetSubgraphID();
        InitializeTensorMemorymap(op);
        op.UpdateSubgraphID(curColor);
        UpdateCopyOpIsCube(op);
    }
    SetTensorBoundary(function);
    // Processing Special Ops
    for (auto &op : opList) {
        if (IsCopyOut(op.GetOpcode()) && op.GetOpcode() != Opcode::OP_COPY_OUT) {
            ProcessSpecialMTEOperation(op);
        }
        if (IsCopyIn(op.GetOpcode()) && op.GetOpcode() != Opcode::OP_COPY_IN) {
            ProcessMoveInOperation(op);
        }
        if ((op.GetOpcode() == Opcode::OP_ASSEMBLE) || (op.GetOpcode() == Opcode::OP_RESHAPE)) {
            // 校验单输入单输出，且输入输出mem类型相同
            if ((op.GetIOperands().size() != 1) || (op.GetOOperands().size() != 1) ||
                (op.GetIOperands().front() == nullptr) || (op.GetOOperands().front() == nullptr) ||
                (op.GetIOperands().front()->GetMemoryTypeOriginal() !=
                    op.GetOOperands().front()->GetMemoryTypeOriginal())) {
                ALOG_ERROR_F("Invalid Op %s[%d], please check.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
                return FAILED;
            }
        }
    }
    ProcessSameInOutOp(function);
    DeleteRedundantAssemble(function);
    ALOG_INFO_F("===> End PreGraph");
    return SUCCESS;
}

Status PreGraphProcess::PreCheck(Function &function) {
    PreGraphProcessChecker checker;
    return checker.DoPreCheck(function);
}

Status PreGraphProcess::PostCheck(Function &function) {
    PreGraphProcessChecker checker;
    return checker.DoPostCheck(function);
}

} // namespace npu::tile_fwk
