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
#include "passes/pass_utils/graph_utils.h"

namespace npu::tile_fwk {

std::vector<OpImmediate> SumOffset(const std::vector<OpImmediate> offset1, const std::vector<OpImmediate> offset2) {
    std::vector<OpImmediate> res;
    for (size_t i = 0; i < offset1.size(); i++) {
        res.push_back(offset1[i] + offset2[i]);
    }
    return res;
}

// 当前op为Copy Out时，需要将后继Assemble上的offset累加到当前op的CopyOpAttr上
void UpdateCopyOutAttr(Operation *op, Operation *opNext) {
    auto opAttr = std::static_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
    auto opNextAttr = std::static_pointer_cast<AssembleOpAttribute>(opNext->GetOpAttribute());
    if (opNextAttr->GetToDynOffset().size() != 0) {
        if (op->GetOpcode() != Opcode::OP_COPY_OUT) {
            opAttr->SetToOffset(OpImmediate::Specified(opNextAttr->GetToDynOffset()));
        } else {
            opAttr->SetToOffset(SumOffset(OpImmediate::Specified(opNextAttr->GetToDynOffset()), opAttr->GetToOffset()));
        }
    }
    opAttr->SetRawShape(OpImmediate::Specified(op->GetOOperands().front()->tensor->GetDynRawShape()));
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

/*
resetDdr: A_MUL_B的清零后DDR输入，数据边表依赖
copyOutOp: 当前A_MUL_B链路最终搬出的L0C_Copy_Out
功能: 当前Matmul链路最终的CopyOut属性需要与清零时的CopyOut属性对齐
限制条件: 依赖前端使能切K场景下，Matmul的tile展开中显示对清零后的Gm按C矩阵的切分大小做切分
*/
void AlignCopyOutAttr(LogicalTensorPtr &resetDdr, Operation *copyOutOp) {
    if (resetDdr->GetProducers().size() == 1) {
        auto ddrResetCopyOut = *resetDdr->GetProducers().begin();
        if (ddrResetCopyOut->GetOpcode() != Opcode::OP_COPY_OUT) {
            ALOG_ERROR_F("DDR reset Op requires to be OP_COPY_OUT, but %s[%d].", 
                ddrResetCopyOut->GetOpcodeStr().c_str(), ddrResetCopyOut->GetOpMagic());
            return;
        }
        auto ddrResetCopyOutAttr = std::static_pointer_cast<CopyOpAttribute>(ddrResetCopyOut->GetOpAttribute());
        auto L0CCopyOutAttr = std::static_pointer_cast<CopyOpAttribute>(copyOutOp->GetOpAttribute());
        ddrResetCopyOutAttr->SetRawShape(L0CCopyOutAttr->GetRawShape());
        ddrResetCopyOutAttr->SetToOffset(L0CCopyOutAttr->GetToOffset());
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

bool PreGraphProcess::IsCandidateAssembleOp(Function &function, Operation &op) const {
    if (op.GetOpcode() != Opcode::OP_ASSEMBLE) {
        return false;
    }
    auto &output = op.GetOOperands().front();
    if (output->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR || op.IsDeleted()) {
        return false;
    }
    for (auto &prod : function.FindProducers(op)) {
        if (prod->GetOpcode() != Opcode::OP_VIEW) {
            return true;
        }
    }
    return false;
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
        if (!IsCandidateAssembleOp(function, op)) {
            continue;
        }
        auto &input = op.GetIOperands().front();
        auto &output = op.GetOOperands().front();
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
                if (!IsCopyOut(producer->GetOpcode())) { 
                    continue;
                }
                UpdateCopyOutAttr(producer, cons);
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
    if (!op.HasStaticAttribute(OpAttributeKey::requiresBoundaryCopy)) {
          return;
    }
    for (auto &input : op.GetIOperands()) {
        if (input->GetProducers().size() == 0 && input->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
            // insert Copy_In before the op
            input->isSubGraphBoundary = false;
            LogicalTensors operandGm;
            LogicalTensorPtr tensorGM = std::make_shared<LogicalTensor>(function, input->Datatype(), input->shape);
            GraphUtils::CopyDynStatus(tensorGM, input);
            tensorGM->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
            tensorGM->SetMemoryTypeToBe(MemoryType::MEM_DEVICE_DDR);
            tensorGM->isSubGraphBoundary = true;
            tensorGM->subGraphID = op.GetSubgraphID();
            operandGm.push_back(tensorGM);
            function.GetTensorMap().Insert(tensorGM);

            LogicalTensors operandUb;
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

void PreGraphProcess::InitializeTensorColor(Operation &op) const {
    const int newColor = op.GetSubgraphID();
    for (auto &input : op.GetIOperands()) {
        if (input->GetProducers().size() == 0) {
            input->subGraphID = newColor;
        }
    }
    for (auto &output : op.GetOOperands()) {
        TileRange range;
        range.memId = output->tensor->GetRawMagic();
        output->subGraphID = newColor;
    }
}

bool IsDiffSubgraphId(int &oriSubgraphId, Operation *op) {
    int opSubgraphId = op->GetSubgraphID();
    if (oriSubgraphId == -1) {
        oriSubgraphId = opSubgraphId;
    }
    if (oriSubgraphId != opSubgraphId) {
        return true;
    }
    return false;
}

bool IsTensorSubgraphBoundary(LogicalTensorPtr t) {
    int subgraphId = -1;
    for (auto &op : t->GetProducers()) {
        if (IsDiffSubgraphId(subgraphId, op) == true) {
            return true;
        }
    }
    for (auto &op : t->GetConsumers()) {
        if (IsDiffSubgraphId(subgraphId, op) == true) {
            return true;
        }
    }

    return false;
}

void PreGraphProcess::SetTensorBoundary(Function &function) const {
    for (auto &op : function.Operations()) {
        /* memory map size > 1 代表该tensor被多个子图使用，那么标记为boundary*/
        for (auto &input : op.GetIOperands()) {
            if (IsTensorSubgraphBoundary(input)) {
                input->isSubGraphBoundary = true;
            }
            if (input->GetProducers().size() == 0) {
                input->isSubGraphBoundary = true;
                InsertTemporaryCopyIn(function, op);
            }
        }
        for (auto &output : op.GetOOperands()) {
            if (IsTensorSubgraphBoundary(output)) {
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

bool PreGraphProcess::IsFloat(const std::shared_ptr<LogicalTensor> tensor) const {
    auto dataType = tensor->Datatype();
    if ((dataType == DT_FP16) || (dataType == DT_FP32) || (dataType == DT_BF16)) {
        return true;
    }
    return false;
}

bool PreGraphProcess::IsInt(const std::shared_ptr<LogicalTensor> tensor) const {
    auto dataType = tensor->Datatype();
    if ((dataType == DT_INT8) || (dataType == DT_INT16) || (dataType == DT_INT32)) {
        return true;
    }
    return false;
}

Status PreGraphProcess::AddL1CopyInAttr(
    const std::shared_ptr<LogicalTensor> input, int nzValue, int mValue, int kValue, int nValue) const {
    auto copyInOp = *(input->GetProducers().begin());
    auto tensorL0 = copyInOp->GetIOperands().front();
    auto L1CopyInOp = *(tensorL0->GetProducers().begin());
    if (L1CopyInOp->GetOpcode() == Opcode::OP_VIEW || L1CopyInOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
        /*
        1. View 对应大包搬运场景
        gm -> L1_COPY_IN -> L1 ---> View ---> L1_partial ---> L1_TO_L0A ---> L0 ---> A_MUL_B
                              \ ---> View ---> L1_partial ---> L1_TO_L0A ---> L0 ---> A_MUL_B
        2. Assemble 对应 Gather On L1 场景
        */
        tensorL0 = L1CopyInOp->GetIOperands().front();
        L1CopyInOp = *(tensorL0->GetProducers().begin());
    }
    if (L1CopyInOp->GetOpcode() != Opcode::OP_COPY_IN) {
        ALOG_DEBUG_F("L0 tesnor[%d] has invalid corresponding L1CopyInOp, please check.", input->magic);
        return FAILED;
    }
    L1CopyInOp->SetAttribute(COPY_IS_NZ, nzValue);
    ALOG_DEBUG_F("Update %s[%d] attr is_Nz: %d", L1CopyInOp->GetOpcodeStr().c_str(), L1CopyInOp->GetOpMagic(), nzValue);
    if (copyInOp->GetOpcode() == Opcode::OP_L1_TO_L0A) {
        L1CopyInOp->SetAttribute(L1_COPY_IN_OUTER, mValue);
        L1CopyInOp->SetAttribute(L1_COPY_IN_INNER, kValue);
        ALOG_DEBUG_F("OP_L1_TO_L0A: Outer: %d, Inner: %d", mValue, kValue);
        return SUCCESS;
    }
    if (copyInOp->GetOpcode() == Opcode::OP_L1_TO_L0B) {
        L1CopyInOp->SetAttribute(L1_COPY_IN_OUTER, kValue);
        L1CopyInOp->SetAttribute(L1_COPY_IN_INNER, nValue);
        ALOG_DEBUG_F("OP_L1_TO_L0B: Outer: %d, Inner: %d", kValue, nValue);
        return SUCCESS;
    }
    if (copyInOp->GetOpcode() == Opcode::OP_L1_TO_L0_AT) {
        L1CopyInOp->SetAttribute(L1_COPY_IN_OUTER, kValue);
        L1CopyInOp->SetAttribute(L1_COPY_IN_INNER, mValue);
        ALOG_DEBUG_F("OP_L1_TO_L0_AT: Outer: %d, Inner: %d", kValue, mValue);
        return SUCCESS;
    }
    if (copyInOp->GetOpcode() == Opcode::OP_L1_TO_L0_BT) {
        L1CopyInOp->SetAttribute(L1_COPY_IN_OUTER, nValue);
        L1CopyInOp->SetAttribute(L1_COPY_IN_INNER, kValue);
        ALOG_DEBUG_F("OP_L1_TO_L0_BT: Outer: %d, Inner: %d", nValue, kValue);
        return SUCCESS;
    }
    ALOG_DEBUG_F("invalid Cube input %d, produced by %s[%d]", input->GetMagic(), copyInOp->GetOpcodeStr().c_str(),
        copyInOp->GetOpMagic());
    return FAILED;
}

Status PreGraphProcess::AddL0cCopyOutAttr(const std::shared_ptr<LogicalTensor> output, int nzValue, int mValue, int nValue) const {
    for (auto &childOp : output->GetConsumers()) {
        if (childOp->GetOpcode() != Opcode::OP_COPY_OUT) {
            continue;
        }
        childOp->SetAttribute(COPY_IS_NZ, nzValue);
        childOp->SetAttribute(L0C_COPY_OUT_OUTER, mValue);
        childOp->SetAttribute(L0C_COPY_OUT_INNER, nValue);
        ALOG_DEBUG_F("Update %s[%d] attr is_Nz: %d, curH: %d, curW: %d",
            childOp->GetOpcodeStr().c_str(), childOp->GetOpMagic(), nzValue, mValue, nValue);
    }
    return SUCCESS;
}

Status PreGraphProcess::UpdateCopyAttr(Operation &op) const {
    int32_t nzAttr = op.GetIntAttribute(MATMUL_NZ_ATTR);
    auto mValue = (op.HasAttr(A_MUL_B_ACT_M)) ? op.GetIntAttribute(A_MUL_B_ACT_M) : 0;
    auto kValue = (op.HasAttr(A_MUL_B_ACT_K)) ? op.GetIntAttribute(A_MUL_B_ACT_K) : 0;
    auto nValue = (op.HasAttr(A_MUL_B_ACT_N)) ? op.GetIntAttribute(A_MUL_B_ACT_N) : 0;

    int aIsNz = nzAttr % 2;
    int bIsNz = (nzAttr >> 1) % 2;
    int cIsNz = (nzAttr >> 2) % 2;
    ALOG_DEBUG_F("Retrive %s[%d] attr done, aIsNz: %d, bIsNz: %d, cIsNz: %d, mValue: %d, kValue: %d, nValue: %d",
            op.GetOpcodeStr().c_str(), op.GetOpMagic(), aIsNz, bIsNz, cIsNz, mValue, kValue, nValue);
    for (auto &input : op.GetIOperands()) {
        if (input->GetMemoryTypeOriginal() == MemoryType::MEM_L0A) {
            if (AddL1CopyInAttr(input, aIsNz, mValue, kValue, nValue) != SUCCESS) {
                ALOG_ERROR_F("Set Attr for matrix A L1_COPY_IN of %s[%d] failed.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
                return FAILED;
            }
            continue;
        }
        if (input->GetMemoryTypeOriginal() == MemoryType::MEM_L0B) {
            if (AddL1CopyInAttr(input, bIsNz, mValue, kValue, nValue) != SUCCESS) {
                ALOG_ERROR_F("Set Attr for matrix B L1_COPY_IN of %s[%d] failed.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
                return FAILED;
            }
        }
    }
    for (auto &output : op.GetOOperands()) {
        if (AddL0cCopyOutAttr(output, cIsNz, mValue, nValue) != SUCCESS) {
            ALOG_ERROR_F("Set Attr for L0C_COPY_OUT of %s[%d] failed.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status PreGraphProcess::CheckValidCube(const Operation &op) {
    /* 校验有且只有一个输出 */
    if (op.GetOOperands().size() != 1) {
        ALOG_ERROR_F("%s[%d] has output num != 1.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
        return FAILED;
    }
    /* 校验输出: 1. 非空，2. mem类型为L0C, 3.有消费者 */
    auto outputL0C = op.GetOOperands().front();
    if (outputL0C == nullptr) {
        ALOG_ERROR_F("%s[%d] output is nullptr.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
        return FAILED;
    }
    if (outputL0C->GetMemoryTypeOriginal() != MemoryType::MEM_L0C) {
        ALOG_ERROR_F("%s[%d] output is NOT L0C.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
        return FAILED;
    }
    if (outputL0C->GetConsumers().size() < 1) {
        ALOG_ERROR_F("%s[%d] output has EMPTY consumers.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
        return FAILED;
    }
    return SUCCESS;
}

Status PreGraphProcess::UpdateL0cDtype(Operation &op) {
    std::pair<DataType, DataType> inputDtypes = std::make_pair(DataType::DT_FP16, DataType::DT_FP16);
    for (auto &input : op.GetIOperands()) {
        if (input->GetMemoryTypeOriginal() == MemoryType::MEM_L0A) {
            inputDtypes.first = input->Datatype();
        } else if (input->GetMemoryTypeOriginal() == MemoryType::MEM_L0B) {
            inputDtypes.second = input->Datatype();
        }
    }
    if (supportDtypeMap.count(inputDtypes)) {
        DataType outDtype = supportDtypeMap.at(inputDtypes);
        for (auto &output: op.GetOOperands()) {
            output->tensor->datatype = outDtype;
        }
        return SUCCESS;
    } else {
        ALOG_ERROR_F("%s[%d] has unsupport input dtypes (L0A: %s, L0B: %s), update L0C dtype Failed.",
            op.GetOpcodeStr().c_str(), op.GetOpMagic(), 
            BriefDataType2String(inputDtypes.first).c_str(),
            BriefDataType2String(inputDtypes.second).c_str());
        return FAILED;
    }
}

std::pair<Operation *, Operation *> PreGraphProcess::GetLastMmCopyOut(Operation &op) {
    auto outputL0C = op.GetOOperands().front();
    auto chainEndCopyOut = *(outputL0C->GetConsumers().begin());
    size_t depth_ = 0;
    // recursively find: MatMul -> L0C -> Copy_Out -> Gm
    while (chainEndCopyOut->GetOpcode() != Opcode::OP_COPY_OUT) {
        outputL0C = chainEndCopyOut->GetOOperands().front();
        chainEndCopyOut = *(outputL0C->GetConsumers().begin());
        depth_ += 1;
    }
    if (chainEndCopyOut == nullptr) {
        ALOG_ERROR_F("%s[%d] has nullptr L0C_Copy_Out.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
        return {nullptr, nullptr};
    }
    if (chainEndCopyOut->GetOOperands().size() != 1) {
        ALOG_ERROR_F("%s[%d] has more than ONE outputs.", chainEndCopyOut->GetOpcodeStr().c_str(), chainEndCopyOut->GetOpMagic());
        return {nullptr, nullptr};
    }
    auto finalOutput = chainEndCopyOut->GetOOperands().front();
    if (finalOutput->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
        ALOG_ERROR_F("%s[%d] has invlid output memType: %s, expect: MEM_DEVICE_DDR.",
            chainEndCopyOut->GetOpcodeStr().c_str(), chainEndCopyOut->GetOpMagic(),
            MemoryTypeToString(finalOutput->GetMemoryTypeOriginal()).c_str());
        return {nullptr, nullptr};
    }
    // Copy_Out 的上游Op即为最后一个Matmul
    auto lastMm = *chainEndCopyOut->ProducerOps().begin();
    return {lastMm, chainEndCopyOut};
}

Status PreGraphProcess::UpdateCubeOp(Function &function) {
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_A_MUL_B && op.GetOpcode() != Opcode::OP_A_MULACC_B) {
            continue;
        }
        if (CheckValidCube(op) != SUCCESS) {
            ALOG_ERROR_F("%s[%d] is invalid.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
            return FAILED;
        }
        auto lastMmCopyOut = GetLastMmCopyOut(op);
        auto lastMm = lastMmCopyOut.first;
        auto chainEndCopyOut = lastMmCopyOut.second;
        if (lastMm == nullptr || chainEndCopyOut == nullptr) {
            ALOG_ERROR_F("Get the last MatMul and L0C_Copy_Out for %s[%d] failed.", 
                op.GetOpcodeStr().c_str(), op.GetOpMagic());
            return FAILED;
        }
        for (auto &input : op.GetIOperands()) {
            if (input->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            // Align copy out GM with the reset GM
            if (function.IsFromInCast(input)) {
                ALOG_WARN_F("PreGraphProcess::UpdateCubeOp: OP_A_MUL_B iOperand tensor[%d] is incast.", input->GetMagic());
                continue;
            }
            auto finalOutput = lastMm->GetOOperands().front();
            input->tensor = finalOutput->tensor;
            /*
            强制要求当前Matmul链路输出的Gm仅存在一个清零的Op，暂时通过指定清零和ReduceAcc使用的vec tilesize与tileM x tileN相同
            后续通过前端提供使能切K的API保证
            */
            AlignCopyOutAttr(input, chainEndCopyOut);
        }
        if (UpdateL0cDtype(op) != SUCCESS) {
            ALOG_ERROR_F("Update L0C dtype for %s[%d] failed.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
            return FAILED;
        }
        if (UpdateCopyAttr(op) != SUCCESS) {
            ALOG_ERROR_F("Set Attr for %s[%d] failed.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status PreGraphProcess::RunOnFunction(Function &function) {
    ALOG_INFO_F("===> start PreGraph");
    PreColorSort(function);
    auto opList = function.Operations();
    for (auto &op : opList) {
        InitializeTensorColor(op);
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
    }
    ProcessSameInOutOp(function);
    DeleteRedundantAssemble(function);
    if (UpdateCubeOp(function) != SUCCESS) {
        ALOG_ERROR_F("Update Cube attr failed.");
        return FAILED;
    }
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
