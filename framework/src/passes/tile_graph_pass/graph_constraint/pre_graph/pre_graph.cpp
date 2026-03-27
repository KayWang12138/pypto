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
 * \file pre_graph.cpp
 * \brief
 */

#include "pre_graph.h"
#include "passes/pass_check/pre_graph_checker.h"
#include "passes/pass_utils/merge_view_assemble_utils.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "PreGraphProcess"

namespace npu::tile_fwk {
void PreGraphProcess::UpdateCopyOpIsCube(Operation &op) const {
    /*
    后续考虑移到InsertCopyOp
    copy_out for producer
    op(copy_in) --> input --> consumerOp(isCube?)
    */
    if (IsCopyIn(op.GetOpcode())) {
        for (const auto &consumerOps : op.ConsumerOps()) {
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
        for (const auto &producerOps : op.ProducerOps()) {
            if ((producerOps->HasAttr(OpAttributeKey::isCube)) &&
                (producerOps->GetSubgraphID() == op.GetSubgraphID())) {
                op.SetAttribute(OpAttributeKey::isCube, producerOps->GetBoolAttribute(OpAttributeKey::isCube));
                break;
            }
        }
    }
}

/**
 * @brief 判断 UB 上的tensor尾轴是否32B对齐
 */
inline bool IsLastDim32BAligned(const LogicalTensorPtr& tensor) {
    // 空shape视为非32B对齐
    if (tensor->shape.empty()) {
        return false;
    }

    size_t lastIdx = tensor->shape.size() - 1;
    size_t lastDim = tensor->shape[lastIdx];
    size_t bytes = BytesOf(tensor->Datatype());
    size_t totalByte = lastDim * bytes;

    // 判断是否32字节对齐
    return (totalByte % 32) == 0;
}

inline size_t GetPaddingValue(LogicalTensorPtr &in) {
    auto bytes = BytesOf(in->Datatype());
    auto paddingIter = BLOCK_PADDING_DIM.find(bytes);
    if (paddingIter == BLOCK_PADDING_DIM.end()) {
        return 1;
    }
    return paddingIter->second;
}

/**
 * @brief 为 UB 上尾轴非32B对齐的tensor做32B对齐操作
 */
inline int64_t Pad(int64_t dim, int64_t padValue) {
    if (padValue == 0) {
        return dim;
    }
    return (dim + padValue - 1) / padValue * padValue;
}

/**
 * @brief 为 UB 内存类型的输入插入拷贝序列 (UB → DDR → UB)
 */
void PreGraphProcess::InsertCopyUBOp(Function &function, Operation *needInsertCopyAssOp, LogicalTensorPtr &input) {
    auto copyShape = input->GetShape();
    auto copyRawShape = input->tensor->GetDynRawShape();
    auto copyDynShape = input->GetDynValidShape();
    Offset offset(copyShape.size(), 0);

    LogicalTensor copyOutOutput(function, input->Datatype(), copyShape);
    copyOutOutput.SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto copyOutOutputPtr = std::make_shared<LogicalTensor>(std::move(copyOutOutput));
    auto &copyOutOp = function.AddOperation(Opcode::OP_COPY_OUT, {input}, {copyOutOutputPtr});

    copyOutOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        input->GetMemoryTypeOriginal(),
        OpImmediate::Specified(offset),
        OpImmediate::Specified(copyShape),
        OpImmediate::Specified(copyRawShape),
        OpImmediate::Specified(copyDynShape)
    ));
    copyOutOp.UpdateSubgraphID(needInsertCopyAssOp->GetSubgraphID());

    LogicalTensor copyInOutput(function, input->Datatype(), copyShape);
    copyInOutput.SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto copyInOutputPtr = std::make_shared<LogicalTensor>(std::move(copyInOutput));
    auto &copyInOp = function.AddOperation(Opcode::OP_COPY_IN, {copyOutOutputPtr}, {copyInOutputPtr});
    copyInOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(offset),
        input->GetMemoryTypeOriginal(),
        OpImmediate::Specified(copyShape),
        OpImmediate::Specified(copyRawShape),
        OpImmediate::Specified(copyDynShape)
    ));
    copyInOp.UpdateSubgraphID(needInsertCopyAssOp->GetSubgraphID());

    needInsertCopyAssOp->ReplaceInput(copyInOutputPtr, input);
}

/**
 * @brief 为 DDR 内存类型的输入插入拷贝序列 (DDR → UB → DDR)
 */
void PreGraphProcess::InsertCopyDDROp(Function &function, Operation *needInsertCopyAssOp, LogicalTensorPtr &input) {
    auto copyShape = input->GetShape();
    auto copyRawShape = input->tensor->GetDynRawShape();
    auto copyDynShape = input->GetDynValidShape();
    Offset offset(copyShape.size(), 0);

    LogicalTensor copyInOutput(function, input->Datatype(), copyShape);
    copyInOutput.SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    const int UB_SIZE_THRESHOLD = static_cast<int>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB));
    auto memType = copyInOutput.GetMemoryTypeOriginal();
    if ((memType == MemoryType::MEM_UB) && (copyInOutput.GetDataSize() > UB_SIZE_THRESHOLD)) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Tensor %d exceeds the UB size limit.", copyInOutput.magic);
        return;
    }
    auto copyInOutputPtr = std::make_shared<LogicalTensor>(std::move(copyInOutput));
    if (memType == MemoryType::MEM_UB && !IsLastDim32BAligned(copyInOutputPtr)) {
        size_t lastIdx = copyInOutputPtr->shape.size() - 1;
        size_t paddingValue = GetPaddingValue(copyInOutputPtr); // 根据数据类型，判断需要pad到几个元素
        
        // 保存rawshape
        copyInOutputPtr->oriShape = copyInOutputPtr->shape;
        copyInOutputPtr->tensor->oriRawshape = copyInOutputPtr->tensor->rawshape;

        // pad 32B
        copyInOutputPtr->shape[lastIdx] = Pad(copyInOutputPtr->shape[lastIdx], paddingValue);
        copyInOutputPtr->tensor->rawshape[lastIdx] = Pad(copyInOutputPtr->tensor->oriRawshape[lastIdx], copyInOutputPtr->shape[lastIdx]);
    }
    auto &copyInOp = function.AddOperation(Opcode::OP_COPY_IN, {input}, {copyInOutputPtr});
    copyInOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(input->GetOffset()),
        MemoryType::MEM_UB,
        OpImmediate::Specified(copyShape),
        OpImmediate::Specified(copyRawShape),
        OpImmediate::Specified(copyDynShape)
    ));
    copyInOp.UpdateSubgraphID(needInsertCopyAssOp->GetSubgraphID());

    LogicalTensor copyOutOutput(function, input->Datatype(), copyShape);
    copyOutOutput.SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto copyOutOutputPtr = std::make_shared<LogicalTensor>(std::move(copyOutOutput));
    auto &copyOutOp = function.AddOperation(Opcode::OP_COPY_OUT, {copyInOutputPtr}, {copyOutOutputPtr});
    copyOutOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        MemoryType::MEM_UB,
        OpImmediate::Specified(offset),
        OpImmediate::Specified(copyShape),
        OpImmediate::Specified(copyRawShape),
        OpImmediate::Specified(copyDynShape)
    ));
    copyOutOp.UpdateSubgraphID(needInsertCopyAssOp->GetSubgraphID());

    needInsertCopyAssOp->ReplaceInput(copyOutOutputPtr, input);
}

/**
 * @brief 递归查找需要插入拷贝的 ASSEMBLE 操作
 */
void PreGraphProcess::FindNeedToCopyAssemble(std::unordered_set<Operation*> &needInsertCopyAssOps, std::unordered_set<int> &visitedAssOps, Operation &op) {
    visitedAssOps.insert(op.GetOpMagic());
    auto assembleIn = op.GetIOperands()[0];
    auto producers = assembleIn->GetProducers();
    if ((!producers.empty()) && (*producers.begin())->GetOpcode() == Opcode::OP_TRANSPOSE_MOVEOUT) {
        return;
    }
    auto consumers = assembleIn->GetConsumers();
    bool sameAssembleOut = true;
    for (const auto &con : consumers) {
        if (con->GetOOperands()[0]->GetMagic() != op.GetOOperands()[0]->GetMagic()) {
            sameAssembleOut = false;
            break;
        }
    }
    if (!sameAssembleOut) {
        for (const auto &con : consumers) {
            if (con->GetOpMagic() != op.GetOpMagic() && con->GetOpcode() == Opcode::OP_ASSEMBLE) {
                visitedAssOps.insert(con->GetOpMagic());
                needInsertCopyAssOps.insert(con);
            }
        }
    }
}

/**
 * @brief 遍历所有 ASSEMBLE 操作，为需要拷贝的操作插入拷贝序列，避免多个 ASSEMBLE 操作共享同一个输入导致的内存冲突
 * Tensor1 ---> Assemble ---> Tensor2
 *         ---> Assemble ---> Tensor3
 *         ---> Assemble ---> Tensor4

 * Tensor1 ---> Reshape ---> Assemble ---> Tensor2

 * Tensor1 ---> View ---> Reshape ---> OP(非CopyOut) ---> Tensor2
 */
void PreGraphProcess::InsertNeedCopy(Function &function) {
    std::unordered_set<int> visitedAssOps;
    std::unordered_set<Operation *> needInsertCopyAssOps;
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE && (!visitedAssOps.count(op.GetOpMagic()))) {
            FindNeedToCopyAssemble(needInsertCopyAssOps, visitedAssOps, op);
        }
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            auto producerOps = op.ProducerOps();
            auto consumerOps = op.ConsumerOps();
            bool flag = true;
            for (auto consumerOp : consumerOps) {
                if (consumerOp->GetOpcode() == Opcode::OP_COPY_OUT) {
                    flag = false;
                    break;
                }
            }
            for (auto producesOp : producerOps) {
                if (producesOp->GetOpcode() == Opcode::OP_VIEW && flag) {
                    needInsertCopyAssOps.insert(&op);
                }
            }
            for (auto consumerOp : consumerOps) {
                if (consumerOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
                    needInsertCopyAssOps.insert(consumerOp);
                }
            }
        }
    }
    std::vector<Operation *> sortedOps(needInsertCopyAssOps.begin(), needInsertCopyAssOps.end());
    std::sort(sortedOps.begin(), sortedOps.end(),
        [](const Operation *a, const Operation *b) { return a->GetOpMagic() < b->GetOpMagic(); });
    for (auto &needInsertCopyAssOp : sortedOps) {
        auto input = needInsertCopyAssOp->GetIOperands()[0];
        if (input->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
            InsertCopyUBOp(function, needInsertCopyAssOp, input);
        } else if (input->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            InsertCopyDDROp(function, needInsertCopyAssOp, input);
        }
    }
}

Status PreGraphProcess::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Operation, "===> start PreGraph.");
    ColorGraph colorGraph;
    colorGraph.PreColorSort(function);
    auto opList = function.Operations();
    for (auto &op : opList) {
        colorGraph.InitializeTensorColor(op);
        UpdateCopyOpIsCube(op);
    }
    SetBoundary setBoundary;
    setBoundary.SetTensorBoundary(function);
    // Processing Special Ops
    SetCopyAttr setCopyAttr;
    for (auto &op : opList) {
        if (IsCopyOut(op.GetOpcode()) && op.GetOpcode() != Opcode::OP_COPY_OUT) {
            setCopyAttr.ProcessSpecialMTEOperation(op);
        }
        if (IsCopyIn(op.GetOpcode()) && op.GetOpcode() != Opcode::OP_COPY_IN && op.GetOpcode() != Opcode::OP_SHMEM_GET_GM2UB) {
            setCopyAttr.ProcessMoveInOperation(op);
        }
    }
    RemoveRedundantAssemble removeRedundantAssemble;
    removeRedundantAssemble.DeleteRedundantAssemble(function);
    CubeProcess cubeProcess;
    if (cubeProcess.UpdateCubeOp(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Update Cube attr failed.");
        return FAILED;
    }
    Status status = MergeViewAssembleUtils::MergeViewAssemble(function);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Merge assemble and view failed.");
        return status;
    }
    InsertNeedCopy(function);
    APASS_LOG_INFO_F(Elements::Operation, "===> End PreGraph.");
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