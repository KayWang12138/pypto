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
 * \file pre_graph_checker.cpp
 * \brief
 */

#include "pre_graph_checker.h"

namespace npu {
namespace tile_fwk {
Status PreGraphProcessChecker::DoPreCheck(Function &function) {
    ALOG_INFO_F("PreCheck for PreGraph");
    if (!function.LoopCheck().empty()) {
        ALOG_ERROR_F("Loopcheck failed before PreGraph");
        return FAILED;
    }
    for (auto &op : function.Operations()) {
        if (op.GetSubgraphID() == NOT_IN_SUBGRAPH) {
            ALOG_ERROR_F("%s[%d] is not partitioned.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
            return FAILED;
        }
        if ((op.GetOpcode() != Opcode::OP_ASSEMBLE) && (op.GetOpcode() != Opcode::OP_VIEW) && 
            (op.GetOpcode() != Opcode::OP_RESHAPE)) {
            continue;
        }
        auto tensorIn = op.GetIOperands().front();
        auto tensorOut = op.GetOOperands().front();
        if (tensorIn->GetMemoryTypeOriginal() != tensorOut->GetMemoryTypeOriginal()) {
            ALOG_ERROR_F("unmatched input output memory type for reshape opmagic: %d, input mem type: %s, output mem type: %s", 
                op.opmagic,
                MemoryTypeToString(tensorIn->GetMemoryTypeOriginal()).c_str(),
                MemoryTypeToString(tensorOut->GetMemoryTypeOriginal()).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status PreGraphProcessChecker::DoPostCheck(Function &function) {
    ALOG_INFO_F("PostCheck for PreGraph");
    // 检测是否成环
    if (!function.LoopCheck().empty()) {
        ALOG_ERROR_F("Loopcheck failed after PreGraph");
        return FAILED;
    }
    std::unordered_set<std::shared_ptr<LogicalTensor>> checkedTensors;
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE && PostCheckAssemble(function, op) != SUCCESS) {
            return FAILED;
        }
        if (op.GetOpcode() == Opcode::OP_VIEW && PostCheckView(function, op) != SUCCESS) {
            return FAILED;
        }
        if (op.GetOpcode() == Opcode::OP_RESHAPE && PostCheckReshape(function, op) != SUCCESS) {
            return FAILED;
        }
        for (const std::shared_ptr<LogicalTensor> &inputTensor : op.GetIOperands()) {
            if (checkedTensors.count(inputTensor) > 0) {
                continue;
            }
            checkedTensors.insert(inputTensor);
            if (PostCheckHelpFunc(*inputTensor) != SUCCESS) {
                return FAILED;
            }
        }
        for (const std::shared_ptr<LogicalTensor> &outputTensor : op.GetOOperands()) {
            if (checkedTensors.count(outputTensor) > 0) {
                continue;
            }
            checkedTensors.insert(outputTensor);
            if (PostCheckHelpFunc(*outputTensor) != SUCCESS) {
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status PreGraphProcessChecker::PostCheckHelpFunc(const LogicalTensor &singleTensor) {
    if (singleTensor.subGraphID == NOT_IN_SUBGRAPH) {
        // tensor 的子图编号是否被设置过
        ALOG_ERROR_F(
            "Tensor magic: %d, its subgraph id should not be %d.", singleTensor.GetMagic(), NOT_IN_SUBGRAPH);
        return FAILED;
    }
    if (singleTensor.GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
        singleTensor.isSubGraphBoundary == false) {
        // gm tensor 是否被标记为boundary
        ALOG_WARN_F("Tensor magic: %d, when memory type is DDR, this tensor should be subgraph boundary.",
            singleTensor.GetMagic());
    }
    if (singleTensor.GetMemoryTypeOriginal() == MemoryType::MEM_L0C &&
        (singleTensor.Datatype() != DataType::DT_FP32 && singleTensor.Datatype() != DataType::DT_INT32)) {
        // L0C tensor 数据类型是否为FP32或INT32
        ALOG_ERROR_F("Tensor magic: %d, when memory type is L0C, this tensor should be fp32 or int32.",
            singleTensor.GetMagic());
        return FAILED;
    }
    if (singleTensor.MemorySize() < 1 && !singleTensor.IsDummy()) {
        // 是否存在 dummy tensor
        ALOG_INFO_F("Tensor magic: %d, its memory size %d should be over than 0, but not.",
            singleTensor.GetMagic(), singleTensor.MemorySize());
    }
    return SUCCESS;
}

bool PreGraphProcessChecker::VerifyAssembleResult (const Operation &op) {
    auto assembleIn = op.GetIOperands().front();
    auto assembleOut = op.GetOOperands().front();
    for (LogicalTensorPtr in : op.GetIOperands()) {
        if (in->GetRawMagic() != assembleIn->GetRawMagic()) {
            return false;
        }
    }
    for (LogicalTensorPtr out : op.GetOOperands()) {
        if (out->GetRawMagic() != assembleOut->GetRawMagic()) {
            return false;
        }
    }
    if (assembleIn->GetRawMagic() != assembleOut->GetRawMagic()) {
        return false;
    }
    return true;
}

Status PreGraphProcessChecker::PostCheckAssemble(Function &function, const Operation &op) {
    if (VerifyAssembleResult(op)) {
        return SUCCESS;
    }
    auto assembleIn = op.GetIOperands().front();
    for (Operation *producer : assembleIn->GetProducers()) {
        if (producer->GetOpcode() == Opcode::OP_VIEW) {
            ALOG_ERROR_F(
                "Assemble[%d] Unsupported OP connection scenaios: assemble input tensor has view producer op[%d]",
                op.GetOpMagic(), producer->GetOpMagic());
            return FAILED;
        }
        if (producer->GetOpcode() == Opcode::OP_RESHAPE) {
            ALOG_ERROR_F(
                "Assemble[%d] Unsupported OP connection scenaios: assemble input tensor has reshape producer op[%d]",
                op.GetOpMagic(), producer->GetOpMagic());
            return FAILED;
        }
    }
    for (Operation *consumer : assembleIn->GetConsumers()) {
        if (consumer->GetOpMagic() == op.GetOpMagic()) {
            continue;
        }
        if (consumer->GetOpcode() == Opcode::OP_ASSEMBLE) {
            ALOG_ERROR_F(
                "Assemble[%d] Unsupported OP connection scenaios: assemble input tensor has other assemble consumer op[%d]",
                op.GetOpMagic(), consumer->GetOpMagic());
            return FAILED;
        }
        if (consumer->GetOpcode() == Opcode::OP_RESHAPE && function.IsFromOutCast(consumer->GetOOperands().front())) {
            ALOG_ERROR_F(
                "Assemble[%d] Unsupported OP connection scenaios: assemble input tensor has reshape->outcast consumer op[%d]",
                op.GetOpMagic(), consumer->GetOpMagic());
            return FAILED;
        }
    }
    ALOG_ERROR_F("Operation magic: %d, assemble op raw magic should not changed.", op.GetOpMagic());
    return FAILED;
}

bool PreGraphProcessChecker::VerifyViewResult (const Operation &op) {
    auto viewIn = op.GetIOperands().front();
    auto viewOut = op.GetOOperands().front();
    for (LogicalTensorPtr in : op.GetIOperands()) {
        if (in->GetRawMagic() != viewIn->GetRawMagic()) {
            return false;
        }
    }
    for (LogicalTensorPtr out : op.GetOOperands()) {
        if (out->GetRawMagic() != viewOut->GetRawMagic()) {
            return false;
        }
    }
    if (viewIn->GetRawMagic() != viewOut->GetRawMagic()) {
        return false;
    }
    return true;
}

Status PreGraphProcessChecker::PostCheckView(Function &function, const Operation &op) {
    if (VerifyViewResult(op)) {
        return SUCCESS;
    }
    auto viewOut = op.GetOOperands().front();
    for (Operation *producer : viewOut->GetProducers()) {
        if (producer->GetOpMagic() == op.GetOpMagic()) {
            continue;
        }
        if (producer->GetOpcode() == Opcode::OP_VIEW) {
            ALOG_ERROR_F(
                "View[%d] Unsupported OP connection scenaios: view input tensor has other view producer op[%d]",
                op.GetOpMagic(), producer->GetOpMagic());
            return FAILED;
        }
        if (producer->GetOpcode() == Opcode::OP_RESHAPE) {
            ALOG_ERROR_F(
                "View[%d] Unsupported OP connection scenaios: view input tensor has reshape producer op[%d]",
                op.GetOpMagic(), producer->GetOpMagic());
            return FAILED;
        }
    }
    for (Operation *consumer : viewOut->GetConsumers()) {
        if (consumer->GetOpcode() == Opcode::OP_ASSEMBLE) {
            ALOG_ERROR_F(
                "View[%d] Unsupported OP connection scenaios: view input tensor has assemble consumer op[%d]",
                op.GetOpMagic(), consumer->GetOpMagic());
            return FAILED;
        }
        if (consumer->GetOpcode() == Opcode::OP_RESHAPE && function.IsFromOutCast(consumer->GetOOperands().front())) {
            ALOG_ERROR_F(
                "View[%d] Unsupported OP connection scenaios: view input tensor has reshape->outcast consumer op[%d]",
                op.GetOpMagic(), consumer->GetOpMagic());
            return FAILED;
        }
    }
    ALOG_ERROR_F("Operation magic: %d, view op raw magic should not changed.", op.GetOpMagic());
    return FAILED;
}

Status PreGraphProcessChecker::HandleScenarioReshapeOutCast(Function &function, const Operation &op, const LogicalTensorPtr reshapeIn) {
    for (Operation *producer : reshapeIn->GetProducers()) {
        if (producer->GetOpcode() == Opcode::OP_VIEW) {
            ALOG_ERROR_F("Reshape[%d]->outcast Unsupported OP connection scenaios: reshape->outcast input tensor "
                         "has view producer op[%d]",
                op.GetOpMagic(), producer->GetOpMagic());
            return FAILED;
        }
        if (producer->GetOpcode() == Opcode::OP_RESHAPE) {
            ALOG_ERROR_F("Reshape[%d]->outcast Unsupported OP connection scenaios: reshape->outcast input tensor "
                         "has reshape producer op[%d]",
                op.GetOpMagic(), producer->GetOpMagic());
            return FAILED;
        }
    }
    for (Operation *consumer : reshapeIn->GetConsumers()) {
        if (consumer->GetOpMagic() == op.GetOpMagic()) {
            continue;
        }
        if (consumer->GetOpcode() == Opcode::OP_ASSEMBLE) {
            ALOG_ERROR_F("Reshape[%d]->outcast Unsupported OP connection scenaios: reshape->outcast input tensor has "
                         "other assemble consumer op[%d]",
                op.GetOpMagic(), consumer->GetOpMagic());
            return FAILED;
        }
        if (consumer->GetOpcode() == Opcode::OP_RESHAPE && function.IsFromOutCast(consumer->GetOOperands().front())) {
            ALOG_ERROR_F("Reshape[%d]->outcast Unsupported OP connection scenaios: reshape->outcast input tensor has "
                         "reshape->outcast consumer op[%d]",
                op.GetOpMagic(), consumer->GetOpMagic());
            return FAILED;
        }
    }
    ALOG_ERROR_F("Operation magic: %d, reshape op's output actual raw magic shoule be same with input raw magic.",
        op.GetOpMagic());
    return FAILED;
}

Status PreGraphProcessChecker::VerifyReshapeResult(
    Function &function, const Operation &op, const LogicalTensorPtr reshapeIn, const LogicalTensorPtr reshapeOut) {
    if (reshapeOut->GetRawMagic() == reshapeIn->GetRawMagic()) {
        return SUCCESS;
    }
    if (function.IsFromOutCast(reshapeOut)) {
        return HandleScenarioReshapeOutCast(function, op, reshapeIn);
    }
    for (Operation *producer : reshapeOut->GetProducers()) {
        if (producer->GetOpMagic() == op.GetOpMagic()) {
            continue;
        }
        if (producer->GetOpcode() == Opcode::OP_VIEW) {
            ALOG_ERROR_F(
                "Reshape[%d] Unsupported OP connection scenaios: reshape input tensor has other view producer op[%d]",
                op.GetOpMagic(), producer->GetOpMagic());
            return FAILED;
        }
        if (producer->GetOpcode() == Opcode::OP_RESHAPE) {
            ALOG_ERROR_F(
                "Reshape[%d] Unsupported OP connection scenaios: reshape input tensor has reshape producer op[%d]",
                op.GetOpMagic(), producer->GetOpMagic());
            return FAILED;
        }
    }
    for (Operation *consumer : reshapeOut->GetConsumers()) {
        if (consumer->GetOpcode() == Opcode::OP_ASSEMBLE) {
            ALOG_ERROR_F(
                "Reshape[%d] Unsupported OP connection scenaios: reshape input tensor has assemble consumer op[%d]",
                op.GetOpMagic(), consumer->GetOpMagic());
            return FAILED;
        }
        if (consumer->GetOpcode() == Opcode::OP_RESHAPE && function.IsFromOutCast(consumer->GetOOperands().front())) {
            ALOG_ERROR_F(
                "Reshape[%d] Unsupported OP connection scenaios: reshape input tensor has reshape->outcast consumer op[%d]",
                op.GetOpMagic(), consumer->GetOpMagic());
            return FAILED;
        }
    }
    ALOG_ERROR_F("Operation magic: %d, reshape op's output actual raw magic shoule be same with input raw magic.",
        op.GetOpMagic());
    return FAILED;
}

Status PreGraphProcessChecker::PostCheckReshape(Function &function, const Operation &op) {
    auto reshapeIn = op.GetIOperands().front();
    auto reshapeOut = op.GetOOperands().front();
    if (VerifyReshapeResult(function, op, reshapeIn, reshapeOut) == FAILED) {
        return FAILED;
    }

    if (reshapeIn->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
        ALOG_DEBUG_F(" reshape on local buffer, opmagic: %d", op.opmagic);
        auto opSubgraphId = op.GetSubgraphID();
        auto inputSubgraphId = reshapeIn->GetSubgraphID();
        auto outSubgraphId = reshapeIn->GetSubgraphID();
        if (opSubgraphId != inputSubgraphId || opSubgraphId != outSubgraphId) {
            // local buffer 上的reshape，输入/输出/op的子图编号相同
            ALOG_ERROR_F("OP_RESHAPE[%d], op subGraphId: %d, input subGraphId: %d, output subGraphId: %d,", 
                op.GetOpMagic(), opSubgraphId, inputSubgraphId, outSubgraphId);
            return FAILED;
        }
        
        // Debug Print
        ALOG_DEBUG_F(" check done, input magic %d (raw %d), output magic %d (raw %d)",
            reshapeIn->magic, reshapeIn->GetRawMagic(), reshapeOut->magic,
            reshapeOut->GetRawMagic());
        auto childOp = *(reshapeOut->GetConsumers().begin());
        ALOG_DEBUG_F(" child op: %s, opmagic: %d", childOp->GetOpcodeStr().c_str(), childOp->opmagic);
        ALOG_DEBUG_F(" child op output magic %d (raw %d)", childOp->GetOOperands()[0]->magic,
            childOp->GetOOperands()[0]->GetRawMagic());
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu