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
 * \file autodiff.cpp
 * \brief Automatic differentiation pass implementation.
 */

#include "autodiff.h"

#include <algorithm>
#include <iostream>
#include <queue>
#include <stack>

#include "interface/utils/log.h"
#include "interface/inner/element.h"
#include "interface/operation/operation.h"
#include "interface/program/program.h"
#include "passes/pass_mgr/pass_registry.h"

namespace npu::tile_fwk {

void AutodiffPass::RegisterGradientTensor(int magic, LogicalTensorPtr tensor) {
    Program::GetInstance().RegisterGradientTensor(magic, tensor);
}

LogicalTensorPtr AutodiffPass::GetGradientTensor(int magic) {
    return Program::GetInstance().GetGradientTensor(magic);
}

void AutodiffPass::ClearGradientRegistry() {
    Program::GetInstance().ClearGradientRegistry();
}

REG_PASS(AutodiffPass);

Status AutodiffPass::PreCheck(Function& function) {
    // Verify function is a tensor graph
    if (function.GetGraphType() != GraphType::TENSOR_GRAPH) {
        ALOG_WARN_F("AutodiffPass only supports TENSOR_GRAPH, skipping.");
        return SUCCESS;
    }
    return SUCCESS;
}

Status AutodiffPass::PostCheck(Function& /* function */) {
    return SUCCESS;
}

Status AutodiffPass::RunOnFunction(Function& function) {
    ALOG_INFO_F("===> Start AutodiffPass for function [%s].", function.GetRawName().c_str());

    // Clear state from previous runs
    valueGrads_.clear();
    requiresGradSet_.clear();
    wrtTensors_.clear();

    // Step 1: Find loss tensor
    LogicalTensorPtr lossTensor = FindLossTensor(function);
    if (lossTensor == nullptr) {
        ALOG_INFO_F("No loss tensor found, skipping autodiff.");
        return SUCCESS;
    }

    // Step 2: Collect tensors requiring gradients
    wrtTensors_ = CollectRequiresGradTensors(function);
    if (wrtTensors_.empty()) {
        ALOG_INFO_F("No tensors require gradients, skipping autodiff.");
        return SUCCESS;
    }

    for (const auto& tensor : wrtTensors_) {
        requiresGradSet_.insert(tensor->GetMagic());
    }

    // Step 3: Initialize loss gradient
    if (InitializeLossGrad(function, lossTensor) != SUCCESS) {
        ALOG_ERROR_F("Failed to initialize loss gradient.");
        return FAILED;
    }

    // Step 4: Get reverse topological order
    std::vector<Operation*> reverseTopoOrder = GetReverseTopoOrder(function);
    if (reverseTopoOrder.empty() && !function.Operations().DuplicatedOpList().empty()) {
        ALOG_ERROR_F("Cycle detected in computation graph, cannot compute gradients.");
        return FAILED;
    }

    // Step 5: Process operations in reverse order
    for (Operation* op : reverseTopoOrder) {
        if (ProcessOperation(function, op) != SUCCESS) {
            ALOG_ERROR_F("Failed to process operation during autodiff.");
            return FAILED;
        }
    }

    // Step 6: Store gradients via tensor attributes
    struct GradInfo {
        LogicalTensorPtr grad;
        std::vector<int64_t> originals;
    };
    std::unordered_map<int, GradInfo> gradInfos;

    for (const auto& tensor : wrtTensors_) {
        auto it = valueGrads_.find(tensor->GetMagic());
        if (it == valueGrads_.end() || it->second == nullptr) {
            continue;
        }
        LogicalTensorPtr grad = it->second;
        ALOG_INFO_F("Setting gradient_magic for tensor magic=%d to grad_magic=%d.",
                    tensor->GetMagic(), grad->GetMagic());
        tensor->SetAttr("gradient_magic", static_cast<int64_t>(grad->GetMagic()));

        auto& entry = gradInfos[grad->GetMagic()];
        if (!entry.grad) {
            entry.grad = grad;
        }
        entry.originals.push_back(static_cast<int64_t>(tensor->GetMagic()));
    }

    for (const auto& kv : gradInfos) {
        const auto& info = kv.second;
        if (!info.grad) {
            continue;
        }
        info.grad->SetAttr("is_gradient", true);
        info.grad->SetAttr("gradient_of_magics", info.originals);
        if (info.originals.size() == 1) {
            info.grad->SetAttr("gradient_of_magic", info.originals.front());
        }

         // Gradient tensors are not added to outCasts_ since slot allocation is already complete.
         // They are tracked via tensor attributes and TensorMap for retrieval.
        function.GetTensorMap().Insert(info.grad, false);

        // Also register in the static gradient registry (persists across TensorMap resets)
        RegisterGradientTensor(info.grad->GetMagic(), info.grad);

        ALOG_INFO_F("Gradient computed for tensor (grad_magic=%d, originals=%zu).",
            info.grad->GetMagic(), info.originals.size());
    }

    ALOG_INFO_F("===> End AutodiffPass for function [%s].", function.GetRawName().c_str());
    return SUCCESS;
}

LogicalTensorPtr AutodiffPass::FindLossTensor(Function& function) {
    // Look for tensor marked with is_loss=true
    // First check outcasts
    for (const auto& outcast : function.outCasts_) {
        if (outcast == nullptr) {
            continue;
        }
        bool isLoss = false;
        if (outcast->GetAttr(ATTR_IS_LOSS, isLoss) && isLoss) {
            return outcast;
        }
    }

    // Check incasts (in case input tensor is marked as loss, e.g., loss[:] = ...)
    for (const auto& incast : function.inCasts_) {
        if (incast == nullptr) {
            continue;
        }
        bool isLoss = false;
        if (incast->GetAttr(ATTR_IS_LOSS, isLoss) && isLoss) {
            return incast;
        }
    }

    // Then check all operations' outputs
    auto ops = function.Operations().DuplicatedOpList();
    for (auto& op : ops) {
        for (const auto& output : op->GetOOperands()) {
            if (output == nullptr) {
                continue;
            }
            bool isLoss = false;
            if (output->GetAttr(ATTR_IS_LOSS, isLoss) && isLoss) {
                return output;
            }
        }
    }

    return nullptr;
}

std::vector<LogicalTensorPtr> AutodiffPass::CollectRequiresGradTensors(Function& function) {
    std::vector<LogicalTensorPtr> result;

    // Check incasts
    for (const auto& incast : function.inCasts_) {
        bool requiresGrad = false;
        if (incast->GetAttr(ATTR_REQUIRES_GRAD, requiresGrad) && requiresGrad) {
            result.push_back(incast);
        }
    }

    // Check all operations' inputs and outputs
    for (auto& op : function.Operations().DuplicatedOpList()) {
        for (const auto& input : op->GetIOperands()) {
            bool requiresGrad = false;
            if (input->GetAttr(ATTR_REQUIRES_GRAD, requiresGrad) && requiresGrad) {
                // Avoid duplicates
                bool found = false;
                for (const auto& existing : result) {
                    if (existing->GetMagic() == input->GetMagic()) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    result.push_back(input);
                }
            }
        }
    }

    return result;
}

Status AutodiffPass::InitializeLossGrad(Function& function, LogicalTensorPtr loss) {
    // Create gradient tensor filled with ones
    LogicalTensorPtr grad = CreateOnesTensor(function, loss->GetShape(), loss->Datatype());
    if (grad == nullptr) {
        return FAILED;
    }

    valueGrads_[loss->GetMagic()] = grad;
    return SUCCESS;
}

std::vector<Operation*> AutodiffPass::GetReverseTopoOrder(Function& function) {
    std::vector<Operation*> result;
    std::unordered_set<Operation*> visited;
    std::vector<Operation*> opList = function.Operations().DuplicatedOpList();

    // Compute in-degrees
    std::unordered_map<Operation*, int> inDegree;
    for (auto* op : opList) {
        inDegree[op] = 0;
    }

    for (auto* op : opList) {
        for (auto* consumer : op->ConsumerOps()) {
            inDegree[consumer]++;
        }
    }

    // Kahn's algorithm for topological sort
    std::queue<Operation*> zeroInDegree;
    for (auto* op : opList) {
        if (inDegree[op] == 0) {
            zeroInDegree.push(op);
        }
    }

    std::vector<Operation*> topoOrder;
    while (!zeroInDegree.empty()) {
        Operation* op = zeroInDegree.front();
        zeroInDegree.pop();
        topoOrder.push_back(op);

        for (auto* consumer : op->ConsumerOps()) {
            inDegree[consumer]--;
            if (inDegree[consumer] == 0) {
                zeroInDegree.push(consumer);
            }
        }
    }

    // Cycle detection: if not all ops were processed, graph has a cycle
    if (topoOrder.size() != opList.size()) {
        ALOG_ERROR_F("Graph contains cycle! Expected %zu ops but got %zu in topoOrder.",
                     opList.size(), topoOrder.size());
        return {};
    }

    // Reverse the order
    std::reverse(topoOrder.begin(), topoOrder.end());
    return topoOrder;
}

Status AutodiffPass::ProcessOperation(Function& function, Operation* op) {
    Opcode opcode = op->GetOpcode();

    // Skip operations that don't need gradient computation
    // Check if any output has a gradient
    bool hasOutputGrad = false;
    for (const auto& output : op->GetOOperands()) {
        if (valueGrads_.find(output->GetMagic()) != valueGrads_.end()) {
            hasOutputGrad = true;
            break;
        }
    }

    if (!hasOutputGrad) {
        return SUCCESS;
    }

    // Check if VJP rule exists
    if (!VJPRegistry::Instance().HasVJP(opcode)) {
        ALOG_WARN_F("No VJP rule for opcode, skipping gradient computation for this op.");
        return SUCCESS;
    }

    // Build context and call VJP
    VJPContext ctx = BuildVJPContext(function, op);
    VJPFunc vjp = VJPRegistry::Instance().GetVJP(opcode);
    VJPResult inputGrads = vjp(ctx);

    // Accumulate gradients for inputs
    const auto& inputs = op->GetIOperands();
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& input = inputs[i];

        // Skip non-differentiable dtypes
        if (!IsDifferentiableDtype(input->Datatype())) {
            continue;
        }

        // Find gradient by name
        std::string inputName = (i == 0) ? "input" : ((i == 1) ? "other" : ("input" + std::to_string(i)));
        auto it = inputGrads.find(inputName);
        if (it != inputGrads.end() && it->second != nullptr) {
            if (AccumulateGrad(function, input, it->second) != SUCCESS) {
                return FAILED;
            }
        }
    }

    return SUCCESS;
}

Status AutodiffPass::AccumulateGrad(Function& function, LogicalTensorPtr tensor, LogicalTensorPtr grad) {
    if (grad == nullptr) {
        return SUCCESS;
    }

    int magic = tensor->GetMagic();
    auto it = valueGrads_.find(magic);

    if (it == valueGrads_.end() || it->second == nullptr) {
        // First gradient, just store it
        valueGrads_[magic] = grad;
    } else {
        // Accumulate: existing + new
        LogicalTensorPtr existing = it->second;

        // Validate shapes match
        if (existing->GetShape() != grad->GetShape()) {
            ALOG_ERROR_F("Shape mismatch when accumulating gradients: existing shape vs grad shape");
            return FAILED;
        }

        // Cast grad to existing dtype if needed
        if (existing->Datatype() != grad->Datatype()) {
            grad = CreateCastOp(function, grad, existing->Datatype());
            if (grad == nullptr) {
                ALOG_ERROR_F("Failed to cast gradient dtype");
                return FAILED;
            }
        }

        LogicalTensorPtr sum = CreateAddOp(function, existing, grad);
        if (sum == nullptr) {
            return FAILED;
        }
        valueGrads_[magic] = sum;
    }

    return SUCCESS;
}

VJPContext AutodiffPass::BuildVJPContext(Function& function, Operation* op) {
    VJPContext ctx;
    ctx.node = op;
    ctx.function = &function;

    // Populate output gradients
    const auto& outputs = op->GetOOperands();
    for (size_t i = 0; i < outputs.size(); ++i) {
        std::string name = (i == 0) ? "output" : ("output" + std::to_string(i));
        auto it = valueGrads_.find(outputs[i]->GetMagic());
        if (it != valueGrads_.end()) {
            ctx.out_grads[name] = it->second;
        }
    }

    // Populate saved tensors (inputs and outputs)
    const auto& inputs = op->GetIOperands();
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::string name = (i == 0) ? "input" : ((i == 1) ? "other" : ("input" + std::to_string(i)));
        ctx.saved_tensors[name] = inputs[i];
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        std::string name = (i == 0) ? "output" : ("output" + std::to_string(i));
        ctx.saved_tensors[name] = outputs[i];
    }

    return ctx;
}

bool AutodiffPass::IsDifferentiableDtype(DataType dtype) const {
    return dtype == DataType::DT_FP32 ||
           dtype == DataType::DT_FP16 ||
           dtype == DataType::DT_BF16;
}

LogicalTensorPtr AutodiffPass::CreateOnesTensor(Function& function, const Shape& shape, DataType dtype) {
    // Create output tensor
    auto output = std::make_shared<LogicalTensor>(function, dtype, shape);

    // Create OP_VEC_DUP to fill with ones
    // OP_VEC_DUP fills a tensor with a scalar value
    Operation& dupOp = function.AddRawOperation(Opcode::OP_VEC_DUP, {}, {output});

    // Set scalar value to 1.0 based on dtype
    Element scalarValue;
    switch (dtype) {
        case DataType::DT_FP32:
            scalarValue = Element(DataType::DT_FP32, 1.0f);
            break;
        case DataType::DT_FP16:
            scalarValue = Element(DataType::DT_FP16, 1.0);
            break;
        case DataType::DT_BF16:
            scalarValue = Element(DataType::DT_BF16, 1.0);
            break;
        default:
            scalarValue = Element(DataType::DT_FP32, 1.0f);
            break;
    }

    dupOp.SetAttr(OpAttributeKey::scalar, scalarValue);

    // Set shape attributes
    std::vector<int64_t> shapeVec(shape.begin(), shape.end());
    dupOp.SetAttr(OP_ATTR_PREFIX + "shape", shapeVec);

    return output;
}

LogicalTensorPtr AutodiffPass::CreateAddOp(Function& function, LogicalTensorPtr a, LogicalTensorPtr b) {
    if (a == nullptr || b == nullptr) {
        return nullptr;
    }

    // Determine output shape (should be the same for gradient accumulation)
    Shape outputShape = a->GetShape();
    DataType dtype = a->Datatype();

    // Create output tensor
    auto output = std::make_shared<LogicalTensor>(function, dtype, outputShape);

    // Add operation
    Operation& addOp = function.AddRawOperation(Opcode::OP_ADD, {a, b}, {output});
    (void)addOp; // Suppress unused variable warning

    return output;
}

LogicalTensorPtr AutodiffPass::CreateCastOp(Function& function, LogicalTensorPtr input, DataType targetDtype) {
    if (input == nullptr) {
        return nullptr;
    }

    if (input->Datatype() == targetDtype) {
        return input;
    }

    auto output = std::make_shared<LogicalTensor>(function, targetDtype, input->GetShape());
    Operation& castOp = function.AddRawOperation(Opcode::OP_CAST, {input}, {output});
    castOp.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_NONE);
    return output;
}

LogicalTensorPtr AutodiffPass::CreateSumOp(Function& function, LogicalTensorPtr input, int64_t axis, bool keepdim) {
    if (input == nullptr) {
        return nullptr;
    }

    Shape inputShape = input->GetShape();
    DataType dtype = input->Datatype();

    int64_t axisNorm = axis;
    if (axisNorm < 0) {
        axisNorm += static_cast<int64_t>(inputShape.size());
    }

    if (inputShape.empty() || axisNorm < 0 || axisNorm >= static_cast<int64_t>(inputShape.size())) {
        return nullptr;
    }

    // ROWSUM_SINGLE outputs keepdim shape (axis dimension becomes 1)
    Shape sumShape = inputShape;
    sumShape[axisNorm] = 1;

    auto sumOutput = std::make_shared<LogicalTensor>(function, dtype, sumShape);

    Operation& sumOp = function.AddRawOperation(Opcode::OP_ROWSUM_SINGLE, {input}, {sumOutput});
    sumOp.SetAttr(OP_ATTR_PREFIX + "AXIS", static_cast<int64_t>(axisNorm));

    if (keepdim) {
        return sumOutput;
    }

    // Remove reduced dimension for keepdim = false
    Shape outputShape;
    for (size_t i = 0; i < sumShape.size(); ++i) {
        if (static_cast<int64_t>(i) != axisNorm) {
            outputShape.push_back(sumShape[i]);
        }
    }
    if (outputShape.empty()) {
        outputShape.push_back(1);
    }
    auto reshaped = std::make_shared<LogicalTensor>(function, dtype, outputShape);
    function.AddRawOperation(Opcode::OP_RESHAPE, {sumOutput}, {reshaped});
    return reshaped;
}

LogicalTensorPtr AutodiffPass::Unbroadcast(Function& function, LogicalTensorPtr grad,
                                            const std::vector<int64_t>& targetShape) {
    if (grad == nullptr) {
        return nullptr;
    }

    const Shape& gradShape = grad->GetShape();
    std::vector<int64_t> gradShapeVec(gradShape.begin(), gradShape.end());

    // Edge case: both are empty (scalar)
    if (gradShapeVec.empty() && targetShape.empty()) {
        return grad;
    }

    // Edge case: target is scalar - sum all dimensions
    if (targetShape.empty()) {
        LogicalTensorPtr result = grad;
        for (int64_t axis = static_cast<int64_t>(gradShapeVec.size()) - 1; axis >= 0; --axis) {
            result = CreateSumOp(function, result, axis, false);
            if (result == nullptr) {
                return nullptr;
            }
        }
        return result;
    }

    // Edge case: grad is scalar - need to broadcast to target (shouldn't happen in normal backward)
    if (gradShapeVec.empty()) {
        ALOG_WARN_F("Unbroadcast: grad is scalar but target is not, returning grad unchanged.");
        return grad;
    }

    // Check if shapes match
    if (gradShapeVec == targetShape) {
        return grad;
    }

    // Edge case: target has more dimensions than grad (invalid for unbroadcast)
    int64_t gradNdim = static_cast<int64_t>(gradShapeVec.size());
    int64_t targetNdim = static_cast<int64_t>(targetShape.size());
    if (targetNdim > gradNdim) {
        ALOG_ERROR_F("Unbroadcast: target has more dimensions (%lld) than grad (%lld).",
                     static_cast<long long>(targetNdim), static_cast<long long>(gradNdim));
        return nullptr;
    }

    // Compute broadcast axes
    std::vector<int64_t> broadcastAxes;

    // Leading dimensions that were added
    for (int64_t i = 0; i < gradNdim - targetNdim; ++i) {
        broadcastAxes.push_back(i);
    }

    // Dimensions that were broadcast from 1
    for (int64_t i = 0; i < targetNdim; ++i) {
        int64_t gradAxis = i + (gradNdim - targetNdim);
        if (targetShape[i] == 1 && gradShapeVec[gradAxis] != 1) {
            broadcastAxes.push_back(gradAxis);
        }
    }

    if (broadcastAxes.empty()) {
        return grad;
    }

    // Sum along broadcast axes (in reverse order to preserve axis indices)
    LogicalTensorPtr result = grad;
    std::sort(broadcastAxes.rbegin(), broadcastAxes.rend());

    for (int64_t axis : broadcastAxes) {
        result = CreateSumOp(function, result, axis, true);
        if (result == nullptr) {
            return nullptr;
        }
    }

    // Reshape to target shape if needed
    const Shape& resultShape = result->GetShape();
    std::vector<int64_t> resultShapeVec(resultShape.begin(), resultShape.end());
    if (resultShapeVec != targetShape) {
        Shape newShape(targetShape.begin(), targetShape.end());
        auto reshaped = std::make_shared<LogicalTensor>(function, result->Datatype(), newShape);
        function.AddRawOperation(Opcode::OP_RESHAPE, {result}, {reshaped});
        result = reshaped;
    }

    return result;
}

} // namespace npu::tile_fwk
