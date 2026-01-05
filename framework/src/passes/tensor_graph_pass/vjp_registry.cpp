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
 * \file vjp_registry.cpp
 * \brief VJP registry implementation.
 */

#include "vjp_registry.h"

#include <stdexcept>

#include "interface/inner/element.h"

namespace npu::tile_fwk {

LogicalTensorPtr VJPContext::GetOutputGrad(const std::string& name) const {
    auto it = out_grads.find(name);
    if (it != out_grads.end()) {
        return it->second;
    }
    return nullptr;
}

LogicalTensorPtr VJPContext::GetSaved(const std::string& name) const {
    auto it = saved_tensors.find(name);
    if (it != saved_tensors.end()) {
        return it->second;
    }
    return nullptr;
}

double VJPContext::GetScalarAttr(double defaultVal) const {
    if (node == nullptr) {
        return defaultVal;
    }
    Element element;
    if (node->GetAttr(OpAttributeKey::scalar, element)) {
        return element.Cast<double>();
    }
    double value = defaultVal;
    if (node->GetAttr(OP_ATTR_PREFIX + "scalar", value)) {
        return value;
    }
    return defaultVal;
}

std::vector<int64_t> VJPContext::GetInputShape(const std::string& name) const {
    if (node == nullptr) {
        return {};
    }

    // Try to find input by name in saved_tensors first
    auto it = saved_tensors.find(name);
    if (it != saved_tensors.end() && it->second != nullptr) {
        const auto& shape = it->second->GetShape();
        return std::vector<int64_t>(shape.begin(), shape.end());
    }

    // Fallback: search by input index
    // Common naming convention: "input" -> index 0, "other" -> index 1
    const auto& inputs = node->GetIOperands();
    size_t index = 0;
    if (name == "input" || name == "input0") {
        index = 0;
    } else if (name == "other" || name == "input1") {
        index = 1;
    } else if (name == "input2") {
        index = 2;
    }

    if (index < inputs.size() && inputs[index] != nullptr) {
        const auto& shape = inputs[index]->GetShape();
        return std::vector<int64_t>(shape.begin(), shape.end());
    }

    return {};
}

LogicalTensorPtr VJPContext::Unbroadcast(LogicalTensorPtr grad, const std::vector<int64_t>& targetShape) const {
    if (grad == nullptr || function == nullptr) {
        return nullptr;
    }

    const Shape& gradShape = grad->GetShape();
    std::vector<int64_t> gradShapeVec(gradShape.begin(), gradShape.end());

    // Check if shapes match
    if (gradShapeVec == targetShape) {
        return grad;
    }

    // Compute broadcast axes
    std::vector<int64_t> broadcastAxes;
    int64_t gradNdim = static_cast<int64_t>(gradShapeVec.size());
    int64_t targetNdim = static_cast<int64_t>(targetShape.size());

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
        // Compute output shape after summing along this axis
        const Shape& inputShape = result->GetShape();
        Shape outputShape;
        for (size_t i = 0; i < inputShape.size(); ++i) {
            if (static_cast<int64_t>(i) == axis) {
                outputShape.push_back(1);  // keepdim = true
            } else {
                outputShape.push_back(inputShape[i]);
            }
        }

        auto sumOutput = std::make_shared<LogicalTensor>(*function, result->Datatype(), outputShape);
        // Use OP_ROWSUM_SINGLE which properly supports AXIS attribute
        Operation& sumOp = function->AddRawOperation(Opcode::OP_ROWSUM_SINGLE, {result}, {sumOutput});
        // Use uppercase AXIS to match opcode.cpp registration
        sumOp.SetAttr(OP_ATTR_PREFIX + "AXIS", axis);
        result = sumOutput;
    }

    // Reshape to target shape if needed
    const Shape& resultShape = result->GetShape();
    std::vector<int64_t> resultShapeVec(resultShape.begin(), resultShape.end());
    if (resultShapeVec != targetShape) {
        Shape newShape(targetShape.begin(), targetShape.end());
        auto reshaped = std::make_shared<LogicalTensor>(*function, result->Datatype(), newShape);
        function->AddRawOperation(Opcode::OP_RESHAPE, {result}, {reshaped});
        result = reshaped;
    }

    return result;
}

VJPRegistry& VJPRegistry::Instance() {
    static VJPRegistry instance;
    return instance;
}

void VJPRegistry::Register(Opcode op, VJPFunc vjp) {
    registry_[op] = std::move(vjp);
}

bool VJPRegistry::HasVJP(Opcode op) const {
    return registry_.find(op) != registry_.end();
}

VJPFunc VJPRegistry::GetVJP(Opcode op) const {
    auto it = registry_.find(op);
    if (it == registry_.end()) {
        throw std::runtime_error("No VJP rule registered for opcode");
    }
    return it->second;
}

std::vector<Opcode> VJPRegistry::GetRegisteredOpcodes() const {
    std::vector<Opcode> result;
    result.reserve(registry_.size());
    for (const auto& pair : registry_) {
        result.push_back(pair.first);
    }
    return result;
}

} // namespace npu::tile_fwk
