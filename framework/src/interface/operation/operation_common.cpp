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
 * \file operation_common.cpp
 * \brief
 */

#include "operation_common.h"
#include "interface/utils/vector_error.h"
#include <algorithm>
#include <unordered_set>

namespace npu::tile_fwk {

inline const std::vector<size_t>& GetShapeLenLimit(const std::string& op)
{
    // if the limit of op is not [1, 4], should add here
    static std::unordered_map<std::string, const std::vector<size_t>> op_shape_len_limit = {
        {"ADD", {1, 4}}, {"CAST", {1, 4}}, {"ISFINITE", {1, 5}}, {"REMR", {1, 5}}, {"DEFAULT", {1, 4}}};
    if (op_shape_len_limit.find(op) == op_shape_len_limit.end()) {
        return op_shape_len_limit.at("DEFAULT");
    }
    return op_shape_len_limit.at(op);
}

void CheckTensorShape(const LogicalTensorPtr& tensor, const std::string& op)
{
    auto shape = tensor->shape;
    // valid input dims must in [1, 4]
    auto shape_len_limit = GetShapeLenLimit(op);
    ASSERT(VectorErrorCode::ERR_PARAM_SHAPE_DIM_UNSUPPORTED,
           shape.size() >= shape_len_limit[0] && shape.size() <= shape_len_limit[1])
        << "The dims of tensor out of range.";
    int64_t shapeSize = 1;
    for (const auto& value : shape) {
        ASSERT(VectorErrorCode::ERR_PARAM_INVALID, value <= INT32_MAX)
            << "The dim value of tensor must less than or equal to INT32_MAX(2,147,483,647)";
        shapeSize *= value;
        ASSERT(VectorErrorCode::ERR_PARAM_INVALID, shapeSize <= INT32_MAX)
            << "The shape size of tensor must less than or equal to INT32_MAX(2,147,483,647)";
    }
}

void CheckTensorDynamicShape(const LogicalTensors iOperands, const Opcode opCode)
{
    const std::string opName = OpcodeManager::Inst().GetOpcodeStr(opCode);
    const auto& inputMemType = OpcodeManager::Inst().GetInputsMemType(opCode);
    if (inputMemType.size() != iOperands.size()) {
        return;
    }
    for (size_t i = 0; i < iOperands.size(); i++) {
        if (inputMemType[i] == MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        for (size_t dimIdx = 0; dimIdx < iOperands[i]->shape.size(); ++dimIdx) {
            ASSERT(VectorErrorCode::ERR_PARAM_INVALID, iOperands[i]->shape[dimIdx] > 0)
                << (!opName.empty() ? "Operation: " + opName : "")
                << " Input operand (name: " << iOperands[i]->tensor->GetSymbol() << ") "
                << " at dimension[" << dimIdx << "] has invalid shape value: " << iOperands[i]->shape[dimIdx];
        }
    }
}

std::vector<int> GetBroadCastShape(LogicalTensorPtr& operand1, LogicalTensorPtr& operand2)
{
    std::vector<int64_t> opShape1(operand1->shape);
    std::vector<int64_t> opShape2(operand2->shape);
    auto maxShapeSize = std::max(opShape1.size(), opShape2.size());
    if (opShape1.size() != maxShapeSize) {
        opShape1.insert(opShape1.begin(), maxShapeSize - opShape1.size(), 1);
    }
    if (opShape2.size() != maxShapeSize) {
        opShape2.insert(opShape2.begin(), maxShapeSize - opShape2.size(), 1);
    }
    std::vector<int> broadCastShape(maxShapeSize, 0);
    for (size_t i = 0; i < maxShapeSize; i++) {
        broadCastShape[i] = std::max(opShape1[i], opShape2[i]);
    }
    return broadCastShape;
}

std::vector<int> GetBroadcastAxes(const Shape& shape1, const Shape& shape2)
{
    Shape shape1_(shape1), shape2_(shape2);
    std::vector<int> result = {};
    auto maxShapeSize = std::max(shape1_.size(), shape2_.size());
    if (shape1_.size() != maxShapeSize) {
        shape1_.insert(shape1_.begin(), maxShapeSize - shape1_.size(), 1);
    }
    if (shape2_.size() != maxShapeSize) {
        shape2_.insert(shape2_.begin(), maxShapeSize - shape2_.size(), 1);
    }
    for (size_t i = 0; i < shape1_.size(); i++) {
        if (shape1_[i] != shape2_[i] && (shape1_[i] == 1 || shape2_[i] == 1)) {
            result.push_back(i);
        }
    }
    return result;
}

void CheckAxisRange(const Tensor& tensor, int& axis)
{
    int shapeSize = tensor.GetShape().size();
    if (axis < 0) {
        axis += shapeSize;
    }
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, axis >= 0 && axis < shapeSize)
        << "Axis is not in the reasonable range!";
}

void CheckTensorDimRange(const LogicalTensorPtr& tensor, size_t minDim, size_t maxDim, const std::string& opName)
{
    auto shape = tensor->shape;
    ASSERT(VectorErrorCode::ERR_PARAM_SHAPE_DIM_UNSUPPORTED,
           shape.size() >= minDim && shape.size() <= maxDim)
        << "The dims of tensor is out of range [" << minDim << ", " << maxDim << "]"
        << ", actual dims: " << shape.size() << " for op: " << opName;
}

void CheckTensorsDimConsistency(const std::vector<LogicalTensorPtr>& tensors, const std::string& opName)
{
    if (tensors.empty()) {
        return;
    }
    auto firstDim = tensors[0]->shape.size();
    for (size_t i = 1; i < tensors.size(); ++i) {
        ASSERT(VectorErrorCode::ERR_PARAM_SHAPE_DIM_UNSUPPORTED,
               tensors[i]->shape.size() == firstDim)
            << "Tensor dim inconsistent, tensor[0] dim: " << firstDim
            << ", tensor[" << i << "] dim: " << tensors[i]->shape.size()
            << " for op: " << opName;
    }
}

void CheckTensorShapeSize(const LogicalTensorPtr& tensor, const std::string& opName)
{
    auto shape = tensor->shape;
    int64_t shapeSize = 1;
    for (const auto& value : shape) {
        if (value > INT32_MAX) {
            ASSERT(VectorErrorCode::ERR_PARAM_INVALID, false)
                << "The dim value of tensor must less than or equal to INT32_MAX(2,147,483,647)"
                << " for op: " << opName;
        }
        if (value > 0) {
            shapeSize *= value;
        }
        if (shapeSize > INT32_MAX) {
            ASSERT(VectorErrorCode::ERR_PARAM_INVALID, false)
                << "The shape size of tensor must less than or equal to INT32_MAX(2,147,483,647)"
                << " for op: " << opName;
        }
    }
}

bool IsShapeConsistentOrBroadcastCompatible(const Shape& shape1, const Shape& shape2)
{
    if (shape1.size() != shape2.size()) {
        return false;
    }
    for (size_t i = 0; i < shape1.size(); ++i) {
        if (shape1[i] != shape2[i] && shape1[i] != 1 && shape2[i] != 1) {
            return false;
        }
    }
    return true;
}

void CheckTensorsShapeConsistencyOrBroadcast(const std::vector<LogicalTensorPtr>& tensors, const std::string& opName)
{
    if (tensors.empty()) {
        return;
    }
    for (size_t i = 1; i < tensors.size(); ++i) {
        Shape shape0 = tensors[0]->shape;
        Shape shapeI = tensors[i]->shape;
        ASSERT(VectorErrorCode::ERR_PARAM_INVALID,
               shape0 == shapeI || IsShapeConsistentOrBroadcastCompatible(shape0, shapeI))
            << "Tensor shape must be consistent or broadcast compatible"
            << ", tensor[0] shape: " << VectorToString(shape0)
            << ", tensor[" << i << "] shape: " << VectorToString(shapeI)
            << " for op: " << opName;
    }
}

void CheckTensorDataType(const LogicalTensorPtr& tensor, const std::unordered_set<DataType>& supportedTypes,
                         const std::string& opName)
{
    auto dtype = tensor->tensor->dtype;
    bool isSupported = supportedTypes.find(dtype) != supportedTypes.end();
    ASSERT(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, isSupported)
        << "Tensor data type " << DataType2String(dtype)
        << " is not in supported types for op: " << opName;
}

void CheckTensorsDataTypeConsistency(const LogicalTensorPtr& tensor1, const LogicalTensorPtr& tensor2,
                                     const std::string& opName)
{
    auto dtype1 = tensor1->tensor->dtype;
    auto dtype2 = tensor2->tensor->dtype;
    ASSERT(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, dtype1 == dtype2)
        << "Tensor data type inconsistent, tensor1 dtype: " << DataType2String(dtype1)
        << ", tensor2 dtype: " << DataType2String(dtype2) << " for op: " << opName;
}

void CheckTensorsFormatConsistency(const LogicalTensorPtr& tensor1, const LogicalTensorPtr& tensor2,
                                   const std::string& opName)
{
    auto format1 = tensor1->Format();
    auto format2 = tensor2->Format();
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, format1 == format2)
        << "Tensor format inconsistent, tensor1 format: " << std::to_string(format1)
        << ", tensor2 format: " << std::to_string(format2) << " for op: " << opName;
}
} // namespace npu::tile_fwk
