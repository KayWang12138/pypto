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
 * \file alignment_utils.cpp
 * \brief
 */

#include "passes/pass_utils/alignment_utils.h"
#include "interface/utils/common.h"

namespace npu::tile_fwk {

bool AlignmentUtils::IsValidForLastDimCheck(const LogicalTensorPtr& tensor)
{
    return tensor != nullptr && tensor->tensor != nullptr && !tensor->shape.empty();
}

bool AlignmentUtils::IsCombinedAxis(const std::vector<bool>& combineAxis, size_t index)
{
    return index < combineAxis.size() && combineAxis[index];
}
//padvalueing
int64_t AlignmentUtils::GetLastDimAlignBase(const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr || tensor->tensor == nullptr) {
        return 0;
    }
    auto bytes = static_cast<int64_t>(BytesOf(tensor->Datatype()));
    if (bytes <= 0) {
        return 0;
    }
    auto iter = BLOCK_PADDING_DIM.find(static_cast<size_t>(bytes));
    if (iter == BLOCK_PADDING_DIM.end()) {
        return 1;
    }
    return iter->second;
}
//IsLastDim32BAligned
bool AlignmentUtils::IsLastDim32BAligned(const LogicalTensorPtr& tensor)
{
    if (!IsValidForLastDimCheck(tensor)) {
        return false;
    }
    auto bytes = static_cast<int64_t>(BytesOf(tensor->Datatype()));
    if (bytes <= 0) {
        return false;
    }
    auto lastDim = tensor->shape.back();
    if (lastDim <= 0) {
        return false;
    }
    return ((lastDim * bytes) % 32) == 0;
}

// 和 /mnt/workspace/gitCode/cann/pypto/framework/src/interface/utils/common.h 的 AlignUp方法一致
int64_t AlignmentUtils::Pad(int64_t dim, int64_t padValue)
{
    if (padValue == 0) {
        return dim;
    }
    return (dim + padValue - 1) / padValue * padValue;
}

void AlignmentUtils::ProcessLastDim32BAligned(LogicalTensorPtr tensor) {
    auto memType = tensor->GetMemoryTypeOriginal();
    if (memType == MemoryType::MEM_UB && !IsLastDim32BAligned(tensor)) {
        size_t lastIdx = tensor->shape.size() - 1;
        size_t paddingValue = GetLastDimAlignBase(tensor); // 根据数据类型，判断需要pad到几个元素

        // 保存rawshape
        tensor->oriShape = tensor->shape;
        tensor->tensor->oriRawshape = tensor->tensor->rawshape;

        // pad 32B
        tensor->shape[lastIdx] = Pad(tensor->shape[lastIdx], paddingValue);
        tensor->tensor->rawshape[lastIdx] = Pad(tensor->tensor->oriRawshape[lastIdx], tensor->shape[lastIdx]);
    }
}

//op
bool AlignmentUtils::NeedPadLastDim(const LogicalTensorPtr& tensor)
{
    if (!IsValidForLastDimCheck(tensor)) {
        return false;
    }
    if (tensor->GetMemoryTypeOriginal() != MemoryType::MEM_UB) {
        return false;
    }
    auto alignBase = AlignmentUtils::GetLastDimAlignBase(tensor);
    auto lastDim = tensor->shape.back();
    return (lastDim > 0) && (alignBase > 0) && ((lastDim % alignBase) != 0);
}
//op
bool AlignmentUtils::IsRawLastDimUnaligned(const LogicalTensorPtr& tensor)
{
    if (!IsValidForLastDimCheck(tensor)) {
        return false;
    }
    const auto& oriRawShape = tensor->tensor->oriRawshape;
    const auto& rawShape = tensor->tensor->rawshape;
    if (oriRawShape.size() != tensor->shape.size() || rawShape.size() != tensor->shape.size()) {
        return false;
    }
    auto lastIdx = tensor->shape.size() - 1;
    return oriRawShape[lastIdx] != rawShape[lastIdx];
}
// op
bool AlignmentUtils::HasUnalignedInputOrOutput(const Operation& op)
{
    std::vector<bool> inputAxis;
    std::vector<bool> outputAxis;
    op.GetAttr(OpAttributeKey::inputCombineAxis, inputAxis);
    op.GetAttr(OpAttributeKey::outputCombineAxis, outputAxis);
    for (size_t i = 0; i < op.GetIOperands().size(); ++i) {
        if (IsCombinedAxis(inputAxis, i)) {
            continue;
        }
        if (AlignmentUtils::IsRawLastDimUnaligned(op.GetIOperands()[i]) ||
            AlignmentUtils::NeedPadLastDim(op.GetIOperands()[i])) {
            return true;
        }
    }
    for (size_t i = 0; i < op.GetOOperands().size(); ++i) {
        if (IsCombinedAxis(outputAxis, i)) {
            continue;
        }
        if (AlignmentUtils::IsRawLastDimUnaligned(op.GetOOperands()[i]) ||
            AlignmentUtils::NeedPadLastDim(op.GetOOperands()[i])) {
            return true;
        }
    }
    return false;
}

} // namespace npu::tile_fwk
