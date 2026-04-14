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
    if (iter != BLOCK_PADDING_DIM.end()) {
        return static_cast<int64_t>(iter->second);
    }
    return BLOCK_SIZE / bytes;
}

int64_t AlignmentUtils::GetLastDimAlignBaseOrOne(const LogicalTensorPtr& tensor)
{
    auto alignBase = AlignmentUtils::GetLastDimAlignBase(tensor);
    return alignBase > 0 ? alignBase : 1;
}

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
    return ((lastDim * bytes) % BLOCK_SIZE) == 0;
}

bool AlignmentUtils::NeedPadLastDim(const LogicalTensorPtr& tensor)
{
    if (!IsValidForLastDimCheck(tensor)) {
        return false;
    }
    if (tensor->GetMemoryTypeOriginal() != MemoryType::MEM_UB) {
        return false;
    }
    auto alignBase = AlignmentUtils::GetLastDimAlignBase(tensor);
    if (alignBase <= 0) {
        return false;
    }
    auto lastDim = tensor->shape.back();
    if (lastDim <= 0) {
        return false;
    }
    return (lastDim % alignBase) != 0;
}

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
