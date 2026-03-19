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
 * \file quantize.h
 * \brief Quantization operation interfaces for INT8 symmetric and asymmetric quantization
 */

#pragma once

#include <string>
#include <optional>
#include "interface/utils/common.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"

namespace npu::tile_fwk {

enum class QuantizeType {
    INT8_SYM,   // Symmetric quantization: FP32 -> INT8
    INT8_ASYM,  // Asymmetric quantization: FP32 -> UINT8
};

template <QuantizeType T>
std::string GetQuantizeOpName() {
    switch (T) {
        case QuantizeType::INT8_SYM: return "QUANTIZE_SYM";
        case QuantizeType::INT8_ASYM: return "QUANTIZE_ASYM";
        default: ASSERT(false && "unknown quantize op type"); return "";
    }
}

template <QuantizeType T>
Opcode GetQuantizeOpCode() {
    switch (T) {
        case QuantizeType::INT8_SYM: return Opcode::OP_QUANTIZE_SYM;
        case QuantizeType::INT8_ASYM: return Opcode::OP_QUANTIZE_ASYM;
        default: ASSERT(false && "unknown quantize op type");
    }
}

void QuantizeOperationOperandCheck(
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand);

/**
 * @brief INT8 Symmetric Quantization operation
 *
 * Formula: int8 = round(fp32 * scale), clamped to [-128, 127]
 *
 * @tparam T QuantizeType (INT8_SYM)
 * @param function The function to add the operation to
 * @param src Input tensor (FP32)
 * @param scale Scale factor tensor
 * @param axis Quantization axis: -1 for per-row, -2 for per-column (default: -1)
 * @return LogicalTensorPtr Output tensor (INT8)
 */
template <QuantizeType T = QuantizeType::INT8_SYM>
LogicalTensorPtr TensorQuantizeSymmetric(
    Function &function,
    LogicalTensorPtr src,
    LogicalTensorPtr scale,
    int64_t axis = -1) {
    static_assert(T == QuantizeType::INT8_SYM,
                  "TensorQuantizeSymmetric only supports INT8_SYM type");

    auto opName = GetQuantizeOpName<T>();
    CheckTensorShape(src, opName);
    CheckTensorShape(scale, opName);

    // Output is INT8 for symmetric quantization
    DataType outDtype = DataType::DT_INT8;
    auto result = std::make_shared<LogicalTensor>(
        function, outDtype, src->shape, src->GetDynValidShape(), src->Format());

    auto &op = function.AddOperation(GetQuantizeOpCode<T>(), {src, scale}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", Element(axis));

    return result;
}

/**
 * @brief INT8 Asymmetric Quantization operation
 *
 * Formula: uint8 = round(fp32 * scale + offset), clamped to [0, 255]
 *
 * @tparam T QuantizeType (INT8_ASYM)
 * @param function The function to add the operation to
 * @param src Input tensor (FP32)
 * @param scale Scale factor tensor
 * @param offset Offset tensor
 * @param axis Quantization axis: -1 for per-row, -2 for per-column (default: -1)
 * @return LogicalTensorPtr Output tensor (UINT8)
 */
template <QuantizeType T = QuantizeType::INT8_ASYM>
LogicalTensorPtr TensorQuantizeAsymmetric(
    Function &function,
    LogicalTensorPtr src,
    LogicalTensorPtr scale,
    LogicalTensorPtr offset,
    int64_t axis = -1) {
    static_assert(T == QuantizeType::INT8_ASYM,
                  "TensorQuantizeAsymmetric only supports INT8_ASYM type");

    auto opName = GetQuantizeOpName<T>();
    CheckTensorShape(src, opName);
    CheckTensorShape(scale, opName);
    CheckTensorShape(offset, opName);

    // Output is UINT8 for asymmetric quantization
    DataType outDtype = DataType::DT_UINT8;
    auto result = std::make_shared<LogicalTensor>(
        function, outDtype, src->shape, src->GetDynValidShape(), src->Format());

    auto &op = function.AddOperation(GetQuantizeOpCode<T>(), {src, scale, offset}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", Element(axis));

    return result;
}

} // namespace npu::tile_fwk
