/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quantize.h
 * \brief Quantize operation interface
 */

#pragma once

#include <string>
#include "interface/utils/common.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"

namespace npu::tile_fwk {

enum class QuantizeType {
    INT8_SYM,   // Symmetric quantization: FP32 -> INT8
    INT8_ASYM   // Asymmetric quantization: FP32 -> UINT8
};

// Type conversion: Framework layer QuantizeType -> ISA layer pto::QuantType string
inline const char* QuantizeTypeToISAString(QuantizeType type) {
    switch (type) {
        case QuantizeType::INT8_SYM: return "pto::QuantType::INT8_SYM";
        case QuantizeType::INT8_ASYM: return "pto::QuantType::INT8_ASYM";
        default: ASSERT(false && "unknown quantize type"); return "";
    }
}

template <QuantizeType T>
std::string GetQuantizeOpName() {
    switch (T) {
        case QuantizeType::INT8_SYM: return "INT8_SYM";
        case QuantizeType::INT8_ASYM: return "INT8_ASYM";
        default: ASSERT(false && "unknown quantize type"); return "";
    }
}

template <QuantizeType T>
Opcode GetQuantizeOpCode() {
    switch (T) {
        case QuantizeType::INT8_SYM: return Opcode::OP_QUANTIZE_SYM;
        case QuantizeType::INT8_ASYM: return Opcode::OP_QUANTIZE_ASYM;
        default: ASSERT(false && "unknown quantize type"); return Opcode::OP_QUANTIZE_SYM;
    }
}

// Parameter validation
void CheckQuantize(const LogicalTensorPtr &input, const LogicalTensorPtr &scale,
                   DataType otype, int axis, const LogicalTensorPtr &zeroPoints);

// Logical tensor operation (non-tiled)
LogicalTensorPtr TensorQuantizeOperation(Function &function, const LogicalTensorPtr &input,
                                        const LogicalTensorPtr &scale, DataType otype,
                                        int axis, const LogicalTensorPtr &zeroPoints);

// Tile operation implementation (for TILE_GRAPH)
template <QuantizeType quantType>
void TiledQuantizeOperation(Function &function, const TileShape &tileShape,
                            const LogicalTensorPtr &input, const LogicalTensorPtr &scale,
                            const LogicalTensorPtr &result, const LogicalTensorPtr &zeroPoints = nullptr, int axis = -1);

// Tile func wrappers for registration
void QuantizeSymOperationTileFunc(Function &function, const TileShape &tileShape,
                                  const std::vector<LogicalTensorPtr> &iOperand,
                                  const std::vector<LogicalTensorPtr> &oOperand,
                                  const Operation &op);

void QuantizeAsymOperationTileFunc(Function &function, const TileShape &tileShape,
                                   const std::vector<LogicalTensorPtr> &iOperand,
                                   const std::vector<LogicalTensorPtr> &oOperand,
                                   const Operation &op);

} // namespace npu::tile_fwk
