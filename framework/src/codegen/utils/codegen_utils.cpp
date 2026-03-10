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
 * \file codegen_utils.cpp
 * \brief
 */

#include "codegen_utils.h"

#include <cstring>
#include <algorithm>
#include <unistd.h>

#include "codegen/codegen_common.h"
#include "interface/utils/log.h"

namespace npu::tile_fwk {
std::vector<int64_t> NormalizeShape(const std::vector<int64_t> &shapeVec, unsigned dim) {
    std::vector<int64_t> normalizedVec(dim, 1);
    for (size_t i = 0; i < shapeVec.size(); i++) {
        ASSERT(OperErr::TENSOR_DIM_EXCEEDED, i < dim) << "exceed dimension limit!";
        normalizedVec[i] = shapeVec[shapeVec.size() - 1 - i];
    }
    std::reverse(normalizedVec.begin(), normalizedVec.end());
    return normalizedVec;
}

std::string FormatFloat(const std::variant<int64_t, uint64_t, double> &v, DataType dtype, int precision) {
    // 定义处理函数
    auto apply = [&](auto &&val) -> std::string {
        std::ostringstream oss;
        using T = std::decay_t<decltype(val)>;
        if constexpr (std::is_same_v<T, double>) {
            if ((dtype == DataType::DT_FP32 || dtype == DataType::DT_FP16 || dtype == DataType::DT_BF16) &&
                (std::isinf(val) || std::isnan(val))) {
                FloatSpecVal fsv = {dtype, val};
                oss << fsv.GetFsVarName() << ".f";
                return oss.str();
            }
        }
        oss << std::setprecision(precision) << val;
        return oss.str();
    };

    return std::visit(apply, v);
}

std::string GetTypeForB16B32(const DataType &dtype) {
    if (BytesOf(dtype) == K_BYTES_OF16_BIT) {
        return "uint16_t";
    }
    if (BytesOf(dtype) == K_BYTES_OF32_BIT) {
        return "uint32_t";
    }
    ASSERT(GenCodeErr::DATA_TYPE_UNSUPPORTED, false) << "can not support dtype: " << DataType2String(dtype);
    return {};
}

std::string GetAddrTypeByOperandType(OperandType type) {
    auto iter = OPERAND_TYPE_TO_ADDR_TYPE.find(type);
    if (iter != OPERAND_TYPE_TO_ADDR_TYPE.end()) {
        return iter->second;
    }
    ASSERT(OperErr::OPERAND_TYPE_UNSUPPORTED, false) << "cannot support current OperandType " << type;
    return "";
}

int64_t CalcLinearOffset(const std::vector<int64_t> &shape, const std::vector<int64_t> &offset) {
    if (shape.empty() || offset.empty() || shape.size() != offset.size()) {
        CODEGEN_LOGE_E(GenCodeErr::TENSOR_SHAPE_INVALID, "Invalid Input! shape: %s, offset: %s",
            IntVecToStr(shape).c_str(), IntVecToStr(offset).c_str());
        return 0;
    }

    int64_t resOffset{0};
    int64_t base = 1;
    for (int i = static_cast<int>(offset.size()) - 1; i >= 0; i--) {
        resOffset += offset[i] * base;
        base *= shape[i];
    }

    return resOffset;
}

void PrintIndent(std::ostringstream &os, int scopeLevel) {
    for (int i = 0; i < scopeLevel; i++) {
        os << "  ";
    }
}

} // namespace npu::tile_fwk
