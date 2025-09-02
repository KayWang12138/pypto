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
 * \file codegen_utils.cpp
 * \brief
 */

#include "codegen_utils.h"

#include <cstring>
#include <algorithm>

#include "codegen/codegen_common.h"

namespace npu::tile_fwk {
std::string JoinString(const std::vector<std::string> &strList, const std::string &conj) {
    std::ostringstream oss;
    std::string prefix = "/*";
    for (size_t i = 0; i < strList.size(); i++) {
        if (i != 0) {
            if (strList[i - 1].substr(0, prefix.length()) == prefix) {
                oss << " ";
            } else {
                oss << conj;
            }
        }
        oss << strList[i];
    }
    return oss.str();
}

std::string PrintParams(const std::pair<std::string, std::string> &delimiter, const std::vector<std::string> &params,
    const std::string &conj) {
    std::ostringstream oss;
    oss << delimiter.first << JoinString(params, conj) << delimiter.second;
    return oss.str();
}

std::vector<int64_t> NormalizeShape(const std::vector<int64_t> &shapeVec, unsigned dim) {
    std::vector<int64_t> normalizedVec(dim, 1);
    for (size_t i = 0; i < shapeVec.size(); i++) {
        ASSERT(i < dim && "exceed dimension limit!");
        normalizedVec[i] = shapeVec[shapeVec.size() - 1 - i];
    }
    std::reverse(normalizedVec.begin(), normalizedVec.end());
    return normalizedVec;
}

std::string GetTypeForB16B32(const DataType &dtype) {
    if (BytesOf(dtype) == K_BYTES_OF16_BIT) {
        return "uint16_t";
    }
    if (BytesOf(dtype) == K_BYTES_OF32_BIT) {
        return "uint32_t";
    }
    ASSERT(false) << "can not support dtype: " << DataType2String(dtype);
    return {};
}

std::string GetAddrTypeByOperandType(OperandType type) {
    auto iter = OPERAND_TYPE_TO_ADDR_TYPE.find(type);
    if (iter != OPERAND_TYPE_TO_ADDR_TYPE.end()) {
        return iter->second;
    }
    ASSERT(false) << "cannot support current OperandType " << type;
    return "";
}
} // namespace npu::tile_fwk
