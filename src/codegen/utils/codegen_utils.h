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
 * \file codegen_utils.h
 * \brief
 */

#ifndef CODEGEN_UTILS_H
#define CODEGEN_UTILS_H

#include <iostream>
#include <vector>

#include "codegen/codegen_common.h"
#include "tilefwk/data_type.h"
#include "interface/utils/common.h"
#include "interface/operation/opcode.h"

namespace npu::tile_fwk {
template <typename T>
inline void FillIntVecWithDummyInHead(std::vector<T> &input, unsigned padNum, T dummy) {
    for (unsigned i = 0; i < padNum; ++i) {
        input.insert(input.begin(), dummy);
    }
}

template <typename T = std::string>
std::string JoinString(const std::vector<T> &strList, const std::string &conj) {
    std::ostringstream oss;
    std::string prefix = "/*";
    for (size_t i = 0; i < strList.size(); i++) {
        if (i != 0) {
            if (std::is_same_v<T, std::string> && strList[i - 1].substr(0, prefix.length()) == prefix) {
                oss << " ";
            } else {
                oss << conj;
            }
        }
        oss << strList[i];
    }
    return oss.str();
}

template <typename T = std::string>
std::string PrintParams(
    const std::pair<std::string, std::string> &delimiter, const std::vector<T> &params, const std::string &conj) {
    std::ostringstream oss;
    oss << delimiter.first << JoinString<T>(params, conj) << delimiter.second;
    return oss.str();
}

std::vector<int64_t> NormalizeShape(const std::vector<int64_t> &shapeVec, unsigned dim);

inline std::vector<int64_t> ToVecInt(const std::vector<int64_t> &shapeVec) {
    std::vector<int64_t> vec;
    for (auto &i : shapeVec) {
        vec.push_back(static_cast<int64_t>(i));
    }
    return vec;
}
std::string GetTypeForB16B32(const DataType &dtype);

inline std::string GetPipeId(PipeType queue) {
    auto res = PIPE_ID.find(queue);
    return res == PIPE_ID.end() ? "" : res->second;
}

inline std::string GetTileOpName(Opcode opCode) {
    const auto &opCfg = OpcodeManager::Inst().GetTileOpCfg(opCode);
    return opCfg.tileOpCode_;
}

std::string GetAddrTypeByOperandType(OperandType type);

} // namespace npu::tile_fwk
#endif
