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
 * \file symbol_handler.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

namespace npu::tile_fwk {

enum class SymbolHandlerId : uint64_t {
    GetInputShapeDimSize,
    GetInputShapeDim,
    GetInputDataInt32Dim1,
    GetInputDataInt32Dim2,
    GetInputDataInt32Dim3,
    GetViewValidShapeDim,
    IsLoopBegin,
    IsLoopEnd
};

const std::unordered_map<std::string, SymbolHandlerId> symbolHandlerIndexDict = {
    {"GetInputShapeDimSize", SymbolHandlerId::GetInputShapeDimSize},
    {"GetInputShapeDim",  SymbolHandlerId::GetInputShapeDim},
    {"GetInputDataInt32Dim1", SymbolHandlerId::GetInputDataInt32Dim1},
    {"GetInputDataInt32Dim2", SymbolHandlerId::GetInputDataInt32Dim2},
    {"GetInputDataInt32Dim3", SymbolHandlerId::GetInputDataInt32Dim3},
    {"GetViewValidShapeDim", SymbolHandlerId::GetViewValidShapeDim},
    {"IsLoopBegin", SymbolHandlerId::IsLoopBegin},
    {"IsLoopEnd", SymbolHandlerId::IsLoopEnd}
};

struct SymbolHandler {
    SymbolHandler(SymbolHandlerId id, uint64_t index) : handlerId(id), symIndex(index) {}
    SymbolHandlerId handlerId;
    uint64_t symIndex;

    static std::string GetNameByHandlerId(SymbolHandlerId tmpHandlerId) {
        for (auto &[name, id] : symbolHandlerIndexDict) {
            if (id == tmpHandlerId) {
                return name;
            }
        }
        return "";
    }
};
} // namespace npu::tile_fwk
