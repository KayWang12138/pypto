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
 * \file codegen_symbol.h
 * \brief
 */

#pragma once

#include <map>
#include <tuple>
#include <cstdint>
#include <string>

#include "interface/utils/log.h"
#include "tilefwk/error.h"
#include "interface/utils/common.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {
class SymbolManager {
public:
    SymbolManager() = default;
    virtual ~SymbolManager() = default;

    using BufferType = enum OperandType;

    enum class AllocKeyIdx : size_t {
        BufType = 0,
        RangeStart = 1,
        RangeEnd = 2,
    };
    using AllocKey = std::tuple<BufferType, int64_t /*RangeStart*/, int64_t /*RangeEnd*/>;
    using AllocRecord = std::pair<uint64_t /*AllocaAddr*/, unsigned /*AllocaSize*/>;

public:
    virtual std::string QueryVariableName(const AllocKey &key);
    SymbolManager(SymbolManager &other) = delete;

    void operator=(const SymbolManager &other) = delete;

    static SymbolManager *GetInstance() {
        static SymbolManager inst;
        return &inst;
    }

    bool BindAddrWithVariableName(const AllocKey &key, const std::string &varName);

    void AddToTensorMap(int magicNum, std::shared_ptr<LogicalTensor> tensor) {
        tensorMap_.insert(std::pair<int, std::shared_ptr<LogicalTensor>>(magicNum, tensor));
    }

    std::shared_ptr<LogicalTensor> GetTensorByMagic(int magicNum) const;
    AllocKey CreateAllocKey(std::shared_ptr<LogicalTensor> tensor) const;
    AllocKey CreateAllocKey(int tensorMagicNum) const;
    static std::string FormatAllocKey(const SymbolManager::AllocKey &key);

private:
    std::map<AllocKey, std::string> key2VariableName_;
    std::unordered_map<int, std::shared_ptr<LogicalTensor>> tensorMap_;
};
} // namespace npu::tile_fwk
