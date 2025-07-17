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
#include "interface/utils/assert.h"
#include "interface/utils/common.h"

namespace npu::tile_fwk {
class SymbolManager {
public:
    SymbolManager() = default;
    virtual ~SymbolManager() = default;

    using BufferType = enum OperandType;

    using AllocKey = std::tuple<BufferType, int64_t /*RangeStart*/, int64_t /*RangeEnd*/>;
    using AllocRecord = std::pair<uint64_t /*AllocaAddr*/, unsigned /*AllocaSize*/>;

public:
    SymbolManager(SymbolManager &other) = delete;

    void operator=(const SymbolManager &other) = delete;

    static SymbolManager *GetInstance() {
        static SymbolManager inst;
        return &inst;
    }

    bool BindAddrWithVariableName(const AllocKey &key, const std::string &varName) {
        auto iter = key2VariableName.find(key);
        if (iter != key2VariableName.end()) {
            return true;
        } else {
            key2VariableName.insert(std::pair<AllocKey, std::string>(key, varName));
        }
        return false;
    }

    virtual std::string QueryVariableName(const AllocKey &key) {
        ALOG_INFO << __FUNCTION__ << ": query varname by key: " << key;

        auto iter = key2VariableName.find(key);
        if (iter != key2VariableName.end()) {
            return iter->second;
        }

        ALOG_ERROR << __FUNCTION__ << ": failed to query by key: " << key;
        ASSERT(false) << " UNDEFINED_VAR !!! ";
        return "UNDEFINED_VAR";
    }

private:
    std::map<AllocKey, std::string> key2VariableName;
};

inline std::ostream &operator<<(std::ostream &os, npu::tile_fwk::SymbolManager::AllocKey key) {
    os << "AllocKey<buf_type = "
       << npu::tile_fwk::OperandTypeToStr(std::get<0>(key)) /* 0 is BUF_TYPE_IDX */
       << ", range_start = " << std::get<1>(key) /* 1 is RANGE_START_IDX */
       << ", range_end = " << std::get<2>(key) /* 2 is RANGE_END_IDX */
       << ">";
    return os;
}
} // namespace npu::tile_fwk
