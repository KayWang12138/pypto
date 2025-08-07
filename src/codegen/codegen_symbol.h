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
#include "codegen_common.h"

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

    bool BindAddrWithVariableName(const AllocKey &key, const std::string &varName) {
        auto iter = key2VariableName_.find(key);
        if (iter != key2VariableName_.end()) {
            return true;
        } else {
            key2VariableName_.insert(std::pair<AllocKey, std::string>(key, varName));
        }
        return false;
    }

    void AddToTensorMap(int magicNum, std::shared_ptr<LogicalTensor> tensor) {
        tensorMap_.insert(std::pair<int, std::shared_ptr<LogicalTensor>>(magicNum, tensor));
    }

    std::shared_ptr<LogicalTensor> GetTensorByMagic(int magicNum) const {
        auto iter = tensorMap_.find(magicNum);
        if (iter != tensorMap_.end()) {
            return iter->second;
        } else {
            ASSERT(false) << "can not find tensor by magicNum:" << magicNum;
            return nullptr;
        }
    }

    AllocKey CreateAllocKey(std::shared_ptr<LogicalTensor> tensor) const;
    AllocKey CreateAllocKey(int tensorMagicNum) const;

private:
    std::map<AllocKey, std::string> key2VariableName_;
    std::unordered_map<int, std::shared_ptr<LogicalTensor>> tensorMap_;
};

inline SymbolManager::AllocKey SymbolManager::CreateAllocKey(std::shared_ptr<LogicalTensor> tensor) const {
    const auto &memMap = tensor->memorymap;
    if (memMap.count(tensor->subGraphID) == 0) {
        ALOG_ERROR_F("%s: can not find subGraphID(%d) in the memorymap of tensor: ", __FUNCTION__, tensor->subGraphID);
        ALOG_ERROR_F("    %s", tensor->Dump().c_str());
        ALOG_ERROR_F("    memorymap size = %d", memMap.size());

        ASSERT(false);
        return {};
    }

    auto memType = tensor->GetMemoryTypeOriginal();
    if (npu::tile_fwk::OPERAND_TYPE_TO_MEMORY_TYPE.count(memType) == 0) {
        ALOG_ERROR_F("%s: invalid memory type(%d) of tensor: ", __FUNCTION__, static_cast<size_t>(memType));
        ALOG_ERROR_F("    %s", tensor->Dump().c_str());

        ASSERT(false);
        return {};
    }

    const npu::tile_fwk::TileRange &range = memMap.at(tensor->subGraphID);
    auto bufferType = npu::tile_fwk::OPERAND_TYPE_TO_MEMORY_TYPE.at(memType);
    SymbolManager::AllocKey key = SymbolManager::AllocKey(bufferType, range.start, range.end);
    return key;
}

inline SymbolManager::AllocKey SymbolManager::CreateAllocKey(int tensorMagicNum) const {
    std::shared_ptr<LogicalTensor> tensor = SymbolManager::GetTensorByMagic(tensorMagicNum);
    if (!tensor) {
        ALOG_ERROR_F("%s: can not query tensor object from tensor magicnum: %d", __FUNCTION__, tensorMagicNum);
        return {};
    }

    return CreateAllocKey(tensor);
}

inline std::string FormatAllocKey(const SymbolManager::AllocKey &key) {
    constexpr size_t bufSize = 256;
    char keyBuf[bufSize] = {};

    int ret = snprintf_s(keyBuf, sizeof(keyBuf), sizeof(keyBuf) - 1,
        "alloc indentifier <buf_type=%s, range_start=%ld, range_end=%ld>",
        OperandTypeToStr(std::get<static_cast<size_t>(SymbolManager::AllocKeyIdx::BufType)>(key)).c_str(),
        std::get<static_cast<size_t>(SymbolManager::AllocKeyIdx::RangeStart)>(key),
        std::get<static_cast<size_t>(SymbolManager::AllocKeyIdx::RangeEnd)>(key));
    if (ret < 0) {
        ALOG_INFO_F("Format alloc indentifier snprintf_s buffer failed %d", ret);
    }
    return std::string(keyBuf);
}

inline std::string SymbolManager::QueryVariableName(const SymbolManager::AllocKey &key) {
    ALOG_INFO_F("%s: query varname by indentifier: %s", __FUNCTION__, FormatAllocKey(key).c_str());

    auto iter = key2VariableName_.find(key);
    if (iter != key2VariableName_.end()) {
        return iter->second;
    }

    ALOG_ERROR_F("%s: failed to query by indentifier: %s", __FUNCTION__, FormatAllocKey(key).c_str());
    ASSERT(false) << " UNDEFINED_VAR !!! ";
    return "UNDEFINED_VAR";
}

} // namespace npu::tile_fwk
