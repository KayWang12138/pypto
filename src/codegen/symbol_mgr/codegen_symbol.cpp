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
 * \file codegen_symbol.cpp
 * \brief
 */

#include "codegen_symbol.h"
#include "codegen/codegen_common.h"

namespace npu::tile_fwk {
SymbolManager::AllocKey SymbolManager::CreateAllocKey(std::shared_ptr<LogicalTensor> tensor) const {
    const auto &memMap = tensor->memorymap;
    if (memMap.count(tensor->subGraphID) == 0) {
        ALOG_ERROR_F("%s: can not find subGraphID(%d) in the memorymap of tensor: ", __FUNCTION__, tensor->subGraphID);
        ALOG_ERROR_F("    %s", tensor->Dump().c_str());
        ALOG_ERROR_F("    memorymap size = %d", memMap.size());

        ASSERT(false);
        return {};
    }

    auto memType = tensor->GetMemoryTypeOriginal();
    if (OPERAND_TYPE_TO_MEMORY_TYPE.count(memType) == 0) {
        ALOG_ERROR_F("%s: invalid memory type(%d) of tensor: ", __FUNCTION__, static_cast<size_t>(memType));
        ALOG_ERROR_F("    %s", tensor->Dump().c_str());

        ASSERT(false);
        return {};
    }

    const TileRange &range = memMap.at(tensor->subGraphID);
    auto bufferType = OPERAND_TYPE_TO_MEMORY_TYPE.at(memType);
    SymbolManager::AllocKey key = SymbolManager::AllocKey(bufferType, range.start, range.end);
    return key;
}

SymbolManager::AllocKey SymbolManager::CreateAllocKey(int tensorMagicNum) const {
    std::shared_ptr<LogicalTensor> tensor = SymbolManager::GetTensorByMagic(tensorMagicNum);
    if (!tensor) {
        ALOG_ERROR_F("%s: can not query tensor object from tensor magicnum: %d", __FUNCTION__, tensorMagicNum);
        return {};
    }

    return CreateAllocKey(tensor);
}

bool SymbolManager::BindAddrWithVariableName(const AllocKey &key, const std::string &varName) {
    auto iter = key2VariableName_.find(key);
    if (iter != key2VariableName_.end()) {
        return true;
    } else {
        key2VariableName_.insert(std::pair<AllocKey, std::string>(key, varName));
    }
    return false;
}

std::shared_ptr<LogicalTensor> SymbolManager::GetTensorByMagic(int magicNum) const {
    auto iter = tensorMap_.find(magicNum);
    if (iter != tensorMap_.end()) {
        return iter->second;
    } else {
        ASSERT(false) << "can not find tensor by magicNum:" << magicNum;
        return nullptr;
    }
}

std::string SymbolManager::FormatAllocKey(const SymbolManager::AllocKey &key) {
    std::ostringstream os;
    os << "alloc identifier <buf_type="
       << OperandTypeToStr(std::get<static_cast<size_t>(SymbolManager::AllocKeyIdx::BufType)>(key)).c_str() << ", ";
    os << "range_start=" << std::get<static_cast<size_t>(SymbolManager::AllocKeyIdx::RangeStart)>(key) << ", ";
    os << "range_end=" << std::get<static_cast<size_t>(SymbolManager::AllocKeyIdx::RangeEnd)>(key) << ">";
    return os.str();
}

std::string SymbolManager::QueryVariableName(const SymbolManager::AllocKey &key) {
    ALOG_INFO_F("%s: query varname by identifier: %s", __FUNCTION__, FormatAllocKey(key).c_str());

    auto iter = key2VariableName_.find(key);
    if (iter != key2VariableName_.end()) {
        return iter->second;
    }

    ALOG_ERROR_F("%s: failed to query by identifier: %s", __FUNCTION__, FormatAllocKey(key).c_str());
    ASSERT(false) << " UNDEFINED_VAR !!! ";
    return "UNDEFINED_VAR";
}

} // namespace npu::tile_fwk
