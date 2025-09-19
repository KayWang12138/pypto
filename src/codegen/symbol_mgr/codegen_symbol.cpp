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
AllocKey SymbolManager::CreateAllocKey(const std::shared_ptr<LogicalTensor> &tensor) const {
    const auto &memMap = tensor->memorymap;
    if (memMap.size() == 0) {
        ALOG_ERROR_F("%s: empty memorymap in tensor: ", __FUNCTION__);
        ALOG_ERROR_F("    %s", tensor->Dump().c_str());

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

    const TileRange &range = memMap.begin()->second;
    auto bufferType = OPERAND_TYPE_TO_MEMORY_TYPE.at(memType);
    AllocKey key = AllocKey(bufferType, range.start, range.end);
    return key;
}

AllocKey SymbolManager::CreateAllocKey(int tensorMagicNum) const {
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

std::string SymbolManager::FormatAllocKey(const AllocKey &key) {
    auto [bufType, start, end] = key;
    std::ostringstream os;
    os << "alloc identifier <buf_type=" << OperandTypeToStr(bufType) << ", ";
    os << "range_start=" << start << ", ";
    os << "range_end=" << end << ">";
    return os.str();
}

std::string SymbolManager::QueryVariableName(const AllocKey &key) {
    ALOG_INFO_F("%s: query varname by identifier: %s", __FUNCTION__, FormatAllocKey(key).c_str());

    auto iter = key2VariableName_.find(key);
    if (iter != key2VariableName_.end()) {
        return iter->second;
    }

    ALOG_ERROR_F("%s: failed to query by identifier: %s", __FUNCTION__, FormatAllocKey(key).c_str());
    ASSERT(false) << " UNDEFINED_VAR !!! ";
    return "UNDEFINED_VAR";
}

std::string SymbolManager::QueryVarNameByTensorMagic(int magic) {
    AllocKey key = CreateAllocKey(magic);
    std::string varName = QueryVariableName(key);
    return varName;
}

std::string SymbolManager::AddTileTensorUsing(const TileTensorUsing &tileTensorUsing) {
    std::string tensorUsingType = tileTensorUsing.GenName();
    tileTensorUsing_.insert({tensorUsingType, tileTensorUsing});
    return tensorUsingType;
}

void SymbolManager::AddTileTensor(const TileTensor &tileTensor) {
    auto result = tileTensor_.insert({tileTensor, tileTensor.tensorName});
    if (result.second) {
        tileTensorByMagic_.insert({tileTensor.magic, tileTensor.tensorName});
    } else {
        tileTensorByMagic_.insert({tileTensor.magic, result.first->second});
    }
}

std::string SymbolManager::QueryTileTensorByMagic(int magic) {
    auto iterByMagic = tileTensorByMagic_.find(magic);
    if (iterByMagic != tileTensorByMagic_.end()) {
        return iterByMagic->second;
    }

    ASSERT(false) << "tensor magic " << magic << " is not found !!! ";
    return "";
}

std::string SymbolManager::GenUsingList() {
    std::ostringstream oss;
    for (const auto &usingPair : tileTensorUsing_) {
        const std::string &usingName = usingPair.first;
        const TileTensorUsing &tileTensorUsing = usingPair.second;
        oss << "using " << usingName << " = " << tileTensorUsing.ToString();
    }
    return oss.str();
}

std::string SymbolManager::GenTileTensorDefList() {
    std::ostringstream oss;
    for (const auto &tensorPair : tileTensor_) {
        const TileTensor &tileTensor = tensorPair.first;
        oss << tileTensor.ToString();
    }
    return oss.str();
}

} // namespace npu::tile_fwk
