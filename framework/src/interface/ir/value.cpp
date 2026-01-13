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
 * \file value.cpp
 * \brief
 */
#include "ir/utils.h"
#include "ir/value.h"

#include <ostream>
#include <variant>
#include <unordered_map>

namespace pto {

static std::unordered_map<MemSpaceKind, std::string> memSpaceNameDict = {
    {MemSpaceKind::DDR,   "DDR"},
    {MemSpaceKind::L2,    "L2"},
    {MemSpaceKind::UB,    "UB"},
    {MemSpaceKind::L1,    "L1"},
    {MemSpaceKind::L0A,   "L0A"},
    {MemSpaceKind::L0B,   "L0B"},
    {MemSpaceKind::L0C,   "L0C"},
    {MemSpaceKind::REG,   "REG"},
    {MemSpaceKind::SHMEM, "SHMEM"},
};
std::string GetMemSpaceKindName(MemSpaceKind kind) {
    if (memSpaceNameDict.count(kind)) {
        return memSpaceNameDict[kind];
    } else {
        return "UNKNOWN";
    }
}

// ========== Value System Implementation ==========

int64_t ScalarValue::GetInt64Value() const {
    if (!HasImmediateValue()) {
        throw std::runtime_error("ScalarValue does not hold a constant value");
    }
    return std::visit([](const auto& val) -> int64_t {
        return static_cast<int64_t>(val);
    }, immediateValue_);
}
} // namespace pto
