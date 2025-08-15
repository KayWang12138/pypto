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
 * \file SimplifiedMemoryAllocator.h
 * \brief
 */

#pragma once
 
#include "codegen/symbol_mgr/codegen_symbol.h"
 
namespace CostModel {
class SimplifiedMemoryAllocator : public npu::tile_fwk::SymbolManager {
public:
    ~SimplifiedMemoryAllocator() override = default;
    std::string QueryVariableName(const AllocKey &key) override {
        (void)key;
        return "charArray" + std::to_string(++memoryId);
    }

private:
    int memoryId = 0;
};

} // CostModel
