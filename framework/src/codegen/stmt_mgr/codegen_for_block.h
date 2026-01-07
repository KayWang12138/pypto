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
 * \file codegen_for_block.h
 * \brief
 */

#pragma once

#include <vector>
struct ForNode {
    std::string loopVar;
    SymbolicScalar start;
    SymbolicScalar extent;
    SymbolicScalar step;
}

class ForBlockManager {
public:
    ForBlockManager() = default;
    ~ForBlockManager() = default;

private:
    std::vector<ForNode> forNodes;
    std::vector<int> tensorNeedSetAddr;
    std::vector<int> opList;
    int64_t lastGroupId{-1};
    int64_t curGroupId{-1};
};