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

namespace npu::tile_fwk {
struct ForNode {
    std::string loopVar;
    SymbolicScalar start;
    SymbolicScalar extent;
    SymbolicScalar step;
}

class ForBlockManager {
public:
    explict ForBlockManager(const std::vector<SymbolicScalar> &axesList);
    ~ForBlockManager() = default;

    void LoopStart() {
        isLoopStart_ = true;
        isInLoop_ = true;
    }
    void LoopEnd() { isLoopEnd_ = true; }

    void OutLoop() {
        isInLoop_ = false;
        isLoopStart_ = false;
        isLoopEnd_ = false;
    }

private:
    const std::vector<SymbolicScalar> axesList_;
    std::vector<ForNode> forNodes_;
    std::vector<std::string> tensorNeedSetAddr_;
    std::vector<int> opList_;
    int64_t lastGroupId_{INVALID_LOOP_GROUPID};
    int64_t curGroupId{INVALID_LOOP_GROUPID};
    bool isInLoop_{false};
    bool isLoopStart_{false};
    bool isLoopEnd_{false};
};
} // namespace npu::tile_fwk