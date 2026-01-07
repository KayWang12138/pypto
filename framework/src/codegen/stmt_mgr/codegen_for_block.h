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

#include "interface/operation/opcode.h"
#include "interface/tensor/symbolic_scalar.h"

namespace npu::tile_fwk {
extern std::unordered_map<Opcode, std::string> SUPPORT_VF_FUSE_OPS;
struct ForNode {
    std::string loopVar;
    SymbolicScalar start;
    SymbolicScalar extent;
    SymbolicScalar step;
}

class ForBlockManager {
public:
    explict ForBlockManager() = default;
    ~ForBlockManager() = default;

    void UpdateAxesList(const std::vector<SymbolicScalar> &axesList);

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

    bool IsInLoop(){
        return isInLoop_;
    }

    void AddTensorInLoopBody(const std::string &tensorName) { tensorNeedSetAddr_.emplace_back(tensorName); }
    void AddOpInLoopBody(const std::string &op) { opList_.emplace_back(op); }

    std::string Print() const;

private:
    const std::vector<SymbolicScalar> axesList_;
    std::vector<ForNode> forNodes_;
    std::vector<std::string> tensorNeedSetAddr_;
    std::vector<std::string> opList_;
    bool isInLoop_{false};
    bool isLoopStart_{false};
    bool isLoopEnd_{false};
};
} // namespace npu::tile_fwk