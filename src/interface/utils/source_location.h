/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <stack>

namespace npu::tile_fwk {

class SourceLocation {
public:
    SourceLocation() = default;
    SourceLocation(const std::string &fname, int lineno) : fname_(fname), lineno_(lineno) {}
    explicit SourceLocation(uint64_t pc) : fname_("??"), lineno_(-1), pc_(pc){};

    int GetLineno() const;
    std::string GetFileName() const;
    uint64_t GetPC() const { return pc_; }

    static void SetLocation(std::shared_ptr<SourceLocation> loc = nullptr) {
        if (loc)
            callStack.push(loc);
        else if (callStack.size())
            callStack.pop();
    }

    static std::shared_ptr<SourceLocation> GetLocation() {
        if (callStack.size())
            return callStack.top();
        return nullptr;
    }
    static void Init(std::vector<std::shared_ptr<SourceLocation>> locs);

private:
    void Init() const;

private:
    mutable std::string fname_;
    mutable int lineno_;
    uint64_t pc_;
    static std::stack<std::shared_ptr<SourceLocation>> callStack;
};

using SourceLocationPtr = std::shared_ptr<SourceLocation>;

} // namespace npu::tile_fwk