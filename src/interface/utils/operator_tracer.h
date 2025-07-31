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

#include <sys/cdefs.h>
#include <cstdint>
#include "source_location.h"

namespace npu::tile_fwk {
class OperatorChecker {
public:
    void PreCheck();
    void PostCheck();

private:
    int preOpCount;
    int preMagic;
    int preOp;
    int preRawMagic;
    int enable;
};

struct OperatorTracer {
    OperatorTracer(const void *lr) {
        if (enableSourceLocation) {
            // lr is return address, we need find caller address, minus 4 here
            SourceLocation::SetLocation(std::make_shared<SourceLocation>((uint64_t)lr - 4));
        }
        if (enableChecker) {
            checker.PreCheck();
        }
    }
    ~OperatorTracer() {
        if (enableSourceLocation)
            SourceLocation::SetLocation(nullptr);
        if (enableChecker)
            checker.PostCheck();
    }
    OperatorChecker checker;

    static void EnableChecker(bool val) { enableChecker = val; }
    static void EnableSourceLocation(bool val) { enableSourceLocation = val; }

    static bool enableChecker;
    static bool enableSourceLocation;
};
} // namespace npu::tile_fwk

#define DECLARE_TRACER()
#define DECLARE_TRACER1()