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
 * \file CallPipeImpl.h
 * \brief
 */

#pragma once

#include "PipeMachineImpl.h"

namespace CostModel
{
    class CallPipeImpl : public PipeMachineImpl
    {
    public:
        uint64_t Simulate(const TileOpPtr &tileOp) override {
            if (tileOp->IsCall()) {
                return 1;
            }
            return 1;
        }
        uint64_t PostSimulate(const TileOpPtr &tileOp) override {
            if (tileOp->IsCall()) {
                return 1;
            }
            return 1;
        }
    };

    inline UnifiedPipeMachinePtr CreateCallPipeImpl() {
        return UnifiedPipeMachinePtr(new CallPipeImpl());
    }
} // namespace CostModel
