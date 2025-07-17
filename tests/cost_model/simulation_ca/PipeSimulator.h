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
 * \file PipeSimulator.h
 * \brief
 */

#pragma once

#include <unordered_map>
#include "simulation/arch/PipeMachineImpl.h"
#include "simulation_ca/A2A3/SimulatorA2A3.h"

namespace CostModel
{
    template <typename Simulator>
    class PipeSimulator : public PipeMachineImpl
    {
    public:
        uint64_t Simulate(const TileOpPtr& tileOp) override;
        uint64_t PostSimulate(const TileOpPtr &tileOp) override;
    private:
        std::unordered_map<std::string, uint64_t> tileopLatencyCacheMp;
    };

    extern "C" UnifiedPipeMachinePtr CreatePipeSimulatorSimulatorA2A3();
} // namespace CostModel
