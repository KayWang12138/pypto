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
 * \file backend.h
 * \brief
 */

#pragma once

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "cost_model/simulation/CostModelInterface.h"

namespace npu::tile_fwk {

class CostModelAgent {
public:
    bool getFunctionFromJson = false;
    std::string agentJsonPath = "";
    std::string topoJsonPath = "";
    void BuildCostModel();
    void SubmitToCostModel(Function *rootFunc);
    void SubmitSingleFuncToCostModel(Function *func);
    void SubmitLeafFunctionsToCostModel();

    uint64_t seqPos = 0;
    uint64_t taskIdPos = 1;
    uint64_t rootIndexPos = 2;
    uint64_t leafIndexPos = 3;
    uint64_t opmagicPos = 4;
    uint64_t coreTypePos = 5;
    uint64_t psgIdPos = 6;
    uint64_t funcHashPos = 7;
    uint64_t succStartPos = 8;
    uint64_t seqNumOffset = 32;
    Json ParseDynTopo(std::string &path);
    void SubmitTopo(std::string &path);
    void RunCostModel();
    void TerminateCostModel();
    void DebugSingleFunc(Function *func);
    void GetFunctionFromJson(const std::string &jsonPath);
    uint64_t GetLeafFunctionTimeCost(uint64_t hash);

private:
    std::shared_ptr<CostModel::CostModelInterface> costModel;
};

} // namespace npu::tile_fwk
