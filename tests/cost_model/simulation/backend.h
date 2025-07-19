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
#include "simulation/CostModelInterface.h"

namespace npu::tile_fwk {

class CostModelAgent {
public:
    bool getFunctionFromJson = false;
    std::string agentJsonPath = "";
    void BuildCostModel();
    void SubmitToCostModel(Function *rootFunc);
    void SubmitSingleFuncToCostModel(Function *func);
    void SubmitLeafFunctionsToCostModel();
    void RunCostModel();
    void TerminateCostModel();
    void DebugSingleFunc(Function *func);
    void GetFunctionFromJson(const std::string &jsonPath);
    uint64_t GetLeafFunctionTimeCost(uint64_t hash);

private:
    std::shared_ptr<CostModel::CostModelInterface> costModel;
};

} // namespace npu::tile_fwk
