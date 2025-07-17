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
 * \file calibrate_cycles.h
 * \brief
 */

#ifndef TILE_FWK_CALIBRATE_CYCLES_H
#define TILE_FWK_CALIBRATE_CYCLES_H

#include "interface/tensor/logical_tensor.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "passes/pass_interface/pass.h"

namespace npu::tile_fwk {

class CalibratedCyclesPass : public Pass {
public:
    CalibratedCyclesPass() : Pass("CalibratedCyclesPass") {}
    ~CalibratedCyclesPass() override {}

private:
    Status RunOnFunction(Function &function) override;

    bool LoadCalibratedCyclesCSV(const std::string csvFilePath,
        std::unordered_map<unsigned long, int>& cyclesMap);
};
}

#endif // TILE_FWK_CALIBRATE_CYCLES_H