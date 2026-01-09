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
 * \file tune_sync_for_vf.h
 * \brief
 */

#ifndef TUNE_SYNC_FOR_VF_H
#define TUNE_SYNC_FOR_VF_H

#include "passes/pass_interface/pass.h"
#include "interface/program/program.h"
#include "interface/function/function.h"
#include "passes/pass_utils/pass_utils.h"

namespace npu::tile_fwk {
constexpr float vfPrarm = 0.8f;
class TuneSyncForVF : public Pass {
public:
    TuneSyncForVF() : Pass("TuneSyncForVF") {}
    ~TuneSyncForVF() override = default;

    Status RunOnFunction(Function &function) override;

private:
    void ChangeOpSeq(Function *subGraphFunc, bool isAIV1);
    bool NeedAdjustSetFlag(Function *subGraphFunc, Operation *vecTileOp0, Operation *vecTileOp1, Operation *setFlag);
    bool NeedAdjustWaitFlag(Function *subGraphFunc, Operation *vecTileOp0, Operation *vecTileOp1, Operation *waitFlag);
    void AdjustSetWaitFlag(Function *subGraphFunc, std::vector<Operation *> &setFlagList, 
        std::vector<Operation *> &waitFlagList, size_t vecTileOp0Idx, size_t vecTileOp1Idx, int groupNum);
    std::vector<Operation *> &opList_;
    std::vector<std::vector<Operation *>> mergedOps;
};
}
#endif // TUNE_SYNC_FOR_VF_H