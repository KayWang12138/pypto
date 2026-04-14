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
 * \file codegen_op.cpp
 * \brief
 */

#include "codegen_op_cloudnpu.h"

#include <algorithm>

#include "codegen/codegen_common.h"
#include "codegen/utils/codegen_utils.h"
#include "securec.h"

namespace npu::tile_fwk {

CodeGenOpCloudNPU::CodeGenOpCloudNPU(const CodeGenOpNPUCtx& ctx) : CodeGenOpNPU(ctx)
{
    mteFixPipeOps_ = mteFixPipeOpsCloudNPU_;
    unaryOps_ = unaryOpsCloudNPU_;
    binaryOps_ = binaryOpsCloudNPU_;
    compositeOps_ = compositeOpsCloudNPU_;
    sortOps_ = sortOpsCloudNPU_;
    cubeOps_ = cubeOpsCloudNPU_;
    syncOps_ = syncOpsCloudNPU_;
    distributeOps_ = distributeOpsCloudNPU_;
    gatherScatterOps_ = gatherScatterOpsCloudNPU_;
    normalVecOps_ = normalVecOpsCloudNPU_;
    perfOps_ = perfOpsCloudNPU_;
    aicpuOps_ = aicpuOpsCloudNPU_;

    InitOpsGenMap();
    forBlkMgr_ = ctx.forBlockManager;
    CodeGenOp::Init(ctx.operation);
    UpdateTileTensorInfo();
    UpdateLoopInfo();
}

} // namespace npu::tile_fwk
