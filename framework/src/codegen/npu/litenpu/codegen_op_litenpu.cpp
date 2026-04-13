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
 * \file codegen_op.cpp
 * \brief
 */

#include "codegen_op_litenpu.h"

#include <algorithm>

#include "codegen/codegen_common.h"
#include "securec.h"

namespace npu::tile_fwk {
// // ensure funcType is static, and isUnderDynamicFunc is false
// CodeGenOpLiteNPU::CodeGenOpLiteNPU(const CodeGenOpCtx &ctx)
//     : CodeGenOp(ctx) {
//     CodeGenOp::Init(ctx.operation);
//     UpdateTileTensorInfo();
// }

// void CodeGenOpLiteNPU::UpdateTileTensorShapeAndStride(int paramIdx, TileTensor &tileTensor, bool isSpillToGm) {
//     (void)isSpillToGm; // TODO...
//     auto newOriginShape = originShape[paramIdx];
//     auto newRawShape = rawShape[paramIdx];
//     auto newDynValidShape = dynamicValidShape[paramIdx];
//     CODEGEN_LOGI("newOriginShape is %s, newRawShape is %s, newDynValidShape is %s", IntVecToStr(newOriginShape).c_str(),
//         IntVecToStr(newRawShape).c_str(), IntVecToStr(newDynValidShape).c_str());

//     tileTensor.rawShape = newRawShape;

//     // ---- static ----
//     if (functionType == FunctionType::STATIC) {
//         for (auto s : newOriginShape) {
//             tileTensor.shape.emplace_back(std::to_string(s));
//         }
//         tileTensor.stride = BuildStride(newRawShape);
//         return;
//     }
// }

// TileTensor CodeGenOpLiteNPU::QueryTileTensorByIdx(int paramIdx) const
// {
//     const int tensorMagic = operandWithMagic[paramIdx];
//     const int opMagic = originalOp.GetOpMagic();
//     const TileTensor* tileTensor = nullptr;
//     // bool isInLoop = forBlkMgr_ != nullptr && forBlkMgr_->IsInLoop();
//     // if (isInLoop) {
//     //     tileTensor = sm->QueryTileTensorInLoopByMagic(tensorMagic, opMagic);
//     //     // some tensor in loop is reused same tensor out of loop
//     //     if (tileTensor == nullptr) {
//     //         tileTensor = sm->QueryTileTensorByMagic(tensorMagic, opMagic);
//     //     }
//     // } else {
//         tileTensor = sm->QueryTileTensorByMagic(tensorMagic, opMagic);
//     // }

//     if (tileTensor != nullptr) {
//         CODEGEN_LOGI("QueryTileTensorByIdx found: %s", tileTensor->ToString().c_str());
//         return *tileTensor;
//     }

//     // ASSERT(GenCodeErr::TENSOR_NOT_FOUND, false)
//     //     << "TileTensor: paramIdx " << paramIdx << ", tensor magic " << tensorMagic << ", op magic " << opMagic
//     //     << ", isInLoop " << isInLoop << " is not found !!!";
//     ASSERT(GenCodeErr::TENSOR_NOT_FOUND, false)
//         << "TileTensor: paramIdx " << paramIdx << ", tensor magic " << tensorMagic << ", op magic " << opMagic
//         << " is not found !!!";
//     static TileTensor emptyTileTensor;
//     return emptyTileTensor;
// }

CodeGenOpLiteNPU::CodeGenOpLiteNPU(const CodeGenOpNPUCtx& ctx) : CodeGenOpNPU(ctx) {}

} // namespace npu::tile_fwk
