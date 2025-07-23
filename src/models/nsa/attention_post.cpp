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
 * \file attention_post.cpp
 * \brief
 */
#include "attention_post.h"

#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "tilefwk/tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/common.h"

namespace npu::tile_fwk {

// b and s is dynamic, support:
// b: 16, 32, 64, 24, 48, 96
// s: 1, 2
void PostCompute(Tensor &input, Tensor &weightUV, Tensor &weightO, Tensor &weightOScale, Tensor &smoothScalesWo,
                 const PostTileConfig &tileConfig, Tensor &postOut) {
    // input: [b,s,n,kvLoraRank], fp16/bf16
    // weightUV: [n,kvLoraRank,vHeadDim], fp16/bf16
    // weightO: [v*kvLoraRank,h], fp16/bf16/int8
    // weightOScale: [1,h], fp32
    // params check
    assert(input->shape.size() == SHAPE_DIM4 && weightUV->shape.size() == SHAPE_DIM3
           && weightO->shape.size() == SHAPE_DIM2);
    auto dtype = weightUV->Datatype();
    auto n = weightUV->shape[0];
    auto kvLoraRank = weightUV->shape[1];
    auto vHeadDim = weightUV->shape[2];
    auto h = weightO->shape[1];

    int tileB = tileConfig.tileB;
    int tileS = tileConfig.tileS;
    int tileBS = tileB * tileS;

    bool isQuant = (weightOScale.GetStorage() != nullptr);
    bool isSmooth = (smoothScalesWo.GetStorage() != nullptr);

    int b = input->shape[0];  // SymbolicScalar b = GetInputShapeDim(input, 0);
    int s = input->shape[1];  // SymbolicScalar s = GetInputShapeDim(input, 1);
    SymbolicScalar bLoop = b / tileB;
    SymbolicScalar sLoop = s / tileS;

    LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bLoop, 1), {}, true) {
        SymbolicScalar bOffset = bIdx * tileB;
        LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sLoop, 1)) {
            SymbolicScalar sOffset = sIdx * tileS;
            std::vector<SymbolicScalar> outOffset = {bOffset, sOffset, 0};

            Program::GetInstance().GetTileShape().SetVecTileShapes({1, 1, 32, kvLoraRank});
            auto inputView = DView(input, {tileB, tileS, n, kvLoraRank}, {bOffset, sOffset, 0, 0});
            ConfigManager::Instance().SetSemanticLabel("postReshape1");
            auto inputRes = Reshape(inputView, {tileBS, n, kvLoraRank});
            Program::GetInstance().GetTileShape().SetVecTileShapes({std::min(32, tileBS), 2, kvLoraRank});
            ConfigManager::Instance().SetSemanticLabel("postTranspose1");
            auto inputTrans = Transpose(inputRes, {0, 1});  // [n,tileBS,kvLoraRank]

            ConfigManager::Instance().SetSemanticLabel("postBmm");
            int c0 = 16;
            int m = (std::min(32, tileBS) + c0 - 1) / c0 * c0;
            Program::GetInstance().GetTileShape().SetCubeTileShapes({m, m},
                {std::min(256, kvLoraRank), std::min(512, kvLoraRank)},
                {vHeadDim, vHeadDim}, true);
            // [n,tileBS,kvLoraRank] @ [n,kvLoraRank,vHeadDim] -> [n,tileBS,vHeadDim]
            auto bmm = Matrix::BatchMatmul(dtype, inputTrans, weightUV);

            ConfigManager::Instance().SetSemanticLabel("postTranspose2");
            Program::GetInstance().GetTileShape().SetVecTileShapes({4, std::min(32, tileBS), vHeadDim});
            auto bmmTrans = Transpose(bmm, {0, 1}); // [n,tileBS,vHeadDim] -> [tileBS,n,vHeadDim]
            ConfigManager::Instance().SetSemanticLabel("postReshape2");
            auto bmmRes = Reshape(bmmTrans, {tileBS, n * vHeadDim});

            if (isQuant) {
                ConfigManager::Instance().SetSemanticLabel("postQuant");
                Program::GetInstance().GetTileShape().SetVecTileShapes({1, n * vHeadDim});
                std::tuple<Tensor, Tensor> quantRes;
                if (isSmooth) {
                    quantRes = Quant(bmmRes, true, true, smoothScalesWo);
                } else {
                    quantRes = Quant(bmmRes, true, false);
                }
                auto bmmResQuant = std::get<0>(quantRes); // [tileBS, n*vHeadDim], int8
                auto scaleDequant = std::get<1>(quantRes); // [tileBS, 1], fp32

                ConfigManager::Instance().SetSemanticLabel("postMm");
                Program::GetInstance().GetTileShape().SetCubeTileShapes({m, m},
                    {std::min(512, n * vHeadDim), std::min(512, n * vHeadDim)},
                    {std::min(64, h), std::min(64, h)}, true);
                // [tileBS, n*vHeadDim] @ [n*vHeadDim, h] -> [tileBS, h], int8 @ int8 -> int32
                Tensor mm = Matrix::Matmul(DataType::DT_INT32, bmmResQuant, weightO);

                ConfigManager::Instance().SetSemanticLabel("postDequant");
                Program::GetInstance().GetTileShape().SetVecTileShapes(
                    {std::min(32, tileBS), std::min(32, h)});
                Tensor res = Cast(mm, DataType::DT_FP32);
                res = Mul(res, scaleDequant);   // [tileBS, h] * [tileBS, 1] -> [tileBS, h]
                res = Mul(res, weightOScale);   // [tileBS, h] * [1, h] -> [tileBS, h]
                Tensor dequantRes = Cast(res, dtype, CAST_RINT);

                ConfigManager::Instance().SetSemanticLabel("postReshape3");
                auto postOutView = Reshape(dequantRes, {tileB, tileS, h});
                Program::GetInstance().GetTileShape().SetVecTileShapes({1, 1, h});
                DAssemble(postOutView, outOffset, postOut);
            } else {
                Program::GetInstance().GetTileShape().SetCubeTileShapes({m, m},
                    {std::min(512, n * vHeadDim), std::min(512, n * vHeadDim)},
                    {std::min(64, h), std::min(64, h)}, true);
                // [tileBS, n*vHeadDim] @ [n*vHeadDim, h] -> [tileBS, h], dtype @ dtype -> dtype
                Tensor mm = Matrix::Matmul(dtype, bmmRes, weightO);

                ConfigManager::Instance().SetSemanticLabel("postReshape3");
                auto postOutView = Reshape(mm, {tileB, tileS, h});
                Program::GetInstance().GetTileShape().SetVecTileShapes({1, 1, h});
                DAssemble(postOutView, outOffset, postOut);
            }
        }
    }
}

void AttentionPost(Tensor &input, Tensor &weightUV, Tensor &weightO, Tensor &weightOScale, Tensor &smoothScalesWo,
                   const PostTileConfig &tileConfig, Tensor &postOut) {
    FUNCTION("POST_MAIN", FunctionType::DYNAMIC, {input, weightUV, weightO, weightOScale, smoothScalesWo}, {postOut}) {
        PostCompute(input, weightUV, weightO, weightOScale, smoothScalesWo, tileConfig, postOut);
    }
}

} // namespace npu::tile_fwk
