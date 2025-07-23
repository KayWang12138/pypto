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
 * \file dynamic_nsa.cpp
 * \brief
 */

#include "models/nsa/dynamic_nsa_v1.h"

#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "tilefwk/tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/common.h"

namespace npu::tile_fwk {

void GenGatedScore(const Tensor &x, const Tensor &gateW1, const Tensor &gateW2, const Tensor &gateSimW1,
    Tensor& gatingScore, GateMode gateMode) {
    (void)gateSimW1;
    (void)gateMode;
    DataType dType = x->Datatype();

    int b = x->shape[0];
    int s = x->shape[1];
    int h = x->shape[2];
    int n1 = gateW2->shape[1] / 3;
    int tileB = b;
    int tileS = s;
    int tileBS = tileB * tileS;

    SymbolicScalar bLoop = b / tileB;
    SymbolicScalar sLoop = s / tileS;
    LOOP("LOOP_L0_bIdx_gated_score", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bLoop, 1)) {
        LOOP("LOOP_L0_sIdx_gated_score", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sLoop, 1)) {
            Program::GetInstance().GetTileShape().SetVecTileShapes({tileB, tileS, h});
            Program::GetInstance().GetTileShape().SetCubeTileShapes(
                {tileBS, tileBS}, {NUM_128, NUM_128}, {NUM_128, NUM_128});
            SymbolicScalar bOfs = bIdx * tileB;
            SymbolicScalar sOfs = sIdx * tileS;
            SymbolicScalar bsOfs = bOfs * sOfs;

            auto xReshape = Reshape(x, {b * s, h});
            auto xView = DView(xReshape, {tileBS, h}, {bsOfs, 0});
            auto mm1Res = Matrix::Matmul(dType, xReshape, gateW1);

            Program::GetInstance().GetTileShape().SetVecTileShapes({1, h});
            auto sigmoidRes = Sigmoid(mm1Res);

            auto mm2Res = Matrix::Matmul(dType, sigmoidRes, gateW2);
            Program::GetInstance().GetTileShape().SetVecTileShapes({tileBS, n1});

            auto res = Reshape(mm2Res, {tileB, tileS, 3, n1});
            Program::GetInstance().GetTileShape().SetVecTileShapes({2, tileS, 3, n1});

            res = Transpose(Cast(res, DataType::DT_FP32), {2, 3});

            DAssemble(Cast(res, DataType::DT_FP16), {bOfs, sIdx, 0, 0}, gatingScore);
        }
    }
}

void GenAttn(Tensor &gatingScore, Tensor &cmpAtten, Tensor &selAtten, Tensor &winAtten, Tensor &attentionOut) {
    int nDimSize = cmpAtten->shape[2]; // n1
    int vDimSize = cmpAtten->shape[3]; // v_dim
    int tileB = 8;
    int tileS = 1;

    SymbolicScalar bDimSize = GetInputShapeDim(cmpAtten, 0);
    SymbolicScalar sDimSize = GetInputShapeDim(cmpAtten, 1);
    SymbolicScalar bLoop = bDimSize / tileB;
    SymbolicScalar sLoop = sDimSize / tileS;
    DataType dType = attentionOut->Datatype();
    LOOP("LOOP_L0_bIdx_gen_attn", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(bLoop), {}, true) {
        SymbolicScalar bOffset = bIdx * tileB;
        SymbolicScalar actualBSize = std::min(tileB, (bDimSize - bIdx * tileB));
        LOOP("LOOP_L1_sIdx_gen_attn", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(sLoop)) {
            SymbolicScalar sOffset = sIdx * tileS;
            std::vector<SymbolicScalar> outOffset = {bOffset, sOffset, 0, 0};
            SymbolicScalar actualsSize = std::min(tileS, (sDimSize - sIdx * tileS));
            Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, NUM_16, vDimSize);
            auto cmpAttenTile = DViewPad(cmpAtten, {tileB, tileS, nDimSize, vDimSize},
                {actualBSize, actualsSize, nDimSize, vDimSize}, {bOffset, sOffset, 0, 0});
            auto selAttenTile = DViewPad(selAtten, {tileB, tileS, nDimSize, vDimSize},
                {actualBSize, actualsSize, nDimSize, vDimSize}, {bOffset, sOffset, 0, 0});
            auto winAttenTile = DViewPad(winAtten, {tileB, tileS, nDimSize, vDimSize},
                {actualBSize, actualsSize, nDimSize, vDimSize}, {bOffset, sOffset, 0, 0});
            auto cmpAttenFP32Tile = Cast(cmpAttenTile, DT_FP32);
            auto selAttenFP32Tile = Cast(selAttenTile, DT_FP32);
            auto winAttenFP32Tile = Cast(winAttenTile, DT_FP32);
            Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, nDimSize, NUM_3);
            auto gatingScoreTile = DViewPad(gatingScore, {tileB, tileS, nDimSize, NUM_3},
                {actualBSize, actualsSize, nDimSize, NUM_3}, {bOffset, sOffset, 0, 0});
            auto gatingScoreFP32 = Cast(gatingScoreTile, DT_FP32);
            auto cmpWeight = View(gatingScoreFP32, {tileB, tileS, nDimSize, 1}, {0, 0, 0, 0});
            auto selWeight = View(gatingScoreFP32, {tileB, tileS, nDimSize, 1}, {0, 0, 0, 1});
            auto winWeight = View(gatingScoreFP32, {tileB, tileS, nDimSize, 1}, {0, 0, 0, 2});
            Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, NUM_16, vDimSize);
            auto mulCmp = Mul(cmpAttenFP32Tile, cmpWeight);
            auto mulSel = Mul(selAttenFP32Tile, selWeight);
            auto mulWin = Mul(winAttenFP32Tile, winWeight);
            auto addCmpSel = Add(mulCmp, mulSel);
            auto outFP32 = Add(addCmpSel, mulWin);
            Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, NUM_16, vDimSize);
            auto attentionOutTile = Cast(outFP32, dType, CAST_RINT);
            DAssemble(attentionOutTile, outOffset, attentionOut);
        }
    }
}

void DynamicNsa(Tensor &topkIndices, Tensor &topkTensorShape, Tensor &kvNopeCache, Tensor &kRopeCache, Tensor &kvActSeqs, Tensor &blockTable,
    int front, int near, int topk, int slcBlockSize, int blockSize, KvSlcTileShapeConfig &kvSlcTileConfig,
    const Tensor &qNope, const Tensor &qRope, Tensor &kvSlcActSeqs, float softmaxScale, SaTileShapeConfig saTileConfig,
    const Tensor &x, const Tensor &gateW1, const Tensor &gateW2, const Tensor &gateSimW1, GateMode gateMode,
    Tensor &cmpAtten, Tensor &winAtten,
    Tensor &weightUV, Tensor &weightO, Tensor &weightOScale, Tensor &smoothScalesWo, const PostTileConfig &postConfig,
    Tensor &kvSlcActSeqOut, Tensor &attentionOut, Tensor &postOut) {
    ASSERT(gateMode == standard); // 当前仅支持standard模式

    FUNCTION("main", FunctionType::DYNAMIC, {topkIndices, topkTensorShape, kvNopeCache, kRopeCache, kvActSeqs, blockTable, // genKvSlc
                                             qNope, qRope, kvSlcActSeqs,  // SlcAttn
                                             x, gateW1, gateW2, gateSimW1,  // gatedScore
                                             cmpAtten, winAtten,  // genAttn
                                             weightUV, weightO, weightOScale, smoothScalesWo}, // paPost
                                            {kvSlcActSeqOut, attentionOut, postOut}) {
        Program::GetInstance().GetConfig().Set<int>(DB_TYPE, 1);
        Program::GetInstance().GetConfig().Set<int>(L1_REUSE, NUM_4);
        Program::GetInstance().GetConfig().Set<std::map<int, int>>(CUBE_NBUFFER_MAP, {{NUM_3, NUM_4}});
        Program::GetInstance().GetConfig().Set<int>(COPYIN_THRESHOLD, NUM_2 * NUM_1024 * NUM_1024);

        int b = x->shape[0];
        int s = x->shape[1]; // s=1
        int n1 = gateW2->shape[1] / 3;
        int n2 = 1;
        int vDim = qNope->shape[1];  // kvLoraRank
        int ropeDim = qRope->shape[1];
        int kDim = vDim + ropeDim;
        auto dtype = x->Datatype();
        int slcSMax = topk * slcBlockSize;

        /*********************************/
        /*有数据依赖的子图一定要加loop_barrier*/
        /** 将整个nsa分为以下进行串联
         * subgragh 0: mla_prolog
         * subgragh 1: gen_win_attn
         * subgragh 2: kv_compression
         * subgragh 3: gen_cmp_atten
         * subgragh 4: gen_slc_atten, 其中包括: gen_kv_slc及slc_attn
         * subgragh 5: gen_gated_score
         * subgragh 6: gen_attn
         * subgragh 7: pa_post
         */
        /** 依赖关系如下：
            {subgraph-1, subgraph-2, subgraph-4}: {subgraph-0}
            subgraph-6: {subgraph-1, subgraph-3, subgraph-4, subgraph-5}
            subgraph-3: {subgraph-2}
            subgraph-7: {subgraph-6}
        */
        /*********************************/

        // subgraph-0

        // Loop_barrier
        // subgraph-1

        // subgraph-2-3

        // subgraph-4
        /********gen kv slc ********/
        config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, false);
        Tensor kSlc(dtype, {b * s * n2 * slcSMax, kDim}, "kSlc"); // 输出两维 [b*s1*n2*topk*slcBlockSize, d]
        Tensor vSlc(dtype, {b * s * n2 * slcSMax, vDim}, "vSlc");
        KvSlcCompute(topkIndices, topkTensorShape, kvNopeCache, kRopeCache, kvActSeqs, front, near, topk, slcBlockSize,
                     n2, blockTable, blockSize, kSlc, vSlc, kvSlcActSeqOut, kvSlcTileConfig);

        // loop_barrier
        config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true); // codegen参数化，for 动态尾块
        /********gen slc atten ********/
        Tensor slcAttn(DT_FP32, {b, s, n1, vDim}, "slcAttn"); // slcAttn 输出四维[b,s,n1,vDim] fp32
        SlcAttnCompute(qNope, qRope, kSlc, vSlc, kvSlcActSeqs, n1, n2, softmaxScale, slcAttn, saTileConfig);

        // subgraph-5
        config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, false); // 非参数化
        /********gen gated_score ********/
        Tensor gatingScore(dtype, {b, s, n1, 3}, "gatingScore");
        GenGatedScore(x, gateW1, gateW2, gateSimW1, gatingScore, gateMode); // GenGatedScore 输出四维[b,s1,n1,3] fp16

        // Loop_barrier
        // subgraph-6
        /******** gen attn ********/
        GenAttn(gatingScore, cmpAtten, slcAttn, winAtten, attentionOut); // [b,s,n1,vDim] fp16

        Program::GetInstance().GetConfig().Set<int>(CYCLE_UPPER_BOUND, 500000);  // 500000
        Program::GetInstance().GetConfig().Set<std::map<int, int>>(CUBE_NBUFFER_MAP, {{0, 4}});
        // Loop_barrier
        // subgraph-7: postOut [b,s,h]
        PostCompute(attentionOut, weightUV, weightO, weightOScale, smoothScalesWo, postConfig, postOut);
    }
}

} // namespace ascend
