/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_mla.cpp
 * \brief
 */

#include <iostream>
#include "interface/configs/config_manager.h"
#include "interface/function/function.h"

using namespace npu::tile_fwk;

namespace mla_prolog {

const int NUM_NEG_2 = -2;
const int NUM_2 = 2;
const int NUM_3 = 3;
const int NUM_4 = 4;
const int NUM_8 = 8;
const int NUM_16 = 16;
const int NUM_32 = 32;
const int NUM_64 = 64;
const int NUM_128 = 128;
const int NUM_256 = 256;
const int NUM_512 = 512;
const int NUM_1024 = 1024;

struct MlaQuantInputs {
    Tensor dequantScaleX;
    Tensor dequantScaleWDq;
    Tensor dequantScaleWUqQr;
    Tensor dequantScaleWDkvKr;
    Tensor quantScaleCkv;
    Tensor quantScaleCkr;
    Tensor smoothScalesCq;
};

struct TestShapeParams {
    int b;
    int s;
    int s2;
    int n;
    int h;
    int qLoraRank;
    int qkNopeHeadDim;
    int qkRopeHeadDim;
    int kvLoraRank;
    int blockSize;
};


struct SimpleParams {
    int b;
    int s;
    int s2;
    int d;
    int m;
    int k;
    int n;
    int n2;
    int right;
    int h;
    int qLoraRank;
    int kvLoraRank;
    int qkRopeHeadDim;
    int qkNopeHeadDim;
    int qHeadDim;
    std::string cacheMode;
    int blockSize;
    std::vector<int> vecTile;
    std::vector<int> cubeMTile;
    std::vector<int> cubeKTile;
    std::vector<int> cubeNTile;
    int tileB;
    static SimpleParams GetCommonParams()
    {
        SimpleParams params;
        params.n2 = 1;
        params.s = 1;
        params.h = 7168;            // 7168
        params.qLoraRank = 1536;    // 1536
        params.kvLoraRank = 512;    // 512
        params.qkRopeHeadDim = 64;  // 64
        params.qkNopeHeadDim = 128; // 128
        params.qHeadDim = params.qkRopeHeadDim + params.qkNopeHeadDim;
        params.cacheMode = "BNSD";
        params.blockSize = 128; // 128
        return params;
    }

    static SimpleParams GetLowParams()
    {
        SimpleParams params = GetCommonParams();
        params.b = 4;    // 4
        params.n = 32;   // 32
        params.s2 = 256; // 256
        return params;
    }

    static SimpleParams GetHighParams()
    {
        SimpleParams params = GetCommonParams();
        params.b = 32;    // 32
        params.n = 128;   // 128
        params.s2 = 4096; // 4096
        return params;
    }
};

// tile config
struct MlaTileConfig {
    int tileB = 8; // tileB is 8
    int tileS = 1;
};


std::vector<Tensor> mlaPre(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wDkvKr,
    const Tensor &gammaCq, float epsilonCq, const MlaQuantInputs &quantInputs, bool splitK, bool isSmooth)
{
    // quant
    Tensor dequantScaleWUqQr = quantInputs.dequantScaleWUqQr;
    bool isQuant = (dequantScaleWUqQr.GetStorage() != nullptr);
    Tensor smoothScalesCq = quantInputs.smoothScalesCq;

    int b = tokenX->shape[0];
    int s = tokenX->shape[1];
    int h = tokenX->shape[2];
    int bs = b * s;
    int qLoraRank = wDq->shape[1];

    DataType dType = tokenX->Datatype();
    DataType dTypeQuantOut = isQuant ? DataType::DT_INT32 : dType;
    std::vector<Tensor> qkvPreRes;

    Tensor input = Reshape(tokenX, {bs, h}); // [b,s,h] -> [b*s,h]

    /******** q ********/
    int c0 = 16;                                                          // 16
    int m = (std::min(32, bs) + c0 - 1) / c0 * c0;                        // 32
    int tieM = std::min(32, m);                                           // 32
    TileShape::Current().SetCubeTile({tieM, tieM}, {256, 256}, {64, 64}); // 256, 64
    // [b*s,h] * [h,qLoraRank] = [b*s,qLoraRank]
    Tensor qMmRes;
    if (splitK) {
        Tensor tmpC(DT_FP32, {bs, qLoraRank}, "tmp_q");
        TileShape::Current().SetVecTile(std::min(32, bs), 128); // 32, 128
        tmpC = MulS(tmpC, Element(DataType::DT_FP32, 0.0f));
        std::vector<Tensor> matmulResult;
        auto kSplit = 7;
        auto kSplitSize = h / kSplit;
        for (int ki = 0; ki < kSplit; ki++) {
            auto input_mk = View(input, {bs, kSplitSize}, {0, ki * kSplitSize});
            auto input_kn = View(wDq, {kSplitSize, qLoraRank}, {ki * kSplitSize, 0});
            auto tmp = Matrix::Matmul(DT_FP32, input_mk, input_kn, tmpC); // [b*s,h/2] * [h/2,qLoraRank]
            matmulResult.emplace_back(tmp);
        }
        Tensor qMmResF32 = npu::tile_fwk::Reduce(matmulResult, ReduceMode::ATOMIC_ADD);
        TileShape::Current().SetVecTile(std::min(32, bs), 128); // 32, 128
        qMmRes = Cast(qMmResF32, dType);
    } else {
        qMmRes = Matrix::Matmul(dType, input, wDq); // bf16
    }

    TileShape::Current().SetVecTile(std::min(8, bs), qLoraRank); // 8
    Tensor normRes = RmsNorm(qMmRes, gammaCq, epsilonCq);

    Tensor normDequantScale;
    std::tuple<Tensor, Tensor> normQuantRes;
    if (isQuant) {
        if (isSmooth) {
            normQuantRes = Quant(normRes, true, true, smoothScalesCq);
        } else {
            normQuantRes = Quant(normRes); // int8
        }
        normRes = std::get<0>(normQuantRes);
        normDequantScale = std::get<1>(normQuantRes);
        TileShape::Current().SetCubeTile({tieM, tieM}, {256, 256}, {256, 256}); // 256
    } else {
        // use tileM will core dump
        TileShape::Current().SetCubeTile({tieM, tieM}, {256, 256}, {64, 64}); // 256, 64
    }
    // [b*s,qLoraRank] * [qLoraRank, n*qHeadDim] = [b*s, n*qHeadDim]
    Tensor q = Matrix::Matmul(dTypeQuantOut, normRes, wUqQr); // bf16  // quant: A8W8O32 -> bf16
    qkvPreRes.emplace_back(q);

    /******** kv ********/
    TileShape::Current().SetCubeTile({m, m}, {256, 256}, {64, 64}); // 256, 64
    // [b*s,h] * [h,kvLoraRank+qkRopeHeadDim] = [b*s,kvLoraRank+qkRopeHeadDim]
    Tensor compressedKv;
    if (splitK) {
        TileShape::Current().SetVecTile(std::min(32, bs), 64); // 32, 64
        int kvN = wDkvKr->shape[1];
        Tensor tmpCKv(DT_FP32, {bs, kvN}, "tmpKv");
        tmpCKv = MulS(tmpCKv, Element(DataType::DT_FP32, 0.0f));
        std::vector<Tensor> matmulResult_kv;
        auto kSplitKv = 7;
        auto kSplitSizeKv = h / kSplitKv;
        for (int ki = 0; ki < kSplitKv; ki++) {
            auto input_mk = View(input, {bs, kSplitSizeKv}, {0, ki * kSplitSizeKv});
            auto input_kn = View(wDkvKr, {kSplitSizeKv, kvN}, {ki * kSplitSizeKv, 0});
            auto tmp = Matrix::Matmul(DT_FP32, input_mk, input_kn, tmpCKv); // [b*s,h/2] * [h/2,kv_n] = [b*s,kv_n]
            matmulResult_kv.emplace_back(tmp);
        }
        Tensor kvMmResF32 = npu::tile_fwk::Reduce(matmulResult_kv, ReduceMode::ATOMIC_ADD);
        TileShape::Current().SetVecTile(std::min(32, bs), 64); // 32, 64
        compressedKv = Cast(kvMmResF32, dType);
    } else {
        compressedKv = Matrix::Matmul(dType, input, wDkvKr); // bf16
    }
    Tensor compressedKvRes = Reshape(compressedKv, {b, s, (int)wDkvKr->shape[1]});
    qkvPreRes.emplace_back(compressedKvRes);

    if (isQuant) {
        qkvPreRes.emplace_back(normDequantScale);
    }

    return qkvPreRes;
}

void MlaProlog(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wUk, const Tensor &wDkvKr,
    const Tensor &gammaCq, const Tensor &gammaCkv, const Tensor &sin, const Tensor &cos, const Tensor &cacheIndex,
    Tensor &kvCache, Tensor &krCache, const MlaQuantInputs &quantInputs, const RoPETileShapeConfigNew &ropeConfig,
    Tensor &queryOut, Tensor &queryRopeOut, Tensor &kvCacheOut, Tensor &krCacheOut, float epsilonCq, float epsilonCkv,
    std::string cacheMode, bool splitK, bool isSmooth)
{
    // params check
    if (tokenX->shape.size() != SHAPE_DIM3 || wUk->shape.size() != SHAPE_DIM3 || sin->shape.size() != SHAPE_DIM3) {
        throw std::invalid_argument("`tokenX`, `wUk` and `sin` should be 3-dimensional.");
    }
    if (cacheMode != "BNSD" && cacheMode != "PA_BSND" && cacheMode != "PA_NZ") {
        throw std::invalid_argument("`cacheMode` should be 'BNSD', 'PA_BSND' or 'PA_NZ'.");
    }
    DataType dType = tokenX->Datatype();
    int b = tokenX->shape[0];
    int s = tokenX->shape[1]; // s=1
    int h = tokenX->shape[2];
    int s2 = kvCache->shape[2];
    // [n, qkNopeHeadDim, kvLoraRank]
    int n = wUk->shape[0];
    int qkNopeHeadDim = wUk->shape[1];
    int kvLoraRank = wUk->shape[2];
    int qkRopeHeadDim = sin->shape[2]; // [b,s,qkRopeHeadDim]
    int qHeadDim = qkNopeHeadDim + qkRopeHeadDim;

    int tileB = b;
    int tileBS = tileB * s;
    SymbolicScalar bLoop = b / tileB;

    FUNCTION("main",
        {tokenX, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos, cacheIndex, kvCache, krCache,
            quantInputs.dequantScaleWUqQr, quantInputs.smoothScalesCq},
        {queryOut, queryRopeOut, kvCacheOut, krCacheOut}) {
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bLoop, 1)) {
            SymbolicScalar bOffset = bIdx * tileB;
            std::vector<SymbolicScalar> outputOffset = {bOffset, 0, 0, 0};

            Tensor dequantScaleWUqQr = quantInputs.dequantScaleWUqQr;
            bool isQuant = (dequantScaleWUqQr.GetStorage() != nullptr);

            auto xView = View(tokenX, {tileB, s, h}, {bOffset, 0, 0});
            auto qKv = mlaPre(xView, wDq, wUqQr, wDkvKr, gammaCq, epsilonCq, quantInputs, splitK, isSmooth);
            Tensor q = qKv[0];     // [b*s, n*qHeadDim]
            Tensor kvTmp = qKv[1]; // [b,s,kvLoraRank+qkRopeHeadDim]

            // dequant: int32 -> fp32 -> *scale -> fp16/bf16
            if (isQuant) {
                std::vector<int64_t> tileShape = {std::min(32, tileBS), 64}; // 32, 64
                TileShape::Current().SetVecTile(tileShape);
                auto qTmpFp32 = Cast(q, DataType::DT_FP32);
                auto qTmpDequantScale = qKv[2];
                auto qTmpDequantPerToken = Mul(qTmpFp32, qTmpDequantScale);
                auto qTmpDequantChannel = Mul(qTmpDequantPerToken, dequantScaleWUqQr);

                q = Cast(qTmpDequantChannel, dType);
            }

            auto qTmp = Reshape(q, {tileB, s, n, qHeadDim});
            std::vector<int64_t> tileShape = {std::min(32, tileB), 1, 1, 64}; // 32, 64
            TileShape::Current().SetVecTile(tileShape);

            /******** q ********/
            Tensor qNope = View(qTmp, {tileB, s, n, qkNopeHeadDim}, {0, 0, 0, 0}); // [b,s,n,qkNopeHeadDim]
            tileShape = {tileB, 1, 1, 128};                                        // 128
            TileShape::Current().SetVecTile(tileShape);
            Tensor qNopeRes = Reshape(qNope, {tileBS, n, qkNopeHeadDim}); // [bs,n,qkNopeHeadDim]
            tileShape = {std::min(32, tileBS), 1, qkNopeHeadDim};         // {2, 32, qkNopeHeadDim}
            TileShape::Current().SetVecTile(tileShape);
            Tensor qNopeTrans = Transpose(qNopeRes, {0, 1}); // [n,bs,qkNopeHeadDim]

            int c0 = 16; // 16
            int m = (std::min(32, tileBS) + c0 - 1) / c0 * c0;
            TileShape::Current().SetCubeTile({m, m}, {128, 128}, {128, 128}); // 128
            // bmm: (n,bs,qkNopeHeadDim) * (n, qkNopeHeadDim, kvLoraRank) = (n, bs, kvLoraRank)
            Tensor qNopeNew = Matrix::BatchMatmul(dType, qNopeTrans, wUk);

            tileShape = {1, std::min(32, tileBS), kvLoraRank}; // 32
            TileShape::Current().SetVecTile(tileShape);
            Tensor qNopeNewTrans = Transpose(qNopeNew, {0, 1});                     // [bs,n,kvLoraRank]
            auto queryOutDview = Reshape(qNopeNewTrans, {tileB, s, n, kvLoraRank}); // [b,s,n,kvLoraRank], output1

            /******** kv ********/
            Tensor compressedKv = View(kvTmp, {tileB, s, kvLoraRank}, {0, 0, 0}); // [b,s,kvLoraRank]
            tileShape = {2, 1, 512};                                              // 512
            TileShape::Current().SetVecTile(tileShape);
            Tensor compressedKvNorm = RmsNorm(compressedKv, gammaCkv, epsilonCkv); // [b,s,kvLoraRank]
            Tensor kNope = Reshape(compressedKvNorm, {tileB, 1, s, kvLoraRank});   // [b,1,s,kvLoraRank]
                                                                                   ////
            /******** RoPE ********/
            Tensor kPeView = View(kvTmp, {tileB, s, qkRopeHeadDim}, {0, 0, kvLoraRank}); // [b,s,qkRopeHeadDim]
            tileShape = {std::min(32, tileB), 1, qkRopeHeadDim};
            TileShape::Current().SetVecTile(tileShape);
            Tensor kPeRes = Reshape(kPeView, {tileB, s, 1, qkRopeHeadDim}); // [b,s,1,qkRopeHeadDim]
            Tensor qPeView = View(qTmp, {tileB, s, n, qkRopeHeadDim}, {0, 0, 0, qkNopeHeadDim});
            Tensor cosView = View(cos, {tileB, s, qkRopeHeadDim}, {bOffset, 0, 0});
            Tensor sinView = View(sin, {tileB, s, qkRopeHeadDim}, {bOffset, 0, 0});
            Tensor kRopeView(kPeRes->Datatype(), {tileB, s, 1, qkRopeHeadDim}, "kRopeView"); // [b,1,s,qkRopeHeadDim]
            Tensor qRopeView(kPeRes->Datatype(), {tileB, s, n, qkRopeHeadDim}, "qRopeView");
            ApplyRotaryPosEmbV2(qPeView, kPeRes, cosView, sinView, qRopeView, kRopeView, 2, ropeConfig); // 2
            Tensor kvCacheOutDview, krCacheOutDview;
            if (cacheMode != "BNSD") {
                int blockNum = kvCache->shape[0];
                int blockSize = kvCache->shape[1];
                int n2 = kvCache->shape[2];
                Tensor kvCacheRes = Reshape(kvCache, {blockNum * blockSize * n2, kvLoraRank});
                Tensor krCacheRes = Reshape(krCache, {blockNum * blockSize * n2, qkRopeHeadDim});
                auto cacheIndexDview = View(cacheIndex, {tileB, s}, {bOffset, 0});
                kNope = Reshape(kNope, {tileB * s, kvLoraRank}); // [b*s,kvLoraRank]
                Tensor kRopeRes = Reshape(kRopeView, {tileB * s * 1, qkRopeHeadDim});

                /******** kvCache ********/
                tileShape = {1, kvLoraRank};
                TileShape::Current().SetVecTile(tileShape);
                // kvCache: [blockNum * blockSize * n2, kvLoraRank], output3
                kvCacheOutDview = ScatterUpdate(kvCacheRes, cacheIndexDview, kNope, NUM_NEG_2, cacheMode, blockSize);

                /******** krCache ********/
                tileShape = {1, qkRopeHeadDim};
                TileShape::Current().SetVecTile(tileShape);
                // krCache: [blockNum * blockSize * n2, qkRopeHeadDim], output4
                krCacheOutDview = ScatterUpdate(
                    krCacheRes, cacheIndexDview, kRopeRes, NUM_NEG_2, cacheMode, blockSize);

                kvCacheOut = Reshape(kvCacheOutDview, {blockNum, blockSize, n2, kvLoraRank});
                krCacheOut = Reshape(krCacheOutDview, {blockNum, blockSize, n2, qkRopeHeadDim});
            } else {
                Tensor kRopeRes = Reshape(kRopeView, {tileB, 1, s, qkRopeHeadDim});
                auto cacheIndexDview = View(cacheIndex, {tileB, s}, {bOffset, 0});
                tileShape = {1, 1, 1, kvLoraRank};
                TileShape::Current().SetVecTile(tileShape);
                auto kvCacheDview = View(kvCache, {tileB, 1, s2, kvLoraRank}, {bOffset, 0, 0, 0});
                kvCacheOut = ScatterUpdate(kvCacheDview, cacheIndexDview, kNope, NUM_NEG_2);

                tileShape = {1, 1, 1, qkRopeHeadDim};
                TileShape::Current().SetVecTile(tileShape);
                auto krCacheDview = View(krCache, {tileB, 1, s2, qkRopeHeadDim}, {bOffset, 0, 0, 0});
                krCacheOut = ScatterUpdate(krCacheDview, cacheIndexDview, kRopeRes, NUM_NEG_2);
            }
            Assemble(queryOutDview, outputOffset, queryOut);
            Assemble(qRopeView, outputOffset, queryRopeOut);
        }
    }
}

Tensor DeQuant(DataType dType, const Tensor &input, const Tensor &scale, const Tensor &wScale)
{
    Tensor dequantRes = Cast(input, DataType::DT_FP32);
    dequantRes = Mul(dequantRes, scale);
    dequantRes = Mul(dequantRes, wScale);
    return Cast(dequantRes, dType);
}

std::vector<Tensor> PreCompute(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wDkvKr,
    const Tensor &gammaCq, float epsilonCq, const MlaQuantInputs &quantInputs)
{
    // quant
    Tensor dequantScaleWDq = quantInputs.dequantScaleWDq;
    Tensor dequantScaleWDkvKr = quantInputs.dequantScaleWDkvKr;
    Tensor dequantScaleWUqQr = quantInputs.dequantScaleWUqQr;
    bool isQuantA = (dequantScaleWDq.GetStorage() != nullptr) && (dequantScaleWDkvKr.GetStorage() != nullptr);
    bool isQuantB = dequantScaleWUqQr.GetStorage() != nullptr;
    Tensor smoothScalesCq = quantInputs.smoothScalesCq;
    bool isSmooth = (smoothScalesCq.GetStorage() != nullptr);

    int b = tokenX->shape[0];
    int s = tokenX->shape[1];
    int h = tokenX->shape[2];
    int bs = b * s;
    int qLoraRank = wDq->shape[1];

    DataType dType = tokenX->Datatype();
    DataType dTypeQuantAOut = isQuantA ? DataType::DT_INT32 : dType;
    DataType dTypeQuantBOut = isQuantB ? DataType::DT_INT32 : dType;
    std::vector<Tensor> qkvPreRes;

    config::SetSemanticLabel("pre_reshape");
    Tensor input = Reshape(tokenX, {bs, h}); // [b,s,h] -> [b*s,h]
    Tensor inputQuant;
    Tensor inputQuantScale;

    /******** q ********/
    int c0 = 16; // 16
    int m = (std::min(32, bs) + c0 - 1) / c0 * c0;
    int mv = std::min(8, bs);
    // [b*s,h] @ [h,qLoraRank] = [b*s,qLoraRank]
    Tensor qAProj;
    if (isQuantA) {
        TileShape::Current().SetVecTile(mv, qLoraRank);
        TileShape::Current().SetCubeTile({m, m}, {256, 256}, {256, 256});
        // no smooth
        config::SetSemanticLabel("Quant_x");
        auto quantRes = Quant(input);
        inputQuant = std::get<0>(quantRes);
        inputQuantScale = std::get<1>(quantRes);
        config::SetSemanticLabel("QuantMatmul_qa");
        qAProj = Matrix::Matmul(dTypeQuantAOut, inputQuant, wDq);
        config::SetSemanticLabel("Dequant_qa");
        qAProj = DeQuant(dType, qAProj, inputQuantScale, dequantScaleWDq);
    } else {
        TileShape::Current().SetCubeTile({m, m}, {256, 256}, {64, 64});
        config::SetSemanticLabel("Matmul_qa");
        qAProj = Matrix::Matmul(dType, input, wDq);
    }

    // rmsnorm
    TileShape::Current().SetVecTile(mv, qLoraRank);
    config::SetSemanticLabel("RmsNorm_qa");
    Tensor normRes = RmsNorm(qAProj, gammaCq, epsilonCq);

    // [b*s,qLoraRank] @ [qLoraRank, n*qHeadDim] = [b*s, n*qHeadDim]
    Tensor qBProj;
    if (isQuantB) {
        Tensor normQuant;
        Tensor normQuantScale;
        TileShape::Current().SetVecTile(mv, qLoraRank);
        TileShape::Current().SetCubeTile({m, m}, {256, 256}, {256, 256});
        config::SetSemanticLabel("Quant_qMmRes");
        std::tuple<Tensor, Tensor> quantRes;
        if (isSmooth) {
            quantRes = Quant(normRes, true, true, smoothScalesCq);
        } else {
            quantRes = Quant(normRes, true, false);
        }
        normQuant = std::get<0>(quantRes);
        normQuantScale = std::get<1>(quantRes);
        config::SetSemanticLabel("QuantMatmul_qb");
        qBProj = Matrix::Matmul(dTypeQuantBOut, normQuant, wUqQr);
        config::SetSemanticLabel("Dequant_qb");
        qBProj = DeQuant(dType, qBProj, normQuantScale, dequantScaleWUqQr);
    } else {
        TileShape::Current().SetCubeTile({m, m}, {256, 256}, {64, 64});
        config::SetSemanticLabel("Matmul_qb");
        qBProj = Matrix::Matmul(dType, normRes, wUqQr);
    }
    qkvPreRes.emplace_back(qBProj);

    /******** kv ********/
    // [b*s,h] @ [h,kvLoraRank+qkRopeHeadDim] = [b*s,kvLoraRank+qkRopeHeadDim]
    Tensor compressedKv;
    if (isQuantA) {
        TileShape::Current().SetVecTile(mv, qLoraRank);
        TileShape::Current().SetCubeTile({m, m}, {256, 256}, {256, 256});
        // no smooth
        config::SetSemanticLabel("QuantMatmul_kva");
        compressedKv = Matrix::Matmul(dTypeQuantAOut, inputQuant, wDkvKr);
        config::SetSemanticLabel("Dequant_kva");
        compressedKv = DeQuant(dType, compressedKv, inputQuantScale, dequantScaleWDkvKr);
    } else {
        TileShape::Current().SetCubeTile({m, m}, {256, 256}, {64, 64});
        config::SetSemanticLabel("Matmul_kva");
        compressedKv = Matrix::Matmul(dType, input, wDkvKr);
    }
    qkvPreRes.emplace_back(compressedKv);

    return qkvPreRes;
}

// NSA MlaProlog, b and s is dynamic, support:
// b: 16, 32, 64, 24, 48, 96
// s: 1, 2
void MlaPrologCompute(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wUk,
    const Tensor &wDkvKr, const Tensor &gammaCq, const Tensor &gammaCkv, const Tensor &sin, const Tensor &cos,
    const Tensor &cacheIndex, Tensor &kvCache, Tensor &krCache, const MlaQuantInputs &quantInputs,
    const MlaTileConfig &tileConfig, Tensor &queryOut, Tensor &queryRopeOut, Tensor &kvCacheOut, Tensor &krCacheOut,
    float epsilonCq, float epsilonCkv, std::string cacheMode)
{
    // params check
    if (tokenX->shape.size() != SHAPE_DIM3 || wUk->shape.size() != SHAPE_DIM3 || sin->shape.size() != SHAPE_DIM3) {
        throw std::invalid_argument("`tokenX`, `wUk` and `sin` should be 3-dimensional.");
    }
    if (kvCache->shape.size() != SHAPE_DIM4 || krCache->shape.size() != SHAPE_DIM4) {
        throw std::invalid_argument("`kvCache` and `krCache` should be 4-dimensional.");
    }
    if (cacheMode != "PA_BSND" && cacheMode != "PA_NZ") {
        throw std::invalid_argument("`cacheMode` should be 'PA_BSND' or 'PA_NZ'.");
    }

    DataType dType = tokenX->Datatype();
    int h = tokenX->shape[2]; // 2
    // [n, qkNopeHeadDim, kvLoraRank]
    int n = wUk->shape[0];
    int qkNopeHeadDim = wUk->shape[1];
    int kvLoraRank = wUk->shape[2];
    int qkRopeHeadDim = sin->shape[2]; // [b,s,qkRopeHeadDim], 2
    int qHeadDim = qkNopeHeadDim + qkRopeHeadDim;
    // kvCache: [block_num, block_size, n2, kv_lora_rank], n2=1
    int blockNum = kvCache->shape[0];
    int blockSize = kvCache->shape[1];
    int n2 = kvCache->shape[2];
    if (qkNopeHeadDim != NUM_128 && qkRopeHeadDim != NUM_64) {
        throw std::invalid_argument("`qkNopeHeadDim` should be 64 or 128.");
    }

    int tileB = tileConfig.tileB;
    int tileS = tileConfig.tileS;
    int tileBS = tileB * tileS;

    RoPETileShapeConfigNew ropeConfig {
        {tileB, tileS, qkRopeHeadDim},           // (b,s,d)
        {tileB, tileS, 1, qkRopeHeadDim},        // (b,s,n,d) Q
        {tileB, tileS, 1, qkRopeHeadDim},        // (b,s,1,d) K
        {tileB, tileS, 1, qkRopeHeadDim / 2, 2}  // (b,s,n,d//2,2)
    };

    SymbolicScalar b = GetInputShape(tokenX, 0);
    SymbolicScalar s = GetInputShape(tokenX, 1);
    SymbolicScalar bLoop = b / tileB;
    SymbolicScalar sLoop = s / tileS;

    LOOP("MLA_LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bLoop, 1)) {
        SymbolicScalar bOffset = bIdx * tileB;
        LOOP("MLA_LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sLoop, 1)) {
            SymbolicScalar sOffset = sIdx * tileS;
            std::vector<SymbolicScalar> outputOffset = {bOffset, sOffset, 0, 0};

            TileShape::Current().SetVecTile({tileB, tileS, 128}); // 128
            auto xView = View(tokenX, {tileB, tileS, h}, {bOffset, sOffset, 0});
            auto qKv = PreCompute(xView, wDq, wUqQr, wDkvKr, gammaCq, epsilonCq, quantInputs);
            Tensor q = qKv[0];     // [b*s, n*qHeadDim]
            Tensor kvTmp = qKv[1]; // [b*s, kvLoraRank+qkRopeHeadDim]
            auto qTmp = Reshape(q, {tileB, tileS, n, qHeadDim});

            /******** q ********/
            config::SetSemanticLabel("Prepare_qNope");
            Tensor qNope = View(qTmp, {tileB, tileS, n, qkNopeHeadDim}, {0, 0, 0, 0}); // [b,s,n,qkNopeHeadDim]
            std::vector<int64_t> tileShape = {tileB, tileS, 1, 128}; // 128
            TileShape::Current().SetVecTile(tileShape);
            Tensor qNopeRes = Reshape(qNope, {tileBS, n, qkNopeHeadDim}); // [bs,n,qkNopeHeadDim]
            tileShape = {std::min(32, tileBS), 1, qkNopeHeadDim}; // 32
            TileShape::Current().SetVecTile(tileShape);
            Tensor qNopeTrans = Transpose(qNopeRes, {0, 1}); // [n,bs,qkNopeHeadDim]

            int c0 = 16; // 16
            int m = (std::min(32, tileBS) + c0 - 1) / c0 * c0; // 32
            config::SetSemanticLabel("Matmul_qNope_wUk");
            TileShape::Current().SetCubeTile({m, m}, {128, 128}, {128, 128}); // 128
            // bmm: (n,bs,qkNopeHeadDim) @ (n, qkNopeHeadDim, kvLoraRank) = (n, bs, kvLoraRank)
            Tensor qNopeNew = Matrix::BatchMatmul(dType, qNopeTrans, wUk);

            config::SetSemanticLabel("queryOut");
            tileShape = {1, std::min(32, tileBS), kvLoraRank}; // 32
            TileShape::Current().SetVecTile(tileShape);
            Tensor qNopeNewTrans = Transpose(qNopeNew, {0, 1}); // [bs,n,kvLoraRank]
            auto queryOutView = Reshape(qNopeNewTrans, {tileB, tileS, n, kvLoraRank}); // [b,s,n,kvLoraRank]

            /******** kv ********/
            Tensor compressedKv = View(kvTmp, {tileBS, kvLoraRank}, {0, 0}); // [b*s,kvLoraRank]
            tileShape = {2, 512}; // 2, 512
            config::SetSemanticLabel("RmsNorm_compressedKv");
            TileShape::Current().SetVecTile(tileShape);
            Tensor kNope = RmsNorm(compressedKv, gammaCkv, epsilonCkv); // [b*s,kvLoraRank]

            /******** RoPE ********/
            config::SetSemanticLabel("RotaryPosEmb");
            Tensor kPeView = View(kvTmp, {tileBS, qkRopeHeadDim}, {0, kvLoraRank}); // [b*s,qkRopeHeadDim]
            Tensor kPeRes = Reshape(kPeView, {tileB, tileS, 1, qkRopeHeadDim}); // [b,s,1,qkRopeHeadDim]
            Tensor qPeView = View(qTmp, {tileB, tileS, n, qkRopeHeadDim}, {0, 0, 0, qkNopeHeadDim});
            Tensor cosView = View(cos, {tileB, tileS, qkRopeHeadDim}, {bOffset, sOffset, 0});
            Tensor sinView = View(sin, {tileB, tileS, qkRopeHeadDim}, {bOffset, sOffset, 0});
            Tensor qRopeView(kPeRes->Datatype(), {tileB, tileS, n, qkRopeHeadDim}, "qRopeView");
            Tensor kRopeView(kPeRes->Datatype(), {tileB, tileS, 1, qkRopeHeadDim}, "kRopeView");
            ApplyRotaryPosEmbV2(qPeView, kPeRes, cosView, sinView, qRopeView, kRopeView,
                2, ropeConfig); // 2 is unsqueeze dim

            // PA_BSND, PA_NZ
            Tensor kvCacheRes = Reshape(kvCache, {blockNum * blockSize * n2, kvLoraRank});
            Tensor krCacheRes = Reshape(krCache, {blockNum * blockSize * n2, qkRopeHeadDim});
            Tensor kRopeRes = Reshape(kRopeView, {tileBS * 1, qkRopeHeadDim});
            Tensor indexView = View(cacheIndex, {tileB, tileS}, {bOffset, sOffset});

            /******** kvCache ********/
            config::SetSemanticLabel("ScatterUpdate_kvCache");
            tileShape = {1, kvLoraRank};
            TileShape::Current().SetVecTile(tileShape);
            // kvCache: [blockNum * blockSize * n2, kvLoraRank], output3
            Tensor kvCacheOutView = ScatterUpdate(kvCacheRes, indexView, kNope, NUM_NEG_2, cacheMode, blockSize);

            /******** krCache ********/
            config::SetSemanticLabel("ScatterUpdate_krCache");
            tileShape = {1, qkRopeHeadDim};
            TileShape::Current().SetVecTile(tileShape);
            // krCache: [blockNum * blockSize * n2, qkRopeHeadDim], output4
            Tensor krCacheOutView = ScatterUpdate(krCacheRes, indexView, kRopeRes, NUM_NEG_2, cacheMode, blockSize);

            /* 输入和输出相同shape时，即 n2 = 1, tensor graph上无法看到该reshape */
            kvCacheOut = Reshape(kvCacheOutView, {blockNum * blockSize, n2 * kvLoraRank});
            krCacheOut = Reshape(krCacheOutView, {blockNum * blockSize, n2 * qkRopeHeadDim});

            config::SetSemanticLabel("Assemble_queryOut");
            TileShape::Current().SetVecTile({1, 1, 32, 128}); // 32, 128
            Assemble(queryOutView, outputOffset, queryOut);  // output1
            config::SetSemanticLabel("Assemble_qRope");
            TileShape::Current().SetVecTile({1, 1, 32, 64});  // 32, 64
            Assemble(qRopeView, outputOffset, queryRopeOut);  // output2
            config::SetSemanticLabel("");
        }
    }
}

void MlaProlog(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wUk,
    const Tensor &wDkvKr, const Tensor &gammaCq, const Tensor &gammaCkv, const Tensor &sin, const Tensor &cos,
    const Tensor &cacheIndex, Tensor &kvCache, Tensor &krCache, const MlaQuantInputs &quantInputs,
    const MlaTileConfig &tileConfig, Tensor &queryOut, Tensor &queryRopeOut, Tensor &kvCacheOut, Tensor &krCacheOut,
    float epsilonCq, float epsilonCkv, std::string cacheMode)
{
    FUNCTION("main",
        {tokenX, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos, cacheIndex, kvCache, krCache,
            quantInputs.dequantScaleWDq, quantInputs.dequantScaleWDkvKr, quantInputs.dequantScaleWUqQr,
            quantInputs.smoothScalesCq},
        {queryOut, queryRopeOut, kvCacheOut, krCacheOut}) {
        // compute
        MlaPrologCompute(tokenX, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos, cacheIndex, kvCache, krCache,
            quantInputs, tileConfig, queryOut, queryRopeOut, kvCacheOut, krCacheOut, epsilonCq, epsilonCkv, cacheMode);
    }
}

void TestDynamicMlaProlog(bool isQuantA, bool isQuantB, bool isSmooth, bool nz, bool usePrefetch,
    const TestShapeParams &params, const MlaTileConfig &tileConfig, std::string cacheMode = "PA_NZ")
{
    config::SetHostConfig(npu::tile_fwk::KEY_ONLY_CODEGEN, true);

    int b = params.b;
    int s = params.s;
    int s2 = params.s2;
    int n = params.n;
    int n2 = 1;
    int h = params.h;
    int qLoraRank = params.qLoraRank;
    int qkNopeHeadDim = params.qkNopeHeadDim;
    int qkRopeHeadDim = params.qkRopeHeadDim;
    int kvLoraRank = params.kvLoraRank;
    int blockSize = params.blockSize;
    int qHeadDim = qkNopeHeadDim + qkRopeHeadDim;

    DataType dType = DT_FP16;
    DataType dTypeQuantA = isQuantA ? DT_INT8 : dType;
    DataType dTypeQuantB = isQuantB ? DT_INT8 : dType;

    std::vector<int64_t> xShape = {b, s, h};
    std::vector<int64_t> wDqShape = {h, qLoraRank};
    std::vector<int64_t> wUqQrShape = {qLoraRank, n * qHeadDim};
    std::vector<int64_t> wDkvKrShape = {h, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> wUkShape = {n, qkNopeHeadDim, kvLoraRank};
    std::vector<int64_t> cosShape = {b, s, qkRopeHeadDim};
    std::vector<int64_t> gammaCqShape = {qLoraRank};
    std::vector<int64_t> gammaCkvShape = {kvLoraRank};
    std::vector<int64_t> kvLenShape = {b, s};
    int blockNum = b * (s2 / blockSize);
    std::vector<int64_t> kvCacheShape = {blockNum, blockSize, n2, kvLoraRank};
    std::vector<int64_t> krCacheShape = {blockNum, blockSize, n2, qkRopeHeadDim};
    std::vector<int64_t> kvCacheOutShape = {blockNum * blockSize, n2 * kvLoraRank};
    std::vector<int64_t> krCacheOutShape = {blockNum, blockSize, n2 * qkRopeHeadDim};
    std::vector<int64_t> scaleWDqShape = {1, qLoraRank};
    std::vector<int64_t> scaleWUqQrShape = {1, n * qHeadDim};
    std::vector<int64_t> scaleWDkvKrShape = {1, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> smoothCqShape{1, qLoraRank};
    // output
    std::vector<int64_t> qOutShape = {b, s, n, kvLoraRank};
    std::vector<int64_t> qRopeOutShape = {b, s, n, qkRopeHeadDim};

    TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;

    Tensor wDq(dTypeQuantA, wDqShape, "wDq", weightFormat);
    Tensor wUqQr(dTypeQuantB, wUqQrShape, "wUqQr", weightFormat);
    if (usePrefetch) {
        wDq.SetCachePolicy(CachePolicy::PREFETCH, true);
        wUqQr.SetCachePolicy(CachePolicy::PREFETCH, true);
    }
    Tensor wDkvKr(dTypeQuantA, wDkvKrShape, "wDkvKr", weightFormat);
    Tensor wUk(dType, wUkShape, "wUk", weightFormat);

    Tensor gammaCq(dType, gammaCqShape, "gammaCq");
    Tensor gammaCkv(dType, gammaCkvShape, "gammaCkv");

    Tensor kvCache(dType, kvCacheShape, "kvCache");
    Tensor krCache(dType, krCacheShape, "krCache");

    Tensor scaleWDq(DT_FP32, scaleWDqShape, "scaleWDq");
    Tensor scaleWUqQr(DT_FP32, scaleWUqQrShape, "scaleWUqQr");
    Tensor scaleWDkvKr(DT_FP32, scaleWDkvKrShape, "scaleWDkvKr");

    Tensor smoothCq(DT_FP32, smoothCqShape, "smoothCq");
    // output
    Tensor outputKvCache(dType, kvCacheOutShape, "outputKvCache");
    Tensor outputKrCache(dType, krCacheOutShape, "outputKrCache");

    // dynamic shape
    Tensor dynamicX(dType, {-1, -1, h}, "dynamicX");
    Tensor dynamicCos(dType, {-1, -1, qkRopeHeadDim}, "dynamicCos");
    Tensor dynamicSin(dType, {-1, -1, qkRopeHeadDim}, "dynamicSin");
    Tensor dynamicCacheIndex(DT_INT64, {-1, -1}, "dynamicCacheIndex"); // int64
    Tensor dynamicOutputQ(dType, {-1, GetInputShape(dynamicX, 1), n, kvLoraRank}, "dynamicOutputQ");
    Tensor dynamicOutputQRope(dType, {-1, GetInputShape(dynamicX, 1), n, qkRopeHeadDim}, "dynamicOutputQRope");

    MlaQuantInputs quantInputs;
    if (isQuantA) {
        quantInputs.dequantScaleWDq = scaleWDq;
        quantInputs.dequantScaleWDkvKr = scaleWDkvKr;
    }
    if (isQuantB) {
        quantInputs.dequantScaleWUqQr = scaleWUqQr;
        if (isSmooth) {
            quantInputs.smoothScalesCq = smoothCq;
        }
    }
    MlaProlog(dynamicX, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, dynamicSin, dynamicCos, dynamicCacheIndex, kvCache,
        krCache, quantInputs, tileConfig, dynamicOutputQ, dynamicOutputQRope, outputKvCache, outputKrCache, 1e-5f,
        1e-5f, cacheMode);
}
} // namespace mla_prolog

int main()
{
    mla_prolog::TestShapeParams params = {1, 2, 128, 6, 512, 64, 64, 64, 32, 32};
    std::string cacheMode = "PA_BSND";
    mla_prolog::MlaTileConfig tileConfig = {32, 1};
    config::SetPassOption(NBUFFER_MERGE_MODE, 1);
    config::SetPassOption(L1_REUSE, mla_prolog::NUM_4);
    config::SetPassOption(CUBE_NBUFFER_MAP, std::map<int64_t, int64_t>{{3, 4}});
    mla_prolog::TestDynamicMlaProlog(false, true, true, false, true, params, tileConfig, cacheMode);
    return 0;
}