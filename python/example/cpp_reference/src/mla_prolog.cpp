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
 * \file mla_prolog.cpp
 * \brief
 */

#include "operator/models/deepseek/mla_prolog.h"
#include "operator/models/deepseek/deepseek_mla.h"
#include "interface/configs/config_manager.h"
using namespace npu::tile_fwk;

namespace npu::tile_fwk {
std::vector<Tensor> QkvPre(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wDkvKr,
    const Tensor &gammaCq, float epsilonCq, MlaQuantInputs quantInputs, bool splitReduceLastDim, bool splitK)
{
    // quant
    Tensor dequantScaleWUqQr = quantInputs.dequantScaleWUqQr;
    bool isQuant = (dequantScaleWUqQr.GetStorage() != nullptr);
    Tensor smoothScalesCq = quantInputs.smoothScalesCq;
    bool hasSmooth = (smoothScalesCq.GetStorage() != nullptr);
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
    int c0 = NUM_16;
    int tieM = (bs + c0 - 1) / c0 * c0;
    TileShape::Current().SetCubeTile({tieM, tieM}, {NUM_256, NUM_256}, {NUM_64, NUM_64});
    // [b*s,h] * [h,q_lora_rank] = [b*s,q_lora_rank]
    Tensor qMmRes;
    if (splitK) {
        Tensor tmpC(DT_FP32, {bs, qLoraRank}, "tmp_q");
        TileShape::Current().SetVecTile(std::min(NUM_32, bs), NUM_128);
        tmpC = MulS(tmpC, Element(DataType::DT_FP32, F_0));
        std::vector<Tensor> matmulResult;
        auto kSplit = 7;
        auto kSplitSize = h / kSplit;
        for (int ki = 0; ki < kSplit; ki++) {
            auto input_mk = View(input, {bs, kSplitSize}, {0, ki * kSplitSize});
            auto input_kn = View(wDq, {kSplitSize, qLoraRank}, {ki * kSplitSize, 0});
            auto tmp = Matrix::Matmul(DT_FP32, input_mk, input_kn, tmpC); // [b*s,h/2] * [h/2,q_lora_rank]
            matmulResult.emplace_back(tmp);
        }
        Tensor qMmResF32 = npu::tile_fwk::Reduce(matmulResult, ReduceMode::ATOMIC_ADD);
        TileShape::Current().SetVecTile(std::min(NUM_32, bs), NUM_128);
        qMmRes = Cast(qMmResF32, dType);
    } else {
        qMmRes = Matrix::Matmul(dType, input, wDq); // bf16
    }

    if (splitReduceLastDim) {
        TileShape::Current().SetVecTile(std::min(NUM_16, bs), NUM_128);
    } else {
        TileShape::Current().SetVecTile(std::min(NUM_8, bs), qLoraRank);
    }

    Tensor normRes = RmsNorm(qMmRes, gammaCq, epsilonCq);
    Tensor normDequantScale;
    std::tuple<Tensor, Tensor> normQuantRes;
    if (isQuant) {
        if (hasSmooth) {
            normQuantRes = Quant(normRes, true, true, smoothScalesCq);
        } else {
            normQuantRes = Quant(normRes);  // int8
        }
        normRes = std::get<0>(normQuantRes);
        normDequantScale = std::get<1>(normQuantRes);
        TileShape::Current().SetCubeTile({tieM, tieM}, {NUM_256, NUM_256}, {NUM_256, NUM_256});
    } else {
        TileShape::Current().SetCubeTile({tieM, tieM}, {NUM_256, NUM_256}, {NUM_64, NUM_64});
    }
    // [b*s,qLoraRank] * [qLoraRank, n*qHeadDim] = [b*s, n*qHeadDim]
    Tensor q = Matrix::Matmul<false, false>(dTypeQuantOut, normRes, wUqQr); // bf16  // quant: A8W8O32 -> bf16
    qkvPreRes.emplace_back(q);

    /******** kv ********/
    TileShape::Current().SetCubeTile({tieM, tieM}, {NUM_256, NUM_256}, {NUM_64, NUM_64});
    // [b*s,h] * [h,kvLoraRank+qkRopeHeadDim] = [b*s,kvLoraRank+qkRopeHeadDim]
    Tensor compressedKv;
    if (splitK) {
        TileShape::Current().SetVecTile(std::min(NUM_32, bs), NUM_64);
        int kvN = wDkvKr->shape[1];
        Tensor tmpCKv(DT_FP32, {bs, kvN}, "tmpKv");
        tmpCKv = MulS(tmpCKv, Element(DataType::DT_FP32, F_0));
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
        TileShape::Current().SetVecTile(std::min(NUM_32, bs), NUM_64);
        compressedKv = Cast(kvMmResF32, dType);
    } else {
        compressedKv = Matrix::Matmul(dType, input, wDkvKr); // bf16
    }
    Tensor compressedKvRes = Reshape(compressedKv, {b, s, wDkvKr->shape[1]});
    qkvPreRes.emplace_back(compressedKvRes);

    if (isQuant) {
        qkvPreRes.emplace_back(normDequantScale);
    }

    return qkvPreRes;
}

void MlaProlog(Tensor tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wUk, const Tensor &wDkvKr,
    const Tensor &gammaCq, const Tensor &gammaCkv, const Tensor &sin, const Tensor &cos, const Tensor &cacheIndex,
    Tensor &kvCache, Tensor &krCache, MlaQuantInputs quantInputs, const RoPETileShapeConfigNew &ropeConfig,
    Tensor &queryOut, Tensor &queryRopeOut, Tensor &kvCacheOut, Tensor &krCacheOut, float epsilonCq, float epsilonCkv,
    std::string cacheMode, bool splitReduceLastDim, bool splitK)
{
    // params check
    assert(tokenX->shape.size() == SHAPE_DIM3 && wUk->shape.size() == SHAPE_DIM3 && sin->shape.size() == SHAPE_DIM3);
    assert(kvCache->shape.size() == SHAPE_DIM4 && krCache->shape.size() == SHAPE_DIM4);
    assert(cacheMode == "BNSD" || cacheMode == "PA_BSND" || cacheMode == "PA_NZ");

    Tensor dequantScaleWUqQr = quantInputs.dequantScaleWUqQr;
    bool isQuant = (dequantScaleWUqQr.GetStorage() != nullptr);
    std::cout << "isQuant +++ " << isQuant << std::endl;

    DataType dType = tokenX->Datatype();
    int b = tokenX->shape[0];
    int s = tokenX->shape[1]; // s=1
    int bs = b * s;
    // [n, qkNopeHeadDim, kvLoraRank]
    int n = wUk->shape[0];
    int qkNopeHeadDim = wUk->shape[1];
    int kvLoraRank = wUk->shape[2];
    int qkRopeHeadDim = sin->shape[2]; // [b,s,qkRopeHeadDim]
    int qHeadDim = qkNopeHeadDim + qkRopeHeadDim;

    auto qKv = QkvPre(tokenX, wDq, wUqQr, wDkvKr, gammaCq, epsilonCq, quantInputs, splitReduceLastDim, splitK);
    Tensor q = qKv[0];     // [b*s, n*qHeadDim]
    Tensor kvTmp = qKv[1]; // [b,s,kvLoraRank+qkRopeHeadDim]

    // dequant: int32 -> fp32 -> *scale -> fp16/bf16
    if (isQuant) {
        std::vector<int64_t> tileShape = {bs, NUM_64};
        TileShape::Current().SetVecTile(tileShape);
        auto qTmpFp32 = Cast(q, DataType::DT_FP32);
        auto qTmpDequantScale = qKv[2];
        auto qTmpDequantPerToken = Mul(qTmpFp32, qTmpDequantScale);
        auto qTmpDequantChannel = Mul(qTmpDequantPerToken, dequantScaleWUqQr);

        q = Cast(qTmpDequantChannel, dType);
    }
    auto qTmp = Reshape(q, {b, s, n, qHeadDim});
    std::vector<int64_t> tileShape = {b, 1, 1, NUM_64};
    TileShape::Current().SetVecTile(tileShape);

    /******** q ********/
    Tensor qNope = View(qTmp, {b, s, n, qkNopeHeadDim}, {0, 0, 0, 0}); // [b,s,n,qkNopeHeadDim]
                                                                      // {NUM_2, 1, NUM_32, NUM_128}
    tileShape = {b, 1, 1, NUM_128};
    TileShape::Current().SetVecTile(tileShape);
    Tensor qNopeRes = Reshape(qNope, {bs, n, qkNopeHeadDim}); // [bs,n,qkNopeHeadDim]
    tileShape = {bs, 1, qkNopeHeadDim};     // {NUM_2, NUM_32, qkNopeHeadDim}
    TileShape::Current().SetVecTile(tileShape);
    Tensor qNopeTrans = Transpose(qNopeRes, {0, 1}); // [n,bs,qkNopeHeadDim]

    int c0 = NUM_16;
    int m = (bs + c0 - 1) / c0 * c0;
    TileShape::Current().SetCubeTile({m, m}, {NUM_128, NUM_128}, {NUM_128, NUM_128});
    // bmm: (n,bs,qkNopeHeadDim) * (n, qkNopeHeadDim, kvLoraRank) = (n, bs, kvLoraRank)
    Tensor qNopeNew = Matrix::BatchMatmul(dType, qNopeTrans, wUk);

    tileShape = {1, bs, kvLoraRank}; // {NUM_16, NUM_2, kvLoraRank}
    TileShape::Current().SetVecTile(tileShape);
    Tensor qNopeNewTrans = Transpose(qNopeNew, {0, 1});       // [bs,n,kvLoraRank]
    queryOut = Reshape(qNopeNewTrans, {b, s, n, kvLoraRank}); // [b,s,n,kvLoraRank], output1

    /******** kv ********/
    Tensor compressedKv = View(kvTmp, {b, s, kvLoraRank}, {0, 0, 0}); // [b,s,kvLoraRank]
    tileShape = {NUM_2, 1, NUM_512};
    TileShape::Current().SetVecTile(tileShape);
    Tensor compressedKvNorm = RmsNorm(compressedKv, gammaCkv, epsilonCkv); // [b,s,kvLoraRank]

    /******** RoPE ********/
    Tensor qPe = View(qTmp, {b, s, n, qkRopeHeadDim}, {0, 0, 0, qkNopeHeadDim}); // [b,s,n,qkRopeHeadDim]
    Tensor kPe = View(kvTmp, {b, s, qkRopeHeadDim}, {0, 0, kvLoraRank});         // [b,s,qkRopeHeadDim]
    tileShape = {bs, 1, NUM_64};
    TileShape::Current().SetVecTile(tileShape);
    Tensor kPeRes = Reshape(kPe, {b, s, 1, qkRopeHeadDim}); // [b,s,1,qkRopeHeadDim]

    Tensor kRope(kPeRes->Datatype(), {b, s, 1, qkRopeHeadDim}, "kRope"); // [b,1,s,qkRopeHeadDim]
    // queryRopeOut: [b,s,n,qkRopeHeadDim], output2
    ApplyRotaryPosEmbV2(qPe, kPeRes, cos, sin, queryRopeOut, kRope, 2, ropeConfig);

    if (cacheMode == "PA_BSND") {
        int blockNum = kvCache->shape[0];
        int blockSize = kvCache->shape[1];
        int n2 = kvCache->shape[2];
        Tensor kvCacheRes = Reshape(kvCache, {blockNum * blockSize * n2, kvLoraRank});
        Tensor krCacheRes = Reshape(krCache, {blockNum * blockSize * n2, qkRopeHeadDim});
        Tensor kNope = Reshape(compressedKvNorm, {b * s, kvLoraRank});       // [b*s,kvLoraRank]
        Tensor kRopeRes = Reshape(kRope, {b * s * 1, qkRopeHeadDim});

        /******** kvCache ********/
        tileShape = {1, kvLoraRank};
        TileShape::Current().SetVecTile(tileShape);
        // kvCache: [blockNum * blockSize * n2, kvLoraRank], output3
        Tensor kvCacheUpdate = ScatterUpdate(kvCacheRes, cacheIndex, kNope, -2, cacheMode);
        kvCacheOut = Reshape(kvCacheUpdate, {blockNum, blockSize, n2, kvLoraRank});

        /******** krCache ********/
        tileShape = {1, qkRopeHeadDim};
        TileShape::Current().SetVecTile(tileShape);
        // krCache: [blockNum * blockSize * n2, qkRopeHeadDim], output4
        Tensor krCacheUpdate = ScatterUpdate(krCacheRes, cacheIndex, kRopeRes, -2, cacheMode);
        krCacheOut = Reshape(krCacheUpdate, {blockNum, blockSize, n2, qkRopeHeadDim});
    } else {
        Tensor kNope = Reshape(compressedKvNorm, {b, 1, s, kvLoraRank});       // [b,1,s,kvLoraRank]
        Tensor kRopeRes = Reshape(kRope, {b, 1, s, qkRopeHeadDim});

        /******** kvCache ********/
        tileShape = {1, 1, 1, kvLoraRank};
        TileShape::Current().SetVecTile(tileShape);
        // kvCache: [b,1,s2,kvLoraRank], output3
        kvCacheOut = ScatterUpdate(kvCache, cacheIndex, kNope, -2); // cacheIndex: [b,s]

        /******** krCache ********/
        tileShape = {1, 1, 1, qkRopeHeadDim};
        TileShape::Current().SetVecTile(tileShape);
        // krCache: [b,1,s2,qkRopeHeadDim], output4
        krCacheOut = ScatterUpdate(krCache, cacheIndex, kRopeRes, -2); // cacheIndex: [b,s]
    }
}

} // namespace npu::tile_fwk

template <typename T = npu::tile_fwk::float16, bool splitReduceLastDim = true,
                        bool splitK = false, bool nz = false, bool usePrefetch = false>
void TestMlaPrologV2(std::vector<int> &params, bool isQuant = false, bool hasSmooth = false,
                     int blockSize = 128, std::string cacheMode = "BNSD")
{
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank
    int b = params[0];
    int s = params[1];
    int s2 = params[2];
    int n = params[3];
    int h = params[4];
    int qLoraRank = params[5];
    int qkNopeHeadDim = params[6];
    int qkRopeHeadDim = params[7];
    int kvLoraRank = params[8];
    int q_head_dim = qkNopeHeadDim + qkRopeHeadDim;

    DataType dType = DataType::DT_FP32;
    if (std::is_same<T, npu::tile_fwk::float16>::value) {
        dType = DataType::DT_FP16;
    } else if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        dType = DataType::DT_BF16;
    } else {
        dType = DataType::DT_FP32;
    }

    DataType dTypeQuantIn = isQuant ? DataType::DT_INT8 : dType;

    std::vector<int64_t> xShape = {b, s, h};
    std::vector<int64_t> wQaShape = {h, qLoraRank};
    std::vector<int64_t> wQbShape = {qLoraRank, n * q_head_dim};
    std::vector<int64_t> wKvAShape = {h, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> wKvBKShape = {n, qkNopeHeadDim, kvLoraRank};
    std::vector<int64_t> cosShape = {b, s, qkRopeHeadDim};
    std::vector<int64_t> gammaCqShape = {qLoraRank};
    std::vector<int64_t> gammaCkvShape = {kvLoraRank};
    std::vector<int64_t> kvLenShape = {b, s};
    std::vector<int64_t> kvCacheShape = {b, 1, s2, kvLoraRank};
    std::vector<int64_t> krCacheShape = {b, 1, s2, qkRopeHeadDim};
    // output
    std::vector<int64_t> qOutShape = {b, s, n, kvLoraRank};
    std::vector<int64_t> qRopeOutShape = {b, s, n, qkRopeHeadDim};
    if (cacheMode == "PA_BSND") {
        int blockNum = b * (s2 / blockSize);
        kvCacheShape = {blockNum, blockSize, 1, kvLoraRank};
        krCacheShape = {blockNum, blockSize, 1, qkRopeHeadDim};
    }

    ConfigManager::Instance();
    PROGRAM("MlaProlog") {
        Tensor x = Tensor(dType, xShape, "x"); // 32_1_7168
        TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
        Tensor wDq = Tensor(dType, wQaShape, "wDq", weightFormat);
        Tensor wUqQr = Tensor(dTypeQuantIn, wQbShape, "wUqQr", weightFormat);
        if constexpr (usePrefetch) {
            wDq.SetCachePolicy(CachePolicy::PREFETCH, true);
            wUqQr.SetCachePolicy(CachePolicy::PREFETCH, true);
        }
        Tensor wDkvKr = Tensor(dType, wKvAShape, "wDkvKr", weightFormat);
        Tensor wUk = Tensor(dType, wKvBKShape, "wUk", weightFormat);
        Tensor gammaCq = Tensor(dType, gammaCqShape, "gammaCq");
        Tensor gammaCkv = Tensor(dType, gammaCkvShape, "gammaCkv");
        Tensor cos = Tensor(dType, cosShape, "cos");
        Tensor sin = Tensor(dType, cosShape, "sin");
        Tensor kvLen = Tensor(DT_INT64, kvLenShape, "kvLen");  // int64
        Tensor kvCache = Tensor(dType, kvCacheShape, "kvCache");
        Tensor krCache = Tensor(dType, krCacheShape, "krCache");
        // output
        Tensor outputQ = Tensor(dType, qOutShape, "outputQ");
        Tensor outputQRope = Tensor(dType, qRopeOutShape, "outputQRope");

        RoPETileShapeConfigNew ropeConfig{
            {b, 1, 64}, // (b,s,d)
            {b, 1, 1, 64}, // Q (b,s,n,d)
            {b, 1, 1, 64}, // K (b,s,1,d)
            {b, 1, 1, 32, 2} // (b,s,n,d//2,2)
        };

        MlaQuantInputs quantInputs;

        if (isQuant) {
            std::vector<int64_t> wQbScaleShape = {1, n * q_head_dim};
            std::vector<int64_t> smoothCqShape = {1, qLoraRank};
            Tensor wQbScale = Tensor(DataType::DT_FP32, wQbScaleShape, "wQbScale");
            quantInputs.dequantScaleWUqQr = wQbScale;
            Tensor smoothCq = Tensor(DT_FP32, smoothCqShape, "smoothCq");
            if (hasSmooth) {
                quantInputs.smoothScalesCq = smoothCq;
                smoothCq.SetCachePolicy(CachePolicy::PREFETCH, true);
            }
            config::SetBuildStatic(true);
            FUNCTION("MlaPrologUt",
                {x, wDq, wUqQr, wQbScale, smoothCq, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos,
                    kvLen, kvCache, krCache, outputQ, outputQRope}) {
                MlaProlog(x, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos, kvLen, kvCache, krCache,
                    quantInputs, ropeConfig, outputQ, outputQRope, kvCache, krCache, 1e-5f, 1e-5f, cacheMode,
                    splitReduceLastDim,  splitK);
            };
        } else {
            config::SetBuildStatic(true);
            FUNCTION("MlaPrologUt",
                {x, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos,
                    kvLen, kvCache, krCache, outputQ, outputQRope}) {
                MlaProlog(x, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos, kvLen, kvCache, krCache,
                    quantInputs, ropeConfig, outputQ, outputQRope, kvCache, krCache, 1e-5f, 1e-5f, cacheMode,
                    splitReduceLastDim,  splitK);
            };
        }
    }
}

int main()
{
    int b = 32;
    int s = 1;
    int s2 = 4096;
    int h = 7168;
    int n = 128;
    int qLoraRank = 1536;
    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;
    int kvLoraRank = 512;

    int blockSize = 128;
    std::string cacheMode = "PA_BSND";

    const bool splitReduceLastDim = false;
    const bool splitK = false;
    const bool nz = true;
    const bool usePrefetch = false;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank};
    TestMlaPrologV2<npu::tile_fwk::bfloat16, splitReduceLastDim, splitK, nz, usePrefetch>(params,
        true, true, blockSize, cacheMode);
}