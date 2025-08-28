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
 * \file deepseek_mla.cpp
 * \brief
 */

#include "operator/models/deepseek/deepseek_mla.h"

#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "tilefwk/tensor.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {
DeepseekAttention::DeepseekAttention(
    std::map<std::string, std::variant<bool, int, float, std::string>> config, AttentionW aw, const int inLayerIdx)
    : layerIdx(inLayerIdx) {
    attentionDropout = std::get<int>(config["attentionDropout"]);
    hiddenSize = std::get<int>(config["hiddenSize"]);
    numHeads = std::get<int>(config["numAttentionHeads"]);
    maxPositionEmbeddings = std::get<int>(config["maxPositionEmbeddings"]);
    ropeTheta = std::get<int>(config["ropeTheta"]);
    qLoraRank = std::get<int>(config["qLoraRank"]);
    qkRopeHeadDim = std::get<int>(config["qkRopeHeadDim"]);
    kvLoraRank = std::get<int>(config["kvLoraRank"]);
    vHeadDim = std::get<int>(config["vHeadDim"]);
    qkNopeHeadDim = std::get<int>(config["qkNopeHeadDim"]);
    qHeadDim = qkNopeHeadDim + qkRopeHeadDim;
    isCausal = true;

    qAProjW = aw.qAProjW;
    qBProjW = aw.qBProjW;
    qBProjWScale = aw.qBProjWScale;
    kvAProjWithMqaW = aw.kvAProjWithMqaW;
    kvBProjWK = aw.kvBProjWK;
    kvBProjWV = aw.kvBProjWV;
    oProjW = aw.oProjW;

    softmaxScale = static_cast<float>(1.0 / std::sqrt(qHeadDim));

    // "ropeScaling": {
    //     "beta_fast": NUM_32,
    //     "beta_slow": 1,
    //     "factor": 40,
    //     "mscale": 1.0,
    //     "mscaleAllDim": 1.0,
    //     "original_max_position_embeddings": 4096,
    //     "type": "yarn"
    //   },
    if (std::get<int>(config["ropeScaling"]) == 1) {
        int factor = 40;
        float mscale = 1.0;
        float mscaleAllDim = 1.0;
        double valuePointOne = 0.1;
        if (mscaleAllDim > 1) {
            mscale = static_cast<float>(valuePointOne * mscale * std::log(factor) + 1.0);
        }
        softmaxScale = softmaxScale * mscale * mscale;
    }
}

Tensor DeepseekAttention::Attention(Tensor q, Tensor kv, Tensor attenMask) {
    // q: [b,numHeads,s, kvLoraRank + qkRopeHeadDim]
    // kv: [b,1,s2, kvLoraRank + qkRopeHeadDim]
    int b = q->shape[0];
    int n2 = kv->shape[1]; // 1
    int s1 = q->shape[2];
    int s2 = kv->shape[2];
    int kvLoraRankV = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    DataType dType = q->Datatype();

    Program::GetInstance().GetTileShape().SetCubeTileShapes({std::min(NUM_128, s1), std::min(NUM_128, s1)}, {NUM_64, NUM_64}, {NUM_128, NUM_128});
    // Program::GetInstance().GetTileShape().SetVecTileShapes({NUM_128, NUM_64, NUM_128, NUM_64}); //  bmm接口增加一个config参数
    //  [b,n,s1, kvLoraRank + qkRopeHeadDim] * [b,1, kvLoraRank + qkRopeHeadDim, s2] = [b,n,s1,s2]
    Tensor qk = Matrix::BatchMatmul<false, true>(dType, q, kv);
    Program::GetInstance().GetTileShape().SetVecTileShapes({1, 1, NUM_128, NUM_64});
    Tensor qkFp32 = Cast(qk, DataType::DT_FP32);
    qkFp32 = MulS(qkFp32, Element(DataType::DT_FP32, static_cast<double>(softmaxScale)));
    qkFp32 = Add(qkFp32, attenMask);
    Tensor qk16 = Cast(qkFp32, dType);
    Tensor softmax = SoftmaxNew(qk16); // [b,n,s1,s2]
    // no drop
    Tensor v = View(kv, {b, n2, s2, kvLoraRankV}, {0, 0, 0, 0});
    Program::GetInstance().GetTileShape().SetCubeTileShapes({std::min(NUM_128, s1), std::min(NUM_128, s1)}, {NUM_64, NUM_64}, {NUM_128, NUM_128});
    // Program::GetInstance().GetTileShape().SetVecTileShapes({NUM_128, NUM_64, NUM_128, NUM_64}); //  bmm接口增加一个config参数
    // [b,n,s1,s2] * [b,1, s2, kvLoraRank] = [b,n,s1,kvLoraRank]
    Tensor attenRes = Matrix::BatchMatmul(dType, softmax, v);
    return attenRes;
}

Tensor DeepseekAttention::AttentionPost(Tensor attenRes) {
    // attenRes: [b,n,s,kvLoraRank]
    int b = attenRes->shape[0];
    int n = attenRes->shape[1];
    int s = attenRes->shape[2];
    int bs = b * s;
    DataType dType = attenRes->Datatype();

    Program::GetInstance().GetTileShape().SetVecTileShapes({1, 1, 1, NUM_512});
    Tensor attenRes0 = Transpose(attenRes, {1, 2});
    Program::GetInstance().GetTileShape().SetVecTileShapes({1, 1, NUM_128, NUM_64});
    Tensor attenRes1 = Reshape(attenRes0, {b * s, n, kvLoraRank});
    Program::GetInstance().GetTileShape().SetVecTileShapes({1, 1, NUM_512});
    Tensor attenRes2 = Transpose(attenRes1, {0, 1});
    Program::GetInstance().GetTileShape().SetVecTileShapes({1, NUM_128, NUM_64});
    Program::GetInstance().GetTileShape().SetCubeTileShapes({std::min(NUM_128, bs), std::min(NUM_128, bs)}, {NUM_128, NUM_128}, {NUM_128, NUM_128});
    Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_128, NUM_64);
    // [n,bs,kvLoraRank] * [n, kvLoraRank, vHeadDim] = [n,bs,vHeadDim]
    Tensor mm7Res = Matrix::BatchMatmul(dType, attenRes2, kvBProjWV);

    Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, NUM_128);
    Tensor mm7Res1 = Transpose(mm7Res, {0, 1}); // [bs,n,vHeadDim]
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, NUM_128, NUM_64);
    Tensor mm7Res2 = Reshape(mm7Res1, {b, s, n * vHeadDim});

    Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_128, NUM_64);
    // [b,s, n*vHeadDim] @ [n*vHeadDim, h] = [b,s,h]
    Tensor attnOutW = Unsqueeze(oProjW, 0);
    Program::GetInstance().GetTileShape().SetCubeTileShapes({std::min(NUM_128, s), std::min(NUM_128, s)}, {NUM_128, NUM_128}, {NUM_128, NUM_128});
    Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_128, NUM_64);
    Tensor attenOutput = Matrix::BatchMatmul(dType, mm7Res2, attnOutW);

    return attenOutput;
}

Tensor DeepseekAttention::AttentionPost2(Tensor attenRes) {
    // attenRes: [b,n,s,kvLoraRank]
    int b = attenRes->shape[0];
    int n = attenRes->shape[1];
    int s = attenRes->shape[2];
    int bs = b * s;
    int h = oProjW->shape[1];
    DataType dType = attenRes->Datatype();

    Program::GetInstance().GetTileShape().SetVecTileShapes({NUM_16, NUM_16, 1, NUM_128});
    Tensor attenRes0 = Transpose(attenRes, {1, 2});
    Program::GetInstance().GetTileShape().SetVecTileShapes({NUM_16, 1, NUM_16, NUM_128});
    Tensor attenRes1 = Reshape(attenRes0, {b * s, n, kvLoraRank});
    Program::GetInstance().GetTileShape().SetVecTileShapes({NUM_16, NUM_16, NUM_128});
    Tensor attenRes2 = Transpose(attenRes1, {0, 1});
    Program::GetInstance().GetTileShape().SetCubeTileShapes({std::min(NUM_128, bs), std::min(NUM_128, bs)}, {NUM_128, NUM_128}, {std::min(NUM_128, h), std::min(NUM_128, h)});
    // [n,bs,kvLoraRank] * [n, kvLoraRank, vHeadDim] = [n,bs,vHeadDim]
    Tensor mm7Res = Matrix::BatchMatmul(dType, attenRes2, kvBProjWV);

    Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_16, NUM_16, NUM_128);
    Tensor mm7Res1 = Transpose(mm7Res, {0, 1}); // [bs,n,vHeadDim]
    Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_16, NUM_16, NUM_128);
    Tensor mm7Res2 = Reshape(mm7Res1, {b, s, n * vHeadDim});

    Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_128, std::min(NUM_256, h));
    // [b,s, n*vHeadDim] @ [n*vHeadDim, h] = [b,s,h]
    Tensor attnOutW = Unsqueeze(oProjW, 0);
    Program::GetInstance().GetTileShape().SetCubeTileShapes({std::min(NUM_128, s), std::min(NUM_128, s)}, {NUM_128, NUM_128}, {std::min(NUM_128, h), std::min(NUM_128, h)});
    Tensor attenOutput = Matrix::BatchMatmul(dType, mm7Res2, attnOutW);

    return attenOutput;
}

std::tuple<Tensor, Tensor> DeepseekAttention::QkvPre(Tensor hiddenStates) {
    int b = hiddenStates->shape[0];
    int s = hiddenStates->shape[1];
    DataType dType = hiddenStates->Datatype();

    Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_128, NUM_64);
    Tensor qAProjW1 = Unsqueeze(qAProjW, 0);
    Tensor qBProjW1 = Unsqueeze(qBProjW, 0);
    Tensor kvAProjWithMqaW1 = Unsqueeze(kvAProjWithMqaW, 0);

    Program::GetInstance().GetTileShape().SetCubeTileShapes({std::min(NUM_128, s), std::min(NUM_128, s)}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
    Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_128, NUM_64); // for Assemble
    // qAProj, bmm: (b, s, h) * (1, h, qLoraRank) = (b, s, qLoraRank)
    Tensor qAProj = Matrix::BatchMatmul(dType, hiddenStates, qAProjW1); // bf16

    Program::GetInstance().GetTileShape().SetVecTileShapes(1, NUM_128, NUM_64);
    Tensor qALayerNorm = RmsNorm(qAProj);

    Program::GetInstance().GetTileShape().SetCubeTileShapes({std::min(NUM_128, s), std::min(NUM_128, s)}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
    Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_128, NUM_64); // for Assemble
    // q_b_proj, bmm: (b, s, qLoraRank) * (qLoraRank, numHeads * qHeadDim) = (b, s, numHeads * qHeadDim)
    Tensor q = Matrix::BatchMatmul(dType, qALayerNorm, qBProjW1);

    Program::GetInstance().GetTileShape().SetVecTileShapes(1, NUM_128, NUM_64);
    // (b, s, numHeads, qHeadDim)
    Tensor q2 = Reshape(q, {b, s, numHeads, qHeadDim});

    Program::GetInstance().GetTileShape().SetCubeTileShapes({std::min(NUM_128, s), std::min(NUM_128, s)}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
    Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_128, NUM_64); // for Assemble
    // kv_a_proj_with_mqa, bmm: (b, s, h) * (h, kvLoraRank + qkRopeHeadDim) = (b, s, kvLoraRank +
    // qkRopeHeadDim)
    Tensor compressedKv = Matrix::BatchMatmul(dType, hiddenStates, kvAProjWithMqaW1); // bf16

    return std::tie(q2, compressedKv);
}

std::tuple<Tensor, Tensor> DeepseekAttention::QkvPreCv(Tensor hiddenStates) {
    int b = hiddenStates->shape[0];
    int s = hiddenStates->shape[1];
    DataType dType = hiddenStates->Datatype();

    Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_128, NUM_64);
    Tensor qAProjW1 = Unsqueeze(qAProjW, 0); // [NUM_256,NUM_512]
    Tensor qBProjW1 = Unsqueeze(qBProjW, 0); // [NUM_512,2*192]
    Tensor kvAProjWithMqaW1 = Unsqueeze(kvAProjWithMqaW, 0); // [NUM_256,576]

    Program::GetInstance().GetTileShape().SetCubeTileShapes({std::min(NUM_128, s), std::min(NUM_128, s)}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
    // qAProj, bmm: (b, s, h) * (1, h, qLoraRank) = (b, s, qLoraRank)
    // [2,1,256] * [1,256,512]
    Tensor qAProj = Matrix::BatchMatmul(dType, hiddenStates, qAProjW1); // bf16  2_1_512

    Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_2, 1, NUM_512);
    Tensor qALayerNorm = RmsNorm(qAProj); // 2_1_512

    Program::GetInstance().GetTileShape().SetCubeTileShapes({std::min(NUM_128, s), std::min(NUM_128, s)}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
    // q_b_proj, bmm: (b, s, qLoraRank) * (qLoraRank, numHeads * qHeadDim) = (b, s, numHeads * qHeadDim)
    // 2_1_512 * 1_512_2*192
    // 2_1_2*192
    Tensor q = Matrix::BatchMatmul(dType, qALayerNorm, qBProjW1);

    Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_2, 1, NUM_384);
    // (b, s, numHeads, qHeadDim) // 2_1_2_192
    Tensor q2 = Reshape(q, {b, s, numHeads, qHeadDim});

    Program::GetInstance().GetTileShape().SetCubeTileShapes({std::min(NUM_128, s), std::min(NUM_128, s)}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
    // Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_128, NUM_64); // for Assemble
    // kv_a_proj_with_mqa, bmm: (b, s, h) * (h, kvLoraRank + qkRopeHeadDim) = (b, s, kvLoraRank +
    // qkRopeHeadDim)
    // 2_1_256  1_256_576
    Tensor compressedKV = Matrix::BatchMatmul(dType, hiddenStates, kvAProjWithMqaW1); // bf16
    // 2_1_32_192 2_1_576
    return std::tie(q2, compressedKV);
}

std::vector<Tensor> DeepseekAttention::QkvPre2(Tensor hiddenStates, bool isQuant) {
    int b = hiddenStates->shape[0];
    int s = hiddenStates->shape[1];
    int h = hiddenStates->shape[2];
    int bs = b * s;

    DataType dType = hiddenStates->Datatype();
    DataType dTypeQuantOut = isQuant ? DataType::DT_INT32 : dType;
    std::vector<Tensor> qkvPre2Res;

    Tensor input = Reshape(hiddenStates, {bs, h});  // [b,s,h] -> [b*s,h]

    int c0 = NUM_16;
    int m = (std::min(NUM_32, bs) + c0 - 1) / c0 * c0;
    int tileM = std::min(NUM_16, m);
    Program::GetInstance().GetTileShape().SetCubeTileShapes({tileM, tileM}, {NUM_256, NUM_256}, {NUM_128, NUM_128});
    // [b*s,h] * [h,qLoraRank] = [b*s,qLoraRank]
    Tensor qAProj = Matrix::Matmul<false, false>(dType, input, qAProjW);  // bf16

    Program::GetInstance().GetTileShape().SetVecTileShapes(std::min(NUM_16, bs), NUM_128);
    Tensor qAProjNorm = RmsNorm(qAProj);

    Tensor qAProjNormScaleDequant;
    if (isQuant) {
        auto qAProjNormQuantRes = Quant(qAProjNorm);    //int8
        qAProjNorm = std::get<0>(qAProjNormQuantRes);
        qAProjNormScaleDequant = std::get<1>(qAProjNormQuantRes);
        Program::GetInstance().GetTileShape().SetCubeTileShapes({tileM, tileM}, {NUM_256, NUM_256}, {NUM_256, NUM_256});
    } else {
        Program::GetInstance().GetTileShape().SetCubeTileShapes({m, m}, {NUM_256, NUM_256}, {NUM_64, NUM_64});
    }
    // [b*s,qLoraRank] * [qLoraRank, N*qHeadDim] = [b*s, N*qHeadDim]
    Tensor q = Matrix::Matmul<false, false>(dTypeQuantOut, qAProjNorm, qBProjW);  // bf16  // quant  A8W8O32  ->  bf16
    qkvPre2Res.emplace_back(q);

    Program::GetInstance().GetTileShape().SetCubeTileShapes({m, m}, {NUM_256, NUM_256}, {NUM_64, NUM_64});
    // [b*s,h] * [h,kvLoraRank+qkRopeHeadDim] = [b*s,kvLoraRank+qkRopeHeadDim]
    Tensor compressedKv = Matrix::Matmul<false, false>(dType, input, kvAProjWithMqaW);  // bf16
    Tensor compressedKvRes = Reshape(compressedKv, {b, s, kvLoraRank + qkRopeHeadDim});
    qkvPre2Res.emplace_back(compressedKvRes);

    if (isQuant) {
        qkvPre2Res.emplace_back(qAProjNormScaleDequant);
    }

    return qkvPre2Res;
}

std::tuple<Tensor, Tensor> DeepseekAttention::QkvPreFp32(Tensor hiddenStates) {
    int b = hiddenStates->shape[0];
    int s = hiddenStates->shape[1];
    int h = hiddenStates->shape[2];
    int bs = b * s;
    DataType dType = hiddenStates->Datatype();

    Tensor input = Reshape(hiddenStates, {bs, h});  // [b,s,h] -> [b*s,h]

    Program::GetInstance().GetTileShape().SetCubeTileShapes({std::min(NUM_64, bs), std::min(NUM_64, bs)}, {NUM_256, NUM_256}, {NUM_128, NUM_128});
    // [b*s,h] * [h,qLoraRank] = [b*s,qLoraRank]
    // [NUM_32*1,NUM_256] * [NUM_256,NUM_512] = [NUM_32*1,NUM_512]
    Tensor qAProjFp32 = Matrix::Matmul<false, false>(DataType::DT_FP32, input, qAProjW);  // fp32

    Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_32, NUM_128);
    Tensor qAProjNormFp32 = RmsNorm(qAProjFp32);  // fp32

    std::vector<int64_t> tileShape = {NUM_32, NUM_128};
    Program::GetInstance().GetTileShape().SetVecTileShapes(tileShape);
    Tensor qAProjNorm = Cast(qAProjNormFp32, dType);  // bf16

    Program::GetInstance().GetTileShape().SetCubeTileShapes({std::min(NUM_64, bs), std::min(NUM_64, bs)}, {NUM_256, NUM_256}, {NUM_64, NUM_64});
    // [b*s,qLoraRank] * [qLoraRank, N*qHeadDim] = [b*s, N*qHeadDim]
    // [NUM_32*1,NUM_512] * [NUM_512, 2*192] = [NUM_32*1, 2*192]
    Tensor qFp32 = Matrix::Matmul<false, false>(DataType::DT_FP32, qAProjNorm, qBProjW);  // fp32
    Tensor qRes = Reshape(qFp32, {b, s, numHeads, qHeadDim});

    Program::GetInstance().GetTileShape().SetCubeTileShapes({std::min(NUM_64, bs), std::min(NUM_64, bs)}, {NUM_256, NUM_256}, {NUM_64, NUM_64});
    // [b*s,h] * [h,kvLoraRank+qkRopeHeadDim] = [b*s,kvLoraRank+qkRopeHeadDim]
    // [NUM_32*1,NUM_256] * [NUM_256,NUM_512+NUM_64] = [NUM_32*1,NUM_512+NUM_64]
    Tensor compressedKvFp32 = Matrix::Matmul<false, false>(DataType::DT_FP32, input, kvAProjWithMqaW);  // fp32
    Tensor compressedKvRes = Reshape(compressedKvFp32, {b, s, kvLoraRank + qkRopeHeadDim});

    return std::tie(qRes, compressedKvRes);
}


// mm/bmm: bf16 in, bf16 out
Tensor DeepseekAttention::Forward(Tensor hiddenStates, Tensor attenMask, Tensor positionIds, Tensor cos, Tensor sin,
    Tensor kvLen, Tensor pastKeyStates, const RoPETileShapeConfig &ropeTileShapeConfig) {
    // hiddenStates: (b,s,h), attention_mask: (b,1,s,s2), positionIds: (b,s)
    int b = hiddenStates->shape[0];
    int s = hiddenStates->shape[1];
    int bs = b * s;
    DataType dType = hiddenStates->Datatype();

    /*** prepare q k v ***/
    auto qKv = QkvPre(hiddenStates);
    Tensor q = std::get<0>(qKv);
    Tensor compressedKv = std::get<1>(qKv);

    Tensor qNope = View(q, {b, s, numHeads, qkNopeHeadDim}, {0, 0, 0, 0});
    Tensor qPe = View(q, {b, s, numHeads, qkRopeHeadDim}, {0, 0, 0, qkNopeHeadDim});
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, 1, NUM_64);
    qPe = Transpose(qPe, {1, 2}); // setTileShapes 4维
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, NUM_128, NUM_64);

    // 先View kPe,防止compress_kv变化影响k_pe
    Tensor kPe = View(compressedKv, {b, s, qkRopeHeadDim}, {0, 0, kvLoraRank}); // (b,s,qkRopeHeadDim)
    compressedKv = View(compressedKv, {b, s, kvLoraRank}, {0, 0, 0});              // (b,s,kvLoraRank)
    // [b,s, qkRopeHeadDim] -> [b,s, 1,qkRopeHeadDim] -> [b,1,s,qkRopeHeadDim]
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, NUM_128, NUM_64);
    kPe = Reshape(kPe, {b, 1, s, qkRopeHeadDim}); // setTileShapes 4维

    Program::GetInstance().GetTileShape().SetVecTileShapes(1, NUM_128, 1, NUM_64); // SetVecTileShapes(1, 1, NUM_128, NUM_64)
    // (b, s, n, qkNopeHeadDim) * (n, qkNopeHeadDim, kvLoraRank) -> (b, s, numHeads, kvLoraRank)
    Tensor qNope1 = Reshape(qNope, {b * s, numHeads, qkNopeHeadDim});
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, NUM_128);
    Tensor qNope2 = Transpose(qNope1, {0, 1}); // (n,bs,d)
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, NUM_128, NUM_64);

    Program::GetInstance().GetTileShape().SetCubeTileShapes({std::min(NUM_128, bs), std::min(NUM_128, bs)}, {NUM_128, NUM_128}, {NUM_128, NUM_128});
    Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_128, NUM_64); // for Assemble
    // bmm: (n,bs,d) * (n, d, kvLoraRank) = (n, bs, kvLoraRank)
    Tensor qNopeNew = Matrix::BatchMatmul(dType, qNope2, kvBProjWK);
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, NUM_512);
    Tensor qNopeNew2 = Transpose(qNopeNew, {0, 1});
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, NUM_128, NUM_64);
    qNopeNew2 = Reshape(qNopeNew2, {b, s, numHeads, kvLoraRank});
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, 1, NUM_512);
    qNopeNew2 = Transpose(qNopeNew2, {1, 2});
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, NUM_128, NUM_64); // (b,n,s,kvLoraRank)

    Program::GetInstance().GetTileShape().SetVecTileShapes(1, NUM_128, NUM_64);
    Tensor kNope = RmsNorm(compressedKv); // (b,s,kvLoraRank)
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, NUM_128, NUM_64);
    kNope = Reshape(kNope, {b, 1, s, kvLoraRank}); // (b,1,s,kvLoraRank)

    Tensor qPeRope(qPe->Datatype(), {b, numHeads, s, qkRopeHeadDim}, "qPeRope");
    // (b,numHeads,s,qkRopeHeadDim)
    Tensor kPeRope(kPe->Datatype(), {b, 1, s, qkRopeHeadDim}, "kPeRope"); // (b,1,s,qkRopeHeadDim)
    ApplyRotaryPosEmb(qPe, kPe, cos, sin, positionIds, qPeRope, kPeRope, 1, ropeTileShapeConfig);
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, NUM_128, NUM_64);

    Tensor queryStates = Concat({qNopeNew2, qPeRope}, -1); // (b,numHeads,s, kvLoraRank + qkRopeHeadDim)
    Tensor keyStates = Concat({kNope, kPeRope}, -1);        // (b,1,s, kvLoraRank + qkRopeHeadDim)

    // pastKeyStates: [b,1,s2, kvLoraRank + qkRopeHeadDim]
    auto pastKeyStatesNew = ScatterUpdate(pastKeyStates, kvLen, keyStates, -2); // 增量
    //
    Tensor attenRes = Attention(queryStates, pastKeyStatesNew, attenMask); // 增量

    return AttentionPost(attenRes);
}

// mm/bmm: bf16 in, bf16 out
std::tuple<Tensor, Tensor>  DeepseekAttention::AtentionPreForward(Tensor hiddenStates, Tensor attenMask,
    Tensor positionIds, Tensor cos, Tensor sin, Tensor kvLen, Tensor pastKeyStates,
    const RoPETileShapeConfig &ropeTileShapeConfig) {
    (void)attenMask;
    // hiddenStates: (b,s,h), attention_mask: (b,1,s,s2), positionIds: (b,s)
    int b = hiddenStates->shape[0];
    int s = hiddenStates->shape[1];
    int bs = b * s;
    DataType dType = hiddenStates->Datatype();

    /*** prepare q k v ***/
    auto qKv = QkvPre(hiddenStates);
    Tensor q = std::get<0>(qKv);
    Tensor compressedKv = std::get<1>(qKv);

    Tensor qNope = View(q, {b, s, numHeads, qkNopeHeadDim}, {0, 0, 0, 0});
    Tensor qPe = View(q, {b, s, numHeads, qkRopeHeadDim}, {0, 0, 0, qkNopeHeadDim});
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, 1, NUM_64);
    qPe = Transpose(qPe, {1, 2}); // setTileShapes 4维
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, NUM_128, NUM_64);

    // 先View kPe,防止compress_kv变化影响k_pe
    Tensor kPe = View(compressedKv, {b, s, qkRopeHeadDim}, {0, 0, kvLoraRank}); // (b,s,qkRopeHeadDim)
    compressedKv = View(compressedKv, {b, s, kvLoraRank}, {0, 0, 0});              // (b,s,kvLoraRank)
    // [b,s, qkRopeHeadDim] -> [b,s, 1,qkRopeHeadDim] -> [b,1,s,qkRopeHeadDim]
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, NUM_128, NUM_64);
    kPe = Reshape(kPe, {b, 1, s, qkRopeHeadDim}); // setTileShapes 4维

    Program::GetInstance().GetTileShape().SetVecTileShapes(1, NUM_128, 1, NUM_64); // SetVecTileShapes(1, 1, NUM_128, NUM_64)
    // (b, s, n, qkNopeHeadDim) * (n, qkNopeHeadDim, kvLoraRank) -> (b, s, numHeads, kvLoraRank)
    Tensor qNope1 = Reshape(qNope, {b * s, numHeads, qkNopeHeadDim});
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, NUM_128);
    Tensor qNope2 = Transpose(qNope1, {0, 1}); // (n,bs,d)
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, NUM_128, NUM_64);

    Program::GetInstance().GetTileShape().SetCubeTileShapes({std::min(NUM_128, bs), std::min(NUM_128, bs)}, {NUM_128, NUM_128}, {NUM_128, NUM_128});
    Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_128, NUM_64); // for Assemble
    // bmm: (n,bs,d) * (n, d, kvLoraRank) = (n, bs, kvLoraRank)
    Tensor qNopeNew = Matrix::BatchMatmul(dType, qNope2, kvBProjWK);
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, NUM_512);
    Tensor qNopeNew2 = Transpose(qNopeNew, {0, 1});
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, NUM_128, NUM_64);
    qNopeNew2 = Reshape(qNopeNew2, {b, s, numHeads, kvLoraRank});
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, 1, NUM_512);
    qNopeNew2 = Transpose(qNopeNew2, {1, 2});
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, NUM_128, NUM_64); // (b,n,s,kvLoraRank)

    Program::GetInstance().GetTileShape().SetVecTileShapes(1, NUM_128, NUM_64);
    Tensor kNope = RmsNorm(compressedKv); // (b,s,kvLoraRank)
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, NUM_128, NUM_64);
    kNope = Reshape(kNope, {b, 1, s, kvLoraRank}); // (b,1,s,kvLoraRank)

    Tensor qPeRope(qPe->Datatype(), {b, numHeads, s, qkRopeHeadDim}, "qPeRope");
    // (b,numHeads,s,qkRopeHeadDim)
    Tensor kPeRope(kPe->Datatype(), {b, 1, s, qkRopeHeadDim}, "kPeRope"); // (b,1,s,qkRopeHeadDim)
    ApplyRotaryPosEmb(qPe, kPe, cos, sin, positionIds, qPeRope, kPeRope, 1, ropeTileShapeConfig);
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, NUM_128, NUM_64);

    Tensor queryStates = Concat({qNopeNew2, qPeRope}, -1); // (b,numHeads,s, kvLoraRank + qkRopeHeadDim)
    Tensor keyStates = Concat({kNope, kPeRope}, -1);        // (b,1,s, kvLoraRank + qkRopeHeadDim)

    // pastKeyStates: [b,1,s2, kvLoraRank + qkRopeHeadDim]
    auto pastKeyStatesNew = ScatterUpdate(pastKeyStates, kvLen, keyStates, -2); // 增量
    return std::tie(queryStates, pastKeyStatesNew);
    //
}

// mm/bmm: bf16 in, bf16 out
std::tuple<Tensor, Tensor>  DeepseekAttention::AtentionPreForwardCv(Tensor hiddenStates, Tensor attenMask,
    Tensor positionIds, Tensor cos, Tensor sin, Tensor kvLen, Tensor pastKeyStates,
    const RoPETileShapeConfig &ropeTileShapeConfig) {
    (void)attenMask;
    // hiddenStates: (b,s,h), attention_mask: (b,1,s,s2), positionIds: (b,s)
    int b = hiddenStates->shape[0];
    int s = hiddenStates->shape[1];
    int bs = b * s;
    DataType dType = hiddenStates->Datatype();

    /*** prepare q k v ***/
    // 2_1_32_192 2_1_576
    auto qKv = QkvPreCv(hiddenStates);
    Tensor q = std::get<0>(qKv); //2_1_32_192
    Tensor compressedKv = std::get<1>(qKv); //2_1_576

    Tensor qNope = View(q, {b, s, numHeads, qkNopeHeadDim}, {0, 0, 0, 0}); // 2_1_32_128
    Tensor qPe = View(q, {b, s, numHeads, qkRopeHeadDim}, {0, 0, 0, qkNopeHeadDim}); // 2_1_32_64
    Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_2, 1, NUM_32, NUM_64);
    qPe = Transpose(qPe, {1, 2}); // setTileShapes 4维 2_32_1_64

    // 先View kPe,防止compress_kv变化影响k_pe
    Tensor kPe = View(compressedKv, {b, s, qkRopeHeadDim}, {0, 0, kvLoraRank}); // (b,s,qkRopeHeadDim) 2_1_64
    compressedKv = View(compressedKv, {b, s, kvLoraRank}, {0, 0, 0});              // (b,s,kvLoraRank) 2_1_512
    // [b,s, qkRopeHeadDim] -> [b,s, 1,qkRopeHeadDim] -> [b,1,s,qkRopeHeadDim]
    Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_2, 1, NUM_64);
    kPe = Reshape(kPe, {b, 1, s, qkRopeHeadDim}); // setTileShapes 4维 2_1_1_64

    Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_2, 1, NUM_32, NUM_128); // SetVecTileShapes(1, 1, NUM_128, NUM_64)
    // (b, s, n, qkNopeHeadDim) * (n, qkNopeHeadDim, kvLoraRank) -> (b, s, numHeads, kvLoraRank)
    Tensor qNope1 = Reshape(qNope, {b * s, numHeads, qkNopeHeadDim}); // 2_32_128
    Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_2, NUM_32, NUM_128);
    Tensor qNope2 = Transpose(qNope1, {0, 1}); // (n,bs,d) 32_2_128
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, NUM_128, NUM_64);

    Program::GetInstance().GetTileShape().SetCubeTileShapes({std::min(NUM_128, bs), std::min(NUM_128, bs)}, {NUM_128, NUM_128}, {NUM_128, NUM_128});
    // Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_128, NUM_64); // for Assemble
    // bmm: (n,bs,d) * (n, d, kvLoraRank) = (n, bs, kvLoraRank)
    //32_2_128 * 32_128_512 = 32_2_512
    Tensor qNopeNew = Matrix::BatchMatmul(dType, qNope2, kvBProjWK);
    Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_16, NUM_2, NUM_512);
    Tensor qNopeNew2 = Transpose(qNopeNew, {0, 1}); // 2_32_512
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, NUM_32, NUM_512);
    qNopeNew2 = Reshape(qNopeNew2, {b, s, numHeads, kvLoraRank}); // 2_1_32_512
    Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_2, 1, NUM_32, NUM_256);
    qNopeNew2 = Transpose(qNopeNew2, {1, 2}); //2_32_1_512

    Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_2, 1, NUM_512);
    Tensor kNope = RmsNorm(compressedKv); // (b,s,kvLoraRank) 2_1_512
    Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_2, 1, NUM_512);
    kNope = Reshape(kNope, {b, 1, s, kvLoraRank}); // (b,1,s,kvLoraRank) 2_1_1_512

    Tensor qPeRope(qPe->Datatype(), {b, numHeads, s, qkRopeHeadDim}, "qPeRope"); // 2_32_1_64
    // (b,numHeads,s,qkRopeHeadDim)
    Tensor kPeRope(kPe->Datatype(), {b, 1, s, qkRopeHeadDim}, "kPeRope"); // (b,1,s,qkRopeHeadDim) 2_1_1_64
    // 2_32_1_64  2_1_1_64  1_64  1_64  2_1  2_32_1_64  2_1_1_64
    ApplyRotaryPosEmb(qPe, kPe, cos, sin, positionIds, qPeRope, kPeRope, 1, ropeTileShapeConfig);
    Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_2, NUM_32, 1, NUM_64);
   //2_32_1_512 + 2_32_1_64 = 2_32_1_576
    Tensor queryStates = Concat({qNopeNew2, qPeRope}, -1); // (b,numHeads,s, kvLoraRank + qkRopeHeadDim)
    //2_32_1_512 + 2_32_1_64 = 2_32_1_576
    Tensor keyStates = Concat({kNope, kPeRope}, -1);        // (b,1,s, kvLoraRank + qkRopeHeadDim)

    // pastKeyStates: [b,1,s2, kvLoraRank + qkRopeHeadDim]
    // 2_1_256_576
    Program::GetInstance().GetTileShape().SetVecTileShapes(NUM_2, 1, NUM_128, NUM_64);
    auto pastKeyStatesNew = ScatterUpdate(pastKeyStates, kvLen, keyStates, -2); // 增量
    return std::tie(queryStates, pastKeyStatesNew);
}

std::tuple<Tensor, Tensor> DeepseekAttention::MlaPrologAbForward(Tensor hiddenStates, Tensor qPeRope, bool isQuant) {
    // hiddenStates: (b,s,h), positionIds: (b,s)
    int b = hiddenStates->shape[0];
    int s = hiddenStates->shape[1];
    int bs = b * s;
    DataType dType = hiddenStates->Datatype();

    auto qKv = QkvPre2(hiddenStates, isQuant);
    Tensor q = qKv[0];     // [b,s,n,qHeadDim]
    Tensor kvTmp = qKv[1];    // [b,s,kvLoraRank+qkRopeHeadDim]

    //dequant int32 -> fp32  -> *scale  -> fp16/bf16
    if (isQuant) {
        std::vector<int64_t> tileShape = {std::min(NUM_32, bs), NUM_64};
        Program::GetInstance().GetTileShape().SetVecTileShapes(tileShape);
        auto qTmpFp32 = Cast(q, DataType::DT_FP32);
        auto qTmpScaleDequant = qKv[2];
        auto qTmpDequantPerToken = Mul(qTmpFp32, qTmpScaleDequant);
        auto qTmpDequantChannel = Mul(qTmpDequantPerToken, qBProjWScale);
        q = Cast(qTmpDequantChannel, dType);
    }
    auto qTmp = Reshape(q, {b, s, numHeads, qHeadDim});

    /******** q ********/
    Tensor qNope = View(qTmp, {b, s, numHeads, qkNopeHeadDim}, {0, 0, 0, 0}); // [b,s,n,qkNopeHeadDim]

    std::vector<int64_t> tileShape = {NUM_2, 1, NUM_32, NUM_128};
    Program::GetInstance().GetTileShape().SetVecTileShapes(tileShape);
    Tensor qNopeR = Reshape(qNope, {bs, numHeads, qkNopeHeadDim}); // [bs,n,qkNopeHeadDim]
    tileShape = {NUM_2, NUM_32, qkNopeHeadDim};
    Program::GetInstance().GetTileShape().SetVecTileShapes(tileShape);
    Tensor qNopeT = Transpose(qNopeR, {0, 1}); // [n,bs,qkNopeHeadDim] 32_2_128

    int c0 = NUM_16;
    int m = (std::min(NUM_32, bs) + c0 - 1) / c0 * c0;
    Program::GetInstance().GetTileShape().SetCubeTileShapes({m, m}, {NUM_128, NUM_128}, {NUM_128, NUM_128});
    // bmm: (n,bs,qkNopeHeadDim) * (n, qkNopeHeadDim, kvLoraRank) = (n, bs, kvLoraRank)
    Tensor qNopeNew = Matrix::BatchMatmul(dType, qNopeT, kvBProjWK);

    tileShape = {NUM_16, NUM_2, kvLoraRank};
    Program::GetInstance().GetTileShape().SetVecTileShapes(tileShape);
    Tensor qNopeNewT = Transpose(qNopeNew, {0, 1}); // [bs,n,kvLoraRank]
    Tensor qNopeNewR = Reshape(qNopeNewT, {b, s, numHeads, kvLoraRank}); // [b,s,n,kvLoraRank]
    tileShape = {NUM_2, 1, NUM_32, kvLoraRank};
    Program::GetInstance().GetTileShape().SetVecTileShapes(tileShape);
    Tensor qNopeNewT2 = Transpose(qNopeNewR, {1, 2}); // [b,n,s,kvLoraRank]

    tileShape = {NUM_2, NUM_32, 1, NUM_64};
    Program::GetInstance().GetTileShape().SetVecTileShapes(tileShape);
    Tensor queryStates = Concat({qNopeNewT2, qPeRope}, -1); // [b,n,s, kvLoraRank + qkRopeHeadDim]

    return {queryStates, kvTmp};
}


std::vector<Tensor> DeepseekAttention::MlaPrologFoward(Tensor hiddenStates, Tensor positionIds, Tensor cos, Tensor sin,
    Tensor kvLen, Tensor pastKeyStates, const RoPETileShapeConfig &ropeTileShapeConfig, bool isQuant) {
    // hiddenStates: (b,s,h), positionIds: (b,s)
    int b = hiddenStates->shape[0];
    int s = hiddenStates->shape[1];
    int bs = b * s;
    DataType dType = hiddenStates->Datatype();

    auto qKv = QkvPre2(hiddenStates, isQuant);
    Tensor q = qKv[0];     // [b,s,n,qHeadDim]
    Tensor kvTmp = qKv[1];    // [b,s,kvLoraRank+qkRopeHeadDim]

    //dequant int32 -> fp32  -> *scale  -> fp16/bf16
    if (isQuant) {
        std::vector<int64_t> tileShape = {std::min(NUM_32, bs), NUM_64};
        Program::GetInstance().GetTileShape().SetVecTileShapes(tileShape);
        auto qTmpFp32 = Cast(q, DataType::DT_FP32);
        auto qTmpScaleDequant = qKv[2];
        auto qTmpDequantPerToken = Mul(qTmpFp32, qTmpScaleDequant);
        auto qTmpDequantChannel = Mul(qTmpDequantPerToken, qBProjWScale);

        q = Cast(qTmpDequantChannel, dType);
    }
    auto qTmp = Reshape(q, {b, s, numHeads, qHeadDim});

    /******** q ********/
    Tensor qNope = View(qTmp, {b, s, numHeads, qkNopeHeadDim}, {0, 0, 0, 0}); // [b,s,n,qkNopeHeadDim]
    std::vector<int64_t> tileShape = {NUM_32, 1, 1, NUM_128};
    Program::GetInstance().GetTileShape().SetVecTileShapes(tileShape);
    Tensor qNopeR = Reshape(qNope, {bs, numHeads, qkNopeHeadDim}); // [bs,n,qkNopeHeadDim]
    tileShape = {NUM_2, NUM_32, qkNopeHeadDim};
    Program::GetInstance().GetTileShape().SetVecTileShapes(tileShape);
    Tensor qNopeT = Transpose(qNopeR, {0, 1}); // [n,bs,qkNopeHeadDim]

    int c0 = NUM_16;
    int m = (std::min(NUM_32, bs) + c0 - 1) / c0 * c0;
    Program::GetInstance().GetTileShape().SetCubeTileShapes({m, m}, {NUM_128, NUM_128}, {NUM_128, NUM_128});
    // bmm: (n,bs,qkNopeHeadDim) * (n, qkNopeHeadDim, kvLoraRank) = (n, bs, kvLoraRank)
    Tensor qNopeNew = Matrix::BatchMatmul(dType, qNopeT, kvBProjWK);

    tileShape = {NUM_16, NUM_2, kvLoraRank};
    Program::GetInstance().GetTileShape().SetVecTileShapes(tileShape);
    Tensor qNopeNewT = Transpose(qNopeNew, {0, 1}); // [bs,n,kvLoraRank]
    Tensor qNopeNewR = Reshape(qNopeNewT, {b, s, numHeads, kvLoraRank}); // [b,s,n,kvLoraRank]
    tileShape = {NUM_2, 1, NUM_32, kvLoraRank};
    Program::GetInstance().GetTileShape().SetVecTileShapes(tileShape);
    Tensor qNopeNewT2 = Transpose(qNopeNewR, {1, 2}); // [b,n,s,kvLoraRank]

    /******** kv ********/
    Tensor compressedKv = View(kvTmp, {b, s, kvLoraRank}, {0, 0, 0}); // [b,s,kvLoraRank]
    tileShape = {NUM_2, 1, NUM_512};
    Program::GetInstance().GetTileShape().SetVecTileShapes(tileShape);
    Tensor compressedKvNorm = RmsNorm(compressedKv); // [b,s,kvLoraRank]
    Tensor kNope = Reshape(compressedKvNorm, {b, 1, s, kvLoraRank}); // [b,1,s,kvLoraRank]

    /******** RoPE ********/
    // [b,s,n,qkRopeHeadDim]
    Tensor qPe = View(qTmp, {b, s, numHeads, qkRopeHeadDim}, {0, 0, 0, qkNopeHeadDim});
    tileShape = {NUM_2, 1, NUM_32, qkNopeHeadDim};
    Program::GetInstance().GetTileShape().SetVecTileShapes(tileShape);
    Tensor qPeT = Transpose(qPe, {1, 2}); // [b,n,s,qkRopeHeadDim]

    Tensor kPe = View(kvTmp, {b, s, qkRopeHeadDim}, {0, 0, kvLoraRank}); // [b,s,qkRopeHeadDim]
    tileShape = {std::min(NUM_32, bs), 1, NUM_64};
    Program::GetInstance().GetTileShape().SetVecTileShapes(tileShape);
    Tensor kPeR = Reshape(kPe, {b, 1, s, qkRopeHeadDim}); // [b,1,s,qkRopeHeadDim]

    Tensor qPeRope(qPeT->Datatype(), {b, numHeads, s, qkRopeHeadDim}, "qPeRope"); // [b,n,s,qkRopeHeadDim]
    Tensor kPeRope(kPeR->Datatype(), {b, 1, s, qkRopeHeadDim}, "kPeRope"); // [b,1,s,qkRopeHeadDim]
    ApplyRotaryPosEmb(qPeT, kPeR, cos, sin, positionIds, qPeRope, kPeRope, 1, ropeTileShapeConfig);

    /******** output q & kv ********/
    tileShape = {NUM_2, NUM_32, 1, NUM_64};
    Program::GetInstance().GetTileShape().SetVecTileShapes(tileShape);
    Tensor queryStates = Concat({qNopeNewT2, qPeRope}, -1); // [b,n,s, kvLoraRank + qkRopeHeadDim]

    tileShape = {1, 1, 1, NUM_64};
    Program::GetInstance().GetTileShape().SetVecTileShapes(tileShape);
    Tensor keyStates = Concat({kNope, kPeRope}, -1); // [b,1,s, kvLoraRank + qkRopeHeadDim]

    tileShape = {1, 1, NUM_256, NUM_64};
    Program::GetInstance().GetTileShape().SetVecTileShapes(tileShape);
    // pastKeyStates: [b,1,s2, kvLoraRank + qkRopeHeadDim]
    pastKeyStates = ScatterUpdate(pastKeyStates, kvLen, keyStates, -2); // increase

    std::vector<Tensor> res = {queryStates, pastKeyStates, qNopeNewT2, qPeRope};
    return res;
}
} // namespace npu::tile_fwk
