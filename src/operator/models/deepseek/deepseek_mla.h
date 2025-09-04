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
 * \file deepseek_mla.h
 * \brief
 */

#pragma once
#ifndef DEEPSEEK_MLA_H
#define DEEPSEEK_MLA_H

#include "common/pre_def.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_storage.h"

namespace npu::tile_fwk {

constexpr int SCATTER_UPADATE_DIM = -2;
constexpr int NUM_1 = 1;
constexpr int NUM_2 = 2;
constexpr int NUM_3 = 3;
constexpr int NUM_4 = 4;
constexpr int NUM_8 = 8;
constexpr int NUM_16 = 16;
constexpr int NUM_20 = 20;
constexpr int NUM_24 = 24;
constexpr int NUM_32 = 32;
constexpr int NUM_48 = 48;
constexpr int NUM_64 = 64;
constexpr int NUM_128 = 128;
constexpr int NUM_256 = 256;
constexpr int NUM_384 = 384;
constexpr int NUM_512 = 512;
constexpr int NUM_1024 = 1024;
constexpr int NUM_1536 = 1536;
constexpr int NUM_1792 = 1792;
constexpr int NUM_4096 = 4096;
constexpr int NUM_6144 = 6144;
constexpr int NUM_8192 = 8192;
constexpr int NUM_7168 = 7168;
constexpr float F_1 = 1.0;
constexpr float F_0 = 0.0;
constexpr float F_NEGA_1 = -1.0;
constexpr double DF_1E_20 = 1e-20;

static std::map<std::string, std::variant<bool, int, float, std::string>> g_deepseekConfig = {
    {          "architectures", "DeepseekForCausalLM"},
    {         "attention_bias",                 false},
    {      "attentionDropout",                     0},
    {             "AutoConfig",      "DeepseekConfig"},
    {              "AutoModel",       "DeepseekModel"},
    {   "AutoModelForCausalLM", "DeepseekForCausalLM"},
    {         "auxLossAlpha",                0.001f},
    {           "bosTokenId",                100000},
    {           "eosTokenId",                100001},
    {                "epSize",                     1},
    {  "firstKDenseReplace",                     3},
    {             "hiddenAct",                "silu"},
    {            "hiddenSize",                  7168},
    {      "initializerRange",                 0.02f},
    {      "intermediateSize",                 18432},
    {           "kvLoraRank",                   512},
    {                "lmHead",                 false},
    {"maxPositionEmbeddings",                  4096},
    {             "modelType",         "deepseek_v3"},
    {  "moeIntermediateSize",                  2048},
    {         "moeLayerFreq",                     1},
    {                "nGroup",                     8},
    {       "nRoutedExperts",                   256},
    {       "nSharedExperts",                     1},
    {         "normTopkProb",                  true},
    {    "numAttentionHeads",                   128},
    {    "numExpertsPerTok",                     8},
    {      "numHiddenLayers",                    61},
    {    "numKeyValueHeads",                   128},
    {         "pretrainingTp",                     1},
    {            "qLoraRank",                  1536},
    {       "qkNopeHeadDim",                   128},
    {       "qkRopeHeadDim",                    64},
    {                "rmHead",                 false},
    {           "rmsNormEps",                1e-06f},
    {           "ropeScaling",                     1},
    {             "ropeTheta",                 10000},
    {  "routedScalingFactor",                  2.5f},
    {           "scoringFunc",             "sigmoid"},
    {                "seqAux",                  true},
    {    "tieWordEmbeddings",                 false},
    {             "topkGroup",                     4},
    {            "topkMethod",            "noaux_tc"},
    {            "torchDtype",            "bfloat16"},
    {   "transformersVersion",              "4.33.1"},
    {              "useCache",                  true},
    {             "vHeadDim",                   128},
    {             "vocabSize",                129280},
    {             "fp8Format",                "e4m3"},
    {        "initFp8Params",                  true}
};

struct AttenTilingData {
    std::vector<int> bmmVec;
    std::vector<int> commonVec;
    int kvLoraRank;
};

struct AttentionW {
    Tensor qAProjW;
    Tensor qBProjW;
    Tensor qBProjWScale;
    Tensor kvAProjWithMqaW;
    Tensor kvBProjWK;
    Tensor kvBProjWV;
    Tensor oProjW;
};

class DeepseekAttention {
public:
    DeepseekAttention(std::map<std::string, std::variant<bool, int, float, std::string>> config,
        AttentionW aw, const int inLayerIdx);
    Tensor Attention(Tensor q, Tensor kv, Tensor attenMask);
    Tensor AttentionPost(Tensor attenRes);
    Tensor AttentionPost2(Tensor attenRes);
    std::tuple<Tensor, Tensor> QkvPre(Tensor hiddenStates);
    std::tuple<Tensor, Tensor> QkvPreCv(Tensor hiddenStates);
    std::vector<Tensor> QkvPre2(Tensor hiddenStates, bool isQuant = false);
    std::tuple<Tensor, Tensor> QkvPreFp32(Tensor hiddenStates);
    Tensor Forward(Tensor hiddenStates, Tensor attenMask, Tensor positionIds, Tensor cos, Tensor sin, Tensor
        kvLen, Tensor pastKeyStates, const RoPETileShapeConfig &ropeTileShapeConfig);
    std::tuple<Tensor, Tensor>  AtentionPreForward(Tensor hiddenStates, Tensor attenMask, Tensor positionIds, Tensor cos, Tensor sin, Tensor
        kvLen, Tensor pastKeyStates, const RoPETileShapeConfig &ropeTileShapeConfig);
    std::tuple<Tensor, Tensor>  AtentionPreForwardCv(Tensor hiddenStates, Tensor attenMask, Tensor positionIds, Tensor cos, Tensor sin, Tensor
        kvLen, Tensor pastKeyStates, const RoPETileShapeConfig &ropeTileShapeConfig);
    std::tuple<Tensor, Tensor> MlaPrologAbForward(Tensor hiddenStates, Tensor qPeRope, bool isQuant = false);
    std::vector<Tensor> MlaPrologFoward(Tensor hiddenStates, Tensor positionIds,
        Tensor cos, Tensor sin, Tensor kvLen, Tensor pastKeyStates,
        const RoPETileShapeConfig &ropeTileShapeConfig, bool isQuant = false);

private:
    int layerIdx = 0;
    int attentionDropout = 0;
    int hiddenSize = 0;
    int numHeads = 0;
    int maxPositionEmbeddings = 0;
    int ropeTheta = 0;
    int qLoraRank = 0;
    int qkRopeHeadDim = 0;
    int kvLoraRank = 0;
    int vHeadDim = 0;
    int qkNopeHeadDim = 0;
    int qHeadDim = 0;
    bool isCausal = true;
    Tensor qAProjW;
    Tensor qBProjW;
    Tensor qBProjWScale;
    Tensor kvAProjWithMqaW;
    Tensor kvBProjWK;
    Tensor kvBProjWV;
    Tensor oProjW;
    float softmaxScale = 0.0f;
};

class DeepseekV2MLP {
public:
    explicit DeepseekV2MLP(std::map<std::string, std::variant<bool, int, float, std::string>> config) {
        hiddenSize = std::get<int>(config["hiddenSize"]);
        intermediateSize = std::get<int>(config["intermediateSize"]);
        gateProjW = Tensor(DataType::DT_FP32, {hiddenSize, intermediateSize});
        upProjW = Tensor(DataType::DT_FP32, {hiddenSize, intermediateSize});
        downProjW = Tensor(DataType::DT_FP32, {intermediateSize, hiddenSize});
    }

    DeepseekV2MLP(int hs, int is) : hiddenSize(hs), intermediateSize(is) {
        gateProjW = Tensor(DataType::DT_FP32, {hiddenSize, intermediateSize});
        upProjW = Tensor(DataType::DT_FP32, {hiddenSize, intermediateSize});
        downProjW = Tensor(DataType::DT_FP32, {intermediateSize, hiddenSize});
    }

    Tensor Forward(Tensor x) {
        // x 可能多维
        auto &xShape = x.GetShape();
        auto mSize =
            std::accumulate(xShape.begin(), xShape.end() - 1, 1, [](const int &a, const int &b) { return a * b; });

        if (xShape.size() > NUM_2) {
            x = Reshape(x, {mSize, xShape[xShape.size() - 1]});
        }
        const Tensor &gateProj = Matrix::Matmul<false, false>(DataType::DT_FP32, x, gateProjW);
        // Silu
        const Tensor &gateSilu = Div(gateProj, AddS(Exp(MulS(gateProj, Element(DataType::DT_FP32, F_NEGA_1))),
            Element(DataType::DT_FP32, F_1)));
        const Tensor &upProj = Matrix::Matmul<false, false>(DataType::DT_FP32, x, upProjW);
        const Tensor &mul = Mul(gateSilu, upProj);
        // (x.shape[:-1], intermediateSize) * (intermediateSize, hiddenSize) = (x.shape[:-1], hiddenSize)
        Tensor downProj = Matrix::Matmul<false, false>(DataType::DT_FP32, mul, downProjW);
        if (xShape.size() > NUM_2) {
            downProj = Reshape(downProj, xShape);
        }
        return downProj;
    }

    Tensor Forward(Tensor x, Tensor ffnWeight1, Tensor ffnWeight2, Tensor ffnWeight3) {
        // static ffn
        auto castRes = Cast(x, DataType::DT_FP16);
        auto gate = Matrix::Matmul<false, false>(DataType::DT_FP32, castRes, ffnWeight1);  // [b*s, n*d] [n*d, n*d*3] => [b*s, n*d*3]

        // swish: x / (1 + e^(-x))
        auto swish = MulS(gate, Element(DataType::DT_FP32, F_NEGA_1));
        swish = Exp(swish);
        swish = AddS(swish, Element(DataType::DT_FP32, F_1));
        swish = Div(gate, swish);

        // upProj
        auto up = Matrix::Matmul<false, false>(DataType::DT_FP32, castRes, ffnWeight2);  // [b*s, n*d] [n*d, n*d*3] => [b*s, n*d*3]
        swish = Mul(swish, up);
        auto swishFp16 = Cast(swish, DataType::DT_FP16);

        // downProj
        Tensor res = Matrix::Matmul<false, true>(DataType::DT_FP32, swishFp16, ffnWeight3);  // [b*s, n*d*3] [n*d, n*d*3]^T => [b*s, n*d]

        return res;
    }

    Tensor ForwardWithQuant(Tensor x, Tensor ffnWeight1, Tensor ffnWeight2, Tensor ffnWeight3, Tensor ffnwight1Scale, Tensor ffnwight2Scale,Tensor ffnwight3Scale) {
        // static ffn
        // quant
        TileShape::Current().SetVecTile({NUM_32, NUM_512});
        auto normQuantRes = Quant(x); // int8
        TileShape::Current().SetVecTile({NUM_256, NUM_256});
        Tensor castRes = std::get<0>(normQuantRes);
        Tensor castResScale = std::get<1>(normQuantRes);
        TileShape::Current().SetCubeTile({NUM_64, NUM_64}, {NUM_128, NUM_128}, {NUM_128, NUM_128});

        auto gateInt32 = Matrix::Matmul<false, false>(DataType::DT_INT32, castRes, ffnWeight1);

        // dequant: int32 -> fp32 -> *scale -> fp16/bf16
        auto gateTmpFp32 = Cast(gateInt32, DataType::DT_FP32);
        auto gateTmpDequantPerToken = Mul(gateTmpFp32, castResScale);
        auto gate = Mul(gateTmpDequantPerToken, ffnwight1Scale);

        // swish: x / (1 + e^(-x))
        auto swish = MulS(gate, Element(DataType::DT_FP32, F_NEGA_1));
        swish = Exp(swish);
        swish = AddS(swish, Element(DataType::DT_FP32, F_1));
        swish = Div(gate, swish);

        auto upInt32 = Matrix::Matmul<false, false>(DataType::DT_INT32, castRes, ffnWeight2);
        // upProj
        auto upTmpFp32 = Cast(upInt32, DataType::DT_FP32);
        auto upTmpDequantPerToken = Mul(upTmpFp32, castResScale);
        auto up = Mul(upTmpDequantPerToken, ffnwight2Scale);

        swish = Mul(swish, up);

        // downProj
        TileShape::Current().SetVecTile({NUM_32, NUM_512});
        auto swishQuantRes = Quant(swish); // int8
        TileShape::Current().SetVecTile({NUM_256, NUM_256});
        Tensor swishRes = std::get<0>(swishQuantRes);
        Tensor swishScale = std::get<1>(swishQuantRes);

        Tensor resInt32 = Matrix::Matmul<false, true>(DataType::DT_INT32, swishRes, ffnWeight3);
        auto resTmpFp32 = Cast(resInt32, DataType::DT_FP32);
        auto resTmpDequantPerToken = Mul(resTmpFp32, swishScale);
        Tensor ffnwight3ScaleTrans = Transpose(ffnwight3Scale, {0, 1});
        auto res = Mul(resTmpDequantPerToken, ffnwight3ScaleTrans);

        return res;
    }
private:
    int hiddenSize = 0;
    int intermediateSize = 0;
    Tensor gateProjW;
    Tensor upProjW;
    Tensor downProjW;
};

class MoEGate {
public:
    explicit MoEGate(std::map<std::string, std::variant<bool, int, float, std::string>> config) {
        nRoutedExperts = std::get<int>(config["nRoutedExperts"]);
        nGroup = std::get<int>(config["nGroup"]);
        topkGroup = std::get<int>(config["topkGroup"]);
        numExpertsPerTok = std::get<int>(config["numExpertsPerTok"]);

        int hiddenSize = std::get<int>(config["hiddenSize"]);
        std::vector<int64_t> biasShape = {1, nRoutedExperts};
        weight = Tensor(DataType::DT_FP32, {nRoutedExperts, hiddenSize});
        eScoreCorrectionBias = Tensor(DataType::DT_FP32, biasShape, "eScoreCorrectionBias");
    }

    std::tuple<Tensor, Tensor> Forward(const Tensor &hiddenStates) {
        // hiddenStates: [b*s,h]
        int bs = hiddenStates->shape[0];

        /* compute gating score */
        auto logits = Matrix::Matmul<false, true>(DataType::DT_FP32, hiddenStates, weight); // [b*s,h] @ [nRoutedExperts,h].t -> [b*s,256]
        auto scores = Sigmoid(logits);                          // [b*s,256]

        /* select top-k experts */
        auto scoresForChoice = Add(scores, eScoreCorrectionBias);
        // [b*s,256]+[1,256]->[b*s,256]
        // groupScores = (View(scoresForChoice, bsz * seq_len, self.nGroup, -1).topk(2, dim=-1)[0].sum())
        // groupIdx = torch.topk(groupScores, k=self.topkGroup, dim=-1, sorted=False)[1]
        std::vector<int64_t> shape = {
            scoresForChoice->shape[0] * nGroup, scoresForChoice->shape[1] / nGroup
        }; // [b*s,256]->[b*s*8,32]
        auto scoresForChoiceNewShape = Reshape(scoresForChoice, shape);
        auto scoresForChoiceIndex = std::get<0>(TopK(scoresForChoiceNewShape, 2, -1));
        // [b*s*8,32]->[b*s*8,2]

        // TileShape::Current().SetVecTile(128, 1); // for Assemble
        auto groupScores = RowSumSingle(scoresForChoiceIndex, 1); // [b*s*8,2]->[b*s*8]
        // TileShape::Current().SetVecTile(128, 64); // for Assemble

        auto groupScoresReshape = Reshape(groupScores, {groupScores->shape[0] / nGroup, nGroup});
        // [b*s*8]->[b*s,8]
        auto groupIdx = std::get<1>(TopK(groupScoresReshape, topkGroup, 1)); // [b*s,8]->[b*s,4]

        // groupMask = torch.zeros_like(groupScores)
        auto groupMask = MulS(groupScoresReshape, Element(DataType::DT_FP32, F_0)); // [b*s,8]
        // groupMask.scatter_(1, groupIdx, 1)
        auto groupMaskScatter = ScatterElement(groupMask, groupIdx, Element(DataType::DT_FP32, F_1), 1); // [b*s,8]
        // scoreMask
        int dim0 = groupMaskScatter->shape[0] * groupMaskScatter->shape[1];

        auto scoreMask = Expand(Reshape(groupMaskScatter, {dim0, 1}), {dim0, nRoutedExperts / nGroup}); // [b*s*8,1] -> [b*s*8,32]
        scoreMask = Reshape(scoreMask, {bs, nRoutedExperts}); // [b*s*8,32]->[b*s,256]
        auto scoreMaskNot = MulS(scoreMask, Element(DataType::DT_FP32, F_NEGA_1));

        // // tmpScores = scoresForChoice.masked_fill(~scoreMask.bool(), 0.0)
        // auto score_mask_bool = LogicalNot(Cast(scoreMask, DT_BOOL));             // [b*s,256]

        auto tmpScores = Mul(scoresForChoice, scoreMaskNot);
        // _, topkIdx = torch.topk(tmpScores, k=self.top_k, dim=-1, sorted=False)
        auto topkIdx = std::get<1>(TopK(tmpScores, numExpertsPerTok, -1)); // [b*s,256]->[b*s,8]
        // topkWeight = scores.gather(1, topkIdx)
        auto topkWeight = GatherElement(scores, topkIdx, 1); // [b*s,8]

        /* norm gate to sum 1 */
        // denominator = topkWeight.sum(dim=-1, keepdim=True) + 1e-20
        auto topkWeightSum = RowSumSingle(topkWeight, 1);      // [b*s,8]->[b*s,1]
        auto denominator = AddS(topkWeightSum, Element(DataType::DT_FP32, DF_1E_20)); // [b*s,1]
        // topkWeight = topkWeight / denominator
        topkWeight = Div(topkWeight, denominator); // [b*s,numExpertsPerTok]

        /* expert-level computation auxiliary loss */
        // aux_loss = None

        return std::make_tuple(topkIdx, topkWeight);
    }

private:
    int nRoutedExperts = 0;
    int nGroup = 0;
    int topkGroup = 0;
    int numExpertsPerTok = 0;

    Tensor weight;                  // [nRoutedExperts, hiddenSize]
    Tensor eScoreCorrectionBias; // [nRoutedExperts]
};

class DeepseekV2MoE {
public:
    explicit DeepseekV2MoE(std::map<std::string, std::variant<bool, int, float, std::string>> config)
        : expert(std::get<int>(config["hiddenSize"]), std::get<int>(config["moeIntermediateSize"])),
          moeGate(config),
          sharedExpert(std::get<int>(config["hiddenSize"]),
              std::get<int>(config["moeIntermediateSize"]) * std::get<int>(config["nSharedExperts"])) {
        numExpertsPerTok = std::get<int>(config["numExpertsPerTok"]);
        epSize = 1;
        expertsPerRank = std::get<int>(config["nRoutedExperts"]);
        epRank = 0;
    }

    Tensor MoeInfer(Tensor x, Tensor topkIds, Tensor topkWeight, Tensor ffnWeight1, Tensor ffnWeight2,
                    Tensor ffnWeight3, int nRoutedExperts) {
        // x: (b*s, h), topkIds, topkWeight: (b*s, num_experts_per_tok)
        int bs = topkIds.GetShape(0);
        int expertPerTok = topkIds.GetShape(1);
        std::vector<int64_t> zerosShape(NUM_2);
        zerosShape[0] = bs;
        zerosShape[1] = nRoutedExperts;
        Tensor randoms(topkIds->Datatype(), zerosShape);

        Tensor cnts = MulS(randoms, Element(DataType::DT_FP32, F_0)); // (b*s, nRoutedExperts)

        cnts = ScatterElement(cnts, topkIds, Element(DataType::DT_FP32, F_1), 1); // (b*s, nRoutedExperts)

        Tensor tokensPerExpert = RowSumSingle(cnts, 0);

        TileShape::Current().SetVecTile(NUM_128);
        // reduce 0维, (b*s, nRoutedExperts)->(nRoutedExperts)
        Tensor idxs = ArgSort(Reshape(topkIds, { bs * expertPerTok }), -1, false); // (b*s*num_experts_per_tok)

        TileShape::Current().SetVecTile({NUM_128, NUM_128});

        Tensor sortedTokens = TensorIndex(x,  Cast(DivS(Cast(idxs, DataType::DT_FP32),
            Element(DataType::DT_FP32, static_cast<double>(expertPerTok))),
            DataType::DT_INT32, CAST_TRUNC));

        auto &sortedTokensShape = sortedTokens->GetShape();

        // tokensPerExpertCpu = tokensPerExpert.cpu().numpy(); 手动设置规避动态图
        // 这里构造总大小为b*s*num_experts_per_tok的vector,模拟选择专家，执行256次Mlp计算
        std::vector<int> tokensPerExpertCpu(NUM_256, 0);
        for (size_t i = 0; i < NUM_8; i++) {
            tokensPerExpertCpu[i] = sortedTokensShape[0] / NUM_8;
        }

        std::vector<Tensor> outputs;
        int startIdx = 0;

        for (size_t i = 0; i < tokensPerExpertCpu.size(); i++) {
            int numTokens = tokensPerExpertCpu[i];
            if (numTokens == 0) {
                continue;
            }
            const int endIdx = startIdx + numTokens;
            // sortedTokens只有两维
            Tensor tokensForThisExpert = View(sortedTokens,  { numTokens, sortedTokensShape[1] }, { startIdx, 0 }); // 选出[B, H]
            std::cout<<"=numTokens===="<<numTokens<<std::endl;
            for (auto n : tokensForThisExpert.GetShape()){
                std::cout<<"=tokensForThisExpert->shape"<< n <<std::endl;
            }

            // 这里没有选对应的expert，默认infer模式下所有的expert相同
            // (numTokens[i], h),每次沿着b*s*num_experts_per_tok的方向不间隔选取num_tokens的大小;最终需要累计选b*s*num_experts_per_tok否则后续计算无法计算
            auto expertOut = expert.Forward(tokensForThisExpert, ffnWeight1, ffnWeight2, ffnWeight3);
            outputs.emplace_back(expertOut);
            startIdx = endIdx;
        }

        Tensor outs = Concat(outputs, 0); // (all_sum_num_tokens, h) = (b*s*num_experts_per_tok, h)
        Tensor newX(outs.GetDataType(), outs.GetShape()); // (b*s*num_experts_per_tok, h)

        for (auto n: outs.GetShape()){
            std::cout << "=outs->shape" << n << std::endl;
        }
        TileShape::Current().SetVecTile({NUM_128, NUM_128});
        // newX[idxs] = outs  -->index_put: (b*s*num_experts_per_tok, h)[b*s*num_experts_per_tok] =
        // (b*s*num_experts_per_tok, h)
        auto newIdxs = Reshape(idxs, {1, idxs.GetShape(0)});
        newX = IndexPut(newX, {newIdxs}, outs);

        int newXSize = std::accumulate(
            newX->shape.begin(), newX->shape.end(), 1, [](const int &a, const int &b) { return a * b; });
        std::cout << "===newXSize" << newXSize << std::endl;

        std::vector<int64_t> newShape = {bs, expertPerTok, newXSize / (bs * expertPerTok)};
        // (b*s, expertPerTok, h)
        auto newXShape = Reshape(newX, newShape);  // [128,256] -> [16,8,256]
        TileShape::Current().SetVecTile(NUM_16, NUM_128, NUM_128);

        auto wShapes = topkWeight->shape;
        wShapes.emplace_back(1);
        auto newW = Unsqueeze(topkWeight, NUM_2); // (b*s, expertPerTok, 1)
        auto newMul = Mul(newXShape, newW);
        // (b*s, expertPerTok, h) * (b*s, expertPerTok, 1) = (b*s, expertPerTok, h)
        auto reduceRes = RowSumSingle(newMul, 1); // reudce轴1 ->(b*s, 1, h)
        for (auto n: reduceRes->GetShape()){
            std::cout << "=reduceRes->shape.shape" << n <<std::endl;
        }

        auto fOut = Reshape(reduceRes, {bs, newXSize / (bs * expertPerTok)});

        return fOut;
    }

    Tensor MoeInferSingleMlp(Tensor x, Tensor topkIds, Tensor topkWeight, Tensor ffnWeight1, Tensor ffnWeight2,
                    Tensor ffnWeight3, int nRoutedExperts) {
        // x: (b*s, h), topkIds, topkWeight: (b*s, num_experts_per_tok)
        (void)topkWeight;
        int bs = topkIds.GetShape(0);
        int expertPerTok = topkIds.GetShape(1);
        std::vector<int64_t> zerosShape(NUM_2);
        zerosShape[0] = bs;
        zerosShape[1] = nRoutedExperts;
        Tensor randoms(topkIds->Datatype(), zerosShape);

        Tensor cnts = MulS(randoms, Element(DataType::DT_FP32, F_0)); // (b*s, nRoutedExperts)

        cnts = ScatterElement(cnts, topkIds, Element(DataType::DT_FP32, F_1), 1); // (b*s, nRoutedExperts)

        Tensor tokensPerExpert = RowSumSingle(cnts, 0);

        TileShape::Current().SetVecTile(NUM_128);
        // reduce 0维, (b*s, nRoutedExperts)->(nRoutedExperts)
        Tensor idxs = ArgSort(Reshape(topkIds, { bs * expertPerTok }), -1, false); // (b*s*num_experts_per_tok)

        TileShape::Current().SetVecTile({NUM_128, NUM_128});

        // Tensor((b*s, h))[Tensor(b*s*num_experts_per_tok)] = (b*s*num_experts_per_tok, h)
        // 没有int类型除法 只能先cast成float做完除法再cast回int
        Tensor sortedTokens = TensorIndex(x,  Cast(DivS(Cast(idxs, DataType::DT_FP32),
            Element(DataType::DT_FP32, static_cast<double>(expertPerTok))),
            DataType::DT_INT32, CAST_TRUNC));

        auto &sortedTokensShape = sortedTokens->GetShape();

        // tokensPerExpertCpu = tokensPerExpert.cpu().numpy(); 手动设置规避动态图
        // 这里构造总大小为b*s*num_experts_per_tok的vector,模拟选择专家，执行256次Mlp计算
        std::vector<int> tokensPerExpertCpu(NUM_256, 0);
        for (size_t i = 0; i < 1; i++) {
            tokensPerExpertCpu[i] = sortedTokensShape[0] / NUM_8;
        }

        std::vector<Tensor> outputs;
        int startIdx = 0;

        for (size_t i = 0; i < tokensPerExpertCpu.size(); i++) {
            int numTokens = tokensPerExpertCpu[i];
            if (numTokens == 0) {
                continue;
            }
            const int endIdx = startIdx + numTokens;
            // sortedTokens只有两维
            Tensor tokensForThisExpert = View(sortedTokens,  { numTokens, sortedTokensShape[1] }, { startIdx, 0 }); // 选出[B, H]
            std::cout<<"=numTokens===="<<numTokens<<std::endl;
            for (auto n : tokensForThisExpert.GetShape()){
                std::cout<<"=tokensForThisExpert->shape.shape"<< n <<std::endl;
            }

            // 这里没有选对应的expert，默认infer模式下所有的expert相同
            // (numTokens[i], h),每次沿着b*s*num_experts_per_tok的方向不间隔选取num_tokens的大小;最终需要累计选b*s*num_experts_per_tok否则后续计算无法计算
            auto expertOut = expert.Forward(tokensForThisExpert, ffnWeight1, ffnWeight2, ffnWeight3);
            outputs.emplace_back(expertOut);
            startIdx = endIdx;
        }

        Tensor outs = Concat(outputs, 0); // (all_sum_num_tokens, h) = (b*s*num_experts_per_tok, h)
        return outs;
    }
    Tensor MoeInferSingleMlpQuant(Tensor x, Tensor topkIds, Tensor topkWeight, Tensor ffnWeight1, Tensor ffnWeight2,
        Tensor ffnWeight3, Tensor ffnwight1Scale, Tensor ffnwight2Scale,Tensor ffnwight3Scale, int nRoutedExperts) {
        (void)topkWeight;
        int bs = topkIds.GetShape(0);
        int expertPerTok = topkIds.GetShape(1);
        std::vector<int64_t> zerosShape(NUM_2);
        zerosShape[0] = bs;
        zerosShape[1] = nRoutedExperts;
        Tensor randoms(topkIds->Datatype(), zerosShape);

        Tensor cnts = MulS(randoms, Element(DataType::DT_FP32, F_0)); // (b*s, nRoutedExperts)

        cnts = ScatterElement(cnts, topkIds, Element(DataType::DT_FP32, F_1), 1); // (b*s, nRoutedExperts)

        Tensor tokensPerExpert = RowSumSingle(cnts, 0);

        TileShape::Current().SetVecTile(NUM_128);
        // reduce 0维, (b*s, nRoutedExperts)->(nRoutedExperts)
        Tensor idxs = ArgSort(Reshape(topkIds, { bs * expertPerTok }), -1, false); // (b*s*num_experts_per_tok)

        TileShape::Current().SetVecTile({NUM_32, NUM_512});

        // Tensor((b*s, h))[Tensor(b*s*num_experts_per_tok)] = (b*s*num_experts_per_tok, h)
        // 没有int类型除法 只能先cast成float做完除法再cast回int
        Tensor sortedTokens = TensorIndex(x,  Cast(DivS(Cast(idxs, DataType::DT_FP32),
            Element(DataType::DT_FP32, static_cast<double>(expertPerTok))),
            DataType::DT_INT32, CAST_TRUNC));

        TileShape::Current().SetVecTile({NUM_256, NUM_256});
        auto &sortedTokensShape = sortedTokens->GetShape();

        // tokensPerExpertCpu = tokensPerExpert.cpu().numpy(); 手动设置规避动态图
        // 这里构造总大小为b*s*num_experts_per_tok的vector,模拟选择专家，执行256次Mlp计算
        std::vector<int> tokensPerExpertCpu(NUM_256, 0);
        for (size_t i = 0; i < 1; i++) {
            tokensPerExpertCpu[i] = sortedTokensShape[0] / NUM_8;
        }

        std::vector<Tensor> outputs;
        int startIdx = 0;

        for (size_t i = 0; i < tokensPerExpertCpu.size(); i++) {
            int numTokens = tokensPerExpertCpu[i];
            if (numTokens == 0) {
                continue;
            }
            const int endIdx = startIdx + numTokens;
            // sortedTokens只有两维
            Tensor tokensForThisExpert = View(sortedTokens,  { numTokens, sortedTokensShape[1] }, { startIdx, 0 }); // 选出[B, H]
            std::cout<<"=numTokens===="<<numTokens<<std::endl;
            for (auto n : tokensForThisExpert.GetShape()){
                std::cout<<"=tokensForThisExpert->shape.shape"<< n <<std::endl;
            }

            // 这里没有选对应的expert，默认infer模式下所有的expert相同
            // (numTokens[i], h),每次沿着b*s*num_experts_per_tok的方向不间隔选取num_tokens的大小;最终需要累计选b*s*num_experts_per_tok否则后续计算无法计算
            auto expertOut = expert.ForwardWithQuant(tokensForThisExpert, ffnWeight1, ffnWeight2, ffnWeight3, ffnwight1Scale, ffnwight2Scale, ffnwight3Scale);
            outputs.emplace_back(expertOut);
            startIdx = endIdx;
        }

        Tensor outs = Concat(outputs, 0); // (all_sum_num_tokens, h) = (b*s*num_experts_per_tok, h)
        return outs;
    }

    Tensor MoeInfer(Tensor x, Tensor topkIds, Tensor topkWeight, Tensor ffnWeight1, Tensor ffnWeight2, Tensor ffnWeight3,
                        Tensor &idxs, Tensor &sortedTokens, Tensor &outs, int nRoutedExperts) {
        // x: (b*s, h), topkIds, topkWeight: (b*s, numExpertsPerTok)
        int bs = topkIds.GetShape(0);
        int expertPerTok = topkIds.GetShape(1);
        std::vector<int64_t> zerosShape(NUM_2);
        zerosShape[0] = bs;
        zerosShape[1] = nRoutedExperts;
        Tensor randoms(topkIds->Datatype(), zerosShape);

        Tensor cnts = MulS(randoms, Element(DataType::DT_FP32, F_0)); // (b*s, nRoutedExperts)

        cnts = ScatterElement(cnts, topkIds, Element(DataType::DT_FP32, F_1), 1); // (b*s, nRoutedExperts)

        Tensor tokensPerExpert = RowSumSingle(cnts, 0);

        TileShape::Current().SetVecTile(NUM_128);
        // reduce 0维, (b*s, nRoutedExperts)->(nRoutedExperts)
        idxs = ArgSort(Reshape(topkIds, { bs * expertPerTok }), -1, false); // (b*s*numExpertsPerTok)

        TileShape::Current().SetVecTile({NUM_64, NUM_64});

        sortedTokens = TensorIndex(x,  Cast(DivS(Cast(idxs, DataType::DT_FP32),
            Element(DataType::DT_FP32, static_cast<double>(expertPerTok))),
            DataType::DT_INT32, CAST_TRUNC));

        auto &sortedTokensShape = sortedTokens->GetShape();

        // tokensPerExpertCpu = tokensPerExpert.cpu().numpy(); 手动设置规避动态图
        // 这里构造总大小为b*s*num_experts_per_tok的vector,模拟选择专家，执行256次Mlp计算
        std::vector<int> tokensPerExpertCpu(NUM_256, 0);
        for (size_t i = 0; i < NUM_8; i++) {
            tokensPerExpertCpu[i] = sortedTokensShape[0] / NUM_8;
        }

        std::vector<Tensor> outputs;
        int startIdx = 0;

        for (size_t i = 0; i < tokensPerExpertCpu.size(); i++) {
            int numTokens = tokensPerExpertCpu[i];
            if (numTokens == 0) {
                continue;
            }
            const int endIdx = startIdx + numTokens;
            // sorted_tokens只有两维
            Tensor tokensForThisExpert = View(sortedTokens,  { numTokens, sortedTokensShape[1] }, { startIdx, 0 }); // 选出[B, H]
            std::cout<<"=numTokens===="<<numTokens<<std::endl;
            for (auto n : tokensForThisExpert.GetShape()) {
                std::cout<<"=tokensForThisExpert->shape.shape"<< n <<std::endl;
            }

            // 这里没有选对应的expert，默认infer模式下所有的expert相同
            // (numTokens[i], h),每次沿着b*s*num_experts_per_tok的方向不间隔选取num_tokens的大小;最终需要累计选b*s*num_experts_per_tok否则后续计算无法计算
            auto expertOut = expert.Forward(tokensForThisExpert, ffnWeight1, ffnWeight2, ffnWeight3);
            outputs.emplace_back(expertOut);
            startIdx = endIdx;
        }

        outs = Concat(outputs, 0); // (all_sum_num_tokens, h) = (b*s*numExpertsPerTok, h)
        Tensor newX(outs.GetDataType(), outs.GetShape()); // (b*s*numExpertsPerTok, h)

        for (auto n : outs.GetShape()){
            std::cout<<"=outs->shape.shape"<< n <<std::endl;
        }
        TileShape::Current().SetVecTile({NUM_128, NUM_128});
        // newX[idxs] = outs  -->index_put: (b*s*numExpertsPerTok, h)[b*s*numExpertsPerTok] =
        // (b*s*numExpertsPerTok, h)
        auto newIdxs = Reshape(idxs, {1, idxs.GetShape(0)});
        newX = IndexPut(newX, {newIdxs}, outs);

        int newXSize = std::accumulate(
            newX->shape.begin(), newX->shape.end(), 1, [](const int &a, const int &b) { return a * b; });
        std::cout<<"===newXSize"<<newXSize<<std::endl;

        std::vector<int64_t> newShape = {bs, expertPerTok, newXSize / (bs * expertPerTok)};
        // (b*s, expertPerTok, h)
        auto newXShape = Reshape(newX, newShape);  // [128,256] -> [16,8,256]
        TileShape::Current().SetVecTile(NUM_16, NUM_64, NUM_64);

        auto wShape = topkWeight->shape;
        wShape.emplace_back(1);
        auto newW = Unsqueeze(topkWeight, NUM_2); // (b*s, expertPerTok, 1)
        auto newMul = Mul(newXShape, newW);
        // (b*s, expertPerTok, h) * (b*s, expertPerTok, 1) = (b*s, expertPerTok, h)
        auto reduceRes = RowSumSingle(newMul, 1); // reudce轴1 ->(b*s, 1, h)
        for (auto n : reduceRes.GetShape()){
            std::cout<<"=reduceRes->shape.shape"<< n <<std::endl;
        }

        auto fOut = Reshape(reduceRes, {bs, newXSize / (bs * expertPerTok)});

        return fOut;
    }

    Tensor MoeInfer(Tensor x, Tensor topkIds, Tensor topkWeight, int nRoutedExperts = 256) {
        // x: (b*s, h), topkIds, topkWeight: (b*s, numExpertsPerTok)
        int bs = topkIds.GetShape(0);
        int expertPerTok = topkIds.GetShape(1);
        const int twoDim = 2;
        std::vector<int64_t> zerosShape(twoDim);
        zerosShape[0] = bs;
        zerosShape[1] = nRoutedExperts;
        Tensor randoms(topkIds->Datatype(), zerosShape);

        Tensor cnts = MulS(randoms, Element(DataType::DT_FP32, F_0)); // (b*s, nRoutedExperts)

        cnts = ScatterElement(cnts, topkIds, Element(DataType::DT_FP32, F_1), 1); // (b*s, nRoutedExperts)

        Tensor tokensPerExpert = RowSumSingle(cnts, 0);
        // reduce 0维, (b*s, nRoutedExperts)->(nRoutedExperts)
        Tensor idxs = ArgSort(Reshape(topkIds, {bs * expertPerTok}), -1); // (b*s*numExpertsPerTok)

        // Tensor((b*s, h))[Tensor(b*s*numExpertsPerTok)] = (b*s*numExpertsPerTok, h)
        Tensor sortedTokens = TensorIndex(x, DivS(idxs, Element(DataType::DT_FP32,
            static_cast<double>(expertPerTok)))); // int64除法
        auto &sortedTokensShape = sortedTokens.GetShape();

        // tokensPerExpertCpu = tokensPerExpert.cpu().numpy(); 手动设置规避动态图
        // 这里构造总大小为b*s*num_experts_per_tok的vector,模拟选择专家，执行256次Mlp计算
        std::vector<int> tokensPerExpertCpu(NUM_256, 0);
        tokensPerExpertCpu[0] = sortedTokensShape[0] / NUM_8;
        // tokensPerExpertCpu[1] = sortedTokensShape[0] / 8;
        // tokensPerExpertCpu[2] = sortedTokensShape[0] / 8;
        // tokensPerExpertCpu[3] = sortedTokensShape[0] / 8;
        // tokensPerExpertCpu[4] = sortedTokensShape[0] / 8;
        // tokensPerExpertCpu[5] = sortedTokensShape[0] / 8;
        // tokensPerExpertCpu[6] = sortedTokensShape[0] / 8;
        // tokensPerExpertCpu[7] = sortedTokensShape[0] / 8;

        std::vector<Tensor> outputs;
        int startIdx = 0;

        for (size_t i = 0; i < tokensPerExpertCpu.size(); i++) {
            int numTokens = tokensPerExpertCpu[i];
            if (numTokens == 0) {
                continue;
            }
            const int endIdx = startIdx + numTokens;
            // sorted_tokens只有两维
            Tensor tokensForThisExpert = View(sortedTokens, {numTokens, sortedTokensShape[1]}, {startIdx, 0});
            // 这里没有选对应的expert，默认infer模式下所有的expert相同
            // (numTokens[i],
            // h),每次沿着b*s*num_experts_per_tok的方向不间隔选取num_tokens的大小;最终需要累计选b*s*num_experts_per_tok否则后续计算无法计算
            auto expertOut = expert.Forward(tokensForThisExpert);
            outputs.emplace_back(expertOut);
            startIdx = endIdx;
        }

        // newX = torch.empty_like(outs)
        auto outs = Concat(outputs, 0); // (all_sum_num_tokens, h) = (b*s*numExpertsPerTok, h)
        Tensor newX(outs.GetDataType(), outs.GetShape()); // (b*s*numExpertsPerTok, h)

        // newX[idxs] = outs  -->index_put: (b*s*numExpertsPerTok, h)[b*s*numExpertsPerTok] =
        // (b*s*numExpertsPerTok, h)
        newX = IndexPut(newX, {idxs}, outs);

        int newXSize = std::accumulate(
            newX->shape.begin(), newX->shape.end(), 1, [](const int &a, const int &b) { return a * b; });
        std::vector<int64_t> newShape = {bs, expertPerTok, newXSize / (bs * expertPerTok)};
        // (b*s, expertPerTok, h)
        auto newXShape = Reshape(newX, newShape);
        TileShape::Current().SetVecTile(NUM_128, NUM_64, NUM_64); // for Assemble
        auto newl = Cast(newXShape, topkWeight.GetDataType());
        auto wShape = topkWeight->shape;
        wShape.emplace_back(1);
        auto newW = Unsqueeze(topkWeight, 2); // (b*s, expertPerTok, 1)
        auto newMul = Mul(newl, newW);
        // (b*s, expertPerTok, h) * (b*s, expertPerTok, 1) = (b*s, expertPerTok, h)
        auto fOut = Cast(RowSumSingle(newMul, 1), newX.GetDataType()); // reudce轴1 ->(b*s, h)
        TileShape::Current().SetVecTile(NUM_128, NUM_64);              // for Assemble

        return fOut;
    }

    Tensor Forward(Tensor hiddenStates) {
        const Tensor identity = hiddenStates;
        const std::vector<int64_t> &origShape = hiddenStates.GetShape();

        // hiddenStates = Reshape(hiddenStates, {b * s, h}); // (b, s, h)->(b * s, h)

        auto moeGateRes = moeGate.Forward(hiddenStates);   // hiddenStates: (b*s, h)
        const Tensor &topkIdx = std::get<0>(moeGateRes);    // (b*s, numExpertsPerTok)
        const Tensor &topkWeight = std::get<1>(moeGateRes); // (b*s, numExpertsPerTok)

        // MoeInfer
        Tensor inferRes = MoeInfer(hiddenStates, topkIdx, topkWeight);
        inferRes = Reshape(inferRes, origShape); // (b*s, h)
        const Tensor &sharedMlp = sharedExpert.Forward(identity); // (b, s, h)

        return Add(inferRes, sharedMlp);
    }

private:
    int numExpertsPerTok = 0;
    int epSize = 0;
    int expertsPerRank = 0;
    int epRank = 0;;
    DeepseekV2MLP expert;
    MoEGate moeGate;
    DeepseekV2MLP sharedExpert;
};

} // namespace npu::tile_fwk

#endif // DEEPSEEK_MLA_H
