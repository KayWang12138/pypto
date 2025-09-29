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
 * \file llama_graph.cpp
 * \brief
 */

#include <iostream>

#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_storage.h"

using namespace npu::tile_fwk;

constexpr int T_SHAPE = 128;
constexpr int NUM_64 = 64;
constexpr int NUM_128 = 128;
constexpr float F_1 = 1.0;
constexpr float F_NEGA_1 = -1.0;

namespace npu::tile_fwk {
struct AttentionDims {
    int b;
    int n;
    int s;
    int d;
    int singleM;
    int singleN;
};

struct AttentionVecTileConfig {
    int defaultVecTileX;
    int defaultVecTileY;
    int softmaxTileX;
    int softmaxTileY;
    int updateTileX;
    int updateTileY;
    int castTileX;
    int castTileY;
};

struct AttentionCubeTileConfig {
    int c1L1M;
    int c1L1K;
    int c1L1N;
    int c2L1M;
    int c2L1K;
    int c2L1N;
    int c1L0 = 128;
    int c2L0 = 128;
};

struct KeyConfig {
    int max;
    int min;
    int dbType;
    int nBuffer;
    int isPartitionCv;
    int cTileX;
    int cTileY;
};

constexpr AttentionVecTileConfig DFS_VEC_CFG = {128, 128, 16, 128, 16, 128, 32, 128};
constexpr AttentionVecTileConfig SMALL_DFS_VEC_CFG = {64, 128, 16, 128, 16, 128, 32, 128};
constexpr AttentionCubeTileConfig DFS_CUBE_CFG = {128, 128, 128, 128, 128, 128};

constexpr AttentionVecTileConfig OOO_VEC_CFG = {64, 128, 16, 512, 32, 128, 32, 128};
constexpr AttentionCubeTileConfig OOO_CUBE_CFG = {128, 128, 512, 64, 256, 64, 128, 64};

constexpr KeyConfig DFT_BASIC_CFG = {8192, 1024, 1, 1, 0, 128, 128};
constexpr int DFT_SINGLE_M = 128;
constexpr int DFT_SINGLE_N = 128;

static inline void SetC1CubeConfig(const AttentionCubeTileConfig &cubeCfg) {
    TileShape::Current().SetCubeTile(
        {cubeCfg.c1L0, cubeCfg.c1L1M}, {cubeCfg.c1L0, cubeCfg.c1L1K}, {cubeCfg.c1L0, cubeCfg.c1L1N});
}

static inline void SetC2CubeConfig(const AttentionCubeTileConfig &cubeCfg) {
    TileShape::Current().SetCubeTile(
        {cubeCfg.c2L0, cubeCfg.c2L1M}, {cubeCfg.c2L0, cubeCfg.c2L1K}, {cubeCfg.c2L0, cubeCfg.c2L1N});
}

void SetDefaultL0CubeConfig() {
    TileShape::Current().SetCubeTile(
        {T_SHAPE, T_SHAPE}, {T_SHAPE, T_SHAPE}, {T_SHAPE, T_SHAPE});
}


Tensor FlashAttention(const Tensor &q, const Tensor &k, const Tensor &v, const Tensor &m, const Tensor &l,
    const AttentionDims &atDims, const AttentionVecTileConfig &vecCfg, const AttentionCubeTileConfig &cubeCfg) {
    (void)m;
    (void)l;
    // q, k, v, result shape: [b*s, n*d]
    int dim0 = q->shape[0];
    int dim1 = q->shape[1];
    int b = atDims.b;
    int n = atDims.n;
    int s = dim0 / b;
    int d = dim1 / n;
    int singleM = atDims.singleM; // 128
    assert(singleM == 128);
    int singleN = atDims.singleN; // 1024
    int s1Loop = s / singleM;
    int s2Loop = s / singleN;

    std::cout << "FlashAttention, B, N, S, D --------" << b << "," << n << "," << s << "," << d << std::endl;
    std::cout << "s1Loop, s2Loop -------" << s1Loop << "," << s2Loop << "," << std::endl;

    auto bns = atDims.b * atDims.n * atDims.s;
    std::vector<int64_t> shapeReduce = {bns, 1};
    std::vector<float> max(bns, 0);
    std::vector<float> sum(bns, 0);

    std::map<std::vector<int64_t>, Tensor> lastOi;
    std::map<std::vector<int64_t>, Tensor> lastMi;
    std::map<std::vector<int64_t>, Tensor> lastLi;
    Tensor result;

    // LLAMA_FUNCTION(FlashAttention_L4) {
    for (int bIdx = 0; bIdx < b; bIdx++) {
        // LLAMA_FUNCTION(FlashAttention_L3) {
        for (int nIdx = 0; nIdx < n; nIdx++) {
            // LLAMA_FUNCTION(FlashAttention_L2) {
            for (int s2Idx = 0; s2Idx < s2Loop; s2Idx++) {
                // LLAMA_FUNCTION(FlashAttention_L1) {
                auto kj = View(k, {singleN, d}, {bIdx * s + s2Idx * singleN, nIdx * d});
                auto vj = View(v, {singleN, d}, {bIdx * s + s2Idx * singleN, nIdx * d});

                for (int s1Idx = 0; s1Idx < s1Loop; s1Idx++) {
                    ASLOGI("inner fa %d %d %d %d", s2Idx, s1Idx, s2Loop, s1Loop);
                    // LLAMA_FUNCTION(FlashAttention_L0) {
                    auto qi = View(q, {singleM, d}, {bIdx * s + s1Idx * singleM, nIdx * d});
                    std::vector<int64_t> oiOffset = {bIdx * s + s1Idx * singleM, nIdx * d};
                    std::vector<int64_t> liOffset = {(bIdx * n + nIdx) * s + s1Idx * singleM, 0};
                    std::vector<int64_t> miOffset = {(bIdx * n + nIdx) * s + s1Idx * singleM, 0};
                    SetC1CubeConfig(cubeCfg);
                    auto sij = Matrix::Matmul<false, true>(DataType::DT_FP32, qi, kj); // [128, 128], [128, 1024] => [128, 1024]

                    TileShape::Current().SetVecTile(
                        vecCfg.softmaxTileX, vecCfg.softmaxTileY);

                    auto tildaMij = RowMaxSingle(sij);
                    auto tsub = Sub(sij, tildaMij);
                    auto tildaPij = Exp(tsub);
                    auto tildaPijF16 = Cast(tildaPij, DataType::DT_FP16);
                    auto tildaLij = RowSumSingle(tildaPij);

                    SetC2CubeConfig(cubeCfg);

                    if (!s2Idx) {
                        auto oiTmp = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16, vj);
                        if (s2Loop == 1) {
                            auto liExpand = Reciprocal(tildaLij);
                            lastOi[oiOffset] = Mul(oiTmp, liExpand);
                        } else {
                            lastOi[oiOffset] = oiTmp;
                        }
                        lastLi[liOffset] = tildaLij;
                        lastMi[miOffset] = tildaMij;
                        continue;
                    }

                    ASSERT(lastOi.count(oiOffset) > 0);
                    ASSERT(lastLi.count(liOffset) > 0);
                    ASSERT(lastMi.count(miOffset) > 0);
                    auto oi = lastOi[oiOffset];
                    auto li = lastLi[liOffset];
                    auto mi = lastMi[miOffset];

                    auto miNew = Maximum(mi, tildaMij); // [128, 1], [128, 1] => [128, 1]
                    auto t1 = Sub(mi, miNew);           // [128, 1], [128, 1] => [128, 1]
                    auto t2 = Exp(t1);                  // [128, 1]
                    auto t3 = Sub(tildaMij, miNew);     // [128, 1], [128, 1] => [128, 1]
                    auto t4 = Exp(t3);                  // [128, 1]
                    auto t5 = Mul(t4, tildaLij);        // [128, 1], [128, 1] => [128, 1]
                    auto t6 = Mul(t2, li);              // [128, 1], [128, 1] => [128, 1]
                    auto liNew = Add(t6, t5);           // [128, 1], [128, 1] => [128, 1]

                    auto q3 = Mul(oi, t2);
                    auto q1 = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16, vj);
                    auto q2 = Mul(q1, t4);
                    auto oiTmp = Add(q3, q2); // [128, 128]
                    if (s2Idx == s2Loop - 1) {
                        lastOi[oiOffset] = Mul(oiTmp, Reciprocal(liNew));
                    } else {
                        lastOi[oiOffset] = oiTmp;
                    }
                    lastLi[liOffset] = liNew;
                    lastMi[miOffset] = miNew;
                }
            }
        }

        std::vector<std::pair<Tensor, std::vector<int64_t>>> aggregation;
        for (auto &[offset, tensor] : lastOi) {
            aggregation.emplace_back(tensor, offset);
        }
        result = Assemble(aggregation);
        assert(result->shape[0] == b * s);
        assert(result->shape[1] == n * d);
    }
    return result;
}


Tensor MultiAttention(const Tensor &hiddenStates, const Tensor &weight, const Tensor &m, const Tensor &l,
    const AttentionDims &atDims, const AttentionVecTileConfig &vecCfg, const AttentionCubeTileConfig &cubeCfg) {
    Tensor result;

    auto x = Cast(hiddenStates, DataType::DT_FP16);

    auto qkv = Matrix::Matmul<false, false>(DataType::DT_FP16, x, weight);
    auto q = View(qkv, hiddenStates->shape, {0, 0});
    auto k = View(qkv, hiddenStates->shape, {0, hiddenStates->shape[1]});
    auto v = View(qkv, hiddenStates->shape, {0, hiddenStates->shape[1] * 2});

    result = FlashAttention(q, k, v, m, l, atDims, vecCfg, cubeCfg);

    return result;
}

Tensor LlamaLayer(Tensor hiddenStates, const Tensor &attnWight, const Tensor &denseWeight, const Tensor &ffnWeight,
    const AttentionDims &atDims, const AttentionVecTileConfig &vecCfg, const AttentionCubeTileConfig &cubeCfg) {
    TileShape::Current().SetVecTile(vecCfg.defaultVecTileX, vecCfg.defaultVecTileY);
    SetDefaultL0CubeConfig();
    auto shape = hiddenStates->shape;
    auto residual = hiddenStates;
    hiddenStates = RmsNorm(hiddenStates);

    auto bns = atDims.b * atDims.n * atDims.s;
    std::vector<int64_t> shapeReduce = {bns, 1};
    std::vector<float> max(bns, 0);
    std::vector<float> sum(bns, 0);

    Tensor m(DataType::DT_FP32, shapeReduce);
    Tensor l(DataType::DT_FP32, shapeReduce);
    auto attentionOut = MultiAttention(hiddenStates, attnWight, m, l, atDims, vecCfg, cubeCfg);

    auto attentionOutFp16 = Cast(attentionOut, DataType::DT_FP16);
    // Dense
    SetDefaultL0CubeConfig();
    auto denseOut = Matrix::Matmul<false, false>(DataType::DT_FP32, attentionOutFp16, denseWeight);
    TileShape::Current().SetVecTile(vecCfg.defaultVecTileX, vecCfg.defaultVecTileY);
    hiddenStates = Add(residual, denseOut);

    // Fully Connected
    residual = hiddenStates;
    hiddenStates = RmsNorm(hiddenStates);

    Tensor mlpRes(DataType::DT_FP32, shape);

    auto a = Cast(hiddenStates, DataType::DT_FP16);
    auto gate = Matrix::Matmul<false, false>(DataType::DT_FP32, a, ffnWeight); // [b*s, n*d] [n*d, n*d*3] => [b*s, n*d*3]

    // swish: x / (1 + e^(-x))
    auto swish = MulS(gate, Element(DataType::DT_FP32, F_NEGA_1));
    swish = Exp(swish);
    swish = AddS(swish, Element(DataType::DT_FP32, F_1));
    swish = Div(gate, swish);

    // up_proj
    // [b*s, n*d] [n*d, n*d*3] => [b*s, n*d*3]
    auto up = Matrix::Matmul<false, false>(DataType::DT_FP32, a, ffnWeight);
    swish = Mul(swish, up);
    auto swishFp16 = Cast(swish, DataType::DT_FP16);

    // down_proj
    // [b*s, n*d*3] [n*d, n*d*3]^T => [b*s, n*d]
    mlpRes = Matrix::Matmul<false, true>(DataType::DT_FP32, swishFp16, ffnWeight);
    hiddenStates = Add(residual, mlpRes);
    return hiddenStates;
}
}

int main() {
    Program::GetInstance().GetConfig().Reset();
    AttentionDims dimsCfg = {1, 1, 128, 128, DFT_SINGLE_M, DFT_SINGLE_N};
    int b = dimsCfg.b;
    int n = dimsCfg.n;
    int s = dimsCfg.s;
    int d = dimsCfg.d;
    Tensor H(DataType::DT_FP32, {b * s, n * d}, "H");
    Tensor AW(DataType::DT_FP16, {n * d, n * d * 3}, "AW");
    Tensor DW(DataType::DT_FP16, {n * d, n * d}, "DW");
    Tensor FW(DataType::DT_FP16, {n * d, n * d * 3}, "FW");
    Tensor Res(DataType::DT_FP32, {b * s, n * d}, "Res");
    FUNCTION("LLAMA", FunctionType::STATIC, {H, AW, DW, FW, Res}) {
        Res = LlamaLayer(H, AW, DW, FW, dimsCfg, SMALL_DFS_VEC_CFG, DFS_CUBE_CFG);
    }

    Res->Dump();
    std::cout << Program::GetInstance().Dump() << std::endl;
}
