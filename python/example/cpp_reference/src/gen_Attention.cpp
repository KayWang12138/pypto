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
 * \file gen_Attention.cpp
 * \brief
 */

#include "interface/function/function.h"
#include "interface/configs/config_manager.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {

constexpr int NUM_3 = 3;
constexpr int NUM_8 = 8;
constexpr int NUM_16 = 16;
constexpr int NUM_128 = 128;
constexpr int NUM_512 = 512;
typedef npu::tile_fwk::bfloat16 T;

struct GenAttenTileShapeConfig {
    int tileBSize;
    int tileS1Size;
    std::array<int, TILE_VEC_FOUR_DIMS> vec1TileShape; // vector op tileshape
    std::array<int, TILE_VEC_FOUR_DIMS> vec2TileShape; // vector op tileshape
};

void GenAttentionCompute(Tensor &cmpAtten, Tensor &selAtten, Tensor &winAtten, Tensor &gatingScore, Tensor &attentionOut, GenAttenTileShapeConfig &tileConfig) {
    int nDimSize = cmpAtten.GetShape()[2];
    int dDimSize = cmpAtten.GetShape()[3];
    int dGateDimSize = gatingScore.GetShape()[3];
    int tileB = tileConfig.tileBSize;
    int tileS = tileConfig.tileS1Size;
    auto v1Tile = tileConfig.vec1TileShape;
    auto v2Tile = tileConfig.vec2TileShape;

    SymbolicScalar bDimSize = GetInputShape(cmpAtten, 0);
    SymbolicScalar sDimSize = GetInputShape(cmpAtten, 1);
    SymbolicScalar bLoop = bDimSize / tileB;
    SymbolicScalar sLoop = sDimSize / tileS;
    DataType dType = cmpAtten.GetStorage()->Datatype();
    LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(bLoop)) {
        SymbolicScalar bOffset = bIdx * tileB;
        SymbolicScalar actualBSize = std::min(tileB, (bDimSize - bIdx * tileB));
        LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(sLoop)) {
            SymbolicScalar sOffset = sIdx * tileS;
            std::vector<SymbolicScalar> outOffset = {bOffset, sOffset, 0, 0};
            SymbolicScalar actualsSize = std::min(tileS, (sDimSize - sIdx * tileS));
            TileShape::Current().SetVecTile(v1Tile[0], v1Tile[1], v1Tile[2], v1Tile[3]);
            auto cmpAttenTile = View(cmpAtten, {tileB, tileS, nDimSize, dDimSize},
                {actualBSize, actualsSize, nDimSize, dDimSize}, {bOffset, sOffset, 0, 0});
            auto selAttenTile = View(selAtten, {tileB, tileS, nDimSize, dDimSize},
                {actualBSize, actualsSize, nDimSize, dDimSize}, {bOffset, sOffset, 0, 0});
            auto winAttenTile = View(winAtten, {tileB, tileS, nDimSize, dDimSize},
                {actualBSize, actualsSize, nDimSize, dDimSize}, {bOffset, sOffset, 0, 0});
            auto cmpAttenFP32Tile = Cast(cmpAttenTile, DT_FP32);
            auto selAttenFP32Tile = Cast(selAttenTile, DT_FP32);
            auto winAttenFP32Tile = Cast(winAttenTile, DT_FP32);
            TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1], v2Tile[2], v2Tile[3]);
            auto gatingScoreTile = View(gatingScore, {tileB, tileS, nDimSize, dGateDimSize},
                {actualBSize, actualsSize, nDimSize, dGateDimSize}, {bOffset, sOffset, 0, 0});
            auto gatingScoreFP32 = Cast(gatingScoreTile, DT_FP32);
            auto cmpWeight = View(gatingScoreFP32, {tileB, tileS, nDimSize, 1}, {0, 0, 0, 0});
            auto selWeight = View(gatingScoreFP32, {tileB, tileS, nDimSize, 1}, {0, 0, 0, 1});
            auto winWeight = View(gatingScoreFP32, {tileB, tileS, nDimSize, 1}, {0, 0, 0, 2});
            TileShape::Current().SetVecTile(v1Tile[0], v1Tile[1], v1Tile[2], v1Tile[3]);
            auto mulCmp = Mul(cmpAttenFP32Tile, cmpWeight);
            auto mulSel = Mul(selAttenFP32Tile, selWeight);
            auto mulWin = Mul(winAttenFP32Tile, winWeight);
            auto addCmpSel = Add(mulCmp, mulSel);
            auto outFP32 = Add(addCmpSel, mulWin);
            TileShape::Current().SetVecTile(v1Tile[0], v1Tile[1], v1Tile[2], v1Tile[3]);
            auto attentionOutTile = Cast(outFP32, dType, CAST_RINT);
            Assemble(attentionOutTile, outOffset, attentionOut);
        }
    }
}

void GenAttention(Tensor &cmpAtten, Tensor &selAtten, Tensor &winAtten, Tensor &gatingScore, Tensor &attentionOut, GenAttenTileShapeConfig &tileConfig) {
    FUNCTION("main", {cmpAtten, selAtten, winAtten, gatingScore}, {attentionOut}) {
        GenAttentionCompute(cmpAtten, selAtten, winAtten, gatingScore, attentionOut, tileConfig);
}
}

} // namespace tile_fwk

int main () {
    int B = NUM_16;
    int N = NUM_128;
    int S1 = 1;
    int D = NUM_512;
    DataType dType;
    if (std::is_same<T, float>::value) {
        dType = DT_FP32;
    } else {
        dType = (std::is_same<T, npu::tile_fwk::float16>::value) ? DT_FP16 : DT_BF16;
    }

    std::vector<int64_t> shape_cmpAtten = {B, S1, N, D};
    std::vector<int64_t> shape_selAtten = {B, S1, N, D};
    std::vector<int64_t> shape_winAtten = {B, S1, N, D};
    std::vector<int64_t> shape_gatingScore = {B, S1, N, NUM_3};
    std::vector<int64_t> shape_attentionOut = {B, S1, N, D};

    Tensor cmpAtten(dType, shape_cmpAtten, "cmpAtten");
    Tensor selAtten(dType, shape_selAtten, "selAtten");
    Tensor winAtten(dType, shape_winAtten, "winAtten");
    Tensor gatingScore(dType, shape_gatingScore, "gatingScore");
    Tensor out_npu(dType, shape_attentionOut, "out_npu");

    GenAttenTileShapeConfig tileConfig;
    const int dTileSize = NUM_512;
    const int nTileSize = NUM_128;
    tileConfig.tileBSize = NUM_8;
    tileConfig.tileS1Size = 1;
    tileConfig.vec1TileShape = {1, 1, NUM_16, dTileSize};
    tileConfig.vec2TileShape = {1, 1, nTileSize, NUM_3};

    GenAttention(cmpAtten, selAtten, winAtten, gatingScore, out_npu, tileConfig);
}
