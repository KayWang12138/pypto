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
 * \file incre_flash_attention.cpp
 * \brief
 */

#include "interface/operation/operation_impl.h"
#include "interface/operation/operation.h"
#include "interface/inner/tilefwk.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {
void IncreFlashAttention(Tensor &qNope, Tensor &kNopeCache, Tensor &vNopeCache, Tensor &qRope, Tensor &kRopeCache,
    std::vector<std::vector<int>> &blockTable, std::vector<int> &actSeqs, float softmaxScale, Tensor &attentionOut,
    IfaTileShapeConfig &tileConfig) {
    auto batchSize = blockTable.size();
    ASSERT(batchSize == actSeqs.size());
    int dN = qNope->shape[1];
    int dR = qRope->shape[1];
    int nQ = 0;
    if (batchSize != 0) {
        nQ = qNope->shape[0] / batchSize;
    }
    int nTile = tileConfig.headNumQTile;
    int blockSize = tileConfig.blockSize;
    int nLoop = CeilDiv(nQ , nTile);
    auto v0Tile = tileConfig.v0TileShape;
    auto c1Tile = tileConfig.c1TileShape;
    auto v1Tile = tileConfig.v1TileShape;
    auto c2Tile = tileConfig.c2TileShape;
    auto v2Tile = tileConfig.v2TileShape;

    std::vector<std::pair<Tensor, std::vector<int>>> aggregation;
    std::vector<Tensor> tiledOut;

    for (size_t bIdx = 0; bIdx < batchSize; bIdx++) {
        int curSeq = actSeqs[bIdx];
        int bnPerBatch = CeilDiv(curSeq, blockSize);
        for (int nIdx = 0; nIdx < nLoop; nIdx++) {
            Tensor oiUpdate;
            Tensor liUpdate;
            Tensor miUpdate;

            auto nTileCur = Min(nTile, nQ - nIdx * nTile);
            auto curOffset = bIdx * nQ + nIdx * nTile;

            TileShape::Current().SetVecTile(v0Tile[0], v0Tile[1]);
            auto qn = View(qNope, {nTileCur, dN}, {static_cast<int>(curOffset), 0});
            auto qr = View(qRope, {nTileCur, dR}, {static_cast<int>(curOffset), 0});
            auto qi = Assemble({{qn, {0,0}},{qr, {0,dN}}});
            for (int bn = 0; bn < bnPerBatch; bn++) {
                auto curBlockIdx = blockTable[bIdx][bn];
                auto s2TileCur = Min(blockSize, curSeq - bn * blockSize);
                TileShape::Current().SetVecTile(v0Tile[0], v0Tile[1]);
                auto kn = View(kNopeCache, {s2TileCur, dN}, {curBlockIdx * blockSize, 0});
                auto kr = View(kRopeCache, {s2TileCur, dR}, {curBlockIdx * blockSize, 0});
                auto kj = Assemble({{kn, {0,0}}, {kr, {0,dN}}});
                auto vj = View(vNopeCache, {s2TileCur, dN}, {curBlockIdx * blockSize, 0});

                TileShape::Current().SetCubeTile(
                    {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]}, true);
                // (nTileCur, dN+dR), (s2TileCur, dN+dR) -> (nTileCur, s2TileCur)
                TileShape::Current().SetMatrixSize({qi.GetShape()[0], 0, kj.GetShape()[0]});
                auto sij = Matrix::Matmul<false, true>(DataType::DT_FP32, qi, kj);

                TileShape::Current().SetVecTile(v1Tile[0], v1Tile[1]);
                auto sijScale = MulS(sij, Element(DataType::DT_FP32, softmaxScale)); // (nTileCur, s2TileCur)
                auto tildaMij = Amax(sijScale);   // (nTileCur, s2TileCur) -> (nTileCur, 1)
                auto tsub = Sub(sijScale, tildaMij); // (nTileCur, s2TileCur) - (nTileCur, 1) -> (nTileCur, s2TileCur)
                auto tildaPij = Exp(tsub);
                auto tildaPijF16 = Cast(tildaPij, DataType::DT_BF16);
                auto tildaLij = Sum(tildaPij); // (nTileCur, s2TileCur) -> (nTileCur, 1)

                if (bn == 0) {
                    TileShape::Current().SetCubeTile(
                        {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]}, true);
                    TileShape::Current().SetMatrixSize(
                        {tildaPijF16.GetShape()[0], tildaPijF16.GetShape()[1], vj.GetShape()[1]});
                    auto oiTmp = Matrix::Matmul<false, false>(DataType::DT_FP32,tildaPijF16, vj);

                    TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                    oiUpdate = (bnPerBatch == 1 ? Div(oiTmp, tildaLij) : oiTmp);
                    liUpdate = tildaLij;
                    miUpdate = tildaMij;
                    continue;
                }

                auto oi = oiUpdate;
                auto li = liUpdate;
                auto mi = miUpdate;

                auto miNew = Maximum(mi, tildaMij); // (nTileCur, 1), (nTileCur, 1) -> (nTileCur, 1)
                auto t1 = Sub(mi, miNew);           // (nTileCur, 1), (nTileCur, 1) -> (nTileCur, 1)
                auto t2 = Exp(t1);
                auto t3 = Sub(tildaMij, miNew); // (nTileCur, 1), (nTileCur, 1) -> (nTileCur, 1)
                auto t4 = Exp(t3);
                auto t5 = Mul(t4, tildaLij); // (nTileCur, 1), (nTileCur, 1) -> (nTileCur, 1)
                auto t6 = Mul(t2, li);       // (nTileCur, 1), (nTileCur, 1) -> (nTileCur, 1)
                auto liNew = Add(t6, t5);    // (nTileCur, 1), (nTileCur, 1) -> (nTileCur, 1)
                auto q3 = Mul(oi, t2); // (nTileCur, dN), (nTileCur, 1) -> (nTileCur, dN)
                TileShape::Current().SetCubeTile(
                    {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]}, true);
                // (nTileCur, s2TileCur), (s2TileCur, dN) -> (nTileCur, dN)
                TileShape::Current().SetMatrixSize(
                    {tildaPijF16.GetShape()[0], tildaPijF16.GetShape()[1], vj.GetShape()[1]});
                auto q1 = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16, vj);
                TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                auto q2 = Mul(q1, t4);    // (nTileCur, dN), (nTileCur, 1) -> (nTileCur, dN)
                auto oiTmp = Add(q3, q2); // (nTileCur, dN), (nTileCur, dN) -> (nTileCur, dN)
                oiUpdate = (bn == bnPerBatch - 1 ?  Div(oiTmp, liNew) : oiTmp);
                liUpdate = liNew;
                miUpdate = miNew;
            }
            tiledOut.push_back(oiUpdate);
        }
    }

    attentionOut = Cat(tiledOut, 0);
}

} // namespace npu::tile_fwk

int main()
{
    std::unordered_map<std::string, int> params = {
        {"b", 4},
        {"nq", 32},
        {"s2", 256}
    };

    IfaTileShapeConfig tileConfig {
        256, // block size
        32,  // nTile
        {256, 128}, // v0 tile for qkv-view-concat, q-S1D:(32,64), k/v-S2D:(256,64), merge 2D to copy
        {32, 32, 256, 256, 128, 128}, // c1 tile for S1D@S2D
        {32, 256}, // v1 tile for S1S2
        {32, 32, 256, 256, 128, 128}, // c2 tile for S1S2@S2D
        {32, 256}, // v2 tile for S1D
    };

    const int b = params["b"];
    const int nq = params["nq"];
    const int s2 = params["s2"];
    const int blockSize = tileConfig.blockSize;
    const int sq = 1;
    const int dn = 512;
    const int dr = 64;
    const int nkv = 1;

    std::vector<int> actSeqs(b, s2);
    const float softmaxScale = static_cast<float>(1.0 / std::sqrt(dn + dr));

    // 根据Per Batch实际的sequence构造blockNum，blockNum >= Sum(blockNumPerBatch)，此处选取相等场景
    int blockNum = 0;
    for (auto s : actSeqs) {
        blockNum += CeilDiv(s, blockSize);
    }

    Tensor qNope(DT_BF16, {b * sq * nq, dn}, "qNope");
    Tensor qRope(DT_BF16, {b * sq * nq, dr}, "qRope");

    Tensor kvNopeCache(DT_BF16, {blockNum * blockSize * nkv, dn}, "kNopeCache", TileOpFormat::TILEOP_NZ);
    Tensor kRopeCache(DT_BF16, {blockNum * blockSize * nkv, dr}, "kRope", TileOpFormat::TILEOP_NZ);

    // blockTable: (b, maxBlockNumPerBatch)
    int maxSeqAllBatch = *(std::max_element(actSeqs.begin(), actSeqs.end()));
    int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);
    std::vector<std::vector<int>> blockTable(b, std::vector<int>(maxBlockNumPerBatch, 0));
    for (int i = 0 ; i < b; ++i) {
        for (int j = 0 ; j < maxBlockNumPerBatch; ++j) {
            blockTable[i][j] = i * maxBlockNumPerBatch + j;
        }
    }

    Tensor attentionOut(DT_FP32, {b * sq * nq, dn}, "attentionOut");

    config::SetBuildStatic(true);
    FUNCTION("IfaStatic", {qNope, kvNopeCache, qRope, kRopeCache, attentionOut}) {
        IncreFlashAttention(qNope, kvNopeCache, kvNopeCache, qRope, kRopeCache, blockTable, actSeqs, softmaxScale,
            attentionOut, tileConfig);
    }
}