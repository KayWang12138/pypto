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
 * \file win_attention.cpp
 * \brief
 */

#include "interface/operation/operation_impl.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/tensor/tensormap.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/common.h"
#include "interface/utils/id_gen.h"
#include "interface/utils/log.h"
#include "win_attention.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {

void WinAttentionCompute(const Tensor &qNope, Tensor &vNopeCache, const Tensor &qRope, Tensor &kRopeCache, int nQ, int nKv,
    Tensor &blockTable, Tensor &actSeqs, int windowSize, int blockSize, float softmaxScale, Tensor &attentionOut,
    WinAttenTileShapeConfig &tileConfig) {
    auto dtype = qNope->Datatype();

    // 入参B*S*N合轴
    int dNopeSize = qNope->shape[1];
    int dRopeSize = qRope->shape[1];
    ASSERT(nKv != 0) << "nKv cant't be zero!";
    auto gGroup = nQ / nKv;
    int gTile = tileConfig.gTile; // 128
    ASSERT(blockSize != 0) << "blockSize can't be zero!";

    auto nopeTile = tileConfig.vNopeTileShape;
    auto ropeTile = tileConfig.vRopeTileShape;
    auto outTile = tileConfig.outTileShape;
    auto c1Tile = tileConfig.c1TileShape;
    auto v1Tile = tileConfig.v1TileShape;
    auto c2Tile = tileConfig.c2TileShape;
    auto v2Tile = tileConfig.v2TileShape;
    // loop config
    SymbolicScalar bSize = blockTable->shape[0];
    SymbolicScalar bTile = 1;
    ASSERT(bTile != 0) << "bTile can't be zero!";
    ASSERT(nQ != 0) << "nQ can't be zero!";
    SymbolicScalar bLoop = bSize / bTile;
    SymbolicScalar s1Size = qNope->shape[0] / bSize / nQ; // [B_s1_N1, D]
    SymbolicScalar s1Tile = 1;
    ASSERT(s1Tile != 0) << "s1Tile can't be zero!";
    SymbolicScalar s1Loop = s1Size / s1Tile;
    SymbolicScalar n2Tile = 1;
    ASSERT(n2Tile != 0) << "n2Tile can't be zero!";
    SymbolicScalar n2Loop = nKv / n2Tile;
    ASSERT(gTile != 0) << "gTile can't be zero!";
    SymbolicScalar gLoop = gGroup / gTile;
    // block tile config
    SymbolicScalar blockStartIndex = 0;
    SymbolicScalar blockStartOffset = 0;
    SymbolicScalar blockEndIndex = 0;
    SymbolicScalar winActualSize = 0;
    SymbolicScalar tableLoop = 0;

    LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bLoop, 1), {}, true) {
        SymbolicScalar curActualSeqSize = GetInputDataInt32Dim1(actSeqs, bIdx);
        LOOP("LOOP_L1_s1Idx", FunctionType::DYNAMIC_LOOP, s1Idx, LoopRange(0, s1Loop, 1)) {
            winActualSize = std::min(windowSize, (curActualSeqSize - s1Size + s1Idx + 1));
            blockEndIndex = (curActualSeqSize + blockSize - 1) / blockSize - 1;
            blockStartIndex = std::max(0, ((curActualSeqSize - winActualSize - s1Size + 1 + s1Idx) / blockSize));
            blockStartOffset = (curActualSeqSize - winActualSize - s1Size + 1 + s1Idx) % blockSize;
            tableLoop = blockEndIndex - blockStartIndex + 1;
            LOOP("LOOP_L2_n2Idx", FunctionType::DYNAMIC_LOOP, n2Idx, LoopRange(0, n2Loop, 1)) {
                LOOP("LOOP_L2_gIdx", FunctionType::DYNAMIC_LOOP, gIdx, LoopRange(0, gLoop, 1)) {
                    std::vector<SymbolicScalar> outOffset = {bIdx , s1Idx , n2Idx * gGroup + gIdx * gTile, 0};
                    Tensor kPart(dtype, {5 * blockSize, (dNopeSize + dRopeSize)}, "kPart");
                    Tensor vPart(dtype, {5 * blockSize, dNopeSize}, "vPart");
                    LOOP("LOOP_L2_tIdx", FunctionType::DYNAMIC_LOOP, tIdx, LoopRange(0, tableLoop, 1), {}, true) {
                        SymbolicScalar curidx = blockStartIndex + tIdx;
                        SymbolicScalar curBlockIdx = GetInputDataInt32Dim2(blockTable, bIdx, curidx);

                        auto kNope = View(vNopeCache, {blockSize, dNopeSize}, {curBlockIdx * blockSize, n2Idx * dNopeSize});
                        Program::GetInstance().GetTileShape().SetVecTileShapes(nopeTile[0], nopeTile[1]);
                        auto tmpK1 = Cast(kNope, DataType::DT_FP32);
                        auto tmpK2 = Cast(tmpK1, dtype);
                        Assemble(tmpK2, {tIdx * blockSize, 0}, kPart);

                        auto kRope = View(kRopeCache, {blockSize, dRopeSize}, {curBlockIdx * blockSize, n2Idx * dRopeSize});
                        Program::GetInstance().GetTileShape().SetVecTileShapes(ropeTile[0], ropeTile[1]);
                        auto tmpKR1 = Cast(kRope, DataType::DT_FP32);
                        auto tmpKR2 = Cast(tmpKR1, dtype);
                        Assemble(tmpKR2, {tIdx * blockSize, dNopeSize}, kPart);

                        auto vNope = View(vNopeCache, {blockSize, dNopeSize}, {curBlockIdx * blockSize, n2Idx * dNopeSize});
                        Program::GetInstance().GetTileShape().SetVecTileShapes(nopeTile[0], nopeTile[1]);
                        auto tmpV1 = Cast(vNope, DataType::DT_FP32);
                        auto tmpV2 = Cast(tmpV1, dtype);
                        Assemble(tmpV2, {tIdx * blockSize, 0}, vPart);
                    }
                    LOOP("LOOP_L2_Idx", FunctionType::DYNAMIC_LOOP, oIdx, LoopRange(1), {}, true) {
                        (void) oIdx;
                        SymbolicScalar curOffset = bIdx * s1Size * nQ + s1Idx * nQ + n2Idx * gGroup + gIdx * gTile;
                        auto kActualPart = View(kPart, {windowSize, dNopeSize + dRopeSize},
                            {winActualSize, dNopeSize + dRopeSize}, {blockStartOffset, 0});
                        auto vActualPart = View(vPart, {windowSize, dNopeSize}, {winActualSize, dNopeSize},
                            {blockStartOffset, 0});
                        Tensor qPart(dtype, {gTile, dNopeSize + dRopeSize}, "qPart");
                        // query
                        auto qNopeL = View(qNope, {gTile, dNopeSize}, {gTile, dNopeSize}, {curOffset, 0});
                        Assemble(qNopeL, {0, 0}, qPart);
                        auto qRopeR = View(qRope, {gTile, dNopeSize}, {gTile, dRopeSize}, {curOffset, 0});
                        Assemble(qRopeR, {0, dNopeSize}, qPart);

                        // matmul_1
                        Program::GetInstance().GetTileShape().SetCubeTileShapes(
                                {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]}, true);
                        auto qKT = Matrix::Matmul<false, true>(DataType::DT_FP32, qPart, kActualPart);
                        Program::GetInstance().GetTileShape().SetVecTileShapes(v1Tile[0], v1Tile[1]);
                        auto qKTScale = MulS(qKT, Element(qKT->Datatype(), softmaxScale));

                        // softmax
                        auto tileMax = RowMaxSingle(qKTScale); // max
                        auto tileSub = Sub(qKTScale, tileMax); // sub max
                        auto tileExp = Exp(tileSub); // exp
                        auto tilSum = RowSumSingle(tileExp);
                        auto tileSoftmx = Div(tileExp, tilSum);
                        auto valueType16 = Cast(tileSoftmx, dtype);

                        // matmul_2
                        Program::GetInstance().GetTileShape().SetCubeTileShapes(
                                {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]}, true);
                        auto out = Matrix::Matmul<false, false>(DataType::DT_FP32, valueType16, vActualPart);

                        // reshape and copyOut
                        Program::GetInstance().GetTileShape().SetVecTileShapes(v2Tile[0], v2Tile[1]);
                        auto outNew = Reshape(out, {bTile, s1Tile, gTile, dNopeSize});
                        Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, outTile[0], outTile[1]);
                        auto outFinal = AddS(outNew, Element(outNew->Datatype(), 0.0));
                        Assemble(outFinal, outOffset, attentionOut);
                    }
                }
            }
        }
    }
}

void WinAttention(const Tensor &qNope, Tensor &vNopeCache, const Tensor &qRope, Tensor &kRopeCache, int nQ, int nKv,
    Tensor &blockTable, Tensor &actSeqs, int windowSize, int blockSize, float softmaxScale, Tensor &attentionOut,
    WinAttenTileShapeConfig &tileConfig) {
    FunctionConfig funConfig;
    FUNCTION("main", funConfig,
        {qNope, vNopeCache, qRope, kRopeCache, blockTable, actSeqs}, {attentionOut}) {
        WinAttentionCompute(qNope, vNopeCache, qRope, kRopeCache, nQ, nKv, blockTable, actSeqs, windowSize, blockSize,
            softmaxScale, attentionOut, tileConfig);
    }
}
} // namespace npu::tile_fwk
