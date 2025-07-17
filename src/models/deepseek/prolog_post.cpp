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
 * \file prolog_post.cpp
 * \brief
 */

#include "models/deepseek/deepseek_mla.h"
#include "interface/operation/operation_impl.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "tilefwk/tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/tensor/tensormap.h"
#include "interface/configs/config_manager.h"
#include "interface/configs/config.h"
#include "interface/utils/common.h"
#include "interface/utils/id_gen.h"
#include "interface/utils/log.h"
#include "page_attention.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {
void PrologPost(Tensor &qNope, Tensor &kNopeCache, Tensor &vNopeCache, Tensor &qRope, Tensor &kRopeCache,
    Tensor &blockTable, Tensor &actSeqs, Tensor &weightUV, Tensor &weightO, int blockSize, float softmaxScale,
    Tensor &postOut, PaTileShapeConfig &tileConfig) {
    auto dtype = qNope->Datatype();
    // 入参B*S*N合轴
    int sQ = 1;
    int dN = qNope->shape[1];
    int dR = qRope->shape[1];
    int tile4 = 4;

    int nTile = tileConfig.headNumQTile;
    int vHeadDim = weightUV->shape[2];  // (nQ, dN, vHeadDim)
    int hiddenSize = weightO->shape[1]; // (nQ*VHeadDim, H)

    auto v0Tile = tileConfig.v0TileShape;
    auto c1Tile = tileConfig.c1TileShape;
    auto v1Tile = tileConfig.v1TileShape;
    auto c2Tile = tileConfig.c2TileShape;
    auto v2Tile = tileConfig.v2TileShape;

    int batchSize = blockTable->shape[0];
    int nQ = qNope->shape[0] / batchSize; // B*1*N
    int nLoop = nQ / nTile;

    Tensor attentionOut(DT_FP32, qNope->shape, "attentionOut");
    
    FUNCTION("main", FunctionType::DYNAMIC,
        {qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs, weightUV, weightO}, {postOut}) {
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(batchSize)) {
            SymbolicScalar curSeq = GetInputDataInt32Dim1(actSeqs, bIdx);
            SymbolicScalar bnPerBatch = curSeq / blockSize; // 暂时仅考虑curSeq是blockSize对齐
            // nLoop是因为B*N*S合轴，计算N时是SymbolicScalar计算，此处不一定需要用Loop
            LOOP("LOOP_L1_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(nLoop)) {
                Tensor oiUpdate(DT_FP32, {nTile, dN}, "oiUpdate");
                Tensor liUpdate(DT_FP32, {nTile, 1}, "liUpdate");
                Tensor miUpdate(DT_FP32, {nTile, 1}, "miUpdate");
                // 当前curOffset没放到更内层循环，避免重复bnPerBatch次的DAssemble操作
                SymbolicScalar curOffset = bIdx * nQ + nIdx * nTile;
                std::vector<SymbolicScalar> oiOffset = {curOffset, 0}; // (B*N*S, d)

                LOOP("LOOP_L2_bn", FunctionType::DYNAMIC_LOOP, bn, LoopRange(bnPerBatch)) {
                    // 当前qn，qr和qi放入内层Loop，避免Concat单独切成一个小图
                    Program::GetInstance().GetTileShape().SetVecTileShapes(v0Tile[0], v0Tile[1]);
                    auto qn = DView(qNope, {nTile, dN}, {curOffset, 0});
                    auto qr = DView(qRope, {nTile, dR}, {curOffset, 0});
                    auto qi = Concat({qn, qr}, 1); // (nTileCur, dN+dR)
                    SymbolicScalar curBlockIdx = GetInputDataInt32Dim2(blockTable, bIdx, bn);

                    auto kn = DView(kNopeCache, {blockSize, dN}, {curBlockIdx * blockSize, 0});
                    auto kr = DView(kRopeCache, {blockSize, dR}, {curBlockIdx * blockSize, 0});
                    auto kj = Concat({kn, kr}, 1); // (s2TileCur, dN+dR)
                    auto vj = DView(vNopeCache, {blockSize, dN}, {curBlockIdx * blockSize, 0});
                    Program::GetInstance().GetTileShape().SetCubeTileShapes(
                        {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]});

                    auto sij =
                        Matrix::Matmul<false, true>(DataType::DT_FP32, qi, kj); // (nTileCur, dN+dR), (s2TileCur, dN+dR) -> (nTileCur, s2TileCur)
                    Program::GetInstance().GetTileShape().SetVecTileShapes(v1Tile[0], v1Tile[1]);
                    auto sijScale = MulS(sij, Element(DataType::DT_FP32, softmaxScale)); // (nTileCur, s2TileCur)

                    auto tildaMij = RowMaxSingle(sijScale); // (nTileCur, s2TileCur) -> (nTileCur, 1)
                    auto tsub =
                        Sub(sijScale, tildaMij); // (nTileCur, s2TileCur) - (nTileCur, 1) -> (nTileCur, s2TileCur)
                    auto tildaPij = Exp(tsub);
                    auto tildaPijF16 = Cast(tildaPij, dtype);
                    auto tildaLij = RowSumSingle(tildaPij); // (nTileCur, s2TileCur) -> (nTileCur, 1)

                    IF(bn == 0) {
                        Program::GetInstance().GetTileShape().SetCubeTileShapes(
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]});
                        auto oiTmp = Matrix::Matmul<false, false>(DataType::DT_FP32, 
                            tildaPijF16, vj); // (nTileCur, s2TileCur), (s2TileCur, dN) -> (nTileCur, dN)
                        Program::GetInstance().GetTileShape().SetVecTileShapes(v2Tile[0], v2Tile[1]);
                        IF(bnPerBatch == 1) {
                            oiUpdate = Div(oiTmp, tildaLij); // (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
                        }
                        ELSE {
                            oiUpdate = oiTmp;
                        }
                        liUpdate = tildaLij;
                        miUpdate = tildaMij;
                    }
                    ELSE {
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
                        Program::GetInstance().GetTileShape().SetCubeTileShapes(
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]});
                        auto q1 = Matrix::Matmul<false, false>(DataType::DT_FP32,
                            tildaPijF16, vj); // (nTileCur, s2TileCur), (s2TileCur, dN) -> (nTileCur, dN)
                        Program::GetInstance().GetTileShape().SetVecTileShapes(v2Tile[0], v2Tile[1]);
                        auto q2 = Mul(q1, t4);    // (nTileCur, dN), (nTileCur, 1) -> (nTileCur, dN)
                        auto oiTmp = Add(q3, q2); // (nTileCur, dN), (nTileCur, dN) -> (nTileCur, dN)
                        IF(bn == bnPerBatch - 1) {
                            oiUpdate = Div(oiTmp, liNew); // (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
                        }
                        ELSE {
                            oiUpdate = oiTmp;
                        }
                        liUpdate = liNew;
                        miUpdate = miNew;
                    }
                    DAssemble(oiUpdate, oiOffset, attentionOut);
                }
            }
        }

        FUNCTION("PaPost", FunctionType::STATIC) {
            Program::GetInstance().GetTileShape().SetVecTileShapes({32, dN});
            auto attenRes = Reshape(attentionOut, {batchSize, nQ, dN}); // (b*sQ*nQ, dN), sQ=1

            Program::GetInstance().GetTileShape().SetVecTileShapes({2, 16, dN});
            auto castOut = Cast(attenRes, dtype);         // (b*sQ, nQ, dN)
            auto attenTrans = Transpose(castOut, {0, 1}); // (b*sQ, nQ, dN) -> (nQ, b*sQ, dN)

            // Bmm的Tile设置也是M, K, N, 与Batch无关；注意，MM的维度不足16时要按照16对齐设置
            Program::GetInstance().GetTileShape().SetCubeTileShapes({16, 16}, {dN, dN}, {vHeadDim, vHeadDim});
            auto bmmRes = Matrix::BatchMatmul<false, false>(DataType::DT_FP32,
                attenTrans, weightUV); // (nQ, b*sQ, dN) * (nQ, dN, vHeadDim) -> (nQ, b*sQ, vHeadDim)

            // cast不支持跳写，这个Transpose必须与上面的tileshape后两维一致； 2、Transpose尾轴不能切
            Program::GetInstance().GetTileShape().SetVecTileShapes(1, tile4, vHeadDim);
            auto bmmTrans = Transpose(bmmRes, {0, 1}); // (nQ, b*sQ, vHeadDim) -> (b*sQ, nQ, vHeadDim)

            Program::GetInstance().GetTileShape().SetVecTileShapes({1, nQ, vHeadDim});
            auto bmmReshape =
                Reshape(bmmTrans, {batchSize * sQ, nQ * vHeadDim}); // (b*sQ, nQ, vHeadDim) -> (b*sQ, nQ*vHeadDim)

            Program::GetInstance().GetTileShape().SetCubeTileShapes({16, 16}, {32, 32}, {hiddenSize, hiddenSize});
            Tensor postMm =
                Matrix::Matmul<false, false>(DataType::DT_FP32, bmmReshape, weightO); // (b*sQ, nQ*vHeadDim) * (nQ*VHeadDim, H) -> (b*sQ, H)

            Program::GetInstance().GetTileShape().SetVecTileShapes({batchSize * sQ, 32});
            postOut = Reshape(postMm, {batchSize, sQ, hiddenSize});
            std::cout << "111111" << std::endl;
        }
    }
}

void PageAttentionAddS(Tensor &qNope, Tensor &kNopeCache, Tensor &vNopeCache, Tensor &qRope, Tensor &kRopeCache,
        Tensor &blockTable, Tensor &actSeqs, int blockSize, float softmaxScale, Tensor &attentionOut, Tensor &postOut,
    PaTileShapeConfig &tileConfig, int maxUnrollTimes) {
    auto dtype = qNope->Datatype();
    // 入参B*S*N合轴
    int dN = qNope->shape[1];
    int dR = qRope->shape[1];
 
    int nTile = tileConfig.headNumQTile;
    auto c1Tile = tileConfig.c1TileShape;
    auto v1Tile = tileConfig.v1TileShape;
    auto c2Tile = tileConfig.c2TileShape;
    auto v2Tile = tileConfig.v2TileShape;
 
    int batchSize = blockTable->shape[0];
    int nQ = qNope->shape[0] / batchSize; // B*1*N
 
    auto N = 128;
    auto kvLoraRank = 512;
    int S = 1;
 
    FUNCTION("main", FunctionType::DYNAMIC,
        {qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs}, {attentionOut, postOut}) {
        SymbolicScalar nLoop = nQ / nTile;
 
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, batchSize, 1)) {
            SymbolicScalar curSeq = GetInputDataInt32Dim1(actSeqs, bIdx);
            SymbolicScalar bnPerBatch = curSeq / blockSize; // 暂时仅考虑curSeq是blockSize对齐
            bnPerBatch.AsIntermediateVariable();
            LOOP("LOOP_L1_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nLoop, 1)) {
                int curNTile = nTile;
                Tensor oiUpdate(DT_FP32, {nTile, dN}, "oiUpdate");
                Tensor liUpdate(DT_FP32, {nTile, 1}, "liUpdate");
                Tensor miUpdate(DT_FP32, {nTile, 1}, "miUpdate");
                // 当前curOffset没放到更内层循环，避免重复bnPerBatch次的DAssemble操作
                SymbolicScalar curOffset = bIdx * nQ + nIdx * nTile;
                std::vector<SymbolicScalar> oiOffset = {curOffset, 0}; // (B*N*S, d)
 
                LOOP("LOOP_L2_bn", FunctionType::DYNAMIC_LOOP, bn, LoopRange(0, bnPerBatch, 1), PowersOf2(maxUnrollTimes)) {
                    // 当前qn，qr和qi放入内层Loop，避免Concat单独切成一个小图
                    int curS2Tile = blockSize;
                    auto qn = DView(qNope, {curNTile, dN}, {curOffset, 0});
                    auto qr = DView(qRope, {curNTile, dR}, {curOffset, 0});
                    Tensor qi(dtype, {curNTile, dN + dR}, "qi");
                    DAssemble(qn, {0, 0}, qi);
                    DAssemble(qr, {0, dN}, qi);
 
                    SymbolicScalar curBlockIdx = GetInputDataInt32Dim2(blockTable, bIdx, bn);
                    curBlockIdx.AsIntermediateVariable();
                    auto kn = DViewPad(kNopeCache, {curS2Tile, dN}, {std::min(curSeq - bn * blockSize, blockSize), dN},
                                                  {curBlockIdx * blockSize, 0});
                    auto kr = DViewPad(kRopeCache, {curS2Tile, dR}, {std::min(curSeq - bn * blockSize, blockSize), dR},
                                                  {curBlockIdx * blockSize, 0});
                    Tensor kj(dtype, {curS2Tile, dN + dR}, "kj");
                    DAssemble(kn, {0, 0}, kj);
                    DAssemble(kr, {0, dN}, kj);
                    auto vj = DViewPad(vNopeCache, {curS2Tile, dN}, {std::min(curSeq - bn * blockSize, blockSize), dN},
                                                  {curBlockIdx * blockSize, 0});
 
                    ConfigManager::Instance().SetSemanticLabel("MatMul");
                    Program::GetInstance().GetTileShape().SetCubeTileShapes(
                        {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]});
                    auto sij = Matrix::Matmul<false, true>(DataType::DT_FP32, qi, kj); // (curNTile, dN+dR), (curS2Tile, dN+dR) -> (curNTile, curS2Tile)
                    Program::GetInstance().GetTileShape().SetVecTileShapes(v1Tile[0], v1Tile[1]);
 
                    ConfigManager::Instance().SetSemanticLabel("SoftMax");
                    auto sijScale = MulS(sij, Element(DataType::DT_FP32, softmaxScale)); // (curNTile, curS2Tile)
 
                    auto tildaMij = RowMaxSingle(sijScale); // (curNTile, curS2Tile) -> (curNTile, 1)
                    auto tsub =
                        Sub(sijScale, tildaMij); // (curNTile, curS2Tile) - (curNTile, 1) -> (curNTile, curS2Tile)
                    auto tildaPij = Exp(tsub);
                    auto tildaPijF16 = Cast(tildaPij, dtype);
                    auto tildaLij = RowSumSingle(tildaPij); // (nTileCur, s2TileCur) -> (nTileCur, 1)
 
                    IF (IsLoopBegin(bn, 0)) {
                        Program::GetInstance().GetTileShape().SetCubeTileShapes(
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]});
                        ConfigManager::Instance().SetSemanticLabel("b1-matmul2");
                        auto oiTmp = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16, vj);; // (curNTile, curS2Tile), (curS2Tile, dN) -> (curNTile, dN)
                        Program::GetInstance().GetTileShape().SetVecTileShapes(v2Tile[0], v2Tile[1]);
                        ConfigManager::Instance().SetSemanticLabel("b1-after-matmul2");
                        IF (IsLoopEnd(bn, bnPerBatch)) {
                            oiUpdate = Div(oiTmp, tildaLij); // (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
                            DAssemble(oiUpdate, oiOffset, attentionOut);
                        } ELSE {
                            oiUpdate = oiTmp;
                        }
                        liUpdate = tildaLij;
                        miUpdate = tildaMij;
                    } ELSE {
                        auto oi = oiUpdate;
                        auto li = liUpdate;
                        auto mi = miUpdate;
 
                        ConfigManager::Instance().SetSemanticLabel("Softmax-acc");
                        auto miNew = Maximum(mi, tildaMij); // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                        auto t1 = Sub(mi, miNew);           // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                        auto t2 = Exp(t1);
                        auto t3 = Sub(tildaMij, miNew); // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                        auto t4 = Exp(t3);
                        auto t5 = Mul(t4, tildaLij); // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                        auto t6 = Mul(t2, li);       // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                        auto liNew = Add(t6, t5);    // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
 
                        auto q3 = Mul(oi, t2); // (curNTile, dN), (curNTile, 1) -> (curNTile, dN)
                        Program::GetInstance().GetTileShape().SetCubeTileShapes(
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]});
                        ConfigManager::Instance().SetSemanticLabel("bn-matmul2");
                        auto q1 = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16, vj); // (curNTile, curS2Tile), (curS2Tile, dN) -> (curNTile, dN)
                        Program::GetInstance().GetTileShape().SetVecTileShapes(v2Tile[0], v2Tile[1]);
                        ConfigManager::Instance().SetSemanticLabel("bn-after-matmul2");
                        auto q2 = Mul(q1, t4);    // (nTileCur, dN), (nTileCur, 1) -> (nTileCur, dN)
                        auto oiTmp = Add(q3, q2); // (nTileCur, dN), (nTileCur, dN) -> (nTileCur, dN)
                        IF (IsLoopEnd(bn, bnPerBatch)) {
                            oiUpdate = Div(oiTmp, liNew); // (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
                            DAssemble(oiUpdate, oiOffset, attentionOut);
                        } ELSE {
                            oiUpdate = oiTmp;
                        }
                        liUpdate = liNew;
                        miUpdate = miNew;
                    }
                }
            }
        }
 
        SymbolicScalar B = attentionOut->shape[0] / N; // S=1
        const int bTile = 32;
        LOOP("PaPost", FunctionType::DYNAMIC_LOOP, papostiter, LoopRange(0, B / bTile, 1), {}, true) {
                auto postInUnit = DView(attentionOut, {bTile * S * N, kvLoraRank}, {papostiter * bTile * S * N, 0});
                Program::GetInstance().GetTileShape().SetVecTileShapes({std::min(64, bTile*S*N), kvLoraRank});// raw (8*1*128, 512)
 
                // 使用AddS看能否进行LooP间数据传递
                auto t1Res = AddS(postInUnit, Element(DataType::DT_FP32, F_0));
 
                std::vector<SymbolicScalar> dynOffset = {papostiter * bTile * S * N, 0};
                DAssemble(t1Res, dynOffset, postOut);
        }
    }
}
 
void PageAttentionAddSSingleOutput(Tensor &qNope, Tensor &kNopeCache, Tensor &vNopeCache, Tensor &qRope, Tensor &kRopeCache,
        Tensor &blockTable, Tensor &actSeqs, int blockSize, float softmaxScale, Tensor &attentionOut, Tensor &postOut,
    PaTileShapeConfig &tileConfig, int maxUnrollTimes) {
    auto dtype = qNope->Datatype();
    // 入参B*S*N合轴
    int dN = qNope->shape[1];
    int dR = qRope->shape[1];
 
    int nTile = tileConfig.headNumQTile;
    auto c1Tile = tileConfig.c1TileShape;
    auto v1Tile = tileConfig.v1TileShape;
    auto c2Tile = tileConfig.c2TileShape;
    auto v2Tile = tileConfig.v2TileShape;
 
    int batchSize = blockTable->shape[0];
    int nQ = qNope->shape[0] / batchSize; // B*1*N
 
    auto N = 128;
    auto kvLoraRank = 512;
    int S = 1;
 
    FUNCTION("main", FunctionType::DYNAMIC,
        {qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs}, {postOut}) {
        SymbolicScalar nLoop = nQ / nTile;
 
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, batchSize, 1)) {
            SymbolicScalar curSeq = GetInputDataInt32Dim1(actSeqs, bIdx);
            SymbolicScalar bnPerBatch = curSeq / blockSize; // 暂时仅考虑curSeq是blockSize对齐
            bnPerBatch.AsIntermediateVariable();
            LOOP("LOOP_L1_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, nLoop, 1)) {
                int curNTile = nTile;
                Tensor oiUpdate(DT_FP32, {nTile, dN}, "oiUpdate");
                Tensor liUpdate(DT_FP32, {nTile, 1}, "liUpdate");
                Tensor miUpdate(DT_FP32, {nTile, 1}, "miUpdate");
                // 当前curOffset没放到更内层循环，避免重复bnPerBatch次的DAssemble操作
                SymbolicScalar curOffset = bIdx * nQ + nIdx * nTile;
                std::vector<SymbolicScalar> oiOffset = {curOffset, 0}; // (B*N*S, d)
 
                LOOP("LOOP_L2_bn", FunctionType::DYNAMIC_LOOP, bn, LoopRange(0, bnPerBatch, 1), PowersOf2(maxUnrollTimes)) {
                    // 当前qn，qr和qi放入内层Loop，避免Concat单独切成一个小图
                    int curS2Tile = blockSize;
                    auto qn = DView(qNope, {curNTile, dN}, {curOffset, 0});
                    auto qr = DView(qRope, {curNTile, dR}, {curOffset, 0});
                    Tensor qi(dtype, {curNTile, dN + dR}, "qi");
                    DAssemble(qn, {0, 0}, qi);
                    DAssemble(qr, {0, dN}, qi);
 
                    SymbolicScalar curBlockIdx = GetInputDataInt32Dim2(blockTable, bIdx, bn);
                    curBlockIdx.AsIntermediateVariable();
                    auto kn = DViewPad(kNopeCache, {curS2Tile, dN}, {std::min(curSeq - bn * blockSize, blockSize), dN},
                                                  {curBlockIdx * blockSize, 0});
                    auto kr = DViewPad(kRopeCache, {curS2Tile, dR}, {std::min(curSeq - bn * blockSize, blockSize), dR},
                                                  {curBlockIdx * blockSize, 0});
                    Tensor kj(dtype, {curS2Tile, dN + dR}, "kj");
                    DAssemble(kn, {0, 0}, kj);
                    DAssemble(kr, {0, dN}, kj);
                    auto vj = DViewPad(vNopeCache, {curS2Tile, dN}, {std::min(curSeq - bn * blockSize, blockSize), dN},
                                                  {curBlockIdx * blockSize, 0});
 
                    ConfigManager::Instance().SetSemanticLabel("MatMul");
                    Program::GetInstance().GetTileShape().SetCubeTileShapes(
                        {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]});
                    auto sij = Matrix::Matmul<false, true>(DataType::DT_FP32, qi, kj); // (curNTile, dN+dR), (curS2Tile, dN+dR) -> (curNTile, curS2Tile)
                    Program::GetInstance().GetTileShape().SetVecTileShapes(v1Tile[0], v1Tile[1]);
 
                    ConfigManager::Instance().SetSemanticLabel("SoftMax");
                    auto sijScale = MulS(sij, Element(DataType::DT_FP32, softmaxScale)); // (curNTile, curS2Tile)
 
                    auto tildaMij = RowMaxSingle(sijScale); // (curNTile, curS2Tile) -> (curNTile, 1)
                    auto tsub =
                        Sub(sijScale, tildaMij); // (curNTile, curS2Tile) - (curNTile, 1) -> (curNTile, curS2Tile)
                    auto tildaPij = Exp(tsub);
                    auto tildaPijF16 = Cast(tildaPij, dtype);
                    auto tildaLij = RowSumSingle(tildaPij); // (nTileCur, s2TileCur) -> (nTileCur, 1)
 
                    IF (IsLoopBegin(bn, 0)) {
                        Program::GetInstance().GetTileShape().SetCubeTileShapes(
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]});
                        ConfigManager::Instance().SetSemanticLabel("b1-matmul2");
                        auto oiTmp = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16, vj);; // (curNTile, curS2Tile), (curS2Tile, dN) -> (curNTile, dN)
                        Program::GetInstance().GetTileShape().SetVecTileShapes(v2Tile[0], v2Tile[1]);
                        ConfigManager::Instance().SetSemanticLabel("b1-after-matmul2");
                        IF (IsLoopEnd(bn, bnPerBatch)) {
                            oiUpdate = Div(oiTmp, tildaLij); // (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
                            DAssemble(oiUpdate, oiOffset, attentionOut);
                        } ELSE {
                            oiUpdate = oiTmp;
                        }
                        liUpdate = tildaLij;
                        miUpdate = tildaMij;
                    } ELSE {
                        auto oi = oiUpdate;
                        auto li = liUpdate;
                        auto mi = miUpdate;
 
                        ConfigManager::Instance().SetSemanticLabel("Softmax-acc");
                        auto miNew = Maximum(mi, tildaMij); // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                        auto t1 = Sub(mi, miNew);           // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                        auto t2 = Exp(t1);
                        auto t3 = Sub(tildaMij, miNew); // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                        auto t4 = Exp(t3);
                        auto t5 = Mul(t4, tildaLij); // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                        auto t6 = Mul(t2, li);       // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
                        auto liNew = Add(t6, t5);    // (curNTile, 1), (curNTile, 1) -> (curNTile, 1)
 
                        auto q3 = Mul(oi, t2); // (curNTile, dN), (curNTile, 1) -> (curNTile, dN)
                        Program::GetInstance().GetTileShape().SetCubeTileShapes(
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]});
                        ConfigManager::Instance().SetSemanticLabel("bn-matmul2");
                        auto q1 = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16, vj); // (curNTile, curS2Tile), (curS2Tile, dN) -> (curNTile, dN)
                        Program::GetInstance().GetTileShape().SetVecTileShapes(v2Tile[0], v2Tile[1]);
                        ConfigManager::Instance().SetSemanticLabel("bn-after-matmul2");
                        auto q2 = Mul(q1, t4);    // (nTileCur, dN), (nTileCur, 1) -> (nTileCur, dN)
                        auto oiTmp = Add(q3, q2); // (nTileCur, dN), (nTileCur, dN) -> (nTileCur, dN)
                        IF (IsLoopEnd(bn, bnPerBatch)) {
                            oiUpdate = Div(oiTmp, liNew); // (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
                            DAssemble(oiUpdate, oiOffset, attentionOut);
                        } ELSE {
                            oiUpdate = oiTmp;
                        }
                        liUpdate = liNew;
                        miUpdate = miNew;
                    }
                }
            }
        }
 
        SymbolicScalar B = attentionOut->shape[0] / N; // S=1
        const int bTile = 32;
        LOOP("PaPost", FunctionType::DYNAMIC_LOOP, papostiter, LoopRange(0, B / bTile, 1), {}, true) {
                auto postInUnit = DView(attentionOut, {bTile * S * N, kvLoraRank}, {papostiter * bTile * S * N, 0});
                Program::GetInstance().GetTileShape().SetVecTileShapes({std::min(64, bTile*S*N), kvLoraRank});// raw (8*1*128, 512)
 
                // 使用AddS看能否进行LooP间数据传递
                auto t1Res = AddS(postInUnit, Element(DataType::DT_FP32, F_0));
 
                std::vector<SymbolicScalar> dynOffset = {papostiter * bTile * S * N, 0};
                DAssemble(t1Res, dynOffset, postOut);
        }
    }
}

} // namespace npu::tile_fwk
