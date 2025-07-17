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
 * \file selected_attention.cpp
 * \brief
 */

#include "interface/operation/operation_impl.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "tilefwk/tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk.h"
#include "interface/program/program.h"
#include "interface/tensor/tensormap.h"
#include "interface/configs/config_manager.h"
#include "interface/configs/config.h"
#include "interface/utils/common.h"
#include "interface/utils/id_gen.h"
#include "interface/utils/log.h"
#include "selected_attention.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {
/**
 * normal attention: q=qNope+qRope, kv是连续的
 * input:
    * qNope: [b*s1*n2*g, k_dim] fp16/bf16
    * qRope: [b*s1*n2*g, rope_dim] fp16/bf16
    * kSlc: [b*s1*n2*s2, k_dim+rope_dim], nope与rope在gen_kv_slc中已经合并起来了 fp16/bf16
    * vSlc: [b*s1*n2*s2, v_dim] fp16/bf16
    * kvSlcActSeqs: [b] int32
 * output:
    * attentionOut: [b*s1*n2*g, v_dim] fp32
*/
void SlcAttn(Tensor &qNope, Tensor &qRope, Tensor &kSlc, Tensor &vSlc, Tensor &kvSlcActSeqs, int nQ, int nKv,
    float softmaxScale, Tensor &attentionOut, SaTileShapeConfig tileConfig) {
    auto dtype = qNope->Datatype();
    int dN = qNope->shape[1];
    int dR = qRope->shape[1];
    int group = nQ / nKv;

    int gTile = tileConfig.gTile;
    int s2Tile = tileConfig.sKvTile;
    auto c1Tile = tileConfig.c1TileShape;
    auto v1Tile = tileConfig.v1TileShape;
    auto c2Tile = tileConfig.c2TileShape;
    auto v2Tile = tileConfig.v2TileShape;

    FUNCTION("SA_MAIN", FunctionType::DYNAMIC, {qNope, qRope, kSlc, vSlc, kvSlcActSeqs}, {attentionOut}) {
        /******** tune params ********/
        // Program::GetInstance().GetConfig().SetCubeNBufferMap({});
        // Program::GetInstance().GetConfig().SetL1Reuse(0);
        // Program::GetInstance().GetConfig().SetCopyInThreshold(1 * NUM_1024 * NUM_1024);
        // Program::GetInstance().GetConfig().SetCycleUpperBound(NUM_100000);
        // Program::GetInstance().GetConfig().SetParallelThreshold(NUM_2);
        // Program::GetInstance().GetConfig().SetCubeNBuffer(NUM_2);
        // config::SetOperationConfig("FORCE_COMBINE_AXIS", true);

        /******** attention ********/
        SymbolicScalar batchSizeSym = GetInputShapeDim(kvSlcActSeqs, 0); // b
        SymbolicScalar s1N2GSym = GetInputShapeDim(qNope, 0) / batchSizeSym; // s1n2
        SymbolicScalar s1Sym = s1N2GSym / nQ; // s1
        SymbolicScalar gLoopSym = group / gTile;

        SymbolicScalar s1N2S2Sym = GetInputShapeDim(kSlc, 0) / batchSizeSym; // s1n2s2
        SymbolicScalar n2S2Sym = s1N2S2Sym / s1Sym; // n2s2
        SymbolicScalar n2Sym = nKv;

        LOOP("SA_LOOP_L0_b", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, batchSizeSym, 1)) {
            SymbolicScalar curKvSlcSeq = GetInputDataInt32Dim1(kvSlcActSeqs, bIdx);

            LOOP("SA_LOOP_L1_s1", FunctionType::DYNAMIC_LOOP, s1Idx, LoopRange(0, s1Sym, 1)) { // 每一个S1的k_slc/v_slc都不同
                SymbolicScalar curSeq = std::max(curKvSlcSeq - s1Sym + 1 + s1Idx, 0); // for MTP s1!= 1 casual计算
                curSeq.AsIntermediateVariable();
                SymbolicScalar bnPerBatch = (curSeq + s2Tile - 1) / s2Tile;

                LOOP("SA_LOOP_L2_n2", FunctionType::DYNAMIC_LOOP, n2Idx, LoopRange(0, n2Sym, 1)) { // GQA场景
                    LOOP("SA_LOOP_L3_g", FunctionType::DYNAMIC_LOOP, gIdx, LoopRange(0, gLoopSym, 1)) {
                        int curGTile = gTile;
                        Tensor oiUpdate(DT_FP32, {curGTile, dN}, "oiUpdate");
                        Tensor liUpdate(DT_FP32, {curGTile, 1}, "liUpdate");
                        Tensor miUpdate(DT_FP32, {curGTile, 1}, "miUpdate");

                        SymbolicScalar curOffset = bIdx * s1N2GSym + s1Idx * nQ + n2Idx * group + gIdx * curGTile;
                        std::vector<SymbolicScalar> oiOffset = {curOffset, 0}; // 按最终结果(B*S1*N1,D)进行assemble

                        LOOP("SA_LOOP_L4_s2", FunctionType::DYNAMIC_LOOP, s2Idx, LoopRange(0, bnPerBatch, 1), PowersOf2(1)) {
                            int curS2Tile = s2Tile;
                            SymbolicScalar curKvOffset = bIdx * s1N2S2Sym + s1Idx * n2S2Sym + s2Idx * curS2Tile;

                            ConfigManager::Instance().SetSemanticLabel("Sa");
                            auto qn = DView(qNope, {curGTile, dN}, {curOffset, 0});
                            auto qr = DView(qRope, {curGTile, dR}, {curOffset, 0});
                            Tensor qi(dtype, {curGTile, dN + dR}, "qi");
                            DAssemble(qn, {0, 0}, qi);
                            DAssemble(qr, {0, dN}, qi);

                            auto kj = DViewPad(kSlc, {curS2Tile, dN + dR}, {std::min(curSeq - s2Idx * curS2Tile, curS2Tile), dN + dR},
                                            {curKvOffset, 0}); // kSlc已经合并了rope和nope
                            auto vj = DViewPad(vSlc, {curS2Tile, dN}, {std::min(curSeq - s2Idx * curS2Tile, curS2Tile), dN},
                                            {curKvOffset, 0});

                            // C1
                            Program::GetInstance().GetTileShape().SetCubeTileShapes(
                                {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]}, true);
                            ConfigManager::Instance().SetSemanticLabel("Sa_QkMM");
                            Program::GetInstance().GetMatrixSize().SetMatrixSize({qi.GetShape()[0], 0, kj.GetShape()[0]});
                            auto sij = Matrix::Matmul<false, true>(DataType::DT_FP32, qi, kj); // (curNTile, dN+dR), (curS2Tile, dN+dR) -> (curNTile, curS2Tile)

                            // V1
                            ConfigManager::Instance().SetSemanticLabel("Sa_Qkvec1");
                            Program::GetInstance().GetTileShape().SetVecTileShapes(v1Tile[0], v1Tile[1]);
                            auto sijScale = MulS(sij, Element(sij->Datatype(), softmaxScale)); // (curNTile, curS2Tile)
                            auto tildaMij = RowMaxSingle(sijScale); // (curNTile, curS2Tile) -> (curNTile, 1)
                            auto tsub = Sub(sijScale, tildaMij);
                            auto tildaPij = Exp(tsub);
                            auto tildaPijF16 = Cast(tildaPij, dtype);
                            auto tildaLij = RowSumSingle(tildaPij); // (nTileCur, s2TileCur) -> (nTileCur, 1)

                            IF (IsLoopBegin(s2Idx, 0)) {
                                // C2
                                Program::GetInstance().GetTileShape().SetCubeTileShapes(
                                    {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]}, true);
                                ConfigManager::Instance().SetSemanticLabel("Sa_KvMm");
                                Program::GetInstance().GetMatrixSize().SetMatrixSize(
                                    {tildaPijF16.GetShape()[0], tildaPijF16.GetShape()[1], vj.GetShape()[1]});
                                auto oiTmp = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16, vj);
                                Program::GetInstance().GetTileShape().SetVecTileShapes(v2Tile[0], v2Tile[1]);
                                IF (IsLoopEnd(s2Idx, bnPerBatch)) {
                                    // V2
                                    ConfigManager::Instance().SetSemanticLabel("Sa_KvVec2");
                                    oiUpdate = Div(oiTmp, tildaLij); // (nTileCur, dN) / (nTileCur, 1) -> (nTileCur, dN)
                                    DAssemble(oiUpdate, oiOffset, attentionOut);
                                } ELSE {
                                    oiUpdate = oiTmp;
                                }
                                liUpdate = tildaLij;
                                miUpdate = tildaMij;
                            } ELSE {
                                ConfigManager::Instance().SetSemanticLabel("Sa_UpdateVec2");
                                auto oi = oiUpdate;
                                auto li = liUpdate;
                                auto mi = miUpdate;

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
                                    {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]}, true);
                                ConfigManager::Instance().SetSemanticLabel("Sa_UpdateMM2");
                                Program::GetInstance().GetMatrixSize().SetMatrixSize(
                                    {tildaPijF16.GetShape()[0], tildaPijF16.GetShape()[1], vj.GetShape()[1]});
                                auto q1 = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16, vj);
                                Program::GetInstance().GetTileShape().SetVecTileShapes(v2Tile[0], v2Tile[1]);
                                auto q2 = Mul(q1, t4);    // (nTileCur, dN), (nTileCur, 1) -> (nTileCur, dN)
                                auto oiTmp = Add(q3, q2); // (nTileCur, dN), (nTileCur, dN) -> (nTileCur, dN)
                                IF (IsLoopEnd(s2Idx, bnPerBatch)) {
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
            }
        }
    }
}
} // namespace npu::tile_fwk
