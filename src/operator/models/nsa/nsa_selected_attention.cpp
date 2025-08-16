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
 * \file nsa_selected_attention.cpp
 * \brief
 */

#include "interface/operation/operation_impl.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "tilefwk/tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/tensormap.h"
#include "interface/configs/config_manager.h"
#include "interface/configs/config_storage.h"
#include "interface/utils/common.h"
#include "interface/utils/id_gen.h"
#include "interface/utils/log.h"
#include "nsa_selected_attention.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {

void SelectedAttentionCompute(Tensor &topKIndcies, Tensor &kvNopeCache, Tensor &kRopeCache, Tensor &kvActSeqs, Tensor &blockTable,
    const Tensor &qNope, const Tensor &qRope, Tensor &attentionOut,
    int nQ, int nKv, float softmaxScale, int front, int near, int topk, int blockSize, int cmpBlockSize, int slcBlockSize,
    SATileShapeConfig saTileConfig) {
    auto dtype = qNope->Datatype();
    int dN = qNope->shape[1];
    int dR = qRope->shape[1];
    int group = nQ / nKv;

    auto v0Tile = saTileConfig.kvSlcV0TileShape;
    int gTile = saTileConfig.gTile;
    int curS2Tile = saTileConfig.sKvTile;
    auto c1Tile = saTileConfig.c1TileShape;
    auto v1Tile = saTileConfig.v1TileShape;
    auto c2Tile = saTileConfig.c2TileShape;
    auto v2Tile = saTileConfig.v2TileShape;

    /******** tune params ********/
    // Program::GetInstance().GetConfig().Set<std::map<int, int>>(CUBE_NBUFFER_MAP, {});
    // Program::GetInstance().GetConfig().Set<int>(L1_REUSE, 0);
    // Program::GetInstance().GetConfig().Set<int>(COPYIN_THRESHOLD, 1 * 1024 * 1024);
    // Program::GetInstance().GetConfig().Set<int>(CYCLE_UPPER_BOUND, 100000);
    // Program::GetInstance().GetConfig().Set<int>(PARALLEL_THRESHOLD, 2);
    // Program::GetInstance().GetConfig().Set<int>(CUBE_NBUFFER, 2);
    // config::SetOperationConfig("FORCE_COMBINE_AXIS", true);

    SymbolicScalar batchSizeSym = topKIndcies->shape[0]; // b
    SymbolicScalar s1N2GSym = qNope->shape[0] / batchSizeSym; // s1n2
    SymbolicScalar s1Sym = s1N2GSym / nQ; // s1
    SymbolicScalar gLoopSym = group / gTile;
    SymbolicScalar n2Sym = nKv;

    LOOP("LOOP_L0_b_SA", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, batchSizeSym, 1), {}, true) {
        SymbolicScalar curActSeq = GetInputDataInt32Dim1(kvActSeqs, bIdx);
        curActSeq.AsIntermediateVariable();
        LOOP("LOOP_L1_s1_SA", FunctionType::DYNAMIC_LOOP, s1Idx, LoopRange(0, s1Sym, 1)) {
            LOOP("LOOP_L2_n2_SA", FunctionType::DYNAMIC_LOOP, n2Idx, LoopRange(0, n2Sym, 1)) { // GQA场景
                Tensor kSlc(dtype, {topk * slcBlockSize, dN + dR}, "kSlc");
                Tensor vSlc(dtype, {topk * slcBlockSize, dN + dR}, "vSlc");

                SymbolicScalar curKvSlcSeq = 0;
                config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, false);
                SymbolicScalar sSlc = (curActSeq - s1Sym + 1 + s1Idx - cmpBlockSize + slcBlockSize) / slcBlockSize;
                LOOP("LOOP_L3_kv_slc_SA", FunctionType::DYNAMIC_LOOP, kvSlcIdx, LoopRange(0, 1, 1), {}, true) { // kv_slc
                    (void)kvSlcIdx;
                    SymbolicScalar positions = 0;
                    SymbolicScalar slcSeqLen = 0;
                    for (int topKIdx = 0; topKIdx < topk; topKIdx++) {
                        if (topKIdx < front) {
                            // 获取到topk的position
                            // 头部的front个
                            positions = topKIdx * slcBlockSize;
                        } else if (topKIdx > (topk - near - front)) {
                            // 尾部的near个
                            positions = (sSlc - near + (topKIdx - (topk - front - near)) - 1) * slcBlockSize;
                        } else {
                            // 中间的topk-front-near个
                            SymbolicScalar topkIndex = GetInputDataInt32Dim3(topKIndcies, bIdx, s1Idx, topKIdx - front);
                            positions = topkIndex * slcBlockSize;
                        }
                        curKvSlcSeq = curKvSlcSeq + std::min(slcBlockSize, curActSeq - positions);
                        SymbolicScalar blockIdxInBatch = positions / blockSize;
                        SymbolicScalar tail = positions % blockSize;
                        SymbolicScalar slcBlockIdx = GetInputDataInt32Dim2(blockTable, bIdx, blockIdxInBatch);
                        Program::GetInstance().GetTileShape().SetVecTileShapes(v0Tile[0], v0Tile[1]);
                        auto kvSlcBlock = DView(kvNopeCache, {slcBlockSize, dN}, {slcBlockIdx * blockSize + tail, n2Idx * dN});
                        auto krSlcBlock = DView(kRopeCache, {slcBlockSize, dR}, {slcBlockIdx * blockSize + tail, n2Idx * dR});
                        Program::GetInstance().GetTileShape().SetVecTileShapes(v0Tile[0], v0Tile[1]);
                        auto kvSlcBlock_fp32 = Cast(kvSlcBlock, DataType::DT_FP32);
                        auto krSlcBlock_fp32 = Cast(krSlcBlock, DataType::DT_FP32);
                        Program::GetInstance().GetTileShape().SetVecTileShapes(v0Tile[0], v0Tile[1]);
                        auto kvSlcBlock_fp16 = Cast(kvSlcBlock_fp32, kSlc->Datatype());
                        auto krSlcBlock_fp16 = Cast(krSlcBlock_fp32, kSlc->Datatype());
                        Program::GetInstance().GetTileShape().SetVecTileShapes(v0Tile[0], v0Tile[1]);

                        SymbolicScalar slcOutSOffset = topKIdx * slcBlockSize; // 需要调整kv_slc拼接时，near和topk的顺序
                        DAssemble(kvSlcBlock_fp16, {slcOutSOffset, 0}, kSlc);
                        DAssemble(krSlcBlock_fp16, {slcOutSOffset, dN}, kSlc);
                        DAssemble(kvSlcBlock_fp16, {slcOutSOffset, 0}, vSlc);
                    }
                }

                config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true); // 参数化，使能动态尾块特性
                LOOP("LOOP_L3_g_SA", FunctionType::DYNAMIC_LOOP, gIdx, LoopRange(0, gLoopSym, 1), {}, true) { // slc_attn
                    int curGTile = gTile;
                    Tensor oiUpdate(DT_FP32, {curGTile, dN}, "oiUpdate");
                    Tensor liUpdate(DT_FP32, {curGTile, 1}, "liUpdate");
                    Tensor miUpdate(DT_FP32, {curGTile, 1}, "miUpdate");

                    SymbolicScalar curOffset = bIdx * s1N2GSym + s1Idx * nQ + n2Idx * group + gIdx * curGTile;
                    std::vector<SymbolicScalar> oiOffset = {bIdx, s1Idx, n2Idx * group + gIdx * curGTile, 0}; // 按最终结果(B,S1,N1,D)进行assemble

                    SymbolicScalar curSeq = std::max(curKvSlcSeq - s1Sym + 1 + s1Idx, 0); // for MTP s1!= 1 casual计算
                    curSeq.AsIntermediateVariable();
                    SymbolicScalar bnPerBatch = (curSeq + curS2Tile - 1) / curS2Tile;
                    LOOP("LOOP_L4_s2_SA", FunctionType::DYNAMIC_LOOP, s2Idx, LoopRange(0, bnPerBatch, 1), PowersOf2(1)) {
                        ConfigManager::Instance().SetSemanticLabel("Sa");
                        // DView, 临时规避改成 DViewPad
                        auto qn = DViewPad(qNope, {curGTile, dN}, {curGTile, dN}, {curOffset, 0});
                        auto qr = DViewPad(qRope, {curGTile, dR}, {curGTile, dR}, {curOffset, 0});
                        Tensor qi(dtype, {curGTile, dN + dR}, "qi");
                        DAssemble(qn, {0, 0}, qi);
                        DAssemble(qr, {0, dN}, qi);

                        auto kj = DViewPad(kSlc, {curS2Tile, dN + dR}, {std::min(curSeq - s2Idx * curS2Tile, curS2Tile), dN + dR},
                                        {s2Idx * curS2Tile, 0}); // kSlc已经合并了rope和nope
                        auto vj = DViewPad(vSlc, {curS2Tile, dN}, {std::min(curSeq - s2Idx * curS2Tile, curS2Tile), dN},
                                        {s2Idx * curS2Tile, 0});

                        // C1
                        Program::GetInstance().GetTileShape().SetCubeTileShapes(
                            {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]}, true);
                        ConfigManager::Instance().SetSemanticLabel("Sa_QkMM");
                        Program::GetInstance().GetMatrixSize().SetMatrixSize({qi.GetShape()[0], 0, kj.GetShape()[0]});
                        auto sij = Matrix::Matmul<false, true>(DataType::DT_FP32, qi, kj);

                        // V1
                        ConfigManager::Instance().SetSemanticLabel("Sa_Qkvec1");
                        Program::GetInstance().GetTileShape().SetVecTileShapes(v1Tile[0], v1Tile[1]);
                        auto sijScale = MulS(sij, Element(sij->Datatype(), softmaxScale));
                        auto tildaMij = RowMaxSingle(sijScale); // (curGTile, curS2Tile) -> (curGTile, 1)
                        auto tsub = Sub(sijScale, tildaMij); // (curGTile, curS2Tile), (curGTile, 1) -> (curGTile, curS2Tile)
                        auto tildaPij = Exp(tsub);  // (curGTile, curS2Tile) -> (curGTile, curS2Tile)
                        auto tildaPijF16 = Cast(tildaPij, dtype);
                        auto tildaLij = RowSumSingle(tildaPij);

                        IF (IsLoopBegin(s2Idx, 0)) {
                            // C2
                            Program::GetInstance().GetTileShape().SetCubeTileShapes(
                                {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]}, true);
                            ConfigManager::Instance().SetSemanticLabel("Sa_KvMm");
                            Program::GetInstance().GetMatrixSize().SetMatrixSize(
                                {tildaPijF16.GetShape()[0], tildaPijF16.GetShape()[1], vj.GetShape()[1]});
                            auto oiTmp = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16, vj);
                            Program::GetInstance().GetTileShape().SetVecTileShapes(v2Tile[0], v2Tile[1]);
                            IF (IsLoopEnd(s2Idx, bnPerBatch)) { // PATH3
                                // V2
                                ConfigManager::Instance().SetSemanticLabel("Sa_KvVec2");
                                oiUpdate = Div(oiTmp, tildaLij);
                                Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, v2Tile[0], v2Tile[1]);
                                auto oiUpdate4Dim = AddS(Reshape(oiUpdate, {1, 1, curGTile, dN}), Element(oiUpdate->Datatype(), float(0)));
                                DAssemble(oiUpdate4Dim, oiOffset, attentionOut);
                            } ELSE { // PATH2
                                oiUpdate = oiTmp;
                            }
                            liUpdate = tildaLij;
                            miUpdate = tildaMij;
                        } ELSE {
                            ConfigManager::Instance().SetSemanticLabel("Sa_UpdateVec2");
                            auto oi = oiUpdate;
                            auto li = liUpdate;
                            auto mi = miUpdate;

                            auto miNew = Maximum(mi, tildaMij);
                            auto t1 = Sub(mi, miNew);
                            auto t2 = Exp(t1);
                            auto t3 = Sub(tildaMij, miNew);
                            auto t4 = Exp(t3);
                            auto t5 = Mul(t4, tildaLij);
                            auto t6 = Mul(t2, li);
                            auto liNew = Add(t6, t5);

                            auto q3 = Mul(oi, t2);
                            Program::GetInstance().GetTileShape().SetCubeTileShapes(
                                {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]}, true);
                            ConfigManager::Instance().SetSemanticLabel("Sa_UpdateMM2");
                            Program::GetInstance().GetMatrixSize().SetMatrixSize(
                                {tildaPijF16.GetShape()[0], tildaPijF16.GetShape()[1], vj.GetShape()[1]});
                            auto q1 = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16, vj);
                            Program::GetInstance().GetTileShape().SetVecTileShapes(v2Tile[0], v2Tile[1]);
                            auto q2 = Mul(q1, t4);
                            auto oiTmp = Add(q3, q2);
                            IF (IsLoopEnd(s2Idx, bnPerBatch)) { // PATH1
                                oiUpdate = Div(oiTmp, liNew);
                                Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, v2Tile[0], v2Tile[1]);
                                auto oiUpdate4Dim = AddS(Reshape(oiUpdate, {1, 1, curGTile, dN}), Element(oiUpdate->Datatype(), float(0)));
                                DAssemble(oiUpdate4Dim, oiOffset, attentionOut);
                            } ELSE { // PATH0
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

void SelectedAttention(Tensor &topKIndcies, Tensor &kvNopeCache, Tensor &kRopeCache, Tensor &kvActSeqs, Tensor &blockTable,
    const Tensor &qNope, const Tensor &qRope, Tensor &attentionOut,
    int nQ, int nKv, float softmaxScale, int front, int near, int topk, int blockSize, int cmpBlockSize, int slcBlockSize,
    SATileShapeConfig saTileConfig) {
    FUNCTION("SA_MAIN", FunctionType::DYNAMIC, 
        {topKIndcies, kvNopeCache, kRopeCache, kvActSeqs, blockTable, qNope, qRope},
        {attentionOut}) {
        SelectedAttentionCompute(topKIndcies, kvNopeCache, kRopeCache, kvActSeqs, blockTable,
            qNope, qRope, attentionOut,
            nQ, nKv, softmaxScale, front, near, topk, blockSize, cmpBlockSize, slcBlockSize, saTileConfig);
    }
}

} // namespace npu::tile_fwk
