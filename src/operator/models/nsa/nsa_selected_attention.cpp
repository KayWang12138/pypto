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
/**
 * normal attention: q=qNope+qRope, kv是连续的
 * input:
    * topKIndcies: [b, s1, topk]
    * kvNopeCache: [blockNum * blockSize, n2 * v_dim]
    * kRopeCache: [blockNum * blockSize, n2 * rope_dim]
    * kvActSeqs: [b]
    * blockTableL {b, maxBlockNumPerBatch}
    * qNope: [b*s1*n2*g, k_dim] fp16/bf16
    * qRope: [b*s1*n2*g, rope_dim] fp16/bf16
 * output:
    * attentionOut: [b, s1, n2, g, v_dim] fp32

 * middle tensor:
    * kSlc: [b*s1*n2*s2, k_dim + rope_dim], nope与rope在gen_kv_slc中已经合并起来了 fp16/bf16
    * vSlc: [b*s1*n2*s2, v_dim] fp16/bf16
    * kvSlcActSeqs: [b, s1] int32
*/
void SelectedAttentionCompute(Tensor &topKIndcies, Tensor &kvNopeCache, Tensor &kRopeCache, Tensor &kvActSeqs, Tensor &blockTable,
    const Tensor &qNope, const Tensor &qRope, Tensor &attentionOut,
    int nQ, int nKv, float softmaxScale, int front, int near, int topk, int blockSize, int cmpBlockSize, int slcBlockSize,
    SATileShapeConfig saTileConfig, bool debug) {
    auto dtype = qNope->Datatype();
    int dN = qNope->shape[1];
    int dR = qRope->shape[1];
    int group = nQ / nKv;

    auto v0Tile = saTileConfig.kvSlcV0TileShape;
    int gTile = saTileConfig.gTile;
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

    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);

    LOOP("LOOP_L0_b_SA", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, batchSizeSym, 1), {}, true) {
        SymbolicScalar curActSeq = GetInputDataInt32Dim1(kvActSeqs, bIdx);
        curActSeq.AsIntermediateVariable();
        LOOP("LOOP_L1_s1_SA", FunctionType::DYNAMIC_LOOP, s1Idx, LoopRange(0, s1Sym, 1)) {
            LOOP("LOOP_L2_n2_SA", FunctionType::DYNAMIC_LOOP, n2Idx, LoopRange(0, n2Sym, 1)) { // GQA场景
                LOOP("LOOP_L3_g_SA", FunctionType::DYNAMIC_LOOP, gIdx, LoopRange(0, gLoopSym, 1)) { // slc_attn
                    int curGTile = gTile;
                    SymbolicScalar curOffset = bIdx * s1N2GSym + s1Idx * nQ + n2Idx * group + gIdx * curGTile;
                    std::vector<SymbolicScalar> oiOffset = {bIdx, s1Idx, n2Idx * group + gIdx * curGTile, 0}; // 按最终结果(B,S1,N1,D)进行assemble

                    LOOP("LOOP_L4_s2_SA", FunctionType::DYNAMIC_LOOP, s2Idx, LoopRange(0, 1, 1), PowersOf2(1)) { // 非Flash
                        int curS2Tile = topk * slcBlockSize;
                        // kv_slc
                        ConfigManager::Instance().SetSemanticLabel("kv_slc");
                        Tensor kSlc(dtype, {topk * slcBlockSize, dN + dR}, "kSlc");
                        SymbolicScalar curKvSlcSeq = 0;
                        SymbolicScalar sSlc = (curActSeq - s1Sym + 1 + s1Idx - cmpBlockSize + slcBlockSize) / slcBlockSize;
                        sSlc.AsIntermediateVariable();
                        SymbolicScalar positions = 0;
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
                                SymbolicScalar topkIndex;
                                if (debug) {
                                    TileShape::Current().SetVecTile(1, 1, NUM16);
                                    topkIndex = GetTensorDataInt32(topKIndcies, bIdx, s1Idx, topKIdx - front);
                                } else {
                                    topkIndex = GetInputDataInt32Dim3(topKIndcies, bIdx, s1Idx, topKIdx - front);
                                }

                                positions = topkIndex * slcBlockSize;
                            }
                            curKvSlcSeq = curKvSlcSeq + std::min(slcBlockSize, curActSeq - positions);
                            SymbolicScalar blockIdxInBatch = positions / blockSize;
                            SymbolicScalar tail = positions % blockSize;
                            SymbolicScalar slcBlockIdx = GetInputDataInt32Dim2(blockTable, bIdx, blockIdxInBatch);
                            TileShape::Current().SetVecTile(v0Tile[0], v0Tile[1]);
                            auto kvSlcBlock = View(kvNopeCache, {slcBlockSize, dN}, {slcBlockIdx * blockSize + tail, n2Idx * dN});
                            auto krSlcBlock = View(kRopeCache, {slcBlockSize, dR}, {slcBlockIdx * blockSize + tail, n2Idx * dR});

                            ConfigManager::Instance().SetSemanticLabel("kv_slc_cast_fp32");
                            TileShape::Current().SetVecTile(v0Tile[0], v0Tile[1]);
                            auto kvSlcBlock_fp32 = Cast(kvSlcBlock, DataType::DT_FP32);
                            auto krSlcBlock_fp32 = Cast(krSlcBlock, DataType::DT_FP32);
                            ConfigManager::Instance().SetSemanticLabel("kv_slc_cast");
                            TileShape::Current().SetVecTile(v0Tile[0], v0Tile[1]);
                            auto kvSlcBlock_fp16 = Cast(kvSlcBlock_fp32, kSlc->Datatype());
                            auto krSlcBlock_fp16 = Cast(krSlcBlock_fp32, kSlc->Datatype());
                            TileShape::Current().SetVecTile(v0Tile[0], v0Tile[1]);

                            SymbolicScalar slcOutSOffset = topKIdx * slcBlockSize;
                            Assemble(kvSlcBlock_fp16, {slcOutSOffset, 0}, kSlc);
                            Assemble(krSlcBlock_fp16, {slcOutSOffset, dN}, kSlc);
                        }

                        // qAssemble
                        ConfigManager::Instance().SetSemanticLabel("Sa");
                        // View, 临时规避改成 View
                        auto qn = View(qNope, {curGTile, dN}, {curGTile, dN}, {curOffset, 0});
                        auto qr = View(qRope, {curGTile, dR}, {curGTile, dR}, {curOffset, 0});
                        Tensor qi(dtype, {curGTile, dN + dR}, "qi");
                        Assemble(qn, {0, 0}, qi);
                        Assemble(qr, {0, dN}, qi);

                        // slc_attn
                        SymbolicScalar curSeq = std::max(curKvSlcSeq - s1Sym + 1 + s1Idx, 0); // for MTP s1!= 1 casual计算
                        curSeq.AsIntermediateVariable();
                        auto kj = View(kSlc, {curS2Tile, dN + dR}, {std::min(curSeq - s2Idx * curS2Tile, curS2Tile), dN + dR},
                                        {s2Idx * curS2Tile, 0}); // kSlc已经合并了rope和nope
                        auto vj = View(kSlc, {curS2Tile, dN}, {std::min(curSeq - s2Idx * curS2Tile, curS2Tile), dN},
                                        {s2Idx * curS2Tile, 0});

                        // C1
                        ConfigManager::Instance().SetSemanticLabel("Sa_QkMM");
                        TileShape::Current().SetCubeTile(
                            {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]}, true);
                        TileShape::Current().SetMatrixSize({qi.GetShape()[0], 0, kj.GetShape()[0]});
                        auto sij = Matrix::Matmul<false, true>(DataType::DT_FP32, qi, kj);

                        // V1
                        ConfigManager::Instance().SetSemanticLabel("Sa_Qkvec1");
                        TileShape::Current().SetVecTile(v1Tile[0], v1Tile[1]);
                        auto sijScale = MulS(sij, Element(sij->Datatype(), softmaxScale));
                        auto tildaMij = RowMaxSingle(sijScale); // (curGTile, curS2Tile) -> (curGTile, 1)
                        auto tsub = Sub(sijScale, tildaMij); // (curGTile, curS2Tile), (curGTile, 1) -> (curGTile, curS2Tile)
                        auto tildaPij = Exp(tsub);  // (curGTile, curS2Tile) -> (curGTile, curS2Tile)
                        auto tildaLij = RowSumSingle(tildaPij);
                        auto tSoftmax = Div(tildaPij, tildaLij);
                        auto tildaPijF16 = Cast(tSoftmax, dtype);

                        // C2
                        ConfigManager::Instance().SetSemanticLabel("Sa_KvMm");
                        TileShape::Current().SetCubeTile(
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]}, true);
                        TileShape::Current().SetMatrixSize(
                            {tildaPijF16.GetShape()[0], tildaPijF16.GetShape()[1], vj.GetShape()[1]});
                        auto oi = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16, vj);

                        // V2
                        ConfigManager::Instance().SetSemanticLabel("Sa_KvVec2");
                        TileShape::Current().SetVecTile(1, 1, v2Tile[0], v2Tile[1]);
                        auto oi4Dim = AddS(Reshape(oi, {1, 1, curGTile, dN}), Element(oi->Datatype(), float(0)));
                        Assemble(oi4Dim, oiOffset, attentionOut);
                    }
                }
            }
        }
    }
}

void SelectedAttentionFlashCompute(Tensor &topKIndcies, Tensor &kvNopeCache, Tensor &kRopeCache, Tensor &kvActSeqs, Tensor &blockTable,
    const Tensor &qNope, const Tensor &qRope, Tensor &attentionOut,
    int nQ, int nKv, float softmaxScale, int front, int near, int topk, int blockSize, int cmpBlockSize, int slcBlockSize,
    SATileShapeConfig saTileConfig, bool debug) {
    auto dtype = qNope->Datatype();
    int b = topKIndcies->shape[0];
    int s1 = topKIndcies->shape[1];
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
    // Program::GetInstance().GetConfig().Set<int>(SG_CYCLE_UPPER_BOUND, 100000);
    // Program::GetInstance().GetConfig().Set<int>(SG_PARALLEL_NUM, 2);
    // Program::GetInstance().GetConfig().Set<int>(CUBE_NBUFFER, 2);
    // config::SetOperationConfig("FORCE_COMBINE_AXIS", true);

    SymbolicScalar batchSizeSym = topKIndcies->shape[0]; // b
    SymbolicScalar s1N2GSym = qNope->shape[0] / batchSizeSym; // s1n2
    SymbolicScalar s1Sym = s1N2GSym / nQ; // s1
    SymbolicScalar gLoopSym = group / gTile;
    SymbolicScalar n2Sym = nKv;

    Tensor kSlc(dtype, {b * s1 * nKv * topk * slcBlockSize, dN + dR}, "kSlc");

    SymbolicScalar s1N2S2Sym = kSlc->shape[0] / batchSizeSym; // s1n2s2
    SymbolicScalar n2S2Sym = s1N2S2Sym / s1Sym; // n2s2
    SymbolicScalar s2Sym = n2S2Sym / n2Sym; // s2

    LOOP("LOOP_L0_b_SA", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, batchSizeSym, 1), {}, true) {
        SymbolicScalar curActSeq = GetInputDataInt32Dim1(kvActSeqs, bIdx);
        curActSeq.AsIntermediateVariable();
        LOOP("LOOP_L1_s1_SA", FunctionType::DYNAMIC_LOOP, s1Idx, LoopRange(0, s1Sym, 1)) {
            LOOP("LOOP_L2_n2_SA", FunctionType::DYNAMIC_LOOP, n2Idx, LoopRange(0, n2Sym, 1)) { // GQA场景

                SymbolicScalar curKvSlcSeq = 0;
                config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, false);
                SymbolicScalar sSlc = (curActSeq - s1Sym + 1 + s1Idx - cmpBlockSize + slcBlockSize) / slcBlockSize;
                sSlc.AsIntermediateVariable();
                LOOP("LOOP_L3_kv_slc_SA", FunctionType::DYNAMIC_LOOP, kvSlcIdx, LoopRange(0, 1, 1)) { // kv_slc
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
                            SymbolicScalar topkIndex;
                            if (debug) {
                                TileShape::Current().SetVecTile(1, 1, NUM16);
                                topkIndex = GetTensorDataInt32(topKIndcies, bIdx, s1Idx, topKIdx - front);
                            } else {
                                topkIndex = GetInputDataInt32Dim3(topKIndcies, bIdx, s1Idx, topKIdx - front);
                            }

                            positions = topkIndex * slcBlockSize;
                        }
                        curKvSlcSeq = curKvSlcSeq + std::min(slcBlockSize, curActSeq - positions);
                        SymbolicScalar blockIdxInBatch = positions / blockSize;
                        SymbolicScalar tail = positions % blockSize;
                        SymbolicScalar slcBlockIdx = GetInputDataInt32Dim2(blockTable, bIdx, blockIdxInBatch);
                        TileShape::Current().SetVecTile(v0Tile[0], v0Tile[1]);
                        auto kvSlcBlock = View(kvNopeCache, {slcBlockSize, dN}, {slcBlockIdx * blockSize + tail, n2Idx * dN});
                        auto krSlcBlock = View(kRopeCache, {slcBlockSize, dR}, {slcBlockIdx * blockSize + tail, n2Idx * dR});
                        SymbolicScalar slcOutSOffset = bIdx * s1N2S2Sym + s1Idx * n2S2Sym + n2Idx * s2Sym + topKIdx * slcBlockSize;
                        Assemble(kvSlcBlock, {slcOutSOffset, 0}, kSlc);
                        Assemble(krSlcBlock, {slcOutSOffset, dN}, kSlc);
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
                    SymbolicScalar bnPerBatch = (topk * slcBlockSize + curS2Tile - 1) / curS2Tile;
                    LOOP("LOOP_L4_s2_SA", FunctionType::DYNAMIC_LOOP, s2Idx, LoopRange(0, bnPerBatch, 1), PowersOf2(1)) {
                        SymbolicScalar curKvOffset = bIdx * s1N2S2Sym + s1Idx * n2S2Sym + n2Idx * s2Sym + s2Idx * curS2Tile;

                        ConfigManager::Instance().SetSemanticLabel("Sa");
                        // View, 临时规避改成 View
                        auto qn = View(qNope, {curGTile, dN}, {curGTile, dN}, {curOffset, 0});
                        auto qr = View(qRope, {curGTile, dR}, {curGTile, dR}, {curOffset, 0});
                        Tensor qi(dtype, {curGTile, dN + dR}, "qi");
                        Assemble(qn, {0, 0}, qi);
                        Assemble(qr, {0, dN}, qi);

                        auto kj = View(kSlc, {curS2Tile, dN + dR}, {std::min(curSeq - s2Idx * curS2Tile, curS2Tile), dN + dR},
                                        {curKvOffset, 0}); // kSlc已经合并了rope和nope
                        auto vj = View(kSlc, {curS2Tile, dN}, {std::min(curSeq - s2Idx * curS2Tile, curS2Tile), dN},
                                        {curKvOffset, 0});

                        // C1
                        TileShape::Current().SetCubeTile(
                            {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]}, true);
                        ConfigManager::Instance().SetSemanticLabel("Sa_QkMM");
                        TileShape::Current().SetMatrixSize({qi.GetShape()[0], 0, kj.GetShape()[0]});
                        auto sij = Matrix::Matmul<false, true>(DataType::DT_FP32, qi, kj);

                        // V1
                        ConfigManager::Instance().SetSemanticLabel("Sa_Qkvec1");
                        TileShape::Current().SetVecTile(v1Tile[0], v1Tile[1]);
                        auto sijScale = MulS(sij, Element(sij->Datatype(), softmaxScale));
                        auto tildaMij = RowMaxSingle(sijScale); // (curGTile, curS2Tile) -> (curGTile, 1)
                        auto tsub = Sub(sijScale, tildaMij); // (curGTile, curS2Tile), (curGTile, 1) -> (curGTile, curS2Tile)
                        auto tildaPij = Exp(tsub);  // (curGTile, curS2Tile) -> (curGTile, curS2Tile)
                        auto tildaPijF16 = Cast(tildaPij, dtype);
                        auto tildaLij = RowSumSingle(tildaPij);

                        IF (IsLoopBegin(s2Idx, 0)) {
                            // C2
                            TileShape::Current().SetCubeTile(
                                {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]}, true);
                            ConfigManager::Instance().SetSemanticLabel("Sa_KvMm");
                            TileShape::Current().SetMatrixSize(
                                {tildaPijF16.GetShape()[0], tildaPijF16.GetShape()[1], vj.GetShape()[1]});
                            auto oiTmp = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16, vj);
                            TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                            IF (IsLoopEnd(s2Idx, bnPerBatch)) { // PATH3
                                // V2
                                ConfigManager::Instance().SetSemanticLabel("Sa_KvVec2");
                                oiUpdate = Div(oiTmp, tildaLij);
                                TileShape::Current().SetVecTile(1, 1, v2Tile[0], v2Tile[1]);
                                auto oiUpdate4Dim = AddS(Reshape(oiUpdate, {1, 1, curGTile, dN}), Element(oiUpdate->Datatype(), float(0)));
                                Assemble(oiUpdate4Dim, oiOffset, attentionOut);
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
                            TileShape::Current().SetCubeTile(
                                {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]}, true);
                            ConfigManager::Instance().SetSemanticLabel("Sa_UpdateMM2");
                            TileShape::Current().SetMatrixSize(
                                {tildaPijF16.GetShape()[0], tildaPijF16.GetShape()[1], vj.GetShape()[1]});
                            auto q1 = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16, vj);
                            TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1]);
                            auto q2 = Mul(q1, t4);
                            auto oiTmp = Add(q3, q2);
                            IF (IsLoopEnd(s2Idx, bnPerBatch)) { // PATH1
                                oiUpdate = Div(oiTmp, liNew);
                                TileShape::Current().SetVecTile(1, 1, v2Tile[0], v2Tile[1]);
                                auto oiUpdate4Dim = AddS(Reshape(oiUpdate, {1, 1, curGTile, dN}), Element(oiUpdate->Datatype(), float(0)));
                                Assemble(oiUpdate4Dim, oiOffset, attentionOut);
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
    FunctionConfig funConfig;
    FUNCTION("SA_MAIN", funConfig,
        {topKIndcies, kvNopeCache, kRopeCache, kvActSeqs, blockTable, qNope, qRope},
        {attentionOut}) {
        SelectedAttentionCompute(topKIndcies, kvNopeCache, kRopeCache, kvActSeqs, blockTable,
            qNope, qRope, attentionOut,
            nQ, nKv, softmaxScale, front, near, topk, blockSize, cmpBlockSize, slcBlockSize, saTileConfig);
    }
}

} // namespace npu::tile_fwk
