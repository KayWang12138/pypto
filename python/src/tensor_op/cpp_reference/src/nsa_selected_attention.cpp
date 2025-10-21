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
#include "interface/operation/operation.h"
#include "interface/configs/config_manager.h"

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
constexpr int NUM_1 = 1;
constexpr int NUM_2 = 2;
constexpr int NUM_16 = 16;
constexpr int NUM_32 = 32;
constexpr int NUM_64 = 64;
constexpr int NUM_128 = 128;
constexpr int NUM_512 = 512;
constexpr int NUM_1024 = 1024;
constexpr int NUM_1536 = 1536;
constexpr int NUM_7168 = 7168;
constexpr int NUM_8192 = 8192;
constexpr int NUM_65536 = 65536;

struct NSASimpleParams {
    int b;
    int s1;
    int s2;
    int n1;
    int n2;
    int h;
    int qLoraRank;
    int kvLoraRank;
    int qkRopeHeadDim;
    int qkNopeHeadDim;
    int qHeadDim;
    int ropeDim;
    int cmpBlockSize;
    int cmpStride;
    int slcBlockSize;
    int front;
    int near;
    int topk;
    std::string cacheMode;
    int blockSize;
    int winSize;
    int vHeadDim;
    float eps;
    static NSASimpleParams GetCommonParams()
    {
        NSASimpleParams params;
        params.h = NUM_7168;
        params.qLoraRank = NUM_1536;
        params.kvLoraRank = NUM_512;
        params.qkRopeHeadDim = NUM_64;
        params.qkNopeHeadDim = NUM_128;
        params.qHeadDim = params.qkRopeHeadDim + params.qkNopeHeadDim;
        params.ropeDim = NUM_64;
        params.cmpBlockSize = NUM_32;
        params.cmpStride = NUM_16;
        params.slcBlockSize = NUM_64;
        params.front = NUM_1;
        params.near = NUM_2;
        params.topk = NUM_16;
        params.cacheMode = "BSND";
        params.blockSize = NUM_128;
        params.winSize = NUM_512;
        params.vHeadDim = NUM_128;
        params.eps = 1e-5f;
        return params;
    }

    static NSASimpleParams GetDecodeParams()
    {
        NSASimpleParams params = GetCommonParams();
        params.b = NUM_32;
        params.s1 = NUM_1;
        params.s2 = NUM_65536;
        params.n1 = NUM_128;
        params.n2 = NUM_1;
        return params;
    }

    static NSASimpleParams GetMTPParams()
    {
        NSASimpleParams params = GetCommonParams();
        params.b = NUM_32;
        params.s1 = NUM_2;
        params.s2 = NUM_65536;
        params.n1 = NUM_128;
        params.n2 = NUM_1;
        return params;
    }
};

struct SATileShapeConfig {
    std::array<int, TILE_VEC_DIMS> kvSlcV0TileShape;

    int gTile;  // 由于没有处理尾块，当前仅支持因子切分
    int sKvTile;
    std::array<int, TILE_CUBE_DIMS> c1TileShape;  // (m, M), (k, K), (n, N)
    std::array<int, TILE_VEC_DIMS> v1TileShape;
    std::array<int, TILE_CUBE_DIMS> c2TileShape;  // (m, M), (k, K), (n, N)
    std::array<int, TILE_VEC_DIMS> v2TileShape;
};

void SelectedAttentionCompute(Tensor &topKIndcies, Tensor &kvNopeCache, Tensor &kRopeCache, Tensor &kvActSeqs, Tensor &blockTable,
                              const Tensor &qNope, const Tensor &qRope, Tensor &attentionOut,
                              int nQ, int nKv, float softmaxScale, int front, int near, int topk, int blockSize, int cmpBlockSize, int slcBlockSize,
                              SATileShapeConfig saTileConfig, bool debug = false) 
{
    auto dtype = qNope->Datatype();
    int dN = qNope->shape[1];
    int dR = qRope->shape[1];
    int group = 0;
    if (nKv != 0) {
        group = nQ / nKv;
    } else {
        std::cout << "nKv cannot be 0" << std::endl;
    }

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

    SymbolicScalar batchSizeSym = topKIndcies->shape[0];      // b
    SymbolicScalar s1N2GSym = qNope->shape[0] / batchSizeSym; // s1n2
    SymbolicScalar s1Sym = 0;                     // s1
    if (nQ != 0) {
        s1Sym = s1N2GSym / nQ;
    } else {
        std::cout << "nQ cannot be 0" << std::endl;
    }
    SymbolicScalar gLoopSym = group / gTile;
    SymbolicScalar n2Sym = nKv;

    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);

    LOOP("LOOP_L0_b_SA", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, batchSizeSym, 1), {}, true)
    {
        SymbolicScalar curActSeq = GetTensorData(kvActSeqs, {bIdx});
        curActSeq.AsIntermediateVariable();
        LOOP("LOOP_L1_s1_SA", FunctionType::DYNAMIC_LOOP, s1Idx, LoopRange(0, s1Sym, 1)){
            LOOP("LOOP_L2_n2_SA", FunctionType::DYNAMIC_LOOP, n2Idx, LoopRange(0, n2Sym, 1)){  // GQA场景
                LOOP("LOOP_L3_g_SA", FunctionType::DYNAMIC_LOOP, gIdx, LoopRange(0, gLoopSym, 1)){  // slc_attn
                    int curGTile = gTile;
                    SymbolicScalar curOffset = bIdx * s1N2GSym + s1Idx * nQ + n2Idx * group + gIdx * curGTile;
                    std::vector<SymbolicScalar> oiOffset = {bIdx, s1Idx, n2Idx * group + gIdx * curGTile, 0};
                    LOOP("LOOP_L4_s2_SA", FunctionType::DYNAMIC_LOOP, s2Idx, LoopRange(0, 1, 1), PowersOf2(1)){
                        int curS2Tile = topk * slcBlockSize;
                        // kv_slc
                        config::SetSemanticLabel("kv_slc");
                        Tensor kSlc(dtype, {topk * slcBlockSize, dN + dR}, "kSlc");
                        SymbolicScalar curKvSlcSeq = 0;
                        SymbolicScalar sSlc = 0;
                        if (slcBlockSize != 0) {
                            sSlc = (curActSeq - s1Sym + 1 + s1Idx - cmpBlockSize + slcBlockSize) / slcBlockSize;
                        } else {
                            std::cout << "slcBlockSize cannot be 0" << std::endl;
                        }
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
                                    topkIndex = GetTensorData(topKIndcies, {bIdx, s1Idx, topKIdx - front});
                                } else {
                                    topkIndex = GetTensorData(topKIndcies, {bIdx, s1Idx, topKIdx - front});
                                }

                                positions = topkIndex * slcBlockSize;
                            }
                            curKvSlcSeq = curKvSlcSeq + std::min(slcBlockSize, curActSeq - positions);
                            SymbolicScalar blockIdxInBatch = 0;
                            if (blockSize != 0) {
                                blockIdxInBatch = positions / blockSize;
                            } else {
                                std::cout << "blockSize cannot be 0" << std::endl;
                            }
                            SymbolicScalar tail = 0;
                            if (blockSize != 0) {
                                tail = positions % blockSize;
                            } else {
                                std::cout << "blockSize cannot be 0" << std::endl;
                            }
                            SymbolicScalar slcBlockIdx = GetTensorData(blockTable, {bIdx, blockIdxInBatch});
                            TileShape::Current().SetVecTile(v0Tile[0], v0Tile[1]);
                            auto kvSlcBlock =
                                View(kvNopeCache, {slcBlockSize, dN}, {slcBlockIdx * blockSize + tail, n2Idx * dN});
                            auto krSlcBlock =
                                View(kRopeCache, {slcBlockSize, dR}, {slcBlockIdx * blockSize + tail, n2Idx * dR});

                            config::SetSemanticLabel("kv_slc_cast_fp32");
                            TileShape::Current().SetVecTile(v0Tile[0], v0Tile[1]);
                            auto kvSlcBlock_fp32 = Cast(kvSlcBlock, DataType::DT_FP32);
                            auto krSlcBlock_fp32 = Cast(krSlcBlock, DataType::DT_FP32);
                            config::SetSemanticLabel("kv_slc_cast");
                            TileShape::Current().SetVecTile(v0Tile[0], v0Tile[1]);
                            auto kvSlcBlock_fp16 = Cast(kvSlcBlock_fp32, kSlc->Datatype());
                            auto krSlcBlock_fp16 = Cast(krSlcBlock_fp32, kSlc->Datatype());
                            TileShape::Current().SetVecTile(v0Tile[0], v0Tile[1]);

                            SymbolicScalar slcOutSOffset = topKIdx * slcBlockSize;
                            Assemble(kvSlcBlock_fp16, {slcOutSOffset, 0}, kSlc);
                            Assemble(krSlcBlock_fp16, {slcOutSOffset, dN}, kSlc);
                        }

                        // qAssemble
                        config::SetSemanticLabel("Sa");
                        // View, 临时规避改成 View
                        auto qn = View(qNope, {curGTile, dN}, {curGTile, dN}, {curOffset, 0});
                        auto qr = View(qRope, {curGTile, dR}, {curGTile, dR}, {curOffset, 0});
                        Tensor qi(dtype, {curGTile, dN + dR}, "qi");
                        Assemble(qn, {0, 0}, qi);
                        Assemble(qr, {0, dN}, qi);

                        // slc_attn
                        SymbolicScalar curSeq =
                            std::max(curKvSlcSeq - s1Sym + 1 + s1Idx, 0); // for MTP s1!= 1 casual计算
                        curSeq.AsIntermediateVariable();
                        auto kj = View(kSlc, {curS2Tile, dN + dR}, 
                                        {std::min(curSeq - s2Idx * curS2Tile, curS2Tile), dN + dR},
                                        {s2Idx * curS2Tile, 0});  // kSlc已经合并了rope和nope
                        auto vj = View(kSlc, {curS2Tile, dN}, {std::min(curSeq - s2Idx * curS2Tile, curS2Tile), dN},
                                        {s2Idx * curS2Tile, 0});

                        // C1
                        config::SetSemanticLabel("Sa_QkMM");
                        TileShape::Current().SetCubeTile(
                                                        {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]}, true);
                        TileShape::Current().SetMatrixSize({qi.GetShape()[0], 0, kj.GetShape()[0]});
                        auto sij = Matrix::Matmul<false, true>(DataType::DT_FP32, qi, kj);

                        // V1
                        config::SetSemanticLabel("Sa_Qkvec1");
                        TileShape::Current().SetVecTile(v1Tile[0], v1Tile[1]);
                        auto sijScale = MulS(sij, Element(sij->Datatype(), softmaxScale));
                        auto tildaMij = RowMaxSingle(sijScale); // (curGTile, curS2Tile) -> (curGTile, 1)
                        auto tsub =
                            Sub(sijScale, tildaMij); // (curGTile, curS2Tile), (curGTile, 1) -> (curGTile, curS2Tile)
                        auto tildaPij = Exp(tsub);  // (curGTile, curS2Tile) -> (curGTile, curS2Tile)
                        auto tildaLij = RowSumSingle(tildaPij);
                        auto tSoftmax = Div(tildaPij, tildaLij);
                        auto tildaPijF16 = Cast(tSoftmax, dtype);

                        // C2
                        config::SetSemanticLabel("Sa_KvMm");
                        TileShape::Current().SetCubeTile(
                            {c2Tile[0], c2Tile[1]}, {c2Tile[2], c2Tile[3]}, {c2Tile[4], c2Tile[5]}, true);
                        TileShape::Current().SetMatrixSize(
                            {tildaPijF16.GetShape()[0], tildaPijF16.GetShape()[1], vj.GetShape()[1]});
                        auto oi = Matrix::Matmul<false, false>(DataType::DT_FP32, tildaPijF16, vj);

                        // V2
                        config::SetSemanticLabel("Sa_KvVec2");
                        TileShape::Current().SetVecTile(1, 1, v2Tile[0], v2Tile[1]);
                        auto oi4Dim = AddS(Reshape(oi, {1, 1, curGTile, dN}), Element(oi->Datatype(), float(0)));
                        Assemble(oi4Dim, oiOffset, attentionOut);
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
    FUNCTION("SA_MAIN",
        {topKIndcies, kvNopeCache, kRopeCache, kvActSeqs, blockTable, qNope, qRope},
        {attentionOut})
    {
        SelectedAttentionCompute(topKIndcies, kvNopeCache, kRopeCache, kvActSeqs, blockTable,
            qNope, qRope, attentionOut,
            nQ, nKv, softmaxScale, front, near, topk, blockSize, cmpBlockSize, slcBlockSize, saTileConfig);
    }
}

template <typename T = npu::tile_fwk::float16>
void TestKvSlcAttn(const NSASimpleParams &params, SATileShapeConfig& saTileConfig) 
{
    int b = params.b;
    int s1 = params.s1;
    int s2 = params.s2;
    int n1 = params.n1;
    int n2 = params.n2;
    int vDim = params.kvLoraRank;
    int dn = vDim;
    int dr = params.ropeDim;
    float softmaxScale = static_cast<float>(1.0 / sqrtf((dn + dr)));
    int blockSize = params.blockSize;
    int cmpBlockSize = params.cmpBlockSize;
    int slcBlockSize = params.slcBlockSize;
    int front = params.front;
    int near = params.near;
    int topk = params.topk;
    int smax = params.topk * params.slcBlockSize;

    std::vector<int> kvCacheActSeqVec(b, s2);
    int blockNum = 0;
    for (auto seqItem : kvCacheActSeqVec) {
        blockNum += CeilDiv(seqItem, blockSize);
    }
    int maxSeqAllBatch = *(std::max_element(kvCacheActSeqVec.begin(), kvCacheActSeqVec.end()));
    int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);

    DataType dType = (std::is_same<T, npu::tile_fwk::float16>::value) ? DT_FP16 : DT_BF16;

    // 1. 设置shape
    std::vector<int64_t> topkIndicesShape = {b, s1, topk - front - near};
    std::vector<int64_t> topkTensorShapeShape = {b, s1};
    std::vector<int64_t> kvNopeCacheShape = {int(blockNum * blockSize), n2 * dn};
    std::vector<int64_t> kRopeCacheShape = {int(blockNum * blockSize), n2 * dr};
    std::vector<int64_t> kvCacheActSeqShape = {b};
    std::vector<int64_t> blockTableShape = {b, maxBlockNumPerBatch};
    std::vector<int64_t> slcActSeqsShape = {b, s1};

    std::vector<int64_t> qNopeShape = {b * s1 * n1, dn};
    std::vector<int64_t> qRopeShape = {b * s1 * n1, dr};
    std::vector<int64_t> kSlcShape = {b * s1 * n2 * smax, dn + dr};
    std::vector<int64_t> vSlcShape = {b * s1 * n2 * smax, dn};

    std::vector<int64_t> shapeSelAtten = {b, s1, n1, vDim};

    // 2. 构造tensor
    Tensor topkIndices(DT_INT32, topkIndicesShape, "topkTensor");
    Tensor topkTensorShape(DT_INT32, topkTensorShapeShape, "topkTensorShape");
    Tensor kvNopeCache(dType, kvNopeCacheShape, "kNopeCache");
    Tensor kRopeCache(dType, kRopeCacheShape, "vNopeCache");
    Tensor kvCacheActSeq(DT_INT32, kvCacheActSeqShape, "kvCacheActSeq");
    Tensor blockTable(DT_INT32, blockTableShape, "blockTable");
    Tensor slcActSeqs(DT_INT32, slcActSeqsShape, "slcActSeqs");

    Tensor qNope(dType, qNopeShape, "qNope");
    Tensor qRope(dType, qRopeShape, "qRope");

    Tensor attenOut(DT_FP32, shapeSelAtten, "attenOut");

    SelectedAttention(topkIndices, kvNopeCache, kRopeCache, kvCacheActSeq, blockTable, qNope, qRope, attenOut, n1, n2,
                      softmaxScale, front, near, topk, blockSize, cmpBlockSize, slcBlockSize, saTileConfig);
}
} // namespace npu::tile_fwk

int main()
{
    NSASimpleParams params = NSASimpleParams::GetDecodeParams();

    params.b = NUM_16;
    params.s1 = NUM_1;
    params.s2 = NUM_8192;
    params.n1 = NUM_128;
    params.n2 = NUM_1;

    SATileShapeConfig saTileConfig;
    saTileConfig.kvSlcV0TileShape = {64, 256}; // slcBlockSize=64
    const int gTile = NUM_128;                 // for gLoop split
    const int sTile = NUM_1024;                // for s2Loop split
    saTileConfig.gTile = gTile;
    saTileConfig.sKvTile = sTile;
    saTileConfig.c1TileShape = {gTile, gTile, 64, 64, 256, 256};    // (n1, dn+dr) @ (s2Tile, dn+dr) -> (n1, s2Tile)
    saTileConfig.v1TileShape = {16, 256};                           // (n1, s2Tile)
    saTileConfig.c2TileShape = {gTile, gTile, 128, 128, 128, 128};  // (n1, s2Tile) @ (s2Tile, dn) -> (n1, d)
    saTileConfig.v2TileShape = {64, 128};                           // (n1, d)

    TestKvSlcAttn<npu::tile_fwk::float16>(params, saTileConfig);
}
