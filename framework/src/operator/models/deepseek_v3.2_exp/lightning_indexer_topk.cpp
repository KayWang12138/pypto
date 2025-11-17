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
 * \file fused_compress_kv_select.cpp
 * \brief
 */

#include "interface/operation/operation_impl.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "tilefwk/tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/program/program.h"
#include "interface/tensor/tensormap.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/common.h"
#include "interface/utils/id_gen.h"
#include "interface/utils/log.h"
#include "lightning_indexer_topk.h"
#include <cfloat>

using namespace npu::tile_fwk;

namespace npu::tile_fwk {

void LightningIndexerTopkQuant(const Tensor &query, const Tensor &key, const Tensor &qScale, const Tensor &kScale,
    const Tensor &weights, const Tensor &actSeqKey, const Tensor &blockTable, Tensor &topkRes,
    const int selectedCount, IndexerTile tileConfig, const std::set<int> &unrollList) {
    LightningIndexerTopkImpl(query, key, true, &qScale, &kScale,
        weights, actSeqKey, blockTable, topkRes,
        selectedCount, tileConfig, unrollList);
}

void LightningIndexerTopk(const Tensor &query, const Tensor &key,
    const Tensor &weights, const Tensor &actSeqKey, const Tensor &blockTable, Tensor &topkRes,
    const int selectedCount, IndexerTile tileConfig, const std::set<int> &unrollList) {
    LightningIndexerTopkImpl(query, key, false, nullptr, nullptr,
        weights, actSeqKey, blockTable, topkRes,
        selectedCount, tileConfig, unrollList);
}

void LightningIndexerTopkImpl(const Tensor &query, const Tensor &key, bool isQuant, const Tensor *qScale, const Tensor *kScale,
    const Tensor &weights, const Tensor &actSeqKey, const Tensor &blockTable, Tensor &topkRes,
    const int selectedCount, IndexerTile tileConfig, const std::set<int> &unrollList,
    Tensor *tmpOut, Tensor *topkValue) {
    /*
    <no quant>
    query: [B, S1, indexN1, indexD], bf16
    key: [blockNum, blockSize, n2, indexD] bf16

    <quant>
    query: [B, S1, indexN1, indexD], int8
    key: [blockNum, blockSize, n2, indexD] int8
    qScale: [B, S1, indexN1, 1], fp16
    kScale: [blockNum, blockSize, n2, 1] fp16

    <common>
    weights: [B, S1, indexN1], bf16
    actSeqKey: [B], int32
    blockTable: [B, maxBlockNum]
    topkRes: [B, s1, N2, selectedCount], int32
    selectedCount: selectedCount num
    */

    // Symbolization
    SymbolicScalar b = GetInputShape(query, 0);
    SymbolicScalar s1 = GetInputShape(query, 1);
    SymbolicScalar blockNum = GetInputShape(key, 0);

    auto indexN1 = query.GetStorage()->shape[SHAPE_DIM2];
    auto indexD = query.GetStorage()->shape[SHAPE_DIM3];
    auto blockSize = key.GetStorage()->shape[1];
    auto n2 = key.GetStorage()->shape[SHAPE_DIM2];
    auto qkDType = query.GetStorage()->Datatype();
    auto scaleDType = isQuant ? (*qScale).GetStorage()->Datatype() : DT_FP16;
    auto wDType = weights.GetStorage()->Datatype();
    auto group = indexN1 / n2; // 暂时考虑整除
    auto c1Tile = tileConfig.c1Tile;
    constexpr int64_t maxBatch = 128;
    constexpr int64_t maxS1 = 4;
    constexpr int64_t maxN2 = 1;
    constexpr int64_t maxS2 = 128 * 1024;

    constexpr float AVOID_FP32_TO_FP16_OVERFLOW_SCALE = 1.0f / 2048;

    Tensor query2D(qkDType, {b * s1 * indexN1, indexD}, "query2D");
    Tensor key2D(qkDType, {blockNum * blockSize, n2 * indexD}, "key2D");
    Tensor qScale2D(scaleDType, {b * s1 * indexN1, 1}, "qScale2D");
    Tensor kScale2D(scaleDType, {blockNum * blockSize, n2}, "kScale2D");
    Tensor weight2D(wDType, {b * s1 * indexN1, 1}, "weight2D");
    Tensor localSum(DT_FP32, {maxBatch * maxS1 * maxN2, maxS2}, "localSum");

    LOOP("INPUT_4D_2_2D", FunctionType::DYNAMIC_LOOP, unUsedIdx, LoopRange(1)) {
        (void)unUsedIdx;
        query2D = Reshape(query, {b * s1 * indexN1, indexD}, true);
        key2D = Reshape(key, {blockNum * blockSize, n2 * indexD}, true);
        weight2D = Reshape(weights, {b * s1 * indexN1, 1}, true);
        if (isQuant) {
            qScale2D = Reshape(*qScale, {b * s1 * indexN1, 1}, true);
            kScale2D = Reshape(*kScale, {blockNum * blockSize, n2}, true);
        }
    }

    LOOP("INDEX_LOOP_BATCH", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(b)) {
        auto curSeq = GetTensorData(actSeqKey, {bIdx});

        LOOP("INDEX_LOOP_S1", FunctionType::DYNAMIC_LOOP, s1Idx, LoopRange(s1)) {
            // 因果推理
            auto casualOffset = s1 - s1Idx - 1;
            auto effSeq = curSeq - casualOffset;
            auto actBlock = (effSeq + blockSize - 1) / blockSize;
            LOOP("INDEX_LOOP_N2", FunctionType::DYNAMIC_LOOP, n2Idx, LoopRange(n2)) {
                auto bs1n2Offset = bIdx * s1 * n2 + s1Idx * n2 + n2Idx;
                auto qOffset = bIdx * s1 * indexN1 + s1Idx * indexN1 + n2Idx * group;

                // unrolling process template
                auto unrollingProcess = [&](int unrollLength, auto &&firstBlockIdx) {
                    // TileShape::Current().SetVecTile(128, 128);
                    auto curQ = View(query2D, {group, indexD}, {qOffset, 0}); // (group, d)
                    std::vector<Tensor> concatSrcs;
                    // static unrolling
                    for (int subblockIdx = 0; subblockIdx < unrollLength; subblockIdx++) {
                        auto blockIdx = firstBlockIdx + subblockIdx;
                        SymbolicScalar curBlockIdx = GetTensorData(blockTable, {bIdx, blockIdx});
                        auto curK = View(key2D, {blockSize, indexD},
                            {std::min(blockSize, effSeq - (blockIdx * blockSize)), indexD},
                            {curBlockIdx * blockSize, n2Idx * indexD});

                        TileShape::Current().SetCubeTile(
                            {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]}, false);
                        auto mmRes =
                            Matrix::Matmul<false, true>(DataType::DT_FP32, curQ, curK); // (group, blockSize)
                        concatSrcs.emplace_back(mmRes); // Matches tile size, no extra Assign needed (?)
                    }

                    TileShape::Current().SetVecTile(tileConfig.weightTile);

                    auto curW = View(weight2D, {group, 1}, {qOffset, 0}); // (group, 1)
                    auto wB32 = Cast(curW, DT_FP32);                      // (group, 1)

                    auto mmRes = Cat(concatSrcs, -1); // (group, superBlockSize)

                    TileShape::Current().SetVecTile(tileConfig.v1Tile);
                    auto reluRes = Maximum(mmRes, Element(DT_FP32, 0.0f));       // (group, superBlockSize)
                    auto mulRes = Mul(reluRes, wB32); // (group, superBlockSize) * (group, 1) -> (group, superBlockSize)
                    auto sumRes = Sum(mulRes, 0, true); // (1, superBlockSize)
                    Assemble(sumRes, {bs1n2Offset, firstBlockIdx * blockSize}, localSum);
                    if (tmpOut != nullptr) {
                        // tmpOut: [B*S1*N2, S2]
                        Assemble(sumRes, {bs1n2Offset, firstBlockIdx * blockSize}, *tmpOut);
                    }
                };

                auto unrollingProcessQuant = [&](int unrollLength, auto &&firstBlockIdx) {
                    // TileShape::Current().SetVecTile(128, 128);
                    auto curQ = View(query2D, {group, indexD}, {qOffset, 0}); // (group, d)
                    Tensor curQScale = View(qScale2D, {group, 1}, {qOffset, 0}); // (group, 1)
                    std::vector<Tensor> mmResQuantConcatSrcs;
                    std::vector<Tensor> kScaleConcatSrcs;

                    // static unrolling
                    for (int subblockIdx = 0; subblockIdx < unrollLength; subblockIdx++) {
                        auto blockIdx = firstBlockIdx + subblockIdx;
                        SymbolicScalar curBlockIdx = GetTensorData(blockTable, {bIdx, blockIdx});
                        auto curK = View(key2D, {blockSize, indexD},
                            {std::min(blockSize, effSeq - (blockIdx * blockSize)), indexD},
                            {curBlockIdx * blockSize, n2Idx * indexD});

                        TileShape::Current().SetCubeTile(
                            {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]}, false);

                        auto mmRes =
                            Matrix::Matmul<false, true>(DataType::DT_INT32, curQ, curK); // (group, blockSize)
                        mmResQuantConcatSrcs.emplace_back(mmRes);

                        auto curKScale = View(kScale2D, {blockSize, 1},
                            {std::min(blockSize, effSeq - (blockIdx * blockSize)), 1},
                            {curBlockIdx * blockSize, n2Idx}); // (blockSize, 1)
                        kScaleConcatSrcs.emplace_back(curKScale);
                    }

                    TileShape::Current().SetVecTile(tileConfig.weightTile);

                    auto curW = View(weight2D, {group, 1}, {qOffset, 0}); // (group, 1)
                    auto wF16 = Cast(curW, DT_FP16);                      // (group, 1)

                    TileShape::Current().SetVecTile(tileConfig.v1Tile);
                    auto curKScale = Cat(kScaleConcatSrcs, 0);
                    auto mmResI32 = Cat(mmResQuantConcatSrcs, -1); // (group, superBlockSize)
                    auto mmResFP32 = Mul(Cast(mmResI32, DT_FP32), Element(DT_FP32, AVOID_FP32_TO_FP16_OVERFLOW_SCALE));
                    auto mmResFP16 = Cast(mmResFP32, DT_FP16);
                    auto mmResDequant = Mul(Mul(mmResFP16, curQScale), Transpose(curKScale, {0, 1}));
                    auto reluRes = Maximum(mmResDequant, Element(DT_FP16, 0.0f)); // (group, superBlockSize)
                    auto mulRes = Mul(reluRes, wF16); // (group, superBlockSize) * (group, 1) -> (group, superBlockSize)

                    // RowSumSingle doesn't support non-4-byte types currently
                    auto sumRes = Sum(Cast(mulRes, DT_FP32), 0, true); // (1, superBlockSize)
                    Assemble(sumRes, {bs1n2Offset, firstBlockIdx * blockSize}, localSum);
                    if (tmpOut != nullptr) {
                        // tmpOut: [B*S1*N2, S2]
                        Assemble(sumRes, {bs1n2Offset, firstBlockIdx * blockSize}, *tmpOut);
                    }
                };

                LOOP("INDEX_LOOP_MATMUL", FunctionType::DYNAMIC_LOOP, loopBlockIdx, LoopRange(actBlock), unrollList) {
                    for (int loopUnrollLength : unrollList) {
                        UNROLL(loopUnrollLength) {
                            if (isQuant) {
                                unrollingProcessQuant(loopUnrollLength, loopBlockIdx);
                            } else {
                                unrollingProcess(loopUnrollLength, loopBlockIdx);
                            }
                        }
                    }
                }
            }
        }
    }

    constexpr int32_t NUMS_2048 = 2048;
    ASSERT(selectedCount == NUMS_2048);
    DataType xdtype = localSum.GetDataType();
    DataType idxdtype = topkRes.GetDataType();
    const int padIdxValue = -1;
    const int tileSize = 8192;
    const bool descending = true;
    const float padValue = descending ? -FLT_MAX : FLT_MAX;
    const int length2K = selectedCount;
    const int length8K = 1024 * 8;
    const int length64K = 1024 * 64;
    const int length128K = maxS2;
    TileShape::Current().SetVecTile({1, tileSize});
    LOOP("INDEX_LOOP_TOPK_bs1n2Offset", FunctionType::DYNAMIC_LOOP, bs1n2Offset, LoopRange(b * s1 * n2)) {    // barrier on
        auto bIdx = bs1n2Offset / (s1 * n2);
        auto s1Idx = (bs1n2Offset % (s1 * n2)) / n2;
        auto n2Idx = bs1n2Offset % n2;

        auto curSeq = GetTensorData(actSeqKey, {bIdx});
        auto casualOffset = s1 - s1Idx - 1;
        auto effSeq = curSeq - casualOffset;

        auto lengthIsLE2K = effSeq <= length2K;
        auto lengthIsGT2K = effSeq > length2K;
        Tensor padX2K(xdtype, {maxBatch * maxS1 * maxN2, length2K}, "padX2K");
        TileShape::Current().SetVecTile({1, tileSize});
        LOOP("2K_LOOP", FunctionType::DYNAMIC_LOOP, unused, LoopRange(lengthIsLE2K)) {
            (void)unused;
            config::SetPassOption(SG_SKIP_PARTITION, true);
            LOOP("2K_PAD", FunctionType::DYNAMIC_LOOP, unused1, LoopRange(1)) {
                (void)unused1;
                TileShape::Current().SetVecTile({1, length2K});
                auto effSumRes = View(localSum, {1, length2K}, {1, effSeq}, {bs1n2Offset, 0});
                auto ax = View(effSumRes, {1, length2K}, {1, effSeq}, {0, 0});
                auto bx = Full(Element(xdtype, padValue), xdtype, {1, length2K}, {1, length2K - effSeq});
                Assemble(Assign(ax), {bs1n2Offset, 0}, padX2K);
                Assemble(bx, {bs1n2Offset, effSeq}, padX2K);
            }
            config::SetPassOption(SG_SKIP_PARTITION, false);
            LOOP("2K_TOPK", FunctionType::DYNAMIC_LOOP, unused2, LoopRange(1)) {
                (void)unused2;
                auto [resValue, resIdx] = TopK(View(padX2K, {1, length2K}, {bs1n2Offset, 0}), selectedCount, 1);
                TileShape::Current().SetVecTile(tileConfig.addsTile);
                auto topk4D = Reshape(
                    View(resIdx, {1, selectedCount}, {1, effSeq}, {0, 0}), {1, 1, 1, selectedCount}, {1, 1, 1, effSeq});
                Assemble(Assign(topk4D), {bIdx, s1Idx, n2Idx, 0}, topkRes);
                auto topkIndicesPad = Full(Element(idxdtype, padIdxValue), idxdtype,
                    {1, 1, 1, selectedCount}, {1, 1, 1, selectedCount - effSeq});
                Assemble(topkIndicesPad, {bIdx, s1Idx, n2Idx, effSeq}, topkRes);

                if (topkValue != nullptr) {
                    auto topk4DValue = Reshape(View(resValue, {1, selectedCount}, {1, effSeq}, {0, 0}),
                        {1, 1, 1, selectedCount}, {1, 1, 1, effSeq});
                    Assemble(Assign(topk4DValue), {bIdx, s1Idx, n2Idx, 0}, *topkValue);
                    auto topkValuePad = Full(Element(DT_FP32, padValue), DT_FP32, {1, 1, 1, selectedCount},
                        {1, 1, 1, selectedCount - effSeq});
                    Assemble(topkValuePad, {bIdx, s1Idx, n2Idx, effSeq}, *topkValue);
                }
                TileShape::Current().SetVecTile({1, tileSize});
            }
        }

        auto lengthIsLE8K = effSeq <= length8K;
        auto lengthIsGT8K = effSeq > length8K;
        Tensor padX8K(xdtype, {maxBatch * maxS1 * maxN2, length8K}, "padX8K");
        LOOP("8K_LOOP", FunctionType::DYNAMIC_LOOP, unused, LoopRange(lengthIsGT2K * lengthIsLE8K), {}, true) {
            UNUSED(unused);
            LOOP("8K_PAD", FunctionType::DYNAMIC_LOOP, unused0, LoopRange(1)) {
                UNUSED(unused0);
                TileShape::Current().SetVecTile({1, tileSize});
                auto ax = View(localSum, {1, length8K}, {1, effSeq}, {bs1n2Offset, 0});
                auto bx = Full(Element(xdtype, padValue), xdtype, {1, length8K}, {1, length8K - effSeq});
                Assemble(Assign(ax), {bs1n2Offset, 0}, padX8K);
                Assemble(bx, {bs1n2Offset, effSeq}, padX8K);
            }

            TileShape::Current().SetVecTile({1, tileSize});
            LOOP("8K_TOPK", FunctionType::DYNAMIC_LOOP, unused1, LoopRange(1)) {
                UNUSED(unused1);
                auto [resValue, resIdx] = TopK(View(padX8K, {1, length8K}, {bs1n2Offset, 0}), selectedCount, 1);
                TileShape::Current().SetVecTile(tileConfig.addsTile);
                auto topk4D = Reshape(resIdx, {1, 1, 1, selectedCount});
                Assemble(Assign(topk4D), {bIdx, s1Idx, n2Idx, 0}, topkRes);
                if (topkValue != nullptr) {
                    TileShape::Current().SetVecTile(tileConfig.addsTile);
                    auto topk4DValue = Reshape(resValue, {1, 1, 1, selectedCount});
                    Assemble(Assign(topk4DValue), {bIdx, s1Idx, n2Idx, 0}, *topkValue);
                }
                TileShape::Current().SetVecTile({1, tileSize});
            }
        }

        auto lengthIsLE64K = effSeq <= length64K;
        auto lengthIsGT64K = effSeq > length64K;
        Tensor padX64K(xdtype, {maxBatch * maxS1 * maxN2, length64K}, "padX64K");
        LOOP("64K_LOOP", FunctionType::DYNAMIC_LOOP, unused, LoopRange(lengthIsGT8K * lengthIsLE64K)) {
            UNUSED(unused);
            LOOP("64K_PAD", FunctionType::DYNAMIC_LOOP, unused0, LoopRange(1)) {
                UNUSED(unused0);
                TileShape::Current().SetVecTile({1, tileSize});
                auto ax = View(localSum, {1, length64K}, {1, effSeq}, {bs1n2Offset, 0});
                auto bx = Full(Element(xdtype, padValue), xdtype, {1, length64K}, {1, length64K - effSeq});
                Assemble(Assign(ax), {bs1n2Offset, 0}, padX64K);
                Assemble(bx, {bs1n2Offset, effSeq}, padX64K);
            }

            TileShape::Current().SetVecTile({1, tileSize});
            LOOP("64K_TOPK", FunctionType::DYNAMIC_LOOP, unused1, LoopRange(1)) {
                UNUSED(unused1);
                auto [resValue, resIdx] = TopK(View(padX64K, {1, length64K}, {bs1n2Offset, 0}), selectedCount, 1);
                TileShape::Current().SetVecTile(tileConfig.addsTile);
                auto topk4D = Reshape(resIdx, {1, 1, 1, selectedCount});
                Assemble(Assign(topk4D), {bIdx, s1Idx, n2Idx, 0}, topkRes);
                if (topkValue != nullptr) {
                    TileShape::Current().SetVecTile(tileConfig.addsTile);
                    auto topk4DValue = Reshape(resValue, {1, 1, 1, selectedCount});
                    Assemble(Assign(topk4DValue), {bIdx, s1Idx, n2Idx, 0}, *topkValue);
                }
                TileShape::Current().SetVecTile({1, tileSize});
            }
        }

        Tensor padX128K(xdtype, {maxBatch * maxS1 * maxN2, length128K}, "padX128K");
        LOOP("128K_LOOP", FunctionType::DYNAMIC_LOOP, unused, LoopRange(lengthIsGT64K)) {
            UNUSED(unused);
            LOOP("128K_PAD", FunctionType::DYNAMIC_LOOP, unused0, LoopRange(1)) {
                UNUSED(unused0);
                TileShape::Current().SetVecTile({1, tileSize});
                auto ax = View(localSum, {1, length128K}, {1, effSeq}, {bs1n2Offset, 0});
                auto bx = Full(Element(xdtype, padValue), xdtype, {1, length128K}, {1, length128K - effSeq});
                Assemble(Assign(ax), {bs1n2Offset, 0}, padX128K);
                Assemble(bx, {bs1n2Offset, effSeq}, padX128K);
            }

            TileShape::Current().SetVecTile({1, tileSize});
            LOOP("128K_TOPK", FunctionType::DYNAMIC_LOOP, unused1, LoopRange(1)) {
                UNUSED(unused1);
                auto [resValue, resIdx] = TopK(View(padX128K, {1, length128K}, {bs1n2Offset, 0}), selectedCount, 1);
                TileShape::Current().SetVecTile(tileConfig.addsTile);
                auto topk4D = Reshape(resIdx, {1, 1, 1, selectedCount});
                Assemble(Assign(topk4D), {bIdx, s1Idx, n2Idx, 0}, topkRes);
                if (topkValue != nullptr) {
                    TileShape::Current().SetVecTile(tileConfig.addsTile);
                    auto topk4DValue = Reshape(resValue, {1, 1, 1, selectedCount});
                    Assemble(Assign(topk4DValue), {bIdx, s1Idx, n2Idx, 0}, *topkValue);
                }
                TileShape::Current().SetVecTile({1, tileSize});
            }
        }
    }
}

} // namespace npu::tile_fwk