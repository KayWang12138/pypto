/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_combine_ffn_fused.cpp
 * \brief
 */

#include "distributed_common.h"
#include "interface/inner/tilefwk.h"
#include "interface/utils/common.h"
#include "tilefwk/data_type.h"
#include "tilefwk/symbolic_distributed.h"
#include "tilefwk/tensor.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/tilefwk_op.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <numeric>

namespace npu::tile_fwk::Distributed {
void MoeDistributedCombineValidate(const Tensor& expandX, const Tensor& assistInfoForCombine, const Tensor& recvCounts,
    const Tensor& expertScales, const char* group, uint32_t epWorldSize, uint32_t moeExpertNum,
    uint32_t sharedExpertNum, uint32_t sharedExpertRankNum, Tensor& out);

namespace {
int32_t GetFfnIntermediateSizeFromShape(const Shape& shape, int32_t hiddenSize)
{
    ASSERT(hiddenSize > 0) << "hiddenSize must be positive, but got " << hiddenSize;
    int64_t weightElements = std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
    int64_t denom = static_cast<int64_t>(hiddenSize) * 3;
    ASSERT(weightElements > 0) << "ffnWeight cannot be empty.";
    ASSERT(weightElements % denom == 0) << "ffnWeight element count must be divisible by 3 * hiddenSize, got "
        << weightElements << " and hiddenSize=" << hiddenSize;
    int64_t intermediateSize = weightElements / denom;
    ASSERT(intermediateSize > 0) << "ffnWeight intermediateSize must be positive, but got " << intermediateSize;
    ASSERT(intermediateSize <= static_cast<int64_t>(std::numeric_limits<uint16_t>::max()))
        << "ffnWeight intermediateSize exceeds uint16_t: " << intermediateSize;
    return static_cast<int32_t>(intermediateSize);
}

int32_t GetFfnIntermediateSize(const Tensor& ffnWeight, int32_t hiddenSize)
{
    return GetFfnIntermediateSizeFromShape(ffnWeight.GetShape(), hiddenSize);
}

int32_t GetChunkSize(int32_t batchSize)
{
    if (batchSize <= 0) {
        return batchSize;
    }
    const char* chunkEnv = std::getenv("PYPTO_MOE_FFN_CHUNK_SIZE");
    if (chunkEnv != nullptr && chunkEnv[0] != '\0') {
        char* end = nullptr;
        long parsed = std::strtol(chunkEnv, &end, 10);
        if (end != chunkEnv && *end == '\0') {
            if (parsed <= 0) {
                return batchSize;
            }
            if (parsed > std::numeric_limits<int32_t>::max()) {
                return batchSize;
            }
            return std::min(batchSize, static_cast<int32_t>(parsed));
        }
    }
    constexpr int32_t kDefaultChunkPerAiv = 16;
    int32_t defaultChunk = std::min(batchSize, AIV_NUM * kDefaultChunkPerAiv);
    return defaultChunk > 0 ? defaultChunk : batchSize;
}
} // namespace

void MoeDistributedCombineFfnFused(const Tensor& expandX, const Tensor& assistInfoForCombine, const Tensor& recvCounts,
    const Tensor& expertScales, const Tensor& ffnWeight, const char* group, uint32_t epWorldSize,
    uint32_t moeExpertNum, uint32_t sharedExpertNum, uint32_t sharedExpertRankNum, Tensor& out)
{
    MoeDistributedCombineValidate(expandX, assistInfoForCombine, recvCounts, expertScales, group, epWorldSize,
        moeExpertNum, sharedExpertNum, sharedExpertRankNum, out);
    ASSERT(ffnWeight.Format() == npu::tile_fwk::TileOpFormat::TILEOP_ND) << "The format of \"ffnWeight\" only "
        << "supports ND, but got NZ";
    ASSERT(ffnWeight.GetDataType() == expandX.GetDataType()) << "The data type of \"ffnWeight\" must be consistent "
        << "with that of \"expandX\", but got " << DataType2String(ffnWeight.GetDataType()) << " and "
        << DataType2String(expandX.GetDataType());

    int32_t batchSize = expertScales.GetShape(0);
    int32_t hiddenSize = expandX.GetShape(1);
    int32_t intermediateSize = GetFfnIntermediateSize(ffnWeight, hiddenSize);
    int32_t chunkSize = GetChunkSize(batchSize);

    Tensor combineOut(expandX.GetDataType(), {batchSize, hiddenSize}, "combineOut");
    MoeDistributedCombine(expandX, assistInfoForCombine, recvCounts, expertScales, group, epWorldSize,
        moeExpertNum, sharedExpertNum, sharedExpertRankNum, combineOut);

    LOOP("MoeCombineFfn", FunctionType::DYNAMIC_LOOP, loopIdx, LoopRange(1)) {
        (void)loopIdx;
        auto setCubeTile = [](int64_t kSize, int64_t nSize) {
            constexpr int64_t kTileLimit = 64;
            constexpr int64_t nTileLimit = 64;
            int64_t kTile = AlignUp(std::min(kSize, kTileLimit), 16L);
            int64_t nTile = AlignUp(std::min(nSize, nTileLimit), 16L);
            int64_t mTile = 16;
            TileShape::Current().SetCubeTile({mTile, mTile}, {kTile, kTile}, {nTile, nTile});
        };
        auto setVecTile1D = [](int64_t length) {
            constexpr int64_t tileLimit = 1024;
            int64_t tile = AlignUp(std::min(length, tileLimit), 16L);
            TileShape::Current().SetVecTile({tile});
        };
        auto setVecTile2D = [](int64_t lastDim) {
            int64_t tile = AlignUp(lastDim, 16L);
            TileShape::Current().SetVecTile({1, tile});
        };

        int64_t weightStride = static_cast<int64_t>(hiddenSize) * intermediateSize;
        auto gateWeightFlat = View(ffnWeight, {weightStride}, {0});
        auto upWeightFlat = View(ffnWeight, {weightStride}, {weightStride});
        auto downWeightFlat = View(ffnWeight, {weightStride}, {2LL * weightStride});
        setVecTile1D(weightStride);
        auto gateWeight = Reshape(gateWeightFlat, {hiddenSize, intermediateSize});
        auto upWeight = Reshape(upWeightFlat, {hiddenSize, intermediateSize});
        auto downWeight = Reshape(downWeightFlat, {intermediateSize, hiddenSize});
        setVecTile2D(std::max<int64_t>(hiddenSize, intermediateSize));
        auto gateWeightFp32 = Cast(gateWeight, DT_FP32);
        auto upWeightFp32 = Cast(upWeight, DT_FP32);
        auto downWeightFp32 = Cast(downWeight, DT_FP32);

        auto runChunkFfn = [&](const Tensor& combineChunk) {
            setVecTile2D(std::max<int64_t>(hiddenSize, intermediateSize));
            auto combineFp32 = Cast(combineChunk, DT_FP32);
            setCubeTile(hiddenSize, intermediateSize);
            auto gateFp32 = Matrix::Matmul(DT_FP32, combineFp32, gateWeightFp32);
            auto upFp32 = Matrix::Matmul(DT_FP32, combineFp32, upWeightFp32);
            auto negGate = Neg(gateFp32);
            auto expNegGate = Exp(negGate);
            auto denom = Add(expNegGate, Element(DT_FP32, 1.0f));
            auto silu = Div(gateFp32, denom);
            auto interFp32 = Mul(silu, upFp32);
            setCubeTile(intermediateSize, hiddenSize);
            auto outFp32 = Matrix::Matmul(DT_FP32, interFp32, downWeightFp32);
            setVecTile2D(hiddenSize);
            return Cast(outFp32, expandX.GetDataType());
        };

        if (batchSize <= chunkSize) {
            auto combineChunk = View(combineOut, {batchSize, hiddenSize}, {0, 0});
            out = runChunkFfn(combineChunk);
            continue;
        }

        for (int32_t rowOffset = 0; rowOffset < batchSize; rowOffset += chunkSize) {
            int32_t rowShape = std::min(chunkSize, batchSize - rowOffset);
            auto combineChunk = View(combineOut, {rowShape, hiddenSize}, {rowOffset, 0});
            auto outChunk = runChunkFfn(combineChunk);
            Assemble(outChunk, {rowOffset, 0}, out);
        }
    }
}
} // namespace npu::tile_fwk::Distributed
