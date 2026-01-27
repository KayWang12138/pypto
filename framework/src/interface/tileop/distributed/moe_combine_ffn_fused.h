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
 * \file moe_combine_ffn_fused.h
 * \brief MOE combine + FFN fused kernel (AIC/AIV split).
 */

#ifndef __DISTRIBUTED_COMBINE_FFN_FUSED__
#define __DISTRIBUTED_COMBINE_FFN_FUSED__

#include "common.h"
#include "hccl_context.h"
#include "moe_combine.h"
#include "../tileop_common.h"

#include <type_traits>

#ifndef __TILE_FWK_HOST__
#if defined(__has_include)
#if __has_include("basic_api/kernel_operator_block_sync_intf.h")
#define PYPTO_USE_ASCENDC_BLOCK_SYNC 1
#include "basic_api/kernel_operator_block_sync_intf.h"
#endif
#endif
#endif

#if defined(PYPTO_USE_ASCENDC_BLOCK_SYNC)
#define PYPTO_CROSS_CORE_SET(mode, pipe, flag) AscendC::CrossCoreSetFlag<mode, pipe>(flag)
#define PYPTO_CROSS_CORE_WAIT(mode, flag) AscendC::CrossCoreWaitFlag<mode>(flag)
#define PYPTO_SYNC_ALL() AscendC::SyncAll<true>()
#else
#define PYPTO_CROSS_CORE_SET(mode, pipe, flag) ((void)(flag))
#define PYPTO_CROSS_CORE_WAIT(mode, flag) ((void)(flag))
#define PYPTO_SYNC_ALL() do { } while (0)
#endif

#ifndef PYPTO_FFN_FUSED_CROSS_CORE
#if defined(PYPTO_USE_ASCENDC_BLOCK_SYNC)
#define PYPTO_FFN_FUSED_CROSS_CORE 1
#else
#define PYPTO_FFN_FUSED_CROSS_CORE 0
#endif
#endif

#ifdef SUPPORT_TILE_TENSOR
using pto::TileLeft;
using pto::TileRight;
using pto::TileAcc;
using pto::BLayout;
using pto::SLayout;
#endif

namespace TileOp::Distributed {

constexpr uint32_t COMBINE_FFN_FLAG_VALUE = 2;
constexpr uint32_t COMBINE_FFN_EXPERT_BATCH = 4;
constexpr uint32_t FFN_SILU_FLAG_OFFSET = 100;

// L0 buffer constraints (32KB each for L0A, L0B, L0C)
constexpr uint32_t L0_BUFFER_SIZE = 32 * 1024;
constexpr uint32_t CUBE_BLOCK_M = 16;
constexpr uint32_t CUBE_BLOCK_N = 16;
constexpr uint32_t CUBE_BLOCK_K = 16;

struct MoeCombineFFNFusedParams {
    uint32_t batchSize;
    uint32_t hiddenSize;
    uint32_t intermediateSize;
    uint32_t topK;
    uint32_t expertNum;
    uint32_t expertPerRank;
    uint32_t rankNum;
    uint32_t rankId;
};

struct FFNWorkspace {
    uint64_t gateResultOffset;
    uint64_t upResultOffset;
    uint64_t intermediateOffset;
    uint64_t totalSize;
};

#ifndef __TILE_FWK_HOST__
#if defined(SUPPORT_TILE_TENSOR) && defined(__DAV_C220_CUBE__)
// Tiled matmul C = A @ B using pto-isa.
template <typename T, typename AccT, uint16_t tileM, uint16_t tileK, uint16_t tileN>
TILEOP void FFNTiledMatMul(
    __gm__ T* output,           // [M, N]
    __gm__ T* input,            // [M, K]
    __gm__ T* weight,           // [K, N]
    uint32_t M,
    uint32_t K,
    uint32_t N,
    uint64_t inputStride,       // stride for input rows
    uint64_t weightStride,      // stride for weight rows
    uint64_t outputStride)      // stride for output rows
{
    // Tile types.
    using TileL0A = pto::TileLeft<T, tileM, tileK, -1, -1>;
    using TileL0B = pto::TileRight<T, tileK, tileN, -1, -1>;
    using TileL0C = pto::TileAcc<AccT, tileM, tileN, -1, -1>;

    // L1 tile types for staging.
    using TileL1Mat = pto::Tile<pto::TileType::Mat, T, tileM, tileK, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    using TileL1Weight = pto::Tile<pto::TileType::Mat, T, tileK, tileN, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    // Global tensor types.
    using GlobalShape = pto::Shape<1, 1, 1, -1, -1>;
    using GlobalStride = pto::Stride<1, 1, 1, -1, -1>;
    using GlobalInput = pto::GlobalTensor<T, GlobalShape, GlobalStride, pto::Layout::ND>;
    using GlobalWeight = pto::GlobalTensor<T, GlobalShape, GlobalStride, pto::Layout::ND>;
    using GlobalOutput = pto::GlobalTensor<T, GlobalShape, GlobalStride, pto::Layout::ND>;

    T l1InputBuf[tileM * tileK];
    T l1WeightBuf[tileK * tileN];
    T l0aBuf[tileM * tileK];
    T l0bBuf[tileK * tileN];
    AccT l0cBuf[tileM * tileN];

    // Iterate over M dimension in tiles.
    for (uint32_t mTile = 0; mTile < M; mTile += tileM) {
        uint32_t curM = (mTile + tileM <= M) ? tileM : (M - mTile);
        // Align M to CUBE_BLOCK_M for TMATMUL requirements.
        uint32_t alignedM = AlignUp<uint32_t>(curM, CUBE_BLOCK_M);

        // Iterate over N dimension in tiles.
        for (uint32_t nTile = 0; nTile < N; nTile += tileN) {
            uint32_t curN = (nTile + tileN <= N) ? tileN : (N - nTile);

            // Initialize L0C accumulator.
            TileL0C l0c(alignedM, curN);
            pto::TASSIGN(l0c, reinterpret_cast<uint64_t>(l0cBuf));

            // Iterate over K dimension in tiles (accumulate).
            for (uint32_t kTile = 0; kTile < K; kTile += tileK) {
                uint32_t curK = (kTile + tileK <= K) ? tileK : (K - kTile);

                // Load input tile [curM, curK] from GM to L1.
                TileL1Mat l1Input(alignedM, curK);
                pto::TASSIGN(l1Input, reinterpret_cast<uint64_t>(l1InputBuf));
                GlobalInput globalInput(
                    input + mTile * inputStride + kTile,
                    GlobalShape(curM, curK),
                    GlobalStride(inputStride, 1));
                pto::TLOAD(l1Input, globalInput);

                // Load weight tile [curK, curN] from GM to L1.
                TileL1Weight l1Weight(curK, curN);
                pto::TASSIGN(l1Weight, reinterpret_cast<uint64_t>(l1WeightBuf));
                GlobalWeight globalWeight(
                    weight + kTile * weightStride + nTile,
                    GlobalShape(curK, curN),
                    GlobalStride(weightStride, 1));
                pto::TLOAD(l1Weight, globalWeight);

                // Extract from L1 to L0A/L0B.
                TileL0A l0a(alignedM, curK);
                TileL0B l0b(curK, curN);
                pto::TASSIGN(l0a, reinterpret_cast<uint64_t>(l0aBuf));
                pto::TASSIGN(l0b, reinterpret_cast<uint64_t>(l0bBuf));
                pto::TEXTRACT(l0a, l1Input, 0, 0);
                pto::TEXTRACT(l0b, l1Weight, 0, 0);

                // MatMul with accumulation: l0c += l0a @ l0b.
                if (kTile == 0) {
                    pto::TMATMUL(l0c, l0a, l0b);
                } else {
                    pto::TMATMUL_ACC(l0c, l0c, l0a, l0b);
                }
            }

            // Store result from L0C to GM (only store valid curM rows).
            GlobalOutput globalOutput(
                output + mTile * outputStride + nTile,
                GlobalShape(curM, curN),
                GlobalStride(outputStride, 1));
            pto::TSTORE(globalOutput, l0c);
        }
    }
}
#endif // SUPPORT_TILE_TENSOR && __DAV_C220_CUBE__
#endif // !__TILE_FWK_HOST__

template <typename T, uint32_t topK, uint16_t colShape, uint16_t paddedColShape>
struct MoeCombineFFNFusedContext {
    __gm__ T* ffnOutput;
    __gm__ T* workspace;
    __ubuf__ float* mulFp32Buffer;
    __ubuf__ float* sumFp32Buffer;
    __ubuf__ T* outBuffer;
    __ubuf__ float* expertScales;
    __gm__ T* ffnWeight;
    __gm__ int32_t* recvCounts;
    __gm__ T* shmemDataBaseAddr;
    __gm__ int32_t* shmemSignalBaseAddr;
    __gm__ int64_t* hcclContext;
    uint64_t thisRankId;
    int64_t rowOffset;
    uint16_t rowShape;
    uint16_t intermediateSize;
};

#ifndef __TILE_FWK_HOST__

template <typename T, uint32_t topK, uint16_t colShape, uint16_t paddedColShape, uint16_t intermediateSize>
TILEOP void MoeCombineFFNFusedAIV(
    MoeCombineFFNFusedContext<T, topK, colShape, paddedColShape>& ctx,
    uint32_t expertIdx)
{
    constexpr uint32_t expertScalesColShape = AlignUp<uint32_t>(sizeof(float) * topK, COPY_BLOCK_BYTE_SIZE) /
        sizeof(float);

    (void)expertIdx;
    __ubuf__ int32_t* signalBuffer = reinterpret_cast<__ubuf__ int32_t*>(ctx.outBuffer);
    uint64_t workspaceStride = static_cast<uint64_t>(colShape) + 3ULL * intermediateSize;
    __gm__ T* workspaceBase = ctx.workspace + static_cast<uint64_t>(ctx.rowOffset) * workspaceStride;

    for (uint64_t tokenId = ctx.rowOffset; tokenId < ctx.rowOffset + ctx.rowShape; tokenId++) {
        uint64_t localTokenId = tokenId - ctx.rowOffset;
        __gm__ int32_t* winSignalAddr = MapVirtualAddr<int32_t>(ctx.hcclContext, ctx.shmemSignalBaseAddr, ctx.thisRankId) +
            MOE_COMBINE_SIGNAL_OFFSET * tokenId;
        int32_t expectedValue = ctx.recvCounts[tokenId];
        int32_t maskVals[topK];
        for (uint32_t idx = 0; idx < topK; ++idx) {
            maskVals[idx] = 0;
        }

        if (expectedValue > 0) {
            MoeDistributedCombineWaitSignal(winSignalAddr, signalBuffer, expectedValue);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            copy_gm_to_ubuf(signalBuffer, winSignalAddr, 0, 1, 2, 0, 0);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            for (uint32_t idx = 0; idx < topK; ++idx) {
                maskVals[idx] = signalBuffer[1 + idx];
            }
        }

        __gm__ T* winDataAddr = MapVirtualAddr<T>(ctx.hcclContext, ctx.shmemDataBaseAddr, ctx.thisRankId) +
            colShape * topK * tokenId;
        MoeDistributedCombineCompute<T, topK, colShape, paddedColShape>(ctx.outBuffer, ctx.mulFp32Buffer,
            ctx.sumFp32Buffer, ctx.expertScales + expertScalesColShape * tokenId, winDataAddr, maskVals);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TileOp::UBCopyOut<T, 1, colShape, colShape, paddedColShape>(workspaceBase + colShape * localTokenId,
            ctx.outBuffer);
    }
}

// AIC FFN matmul stage for one expert batch.
template <typename T, uint32_t topK, uint16_t colShape, uint16_t paddedColShape, uint16_t intermediateSize>
TILEOP void MoeCombineFFNFusedAIC(
    MoeCombineFFNFusedContext<T, topK, colShape, paddedColShape>& ctx,
    uint32_t expertIdx,
    uint32_t tokenStart,
    uint32_t tokenCount)
{
    if (tokenCount == 0) {
        return;
    }

    const uint32_t M = tokenCount;
    const uint32_t K = colShape;
    const uint32_t N = intermediateSize;

    uint64_t workspaceStride = static_cast<uint64_t>(colShape) + 3ULL * intermediateSize;
    __gm__ T* workspaceBase = ctx.workspace + static_cast<uint64_t>(ctx.rowOffset) * workspaceStride;
    uint64_t combineSize = static_cast<uint64_t>(ctx.rowShape) * colShape;
    uint64_t localTokenStart = static_cast<uint64_t>(tokenStart) - static_cast<uint64_t>(ctx.rowOffset);
    __gm__ T* inputPtr = workspaceBase + localTokenStart * colShape;
    uint64_t gateOffset = 0;
    uint64_t upOffset = static_cast<uint64_t>(colShape) * intermediateSize;
    uint64_t downOffset = 2ULL * static_cast<uint64_t>(colShape) * intermediateSize;
    __gm__ T* gateWeightPtr = ctx.ffnWeight + gateOffset;
    __gm__ T* upWeightPtr = ctx.ffnWeight + upOffset;
    __gm__ T* downWeightPtr = ctx.ffnWeight + downOffset;
    __gm__ T* outputPtr = ctx.ffnOutput + static_cast<uint64_t>(tokenStart) * colShape;

    __gm__ T* gateResultPtr = workspaceBase + combineSize;
    __gm__ T* upResultPtr = gateResultPtr + M * N;
    __gm__ T* intermediatePtr = upResultPtr + M * N;

    (void)gateResultPtr;
    (void)upResultPtr;
    (void)intermediatePtr;

    // Stage 1: gate/up matmul.
#ifdef SUPPORT_TILE_TENSOR
#ifdef __DAV_C220_CUBE__
    // Tile sizes chosen to fit L0 buffers.
    constexpr uint16_t tileM = 16;
    constexpr uint16_t tileK = 64;
    constexpr uint16_t tileN = 64;

    // Determine accumulator type based on input type.
    using AccType = std::conditional_t<
        std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>,
        float,
        T>;

    // Gate projection.
    FFNTiledMatMul<T, AccType, tileM, tileK, tileN>(
        gateResultPtr,      // output [M, N]
        inputPtr,           // input  [M, K]
        gateWeightPtr,      // weight [K, N]
        M, K, N,
        K,                  // inputStride = K (row-major input)
        N,                  // weightStride = N (row-major weight)
        N);                 // outputStride = N (row-major output)

    // Up projection.
    FFNTiledMatMul<T, AccType, tileM, tileK, tileN>(
        upResultPtr,        // output [M, N]
        inputPtr,           // input  [M, K]
        upWeightPtr,        // weight [K, N]
        M, K, N,
        K,                  // inputStride
        N,                  // weightStride
        N);                 // outputStride
#endif
#endif

    // Stage 2: signal AIV for SiLU.
#ifdef __DAV_C220_CUBE__
    PYPTO_SYNC_ALL();
    PYPTO_CROSS_CORE_SET(0x2, PIPE_MTE3, FFN_SILU_FLAG_OFFSET + expertIdx);
#endif

    // Stage 3: wait SiLU, then down projection.
#ifdef __DAV_C220_CUBE__
    PYPTO_CROSS_CORE_WAIT(0x2, FFN_SILU_FLAG_OFFSET + expertIdx + 1);
#endif

#ifdef SUPPORT_TILE_TENSOR
#ifdef __DAV_C220_CUBE__
    // Down projection.
    FFNTiledMatMul<T, AccType, tileM, tileN, tileK>(
        outputPtr,          // output [M, K]
        intermediatePtr,    // input  [M, N]
        downWeightPtr,      // weight [N, K]
        M, N, K,
        N,                  // inputStride = N
        K,                  // weightStride = K
        K);                 // outputStride = K
#endif
#endif
}

template <typename T>
TILEOP void MoeCombineFFNSiLUFusionDispatch(
    __gm__ T* intermediate,
    __gm__ T* gateResult,
    __gm__ T* upResult,
    __ubuf__ float* ubBuffer,
    uint64_t elementCount);

// Single-core FFN stage: gate/up -> SiLU -> down.
template <typename T, uint32_t topK, uint16_t colShape, uint16_t paddedColShape, uint16_t intermediateSize>
TILEOP void MoeCombineFFNFusedAICSingleCore(
    MoeCombineFFNFusedContext<T, topK, colShape, paddedColShape>& ctx,
    uint32_t tokenStart,
    uint32_t tokenCount)
{
#ifdef __DAV_C220_CUBE__
    if (tokenCount == 0) {
        return;
    }

    const uint32_t M = tokenCount;
    const uint32_t K = colShape;
    const uint32_t N = intermediateSize;

    uint64_t workspaceStride = static_cast<uint64_t>(colShape) + 3ULL * intermediateSize;
    __gm__ T* workspaceBase = ctx.workspace + static_cast<uint64_t>(ctx.rowOffset) * workspaceStride;
    uint64_t combineSize = static_cast<uint64_t>(ctx.rowShape) * colShape;
    uint64_t localTokenStart = static_cast<uint64_t>(tokenStart) - static_cast<uint64_t>(ctx.rowOffset);
    __gm__ T* inputPtr = workspaceBase + localTokenStart * colShape;
    uint64_t gateOffset = 0;
    uint64_t upOffset = static_cast<uint64_t>(colShape) * intermediateSize;
    uint64_t downOffset = 2ULL * static_cast<uint64_t>(colShape) * intermediateSize;
    __gm__ T* gateWeightPtr = ctx.ffnWeight + gateOffset;
    __gm__ T* upWeightPtr = ctx.ffnWeight + upOffset;
    __gm__ T* downWeightPtr = ctx.ffnWeight + downOffset;
    __gm__ T* outputPtr = ctx.ffnOutput + static_cast<uint64_t>(tokenStart) * colShape;

    __gm__ T* gateResultPtr = workspaceBase + combineSize;
    __gm__ T* upResultPtr = gateResultPtr + M * N;
    __gm__ T* intermediatePtr = upResultPtr + M * N;

    // Stage 1: gate/up matmul.
#ifdef SUPPORT_TILE_TENSOR
    // Tile sizes chosen to fit L0 buffers.
    constexpr uint16_t tileM = 16;
    constexpr uint16_t tileK = 64;
    constexpr uint16_t tileN = 64;

    // Determine accumulator type based on input type.
    using AccType = std::conditional_t<
        std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>,
        float,
        T>;

    FFNTiledMatMul<T, AccType, tileM, tileK, tileN>(
        gateResultPtr,      // output [M, N]
        inputPtr,           // input  [M, K]
        gateWeightPtr,      // weight [K, N]
        M, K, N,
        K,                  // inputStride = K (row-major input)
        N,                  // weightStride = N (row-major weight)
        N);                 // outputStride = N (row-major output)

    FFNTiledMatMul<T, AccType, tileM, tileK, tileN>(
        upResultPtr,        // output [M, N]
        inputPtr,           // input  [M, K]
        upWeightPtr,        // weight [K, N]
        M, K, N,
        K,                  // inputStride
        N,                  // weightStride
        N);                 // outputStride
#endif

    // Stage 2: SiLU fusion on vector pipeline.
    MoeCombineFFNSiLUFusionDispatch<T>(
        intermediatePtr, gateResultPtr, upResultPtr, ctx.mulFp32Buffer, M * N);

    // Stage 3: down projection.
#ifdef SUPPORT_TILE_TENSOR
    FFNTiledMatMul<T, AccType, tileM, tileN, tileK>(
        outputPtr,          // output [M, K]
        intermediatePtr,    // input  [M, N]
        downWeightPtr,      // weight [N, K]
        M, N, K,
        N,                  // inputStride = N
        K,                  // weightStride = K
        K);                 // outputStride = K
#endif
#else
    (void)ctx;
    (void)tokenStart;
    (void)tokenCount;
#endif
}

// AIV SiLU fusion: intermediate = SiLU(gate) * up.
template <typename T, uint16_t tileSize>
TILEOP void MoeCombineFFNSiLUFusionTiles(
    __gm__ T* intermediate,
    __gm__ T* gateResult,
    __gm__ T* upResult,
    __ubuf__ float* ubBuffer,
    uint64_t elementCount)
{
    constexpr uint16_t paddedTileSize = AlignUp<uint16_t>(tileSize, VECTOR_INSTRUCTION_BYTE_SIZE / sizeof(float));
    uint8_t repeat = static_cast<uint8_t>(paddedTileSize * sizeof(float) / VECTOR_INSTRUCTION_BYTE_SIZE);

    __ubuf__ float* gateBuffer = ubBuffer;
    __ubuf__ float* upBuffer = ubBuffer + paddedTileSize;
    __ubuf__ float* sigmoidBuffer = ubBuffer + 2 * paddedTileSize;
    __ubuf__ T* inputBuffer = reinterpret_cast<__ubuf__ T*>(ubBuffer + 3 * paddedTileSize);
    __ubuf__ T* outputBuffer = reinterpret_cast<__ubuf__ T*>(ubBuffer + 3 * paddedTileSize + paddedTileSize / 2);

    if (elementCount == 0 || elementCount % tileSize != 0) {
        return;
    }

    for (uint64_t offset = 0; offset < elementCount; offset += tileSize) {
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        TileOp::UBCopyIn<T, 1, tileSize, paddedTileSize / 2, tileSize>(inputBuffer, gateResult + offset);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        vconv_bf162f32(gateBuffer, inputBuffer, repeat, 1, 1, 8, 4);

        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TileOp::UBCopyIn<T, 1, tileSize, paddedTileSize / 2, tileSize>(inputBuffer, upResult + offset);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        vconv_bf162f32(upBuffer, inputBuffer, repeat, 1, 1, 8, 4);

        pipe_barrier(PIPE_V);
        vmuls(sigmoidBuffer, gateBuffer, -1.0f, repeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vexp(sigmoidBuffer, sigmoidBuffer, repeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vadds(sigmoidBuffer, sigmoidBuffer, 1.0f, repeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vdiv(sigmoidBuffer, gateBuffer, sigmoidBuffer, repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(sigmoidBuffer, sigmoidBuffer, upBuffer, repeat, 1, 1, 1, 8, 8, 8);

        pipe_barrier(PIPE_V);
        vconv_f322bf16a(outputBuffer, sigmoidBuffer, repeat, 1, 1, 4, 8);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TileOp::UBCopyOut<T, 1, tileSize, tileSize, paddedTileSize / 2>(intermediate + offset, outputBuffer);
    }
}

template <typename T>
TILEOP void MoeCombineFFNSiLUFusionDispatch(
    __gm__ T* intermediate,
    __gm__ T* gateResult,
    __gm__ T* upResult,
    __ubuf__ float* ubBuffer,
    uint64_t elementCount)
{
    uint64_t offset = 0;
    uint64_t remaining = elementCount;

    if (remaining >= 1024) {
        uint64_t block = (remaining / 1024) * 1024;
        MoeCombineFFNSiLUFusionTiles<T, 1024>(intermediate + offset, gateResult + offset, upResult + offset,
            ubBuffer, block);
        offset += block;
        remaining -= block;
    }
    if (remaining >= 256) {
        uint64_t block = (remaining / 256) * 256;
        MoeCombineFFNSiLUFusionTiles<T, 256>(intermediate + offset, gateResult + offset, upResult + offset,
            ubBuffer, block);
        offset += block;
        remaining -= block;
    }
    if (remaining >= 64) {
        uint64_t block = (remaining / 64) * 64;
        MoeCombineFFNSiLUFusionTiles<T, 64>(intermediate + offset, gateResult + offset, upResult + offset,
            ubBuffer, block);
        offset += block;
        remaining -= block;
    }
    if (remaining >= 16) {
        uint64_t block = (remaining / 16) * 16;
        MoeCombineFFNSiLUFusionTiles<T, 16>(intermediate + offset, gateResult + offset, upResult + offset,
            ubBuffer, block);
        offset += block;
        remaining -= block;
    }
    if (remaining != 0) {
        return;
    }
}

template <typename T, uint32_t topK, uint16_t rowShape, uint16_t colShape, uint16_t paddedColShape, uint16_t intermediateSize>
TILEOP void MoeCombineFFNFusedKernel(
    __gm__ T* ffnOutput,
    __gm__ T* workspace,
    __ubuf__ float* mulFp32Buffer,
    __ubuf__ float* sumFp32Buffer,
    __ubuf__ T* outBuffer,
    __ubuf__ float* expertScales,
    __gm__ T* ffnWeight,
    __gm__ int32_t* recvCounts,
    __gm__ T* shmemDataBaseAddr,
    __gm__ int32_t* shmemSignalBaseAddr,
    uint64_t shmemDataOffset0,
    uint64_t shmemDataOffset1,
    uint64_t shmemDataOffset2,
    uint64_t shmemDataOffset3,
    int64_t rowOffset,
    __gm__ int64_t* hcclContext)
{
    (void)shmemDataOffset1;
    (void)shmemDataOffset2;
    (void)shmemDataOffset3;
    MoeCombineFFNFusedContext<T, topK, colShape, paddedColShape> ctx;
    ctx.ffnOutput = ffnOutput;
    ctx.workspace = workspace;
    ctx.mulFp32Buffer = mulFp32Buffer;
    ctx.sumFp32Buffer = sumFp32Buffer;
    ctx.outBuffer = outBuffer;
    ctx.expertScales = expertScales;
    ctx.ffnWeight = ffnWeight;
    ctx.recvCounts = recvCounts;
    ctx.shmemDataBaseAddr = shmemDataBaseAddr;
    ctx.shmemSignalBaseAddr = shmemSignalBaseAddr;
    ctx.hcclContext = hcclContext;
    ctx.thisRankId = shmemDataOffset0;
    ctx.rowOffset = rowOffset;
    ctx.rowShape = rowShape;
    ctx.intermediateSize = intermediateSize;

    const uint64_t M = rowShape;
    const uint64_t N = intermediateSize;

    uint32_t expertIdx = 0;
#if PYPTO_FFN_FUSED_CROSS_CORE
#if defined(__DAV_C220_VEC__)
    MoeCombineFFNFusedAIV<T, topK, colShape, paddedColShape, intermediateSize>(ctx, expertIdx);
    PYPTO_SYNC_ALL();
    PYPTO_CROSS_CORE_SET(0x2, PIPE_MTE3, expertIdx + 1);

    PYPTO_CROSS_CORE_WAIT(0x2, FFN_SILU_FLAG_OFFSET + expertIdx);

    uint64_t combineSize = static_cast<uint64_t>(rowShape) * colShape;
    __gm__ T* gateResultPtr = workspace + combineSize;
    __gm__ T* upResultPtr = gateResultPtr + M * N;
    __gm__ T* intermediatePtr = upResultPtr + M * N;

    MoeCombineFFNSiLUFusionDispatch<T>(
        intermediatePtr, gateResultPtr, upResultPtr, mulFp32Buffer, M * N);

    PYPTO_SYNC_ALL();
    PYPTO_CROSS_CORE_SET(0x2, PIPE_MTE3, FFN_SILU_FLAG_OFFSET + expertIdx + 1);
#endif
#if defined(__DAV_C220_CUBE__)
    PYPTO_CROSS_CORE_WAIT(0x2, expertIdx + 1);
    MoeCombineFFNFusedAIC<T, topK, colShape, paddedColShape, intermediateSize>(
        ctx, expertIdx, static_cast<uint32_t>(rowOffset), rowShape);
#endif
#else
#if defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
    MoeCombineFFNFusedAIV<T, topK, colShape, paddedColShape, intermediateSize>(ctx, expertIdx);
#endif
#if defined(__DAV_C220_CUBE__)
    MoeCombineFFNFusedAICSingleCore<T, topK, colShape, paddedColShape, intermediateSize>(
        ctx, static_cast<uint32_t>(rowOffset), rowShape);
#else
    (void)M;
    (void)N;
    (void)expertIdx;
#endif
#endif
}

#endif // !__TILE_FWK_HOST__

} // namespace TileOp::Distributed

#endif // __DISTRIBUTED_COMBINE_FFN_FUSED__
