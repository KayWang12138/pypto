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
 * \brief MoE FFN kernel for combined-output input (AIC only, no cross-core protocol).
 */

#ifndef __DISTRIBUTED_COMBINE_FFN_FUSED__
#define __DISTRIBUTED_COMBINE_FFN_FUSED__

#include "common.h"
#include "../tileop_common.h"

#include <type_traits>

#ifdef SUPPORT_TILE_TENSOR
using pto::TileLeft;
using pto::TileRight;
using pto::TileAcc;
using pto::BLayout;
using pto::SLayout;
#endif

namespace TileOp::Distributed {

// L0 buffer constraints (32KB each for L0A, L0B, L0C)
constexpr uint32_t L0_BUFFER_SIZE = 32 * 1024;
constexpr uint32_t CUBE_BLOCK_M = 16;
constexpr uint32_t CUBE_BLOCK_N = 16;
constexpr uint32_t CUBE_BLOCK_K = 16;

#ifndef __TILE_FWK_HOST__
#if defined(SUPPORT_TILE_TENSOR) && defined(__AIC__)
// Tiled matmul C = A @ B using dynamic arch32 cube primitives.
template <typename T, typename AccT, uint16_t tileM, uint16_t tileK, uint16_t tileN>
TILEOP void FFNTiledMatMul(
    __gm__ T* output,
    __gm__ T* input,
    __gm__ T* weight,
    uint32_t M,
    uint32_t K,
    uint32_t N,
    uint64_t inputStride,
    uint64_t weightStride,
    uint64_t outputStride)
{
    __cbuf__ T* l1InputBuf = reinterpret_cast<__cbuf__ T*>(get_imm(0x0000));
    __cbuf__ T* l1WeightBuf = reinterpret_cast<__cbuf__ T*>(get_imm(0x1000));
    __ca__ T* l0aBuf = reinterpret_cast<__ca__ T*>(get_imm(0x0000));
    __cb__ T* l0bBuf = reinterpret_cast<__cb__ T*>(get_imm(0x0000));
    __cc__ AccT* l0cBuf = reinterpret_cast<__cc__ AccT*>(get_imm(0x0000));

    // Reinitialize local event state to avoid stale PIPE_M/PIPE_MTE1/PIPE_FIX handshakes across invocations.
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID2);

    for (uint32_t mTile = 0; mTile < M; mTile += tileM) {
        uint32_t curM = (mTile + tileM <= M) ? tileM : (M - mTile);
        uint32_t alignedM = AlignUp<uint32_t>(curM, CUBE_BLOCK_M);

        for (uint32_t nTile = 0; nTile < N; nTile += tileN) {
            uint32_t curN = (nTile + tileN <= N) ? tileN : (N - nTile);

            for (uint32_t kTile = 0; kTile < K; kTile += tileK) {
                uint32_t curK = (kTile + tileK <= K) ? tileK : (K - kTile);

                TileOp::DynL1CopyIn<T, T, CopyInMode::ND2NZ>(
                    l1InputBuf, input, curM, curK, M, inputStride, mTile, kTile, 0);
                TileOp::DynL1CopyIn<T, T, CopyInMode::ND2NZ>(
                    l1WeightBuf, weight, curK, curN, K, weightStride, kTile, nTile, 0);

                TileOp::DynL1ToL0A<T, 0, 0>(l0aBuf, l1InputBuf, curM, curK, curM, curK);
                TileOp::DynL1ToL0B<T, 0, 0>(l0bBuf, l1WeightBuf, curK, curN, curK, curN);

                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);

                bool isAcc = (kTile != 0);
                TileOp::DynTmad<AccT, T, T, 0, 0>(
                    l0cBuf, l0aBuf, l0bBuf, curM, curK, curN, isAcc, 0, alignedM, curN);

                set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
                wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
            }

            set_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID2);

            TileOp::DynL0CCopyOut<T, AccT, true, 0>(
                output, l0cBuf, curM, curN, M, outputStride, mTile, nTile, M, outputStride, 0);

            set_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
        }
    }
}
#endif // SUPPORT_TILE_TENSOR && __AIC__
#endif // !__TILE_FWK_HOST__

#ifndef __TILE_FWK_HOST__

// SiLU fusion: intermediate = SiLU(gate) * up.
#if defined(__DAV_C220_VEC__)
template <typename T, uint16_t tileSize>
TILEOP void MoeFfnSiLUFusionTiles(
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
TILEOP void MoeFfnSiLUFusionDispatch(
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
        MoeFfnSiLUFusionTiles<T, 1024>(intermediate + offset, gateResult + offset, upResult + offset,
            ubBuffer, block);
        offset += block;
        remaining -= block;
    }
    if (remaining >= 256) {
        uint64_t block = (remaining / 256) * 256;
        MoeFfnSiLUFusionTiles<T, 256>(intermediate + offset, gateResult + offset, upResult + offset,
            ubBuffer, block);
        offset += block;
        remaining -= block;
    }
    if (remaining >= 64) {
        uint64_t block = (remaining / 64) * 64;
        MoeFfnSiLUFusionTiles<T, 64>(intermediate + offset, gateResult + offset, upResult + offset,
            ubBuffer, block);
        offset += block;
        remaining -= block;
    }
    if (remaining >= 16) {
        uint64_t block = (remaining / 16) * 16;
        MoeFfnSiLUFusionTiles<T, 16>(intermediate + offset, gateResult + offset, upResult + offset,
            ubBuffer, block);
        offset += block;
        remaining -= block;
    }
    if (remaining != 0) {
        return;
    }
}
#else
template <typename T>
INLINE float MoeFfnToFp32(T val)
{
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        return Bf16ToFp32(val);
    } else {
        return static_cast<float>(val);
    }
}

template <typename T>
INLINE T MoeFfnFromFp32(float val)
{
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        return Fp32ToBf16R(val);
    } else {
        return static_cast<T>(val);
    }
}

template <typename T>
TILEOP void MoeFfnSiLUFusionDispatch(
    __gm__ T* intermediate,
    __gm__ T* gateResult,
    __gm__ T* upResult,
    __ubuf__ float* ubBuffer,
    uint64_t elementCount)
{
    (void)ubBuffer;
    for (uint64_t i = 0; i < elementCount; ++i) {
        float gateVal = MoeFfnToFp32(gateResult[i]);
        float upVal = MoeFfnToFp32(upResult[i]);
        float absGate = gateVal >= 0.0f ? gateVal : -gateVal;
        float sigmoid = 0.5f * (gateVal / (1.0f + absGate) + 1.0f);
        intermediate[i] = MoeFfnFromFp32<T>(gateVal * sigmoid * upVal);
    }
}
#endif

template <typename T, uint16_t colShape, uint16_t intermediateSize>
struct MoeFfnFusedContext {
    __gm__ T* ffnOutput;
    __gm__ T* workspace;
    __ubuf__ float* ubBuffer;
    __gm__ T* combineInput;
    __gm__ T* ffnWeight;
};

// Single-core FFN stage: gate/up -> SiLU -> down.
template <typename T, uint16_t colShape, uint16_t intermediateSize>
TILEOP void MoeFfnFusedAICSingleCore(
    MoeFfnFusedContext<T, colShape, intermediateSize>& ctx,
    uint32_t tokenCount)
{
#ifdef __DAV_C220_CUBE__
    if (tokenCount == 0) {
        return;
    }

    const uint32_t M = tokenCount;
    const uint32_t K = colShape;
    const uint32_t N = intermediateSize;
    const uint32_t alignedM = AlignUp<uint32_t>(M, CUBE_BLOCK_M);

    __gm__ T* inputPtr = ctx.combineInput;
    uint64_t gateOffset = 0;
    uint64_t upOffset = static_cast<uint64_t>(colShape) * intermediateSize;
    uint64_t downOffset = 2ULL * static_cast<uint64_t>(colShape) * intermediateSize;
    __gm__ T* gateWeightPtr = ctx.ffnWeight + gateOffset;
    __gm__ T* upWeightPtr = ctx.ffnWeight + upOffset;
    __gm__ T* downWeightPtr = ctx.ffnWeight + downOffset;
    __gm__ T* outputPtr = ctx.ffnOutput;

    __gm__ T* paddedInputPtr = ctx.workspace;
    __gm__ T* gateResultPtr = paddedInputPtr + static_cast<uint64_t>(alignedM) * K;
    __gm__ T* upResultPtr = gateResultPtr + static_cast<uint64_t>(alignedM) * N;
    __gm__ T* intermediatePtr = upResultPtr + static_cast<uint64_t>(alignedM) * N;

#ifdef SUPPORT_TILE_TENSOR
    constexpr uint16_t tileM = 16;
    constexpr uint16_t tileK = 64;
    constexpr uint16_t tileN = 64;

    using AccType = std::conditional_t<
        std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>,
        float,
        T>;

    for (uint32_t m = 0; m < alignedM; ++m) {
        uint64_t srcBase = static_cast<uint64_t>(m) * K;
        for (uint32_t k = 0; k < K; ++k) {
            paddedInputPtr[srcBase + k] = (m < M) ? inputPtr[srcBase + k] : MoeFfnFromFp32<T>(0.0f);
        }
    }

    FFNTiledMatMul<T, AccType, tileM, tileK, tileN>(
        gateResultPtr,
        paddedInputPtr,
        gateWeightPtr,
        alignedM,
        K,
        N,
        K,
        N,
        N);

    FFNTiledMatMul<T, AccType, tileM, tileK, tileN>(
        upResultPtr,
        paddedInputPtr,
        upWeightPtr,
        alignedM,
        K,
        N,
        K,
        N,
        N);
#endif

    // Ensure gate/up GM writes from MTE3 are visible before scalar SiLU reads.
    pipe_barrier(PIPE_MTE3);
    pipe_barrier(PIPE_ALL);

    MoeFfnSiLUFusionDispatch<T>(
        intermediatePtr, gateResultPtr, upResultPtr, ctx.ubBuffer, static_cast<uint64_t>(M) * N);

    // Ensure scalar GM writes of intermediate are visible before down matmul MTE2 loads.
    pipe_barrier(PIPE_ALL);

#ifdef SUPPORT_TILE_TENSOR
    FFNTiledMatMul<T, AccType, tileM, tileN, tileK>(
        outputPtr,
        intermediatePtr,
        downWeightPtr,
        M,
        N,
        K,
        N,
        K,
        K);
#endif
#else
    (void)ctx;
    (void)tokenCount;
#endif
}

template <typename T, uint16_t rowShape, uint16_t colShape, uint16_t paddedColShape, uint16_t intermediateSize>
TILEOP void MoeFfnFusedKernel(
    __gm__ T* ffnOutput,
    __gm__ T* workspace,
    __ubuf__ float* ubBuffer,
    __gm__ T* combineInput,
    __gm__ T* ffnWeight,
    __gm__ int64_t* hcclContext)
{
    (void)paddedColShape;
    (void)hcclContext;

    MoeFfnFusedContext<T, colShape, intermediateSize> ctx;
    ctx.ffnOutput = ffnOutput;
    ctx.workspace = workspace;
    ctx.ubBuffer = ubBuffer;
    ctx.combineInput = combineInput;
    ctx.ffnWeight = ffnWeight;

    MoeFfnFusedAICSingleCore<T, colShape, intermediateSize>(ctx, rowShape);
}

#endif // !__TILE_FWK_HOST__

} // namespace TileOp::Distributed

#endif // __DISTRIBUTED_COMBINE_FFN_FUSED__
