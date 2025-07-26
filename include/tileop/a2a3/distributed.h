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
 * \file distributed.h
 * \brief
 */

#ifndef __LOGICALTENSOR_TILEOP_DIST__
#define __LOGICALTENSOR_TILEOP_DIST__

#include "tileop_common.h"
#include "hccl_context.h"

#include <type_traits>

namespace TileOp {
namespace Distributed {
// 以下 ATOMIC_ADD_BLOCK_BYTE_SIZE 和 FLAG_BYTE_SIZE 的定义与 comm_wait_flag.h 中的定义一致
constexpr uint32_t ATOMIC_ADD_BLOCK_BYTE_SIZE = 32; // AtomicAdd 每次操作 32B 的数据，对同一 32B 的数据进行 AtomicAdd 需要排队
constexpr uint32_t FLAG_BYTE_SIZE = ATOMIC_ADD_BLOCK_BYTE_SIZE * 4; // 为了消除 AtomicAdd 并发，以 32B 为最小单位，视情况调节每个 flag 占用的字节数
constexpr uint32_t COMBINE_FLAG_OFFSET = 128;   // combine算子flag的偏移512B，flag是int32类型
constexpr uint32_t COMBINE_INFO_NUM = 3;        // combine算子的输入的combineInfo信息有三个，attnId, tokenId and kOffset

#define GM_ADDR __gm__ uint8_t *
#define UB_ADDR __ubuf__ uint8_t *

template <typename T>
TILEOP void SetAttomicType()
{
    if constexpr (std::is_same_v<T, float>) {
        set_atomic_f32();
    } else if constexpr (std::is_same_v<T, half>) {
        set_atomic_f16();
    } else if constexpr (std::is_same_v<T, int16_t>) {
        set_atomic_s16();
    } else if constexpr (std::is_same_v<T, int32_t>) {
        set_atomic_s32();
    } else if constexpr (std::is_same_v<T, int8_t>) {
        set_atomic_s8();
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        set_atomic_bf16();
    }
}

struct TilingInfo {
    int tileIndex;
    int groupIndex;
    int rowPerRank;
    int colPerRank;
    int rankShape;
    int rankOffset;
    int rowShape;
    int rowOffset;
    int colShape;
    int colOffset;
    int totalTileNum;
    int shareRankCnt;
};

/* UB 清 0 */
TILEOP void ClearFlagBuf(__ubuf__ int32_t *flagBuf)
{
    /*
    每次处理 8 个 block，8 * 32 = 256B，所以使用 vector_dup 时建议 flag 内存对齐 256B
    BlockStride 是每次迭代内 block 的距离（stride，前一个头和后一个头，0 会按照 1 来处理），单位是 block
    RepeatStride 是每次迭代间 block 的距离，如果内存是连续的，值一般是 8
    */

    uint8_t repeat = 1;
    int32_t src = 0;
    uint16_t dstBlockStride = 0;
    uint16_t srcBlockStride = 0;
    uint8_t dstRepeatStride = 8;
    uint8_t srcRepeatStride = 0;
    vector_dup(flagBuf, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}

/* win 连续排布 */
TILEOP void SetFlag(__ubuf__ int32_t *flag, GM_ADDR winFlagBaseAddr, uint32_t flagOffset)
{
    /* 需要手动清空 flag 所使用的内存 */
    ClearFlagBuf(flag);

    GM_ADDR winFlagAddr = winFlagBaseAddr + flagOffset * FLAG_BYTE_SIZE;

    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    flag[0] = 1;
    /*
    copy_ubuf_to_gm 参数：
        nBurst：搬运次数，flag ub 较小，可以一次搬完
        lenBurst：每次搬运块数，单位为 block, 32B
        srcStride：前后数据块的间隔，单位为 block
        dstStride：前后数据块的间隔，单位为 block
    */
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_gm(winFlagAddr, flag, 0, 1, 1, 0, 0);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
}

TILEOP void WaitFlag(__ubuf__ int32_t *flag, GM_ADDR winFlagBaseAddr, uint32_t flagOffset)
{
    GM_ADDR winFlagAddr = winFlagBaseAddr + flagOffset * FLAG_BYTE_SIZE;
    uint32_t value = 1;
    uint32_t status = 0;
    do {
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        copy_gm_to_ubuf(flag, winFlagAddr, 0, 1, 1, 0, 0);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        status = flag[0];
    } while (status != value);
}

TILEOP void ClearFlag(__ubuf__ int32_t *flag, GM_ADDR winFlagBaseAddr, uint32_t flagOffset)
{
    GM_ADDR winFlagAddr = winFlagBaseAddr + flagOffset * FLAG_BYTE_SIZE;
    ClearFlagBuf(flag);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_gm(winFlagAddr, flag, 0, 1, 1, 0, 0);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
}

template <typename T>
TILEOP void WriteRemote(
    __ubuf__ int32_t *flag, __ubuf__ T *in, __ubuf__ int32_t *tilingData, __gm__ int64_t *hcclContext)
{
    __ubuf__ TilingInfo *tilingInfo = reinterpret_cast<__ubuf__ TilingInfo *>(tilingData);
    uint16_t nBurst = tilingInfo->rowShape;                    // 次数
    uint16_t lenBurst = tilingInfo->colShape * sizeof(T) / 32; // 块数，block 为单位, 32B，这里需要注意是否会越界
    uint16_t srcStride = 0;                                    // 前后数据块的间隔，单位为 block
    uint16_t dstStride = 0;                                    // 前后数据块的间隔，单位为 block

    // win
    __gm__ HcclCombinOpParam *winContext = (__gm__ HcclCombinOpParam *)(hcclContext[tilingInfo->groupIndex]);
    uint32_t localUsrRankId = winContext->rankId;
    uint32_t rankSize = winContext->rankNum;
    for (uint32_t remoteRankId = tilingInfo->rankOffset; remoteRankId < tilingInfo->rankShape + tilingInfo->rankOffset;
        remoteRankId++) { // tileOp 只会处理一部分 rank
        if (remoteRankId == localUsrRankId) {
            continue;
        }
        GM_ADDR remoteRankWinGMBaseAddr = (GM_ADDR)(winContext->windowsIn[remoteRankId]);
        uint64_t eachRankAddrSizeInRemoteWindow =
            tilingInfo->rowPerRank * tilingInfo->colPerRank *
            sizeof(T); // 每张卡在 win 上占据的大小，一共是一张卡的原始大小；或者 win 200M 按照 rank 均分

        GM_ADDR remoteRankWinGMAddr =
            remoteRankWinGMBaseAddr                           /* 远端 win 基地址 */
            + eachRankAddrSizeInRemoteWindow * localUsrRankId /* 当前 rank 在远端 win 的偏移地址 */
            + (tilingInfo->rowOffset * tilingInfo->colPerRank + tilingInfo->rowShape * tilingInfo->colOffset) *
                  sizeof(T) /* 当前 tile 块的偏移 */;

        copy_ubuf_to_gm(remoteRankWinGMAddr, in, 0 /* sid */, nBurst, lenBurst, srcStride, dstStride);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        /* 把 flag 写到远端卡的 win，flag 在 win 区的基地址 */
        GM_ADDR winFlagBaseAddr = (GM_ADDR)(winContext->windowsExp[remoteRankId]);
        /* flag 的偏移，flag 在 win 区连续排布 */
        uint32_t flagOffset = tilingInfo->tileIndex * rankSize + localUsrRankId; // 写到远端的本卡位置
        SetFlag(flag, winFlagBaseAddr, flagOffset);
    }

    return;
}

// WinGM 拷贝到 GM, Win区大小为200M，offset用uint32_t
template <typename T, bool enableAtomicAdd>
TILEOP void WinGMCopyOut(__gm__ T *out, __ubuf__ T *in, GM_ADDR WinBaseAddr, uint32_t offset, uint16_t nBurst,
    uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride)
{
    // 获取WinGM数据，Win区数据和UB数据是连续的，stride都为0
    copy_gm_to_ubuf(in, reinterpret_cast<__gm__ T *>(WinBaseAddr + offset), 0 /* sid */, nBurst, lenBurst, 0, 0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    if constexpr (enableAtomicAdd) {
        set_atomic_add();
        SetAttomicType<T>();
    }
    copy_ubuf_to_gm(out, in, 0 /* sid */, nBurst, lenBurst, srcStride, dstStride);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    if constexpr (enableAtomicAdd) {
        set_atomic_none();
    }
}

template <typename T, bool enableAtomicAdd, bool needWaitFlag>
TILEOP void RemoteGroupRecv(
    __gm__ T *out, __ubuf__ T *in, __ubuf__ int32_t *flag, __ubuf__ int32_t *tilingData, __gm__ int64_t *hcclContext)
{
    __ubuf__ TilingInfo *tilingInfo = reinterpret_cast<__ubuf__ TilingInfo *>(tilingData);
    uint16_t nBurst = tilingInfo->rowShape;
    uint16_t lenBurst = tilingInfo->colShape * sizeof(T) / 32;
    uint16_t srcStride = 0;
    uint16_t dstStride = ((tilingInfo->colPerRank - tilingInfo->colShape) * sizeof(T)) / 32;

    __gm__ HcclCombinOpParam *winContext = (__gm__ HcclCombinOpParam *)(hcclContext[tilingInfo->groupIndex]);
    uint32_t localUsrRankId = winContext->rankId;
    uint32_t rankSize = winContext->rankNum;
    GM_ADDR localWinGM = (GM_ADDR)(winContext->windowsIn[localUsrRankId]);
    GM_ADDR winFlagBaseAddr = (GM_ADDR)(winContext->windowsExp[localUsrRankId]); // flag 在 win 区的基地址

    uint32_t rankByteSize = tilingInfo->rowPerRank * tilingInfo->colPerRank * sizeof(T);
    uint32_t tileOffset =
        (tilingInfo->rowOffset * tilingInfo->colPerRank + tilingInfo->colOffset * tilingInfo->rowShape) * sizeof(T);
    uint32_t tileIndexOffset = tilingInfo->tileIndex * rankSize;

    for (uint32_t remoteRankId = tilingInfo->rankOffset; remoteRankId < tilingInfo->rankOffset + tilingInfo->rankShape;
        remoteRankId++) {
        if (remoteRankId == localUsrRankId) {
            continue;
        }

        uint32_t flagOffset = tileIndexOffset + remoteRankId;

        if constexpr (needWaitFlag) {
            // AIV等flag
            WaitFlag(flag, winFlagBaseAddr, flagOffset);
        }

        // 拷贝Win区数据
        uint32_t winGMOffset = rankByteSize * remoteRankId + tileOffset;
        WinGMCopyOut<T, enableAtomicAdd>(out, in, localWinGM, winGMOffset, nBurst, lenBurst, srcStride, dstStride);
        if constexpr (not enableAtomicAdd) {
            out += tilingInfo->rowShape * tilingInfo->colShape;
        }

        // 清除flag
        ClearFlag(flag, winFlagBaseAddr, flagOffset);
    }
}

template <typename T, bool needWaitFlag = false>
TILEOP void RemoteGather(
    __gm__ T *out, __ubuf__ T *in, __ubuf__ int32_t *flag, __ubuf__ int32_t *tilingData, __gm__ int64_t *hcclContext)
{
    RemoteGroupRecv<T, false, needWaitFlag>(out, in, flag, tilingData, hcclContext);
    return;
}

template <typename T, bool needWaitFlag = false>
TILEOP void RemoteReduce(__gm__ T *out, __ubuf__ T *tmp, __gm__ T *in, __ubuf__ int32_t *flag, 
    __ubuf__ int32_t *tilingData, __gm__ int64_t *hcclContext)
{
    RemoteGroupRecv<T, true, needWaitFlag>(out, tmp, flag, tilingData, hcclContext);
}

template <typename T, bool isLocalReduce = false>
TILEOP void LocalCopyOut(__gm__ T *out, __ubuf__ T *in, __ubuf__ int32_t *tilingData, __gm__ int64_t *hcclContext)
{
    __ubuf__ TilingInfo *tilingInfo = reinterpret_cast<__ubuf__ TilingInfo *>(tilingData);
    uint16_t nBurst = tilingInfo->rowShape;
    uint16_t lenBurst = tilingInfo->colShape * sizeof(T) / 32;
    uint16_t srcStride = 0;
    uint16_t dstStride = ((tilingInfo->colPerRank - tilingInfo->colShape) * sizeof(T)) / 32;

    copy_ubuf_to_gm(out, in, 0 /* sid */, nBurst, lenBurst, srcStride, dstStride);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    return;
}

TILEOP void CombineWaitFlag(__ubuf__ int32_t *flag, __gm__ int32_t *winFlagAddr, int32_t expectValue)
{
    int32_t status = 0;
    do {
        copy_gm_to_ubuf(flag, winFlagAddr, 0 /* sid */, 1, 1, 0, 0);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        status = flag[0];
    } while (status != expectValue);
    winFlagAddr[0] = 0;         // clear flag
    dcci(winFlagAddr, 0, 2);    // 0: SINGLE_CACHE_LINE, 2: CACHELINE_OUT
}

template <typename T, uint32_t topk>
TILEOP void FFN2Attn(__ubuf__ int32_t *flag, __ubuf__ T *in, __ubuf__ int32_t *combineInfo,
    __ubuf__ int32_t *tilingData, __gm__ int64_t *hcclContext)
{
    __ubuf__ TilingInfo *tilingInfo = reinterpret_cast<__ubuf__ TilingInfo *>(tilingData);
    __gm__ HcclCombinOpParam *winContext = (__gm__ HcclCombinOpParam *)(hcclContext[tilingInfo->groupIndex]);

    int rowShape = tilingInfo->rowShape;
    int colShape = tilingInfo->colShape;

    uint16_t lenBurst = colShape * sizeof(T) / 32;
    uint32_t winTokenSize = topk + 1;        // 一个token在win区需要 topk + 1 个 token大小

    int32_t combineInfoOffset = tilingInfo->rowOffset * COMBINE_INFO_NUM;      // combineInfo是完整传入，需要手动偏移

    // Flag清零
    vector_dup(flag, 0, 1, 1, 1, 8, 0);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    flag[0] = 1;            // flag第一位固定为1, 后面通过atomic add实现flag累加

    // 每次发送1个tokenTensor
    for (int row = 0; row < rowShape; row++) {
        int32_t attnId = combineInfo[combineInfoOffset];
        if (attnId < 0) {
            break;  // 静态图无效数据，对应attnId为-1
        }
        int32_t tokenId = combineInfo[combineInfoOffset + 1];
        int32_t kOffset = combineInfo[combineInfoOffset + 2];

        __gm__ T * winGM = (__gm__ T *)(winContext->windowsIn[attnId]) +
                           (uint64_t)((tokenId * winTokenSize + kOffset) * colShape);
        __gm__ int32_t* winFlagAddr = (__gm__ int32_t*)(winContext->windowsExp[attnId]) +
                                      (uint64_t)(tokenId * COMBINE_FLAG_OFFSET);            // flag 占用 512Byte

        // send data
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        copy_ubuf_to_gm(winGM, in, 0 /* sid */, 1, lenBurst, 0, 0);
        pipe_barrier(PIPE_MTE3);

        // set flag
        set_atomic_add();
        set_atomic_s32();
        copy_ubuf_to_gm(winFlagAddr, flag, 0 /* sid */, 1, 1, 0, 0);
        set_atomic_none();
        pipe_barrier(PIPE_MTE3);

        // update Offset
        in += colShape;
        combineInfoOffset += COMBINE_INFO_NUM;
    }
}

template <typename T>
TILEOP void SharedCombineCompute(__ubuf__ T *out, __ubuf__ float *tmpFP32, __ubuf__ float *sumFP32,
    __gm__ T * winGMAddr, int offset, uint16_t lenBurst, int repeat)
{
    __ubuf__ T *curOut = out;
    __ubuf__ float *curSumFP32 = sumFP32;
    copy_gm_to_ubuf(out, winGMAddr, 0 /* sid */, 1, lenBurst, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    for (int j = 0; j < repeat; j++) {
        vconv_bf162f32(tmpFP32, curOut, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vadd(curSumFP32, tmpFP32, curSumFP32, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        curSumFP32 += offset;
        curOut += offset;
    }
}

template <typename T>
TILEOP void AttmCombineCompute(__ubuf__ T *out, __ubuf__ float *scale, __ubuf__ float *mulFP32, __ubuf__ float *sumFP32,
    __gm__ T * winGMAddr, uint32_t topk, uint32_t colShape)
{
    uint16_t lenBurst = colShape * sizeof(T) / 32;
    float scaleVal = 0.0f;
    int repeat = colShape * sizeof(float) / 32 / 8;     // vector 一次处理8个block(32B)
    int offset = 32 * 8 / sizeof(float);

    __ubuf__ T *curOut = out;
    __ubuf__ float *curSumFP32 = sumFP32;

    // init sumFP32
    for (int j = 0; j < repeat; j++) {
        vector_dup(curSumFP32, 0.0f, 1, 1, 1, 8, 8);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        curSumFP32 += offset;
    }

    // moe专家
    for (int i = 0; i < topk; i++) {
        scaleVal = scale[i];
        curOut = out;
        curSumFP32 = sumFP32;
        copy_gm_to_ubuf(out, winGMAddr, 0 /* sid */, 1, lenBurst, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        for (int j = 0; j < repeat; j++) {
            vconv_bf162f32(mulFP32, curOut, 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vmuls(mulFP32, mulFP32, scaleVal, 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vadd(curSumFP32, mulFP32, curSumFP32, 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            curSumFP32 += offset;
            curOut += offset;
        }
        winGMAddr += colShape;
    }

    // 共享专家
    SharedCombineCompute(out, mulFP32, sumFP32, winGMAddr, offset, lenBurst, repeat);

    curOut = out;
    curSumFP32 = sumFP32;
    for (int j = 0; j < repeat; j++) {
        vconv_f322bf16a(curOut, curSumFP32, 1, 1, 1, 8, 8);     // cast round mode
        pipe_barrier(PIPE_V);
        curOut += offset;
        curSumFP32 += offset;
    }
}

template <typename T, uint32_t topk, uint32_t bs>
TILEOP void AttnCombine(__ubuf__ T *out, __ubuf__ float *scale, __ubuf__ float *mulFP32, __ubuf__ float *sumFP32,
    __gm__ float *scaleGM, __ubuf__ int32_t *tilingData, __gm__ int64_t *hcclContext)
{
    __ubuf__ TilingInfo *tilingInfo = reinterpret_cast<__ubuf__ TilingInfo *>(tilingData);
    __gm__ HcclCombinOpParam *winContext = (__gm__ HcclCombinOpParam *)(hcclContext[tilingInfo->groupIndex]);

    int32_t tokenId = tilingInfo->rowOffset;    // rowOffset corresponds to tokenId
    int32_t colShape = tilingInfo->colShape;
    uint32_t winTokenSize = topk + 1;            // requires topk+1 token tensor spaces in WinGM

    // 临时，UBCopyIn拷贝scale的shape大小为[8, 4]数据有问题，
    uint16_t lenBurst = bs * topk * sizeof(float) / 32;
    copy_gm_to_ubuf(scale, scaleGM, 0 /* sid */, 1, lenBurst, 0, 0);
    pipe_barrier(PIPE_MTE2);

    // 等待flag，out 复用为 flag, 不用申请额外的UB给flag用
    __ubuf__ int32_t* flag = reinterpret_cast<__ubuf__ int32_t*>(out);
    __gm__ int32_t* winFlagAddr = (__gm__ int32_t*)(winContext->windowsExp[winContext->rankId]) +
                                  (uint64_t)(tokenId * COMBINE_FLAG_OFFSET);
    CombineWaitFlag(flag, winFlagAddr, winTokenSize);

    // Combine的相关计算
    scale += tokenId * topk;        // scale是完整传入，需要手动偏移
    __gm__ T * winGMAddr = (__gm__ T *)(winContext->windowsIn[winContext->rankId]) +
        (uint64_t)(tokenId * winTokenSize * colShape);
    AttmCombineCompute<T>(out, scale, mulFP32, sumFP32, winGMAddr, topk, colShape);
    pipe_barrier(PIPE_ALL);     // necessary
}

template <typename T>
constexpr TILEOP T AlignUp(const T value, const T alignment)
{
    if (alignment == 0) {
        return value;
    }
    return (value + alignment - 1) / alignment * alignment;
}

TILEOP int32_t CalcOccurrences(__ubuf__ int32_t *expertTable, uint32_t dstExpertId, uint32_t cnt,
    __ubuf__ int32_t *tmpBuf)
{
    if (cnt == 0) {
        return 0;
    }
    int num = 0;
    __ubuf__ int32_t *tmp = expertTable;
    for (int32_t i = 0; i < cnt; i++) {
        if ((*tmp++) == dstExpertId) {
            num++;
        }
    }
    return num;
}

TILEOP int32_t CalcOccurrencesVector(__ubuf__ int32_t *expertTable, uint32_t dstExpertId, uint32_t cnt,
    __ubuf__ int32_t *tmpBuf)
{
    if (cnt == 0) {
        return 0;
    }
    int32_t bufferLen = AlignUp<int32_t>(cnt * sizeof(int32_t), 32);
    uint32_t repeatCnt = bufferLen / 32;
    if (bufferLen % 32 != 0) {
        repeatCnt++;
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
    __ubuf__ int32_t *subBuf = tmpBuf + bufferLen;
    // dst, srcImm, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride
    vector_dup(tmpBuf, dstExpertId, repeatCnt, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    // dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride
    vsub(subBuf, expertTable, tmpBuf, repeatCnt, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    // dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride
    vabs((__ubuf__ float*)tmpBuf, (__ubuf__ float*)subBuf, repeatCnt, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    // dst, src0, src1, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride
    vmins(subBuf, tmpBuf, 1, repeatCnt, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);

    // 求和
    set_mask_count(); // 设置 counter mode
    set_vector_mask(0, cnt); // 只计算 cnt 个
    // dst, src, repeat, dstRepeatStride, srcBlockStride, srcRepeatStride
    vcadd((__ubuf__ float*)tmpBuf, (__ubuf__ float*)subBuf, 1, 8, 1, 8, 0);
    set_mask_norm();
    set_vector_mask(-1, -1);
    pipe_barrier(PIPE_V);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    return cnt - tmpBuf[0];
}

struct DispatchTilingInfo {
    int tileIndex{0};
    int groupIndex{0};
    int shape{0};
    int offset{0};
    int totalTileNum{0};
};

template <typename T, int32_t bs, int32_t axisH, int32_t topK>
TILEOP void SendToRoutingExpert(__gm__ int32_t *syncTensor, __ubuf__ T *tokenBuffer, __ubuf__ int32_t *expertTableUb,
    __ubuf__ int32_t *expertBuffer, __gm__ T *token, __gm__ int32_t *expertTable, __ubuf__ int32_t *tilingData,
    __gm__ int64_t *hcclContext)
{
    auto tilingInfo = reinterpret_cast<__ubuf__ DispatchTilingInfo *>(tilingData);
    constexpr int32_t hOutSize = axisH * sizeof(T); // 如有量化，需要量化后通信
    constexpr int32_t hCommuSize = hOutSize;
    constexpr int32_t tokenQuantAlign32 = AlignUp<int32_t>(hOutSize + sizeof(float), 32) / sizeof(int32_t);
    constexpr int32_t expertPerSizeOnWin = bs * hCommuSize;

    __gm__ HcclCombinOpParam *winContext = (__gm__ HcclCombinOpParam *)(hcclContext[tilingInfo->groupIndex]);
    int32_t localUsrRankId = static_cast<int32_t>(winContext->rankId);

    constexpr int32_t expertTblSize = bs * topK;
    constexpr int32_t lenBurst = AlignUp<int32_t>(expertTblSize * sizeof(int32_t), 32) / 32;
    copy_gm_to_ubuf(expertTableUb, expertTable, 0, 1, lenBurst, 0, 0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

    __ubuf__ int32_t *tmpTokenBuffer = reinterpret_cast<__ubuf__ int32_t *>(tokenBuffer);
    for (int32_t tableIndex = tilingInfo->offset; tableIndex < tilingInfo->offset + tilingInfo->shape; ++tableIndex) {
        int32_t row = tableIndex / topK;
        copy_gm_to_ubuf(tokenBuffer, token + row * axisH, 0, 1, hOutSize / 32, 0, 0);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        tmpTokenBuffer[tokenQuantAlign32] = localUsrRankId;
        tmpTokenBuffer[tokenQuantAlign32 + 1] = row;
        tmpTokenBuffer[tokenQuantAlign32 + 2] = tableIndex % topK;
        uint32_t remoteRankId = *(expertTableUb + tableIndex);
        uint32_t tokenWinOffset = CalcOccurrencesVector(expertTableUb, remoteRankId, tableIndex, expertBuffer);
        GM_ADDR remoteRankWinGMBaseAddr = (GM_ADDR)(winContext->windowsIn[remoteRankId]);
        GM_ADDR dstWinGMAddr = remoteRankWinGMBaseAddr + expertPerSizeOnWin * localUsrRankId +
            (tokenWinOffset * hCommuSize);
        copy_ubuf_to_gm(dstWinGMAddr, tokenBuffer, 0, 1, hCommuSize / 32, 0, 0);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    }
}

template <typename T, int32_t bs, int32_t axisH>
TILEOP void SendToSharedExpert(__gm__ int32_t *syncTensor, __ubuf__ T *tokenBuffer, __gm__ T *token,
    __ubuf__ int32_t *tilingData, __gm__ int64_t *hcclContext)
{
    auto tilingInfo = reinterpret_cast<__ubuf__ DispatchTilingInfo *>(tilingData);
    constexpr int32_t hOutSize = axisH * sizeof(T);
    constexpr int32_t scaleParamPad = 128; // 预留128B给量化参数+expandidx索引，实际使用量化 4B(fp32) + 3 * 4，非量化 3 * 4
    constexpr int32_t hCommuSize = hOutSize; // + scaleParamPad; // 暂时先不处理 scale
    constexpr int32_t tokenQuantAlign32 = AlignUp<int32_t>(hOutSize + sizeof(float), 32) / sizeof(int32_t);
    constexpr int32_t expertPerSizeOnWin = bs * hCommuSize;

    __gm__ HcclCombinOpParam *winContext = (__gm__ HcclCombinOpParam *)(hcclContext[tilingInfo->groupIndex]);
    int32_t localUsrRankId = static_cast<int32_t>(winContext->rankId);
    int32_t rankSize = static_cast<int32_t>(winContext->rankNum);
    int32_t shareRankSize = 1; // 在前端赋值 shareRankCnt
    int32_t moeRankSize = rankSize - shareRankSize;
    int32_t shareOpProcessRankSize = moeRankSize / shareRankSize;

    __ubuf__ int32_t *tmpTokenBuffer = reinterpret_cast<__ubuf__ int32_t *>(tilingData); // 类型转换使用
    for (int32_t bsId = tilingInfo->offset; bsId < tilingInfo->offset + tilingInfo->shape; ++bsId) {
        copy_gm_to_ubuf(tokenBuffer, token + bsId * axisH, 0, 1, hOutSize / 32, 0, 0);
        tmpTokenBuffer[tokenQuantAlign32] = localUsrRankId; // 暂时先不考虑 A3 数组，前端大小可以暂时用 token[1],暂不修改
        tmpTokenBuffer[tokenQuantAlign32 + 1] = bsId;
        tmpTokenBuffer[tokenQuantAlign32 + 2] = 1;
        // 对齐后续 sched 使用方式，连续的 moe 卡发给同一个共享专家
        uint32_t remoteRankId = (localUsrRankId - shareRankSize) / shareOpProcessRankSize; \
        GM_ADDR remoteRankWinGMBaseAddr = (GM_ADDR)(winContext->windowsIn[remoteRankId]);
        GM_ADDR dstWinGMAddr = remoteRankWinGMBaseAddr + expertPerSizeOnWin * localUsrRankId +
            (bsId * hCommuSize);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        copy_ubuf_to_gm(dstWinGMAddr, tokenBuffer, 0, 1, hCommuSize / 32, 0, 0);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
}

template <typename T, int32_t bs, int32_t axisH>
TILEOP void CopyToLocalExpert(__gm__ T *out, __ubuf__ T *tokenBuffer, __gm__ T *token, __ubuf__ int32_t *tilingData,
    __gm__ int64_t *hcclContext)
{
    auto tilingInfo = reinterpret_cast<__ubuf__ DispatchTilingInfo *>(tilingData);
    constexpr int32_t hOutSize = axisH * sizeof(T);
    constexpr int32_t hCommuSize = hOutSize;

    __gm__ HcclCombinOpParam *winContext = (__gm__ HcclCombinOpParam *)(hcclContext[tilingInfo->groupIndex]);
    uint32_t localUsrRankId = winContext->rankId;

    for (int32_t bsId = tilingInfo->offset; bsId < tilingInfo->offset + tilingInfo->shape; ++bsId) {
        copy_gm_to_ubuf(tokenBuffer, token + bsId * axisH, 0, 1, hOutSize / 32, 0, 0);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        copy_ubuf_to_gm(out + bsId * axisH, tokenBuffer, 0, 1, hOutSize / 32, 0, 0);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    }
}

template <typename T, int32_t bs, int32_t topK>
TILEOP void DispatchSetFlag(__gm__ int32_t *dummy, __ubuf__ int32_t *statusTensor, __ubuf__ int32_t *expertTableUb,
    __ubuf__ int32_t *expertBuffer, __gm__ T *syncTensor, __gm__ int32_t *expertTable, __ubuf__ int32_t *tilingData,
    __gm__ int64_t *hcclContext)
{
    auto tilingInfo = reinterpret_cast<__ubuf__ DispatchTilingInfo *>(tilingData);
    __gm__ HcclCombinOpParam *winContext = (__gm__ HcclCombinOpParam *)(hcclContext[tilingInfo->groupIndex]);

    constexpr int32_t expertTblSize = bs * topK;
    constexpr int32_t lenBurst = AlignUp<int32_t>(expertTblSize * sizeof(int32_t), 32) / 32;
    copy_gm_to_ubuf(expertTableUb, expertTable, 0, 1, lenBurst, 0, 0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

    uint32_t localUsrRankId = winContext->rankId;
    for (int32_t dstExpertId = tilingInfo->offset; dstExpertId < tilingInfo->offset + tilingInfo->shape; 
        ++dstExpertId) {
        GM_ADDR winFlagBaseAddr = (GM_ADDR)(winContext->windowsExp[dstExpertId]);
        GM_ADDR dstWinGMAddr = winFlagBaseAddr + 512 * localUsrRankId;

        statusTensor[dstExpertId * 8] = 1;
        statusTensor[dstExpertId * 8 + 1] = CalcOccurrencesVector(expertTableUb, dstExpertId, expertTblSize,
            expertBuffer);
        copy_ubuf_to_gm(dstWinGMAddr, statusTensor + dstExpertId * 8, 0, 1, 1, 0, 0);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    }
}

struct DataCopyParams {
    uint8_t sid;
    uint16_t nBurst;
    uint16_t lenBurst;
    uint16_t srcStride;
    uint16_t dstStride;
};
 
struct GatherMaskParams {
    uint16_t repeat;
    uint8_t src0BlockStride;
    uint8_t patternMode;
    uint16_t src0RepeatStride;
    uint8_t src1RepeatStride;
};
 
struct SumParams {
    uint8_t repeat;
    uint16_t dstRepeatStride;
    uint16_t srcBlockStride;
    uint16_t srcRepeatStride;
};
 
constexpr uint32_t MASK_SELECT_SEND_FLAG = 0x1010101; // 每 8 个数取第一个
constexpr uint32_t MASK_SELECT_SEND_COUNT = 0x2020202; // 每 8 个数取第二个
constexpr uint32_t MASK_SELECT_RECV_TOKEN_CNT = 0x1010101; // 每 8 个数取第一个
 
TILEOP void GatherMask(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src0, __ubuf__ uint32_t *src1,
    GatherMaskParams &gatherMaskParams)
{
    set_mask_norm();
    set_vector_mask(-1, -1);
    vreducev2(dst, src0, src1, gatherMaskParams.repeat, gatherMaskParams.src0BlockStride, gatherMaskParams.patternMode,
        gatherMaskParams.src0RepeatStride, gatherMaskParams.src1RepeatStride);
    set_mask_norm();
    set_vector_mask(-1, -1); // 重置 mask
}
 
TILEOP void Sum(__ubuf__ float *result, __ubuf__ float *src, SumParams &sumParams, uint32_t cnt)
{
    set_mask_count(); // 设置 counter mode
    set_vector_mask(0, cnt); // 只计算 cnt 个
    vcadd(result, src, sumParams.repeat, sumParams.dstRepeatStride, sumParams.srcBlockStride,
        sumParams.srcRepeatStride, 0);
    set_mask_norm(); // 重置 mode
    set_vector_mask(-1, -1); // 重置 mask
}
 
TILEOP void ReadFlagV2(__ubuf__ uint32_t *flag, uint32_t offset, uint32_t repeat, __ubuf__ int32_t *tilingData,
    __gm__ int64_t *hcclContext)
{
    __ubuf__ TilingInfo *tilingInfo = reinterpret_cast<__ubuf__ TilingInfo *>(tilingData);
    __gm__ HcclCombinOpParam *winContext = (__gm__ HcclCombinOpParam *)(hcclContext[tilingInfo->groupIndex]);
    uint32_t localUsrRankId = winContext->rankId;
    GM_ADDR winFlagBaseAddr = (GM_ADDR)(winContext->windowsExp[localUsrRankId]); // flag 在 win 区的基地址
    GM_ADDR winFlagReadStartAddr = winFlagBaseAddr + offset;
 
    DataCopyParams dataCopyParams;
    dataCopyParams.sid = 0;
    dataCopyParams.nBurst = repeat; // 搬运次数
    dataCopyParams.lenBurst = 1; // 每次搬运的数据量大小，32B 为单位，1 表示每次搬运 32B
    dataCopyParams.srcStride = 15; // 前一个尾巴和下一个的开头，gap，15 * 32 = 480B
    dataCopyParams.dstStride = 0; // 前一个尾巴和下一个开头，gap，搬到 UB 上需要连续
    copy_gm_to_ubuf(flag, winFlagReadStartAddr, dataCopyParams.sid, dataCopyParams.nBurst, dataCopyParams.lenBurst,
        dataCopyParams.srcStride, dataCopyParams.dstStride);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
}
 
template <typename T>
TILEOP void GatherMaskAndSum(__gm__ T *out, __ubuf__ uint32_t *src0, __ubuf__ uint32_t *src1, __ubuf__ uint32_t *dst,
    uint32_t mask, uint32_t cnt, __ubuf__ int32_t *tilingData, __gm__ int64_t *hcclContext)
{
    ClearFlagBuf(reinterpret_cast<__ubuf__ int32_t *>(src1));
    ClearFlagBuf(reinterpret_cast<__ubuf__ int32_t *>(dst));
 
    src1[0] = mask; // 设置前 32 个数
    src1[1] = mask; // 设置后 32 个数，共计 64 个数，256B
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

    GatherMaskParams gatherMaskParams;
    uint32_t gatherMaskRepeat = (cnt * 32 + 255) / 256; // 重复次数，向上对齐 256B（gather 每次处理 256B）
    gatherMaskParams.repeat = gatherMaskRepeat; // 重复次数，最多只会处理 8 个数，每 8 个取一个，正好 64 个，256B
    // 单次迭代内 blk stride，表示 mask 后 32 个数相对前 32 个数的 stride（u32 类型），1 表示连续，0 表示两次处理同一块，一般取 1
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.patternMode = 0; // 自定义模式需为 0
    // 可能可以调为 1，winFlag 搬运进 UB 的时候可以按照 32B 对齐，不搬运全部 512B 大小
    gatherMaskParams.src0RepeatStride = 16; // 迭代间 stride，16 * 32 = 512B，符合 dispatch 的 flag 排序要求
    gatherMaskParams.src1RepeatStride = 0; // 迭代间 stride，0 表示每次 repeat 都取同样的 src1 mask
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    GatherMask(dst, src0, src1, gatherMaskParams);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0); 

    __ubuf__ float *sumSrc = reinterpret_cast<__ubuf__ float *>(dst);
    ClearFlagBuf(reinterpret_cast<__ubuf__ int32_t *>(src1)); // 可以不加，其余位置的数据并不重要，可以不清空
    __ubuf__ float *sumDst = reinterpret_cast<__ubuf__ float *>(src1);
 
    SumParams sumParams;
    sumParams.repeat = 1; // 只计算一次
    sumParams.dstRepeatStride = 8; // 不重要
    sumParams.srcBlockStride = 1; // 表示 src 连续取值
    sumParams.srcRepeatStride = 8; // 不重要
    Sum(sumDst, sumSrc, sumParams, cnt); // sum 的输出这里是一个 float 值
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
}
 
TILEOP void CopyOutRecvTokenCnt(GM_ADDR outRecvTokenCntAddr, UB_ADDR recvTokenCntAddr, uint32_t tileIndex,
    uint32_t totalTileNum)
{
    DataCopyParams dataCopyParams;
    dataCopyParams.sid = 0;
    dataCopyParams.nBurst = 1; // 搬运次数
    dataCopyParams.lenBurst = 1; // 每次搬运的数据量大小，32B 为单位，1 表示每次搬运 32B
    dataCopyParams.srcStride = 0; // 前一个尾巴和下一个的开头，gap，不重要
    dataCopyParams.dstStride = 15; // 前一个尾巴和下一个开头，gap，不重要
 
    uint32_t offset = totalTileNum * 512; // 每个 op 写 48 个 512B 大小的地址
    GM_ADDR outRecvTokenCntStartAddr = outRecvTokenCntAddr + tileIndex * offset; // 本 op 偏移地址
    // 搬运需要使用同一个 src，所以需要手动循环
    for (int i = 0; i < totalTileNum; i++) {
        copy_ubuf_to_gm(outRecvTokenCntStartAddr, recvTokenCntAddr, dataCopyParams.sid, dataCopyParams.nBurst,
            dataCopyParams.lenBurst, dataCopyParams.srcStride, dataCopyParams.dstStride);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        outRecvTokenCntStartAddr += 512; // 需要将同一个 src 连续写 48 次，所以 src 不变化，dst 每次手动偏移 512B
    }
}

template <typename T>
TILEOP void ConstructOutRecvTokenCnt(__gm__ T *out, __ubuf__ uint32_t *src0, __ubuf__ uint32_t *src1,
    __ubuf__ uint32_t *dst, uint32_t cnt, __ubuf__ int32_t *tilingData, __gm__ int64_t *hcclContext)
{
    GatherMaskAndSum(out, src0, src1, dst, MASK_SELECT_SEND_COUNT, cnt, tilingData, hcclContext);
 
    GM_ADDR outRecvTokenCntAddr = reinterpret_cast<GM_ADDR>(out);
    UB_ADDR recvTokenCntAddr = reinterpret_cast<UB_ADDR>(src1); // src1 复用为了 sum 最终的输出
    __ubuf__ TilingInfo *tilingInfo = reinterpret_cast<__ubuf__ TilingInfo *>(tilingData);
 
    CopyOutRecvTokenCnt(outRecvTokenCntAddr, recvTokenCntAddr, tilingInfo->tileIndex, tilingInfo->totalTileNum);
}
 
/*
 * src0: store flag, 288 * 512B, if int32, shape is [rankSize, 128]
 * src1: mask, dtype is uint32, shape can be [1, 8], 32B
 * dst: vreducev2 out, rankSize * 4B
 * 需要注意 tensor 的大小一定得符合预期，否则结果可能不符合预期
*/
template <typename T>
TILEOP void WaitFlagV2(__gm__ T *out, __ubuf__ uint32_t *src0, __ubuf__ uint32_t *src1, __ubuf__ uint32_t *dst,
    uint32_t cnt, __ubuf__ int32_t *tilingData, __gm__ int64_t *hcclContext)
{
    GatherMaskAndSum(out, src0, src1, dst, MASK_SELECT_SEND_FLAG, cnt, tilingData, hcclContext);
}
 
TILEOP void ClearFlagV2(__ubuf__ int32_t *flag, uint32_t offset, uint32_t repeat, __ubuf__ int32_t *tilingData,
    __gm__ int64_t *hcclContext)
{
    __ubuf__ TilingInfo *tilingInfo = reinterpret_cast<__ubuf__ TilingInfo *>(tilingData);
    __gm__ HcclCombinOpParam *winContext = (__gm__ HcclCombinOpParam *)(hcclContext[tilingInfo->groupIndex]);
    uint32_t localUsrRankId = winContext->rankId;
    GM_ADDR winFlagBaseAddr = (GM_ADDR)(winContext->windowsExp[localUsrRankId]); // flag 在 win 区的基地址
    GM_ADDR winFlagReadStartAddr = winFlagBaseAddr + offset;
 
    ClearFlagBuf(flag);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    flag[0] = -1;
    DataCopyParams dataCopyParams;
    dataCopyParams.sid = 0;
    dataCopyParams.nBurst = 1; // 搬运次数
    dataCopyParams.lenBurst = 1; // 每次搬运的数据量大小，32B 为单位，1 表示每次搬运 32B
    dataCopyParams.srcStride = 0; // 前一个尾巴和下一个的开头，gap
    dataCopyParams.dstStride = 15; // 前一个尾巴和下一个开头，gap
    set_atomic_s32();
    for (int i = 0; i < repeat; i++) {
        copy_ubuf_to_gm(winFlagReadStartAddr, flag, dataCopyParams.sid, dataCopyParams.nBurst, dataCopyParams.lenBurst,
            dataCopyParams.srcStride, dataCopyParams.dstStride);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        winFlagReadStartAddr += 512;
    }
    set_atomic_none();
}
 
template <typename T>
TILEOP void MoeRankWaitFlag(__gm__ T *out, __ubuf__ uint32_t *src0, __ubuf__ uint32_t *src1, __ubuf__ uint32_t *dst,
    __ubuf__ int32_t *tilingData, __gm__ int64_t *hcclContext, uint32_t processRankSize)
{
    __ubuf__ TilingInfo *tilingInfo = reinterpret_cast<__ubuf__ TilingInfo *>(tilingData);
    uint32_t cnt = processRankSize; // 计算次数，根据处理的卡数 shapeR 来确定
    uint32_t flagSum = 0;
    uint32_t offset = tilingInfo->rankOffset * 512;
    while (flagSum != cnt) {
        ReadFlagV2(src0, offset, cnt, tilingData, hcclContext);
        WaitFlagV2<T>(out, src0, src1, dst, cnt, tilingData, hcclContext);
        // src1 复用为 sum 的输出
        flagSum = src1[0];
    }
    // 理论上这里不需要再读，前面已经确保写上去了
    ConstructOutRecvTokenCnt<T>(out, src0, src1, dst, cnt, tilingData, hcclContext);
    ClearFlagV2(reinterpret_cast<__ubuf__ int32_t *>(src0), offset, cnt, tilingData, hcclContext); // 暂时放在读完之后就清 flag
}
 
template <typename T>
TILEOP void ShareRankWaitFlag(__gm__ T *out, __ubuf__ uint32_t *src0, __ubuf__ uint32_t *src1, __ubuf__ uint32_t *dst,
    __ubuf__ int32_t *tilingData, __gm__ int64_t *hcclContext, uint32_t processRankSize)
{
    __ubuf__ TilingInfo *tilingInfo = reinterpret_cast<__ubuf__ TilingInfo *>(tilingData);
    // 当前 share rank 接收的起始的 moe rank id
    uint32_t startMoeRankId = tilingInfo->tileIndex * processRankSize + tilingInfo->shareRankCnt;
    if (tilingInfo->tileIndex != 0) {
        return;
    }
    uint32_t cnt = processRankSize; // 计算次数，共享专家固定处理 8 个卡
    uint32_t flagSum = 0;
    uint32_t offset = startMoeRankId * 512; // 每张卡占据 512B，flag 读取偏移为 rankID * 512B
    while (flagSum != cnt) {
        ReadFlagV2(src0, offset, cnt, tilingData, hcclContext);
        WaitFlagV2<T>(out, src0, src1, dst, cnt, tilingData, hcclContext);
        // src1 复用为 sum 的输出
        flagSum = src1[0];
    }
    ClearFlagV2(reinterpret_cast<__ubuf__ int32_t *>(src0), offset, cnt, tilingData, hcclContext); // 暂时放在读完之后就清 flag
}
 
// tileIndex 覆写为了 tileOpIndex 计数
template<typename T, bool isSharedRank = false>
TILEOP void FFNSched(__gm__ T *out, __ubuf__ int32_t *buffer, __gm__ int32_t *dummy, __ubuf__ int32_t *tilingData,
    __gm__ int64_t *hcclContext)
{
    __ubuf__ TilingInfo *tilingInfo = reinterpret_cast<__ubuf__ TilingInfo *>(tilingData);
    __gm__ HcclCombinOpParam *winContext = (__gm__ HcclCombinOpParam *)(hcclContext[tilingInfo->groupIndex]);
    __ubuf__ uint8_t *tmpUb = reinterpret_cast<__ubuf__ uint8_t *>(buffer);
    uint32_t offset = 0;
    __ubuf__ uint32_t *src0 = reinterpret_cast<__ubuf__ uint32_t *>(tmpUb + offset);
    uint32_t moeOpProcessRankSize = tilingInfo->rankShape;
    uint32_t shareOpProcessRankSize = (winContext->rankNum - tilingInfo->shareRankCnt) / tilingInfo->shareRankCnt;
    uint32_t maxProcessRankSize = (moeOpProcessRankSize < shareOpProcessRankSize) ?
        shareOpProcessRankSize : moeOpProcessRankSize; // 共享专家和 moe 专家每个 op 处理的最大卡数，在 288 卡的时候，
        // 共享专家处理 8 个，moe 处理 6 个，最大是 8 个
    uint32_t src0Size = maxProcessRankSize * 32; // 每个 op 最多等待的 flag 卡数，最多是 8 个，8 * 32 = 256B
    offset += src0Size;
    __ubuf__ uint32_t *src1 = reinterpret_cast<__ubuf__ uint32_t *>(tmpUb + offset);
    uint32_t src1Size = 32; // 第一次是 mask，32B，第二次复用为 sum 的结果，一个 float 4B；所以最大为 32B
    offset += src1Size;
    __ubuf__ uint32_t *dst = reinterpret_cast<__ubuf__ uint32_t *>(tmpUb + offset);
    uint32_t dstSize = (maxProcessRankSize * 4 + 31) / 32 * 32; // src0 中挑出来的 int 个数，最大是 8 个，正好是 32B
    offset += dstSize;
 
    if constexpr (isSharedRank) {
        ShareRankWaitFlag<T>(out, src0, src1, dst, tilingData, hcclContext, shareOpProcessRankSize);
    } else {
        MoeRankWaitFlag<T>(out, src0, src1, dst, tilingData, hcclContext, moeOpProcessRankSize);
    }
}

TILEOP void ReadRecvTokenCnt(__ubuf__ uint32_t *recvTokenCnt, __gm__ uint32_t *src,
    __ubuf__ int32_t *tilingData, __gm__ int64_t *hcclContext, uint32_t tileCnt)
{
    __ubuf__ TilingInfo *tilingInfo = reinterpret_cast<__ubuf__ TilingInfo *>(tilingData);
    DataCopyParams gmToUbParams;
    gmToUbParams.sid = 0;
    gmToUbParams.nBurst = tilingInfo->tileIndex; // 搬运次数，只搬运能用到的数据即可
    // 对于 index 0，需要全部搬运，其会计算全部；likely 是为了降低快核，提高慢核性能
    if (tilingInfo->tileIndex == 0) [[likely]] {
        gmToUbParams.nBurst = tileCnt;
    }
    gmToUbParams.lenBurst = 1; // cnt 有效数据只有 4B，搬运 32B 即可
    // 前一个尾巴和下一个的开头，gap，最大值 65535，每次偏移 48*512B，48 * 512 / 32 - 1
    gmToUbParams.srcStride = tilingInfo->totalTileNum * 512 / 32 - 1;
    gmToUbParams.dstStride = 0; // 前一个尾巴和下一个开头，gap，搬运成连续
 
    // 每个 tile 都读自己 index 的 gm 地址，避免地址交织
    GM_ADDR thisTileStartSrcAddr = reinterpret_cast<GM_ADDR>(src) + tilingInfo->tileIndex * 512;
 
    copy_gm_to_ubuf(recvTokenCnt, thisTileStartSrcAddr, gmToUbParams.sid, gmToUbParams.nBurst, gmToUbParams.lenBurst,
        gmToUbParams.srcStride, gmToUbParams.dstStride);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
}
 
TILEOP void CumSum(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src, __ubuf__ uint32_t *gatherMaskDst, uint32_t mask,
    uint32_t cnt)
{
    __ubuf__ uint32_t *gatherMask = reinterpret_cast<__ubuf__ uint32_t *>(dst);
    // 主逻辑
    gatherMask[0] = mask; // 每 8 个数挑一个，一次处理 64 个数，256B，一共需要处理 32 * 48 / 256 = 6 次
    gatherMask[1] = mask;
    pipe_barrier(PIPE_ALL);
    GatherMaskParams gatherMaskParams;
    gatherMaskParams.repeat = (cnt * 32 + 255) / 256; // 重复次数，只搬运所需要的部分
    // 单次迭代内 blk stride，表示 mask 后 32 个数相对前 32 个数的 stride（u32 类型），单位应该是 128B，1 表示连续，0 表示两次处理同一块，一般取 1
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.patternMode = 0; // 自定义模式需为 0
    gatherMaskParams.src0RepeatStride = 8; // 迭代间 stride，搬运进 UB 后调整为 32B 间隔，64 个数 * 4 = 256B / 32 = 8
    gatherMaskParams.src1RepeatStride = 0; // 迭代间 stride，0 表示每次 repeat 都取同样的 src1 mask
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    GatherMask(gatherMaskDst, src, gatherMask, gatherMaskParams); // 把所有的 src 都收集起来了，共计 48 个
    set_flag(PIPE_V, PIPE_S, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID1); 
    __ubuf__ float *sumSrc = reinterpret_cast<__ubuf__ float *>(gatherMaskDst);
    // 在 TileOpIndex = 0 的场景下，cnt 是 0，需要手动清空上面赋值的 mask，否则后续结果错误
    ClearFlagBuf(reinterpret_cast<__ubuf__ int32_t *>(dst));
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    __ubuf__ float *sumDst = reinterpret_cast<__ubuf__ float *>(dst); // GatherMask 的 mask 复用为 Sum 的输出
 
    SumParams sumParams;
    sumParams.repeat = 1; // 只计算一次
    sumParams.dstRepeatStride = 8; // 不重要
    sumParams.srcBlockStride = 1; // 表示 src 连续取值
    sumParams.srcRepeatStride = 8; // 不重要
    Sum(sumDst, sumSrc, sumParams, cnt); // sum 的输出这里是一个 float 值
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
}
 
TILEOP void CopyGmToGm(__gm__ void *dst, __gm__ void *src, __ubuf__ void *tmpUbuf, DataCopyParams gmToUbParams,
    DataCopyParams ubToGmParams)
{
    copy_gm_to_ubuf(tmpUbuf, src, gmToUbParams.sid, gmToUbParams.nBurst, gmToUbParams.lenBurst,
        gmToUbParams.srcStride, gmToUbParams.dstStride);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

    copy_ubuf_to_gm(dst, tmpUbuf, ubToGmParams.sid, ubToGmParams.nBurst, ubToGmParams.lenBurst,
        ubToGmParams.srcStride, ubToGmParams.dstStride);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
}

template<typename T>
TILEOP void MoeRankWinCopyOut(__gm__ T *expandX, __ubuf__ uint8_t *buffer, __ubuf__ int32_t *tilingData,
    __gm__ int64_t *hcclContext)
{
    __ubuf__ TilingInfo *tilingInfo = reinterpret_cast<__ubuf__ TilingInfo *>(tilingData);
    uint32_t offset = 0;
    __ubuf__ T *token = reinterpret_cast<__ubuf__ T *>(buffer + offset);
    uint32_t tokenSize = tilingInfo->colShape * sizeof(T); // 这里需要为真实的 colShape
    offset = offset + tokenSize;
    __ubuf__ uint32_t *flag = reinterpret_cast<__ubuf__ uint32_t *>(buffer + offset);
    uint32_t flagSize = 32; // 读取的一张卡的 flag，一共两个 int 有效数字，32B 即可存放
    offset = offset + flagSize;
 
    __gm__ HcclCombinOpParam *winContext = (__gm__ HcclCombinOpParam *)(hcclContext[tilingInfo->groupIndex]);
    uint32_t localUsrRankId = winContext->rankId;
    GM_ADDR winTokenAddr = (GM_ADDR)(winContext->windowsIn[localUsrRankId]);
 
    uint32_t tokenCnt = 0; // 本 op 处理的 cnt 总数
    for (int rank = tilingInfo->rankOffset; rank < tilingInfo->rankOffset + tilingInfo->rankShape; rank++) {
        // 当前处理 rank 的 gm 地址，类型为 GM_ADDR
        GM_ADDR thisRankOutAddr = reinterpret_cast<GM_ADDR>(expandX) + tokenCnt * tilingInfo->colShape * sizeof(T);
        // winTokenAddr 类型为 GM_ADDR，类型为 u8，需要乘以 type size; 当前 rank 的偏移为 bs * size(token) * rank，colShape 需要为真实的 shape
        GM_ADDR thisRankWinTokenAddr = winTokenAddr + rank * tilingInfo->rowShape * tilingInfo->colShape * sizeof(T);
 
        uint32_t thisRankFlagOffset = rank * 512; // 当前 rank 在 flag 的偏移
        ReadFlagV2(flag, thisRankFlagOffset, 1, tilingData, hcclContext); // 每次读取一张卡的 flag, 512B, 存到 UB 是 32B
        uint32_t thisRankSendTokenCnt = flag[1]; // count 在 flag 第二个数
        tokenCnt += thisRankSendTokenCnt; // 最好一次一个 token，否则 UB 占用可能超
 
        DataCopyParams gmToUbParams;
        gmToUbParams.sid = 0;
        gmToUbParams.nBurst = 1; // 搬运次数，最大为一张卡的 token 全部发送过来，bs，防止超 UB
        gmToUbParams.lenBurst = tilingInfo->colShape * sizeof(T) / 32; // col 就是 token 实际列宽
        gmToUbParams.srcStride = 240; // 前一个尾巴和下一个的开头，gap，这里根据 token 结构需要跳跃 7168+512
        gmToUbParams.dstStride = 0; // 前一个尾巴和下一个开头，gap
 
        DataCopyParams ubToGmParams;
        ubToGmParams.sid = 0;
        ubToGmParams.nBurst = 1; // 搬运次数
        ubToGmParams.lenBurst = tilingInfo->colShape * sizeof(T) / 32; // col 就是 token 实际列宽
        ubToGmParams.srcStride = 0; // 前一个尾巴和下一个的开头，gap
        ubToGmParams.dstStride = 0; // 前一个尾巴和下一个开头，gap
 
        for (int i = 0; i < thisRankSendTokenCnt; i++) {
            CopyGmToGm(thisRankOutAddr, thisRankWinTokenAddr, token, gmToUbParams, ubToGmParams);
            thisRankWinTokenAddr += tilingInfo->colShape * sizeof(T); // colShape 需要为真实的 shape
            thisRankOutAddr += tilingInfo->colShape * sizeof(T);
        }
    }
}
 
// 对于 MOE 专家卡，搬出需要确定当前 TileOp 收到了多少个 token，获取偏移
template<typename T>
TILEOP void MoeRankCopyOut(__gm__ T *expandX, __gm__ uint32_t *validCnt, __ubuf__ uint8_t *buffer,
    __gm__ uint32_t *gmRecvTokenCnt, __ubuf__ int32_t *tilingData, __gm__ int64_t *hcclContext)
{
    __ubuf__ TilingInfo *tilingInfo = reinterpret_cast<__ubuf__ TilingInfo *>(tilingData);
    int tileCnt = tilingInfo->totalTileNum;
    uint32_t offset = 0;
    __ubuf__ uint32_t *recvTokenCnt = reinterpret_cast<__ubuf__ uint32_t *>(buffer + offset);
    uint32_t recvTokenCntSize = (tileCnt * 32 + 255) / 256 * 256; // 48 个 TileOp，每个 32B 大小
    offset = offset + recvTokenCntSize;
    // 第一次作为 GatherMask 的 mask，只有两个数；第二次作为 cumSum 的输出，只有一个数；但是会使用指令清空，最小 256B；所以最终 256B
    __ubuf__ uint32_t *cumSumDst = reinterpret_cast<__ubuf__ uint32_t *>(buffer + offset);
    uint32_t cumSumDstSize = 256;
    offset = offset + cumSumDstSize;
    __ubuf__ uint32_t *gatherMaskDst = reinterpret_cast<__ubuf__ uint32_t *>(buffer + offset);
    // 作为 gathermask 的 dst，最多会存放 48 个 int，也就是最大是 48 * 4 = 192B
    uint32_t gatherMaskDstSize = (tileCnt * 4 + 31) / 32 * 32;
    offset = offset + gatherMaskDstSize;
 
    ReadRecvTokenCnt(recvTokenCnt, gmRecvTokenCnt, tilingData, hcclContext, tileCnt);
 
    // 计算的最终结果在 cumSumDst 中，一个 float 类型数据
    // tileIndex 就是计算 recvTokenOffset 时 sum 的计算 cnt
    CumSum(cumSumDst, recvTokenCnt, gatherMaskDst, MASK_SELECT_RECV_TOKEN_CNT, tilingInfo->tileIndex);
    uint32_t recvTokenOffset = cumSumDst[0];
    if (tilingInfo->tileIndex == 0) [[likely]] { // 默认只有一个 TileOp 做计算，提升慢核性能
        CumSum(cumSumDst, recvTokenCnt, gatherMaskDst, MASK_SELECT_RECV_TOKEN_CNT, tileCnt);
        validCnt[0] = cumSumDst[0];
    }
    uint32_t outOffset = recvTokenOffset * tilingInfo->colShape; // 单位是元素个数
    MoeRankWinCopyOut<T>(expandX + outOffset, buffer, tilingData, hcclContext);
}

template<typename T>
TILEOP void ShareRankWinCopyOut(__gm__ T *expandX, __ubuf__ uint8_t *buffer, uint32_t processTokenCnt,
    uint32_t recvTokenOffset, __ubuf__ int32_t *tilingData, __gm__ int64_t *hcclContext, uint32_t processMoeRankCnt)
{
    if (processTokenCnt == 0) { // bs 泛化场景下可能存在不够 48 个 op 切分的场景
        return;
    }
    __ubuf__ TilingInfo *tilingInfo = reinterpret_cast<__ubuf__ TilingInfo *>(tilingData);
    __gm__ HcclCombinOpParam *winContext = (__gm__ HcclCombinOpParam *)(hcclContext[tilingInfo->groupIndex]);
    uint32_t localUsrRankId = winContext->rankId;
 
    GM_ADDR winTokenAddr = (GM_ADDR)(winContext->windowsIn[localUsrRankId]); // winIn 起始地址
    uint32_t shareRankCnt = tilingInfo->shareRankCnt;
    uint32_t processMoeRankStartIdx = localUsrRankId * processMoeRankCnt + shareRankCnt; // 本共享专家接收的 moe 专家起始 rank id
    // 每张卡在 win 区都占据完整大小，row，col，共享专家处理从 startIdx 开始的连续 64 个 token（处理 scale + 三元组）
    GM_ADDR thisRankWinTokenAddr = winTokenAddr + processMoeRankStartIdx * tilingInfo->rowShape * tilingInfo->colShape
        * sizeof(T);
    GM_ADDR thisTileWinTokenAddr = thisRankWinTokenAddr + recvTokenOffset * tilingInfo->colShape * sizeof(T);
 
    uint32_t offset = 0;
    __ubuf__ T *token = reinterpret_cast<__ubuf__ T *>(buffer + offset);
    uint32_t tokenSize = tilingInfo->colShape * sizeof(T);
    offset = offset + tokenSize;
 
    DataCopyParams gmToUbParams;
    gmToUbParams.sid = 0;
    gmToUbParams.nBurst = 1; // 搬运次数，防止 UB 越界，也为了避免 UB 大小不固定
    gmToUbParams.lenBurst = tilingInfo->colShape * sizeof(T) / 32; // x 实际 col 大小，不包括后面追加的东西
    gmToUbParams.srcStride = 240; // 前一个尾巴和下一个的开头，gap，这里根据 token 结构需要跳跃 7168+512
    gmToUbParams.dstStride = 0; // 前一个尾巴和下一个开头，gap
 
    DataCopyParams ubToGmParams;
    ubToGmParams.sid = 0;
    ubToGmParams.nBurst = 1; // 搬运次数，
    ubToGmParams.lenBurst = tilingInfo->colShape * sizeof(T) / 32; // col 就是 token 实际列宽
    ubToGmParams.srcStride = 0; // 前一个尾巴和下一个的开头，gap
    ubToGmParams.dstStride = 0; // 前一个尾巴和下一个开头，gap
 
    GM_ADDR thisTileOutAddr = reinterpret_cast<GM_ADDR>(expandX);
    for (int i = 0; i < processTokenCnt; i++) {
        CopyGmToGm(thisTileOutAddr, thisTileWinTokenAddr, token, gmToUbParams, ubToGmParams);
        thisTileWinTokenAddr += tilingInfo->colShape * sizeof(T); // colShape 需要修改为冗余长度的 shape
        thisTileOutAddr += tilingInfo->colShape * sizeof(T);
    }
}

// 对于共享专家卡，其收到的 token 个数是固定的，直接切分搬出即可，不需要 cumsum 计算
template<typename T>
TILEOP void ShareRankCopyOut(__gm__ T *expandX, __ubuf__ uint8_t *buffer, __ubuf__ int32_t *tilingData,
    __gm__ int64_t *hcclContext)
{
    __ubuf__ TilingInfo *tilingInfo = reinterpret_cast<__ubuf__ TilingInfo *>(tilingData);
    __gm__ HcclCombinOpParam *winContext = (__gm__ HcclCombinOpParam *)(hcclContext[tilingInfo->groupIndex]);
 
    // 理论上不需要切分 TileOp, 但是为了和 moe 专家切分保持一致，按照 token cnt 切分
    uint32_t vectorCnt = tilingInfo->totalTileNum; // 理论上的核数，device 实际核数不是 48 也不影响
    // 每个共享专家处理的 moe 卡数
    uint32_t processMoeRankCnt = (winContext->rankNum - tilingInfo->shareRankCnt) / tilingInfo->shareRankCnt;
    // bs * processMoeRankCnt, bs 典型值 8，从某张卡开始连续的 64 个 token
    uint32_t tokenCntRecvFromMoeRank = processMoeRankCnt * tilingInfo->rowShape;
    uint32_t tailProcessTokenCnt = tokenCntRecvFromMoeRank / vectorCnt; // 尾块处理 token 个数
    uint32_t tileCnt = tokenCntRecvFromMoeRank % vectorCnt; // 整块个数
    uint32_t tileProcessTokenCnt = tailProcessTokenCnt + 1; // 整块处理 token 个数
 
    uint32_t recvTokenOffset = 0; // 本 TileOp 处理的 token 的偏移
    uint32_t processTokenCnt = 0;
    if (tilingInfo->tileIndex < tileCnt) {
        recvTokenOffset = tilingInfo->tileIndex * tileProcessTokenCnt;
        processTokenCnt = tileProcessTokenCnt;
    } else {
        recvTokenOffset = tileCnt * tileProcessTokenCnt + (tilingInfo->tileIndex - tileCnt) * tailProcessTokenCnt;
        processTokenCnt = tailProcessTokenCnt;
    }
    uint32_t outOffset = tilingInfo->rowShape * tilingInfo->colShape; // 共享专家本身的 8 个 token 放在最开头
    outOffset += recvTokenOffset * tilingInfo->colShape; // 元素个数
    ShareRankWinCopyOut<T>(expandX + outOffset, buffer, processTokenCnt, recvTokenOffset, tilingData, hcclContext,
        processMoeRankCnt);
}

// __gm__ half *expandX, __gm__ half *validCnt 类型有误
template<typename T, bool isSharedRank = false>
TILEOP void FFNBatching(__gm__ T *expandX, __gm__ int32_t *validCnt, __ubuf__ int32_t *buffer,
    __gm__ int32_t *gmRecvTokenCnt, __ubuf__ int32_t *tilingData, __gm__ int64_t *hcclContext)
{
    if constexpr (isSharedRank) {
        ShareRankCopyOut<T>(expandX, reinterpret_cast<__ubuf__ uint8_t *>(buffer), tilingData, hcclContext);
    } else {
        MoeRankCopyOut<T>(expandX, reinterpret_cast<__gm__ uint32_t *>(validCnt),
            reinterpret_cast<__ubuf__ uint8_t *>(buffer), reinterpret_cast<__gm__ uint32_t *>(gmRecvTokenCnt),
            tilingData, hcclContext);
    }
}

} // namespace Distributed
} // namespace TileOp

#endif
