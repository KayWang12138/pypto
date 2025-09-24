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
 * \file cube_dyn.h
 * \brief
 */

#ifndef TILE_FWK_CUBE_DYN_H
#define TILE_FWK_CUBE_DYN_H

#include "tileop_common.h"

#include <type_traits>

namespace TileOp {
constexpr uint16_t BLOCK_CUBE_M_N = 16;
constexpr uint16_t BLOCK_ALIGN_BYTE = 32;

template <typename T>
INLINE T CeilAlign(T num_1, T num_2) {
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

template <typename T>
INLINE T CeilDiv(T num_1, T num_2) {
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2;
}
/*
 * brief: dynamic l1 copy in nz2nz functions with batch
 */
template <typename GMT, typename L1T, unsigned TShape0, unsigned TShape1>
TILEOP void DynL1CopyInNZ2NZ(__cbuf__ L1T *dst, __gm__ GMT *src, unsigned GmShape0, unsigned GmShape1,
    unsigned GmOffset0, unsigned GmOffset1, unsigned curH, unsigned curW, int reserved) {
    auto inputC0Size = 32 / sizeof(L1T);
    // 计算在那个Batch块中;
    auto batchSize = curH * curW;
     // 计算当前的偏移点
    auto offsetElem = GmOffset0 * GmShape1 +GmOffset1;
    auto batchIndex = offsetElem / batchSize;
    // auto batchIndex = CalcLinearOffset(GmShape1, GmOffset0, GmOffset1) / batchSize;
    // NZ的offset转换
    auto offsetWithNZ = batchIndex * batchSize + (GmOffset1 * curH) + (GmOffset0 - batchIndex * curH) * inputC0Size;
    int32_t C0 = 32 / sizeof(GMT);
    uint16_t nBurst = TShape1 / C0;
    uint16_t lenBurst = TShape0 * C0 * sizeof(GMT) / 32;
    uint16_t srcStride = (curH - TShape0) * C0 * sizeof(GMT) / 32;
    uint16_t dstStride = 0;
    copy_gm_to_cbuf(dst, src + offsetWithNZ, 0 /*sid*/, nBurst, lenBurst, srcStride, dstStride, PAD_NONE);
}

/*
 * brief: dynamic l1 copy in nd2nz functions
 */
template <typename GMT, typename L1T, unsigned TShape0, unsigned TShape1>
TILEOP void DynL1CopyIn(__cbuf__ L1T *dst, __gm__ GMT *src, unsigned GmShape0, unsigned GmShape1, unsigned GmOffset0,
    unsigned GmOffset1, int reserved) { // ND2NZ
    src += CalcLinearOffset(GmShape1, GmOffset0, GmOffset1);

    constexpr uint16_t ndNum = 1;
    constexpr uint16_t nValue = TShape0;      // n
    constexpr uint16_t dValue = TShape1;      // d
    constexpr uint16_t srcNdMatrixStride = 0; //
    uint16_t srcDValue = GmShape1;            // D
    auto c0Size = 32 / sizeof(GMT);
    constexpr uint16_t dstNzC0Stride = TShape0; // n
    constexpr uint16_t dstNzNStride = 1;
    constexpr uint16_t dstNzMatrixStride = 1;
    if constexpr (std::is_same<GMT, int8_t>::value) {
        copy_gm_to_cbuf_multi_nd2nz_b8((__cbuf__ L1T *)dst, (__gm__ GMT *)src, 0 /*sid*/, ndNum, nValue, dValue,
            srcNdMatrixStride, srcDValue, dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    }

    if constexpr (std::is_same<GMT, half>::value || std::is_same<GMT, bfloat16_t>::value) {
        copy_gm_to_cbuf_multi_nd2nz_b16((__cbuf__ L1T *)dst, (__gm__ GMT *)src, 0 /*sid*/, ndNum, nValue, dValue,
            srcNdMatrixStride, srcDValue, dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    }

    if constexpr (std::is_same<GMT, float>::value) {
        copy_gm_to_cbuf_multi_nd2nz_b32s((__cbuf__ L1T *)dst, (__gm__ GMT *)src, 0 /*sid*/, ndNum, nValue, dValue,
            srcNdMatrixStride, srcDValue, dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    }
}

// L1 spill out scene
template <typename GMT, typename L1T, unsigned TShape0, unsigned TShape1, unsigned GmShape0, unsigned GmShape1>
TILEOP void DynL1CopyOutND(__gm__ GMT *dst, __cbuf__ L1T *src, int reserved) {
    uint16_t nBurst = TShape0;
    uint16_t lenBurst = TShape1 * sizeof(GMT) / BLOCK_SIZE;
    uint16_t srcStride = 0;
    uint16_t dstStride = (GmShape1 - TShape1) * sizeof(GMT) / BLOCK_SIZE;

    if (lenBurst == 0) {
        nBurst = 1;
        lenBurst = TShape0 * TShape1 * sizeof(GMT);
        if (lenBurst == 0) {
            lenBurst = 1;
        }
        srcStride = 0;
        dstStride = 0;
    }
    copy_cbuf_to_gm(dst, src, 0 /*sid*/, nBurst, lenBurst, srcStride, dstStride);
}

// Currently 'L1CopyOut' is ONLY used when spilling occurred, and does NOT need data format conversion. the impl
// redirect this function to L1CopyOutND directly.
template <typename GMT, typename L1T, unsigned TShape0, unsigned TShape1, unsigned GmShape0, unsigned GmShape1>
TILEOP void DynL1CopyOut(__gm__ GMT *dst, __cbuf__ L1T *src, int reserved) {
    TileOp::DynL1CopyOutND<GMT, L1T, TShape0, TShape1, GmShape0, GmShape1>(dst, src, reserved);
}

// L1 spill out scene
template <typename GMT, typename L1T, unsigned TShape0, unsigned TShape1>
TILEOP void DynL1CopyInND(__cbuf__ L1T *dst, __gm__ GMT *src, unsigned GmShape0, unsigned GmShape1, int reserved) {
    uint16_t nBurst = TShape0;
    uint16_t lenBurst = TShape1 * sizeof(GMT) / 32;
    uint16_t srcStride = 0;
    uint16_t dstStride = (GmShape1 - TShape1) * sizeof(GMT) / BLOCK_SIZE;

    if (lenBurst == 0) {
        nBurst = 1;
        lenBurst = TShape0 * TShape1 * sizeof(GMT);
        if (lenBurst == 0) {
            lenBurst = 1;
        }
        srcStride = 0;
        dstStride = 0;
    }
    copy_gm_to_cbuf(dst, src, 0 /*sid*/, nBurst, lenBurst, srcStride, dstStride, PAD_NONE);
}

template <typename GMT, typename L0CT, unsigned TShape0, unsigned TShape1, unsigned oriTShape0, unsigned oriTShape1>
TILEOP void DynL0CCopyOut(__gm__ GMT *dst, __cc__ L0CT *src, unsigned GmShape0, unsigned GmShape1, unsigned GmOffset0,
    unsigned GmOffset1, int uf) { // NZ2ND
    uint16_t MSize = oriTShape0 < (GmShape0 - GmOffset0) ? oriTShape0 : (GmShape0 - GmOffset0);
    uint16_t NSize = TShape1 < (GmShape1 - GmOffset1) ? TShape1 : (GmShape1 - GmOffset1);
    uint32_t dstStride_dst_D = GmShape1;
    uint16_t srcStride = TShape0;
    uint64_t ndNum = 1;
    uint64_t src_nd_stride = 0;
    uint64_t dst_nd_stride = 0;

    uint8_t UnitFlagMode = uf;
    uint64_t QuantPRE = NoQuant;
    uint8_t ReLUPRE = 0;
    bool channelSplit = false;
    bool NZ2ND_EN = true;

    uint64_t config = 0, nd_para = 0;
    nd_para = nd_para | (ndNum & 0xffff);
    nd_para = nd_para | ((src_nd_stride & 0xffff) << 16);
    nd_para = nd_para | ((dst_nd_stride & 0xffff) << 32);

    if (std::is_same<L0CT, float>::value) {
        if (std::is_same<GMT, half>::value) {
            QuantPRE = QuantMode_t::F322F16;
        } else if (std::is_same<GMT, bfloat16_t>::value) {
            QuantPRE = QuantMode_t::F322BF16;
        } else {
            QuantPRE = QuantMode_t::NoQuant;
        }
    }
    set_nd_para(nd_para);
    copy_matrix_cc_to_gm((__gm__ GMT *)(dst + (GmOffset0 * GmShape1) + GmOffset1), (__cc__ L0CT *)src, 0, NSize, MSize,
        dstStride_dst_D, srcStride, UnitFlagMode, QuantPRE, ReLUPRE, channelSplit, NZ2ND_EN);
}

template <typename GMT, typename L0CT, unsigned TShape0, unsigned TShape1, unsigned oriTShape0, unsigned oriTShape1,
    int isAcc>
TILEOP void DynL0CCopyOut(__gm__ GMT *dst, __cc__ L0CT *src, unsigned GmShape0, unsigned GmShape1, unsigned GmOffset0,
    unsigned GmOffset1, int uf) {
    SetAtomicAdd<GMT>();
    DynL0CCopyOut<GMT, L0CT, TShape0, TShape1, oriTShape0, oriTShape1>(
        dst, src, GmShape0, GmShape1, GmOffset0, GmOffset1, uf);
    if constexpr (isAcc == 1) {
        set_atomic_none();
    }
}

/* -------------------------------------------- support unaligned scene --------------------------------------------*/
/*
 * brief: dynamic l1 copy in nz2nz functions with batch
 */
template <typename GMT, typename L1T>
TILEOP void DynL1CopyInNZ2NZ(__cbuf__ L1T *dst, __gm__ GMT *src, unsigned TShape0, unsigned TShape1, unsigned GmShape0,
    unsigned GmShape1, unsigned GmOffset0, unsigned GmOffset1, unsigned curH, unsigned curW, int reserved) {
    auto inputC0Size = 32 / sizeof(L1T);
    // 计算在那个Batch块中;
    auto batchSize = curH * curW;
    // 计算当前的偏移点
    auto offsetElem = GmOffset0 * GmShape1 + GmOffset1;
    auto batchIndex = offsetElem / batchSize;
    // auto batchIndex = CalcLinearOffset(GmShape1, GmOffset0, GmOffset1) / batchSize;
    // NZ的offset转换
    auto offsetWithNZ = batchIndex * batchSize + (GmOffset1 * curH) + (GmOffset0 - batchIndex * curH) * inputC0Size;
    int32_t C0 = 32 / sizeof(GMT);
    uint16_t nBurst = TShape1 / C0;
    uint16_t lenBurst = TShape0 * C0 * sizeof(GMT) / 32;
    uint16_t srcStride = (curH - TShape0) * C0 * sizeof(GMT) / 32;
    uint16_t dstStride = 0;
    copy_gm_to_cbuf(dst, src + offsetWithNZ, 0 /*sid*/, nBurst, lenBurst, srcStride, dstStride, PAD_NONE);
}

/*
 * brief: dynamic l1 copy in nd2nz functions
 */
template <typename GMT, typename L1T>
TILEOP void DynL1CopyIn(__cbuf__ L1T *dst, __gm__ GMT *src, unsigned TShape0, unsigned TShape1, unsigned GmShape0,
    unsigned GmShape1, unsigned GmOffset0, unsigned GmOffset1, int reserved) { // ND2NZ
    src += CalcLinearOffset(GmShape1, GmOffset0, GmOffset1);
    uint16_t nValue = TShape0;
    uint16_t dValue = TShape1;
    uint16_t srcDValue = GmShape1;
    uint16_t dstNzC0Stride = CeilAlign<uint16_t>(TShape0, BLOCK_CUBE_M_N);

    constexpr uint16_t ndNum = 1;
    constexpr uint16_t srcNdMatrixStride = 0;
    constexpr uint16_t dstNzNStride = 1;
    constexpr uint16_t dstNzMatrixStride = 1;

    if constexpr (std::is_same<GMT, int8_t>::value) {
        copy_gm_to_cbuf_multi_nd2nz_b8((__cbuf__ L1T *)dst, (__gm__ GMT *)src, 0 /*sid*/, ndNum, nValue, dValue,
            srcNdMatrixStride, srcDValue, dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    }

    if constexpr (std::is_same<GMT, half>::value || std::is_same<GMT, bfloat16_t>::value) {
        copy_gm_to_cbuf_multi_nd2nz_b16((__cbuf__ L1T *)dst, (__gm__ GMT *)src, 0 /*sid*/, ndNum, nValue, dValue,
            srcNdMatrixStride, srcDValue, dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    }

    if constexpr (std::is_same<GMT, float>::value) {
        copy_gm_to_cbuf_multi_nd2nz_b32s((__cbuf__ L1T *)dst, (__gm__ GMT *)src, 0 /*sid*/, ndNum, nValue, dValue,
            srcNdMatrixStride, srcDValue, dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    }
}

// L1 spill out scene
template <typename GMT, typename L1T>
TILEOP void DynL1CopyOutND(__gm__ GMT *dst, __cbuf__ L1T *src, unsigned TShape0, unsigned TShape1, unsigned GmShape0,
    unsigned GmShape1, int reserved) {
    uint16_t nBurst = TShape0;
    uint16_t lenBurst = TShape1 * sizeof(GMT) / BLOCK_SIZE;
    uint16_t srcStride = 0;
    uint16_t dstStride = (GmShape1 - TShape1) * sizeof(GMT) / BLOCK_SIZE;

    if (lenBurst == 0) {
        nBurst = 1;
        lenBurst = TShape0 * TShape1 * sizeof(GMT);
        if (lenBurst == 0) {
            lenBurst = 1;
        }
        srcStride = 0;
        dstStride = 0;
    }
    copy_cbuf_to_gm(dst, src, 0 /*sid*/, nBurst, lenBurst, srcStride, dstStride);
}

// Currently 'L1CopyOut' is ONLY used when spilling occurred, and does NOT need data format conversion. the impl
// redirect this function to L1CopyOutND directly.
template <typename GMT, typename L1T>
TILEOP void DynL1CopyOut(__gm__ GMT *dst, __cbuf__ L1T *src, unsigned TShape0, unsigned TShape1, unsigned GmShape0,
    unsigned GmShape1, int reserved) {
    TileOp::DynL1CopyOutND<GMT, L1T>(dst, src, TShape0, TShape1, GmShape0, GmShape1, reserved);
}

// L1 spill out scene
template <typename GMT, typename L1T>
TILEOP void DynL1CopyInND(__cbuf__ L1T *dst, __gm__ GMT *src, unsigned TShape0, unsigned TShape1, unsigned GmShape0,
    unsigned GmShape1, int reserved) {
    uint16_t nBurst = TShape0;
    uint16_t lenBurst = TShape1 * sizeof(GMT) / 32;
    uint16_t srcStride = 0;
    uint16_t dstStride = (GmShape1 - TShape1) * sizeof(GMT) / BLOCK_SIZE;

    if (lenBurst == 0) {
        nBurst = 1;
        lenBurst = TShape0 * TShape1 * sizeof(GMT);
        if (lenBurst == 0) {
            lenBurst = 1;
        }
        srcStride = 0;
        dstStride = 0;
    }
    copy_gm_to_cbuf(dst, src, 0 /*sid*/, nBurst, lenBurst, srcStride, dstStride, PAD_NONE);
}

template <typename GMT, typename L0CT, bool enableNZ2ND>
TILEOP void DynL0CCopyOut(__gm__ GMT *dst, __cc__ L0CT *src, unsigned oriTShape0, unsigned oriTShape1, unsigned GmShape0,
    unsigned GmShape1, unsigned GmOffset0, unsigned GmOffset1, unsigned curH, unsigned curW, int uf) {
    uint16_t mSize = oriTShape0;
    uint16_t nSize = oriTShape1; // should be multiples of 8 when fp32 & channel split
    uint32_t dstStrideDstD = GmShape1;
    uint16_t srcStride = CeilAlign<uint16_t>(oriTShape0, BLOCK_CUBE_M_N);
    bool channelSplit = false;
    int64_t gmOffset = (GmOffset0 * GmShape1) + GmOffset1;

    if constexpr (!enableNZ2ND) {
        // s32搬出不涉及channel split，因此C0=16
        int64_t c0Size = std::is_same<GMT, int32_t>::value ? BLOCK_CUBE_M_N : BLOCK_ALIGN_BYTE / sizeof(GMT);
        nSize = CeilAlign<uint16_t>(nSize, c0Size);
        int64_t wAlign = CeilAlign<int64_t>(curW, c0Size);
        int64_t elemPerBatch = curH * wAlign;
        int64_t batchIdx = gmOffset / elemPerBatch;
        gmOffset = batchIdx * elemPerBatch + (GmOffset1 * curH) + (GmOffset0 - batchIdx * curH) * c0Size;
        // fp32搬出默认开启channel split
        channelSplit = std::is_same<GMT, float>::value;
        // dst stride between the start addresses of different bursts in unit of 32B
        // 2含义：int32 NZ搬出场景（C0=16），此处dstStride需乘2使得内轴按64B为单位做偏移计算
        dstStrideDstD = std::is_same<GMT, int32_t>::value ? curH * 2 : curH;
    }

    uint64_t ndNum = 1;
    uint64_t srcNdStride = 0;
    uint64_t dstNdStride = 0;
    uint64_t ndPara = 0;
    ndPara = ndPara | (ndNum & 0xffff);
    ndPara = ndPara | ((srcNdStride & 0xffff) << 16);
    ndPara = ndPara | ((dstNdStride & 0xffff) << 32);
    set_nd_para(ndPara);

    uint8_t unitFlagMode = uf;
    uint64_t quantPre = NoQuant;
    uint8_t reluPre = 0;
    if (std::is_same<L0CT, float>::value) {
        if (std::is_same<GMT, half>::value) {
            quantPre = QuantMode_t::F322F16;
        } else if (std::is_same<GMT, bfloat16_t>::value) {
            quantPre = QuantMode_t::F322BF16;
        } else {
            quantPre = QuantMode_t::NoQuant;
        }
    }

    copy_matrix_cc_to_gm((__gm__ GMT *)(dst + gmOffset), (__cc__ L0CT *)src, 0, nSize, mSize, dstStrideDstD,
        srcStride, unitFlagMode, quantPre, reluPre, channelSplit, enableNZ2ND);
}

template <typename GMT, typename L0CT, int isAcc, bool enableNZ2ND>
TILEOP void DynL0CCopyOut(__gm__ GMT *dst, __cc__ L0CT *src, unsigned oriTShape0, unsigned oriTShape1,
    unsigned GmShape0, unsigned GmShape1, unsigned GmOffset0, unsigned GmOffset1, unsigned curH, unsigned curW,
    int uf) {
    SetAtomicAdd<GMT>();
    DynL0CCopyOut<GMT, L0CT, enableNZ2ND>(
        dst, src, oriTShape0, oriTShape1, GmShape0, GmShape1, GmOffset0, GmOffset1, curH, curW, uf);
    if constexpr (isAcc == 1) {
        set_atomic_none();
    }
}

// Nz2Zz
template <typename T, unsigned Offset0, unsigned Offset1>
TILEOP void DynL1ToL0A(__ca__ T *dst, __cbuf__ T *src, unsigned dstM, unsigned dstK, unsigned srcM, unsigned srcK) {
    constexpr uint16_t blockCubeK = BLOCK_ALIGN_BYTE / sizeof(T);
    dstM = CeilAlign<uint16_t>(dstM, BLOCK_CUBE_M_N);
    dstK = CeilAlign<uint16_t>(dstK, blockCubeK);
    srcM = CeilAlign<uint16_t>(srcM, BLOCK_CUBE_M_N);
    srcK = CeilAlign<uint16_t>(srcK, blockCubeK);

    uint8_t repeat = dstK / blockCubeK;
    uint16_t srcStride = srcM / BLOCK_CUBE_M_N;
    uint16_t dstStride = 0;
    int32_t dstOffset = 0;
    int32_t dstOffsetStep = BLOCK_CUBE_M_N * dstK;
    int32_t srcOffset = Offset0 * blockCubeK + Offset1 *srcM;
    int32_t srcOffsetStep = BLOCK_CUBE_M_N * blockCubeK;

    if constexpr (std::is_same<T, float>::value) {
        uint64_t config = srcM | (1 << 16);
        set_fmatrix(config);
        dstK = CeilAlign<uint16_t>(dstK, BLOCK_CUBE_M_N);
        img2colv2_cbuf_to_ca(dst,src, dstK, dstM, 0, 0, 1, 1, 1, 1, 1, 1, false, false, false, false, srcK);
        return;
    }

    for (int32_t mIdx = 0; mIdx < static_cast<int32_t>(dstM / BLOCK_CUBE_M_N); ++mIdx) {
        load_cbuf_to_ca(dst + dstOffset, src + srcOffset, 0, repeat, srcStride, dstStride, 0, 0, inc);
        dstOffset += dstOffsetStep;
        srcOffset += srcOffsetStep;
    }
}

// Nz2Zz
template <typename T, unsigned Offset0, unsigned Offset1>
TILEOP void DynL1ToL0At(__ca__ T *dst, __cbuf__ T *src, unsigned dstM, unsigned dstK, unsigned srcK, unsigned srcM) {
    constexpr uint16_t c0Size = BLOCK_ALIGN_BYTE / sizeof(T);
    dstM = CeilAlign<uint16_t>(dstM, c0Size);
    dstK = CeilAlign<uint16_t>(dstK, BLOCK_CUBE_M_N);
    srcM = CeilAlign<uint16_t>(srcM, c0Size);
    srcK = CeilAlign<uint16_t>(srcK, BLOCK_CUBE_M_N);

    if constexpr (std::is_same<T, float>::value) {
        uint64_t config = srcK | (1 << 16);
        set_fmatrix(config);
        img2colv2_cbuf_to_ca(dst,src, dstM, dstK, 0, 0, 1, 1, 1, 1, 1, 1, false, false, true, false, srcM);
        return;
    }

    if constexpr (std::is_same<T, int8_t>::value) {
        uint8_t repeat = dstK / c0Size;
        uint16_t srcStride = 1;
        uint16_t dstStride = 0;
        uint16_t dstFracStride = dstK / c0Size - 1;
        int32_t dstOffset = 0;
        int32_t srcOffset = 0;
        int32_t dstOffsetStep = dstK * c0Size;
        int32_t srcOffsetStep = srcK * c0Size;

        for (int32_t mIdx = 0; mIdx < static_cast<int32_t>(dstM / c0Size); ++mIdx) {
            load_cbuf_to_ca_transpose(dst + dstOffset, src + srcOffset, 0,
                repeat, srcStride, dstStride, inc, dstFracStride);
                dstOffset += dstOffsetStep;
                srcOffset += srcOffsetStep;
        }
        return;
    }
    if (dstK == srcK) {
        uint8_t repeat = (dstK / BLOCK_CUBE_M_N) * (dstM / c0Size); // 表示搬运的次数
        uint16_t srcStride = 1;                                     // fract between fract
        uint16_t dstStride = 0;                                     // gap between fract
        load_cbuf_to_ca(dst, src, 0, repeat, srcStride, dstStride, 0, 1, inc);
    } else {
        uint8_t repeat = dstK / BLOCK_CUBE_M_N;
        uint16_t srcStride = 1;
        uint16_t dstStride = 0;
        int32_t dstOffset = 0;
        int32_t dstOffsetStep = BLOCK_CUBE_M_N * dstK;
        int32_t srcOffset = Offset0 * c0Size + Offset1 * srcK;
        int32_t srcOffsetStep = srcK * c0Size;

        for (int32_t mIdx = 0; mIdx < static_cast<int32_t>(dstM / c0Size); ++mIdx) {
            load_cbuf_to_ca(dst + dstOffset, src + srcOffset, 0, repeat, srcStride, dstStride, 0, 1, inc);
            dstOffset += dstOffsetStep;
            srcOffset += srcOffsetStep;
        }
    }
}

// Nz2Zn
template <typename T, unsigned Offset0, unsigned Offset1>
TILEOP void DynL1ToL0B(__cb__ T *dst, __cbuf__ T *src, unsigned dstK, unsigned dstN, unsigned srcK, unsigned srcN) {
    auto nBlockSize = 32;
    int64_t frac_num = 32 / sizeof(T);
    dstK = (dstK + frac_num - 1) / frac_num * frac_num;
    dstN = (dstN + frac_num - 1) / frac_num * frac_num;
    srcN = (srcN + frac_num - 1) / frac_num * frac_num;
    srcK = (srcK + frac_num - 1) / frac_num * frac_num;

    if constexpr (std::is_same<T, float>::value) {
        constexpr uint16_t C0Size = BLOCK_ALIGN_BYTE / sizeof(T);
        // need to enable k-alignment in mmad to copy with even numbers align to 8 should actually be odd numbers to 8.
        // load3D automatically to 16
        dstK = CeilAlign<uint16_t>(dstK, BLOCK_CUBE_M_N);
        dstN = CeilAlign<uint16_t>(dstN, BLOCK_CUBE_M_N);
        srcN = CeilAlign<uint16_t>(srcN, C0Size);
        srcK = CeilAlign<uint16_t>(srcK, BLOCK_CUBE_M_N);
        // set featureMap 0-15 for w(srcK) 16-31(1)
        uint64_t config = srcK | (1 << 16);
        set_fmatrix_b(config);
        img2colv2_cbuf_to_cb(dst,src, dstN, dstK, 0, 0, 1, 1, 1, 1, 1, 1, false, false, false, true, srcN);
        return;
    }

    if constexpr (std::is_same<T, int8_t>::value) {
        uint8_t repeat = dstK / nBlockSize;
        uint16_t srcStride = 1;
        uint16_t dstStride = dstN / BLOCK_CUBE_M_N - 1;
        uint16_t dstFracStride = 0;
        int32_t dstOffset = 0;
        int32_t srcOffset = 0;
        int32_t dstOffsetStep = nBlockSize * nBlockSize;
        int32_t srcOffsetStep = srcK * nBlockSize;

        for (int32_t nIdx = 0; nIdx < static_cast<int32_t>(dstN / nBlockSize); ++nIdx) {
            load_cbuf_to_cb_transpose(dst + dstOffset, src + srcOffset, 0,
                repeat, srcStride, dstStride, inc, dstFracStride);
                dstOffset += dstOffsetStep;
                srcOffset += srcOffsetStep;
        }
        return;
    }
    if constexpr (std::is_same<T, float>::value) {
        uint8_t repeat = dstN / BLOCK_CUBE_M_N;
        uint16_t srcStride = 1;
        uint16_t dstStride = 0;
        uint16_t dstFracStride = repeat - 1;
        load_cbuf_to_cb_transpose(dst, src, 0, repeat, srcStride, dstStride, inc, dstFracStride);
        return;
    }
    // L1 n1k1k0no   -> l0b  k1n1n0k0
    int64_t k_frac = dstK / frac_num; // B32
    uint8_t repeat = dstN / 16; //
    uint16_t srcStride = srcK / frac_num;
    uint16_t dstStride = 0; // gap;

    for (int64_t k_idx = 0; k_idx < k_frac; ++k_idx) {
        load_cbuf_to_cb(dst + k_idx * frac_num * dstN,
            src + k_idx * 16 * frac_num + (Offset0 * 16 + Offset1 * srcK), 0, repeat, srcStride, dstStride, 0, 1,
            inc);
    }
}

// Nz2Zz
template <typename T, unsigned Offset0, unsigned Offset1>
TILEOP void DynL1ToL0Bt(__cb__ T *dst, __cbuf__ T *src, unsigned dstK, unsigned dstN, unsigned srcN, unsigned srcK) {
    int64_t frac_num = 32 / sizeof(T);

    // no need to use load3D as the frac is same for l1 and l0
    constexpr uint16_t c0Size = BLOCK_ALIGN_BYTE / sizeof(T);
    dstN = CeilAlign<uint16_t>(dstN, BLOCK_CUBE_M_N);
    dstK = CeilAlign<uint16_t>(dstK, c0Size);
    srcN = CeilAlign<uint16_t>(srcN, BLOCK_CUBE_M_N);
    srcK = CeilAlign<uint16_t>(srcK, c0Size);

    if (dstN == srcN) {
        uint8_t repeat = dstN / BLOCK_CUBE_M_N * dstK / c0Size;
        constexpr uint16_t srcStride = 1;
        constexpr uint16_t dstStride = 0;

        load_cbuf_to_cb(dst, src + (Offset0 * frac_num + Offset1 * srcN), (uint16_t)0, repeat, srcStride, dstStride,
            (uint8_t)0, (bool)0, (addr_cal_mode_t)0);
    } else {
        int64_t k_frac = dstK / frac_num;
        uint8_t repeat = dstN / 16;
        constexpr uint16_t srcStride = 1;
        constexpr uint16_t dstStride = 0;

        for (int64_t k_idx = 0; k_idx < k_frac; ++k_idx) {
            load_cbuf_to_cb(dst + k_idx * frac_num * dstN,
                src + k_idx * srcN * frac_num + (Offset0 * frac_num + Offset1 * srcN), (uint16_t)0, repeat, srcStride,
                dstStride, (uint8_t)0, (bool)0, (addr_cal_mode_t)0);
        }
    }
}

template <typename Tc, typename Ta, typename Tb, unsigned Offset0, unsigned Offset1>
TILEOP void DynTmad(__cc__ Tc *c, __ca__ Ta *a, __cb__ Tb *b, uint16_t m, uint16_t k, uint16_t n, bool zero_C, int uf,
    unsigned L0CShape0, unsigned L0CShape1) {
    uint8_t unitFlag = uf;
    bool kDirectionAlign = true; // aligned to 8 for fp32
    bool cmatrixSource = false;  // true means bias
    m = CeilAlign<uint16_t>(m, BLOCK_CUBE_M_N);
    if constexpr (std::is_same<Tb, int8_t>::value) {
        n = CeilAlign<uint16_t>(n, 32);
    }

    mad((__cc__ Tc *)(c + (Offset0 * 16) + Offset1 * L0CShape0), a, b, m, k, n, unitFlag, kDirectionAlign,
        cmatrixSource, zero_C);
    pipe_barrier(PIPE_M);
}

} // namespace TileOp
#endif // TILE_FWK_CUBE_DYN_H
