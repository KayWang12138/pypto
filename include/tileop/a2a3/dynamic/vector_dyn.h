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
 * \file vector_dyn.h
 * \brief
 */

#include "tileop_common.h"
#include "vector.h"

#include <type_traits>

#ifndef TILE_FWK_VECTOR_DYN_H
#define TILE_FWK_VECTOR_DYN_H

namespace TileOp {
// ADD
#define T_BIN DynTadd_
#define V_BIN_FUNC vadd
#define T_BIN_VS DynTadds_
#define V_BIN_FUNC_VS vadds
#include "vector_bin_dyn.h"
#undef T_BIN_VS
#undef V_BIN_FUNC_VS
#undef T_BIN
#undef V_BIN_FUNC
// SUB
#define T_BIN DynTsub_
#define V_BIN_FUNC vsub
#define T_BIN_VS DynTsubs_
#define V_BIN_FUNC_VS vadds
#define VS_SUB
#include "vector_bin_dyn.h"
#undef VS_SUB
#undef T_BIN_VS
#undef V_BIN_FUNC_VS
#undef T_BIN
#undef V_BIN_FUNC
// MUL
#define T_BIN DynTmul_
#define V_BIN_FUNC vmul
#define T_BIN_VS DynTmuls_
#define V_BIN_FUNC_VS vmuls
#include "vector_bin_dyn.h"
#undef T_BIN_VS
#undef V_BIN_FUNC_VS
#undef T_BIN
#undef V_BIN_FUNC
// DIV
#define T_BIN DynTdiv_
#define V_BIN_FUNC vdiv
#define T_BIN_VS DynTdivs_
#define V_BIN_FUNC_VS vmuls
#define VS_DIV
#include "vector_bin_dyn.h"
#undef VS_DIV
#undef T_BIN_VS
#undef V_BIN_FUNC_VS
#undef T_BIN
#undef V_BIN_FUNC
// MAX
#define T_BIN DynTmax_
#define V_BIN_FUNC vmax
#include "vector_bin_dyn.h"
#undef T_BIN
#undef V_BIN_FUNC

// ADDWITHBRC
#define T_BIN_BRC DynTaddbrc_
#define V_BIN_BRC_FUNC vadd
#include "vector_bin_brc_dyn.h"
#undef T_BIN_BRC
#undef V_BIN_BRC_FUNC

// SUBWITHBRC
#define T_BIN_BRC DynTsubbrc_
#define V_BIN_BRC_FUNC vsub
#include "vector_bin_brc_dyn.h"
#undef T_BIN_BRC
#undef V_BIN_BRC_FUNC

// MULWITHBRC
#define T_BIN_BRC DynTmulbrc_
#define V_BIN_BRC_FUNC vmul
#include "vector_bin_brc_dyn.h"
#undef T_BIN_BRC
#undef V_BIN_BRC_FUNC

// MAXWITHBRC
#define T_BIN_BRC DynTmaxbrc_
#define V_BIN_BRC_FUNC vmax
#include "vector_bin_brc_dyn.h"
#undef T_BIN_BRC
#undef V_BIN_BRC_FUNC

// DIVWITHBRC
#define T_BIN_BRC DynTdivbrc_
#define V_BIN_BRC_FUNC vdiv
#include "vector_bin_brc_dyn.h"
#undef T_BIN_BRC
#undef V_BIN_BRC_FUNC

// Unary op
// EXP
#define T_UNA DynTexp_
#define V_UNA_FUNC vexp
#include "vector_una_dyn.h"
#undef T_UNA
#undef V_UNA_FUNC
// RECIPROCAL
#define T_UNA DynTrec_
#define V_UNA_FUNC vrec
#include "vector_una_dyn.h"
#undef T_UNA
#undef V_UNA_FUNC
// SQRT
#define T_UNA DynTsqrt_
#define V_UNA_FUNC vsqrt
#include "vector_una_dyn.h"
#undef T_UNA
#undef V_UNA_FUNC
// ABS
#define T_UNA DynTabs_
#define V_UNA_FUNC vabs
#include "vector_una_dyn.h"
#undef T_UNA
#undef V_UNA_FUNC

template <typename T, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned srcRawShape1,
    unsigned srcRawShape2, unsigned axis0, unsigned axis1>
TILEOP void DynTtransposeDataMoveBase(__gm__ T *dst, __ubuf__ T *src, unsigned dstShape1, unsigned dstShape2) {
    if constexpr (axis0 == 0 && axis1 == 1) {
        __gm__ T *dst_ = dst;
        __ubuf__ T *src_ = src;
        unsigned nBurst = TShape1;
        unsigned lenBurst = TShape2 * sizeof(T);
        unsigned srcStride = 0;
        unsigned dstStride = (dstShape1 - 1) * dstShape2 * sizeof(T);
        for (int b = 0; b < TShape0; b++) {
            copy_ubuf_to_gm_align_b32(dst_, src_, 0, nBurst, lenBurst, 0, 0, srcStride, dstStride);
            dst_ += dstShape2;
            src_ += srcRawShape1 * srcRawShape2;
        }
    } else {
        static_assert(sizeof(T) == 0, "Unsupport transpose axis");
    }
}

template <typename T, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned TShape3, unsigned srcRawShape1,
    unsigned srcRawShape2, unsigned srcRawShape3, unsigned axis0, unsigned axis1>
TILEOP void DynTtransposeDataMove(__gm__ T *dst, __ubuf__ T *src, unsigned dstShape0, unsigned dstShape1,
    unsigned dstShape2, unsigned dstShape3, unsigned GmOffset0, unsigned GmOffset1, unsigned GmOffset2,
    unsigned GmOffset3) {
    if constexpr (axis0 == 1 && axis1 == 2) {
        __gm__ T *dst_ =
            dst + CalcLinearOffset(dstShape1, dstShape2, dstShape3, GmOffset0, GmOffset1, GmOffset2, GmOffset3);
        __ubuf__ T *src_ = src;
        for (int b = 0; b < TShape0; b++) {
            DynTtransposeDataMoveBase<T, TShape1, TShape2, TShape3, srcRawShape2, srcRawShape3, axis0 - 1, axis1 - 1>(
                dst_, src_, dstShape2, dstShape3);
            dst_ += dstShape1 * dstShape2 * dstShape3;
            src_ += srcRawShape1 * srcRawShape2 * srcRawShape3;
        }
    } else {
        static_assert(sizeof(T) == 0, "Unsupport transpose axis");
    }
}

/* ------------------------------------- support unaligned scene -------------------------------------*/

// The src data remains unchanged.
// T: fp32. support: OS0 <= REPEAT_MAX
template <typename T, unsigned DS, unsigned SS, unsigned TBS>
TILEOP void DynTrowsumsingle_(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *tmp, unsigned OS0, unsigned OS1) {
    //    OS0 <= REPEAT_MAX
    uint64_t srcRepeatPerRow = static_cast<uint64_t>(OS1 * sizeof(T) / REPEAT_BYTE);
    uint8_t srcRepeatStride = SS * sizeof(T) / BLOCK_SIZE;
    constexpr unsigned nElemPerRepeat = REPEAT_BYTE / sizeof(T);
    unsigned remain = OS1 % nElemPerRepeat;
    if (srcRepeatPerRow == 1 && remain == 0) {
        vcadd(dst, src, OS0, (uint16_t)DS, (uint16_t)1ULL, (uint16_t)srcRepeatStride, false);
        return;
    }

    if (OS0 <= REPEAT_MAX && srcRepeatPerRow < 1 && remain > 0) {
        SetContinuousMask(OS1);
        vcadd(dst, src, OS0, (uint16_t)DS, (uint16_t)1ULL, (uint16_t)srcRepeatStride, false);
        set_vector_mask(-1, -1);
        return;
    }

    // NEXTNEXT: delete when new tileOp which force to combine axis is complete
    if (OS1 == SS) {
        if (OS0 % 8 == 0 && SS == 1024 && OS0 * 16 <= REPEAT_MAX) {
            vcgadd(tmp, src, OS0 * 16, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL); // [m,1024] -> [m,128]
            pipe_barrier(PIPE_V);
            vadd(tmp, tmp + 64, tmp, OS0, 1, 1, 1, 16, 16, 16); // [m,128] -> [m,64]
            pipe_barrier(PIPE_V);
            vcgadd(tmp, tmp, OS0, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)16ULL); // [m,64] -> [m,8]
            pipe_barrier(PIPE_V);
            vcgadd(dst, tmp, OS0 / 8, (uint16_t)DS, (uint16_t)1ULL, (uint16_t)8ULL); // [m,8] -> [m,1]
            pipe_barrier(PIPE_V);
            return;

        } else if (OS0 % 8 == 0 && SS == 512 && OS0 * 8 <= REPEAT_MAX) {
            vcgadd(tmp, src, OS0 * 8, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL); // [m,512] -> [m,64]
            pipe_barrier(PIPE_V);
            vcgadd(tmp, tmp, OS0, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL); // [m,64] -> [m,8]
            pipe_barrier(PIPE_V);
            vcgadd(dst, tmp, OS0 / 8, (uint16_t)DS, (uint16_t)1ULL, (uint16_t)8ULL); // [m,8] -> [m,1]
            pipe_barrier(PIPE_V);
            return;
        }
    }

    constexpr uint16_t tmpRepeatStride = TBS * sizeof(T) / BLOCK_SIZE;
    if constexpr (tmpRepeatStride < BLOCK_MAX_PER_REPEAT) {
        // work around for ccec compiling check; If delete will cause compiling error for "copy_ubuf_to_ubuf" after
        // which reminder the range of 7th parameter must be [0, 65535]
        return;
    }

    unsigned curLen = srcRepeatPerRow;
    for (unsigned i = 0; i < curLen / 2; i++) {
        vadd(tmp + i * nElemPerRepeat, src + i * 2 * nElemPerRepeat, src + (i * 2 + 1) * nElemPerRepeat, OS0, 1, 1, 1,
            tmpRepeatStride, srcRepeatStride, srcRepeatStride);
    }
    pipe_barrier(PIPE_V);
    if (curLen == 1 && remain > 0) {
        copy_ubuf_to_ubuf(tmp, src, 0, OS0, BLOCK_MAX_PER_REPEAT, srcRepeatStride - BLOCK_MAX_PER_REPEAT,
            tmpRepeatStride - BLOCK_MAX_PER_REPEAT);
        pipe_barrier(PIPE_V);
    } else if (curLen % 2 > 0) {
        vadd(tmp, tmp, src + (curLen - 1) * nElemPerRepeat, OS0, 1, 1, 1, tmpRepeatStride, tmpRepeatStride,
            srcRepeatStride);
        pipe_barrier(PIPE_V);
    }

    if (remain > 0) {
        unsigned repeatOffset = curLen == 1 ? 0 : curLen / 2 - 1;
        SetContinuousMask(remain);
        vadd(tmp + repeatOffset * nElemPerRepeat, src + curLen * nElemPerRepeat, tmp + repeatOffset * nElemPerRepeat,
            OS0, 1, 1, 1, tmpRepeatStride, srcRepeatStride, tmpRepeatStride);
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
    }

    curLen = curLen / 2;
    bool mergeLast = true;
    while (curLen > 1) {
        for (unsigned i = 0; i < curLen / 2; i++) {
            vadd(tmp + i * nElemPerRepeat, tmp + i * 2 * nElemPerRepeat, tmp + (i * 2 + 1) * nElemPerRepeat, OS0, 1, 1,
                1, tmpRepeatStride, tmpRepeatStride, tmpRepeatStride);
        }
        unsigned loopRemain = curLen % 2;
        curLen = curLen / 2;
        if (loopRemain > 0) {
            pipe_barrier(PIPE_V);
            vadd(tmp + (curLen - 1) * nElemPerRepeat /*last repeat of new curLen*/,
                tmp + curLen * 2 * nElemPerRepeat /*remain repeat*/, tmp + (curLen - 1) * nElemPerRepeat, OS0, 1, 1, 1,
                tmpRepeatStride, tmpRepeatStride, tmpRepeatStride);
        }
        pipe_barrier(PIPE_V);
    }
    pipe_barrier(PIPE_V);
    vcadd(dst, tmp, OS0, (uint16_t)DS, (uint16_t)1ULL, tmpRepeatStride, false);
}

template <typename T, unsigned DS1, unsigned DS2, unsigned DS3, unsigned SS1, unsigned SS2, unsigned SS3, unsigned TBS3>
TILEOP void DynTrowsumsingle_(
    __ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *tmp, unsigned OS0, unsigned OS1, unsigned OS2, unsigned OS3) {
    static_assert(SS3 * sizeof(T) % BLOCK_SIZE == 0);
    for (int i = 0; i < OS0; ++i) {
        __ubuf__ T *dst_ = dst;
        __ubuf__ T *src_ = src;
        for (int j = 0; j < OS1; ++j) {
            TileOp::DynTrowsumsingle_<T, DS3, SS3, TBS3>(dst_, src_, tmp, OS2, OS3);
            dst_ += DS3 * DS2;
            src_ += SS3 * SS2;
            pipe_barrier(PIPE_V);
        }
        dst += DS1 * DS2 * DS3;
        src += SS1 * SS2 * SS3;
    }
}

// The src data remains unchanged.
// T: fp32. support: OS0 <= REPEAT_MAX
template <typename T, unsigned DS, unsigned SS, unsigned TBS>
TILEOP void DynTrowmaxsingle_(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *tmp, unsigned OS0, unsigned OS1) {
    //    OS0 <= REPEAT_MAX
    uint64_t srcRepeatPerRow = static_cast<uint64_t>(OS1 * sizeof(T) / REPEAT_BYTE);
    unsigned srcRepeatStride = SS * sizeof(T) / BLOCK_SIZE;
    constexpr unsigned nElemPerRepeat = REPEAT_BYTE / sizeof(T);
    unsigned remain = OS1 % nElemPerRepeat;
    if (srcRepeatPerRow == 1 && OS0 <= REPEAT_MAX && remain == 0) {
        vcmax(dst, src, OS0, (uint16_t)DS, (uint16_t)1ULL, (uint16_t)srcRepeatStride, ONLY_VALUE);
        return;
    }

    if (OS0 <= REPEAT_MAX && srcRepeatPerRow < 1 && remain > 0) {
        SetContinuousMask(OS1);
        vcmax(dst, src, OS0, (uint16_t)DS, (uint16_t)1ULL, (uint16_t)srcRepeatStride, ONLY_VALUE);
        set_vector_mask(-1, -1);
        return;
    }

    // NEXTNEXT: delete when new tileOp which force to combine axis is complete
    if (OS1 == SS) {
        if (OS0 % 8 == 0 && SS == 1024 && OS0 * 16 <= REPEAT_MAX) {
            vcgmax(tmp, src, OS0 * 16, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL); // [m,1024] -> [m,128]
            pipe_barrier(PIPE_V);
            vmax(tmp, tmp + 64, tmp, OS0, 1, 1, 1, 16, 16, 16); // [m,128] -> [m,64]
            pipe_barrier(PIPE_V);
            vcgmax(tmp, tmp, OS0, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)16ULL); // [m,64] -> [m,8]
            pipe_barrier(PIPE_V);
            vcgmax(dst, tmp, OS0 / 8, (uint16_t)DS, (uint16_t)1ULL, (uint16_t)8ULL); // [m,8] -> [m,1]
            pipe_barrier(PIPE_V);
            return;

        } else if (OS0 % 8 == 0 && SS == 512 && OS0 * 8 <= REPEAT_MAX) {
            vcgmax(tmp, src, OS0 * 8, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL); // [m,512] -> [m,64]
            pipe_barrier(PIPE_V);
            vcgmax(tmp, tmp, OS0, (uint16_t)1ULL, (uint16_t)1ULL, (uint16_t)8ULL); // [m,64] -> [m,8]
            pipe_barrier(PIPE_V);
            vcgmax(dst, tmp, OS0 / 8, (uint16_t)DS, (uint16_t)1ULL, (uint16_t)8ULL); // [m,8] -> [m,1]
            pipe_barrier(PIPE_V);
            return;
        }
    }
    uint16_t tmpRepeatStride = TBS * sizeof(T) / BLOCK_SIZE;
    if (srcRepeatPerRow == 1 && remain > 0) {
        copy_ubuf_to_ubuf(tmp, src, 0, OS0, BLOCK_MAX_PER_REPEAT, srcRepeatStride - BLOCK_MAX_PER_REPEAT,
            tmpRepeatStride - BLOCK_MAX_PER_REPEAT);
    } else {
        if ((tmpRepeatStride <= REPEAT_MAX) && (srcRepeatStride <= REPEAT_MAX)) {
            vmax(tmp, src, src + nElemPerRepeat, OS0, 1, 1, 1, tmpRepeatStride, srcRepeatStride, srcRepeatStride);
        } else {
            for (int i = 0; i < OS0; i++) {
                vmax(tmp + i * TBS, src + i * SS, src + i * SS + nElemPerRepeat, 1, 1, 1, 1, 1, 1, 1);
            }
        }
    }
    pipe_barrier(PIPE_V);

    for (int i = 2; i < srcRepeatPerRow; i++) {
        if ((tmpRepeatStride <= REPEAT_MAX) && (srcRepeatStride <= REPEAT_MAX)) {
            vmax(tmp, src + i * nElemPerRepeat, tmp, OS0, 1, 1, 1, tmpRepeatStride, srcRepeatStride, tmpRepeatStride);
        } else {
            for (int j = 0; j < OS0; j++) {
                vmax(tmp + j * TBS, src + i * nElemPerRepeat + j * SS, tmp + j * TBS, 1, 1, 1, 1, 1, 1, 1);
            }
        }
        pipe_barrier(PIPE_V);
    }
    if (remain > 0) {
        SetContinuousMask(remain);
        if ((tmpRepeatStride <= REPEAT_MAX) && (srcRepeatStride <= REPEAT_MAX)) {
            vmax(tmp, src + srcRepeatPerRow * nElemPerRepeat, tmp, OS0, 1, 1, 1, tmpRepeatStride, srcRepeatStride,
                tmpRepeatStride);
        } else {
            for (int j = 0; j < OS0; j++) {
                vmax(
                    tmp + j * TBS, src + srcRepeatPerRow * nElemPerRepeat + j * SS, tmp + j * TBS, 1, 1, 1, 1, 1, 1, 1);
            }
        }
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
    }
    vcmax(dst, tmp, OS0, (uint16_t)DS, (uint16_t)1ULL, tmpRepeatStride, ONLY_VALUE);
    pipe_barrier(PIPE_V);
}

template <typename T, unsigned DS1, unsigned DS2, unsigned DS3, unsigned SS1, unsigned SS2, unsigned SS3, unsigned TBS3>
TILEOP void DynTrowmaxsingle_(
    __ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *tmp, unsigned OS0, unsigned OS1, unsigned OS2, unsigned OS3) {
    static_assert(SS3 * sizeof(T) % BLOCK_SIZE == 0);
    for (int i = 0; i < OS0; ++i) {
        __ubuf__ T *dst_ = dst;
        __ubuf__ T *src_ = src;
        for (int j = 0; j < OS1; ++j) {
            TileOp::DynTrowmaxsingle_<T, DS3, SS3, TBS3>(dst_, src_, tmp, OS2, OS3);
            dst_ += DS3 * DS2;
            src_ += SS3 * SS2;
            pipe_barrier(PIPE_V);
        }
        dst += DS1 * DS2 * DS3;
        src += SS1 * SS2 * SS3;
    }
}

// dim2
template <typename T, unsigned dstRawShape1, unsigned srcRawShape1, unsigned axis>
TILEOP void DynTexpand_(
    __ubuf__ T *dst, __ubuf__ T *src, unsigned dstShape0, unsigned dstShape1, unsigned srcShape0, unsigned srcShape1) {
    if (axis == 0) {
        // 1 16 -> 16 16
        uint64_t blockLen = (dstShape1 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (int i = 0; i < dstShape0; i++) {
            copy_ubuf_to_ubuf(dst + i * dstRawShape1, src, 0, 1, blockLen, 1, 1);
        }
    } else if (axis == 1) {
        uint64_t shape1Repeat = static_cast<uint64_t>(dstShape1 * sizeof(T) / REPEAT_BYTE);
        if (shape1Repeat < 1) {
            // 16 1 -> 16 16
            SetContinuousMask(dstShape1);
            for (int i = 0; i < dstShape0; i++) {
                set_flag(PIPE_V, PIPE_S, EVENT_ID7);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
                T tmp = (T)(*(src + i * srcRawShape1));
                set_flag(PIPE_S, PIPE_V, EVENT_ID7);
                wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
                vector_dup(dst + i * dstRawShape1, tmp, 1, 1, 0, 0, 0);
            }
            set_vector_mask(-1, -1);
        } else {
            // 16 1 -> 16 64
            constexpr unsigned reptEleNum = REPEAT_BYTE / sizeof(T);
            uint64_t remainNum = static_cast<uint64_t>(dstShape1 % reptEleNum);
            unsigned numLoop = shape1Repeat / REPEAT_MAX;
            unsigned remainAfterLoop = shape1Repeat % REPEAT_MAX;
            for (int i = 0; i < dstShape0; i++) {
                set_flag(PIPE_V, PIPE_S, EVENT_ID7);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
                T tmp = (T)(*(src + i * srcRawShape1));
                set_flag(PIPE_S, PIPE_V, EVENT_ID7);
                wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
                if (numLoop) {
                    for (int j = 0; j < numLoop; j++) {
                        vector_dup(dst + i * dstRawShape1 + j * reptEleNum * REPEAT_MAX, tmp, REPEAT_MAX, 1, 1, 8, 0);
                    }
                }
                if (remainAfterLoop) {
                    vector_dup(
                        dst + i * dstRawShape1 + numLoop * reptEleNum * REPEAT_MAX, tmp, remainAfterLoop, 1, 1, 8, 0);
                }
                if (remainNum) {
                    SetContinuousMask(remainNum);
                    vector_dup(dst + i * dstRawShape1 + shape1Repeat * reptEleNum, tmp, 1, 1, 1, 8, 0);
                    set_vector_mask(-1, -1);
                }
            }
        }
    }
}
// dim3
template <typename T, unsigned dstRawShape1, unsigned dstRawShape2, unsigned srcRawShape1, unsigned srcRawShape2,
    unsigned axis>
TILEOP void DynTexpand_(__ubuf__ T *dst, __ubuf__ T *src, unsigned dstShape0, unsigned dstShape1, unsigned dstShape2,
    unsigned srcShape0, unsigned srcShape1, unsigned srcShape2) {
    if (axis == 1 || axis == 2) {
        // 16 1 16 -> 16 16 16 or 16 16 1 -> 16 16 16
        for (unsigned i = 0; i < dstShape0; i++) {
            TileOp::DynTexpand_<T, dstRawShape2, srcRawShape2, axis - 1>(
                dst, src, dstShape1, dstShape2, srcShape1, srcShape2);
            dst += dstRawShape1 * dstRawShape2;
            src += srcRawShape1 * srcRawShape2;
        }
    } else if (axis == 0) {
        // 1 16 16 -> 16 16 16
        uint64_t blockLen = (dstShape2 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        uint64_t srcGap = (srcRawShape2 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE - blockLen;
        uint64_t dstGap = (dstRawShape2 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE - blockLen;
        for (unsigned i = 0; i < dstShape0; i++) {
            copy_ubuf_to_ubuf(dst + i * dstRawShape1 * dstRawShape2, src, 0, dstShape1, blockLen, srcGap, dstGap);
        }
    }
}
// dim4
template <typename T, unsigned dstRawShape1, unsigned dstRawShape2, unsigned dstRawShape3, unsigned srcRawShape1,
    unsigned srcRawShape2, unsigned srcRawShape3, unsigned axis>
TILEOP void DynTexpand_(__ubuf__ T *dst, __ubuf__ T *src, unsigned dstShape0, unsigned dstShape1, unsigned dstShape2,
    unsigned dstShape3, unsigned srcShape0, unsigned srcShape1, unsigned srcShape2, unsigned srcShape3) {
    if (axis == 0) {
        // 1 16 16 16 -> 16 16 16 16
        uint64_t blockLen = (dstShape3 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        uint64_t srcGap = (srcRawShape3 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE - blockLen;
        uint64_t dstGap = (dstRawShape3 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE - blockLen;
        for (unsigned i = 0; i < dstShape0; ++i) {
            for (unsigned j = 0; j < dstShape1; j++) {
                copy_ubuf_to_ubuf(
                    dst + i * dstRawShape1 * dstRawShape2 * dstRawShape3 + j * dstRawShape2 * dstRawShape3,
                    src + j * srcRawShape2 * srcRawShape3, 0, dstShape2, blockLen, srcGap, dstGap);
            }
        }
    } else if (axis == 1 || axis == 2 || axis == 3) {
        for (unsigned i = 0; i < dstShape0; ++i) {
            TileOp::DynTexpand_<T, dstRawShape2, dstRawShape3, srcRawShape2, srcRawShape3, axis - 1>(
                dst, src, dstShape1, dstShape2, dstShape3, srcShape1, srcShape2, srcShape3);
            dst += dstRawShape1 * dstRawShape2 * dstRawShape3;
            src += srcRawShape1 * srcRawShape2 * srcRawShape3;
        }
    }
}

template <typename Td, typename Ts, unsigned DS, unsigned SS, unsigned Mode>
TILEOP void DynTcast_(__ubuf__ Td *dst, __ubuf__ Ts *src, unsigned T0, unsigned T1) {
    // Now support fp32<->fp16, fp32<->bf16, fp32<->int16, fp32<->int32, int32<->fp16, fp16<->int8, fp32->fp32,
    // bf16->int32
    uint64_t repeatWidth = static_cast<uint64_t>(max(sizeof(Td), sizeof(Ts)));

    unsigned elementsPerRepeat = REPEAT_BYTE / repeatWidth;
    unsigned numRepeatPerLine = T1 / elementsPerRepeat;
    unsigned numRemainPerLine = T1 % elementsPerRepeat;
    unsigned dstRepeatStride =
        repeatWidth == sizeof(Td) ? BLOCK_MAX_PER_REPEAT : (BLOCK_MAX_PER_REPEAT / sizeof(Ts) * sizeof(Td));
    unsigned srcRepeatStride =
        repeatWidth == sizeof(Ts) ? BLOCK_MAX_PER_REPEAT : (BLOCK_MAX_PER_REPEAT / sizeof(Td) * sizeof(Ts));
    constexpr unsigned dstNElemPerBlock = BLOCK_SIZE / sizeof(Td);
    constexpr unsigned srcNElemPerBlock = BLOCK_SIZE / sizeof(Ts);

    if (numRepeatPerLine > 0) {
        unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
        unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
        for (int i = 0; i < T0; i++) {
            if (numLoop) {
                for (int j = 0; j < numLoop; j++) {
                    GenCastCall<Td, Ts, Mode>(dst + i * DS + j * elementsPerRepeat * REPEAT_MAX,
                        src + i * SS + j * elementsPerRepeat * REPEAT_MAX, (uint8_t)REPEAT_MAX, 1, 1,
                        (uint16_t)dstRepeatStride, (uint16_t)srcRepeatStride);
                }
            }
            if (remainAfterLoop) {
                GenCastCall<Td, Ts, Mode>(dst + i * DS + numLoop * elementsPerRepeat * REPEAT_MAX,
                    src + i * SS + numLoop * elementsPerRepeat * REPEAT_MAX, (uint8_t)remainAfterLoop, 1, 1,
                    (uint16_t)dstRepeatStride, (uint16_t)srcRepeatStride);
            }
        }
    }

    // shift to deal with tail
    dst += numRepeatPerLine * elementsPerRepeat;
    src += numRepeatPerLine * elementsPerRepeat;

    if (numRemainPerLine) {
        unsigned numLoop = T0 / REPEAT_MAX;
        unsigned remainAfterLoop = T0 % REPEAT_MAX;
        SetContinuousMask(numRemainPerLine);
        if (numLoop) {
            for (int i = 0; i < numLoop; i++) {
                GenCastCall<Td, Ts, Mode>(dst + i * REPEAT_MAX * DS, src + i * REPEAT_MAX * SS, (uint8_t)REPEAT_MAX, 1,
                    1, (uint16_t)DS / dstNElemPerBlock, (uint16_t)SS / srcNElemPerBlock);
            }
        }
        if (remainAfterLoop) {
            GenCastCall<Td, Ts, Mode>(dst + numLoop * REPEAT_MAX * DS, src + numLoop * REPEAT_MAX * SS,
                (uint8_t)remainAfterLoop, 1, 1, (uint16_t)DS / dstNElemPerBlock, (uint16_t)SS / srcNElemPerBlock);
        }
        set_vector_mask(-1, -1);
    }
}

template <typename Td, typename Ts, unsigned DS0, unsigned DS1, unsigned DS2, unsigned SS0, unsigned SS1, unsigned SS2,
    unsigned Mode>
TILEOP void DynTcast_(__ubuf__ Td *dst, __ubuf__ Ts *src, unsigned T0, unsigned T1, unsigned T2, unsigned T3) {
    static_assert((DS2 * sizeof(Td)) % BLOCK_SIZE == 0);
    static_assert((SS2 * sizeof(Ts)) % BLOCK_SIZE == 0);
    for (int i = 0; i < T0; i++) {
        __ubuf__ Td *dst_ = dst;
        __ubuf__ Ts *src_ = src;
        for (int j = 0; j < T1; j++) {
            DynTcast_<Td, Ts, DS2, SS2, Mode>(dst_, src_, T2, T3);
            dst_ += DS1 * DS2;
            src_ += SS1 * SS2;
        }
        dst += DS0 * DS1 * DS2;
        src += SS0 * SS1 * SS2;
    }
}

template <typename T, unsigned Ds>
TILEOP void DynTduplicate_(__ubuf__ T *dst, T value, unsigned T0, unsigned T1) {
    constexpr unsigned npr = REPEAT_BYTE / sizeof(T);
    unsigned numRepeatPerLine = T1 / npr;
    unsigned numRemainPerLine = T1 % npr;
    constexpr unsigned blockSizeElem = BLOCK_SIZE / sizeof(T);
    if (numRepeatPerLine > 0) {
        for (unsigned i = 0; i < T0; i++) {
            vector_dup(dst + i * Ds, value, numRepeatPerLine, 1, 1, BLOCK_MAX_PER_REPEAT, (int64_t)0);
        }
    }

    // shift to deal with tail
    dst += numRepeatPerLine * npr;
    if (numRemainPerLine) {
        unsigned numLoop = T0 / REPEAT_MAX;
        unsigned remainAfterLoop = T0 % REPEAT_MAX;
        if (numRemainPerLine >= HALF_MASK) {
            set_vector_mask((((static_cast<uint64_t>(1)) << static_cast<uint32_t>(numRemainPerLine - HALF_MASK)) - 1UL),
                0xffffffffffffffffUL);
        } else {
            set_vector_mask(0, (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(numRemainPerLine)) - 1UL));
        }
        if (numLoop) {
            for (unsigned i = 0; i < numLoop; i++) {
                vector_dup(dst + i * REPEAT_MAX * Ds, value, REPEAT_MAX, 1, 1, Ds / blockSizeElem, (int64_t)0);
            }
        }
        if (remainAfterLoop) {
            vector_dup(dst + numLoop * REPEAT_MAX * Ds, value, remainAfterLoop, 1, 1, Ds / blockSizeElem, (int64_t)0);
        }
        set_vector_mask(-1, -1);
    }
}

// dim4
template <typename T, unsigned Ds0, unsigned Ds1, unsigned Ds2>
TILEOP void DynTduplicate_(__ubuf__ T *dst, T value, unsigned T0, unsigned T1, unsigned T2, unsigned T3) {
    static_assert((Ds2 * sizeof(T)) % BLOCK_SIZE == 0);
    for (unsigned i = 0; i < T0; i++) {
        __ubuf__ T *dst_ = dst;
        for (unsigned j = 0; j < T1; j++) {
            DynTduplicate_<T, Ds2>(dst_, value, T2, T3);
            dst_ += Ds1 * Ds2;
        }
        dst += Ds0 * Ds1 * Ds2;
    }
}

// dim2
template <typename T, unsigned srcRawShape1>
TILEOP void DynTrowsumline_(__ubuf__ T *dst, __ubuf__ T *src0, unsigned TShape0, unsigned TShape1) {
    static_assert(sizeof(T) == 4);
    uint32_t rptElm = REPEAT_BYTE / sizeof(T);
    uint32_t repeatTime = (TShape1 + rptElm - 1) / rptElm;
    uint32_t remainElm = TShape1 % rptElm;
    if (!remainElm) {
        vcopy((__ubuf__ uint32_t *)dst, (__ubuf__ uint32_t *)src0, repeatTime, 1, 1, 8, 8);
    } else {
        if (repeatTime == 1) {
            SetContinuousMask(remainElm);
            vcopy((__ubuf__ uint32_t *)dst, (__ubuf__ uint32_t *)src0, 1, 1, 1, 8, 8);
            set_vector_mask(-1, -1);
        } else {
            vcopy((__ubuf__ uint32_t *)dst, (__ubuf__ uint32_t *)src0, repeatTime - 1, 1, 1, 8, 8);
            SetContinuousMask(remainElm);
            vcopy((__ubuf__ uint32_t *)(dst + (repeatTime - 1) * rptElm),
                (__ubuf__ uint32_t *)(src0 + (repeatTime - 1) * rptElm), 1, 1, 1, 8, 8);
            set_vector_mask(-1, -1);
        }
    }
    pipe_barrier(PIPE_V);

    for (uint32_t j = 1; j < TShape0; j++) {
        if (!remainElm) {
            vadd(dst, dst, src0 + j * srcRawShape1, repeatTime, 1, 1, 1, 8, 8, 8);
        } else {
            if (repeatTime == 1) {
                SetContinuousMask(remainElm);
                vadd(dst, dst, src0 + j * srcRawShape1, 1, 1, 1, 1, 8, 8, 8);
                set_vector_mask(-1, -1);
            } else {
                vadd(dst, dst, src0 + j * srcRawShape1, repeatTime - 1, 1, 1, 1, 8, 8, 8);
                SetContinuousMask(remainElm);
                vadd(dst + (repeatTime - 1) * rptElm, dst + (repeatTime - 1) * rptElm,
                    src0 + j * srcRawShape1 + (repeatTime - 1) * rptElm, 1, 1, 1, 1, 8, 8, 8);
                set_vector_mask(-1, -1);
            }
        }
        pipe_barrier(PIPE_V);
    }
}

// dim3
template <typename T, unsigned srcRawShape1, unsigned srcRawShape2, unsigned dstRawShape1, unsigned dstRawShape2,
    unsigned axis>
TILEOP void DynTrowsumline_(__ubuf__ T *dst, __ubuf__ T *src0, unsigned TShape0, unsigned TShape1, unsigned TShape2) {
    static_assert(sizeof(T) == 4);
    if (axis == 0) {
        uint32_t rptElm = REPEAT_BYTE / sizeof(T);
        uint32_t repeatTime = (TShape2 + rptElm - 1) / rptElm;
        uint32_t remainElm = TShape2 % rptElm;
        for (unsigned i = 0; i < TShape1; i++) {
            if (!remainElm) {
                vcopy((__ubuf__ uint32_t *)(dst + i * dstRawShape2), (__ubuf__ uint32_t *)(src0 + i * srcRawShape2),
                    repeatTime, 1, 1, 8, 8);
            } else {
                if (repeatTime == 1) {
                    SetContinuousMask(remainElm);
                    vcopy((__ubuf__ uint32_t *)(dst + i * dstRawShape2), (__ubuf__ uint32_t *)(src0 + i * srcRawShape2),
                        1, 1, 1, 8, 8);
                    set_vector_mask(-1, -1);
                } else {
                    vcopy((__ubuf__ uint32_t *)(dst + i * dstRawShape2), (__ubuf__ uint32_t *)(src0 + i * srcRawShape2),
                        repeatTime - 1, 1, 1, 8, 8);
                    SetContinuousMask(remainElm);
                    vcopy((__ubuf__ uint32_t *)(dst + i * dstRawShape2 + (repeatTime - 1) * rptElm),
                        (__ubuf__ uint32_t *)(src0 + i * srcRawShape2 + (repeatTime - 1) * rptElm), 1, 1, 1, 8, 8);
                    set_vector_mask(-1, -1);
                }
            }
        }
        pipe_barrier(PIPE_V);
        for (unsigned i = 1; i < TShape0; i++) {
            for (unsigned j = 0; j < TShape1; j++) {
                if (!remainElm) {
                    vadd(dst + j * dstRawShape2, dst + j * dstRawShape2,
                        src0 + i * srcRawShape1 * srcRawShape2 + j * srcRawShape2, repeatTime, 1, 1, 1, 8, 8, 8);
                } else {
                    if (repeatTime == 1) {
                        set_vector_mask(0, (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(remainElm)) - 1UL));
                        vadd(dst + j * dstRawShape2, dst + j * dstRawShape2,
                            src0 + i * srcRawShape1 * srcRawShape2 + j * srcRawShape2, repeatTime, 1, 1, 1, 8, 8, 8);
                        set_vector_mask(-1, -1);
                    } else {
                        vadd(dst + j * dstRawShape2, dst + j * dstRawShape2,
                            src0 + i * srcRawShape1 * srcRawShape2 + j * srcRawShape2, repeatTime - 1, 1, 1, 1, 8, 8,
                            8);
                        set_vector_mask(0, (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(remainElm)) - 1UL));
                        vadd(dst + j * dstRawShape2 + (repeatTime - 1) * rptElm,
                            dst + j * dstRawShape2 + (repeatTime - 1) * rptElm,
                            src0 + i * srcRawShape1 * srcRawShape2 + j * srcRawShape2 + (repeatTime - 1) * rptElm, 1, 1,
                            1, 1, 8, 8, 8);
                        set_vector_mask(-1, -1);
                    }
                }
                pipe_barrier(PIPE_V);
            }
        }
    } else if (axis == 1) {
        for (unsigned i = 0; i < TShape0; i++) {
            DynTrowsumline_<T, srcRawShape2>(
                dst + i * dstRawShape1 * dstRawShape2, src0 + i * srcRawShape1 * srcRawShape2, TShape1, TShape2);
        }
    }
}

// dim4
template <typename T, unsigned srcRawShape1, unsigned srcRawShape2, unsigned srcRawShape3, unsigned dstRawShape1,
    unsigned dstRawShape2, unsigned dstRawShape3, unsigned axis>
TILEOP void DynTrowsumline_(
    __ubuf__ T *dst, __ubuf__ T *src0, unsigned TShape0, unsigned TShape1, unsigned TShape2, unsigned TShape3) {
    static_assert(sizeof(T) == 4);
    if (axis == 0) {
        uint32_t rptElm = REPEAT_BYTE / sizeof(T);
        uint32_t repeatTime = (TShape3 + rptElm - 1) / rptElm;
        uint32_t remainElm = TShape3 % rptElm;
        for (unsigned i = 0; i < TShape1; i++) {
            for (unsigned j = 0; j < TShape2; j++) {
                if (!remainElm) {
                    vcopy((__ubuf__ uint32_t *)(dst + i * dstRawShape2 * dstRawShape3 + j * dstRawShape3),
                        (__ubuf__ uint32_t *)(src0 + i * srcRawShape2 * srcRawShape3 + j * srcRawShape3), repeatTime, 1,
                        1, 8, 8);
                } else {
                    if (repeatTime == 1) {
                        SetContinuousMask(remainElm);
                        vcopy((__ubuf__ uint32_t *)(dst + i * dstRawShape2 * dstRawShape3 + j * dstRawShape3),
                            (__ubuf__ uint32_t *)(src0 + i * srcRawShape2 * srcRawShape3 + j * srcRawShape3), 1, 1, 1,
                            8, 8);
                        set_vector_mask(-1, -1);
                    } else {
                        vcopy((__ubuf__ uint32_t *)(dst + i * dstRawShape2 * dstRawShape3 + j * dstRawShape3),
                            (__ubuf__ uint32_t *)(src0 + i * srcRawShape2 * srcRawShape3 + j * srcRawShape3),
                            repeatTime - 1, 1, 1, 8, 8);
                        SetContinuousMask(remainElm);
                        vcopy((__ubuf__ uint32_t *)(dst + i * dstRawShape2 * dstRawShape3 + j * dstRawShape3 +
                                                    (repeatTime - 1) * rptElm),
                            (__ubuf__ uint32_t *)(src0 + i * srcRawShape2 * srcRawShape3 + j * srcRawShape3 +
                                                  (repeatTime - 1) * rptElm),
                            1, 1, 1, 8, 8);
                        set_vector_mask(-1, -1);
                    }
                }
            }
        }
        pipe_barrier(PIPE_V);
        for (unsigned i = 1; i < TShape0; i++) {
            for (unsigned j = 0; j < TShape1; j++) {
                for (unsigned k = 0; k < TShape2; k++) {
                    if (!remainElm) {
                        vadd(dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3,
                            dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3,
                            src0 + i * srcRawShape1 * srcRawShape2 * srcRawShape3 + j * srcRawShape2 * srcRawShape3 +
                                k * srcRawShape3,
                            repeatTime, 1, 1, 1, 8, 8, 8);
                    } else {
                        if (repeatTime == 1) {
                            set_vector_mask(
                                0, (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(remainElm)) - 1UL));
                            vadd(dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3,
                                dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3,
                                src0 + i * srcRawShape1 * srcRawShape2 * srcRawShape3 +
                                    j * srcRawShape2 * srcRawShape3 + k * srcRawShape3,
                                repeatTime, 1, 1, 1, 8, 8, 8);
                            set_vector_mask(-1, -1);
                        } else {
                            vadd(dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3,
                                dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3,
                                src0 + i * srcRawShape1 * srcRawShape2 * srcRawShape3 +
                                    j * srcRawShape2 * srcRawShape3 + k * srcRawShape3,
                                repeatTime - 1, 1, 1, 1, 8, 8, 8);
                            set_vector_mask(
                                0, (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(remainElm)) - 1UL));
                            vadd(dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3 + (repeatTime - 1) * rptElm,
                                dst + j * dstRawShape2 * dstRawShape3 + k * dstRawShape3 + (repeatTime - 1) * rptElm,
                                src0 + i * srcRawShape1 * srcRawShape2 * srcRawShape3 +
                                    j * srcRawShape2 * srcRawShape3 + k * srcRawShape3 + (repeatTime - 1) * rptElm,
                                1, 1, 1, 1, 8, 8, 8);
                            set_vector_mask(-1, -1);
                        }
                    }
                }
            }
        }
    } else if (axis == 1 || axis == 2) {
        for (unsigned i = 0; i < TShape0; i++) {
            DynTrowsumline_<T, srcRawShape2, srcRawShape3, dstRawShape2, dstRawShape3, axis - 1>(
                dst + i * dstRawShape1 * dstRawShape2 * dstRawShape3,
                src0 + i * srcRawShape1 * srcRawShape2 * srcRawShape3, TShape1, TShape2, TShape3);
        }
    }
}

template <typename T, unsigned srcRawShape1, unsigned srcRawShape2, unsigned axis0, unsigned axis1>
TILEOP void DynTtransposeDataMoveBase_(__gm__ T *dst, __ubuf__ T *src, unsigned TShape0, unsigned TShape1,
    unsigned TShape2, unsigned dstShape1, unsigned dstShape2) {
    if constexpr (axis0 == 0 && axis1 == 1) {
        __gm__ T *dst_ = dst;
        __ubuf__ T *src_ = src;
        unsigned nBurst = TShape1;
        unsigned lenBurst = TShape2 * sizeof(T);
        unsigned srcStride = 0;
        unsigned dstStride = (dstShape1 - 1) * dstShape2 * sizeof(T);
        for (int b = 0; b < TShape0; b++) {
            copy_ubuf_to_gm_align_b32(dst_, src_, 0, nBurst, lenBurst, 0, 0, srcStride, dstStride);
            dst_ += dstShape2;
            src_ += srcRawShape1 * srcRawShape2;
        }
    } else {
        static_assert(sizeof(T) == 0, "Unsupport transpose axis");
    }
}

template <typename T, unsigned srcRawShape1, unsigned srcRawShape2, unsigned srcRawShape3, unsigned axis0,
    unsigned axis1>
TILEOP void DynTtransposeDataMove_(__gm__ T *dst, __ubuf__ T *src, unsigned TShape0, unsigned TShape1,
    unsigned TShape2, unsigned TShape3, unsigned dstShape0, unsigned dstShape1, unsigned dstShape2, unsigned dstShape3,
    unsigned GmOffset0, unsigned GmOffset1, unsigned GmOffset2, unsigned GmOffset3) {
    if constexpr (axis0 == 1 && axis1 == 2) {
        __gm__ T *dst_ =
            dst + CalcLinearOffset(dstShape1, dstShape2, dstShape3, GmOffset0, GmOffset1, GmOffset2, GmOffset3);
        __ubuf__ T *src_ = src;
        for (int b = 0; b < TShape0; b++) {
            DynTtransposeDataMoveBase_<T, srcRawShape2, srcRawShape3, axis0 - 1, axis1 - 1>(
                dst_, src_, TShape1, TShape2, TShape3, dstShape2, dstShape3);
            dst_ += dstShape1 * dstShape2 * dstShape3;
            src_ += srcRawShape1 * srcRawShape2 * srcRawShape3;
        }
    } else {
        static_assert(sizeof(T) == 0, "Unsupport transpose axis");
    }
}

template <typename T, typename T2, unsigned src0Shape1, unsigned dstShape1, unsigned axis>
TILEOP void DynTgatherElement(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T2 *src1, unsigned TShape0, unsigned TShape1) {
    constexpr uint16_t lenBurst = 1;
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    for (int i = 0; i < TShape0; ++i) {
        for (int j = 0; j < TShape1; ++j) {
            T2 index = (T2)(*(src1 + i * TShape1 + j)); // src1[i,j]
            int srcOffset = 0;
            if constexpr (axis == 0) {
                srcOffset = index * src0Shape1 + j;
            } else {
                srcOffset = i * src0Shape1 + index;
            }
            dst[i * dstShape1 + j] = src0[srcOffset];
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

template <typename T, unsigned DS, unsigned SS>
TILEOP void DynTtranspose_vnchwconv_(
    __ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *tmp, unsigned T0, unsigned T1, unsigned TS) {
    if (((DS % 16) != 0) || ((SS % 8) != 0) || ((TS % 16) != 0)) {
        set_flag(PIPE_V, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
        for (int i = 0; i < T0; i++) {
            for (int j = 0; j < T1; j++) {
                dst[j * DS + i] = src[i * SS + j];
            }
        }
        set_flag(PIPE_S, PIPE_V, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
        return;
    }
    static_assert(sizeof(T) == 4);
    // 16 x 32B subtile
    constexpr int block_elem = BLOCK_SIZE / sizeof(T);
    // go by subtile column, a.k.a. iter in row direction
    int num_subtile_x = (T1 + block_elem - 1) / block_elem;
    int num_subtile_y = T0 / 16;
    if (num_subtile_y) {
        for (int i = 0; i < num_subtile_x; i++) {
            uint64_t srcUb[16] = {0}, tmpUb[16] = {0};
            for (int j = 0; j < 16; j++) {
                srcUb[j] = (uint64_t)(src + i * 8 + j * SS);
                tmpUb[j] = (uint64_t)(tmp + ((j >> 1) + i * 8) * TS + (j & 1) * block_elem);
            }
            set_va_reg_sb(VA2, srcUb);
            set_va_reg_sb(VA3, &srcUb[8]);
            set_va_reg_sb(VA0, tmpUb);
            set_va_reg_sb(VA1, &tmpUb[8]);
            if (num_subtile_y == 1) {
                scatter_vnchwconv_b32(VA0, VA2, 1, 0, 0);
            } else {
                scatter_vnchwconv_b32(VA0, VA2, num_subtile_y, 2, 16 * SS * sizeof(T) / BLOCK_SIZE);
            }
        }
    }
    // tail
    int remain_y = T0 % 16;
    if (remain_y) {
        uint64_t srcUb[16] = {0}, tmpUb[16] = {0};
        for (int i = 0; i < remain_y; i++) {
            srcUb[i] = (uint64_t)(src + (num_subtile_y * 16 + i) * SS);
        }
        for (int i = 0; i <16; i++) {
            tmpUb[i] = (uint64_t)(tmp + num_subtile_y * 16 + (i & 1) * block_elem + (i >> 1) * TS);
        }
        set_va_reg_sb(VA2, srcUb);
        set_va_reg_sb(VA3, &srcUb[8]);
        set_va_reg_sb(VA0, tmpUb);
        set_va_reg_sb(VA1, &tmpUb[8]);
        if (num_subtile_x == 1) {
            scatter_vnchwconv_b32(VA0, VA2, 1, 0, 0);
        } else {
            scatter_vnchwconv_b32(VA0, VA2, num_subtile_x, 8 * TS * sizeof(T) / BLOCK_SIZE, 1);
        }
    }
    // copy to dst
    pipe_barrier(PIPE_V);
    uint16_t lenBurst = (T0 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint16_t srcGap = TS * sizeof(T) / BLOCK_SIZE - lenBurst;
    uint16_t dstGap = DS * sizeof(T) / BLOCK_SIZE - lenBurst;
    copy_ubuf_to_ubuf(dst, tmp, 0, T1, lenBurst, srcGap, dstGap);
}

template <typename T, unsigned DS1, unsigned DS2, unsigned DS3, unsigned DS4, unsigned SS1, unsigned SS2, unsigned SS3,
    unsigned SS4>
TILEOP void DynTtranspose_vnchwconv_(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *tmp, unsigned T0, unsigned T1,
    unsigned T2, unsigned T3, unsigned T4) {
    unsigned TS1 = DS1;
    unsigned TS2 = DS2;
    unsigned TS3 = (T4 + 7) / 8 * 8;
    unsigned TS4 = (T3 + 15) / 16 * 16;
    for (unsigned i = 0; i < T0; i++) {
        __ubuf__ T *dst0 = dst;
        __ubuf__ T *src0 = src;
        __ubuf__ T *tmp0 = tmp;
        for (unsigned j = 0; j < T1; j++) {
            __ubuf__ T *dst1 = dst0;
            __ubuf__ T *src1 = src0;
            __ubuf__ T *tmp1 = tmp0;
            for (unsigned k = 0; k < T2; k++) {
                DynTtranspose_vnchwconv_<T, DS4, SS4>(dst1, src1, tmp1, T3, T4, TS4);
                dst1 += DS3 * DS4;
                src1 += SS3 * SS4;
                tmp1 += TS3 * TS4;
            }
            dst0 += DS2 * DS3 * DS4;
            src0 += SS2 * SS3 * SS4;
            tmp0 += TS2 * TS3 * TS4;
        }
        dst += DS1 * DS2 * DS3 * DS4;
        src += SS1 * SS2 * SS3 * SS4;
        tmp += TS1 * TS2 * TS3 * TS4;
    }
}

// [case1] params: [src0Shape0,src0Shape1], indices: [TShape0], axis: 0, output: [TShape0,TShape1]
// [case2] params: [src0Shape0,src0Shape1], indices: [TShape0,TShape1], axis: 0, output: [TShape0,TShape1,TShape2]
template <typename T, typename T2, unsigned src0Shape2, unsigned dst0Shape2>
TILEOP void DynTgather_(
    __ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T2 *src1, unsigned TShape0, unsigned TShape1, unsigned TShape2) {
    const uint16_t lenBurst = (TShape2 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int i = 0; i < TShape0; ++i) {
        for (int j = 0; j < TShape1; ++j) {
            set_flag(PIPE_V, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
            T2 index = (T2)(*(src1 + j)); // src1[i,j]
            set_flag(PIPE_S, PIPE_V, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID7);

            // dst, src, sid, nBurst, lenBurst, srcStride, dstStride
            copy_ubuf_to_ubuf(dst + j * dst0Shape2, src0 + index * src0Shape2, 0, 1, lenBurst, 1, 1);
        }

        dst += TShape1 * dst0Shape2;
        src1 += TShape1;
    }
}

template <typename T, unsigned dstShape0, unsigned dstShape1,
 unsigned srcShape0, unsigned srcShape1, unsigned reverseOperand>
TILEOP void DynTSadds(__ubuf__ T *dst, __ubuf__ T *src, float scalar,unsigned TShape0, unsigned TShape1) {
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    for (int i = 0; i < TShape0; ++i) {
        for (int j = 0; j < TShape1; ++j) {
            T value = (T)(*(src + i * srcShape1 + j));
            int dstOffset = i * dstShape1 + j;
            dst[dstOffset] = scalar + value;
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

template <typename T, unsigned dstShape0, unsigned dstShape1,
 unsigned srcShape0, unsigned srcShape1, unsigned reverseOperand>
TILEOP void DynTSsubs(__ubuf__ T *dst, __ubuf__ T *src, float scalar,unsigned TShape0, unsigned TShape1) {
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    for (int i = 0; i < TShape0; ++i) {
        for (int j = 0; j < TShape1; ++j) {
            T value = (T)(*(src + i * srcShape1 + j));
            int dstOffset = i * dstShape1 + j;
            dst[dstOffset] = reverseOperand == 1 ? scalar - value : value - scalar;
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

template <typename T, unsigned dstShape0, unsigned dstShape1,
 unsigned srcShape0, unsigned srcShape1, unsigned reverseOperand>
TILEOP void DynTSmuls(__ubuf__ T *dst, __ubuf__ T *src, float scalar, unsigned TShape0, unsigned TShape1) {
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    for (int i = 0; i < TShape0; ++i) {
        for (int j = 0; j < TShape1; ++j) {
            T value = (T)(*(src + i * srcShape1 + j));
            int dstOffset = i * dstShape1 + j;
            dst[dstOffset] = value * scalar;
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

template <typename T, unsigned dstShape0, unsigned dstShape1, unsigned srcShape0, unsigned srcShape1,
    unsigned reverseOperand>
TILEOP void DynTSdivs(__ubuf__ T *dst, __ubuf__ T *src, float scalar, unsigned TShape0, unsigned TShape1) {
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    for (int i = 0; i < TShape0; ++i) {
        for (int j = 0; j < TShape1; ++j) {
            int dstOffset = i * dstShape1 + j;
            T value = (T)(*(src + dstOffset));
            dst[dstOffset] = reverseOperand == 1 ? scalar / value : value / scalar;
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

template <typename T, unsigned dstShape0, unsigned dstShape1, unsigned dstShape2, unsigned srcShape0,
    unsigned srcShape1, unsigned srcShape2, unsigned reverseOperand>
TILEOP void DynTSdivs(
    __ubuf__ T *dst, __ubuf__ T *src, float scalar, unsigned TShape0, unsigned TShape1, unsigned TShape2) {
    int dstOffset = dstShape1 * dstShape2;
    for (int i = 0; i < TShape0; i++) {
        TileOp::DynTSdivs<T, dstShape0, dstShape1, srcShape0, srcShape1, reverseOperand>(
            dst + i * dstOffset, src + i * dstOffset, scalar, TShape1, TShape2);
    }
}

template <typename T, unsigned dstShape0, unsigned dstShape1,
 unsigned srcShape0, unsigned srcShape1, unsigned reverseOperand>
TILEOP void DynTSmaxs(__ubuf__ T *dst, __ubuf__ T *src, float scalar, unsigned TShape0, unsigned TShape1) {
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    for (int i = 0; i < TShape0; ++i) {
        for (int j = 0; j < TShape1; ++j) {
            T value = (T)(*(src + i * srcShape1 + j));
            int dstOffset = i * dstShape1 + j;
            dst[dstOffset] = value > scalar ? value : scalar;
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

template <typename T, unsigned dstShape0, unsigned dstShape1, unsigned srcShape0, unsigned srcShape1, int axis, int isLargest>
TILEOP void DynBitSort(__ubuf__ T *dst, __ubuf__ T *src, unsigned oriShape0, unsigned oriShape1) {
    // 生成index数据,首先创建一个1~8的数组,之后扩展到TShape1,构成0~TShape1的index数组
    // pipe_barrier(PIPE_ALL); // 当前OP无法描述两条流水,UB复用场景存在问题,暂时按照pipe_all规避
    int32_t srcShape1Align = (oriShape1 + 31) / 32 * 32;
    __ubuf__ uint32_t *idx = (__ubuf__ uint32_t *)dst + 2 * srcShape1Align;
    for (int32_t j = 0; j < oriShape1; j++) {
        *(idx + j) = j;
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);

    // 对于不满足32元素对齐场景,首先将src拷贝到dst的3*srcShape1位置
    if (oriShape1 < 32) {
        uint64_t srcShape1_Align_Block_Num = (oriShape1 * sizeof(float) + 31) / 32;
        uint64_t dstShape1_Block_Num = dstShape1 * sizeof(float) / 32;
        copy_ubuf_to_ubuf((__ubuf__ float *)dst + 3 * srcShape1Align, (__ubuf__ void *)src, 0, oriShape0,
            srcShape1_Align_Block_Num, 0, dstShape1_Block_Num - srcShape1_Align_Block_Num);
        pipe_barrier(PIPE_V);
        if constexpr (isLargest == 0) {
            set_mask_count();
            set_vector_mask(0, oriShape1);
            // 按照升序排列时,需要首先将数据乘以-1,同时不可以污染src
            vmuls((__ubuf__ float *)dst + 3 * srcShape1Align, (__ubuf__ float *)dst + 3 * srcShape1Align, -1.0f, 1, 1,
                1, 8, 8);
            pipe_barrier(PIPE_V);
            set_mask_norm();
            set_vector_mask(-1, -1);
        }
        // 需要将尾块部分置为-inf，之后再排序
        // 计算duplicate的mask
        uint64_t mask = ~(((static_cast<uint64_t>(1)) << oriShape1) - 1);
        mask = mask & 0xFFFFFFFF;
        float FLOAT_MIN = -1.0e38f;
        set_mask_norm();
        set_vector_mask(0, mask);
        vector_dup(dst + 3 * srcShape1Align, FLOAT_MIN, oriShape0, 1, 1, dstShape1 * sizeof(float) / 32, (int64_t)0);
        pipe_barrier(PIPE_V);
        for (int rowIdx = 0; rowIdx < oriShape0; rowIdx++) {
            vbitsort((__ubuf__ float *)dst + rowIdx * dstShape1,
                (__ubuf__ float *)dst + rowIdx * dstShape1 + 3 * srcShape1Align, (__ubuf__ uint32_t *)idx, 1);
        }
        pipe_barrier(PIPE_V);
        set_vector_mask(-1, -1);
    }

    if (oriShape1 == 32) {
        for (int rowIdx = 0; rowIdx < oriShape0; rowIdx++) {
            // 32个数时，一次完成排序
            __ubuf__ float *srcData = reinterpret_cast<__ubuf__ float *>(src) + rowIdx * srcShape1;
            __ubuf__ float *dstData = reinterpret_cast<__ubuf__ float *>(dst) + rowIdx * dstShape1;
            if constexpr (isLargest == 0) {
                set_mask_count();
                set_vector_mask(0, oriShape1);
                // 按照升序排列时,需要首先将数据乘以-1,同时不可以污染src
                srcData = reinterpret_cast<__ubuf__ float *>(dst) + rowIdx * dstShape1 + 3 * srcShape1Align;
                vmuls(srcData, reinterpret_cast<__ubuf__ float *>(src) + rowIdx * srcShape1, -1.0f, 1, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                set_mask_norm();
                set_vector_mask(-1, -1);
            }
            vbitsort(dstData, srcData, idx, 1);
            pipe_barrier(PIPE_V);
        }
    }

    if (oriShape1 > 32) {
        int32_t repeat_sort32 = oriShape1 / 32;
        int32_t tail_sort32 = oriShape1 % 32;
        for (int rowIdx = 0; rowIdx < oriShape0; rowIdx++) {
            __ubuf__ float *srcData = reinterpret_cast<__ubuf__ float *>(src) + rowIdx * srcShape1;
            __ubuf__ float *dstData = reinterpret_cast<__ubuf__ float *>(dst) + rowIdx * dstShape1;
            if constexpr (isLargest == 0) {
                set_mask_count();
                set_vector_mask(0, oriShape1);
                // 按照升序排列时,需要首先将数据乘以-1,同时不可以污染src
                srcData = reinterpret_cast<__ubuf__ float *>(dst) + rowIdx * dstShape1 + 3 * srcShape1Align;
                vmuls(srcData, reinterpret_cast<__ubuf__ float *>(src) + rowIdx * srcShape1, -1.0f, 1, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                set_mask_norm();
                set_vector_mask(-1, -1);
            }
            // 首先逐32个数进行排序,需要补齐不对齐的部分
            if (tail_sort32 > 0) {
                // 非整块的时候,首先对尾部补充-inf
                float FLOAT_MIN = -1.0e38f;
                uint64_t mask = ~(((static_cast<uint64_t>(1)) << ( tail_sort32)) - 1);
                set_mask_norm();
                set_vector_mask(0, mask);
                vector_dup(srcData + repeat_sort32 * 32, FLOAT_MIN, 1, 1, 1, 8, (int64_t)0);
                pipe_barrier(PIPE_V);
                vbitsort(dstData, srcData, idx, repeat_sort32 + 1);
                pipe_barrier(PIPE_V);
                set_vector_mask(-1, -1);
            } else {
                // 整块时,直接进行逐32元素排序
                vbitsort(dstData, srcData, idx, repeat_sort32);
                pipe_barrier(PIPE_V);
            }
            pipe_barrier(PIPE_V);
        }
    }
}

template <typename T, unsigned dstShape0, unsigned dstShape1, unsigned srcShape0, unsigned srcShape1, int axis, int k, int isLargest>
TILEOP void DynMrgSort(__ubuf__ T *dst, __ubuf__ T *src, unsigned oriShape0, unsigned oriShape1) {
    constexpr int32_t kAlign = (k + 3) / 4 * 4; // k需要向32Bytes取整,否则最后搬运出问题
    int32_t totalNum = oriShape1 / 4;
    for (int rowIdx = 0; rowIdx < dstShape0; rowIdx++) {
        // 每4个合并,计算整块
        int32_t z = 32;
        for (; z * 4 <= totalNum; z *= 4) {
            __ubuf__ float *srcData = reinterpret_cast<__ubuf__ float *>(src) + rowIdx * srcShape1;
            __ubuf__ float *dstData = reinterpret_cast<__ubuf__ float *>(src) + rowIdx * srcShape1 + totalNum * 2;
            uint64_t config = 0;
            uint32_t repeat_mrg = totalNum / (z * 4);
            config |= uint64_t(totalNum / (z * 4)); // Xt[7:0]: repeat time
            config |= (uint64_t(0b1111) << 8);      // Xt[11:8]: 4-bit mask signal
            config |= (uint64_t(0b0) << 12);        // Xt[12]: 1-enable input list exhausted suspension

            // 每次计算的数据
            uint64_t src1 = 0;
            src1 |= (uint64_t(z));
            src1 |= (uint64_t(z) << 16);
            src1 |= (uint64_t(z) << 32);
            src1 |= (uint64_t(z) << 48);

            __ubuf__ float *addr_array[4] = {(__ubuf__ float *)(srcData + 0 * z * 2),
                (__ubuf__ float *)(srcData + 1 * z * 2), (__ubuf__ float *)(srcData + 2 * z * 2),
                (__ubuf__ float *)(srcData + 3 * z * 2)};
            pipe_barrier(PIPE_V);
            vmrgsort4(dstData, addr_array, src1, config);
            pipe_barrier(PIPE_V);
            copy_ubuf_to_ubuf(
                (__ubuf__ void *)srcData, (__ubuf__ void *)dstData, 0, z * 4 * repeat_mrg * 2 / 8, 1, 0, 0);
            pipe_barrier(PIPE_V);
        }
        // 合并尾块
        if (z < totalNum) {
            int32_t arrayCount = 0;
            int32_t mrgArray[15] = {0};
            int32_t tmpInner = totalNum;
            for (int32_t i = z; i >= 32; i /= 4) {
                int32_t count;
                for (count = 0; count < tmpInner / i; count++) {
                    mrgArray[arrayCount++] = i;
                }
                tmpInner -= count * i;
            }
            uint16_t mrgSortedLen = 0;
            for (int32_t i = 0; i < arrayCount - 1; ++i) {
                __ubuf__ float *srcData = reinterpret_cast<__ubuf__ float *>(src) + rowIdx * srcShape1;
                __ubuf__ float *dstData = reinterpret_cast<__ubuf__ float *>(src) + rowIdx * srcShape1 + totalNum * 2;
                mrgSortedLen += static_cast<uint16_t>(mrgArray[i]);
                uint64_t tmpMrgSortedLen = mrgSortedLen;
                uint64_t tmpMrgArray = mrgArray[i + 1];
                if (mrgSortedLen > k) {
                    tmpMrgSortedLen = k;
                }
                if (mrgArray[i + 1] > k) {
                    tmpMrgArray = k;
                }
                uint64_t config = 0;
                config |= uint64_t(1);           // Xt[7:0]: repeat time
                config |= (uint64_t(0b11) << 8); // Xt[11:8]: 4-bit mask signal
                config |= (uint64_t(0b0) << 12); // Xt[12]: 1-enable input list exhausted suspension

                // 每次计算的数据
                uint64_t src1 = 0;
                src1 |= (uint64_t(tmpMrgSortedLen));
                src1 |= (uint64_t(tmpMrgArray) << 16);
                __ubuf__ float *addr_array[4] = {(__ubuf__ float *)(srcData),
                    (__ubuf__ float *)(srcData + mrgSortedLen * 2), (__ubuf__ float *)0, (__ubuf__ float *)0};
                pipe_barrier(PIPE_V);
                vmrgsort4(dstData, addr_array, src1, config);
                pipe_barrier(PIPE_V);
                copy_ubuf_to_ubuf((__ubuf__ void *)srcData, (__ubuf__ void *)dstData, 0,
                    (tmpMrgSortedLen + tmpMrgArray) * 2 / 8, 1, 0, 0);
                pipe_barrier(PIPE_V);
            }
        }
        copy_ubuf_to_ubuf((__ubuf__ float *)dst + rowIdx * dstShape1, (__ubuf__ float *)src + rowIdx * srcShape1, 0,
            kAlign / 4, 1, 0, 0);
        pipe_barrier(PIPE_V);
    }
}

template <typename T, typename U, int k, int extractMode, int isLargest>
TILEOP void DynExtract(__ubuf__ T *dst, __ubuf__ U *src, unsigned TShape0, unsigned TShape1) {
    constexpr uint64_t repeat = static_cast<uint64_t>(TShape0 * TShape1 * 2 * sizeof(T) / REPEAT_BYTE);
    constexpr uint8_t dstBlockStride = 1;
    constexpr uint8_t srcBlockStride = 1;
    constexpr uint8_t dstRepeatStride = 8;
    constexpr uint8_t srcRepeatStride = 8;
    // mode trans, extractMode == 0 取奇数位， extractMode == 1 取偶数位
    int patternMode = 1;
    if constexpr (extractMode == 1) {
        patternMode = 2;
    }
    __ubuf__ U *nullsrc1 = REPEAT_BYTE * sizeof(U) + src;
    if constexpr (repeat < 1) {
        uint64_t elems = TShape0 * TShape1;
        set_mask_count();
        set_vector_mask(0, elems * 2);
        vreducev2((__ubuf__ uint32_t *)dst, (__ubuf__ uint32_t *)src, (__ubuf__ uint32_t *)nullsrc1, 1, srcBlockStride,
            patternMode, srcRepeatStride, 0);
        set_mask_norm();
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
    } else {
        uint8_t repeatMod = static_cast<uint8_t>(repeat % REPEAT_MAX);
        if (repeatMod != 0) {
            uint64_t elems = TShape0 * TShape1;
            set_mask_norm();
            set_vector_mask(-1, -1);
            vreducev2((__ubuf__ uint32_t *)(dst), (__ubuf__ uint32_t *)(src), (__ubuf__ uint32_t *)nullsrc1, repeatMod,
                srcBlockStride, patternMode, srcRepeatStride, 0);
            pipe_barrier(PIPE_V);
        }
    }

    if constexpr (extractMode == 0 && isLargest == 0) {
        // 按照升序排序时,对于value需要乘以-1,恢复原始值
        set_mask_count();
        set_vector_mask(0, TShape0 * TShape1);
        vmuls((__ubuf__ float *)dst, (__ubuf__ float *)dst, -1.0f, 1, 1, 1, 8, 8);
        set_mask_norm();
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
    }
}

} // namespace TileOp

#endif // TILE_FWK_VECTOR_DYN_H
