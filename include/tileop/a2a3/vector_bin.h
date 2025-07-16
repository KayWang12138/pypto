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
 * \file vector_bin.h
 * \brief
 */

// dim2 & dim1 (T0 = 1 for dim1)
template <typename T, unsigned T0, unsigned T1, unsigned DS, unsigned SS0, unsigned SS1>
TILEOP void T_BIN(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1) {
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned numRepeatPerLine = T1 / elementsPerRepeat;
    constexpr unsigned numRemainPerLine = T1 % elementsPerRepeat;
    constexpr unsigned blockSizeElem = BLOCK_SIZE / sizeof(T);

    if constexpr (numRepeatPerLine > 0) {
        constexpr unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
        constexpr unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
        for (int i = 0; i < T0; i++) {
            if constexpr (numLoop) {
                for (int j = 0; j < numLoop; j++) {
                    V_BIN_FUNC(dst + i * DS + j * elementsPerRepeat * REPEAT_MAX,
                        src0 + i * SS0 + j * elementsPerRepeat * REPEAT_MAX,
                        src1 + i * SS1 + j * elementsPerRepeat * REPEAT_MAX, REPEAT_MAX, 1, 1, 1, 8, 8, 8);
                }
            }
            if constexpr (remainAfterLoop) {
                V_BIN_FUNC(dst + i * DS + numLoop * elementsPerRepeat * REPEAT_MAX,
                    src0 + i * SS0 + numLoop * elementsPerRepeat * REPEAT_MAX,
                    src1 + i * SS1 + numLoop * elementsPerRepeat * REPEAT_MAX, remainAfterLoop, 1, 1, 1, 8, 8, 8);
            }
        }
    }

    // shift to deal with tail
    dst += numRepeatPerLine * elementsPerRepeat;
    src0 += numRepeatPerLine * elementsPerRepeat;
    src1 += numRepeatPerLine * elementsPerRepeat;

    if constexpr (numRemainPerLine) {
        constexpr unsigned numLoop = T0 / REPEAT_MAX;
        constexpr unsigned remainAfterLoop = T0 % REPEAT_MAX;
        bool strideOverFlag = (DS / blockSizeElem > REPEAT_STRIDE_MAX) || (SS0 / blockSizeElem > REPEAT_STRIDE_MAX) ||
                              (SS1 / blockSizeElem > REPEAT_STRIDE_MAX);
        SetContinuousMask(numRemainPerLine);
        if constexpr (numLoop) {
            for (int i = 0; i < numLoop; i++) {
                if (strideOverFlag) {
                    for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                        V_BIN_FUNC(dst + i * REPEAT_MAX * DS + j * DS, src0 + i * REPEAT_MAX * SS0 + j * SS0,
                            src1 + i * REPEAT_MAX * SS1 + j * SS1, 1, 1, 1, 1, 1, 1, 1);
                    }
                } else {
                    V_BIN_FUNC(dst + i * REPEAT_MAX * DS, src0 + i * REPEAT_MAX * SS0, src1 + i * REPEAT_MAX * SS1,
                        REPEAT_MAX, 1, 1, 1, DS / blockSizeElem, SS0 / blockSizeElem, SS1 / blockSizeElem);
                }
            }
        }
        if constexpr (remainAfterLoop) {
            if (strideOverFlag) {
                for (unsigned j = 0; j < remainAfterLoop; j++) {
                    V_BIN_FUNC(dst + numLoop * REPEAT_MAX * DS + j * DS, src0 + numLoop * REPEAT_MAX * SS0 + j * SS0,
                        src1 + numLoop * REPEAT_MAX * SS1 + j * SS1, 1, 1, 1, 1, 1, 1, 1);
                }
            } else {
                V_BIN_FUNC(dst + numLoop * REPEAT_MAX * DS, src0 + numLoop * REPEAT_MAX * SS0,
                    src1 + numLoop * REPEAT_MAX * SS1, remainAfterLoop, 1, 1, 1, DS / blockSizeElem,
                    SS0 / blockSizeElem, SS1 / blockSizeElem);
            }
        }
        set_vector_mask(-1, -1);
    }
}

// dim3
template <typename T, unsigned T0, unsigned T1, unsigned T2,
         unsigned DS0, unsigned DS1,
         unsigned S0S0, unsigned S0S1,
         unsigned S1S0, unsigned S1S1>
TILEOP void T_BIN(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1) {
    static_assert((DS1 * sizeof(T)) % BLOCK_SIZE == 0);
    static_assert((S0S1 * sizeof(T)) % BLOCK_SIZE == 0);
    static_assert((S1S1 * sizeof(T)) % BLOCK_SIZE == 0);
    for (int i = 0; i < T0; i++) {
        T_BIN<T, T1, T2, DS1, S0S1, S1S1>(dst, src0, src1);
        dst += DS0 * DS1;
        src0 += S0S0 * S0S1;
        src1 += S1S0 * S1S1;
    }
}

// dim4
template <typename T, unsigned T0, unsigned T1, unsigned T2, unsigned T3,
         unsigned DS0, unsigned DS1, unsigned DS2,
         unsigned S0S0, unsigned S0S1, unsigned S0S2,
         unsigned S1S0, unsigned S1S1, unsigned S1S2>
TILEOP void T_BIN(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1) {
    static_assert((DS2 * sizeof(T)) % BLOCK_SIZE == 0);
    static_assert((S0S2 * sizeof(T)) % BLOCK_SIZE == 0);
    static_assert((S1S2 * sizeof(T)) % BLOCK_SIZE == 0);
    for (int i = 0; i < T0; i++) {
        __ubuf__ T *dst_ = dst;
        __ubuf__ T *src0_ = src0;
        __ubuf__ T *src1_ = src1;
        for (int j = 0; j < T1; j++) {
            T_BIN<T, T2, T3, DS2, S0S2, S1S2>(dst_, src0_, src1_);
            dst_ += DS1 * DS2;
            src0_ += S0S1 * S0S2;
            src1_ += S1S1 * S1S2;
        }
        dst += DS0 * DS1 * DS2;
        src0 += S0S0 * S0S1 * S0S2;
        src1 += S1S0 * S1S1 * S1S2;
    }
}

template <typename T, unsigned T0, unsigned T1, unsigned DS, unsigned SS0>
TILEOP void T_BIN_VS(__ubuf__ T *dst, __ubuf__ T *src0, T src1) {
#ifdef VS_SUB
    src1 = src1 * (-1);
#endif
#ifdef VS_DIV
    if (src1 != 0) {
        src1 = (float)1.0 / src1;
    }
#endif
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned numRepeatPerLine = T1 / elementsPerRepeat;
    constexpr unsigned numRemainPerLine = T1 % elementsPerRepeat;
    constexpr unsigned blockSizeElem = BLOCK_SIZE / sizeof(T);

    if constexpr (numRepeatPerLine > 0) {
        constexpr unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
        constexpr unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
        for (int i = 0; i < T0; i++) {
            if constexpr (numLoop) {
                for (int j = 0; j < numLoop; j++) {
                    V_BIN_FUNC_VS(dst + i * DS + j * elementsPerRepeat * REPEAT_MAX,
                        src0 + i * SS0 + j * elementsPerRepeat * REPEAT_MAX, src1, REPEAT_MAX, 1, 1, 8, 8);
                }
            }
            if constexpr (remainAfterLoop) {
                V_BIN_FUNC_VS(dst + i * DS + elementsPerRepeat * REPEAT_MAX * numLoop,
                    src0 + i * SS0 + elementsPerRepeat * REPEAT_MAX * numLoop, src1, remainAfterLoop, 1, 1, 8, 8);
            }
        }
    }

    // shift to deal with tail
    dst += numRepeatPerLine * elementsPerRepeat;
    src0 += numRepeatPerLine * elementsPerRepeat;

    if constexpr (numRemainPerLine) {
        constexpr unsigned numLoop = T0 / REPEAT_MAX;
        constexpr unsigned remainAfterLoop = T0 % REPEAT_MAX;
        bool strideOverFlag = (DS / blockSizeElem > REPEAT_STRIDE_MAX) || (SS0 / blockSizeElem > REPEAT_STRIDE_MAX);
        SetContinuousMask(numRemainPerLine);
        if constexpr (numLoop) {
            for (int i = 0; i < numLoop; i++) {
                if (strideOverFlag) {
                    for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                        V_BIN_FUNC_VS(dst + i * REPEAT_MAX * DS + j * DS, src0 + i * REPEAT_MAX * SS0 + j * SS0, src1,
                            1, 1, 1, 1, 1);
                    }
                } else {
                    V_BIN_FUNC_VS(dst + i * REPEAT_MAX * DS, src0 + i * REPEAT_MAX * SS0, src1, REPEAT_MAX, 1, 1,
                        DS / blockSizeElem, SS0 / blockSizeElem);
                }
            }
        }
        if constexpr (remainAfterLoop) {
            if (strideOverFlag) {
                for (unsigned j = 0; j < REPEAT_MAX; j++) {
                    V_BIN_FUNC_VS((__ubuf__ T *)(dst + numLoop * REPEAT_MAX * DS + j * DS),
                        src0 + numLoop * REPEAT_MAX * SS0 + j * SS0, src1, 1, 1, 1, 1, 1);
                }
            } else {
                V_BIN_FUNC_VS((__ubuf__ T *)(dst + numLoop * REPEAT_MAX * DS), src0 + numLoop * REPEAT_MAX * SS0, src1,
                    remainAfterLoop, 1, 1, DS / blockSizeElem, SS0 / blockSizeElem);
            }
        }
        set_vector_mask(-1, -1);
    }
}

// dim4
template <typename T, unsigned T0, unsigned T1, unsigned T2, unsigned T3, unsigned DS0, unsigned DS1, unsigned DS2,
    unsigned S0S0, unsigned S0S1, unsigned S0S2>
TILEOP void T_BIN_VS(__ubuf__ T *dst, __ubuf__ T *src0, T src1) {
    static_assert((DS2 * sizeof(T)) % BLOCK_SIZE == 0);
    static_assert((S0S2 * sizeof(T)) % BLOCK_SIZE == 0);
    for (int i = 0; i < T0; i++) {
        __ubuf__ T *dst_ = dst;
        __ubuf__ T *src0_ = src0;
        for (int j = 0; j < T1; j++) {
            T_BIN_VS<T, T2, T3, DS2, S0S2>(dst_, src0_, src1);
            dst_ += DS1 * DS2;
            src0_ += S0S1 * S0S2;
        }
        dst += DS0 * DS1 * DS2;
        src0 += S0S0 * S0S1 * S0S2;
    }
}