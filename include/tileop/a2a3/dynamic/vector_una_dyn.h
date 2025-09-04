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
 * \file vector_una_dyn.h
 * \brief
 */
#ifndef TILE_FWK_VECTOR_UNA_DYN_H
#define TILE_FWK_VECTOR_UNA_DYN_H

#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <UnaryOp op, typename T>
TILEOP void UnaryCompute(__ubuf__ T *dst, __ubuf__ T *src, unsigned L0, unsigned S2, unsigned S3) {
    constexpr unsigned simdw = REPEAT_BYTE / sizeof(T);
    if constexpr (op == UnaryOp::ABS) {
        vabs(dst, src, L0, 1, 1, S2, S3);
    }
    if constexpr (op == UnaryOp::EXP) {
        vexp(dst, src, L0, 1, 1, S2, S3);
    }
    if constexpr (op == UnaryOp::REC) {
        vrec(dst, src, L0, 1, 1, S2, S3);
    }
    if constexpr (op == UnaryOp::SQRT) {
        vsqrt(dst, src, L0, 1, 1, S2, S3);
    }
}

template <typename T, size_t N>
TILEOP inline int GetShape(const T t) {
    auto layout = (typename T::LayoutType)t.GetLayout();
    return Std::get<N>(layout.GetShape());
}

template <typename T, size_t N>
TILEOP inline int GetStride(const T t) {
    auto layout = (typename T::LayoutType)t.GetLayout();
    return Std::get<N>(layout.GetStride());
}

template <UnaryOp op, typename DT, typename ST>
TILEOP void UnaryOpRepeatPerLine(DT dst, ST src) {
    constexpr unsigned simdw = REPEAT_BYTE / sizeof(typename DT::Type);
    unsigned numRepeatPerLine = GetShape<DT, 1>(dst) / simdw;
    if (numRepeatPerLine == 0) {
        return;
    }
    using VecType = typename DT::Type;
    const unsigned DS = GetStride<DT, 1>(dst);
    const unsigned SS = GetStride<ST, 1>(src);
    unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
    unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
    auto dst_base = (__ubuf__ VecType *)dst.GetAddr();
    auto src_base = (__ubuf__ VecType *)src.GetAddr();
    for (int i = 0; i < GetShape<DT, 0>(dst); i++) {
        if (numLoop) {
            for (int j = 0; j < numLoop; j++) {
                auto _dst = dst_base + i * DS + j * simdw * REPEAT_MAX;
                auto _src = src_base + i * SS + j * simdw * REPEAT_MAX;
                UnaryCompute<op, typename DT::Type>(_dst, _src, REPEAT_MAX, 8, 8);
            }
        }
        if (remainAfterLoop) {
            auto _dst = dst_base + i * DS + simdw * REPEAT_MAX * numLoop;
            auto _src = src_base + i * SS + simdw * REPEAT_MAX * numLoop;
            UnaryCompute<op, typename DT::Type>(_dst, _src, remainAfterLoop, 8, 8);
        }
    }
}

template <UnaryOp op, typename DT, typename ST>
TILEOP void UnaryOpRemainPerLine(DT dst, ST src) {
    constexpr unsigned simdw = REPEAT_BYTE / sizeof(typename DT::Type);
    const unsigned numRemainPerLine = GetShape<DT, 1>(dst) % simdw;
    if (numRemainPerLine == 0) {
        return;
    }
    using VecType = typename DT::Type;
    const unsigned DS = GetStride<DT, 1>(dst);
    const unsigned SS = GetStride<ST, 1>(src);
    constexpr unsigned nElemPerBlock = BLOCK_SIZE / sizeof(typename DT::Type);
    const unsigned T0 = GetShape<ST, 0>(src);
    const unsigned numLoop = T0 / REPEAT_MAX;
    const unsigned remainAfterLoop = T0 % REPEAT_MAX;
    bool strideOverFlag = (DS / nElemPerBlock > REPEAT_STRIDE_MAX) || (SS / nElemPerBlock > REPEAT_STRIDE_MAX);
    SetContinuousMask(numRemainPerLine);
    // shift to deal with tail
    const unsigned numRepeatPerLine = GetShape<DT, 1>(dst) / simdw;
    auto dst_base = (__ubuf__ VecType *)dst.GetAddr() + numRepeatPerLine * simdw;
    auto src_base = (__ubuf__ VecType *)src.GetAddr() + numRepeatPerLine * simdw;
    for (int i = 0; i < numLoop; i++) {
        if (strideOverFlag) {
            for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                auto _dst = dst_base + i * REPEAT_MAX * DS + j * DS;
                auto _src = src_base + i * REPEAT_MAX * SS + j * SS;
                UnaryCompute<op, typename DT::Type>(_dst, _src, 1, 1, 1);
            }
        } else {
            auto _dst = dst_base + i * REPEAT_MAX * DS;
            auto _src = src_base + i * REPEAT_MAX * SS;
            UnaryCompute<op, typename DT::Type>(_dst, _src, REPEAT_MAX, DS / nElemPerBlock, SS / nElemPerBlock);
        }
    }
    if (remainAfterLoop == 0) {
        return;
    }
    if (strideOverFlag) {
        for (unsigned j = 0; j < remainAfterLoop; j++) {
            auto _dst = dst_base + numLoop * REPEAT_MAX * DS + j * DS;
            auto _src = src_base + numLoop * REPEAT_MAX * SS + j * SS;
            UnaryCompute<op, typename DT::Type>(_dst, _src, 1, 1, 1);
        }
    } else {
        auto _dst = dst_base + numLoop * REPEAT_MAX * DS;
        auto _src = src_base + numLoop * REPEAT_MAX * SS;
        UnaryCompute<op, typename DT::Type>(_dst, _src, remainAfterLoop, DS / nElemPerBlock, SS / nElemPerBlock);
    }
    set_vector_mask(-1, -1);
}

// dim2 & dim1 (T0 = 1 for dim1)
template <UnaryOp op, typename DT, typename ST>
TILEOP void UnaryOperation2Dim(DT dst, ST src) {
    UnaryOpRepeatPerLine<op, DT, ST>(dst, src);
    UnaryOpRemainPerLine<op, DT, ST>(dst, src);
}

// dim3
template <UnaryOp op, typename DT, typename ST>
TILEOP void UnaryOperation3Dim(DT dst, ST src) {
    const auto DS1 = GetStride<DT, 1>(dst);
    const auto DS2 = GetStride<DT, 2>(dst);
    const auto SS1 = GetStride<ST, 1>(src);
    const auto SS2 = GetStride<ST, 2>(src);
    auto subDstShape = MakeShape<int, int>(GetShape<DT, 1>(dst), GetShape<DT, 2>(dst));
    auto subDstStride = MakeStride<int, int>(DS1, DS2);
    auto subDstLayout = MakeLayout<decltype(subDstShape), decltype(subDstStride)>(subDstShape, subDstStride);
    auto subSrcShape = MakeShape<int, int>(GetShape<ST, 1>(src), GetShape<ST, 2>(src));
    auto subSrcStride = MakeStride<int, int>(SS1, SS2);
    auto subSrcLayout = MakeLayout<decltype(subSrcShape), decltype(subSrcStride)>(subSrcShape, subSrcStride);
    __ubuf__ typename DT::Type *_dst = dst.GetAddr();
    __ubuf__ typename ST::Type *_src = src.GetAddr();
    for (int i = 0; i < GetShape<DT, 0>(dst); i++) {
        auto dstTensor = MakeTensor<typename DT::Type, Layout2Dim>(_dst, subDstLayout);
        auto srcTensor = MakeTensor<typename ST::Type, Layout2Dim>(_src, subSrcLayout);
        UnaryOperation2Dim<op, decltype(dstTensor), decltype(srcTensor)>(dstTensor, srcTensor);
        _dst += DS1 * DS2;
        _src += SS1 * SS2;
    }
}

// dim4
template <UnaryOp op, typename DT, typename ST>
TILEOP void UnaryOperation4Dim(DT dst, ST src) {
    const auto DS1 = GetStride<DT, 1>(dst);
    const auto DS2 = GetStride<DT, 2>(dst);
    const auto DS3 = GetStride<DT, 3>(dst);
    const auto SS1 = GetStride<ST, 1>(src);
    const auto SS2 = GetStride<ST, 2>(src);
    const auto SS3 = GetStride<ST, 3>(src);
    auto subDstShape = MakeShape<int, int>(GetShape<DT, 2>(dst), GetShape<DT, 3>(dst));
    auto subDstStride = MakeStride<int, int>(DS2, DS3);
    auto subDstLayout = MakeLayout<decltype(subDstShape), decltype(subDstStride)>(subDstShape, subDstStride);
    auto subSrcShape = MakeShape<int, int>(GetShape<ST, 2>(src), GetShape<ST, 3>(src));
    auto subSrcStride = MakeStride<int, int>(SS2, SS3);
    auto subSrcLayout = MakeLayout<decltype(subSrcShape), decltype(subSrcStride)>(subSrcShape, subSrcStride);
    __ubuf__ typename DT::Type *_dst = dst.GetAddr();
    __ubuf__ typename ST::Type *_src = src.GetAddr();
    for (int i = 0; i < GetShape<DT, 0>(dst); i++) {
        auto dst0 = _dst;
        auto src0 = _src;
        for (int j = 0; j < GetShape<DT, 1>(dst); j++) {
            auto dstTensor = MakeTensor<typename DT::Type, Layout2Dim>(dst0, subDstLayout);
            auto srcTensor = MakeTensor<typename ST::Type, Layout2Dim>(src0, subSrcLayout);
            UnaryOperation2Dim<op, decltype(dstTensor), decltype(srcTensor)>(dstTensor, srcTensor);
            dst0 += DS2 * DS3;
            src0 += SS2 * SS3;
        }
        _dst += DS1 * DS2 * DS3;
        _src += SS1 * SS2 * SS3;
    }
}

template <UnaryOp op, typename DT, typename ST>
TILEOP void UnaryOperation(DT dst, ST src) {
    auto dstLayout = (typename DT::LayoutType)dst.GetLayout();
    constexpr auto dims = Std::tuple_size<decltype(dstLayout.GetShape())>::value;
    static_assert(dims > 1 && dims < 5);

    if constexpr (dims == 2) {
        UnaryOperation2Dim<op, DT, ST>(dst, src);
    }
    if constexpr (dims == 3) {
        UnaryOperation3Dim<op, DT, ST>(dst, src);
    }
    if constexpr (dims == 4) {
        UnaryOperation4Dim<op, DT, ST>(dst, src);
    }
}

template <typename DT, typename ST>
TILEOP void Abs(DT dst, ST src) {
    UnaryOperation<UnaryOp::ABS, DT, ST>(dst, src);
}

template <typename DT, typename ST>
TILEOP void Exp(DT dst, ST src) {
    UnaryOperation<UnaryOp::EXP, DT, ST>(dst, src);
}

template <typename DT, typename ST>
TILEOP void Rec(DT dst, ST src) {
    UnaryOperation<UnaryOp::REC, DT, ST>(dst, src);
}

template <typename DT, typename ST>
TILEOP void Sqrt(DT dst, ST src) {
    UnaryOperation<UnaryOp::SQRT, DT, ST>(dst, src);
}
#endif // TILE_FWK_VECTOR_UNA_DYN_H

// dim2 & dim1 (T0 = 1 for dim1)
template <typename T, unsigned DS, unsigned SS>
TILEOP void T_UNA(__ubuf__ T *dst, __ubuf__ T *src, unsigned T0, unsigned T1) {
    auto subDstShape = MakeShape<int, int>(T0, T1);
    auto subDstStride = MakeStride<int, int>(1, DS);
    auto subDstLayout = MakeLayout<decltype(subDstShape), decltype(subDstStride)>(subDstShape, subDstStride);
    auto subSrcShape = MakeShape<int, int>(T0, T1);
    auto subSrcStride = MakeStride<int, int>(1, SS);
    auto subSrcLayout = MakeLayout<decltype(subSrcShape), decltype(subSrcStride)>(subSrcShape, subSrcStride);
    auto dstTensor = MakeTensor<T, decltype(subDstLayout)>(dst, subDstLayout);
    auto srcTensor = MakeTensor<T, decltype(subSrcLayout)>(src, subSrcLayout);
    V_UNA_FUNC<decltype(dstTensor), decltype(srcTensor)>(dstTensor, srcTensor);
}

// dim3
template <typename T, unsigned DS0, unsigned DS1, unsigned SS0, unsigned SS1>
TILEOP void T_UNA(__ubuf__ T *dst, __ubuf__ T *src, unsigned T0, unsigned T1, unsigned T2) {
    auto subDstShape = MakeShape<int, int, int>(T0, T1, T2);
    auto subDstStride = MakeStride<int, int, int>(1, DS0, DS1);
    auto subDstLayout = MakeLayout<decltype(subDstShape), decltype(subDstStride)>(subDstShape, subDstStride);
    auto subSrcShape = MakeShape<int, int, int>(T0, T1, T2);
    auto subSrcStride = MakeStride<int, int, int>(1, SS0, SS1);
    auto subSrcLayout = MakeLayout<decltype(subSrcShape), decltype(subSrcStride)>(subSrcShape, subSrcStride);
    auto dstTensor = MakeTensor<T, decltype(subDstLayout)>(dst, subDstLayout);
    auto srcTensor = MakeTensor<T, decltype(subSrcLayout)>(src, subSrcLayout);
    V_UNA_FUNC<decltype(dstTensor), decltype(srcTensor)>(dstTensor, srcTensor);
}

// dim4
template <typename T, unsigned DS0, unsigned DS1, unsigned DS2, unsigned SS0, unsigned SS1, unsigned SS2>
TILEOP void T_UNA(__ubuf__ T *dst, __ubuf__ T *src, unsigned T0, unsigned T1, unsigned T2, unsigned T3) {
    auto subDstShape = MakeShape<int, int, int, int>(T0, T1, T2, T3);
    auto subDstStride = MakeStride<int, int, int, int>(1, DS0, DS1, DS2);
    auto subDstLayout = MakeLayout<decltype(subDstShape), decltype(subDstStride)>(subDstShape, subDstStride);
    auto subSrcShape = MakeShape<int, int, int, int>(T0, T1, T2, T3);
    auto subSrcStride = MakeStride<int, int, int, int>(1, SS0, SS1, SS2);
    auto subSrcLayout = MakeLayout<decltype(subSrcShape), decltype(subSrcStride)>(subSrcShape, subSrcStride);
    auto dstTensor = MakeTensor<T, decltype(subDstLayout)>(dst, subDstLayout);
    auto srcTensor = MakeTensor<T, decltype(subSrcLayout)>(src, subSrcLayout);
    V_UNA_FUNC<decltype(dstTensor), decltype(srcTensor)>(dstTensor, srcTensor);
}
