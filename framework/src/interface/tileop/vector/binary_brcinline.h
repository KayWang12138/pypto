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
 * \file binary_brcinline.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_BINARY_BRCINLINE__H
#define TILEOP_TILE_OPERATOR_BINARY_BRCINLINE__H
#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <BinaryOp op, typename LastUse, typename T0, typename T1, typename Scalar>
TILEOP void BinaryLeftScalarComputeImpl(T0 dst, Scalar src0, T1 src1) {
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;
    constexpr auto n3 = Std::tuple_element<DIM_3RD, LastUse>::type::value;
    if constexpr (op == BinaryOp::ADD) {
        PTO_WITH_LAST_USE(pto::TADDS(dst, src1, src0), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::SUB) {
        PTO_WITH_LAST_USE(pto::TNEG(dst, src1), n1, n2, n3);
        #ifdef __DAV_V220
        pipe_barrier(PIPE_V);
        #endif
        PTO_WITH_LAST_USE(pto::TADDS(dst, dst, scr0), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::MUL) {
        PTO_WITH_LAST_USE(pto::TMULS(dst, src1, src0), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::DIV) {
        PTO_WITH_LAST_USE(pto::TDIVS(dst, src1, src0), n1, n2, n3);   
        // 存在问题，scalar/tensor不支持，如果要用TRECIP 不允许 dst 和 src 指向同一块内存，这里地址一定会复用的
        return;
    }

    if constexpr (op == BinaryOp::MAX) {
        PTO_WITH_LAST_USE(pto::TMAXS(dst, src1, src0), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::MIN) {
        PTO_WITH_LAST_USE(pto::TMINS(dst, src1, src0), n1, n2, n3);
        return;
    }
}

template <BinaryOp op, typename LastUse, typename T0, typename T1, typename Scalar>
TILEOP void BinaryRightScalarComputeImpl(T0 dst, T1 src0, Scalar src1) {
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;
    constexpr auto n3 = Std::tuple_element<DIM_3RD, LastUse>::type::value;
    if constexpr (op == BinaryOp::ADD) {
        PTO_WITH_LAST_USE(pto::TADDS(dst, src0, src1), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::SUB) {
        PTO_WITH_LAST_USE(pto::TADDS(dst, src0, -src1), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::MUL) {
        PTO_WITH_LAST_USE(pto::TMULS(dst, src0, src1), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::DIV) {
        PTO_WITH_LAST_USE(pto::TDIVS(dst, src0, src1), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::MAX) {
        PTO_WITH_LAST_USE(pto::TMAXS(dst, src0, src1), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::MIN) {
        PTO_WITH_LAST_USE(pto::TMINS(dst, src0, src1), n1, n2, n3);
        return;
    }
}

template <BinaryOp op, typename LastUse, typename T0, typename T1, typename T2>
TILEOP void BinaryRowExpandComputeImpl(T0 dst, T1 src0, T2 src1) {
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;
    constexpr auto n3 = Std::tuple_element<DIM_3RD, LastUse>::type::value;
    
    if constexpr (op == BinaryOp::ADD) {
        PTO_WITH_LAST_USE(pto::TROWEXPANDADD(dst, src0, src1), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::SUB) {
        PTO_WITH_LAST_USE(pto::TROWEXPANDSUB(dst, src0, src1), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::MUL) {
        PTO_WITH_LAST_USE(pto::TROWEXPANDMUL(dst, src0, src1), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::DIV) {
        PTO_WITH_LAST_USE(pto::TROWEXPANDDIV(dst, src0, src1), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::MAX) {
        PTO_WITH_LAST_USE(pto::TROWEXPANDMAX(dst, src0, src1), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::MIN) {
        PTO_WITH_LAST_USE(pto::TROWEXPANDMIN(dst, src0, src1), n1, n2, n3);
        return;
    }
}

template <BinaryOp op, typename LastUse, typename T0, typename T1, typename T2>
TILEOP void BinaryColExpandComputeImpl(T0 dst, T1 full, T2 brc) {
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;
    constexpr auto n3 = Std::tuple_element<DIM_3RD, LastUse>::type::value;
    
    if constexpr (op == BinaryOp::ADD) {
        PTO_WITH_LAST_USE(pto::TCOLEXPANDADD(dst, full, brc), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::SUB) {
        PTO_WITH_LAST_USE(pto::TCOLEXPANDSUB(dst, full, brc), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::MUL) {
        PTO_WITH_LAST_USE(pto::TCOLEXPANDMUL(dst, full, brc), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::DIV) {
        PTO_WITH_LAST_USE(pto::TCOLEXPANDDIV(dst, full, brc), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::MAX) {
        PTO_WITH_LAST_USE(pto::TCOLEXPANDMAX(dst, full, brc), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::MIN) {
        PTO_WITH_LAST_USE(pto::TCOLEXPANDMIN(dst, full, brc), n1, n2, n3);
        return;
    }
}

#endif