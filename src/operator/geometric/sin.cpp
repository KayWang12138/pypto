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
 * \file sin.cpp
 * \brief
 */

#include "interface/operation/operation_impl.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "tilefwk/tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/tensor/tensormap.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/common.h"
#include "interface/utils/id_gen.h"
#include "interface/utils/log.h"
using namespace npu::tile_fwk;

namespace npu::tile_fwk {
Tensor Sin(Tensor operand) {
    // An algorithm guarante data precision from -10^10 to 10^10
    auto dType = operand->Datatype();
    if (dType != DataType::DT_FP32) {
        operand = Cast(operand, DataType::DT_FP32);
    }

    constexpr float number2048 = 2048.0f;
    constexpr float oneOverN = 1.0 / 2048.0;
    constexpr float invHalfPi = 0.63661975f;
    constexpr float pi0 = 1.5708008f;
    constexpr float pi1 = -0.0000044535846f;
    constexpr float pi2 = -8.706138e-10f;
    constexpr float F_025 = 0.25;
    constexpr float F_05 = 0.5;
    constexpr float F_4 = 4.0;
    constexpr float F_1 = 1.0;
    constexpr float F_NEGA_1 = -1.0;
    constexpr float F_NEGA_2 = -2.0;

    auto xScaled = MulS(operand, Element(DataType::DT_FP32, oneOverN));
    auto xOverpi = MulS(xScaled, Element(DataType::DT_FP32, invHalfPi));
    auto n = Cast(xOverpi, DataType::DT_FP32, CAST_ROUND);
    auto n0 = MulS(xOverpi, Element(DataType::DT_FP32, oneOverN));
    n0 = Cast(n0, DataType::DT_FP32, CAST_ROUND);
    n0 = MulS(n0, Element(DataType::DT_FP32, number2048));

    auto n1 = Sub(n, n0);

    auto fix = MulS(n0, Element(DataType::DT_FP32, pi0));
    auto xFix = Sub(xScaled, fix);
    fix = MulS(n1, Element(DataType::DT_FP32, pi0));
    xFix = Sub(xFix, fix);
    fix = MulS(n0, Element(DataType::DT_FP32, pi1));
    xFix = Sub(xFix, fix);
    fix = MulS(n1, Element(DataType::DT_FP32, pi1));
    xFix = Sub(xFix, fix);
    fix = MulS(n0, Element(DataType::DT_FP32, pi2));
    xFix = Sub(xFix, fix);

    constexpr float PI_02 = 1.5703125f;
    constexpr float PI_12 = 0.0004837513f;

    auto remainX = MulS(xFix, Element(DataType::DT_FP32, number2048));
    auto temp = MulS(remainX, Element(DataType::DT_FP32, invHalfPi));
    auto n2 = Cast(temp, DataType::DT_FP32, CAST_ROUND);

    n0 = MulS(n0, Element(DataType::DT_FP32, number2048));
    n1 = MulS(n1, Element(DataType::DT_FP32, number2048));
    fix = MulS(n0, Element(DataType::DT_FP32, PI_02));
    xFix = Sub(operand, fix);
    fix = MulS(n1, Element(DataType::DT_FP32, PI_02));
    xFix = Sub(xFix, fix);
    fix = MulS(n0, Element(DataType::DT_FP32, PI_12));
    xFix = Sub(xFix, fix);

    constexpr float PI_22 = 0.000000075495336f;
    fix = MulS(n2, Element(DataType::DT_FP32, PI_02));
    xFix = Sub(xFix, fix);
    fix = MulS(n1, Element(DataType::DT_FP32, PI_12));
    xFix = Sub(xFix, fix);
    fix = MulS(n0, Element(DataType::DT_FP32, PI_22));
    xFix = Sub(xFix, fix);

    constexpr float PI_32 = 2.5579538e-12f;
    fix = MulS(n2, Element(DataType::DT_FP32, PI_12));
    xFix = Sub(xFix, fix);
    fix = MulS(n1, Element(DataType::DT_FP32, PI_22));
    xFix = Sub(xFix, fix);
    fix = MulS(n0, Element(DataType::DT_FP32, PI_32));
    xFix = Sub(xFix, fix);

    constexpr float PI_42 = 5.389786e-15f;
    fix = MulS(n2, Element(DataType::DT_FP32, PI_22));
    xFix = Sub(xFix, fix);
    fix = MulS(n1, Element(DataType::DT_FP32, PI_32));
    xFix = Sub(xFix, fix);
    fix = MulS(n0, Element(DataType::DT_FP32, PI_42));
    xFix = Sub(xFix, fix);

    constexpr float PI_52 = 5.166901e-19f;
    fix = MulS(n2, Element(DataType::DT_FP32, PI_32));
    xFix = Sub(xFix, fix);
    fix = MulS(n1, Element(DataType::DT_FP32, PI_42));
    xFix = Sub(xFix, fix);
    fix = MulS(n0, Element(DataType::DT_FP32, PI_52));
    xFix = Sub(xFix, fix);

    constexpr float PI_62 = 3.281839e-22f;
    fix = MulS(n2, Element(DataType::DT_FP32, PI_42));
    xFix = Sub(xFix, fix);
    fix = MulS(n1, Element(DataType::DT_FP32, PI_52));
    xFix = Sub(xFix, fix);
    fix = MulS(n0, Element(DataType::DT_FP32, PI_62));
    xFix = Sub(xFix, fix);

    fix = MulS(n2, Element(DataType::DT_FP32, PI_52));
    xFix = Sub(xFix, fix);
    fix = MulS(n1, Element(DataType::DT_FP32, PI_62));
    xFix = Sub(xFix, fix);
    fix = MulS(n2, Element(DataType::DT_FP32, PI_62));
    xFix = Sub(xFix, fix);

    auto halfN2 = MulS(n2, Element(DataType::DT_FP32, F_05));
    auto half4N2 = MulS(n2, Element(DataType::DT_FP32, F_025));
    auto nHalf2 = Cast(halfN2, DataType::DT_FP32, CAST_FLOOR);
    auto nHalf4 = Cast(half4N2, DataType::DT_FP32, CAST_FLOOR);

    auto k1 = MulS(nHalf2, Element(DataType::DT_FP32, F_NEGA_2));
    auto k2 = MulS(nHalf4, Element(DataType::DT_FP32, F_4));
    auto sign = Add(k1, k2);
    sign = AddS(sign, Element(DataType::DT_FP32, F_1));

    auto ifcos = Add(n2, k1);
    auto ifsin = MulS(ifcos, Element(DataType::DT_FP32, F_NEGA_1));
    ifsin = AddS(ifsin, Element(DataType::DT_FP32, F_1));

    constexpr float scoef4 = 0.0000027183114939898219064f;
    constexpr float scoef3 = -0.000198393348360966317347f;
    constexpr float scoef2 = 0.0083333293858894631756f;
    constexpr float scoef1 = -0.166666666416265235595f;
    auto xPow = Mul(xFix, xFix);
    auto sinPoly = MulS(xPow, Element(DataType::DT_FP32, scoef4));
    sinPoly = AddS(sinPoly, Element(DataType::DT_FP32, scoef3));
    sinPoly = Mul(xPow, sinPoly);
    sinPoly = AddS(sinPoly, Element(DataType::DT_FP32, scoef2));
    sinPoly = Mul(xPow, sinPoly);
    sinPoly = AddS(sinPoly, Element(DataType::DT_FP32, scoef1));
    sinPoly = Mul(xPow, sinPoly);
    sinPoly = AddS(sinPoly, Element(DataType::DT_FP32, F_1));
    sinPoly = Mul(xFix, sinPoly);

    constexpr float ccoef4 = 0.0000243904487962774090654f;
    constexpr float ccoef3 = -0.00138867637746099294692f;
    constexpr float ccoef2 = 0.0416666233237390631894f;
    constexpr float ccoef1 = -0.499999997251031003120f;
    auto cosPoly = MulS(xPow, Element(DataType::DT_FP32, ccoef4));
    cosPoly = AddS(cosPoly, Element(DataType::DT_FP32, ccoef3));
    cosPoly = Mul(xPow, cosPoly);
    cosPoly = AddS(cosPoly, Element(DataType::DT_FP32, ccoef2));
    cosPoly = Mul(xPow, cosPoly);
    cosPoly = AddS(cosPoly, Element(DataType::DT_FP32, ccoef1));
    cosPoly = Mul(xPow, cosPoly);
    cosPoly = AddS(cosPoly, Element(DataType::DT_FP32, F_1));

    auto temp1 = Mul(sinPoly, ifsin);
    cosPoly = Mul(cosPoly, ifcos);
    auto res = Add(temp1, cosPoly);
    res = Mul(res, sign);
    if (dType != res->Datatype()) {
        res = Cast(res, dType);
    }
    return res;
}
} // namespace npu::tile_fwk