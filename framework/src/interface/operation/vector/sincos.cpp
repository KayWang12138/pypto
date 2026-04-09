/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sincos.cpp
 * \brief
 */

#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "interface/utils/vector_error.h"

namespace npu::tile_fwk {

const uint8_t HALF_CALC_PROCEDURE = 4;
const uint8_t FLOAT_NOREUSE_CALC_PROCEDURE = 3;
const uint8_t FLOAT_REUSE_CALC_PROCEDURE = 2;

// define the number of x div pi
constexpr float PI_FOR_X_TODIV = 0.3183098733425140380859375f;
// define the PI for compute
constexpr float PI_V2 = 3.140625;
constexpr float KPI_FIRS_PI_MULS = 0.0009670257568359375f;
constexpr float KPI_TWI_PI_MULS = 6.2771141529083251953125e-7f;
constexpr float KPI_THIR_PI_MULS = 1.21644916362129151821136474609375e-10f;
constexpr float KPI_FOR_PI_MULS = -1.0290623200529979163359041220560e-13f;
// define the number of down of pi_div
constexpr float PI_DOWN = 1.57079637050628662109375f;
// kpi_2
constexpr float PI_RESDOWN_ADDS_NEG = -0.00000004371139000189375f;
constexpr float RES_MULTI_SCA = 2.604926501e-6f;
constexpr float RES_ADDICT_UP = -0.0001980894471f;
constexpr float ADD2S = 0.008333049340f;
constexpr float ADD3S = -0.1666665792f;
constexpr float POINT_FIVE = 0.5;
constexpr float M4_SCA = 4.0;
constexpr float K2_SCA = -2.0;

static Tensor SinCosRangeReduction(const Tensor& self, const Tensor& k, const std::string& op)
{
    Tensor roundSelf = self;
    // x -= k * pi_0
    Tensor kpi = Mul(k, Element(DT_FP32, PI_V2));
    roundSelf = Sub(roundSelf, kpi);

    // x -= k * pi_1
    kpi = Mul(k, Element(DT_FP32, KPI_FIRS_PI_MULS));
    roundSelf = Sub(roundSelf, kpi);

    // x = x + PI_DOWN
    if (op == "COS") {
        roundSelf = Add(roundSelf, Element(DT_FP32, PI_DOWN));
    }

    // x -= k * pi_2
    kpi = Mul(k, Element(DT_FP32, KPI_TWI_PI_MULS));
    roundSelf = Sub(roundSelf, kpi);

    // x -= k * pi_3
    kpi = Mul(k, Element(DT_FP32, KPI_THIR_PI_MULS));
    roundSelf = Sub(roundSelf, kpi);

    if (op == "COS") {
        // x -= k * pi_4
        kpi = Mul(k, Element(DT_FP32, KPI_FOR_PI_MULS));
        roundSelf = Sub(roundSelf, kpi);

        // x = x + PI_RESDOWN_ADDS_NEG
        roundSelf = Add(roundSelf, Element(DT_FP32, PI_RESDOWN_ADDS_NEG));
    }
    return roundSelf;
}

static Tensor reduceKCompute(const Tensor& self, const std::string& op)
{
    //!注意内存复用
    //  k=round(x/π), x0=x-kπ, x0 belongs to [-π/2, π/2]   
    //  cos(x) = (-1)^k * sin(x0 + π/2)
    Tensor k = Mul(self, Element(DT_FP32, PI_FOR_X_TODIV));
    Tensor castInt32K = k;
    if (op == "COS") {
        k = Add(k, Element(DT_FP32, POINT_FIVE));
        castInt32K = Cast(k, DataType::DT_INT32, CastMode::CAST_RINT);
    } else {
        castInt32K = Cast(k, DataType::DT_INT32, CastMode::CAST_ROUND);
    }
    k = Cast(castInt32K, DataType::DT_FP32, CastMode::CAST_NONE);
    return k;
}

static Tensor SinCosCompute(const Tensor& self, const std::string& op)
{
    Tensor k = reduceKCompute(self, op);
    Tensor roundSelf = SinCosRangeReduction(self, k, op);
    // kover2
    Tensor result = Mul(k, Element(DT_FP32, POINT_FIVE));
    Tensor castInt32Result = Cast(result, DataType::DT_INT32, CastMode::CAST_FLOOR);
    result = Cast(castInt32Result, DataType::DT_FP32, CastMode::CAST_NONE);

    // kover2floorm4
    result = Mul(result, Element(DT_FP32, M4_SCA));
    //k2
    k = Mul(k, Element(DT_FP32, K2_SCA));
    //sign
    result = Add(result, k);
    result = Add(result, Element(DT_FP32, 1.0f));
    // x^2
    Tensor squareX = Mul(roundSelf, roundSelf);
    // sin(x) = x * P(x)
    // P(x) = (((x^2 * R0 + R1) * x^2 + R2) * x^2 + R3) * x^2 + 1.0
    // roundTensor = mul(x^2, 2.604926501e-6)
    Tensor roundTensor = Mul(squareX, Element(DT_FP32, RES_MULTI_SCA));
    roundTensor = Add(roundTensor, Element(DT_FP32, RES_ADDICT_UP));

    // roundTensor = mul(roundTensor, x^2)
    roundTensor = Mul(squareX, roundTensor);
    roundTensor = Add(roundTensor, Element(DT_FP32, ADD2S));

    // roundTensor = mul(roundTensor, x^2)
    roundTensor = Mul(squareX, roundTensor);
    roundTensor = Add(roundTensor, Element(DT_FP32, ADD3S));

    // roundTensor = mul(roundTensor, x^2)
    roundTensor = Mul(squareX, roundTensor);
    roundTensor = Add(roundTensor, Element(DT_FP32, 1.0f));

    // sin(x) = x * P(x)
    roundTensor = Mul(roundSelf, roundTensor);
    result = Mul(result, roundTensor);

    // result = Min(result, Element(DT_FP32, 1.0f));
    // result = Max(result, Element(DT_FP32, -1.0f));
    return result;
}



Tensor Sin(const Tensor& self)
{
    DECLARE_TRACER();

    auto shapeSize = self.GetShape().size();
    auto dataType = self.GetDataType();
    ASSERT(SHAPE_DIM2 <= shapeSize && shapeSize <= SHAPE_DIM4) << "The shape.size() only support 2~4";
    std::vector<DataType> EXPM1_SUPPORT_DATATYPES = {
        DataType::DT_FP32, DataType::DT_FP16};
    ASSERT(
        std::find(EXPM1_SUPPORT_DATATYPES.begin(), EXPM1_SUPPORT_DATATYPES.end(), dataType) !=
        EXPM1_SUPPORT_DATATYPES.end())
        << "The datatype is not supported";
    Tensor castSelf = self;
    if (self.GetDataType() == DataType::DT_FP16) {
        castSelf = Cast(self, DataType::DT_FP32, CastMode::CAST_NONE);
    }
    Tensor result=SinCosCompute(castSelf, "SIN");
    Tensor castResult = result;
    if (self.GetDataType() == DataType::DT_FP16) {
        castResult = Cast(result, DataType::DT_FP16, CastMode::CAST_NONE);
    }
    return castResult;
    // RETURN_CALL(Expm1, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor Cos(const Tensor& self)
{
    DECLARE_TRACER();

    auto shapeSize = self.GetShape().size();
    auto dataType = self.GetDataType();
    ASSERT(SHAPE_DIM2 <= shapeSize && shapeSize <= SHAPE_DIM4) << "The shape.size() only support 2~4";
    std::vector<DataType> EXPM1_SUPPORT_DATATYPES = {
        DataType::DT_FP32, DataType::DT_FP16};
    ASSERT(
        std::find(EXPM1_SUPPORT_DATATYPES.begin(), EXPM1_SUPPORT_DATATYPES.end(), dataType) !=
        EXPM1_SUPPORT_DATATYPES.end())
        << "The datatype is not supported";
    Tensor castSelf = self;
    if (self.GetDataType() == DataType::DT_FP16) {
        castSelf = Cast(self, DataType::DT_FP32, CastMode::CAST_NONE);
    }
    Tensor result=SinCosCompute(castSelf, "COS");

    Tensor castResult = result;
    if (self.GetDataType() == DataType::DT_FP16) {
        castResult = Cast(result, DataType::DT_FP16, CastMode::CAST_NONE);
    }
    return castResult;
    // RETURN_CALL(Expm1, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}
} // namespace npu::tile_fwk