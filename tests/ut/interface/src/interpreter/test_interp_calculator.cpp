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
 * \file test_pool.cpp
 * \brief
 */

#include <gtest/gtest.h>

#include <math.h>
#include "interface/utils/log.h"
#include "interface/interpreter/thread_pool.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/interpreter/calculator.h"
#include "interface/tensor/float.h"

#include <chrono>

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::util;

namespace {
TEST(ThreadPoolTest, Dispatch) {
    const int nproc = 2;
    struct Handler {
        static void Entry(void *ctx) {
            int threadIndex = (intptr_t)ctx;
            ALOG_INFO("Before: ", threadIndex);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            ALOG_INFO("After: ", threadIndex);
        }
    };
    {
        ThreadPool pool(nproc);
        for (int i = 0; i < nproc * 2; i++) {
            pool.SubmitTask((void *)(intptr_t)i, Handler::Entry);
        }
        pool.NotifyAll();
        pool.WaitForAll();
        pool.Stop();
    }
}

#define EXPECT_TENSOR_DATA(data, size, value_type, expect_type, expect_value) \
    do {                                                                      \
        EXPECT_EQ(data->GetSize(), size);                                     \
        for (int i = 0; i < size; i++) {                                      \
            EXPECT_EQ((expect_type)data->Get<value_type>(i), expect_value);   \
        }                                                                     \
    } while (0)
#define EXPECT_TENSOR_LISTDATA(data, size, value_type, expect_type, expect_value) \
    do {                                                                          \
        EXPECT_EQ(data->GetSize(), size);                                         \
        for (int i = 0; i < size; i++) {                                          \
            EXPECT_EQ((expect_type)data->Get<value_type>(i), expect_value[i]);    \
        }                                                                         \
    } while (0)

TEST(ThreadPoolTest, ElementWise) {
    const int nproc = 2;
    ThreadPool pool(nproc);
    Tensor lhsTensor(DT_BF16, {64, 64}, "lhs");
    Tensor rhsTensor(DT_BF16, {64, 64}, "rhs");

    auto lhsData = RawTensorData::CreateConstantTensor<npu::tile_fwk::bfloat16>(lhsTensor, npu::tile_fwk::bfloat16(2.5));
    auto rhsData = RawTensorData::CreateConstantTensor<npu::tile_fwk::bfloat16>(rhsTensor, npu::tile_fwk::bfloat16(2.0));
    LogicalTensorData lhs(lhsData);
    LogicalTensorData rhs(rhsData);

    auto add = LogicalTensorData::CreateEmpty(lhs.GetDataType(), lhs.GetShape(), std::vector<int64_t>(0));
    auto sub = LogicalTensorData::CreateEmpty(lhs.GetDataType(), lhs.GetShape(), std::vector<int64_t>(0));
    auto mul = LogicalTensorData::CreateEmpty(lhs.GetDataType(), lhs.GetShape(), std::vector<int64_t>(0));
    auto div = LogicalTensorData::CreateEmpty(lhs.GetDataType(), lhs.GetShape(), std::vector<int64_t>(0));
    auto max = LogicalTensorData::CreateEmpty(lhs.GetDataType(), lhs.GetShape(), std::vector<int64_t>(0));
    auto min = LogicalTensorData::CreateEmpty(lhs.GetDataType(), lhs.GetShape(), std::vector<int64_t>(0));

    Calculator::CalcAdd(add.get(), &lhs, &rhs, &pool);
    Calculator::CalcSub(sub.get(), &lhs, &rhs, &pool);
    Calculator::CalcMul(mul.get(), &lhs, &rhs, &pool);
    Calculator::CalcDiv(div.get(), &lhs, &rhs, &pool);
    Calculator::CalcMax(max.get(), &lhs, &rhs, &pool);
    Calculator::CalcMax(min.get(), &lhs, &rhs, &pool);

    EXPECT_EQ(64 * 64, lhs.GetSize());

    EXPECT_TENSOR_DATA(add, lhs.GetSize(), npu::tile_fwk::bfloat16, float, 2.5 + 2);
    EXPECT_TENSOR_DATA(sub, lhs.GetSize(), npu::tile_fwk::bfloat16, float, 2.5 - 2);
    EXPECT_TENSOR_DATA(mul, lhs.GetSize(), npu::tile_fwk::bfloat16, float, 2.5 * 2);
    EXPECT_TENSOR_DATA(div, lhs.GetSize(), npu::tile_fwk::bfloat16, float, 2.5 / 2);
    EXPECT_TENSOR_DATA(max, lhs.GetSize(), npu::tile_fwk::bfloat16, float, 2.5);

    auto rhss = Element(DT_BF16, 2.0);

    auto adds = LogicalTensorData::CreateEmpty(lhs.GetDataType(), lhs.GetShape(), std::vector<int64_t>(0));
    auto subs = LogicalTensorData::CreateEmpty(lhs.GetDataType(), lhs.GetShape(), std::vector<int64_t>(0));
    auto muls = LogicalTensorData::CreateEmpty(lhs.GetDataType(), lhs.GetShape(), std::vector<int64_t>(0));
    auto divs = LogicalTensorData::CreateEmpty(lhs.GetDataType(), lhs.GetShape(), std::vector<int64_t>(0));
    auto maxs = LogicalTensorData::CreateEmpty(lhs.GetDataType(), lhs.GetShape(), std::vector<int64_t>(0));
    auto mins = LogicalTensorData::CreateEmpty(lhs.GetDataType(), lhs.GetShape(), std::vector<int64_t>(0));
    Calculator::CalcAddS(adds.get(), &lhs, &rhss, false, &pool);
    Calculator::CalcSubS(subs.get(), &lhs, &rhss, false, &pool);
    Calculator::CalcMulS(muls.get(), &lhs, &rhss, false, &pool);
    Calculator::CalcDivS(divs.get(), &lhs, &rhss, false, &pool);
    Calculator::CalcMaxS(maxs.get(), &lhs, &rhss, false, &pool);
    Calculator::CalcMinS(mins.get(), &lhs, &rhss, false, &pool);

    EXPECT_TENSOR_DATA(adds, lhs.GetSize(), npu::tile_fwk::bfloat16, float, 2.5 + 2);
    EXPECT_TENSOR_DATA(subs, lhs.GetSize(), npu::tile_fwk::bfloat16, float, 2.5 - 2);
    EXPECT_TENSOR_DATA(muls, lhs.GetSize(), npu::tile_fwk::bfloat16, float, 2.5 * 2);
    EXPECT_TENSOR_DATA(divs, lhs.GetSize(), npu::tile_fwk::bfloat16, float, 2.5 / 2);
    EXPECT_TENSOR_DATA(maxs, lhs.GetSize(), npu::tile_fwk::bfloat16, float, 2.5);
    EXPECT_TENSOR_DATA(mins, lhs.GetSize(), npu::tile_fwk::bfloat16, float, 2);

    auto exps = LogicalTensorData::CreateEmpty(lhs.GetDataType(), lhs.GetShape(), std::vector<int64_t>(0));
    Calculator::CalcExp(exps.get(), &lhs, &pool);
    EXPECT_TENSOR_DATA(exps, lhs.GetSize(), npu::tile_fwk::bfloat16, npu::tile_fwk::bfloat16, npu::tile_fwk::bfloat16(exp(2.5)));

    auto sqrts = LogicalTensorData::CreateEmpty(lhs.GetDataType(), lhs.GetShape(), std::vector<int64_t>(0));
    Calculator::CalcSqrt(sqrts.get(), &lhs, &pool);
    EXPECT_TENSOR_DATA(sqrts, lhs.GetSize(), npu::tile_fwk::bfloat16, npu::tile_fwk::bfloat16, npu::tile_fwk::bfloat16(sqrt(2.5)));

    auto castu8 = LogicalTensorData::CreateEmpty(DT_UINT8, lhs.GetShape(), std::vector<int64_t>(0));
    auto castu16 = LogicalTensorData::CreateEmpty(DT_UINT16, lhs.GetShape(), std::vector<int64_t>(0));
    auto castu32 = LogicalTensorData::CreateEmpty(DT_UINT32, lhs.GetShape(), std::vector<int64_t>(0));
    auto castu64 = LogicalTensorData::CreateEmpty(DT_UINT64, lhs.GetShape(), std::vector<int64_t>(0));
    auto casts8 = LogicalTensorData::CreateEmpty(DT_INT8, lhs.GetShape(), std::vector<int64_t>(0));
    auto casts16 = LogicalTensorData::CreateEmpty(DT_INT16, lhs.GetShape(), std::vector<int64_t>(0));
    auto casts32 = LogicalTensorData::CreateEmpty(DT_INT32, lhs.GetShape(), std::vector<int64_t>(0));
    auto casts64 = LogicalTensorData::CreateEmpty(DT_INT64, lhs.GetShape(), std::vector<int64_t>(0));
    auto castf16 = LogicalTensorData::CreateEmpty(DT_FP16, lhs.GetShape(), std::vector<int64_t>(0));
    auto castf32 = LogicalTensorData::CreateEmpty(DT_FP32, lhs.GetShape(), std::vector<int64_t>(0));
    auto castf64 = LogicalTensorData::CreateEmpty(DT_DOUBLE, lhs.GetShape(), std::vector<int64_t>(0));
    auto castbf16 = LogicalTensorData::CreateEmpty(DT_BF16, lhs.GetShape(), std::vector<int64_t>(0));

    Calculator::CalcCast(castu8.get(), &lhs, DT_UINT8, CastMode::CAST_NONE, &pool);
    Calculator::CalcCast(castu16.get(), &lhs, DT_UINT16, CastMode::CAST_NONE, &pool);
    Calculator::CalcCast(castu32.get(), &lhs, DT_UINT32, CastMode::CAST_NONE, &pool);
    Calculator::CalcCast(castu64.get(), &lhs, DT_UINT64, CastMode::CAST_NONE, &pool);
    Calculator::CalcCast(casts8.get(), &lhs, DT_INT8, CastMode::CAST_NONE, &pool);
    Calculator::CalcCast(casts16.get(), &lhs, DT_INT16, CastMode::CAST_NONE, &pool);
    Calculator::CalcCast(casts32.get(), &lhs, DT_INT32, CastMode::CAST_NONE, &pool);
    Calculator::CalcCast(casts64.get(), &lhs, DT_INT64, CastMode::CAST_NONE, &pool);
    Calculator::CalcCast(castf16.get(), &lhs, DT_FP16, CastMode::CAST_NONE, &pool);
    Calculator::CalcCast(castf32.get(), &lhs, DT_FP32, CastMode::CAST_NONE, &pool);
    Calculator::CalcCast(castf64.get(), &lhs, DT_DOUBLE, CastMode::CAST_NONE, &pool);
    Calculator::CalcCast(castbf16.get(), &lhs, DT_BF16, CastMode::CAST_NONE, &pool);
    EXPECT_TENSOR_DATA(castu8, lhs.GetSize(), uint8_t, uint8_t, 2);
    EXPECT_TENSOR_DATA(castu16, lhs.GetSize(), uint16_t, uint16_t, 2);
    EXPECT_TENSOR_DATA(castu32, lhs.GetSize(), uint32_t, uint32_t, 2);
    EXPECT_TENSOR_DATA(castu64, lhs.GetSize(), uint64_t, uint64_t, 2);
    EXPECT_TENSOR_DATA(casts8, lhs.GetSize(), int8_t, int8_t, 2);
    EXPECT_TENSOR_DATA(casts16, lhs.GetSize(), int16_t, int16_t, 2);
    EXPECT_TENSOR_DATA(casts32, lhs.GetSize(), int32_t, int32_t, 2);
    EXPECT_TENSOR_DATA(casts64, lhs.GetSize(), int64_t, int64_t, 2);
    EXPECT_TENSOR_DATA(castf16, lhs.GetSize(), npu::tile_fwk::float16, float, 2.5);
    EXPECT_TENSOR_DATA(castf32, lhs.GetSize(), float, float, 2.5);
    EXPECT_TENSOR_DATA(castf64, lhs.GetSize(), double, float, 2.5);
    EXPECT_TENSOR_DATA(castbf16, lhs.GetSize(), npu::tile_fwk::bfloat16, float, 2.5);

    auto copy = LogicalTensorData::CreateEmpty(lhs.GetDataType(), lhs.GetShape(), std::vector<int64_t>(0));
    Calculator::CalcCopy(copy.get(), &lhs, &pool);
    EXPECT_TENSOR_DATA(copy, lhs.GetSize(), npu::tile_fwk::bfloat16, npu::tile_fwk::bfloat16, npu::tile_fwk::bfloat16(2.5));

    auto vecdup = LogicalTensorData::CreateEmpty(lhs.GetDataType(), lhs.GetShape(), std::vector<int64_t>(0));
    Calculator::CalcVecDup(vecdup.get(), &rhss, &pool);
    EXPECT_TENSOR_DATA(vecdup, lhs.GetSize(), npu::tile_fwk::bfloat16, npu::tile_fwk::bfloat16, npu::tile_fwk::bfloat16(2.0));

    std::vector<LogicalTensorData *> operList = {
        &lhs,
        &rhs,
        &lhs,
        &rhs,
    };
    auto reduceacc = LogicalTensorData::CreateEmpty(lhs.GetDataType(), lhs.GetShape(), std::vector<int64_t>(0));
    Calculator::CalcReduceAcc(reduceacc.get(), &operList, &pool);
    EXPECT_TENSOR_DATA(reduceacc, lhs.GetSize(), npu::tile_fwk::bfloat16, npu::tile_fwk::bfloat16, npu::tile_fwk::bfloat16(2.5 + 2 + 2.5 + 2));
}

TEST(ThreadPoolTest, ElementWiseFp16) {
    const int nproc = 2;
    ThreadPool pool(nproc);
    Tensor lhsTensor(DT_FP16, {64, 64}, "lhs");
    Tensor rhsTensor(DT_FP16, {64, 64}, "rhs");

    auto lhsData = RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(lhsTensor, npu::tile_fwk::float16(2.5));
    auto rhsData = RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(rhsTensor, npu::tile_fwk::float16(2.0));
    LogicalTensorData lhs(lhsData);
    LogicalTensorData rhs(rhsData);

    auto add = LogicalTensorData::CreateEmpty(lhs.GetDataType(), lhs.GetShape(), std::vector<int64_t>(0));
    auto sub = LogicalTensorData::CreateEmpty(lhs.GetDataType(), lhs.GetShape(), std::vector<int64_t>(0));
    auto mul = LogicalTensorData::CreateEmpty(lhs.GetDataType(), lhs.GetShape(), std::vector<int64_t>(0));
    auto div = LogicalTensorData::CreateEmpty(lhs.GetDataType(), lhs.GetShape(), std::vector<int64_t>(0));
    auto max = LogicalTensorData::CreateEmpty(lhs.GetDataType(), lhs.GetShape(), std::vector<int64_t>(0));
    auto min = LogicalTensorData::CreateEmpty(lhs.GetDataType(), lhs.GetShape(), std::vector<int64_t>(0));

    Calculator::CalcAdd(add.get(), &lhs, &rhs, &pool);
    Calculator::CalcSub(sub.get(), &lhs, &rhs, &pool);
    Calculator::CalcMul(mul.get(), &lhs, &rhs, &pool);
    Calculator::CalcDiv(div.get(), &lhs, &rhs, &pool);
    Calculator::CalcMax(max.get(), &lhs, &rhs, &pool);
    Calculator::CalcMax(min.get(), &lhs, &rhs, &pool);

    EXPECT_EQ(64 * 64, lhs.GetSize());

    EXPECT_TENSOR_DATA(add, lhs.GetSize(), npu::tile_fwk::float16, float, 2.5 + 2);
    EXPECT_TENSOR_DATA(sub, lhs.GetSize(), npu::tile_fwk::float16, float, 2.5 - 2);
    EXPECT_TENSOR_DATA(mul, lhs.GetSize(), npu::tile_fwk::float16, float, 2.5 * 2);
    EXPECT_TENSOR_DATA(div, lhs.GetSize(), npu::tile_fwk::float16, float, 2.5 / 2);
    EXPECT_TENSOR_DATA(max, lhs.GetSize(), npu::tile_fwk::float16, float, 2.5);
}

TEST(ThreadPoolTest, CastMode) {
    const int nproc = 2;
    ThreadPool pool(nproc);
    Tensor lhsTensor(DT_FP16, {1, 1}, "lhs");

    {
        LogicalTensorData lhs(RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(lhsTensor, npu::tile_fwk::float16(2.5)));
        auto cast_none = LogicalTensorData::CreateEmpty(DT_INT16, lhs.GetShape(), std::vector<int64_t>(0));
        Calculator::CalcCast(cast_none.get(), &lhs, DT_INT16, CastMode::CAST_NONE, &pool);
        EXPECT_TENSOR_DATA(cast_none, lhs.GetSize(), int16_t, int16_t, 2);
    }

    {
        LogicalTensorData lhs(RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(lhsTensor, npu::tile_fwk::float16(2.5)));
        auto cast_none = LogicalTensorData::CreateEmpty(DT_INT16, lhs.GetShape(), std::vector<int64_t>(0));
        Calculator::CalcCast(cast_none.get(), &lhs, DT_INT16, CastMode::CAST_RINT, &pool);
        EXPECT_TENSOR_DATA(cast_none, lhs.GetSize(), int16_t, int16_t, 2);
    }
    {
        LogicalTensorData lhs(RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(lhsTensor, npu::tile_fwk::float16(3.5)));
        auto cast_none = LogicalTensorData::CreateEmpty(DT_INT16, lhs.GetShape(), std::vector<int64_t>(0));
        Calculator::CalcCast(cast_none.get(), &lhs, DT_INT16, CastMode::CAST_RINT, &pool);
        EXPECT_TENSOR_DATA(cast_none, lhs.GetSize(), int16_t, int16_t, 4);
    }
    {
        LogicalTensorData lhs(RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(lhsTensor, npu::tile_fwk::float16(3.4)));
        auto cast_none = LogicalTensorData::CreateEmpty(DT_INT16, lhs.GetShape(), std::vector<int64_t>(0));
        Calculator::CalcCast(cast_none.get(), &lhs, DT_INT16, CastMode::CAST_RINT, &pool);
        EXPECT_TENSOR_DATA(cast_none, lhs.GetSize(), int16_t, int16_t, 3);
    }
    {
        LogicalTensorData lhs(RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(lhsTensor, npu::tile_fwk::float16(-3.4)));
        auto cast_none = LogicalTensorData::CreateEmpty(DT_INT16, lhs.GetShape(), std::vector<int64_t>(0));
        Calculator::CalcCast(cast_none.get(), &lhs, DT_INT16, CastMode::CAST_RINT, &pool);
        EXPECT_TENSOR_DATA(cast_none, lhs.GetSize(), int16_t, int16_t, -3);
    }

    {
        LogicalTensorData lhs(RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(lhsTensor, npu::tile_fwk::float16(2.5)));
        auto cast_none = LogicalTensorData::CreateEmpty(DT_INT16, lhs.GetShape(), std::vector<int64_t>(0));
        Calculator::CalcCast(cast_none.get(), &lhs, DT_INT16, CastMode::CAST_ROUND, &pool);
        EXPECT_TENSOR_DATA(cast_none, lhs.GetSize(), int16_t, int16_t, 3);
    }
    {
        LogicalTensorData lhs(RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(lhsTensor, npu::tile_fwk::float16(3.5)));
        auto cast_none = LogicalTensorData::CreateEmpty(DT_INT16, lhs.GetShape(), std::vector<int64_t>(0));
        Calculator::CalcCast(cast_none.get(), &lhs, DT_INT16, CastMode::CAST_ROUND, &pool);
        EXPECT_TENSOR_DATA(cast_none, lhs.GetSize(), int16_t, int16_t, 4);
    }
    {
        LogicalTensorData lhs(RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(lhsTensor, npu::tile_fwk::float16(-2.5)));
        auto cast_none = LogicalTensorData::CreateEmpty(DT_INT16, lhs.GetShape(), std::vector<int64_t>(0));
        Calculator::CalcCast(cast_none.get(), &lhs, DT_INT16, CastMode::CAST_ROUND, &pool);
        EXPECT_TENSOR_DATA(cast_none, lhs.GetSize(), int16_t, int16_t, -3);
    }
    {
        LogicalTensorData lhs(RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(lhsTensor, npu::tile_fwk::float16(-3.5)));
        auto cast_none = LogicalTensorData::CreateEmpty(DT_INT16, lhs.GetShape(), std::vector<int64_t>(0));
        Calculator::CalcCast(cast_none.get(), &lhs, DT_INT16, CastMode::CAST_ROUND, &pool);
        EXPECT_TENSOR_DATA(cast_none, lhs.GetSize(), int16_t, int16_t, -4);
    }
}

TEST(ThreadPoolTest, Reduce) {
    const int nproc = 2;
    ThreadPool pool(nproc);
    int n = 128;
    int m = 64;
    Tensor operTensor(DT_INT32, {n, m}, "oper");

    std::vector<int32_t> operDataValue(n * m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            operDataValue[i * m + j] = j;
        }
    }
    auto operData = RawTensorData::CreateTensor<int32_t>(operTensor, operDataValue);
    LogicalTensorData oper(operData);

    auto rowsum0Shape = oper.GetShape();
    rowsum0Shape[0] = 1;
    auto rowsum0 = LogicalTensorData::CreateEmpty(DT_INT32, rowsum0Shape, std::vector<int64_t>(0));
    Calculator::CalcRowSumLine(rowsum0.get(), &oper, 0, &pool);

    std::vector<int32_t> rowsum0DataValue(m);
    for (int i = 0; i < m; i++) {
        rowsum0DataValue[i] = i * n;
    }
    EXPECT_TENSOR_LISTDATA(rowsum0, m, int32_t, int32_t, rowsum0DataValue);

    auto rowsum1Shape = oper.GetShape();
    rowsum1Shape[1] = 1;
    auto rowsum1 = LogicalTensorData::CreateEmpty(DT_INT32, rowsum1Shape, std::vector<int64_t>(0));
    Calculator::CalcRowSumLine(rowsum1.get(), &oper, 1, &pool);

    std::vector<int32_t> rowsum1DataValue(n);
    for (int i = 0; i < n; i++) {
        rowsum1DataValue[i] = (m - 1) * m / 2;
    }
    EXPECT_TENSOR_LISTDATA(rowsum1, n, int32_t, int32_t, rowsum1DataValue);

    auto rowsum = LogicalTensorData::CreateEmpty(DT_INT32, rowsum1Shape, std::vector<int64_t>(0));
    Calculator::CalcRowSumSingle(rowsum.get(), &oper, rowsum1Shape.size() - 1, &pool);
    EXPECT_TENSOR_LISTDATA(rowsum, n, int32_t, int32_t, rowsum1DataValue);

    auto rowmax = LogicalTensorData::CreateEmpty(DT_INT32, rowsum1Shape, std::vector<int64_t>(0));
    Calculator::CalcRowMaxSingle(rowmax.get(), &oper, rowsum1Shape.size() - 1, &pool);
    std::vector<int32_t> rowmaxDataValue(n, m - 1);
    EXPECT_TENSOR_LISTDATA(rowmax, n, int32_t, int32_t, rowmaxDataValue);
}

TEST(ThreadPoolTest, RowSumSingle) {
    const int nproc = 1;
    ThreadPool pool(nproc);
    int n = 128;
    int m = 64;
    Tensor operTensor(DT_INT32, {n, m}, "oper");

    std::vector<int32_t> operDataValue(n * m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            operDataValue[i * m + j] = i;
        }
    }
    auto operData = RawTensorData::CreateTensor<int32_t>(operTensor, operDataValue);
    LogicalTensorData oper(operData);

    auto rowsum1Shape = oper.GetShape();
    rowsum1Shape[1] = 1;

    std::vector<int32_t> rowsum1DataValue(n);
    for (int i = 0; i < n; i++) {
        rowsum1DataValue[i] = i * m;
        ;
    }
    auto rowsum = LogicalTensorData::CreateEmpty(DT_INT32, rowsum1Shape, std::vector<int64_t>(0));
    Calculator::CalcRowSumSingle(rowsum.get(), &oper, rowsum1Shape.size() - 1, &pool);
    EXPECT_TENSOR_LISTDATA(rowsum, n, int32_t, int32_t, rowsum1DataValue);
}

TEST(ThreadPoolTest, Broadcast) {
    const int nproc = 2;
    ThreadPool pool(nproc);
    int b = 4;
    int n = 16;
    int m = 32;
    std::vector<int32_t> operDataValue(n * m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            operDataValue[i * m + j] = i * m + j;
        }
    }

    {
        Tensor oper0Tensor(DT_INT32, {1, n, m}, "oper");
        auto oper0Data = RawTensorData::CreateTensor<int32_t>(oper0Tensor, operDataValue);
        LogicalTensorData oper0(oper0Data);

        auto expand0Shape = std::vector<int64_t>({b, n, m});
        auto expand0 = LogicalTensorData::CreateEmpty(DT_INT32, expand0Shape, std::vector<int64_t>(0));
        Calculator::CalcExpand(expand0.get(), &oper0, 0, &pool);
        std::vector<int32_t> expand0DataValue(b * n * m);
        for (int k = 0; k < b; k++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    expand0DataValue[k * n * m + i * m + j] = i * m + j;
                }
            }
        }
        EXPECT_TENSOR_LISTDATA(expand0, b * n * m, int32_t, int32_t, expand0DataValue);
    }

    {
        Tensor oper1Tensor(DT_INT32, {n, 1, m}, "oper");
        auto oper1Data = RawTensorData::CreateTensor<int32_t>(oper1Tensor, operDataValue);
        LogicalTensorData oper1(oper1Data);

        auto expand1Shape = std::vector<int64_t>({n, b, m});
        auto expand1 = LogicalTensorData::CreateEmpty(DT_INT32, expand1Shape, std::vector<int64_t>(0));
        Calculator::CalcExpand(expand1.get(), &oper1, 1, &pool);
        std::vector<int32_t> expand1DataValue(n * b * m);
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < b; k++) {
                for (int j = 0; j < m; j++) {
                    expand1DataValue[i * b * m + k * m + j] = i * m + j;
                }
            }
        }
        EXPECT_TENSOR_LISTDATA(expand1, n * b * m, int32_t, int32_t, expand1DataValue);
    }

    {
        Tensor oper2Tensor(DT_INT32, {n, m, 1}, "oper");
        auto oper2Data = RawTensorData::CreateTensor<int32_t>(oper2Tensor, operDataValue);
        LogicalTensorData oper2(oper2Data);

        auto expand2Shape = std::vector<int64_t>({n, m, b});
        auto expand2 = LogicalTensorData::CreateEmpty(DT_INT32, expand2Shape, std::vector<int64_t>(0));
        Calculator::CalcExpand(expand2.get(), &oper2, 2, &pool);
        std::vector<int32_t> expand2DataValue(n * m * b);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                for (int k = 0; k < b; k++) {
                    expand2DataValue[i * m * b + j * b + k] = i * m + j;
                }
            }
        }
        EXPECT_TENSOR_LISTDATA(expand2, n * m * b, int32_t, int32_t, expand2DataValue);
    }
}

TEST(ThreadPoolTest, IndexOutcast) {
    const int nproc = 1;
    ThreadPool pool(nproc);
    int s = 2;
    int b = 2;
    int n = 4;
    int m = 4;
    std::vector<int32_t> srcDataValue = {
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
    };
    Tensor srcTensor(DT_INT32, {s, b, 1, m}, "src");
    auto srcData = RawTensorData::CreateTensor<int32_t>(srcTensor, srcDataValue);
    LogicalTensorData src(srcData);

    std::vector<int32_t> indexDataValue = {
        0,
        1,
        2,
        3,
    };
    Tensor indexTensor(DT_INT32, {s, b}, "index");
    auto indexData = RawTensorData::CreateTensor<int32_t>(indexTensor, indexDataValue);
    LogicalTensorData index(indexData);

    std::vector<int32_t> dstDataValue(s * b * n * m, 20);
    Tensor dstTensor(DT_INT32, {s, b, n, m}, "dst");
    auto dstData = RawTensorData::CreateTensor<int32_t>(dstTensor, dstDataValue);
    LogicalTensorData dst(dstData);

    {
        auto indexOutcast0Shape = std::vector<int64_t>({s, b, n, m});
        auto indexOutcast0 = LogicalTensorData::CreateEmpty(DT_INT32, indexOutcast0Shape, std::vector<int64_t>(0));
        Calculator::CalcIndexCopy(indexOutcast0.get(), &src, &index, &dst, 2, &pool);
        std::vector<int32_t> goldenDataValue = {
            0,
            1,
            2,
            3,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,

            20,
            20,
            20,
            20,
            4,
            5,
            6,
            7,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,

            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            8,
            9,
            10,
            11,
            20,
            20,
            20,
            20,

            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            12,
            13,
            14,
            15,
        };
        EXPECT_TENSOR_LISTDATA(indexOutcast0, s * b * n * m, int32_t, int32_t, goldenDataValue);
    }
}

TEST(ThreadPoolTest, Transpose) {
    const int nproc = 1;
    ThreadPool pool(nproc);

    int b = 2;
    int n = 3;
    int m = 4;
    std::vector<int32_t> operDataValue = {
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,

        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
    };

    Tensor operTensor(DT_INT32, {b, n, m}, "oper");
    auto operData = RawTensorData::CreateTensor<int32_t>(operTensor, operDataValue);
    LogicalTensorData oper(operData);

    {
        auto transpose0Shape = std::vector<int64_t>({b, m, n});
        auto transpose0 = LogicalTensorData::CreateEmpty(DT_INT32, transpose0Shape, std::vector<int64_t>(0));
        Calculator::CalcTransposeAdjDim(transpose0.get(), &oper, 1, &pool);
        std::vector<int32_t> transpose0DataValue = {
            0,
            4,
            8,
            1,
            5,
            9,
            2,
            6,
            10,
            3,
            7,
            11,

            12,
            16,
            20,
            13,
            17,
            21,
            14,
            18,
            22,
            15,
            19,
            23,
        };
        EXPECT_TENSOR_LISTDATA(transpose0, b * n * m, int32_t, int32_t, transpose0DataValue);
    }
    {
        auto transpose1Shape = std::vector<int64_t>({n, b, m});
        auto transpose1 = LogicalTensorData::CreateEmpty(DT_INT32, transpose1Shape, std::vector<int64_t>(0));
        Calculator::CalcTransposeAdjDim(transpose1.get(), &oper, 0, &pool);
        std::vector<int32_t> transpose1DataValue = {
            0,
            1,
            2,
            3,
            12,
            13,
            14,
            15,

            4,
            5,
            6,
            7,
            16,
            17,
            18,
            19,

            8,
            9,
            10,
            11,
            20,
            21,
            22,
            23,
        };
        EXPECT_TENSOR_LISTDATA(transpose1, b * n * m, int32_t, int32_t, transpose1DataValue);
    }
}

TEST(ThreadPoolTest, MatMul) {
    const int nproc = 2;
    ThreadPool pool(nproc);

    int n = 64;
    int k = 128;
    int m = 256;

    Tensor lhsTensor(DT_INT32, {n, k}, "lhs");
    Tensor rhsTensor(DT_INT32, {k, m}, "rhs");
    Tensor rhsTransTensor(DT_INT32, {m, k}, "rhsTrans");
    Tensor accTensor(DT_INT32, {n, m}, "acc");

    std::vector<int32_t> lhsDataValue(n * k);
    std::vector<int32_t> rhsDataValue(k * m);
    std::vector<int32_t> rhsTransDataValue(m * k);
    std::vector<int32_t> accDataValue(n * m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            lhsDataValue[i * k + j] = j;
        }
    }
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < m; j++) {
            rhsDataValue[i * m + j] = j;
        }
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            rhsTransDataValue[i * k + j] = i;
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            accDataValue[i * m + j] = i + j;
        }
    }
    auto lhsData = RawTensorData::CreateTensor<int32_t>(lhsTensor, lhsDataValue);
    auto rhsData = RawTensorData::CreateTensor<int32_t>(rhsTensor, rhsDataValue);
    auto rhsTransData = RawTensorData::CreateTensor<int32_t>(rhsTransTensor, rhsTransDataValue);
    auto accData = RawTensorData::CreateTensor<int32_t>(accTensor, accDataValue);

    LogicalTensorData lhs(lhsData);
    LogicalTensorData rhs(rhsData);
    LogicalTensorData rhsTrans(rhsTransData);
    LogicalTensorData acc(accData);

    std::vector<int64_t> matmulShape = {n, m};
    auto matmul = LogicalTensorData::CreateEmpty(DT_INT32, matmulShape, std::vector<int64_t>(0));
    auto matmulAcc = LogicalTensorData::CreateEmpty(DT_INT32, matmulShape, std::vector<int64_t>(0));
    auto matmulTrans = LogicalTensorData::CreateEmpty(DT_INT32, matmulShape, std::vector<int64_t>(0));
    auto matmulTransAcc = LogicalTensorData::CreateEmpty(DT_INT32, matmulShape, std::vector<int64_t>(0));

    Calculator::CalcMatMul(matmul.get(), &lhs, &rhs, k / 2, &pool);
    Calculator::CalcMatMulTrans(matmulTrans.get(), &lhs, &rhsTrans, k / 4, &pool);
    Calculator::CalcMatMulAcc(matmulAcc.get(), &lhs, &rhs, k / 8, &acc, &pool);
    Calculator::CalcMatMulTransAcc(matmulTransAcc.get(), &lhs, &rhsTrans, k / 16, &acc, &pool);

    std::vector<int32_t> matmulDataValue(n * m);
    std::vector<int32_t> matmulAccDataValue(n * m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            matmulDataValue[i * m + j] = (k - 1) * k / 2 * j;
            matmulAccDataValue[i * m + j] = (k - 1) * k / 2 * j + i + j;
        }
    }
    EXPECT_TENSOR_LISTDATA(matmul, n * m, int32_t, int32_t, matmulDataValue);
    EXPECT_TENSOR_LISTDATA(matmulTrans, n * m, int32_t, int32_t, matmulDataValue);

    EXPECT_TENSOR_LISTDATA(matmulAcc, n * m, int32_t, int32_t, matmulAccDataValue);
    EXPECT_TENSOR_LISTDATA(matmulTransAcc, n * m, int32_t, int32_t, matmulAccDataValue);
}

TEST(ThreadPoolTest, Float16) {
    static_assert(npu::tile_fwk::float16::BaseFromFp32(npu::tile_fwk::float16::BaseToFp32(0x03ff)) == 0x03ff, "");
    const float epsilon = 1e-6f;

    for (uint32_t v = 0; v <= 0xffff; v++) {
        uint32_t v32 = npu::tile_fwk::float16::BaseToFp32(v);
        uint16_t v2 = npu::tile_fwk::float16::BaseFromFp32(v32);
        EXPECT_EQ(v, v2);
    }
    std::vector<float> vs;
    for (int16_t v = 0x7c00 - 1; v >= 0; v--) {
        vs.push_back(npu::tile_fwk::float16::FromBase(v | 0x8000));
    }
    for (int16_t v = 0; v < 0x7c00; v++) {
        vs.push_back(npu::tile_fwk::float16::FromBase(v));
    }
    float subnormal = 0;
    for (int frac = 0; frac < (1 << 10); frac++) {
        subnormal = (float)(1.0 / (1 << 14) / (1 << 10) * frac);
        EXPECT_EQ(subnormal, static_cast<float>(npu::tile_fwk::float16(subnormal)));
        subnormal = (float)(-1.0 / (1 << 14) / (1 << 10) * frac);
        EXPECT_EQ(subnormal, static_cast<float>(npu::tile_fwk::float16(subnormal)));
    }
    float normal = 0;
    for (int exp = 1; exp < 31; exp++) {
        for (int frac = 0; frac < (1 << 10); frac++) {
            normal = static_cast<float>(1.0 * pow(2, exp - 15) * (1 + 1.0 / (1 << 10) * frac));
            EXPECT_EQ(normal, static_cast<float>(npu::tile_fwk::float16(normal)));
            normal = static_cast<float>(-1.0 * pow(2, exp - 15) * (1 + 1.0 / (1 << 10) * frac));
            EXPECT_EQ(normal, static_cast<float>(npu::tile_fwk::float16(normal)));
        }
    }
    float min = 1.0 / (1 << 14) / (1 << 10);
    float max = 1.0 * pow(2, 30 - 15) * ((2 << 10) - 1.0) / (1 << 10);
    EXPECT_TRUE(std::abs(0 - static_cast<float>(npu::tile_fwk::float16(min * static_cast<float>(0.99)))) < epsilon);
    EXPECT_TRUE(isinf(static_cast<float>(npu::tile_fwk::float16(max * static_cast<float>(1.01)))));
}

TEST(ThreadPoolTest, Bfloat16) {
    static_assert(npu::tile_fwk::bfloat16::BaseFromFp32(npu::tile_fwk::bfloat16::BaseToFp32(0x03ff)) == 0x03ff, "");

    // 最接近偶数舍入法校验
    uint32_t u32 = 0x3ed2f1aa; // 0xf1aa > 0x8000   +1
    uint16_t u16 = npu::tile_fwk::bfloat16::BaseFromFp32(u32); // 0x3ed3
    EXPECT_EQ(u16, 0x3ed3);
    // 0x8 是十六进制0xF的 半拉值
    u32 = static_cast<uint32_t>(0x3ed28000); // 0x8000 = 0x8000  最接近偶数舍入法,舍入到最近的偶数(取决于保留部分的最低有效位+0)
    u16 = npu::tile_fwk::bfloat16::BaseFromFp32(u32); // 0x3ed2
    EXPECT_EQ(u16, 0x3ed2);

    u32 = static_cast<uint32_t>(0x3ed38000); // 0x8000 = 0x8000  最接近偶数舍入法,舍入到最近的偶数(取决于保留部分的最低有效位+1)
    u16 = npu::tile_fwk::bfloat16::BaseFromFp32(u32); // 0x3ed4
    EXPECT_EQ(u16, 0x3ed4);

    u32 = static_cast<uint32_t>(0x3ed20002); // 0x0002 < 0x8000  +0
    u16 = npu::tile_fwk::bfloat16::BaseFromFp32(u32); // 0x3ed2
    EXPECT_EQ(u16, 0x3ed2);

    u32 = static_cast<uint32_t>(0x3ed30002); // 0x0002 < 0x8000  +0
    u16 = npu::tile_fwk::bfloat16::BaseFromFp32(u32); // 0x3ed3
    EXPECT_EQ(u16, 0x3ed3);

    const float epsilon = 1e-41f;
    for (uint32_t v = 0x0; v <= static_cast<uint32_t>(0xffff); v++) {
        uint32_t v32 = npu::tile_fwk::bfloat16::BaseToFp32(v);
        uint16_t v2 = npu::tile_fwk::bfloat16::BaseFromFp32(v32);
        EXPECT_EQ(v, v2);
    }

    float subnormal = 0;
    for (int frac = 0; frac < (1 << 7); frac++) {
        subnormal = static_cast<float>(1.0 / pow(2.0, static_cast<double>(126 + 7)) * frac);
        EXPECT_EQ(subnormal, static_cast<float>(npu::tile_fwk::bfloat16(subnormal)));
        subnormal = static_cast<float>(-1.0 / pow(2.0, static_cast<double>(126 + 7)) * frac);
        EXPECT_EQ(subnormal, static_cast<float>(npu::tile_fwk::bfloat16(subnormal)));
    }

    float normal = 0;
    for (int exp = 1; exp < 255; exp++) {
        for (int frac = 0; frac < (1 << 7); frac++) {
            normal = static_cast<float>(1.0 * static_cast<float>(pow(2.0, static_cast<double>(exp - 127))) *
                (1.0 + 1.0 / static_cast<float>(1 << 7) * static_cast<float>(frac)));
            EXPECT_EQ(normal, static_cast<float>(npu::tile_fwk::bfloat16(normal)));
            normal = static_cast<float>(-1.0 * static_cast<float>(pow(2.0, static_cast<double>(exp - 127))) *
                (1.0 + 1.0 / static_cast<float>(1 << 7) * static_cast<float>(frac)));
            EXPECT_EQ(normal, static_cast<float>(npu::tile_fwk::bfloat16(normal)));
        }
    }

    float min = static_cast<float>(1.0 / pow(2.0, static_cast<double>(133)));
    float max = static_cast<float>(1.0 * pow(2.0, static_cast<double>(254 - 127))) *
        (static_cast<float>(2 << 7) - 1.0f) /
        static_cast<float>(1 << 7);
    float eps = std::abs(0 - static_cast<float>(npu::tile_fwk::bfloat16(min * static_cast<float>(0.50))));
    EXPECT_TRUE(eps < epsilon);
    EXPECT_TRUE(isinf(static_cast<float>(npu::tile_fwk::bfloat16(max * static_cast<float>(1.01)))));
}
} // namespace
