/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_gather_in_l1.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <random>
#include "interface/tensor/float.h"
#include "tilefwk/data_type.h"
#include "tilefwk/symbolic_scalar.h"
#include "interface/program/program.h"
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class GatherInUBTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {
    void SetUp() override {
        TestSuite_STest_Ops_Aihac::SetUp();
        config::SetHostOption(ONLY_CODEGEN, true);
        config::SetCodeGenOption(SUPPORT_DYNAMIC_UNALIGNED, true);
        rtSetDevice(GetDeviceIdByEnvVar());
    }
    void TearDown() override {
        config::SetHostOption(ONLY_CODEGEN, false);
        config::SetCodeGenOption(SUPPORT_DYNAMIC_UNALIGNED, false);
        TestSuite_STest_Ops_Aihac::TearDown();
    }
};

void BasicGatherTest(int64_t SRC0, int64_t SRC1, int64_t DST0, int64_t DST1) {
    random_device seed;
    ranlux48 engine(seed());
    uniform_real_distribution<float> floatRandom(-100, 100);

    ASSERT(SRC1 == DST1);

    auto TotalSize = [](const Shape &shapes) {
        size_t res = 1;
        for (auto v : shapes) {
            res *= v;
        }
        return res;
    };

    auto Simu = [&](const std::vector<float16> &src, const std::vector<int32_t> &offsets, int64_t copyLen) {
        ASSERT(offsets.size() == static_cast<size_t>(DST0));
        ASSERT(copyLen <= DST1);
        std::vector<float16> a(DST0 * DST1);
        for (int64_t i = 0; i < DST0; i++) {
            ASSERT(static_cast<size_t>(offsets[i] * SRC1 + copyLen) <= src.size());
            for (int64_t j = 0; j < copyLen; j++) {
                a[i * DST1 + j] = src[offsets[i] * SRC1 + j];
            }
        }
        return a;
    };

    Shape srcShapes{SRC0, SRC1};
    Shape dstShapes{DST0, DST1};
    Shape offsetsShapes{1, DST0};

    Tensor src(DT_FP16, srcShapes, "src");
    Tensor offsets(DT_INT32, offsetsShapes, "offsets");
    Tensor dst(DT_FP16, dstShapes, "dst");
    std::vector<float16> srcData(TotalSize(srcShapes));
    for (auto &v : srcData) {
        v = floatRandom(engine);
    }
    uniform_int_distribution<int32_t> intRandom(0, SRC0 - 1);
    std::vector<int32_t> offsetsData(TotalSize(offsetsShapes));
    for (auto &v : offsetsData) {
        v = intRandom(engine);
    }
    auto golden = Simu(srcData, offsetsData, DST1);
    std::cout << "simu finished" << std::endl;

    FUNCTION("test", {src, offsets}, {dst}) {
        LOOP("LOOP", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, 1, 1)) {
            (void)sIdx;
            TileShape::Current().SetVecTile({32, 64});

            std::vector<SymbolicScalar> srcValidShape = {src.GetShape()[0], src.GetShape()[1]};
            Tensor dynSrc = View(src, src.GetShape(), srcValidShape, {0, 0});

            std::vector<SymbolicScalar> offsetsValidShape = {offsets.GetShape()[0], offsets.GetShape()[1]};
            Tensor dynOffsets = View(offsets, offsets.GetShape(), offsetsValidShape, {0, 0});

            dst = internal::GatherInUB(dynSrc, dynOffsets, DST1);
        }
    }
    std::cout << "compile finished" << std::endl;

    ProgramData::GetInstance().AppendInputs({RawTensorData::CreateTensor<float16>(src, srcData),
        RawTensorData::CreateTensor<int32_t>(offsets, offsetsData)});
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float16>(dst, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto out = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    int maxErrorPrintNum = 50;
    int curErrorPrintNum = 0;
    float eps = 1e-6f;
    for (size_t i = 0; i < golden.size(); i++) {
        auto actual = ((float16 *)out->data())[i];
        auto expect = golden[i];
        if (fabs(actual - expect) > eps && curErrorPrintNum < maxErrorPrintNum) {
            std::cout << i << ": output: " << actual << "; expect: " << expect << std::endl;
            curErrorPrintNum++;
        }
    }
    EXPECT_TRUE(resultCmp(golden, (float16 *)out->data(), eps));
}

TEST_F(GatherInUBTest, gather_in_a_) {
    BasicGatherTest(32, 64, 5, 64);
}
TEST_F(GatherInUBTest, gather_in_a) {
    BasicGatherTest(2048, 256, 512, 256);
}

TEST_F(GatherInUBTest, gather_in_a_unalign) {
    BasicGatherTest(2048, 255, 512, 255);
}

TEST_F(GatherInUBTest, gather_in_a_with_valid_shape) {
    random_device seed;
    ranlux48 engine(seed());
    uniform_real_distribution<float> floatRandom(-100, 100);

    constexpr int64_t SRC0 = 2048;
    constexpr int64_t SRC1 = 128;
    constexpr int64_t DST0 = 512;
    constexpr int64_t DST1 = 128;
    static_assert(SRC1 == DST1);

    auto TotalSize = [](const Shape &shapes) {
        size_t res = 1;
        for (auto v : shapes) {
            res *= v;
        }
        return res;
    };

    auto Simu = [&](const std::vector<float16> &src, const std::vector<int32_t> &offsets,
                    const std::vector<int32_t> &info, int64_t copyLen) {
        ASSERT(info.size() == 2);
        ASSERT(copyLen <= DST1);
        int len = info[0];
        int start = info[1];
        ASSERT(len + start <= DST0);
        std::vector<float16> a(DST0 * DST1);
        for (int64_t i = 0; i < len; i++) {
            ASSERT(static_cast<size_t>(offsets[i] * SRC1 + copyLen) <= src.size());
            for (int64_t j = 0; j < copyLen; j++) {
                a[i * DST1 + j] = src[offsets[i + start] * SRC1 + j];
            }
        }
        return a;
    };

    Shape srcShapes{SRC0, SRC1};
    Shape dstShapes{DST0, DST1};
    Shape offsetsShapes{1, DST0};

    Tensor src(DT_FP16, srcShapes, "src");
    Tensor offsets(DT_INT32, offsetsShapes, "offsets");
    Tensor dst(DT_FP16, dstShapes, "dst");
    Tensor info(DT_INT32, {1, 2}, "info");

    std::vector<float16> srcData(TotalSize(srcShapes));
    for (auto &v : srcData) {
        v = floatRandom(engine);
    }

    uniform_int_distribution<int32_t> intRandom(0, SRC0 - 1);
    std::vector<int32_t> offsetsData(TotalSize(offsetsShapes));
    for (auto &v : offsetsData) {
        v = intRandom(engine);
    }
    std::vector<int32_t> infoData{177, 234};

    auto golden = Simu(srcData, offsetsData, infoData, DST1);
    std::cout << "simu finished" << std::endl;

    FUNCTION("test", {src, offsets,  info}, {dst}) {
        LOOP("LOOP", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, 1, 1)) {
            (void)sIdx;
            auto len = GetTensorData(info, {0, 0});
            auto start = GetTensorData(info, {0, 1});
            TileShape::Current().SetVecTile({32, 64});

            Tensor subOffsets = View(offsets, offsets.GetShape(), {1, len}, {0, start});
            dst = internal::GatherInUB(src, subOffsets, -2);
        }
    }
    std::cout << "compile finished" << std::endl;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float16>(src, srcData),
        RawTensorData::CreateTensor<int32_t>(offsets, offsetsData),
        RawTensorData::CreateTensor<int32_t>(info, infoData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float16>(dst, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto out = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    int maxErrorPrintNum = 200;
    int curErrorPrintNum = 0;
    float eps = 1e-6f;
    for (size_t i = 0; i < golden.size(); i++) {
        auto actual = ((float16 *)out->data())[i];
        auto expect = golden[i];
        if (fabs(actual - expect) > eps && curErrorPrintNum < maxErrorPrintNum) {
            std::cout << i << ": output: " << actual << "; expect: " << expect << std::endl;
            curErrorPrintNum++;
        }
    }
    EXPECT_TRUE(resultCmp(golden, (float16 *)out->data(), eps));
}
