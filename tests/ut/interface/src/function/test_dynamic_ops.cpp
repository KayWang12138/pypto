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
 * \file test_dynamic_ops.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "interface/configs/config_manager.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/interpreter/calc.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"

using namespace npu::tile_fwk;

class DynamicOpsTest : public testing::Test {
public:
    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        ProgramData::GetInstance().Reset();
        config::SetHostOption(ONLY_CODEGEN, true);
        if (strcmp(calc::Model(), "torch")) {
            GTEST_SKIP() << "torch missing skip the verify test";
        }
        TileShape::Current().SetVecTile(32, 32);
        TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    }

    void TearDown() override {
        config::SetPlatformConfig(KEY_EXTRACT_TENSOR_GRAPH_THEN_COMPILE, false);
        config::SetPlatformConfig(KEY_VERIFY_TENSOR_GRAPH, false);
        config::SetPlatformConfig(KEY_VERIFY_TENSOR_GRAPH_DUMP_OPERATION, false);
        config::SetPlatformConfig(KEY_VERIFY_TENSOR_GRAPH_DUMP_TENSOR, false);
        config::SetPlatformConfig(KEY_VERIFY_TENSOR_GRAPH_CHECK_PRECISION, false);
        config::SetPlatformConfig(KEY_VERIFY_PASS, false);
        config::SetPlatformConfig(KEY_VERIFY_PASS_DUMP_OPERATION, false);
        config::SetPlatformConfig(KEY_VERIFY_PASS_DUMP_TENSOR, false);
        config::SetPlatformConfig(KEY_VERIFY_EXECUTE_GRAPH, false);
        config::SetPlatformConfig(KEY_VERIFY_EXECUTE_GRAPH_DUMP_OPERATION, false);
        config::SetPlatformConfig(KEY_VERIFY_EXECUTE_GRAPH_DUMP_TENSOR, false);
        config::SetPlatformConfig(KEY_VERIFY_EXECUTE_GRAPH_CHECK_PRECISION, false);
    }
};

TEST_F(DynamicOpsTest, Assemble) {
    config::SetPlatformConfig(KEY_EXTRACT_TENSOR_GRAPH_THEN_COMPILE, true);
    config::SetPlatformConfig(KEY_VERIFY_TENSOR_GRAPH, true);
    config::SetPlatformConfig(KEY_VERIFY_PASS, true);
    config::SetPlatformConfig(KEY_VERIFY_EXECUTE_GRAPH, true);

    int s = 32;
    int n = 2;
    int m = 1;
    Tensor t0(DT_FP32, {n * s, m * s}, "t0");
    Tensor t1(DT_FP32, {n * s, m * s}, "t1");
    Tensor out(DT_FP32, {n * s, m * s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0),
        RawTensorData::CreateConstantTensor<float>(t1, 2.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<float>(out, 3.0),
    });

    FUNCTION("main", {t0, t1}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t0a = View(t0, {s, s}, {0, 0});
            auto t0b = View(t0, {s, s}, {s, 0});
            auto t1a = View(t1, {s, s}, {0, 0});
            auto t1b = View(t1, {s, s}, {s, 0});
            auto t2a = Add(t0a, t1a);
            auto t2b = Add(t0b, t1b);
            std::vector<std::pair<Tensor, std::vector<int64_t>>> data = {
                {t2a, {0, 0}},
                {t2b, {s, 0}},
            };
            out = Assemble(data);
        }
    }
}

TEST_F(DynamicOpsTest, AssembleFp16) {
    config::SetPlatformConfig(KEY_EXTRACT_TENSOR_GRAPH_THEN_COMPILE, true);
    config::SetPlatformConfig(KEY_VERIFY_TENSOR_GRAPH, true);
    config::SetPlatformConfig(KEY_VERIFY_PASS, true);
    config::SetPlatformConfig(KEY_VERIFY_EXECUTE_GRAPH, true);

    int s = 32;
    int n = 2;
    int m = 1;
    Tensor t0(DT_FP16, {n * s, m * s}, "t0");
    Tensor t1(DT_FP16, {n * s, m * s}, "t1");
    Tensor out(DT_FP16, {n * s, m * s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(t0, 1.0),
        RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(t1, 2.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(out, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(out, 3.0),
    });

    FUNCTION("main", {t0, t1}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(3)) {
            auto t0a = View(t0, {s, s}, {0, 0});
            auto t0b = View(t0, {s, s}, {s, 0});
            auto t1a = View(t1, {s, s}, {0, 0});
            auto t1b = View(t1, {s, s}, {s, 0});
            auto t2a = Add(t0a, t1a);
            auto t2b = Add(t0b, t1b);
            ToFile(t2b, "t2b_%d.bin", {i});
            PrintIf(i == 1,"t2b=", t2b);
            std::vector<std::pair<Tensor, std::vector<int64_t>>> data = {
                {t2a, {0, 0}},
                {t2b, {s, 0}},
            };
            out = Assemble(data);
        }
    }
}

TEST_F(DynamicOpsTest, OpsElementWise) {
    config::SetPlatformConfig(KEY_EXTRACT_TENSOR_GRAPH_THEN_COMPILE, true);
    config::SetPlatformConfig(KEY_VERIFY_TENSOR_GRAPH, true);
    config::SetPlatformConfig(KEY_VERIFY_TENSOR_GRAPH_DUMP_OPERATION, true);
    config::SetPlatformConfig(KEY_VERIFY_TENSOR_GRAPH_DUMP_TENSOR, true);
    config::SetPlatformConfig(KEY_VERIFY_TENSOR_GRAPH_CHECK_PRECISION, true);
    config::SetPlatformConfig(KEY_VERIFY_PASS, true);
    config::SetPlatformConfig(KEY_VERIFY_PASS_DUMP_OPERATION, true);
    config::SetPlatformConfig(KEY_VERIFY_PASS_DUMP_TENSOR, true);
    config::SetPlatformConfig(KEY_VERIFY_EXECUTE_GRAPH, true);
    config::SetPlatformConfig(KEY_VERIFY_EXECUTE_GRAPH_DUMP_OPERATION, true);
    config::SetPlatformConfig(KEY_VERIFY_EXECUTE_GRAPH_DUMP_TENSOR, true);

    std::vector<uint8_t> devProgBinary;

    int s = 32;
    int n = 2;
    int m = 2;
    Tensor t0(DT_FP32, {n * s, m * s}, "t0");
    Tensor t1(DT_FP32, {n * s, m * s}, "t1");
    Tensor t2(DT_FP32, {n * s, m * s}, "t2");
    Tensor t3(DT_FP32, {n * s, m * s}, "t3");
    Tensor t4(DT_FP32, {n * s, m * s}, "t4");
    Tensor t5(DT_INT32, {n, s, m * s}, "t5");
    Tensor out(DT_FP32, {n * s, m * s}, "out");

    float t0Data = 10.0;
    float t1Data = 20.0;
    float t2Data = 30.0;
    float t3Data = 1.0;
    float t4Data = 50.0;

    std::vector<int> t5Data(n * s * m * s, 1);
    t5Data[n * s * m * s - 1] = 3; // 3: loop cnt

    float r0Data = 0;
    int loopCount = 3;
    int condThreshold = 2;

    for (int i = 0; i < loopCount; i++) {
        if (i == 0) {
            r0Data = t0Data + t1Data;
        }  else {
            r0Data = r0Data + t1Data; // +t0, +t1
            if (i < condThreshold) {
                r0Data = r0Data + t2Data; // +t2 * 5
            } else {
                r0Data = r0Data * t3Data; // +t3 * 2
            }
            r0Data = r0Data + t4Data;
        }
    }
    for (int i = 0; i < loopCount; i++) {
        r0Data = r0Data + t1Data; // +t1
        if (i < condThreshold) {
            r0Data = r0Data + t2Data; // +t2 * 5
        } else {
            r0Data = r0Data * t3Data; // +t3 * 2
        }
        r0Data = r0Data + t4Data;
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, t0Data),
        RawTensorData::CreateConstantTensor<float>(t1, t1Data),
        RawTensorData::CreateConstantTensor<float>(t2, t2Data),
        RawTensorData::CreateConstantTensor<float>(t3, t3Data),
        RawTensorData::CreateConstantTensor<float>(t4, t4Data),
        RawTensorData::CreateTensor<int>(t5, t5Data),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<float>(out, r0Data),
    });

    FUNCTION("main", {t0, t1, t2, t3, t4, t5}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(npu::tile_fwk::GetTensorData(t5, {n - 1, s - 1, m * s - 1}))) {
            IF (i == 0) {
                out = Add(t0, t1); // +t0, +t1
            } ELSE {
                out = Add(out, t1); // +t1 * 7
                IF(i < condThreshold) {
                    out = Add(out, t2); // +t2 * 5
                }
                ELSE {
                    out = Mul(out, t3); // +t3 * 2
                }
                out = Add(out, t4);
            }
        }
        LOOP("L1", FunctionType::DYNAMIC_LOOP, i, LoopRange(loopCount)) {
            out = Add(out, t1); // +t1
            IF(i < condThreshold) {
                out = Add(out, t2); // +t2 * 5
            }
            ELSE {
                out = Mul(out, t3); // +t3 * 2
            }
            out = Add(out, t4);
        }
    }
}

TEST_F(DynamicOpsTest, OpsElementWiseFp16) {
    config::SetPlatformConfig(KEY_EXTRACT_TENSOR_GRAPH_THEN_COMPILE, true);
    config::SetPlatformConfig(KEY_VERIFY_TENSOR_GRAPH, true);
    config::SetPlatformConfig(KEY_VERIFY_TENSOR_GRAPH_DUMP_OPERATION, true);
    config::SetPlatformConfig(KEY_VERIFY_TENSOR_GRAPH_DUMP_TENSOR, true);
    config::SetPlatformConfig(KEY_VERIFY_TENSOR_GRAPH_CHECK_PRECISION, true);
    config::SetPlatformConfig(KEY_VERIFY_PASS, true);
    config::SetPlatformConfig(KEY_VERIFY_PASS_DUMP_OPERATION, true);
    config::SetPlatformConfig(KEY_VERIFY_PASS_DUMP_TENSOR, true);
    config::SetPlatformConfig(KEY_VERIFY_EXECUTE_GRAPH, true);
    config::SetPlatformConfig(KEY_VERIFY_EXECUTE_GRAPH_DUMP_OPERATION, true);
    config::SetPlatformConfig(KEY_VERIFY_EXECUTE_GRAPH_DUMP_TENSOR, true);
    config::SetPlatformConfig(KEY_VERIFY_EXECUTE_GRAPH_CHECK_PRECISION, true);

    std::vector<uint8_t> devProgBinary;

    int s = 32;
    int n = 2;
    int m = 2;
    Tensor t0(DT_FP16, {n * s, m * s}, "t0");
    Tensor t1(DT_FP16, {n * s, m * s}, "t1");
    Tensor t2(DT_FP16, {n * s, m * s}, "t2");
    Tensor t3(DT_FP16, {n * s, m * s}, "t3");
    Tensor t4(DT_FP16, {n * s, m * s}, "t4");
    Tensor out(DT_FP16, {n * s, m * s}, "out");

    npu::tile_fwk::float16 t0Data = 10.0;
    npu::tile_fwk::float16 t1Data = 20.0;
    npu::tile_fwk::float16 t2Data = 30.0;
    npu::tile_fwk::float16 t3Data = 1.0;
    npu::tile_fwk::float16 t4Data = 50.0;

    npu::tile_fwk::float16 r0Data = 0;
    int loopCount = 3;
    int condThreshold = 2;

    for (int i = 0; i < loopCount; i++) {
        if (i == 0) {
            r0Data = t0Data + t1Data;
        }  else {
            r0Data = r0Data + t1Data; // +t0, +t1
            if (i < condThreshold) {
                r0Data = r0Data + t2Data; // +t2 * 5
            } else {
                r0Data = r0Data * t3Data; // +t3 * 2
            }
            r0Data = r0Data + t4Data;
        }
    }
    for (int i = 0; i < loopCount; i++) {
        r0Data = r0Data + t1Data; // +t1
        if (i < condThreshold) {
            r0Data = r0Data + t2Data; // +t2 * 5
        } else {
            r0Data = r0Data * t3Data; // +t3 * 2
        }
        r0Data = r0Data + t4Data;
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(t0, t0Data),
        RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(t1, t1Data),
        RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(t2, t2Data),
        RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(t3, t3Data),
        RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(t4, t4Data),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(out, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(out, r0Data),
    });

    FUNCTION("main", {t0, t1, t2, t3, t4}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(loopCount)) {
            IF (i == 0) {
                out = Add(t0, t1); // +t0, +t1
            } ELSE {
                out = Add(out, t1); // +t1 * 7
                IF(i < condThreshold) {
                    out = Add(out, t2); // +t2 * 5
                }
                ELSE {
                    out = Mul(out, t3); // +t3 * 2
                }
                out = Add(out, t4);
            }
        }
        LOOP("L1", FunctionType::DYNAMIC_LOOP, i, LoopRange(loopCount)) {
            out = Add(out, t1); // +t1
            IF(i < condThreshold) {
                out = Add(out, t2); // +t2 * 5
            }
            ELSE {
                out = Mul(out, t3); // +t3 * 2
            }
            out = Add(out, t4);
        }
    }
}

TEST_F(DynamicOpsTest, Cube) {
    config::SetPlatformConfig(KEY_EXTRACT_TENSOR_GRAPH_THEN_COMPILE, true);
    config::SetPlatformConfig(KEY_VERIFY_TENSOR_GRAPH, true);
    config::SetPlatformConfig(KEY_VERIFY_TENSOR_GRAPH_DUMP_TENSOR, true);
    config::SetPlatformConfig(KEY_VERIFY_PASS, false);
    config::SetPlatformConfig(KEY_VERIFY_PASS_DUMP_TENSOR, true);
    config::SetPlatformConfig(KEY_VERIFY_EXECUTE_GRAPH, false);
    config::SetPlatformConfig(KEY_VERIFY_EXECUTE_GRAPH_DUMP_TENSOR, true);

    int n = 4;
    int k = 1024;
    int m = 576;

    using EltType = float;
    DataType eltType = DT_FP32;
    Tensor t0(eltType, {n, k}, "t0");
    Tensor t1(eltType, {k, m}, "t1");
    Tensor t2(eltType, {n, m}, "t2");

    std::vector<EltType> t0Data(n * k);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            t0Data[i * k + j] = (i / 32) * 4 + (j / 64);
        }
    }
    std::vector<EltType> t1Data(k * m);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < m; j++) {
            t1Data[i * m + j] = (i / 64) * 4 + (j / 64);
        }
    }
    std::vector<EltType> t2Data(n * m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            EltType sum = 0;
            for (int x = 0; x < k; x++) {
                sum += t0Data[i * k + x] * t1Data[x * m + j];
            }
            t2Data[i * m + j] = sum;
        }
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<EltType>(t0, t0Data),
        RawTensorData::CreateTensor<EltType>(t1, t1Data),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<EltType>(t2, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<EltType>(t2, t2Data),
    });

    FUNCTION("main", {t0, t1}, {t2}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            t2 = Matrix::Matmul(eltType, t0, t1); // int32
        }
    }
}

TEST_F(DynamicOpsTest, TestGetAndSetTensorDataExpr) {
    config::SetCodeGenOption(CODEGEN_EXPRESSION_FUSION, true);

    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling, tiling);

    int n = tiling * 1;
    int init = 10;
    Tensor input(DT_INT32, {n, n, n}, "input");
    Tensor output(DT_INT32, {n, n, n}, "output");
    std::vector<int32_t> golden(n * n * n);
    for (int i = 0; i < n * n * n; i++) {
        golden[i] = init + init + i;
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(input, init),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<int32_t>(output, golden),
    });

    FUNCTION("main", {input}, {output}) {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(n)) {
            LOOP("Step1", FunctionType::DYNAMIC_LOOP, j, LoopRange(n)) {
                auto add = Add(input, input);
                for (int k = 0; k < n; k++) {
                    SymbolicScalar s = GetTensorData(add, {i, j, k});
                    SetTensorData(s + i * tiling * tiling + j * tiling + k, {i, j, k}, output);
                }
            }
        }
    }
}

TEST_F(DynamicOpsTest, ElementScalar) {
    auto floatElement = Element(DT_BF16, 2.0);
    auto intElement = Element(DT_INT32, static_cast<long>(2));

    auto floatRes = floatElement + floatElement;
    EXPECT_EQ(floatRes.GetFloatData(), 4.0);

    floatRes = floatElement - floatElement;
    EXPECT_EQ(floatRes.GetFloatData(), 0.0);

    floatRes = floatElement * floatElement;
    EXPECT_EQ(floatRes.GetFloatData(), 4.0);

    floatRes = floatElement / floatElement;
    EXPECT_EQ(floatRes.GetFloatData(), 1.0);

    auto intRes = intElement % intElement;
    EXPECT_EQ(intRes.GetSignedData(), 0);

    intRes = intElement + intElement;
    EXPECT_EQ(intRes.GetSignedData(), 4);

    intRes = intElement - intElement;
    EXPECT_EQ(intRes.GetSignedData(), 0);

    intRes = intElement * intElement;
    EXPECT_EQ(intRes.GetSignedData(), 4);

    intRes = intElement / intElement;
    EXPECT_EQ(intRes.GetSignedData(), 1);

    auto BoolRes = floatElement > floatElement;
    EXPECT_EQ(BoolRes, false);

    BoolRes = floatElement == floatElement;
    EXPECT_EQ(BoolRes, true);

    BoolRes = floatElement != floatElement;
    EXPECT_EQ(BoolRes, false);

    BoolRes = floatElement < floatElement;
    EXPECT_EQ(BoolRes, false);

    BoolRes = floatElement <= floatElement;
    EXPECT_EQ(BoolRes, true);

    BoolRes = floatElement > floatElement;
    EXPECT_EQ(BoolRes, false);

    BoolRes = floatElement >= floatElement;
    EXPECT_EQ(BoolRes, true);

    BoolRes = floatElement > floatElement;
    EXPECT_EQ(BoolRes, false);

    BoolRes = intElement == intElement;
    EXPECT_EQ(BoolRes, true);

    BoolRes = intElement != intElement;
    EXPECT_EQ(BoolRes, false);

    BoolRes = intElement < intElement;
    EXPECT_EQ(BoolRes, false);

    BoolRes = intElement <= intElement;
    EXPECT_EQ(BoolRes, true);

    BoolRes = intElement > intElement;
    EXPECT_EQ(BoolRes, false);

    BoolRes = intElement >= intElement;
    EXPECT_EQ(BoolRes, true);
}

TEST_F(DynamicOpsTest, MatmulAcc) {
    config::SetPlatformConfig(KEY_EXTRACT_TENSOR_GRAPH_THEN_COMPILE, true);
    config::SetPlatformConfig(KEY_VERIFY_TENSOR_GRAPH, true);
    config::SetPlatformConfig(KEY_VERIFY_TENSOR_GRAPH_CHECK_PRECISION, true);
    config::SetPlatformConfig(KEY_VERIFY_PASS, true);
    config::SetPlatformConfig(KEY_VERIFY_PASS_CHECK_PRECISION, true);

    Tensor t0(DT_FP32, {128, 128}, "t0");
    Tensor t1(DT_FP32, {128, 128}, "t1");
    Tensor t2(DT_FP32, {64, 64}, "t2");
    Tensor out(DT_FP32, {64, 64}, "out");

    auto d0 = RawTensorData::CreateConstantTensor<float>(t0, 1.0f);
    auto d1 = RawTensorData::CreateConstantTensor<float>(t1, 1.0f);
    auto d2 = RawTensorData::CreateConstantTensor<float>(t2, 1.0f);
    auto out0 = RawTensorData::CreateConstantTensor<float>(out, 1.0f);
    auto golden = RawTensorData::CreateConstantTensor<float>(out, 65.0f);
    ProgramData::GetInstance().PrepareData({d0, d1, d2}, {out0}, {golden});

    FUNCTION("main", {t0, t1, t2}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto v0 = View(t0, {64, 64}, {64, 64});
            auto v1 = View(t1, {64, 64}, {64, 64});
            auto m0 = Matrix::Matmul<false, true>(DT_FP32, v0, v1);
            out = Add(m0, t2);
        }
    }
}

TEST_F(DynamicOpsTest, GetTensorData) {
    config::SetPlatformConfig(KEY_EXTRACT_TENSOR_GRAPH_THEN_COMPILE, true);
    config::SetPlatformConfig(KEY_VERIFY_TENSOR_GRAPH, true);
    config::SetPlatformConfig(KEY_VERIFY_TENSOR_GRAPH_CHECK_PRECISION, true);
    config::SetPlatformConfig(KEY_VERIFY_PASS, true);
    config::SetPlatformConfig(KEY_VERIFY_PASS_CHECK_PRECISION, true);
    config::SetPlatformConfig(KEY_VERIFY_DUMP_PERF_DATA, true);

    Tensor t0(DT_FP32, {32, 32}, "t0");
    Tensor out(DT_FP32, {64, 64}, "out");

    auto t0Data = RawTensorData::CreateConstantTensor<float>(t0, 1.0f);
    auto outData = RawTensorData::CreateConstantTensor<float>(out, 0.0f);
    auto golden = RawTensorData::CreateConstantTensor<float>(out, 2.0f);

    ProgramData::GetInstance().PrepareData({t0Data}, {outData}, {golden});

    FUNCTION("main", {t0}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(2)) {
            auto v = VectorDuplicate(Element(DT_INT32, 32), DT_INT32, {16, 16});
            auto index = GetTensorData(v, {0, 0});
            Print("i=", i, " index=", index, " v=", v);
            auto d = Add(t0, t0);
            Assemble(d, {index * i, 0}, out);
            Assemble(d, {index * i, 32}, out);
        }
    }
}