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
 * \file test_dynamic_mla_prolog.cpp
 * \brief
 */

#include "test_dynamic.h"
#include "test_suite_stest_ops.h"
#include "models/deepseek/dynamic_mla.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class MlaPrologSTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

namespace {

struct TestShapeParams {
    int b;
    int s;
    int s2;
    int n;
    int h;
    int qLoraRank;
    int qkNopeHeadDim;
    int qkRopeHeadDim;
    int kvLoraRank;
    int blockSize;
};

void PerformanceConfig() {
    Program::GetInstance().GetConfig().Set<int>(DB_TYPE, 1);
    Program::GetInstance().GetConfig().Set<int>(L1_REUSE, 4);
    Program::GetInstance().GetConfig().Set<std::map<int, int>>(CUBE_NBUFFER_MAP, {
        {3, 4}
    });
    Program::GetInstance().GetConfig().Set<int>(COPYIN_THRESHOLD, 2 * 1024 * 1024);
}

template <typename T>
static std::shared_ptr<RawTensorData> CreateTensorData(Tensor tensor, std::vector<int> shape, std::string fileName) {
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> values(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, values);
    return RawTensorData::CreateTensor<T>(tensor, values);
}

template <typename T>
static std::vector<T> getGoldenVec(std::vector<int> shape, std::string fileName) {
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> golden(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, golden);
    return golden;
}

template <typename T = npu::tile_fwk::float16, bool nz = true, typename wDtype = int8_t,
    bool isSmooth = true, bool usePrefetch = true>
void TestDynamicMlaProlog(const TestShapeParams &params, const MlaTileConfig &tileConfig,
    std::string cacheMode = "PA_NZ") {
    config::SetHostConfig(npu::tile_fwk::KEY_ONLY_CODEGEN, true);

    int b = params.b;
    int s = params.s;
    int s2 = params.s2;
    int n = params.n;
    int n2 = 1;
    int h = params.h;
    int qLoraRank = params.qLoraRank;
    int qkNopeHeadDim = params.qkNopeHeadDim;
    int qkRopeHeadDim = params.qkRopeHeadDim;
    int kvLoraRank = params.kvLoraRank;
    int blockSize = params.blockSize;
    int qHeadDim = qkNopeHeadDim + qkRopeHeadDim;

    DataType dType = (std::is_same<T, npu::tile_fwk::float16>::value) ? DT_FP16 : DT_BF16;
    bool isQuant = std::is_same<wDtype, int8_t>::value;
    DataType dTypeQuant = isQuant ? DT_INT8 : dType;

    std::vector<int> xShape = {b, s, h};
    std::vector<int> wDqShape = {h, qLoraRank};
    std::vector<int> wUqQrShape = {qLoraRank, n * qHeadDim};
    std::vector<int> wDkvKrShape = {h, kvLoraRank + qkRopeHeadDim};
    std::vector<int> wUkShape = {n, qkNopeHeadDim, kvLoraRank};
    std::vector<int> cosShape = {b, s, qkRopeHeadDim};
    std::vector<int> gammaCqShape = {qLoraRank};
    std::vector<int> gammaCkvShape = {kvLoraRank};
    std::vector<int> kvLenShape = {b, s};
    std::vector<int> kvCacheShape = {b, n2, s2, kvLoraRank};
    std::vector<int> krCacheShape = {b, n2, s2, qkRopeHeadDim};
    std::vector<int> kvCacheOutShape = {b, n2, s2, kvLoraRank};
    std::vector<int> krCacheOutShape = {b, n2, s2, qkRopeHeadDim};
    if (cacheMode != "BNSD") {
        int blockNum = b * (s2 / blockSize);
        kvCacheShape = {blockNum, blockSize, n2, kvLoraRank};
        krCacheShape = {blockNum, blockSize, n2, qkRopeHeadDim};
        kvCacheOutShape = {blockNum * blockSize, n2 * kvLoraRank};
        krCacheOutShape = {blockNum * blockSize, n2 * qkRopeHeadDim};
    }
    std::vector<int> wQbScaleShape = {1, n * qHeadDim};
    std::vector<int> smoothCqShape{1, qLoraRank};
    // output
    std::vector<int> qOutShape = {b, s, n, kvLoraRank};
    std::vector<int> qRopeOutShape = {b, s, n, qkRopeHeadDim};

    Tensor x(dType, xShape, "x");
    TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    Tensor wDq(dType, wDqShape, "wDq", NodeType::LOCAL, weightFormat);
    Tensor wUqQr(dTypeQuant, wUqQrShape, "wUqQr", NodeType::LOCAL, weightFormat);
    if constexpr (usePrefetch) {  // TODO 放到接口实现里
        wDq.Prefetch();
        wUqQr.Prefetch();
    }
    Tensor wDkvKr(dType, wDkvKrShape, "wDkvKr", NodeType::LOCAL, weightFormat);
    Tensor wUk(dType, wUkShape, "wUk", NodeType::LOCAL, weightFormat);
    Tensor gammaCq(dType, gammaCqShape, "gammaCq");
    Tensor gammaCkv(dType, gammaCkvShape, "gammaCkv");
    Tensor cos(dType, cosShape, "cos");
    Tensor sin(dType, cosShape, "sin");
    Tensor cacheIndex(DT_INT64, kvLenShape, "cacheIndex"); // int64
    Tensor kvCache(dType, kvCacheShape, "kvCache");
    Tensor krCache(dType, krCacheShape, "krCache");
    Tensor wQbScale(DT_FP32, wQbScaleShape, "wQbScale");
    Tensor smoothCq(DT_FP32, smoothCqShape, "smoothCq");

    // output
    Tensor outputKvCache(dType, kvCacheOutShape, "outputKvCache");
    Tensor outputKrCache(dType, krCacheOutShape, "outputKrCache");
    Tensor outputQ(dType, qOutShape, "outputQ");
    Tensor outputQRope(dType, qRopeOutShape, "outputQRope");

    // output
    std::vector<T> golden1 = getGoldenVec<T>(qOutShape, "/q_golden.bin");
    std::vector<T> golden2 = getGoldenVec<T>(qRopeOutShape, "/q_rope_golden.bin");
    std::vector<T> golden3 = getGoldenVec<T>(kvCacheOutShape, "/kv_cache_golden.bin");
    std::vector<T> golden4 = getGoldenVec<T>(krCacheOutShape, "/kr_cache_golden.bin");

    auto xData = CreateTensorData<T>(x, xShape, "/x.bin");
    auto wDqData = CreateTensorData<T>(wDq, wDqShape, "/wDq.bin");
    auto wUqQrData = CreateTensorData<wDtype>(wUqQr, wUqQrShape, "/wUqQr.bin");
    auto wUkData = CreateTensorData<T>(wUk, wUkShape, "/wUk.bin");
    auto wDkvKrData = CreateTensorData<T>(wDkvKr, wDkvKrShape, "/wDkvKr.bin");
    auto gammaCqData = CreateTensorData<T>(gammaCq, gammaCqShape, "/gamma_cq.bin");
    auto gammaCkvData = CreateTensorData<T>(gammaCkv, gammaCkvShape, "/gamma_ckv.bin");
    auto cosData = CreateTensorData<T>(cos, cosShape, "/cos.bin");
    auto sinData = CreateTensorData<T>(sin, cosShape, "/sin.bin");
    auto kvLenData = CreateTensorData<int64_t>(cacheIndex, kvLenShape, "/kv_len.bin");
    auto kvCacheData = CreateTensorData<T>(kvCache, kvCacheShape, "/kv_cache.bin");
    auto krCacheData = CreateTensorData<T>(krCache, krCacheShape, "/kr_cache.bin");
    auto outKvCacheData = CreateTensorData<T>(outputKvCache, kvCacheOutShape, "/kv_cache.bin");
    auto outKrCacheData = CreateTensorData<T>(outputKrCache, krCacheOutShape, "/kr_cache.bin");
    auto outputQData = RawTensorData::CreateConstantTensor<T>(outputQ, 0.0);
    auto outputQRopeData = RawTensorData::CreateConstantTensor<T>(outputQRope, 0.0);

    std::vector<RawTensorDataPtr> outputDataList = {outputQData, outputQRopeData, outKvCacheData, outKrCacheData};
    std::vector<RawTensorDataPtr> inputDataList =
        {xData, wDqData, wUqQrData, wUkData, wDkvKrData, gammaCqData, gammaCkvData, sinData, cosData, kvLenData,
         kvCacheData, krCacheData};
    MlaQuantInputs quantInputs;
    if (isQuant) {
        auto wQbScaleData = CreateTensorData<float>(wQbScale, wQbScaleShape, "/w_qb_scale.bin");
        inputDataList.emplace_back(wQbScaleData);
        quantInputs.dequantScaleWUqQr = wQbScale;
        if (isSmooth) {
            auto smoothCqData = CreateTensorData<float>(smoothCq, smoothCqShape, "/smooth_cq.bin");
            inputDataList.emplace_back(smoothCqData);
            quantInputs.smoothScalesCq = smoothCq;
        }
    } else {
        inputDataList.emplace_back(nullptr); // quantInputs.dequantScaleWUqQr
        inputDataList.emplace_back(nullptr); // quantInputs.smoothScalesCq
    }

    MlaProlog(x, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos, cacheIndex, kvCache, krCache, quantInputs,
        tileConfig, outputQ, outputQRope, outputKvCache, outputKrCache, 1e-5f, 1e-5f, cacheMode);

    auto funcOp = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
#ifndef AC_ENABLE_FRAMEWORK_WITHOUT_CANN
    DynFuncRunner::Run(funcOp, inputDataList, outputDataList);

    std::cout << "qNope ====== " << std::endl;
    EXPECT_TRUE(resultCmp<T>(golden1, (T *)outputQData->data(), 0.008f));
    std::cout << "qRope ======" << std::endl;
    EXPECT_TRUE(resultCmp<T>(golden2, (T *)outputQRopeData->data(), 0.005f));
    std::cout << "kv ====== " << std::endl;
    EXPECT_TRUE(resultCmp<T>(golden3, (T *)outKvCacheData->data(), 0.003f));
    std::cout << "kr ====== " << std::endl;
    EXPECT_TRUE(resultCmp<T>(golden4, (T *)outKrCacheData->data(), 0.003f));
#endif
}

////// fp16, quant, weight nz, "PA_NZ"
TEST_F(MlaPrologSTest, b16_s1_pa_nz_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {16, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {16, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, true, int8_t, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b16_s2_pa_nz_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {16, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {16, 2};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, true, int8_t, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b32_s1_pa_nz_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, true, int8_t, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b32_s2_pa_nz_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, true, int8_t, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b64_s1_pa_nz_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {64, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, true, int8_t, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b64_s2_pa_nz_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {64, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, true, int8_t, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b24_s1_pa_nz_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {24, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {24, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, true, int8_t, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b24_s2_pa_nz_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {24, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {24, 2};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, true, int8_t, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b48_s1_pa_nz_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {48, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {48, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, true, int8_t, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b48_s2_pa_nz_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {48, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {48, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, true, int8_t, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b96_s1_pa_nz_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {96, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, true, int8_t, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b96_s2_pa_nz_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {96, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, true, int8_t, true, true>(params, tileConfig, cacheMode);
}

////// bf16, quant, weight nz, "PA_NZ"
TEST_F(MlaPrologSTest, b16_s1_pa_nz_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {16, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {16, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, true, int8_t, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b16_s2_pa_nz_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {16, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {16, 2};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, true, int8_t, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b32_s1_pa_nz_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, true, int8_t, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b32_s2_pa_nz_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, true, int8_t, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b64_s1_pa_nz_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {64, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, true, int8_t, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b64_s2_pa_nz_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {64, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, true, int8_t, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b24_s1_pa_nz_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {24, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {24, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, true, int8_t, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b24_s2_pa_nz_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {24, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {24, 2};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, true, int8_t, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b48_s1_pa_nz_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {48, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {48, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, true, int8_t, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b48_s2_pa_nz_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {48, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {48, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, true, int8_t, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b96_s1_pa_nz_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {96, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, true, int8_t, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b96_s2_pa_nz_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {96, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, true, int8_t, true, true>(params, tileConfig, cacheMode);
}

////// fp16, quant, weight nd, "PA_BSND"
TEST_F(MlaPrologSTest, b32_s1_pa_nd_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_BSND";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, false, int8_t, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b32_s2_pa_nd_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_BSND";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, false, int8_t, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b48_s1_pa_nd_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {48, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_BSND";
    MlaTileConfig tileConfig = {48, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, false, int8_t, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b48_s2_pa_nd_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {48, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_BSND";
    MlaTileConfig tileConfig = {48, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, false, int8_t, true, true>(params, tileConfig, cacheMode);
}

////// bf16, quant, weight nd, "PA_BSND"
TEST_F(MlaPrologSTest, b64_s1_pa_nd_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {64, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_BSND";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, false, int8_t, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b64_s2_pa_nd_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {64, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_BSND";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, false, int8_t, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b96_s1_pa_nd_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {96, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_BSND";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, false, int8_t, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b96_s2_pa_nd_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {96, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_BSND";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, false, int8_t, true, true>(params, tileConfig, cacheMode);
}

////// fp16, no quant, weight nz, "PA_NZ"
TEST_F(MlaPrologSTest, b32_s1_pa_nz_fp16) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b32_s2_pa_nz_fp16) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, true>(params, tileConfig, cacheMode);
}

////// fp16, no quant, weight nd, "PA_BSND"
TEST_F(MlaPrologSTest, b32_s1_pa_nd_fp16) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_BSND";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, false, npu::tile_fwk::float16, false, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b32_s2_pa_nd_fp16) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_BSND";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, false, npu::tile_fwk::float16, false, true>(params, tileConfig, cacheMode);
}

// small shape
TEST_F(MlaPrologSTest, b16_s2_pa_nd_fp16_small) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {16, 2, 256, 128, 256, 256, 128, 64, 512, 128};
    std::string cacheMode = "PA_BSND";
    MlaTileConfig tileConfig = {16, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, false, npu::tile_fwk::float16, false, true>(params, tileConfig, cacheMode);
}

} // namespace
