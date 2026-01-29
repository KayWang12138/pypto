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
 * \file test_dynamic_mla_prolog.cpp
 * \brief
 */

#include "test_dev_func_runner.h"
#include "test_suite_stest_ops.h"
#include "operator/models/deepseek/dynamic_mla.h"
#include "test_cost_macro.h"

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
    config::SetPassOption(VEC_NBUFFER_MODE, 1);
    config::SetPassOption(CUBE_L1_REUSE_SETTING, std::map<int64_t, int64_t>{{-1, 4}});
    config::SetPassOption(CUBE_NBUFFER_SETTING, std::map<int64_t, int64_t>{{3, 4}});
    config::SetPassOption(MG_COPYIN_UPPER_BOUND, 2 * 1024 * 1024);
}

template <typename T>
static std::shared_ptr<RawTensorData> CreateTensorData(Tensor tensor, std::vector<int64_t> shape, std::string fileName) {
    uint64_t capacity = std::accumulate(shape.begin(), shape.end(), uint64_t{1}, std::multiplies<uint64_t>());
    std::vector<T> values(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, values);
    return RawTensorData::CreateTensor<T>(tensor, values);
}

template <typename T>
static std::vector<T> getGoldenVec(std::vector<int64_t> shape, std::string fileName) {
    uint64_t capacity = std::accumulate(shape.begin(), shape.end(), uint64_t{1}, std::multiplies<uint64_t>());
    std::vector<T> golden(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, golden);
    return golden;
}

struct MlaPrologShapes {
    std::vector<int64_t> xShape;
    std::vector<int64_t> wDqShape;
    std::vector<int64_t> wUqQrShape;
    std::vector<int64_t> wDkvKrShape;
    std::vector<int64_t> wUkShape;
    std::vector<int64_t> cosShape;
    std::vector<int64_t> gammaCqShape;
    std::vector<int64_t> gammaCkvShape;
    std::vector<int64_t> kvLenShape;
    std::vector<int64_t> kvCacheShape;
    std::vector<int64_t> krCacheShape;
    std::vector<int64_t> kvCacheOutShape;
    std::vector<int64_t> krCacheOutShape;
    std::vector<int64_t> scaleWDqShape;
    std::vector<int64_t> scaleWUqQrShape;
    std::vector<int64_t> scaleWDkvKrShape;
    std::vector<int64_t> smoothCqShape;
    std::vector<int64_t> qOutShape;
    std::vector<int64_t> qRopeOutShape;
    int blockNum;
    int n2;
};

inline MlaPrologShapes CalculateMlaPrologShapes(const TestShapeParams &params) {
    int b = params.b;
    int s = params.s;
    int s2 = params.s2;
    int n = params.n;
    int h = params.h;
    int qLoraRank = params.qLoraRank;
    int qkNopeHeadDim = params.qkNopeHeadDim;
    int qkRopeHeadDim = params.qkRopeHeadDim;
    int kvLoraRank = params.kvLoraRank;
    int blockSize = params.blockSize;
    int qHeadDim = qkNopeHeadDim + qkRopeHeadDim;
    int n2 = 1;
    int blockNum = b * (s2 / blockSize);

    return {
        {b, s, h},
        {h, qLoraRank},
        {qLoraRank, n * qHeadDim},
        {h, kvLoraRank + qkRopeHeadDim},
        {n, qkNopeHeadDim, kvLoraRank},
        {b, s, qkRopeHeadDim},
        {qLoraRank},
        {kvLoraRank},
        {b, s},
        {blockNum, blockSize, n2, kvLoraRank},
        {blockNum, blockSize, n2, qkRopeHeadDim},
        {blockNum * blockSize, n2 * kvLoraRank},
        {blockNum * blockSize, n2 * qkRopeHeadDim},
        {1, qLoraRank},
        {1, n * qHeadDim},
        {1, kvLoraRank + qkRopeHeadDim},
        {1, qLoraRank},
        {b, s, n, kvLoraRank},
        {b, s, n, qkRopeHeadDim},
        blockNum,
        n2
    };
}

template <typename T>
struct MlaPrologTensors {
    Tensor x;
    Tensor wDq;
    Tensor wUqQr;
    Tensor wDkvKr;
    Tensor wUk;
    Tensor gammaCq;
    Tensor gammaCkv;
    Tensor cos;
    Tensor sin;
    Tensor cacheIndex;
    Tensor kvCache;
    Tensor krCache;
    Tensor scaleWDq;
    Tensor scaleWUqQr;
    Tensor scaleWDkvKr;
    Tensor smoothCq;
    Tensor outputKvCache;
    Tensor outputKrCache;
    Tensor outputQ;
    Tensor outputQRope;
    Tensor dynamicX;
    Tensor dynamicCos;
    Tensor dynamicSin;
    Tensor dynamicCacheIndex;
    Tensor dynamicOutputQ;
    Tensor dynamicOutputQRope;
};

template <typename T, typename wDtype, bool isQuantA, bool isQuantB, bool nz, bool usePrefetch>
MlaPrologTensors<T> CreateMlaPrologTensors(const MlaPrologShapes &shapes, const TestShapeParams &params) {
    DataType dType = (std::is_same<T, npu::tile_fwk::float16>::value) ? DT_FP16 : DT_BF16;
    DataType dTypeQuantA = (std::is_same<wDtype, int8_t>::value && isQuantA) ? DT_INT8 : dType;
    DataType dTypeQuantB = (std::is_same<wDtype, int8_t>::value && isQuantB) ? DT_INT8 : dType;
    TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;

    int n = params.n;
    int qkRopeHeadDim = params.qkRopeHeadDim;
    int kvLoraRank = params.kvLoraRank;
    int h = params.h;

    Tensor x(dType, shapes.xShape, "x");
    Tensor wDq(dTypeQuantA, shapes.wDqShape, "wDq", weightFormat);
    Tensor wUqQr(dTypeQuantB, shapes.wUqQrShape, "wUqQr", weightFormat);
    
    if constexpr (usePrefetch) {
        wDq.SetCachePolicy(CachePolicy::PREFETCH, true);
        wUqQr.SetCachePolicy(CachePolicy::PREFETCH, true);
    }
    
    Tensor wDkvKr(dTypeQuantA, shapes.wDkvKrShape, "wDkvKr", weightFormat);
    Tensor wUk(dType, shapes.wUkShape, "wUk", weightFormat);
    Tensor gammaCq(dType, shapes.gammaCqShape, "gammaCq");
    Tensor gammaCkv(dType, shapes.gammaCkvShape, "gammaCkv");
    Tensor cos(dType, shapes.cosShape, "cos");
    Tensor sin(dType, shapes.cosShape, "sin");
    Tensor cacheIndex(DT_INT64, shapes.kvLenShape, "cacheIndex");
    Tensor kvCache(dType, shapes.kvCacheShape, "kvCache");
    Tensor krCache(dType, shapes.krCacheShape, "krCache");
    Tensor scaleWDq(DT_FP32, shapes.scaleWDqShape, "scaleWDq");
    Tensor scaleWUqQr(DT_FP32, shapes.scaleWUqQrShape, "scaleWUqQr");
    Tensor scaleWDkvKr(DT_FP32, shapes.scaleWDkvKrShape, "scaleWDkvKr");
    Tensor smoothCq(DT_FP32, shapes.smoothCqShape, "smoothCq");
    Tensor outputKvCache(dType, shapes.kvCacheOutShape, "outputKvCache");
    Tensor outputKrCache(dType, shapes.krCacheOutShape, "outputKrCache");
    Tensor outputQ(dType, shapes.qOutShape, "outputQ");
    Tensor outputQRope(dType, shapes.qRopeOutShape, "outputQRope");

    Tensor dynamicX(dType, {-1, -1, h}, "dynamicX");
    Tensor dynamicCos(dType, {-1, -1, qkRopeHeadDim}, "dynamicCos");
    Tensor dynamicSin(dType, {-1, -1, qkRopeHeadDim}, "dynamicSin");
    Tensor dynamicCacheIndex(DT_INT64, {-1, -1}, "dynamicCacheIndex");
    Tensor dynamicOutputQ(dType, {-1, GetInputShape(dynamicX, 1), n, kvLoraRank}, "dynamicOutputQ");
    Tensor dynamicOutputQRope(dType, {-1, GetInputShape(dynamicX, 1), n, qkRopeHeadDim}, "dynamicOutputQRope");

    return {x, wDq, wUqQr, wDkvKr, wUk, gammaCq, gammaCkv, cos, sin, cacheIndex,
            kvCache, krCache, scaleWDq, scaleWUqQr, scaleWDkvKr, smoothCq,
            outputKvCache, outputKrCache, outputQ, outputQRope,
            dynamicX, dynamicCos, dynamicSin, dynamicCacheIndex, dynamicOutputQ, dynamicOutputQRope};
}

template <typename T, typename wDtype, bool isQuantA, bool isQuantB>
struct MlaPrologData {
    std::vector<T> golden1;
    std::vector<T> golden2;
    std::vector<T> golden3;
    std::vector<T> golden4;
    std::vector<RawTensorDataPtr> inputDataList;
    std::vector<RawTensorDataPtr> outputDataList;
    MlaQuantInputs quantInputs;
};

template <typename T, typename wDtype, bool isQuantA, bool isQuantB, bool isSmooth>
MlaPrologData<T, wDtype, isQuantA, isQuantB> PrepareMlaPrologData(const MlaPrologTensors<T> &tensors,
                                                                     const MlaPrologShapes &shapes) {
    using wDtypeA = typename std::conditional<isQuantA, wDtype, T>::type;
    using wDtypeB = typename std::conditional<isQuantB, wDtype, T>::type;

    auto golden1 = getGoldenVec<T>(shapes.qOutShape, "/q_golden.bin");
    auto golden2 = getGoldenVec<T>(shapes.qRopeOutShape, "/q_rope_golden.bin");
    auto golden3 = getGoldenVec<T>(shapes.kvCacheOutShape, "/kv_cache_golden.bin");
    auto golden4 = getGoldenVec<T>(shapes.krCacheOutShape, "/kr_cache_golden.bin");

    auto xData = CreateTensorData<T>(tensors.x, shapes.xShape, "/x.bin");
    auto wDqData = CreateTensorData<wDtypeA>(tensors.wDq, shapes.wDqShape, "/wDq.bin");
    auto wUqQrData = CreateTensorData<wDtypeB>(tensors.wUqQr, shapes.wUqQrShape, "/wUqQr.bin");
    auto wUkData = CreateTensorData<T>(tensors.wUk, shapes.wUkShape, "/wUk.bin");
    auto wDkvKrData = CreateTensorData<wDtypeA>(tensors.wDkvKr, shapes.wDkvKrShape, "/wDkvKr.bin");
    auto gammaCqData = CreateTensorData<T>(tensors.gammaCq, shapes.gammaCqShape, "/gamma_cq.bin");
    auto gammaCkvData = CreateTensorData<T>(tensors.gammaCkv, shapes.gammaCkvShape, "/gamma_ckv.bin");
    auto cosData = CreateTensorData<T>(tensors.cos, shapes.cosShape, "/cos.bin");
    auto sinData = CreateTensorData<T>(tensors.sin, shapes.cosShape, "/sin.bin");
    auto kvLenData = CreateTensorData<int64_t>(tensors.cacheIndex, shapes.kvLenShape, "/kv_len.bin");
    auto kvCacheData = CreateTensorData<T>(tensors.kvCache, shapes.kvCacheShape, "/kv_cache.bin");
    auto krCacheData = CreateTensorData<T>(tensors.krCache, shapes.krCacheShape, "/kr_cache.bin");
    auto outKvCacheData = CreateTensorData<T>(tensors.outputKvCache, shapes.kvCacheOutShape, "/kv_cache.bin");
    auto outKrCacheData = CreateTensorData<T>(tensors.outputKrCache, shapes.krCacheOutShape, "/kr_cache.bin");
    auto outputQData = RawTensorData::CreateConstantTensor<T>(tensors.outputQ, 0.0);
    auto outputQRopeData = RawTensorData::CreateConstantTensor<T>(tensors.outputQRope, 0.0);

    std::vector<RawTensorDataPtr> inputDataList = {xData, wDqData, wUqQrData, wUkData, wDkvKrData, gammaCqData, gammaCkvData,
                                                   sinData, cosData, kvLenData, kvCacheData, krCacheData};
    std::vector<RawTensorDataPtr> outputDataList = {outputQData, outputQRopeData, outKvCacheData, outKrCacheData};

    MlaQuantInputs quantInputs;
    if (isQuantA) {
        auto scaleWDqData = CreateTensorData<float>(tensors.scaleWDq, shapes.scaleWDqShape, "/w_qa_scale.bin");
        auto scaleWDkvKrData = CreateTensorData<float>(tensors.scaleWDkvKr, shapes.scaleWDkvKrShape, "/w_kva_scale.bin");
        inputDataList.emplace_back(scaleWDqData);
        inputDataList.emplace_back(scaleWDkvKrData);
        quantInputs.dequantScaleWDq = tensors.scaleWDq;
        quantInputs.dequantScaleWDkvKr = tensors.scaleWDkvKr;
    } else {
        inputDataList.emplace_back(nullptr);
        inputDataList.emplace_back(nullptr);
    }
    if (isQuantB) {
        auto scaleWUqQrData = CreateTensorData<float>(tensors.scaleWUqQr, shapes.scaleWUqQrShape, "/w_qb_scale.bin");
        inputDataList.emplace_back(scaleWUqQrData);
        quantInputs.dequantScaleWUqQr = tensors.scaleWUqQr;
        if (isSmooth) {
            auto smoothCqData = CreateTensorData<float>(tensors.smoothCq, shapes.smoothCqShape, "/smooth_cq.bin");
            inputDataList.emplace_back(smoothCqData);
            quantInputs.smoothScalesCq = tensors.smoothCq;
        }
    } else {
        inputDataList.emplace_back(nullptr);
        inputDataList.emplace_back(nullptr);
    }

    return {golden1, golden2, golden3, golden4, inputDataList, outputDataList, quantInputs};
}

inline void SetupMlaPrologProgram(const MlaPrologTensors<npu::tile_fwk::float16> &tensors,
                                  const std::vector<npu::tile_fwk::float16> &golden1,
                                  const std::vector<npu::tile_fwk::float16> &golden2,
                                  const std::vector<npu::tile_fwk::float16> &golden3,
                                  const std::vector<npu::tile_fwk::float16> &golden4,
                                  const std::vector<RawTensorDataPtr> &inputDataList,
                                  const std::vector<RawTensorDataPtr> &outputDataList) {
    ProgramData::GetInstance().AppendInputs({inputDataList});
    ProgramData::GetInstance().AppendOutputs({outputDataList});
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<npu::tile_fwk::float16>(tensors.outputQ, golden1),
        RawTensorData::CreateTensor<npu::tile_fwk::float16>(tensors.outputQRope, golden2),
        RawTensorData::CreateTensor<npu::tile_fwk::float16>(tensors.outputKvCache, golden3),
        RawTensorData::CreateTensor<npu::tile_fwk::float16>(tensors.outputKrCache, golden4),
    });
}

inline void BuildMlaPrologGraph(const MlaPrologTensors<npu::tile_fwk::float16> &tensors,
                                 const MlaQuantInputs &quantInputs,
                                 const MlaTileConfig &tileConfig,
                                 const std::string &cacheMode) {
    MlaProlog(tensors.dynamicX, tensors.wDq, tensors.wUqQr, tensors.wUk, tensors.wDkvKr,
              tensors.gammaCq, tensors.gammaCkv, tensors.dynamicSin, tensors.dynamicCos, tensors.dynamicCacheIndex,
              tensors.kvCache, tensors.krCache, quantInputs, tileConfig,
              tensors.dynamicOutputQ, tensors.dynamicOutputQRope, tensors.outputKvCache, tensors.outputKrCache,
              1e-5f, 1e-5f, cacheMode);
}

template <typename T>
inline void VerifyMlaPrologResult(const std::vector<T> &golden1,
                                   const std::vector<T> &golden2,
                                   const std::vector<T> &golden3,
                                   const std::vector<T> &golden4,
                                   const std::vector<RawTensorDataPtr> &outputDataList) {
#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(),
                      ProgramData::GetInstance().GetInputData(),
                      outputDataList);
    std::cout << "qNope ====== " << std::endl;
    EXPECT_TRUE(resultCmp<T>(golden1, (T *)outputDataList[0]->data(), 0.008f));
    std::cout << "qRope ======" << std::endl;
    EXPECT_TRUE(resultCmp<T>(golden2, (T *)outputDataList[1]->data(), 0.005f));
    std::cout << "kv ====== " << std::endl;
    EXPECT_TRUE(resultCmp<T>(golden3, (T *)outputDataList[2]->data(), 0.003f));
    std::cout << "kr ====== " << std::endl;
    EXPECT_TRUE(resultCmp<T>(golden4, (T *)outputDataList[3]->data(), 0.003f));
#endif
}

template <typename T = npu::tile_fwk::float16,  typename wDtype = int8_t, bool isQuantA = false, bool isQuantB = true,
    bool isSmooth = true, bool nz = true, bool usePrefetch = true>
void TestDynamicMlaProlog(
    const TestShapeParams &params, const MlaTileConfig &tileConfig, std::string cacheMode = "PA_NZ") {
    SetInterpreterConfig();

    auto shapes = CalculateMlaPrologShapes(params);
    auto tensors = CreateMlaPrologTensors<T, wDtype, isQuantA, isQuantB, nz, usePrefetch>(shapes, params);
    auto testData = PrepareMlaPrologData<T, wDtype, isQuantA, isQuantB, isSmooth>(tensors, shapes);

    SetupMlaPrologProgram(tensors, testData.golden1, testData.golden2, testData.golden3, testData.golden4,
                          testData.inputDataList, testData.outputDataList);

    BuildMlaPrologGraph(tensors, testData.quantInputs, tileConfig, cacheMode);

    VerifyMlaPrologResult<T>(testData.golden1, testData.golden2, testData.golden3, testData.golden4,
                             testData.outputDataList);
}

////// fp16, quant, weight nz, "PA_NZ"
TEST_F(MlaPrologSTest, b16_s1_pa_nz_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {16, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {16, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, int8_t, false, true, true, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b16_s2_pa_nz_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {16, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {16, 2};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, int8_t, false, true, true, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b32_s1_pa_nz_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, int8_t, false, true, true, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b32_s2_pa_nz_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, int8_t, false, true, true, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b64_s1_pa_nz_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {64, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, int8_t, false, true, true, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b64_s2_pa_nz_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {64, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, int8_t, false, true, true, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b24_s1_pa_nz_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {24, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {24, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, int8_t, false, true, true, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b24_s2_pa_nz_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {24, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {24, 2};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, int8_t, false, true, true, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b48_s1_pa_nz_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {48, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {48, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, int8_t, false, true, true, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b48_s2_pa_nz_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {48, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {48, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, int8_t, false, true, true, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b96_s1_pa_nz_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {96, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, int8_t, false, true, true, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b96_s2_pa_nz_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {96, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, int8_t, false, true, true, true, true>(params, tileConfig, cacheMode);
}

////// bf16, quant, weight nz, "PA_NZ"
TEST_F(MlaPrologSTest, b16_s1_pa_nz_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {16, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {16, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, int8_t, false, true, true, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b16_s2_pa_nz_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {16, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {16, 2};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, int8_t, false, true, true, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b32_s1_pa_nz_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, int8_t, false, true, true, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b32_s2_pa_nz_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, int8_t, false, true, true, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b64_s1_pa_nz_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {64, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, int8_t, false, true, true, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b64_s2_pa_nz_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {64, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, int8_t, false, true, true, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b24_s1_pa_nz_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {24, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {24, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, int8_t, false, true, true, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b24_s2_pa_nz_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {24, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {24, 2};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, int8_t, false, true, true, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b48_s1_pa_nz_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {48, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {48, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, int8_t, false, true, true, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b48_s2_pa_nz_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {48, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {48, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, int8_t, false, true, true, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b96_s1_pa_nz_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {96, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, int8_t, false, true, true, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b96_s2_pa_nz_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {96, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, int8_t, false, true, true, true, true>(params, tileConfig, cacheMode);
}

////// fp16, quant, weight nd, "PA_BSND"
TEST_F(MlaPrologSTest, b32_s1_pa_nd_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_BSND";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, int8_t, false, true, true, false, true>(params, tileConfig, cacheMode);
}

TEST_F_WITH_COST(MlaPrologSTest, b32_s2_pa_nd_fp16_quant, 15) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_BSND";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, int8_t, false, true, true, false, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b48_s1_pa_nd_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {48, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_BSND";
    MlaTileConfig tileConfig = {48, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, int8_t, false, true, true, false, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b48_s2_pa_nd_fp16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {48, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_BSND";
    MlaTileConfig tileConfig = {48, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, int8_t, false, true, true, false, true>(params, tileConfig, cacheMode);
}

////// bf16, quant, weight nd, "PA_BSND"
TEST_F(MlaPrologSTest, b64_s1_pa_nd_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {64, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_BSND";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, int8_t, false, true, true, false, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b64_s2_pa_nd_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {64, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_BSND";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, int8_t, false, true, true, false, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b96_s1_pa_nd_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {96, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_BSND";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, int8_t, false, true, true, false, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b96_s2_pa_nd_bf16_quant) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {96, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_BSND";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, int8_t, false, true, true, false, true>(params, tileConfig, cacheMode);
}

////// fp16, no quant, weight nz, "PA_NZ"
TEST_F(MlaPrologSTest, b32_s1_pa_nz_fp16) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, npu::tile_fwk::float16, false, false, false, true, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b32_s2_pa_nz_fp16) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_NZ";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, npu::tile_fwk::float16, false, false, false, true, true>(params, tileConfig, cacheMode);
}

////// fp16, no quant, weight nd, "PA_BSND"
TEST_F(MlaPrologSTest, b32_s1_pa_nd_fp16) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 1, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_BSND";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, npu::tile_fwk::float16, false, false, false, false, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b32_s2_pa_nd_fp16) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {32, 2, 8192, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_BSND";
    MlaTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, npu::tile_fwk::float16, false, false, false, false, true>(params, tileConfig, cacheMode);
}

// small shape
TEST_F(MlaPrologSTest, b16_s2_pa_nd_fp16_small) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {16, 2, 256, 128, 256, 256, 128, 64, 512, 128};
    std::string cacheMode = "PA_BSND";
    MlaTileConfig tileConfig = {16, 1};

    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::float16, npu::tile_fwk::float16, false, false, false, false, true>(params, tileConfig, cacheMode);
}

TEST_F(MlaPrologSTest, b16_s1_pa_nd_bf16_allquant) {
    // b, s1, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, blockSize
    TestShapeParams params = {16, 1, 1024 * 8, 128, 7168, 1536, 128, 64, 512, 128};
    std::string cacheMode = "PA_BSND";
    MlaTileConfig tileConfig = {16, 1};
    PerformanceConfig();
    TestDynamicMlaProlog<npu::tile_fwk::bfloat16, int8_t, true, true, true, true, true>(params, tileConfig, cacheMode);
}

} // namespace
