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
 * \file test_dynamic_mla.cpp
 * \brief
 */

#include "test_dev_func_runner.h"
#include "test_suite_stest_ops.h"
#include "operator/models/deepseek/dynamic_mla.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class DyMla : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

namespace {

struct MlaPrologShapes {
    std::vector<int64_t> x;
    std::vector<int64_t> wDq;
    std::vector<int64_t> wUqQr;
    std::vector<int64_t> wDkvKr;
    std::vector<int64_t> wUk;
    std::vector<int64_t> cos;
    std::vector<int64_t> gammaCq;
    std::vector<int64_t> gammaCkv;
    std::vector<int64_t> kvLen;
    std::vector<int64_t> kvCache;
    std::vector<int64_t> krCache;
    std::vector<int64_t> wQbScale;
    std::vector<int64_t> smoothCq;
    std::vector<int64_t> qOut;
    std::vector<int64_t> qRopeOut;
    std::vector<int64_t> kvCacheOut;
    std::vector<int64_t> krCacheOut;
};

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
    Tensor kvLen;
    Tensor kvCache;
    Tensor krCache;
    Tensor wQbScale;
    Tensor smoothCq;
    Tensor outputKvCache;
    Tensor outputKrCache;
    Tensor outputQ;
    Tensor outputQRope;
};

void pre() {

}

void performanceConfig() {
    config::SetPassOption(VEC_NBUFFER_MODE, 1);
    config::SetPassOption(CUBE_L1_REUSE_SETTING, std::map<int64_t, int64_t>{{-1, 4}});
    config::SetPassOption(CUBE_NBUFFER_SETTING, std::map<int64_t, int64_t>{{3, 4}});
    config::SetPassOption(MG_COPYIN_UPPER_BOUND, 2 * 1024 * 1024);
}

template <typename T>
static std::shared_ptr<RawTensorData> CreateTensorData(Tensor tensor, std::vector<int64_t> shape, std::string fileName) {
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> values(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, values);
    return RawTensorData::CreateTensor<T>(tensor, values);
}

template <typename T>
static std::vector<T> getGoldenVec(std::vector<int64_t> shape, std::string fileName) {
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> golden(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, golden);
    return golden;
}

template <typename T, typename wDtype>
static MlaPrologShapes CalculateMlaPrologShapes(const SimpleParams &params) {
    MlaPrologShapes shapes;
    int b = params.b;
    int s = params.s;
    int s2 = params.s2;
    int n = params.n;
    int h = params.h;
    int qLoraRank = params.q_lora_rank;
    int qkNopeHeadDim = params.qk_nope_head_dim;
    int qkRopeHeadDim = params.qk_rope_head_dim;
    int kvLoraRank = params.kv_lora_rank;
    int q_head_dim = params.q_head_dim;

    shapes.x = {b, s, h};
    shapes.wDq = {h, qLoraRank};
    shapes.wUqQr = {qLoraRank, n * q_head_dim};
    shapes.wDkvKr = {h, kvLoraRank + qkRopeHeadDim};
    shapes.wUk = {n, qkNopeHeadDim, kvLoraRank};
    shapes.cos = {b, s, qkRopeHeadDim};
    shapes.gammaCq = {qLoraRank};
    shapes.gammaCkv = {kvLoraRank};
    shapes.kvLen = {b, s};
    shapes.kvCache = {b, 1, s2, kvLoraRank};
    shapes.krCache = {b, 1, s2, qkRopeHeadDim};
    if (params.cacheMode != "BNSD") {
        int blockNum = b * (s2 / params.blockSize);
        shapes.kvCache = {blockNum, params.blockSize, 1, kvLoraRank};
        shapes.krCache = {blockNum, params.blockSize, 1, qkRopeHeadDim};
    }
    shapes.wQbScale = {1, n * q_head_dim};
    shapes.smoothCq = {1, qLoraRank};
    shapes.qOut = {b, s, n, kvLoraRank};
    shapes.qRopeOut = {b, s, n, qkRopeHeadDim};
    shapes.kvCacheOut = {b, 1, s2, kvLoraRank};
    shapes.krCacheOut = {b, 1, s2, qkRopeHeadDim};
    return shapes;
}

template <typename T, typename wDtype, bool nz, bool usePrefetch>
static MlaPrologTensors CreateMlaPrologTensors(const SimpleParams &params, const MlaPrologShapes& shapes) {
    MlaPrologTensors tensors;
    int b = params.b;

    DataType dType = (std::is_same<T, npu::tile_fwk::float16>::value) ? DT_FP16 : DT_BF16;
    bool isQuant = std::is_same<wDtype, int8_t>::value;
    DataType dTypeQuant = isQuant ? DT_INT8 : dType;

    TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;

    tensors.x = Tensor(dType, shapes.x, "x");
    tensors.wDq = Tensor(dType, shapes.wDq, "wDq", weightFormat);
    tensors.wUqQr = Tensor(dTypeQuant, shapes.wUqQr, "wUqQr", weightFormat);
    if constexpr (usePrefetch) {
        tensors.wDq.SetCachePolicy(CachePolicy::PREFETCH, true);
        tensors.wUqQr.SetCachePolicy(CachePolicy::PREFETCH, true);
    }
    tensors.wDkvKr = Tensor(dType, shapes.wDkvKr, "wDkvKr", weightFormat);
    tensors.wUk = Tensor(dType, shapes.wUk, "wUk", weightFormat);
    tensors.gammaCq = Tensor(dType, shapes.gammaCq, "gamma_cq");
    tensors.gammaCkv = Tensor(dType, shapes.gammaCkv, "gamma_ckv");
    tensors.cos = Tensor(dType, shapes.cos, "cos");
    tensors.sin = Tensor(dType, shapes.cos, "sin");
    tensors.kvLen = Tensor(DT_INT64, shapes.kvLen, "kv_len");
    tensors.kvCache = Tensor(dType, shapes.kvCache, "kv_cache");
    tensors.krCache = Tensor(dType, shapes.krCache, "kr_cache");
    tensors.wQbScale = Tensor(DT_FP32, shapes.wQbScale, "w_qb_scale");
    tensors.smoothCq = Tensor(DT_FP32, shapes.smoothCq, "smooth_cq");
    tensors.outputKvCache = Tensor(dType, shapes.kvCache, "output_kv_cache");
    tensors.outputKrCache = Tensor(dType, shapes.krCache, "output_kr_cache");
    tensors.outputQ = Tensor(dType, shapes.qOut, "output_q");
    tensors.outputQRope = Tensor(dType, shapes.qRopeOut, "output_q_rope");
    return tensors;
}

template <typename T, typename wDtype>
static void ReadMlaPrologData(const MlaPrologShapes& shapes, const MlaPrologTensors& tensors,
                              std::vector<T>& golden1, std::vector<T>& golden2,
                              std::vector<T>& golden3, std::vector<T>& golden4,
                              std::vector<RawTensorDataPtr>& inputDataList,
                              std::vector<RawTensorDataPtr>& outputDataList,
                              std::vector<RawTensorDataPtr>& goldenDataList)
{
    auto xData = CreateTensorData<T>(tensors.x, shapes.x, "/x.bin");
    auto wDqData = CreateTensorData<T>(tensors.wDq, shapes.wDq, "/wDq.bin");
    auto wUqQrData = CreateTensorData<wDtype>(tensors.wUqQr, shapes.wUqQr, "/wUqQr.bin");
    auto wUkData = CreateTensorData<T>(tensors.wUk, shapes.wUk, "/wUk.bin");
    auto wDkvKrData = CreateTensorData<T>(tensors.wDkvKr, shapes.wDkvKr, "/wDkvKr.bin");
    auto gammaCqData = CreateTensorData<T>(tensors.gammaCq, shapes.gammaCq, "/gamma_cq.bin");
    auto gammaCkvData = CreateTensorData<T>(tensors.gammaCkv, shapes.gammaCkv, "/gamma_ckv.bin");
    auto cosData = CreateTensorData<T>(tensors.cos, shapes.cos, "/cos.bin");
    auto sinData = CreateTensorData<T>(tensors.sin, shapes.cos, "/sin.bin");
    auto kvLenData = CreateTensorData<int64_t>(tensors.kvLen, shapes.kvLen, "/kv_len.bin");
    auto kvCacheData = CreateTensorData<T>(tensors.kvCache, shapes.kvCache, "/kv_cache.bin");
    auto krCacheData = CreateTensorData<T>(tensors.krCache, shapes.krCache, "/kr_cache.bin");
    auto wQbScaleData = CreateTensorData<float>(tensors.wQbScale, shapes.wQbScale, "/w_qb_scale.bin");
    auto smoothCqData = CreateTensorData<float>(tensors.smoothCq, shapes.smoothCq, "/smooth_cq.bin");
    auto outputQData = RawTensorData::CreateConstantTensor<T>(tensors.outputQ, 0.0);
    auto outputQRopeData = RawTensorData::CreateConstantTensor<T>(tensors.outputQRope, 0.0);

    golden1 = getGoldenVec<T>(shapes.qOut, "/q_golden.bin");
    golden2 = getGoldenVec<T>(shapes.qRopeOut, "/q_rope_golden.bin");
    golden3 = getGoldenVec<T>(shapes.kvCache, "/kv_cache_golden.bin");
    golden4 = getGoldenVec<T>(shapes.krCache, "/kr_cache_golden.bin");

    auto golden1Data = CreateTensorData<T>(tensors.outputQ, shapes.qOut, "/q_golden.bin");
    auto golden2Data = CreateTensorData<T>(tensors.outputQRope, shapes.qRopeOut, "/q_rope_golden.bin");
    auto golden3Data = CreateTensorData<T>(tensors.kvCache, shapes.kvCache, "/kv_cache_golden.bin");
    auto golden4Data = CreateTensorData<T>(tensors.krCache, shapes.krCache, "/kr_cache_golden.bin");

    inputDataList = {xData, wDqData, wUqQrData, wUkData, wDkvKrData, gammaCqData, gammaCkvData,
                    sinData, cosData, kvLenData, kvCacheData, krCacheData, wQbScaleData, smoothCqData};
    outputDataList = {outputQData, outputQRopeData, kvCacheData, krCacheData};
    goldenDataList = {golden1Data, golden2Data, golden3Data, golden4Data};
}

template <typename T, typename wDtype, bool isQuant, bool isSmooth>
static void PrepareMlaPrologQuantInputs(const MlaPrologTensors& tensors, MlaQuantInputs& quantInputs)
{
    if (isQuant) {
        quantInputs.dequantScaleWUqQr = tensors.wQbScale;
        if (isSmooth) {
            quantInputs.smoothScalesCq = tensors.smoothCq;
        }
    }
}

template <typename T, typename wDtype, bool splitK, bool isSmooth>
static void ExecuteMlaPrologAndVerify(const MlaPrologTensors& tensors, const SimpleParams &params,
                                      const std::vector<T>& golden1, const std::vector<T>& golden2,
                                      const std::vector<T>& golden3, const std::vector<T>& golden4,
                                      const MlaQuantInputs& quantInputs,
                                      const std::vector<RawTensorDataPtr>& inputDataList,
                                      const std::vector<RawTensorDataPtr>& outputDataList)
{
    int b = params.b;

    RoPETileShapeConfigNew ropeConfig{
        {b, 1, 64},
        {b, 1, 1, 64},
        {b, 1, 1, 64},
        {b, 1, 1, 32, 2}
    };

    config::SetPassConfig("PVC2_OOO", "InferMemoryConflict", KEY_DISABLE_PASS, true);
    MlaProlog(tensors.x, tensors.wDq, tensors.wUqQr, tensors.wUk, tensors.wDkvKr,
        tensors.gammaCq, tensors.gammaCkv, tensors.sin, tensors.cos, tensors.kvLen,
        tensors.kvCache, tensors.krCache, quantInputs, ropeConfig, tensors.outputQ,
        tensors.outputQRope, tensors.outputKvCache, tensors.outputKrCache, 1e-5f, 1e-5f,
        params.cacheMode, splitK, isSmooth);

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), inputDataList, outputDataList);
    std::cout << "qNope ====== " << std::endl;
    EXPECT_TRUE(resultCmp<T>(golden1, (T *)outputDataList[0]->data(), 0.008f, 16));
    std::cout << "qRope ======" << std::endl;
    EXPECT_TRUE(resultCmp<T>(golden2, (T *)outputDataList[1]->data(), 0.005f, 16));
    std::cout << "kv ====== " << std::endl;
    EXPECT_TRUE(resultCmp<T>(golden3, (T *)outputDataList[2]->data(), 0.003f, 16));
    std::cout << "kr ====== " << std::endl;
    EXPECT_TRUE(resultCmp<T>(golden4, (T *)outputDataList[3]->data(), 0.003f, 16));
#endif
}

template <typename T = npu::tile_fwk::float16, typename wDtype = int8_t, bool splitK = false, bool nz = true,
    bool isSmooth = true, bool usePrefetch = true>
void TestMlaPrologV2(const SimpleParams &params) {
    SetInterpreterConfig();
    pre();

    MlaPrologShapes shapes = CalculateMlaPrologShapes<T, wDtype>(params);
    MlaPrologTensors tensors = CreateMlaPrologTensors<T, wDtype, nz, usePrefetch>(params, shapes);

    bool isQuant = std::is_same<wDtype, int8_t>::value;
    std::vector<T> golden1, golden2, golden3, golden4;
    std::vector<RawTensorDataPtr> inputDataList, outputDataList, goldenDataList;
    ReadMlaPrologData<T, wDtype>(shapes, tensors, golden1, golden2, golden3, golden4, inputDataList, outputDataList, goldenDataList);

    MlaQuantInputs quantInputs;
    PrepareMlaPrologQuantInputs<T, wDtype, isQuant, isSmooth>(tensors, quantInputs);

    ProgramData::GetInstance().PrepareData(inputDataList, outputDataList, goldenDataList);

    ExecuteMlaPrologAndVerify<T, wDtype, splitK, isSmooth>(tensors, params, golden1, golden2, golden3, golden4,
                                                            quantInputs, inputDataList, outputDataList);
}

TEST_F(DyMla, low) {
    // verifyConfig();
    performanceConfig();
    TestMlaPrologV2<npu::tile_fwk::float16>(SimpleParams::getLowParams());
}

TEST_F(DyMla, low_PA_BSND) {
    // verifyConfig();
    performanceConfig();
    npu::tile_fwk::SimpleParams params = SimpleParams::getLowParams();
    params.cacheMode = "PA_BSND";
    TestMlaPrologV2<npu::tile_fwk::float16>(params);
}

TEST_F(DyMla, low_PA_NZ) {
    // verifyConfig();
    performanceConfig();
    npu::tile_fwk::SimpleParams params = SimpleParams::getLowParams();
    params.cacheMode = "PA_NZ";
    TestMlaPrologV2<npu::tile_fwk::float16>(params);
}

TEST_F(DyMla, low_bf) {
    performanceConfig();
    TestMlaPrologV2<npu::tile_fwk::bfloat16>(SimpleParams::getLowParams());
}

TEST_F(DyMla, high) {
    performanceConfig();
    TestMlaPrologV2<npu::tile_fwk::float16>(SimpleParams::getHighParams());
}

TEST_F(DyMla, high_PA_NZ) {
    performanceConfig();
    npu::tile_fwk::SimpleParams params = SimpleParams::getHighParams();
    params.cacheMode = "PA_NZ";
    TestMlaPrologV2<npu::tile_fwk::float16>(params);
}

} // namespace
