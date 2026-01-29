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
 * \file test_dynamic_attention.cpp
 * \brief
 */

#include "test_dev_func_runner.h"
#include "test_suite_stest_ops.h"
#include "operator/models/deepseek/attention.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class DynamicAttention : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

namespace {

struct AttentionShapes {
    std::vector<int64_t> x;
    std::vector<int64_t> wQa;
    std::vector<int64_t> wQb;
    std::vector<int64_t> wKvA;
    std::vector<int64_t> wKvBK;
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
    std::vector<int64_t> fakeOut;
    std::vector<int64_t> fakeOut1;
};

struct AttentionCapacities {
    int x;
    int wDq;
    int wUqQr;
    int wDkvKr;
    int wUk;
    int cos;
    int gammaCq;
    int gammaCkv;
    int kvLen;
    int kvCache;
    int krCache;
    int wQbScale;
    int smoothCq;
    int qOut;
    int qRopeOut;
    int fakeOut;
    int fakeOut1;
};

static void CalculateAttentionShapes(int b, int s, int s2, int n, int h, int qLoraRank, int qkNopeHeadDim,
    int qkRopeHeadDim, int kvLoraRank, int vHeadDim, int blockSize, int blockNum, int maxBlockNumPerBatch,
    bool isQuant, int qHeadDim, const std::string& cacheMode, AttentionShapes& shapes)
{
    shapes.x = {b, s, h};
    shapes.wQa = {h, qLoraRank};
    shapes.wQb = {qLoraRank, n * qHeadDim};
    shapes.wKvA = {h, kvLoraRank + qkRopeHeadDim};
    shapes.wKvBK = {n, qkNopeHeadDim, kvLoraRank};
    shapes.cos = {b, s, qkRopeHeadDim};
    shapes.gammaCq = {qLoraRank};
    shapes.gammaCkv = {kvLoraRank};
    shapes.kvLen = {b, s};
    shapes.kvCache = {b, 1, s2, kvLoraRank};
    shapes.krCache = {b, 1, s2, qkRopeHeadDim};
    if (cacheMode != "BNSD") {
        shapes.kvCache = {blockNum, blockSize, 1, kvLoraRank};
        shapes.krCache = {blockNum, blockSize, 1, qkRopeHeadDim};
    }
    shapes.wQbScale = {1, n * qHeadDim};
    shapes.smoothCq = {1, qLoraRank};
    shapes.qOut = {b, s, n, kvLoraRank};
    shapes.qRopeOut = {b, s, n, qkRopeHeadDim};
    shapes.kvCacheOut = {b, 1, s2, kvLoraRank};
    shapes.krCacheOut = {b, 1, s2, qkRopeHeadDim};
    shapes.fakeOut = {b, s, kvLoraRank + qkRopeHeadDim};
    shapes.fakeOut1 = {n, b * s, qkNopeHeadDim};
}

static void CalculateAttentionCapacities(const AttentionShapes& shapes, AttentionCapacities& capacities)
{
    capacities.x = std::accumulate(shapes.x.begin(), shapes.x.end(), 1, std::multiplies<>());
    capacities.wDq = std::accumulate(shapes.wQa.begin(), shapes.wQa.end(), 1, std::multiplies<>());
    capacities.wUqQr = std::accumulate(shapes.wQb.begin(), shapes.wQb.end(), 1, std::multiplies<>());
    capacities.wDkvKr = std::accumulate(shapes.wKvA.begin(), shapes.wKvA.end(), 1, std::multiplies<>());
    capacities.wUk = std::accumulate(shapes.wKvBK.begin(), shapes.wKvBK.end(), 1, std::multiplies<>());
    capacities.cos = std::accumulate(shapes.cos.begin(), shapes.cos.end(), 1, std::multiplies<>());
    capacities.gammaCq = std::accumulate(shapes.gammaCq.begin(), shapes.gammaCq.end(), 1, std::multiplies<>());
    capacities.gammaCkv = std::accumulate(shapes.gammaCkv.begin(), shapes.gammaCkv.end(), 1, std::multiplies<>());
    capacities.kvLen = std::accumulate(shapes.kvLen.begin(), shapes.kvLen.end(), 1, std::multiplies<>());
    capacities.kvCache = std::accumulate(shapes.kvCache.begin(), shapes.kvCache.end(), 1, std::multiplies<>());
    capacities.krCache = std::accumulate(shapes.krCache.begin(), shapes.krCache.end(), 1, std::multiplies<>());
    capacities.wQbScale = std::accumulate(shapes.wQbScale.begin(), shapes.wQbScale.end(), 1, std::multiplies<>());
    capacities.smoothCq = std::accumulate(shapes.smoothCq.begin(), shapes.smoothCq.end(), 1, std::multiplies<>());
    capacities.qOut = std::accumulate(shapes.qOut.begin(), shapes.qOut.end(), 1, std::multiplies<>());
    capacities.qRopeOut = std::accumulate(shapes.qRopeOut.begin(), shapes.qRopeOut.end(), 1, std::multiplies<>());
    capacities.fakeOut = std::accumulate(shapes.fakeOut.begin(), shapes.fakeOut.end(), 1, std::multiplies<>());
    capacities.fakeOut1 = std::accumulate(shapes.fakeOut1.begin(), shapes.fakeOut1.end(), 1, std::multiplies<>());
}

template <typename T, typename wDtype, bool nz, bool usePrefetch>
static void CreateAttentionTensors(DataType dType, DataType dTypeQuantIn, const AttentionShapes& shapes,
    int b, int s, int s2, int n, int h, int kvLoraRank, int qkRopeHeadDim, int maxBlockNumPerBatch, int blockSize,
    int vHeadDim, bool isQuant, const std::string& cacheMode,
    Tensor& x, Tensor& wDq, Tensor& wUqQr, Tensor& wDkvKr, Tensor& wUk,
    Tensor& gamma_cq, Tensor& gamma_ckv, Tensor& cos, Tensor& sin, Tensor& kv_len,
    Tensor& kv_cache, Tensor& kr_cache, Tensor& w_qb_scale, Tensor& smooth_cq,
    Tensor& output_q, Tensor& output_q_rope, Tensor& output_kv_cache, Tensor& output_kr_cache,
    Tensor& blockTable, Tensor& actSeqs, Tensor& paOut,
    Tensor& weightUV, Tensor& weightO, Tensor& weightOScaleW, Tensor& postOut)
{
    TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    TileOpFormat paFormat = cacheMode == "PA_NZ" ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;

    x = Tensor(dType, shapes.x, "x");
    wDq = Tensor(dType, shapes.wQa, "wDq", weightFormat);
    wUqQr = Tensor(dTypeQuantIn, shapes.wQb, "wUqQr", weightFormat);
    if constexpr (usePrefetch) {
        wDq.SetCachePolicy(CachePolicy::PREFETCH, true);
        wUqQr.SetCachePolicy(CachePolicy::PREFETCH, true);
    }
    wDkvKr = Tensor(dType, shapes.wKvA, "wDkvKr", weightFormat);
    wUk = Tensor(dType, shapes.wKvBK, "wUk", weightFormat);
    gamma_cq = Tensor(dType, shapes.gammaCq, "gamma_cq");
    gamma_ckv = Tensor(dType, shapes.gammaCkv, "gamma_ckv");
    cos = Tensor(dType, shapes.cos, "cos");
    sin = Tensor(dType, shapes.cos, "sin");
    kv_len = Tensor(DT_INT64, shapes.kvLen, "kv_len");
    kv_cache = Tensor(dType, shapes.kvCache, "kv_cache", paFormat);
    kr_cache = Tensor(dType, shapes.krCache, "kr_cache", paFormat);

    if (isQuant) {
        w_qb_scale = Tensor(DT_FP32, shapes.wQbScale, "w_qb_scale");
        smooth_cq = Tensor(DT_FP32, shapes.smoothCq, "smooth_cq");
    }

    output_q = Tensor(dType, {b*s*n, kvLoraRank}, "output_q");
    output_q_rope = Tensor(dType, {b*s*n, qkRopeHeadDim}, "output_q_rope");
    output_kv_cache = Tensor(dType, {b * 1 * s2, kvLoraRank}, "output_kv_cache", paFormat);
    output_kr_cache = Tensor(dType, {b * 1 * s2, qkRopeHeadDim}, "output_kr_cache", paFormat);

    blockTable = Tensor(DT_INT32, {b, maxBlockNumPerBatch}, "blockTable");
    actSeqs = Tensor(DT_INT32, {b}, "actSeqs");
    paOut = Tensor(DT_FP32, {b * n * s, kvLoraRank}, "paOut");

    weightUV = Tensor(dType, {n, kvLoraRank, vHeadDim}, "weightUV");
    weightUV.SetCachePolicy(CachePolicy::NONE_CACHEABLE, true);
    weightO = Tensor(DT_INT8, {n * vHeadDim, h}, "weightO", weightFormat);
    weightO.SetCachePolicy(CachePolicy::NONE_CACHEABLE, true);
    weightOScaleW = Tensor(DT_FP32, {1, h}, "weightOScaleW");
    weightOScaleW.SetCachePolicy(CachePolicy::NONE_CACHEABLE, true);
    postOut = Tensor(dType, shapes.x, "postOut");
    postOut.SetCachePolicy(CachePolicy::NONE_CACHEABLE, true);
}

template <typename T, typename wDtype>
static void ReadAttentionData(const std::string& dataPath, const AttentionCapacities& capacities,
    int b, int n, int kvLoraRank, int vHeadDim, int h, int maxBlockNumPerBatch, int s2, bool isQuant,
    std::vector<T>& xValue, std::vector<T>& wDqValue, std::vector<wDtype>& wUqQrValue,
    std::vector<T>& wUkValue, std::vector<T>& wDkvKrValue, std::vector<T>& gammaCqValue,
    std::vector<T>& gammaCkvValue, std::vector<T>& sinValue, std::vector<T>& cosValue,
    std::vector<int64_t>& kvLenValue, std::vector<T>& kvCacheValue, std::vector<T>& krCacheValue,
    std::vector<float>& wQbScaleValue, std::vector<float>& smoothCqValue,
    std::vector<int32_t>& blockTableValue, std::vector<int32_t>& actSeqsValue,
    std::vector<T>& weightUVValue, std::vector<int8_t>& weightOValue, std::vector<float>& weightOScaleWValue)
{
    xValue.resize(capacities.x);
    wDqValue.resize(capacities.wDq);
    wUqQrValue.resize(capacities.wUqQr);
    wUkValue.resize(capacities.wUk);
    wDkvKrValue.resize(capacities.wDkvKr);
    gammaCqValue.resize(capacities.gammaCq);
    gammaCkvValue.resize(capacities.gammaCkv);
    sinValue.resize(capacities.cos);
    cosValue.resize(capacities.cos);
    kvLenValue.resize(capacities.kvLen);
    kvCacheValue.resize(capacities.kvCache);
    krCacheValue.resize(capacities.krCache);
    wQbScaleValue.resize(capacities.wQbScale);
    smoothCqValue.resize(capacities.smoothCq);

    blockTableValue.resize(b * maxBlockNumPerBatch);
    actSeqsValue.resize(b);
    weightUVValue.resize(n * kvLoraRank * vHeadDim);
    weightOValue.resize(n * vHeadDim * h);
    weightOScaleWValue.resize(h);

    readInput<T>(dataPath + "/x.bin", xValue);
    readInput<T>(dataPath + "/wDq.bin", wDqValue);
    readInput<wDtype>(dataPath + "/wUqQr.bin", wUqQrValue);
    readInput<T>(dataPath + "/wUk.bin", wUkValue);
    readInput<T>(dataPath + "/wDkvKr.bin", wDkvKrValue);
    readInput<T>(dataPath + "/gamma_cq.bin", gammaCqValue);
    readInput<T>(dataPath + "/gamma_ckv.bin", gammaCkvValue);
    readInput<T>(dataPath + "/sin.bin", sinValue);
    readInput<T>(dataPath + "/cos.bin", cosValue);
    readInput<int64_t>(dataPath + "/kv_len.bin", kvLenValue);
    readInput<T>(dataPath + "/kv_cache.bin", kvCacheValue);
    readInput<T>(dataPath + "/kr_cache.bin", krCacheValue);

    if (isQuant) {
        readInput<float>(dataPath + "/w_qb_scale.bin", wQbScaleValue);
        readInput<float>(dataPath + "/smooth_cq.bin", smoothCqValue);
    }

    readInput<int32_t>(dataPath + "/block_table.bin", blockTableValue);
    readInput<int32_t>(dataPath + "/actual_seq_len.bin", actSeqsValue);
    readInput<T>(dataPath + "/w_uv.bin", weightUVValue);
    readInput<int8_t>(dataPath + "/w_o.bin", weightOValue);
    readInput<float>(dataPath + "/w_o_scale_w.bin", weightOScaleWValue);
}

template <typename T>
static void ReadAttentionGoldenData(const std::string& dataPath, const AttentionCapacities& capacities,
    int b, int n, int s, int kvLoraRank, int capacityX,
    std::vector<T>& qGolden, std::vector<T>& qRopeGolden, std::vector<T>& kvCacheGolden,
    std::vector<T>& krCacheGolden, std::vector<float>& attenOutGolden, std::vector<T>& attnOutputGolden)
{
    qGolden.resize(capacities.qOut);
    qRopeGolden.resize(capacities.qRopeOut);
    kvCacheGolden.resize(capacities.kvCache);
    krCacheGolden.resize(capacities.krCache);
    attenOutGolden.resize(b * n * s * kvLoraRank);
    attnOutputGolden.resize(capacityX);

    readInput<T>(dataPath + "/q_golden.bin", qGolden);
    readInput<T>(dataPath + "/q_rope_golden.bin", qRopeGolden);
    readInput<T>(dataPath + "/kv_cache_golden.bin", kvCacheGolden);
    readInput<T>(dataPath + "/kr_cache_golden.bin", krCacheGolden);
    readInput<float>(dataPath + "/atten_out.bin", attenOutGolden);
    readInput<T>(dataPath + "/attn_output.bin", attnOutputGolden);
}

template <typename T, typename wDtype>
static void CreateAttentionTensorData(const Tensor& x, const Tensor& wDq, const Tensor& wUqQr,
    const Tensor& wUk, const Tensor& wDkvKr, const Tensor& gamma_cq, const Tensor& gamma_ckv,
    const Tensor& cos, const Tensor& sin, const Tensor& kv_len, const Tensor& kv_cache, const Tensor& kr_cache,
    const Tensor& w_qb_scale, const Tensor& smooth_cq, const Tensor& blockTable, const Tensor& actSeqs,
    const Tensor& weightUV, const Tensor& weightO, const Tensor& weightOScaleW, const Tensor& postOut,
    const std::vector<T>& xValue, const std::vector<T>& wDqValue, const std::vector<wDtype>& wUqQrValue,
    const std::vector<T>& wUkValue, const std::vector<T>& wDkvKrValue, const std::vector<T>& gammaCqValue,
    const std::vector<T>& gammaCkvValue, const std::vector<T>& sinValue, const std::vector<T>& cosValue,
    const std::vector<int64_t>& kvLenValue, const std::vector<T>& kvCacheValue, const std::vector<T>& krCacheValue,
    const std::vector<float>& wQbScaleValue, const std::vector<float>& smoothCqValue,
    const std::vector<int32_t>& blockTableValue, const std::vector<int32_t>& actSeqsValue,
    const std::vector<T>& weightUVValue, const std::vector<int8_t>& weightOValue,
    const std::vector<float>& weightOScaleWValue, bool isQuant, bool isSmooth,
    std::vector<std::shared_ptr<RawTensorData>>& inputDataList, std::shared_ptr<RawTensorData>& postOutData)
{
    auto xData = RawTensorData::CreateTensor<T>(x, xValue);
    auto wDqData = RawTensorData::CreateTensor<T>(wDq, wDqValue);
    auto wUqQrData = RawTensorData::CreateTensor<wDtype>(wUqQr, wUqQrValue);
    auto wUkData = RawTensorData::CreateTensor<T>(wUk, wUkValue);
    auto wDkvKrData = RawTensorData::CreateTensor<T>(wDkvKr, wDkvKrValue);
    auto gammaCqData = RawTensorData::CreateTensor<T>(gamma_cq, gammaCqValue);
    auto gammaCkvData = RawTensorData::CreateTensor<T>(gamma_ckv, gammaCkvValue);
    auto cosData = RawTensorData::CreateTensor<T>(cos, cosValue);
    auto sinData = RawTensorData::CreateTensor<T>(sin, sinValue);
    auto kvLenData = RawTensorData::CreateTensor<int64_t>(kv_len, kvLenValue);
    auto kvCacheData = RawTensorData::CreateTensor<T>(kv_cache, kvCacheValue);
    auto krCacheData = RawTensorData::CreateTensor<T>(kr_cache, krCacheValue);

    std::shared_ptr<RawTensorData> wQbScaleData;
    std::shared_ptr<RawTensorData> smoothCqData;
    if (isQuant) {
        wQbScaleData = RawTensorData::CreateTensor<float>(w_qb_scale, wQbScaleValue);
        if (isSmooth) {
            smoothCqData = RawTensorData::CreateTensor<float>(smooth_cq, smoothCqValue);
        }
    }

    auto blockTableData = RawTensorData::CreateTensor<int32_t>(blockTable, blockTableValue);
    auto actSeqsData = RawTensorData::CreateTensor<int32_t>(actSeqs, actSeqsValue);
    auto weightUVData = RawTensorData::CreateTensor<T>(weightUV, weightUVValue);
    auto weightOData = RawTensorData::CreateTensor<int8_t>(weightO, weightOValue);
    auto weightOScaleWData = RawTensorData::CreateTensor<float>(weightOScaleW, weightOScaleWValue);
    postOutData = RawTensorData::CreateConstantTensor<T>(postOut, 0.0);

    inputDataList = {xData, wDqData, wUqQrData, wUkData, wDkvKrData, gammaCqData, gammaCkvData,
                     sinData, cosData, kvLenData, kvCacheData, krCacheData, wQbScaleData, smoothCqData,
                     blockTableData, actSeqsData, weightUVData, weightOData, weightOScaleWData};
}

template <typename T = npu::tile_fwk::float16, typename wDtype = int8_t, bool splitK = false, bool nz = false, bool usePrefetch = false>
void TestDynamicAttention(std::vector<int> &params, PaTileShapeConfig &paTileConfig, string dataPath,
        uint64_t timeThreshold, bool isQuant = false, bool isSmooth = false) {
    (void) timeThreshold;

    config::SetRuntimeOption(DEVICE_SCHED_MODE, static_cast<uint8_t>(MachineScheduleConfig::L2CACHE_AFFINITY_SCH));
    config::SetRuntimeOption(STITCH_FUNCTION_NUM_INITIAL, 128);
    std::string cacheMode = "PA_NZ";

    int b = params[0];
    int s = params[1];
    int s2 = params[2];
    int n = params[3];
    int h = params[4];
    int qLoraRank = params[5];
    int qkNopeHeadDim = params[6];
    int qkRopeHeadDim = params[7];
    int kvLoraRank = params[8];
    int vHeadDim = params[9];
    int blockSize = params[10];
    int qHeadDim = qkNopeHeadDim + qkRopeHeadDim;

    std::vector<int> atcSeqs(b);
    readInput<int>(dataPath + "/actual_seq_len.bin", atcSeqs);

    int blockNum = 0;
    for (auto seq : atcSeqs) {
        blockNum += CeilDiv(seq, blockSize);
    }

    float softmaxScale = static_cast<float>(1.0 / sqrtf((kvLoraRank + qkRopeHeadDim)));
    int maxSeqAllBatch = *(std::max_element(atcSeqs.begin(), atcSeqs.end()));
    int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);

    DataType dType = DT_FP32;
    if (std::is_same<T, npu::tile_fwk::float16>::value) {
        dType = DT_FP16;
    } else if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        dType = DT_BF16;
    } else {
        dType = DT_FP32;
    }
    DataType dTypeQuantIn = isQuant ? DT_INT8 : dType;

    AttentionShapes shapes;
    CalculateAttentionShapes(b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank,
        vHeadDim, blockSize, blockNum, maxBlockNumPerBatch, isQuant, qHeadDim, cacheMode, shapes);

    AttentionCapacities capacities;
    CalculateAttentionCapacities(shapes, capacities);

    Tensor x, wDq, wUqQr, wDkvKr, wUk, gamma_cq, gamma_ckv, cos, sin, kv_len;
    Tensor kv_cache, kr_cache, w_qb_scale, smooth_cq;
    Tensor output_q, output_q_rope, output_kv_cache, output_kr_cache;
    Tensor blockTable, actSeqs, paOut;
    Tensor weightUV, weightO, weightOScaleW, postOut;

    CreateAttentionTensors<T, wDtype, nz, usePrefetch>(dType, dTypeQuantIn, shapes, b, s, s2, n, h,
        kvLoraRank, qkRopeHeadDim, maxBlockNumPerBatch, blockSize, vHeadDim, isQuant, cacheMode,
        x, wDq, wUqQr, wDkvKr, wUk, gamma_cq, gamma_ckv, cos, sin, kv_len,
        kv_cache, kr_cache, w_qb_scale, smooth_cq, output_q, output_q_rope, output_kv_cache, output_kr_cache,
        blockTable, actSeqs, paOut, weightUV, weightO, weightOScaleW, postOut);

    std::vector<T> xValue, wDqValue, wUkValue, wDkvKrValue, gammaCqValue, gammaCkvValue;
    std::vector<wDtype> wUqQrValue;
    std::vector<T> sinValue, cosValue, kvCacheValue, krCacheValue, weightUVValue;
    std::vector<int64_t> kvLenValue;
    std::vector<float> wQbScaleValue, smoothCqValue, weightOScaleWValue;
    std::vector<int32_t> blockTableValue, actSeqsValue;
    std::vector<int8_t> weightOValue;

    ReadAttentionData<T, wDtype>(dataPath, capacities, b, n, kvLoraRank, vHeadDim, h, maxBlockNumPerBatch,
        s2, isQuant, xValue, wDqValue, wUqQrValue, wUkValue, wDkvKrValue, gammaCqValue, gammaCkvValue,
        sinValue, cosValue, kvLenValue, kvCacheValue, krCacheValue, wQbScaleValue, smoothCqValue,
        blockTableValue, actSeqsValue, weightUVValue, weightOValue, weightOScaleWValue);

    std::vector<T> qGolden, qRopeGolden, kvCacheGolden, krCacheGolden, attnOutputGolden;
    std::vector<float> attenOutGolden;
    ReadAttentionGoldenData<T>(dataPath, capacities, b, n, s, kvLoraRank, capacities.x,
        qGolden, qRopeGolden, kvCacheGolden, krCacheGolden, attenOutGolden, attnOutputGolden);

    std::vector<std::shared_ptr<RawTensorData>> inputDataList;
    std::shared_ptr<RawTensorData> postOutData;
    CreateAttentionTensorData<T, wDtype>(x, wDq, wUqQr, wUk, wDkvKr, gamma_cq, gamma_ckv, cos, sin,
        kv_len, kv_cache, kr_cache, w_qb_scale, smooth_cq, blockTable, actSeqs, weightUV, weightO,
        weightOScaleW, postOut, xValue, wDqValue, wUqQrValue, wUkValue, wDkvKrValue, gammaCqValue,
        gammaCkvValue, sinValue, cosValue, kvLenValue, kvCacheValue, krCacheValue, wQbScaleValue,
        smoothCqValue, blockTableValue, actSeqsValue, weightUVValue, weightOValue, weightOScaleWValue,
        isQuant, isSmooth, inputDataList, postOutData);

    int tileB = b;
    RoPETileShapeConfigNew ropeConfig {
        {tileB, 1, 64},
        {tileB, 1, 1, 64},
        {tileB, 1, 1, 64},
        {tileB, 1, 1, 32, 2}
    };

    MlaQuantInputs quantInputs;
    if (isQuant) {
        quantInputs.dequantScaleWUqQr = w_qb_scale;
        if (isSmooth) {
            quantInputs.smoothScalesCq = smooth_cq;
        }
    }

    Attention(x, wDq, wUqQr, wUk, wDkvKr, gamma_cq, gamma_ckv, sin, cos, kv_len, kv_cache, kr_cache,
            output_q, output_q_rope, output_kv_cache, output_kr_cache, quantInputs, ropeConfig,
            blockTable, actSeqs, paOut, blockSize, softmaxScale, paTileConfig,
            weightUV, weightO, weightOScaleW, postOut, 1e-5f, 1e-5f, cacheMode);

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), inputDataList, {postOutData});

    auto kvCacheData = inputDataList[10];
    auto krCacheData = inputDataList[11];

    std::cout << "====== kvCacheData out: " << std::endl;
    EXPECT_TRUE(resultCmp<T>(kvCacheGolden, (T *)kvCacheData->data(), 0.001f));
    std::cout << "====== krCacheData out: " << std::endl;
    EXPECT_TRUE(resultCmp<T>(krCacheGolden, (T *)krCacheData->data(), 0.001f));
    std::cout << "====== postOutData out: " << std::endl;
    EXPECT_TRUE(resultCmp<T>(attnOutputGolden, (T *)postOutData->data(), 0.03f, 0, 1000, false, true, 0));
#endif
}

TEST_F(DynamicAttention, dynamic_attention_low) { // b_n_s_s2_h_q_lora_rank
    int b = 4;
    int s = 1;
    int s2 = 256;
    int h = 7168;   //7168
    int n = 32;
    int qLoraRank = 1536;   //1536
    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;
    int kvLoraRank = 512;
    int vHeadDim = 128;
    int blockSize = 256;
    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, vHeadDim, blockSize};

    const bool splitK = false;
    const bool nz = false;

    PaTileShapeConfig tileConfig;
    const int nTile = 32;
    tileConfig.headNumQTile = nTile;
    tileConfig.v0TileShape = {nTile, 64};
    tileConfig.c1TileShape = {nTile, nTile, 64, 64, 128, 128};
    tileConfig.v1TileShape = {nTile, 64};
    tileConfig.c2TileShape = {nTile, nTile, 64, 64, 128, 128};
    tileConfig.v2TileShape = {nTile, 64};

    TestDynamicAttention<npu::tile_fwk::float16, npu::tile_fwk::float16, splitK, nz>(params, tileConfig, GetGoldenDir(), 10000, false);
}

TEST_F(DynamicAttention, dynamic_attention_high) { // b_n_s_s2_h_q_lora_rank
    int b = 32;
    int s = 1;
    int s2 = 4096;
    int h = 7168;
    int n = 128;
    int qLoraRank = 1536;
    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;
    int kvLoraRank = 512;
    int vHeadDim = 128;
    int blockSize = 4096;
    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, vHeadDim, blockSize};

    const bool splitK = false;
    const bool nz = false;

    PaTileShapeConfig paTileConfig;
    const int nTile = 128;
    paTileConfig.headNumQTile = nTile;

    paTileConfig.c1TileShape = {nTile, nTile, 128, 256, 128, 256};
    paTileConfig.v1TileShape = {16, 256};
    paTileConfig.c2TileShape = {nTile, nTile, 256, 256, 128, 128};
    paTileConfig.v2TileShape = {16, 256};

    TestDynamicAttention<npu::tile_fwk::float16, npu::tile_fwk::float16, splitK, nz>(params, paTileConfig, GetGoldenDir(), 10000, false);
}

TEST_F(DynamicAttention, low_latency_quant_smooth_nz) { // b_n_s_s2_h_q_lora_rank
    int b = 4;
    int s = 1;
    int s2 = 256;
    int h = 7168;
    int n = 32;
    int qLoraRank = 1536;
    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;
    int kvLoraRank = 512;
    int vHeadDim = 128;
    int blockSize = 256;
    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, vHeadDim, blockSize};

    const bool splitK = false;
    const bool nz = true;
    const bool isQuant = true;
    const bool isSmooth = true;

    PaTileShapeConfig paTileConfig;
    const int nTile = 32;
    paTileConfig.headNumQTile = nTile;
    paTileConfig.v0TileShape = {nTile, 64};
    paTileConfig.c1TileShape = {nTile, nTile, 64, 64, 128, 128};
    paTileConfig.v1TileShape = {nTile, 64};
    paTileConfig.c2TileShape = {nTile, nTile, 64, 64, 128, 128};
    paTileConfig.v2TileShape = {nTile, 64};

    TestDynamicAttention<npu::tile_fwk::float16, int8_t, splitK, nz>(params, paTileConfig, GetGoldenDir(), 10000,
            isQuant, isSmooth);
}

TEST_F(DynamicAttention, high_throughput_quant_smooth_nz) { // b_n_s_s2_h_q_lora_rank
    int b = 32;
    int s = 1;
    int s2 = 4096;
    int h = 7168;
    int n = 128;
    int qLoraRank = 1536;
    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;
    int kvLoraRank = 512;
    int vHeadDim = 128;
    int blockSize = 4096;
    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, vHeadDim, blockSize};

    const bool splitK = false;
    const bool nz = true;
    const bool isQuant = true;
    const bool isSmooth = true;

    PaTileShapeConfig paTileConfig;
    const int nTile = 128;
    paTileConfig.headNumQTile = nTile;

    paTileConfig.c1TileShape = {nTile, nTile, 128, 256, 128, 256};
    paTileConfig.v1TileShape = {16, 256};
    paTileConfig.c2TileShape = {nTile, nTile, 256, 256, 128, 128};
    paTileConfig.v2TileShape = {16, 256};

    // Set Stitching window optimization
    config::SetRuntimeOption<int>(STITCH_FUNCTION_INNER_MEMORY, 11);
    config::SetRuntimeOption<int>(STITCH_FUNCTION_OUTCAST_MEMORY, 32);

    TestDynamicAttention<npu::tile_fwk::float16, int8_t, splitK, nz>(params, paTileConfig, GetGoldenDir(), 10000,
            isQuant, isSmooth);
}

} // namespace
