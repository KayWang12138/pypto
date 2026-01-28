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
struct AttentionTestParams {
    int b;
    int s;
    int s2;
    int n;
    int h;
    int qLoraRank;
    int qkNopeHeadDim;
    int qkRopeHeadDim;
    int kvLoraRank;
    int vHeadDim;
    int blockSize;
    int q_head_dim;
    int maxSeqAllBatch;
    int maxBlockNumPerBatch;
    int blockNum;
    float softmaxScale;
    std::vector<int> atcSeqs;
};

void SetupRuntimeConfig() {
    config::SetRuntimeOption(DEVICE_SCHED_MODE, static_cast<uint8_t>(MachineScheduleConfig::L2CACHE_AFFINITY_SCH));
    config::SetRuntimeOption(STITCH_FUNCTION_NUM_INITIAL, 128);
}

AttentionTestParams ParseParams(const std::vector<int> &params, const std::string &dataPath) {
    AttentionTestParams parsed;

    parsed.b = params[0];
    parsed.s = params[1];
    parsed.s2 = params[2];
    parsed.n = params[3];
    parsed.h = params[4];
    parsed.qLoraRank = params[5];
    parsed.qkNopeHeadDim = params[6];
    parsed.qkRopeHeadDim = params[7];
    parsed.kvLoraRank = params[8];
    parsed.vHeadDim = params[9];
    parsed.blockSize = params[10];
    parsed.q_head_dim = parsed.qkNopeHeadDim + parsed.qkRopeHeadDim;

    parsed.atcSeqs.resize(parsed.b);
    readInput<int>(dataPath + "/actual_seq_len.bin", parsed.atcSeqs);

    parsed.blockNum = 0;
    for (auto seq : parsed.atcSeqs) {
        parsed.blockNum += CeilDiv(seq, parsed.blockSize);
    }

    parsed.softmaxScale = static_cast<float>(1.0 / sqrtf((parsed.kvLoraRank + parsed.qkRopeHeadDim)));
    parsed.maxSeqAllBatch = *(std::max_element(parsed.atcSeqs.begin(), parsed.atcSeqs.end()));
    parsed.maxBlockNumPerBatch = CeilDiv(parsed.maxSeqAllBatch, parsed.blockSize);

    return parsed;
}

DataType GetDtype(bool isQuant, DataType baseDtype) {
    return isQuant ? DT_INT8 : baseDtype;
}

template <typename T>
DataType GetTemplateDtype() {
    if (std::is_same<T, npu::tile_fwk::float16>::value) {
        return DT_FP16;
    } else if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        return DT_BF16;
    }
    return DT_FP32;
}

template <typename T, typename wDtype, bool usePrefetch>
struct AttentionTestTensors {
    // Input tensors
    Tensor x, wDq, wUqQr, wDkvKr, wUk, gamma_cq, gamma_ckv, cos, sin, kv_len, kv_cache, kr_cache, w_qb_scale, smooth_cq;
    // Output tensors
    Tensor output_q, output_q_rope, output_kv_cache, output_kr_cache;
    // PA tensors
    Tensor blockTable, actSeqs, paOut;
    // Post tensors
    Tensor weightUV, weightO, weightOScaleW, postOut;
    RoPETileShapeConfigNew ropeConfig;
};

struct AttentionTestCapacities {
    int64_t capacity_x, wDqCapacity, wUqQrCapacity, wDkvKrCapacity, wUkCapacity;
    int64_t capacity_cos, capacity_gamma_cq, capacity_gamma_ckv, capacity_kv_len;
    int64_t capacity_kv_cache, capacity_kr_cache, capacity_w_qb_scale, capacity_smooth_cq;
    int64_t capacity_q_out, capacity_q_rope_out, capacity_fake_out, capacity_fake_out1;
};

AttentionTestCapacities DefineTensorShapesAndCapacities(const AttentionTestParams &p, const std::string &cacheMode,
    std::vector<int64_t> &kv_cache_shape, std::vector<int64_t> &kr_cache_shape) {
    AttentionTestCapacities caps;

    kv_cache_shape = {p.b, 1, p.s2, p.kvLoraRank};
    kr_cache_shape = {p.b, 1, p.s2, p.qkRopeHeadDim};
    if (cacheMode != "BNSD") {
        kv_cache_shape = {p.blockNum, p.blockSize, 1, p.kvLoraRank};
        kr_cache_shape = {p.blockNum, p.blockSize, 1, p.qkRopeHeadDim};
    }

    std::vector<int64_t> x_shape = {p.b, p.s, p.h};
    std::vector<int64_t> w_qa_shape = {p.h, p.qLoraRank};
    std::vector<int64_t> w_qb_shape = {p.qLoraRank, p.n * p.q_head_dim};
    std::vector<int64_t> w_kv_a_shape = {p.h, p.kvLoraRank + p.qkRopeHeadDim};
    std::vector<int64_t> w_kv_b_k_shape = {p.n, p.qkNopeHeadDim, p.kvLoraRank};
    std::vector<int64_t> cos_shape = {p.b, p.s, p.qkRopeHeadDim};
    std::vector<int64_t> gamma_cq_shape = {p.qLoraRank};
    std::vector<int64_t> gamma_ckv_shape = {p.kvLoraRank};
    std::vector<int64_t> kv_len_shape = {p.b, p.s};
    std::vector<int64_t> w_qb_scale_shape = {1, p.n * p.q_head_dim};
    std::vector<int64_t> smooth_cq_shape = {1, p.qLoraRank};
    std::vector<int64_t> q_out_shape = {p.b, p.s, p.n, p.kvLoraRank};
    std::vector<int64_t> q_rope_out_shape = {p.b, p.s, p.n, p.qkRopeHeadDim};
    std::vector<int64_t> fake_out_shape = {p.b, p.s, p.kvLoraRank + p.qkRopeHeadDim};
    std::vector<int64_t> fake_out_shape1 = {p.n, p.b * p.s, p.qkNopeHeadDim};

    auto calc_capacity = [](const std::vector<int64_t> &shape) {
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    };

    caps.capacity_x = calc_capacity(x_shape);
    caps.wDqCapacity = calc_capacity(w_qa_shape);
    caps.wUqQrCapacity = calc_capacity(w_qb_shape);
    caps.wDkvKrCapacity = calc_capacity(w_kv_a_shape);
    caps.wUkCapacity = calc_capacity(w_kv_b_k_shape);
    caps.capacity_cos = calc_capacity(cos_shape);
    caps.capacity_gamma_cq = calc_capacity(gamma_cq_shape);
    caps.capacity_gamma_ckv = calc_capacity(gamma_ckv_shape);
    caps.capacity_kv_len = calc_capacity(kv_len_shape);
    caps.capacity_kv_cache = calc_capacity(kv_cache_shape);
    caps.capacity_kr_cache = calc_capacity(kr_cache_shape);
    caps.capacity_w_qb_scale = calc_capacity(w_qb_scale_shape);
    caps.capacity_smooth_cq = calc_capacity(smooth_cq_shape);
    caps.capacity_q_out = calc_capacity(q_out_shape);
    caps.capacity_q_rope_out = calc_capacity(q_rope_out_shape);
    caps.capacity_fake_out = calc_capacity(fake_out_shape);
    caps.capacity_fake_out1 = calc_capacity(fake_out_shape1);

    return caps;
}

template <typename T, typename wDtype, bool usePrefetch>
void CreateAllTensors(const AttentionTestParams &p, DataType dType, DataType dTypeQuantIn,
    TileOpFormat weightFormat, TileOpFormat paFormat, bool isQuant, bool isSmooth,
    const std::string &cacheMode, const std::vector<int64_t> &kv_cache_shape,
    const std::vector<int64_t> &kr_cache_shape, AttentionTestTensors<T, wDtype, usePrefetch> &tensors) {

    std::vector<int64_t> x_shape = {p.b, p.s, p.h};
    std::vector<int64_t> w_qa_shape = {p.h, p.qLoraRank};
    std::vector<int64_t> w_qb_shape = {p.qLoraRank, p.n * p.q_head_dim};
    std::vector<int64_t> w_kv_a_shape = {p.h, p.kvLoraRank + p.qkRopeHeadDim};
    std::vector<int64_t> w_kv_b_k_shape = {p.n, p.qkNopeHeadDim, p.kvLoraRank};
    std::vector<int64_t> cos_shape = {p.b, p.s, p.qkRopeHeadDim};
    std::vector<int64_t> gamma_cq_shape = {p.qLoraRank};
    std::vector<int64_t> gamma_ckv_shape = {p.kvLoraRank};
    std::vector<int64_t> kv_len_shape = {p.b, p.s};
    std::vector<int64_t> w_qb_scale_shape = {1, p.n * p.q_head_dim};
    std::vector<int64_t> smooth_cq_shape = {1, p.qLoraRank};

    tensors.x = Tensor(dType, x_shape, "x");
    tensors.wDq = Tensor(dType, w_qa_shape, "wDq", weightFormat);
    tensors.wUqQr = Tensor(dTypeQuantIn, w_qb_shape, "wUqQr", weightFormat);
    if constexpr (usePrefetch) {
        tensors.wDq.SetCachePolicy(CachePolicy::PREFETCH, true);
        tensors.wUqQr.SetCachePolicy(CachePolicy::PREFETCH, true);
    }
    tensors.wDkvKr = Tensor(dType, w_kv_a_shape, "wDkvKr", weightFormat);
    tensors.wUk = Tensor(dType, w_kv_b_k_shape, "wUk", weightFormat);
    tensors.gamma_cq = Tensor(dType, gamma_cq_shape, "gamma_cq");
    tensors.gamma_ckv = Tensor(dType, gamma_ckv_shape, "gamma_ckv");
    tensors.cos = Tensor(dType, cos_shape, "cos");
    tensors.sin = Tensor(dType, cos_shape, "sin");
    tensors.kv_len = Tensor(DT_INT64, kv_len_shape, "kv_len");
    tensors.kv_cache = Tensor(dType, kv_cache_shape, "kv_cache", paFormat);
    tensors.kr_cache = Tensor(dType, kr_cache_shape, "kr_cache", paFormat);
    if (isQuant) {
        tensors.w_qb_scale = Tensor(DT_FP32, w_qb_scale_shape, "w_qb_scale");
        if (isSmooth) {
            tensors.smooth_cq = Tensor(DT_FP32, smooth_cq_shape, "smooth_cq");
        }
    }

    tensors.output_q = Tensor(dType, {p.b*p.s*p.n, p.kvLoraRank}, "output_q");
    tensors.output_q_rope = Tensor(dType, {p.b*p.s*p.n, p.qkRopeHeadDim}, "output_q_rope");
    tensors.output_kv_cache = Tensor(dType, {p.b * 1 * p.s2, p.kvLoraRank}, "output_kv_cache", paFormat);
    tensors.output_kr_cache = Tensor(dType, {p.b * 1 * p.s2, p.qkRopeHeadDim}, "output_kr_cache", paFormat);

    tensors.blockTable = Tensor(DT_INT32, {p.b, p.maxBlockNumPerBatch}, "blockTable");
    tensors.actSeqs = Tensor(DT_INT32, {p.b}, "actSeqs");
    tensors.paOut = Tensor(DT_FP32, {p.b * p.n * p.s, p.kvLoraRank}, "paOut");

    tensors.weightUV = Tensor(dType, {p.n, p.kvLoraRank, p.vHeadDim}, "weightUV");
    tensors.weightUV.SetCachePolicy(CachePolicy::NONE_CACHEABLE, true);
    tensors.weightO = Tensor(DT_INT8, {p.n * p.vHeadDim, p.h}, "weightO", weightFormat);
    tensors.weightO.SetCachePolicy(CachePolicy::NONE_CACHEABLE, true);
    tensors.weightOScaleW = Tensor(DT_FP32, {1, p.h}, "weightOScaleW");
    tensors.weightOScaleW.SetCachePolicy(CachePolicy::NONE_CACHEABLE, true);

    tensors.postOut = Tensor(dType, x_shape, "postOut");
    tensors.postOut.SetCachePolicy(CachePolicy::NONE_CACHEABLE, true);

    int tileB = p.b;
    tensors.ropeConfig = RoPETileShapeConfigNew {
        {tileB, 1, 64},
        {tileB, 1, 1, 64},
        {tileB, 1, 1, 64},
        {tileB, 1, 1, 32, 2}
    };
}

template <typename T, typename wDtype>
struct AttentionTestDataVectors {
    std::vector<T> xValue, wDqValue, wUkValue, wDkvKrValue, gammaCqValue, gammaCkvValue, sinValue, cosValue;
    std::vector<wDtype> wUqQrValue;
    std::vector<int64_t> kvLenValue;
    std::vector<T> kvCacheValue, krCacheValue;
    std::vector<float> wQbScaleValue, smoothCqValue;
    std::vector<int32_t> blockTableValue, actSeqsValue;
    std::vector<T> weightUVValue;
    std::vector<int8_t> weightOValue;
    std::vector<float> weightOScaleWValue;
    std::vector<T> q_golden, q_rope_golden, kv_cache_golden, kr_cache_golden, golden5, golden6;
    std::vector<float> atten_out_golden;
    std::vector<T> attn_output_golden;
};

template <typename T, typename wDtype>
void CreateAllDataVectors(const AttentionTestParams &p, const AttentionTestCapacities &caps,
    AttentionTestDataVectors<T, wDtype> &data) {
    data.xValue.resize(caps.capacity_x);
    data.wDqValue.resize(caps.wDqCapacity);
    data.wUqQrValue.resize(caps.wUqQrCapacity);
    data.wUkValue.resize(caps.wUkCapacity);
    data.wDkvKrValue.resize(caps.wDkvKrCapacity);
    data.gammaCqValue.resize(caps.capacity_gamma_cq);
    data.gammaCkvValue.resize(caps.capacity_gamma_ckv);
    data.sinValue.resize(caps.capacity_cos);
    data.cosValue.resize(caps.capacity_cos);
    data.kvLenValue.resize(caps.capacity_kv_len);
    data.kvCacheValue.resize(caps.capacity_kv_cache);
    data.krCacheValue.resize(caps.capacity_kr_cache);
    data.wQbScaleValue.resize(caps.capacity_w_qb_scale);
    data.smoothCqValue.resize(caps.capacity_smooth_cq);
    data.blockTableValue.resize(p.b * p.maxBlockNumPerBatch);
    data.actSeqsValue.resize(p.b * p.s2);
    data.weightUVValue.resize(p.n * p.kvLoraRank * p.vHeadDim);
    data.weightOValue.resize(p.n * p.vHeadDim * p.h);
    data.weightOScaleWValue.resize(p.h);
    data.q_golden.resize(caps.capacity_q_out);
    data.q_rope_golden.resize(caps.capacity_q_rope_out);
    data.kv_cache_golden.resize(caps.capacity_kv_cache);
    data.kr_cache_golden.resize(caps.capacity_kr_cache);
    data.golden5.resize(caps.capacity_fake_out);
    data.golden6.resize(caps.capacity_fake_out1);
    data.atten_out_golden.resize(p.b * p.n * p.s * p.kvLoraRank);
    data.attn_output_golden.resize(caps.capacity_x);
}

template <typename T, typename wDtype>
void LoadAllData(const std::string &dataPath, const AttentionTestParams &p, bool isQuant, bool isSmooth,
    AttentionTestDataVectors<T, wDtype> &data) {
    readInput<T>(dataPath + "/x.bin", data.xValue);
    readInput<T>(dataPath + "/wDq.bin", data.wDqValue);
    readInput<wDtype>(dataPath + "/wUqQr.bin", data.wUqQrValue);
    readInput<T>(dataPath + "/wUk.bin", data.wUkValue);
    readInput<T>(dataPath + "/wDkvKr.bin", data.wDkvKrValue);
    readInput<T>(dataPath + "/gamma_cq.bin", data.gammaCqValue);
    readInput<T>(dataPath + "/gamma_ckv.bin", data.gammaCkvValue);
    readInput<T>(dataPath + "/sin.bin", data.sinValue);
    readInput<T>(dataPath + "/cos.bin", data.cosValue);
    readInput<int64_t>(dataPath + "/kv_len.bin", data.kvLenValue);
    readInput<T>(dataPath + "/kv_cache.bin", data.kvCacheValue);
    readInput<T>(dataPath + "/kr_cache.bin", data.krCacheValue);
    if (isQuant) {
        readInput<float>(dataPath + "/w_qb_scale.bin", data.wQbScaleValue);
        if (isSmooth) {
            readInput<float>(dataPath + "/smooth_cq.bin", data.smoothCqValue);
        }
    }
    readInput<int32_t>(dataPath + "/block_table.bin", data.blockTableValue);
    readInput<int32_t>(dataPath + "/actual_seq_len.bin", data.actSeqsValue);
    readInput<T>(dataPath + "/w_uv.bin", data.weightUVValue);
    readInput<int8_t>(dataPath + "/w_o.bin", data.weightOValue);
    readInput<float>(dataPath + "/w_o_scale_w.bin", data.weightOScaleWValue);
    readInput<T>(dataPath + "/q_golden.bin", data.q_golden);
    readInput<T>(dataPath + "/q_rope_golden.bin", data.q_rope_golden);
    readInput<T>(dataPath + "/kv_cache_golden.bin", data.kv_cache_golden);
    readInput<T>(dataPath + "/kr_cache_golden.bin", data.kr_cache_golden);
    readInput<float>(dataPath + "/atten_out.bin", data.atten_out_golden);
    readInput<T>(dataPath + "/attn_output.bin", data.attn_output_golden);
}

template <typename T, typename wDtype, bool usePrefetch>
struct AttentionTestTensorData {
    std::shared_ptr<RawTensorData> xData, wDqData, wUqQrData, wUkData, wDkvKrData;
    std::shared_ptr<RawTensorData> gammaCqData, gammaCkvData, cosData, sinData, kvLenData;
    std::shared_ptr<RawTensorData> kvCacheData, krCacheData, wQbScaleData, smoothCqData;
    std::shared_ptr<RawTensorData> outputQData, outputQRopeData;
    std::shared_ptr<RawTensorData> blockTableData, actSeqsData, paOutData;
    std::shared_ptr<RawTensorData> weightUVData, weightOData, weightOScaleWData, postOutData;
};

template <typename T, typename wDtype>
void CreateAllTensorData(const AttentionTestTensors<T, wDtype, false> &tensors,
    const AttentionTestDataVectors<T, wDtype> &data, bool isQuant, bool isSmooth,
    AttentionTestTensorData<T, wDtype, false> &tensorData) {
    tensorData.xData = RawTensorData::CreateTensor<T>(tensors.x, data.xValue);
    tensorData.wDqData = RawTensorData::CreateTensor<T>(tensors.wDq, data.wDqValue);
    tensorData.wUqQrData = RawTensorData::CreateTensor<wDtype>(tensors.wUqQr, data.wUqQrValue);
    tensorData.wUkData = RawTensorData::CreateTensor<T>(tensors.wUk, data.wUkValue);
    tensorData.wDkvKrData = RawTensorData::CreateTensor<T>(tensors.wDkvKr, data.wDkvKrValue);
    tensorData.gammaCqData = RawTensorData::CreateTensor<T>(tensors.gamma_cq, data.gammaCqValue);
    tensorData.gammaCkvData = RawTensorData::CreateTensor<T>(tensors.gamma_ckv, data.gammaCkvValue);
    tensorData.cosData = RawTensorData::CreateTensor<T>(tensors.cos, data.cosValue);
    tensorData.sinData = RawTensorData::CreateTensor<T>(tensors.sin, data.sinValue);
    tensorData.kvLenData = RawTensorData::CreateTensor<int64_t>(tensors.kv_len, data.kvLenValue);
    tensorData.kvCacheData = RawTensorData::CreateTensor<T>(tensors.kv_cache, data.kvCacheValue);
    tensorData.krCacheData = RawTensorData::CreateTensor<T>(tensors.kr_cache, data.krCacheValue);

    if (isQuant) {
        tensorData.wQbScaleData = RawTensorData::CreateTensor<float>(tensors.w_qb_scale, data.wQbScaleValue);
        if (isSmooth) {
            tensorData.smoothCqData = RawTensorData::CreateTensor<float>(tensors.smooth_cq, data.smoothCqValue);
        }
    }

    tensorData.outputQData = RawTensorData::CreateConstantTensor<T>(tensors.output_q, 0.0);
    tensorData.outputQRopeData = RawTensorData::CreateConstantTensor<T>(tensors.output_q_rope, 0.0);
    tensorData.blockTableData = RawTensorData::CreateTensor<int32_t>(tensors.blockTable, data.blockTableValue);
    tensorData.actSeqsData = RawTensorData::CreateTensor<int32_t>(tensors.actSeqs, data.actSeqsValue);
    tensorData.paOutData = RawTensorData::CreateConstantTensor<float>(tensors.paOut, 0.0);
    tensorData.weightUVData = RawTensorData::CreateTensor<T>(tensors.weightUV, data.weightUVValue);
    tensorData.weightOData = RawTensorData::CreateTensor<int8_t>(tensors.weightO, data.weightOValue);
    tensorData.weightOScaleWData = RawTensorData::CreateTensor<float>(tensors.weightOScaleW, data.weightOScaleWValue);
    tensorData.postOutData = RawTensorData::CreateConstantTensor<T>(tensors.postOut, 0.0);
}

template <typename T, typename wDtype>
void RunAndVerify(const AttentionTestTensors<T, wDtype, false> &tensors,
    const AttentionTestDataVectors<T, wDtype> &data, const AttentionTestTensorData<T, wDtype, false> &tensorData,
    const AttentionTestParams &p, bool isQuant, bool isSmooth, PaTileShapeConfig &paTileConfig,
    const std::string &cacheMode) {
    MlaQuantInputs quantInputs;
    if (isQuant) {
        quantInputs.dequantScaleWUqQr = tensors.w_qb_scale;
        if (isSmooth) {
            quantInputs.smoothScalesCq = tensors.smooth_cq;
        }
    }

    Attention(tensors.x, tensors.wDq, tensors.wUqQr, tensors.wUk, tensors.wDkvKr,
            tensors.gamma_cq, tensors.gamma_ckv, tensors.sin, tensors.cos, tensors.kv_len,
            tensors.kv_cache, tensors.kr_cache, tensors.output_q, tensors.output_q_rope,
            tensors.output_kv_cache, tensors.output_kr_cache, quantInputs, tensors.ropeConfig,
            tensors.blockTable, tensors.actSeqs, tensors.paOut, p.blockSize, p.softmaxScale,
            paTileConfig, tensors.weightUV, tensors.weightO, tensors.weightOScaleW, tensors.postOut,
            1e-5f, 1e-5f, cacheMode);

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(),
        {tensorData.xData, tensorData.wDqData, tensorData.wUqQrData, tensorData.wUkData,
         tensorData.wDkvKrData, tensorData.gammaCqData, tensorData.gammaCkvData,
         tensorData.sinData, tensorData.cosData, tensorData.kvLenData, tensorData.kvCacheData,
         tensorData.krCacheData, tensorData.wQbScaleData, tensorData.smoothCqData,
         tensorData.blockTableData, tensorData.actSeqsData, tensorData.weightUVData,
         tensorData.weightOData, tensorData.weightOScaleWData}, {tensorData.postOutData});

    std::cout << "====== kvCacheData out: " << std::endl;
    EXPECT_TRUE(resultCmp<T>(data.kv_cache_golden, (T *)tensorData.kvCacheData->data(), 0.001f));
    std::cout << "====== krCacheData out: " << std::endl;
    EXPECT_TRUE(resultCmp<T>(data.kr_cache_golden, (T *)tensorData.krCacheData->data(), 0.001f));
    std::cout << "====== postOutData out: " << std::endl;
    EXPECT_TRUE(resultCmp<T>(data.attn_output_golden, (T *)tensorData.postOutData->data(), 0.03f, 0, 1000, false, true, 0));
#endif
}

template <typename T = npu::tile_fwk::float16, typename wDtype = int8_t, bool splitK = false, bool nz = false, bool usePrefetch = false>
void TestDynamicAttention(std::vector<int> &params, PaTileShapeConfig &paTileConfig, string dataPath,
        uint64_t timeThreshold, bool isQuant = false, bool isSmooth = false) {
    (void) timeThreshold;
    (void) splitK;

    SetupRuntimeConfig();
    std::string cacheMode = "PA_NZ";
    AttentionTestParams p = ParseParams(params, dataPath);

    DataType dType = GetTemplateDtype<T>();
    DataType dTypeQuantIn = GetDtype(isQuant, dType);
    TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    TileOpFormat paFormat = cacheMode == "PA_NZ" ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;

    std::vector<int64_t> kv_cache_shape, kr_cache_shape;
    AttentionTestCapacities caps = DefineTensorShapesAndCapacities(p, cacheMode, kv_cache_shape, kr_cache_shape);

    AttentionTestTensors<T, wDtype, usePrefetch> tensors;
    CreateAllTensors(p, dType, dTypeQuantIn, weightFormat, paFormat, isQuant, isSmooth, cacheMode,
                     kv_cache_shape, kr_cache_shape, tensors);

    AttentionTestDataVectors<T, wDtype> data;
    CreateAllDataVectors(p, caps, data);
    LoadAllData(dataPath, p, isQuant, isSmooth, data);

    AttentionTestTensorData<T, wDtype, usePrefetch> tensorData;
    CreateAllTensorData(tensors, data, isQuant, isSmooth, tensorData);

    RunAndVerify(tensors, data, tensorData, p, isQuant, isSmooth, paTileConfig, cacheMode);
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
