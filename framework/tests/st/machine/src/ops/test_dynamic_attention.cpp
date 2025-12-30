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
template <typename T = npu::tile_fwk::float16, typename wDtype = int8_t, bool splitK = false, bool nz = false, bool usePrefetch = false>


DataType GetAttentionDataType(bool isQuant) {
    if (std::is_same<T, npu::tile_fwk::float16>::value) {
        return DT_FP16;
    } else if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        return DT_BF16;
    } else {
        return DT_FP32;
    }
}


struct AttentionShapes {
    std::vector<int64_t> x_shape;
    std::vector<int64_t> w_qa_shape;
    std::vector<int64_t> w_qb_shape;
    std::vector<int64_t> w_kv_a_shape;
    std::vector<int64_t> w_kv_b_k_shape;
    std::vector<int64_t> cos_shape;
    std::vector<int64_t> gamma_cq_shape;
    std::vector<int64_t> gamma_ckv_shape;
    std::vector<int64_t> kv_len_shape;
    std::vector<int64_t> kv_cache_shape;
    std::vector<int64_t> kr_cache_shape;
    std::vector<int64_t> w_qb_scale_shape;
    std::vector<int64_t> smooth_cq_shape;
    std::vector<int64_t> blockTableShape;
    std::vector<int64_t> q_out_shape;
    std::vector<int64_t> q_rope_out_shape;
    std::vector<int64_t> kv_cache_out_shape;
    std::vector<int64_t> kr_cache_out_shape;
    std::vector<int64_t> fake_out_shape;
    std::vector<int64_t> fake_out_shape1;
};


struct AttentionCapacities {
    int x;
    int wDq;
    int wUqQr;
    int wDkvKr;
    int wUk;
    int cos;
    int gamma_cq;
    int gamma_ckv;
    int kv_len;
    int kv_cache;
    int kr_cache;
    int w_qb_scale;
    int smooth_cq;
    int q_out;
    int q_rope_out;
    int fake_out;
    int fake_out1;
    int block_table;
    int act_seqs;
    int weight_uv;
    int weight_o;
    int weight_oscale_w;
};


template<typename T>
struct AttentionDataValues {
    std::vector<T> xValue;
    std::vector<T> wDqValue;
    std::vector<T> wUkValue;
    std::vector<T> wDkvKrValue;
    std::vector<T> gammaCqValue;
    std::vector<T> gammaCkvValue;
    std::vector<T> sinValue;
    std::vector<T> cosValue;
    std::vector<T> kvCacheValue;
    std::vector<T> krCacheValue;
    std::vector<float> wQbScaleValue;
    std::vector<float> smoothCqValue;
    std::vector<int32_t> blockTableValue;
    std::vector<int32_t> actSeqsValue;
    std::vector<T> weightUVValue;
    std::vector<int8_t> weightOValue;
    std::vector<float> weightOScaleWValue;
    std::vector<wDtype> wUqQrValue;
    std::vector<int64_t> kvLenValue;
};


AttentionShapes DefineAttentionShapes(int b, int s, int s2, int n, int h, int qLoraRank,
                                      int qkNopeHeadDim, int qkRopeHeadDim, int kvLoraRank,
                                      const std::string& cacheMode, int blockNum, int blockSize, int q_head_dim) {
    AttentionShapes shapes;
    shapes.x_shape = {b, s, h};
    shapes.w_qa_shape = {h, qLoraRank};
    shapes.w_qb_shape = {qLoraRank, n * q_head_dim};
    shapes.w_kv_a_shape = {h, kvLoraRank + qkRopeHeadDim};
    shapes.w_kv_b_k_shape = {n, qkNopeHeadDim, kvLoraRank};
    shapes.cos_shape = {b, s, qkRopeHeadDim};
    shapes.gamma_cq_shape = {qLoraRank};
    shapes.gamma_ckv_shape = {kvLoraRank};
    shapes.kv_len_shape = {b, s};
    shapes.kv_cache_shape = {b, 1, s2, kvLoraRank};
    shapes.kr_cache_shape = {b, 1, s2, qkRopeHeadDim};
    if (cacheMode != "BNSD") {
        shapes.kv_cache_shape = {blockNum, blockSize, 1, kvLoraRank};
        shapes.kr_cache_shape = {blockNum, blockSize, 1, qkRopeHeadDim};
    }
    shapes.w_qb_scale_shape = {1, n * q_head_dim};
    shapes.smooth_cq_shape = {1, qLoraRank};
    shapes.blockTableShape = {b, 1, s2, qkRopeHeadDim};
    shapes.q_out_shape = {b, s, n, kvLoraRank};
    shapes.q_rope_out_shape = {b, s, n, qkRopeHeadDim};
    shapes.kv_cache_out_shape = {b, 1, s2, kvLoraRank};
    shapes.kr_cache_out_shape = {b, 1, s2, qkRopeHeadDim};
    shapes.fake_out_shape = {b, s, kvLoraRank + qkRopeHeadDim};
    shapes.fake_out_shape1 = {n, b * s, qkNopeHeadDim};
    return shapes;
}


AttentionCapacities CalculateAttentionCapacities(const AttentionShapes& shapes) {
    AttentionCapacities caps;
    caps.x = std::accumulate(shapes.x_shape.begin(), shapes.x_shape.end(), 1, std::multiplies<>());
    caps.wDq = std::accumulate(shapes.w_qa_shape.begin(), shapes.w_qa_shape.end(), 1, std::multiplies<>());
    caps.wUqQr = std::accumulate(shapes.w_qb_shape.begin(), shapes.w_qb_shape.end(), 1, std::multiplies<>());
    caps.wDkvKr = std::accumulate(shapes.w_kv_a_shape.begin(), shapes.w_kv_a_shape.end(), 1, std::multiplies<>());
    caps.wUk = std::accumulate(shapes.w_kv_b_k_shape.begin(), shapes.w_kv_b_k_shape.end(), 1, std::multiplies<>());
    caps.cos = std::accumulate(shapes.cos_shape.begin(), shapes.cos_shape.end(), 1, std::multiplies<>());
    caps.gamma_cq = std::accumulate(shapes.gamma_cq_shape.begin(), shapes.gamma_cq_shape.end(), 1, std::multiplies<>());
    caps.gamma_ckv = std::accumulate(shapes.gamma_ckv_shape.begin(), shapes.gamma_ckv_shape.end(), 1, std::multiplies<>());
    caps.kv_len = std::accumulate(shapes.kv_len_shape.begin(), shapes.kv_len_shape.end(), 1, std::multiplies<>());
    caps.kv_cache = std::accumulate(shapes.kv_cache_shape.begin(), shapes.kv_cache_shape.end(), 1, std::multiplies<>());
    caps.kr_cache = std::accumulate(shapes.kr_cache_shape.begin(), shapes.kr_cache_shape.end(), 1, std::multiplies<>());
    caps.w_qb_scale = std::accumulate(shapes.w_qb_scale_shape.begin(), shapes.w_qb_scale_shape.end(), 1, std::multiplies<>());
    caps.smooth_cq = std::accumulate(shapes.smooth_cq_shape.begin(), shapes.smooth_cq_shape.end(), 1, std::multiplies<>());
    caps.q_out = std::accumulate(shapes.q_out_shape.begin(), shapes.q_out_shape.end(), 1, std::multiplies<>());
    caps.q_rope_out = std::accumulate(shapes.q_rope_out_shape.begin(), shapes.q_rope_out_shape.end(), 1, std::multiplies<>());
    caps.fake_out = std::accumulate(shapes.fake_out_shape.begin(), shapes.fake_out_shape.end(), 1, std::multiplies<>());
    caps.fake_out1 = std::accumulate(shapes.fake_out_shape1.begin(), shapes.fake_out_shape1.end(), 1, std::multiplies<>());
    caps.block_table = std::accumulate(shapes.blockTableShape.begin(), shapes.blockTableShape.end(), 1, std::multiplies<>());
    caps.act_seqs = shapes.blockTableShape[0] * shapes.blockTableShape[1];
    caps.weight_uv = std::accumulate({static_cast<int64_t>(shapes.fake_out_shape[1])}, 1, std::multiplies<>());
    caps.weight_o = shapes.fake_out_shape1[0] * shapes.fake_out_shape1[1] * shapes.fake_out_shape1[2];
    caps.weight_oscale_w = shapes.w_qb_scale_shape[0];
    return caps;
}


void CreateAttentionTensors(DataType dType, DataType dTypeQuantIn, TileOpFormat weightFormat,
                            TileOpFormat paFormat, const AttentionShapes& shapes,
                            int b, int s, int s2, int n, int kvLoraRank, int qkRopeHeadDim,
                            int vHeadDim, int h, int maxBlockNumPerBatch, int blockSize,
                            bool isQuant, bool isSmooth,
                            std::vector<int64_t>& kv_cache_shape, std::vector<int64_t>& kr_cache_shape,
                            Tensor& x, Tensor& wDq, Tensor& wUqQr, Tensor& wDkvKr, Tensor& wUk,
                            Tensor& gamma_cq, Tensor& gamma_ckv, Tensor& cos, Tensor& sin,
                            Tensor& kv_len, Tensor& kv_cache, Tensor& kr_cache,
                            Tensor& w_qb_scale, Tensor& smooth_cq,
                            Tensor& output_q, Tensor& output_q_rope, Tensor& output_kv_cache, Tensor& output_kr_cache,
                            Tensor& blockTable, Tensor& actSeqs, Tensor& paOut,
                            Tensor& weightUV, Tensor& weightO, Tensor& weightOScaleW, Tensor& postOut) {
    // mla_prolog
    x = Tensor(dType, shapes.x_shape, "x");
    wDq = Tensor(dType, shapes.w_qa_shape, "wDq", weightFormat);
    wUqQr = Tensor(dTypeQuantIn, shapes.w_qb_shape, "wUqQr", weightFormat);
    if constexpr (usePrefetch) {
        wDq.SetCachePolicy(CachePolicy::PREFETCH, true);
        wUqQr.SetCachePolicy(CachePolicy::PREFETCH, true);
    }
    wDkvKr = Tensor(dType, shapes.w_kv_a_shape, "wDkvKr", weightFormat);
    wUk = Tensor(dType, shapes.w_kv_b_k_shape, "wUk", weightFormat);
    gamma_cq = Tensor(dType, shapes.gamma_cq_shape, "gamma_cq");
    gamma_ckv = Tensor(dType, shapes.gamma_ckv_shape, "gamma_ckv");
    cos = Tensor(dType, shapes.cos_shape, "cos");
    sin = Tensor(dType, shapes.cos_shape, "sin");
    kv_len = Tensor(DT_INT64, shapes.kv_len_shape, "kv_len"); // int64
    kv_cache = Tensor(dType, kv_cache_shape, "kv_cache", paFormat);
    kr_cache = Tensor(dType, kr_cache_shape, "kr_cache", paFormat);
    
    if (isQuant) {
        w_qb_scale = Tensor(DT_FP32, shapes.w_qb_scale_shape, "w_qb_scale");
        if (isSmooth) {
            smooth_cq = Tensor(DT_FP32, shapes.smooth_cq_shape, "smooth_cq");
        }
    }

    output_q = Tensor(dType, {b*s*n, kvLoraRank}, "output_q");
    output_q_rope = Tensor(dType, {b*s*n, qkRopeHeadDim}, "output_q_rope");
    output_kv_cache = Tensor(dType, {b * 1 * s2, kvLoraRank}, "output_kv_cache", paFormat);
    output_kr_cache = Tensor(dType, {b * 1 * s2, qkRopeHeadDim}, "output_kr_cache", paFormat);

    // pa
    blockTable = Tensor(DT_INT32, {b, maxBlockNumPerBatch}, "blockTable");
    actSeqs = Tensor(DT_INT32, {b}, "actSeqs");
    // pa output
    paOut = Tensor(DT_FP32, {b * n * s, kvLoraRank}, "paOut");
    // post
    weightUV = Tensor(dType, {n, kvLoraRank, vHeadDim}, "weightUV");
    weightUV.SetCachePolicy(CachePolicy::NONE_CACHEABLE, true);
    weightO = Tensor(DT_INT8, {n * vHeadDim, h}, "weightO", weightFormat); // NZ
    weightO.SetCachePolicy(CachePolicy::NONE_CACHEABLE, true);
    weightOScaleW = Tensor(DT_FP32, {1, h}, "weightOScaleW");
    weightOScaleW.SetCachePolicy(CachePolicy::NONE_CACHEABLE, true);
    // output
    postOut = Tensor(dType, shapes.x_shape, "postOut");
    postOut.SetCachePolicy(CachePolicy::NONE_CACHEABLE, true);
}


template<typename T>
AttentionDataValues<T> InitializeAttentionDataValues(const AttentionCapacities& caps, int b, int s2, int maxBlockNumPerBatch, int kvLoraRank, int qkRopeHeadDim, int vHeadDim, int h) {
    AttentionDataValues<T> values;
    values.xValue = std::vector<T>(caps.x, 0);
    values.wDqValue = std::vector<T>(caps.wDq, 0);
    values.wUkValue = std::vector<T>(caps.wUk, 0);
    values.wDkvKrValue = std::vector<T>(caps.wDkvKr, 0);
    values.gammaCqValue = std::vector<T>(caps.gamma_cq, 0);
    values.gammaCkvValue = std::vector<T>(caps.gamma_ckv, 0);
    values.sinValue = std::vector<T>(caps.cos, 0);
    values.cosValue = std::vector<T>(caps.cos, 0);
    values.kvCacheValue = std::vector<T>(caps.kv_cache, 0);
    values.krCacheValue = std::vector<T>(caps.kr_cache, 0);
    values.wQbScaleValue = std::vector<float>(caps.w_qb_scale, 0);
    values.smoothCqValue = std::vector<float>(caps.smooth_cq, 0);
    values.blockTableValue = std::vector<int32_t>(caps.block_table, 0);
    values.actSeqsValue = std::vector<int32_t>(b, s2);
    values.weightUVValue = std::vector<T>(caps.weight_uv, 0);
    values.weightOValue = std::vector<int8_t>(caps.weight_o, 0);
    values.weightOScaleWValue = std::vector<float>(h, 0);
    values.wUqQrValue = std::vector<int8_t>(caps.wUqQr, 0);
    values.kvLenValue = std::vector<int64_t>(caps.kv_len, 0);
    return values;
}


template<typename T>
void ReadAttentionInputs(const std::string& dataPath, const AttentionShapes& shapes, bool isQuant, bool isSmooth,
                         AttentionDataValues<T>& values, std::vector<int64_t>& kv_lenValue,
                         std::vector<T>& q_golden, std::vector<T>& q_rope_golden,
                         std::vector<T>& kv_cache_golden, std::vector<T>& kr_cache_golden,
                         std::vector<float>& atten_out_golden, std::vector<T>& attn_output_golden) {
    int capacity_q_out = std::accumulate(shapes.q_out_shape.begin(), shapes.q_out_shape.end(), 1, std::multiplies<>());
    int capacity_q_rope_out = std::accumulate(shapes.q_rope_out_shape.begin(), shapes.q_rope_out_shape.end(), 1, std::multiplies<>());
    int capacity_kv_cache = std::accumulate(shapes.kv_cache_out_shape.begin(), shapes.kv_cache_out_shape.end(), 1, std::multiplies<>());
    int capacity_kr_cache = std::accumulate(shapes.kr_cache_out_shape.begin(), shapes.kr_cache_out_shape.end(), 1, std::multiplies<>());
    int capacity_fake_out = std::accumulate(shapes.fake_out_shape.begin(), shapes.fake_out_shape.end(), 1, std::multiplies<>());

    q_golden = std::vector<T>(capacity_q_out, 0);
    q_rope_golden = std::vector<T>(capacity_q_rope_out, 0);
    kv_cache_golden = std::vector<T>(capacity_kv_cache, 0);
    kr_cache_golden = std::vector<T>(capacity_kr_cache, 0);
    atten_out_golden = std::vector<float>(values.xValue.size() * 1 * 1 * 1 * 1, 0);
    attn_output_golden = std::vector<T>(values.xValue.size(), 0);

    readInput<T>(dataPath + "/x.bin", values.xValue);
    readInput<T>(dataPath + "/wDq.bin", values.wDqValue);
    readInput<int8_t>(dataPath + "/wUqQr.bin", values.wUqQrValue);
    readInput<T>(dataPath + "/wUk.bin", values.wUkValue);
    readInput<T>(dataPath + "/wDkvKr.bin", values.wDkvKrValue);
    readInput<T>(dataPath + "/gamma_cq.bin", values.gammaCqValue);
    readInput<T>(dataPath + "/gamma_ckv.bin", values.gammaCkvValue);
    readInput<T>(dataPath + "/sin.bin", values.sinValue);
    readInput<T>(dataPath + "/cos.bin", values.cosValue);
    readInput<int64_t>(dataPath + "/kv_len.bin", kv_lenValue);
    readInput<T>(dataPath + "/kv_cache.bin", values.kvCacheValue);
    readInput<T>(dataPath + "/kr_cache.bin", values.krCacheValue);
    if (isQuant) {
        readInput<float>(dataPath + "/w_qb_scale.bin", values.wQbScaleValue);
        if (isSmooth) {
            readInput<float>(dataPath + "/smooth_cq.bin", values.smoothCqValue);
        }
    }
    readInput<int32_t>(dataPath + "/block_table.bin", values.blockTableValue);
    readInput<int32_t>(dataPath + "/actual_seq_len.bin", values.actSeqsValue);
    readInput<T>(dataPath + "/w_uv.bin", values.weightUVValue);
    readInput<int8_t>(dataPath + "/w_o.bin", values.weightOValue); // NZ
    readInput<float>(dataPath + "/w_o_scale_w.bin", values.weightOScaleWValue);

    readInput<T>(dataPath + "/q_golden.bin", q_golden);
    readInput<T>(dataPath + "/q_rope_golden.bin", q_rope_golden);
    readInput<T>(dataPath + "/kv_cache_golden.bin", kv_cache_golden);
    readInput<T>(dataPath + "/kr_cache_golden.bin", kr_cache_golden);
    readInput<float>(dataPath + "/atten_out.bin", atten_out_golden); // pa out
    readInput<T>(dataPath + "/attn_output.bin", attn_output_golden); // attention out
}


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
    int q_head_dim = qkNopeHeadDim + qkRopeHeadDim;

    std::vector<int> atcSeqs(b);
    readInput<int>(dataPath + "/actual_seq_len.bin", atcSeqs);

    int blockNum = 0;
    for (auto seq : atcSeqs) {
        blockNum += CeilDiv(seq, blockSize);
    }

    float softmaxScale = static_cast<float>(1.0 / sqrtf((kvLoraRank + qkRopeHeadDim)));
    int maxSeqAllBatch = *(std::max_element(atcSeqs.begin(), atcSeqs.end()));
    int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);

    DataType dType = GetAttentionDataType<T>(false);
    DataType dTypeQuantIn = isQuant ? DT_INT8 : dType;

    AttentionShapes shapes = DefineAttentionShapes(b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, cacheMode, blockNum, blockSize, q_head_dim);
    AttentionCapacities caps = CalculateAttentionCapacities(shapes);

    TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    TileOpFormat paFormat = cacheMode == "PA_NZ" ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;

    // Create all tensors
    Tensor x, wDq, wUqQr, wDkvKr, wUk, gamma_cq, gamma_ckv, cos, sin;
    Tensor kv_len, kv_cache, kr_cache, w_qb_scale, smooth_cq;
    Tensor output_q, output_q_rope, output_kv_cache, output_kr_cache;
    Tensor blockTable, actSeqs, paOut;
    Tensor weightUV, weightO, weightOScaleW, postOut;
    
    CreateAttentionTensors(dType, dTypeQuantIn, weightFormat, paFormat, shapes, b, s, s2, n, kvLoraRank, qkRopeHeadDim,
                           vHeadDim, h, maxBlockNumPerBatch, blockSize, isQuant, isSmooth,
                           kv_cache.shape, kr_cache.shape,
                           x, wDq, wUqQr, wDkvKr, wUk, gamma_cq, gamma_ckv, cos, sin,
                           kv_len, kv_cache, kr_cache, w_qb_scale, smooth_cq,
                           output_q, output_q_rope, output_kv_cache, output_kr_cache,
                           blockTable, actSeqs, paOut, weightUV, weightO, weightOScaleW, postOut);

    int tileB = b;
    RoPETileShapeConfigNew ropeConfig {
        {tileB, 1, 64}, // (b,s,d)
        {tileB, 1, 1, 64}, // Q (b,s,n,d)
        {tileB, 1, 1, 64}, // K (b,s,1,d)
        {tileB, 1, 1, 32, 2} // (b,s,n,d//2,2)
    };

    // Initialize data values and read inputs
    AttentionDataValues<T> values = InitializeAttentionDataValues<T>(caps, b, s2, maxBlockNumPerBatch, kvLoraRank, qkRopeHeadDim, vHeadDim, h);
    std::vector<T> q_golden, q_rope_golden, kv_cache_golden, kr_cache_golden, golden5, golden6;
    std::vector<float> atten_out_golden;
    std::vector<T> attn_output_golden;

    ReadAttentionInputs<T>(dataPath, shapes, isQuant, isSmooth, values, values.kvLenValue,
                        q_golden, q_rope_golden, kv_cache_golden, kr_cache_golden, atten_out_golden, attn_output_golden);

    // Create RawTensorData objects
    auto xData = RawTensorData::CreateTensor<T>(x, values.xValue);
    auto wDqData = RawTensorData::CreateTensor<T>(wDq, values.wDqValue);
    auto wUqQrData = RawTensorData::CreateTensor<int8_t>(wUqQr, values.wUqQrValue);
    auto wUkData = RawTensorData::CreateTensor<T>(wUk, values.wUkValue);
    auto wDkvKrData = RawTensorData::CreateTensor<T>(wDkvKr, values.wDkvKrValue);
    auto gammaCqData = RawTensorData::CreateTensor<T>(gamma_cq, values.gammaCqValue);
    auto gammaCkvData = RawTensorData::CreateTensor<T>(gamma_ckv, values.gammaCkvValue);
    auto cosData = RawTensorData::CreateTensor<T>(cos, values.cosValue);
    auto sinData = RawTensorData::CreateTensor<T>(sin, values.sinValue);
    auto kvLenData = RawTensorData::CreateTensor<int64_t>(kv_len, values.kvLenValue);
    auto kvCacheData = RawTensorData::CreateTensor<T>(kv_cache, values.kvCacheValue);
    auto krCacheData = RawTensorData::CreateTensor<T>(kr_cache, values.krCacheValue);
    std::shared_ptr<RawTensorData> wQbScaleData;
    std::shared_ptr<RawTensorData> smoothCqData;
    if (isQuant) {
        wQbScaleData = RawTensorData::CreateTensor<float>(w_qb_scale, values.wQbScaleValue);
        if (isSmooth) {
            smoothCqData = RawTensorData::CreateTensor<float>(smooth_cq, values.smoothCqValue);
        }
    }

    auto outputQData = RawTensorData::CreateConstantTensor<T>(output_q, 0.0);
    auto outputQRopeData = RawTensorData::CreateConstantTensor<T>(output_q_rope, 0.0);
    auto blockTableData = RawTensorData::CreateTensor<int32_t>(blockTable, values.blockTableValue);
    auto actSeqsData = RawTensorData::CreateTensor<int32_t>(actSeqs, values.actSeqsValue);
    auto paOutData = RawTensorData::CreateConstantTensor<float>(paOut, 0.0);
    auto weightUVData = RawTensorData::CreateTensor<T>(weightUV, values.weightUVValue);
    auto weightOData = RawTensorData::CreateTensor<int8_t>(weightO, values.weightOValue);
    auto weightOScaleWData = RawTensorData::CreateTensor<float>(weightOScaleW, values.weightOScaleWValue);
    auto postOutData = RawTensorData::CreateConstantTensor<T>(postOut, 0.0);

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
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(),
        {xData, wDqData, wUqQrData, wUkData, wDkvKrData, gammaCqData, gammaCkvData, sinData, cosData, kvLenData,
         kvCacheData, krCacheData, wQbScaleData, smoothCqData,
         blockTableData, actSeqsData, weightUVData, weightOData, weightOScaleWData},
        {postOutData});

    std::cout << "====== kvCacheData out: " << std::endl;
    EXPECT_TRUE(resultCmp<T>(kv_cache_golden, (T *)kvCacheData->data(), 0.001f));
    std::cout << "====== krCacheData out: " << std::endl;
    EXPECT_TRUE(resultCmp<T>(kr_cache_golden, (T *)krCacheData->data(), 0.001f));
    std::cout << "====== postOutData out: " << std::endl;
    EXPECT_TRUE(resultCmp<T>(attn_output_golden, (T *)postOutData->data(), 0.03f, 0, 1000, false, true, 0));
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
