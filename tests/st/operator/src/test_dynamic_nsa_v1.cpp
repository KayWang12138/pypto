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
 * \file test_dynamic_nsa.cpp
 * \brief
 */

#include "test_dynamic.h"
#include "test_common.h"
#include "test_suite_stest_ops.h"
#include "operator/models/nsa/dynamic_nsa_v1.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class DynamicNSATest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

void SetPreConfig() {
    config::SetHostConfig(KEY_ONLY_CODEGEN, true);
}

template <typename T>
static std::vector<T> getGoldenVec(std::vector<int> shape, std::string fileName) {
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> golden(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, golden);
    return golden;
}

template <typename T>
static std::shared_ptr<RawTensorData> CreateTensorData(Tensor tensor, std::string fileName) {
    auto shape = tensor.GetShape();
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> values(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, values);
    return RawTensorData::CreateTensor<T>(tensor, values);
}

template <typename T = npu::tile_fwk::float16, typename wDtype = int8_t, bool isSmooth = false, bool nz = false>
void TestNsa(const NSASimpleParams &params, const MlaTileConfig &prologConfig, WinAttenTileShapeConfig &winAttntileConfig, SaTileShapeConfig& saTileConfig,
    KvSlcTileShapeConfig& kvSlcTileConfig, PostTileConfig& postConfig, float precision, std::string cacheMode = "PA_BSND") {
    SetPreConfig();

    float eps = params.eps;
    int b = params.b;
    int s1 = params.s1;
    int s2 = params.s2;
    int n1 = params.n1;
    int n2 = params.n2;
    int h = params.h;
    int v_dim = params.kv_lora_rank;
    int qLoraRank = params.q_lora_rank;
    int qkNopeHeadDim = params.qk_nope_head_dim;
    int qkRopeHeadDim = params.qk_rope_head_dim;
    int qHeadDim = qkNopeHeadDim + qkRopeHeadDim;
    int smax = params.topk * params.slcBlockSize;
    int dn = v_dim;
    int dr = params.rope_dim;
    float softmaxScale = static_cast<float>(1.0 / sqrtf((dn + dr)));
    int blockSize = params.blockSize;
    int winSize = params.winSize;
    int slcBlockSize = params.slcBlockSize;
    int front = params.front;
    int near = params.near;
    int topk = params.topk;

    std::vector<int> kvCacheActSeqVec(b);
    readInput<int>(GetGoldenDir() + "/kv_cache_actual_seq_len.bin", kvCacheActSeqVec);
    int blockNum = 0;
    for (auto seqItem : kvCacheActSeqVec) {
        blockNum += CeilDiv(seqItem, blockSize);
    }
    std::cout << "========= blockNum " << blockNum << std::endl;
    int maxSeqAllBatch = *(std::max_element(kvCacheActSeqVec.begin(), kvCacheActSeqVec.end()));
    int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);

    int vHeadDim = params.vHeadDim;
    DataType dType = (std::is_same<T, npu::tile_fwk::float16>::value) ? DT_FP16 : DT_BF16;
    bool isQuant = std::is_same<wDtype, int8_t>::value;
    DataType dTypeQuant = isQuant ? DT_INT8 : dType;

    // 1. 设置shape
    // MlaProlog
    std::vector<int> xShape = {b, s1, h};
    std::vector<int> wDqShape = {h, qLoraRank};
    std::vector<int> wUqQrShape = {qLoraRank, n1 * qHeadDim};
    std::vector<int> wDkvKrShape = {h, v_dim + qkRopeHeadDim};
    std::vector<int> wUkShape = {n1, qkNopeHeadDim, v_dim};
    std::vector<int> cosShape = {b, s1, qkRopeHeadDim};
    std::vector<int> gammaCqShape = {qLoraRank};
    std::vector<int> gammaCkvShape = {v_dim};
    std::vector<int> kvLenShape = {b, s1};
    std::vector<int> kvCacheShape = {b, n2, s2, v_dim};
    std::vector<int> krCacheShape = {b, n2, s2, qkRopeHeadDim};
    std::vector<int> kvCacheOutShape = {b, n2, s2, v_dim};
    std::vector<int> krCacheOutShape = {b, n2, s2, qkRopeHeadDim};
    if (cacheMode != "BNSD") {
        int blockNum2 = b * (s2 / blockSize);
        std::cout << "========= blockNum2 " << blockNum2 << std::endl;
        kvCacheShape = {blockNum, blockSize, n2, v_dim};
        krCacheShape = {blockNum, blockSize, n2, qkRopeHeadDim};
        kvCacheOutShape = {blockNum * blockSize, n2 * v_dim};
        krCacheOutShape = {blockNum * blockSize, n2 * qkRopeHeadDim};
    }
    std::vector<int> wQbScaleShape = {1, n1 * qHeadDim};
    std::vector<int> smoothCqShape{1, qLoraRank};
    std::vector<int> qOutShape = {b, s1, n1, v_dim};
    std::vector<int> qRopeOutShape = {b, s1, n1, qkRopeHeadDim};

    std::vector<int> topkIndicesShape = {b, s1, topk - front - near};
    std::vector<int> topkTensorShapeShape = {b, s1};
    std::vector<int> kvNopeCacheShape = {int(blockNum * blockSize), n2 * dn};
    std::vector<int> kRopeCacheShape = {int(blockNum * blockSize), n2 * dr};
    std::vector<int> kvCacheActSeqShape = {b};
    std::vector<int> blockTableShape = {b, maxBlockNumPerBatch};

    std::vector<int> slcActSeqsShape = {b, s1};
    std::vector<int> qNopeShape = {b * s1 * n1, dn};
    std::vector<int> qRopeShape = {b * s1 * n1, dr};
    std::vector<int> kSlcShape = {b * s1 * n2 * smax, dn + dr};
    std::vector<int> vSlcShape = {b * s1 * n2 * smax, dn};

    std::vector<int> gateW1Shape = {h, 4 * h};
    std::vector<int> gateW2Shape = {4 * h, 3 * n1};
    std::vector<int> gateSimW1Shape = {h, 3 * n1};
    // std::vector<int> gatingScoreShape = {b, s1, n1, 3};

    std::vector<int> shape_cmpAtten = {b, s1, n1, v_dim};
    std::vector<int> shape_selAtten = {b, s1, n1, v_dim};
    std::vector<int> shape_winAtten = {b, s1, n1, v_dim};
    std::vector<int> shape_attentionOut = {b, s1, n1, v_dim};

    // post: shape
    std::vector<int> wUvShape = {n1, v_dim, vHeadDim};
    std::vector<int> woShape = {n1 * vHeadDim, h};
    std::vector<int> woScaleShape = {1, h};
    std::vector<int> smoothWoShape = {1, n1 * vHeadDim};
    std::vector<int> outShape = {b, s1, h};

    // 2. 构造tensor
    // MlaProlog
    Tensor x(dType, xShape, "x");
    TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    Tensor wDq(dType, wDqShape, "wDq", NodeType::LOCAL, weightFormat);
    Tensor wUqQr(dTypeQuant, wUqQrShape, "wUqQr", NodeType::LOCAL, weightFormat);
    const bool usePrefetch = true;
    if constexpr (usePrefetch) {
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
    // MlaProlog output
    Tensor outputKvCache(dType, kvCacheOutShape, "outputKvCache");
    Tensor outputKrCache(dType, krCacheOutShape, "outputKrCache");
    Tensor outputQ(dType, qOutShape, "outputQ");
    Tensor outputQRope(dType, qRopeOutShape, "outputQRope");

    Tensor topkIndices(DT_INT32, topkIndicesShape, "topkTensor");
    Tensor topkTensorShape(DT_INT32, topkTensorShapeShape, "topkTensorShape");
    Tensor kvNopeCache(dType, kvNopeCacheShape, "kNopeCache");
    Tensor kRopeCache(dType, kRopeCacheShape, "vNopeCache");
    Tensor kvCacheActSeq(DT_INT32, kvCacheActSeqShape, "kvCacheActSeq");
    Tensor blockTable(DT_INT32, blockTableShape, "blockTable");

    Tensor slcActSeqs(DT_INT32, slcActSeqsShape, "slcActSeqs");
    Tensor qNope(dType, qNopeShape, "qNope");
    Tensor qRope(dType, qRopeShape, "qRope");
    Tensor kSlc(dType, kSlcShape, "kSlc");
    Tensor vSlc(dType, vSlcShape, "vSlc");

    Tensor gateW1(dType, gateW1Shape, "gateW1");
    Tensor gateW2(dType, gateW2Shape, "gateW2");
    Tensor gateSimW1(dType, gateSimW1Shape, "gateSimW1");
    // Tensor gatingScore(dType, gatingScoreShape, "gatingScore");

    Tensor cmpAtten(dType, shape_cmpAtten, "cmpAtten");
    Tensor slcAttn(DT_FP32, shape_selAtten, "selAtten"); // fp32输入
    Tensor winAtten(DT_FP32, shape_winAtten, "winAtten");

    Tensor kvSlcActSeqsMidOut(DT_INT32, slcActSeqsShape, "kvSlcActSeqsMidOut");
    Tensor attenOut(dType, shape_attentionOut, "attenOut");

    // post: Tensor
    Tensor wUv(dType, wUvShape, "wUv");
    Tensor wo(dTypeQuant, woShape, "wo", NodeType::LOCAL, weightFormat);
    Tensor woScale;
    Tensor smoothWo;
    Tensor postOut(dType, outShape, "postOut");

    // 3. 为输入填充数据
    auto xData = CreateTensorData<T>(x, "/x.bin");
    auto wDqData = CreateTensorData<T>(wDq, "/wDq.bin");
    auto wUqQrData = CreateTensorData<wDtype>(wUqQr, "/wUqQr.bin");
    auto wUkData = CreateTensorData<T>(wUk, "/wUk.bin");
    auto wDkvKrData = CreateTensorData<T>(wDkvKr, "/wDkvKr.bin");
    auto gammaCqData = CreateTensorData<T>(gammaCq, "/gamma_cq.bin");
    auto gammaCkvData = CreateTensorData<T>(gammaCkv, "/gamma_ckv.bin");
    auto cosData = CreateTensorData<T>(cos, "/cos.bin");
    auto sinData = CreateTensorData<T>(sin, "/sin.bin");
    auto kvLenData = CreateTensorData<int64_t>(cacheIndex, "/kv_len.bin");
    auto kvCacheData = CreateTensorData<T>(kvCache, "/kv_cache.bin");
    auto krCacheData = CreateTensorData<T>(krCache, "/kr_cache.bin");
    auto outKvCacheData = CreateTensorData<T>(outputKvCache, "/kv_cache.bin");
    auto outKrCacheData = CreateTensorData<T>(outputKrCache, "/kr_cache.bin");
    auto outputQData = RawTensorData::CreateConstantTensor<T>(outputQ, 0.0);
    auto outputQRopeData = RawTensorData::CreateConstantTensor<T>(outputQRope, 0.0);

    auto topkIndicesData = CreateTensorData<int32_t>(topkIndices, "/topk_tensor.bin");
    auto topkTensorShapeData = CreateTensorData<int32_t>(topkTensorShape, "/topk_tensor_shape.bin");
    // auto kvNopeCacheData = CreateTensorData<T>(kvNopeCache, "/kv_nope_cache.bin");
    // auto kRopeCacheData = CreateTensorData<T>(kRopeCache, "/k_rope_cache.bin");
    auto kvCacheActSeqData = CreateTensorData<int32_t>(kvCacheActSeq, "/kv_cache_actual_seq_len.bin");
    auto blockTableData = CreateTensorData<int32_t>(blockTable, "/block_table.bin");

    auto slcActSeqsData = CreateTensorData<int32_t>(slcActSeqs, "/kv_slc_actual_seqs.bin");
    // auto qNopeData = CreateTensorData<T>(qNope, "/q_nope.bin");
    // auto qRopeData = CreateTensorData<T>(qRope, "/q_rope.bin");
    // auto kSlcData = CreateTensorData<T>(kSlc, "/k_slc.bin");
    // auto vSlcData = CreateTensorData<T>(vSlc, "/v_slc.bin");

    // auto xData = CreateTensorData<T>(x, "/x.bin");
    auto gateW1Data = CreateTensorData<T>(gateW1, "/gate_w1.bin");
    auto gateW2Data = CreateTensorData<T>(gateW2, "/gate_w2.bin");
    auto gateSimW1Data = CreateTensorData<T>(gateSimW1, "/gate_sim_w1.bin");

    auto cmpAttenData = CreateTensorData<T>(cmpAtten, "/cmp_atten.bin");
    // auto selAttenData = CreateTensorData<T>(selAtten, "/sel_atten.bin");
    // auto winAttenData = CreateTensorData<T>(winAtten, "/win_atten.bin");

    // post: data
    auto wUvData = CreateTensorData<T>(wUv, "/w_uv.bin");
    auto woData = CreateTensorData<wDtype>(wo, "/w_o.bin");

    // auto gatingScoreZeroData = RawTensorData::CreateConstantTensor<T>(gatingScore, 0.0);
    auto kvSlcActSeqsMidOutZeroData = RawTensorData::CreateConstantTensor<int32_t>(kvSlcActSeqsMidOut, 0.0);
    auto kSlcZeroData = RawTensorData::CreateConstantTensor<T>(kSlc, 0.0);
    auto vSlcZeroData = RawTensorData::CreateConstantTensor<T>(vSlc, 0.0);
    auto winAttenData = RawTensorData::CreateConstantTensor<float>(winAtten, 0.0);
    auto slcAttnZeroData = RawTensorData::CreateConstantTensor<float>(slcAttn, 0.0);
    auto attenOutZeroData = RawTensorData::CreateConstantTensor<T>(attenOut, 0.0);
    auto outputData = RawTensorData::CreateConstantTensor<T>(postOut, 0.0);

    auto qNopeData = RawTensorData::CreateConstantTensor<T>(qNope, 0.0);
    auto qRopeData = RawTensorData::CreateConstantTensor<T>(qRope, 0.0);


    // MlaProlog output golden
    std::vector<T> golden1 = getGoldenVec<T>(qOutShape, "/q_golden.bin");
    std::vector<T> golden2 = getGoldenVec<T>(qRopeOutShape, "/q_rope_golden.bin");
    std::vector<T> golden3 = getGoldenVec<T>(kvCacheOutShape, "/kv_cache_golden.bin");
    std::vector<T> golden4 = getGoldenVec<T>(krCacheOutShape, "/kr_cache_golden.bin");
    // std::vector<T> gatingScoreGolden = getGoldenVec<T>(gatingScoreShape, "/gating_score.bin");
    std::vector<int32_t> kvSlcActSeqMidOutGolden = getGoldenVec<int32_t>(slcActSeqsShape, "/kv_slc_actual_seqs.bin");
    std::vector<T> kSlcOutGolden = getGoldenVec<T>(kSlcShape, "/kv_slc_out.bin");
    std::vector<T> vSlcOutGolden = getGoldenVec<T>(vSlcShape, "/kr_slc_out.bin");
    std::vector<float> winAttnGolden = getGoldenVec<float>(shape_winAtten, "/winAttn.bin");
    std::vector<float> slcAttnOutGolden = getGoldenVec<float>(shape_selAtten, "/sel_atten.bin");
    std::vector<T> attenOutGolden = getGoldenVec<T>(shape_attentionOut, "/attention_out.bin");
    // Post output golden
    std::vector<T> postGolden = getGoldenVec<T>(outShape, "/golden_output.bin");

    std::vector<RawTensorDataPtr> outputDataList = {
        outputQData, outputQRopeData, outKvCacheData, outKrCacheData, qNopeData, qRopeData, winAttenData,
        kvSlcActSeqsMidOutZeroData, kSlcZeroData, vSlcZeroData, slcAttnZeroData, attenOutZeroData, outputData};
    std::vector<RawTensorDataPtr> inputDataList =
        {xData, wDqData, wUqQrData, wUkData, wDkvKrData, gammaCqData, gammaCkvData, sinData, cosData, kvLenData,
         kvCacheData, krCacheData};
    MlaQuantInputs quantInputs;
    if (isQuant) {
        auto wQbScaleData = CreateTensorData<float>(wQbScale, "/w_qb_scale.bin");
        inputDataList.emplace_back(wQbScaleData);
        quantInputs.dequantScaleWUqQr = wQbScale;
        if (isSmooth) {
            auto smoothCqData = CreateTensorData<float>(smoothCq, "/smooth_cq.bin");
            inputDataList.emplace_back(smoothCqData);
            quantInputs.smoothScalesCq = smoothCq;
        }
    } else {
        inputDataList.emplace_back(nullptr); // quantInputs.dequantScaleWUqQr
        inputDataList.emplace_back(nullptr); // quantInputs.smoothScalesCq
    }
    std::vector<RawTensorDataPtr> tmpInputDataList = {
        topkIndicesData, topkTensorShapeData, /*kvNopeCacheData, kRopeCacheData,*/ kvCacheActSeqData, blockTableData, // genkvSlc
        /*qNopeData, qRopeData,*/ slcActSeqsData, // slcAtten
        /*xData, */gateW1Data, gateW2Data, gateSimW1Data, // gatedScore
        cmpAttenData, /*winAttenData,*/ // genAttn
        wUvData, woData, // post
    };
    inputDataList.insert(inputDataList.end(), tmpInputDataList.begin(), tmpInputDataList.end());
    if (isQuant) {
        Tensor scale(DT_FP32, woScaleShape, "woScale");
        woScale = scale;
        auto woScaleData = CreateTensorData<float>(woScale, "/w_o_scale.bin");
        inputDataList.emplace_back(woScaleData);
        if (isSmooth) {
            Tensor smooth(DT_FP32, smoothWoShape, "smoothWo");
            smoothWo = smooth;
            auto smoothWoData = CreateTensorData<float>(smoothWo, "/smooth_wo.bin");
            inputDataList.emplace_back(smoothWoData);
        }
    } else {
        inputDataList.emplace_back(nullptr); // woScaleData
        inputDataList.emplace_back(nullptr); // smoothWoData
    }

    // 4. 计算接口
    DynamicNsa(x, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos, cacheIndex, kvCache, krCache, quantInputs,
        prologConfig, eps, eps, cacheMode,
        topkIndices, topkTensorShape, /*kvNopeCache, kRopeCache,*/ kvCacheActSeq, blockTable, front, near, topk, slcBlockSize, blockSize, kvSlcTileConfig, // genKvSlc
        /*qNope, qRope,*/ slcActSeqs, softmaxScale, saTileConfig, // slcAttn
        /*x, */gateW1, gateW2, gateSimW1, GateMode::standard, // gatedscore
        cmpAtten, winAtten, winSize, winAttntileConfig,// gen win
        wUv, wo, woScale, smoothWo, postConfig, // post
        outputQ, outputQRope, outputKvCache, outputKrCache, qNope, qRope, kvSlcActSeqsMidOut, kSlc, vSlc, slcAttn, attenOut, postOut);

    auto funcOp = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
#ifndef AC_ENABLE_FRAMEWORK_WITHOUT_CANN
    // 5. 更新输入输出list
    DynFuncRunner::Run(funcOp, inputDataList, outputDataList); // output list

    // attenOutZeroData->ToFile(GetGoldenDir() + "/attenOutNpu.bin");


    std::cout << "MlaProlog qNope ====== " << std::endl;
    EXPECT_TRUE(resultCmp<T>(golden1, (T *)outputQData->data(), 0.008f));
    std::cout << "MlaProlog qRope ======" << std::endl;
    EXPECT_TRUE(resultCmp<T>(golden2, (T *)outputQRopeData->data(), 0.005f));
    std::cout << "MlaProlog kv ====== " << std::endl;
    EXPECT_TRUE(resultCmp<T>(golden3, (T *)outKvCacheData->data(), 0.003f));
    std::cout << "MlaProlog kr ====== " << std::endl;
    EXPECT_TRUE(resultCmp<T>(golden4, (T *)outKrCacheData->data(), 0.003f));

    std::cout << "MlaProlog qNope Reshape ============ " << std::endl;
    EXPECT_TRUE(resultCmp<T>(golden1, (T *)qNopeData->data(), 0.008f));
    std::cout << "MlaProlog qRope Reshape ============ " << std::endl;
    EXPECT_TRUE(resultCmp<T>(golden2, (T *)qRopeData->data(), 0.005f));

    // EXPECT_TRUE(resultCmp<T>(gatingScoreGolden, (T *)gatingScoreZeroData->data(), 0.008f, 16)); // gatedScore
    std::cout << "winAttnOut ====== " << std::endl;
    EXPECT_TRUE(resultCmp<float>(winAttnGolden, (float *)winAttenData->data(), 0.0005f));
    std::cout << "kvSlcActSeqMidOut ====== " << std::endl;
    EXPECT_TRUE(resultCmp<int32_t>(kvSlcActSeqMidOutGolden, (int32_t *)kvSlcActSeqsMidOutZeroData->data(), 0.0005f));
    std::cout << "kSlc ====== " << std::endl;
    EXPECT_TRUE(resultCmp<T>(kSlcOutGolden, (T *)kSlcZeroData->data(), 0.0005f));
    std::cout << "vSlc ====== " << std::endl;
    EXPECT_TRUE(resultCmp<T>(vSlcOutGolden, (T *)vSlcZeroData->data(), 0.0005f));
    std::cout << "slcAttnOut ====== " << std::endl;
    EXPECT_TRUE(resultCmp<float>(slcAttnOutGolden, (float *)slcAttnZeroData->data(), 0.005f));

    std::cout << "attenOut ====== " << std::endl;
    EXPECT_TRUE(resultCmp<T>(attenOutGolden, (T *)attenOutZeroData->data(), 0.008f));
    std::cout << "post out ====== " << std::endl;
    EXPECT_TRUE(resultCmp<T>(postGolden, (T *)outputData->data(), precision));
    std::cout << "post out ====== print" << std::endl;
    resultCmp<T>(postGolden, (T *)outputData->data(), precision, int(b * s1 * h * precision), 1000, false, false, 16);
#endif
}

TEST_F(DynamicNSATest, nsa_b_16_s1_1_s2_8192_h_7168_fp16) {
    NSASimpleParams params = NSASimpleParams::getDecodeParams();

    int paramsSize = 7;
    std::vector<int> inputParams(paramsSize);
    readInput<int>(GetGoldenDir() + "/input_params.bin", inputParams); // 在golden中保存了变化的参数，便于调试
    params.b = inputParams[0]; // 16
    params.s1 = inputParams[1];
    params.s2 = inputParams[2];
    params.n1 = inputParams[3];
    params.n2 = inputParams[4];
    int isQuant = inputParams[5];
    int isSmooth = inputParams[6];
    std::cout << "===========nsa_b_16_s1_1_s2_8192_h_7168_fp16: isQuant: " << isQuant << ", isSmooth: " << isSmooth << std::endl;

    SaTileShapeConfig saTileConfig;
    const int gTile = 128; // for gLoop split
    const int sTile = 1024; // for s2Loop split
    saTileConfig.gTile = gTile;
    saTileConfig.sKvTile = sTile;
    saTileConfig.c1TileShape = {gTile, gTile, 64, 64, 128, 128}; // (n1, dn+dr) @ (s2Tile, dn+dr) -> (n1, s2Tile)
    saTileConfig.v1TileShape = {16, 256}; // (n1, s2Tile)
    saTileConfig.c2TileShape = {gTile, gTile, 64, 64, 128, 128}; // (n1, s2Tile) @ (s2Tile, dn) -> (n1, d)
    saTileConfig.v2TileShape = {16, 256}; // (n1, d)

    WinAttenTileShapeConfig winAttnTileConfig;
    const int gTileSize = NUM_128; // for gLoop split
    winAttnTileConfig.gTile = gTileSize;
    winAttnTileConfig.vNopeTileShape = {NUM_16, NUM_256};
    winAttnTileConfig.vRopeTileShape = {NUM_128, NUM_64};
    winAttnTileConfig.outTileShape = {NUM_16, NUM_256};
    winAttnTileConfig.c1TileShape = {gTileSize, gTileSize, NUM_64, NUM_64, NUM_128, NUM_128}; // (n1, dN+dR) @ (winSize, dN+dR) -> (n1, s2Tile)
    winAttnTileConfig.v1TileShape = {NUM_16, NUM_256}; // (n1, s2Tile)
    winAttnTileConfig.c2TileShape = {gTileSize, gTileSize, NUM_64, NUM_64, NUM_128, NUM_128}; // (n1, winSize) @ (winSize, dN) -> (n1, d)
    winAttnTileConfig.v2TileShape = {NUM_16, NUM_256}; // (n1, d)

    KvSlcTileShapeConfig kvSlcTileConfig;
    kvSlcTileConfig.v0TileShape = {32, 32};

    PostTileConfig postConfig = {16, 1};
    MlaTileConfig prologConfig = {16, 1};

    std::string cacheMode = "PA_BSND";
    if (isQuant == 1) {
        if (isSmooth == 1) {
            TestNsa<npu::tile_fwk::float16, int8_t, true>(params, prologConfig, winAttnTileConfig, saTileConfig, kvSlcTileConfig, postConfig, 0.06f, cacheMode);
        } else {
            TestNsa<npu::tile_fwk::float16, int8_t, false>(params, prologConfig, winAttnTileConfig, saTileConfig, kvSlcTileConfig, postConfig, 0.06f, cacheMode);
        }
    } else {
        TestNsa<npu::tile_fwk::float16, npu::tile_fwk::float16, false>(params, prologConfig, winAttnTileConfig, saTileConfig, kvSlcTileConfig, postConfig, 0.02f, cacheMode);
    }
}

TEST_F(DynamicNSATest, nsa_b_16_s1_1_s2_8192_h_7168_fp16_quant) {
    NSASimpleParams params = NSASimpleParams::getDecodeParams();

    int paramsSize = 7;
    std::vector<int> inputParams(paramsSize);
    readInput<int>(GetGoldenDir() + "/input_params.bin", inputParams); // 在golden中保存了变化的参数，便于调试
    params.b = inputParams[0]; // 16
    params.s1 = inputParams[1];
    params.s2 = inputParams[2];
    params.n1 = inputParams[3];
    params.n2 = inputParams[4];
    int isQuant = inputParams[5];
    int isSmooth = inputParams[6];
    std::cout << "===========nsa_b_16_s1_1_s2_8192_h_7168_fp16_quant: isQuant: " << isQuant << ", isSmooth: " << isSmooth << std::endl;

    SaTileShapeConfig saTileConfig;
    const int gTile = 128; // for gLoop split
    const int sTile = 1024; // for s2Loop split
    saTileConfig.gTile = gTile;
    saTileConfig.sKvTile = sTile;
    saTileConfig.c1TileShape = {gTile, gTile, 64, 64, 128, 128}; // (n1, dn+dr) @ (s2Tile, dn+dr) -> (n1, s2Tile)
    saTileConfig.v1TileShape = {16, 256}; // (n1, s2Tile)
    saTileConfig.c2TileShape = {gTile, gTile, 64, 64, 128, 128}; // (n1, s2Tile) @ (s2Tile, dn) -> (n1, d)
    saTileConfig.v2TileShape = {16, 256}; // (n1, d)

    WinAttenTileShapeConfig winAttnTileConfig;
    const int gTileSize = NUM_128; // for gLoop split
    winAttnTileConfig.gTile = gTileSize;
    winAttnTileConfig.vNopeTileShape = {NUM_16, NUM_256};
    winAttnTileConfig.vRopeTileShape = {NUM_128, NUM_64};
    winAttnTileConfig.outTileShape = {NUM_16, NUM_256};
    winAttnTileConfig.c1TileShape = {gTileSize, gTileSize, NUM_64, NUM_64, NUM_128, NUM_128}; // (n1, dN+dR) @ (winSize, dN+dR) -> (n1, s2Tile)
    winAttnTileConfig.v1TileShape = {NUM_16, NUM_256}; // (n1, s2Tile)
    winAttnTileConfig.c2TileShape = {gTileSize, gTileSize, NUM_64, NUM_64, NUM_128, NUM_128}; // (n1, winSize) @ (winSize, dN) -> (n1, d)
    winAttnTileConfig.v2TileShape = {NUM_16, NUM_256}; // (n1, d)

    KvSlcTileShapeConfig kvSlcTileConfig;
    kvSlcTileConfig.v0TileShape = {32, 32};

    PostTileConfig postConfig = {16, 1};
    MlaTileConfig prologConfig = {16, 1};

    std::string cacheMode = "PA_BSND";
    if (isQuant == 1) {
        if (isSmooth == 1) {
            TestNsa<npu::tile_fwk::float16, int8_t, true>(params, prologConfig, winAttnTileConfig, saTileConfig, kvSlcTileConfig, postConfig, 0.06f, cacheMode);
        } else {
            TestNsa<npu::tile_fwk::float16, int8_t, false>(params, prologConfig, winAttnTileConfig, saTileConfig, kvSlcTileConfig, postConfig, 0.06f, cacheMode);
        }
    } else {
        TestNsa<npu::tile_fwk::float16, npu::tile_fwk::float16, false>(params, prologConfig, winAttnTileConfig, saTileConfig, kvSlcTileConfig, postConfig, 0.02f, cacheMode);
    }
}
