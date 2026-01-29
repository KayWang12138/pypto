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
 * \file test_lightning_indexer.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "tilefwk/data_type.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "machine/device/dynamic/device_utils.h"
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/deepseek_v3.2_exp/lightning_indexer.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class LightningIndexerSTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

template <typename T>
static std::vector<T> getGoldenVec(std::vector<int64_t> shape, std::string fileName) {
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> golden(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, golden);
    return golden;
}

template <typename T>
static std::shared_ptr<RawTensorData> CreateTensorData(
    Tensor tensor, std::vector<int64_t> shape, std::string fileName) {
    uint64_t capacity = std::accumulate(shape.begin(), shape.end(), uint64_t{1}, std::multiplies<uint64_t>());
    std::vector<T> values(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, values);
    return RawTensorData::CreateTensor<T>(tensor, values);
}

struct LightningIndexerParams {
    int64_t b;
    int64_t s1;
    int64_t n1;
    int64_t d;
    int64_t blockNum;
    int64_t blockSize;
    int64_t n2;
    int64_t maxBlockNum;
    int64_t selectedCount;
};

inline LightningIndexerParams LoadLightningIndexerParams() {
    int paramsSize = 9;
    std::vector<int> input_param(paramsSize);
    readInput<int>(GetGoldenDir() + "/input_params.bin", input_param);
    return {input_param[0], input_param[1], input_param[2], input_param[3],
            input_param[4], input_param[5], input_param[6], input_param[7], input_param[8]};
}

struct LightningIndexerTensors {
    Tensor staticQuery;
    Tensor staticKey;
    Tensor staticQScale;
    Tensor staticKScale;
    Tensor staticWeights;
    Tensor staticActSeq;
    Tensor staticBlockTable;
    Tensor staticTopkRes;
    Tensor staticFirstMm;
    Tensor staticMmOut;
    Tensor staticTopkValue;
    Tensor query;
    Tensor qScale;
    Tensor key;
    Tensor kScale;
    Tensor weights;
    Tensor actSeq;
    Tensor blockTable;
    Tensor firstMm;
    Tensor mmOut;
    Tensor topkRes;
    Tensor topkValue;
};

inline LightningIndexerTensors CreateLightningIndexerTensors(const LightningIndexerParams &params) {
    const int64_t b = params.b;
    const int64_t s1 = params.s1;
    const int64_t n1 = params.n1;
    const int64_t d = params.d;
    const int64_t blockNum = params.blockNum;
    const int64_t blockSize = params.blockSize;
    const int64_t n2 = params.n2;
    const int64_t maxBlockNum = params.maxBlockNum;
    const int64_t selectedCount = params.selectedCount;

    return {
        Tensor(DT_INT8, {b * s1, n1, d}, "staticQuery"),
        Tensor(DT_INT8, {blockNum, blockSize, n2, d}, "staticKey"),
        Tensor(DT_FP16, {b * s1, n1}, "staticQScale"),
        Tensor(DT_FP16, {blockNum, blockSize, n2}, "staticKScale"),
        Tensor(DT_FP16, {b * s1, n1}, "staticWeights"),
        Tensor(DT_INT32, {b}, "staticActSeq"),
        Tensor(DT_INT32, {b, maxBlockNum}, "staticBlockTable"),
        Tensor(DT_INT32, {b * s1, n2, selectedCount}, "staticTopkRes"),
        Tensor(DT_FP16, {b * s1 * n1, maxBlockNum * blockSize}, "staticFirstMm"),
        Tensor(DT_FP32, {b * s1 * n2, maxBlockNum * blockSize}, "staticMmOut"),
        Tensor(DT_FP32, {b * s1, n2, selectedCount}, "staticTopkValue"),
        Tensor(DT_INT8, {-1, n1, d}, "query"),
        Tensor(DT_FP16, {-1, n1}, "qScale"),
        Tensor(DT_INT8, {-1, blockSize, n2, d}, "key"),
        Tensor(DT_FP16, {-1, blockSize, n2}, "kScale"),
        Tensor(DT_FP16, {-1, n1}, "weights"),
        Tensor(DT_INT32, {-1}, "actSeq"),
        Tensor(DT_INT32, {-1, -1}, "blockTable"),
        Tensor(DT_FP16, {}, "firstMm"),
        Tensor(DT_FP32, {}, "mmOut"),
        Tensor(DT_INT32, {}, "topkRes"),
        Tensor(DT_FP32, {}, "topkValue")
    };
}

struct LightningIndexerData {
    std::vector<npu::tile_fwk::float16> firstMmGolden;
    std::vector<float> mmGolden;
    std::vector<int32_t> topkResGolden;
    std::vector<float> topkValueGolden;
    std::vector<RawTensorDataPtr> inputDataList;
    std::vector<RawTensorDataPtr> outputDataList;
};

inline LightningIndexerData LoadLightningIndexerData(const LightningIndexerParams &params,
                                                   LightningIndexerTensors &tensors) {
    const int64_t b = params.b;
    const int64_t s1 = params.s1;
    const int64_t n1 = params.n1;
    const int64_t d = params.d;
    const int64_t blockNum = params.blockNum;
    const int64_t blockSize = params.blockSize;
    const int64_t n2 = params.n2;
    const int64_t maxBlockNum = params.maxBlockNum;
    const int64_t selectedCount = params.selectedCount;

    auto firstMmGolden = getGoldenVec<npu::tile_fwk::float16>({b * s1 * n1, maxBlockNum * blockSize}, "/first_mm.bin");
    auto mmGolden = getGoldenVec<float>({b * s1 * n2, maxBlockNum * blockSize}, "/mm_out.bin");
    auto topkResGolden = getGoldenVec<int32_t>({b * s1, n2, selectedCount}, "/topk_res.bin");
    auto topkValueGolden = getGoldenVec<float>({b * s1, n2, selectedCount}, "/topk_value.bin");

    auto qData = CreateTensorData<int8_t>(tensors.staticQuery, {b * s1, n1, d}, "/query.bin");
    auto kData = CreateTensorData<int8_t>(tensors.staticKey, {blockNum, blockSize, n2, d}, "/key.bin");
    auto qsData = CreateTensorData<npu::tile_fwk::float16>(tensors.staticQScale, {b * s1, n1}, "/q_scale.bin");
    auto ksData = CreateTensorData<npu::tile_fwk::float16>(tensors.staticKScale, {blockNum, blockSize, n2}, "/k_scale.bin");
    auto wData = CreateTensorData<npu::tile_fwk::float16>(tensors.staticWeights, {b * s1, n1}, "/weights.bin");
    auto sData = CreateTensorData<int32_t>(tensors.staticActSeq, {b}, "/act_seq.bin");
    auto bData = CreateTensorData<int32_t>(tensors.staticBlockTable, {b, maxBlockNum}, "/block_table.bin");

    auto firstMmData = RawTensorData::CreateConstantTensor<npu::tile_fwk::float16>(tensors.staticFirstMm, 0.0f);
    auto mmData = RawTensorData::CreateConstantTensor<float>(tensors.staticMmOut, 0.0f);
    auto topkResData = RawTensorData::CreateConstantTensor<int32_t>(tensors.staticTopkRes, 0);
    auto topkValueData = RawTensorData::CreateConstantTensor<float>(tensors.staticTopkValue, 0.0f);

    std::vector<RawTensorDataPtr> inputDataList = {qData, qsData, kData, ksData, wData, sData, bData};
    std::vector<RawTensorDataPtr> outputDataList = {topkResData, firstMmData, mmData, topkValueData};

    return {firstMmGolden, mmGolden, topkResGolden, topkValueGolden, inputDataList, outputDataList};
}

inline void BuildLightningIndexerGraph(LightningIndexerTensors &tensors, const LightningIndexerParams &params,
                                      LightningIndexerConfigs &tileConfig, const std::set<int> &unrollList) {
    const int64_t n1 = params.n1;
    const int64_t n2 = params.n2;
    const int64_t blockSize = params.blockSize;
    const int64_t selectedCount = params.selectedCount;

    auto symT = GetInputShape(tensors.query, 0);
    auto symMaxBlock = GetInputShape(tensors.blockTable, 1);

    tensors.firstMm = Tensor(DT_FP16, {symT * n1, symMaxBlock * blockSize}, "firstMm");
    tensors.mmOut = Tensor(DT_FP32, {symT * n2, symMaxBlock * blockSize}, "MmOut");
    tensors.topkRes = Tensor(DT_INT32, {symT, n2, selectedCount}, "topkRes");
    tensors.topkValue = Tensor(DT_FP32, {symT, n2, selectedCount}, "topkValue");

    FUNCTION("LightningIndexer", {tensors.query, tensors.qScale, tensors.key, tensors.kScale,
                              tensors.weights, tensors.actSeq, tensors.blockTable},
                              {tensors.topkRes, tensors.firstMm, tensors.mmOut, tensors.topkValue}) {
        LightningIndexerImpl(tensors.query, tensors.qScale, tensors.key, tensors.kScale, tensors.weights,
                           tensors.actSeq, tensors.blockTable, selectedCount, tensors.topkRes,
                           tileConfig, unrollList, &tensors.firstMm, &tensors.mmOut, &tensors.topkValue);
    }
}

inline void VerifyLightningIndexerResult(const std::vector<npu::tile_fwk::float16> &firstMmGolden,
                                        const std::vector<float> &mmGolden,
                                        const std::vector<int32_t> &topkResGolden,
                                        const std::vector<float> &topkValueGolden,
                                        const std::vector<RawTensorDataPtr> &outputDataList,
                                        int64_t selectedCount) {
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(),
                      ProgramData::GetInstance().GetInputData(),
                      outputDataList, DeviceLauncherConfig(0));
    constexpr float PRE_TAIL = 1e-4f;
    constexpr int TOPK_COUNT = 100;
    constexpr float ratio = 5e-3f;
    std::cout << "=======================firstMm===============================" << std::endl;
    EXPECT_TRUE(resultCmp(firstMmGolden, (npu::tile_fwk::float16 *)outputDataList[1]->data(),
                       PRE_TAIL, 0, TOPK_COUNT, false, false));
    std::cout << "=======================mmOut===============================" << std::endl;
    EXPECT_TRUE(resultCmp(mmGolden, (float *)outputDataList[2]->data(),
                       PRE_TAIL, 0, TOPK_COUNT, false, false));
    std::cout << "=======================topkValue===============================" << std::endl;
    EXPECT_TRUE(resultCmp(topkValueGolden, (float *)outputDataList[3]->data(),
                       PRE_TAIL, 0, TOPK_COUNT, false, false));
    std::cout << "=======================topkRes===============================" << std::endl;
    EXPECT_TRUE(resultCmp4TopK(topkResGolden, (int32_t *)outputDataList[0]->data(), selectedCount, ratio));
}

void TestLightningIndexer(LightningIndexerConfigs &tileConfig) {
    std::set<int> unrollList = {32, 16, 8, 4, 1};

    auto params = LoadLightningIndexerParams();
    auto tensors = CreateLightningIndexerTensors(params);
    auto data = LoadLightningIndexerData(params, tensors);

    BuildLightningIndexerGraph(tensors, params, tileConfig, unrollList);
    VerifyLightningIndexerResult(data.firstMmGolden, data.mmGolden, data.topkResGolden,
                               data.topkValueGolden, data.outputDataList, params.selectedCount);
}

// LightningIndexerSTest.lightning_indexer_quant_4_b_2_s1_64k_s2
TEST_F(LightningIndexerSTest, lightning_indexer_quant_4_b_2_s1_64k_s2) {
    LightningIndexerConfigs config;
    config.s1Tile = 2; // s1Tile = s1
    config.topkTile = 8192;
    config.c1Tile = {128, 128, 128, 128, 128, 128}; // (m, M), (k, K), (n, N)
    config.c2Tile = {64, 64, 128, 128, 128, 128};   // (m, M), (k, K), (n, N)
    config.extendParam.reluType = npu::tile_fwk::Matrix::ReLuType::ReLu;
    float scale = 2048.0;
    config.extendParam.scaleValue = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&scale));

    TestLightningIndexer(config);
}