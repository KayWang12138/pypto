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
 * \file test_dynamic_fused_compress_kv_select.cpp
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
#include "operator/models/deepseek_v3.2_exp/lightning_indexer_topk.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class DynamicIndexerTopk : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

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

struct IndexerTopkParams {
    int b;
    int s1;
    int n1;
    int d;
    int blockNum;
    int blockSize;
    int n2;
    int maxBlockNum;
    int selectedCount;
};

inline IndexerTopkParams LoadIndexerTopkParams() {
    int paramsSize = 9;
    std::vector<int> input_param(paramsSize);
    readInput<int>(GetGoldenDir() + "/input_params.bin", input_param);
    return {input_param[0], input_param[1], input_param[2], input_param[3],
            input_param[4], input_param[5], input_param[6], input_param[7], input_param[8]};
}

struct IndexerTopkTensors {
    Tensor staticQuery;
    Tensor staticKey;
    Tensor staticQScale;
    Tensor staticKScale;
    Tensor staticWeights;
    Tensor staticActSeq;
    Tensor staticBlockTable;
    Tensor staticTopkRes;
    Tensor staticTmpOut;
    Tensor staticTopkValue;
    Tensor query;
    Tensor key;
    Tensor qScale;
    Tensor kScale;
    Tensor weights;
    Tensor actSeq;
    Tensor blockTable;
    Tensor topkRes;
    Tensor tmpOut;
    Tensor topkValue;
};

inline IndexerTopkTensors CreateIndexerTopkTensors(const IndexerTopkParams &params) {
    const int b = params.b;
    const int s1 = params.s1;
    const int n1 = params.n1;
    const int d = params.d;
    const int blockNum = params.blockNum;
    const int blockSize = params.blockSize;
    const int n2 = params.n2;
    const int maxBlockNum = params.maxBlockNum;
    const int selectedCount = params.selectedCount;

    return {
        Tensor(DT_INT8, {b, s1, n1, d}, "staticQuery"),
        Tensor(DT_INT8, {blockNum, blockSize, n2, d}, "staticKey"),
        Tensor(DT_FP16, {b, s1, n1, 1}, "staticQScale"),
        Tensor(DT_FP16, {blockNum, blockSize, n2, 1}, "staticKScale"),
        Tensor(DT_FP16, {b, s1, n1}, "staticWeights"),
        Tensor(DT_INT32, {b}, "staticActSeq"),
        Tensor(DT_INT32, {b, maxBlockNum}, "staticBlockTable"),
        Tensor(DT_INT32, {b, s1, n2, selectedCount}, "staticTopkRes"),
        Tensor(DT_FP32, {b * s1 * n2, maxBlockNum * blockSize}, "staticTmpOut"),
        Tensor(DT_FP32, {b, s1, n2, selectedCount}, "staticTopkValue"),
        Tensor(DT_INT8, {-1, -1, n1, d}, "query"),
        Tensor(DT_INT8, {-1, blockSize, n2, d}, "key"),
        Tensor(DT_FP16, {-1, -1, n1, 1}, "qScale"),
        Tensor(DT_FP16, {-1, blockSize, n2, 1}, "kScale"),
        Tensor(DT_FP16, {-1, -1, n1}, "weights"),
        Tensor(DT_INT32, {-1}, "actSeq"),
        Tensor(DT_INT32, {-1, -1}, "blockTable"),
        Tensor(DT_INT32, {}, "topkRes"),
        Tensor(DT_FP32, {}, "tmpOut"),
        Tensor(DT_FP32, {}, "topkValue")
    };
}

struct IndexerTopkData {
    std::vector<int32_t> topkResGolden;
    std::vector<float> tmpGolden;
    std::vector<float> topkValueGolden;
    std::vector<RawTensorDataPtr> inputDataList;
    std::vector<RawTensorDataPtr> outputDataList;
};

inline IndexerTopkData LoadIndexerTopkData(const IndexerTopkParams &params, IndexerTopkTensors &tensors) {
    const int b = params.b;
    const int s1 = params.s1;
    const int n1 = params.n1;
    const int d = params.d;
    const int blockNum = params.blockNum;
    const int blockSize = params.blockSize;
    const int n2 = params.n2;
    const int maxBlockNum = params.maxBlockNum;
    const int selectedCount = params.selectedCount;

    auto topkResGolden = getGoldenVec<int32_t>({b, s1, n2, selectedCount}, "/topk_res.bin");
    auto tmpGolden = getGoldenVec<float>({b * s1 * n2, maxBlockNum * blockSize}, "/tmp_out.bin");
    auto topkValueGolden = getGoldenVec<float>({b, s1, n2, selectedCount}, "/topk_value.bin");

    auto qData = CreateTensorData<int8_t>(tensors.staticQuery, {b, s1, n1, d}, "/query.bin");
    auto kData = CreateTensorData<int8_t>(tensors.staticKey, {blockNum, blockSize, n2, d}, "/key.bin");
    auto qsData = CreateTensorData<npu::tile_fwk::float16>(tensors.staticQScale, {b, s1, n1, 1}, "/q_scale.bin");
    auto ksData = CreateTensorData<npu::tile_fwk::float16>(tensors.staticKScale, {blockNum, blockSize, n2, 1}, "/k_scale.bin");
    auto wData = CreateTensorData<npu::tile_fwk::float16>(tensors.staticWeights, {b, s1, n1}, "/weights.bin");
    auto sData = CreateTensorData<int32_t>(tensors.staticActSeq, {b}, "/act_seq.bin");
    auto bData = CreateTensorData<int32_t>(tensors.staticBlockTable, {b, maxBlockNum}, "/block_table.bin");
    auto topkResData = RawTensorData::CreateConstantTensor<int32_t>(tensors.staticTopkRes, 0);
    auto tmpData = RawTensorData::CreateConstantTensor<float>(tensors.staticTmpOut, 0);
    auto topkValueData = RawTensorData::CreateConstantTensor<float>(tensors.staticTopkValue, 0);

    std::vector<RawTensorDataPtr> inputDataList = {qData, kData, qsData, ksData, wData, sData, bData};
    std::vector<RawTensorDataPtr> outputDataList = {topkResData, tmpData, topkValueData};

    return {topkResGolden, tmpGolden, topkValueGolden, inputDataList, outputDataList};
}

inline void BuildIndexerTopkGraph(IndexerTopkTensors &tensors, const IndexerTopkParams &params,
                                  IndexerTile &tileConfig, const std::set<int> &unrollList) {
    const int n2 = params.n2;
    const int selectedCount = params.selectedCount;
    const int blockSize = params.blockSize;

    auto symB = GetInputShape(tensors.query, 0);
    auto symS1 = GetInputShape(tensors.query, 1);
    auto symMaxBlock = GetInputShape(tensors.blockTable, 1);

    tensors.topkRes = Tensor(DT_INT32, {symB, symS1, n2, selectedCount}, "topkRes");
    tensors.tmpOut = Tensor(DT_FP32, {symB * symS1 * n2, symMaxBlock * blockSize}, "tmpOut");
    tensors.topkValue = Tensor(DT_FP32, {symB, symS1, n2, selectedCount}, "topkValue");

    FUNCTION("IndexerTopk", {tensors.query, tensors.key, tensors.qScale, tensors.kScale, tensors.weights,
                              tensors.actSeq, tensors.blockTable},
                              {tensors.topkRes, tensors.tmpOut, tensors.topkValue}) {
        LightningIndexerTopkImpl(tensors.query, tensors.key, true, &tensors.qScale, &tensors.kScale,
            tensors.weights, tensors.actSeq, tensors.blockTable, tensors.topkRes, selectedCount,
            tileConfig, unrollList, &tensors.tmpOut, &tensors.topkValue);
    }
}

inline void VerifyIndexerTopkResult(const std::vector<int32_t> &topkResGolden,
                                    const std::vector<float> &tmpGolden,
                                    const std::vector<float> &topkValueGolden,
                                    const std::vector<RawTensorDataPtr> &outputDataList,
                                    int selectedCount) {
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(),
                      ProgramData::GetInstance().GetInputData(),
                      outputDataList, DeviceLauncherConfig(0));
    constexpr float PRE_TAIL = 1e-5f;
    constexpr int TOPK_COUNT = 100;
    constexpr float ratio = 5e-3f;
    std::cout << "=======================topkValue===============================" << std::endl;
    EXPECT_TRUE(resultCmp(topkValueGolden, (float *)outputDataList[2]->data(), PRE_TAIL, 0, TOPK_COUNT, false, true));
    std::cout << "=======================topkRes===============================" << std::endl;
    EXPECT_TRUE(resultCmp4TopK(topkResGolden, (int32_t *)outputDataList[0]->data(), selectedCount, ratio));
}

void TestLightningIndexerTopkQuant(IndexerTile &tileConfig) {
    std::set<int> unrollList = {64, 32, 16, 8, 4, 2, 1};

    auto params = LoadIndexerTopkParams();
    auto tensors = CreateIndexerTopkTensors(params);
    auto data = LoadIndexerTopkData(params, tensors);

    BuildIndexerTopkGraph(tensors, params, tileConfig, unrollList);
    VerifyIndexerTopkResult(data.topkResGolden, data.tmpGolden, data.topkValueGolden,
                            data.outputDataList, params.selectedCount);
}

// DynamicIndexerTopk.indexer_topk_quant_4_b_1_s1_64k_s2
TEST_F(DynamicIndexerTopk, indexer_topk_quant_4_b_1_s1_64k_s2) {
    config::SetPassOption(MG_COPYIN_UPPER_BOUND, 100 * 1024 * 1024); // mistake
    config::SetPassOption(SG_PG_LOWER_BOUND, 1024);
    config::SetPassOption(SG_PG_UPPER_BOUND, 1024 * 1024);
    config::SetPassOption(CUBE_L1_REUSE_SETTING, std::map<int64_t, int64_t>{{-1, 32}});
    config::SetPassOption(SG_PARALLEL_NUM, 2);
    config::SetPassOption(VEC_NBUFFER_MODE, 1);
    config::SetRuntimeOption<uint8_t>(
        DEVICE_SCHED_MODE, static_cast<uint8_t>(MachineScheduleConfig::L2CACHE_AFFINITY_SCH) |
                            static_cast<uint8_t>(MachineScheduleConfig::MULTI_CORE_FAIR_SCH));

    config::SetRuntimeOption(STITCH_FUNCTION_INNER_MEMORY, 128);
    config::SetRuntimeOption(STITCH_FUNCTION_OUTCAST_MEMORY, 128);
    IndexerTile config;

    config.weightTile = {64, 128};
    config.c1Tile = {64, 64, 128, 128, 128, 128}; // (m, M), (k, K), (n, N)
    config.v1Tile = {64, 128};
    config.topkTile = {1, 4096};
    config.addsTile = {1, 1, 1, 4096};

    TestLightningIndexerTopkQuant(config);
}