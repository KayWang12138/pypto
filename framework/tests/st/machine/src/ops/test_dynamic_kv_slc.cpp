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
 * \file test_dynamic_slc.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "tilefwk/data_type.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "interface/program/program.h"
#include "machine/device/dynamic/device_utils.h"
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/deepseek/gen_kv_slc.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class DynamicSlcTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

struct KvSlcParams {
    int blockSize;
    int b;
    int s;
    int n2;
    int kv_lora_rank;
    int rope_dim;
    int front;
    int near;
    int topK;
    int l_prime;
    int blockNum;
    int maxBlockNumPerBatch;
};

KvSlcParams ReadKvSlcParams() {
    KvSlcParams params;
    int paramsSize = 10;
    std::vector<int> input_param(paramsSize);
    readInput<int>(GetGoldenDir() + "/input_param.bin", input_param);

    params.blockSize = input_param[9];
    params.b = input_param[0];
    params.s = input_param[1];
    params.n2 = input_param[2];
    params.kv_lora_rank = input_param[3];
    params.rope_dim = input_param[4];
    params.front = input_param[5];
    params.near = input_param[6];
    params.topK = input_param[7];
    params.l_prime = input_param[8];

    std::vector<int> seq(params.b);
    readInput<int>(GetGoldenDir() + "/actual_seq_len.bin", seq);

    params.blockNum = 0;
    for (auto seq_item : seq) {
        params.blockNum += CeilDiv(seq_item, params.blockSize);
    }
    int maxSeqAllBatch = *(std::max_element(seq.begin(), seq.end()));
    params.maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, params.blockSize);

    return params;
}

struct KvSlcTensors {
    Tensor topk_tensor;
    Tensor topk_tensor_shape;
    Tensor kvNopeCache;
    Tensor kRopeCache;
    Tensor kvActSeqs;
    Tensor blockTable;
    Tensor k_slcOut;
    Tensor v_slcOut;
    Tensor kvSlcActSeqs;
};

KvSlcTensors CreateKvSlcTensors(const KvSlcParams& params, DataType tensorType) {
    KvSlcTensors tensors;
    tensors.topk_tensor = Tensor(DT_INT32, {params.b, params.s, params.topK - params.front - params.near}, "topk_tensor");
    tensors.topk_tensor_shape = Tensor(DT_INT32, {params.b, params.s}, "topk_tensor_shape");
    tensors.kvNopeCache = Tensor(tensorType, {int(params.blockNum * params.blockSize), params.n2 * params.kv_lora_rank}, "kNopeCache");
    tensors.kRopeCache = Tensor(tensorType, {int(params.blockNum * params.blockSize), params.n2 * params.rope_dim}, "vNopeCache");
    tensors.kvActSeqs = Tensor(DT_INT32, {params.b}, "kvActSeqs");
    tensors.blockTable = Tensor(DT_INT32, {params.b, params.maxBlockNumPerBatch}, "blockTable");
    tensors.k_slcOut = Tensor(tensorType, {params.b * params.s * params.n2 * params.topK * params.l_prime, params.rope_dim + params.kv_lora_rank}, "k_slcOut");
    tensors.v_slcOut = Tensor(tensorType, {params.b * params.s * params.n2 * params.topK * params.l_prime, params.kv_lora_rank}, "v_slcOut");
    tensors.kvSlcActSeqs = Tensor(DT_INT32, {params.b, params.s}, "kvSlcActSeqs");
    return tensors;
}

template <typename T>
struct KvSlcInputData {
    std::vector<int32_t> topkTensorData;
    std::vector<int32_t> topkTensorShapeData;
    std::vector<T> kvNopeCacheData;
    std::vector<T> kRopeCacheData;
    std::vector<int32_t> kvActSeqsData;
    std::vector<int32_t> blockTableData;
    std::vector<int32_t> kvSlcActSeqsData;
};

template <typename T>
KvSlcInputData<T> ReadKvSlcInputData(const KvSlcParams& params) {
    KvSlcInputData<T> data;
    data.topkTensorData.resize(params.b * params.s * (params.topK - params.front - params.near), 0);
    data.topkTensorShapeData.resize(params.b * params.s, 0);
    data.kvNopeCacheData.resize(params.blockNum * params.blockSize * params.n2 * params.kv_lora_rank, 0);
    data.kRopeCacheData.resize(params.blockNum * params.blockSize * params.n2 * params.rope_dim, 0);
    data.kvActSeqsData.resize(params.b, 0);
    data.blockTableData.resize(params.b * params.maxBlockNumPerBatch, 0);
    data.kvSlcActSeqsData.resize(params.b * params.s, 0);

    readInput<int32_t>(GetGoldenDir() + "/topk_tensor.bin", data.topkTensorData);
    readInput<int32_t>(GetGoldenDir() + "/topk_tensor_shape.bin", data.topkTensorShapeData);
    readInput<T>(GetGoldenDir() + "/kv_nope_cache.bin", data.kvNopeCacheData);
    readInput<T>(GetGoldenDir() + "/k_rope_cache.bin", data.kRopeCacheData);
    readInput<int32_t>(GetGoldenDir() + "/actual_seq_len.bin", data.kvActSeqsData);
    readInput<int32_t>(GetGoldenDir() + "/block_table.bin", data.blockTableData);
    return data;
}

template <typename T>
struct KvSlcGoldenData {
    std::vector<T> k_golden;
    std::vector<T> v_golden;
    std::vector<int32_t> kvSlcActSeqs_golden;
};

template <typename T>
KvSlcGoldenData<T> ReadKvSlcGoldenData(const KvSlcParams& params) {
    KvSlcGoldenData<T> golden;
    golden.k_golden.resize(params.b * params.s * params.n2 * params.topK * params.l_prime * (params.rope_dim + params.kv_lora_rank), 0);
    golden.v_golden.resize(params.b * params.s * params.n2 * params.topK * params.l_prime * params.kv_lora_rank, 0);
    golden.kvSlcActSeqs_golden.resize(params.b * params.s, 0);

    readInput(GetGoldenDir() + "/k_slc_out.bin", golden.k_golden);
    readInput(GetGoldenDir() + "/v_slc_out.bin", golden.v_golden);
    readInput(GetGoldenDir() + "/kv_slc_actual_seqs.bin", golden.kvSlcActSeqs_golden);
    return golden;
}

template <typename T>
void SetupKvSlcProgramData(const KvSlcTensors& tensors, const KvSlcInputData<T>& inputData, const KvSlcGoldenData<T>& goldenData) {
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int32_t>(tensors.topk_tensor, inputData.topkTensorData),
        RawTensorData::CreateTensor<int32_t>(tensors.topk_tensor_shape, inputData.topkTensorShapeData),
        RawTensorData::CreateTensor<T>(tensors.kvNopeCache, inputData.kvNopeCacheData),
        RawTensorData::CreateTensor<T>(tensors.kRopeCache, inputData.kRopeCacheData),
        RawTensorData::CreateTensor<int32_t>(tensors.kvActSeqs, inputData.kvActSeqsData),
        RawTensorData::CreateTensor<int32_t>(tensors.blockTable, inputData.blockTableData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<T>(tensors.k_slcOut, 0),
        RawTensorData::CreateConstantTensor<T>(tensors.v_slcOut, 0),
        RawTensorData::CreateConstantTensor<int32_t>(tensors.kvSlcActSeqs, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<T>(tensors.k_slcOut, goldenData.k_golden),
        RawTensorData::CreateTensor<T>(tensors.v_slcOut, goldenData.v_golden),
        RawTensorData::CreateTensor<int32_t>(tensors.kvSlcActSeqs, goldenData.kvSlcActSeqs_golden),
    });
}

template <typename T>
void RunAndVerifyKvSlc(const KvSlcGoldenData<T>& goldenData) {
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto k_Out = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    auto v_Out = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(1);
    auto kvSlcActSeqs_out = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(2);
    EXPECT_TRUE(resultCmp(goldenData.k_golden, (T *)k_Out->data(), 0.0005f));
    EXPECT_TRUE(resultCmp(goldenData.v_golden, (T *)v_Out->data(), 0.0005f));
    EXPECT_TRUE(resultCmp(goldenData.kvSlcActSeqs_golden, (int32_t *)kvSlcActSeqs_out->data(), 0.0005f));
}

template <typename T = npu::tile_fwk::float16, DataType tensorType = DataType::DT_FP16>
void testSlc(KvSlcTileShapeConfig& tileConfig) {
    SetInterpreterConfig();

    KvSlcParams params = ReadKvSlcParams();
    KvSlcTensors tensors = CreateKvSlcTensors(params, tensorType);

    GenKvSlc(tensors.topk_tensor, tensors.topk_tensor_shape, tensors.kvNopeCache, tensors.kRopeCache, tensors.kvActSeqs,
             params.front, params.near, params.topK, params.l_prime, params.n2, tensors.blockTable,
             params.blockSize, tensors.k_slcOut, tensors.v_slcOut, tensors.kvSlcActSeqs, tileConfig);

    auto inputData = ReadKvSlcInputData<T>(params);
    auto goldenData = ReadKvSlcGoldenData<T>(params);

    SetupKvSlcProgramData(tensors, inputData, goldenData);
    RunAndVerifyKvSlc(goldenData);
}

TEST_F(DynamicSlcTest, dynamic_p_slc_fp16) {
    KvSlcTileShapeConfig tileConfig;
    tileConfig.v0TileShape = {32, 32};
    testSlc<npu::tile_fwk::float16, DT_FP16>(tileConfig);
}

TEST_F(DynamicSlcTest, dynamic_p_slc_bf16) {
    KvSlcTileShapeConfig tileConfig;
    tileConfig.v0TileShape = {32, 32};
    testSlc<npu::tile_fwk::bfloat16, DT_BF16>(tileConfig);
}