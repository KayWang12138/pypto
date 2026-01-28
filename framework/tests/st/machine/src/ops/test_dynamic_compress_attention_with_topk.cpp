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
 * \file test_dynamic_compress_attention_with_topk.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "interface/tensor/float.h"
#include "tilefwk/data_type.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "machine/device/dynamic/device_utils.h"
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/nsa/compress_attention_with_topk.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class CmpAttnTopk : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

template <typename T>
static std::shared_ptr<RawTensorData> CreateTensorData(Tensor tensor, std::string fileName) {
    auto shape = tensor.GetShape();
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> values(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, values);
    return RawTensorData::CreateTensor<T>(tensor, values);
}

template <typename T>
DataType GetCmpAttnTopkDataType() {
    if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        return DT_BF16;
    } else if (std::is_same<T, npu::tile_fwk::float16>::value) {
        return DT_FP16;
    }
    return DT_FP32;
}

struct CmpAttnTopkParams {
    int b;
    int s1;
    int n1;
    int dn;
    int dr;
    int n2;
    int blockSize;
    int cmpBlockSize;
    int cmpStride;
    int slcBlockSize;
    int topk;
    int front;
    int near;
    float softmaxScale;
};

CmpAttnTopkParams ReadCmpAttnTopkParams() {
    CmpAttnTopkParams params;
    int paramsSize = 13;
    std::vector<int> input_param(paramsSize);
    readInput<int>(GetGoldenDir() + "/in_params.bin", input_param);

    params.b = input_param[0];
    params.s1 = input_param[1];
    params.n1 = input_param[2];
    params.dn = input_param[3];
    params.dr = input_param[4];
    params.n2 = input_param[5];
    params.blockSize = input_param[6];
    params.cmpBlockSize = input_param[7];
    params.cmpStride = input_param[8];
    params.slcBlockSize = input_param[9];
    params.topk = input_param[10];
    params.front = input_param[11];
    params.near = input_param[12];
    params.softmaxScale = static_cast<float>(1.0 / sqrtf((params.dn + params.dr)));

    ALOG_EVENT_F(R"(
TestCmpAttnTopk params:
    b=%d
    s1=%d
    n1=%d
    dn=%d
    dr=%d
    n2=%d
    blockSize=%d
    cmpBlockSize=%d
    cmpStride=%d
    slcBlockSize=%d
    topk=%d
    front=%d
    near=%d
)",
        params.b, params.s1, params.n1, params.dn, params.dr, params.n2, params.blockSize, params.cmpBlockSize,
        params.cmpStride, params.slcBlockSize, params.topk, params.front, params.near);

    return params;
}

struct CmpAttnSeqInfo {
    std::vector<int> actCmpSeq;
    int cmpBlockNum;
    int maxCmpSeq;
    int maxCmpBlockNum;
    int slcSize;
    int blockSlcNum;
};

CmpAttnSeqInfo ProcessCmpAttnSeq(const CmpAttnTopkParams& params) {
    CmpAttnSeqInfo seqInfo;
    std::vector<int> actSeqLen(params.b);
    readInput<int>(GetGoldenDir() + "/act_seq.bin", actSeqLen);

    for (auto curSeq : actSeqLen) {
        auto curCmpSeq = (curSeq - params.cmpBlockSize) / params.cmpStride + 1;
        seqInfo.actCmpSeq.emplace_back(curCmpSeq);
    }

    seqInfo.cmpBlockNum = 0;
    for (auto s : seqInfo.actCmpSeq) {
        seqInfo.cmpBlockNum += CeilDiv(s, params.blockSize);
    }
    seqInfo.maxCmpSeq = *(std::max_element(seqInfo.actCmpSeq.begin(), seqInfo.actCmpSeq.end()));
    seqInfo.maxCmpBlockNum = CeilDiv(seqInfo.maxCmpSeq, params.blockSize);
    seqInfo.slcSize = params.slcBlockSize / params.cmpStride;
    seqInfo.blockSlcNum = params.blockSize / seqInfo.slcSize;

    return seqInfo;
}

struct CmpAttnTopkTensors {
    Tensor qNope;
    Tensor qRope;
    Tensor cmpKvCache;
    Tensor cmpKrCache;
    Tensor cmpBlockTable;
    Tensor actSeq;
    Tensor auxTensor;
    Tensor incSeq;
    Tensor cmpAttn;
    Tensor topkRes;
};

CmpAttnTopkTensors CreateCmpAttnTopkTensors(const CmpAttnTopkParams& params, const CmpAttnSeqInfo& seqInfo, DataType dType) {
    CmpAttnTopkTensors tensors;
    tensors.qNope = Tensor(dType, {params.b * params.s1 * params.n1, params.dn}, "qNope");
    tensors.qRope = Tensor(dType, {params.b * params.s1 * params.n1, params.dr}, "qRope");
    tensors.cmpKvCache = Tensor(dType, {seqInfo.cmpBlockNum, params.blockSize, params.n2, params.dn}, "cmpKvCache");
    tensors.cmpKrCache = Tensor(dType, {seqInfo.cmpBlockNum, params.blockSize, params.n2, params.dr}, "cmpKrCache");
    tensors.cmpBlockTable = Tensor(DT_INT32, {params.b, seqInfo.maxCmpBlockNum}, "cmpBlockTable");
    tensors.actSeq = Tensor(DT_INT32, {params.b}, "actSeq");
    tensors.auxTensor = Tensor(DT_FP32, {params.slcBlockSize / params.cmpStride + params.cmpBlockSize / params.cmpStride - 1, params.n1}, "auxTensor");
    tensors.incSeq = Tensor(DT_INT32, {1, 1, seqInfo.maxCmpBlockNum * seqInfo.blockSlcNum}, "incSeq");
    tensors.cmpAttn = Tensor(DT_FP32, {params.b, params.s1, params.n1, params.dn}, "cmpAttnOut");
    tensors.topkRes = Tensor(DT_INT32, {params.b, params.s1, params.topk}, "topkRes");

    return tensors;
}

struct CmpAttnTopkGoldenData {
    std::vector<float> attnGolden;
    std::vector<int32_t> topkGolden;
};

CmpAttnTopkGoldenData ReadCmpAttnTopkGoldens(const CmpAttnTopkParams& params) {
    CmpAttnTopkGoldenData golden;
    golden.attnGolden.resize(params.b * params.s1 * params.n1 * params.dn, 0.0);
    golden.topkGolden.resize(params.b * params.s1 * params.topk, 0);

    readInput(GetGoldenDir() + "/cmp_attn_out.bin", golden.attnGolden);
    readInput(GetGoldenDir() + "/topk_res.bin", golden.topkGolden);

    return golden;
}

template <typename T>
struct CmpAttnTopkInputData {
    std::shared_ptr<RawTensorData> qNopeData;
    std::shared_ptr<RawTensorData> qRopeData;
    std::shared_ptr<RawTensorData> cmpKvCacheData;
    std::shared_ptr<RawTensorData> cmpKrCacheData;
    std::shared_ptr<RawTensorData> cmpBlockTableData;
    std::shared_ptr<RawTensorData> actSeqData;
    std::shared_ptr<RawTensorData> auxData;
    std::shared_ptr<RawTensorData> cmpAttnData;
    std::shared_ptr<RawTensorData> topkResData;
};

template <typename T>
CmpAttnTopkInputData<T> PrepareCmpAttnTopkData(const CmpAttnTopkTensors& tensors) {
    CmpAttnTopkInputData<T> data;

    data.qNopeData = CreateTensorData<T>(tensors.qNope, "/q_nope.bin");
    data.qRopeData = CreateTensorData<T>(tensors.qRope, "/q_rope.bin");
    data.cmpKvCacheData = CreateTensorData<T>(tensors.cmpKvCache, "/cmp_kv_cache.bin");
    data.cmpKrCacheData = CreateTensorData<T>(tensors.cmpKrCache, "/cmp_kr_cache.bin");
    data.cmpBlockTableData = CreateTensorData<int32_t>(tensors.cmpBlockTable, "/cmp_block_table.bin");
    data.actSeqData = CreateTensorData<int32_t>(tensors.actSeq, "/act_seq.bin");
    data.auxData = CreateTensorData<float>(tensors.auxTensor, "/aux_tensor.bin");
    data.cmpAttnData = RawTensorData::CreateConstantTensor<float>(tensors.cmpAttn, 0.0f);
    data.topkResData = RawTensorData::CreateConstantTensor<int32_t>(tensors.topkRes, 0);

    return data;
}

template <typename T>
void SetupCmpAttnTopkProgram(const CmpAttnTopkTensors& tensors, const CmpAttnTopkInputData<T>& data,
                             const CmpAttnTopkParams& params, CmpAttnTopkTile& tileConfig) {
    std::vector<RawTensorDataPtr> inputDataList = {
        data.qNopeData, data.qRopeData, data.cmpKvCacheData, data.cmpKrCacheData,
        data.cmpBlockTableData, data.actSeqData, data.auxData
    };
    std::vector<RawTensorDataPtr> outputDataList = {data.cmpAttnData, data.topkResData};

    FUNCTION("CompressAttentionWithTopK",
        {tensors.qNope, tensors.qRope, tensors.cmpKvCache, tensors.cmpKrCache, tensors.cmpBlockTable,
         tensors.actSeq, tensors.auxTensor}, {tensors.cmpAttn, tensors.topkRes}) {
        CompressAttentionWithTopK(tensors.qNope, tensors.qRope, tensors.cmpKvCache, tensors.cmpKrCache,
            tensors.cmpBlockTable, tensors.actSeq, tensors.auxTensor, tensors.cmpAttn, tensors.topkRes,
            params.blockSize, params.cmpBlockSize, params.cmpStride, params.slcBlockSize, params.softmaxScale,
            params.n1, params.topk, params.front, params.near, tileConfig);
    }

    auto funcop = Program::GetInstance().GetLastFunction();
    ProgramData::GetInstance().AppendInputs(inputDataList);
    ProgramData::GetInstance().AppendOutputs(outputDataList);
    DevFuncRunner::Run(funcop);
}

template <typename T>
void RunAndVerifyCmpAttnTopk(const CmpAttnTopkInputData<T>& data, const CmpAttnTopkGoldenData& golden) {
    float eps = 3e-3f;
    std::cout << "=======================attnOut===============================" << std::endl;
    EXPECT_TRUE(resultCmp(golden.attnGolden, (float *)data.cmpAttnData->data(), eps, 100));
    std::cout << "=======================topkRes===============================" << std::endl;
    EXPECT_TRUE(resultCmp(golden.topkGolden, (int32_t *)data.topkResData->data(), 0, 0, 3, false, false, 128));
}

template <typename T = npu::tile_fwk::bfloat16>
void TestCmpAttnTopk(CmpAttnTopkTile &tileConfig) {
    DataType dType = GetCmpAttnTopkDataType<T>();
    CmpAttnTopkParams params = ReadCmpAttnTopkParams();
    CmpAttnSeqInfo seqInfo = ProcessCmpAttnSeq(params);
    CmpAttnTopkTensors tensors = CreateCmpAttnTopkTensors(params, seqInfo, dType);

    auto goldenData = ReadCmpAttnTopkGoldens(params);
    auto inputData = PrepareCmpAttnTopkData<T>(tensors);

    SetupCmpAttnTopkProgram(tensors, inputData, params, tileConfig);
    RunAndVerifyCmpAttnTopk(inputData, goldenData);
}

void CommonTestConfig() {

    CmpAttnTopkTile config;
    config.topkTile = {1, 1, 128};
    config.cmpTile.c1Tile = {128, 128, 128, 128, 128, 128};
    config.cmpTile.v1Tile = {128, 128};
    config.cmpTile.c2Tile = {128, 128, 128, 128, 128, 128};
    config.cmpTile.v2Tile = {128, 64};

    TestCmpAttnTopk<npu::tile_fwk::bfloat16>(config);
}

TEST_F(CmpAttnTopk, cmp_attn_with_topk_singleop_bf16) {
    CommonTestConfig();
}