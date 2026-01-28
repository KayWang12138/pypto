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
 * \file test_dynamic_win_atten.cpp
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
#include "operator/models/nsa/win_attention.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class DynamicWinAttenTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

constexpr int NUM_2 = 2;
constexpr int NUM_16 = 16;
constexpr int NUM_32 = 32;
constexpr int NUM_64 = 64;
constexpr int NUM_128 = 128;
constexpr int NUM_256 = 256;
constexpr int NUM_512 = 512;
constexpr int NUM_1024 = 1024;

template <typename T>
DataType GetDataType() {
    if (std::is_same<T, npu::tile_fwk::float16>::value) {
        return DT_FP16;
    } else if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        return DT_BF16;
    }
    return DT_FP32;
}

struct WinAttenParams {
    int b;
    int sQ;
    int nQ;
    int nKV;
    int sMax;
    int dN;
    int dR;
    int blockSize;
    int windowSize;
    float softmaxScale;
    int maxBlock;
};

WinAttenParams ReadWinAttenParams() {
    WinAttenParams params;
    int paramsSize = 9;
    std::vector<int> inputParam(paramsSize);
    readInput<int>(GetGoldenDir() + "/input_param.bin", inputParam);

    params.b = inputParam[0];
    params.sQ = inputParam[1];
    params.nQ = inputParam[2];
    params.nKV = inputParam[3];
    params.sMax = inputParam[4];
    params.dN = inputParam[5];
    params.dR = inputParam[6];
    params.blockSize = inputParam[7];
    params.windowSize = inputParam[8];
    params.softmaxScale = static_cast<float>(1.0 / sqrtf((params.dN + params.dR)));
    params.maxBlock = (params.sMax + params.blockSize - 1) / params.blockSize;

    std::cout << "====input param==== " << std::endl;
    std::cout << " b = " << params.b << " sQ = " << params.sQ << " nQ = " << params.nQ
        << " nKV = " << params.nKV << " sMax =" << params.sMax << " dN = " << params.dN
        << " dR = " << params.dR << " blockSize = " << params.blockSize
        << " windowSize = " << params.windowSize << std::endl;

    return params;
}

struct WinAttenTensors {
    Tensor actSeqs;
    Tensor qNope;
    Tensor qRope;
    Tensor vNopeCache;
    Tensor kRopeCache;
    Tensor blockTable;
    Tensor attentionOut;
};

WinAttenTensors CreateWinAttenTensors(const WinAttenParams& params, DataType dType) {
    WinAttenTensors tensors;
    std::vector<int64_t> qNopeShape = {params.b * params.sQ * params.nQ, params.dN};
    std::vector<int64_t> qRopeShape = {params.b * params.sQ * params.nQ, params.dR};
    std::vector<int64_t> vNopeCacheShape = {params.b * params.maxBlock * params.blockSize, params.nKV * params.dN};
    std::vector<int64_t> kRopeCacheShape = {params.b * params.maxBlock * params.blockSize, params.nKV * params.dR};
    std::vector<int64_t> attentionOutShape = {params.b, params.sQ, params.nQ, params.dN};
    std::vector<int64_t> blockTableShape = {params.b, params.maxBlock};

    tensors.actSeqs = Tensor(DT_INT32, {params.b}, "actSeqs");
    tensors.qNope = Tensor(dType, qNopeShape, "qNope");
    tensors.qRope = Tensor(dType, qRopeShape, "qRope");
    tensors.vNopeCache = Tensor(dType, vNopeCacheShape, "vNopeCache");
    tensors.kRopeCache = Tensor(dType, kRopeCacheShape, "kRopeCache");
    tensors.blockTable = Tensor(DT_INT32, blockTableShape, "blockTable");
    tensors.attentionOut = Tensor(DT_FP32, attentionOutShape, "attentionOut");

    return tensors;
}

template <typename T>
struct WinAttenInputData {
    std::vector<int> seq;
    std::vector<T> qNopeData;
    std::vector<T> qRopeData;
    std::vector<T> vNopeCacheData;
    std::vector<T> kRopeCacheData;
    std::vector<int> blockTableData;
};

template <typename T>
WinAttenInputData<T> ReadWinAttenInputData(const WinAttenParams& params) {
    WinAttenInputData<T> data;

    int qNopeSize = params.b * params.sQ * params.nQ * params.dN;
    int qRopeSize = params.b * params.sQ * params.nQ * params.dR;
    int vNopeCacheSize = params.b * params.maxBlock * params.blockSize * params.nKV * params.dN;
    int kRopeCacheSize = params.b * params.maxBlock * params.blockSize * params.nKV * params.dR;
    int blockTableSize = params.b * params.maxBlock;

    data.seq.resize(params.b);
    data.qNopeData.resize(qNopeSize, 0);
    data.qRopeData.resize(qRopeSize, 0);
    data.vNopeCacheData.resize(vNopeCacheSize, 0);
    data.kRopeCacheData.resize(kRopeCacheSize, 0);
    data.blockTableData.resize(blockTableSize, 0);

    readInput<int>(GetGoldenDir() + "/actual_seq_list.bin", data.seq);
    readInput<T>(GetGoldenDir() + "/q_nope.bin", data.qNopeData);
    readInput<T>(GetGoldenDir() + "/q_rope.bin", data.qRopeData);
    readInput<T>(GetGoldenDir() + "/k_cache_nope.bin", data.vNopeCacheData);
    readInput<T>(GetGoldenDir() + "/k_cache_rope.bin", data.kRopeCacheData);
    readInput<int>(GetGoldenDir() + "/block_table.bin", data.blockTableData);

    return data;
}

template <typename T>
void SetupWinAttenProgramData(const WinAttenTensors& tensors, const WinAttenInputData<T>& inputData,
                              const std::vector<float>& golden) {
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(tensors.qNope, inputData.qNopeData),
        RawTensorData::CreateTensor<T>(tensors.vNopeCache, inputData.vNopeCacheData),
        RawTensorData::CreateTensor<T>(tensors.qRope, inputData.qRopeData),
        RawTensorData::CreateTensor<T>(tensors.kRopeCache, inputData.kRopeCacheData),
        RawTensorData::CreateTensor<int32_t>(tensors.blockTable, inputData.blockTableData),
        RawTensorData::CreateTensor<int32_t>(tensors.actSeqs, inputData.seq),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(tensors.attentionOut, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(tensors.attentionOut, golden),
    });
}

void RunAndVerifyWinAtten(const std::vector<float>& golden) {
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.0005f));
}

template <typename T = npu::tile_fwk::float16>
void TestWinAtten(WinAttenTileShapeConfig& tileConfig) {
    SetInterpreterConfig();

    DataType dType = GetDataType<T>();
    WinAttenParams params = ReadWinAttenParams();
    WinAttenTensors tensors = CreateWinAttenTensors(params, dType);

    auto inputData = ReadWinAttenInputData<T>(params);

    int winAttenOutSize = params.b * params.sQ * params.nQ * params.dN;
    std::vector<float> golden(winAttenOutSize, 0);
    readInput(GetGoldenDir() + "/atten_out.bin", golden);

    SetupWinAttenProgramData(tensors, inputData, golden);

    WinAttention(tensors.qNope, tensors.vNopeCache, tensors.qRope, tensors.kRopeCache,
                 params.nQ, params.nKV, tensors.blockTable, tensors.actSeqs, params.windowSize,
                 params.blockSize, params.softmaxScale, tensors.attentionOut, tileConfig);

    RunAndVerifyWinAtten(golden);
}

TEST_F(DynamicWinAttenTest, test_DynAttn_nas_win_attn_s1_2_actseqlen_1024_mla_fp16_v1) {
    WinAttenTileShapeConfig tileConfig;
    const int gTileSize = NUM_128; // for gLoop split
    const int skvTileSize = NUM_512; // for flash split
    tileConfig.gTile = gTileSize;
    tileConfig.skvTile = skvTileSize;
    tileConfig.vNopeTileShape = {NUM_32, NUM_512};
    tileConfig.vRopeTileShape = {NUM_128, NUM_64};
    tileConfig.c1TileShape = {gTileSize, gTileSize, NUM_64, NUM_64, NUM_256, NUM_256}; // (n1, dN+dR) @ (s2Tile, dN+dR) -> (n1, s2Tile)
    tileConfig.v1TileShape = {NUM_32, NUM_256}; // (n1, s2Tile)
    tileConfig.c2TileShape = {gTileSize, gTileSize, NUM_256, NUM_256, NUM_128, NUM_128}; // (n1, s2Tile) @ (s2Tile, dN) -> (n1, d)
    tileConfig.v2TileShape = {NUM_32, NUM_512}; // (n1, d)
    // WinConfig config;
    TestWinAtten<npu::tile_fwk::float16>(tileConfig);
}

TEST_F(DynamicWinAttenTest, test_DynAttn_nas_win_attn_s1_2_actseqlen_1023_mla_fp16_unalign) {
    WinAttenTileShapeConfig tileConfig;
    const int gTileSize = NUM_128; // for gLoop split
    const int skvTileSize = NUM_512; // for flash split
    tileConfig.gTile = gTileSize;
    tileConfig.skvTile = skvTileSize;
    tileConfig.vNopeTileShape = {NUM_32, NUM_512};
    tileConfig.vRopeTileShape = {NUM_128, NUM_64};
    tileConfig.c1TileShape = {gTileSize, gTileSize, NUM_64, NUM_64, NUM_256, NUM_256}; // (n1, dN+dR) @ (s2Tile, dN+dR) -> (n1, s2Tile)
    tileConfig.v1TileShape = {NUM_32, NUM_256}; // (n1, s2Tile)
    tileConfig.c2TileShape = {gTileSize, gTileSize, NUM_256, NUM_256, NUM_128, NUM_128}; // (n1, s2Tile) @ (s2Tile, dN) -> (n1, d)
    tileConfig.v2TileShape = {NUM_32, NUM_512}; // (n1, d)
    // WinConfig config;
    TestWinAtten<npu::tile_fwk::float16>(tileConfig);
}

TEST_F(DynamicWinAttenTest, test_DynAttn_nas_win_attn_s1_2_actseqlen_1024_mla_bf16) {
    WinAttenTileShapeConfig tileConfig;
    const int gTileSize = NUM_128; // for gLoop split
    const int skvTileSize = NUM_512; // for flash split
    tileConfig.gTile = gTileSize;
    tileConfig.skvTile = skvTileSize;
    tileConfig.vNopeTileShape = {NUM_32, NUM_512};
    tileConfig.vRopeTileShape = {NUM_128, NUM_64};
    tileConfig.c1TileShape = {gTileSize, gTileSize, NUM_64, NUM_64, NUM_256, NUM_256}; // (n1, dN+dR) @ (s2Tile, dN+dR) -> (n1, s2Tile)
    tileConfig.v1TileShape = {NUM_32, NUM_256}; // (n1, s2Tile)
    tileConfig.c2TileShape = {gTileSize, gTileSize, NUM_256, NUM_256, NUM_128, NUM_128}; // (n1, s2Tile) @ (s2Tile, dN) -> (n1, d)
    tileConfig.v2TileShape = {NUM_32, NUM_512}; // (n1, d)
    // WinConfig config;
    TestWinAtten<npu::tile_fwk::bfloat16>(tileConfig);
}

TEST_F(DynamicWinAttenTest, test_DynAttn_nas_win_attn_s1_2_actseqlen_1023_mla_bf16_unalign) {
    WinAttenTileShapeConfig tileConfig;
    const int gTileSize = NUM_128; // for gLoop split
    const int skvTileSize = NUM_512; // for flash split
    tileConfig.gTile = gTileSize;
    tileConfig.skvTile = skvTileSize;
    tileConfig.vNopeTileShape = {NUM_32, NUM_512};
    tileConfig.vRopeTileShape = {NUM_128, NUM_64};
    tileConfig.c1TileShape = {gTileSize, gTileSize, NUM_64, NUM_64, NUM_256, NUM_256}; // (n1, dN+dR) @ (s2Tile, dN+dR) -> (n1, s2Tile)
    tileConfig.v1TileShape = {NUM_32, NUM_256}; // (n1, s2Tile)
    tileConfig.c2TileShape = {gTileSize, gTileSize, NUM_256, NUM_256, NUM_128, NUM_128}; // (n1, s2Tile) @ (s2Tile, dN) -> (n1, d)
    tileConfig.v2TileShape = {NUM_32, NUM_512}; // (n1, d)
    // WinConfig config;
    TestWinAtten<npu::tile_fwk::bfloat16>(tileConfig);
}