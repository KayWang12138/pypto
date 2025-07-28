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
 * \file test_nsa_slc_attention.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "operation/tilefwk_op.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_storage.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "models/nsa/dynamic_nsa_v1.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/float.h"

using namespace npu::tile_fwk;

class NSAUtest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
        // skip pass, ut only execute model op code
        config::SetPassDefaultConfig(KEY_DISABLE_PASS, true);
    }

    void TearDown() override {}
};

template <typename T = npu::tile_fwk::float16, typename wDtype = int8_t, bool isSmooth = false, bool nz = false>
void TestNsa(const NSASimpleParams &params, const MlaTileConfig &prologConfig, SaTileShapeConfig& saTileConfig,
    KvSlcTileShapeConfig& kvSlcTileConfig, PostTileConfig& postConfig, std::string cacheMode = "PA_BSND") {
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
    int slcBlockSize = params.slcBlockSize;
    int front = params.front;
    int near = params.near;
    int topk = params.topk;

    std::vector<int> kvCacheActSeqVec(b, s2);
    int blockNum = 0;
    for (auto seqItem : kvCacheActSeqVec) {
        blockNum += CeilDiv(seqItem, blockSize);
    }
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
    Tensor winAtten(dType, shape_winAtten, "winAtten");

    Tensor kvSlcActSeqsMidOut(DT_INT32, slcActSeqsShape, "kvSlcActSeqsMidOut");
    Tensor attenOut(dType, shape_attentionOut, "attenOut");

    // post: Tensor
    Tensor wUv(dType, wUvShape, "wUv");
    Tensor wo(dTypeQuant, woShape, "wo", NodeType::LOCAL, weightFormat);
    Tensor woScale;
    Tensor smoothWo;
    Tensor postOut(dType, outShape, "postOut");

    MlaQuantInputs quantInputs;
    if (isQuant) {
        std::vector<int> w_qb_scale_shape = {1, n1 * qHeadDim};
        Tensor w_qb_scale = Tensor(DataType::DT_FP32, w_qb_scale_shape, "w_qb_scale");
        quantInputs.dequantScaleWUqQr = w_qb_scale;
        if (isSmooth) {
            std::vector<int> smooth_cq_shape = {1, qLoraRank};
            Tensor smooth_cq = Tensor(DT_FP32, smooth_cq_shape, "smooth_cq");
            quantInputs.smoothScalesCq = smooth_cq;
        }
    }

    // 3. 计算接口
    DynamicNsa(x, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos, cacheIndex, kvCache, krCache, quantInputs,
        prologConfig, eps, eps, cacheMode,
        topkIndices, topkTensorShape, /*kvNopeCache, kRopeCache,*/ kvCacheActSeq, blockTable, front, near, topk, slcBlockSize, blockSize, kvSlcTileConfig, // genKvSlc
        /*qNope, qRope,*/ slcActSeqs, softmaxScale, saTileConfig, // slcAttn
        /*x, */gateW1, gateW2, gateSimW1, GateMode::standard, // gatedscore
        cmpAtten, winAtten, // gen win
        wUv, wo, woScale, smoothWo, postConfig, // post
        outputQ, outputQRope, outputKvCache, outputKrCache, qNope, qRope, kvSlcActSeqsMidOut, kSlc, vSlc, slcAttn, attenOut, postOut);
}

TEST_F(NSAUtest, nsa_b_16_fp16) {
    NSASimpleParams params = NSASimpleParams::getDecodeParams();

    std::vector<int> inputParams = {16, 1, 8192, 128, 1, 0, 0};

    params.b = inputParams[0]; // 16
    params.s1 = inputParams[1];
    params.s2 = inputParams[2];
    params.n1 = inputParams[3];
    params.n2 = inputParams[4];
    int isQuant = inputParams[5];
    int isSmooth = inputParams[6];

    SaTileShapeConfig saTileConfig;
    const int gTile = 128; // for gLoop split
    const int sTile = 1024; // for s2Loop split
    saTileConfig.gTile = gTile;
    saTileConfig.sKvTile = sTile;
    saTileConfig.c1TileShape = {gTile, gTile, 64, 64, 128, 128}; // (n1, dn+dr) @ (s2Tile, dn+dr) -> (n1, s2Tile)
    saTileConfig.v1TileShape = {16, 256}; // (n1, s2Tile)
    saTileConfig.c2TileShape = {gTile, gTile, 64, 64, 128, 128}; // (n1, s2Tile) @ (s2Tile, dn) -> (n1, d)
    saTileConfig.v2TileShape = {16, 256}; // (n1, d)

    KvSlcTileShapeConfig kvSlcTileConfig;
    kvSlcTileConfig.v0TileShape = {32, 32};

    PostTileConfig postConfig = {16, 1};
    MlaTileConfig prologConfig = {16, 1};

    std::string cacheMode = "PA_BSND";
    if (isQuant == 1) {
        if (isSmooth == 1) {
            TestNsa<npu::tile_fwk::float16, int8_t, true>(params, prologConfig, saTileConfig, kvSlcTileConfig, postConfig, cacheMode);
        } else {
            TestNsa<npu::tile_fwk::float16, int8_t, false>(params, prologConfig, saTileConfig, kvSlcTileConfig, postConfig, cacheMode);
        }
    } else {
        TestNsa<npu::tile_fwk::float16, npu::tile_fwk::float16, false>(params, prologConfig, saTileConfig, kvSlcTileConfig, postConfig, cacheMode);
    }
}
