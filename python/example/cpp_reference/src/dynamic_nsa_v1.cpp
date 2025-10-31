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
 * \file dynamic_nsa.cpp
 * \brief
 */

#include "interface/function/function.h"
#include "interface/configs/config_manager.h"

using namespace npu::tile_fwk;

constexpr int NUM_65536 = 65536;
constexpr int NUM_128 = 128;
constexpr int NUM_2 = 2;
constexpr int NUM_16 = 16;

namespace npu::tile_fwk {
enum GateMode {
    standard,
    simple
};

struct SimpleParams {
    int b;
    int s;
    int s2;
    int d;
    int m;
    int k;
    int n;
    int n2;
    int right;
    int h;
    int q_lora_rank;
    int kv_lora_rank;
    int qk_rope_head_dim;
    int qk_nope_head_dim;
    int q_head_dim;
    std::string cacheMode;
    int blockSize;
    std::vector<int> vecTile;
    std::vector<int> cubeMTile;
    std::vector<int> cubeKTile;
    std::vector<int> cubeNTile;
    int tileB;
    static SimpleParams getCommonParams() {
        SimpleParams params;
        params.n2 = 1;
        params.s = 1;
        params.h = 7168; // 7168
        params.q_lora_rank = 1536; // 1536
        params.kv_lora_rank = 512; // 512
        params.qk_rope_head_dim = 64; // 64
        params.qk_nope_head_dim = 128; // 128
        params.q_head_dim = params.qk_rope_head_dim + params.qk_nope_head_dim;
        params.cacheMode = "BNSD";
        params.blockSize = 128; // 128
        return params;
    }

    static SimpleParams getLowParams() {
        SimpleParams params = getCommonParams();
        params.b = 4; // 4
        params.n = 32; // 32
        params.s2 = 256; // 256
        return params;
    }

    static SimpleParams getHighParams() {
        SimpleParams params = getCommonParams();
        params.b = 32; // 32
        params.n = 128; // 128
        params.s2 = 4096; // 4096
        return params;
    }
};


void GenGatedScore(const Tensor &x, const Tensor &gateW1, const Tensor &gateW2, const Tensor &gateSimW1,
    Tensor &gatingScore, GateMode gateMode) {
    (void)gateSimW1;
    (void)gateMode;
    DataType dType = x.GetStorage()->Datatype();

    int b = x.GetShape()[0];
    int s = x.GetShape()[1];
    int h = x.GetShape()[2];
    int n1 = gateW2.GetShape()[1] / 3;
    int tileB = b;
    int tileS = s;
    int tileBS = tileB * tileS;

    SymbolicScalar bLoop = b / tileB;
    SymbolicScalar sLoop = s / tileS;
    LOOP("LOOP_L0_bIdx_gated_score", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bLoop, 1)) {
        LOOP("LOOP_L0_sIdx_gated_score", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sLoop, 1)) {
            TileShape::Current().SetVecTile({tileB, tileS, h});
            TileShape::Current().SetCubeTile({tileBS, tileBS}, {NUM_128, NUM_128}, {NUM_128, NUM_128});
            SymbolicScalar bOfs = bIdx * tileB;
            SymbolicScalar sOfs = sIdx * tileS;
            SymbolicScalar bsOfs = bOfs * sOfs;

            auto xReshape = Reshape(x, {b * s, h});
            auto xView = View(xReshape, {tileBS, h}, {bsOfs, 0});
            auto mm1Res = Matrix::Matmul(DT_FP32, xReshape, gateW1);

            TileShape::Current().SetVecTile({1, h});
            auto sigmoidRes = Sigmoid(mm1Res);
            sigmoidRes = Cast(sigmoidRes, dType);
            TileShape::Current().SetCubeTile({tileBS, tileBS}, {NUM_128, NUM_128}, {NUM_16, NUM_16});
            auto mm2Res = Matrix::Matmul(DT_FP32, sigmoidRes, gateW2);
            TileShape::Current().SetVecTile({tileBS, n1});

            auto res = Reshape(mm2Res, {tileB, tileS, 3, n1});
            TileShape::Current().SetVecTile({1, tileS, 3, n1});

            res = Transpose(res, {2, 3});
            if (gatingScore.GetStorage()->Datatype() != DT_FP32) {
                res = Cast(res, dType);
            }
            Assemble(res, {bOfs, sIdx, 0, 0}, gatingScore);
        }
    }
}

void GenGatedScoreCompute(const Tensor &x, const Tensor &gateW1, const Tensor &gateW2, const Tensor &gateSimW1,
    Tensor &gatingScore, GateMode gateMode) {
    FUNCTION("FusedCompressKvSelect", {x, gateW1, gateW2, gateSimW1}, {gatingScore}) {
        GenGatedScore(x, gateW1, gateW2, gateSimW1, gatingScore, gateMode);
    }
}


template <typename T = npu::tile_fwk::float16, typename wDtype = int8_t, bool splitK = false, bool nz = true,
    bool isSmooth = true, bool usePrefetch = true>
void TestNsa(const SimpleParams &params) {
    int b = params.b;
    int s = params.s;
    int n = params.n;
    int h = params.h;

    DataType dType = (std::is_same<T, npu::tile_fwk::float16>::value) ? DT_FP16 : DT_BF16;

    std::vector<int64_t> x_shape = {b, s, h};
    std::vector<int64_t> gateW1Shape = {h, 4 * h};
    std::vector<int64_t> gateW2Shape = {4 * h, 3 * n};
    std::vector<int64_t> gateSimW1Shape = {h, 3 * n};
    //    std::vector<int64_t> gatingScoreShape = {b, n, s, 3};
    std::vector<int64_t> gatingScoreShape = {b, s, n, 3};
    std::vector<int64_t> tempShape = {b * s, n * 3};
    std::vector<int64_t> mm1Shape = {b * s, 4 * h};

    Tensor x(dType, x_shape, "x");
    Tensor gateW1(dType, gateW1Shape, "gateW1");
    Tensor gateW2(dType, gateW2Shape, "gateW2");
    Tensor gateSimW1(dType, gateSimW1Shape, "gateSimW1");
    Tensor gatingScore(dType, gatingScoreShape, "gatingScore");
    GenGatedScoreCompute(x, gateW1, gateW2, gateSimW1, gatingScore, GateMode::standard);
}

} // namespace npu::tile_fwk


int main() {
    SimpleParams params = SimpleParams::getHighParams();
    params.h = NUM_128;
    params.s = NUM_2;
    TestNsa<npu::tile_fwk::float16>(params);
}