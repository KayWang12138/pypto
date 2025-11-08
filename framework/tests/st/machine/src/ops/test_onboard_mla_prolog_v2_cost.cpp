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
 * \file test_onboard_mla_prolog_v2_cost.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "interface/inner/tilefwk/tilefwk_api.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "tilefwk/data_type.h"
#include "test_common.h"
#include "test_suite_stest_ops.h"
#include "interface/tensor/float.h"
#include "operator/models/deepseek/mla_prolog.h"
#include "runtime.h"
#include "device_runner.h"
#include "tilefwk_runtime_api.h"

using namespace npu::tile_fwk;

class MlaPrologV2OnBoardCostTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

template <typename T = npu::tile_fwk::float16, bool splitReduceLastDim = true, bool splitK = false, bool nz= false>
void TestMlaPrologV2(std::vector<int> &params, string dataPath, uint64_t timeThreshold, bool isQuant = false, bool hasSmooth = false) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, vHeadDim
    int b = params[0];
    int s = params[1];
    int s2 = params[2];
    int n = params[3];
    int h = params[4];
    int qLoraRank = params[5];
    int qkNopeHeadDim = params[6];
    int qkRopeHeadDim = params[7];
    int kvLoraRank = params[8];
    int q_head_dim = qkNopeHeadDim + qkRopeHeadDim;

    DataType dType = DT_FP32;
    if (std::is_same<T, npu::tile_fwk::float16>::value) {
        dType = DT_FP16;
    } else if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        dType = DT_BF16;
    } else {
        dType = DT_FP32;
    }

    DataType dTypeQuantIn = isQuant ? DT_INT8 : dType;
    // typedef float outDtype;
    typedef T outDtype;
    typedef int8_t wDtype;

    std::vector<int64_t> x_shape = {b, s, h};
    std::vector<int64_t> w_qa_shape = {h, qLoraRank};
    std::vector<int64_t> w_qb_shape = {qLoraRank, n * q_head_dim};
    std::vector<int64_t> w_kv_a_shape = {h, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> w_kv_b_k_shape = {n, qkNopeHeadDim, kvLoraRank};
    std::vector<int64_t> cos_shape = {b, s, qkRopeHeadDim};
    std::vector<int64_t> gamma_cq_shape = {qLoraRank};
    std::vector<int64_t> gamma_ckv_shape = {kvLoraRank};
    std::vector<int64_t> kv_len_shape = {b, s};
    std::vector<int64_t> kv_cache_shape = {b, 1, s2, kvLoraRank};
    std::vector<int64_t> kr_cache_shape = {b, 1, s2, qkRopeHeadDim};
    // output
    std::vector<int64_t> q_out_shape = {b, s, n, kvLoraRank};
    std::vector<int64_t> q_rope_out_shape = {b, s, n, qkRopeHeadDim};
    std::vector<int64_t> kv_cache_out_shape = {b, 1, s2, kvLoraRank};
    std::vector<int64_t> kr_cache_out_shape = {b, 1, s2, qkRopeHeadDim};

    int capacity_x = std::accumulate(x_shape.begin(), x_shape.end(), 1, std::multiplies<>());
    int wDqCapacity = std::accumulate(w_qa_shape.begin(), w_qa_shape.end(), 1, std::multiplies<>());
    int wUqQrCapacity = std::accumulate(w_qb_shape.begin(), w_qb_shape.end(), 1, std::multiplies<>());
    int wDkvKrCapacity = std::accumulate(w_kv_a_shape.begin(), w_kv_a_shape.end(), 1, std::multiplies<>());
    int wUkCapacity = std::accumulate(w_kv_b_k_shape.begin(), w_kv_b_k_shape.end(), 1, std::multiplies<>());
    int capacity_cos = std::accumulate(cos_shape.begin(), cos_shape.end(), 1, std::multiplies<>());
    int capacity_gamma_cq = std::accumulate(gamma_cq_shape.begin(), gamma_cq_shape.end(), 1, std::multiplies<>());
    int capacity_gamma_ckv = std::accumulate(gamma_ckv_shape.begin(), gamma_ckv_shape.end(), 1, std::multiplies<>());
    int capacity_kv_len = std::accumulate(kv_len_shape.begin(), kv_len_shape.end(), 1, std::multiplies<>());
    int capacity_kv_cache = std::accumulate(kv_cache_shape.begin(), kv_cache_shape.end(), 1, std::multiplies<>());
    int capacity_kr_cache = std::accumulate(kr_cache_shape.begin(), kr_cache_shape.end(), 1, std::multiplies<>());
    // output
    int capacity_q_out = std::accumulate(q_out_shape.begin(), q_out_shape.end(), 1, std::multiplies<>());
    int capacity_q_rope_out = std::accumulate(q_rope_out_shape.begin(), q_rope_out_shape.end(), 1, std::multiplies<>());
    int capacity_kv_out = std::accumulate(kv_cache_out_shape.begin(), kv_cache_out_shape.end(), 1, std::multiplies<>());
    int capacity_kr_out = std::accumulate(kr_cache_out_shape.begin(), kr_cache_out_shape.end(), 1, std::multiplies<>());

    std::vector<int64_t> w_qb_scale_shape;
    int capacity_w_qb_scale;
    if (isQuant) {
        w_qb_scale_shape = {1, n * q_head_dim};
        capacity_w_qb_scale = std::accumulate(w_qb_scale_shape.begin(), w_qb_scale_shape.end(), 1, std::multiplies<>());
    }
    std::vector<int64_t> smooth_cq_shape{1, qLoraRank};
    int capacity_smooth_cq = std::accumulate(smooth_cq_shape.begin(), smooth_cq_shape.end(), 1, std::multiplies<>());

    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    TileFwkInit("");

    uint64_t outputSize0 = capacity_q_out * sizeof(T);
    uint64_t outputSize1 = capacity_q_rope_out * sizeof(T);
    uint64_t outputSize2 = capacity_kv_out * sizeof(T);
    uint64_t outputSize3 = capacity_kr_out * sizeof(T);
    uint8_t *q_out_ptr = allocDevAddr(outputSize0);
    uint8_t *q_rope_out_ptr = allocDevAddr(outputSize1);
    void *kv_cache_ptr = readToDev<T>(dataPath + "/kv_cache.bin", capacity_kv_cache);
    void *kr_cache_ptr = readToDev<T>(dataPath + "/kr_cache.bin", capacity_kr_cache);


    void *x_ptr = readToDev<T>(dataPath + "/x.bin", capacity_x);
    void *wDqPtr = readToDev<T>(dataPath + "/wDq.bin", wDqCapacity);
    void *wUqQrPtr = isQuant ? readToDev<wDtype>(dataPath + "/wUqQr.bin", wUqQrCapacity) :
                               readToDev<T>(dataPath + "/wUqQr.bin", wUqQrCapacity);
    void *wDkvKrPtr = readToDev<T>(dataPath + "/wDkvKr.bin", wDkvKrCapacity);
    void *wUkPtr = readToDev<T>(dataPath + "/wUk.bin", wUkCapacity);
    void *gamma_cq_ptr = readToDev<T>(dataPath + "/gamma_cq.bin", capacity_gamma_cq);
    void *gamma_ckv_ptr = readToDev<T>(dataPath + "/gamma_ckv.bin", capacity_gamma_ckv);
    void *cos_ptr = readToDev<T>(dataPath + "/cos.bin", capacity_cos);
    void *sin_ptr = readToDev<T>(dataPath + "/sin.bin", capacity_cos);
    void *kv_len_ptr = readToDev<int64_t>(dataPath + "/kv_len.bin", capacity_kv_len); // int64

    Tensor x(dType, x_shape, (uint8_t *)x_ptr, "x");
    TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    Tensor wDq(dType, w_qa_shape, (uint8_t *)wDqPtr, "wDq", weightFormat);
    Tensor wUqQr(dTypeQuantIn, w_qb_shape, (uint8_t *)wUqQrPtr, "wUqQr", weightFormat);
    Tensor wDkvKr(dType, w_kv_a_shape, (uint8_t *)wDkvKrPtr, "wDkvKr", weightFormat);
    Tensor wUk(dType, w_kv_b_k_shape, (uint8_t *)wUkPtr, "wUk", weightFormat);
    Tensor gamma_cq(dType, gamma_cq_shape, (uint8_t *)gamma_cq_ptr, "gamma_cq");
    Tensor gamma_ckv(dType, gamma_ckv_shape, (uint8_t *)gamma_ckv_ptr, "gamma_ckv");
    Tensor cos(dType, cos_shape, (uint8_t *)cos_ptr, "cos");
    Tensor sin(dType, cos_shape, (uint8_t *)sin_ptr, "sin");
    Tensor kv_len(DT_INT64, kv_len_shape, (uint8_t *)kv_len_ptr, "kv_len"); // int64
    Tensor kv_cache(dType, kv_cache_shape, (uint8_t *)kv_cache_ptr, "kv_cache");
    Tensor kr_cache(dType, kr_cache_shape, (uint8_t *)kr_cache_ptr, "kr_cache");
    // output
    Tensor output_q(dType, q_out_shape, q_out_ptr, "output_q");
    Tensor output_q_rope(dType, q_rope_out_shape, q_rope_out_ptr, "output_q_rope");
    void *smooth_cq_ptr = readToDev<float>(dataPath + "/smooth_cq.bin", capacity_smooth_cq);
    Tensor smooth_cq(DT_FP32, smooth_cq_shape, (uint8_t *)smooth_cq_ptr, "smooth_cq");

    RoPETileShapeConfigNew ropeConfig{
        {b, 1, 64}, // (b,s,d)
        {b, 1, 1, 64}, // Q (b,s,n,d)
        {b, 1, 1, 64}, // K (b,s,1,d)
        {b, 1, 1, 32, 2} // (b,s,n,d//2,2)
    };

    MlaQuantInputs quantInputs;
    std::vector<void *> opArgsRun = {};

    if (isQuant) {
        void *w_qb_scale_ptr = readToDev<float>(dataPath + "/w_qb_scale.bin", capacity_w_qb_scale);
        Tensor w_qb_scale = Tensor(DT_FP32, w_qb_scale_shape, (uint8_t *)w_qb_scale_ptr, "w_qb_scale");
        quantInputs.dequantScaleWUqQr = w_qb_scale;
        if (hasSmooth) {
            quantInputs.smoothScalesCq = smooth_cq;
        }
        TileFwkBeginFunction("MlaProlog_T", {x, wDq, wUqQr, w_qb_scale,smooth_cq, wUk, wDkvKr, gamma_cq, gamma_ckv, sin, cos,
                                            kv_len, kv_cache, kr_cache, output_q, output_q_rope});
        {
            MlaProlog(x, wDq, wUqQr, wUk, wDkvKr, gamma_cq, gamma_ckv, sin, cos, kv_len, kv_cache, kr_cache,
                quantInputs, ropeConfig, output_q, output_q_rope, kv_cache, kr_cache, 1e-5f, 1e-5f,  "BNSD", splitReduceLastDim,  splitK);
        };
        TileFwkEndFunction();

        opArgsRun = {x_ptr, wDqPtr, wUqQrPtr, w_qb_scale_ptr, smooth_cq_ptr,wUkPtr, wDkvKrPtr, gamma_cq_ptr, gamma_ckv_ptr,
            sin_ptr, cos_ptr, kv_len_ptr, kv_cache_ptr, kr_cache_ptr, q_out_ptr, q_rope_out_ptr};

    } else {
        TileFwkBeginFunction("MlaProlog_T", {x, wDq, wUqQr, wUk, wDkvKr, gamma_cq, gamma_ckv, sin, cos, kv_len,
                                            kv_cache, kr_cache, output_q, output_q_rope});
        {
            MlaProlog(x, wDq, wUqQr, wUk, wDkvKr, gamma_cq, gamma_ckv, sin, cos, kv_len, kv_cache, kr_cache,
                quantInputs, ropeConfig, output_q, output_q_rope, kv_cache, kr_cache, 1e-5f, 1e-5f,  "BNSD", splitReduceLastDim,  splitK);
        };
        TileFwkEndFunction();

        opArgsRun = {x_ptr, wDqPtr, wUqQrPtr, wUkPtr, wDkvKrPtr, gamma_cq_ptr, gamma_ckv_ptr, sin_ptr, cos_ptr,
            kv_len_ptr, kv_cache_ptr, kr_cache_ptr, q_out_ptr, q_rope_out_ptr};
    }

    /* torch compile */
    void *handle = TileFwkCompile();

    /* torch prepare workspace memory */
    uint8_t *workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    TileFwkGetWorkspaceSize(handle, &workspaceSize);
    machine::GetRA()->AllocDevAddr(&workspaceAddr, workspaceSize);

    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(), opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("MlaProlog function aicpu stream sync failed");
    }

    std::vector<outDtype> q_golden(capacity_q_out);
    std::vector<outDtype> q_npu(capacity_q_out);
    std::vector<outDtype> q_rope_golden(capacity_q_rope_out);
    std::vector<outDtype> q_rope_npu(capacity_q_rope_out);
    std::vector<T> kv_golden(capacity_kv_out);
    std::vector<T> kv_npu(capacity_kv_out);
    std::vector<T> kr_golden(capacity_kr_out);
    std::vector<T> kr_npu(capacity_kr_out);

    readInput<outDtype>(dataPath + "/q_golden.bin", q_golden);
    readInput<outDtype>(dataPath + "/q_rope_golden.bin", q_rope_golden);
    readInput<T>(dataPath + "/kv_cache_golden.bin", kv_golden);
    readInput<T>(dataPath + "/kr_cache_golden.bin", kr_golden);
    machine::GetRA()->CopyFromTensor((uint8_t *)q_npu.data(), (uint8_t *)q_out_ptr, outputSize0);
    machine::GetRA()->CopyFromTensor((uint8_t *)q_rope_npu.data(), (uint8_t *)q_rope_out_ptr, outputSize1);
    machine::GetRA()->CopyFromTensor((uint8_t *)kv_npu.data(), (uint8_t *)kv_cache_ptr, outputSize2);
    machine::GetRA()->CopyFromTensor((uint8_t *)kr_npu.data(), (uint8_t *)kr_cache_ptr, outputSize3);

    std::cout << "\n====== resultCmp: output q start" << std::endl;
    int ret0 = resultCmp<outDtype>(q_golden, q_npu, 0.008f, 16);
    EXPECT_EQ(ret0, true);

    std::cout << "\n====== resultCmp: output q_rope start" << std::endl;
    int ret1 = resultCmp<outDtype>(q_rope_golden, q_rope_npu, 0.005f, 16);
    EXPECT_EQ(ret1, true);

    std::cout << "\n====== resultCmp: output kv_cache start" << std::endl;
    int ret2 = resultCmp<T>(kv_golden, kv_npu, 0.003f, 16);
    EXPECT_EQ(ret2, true);

    std::cout << "\n====== resultCmp: output kr_cache start" << std::endl;
    int ret3 = resultCmp<T>(kr_golden, kr_npu, 0.003f, 16);
    EXPECT_EQ(ret3, true);

    std::cout << "\n====== run time, start" << std::endl;
    uint64_t taskTime = npu::tile_fwk::DeviceRunner::Get().GetTasksTime();
    std::cout << "timeThreshold:" << timeThreshold << ", cost:" << taskTime << std::endl;
}

TEST_F(MlaPrologV2OnBoardCostTest, test_mla_bf16_low_quant_smooth) { // b_n_s_s2_h_q_lora_rank
    config::SetPassOption(L1_REUSE, 4);
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
    const bool splitReduceLastDim = false;
    const bool splitK = false;
    const bool nz = true;
    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaPrologV2<npu::tile_fwk::bfloat16,splitReduceLastDim,splitK,nz>(params, GetGoldenDir(), 3700, true, true);
}

TEST_F(MlaPrologV2OnBoardCostTest, test_mla_bf16_high_quant_smooth) {  // b_n_s_s2_h_q_lora_rank
    config::SetPassOption(NBUFFER_MERGE_MODE, 1);
    config::SetPassOption(L1_REUSE, 4);
    config::SetPassOption(CUBE_NBUFFER_MAP, std::map<int64_t, int64_t>{{3,4}});
    config::SetPassOption(COPYIN_THRESHOLD, 2*1024*1024);
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
    const bool splitReduceLastDim = false;
    const bool splitK = false;
    const bool nz = true;
    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, vHeadDim};

    TestMlaPrologV2<npu::tile_fwk::bfloat16, splitReduceLastDim, splitK, nz>(params, GetGoldenDir(), 6300, true, true);

}
