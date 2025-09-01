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
 * \file test_onboard_attention_post_cost.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "interface/inner/tilefwk/tilefwk_api.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "tilefwk/data_type.h"
#include "test_common.h"
#include "test_suite_stest_ops.h"
#include "runtime.h"
#include "device_runner.h"
#include "tilefwk_runtime_api.h"

using namespace npu::tile_fwk;

class OnBoardCostTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(OnBoardCostTest, test_attention_post_bf16_real_quant_batch4_onlymm5) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit("");

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];
    int dtypeNum = params[6];

    DataType dTypeInt8 = DT_INT8;
    DataType dTypeFp32 = DT_FP32;
    DataType dType = DT_FP32;
    if (dtypeNum == 0) {
        dType = DT_FP32;
    } else if (dtypeNum == 1) {
        dType = DT_FP16;
    } else if (dtypeNum == 2) {
        dType = DT_BF16;
    }
    int dtypeSize = BytesOf(dType);

    typedef npu::tile_fwk::bfloat16 T;
    typedef int8_t T_INT8;
    typedef float T_FLOAT;

    int inputSize = B * N * S * kvLoraRank;
    int wUvSize = N * kvLoraRank * vHeadDim;
    int wUvScaleWSize = N * 1 * vHeadDim;
    int wOSize = N * vHeadDim * H;
    int wOScaleWSize = H;
    int outputSize = B * S * H;
    int t1Size = B * S * N * kvLoraRank;

    uint64_t outputByteSize = outputSize * dtypeSize;
    uint8_t* outPtr = allocDevAddr(outputByteSize);

    std::vector<int64_t> inputShape = {B,N,S,kvLoraRank};
    std::vector<int64_t> wUvShape = {N,kvLoraRank,vHeadDim};
    std::vector<int64_t> wUvScaleWShape = {N,1,vHeadDim};
    std::vector<int64_t> wOShape = {N*vHeadDim,H};
    std::vector<int64_t> wOScaleWShape = {H};
    std::vector<int64_t> outputShapeT = {B, S, H};
    std::vector<int64_t> t1Shape = {B, S, N, kvLoraRank};
    void *inputPtr = readToDev<T>(GetGoldenDir() + "/input.bin", inputSize);
    void *wUvPtr = readToDev<T>(GetGoldenDir() + "/w_uv.bin", wUvSize);
    void *wUvScaleWPtr = readToDev<T_FLOAT>(GetGoldenDir() + "/w_uv_scale_w.bin", wUvScaleWSize);
    void *wOPtr = readToDev<T_INT8>(GetGoldenDir() + "/w_o.bin", wOSize);
    void *wOScaleWPtr = readToDev<T_FLOAT>(GetGoldenDir() + "/w_o_scale_w.bin", wOScaleWSize);

    void *t1Ptr = readToDev<T>(GetGoldenDir() + "/t1.bin", t1Size);

    Tensor inputI(dType, inputShape, (uint8_t *)inputPtr, "A");
    Tensor wUvI(dType, wUvShape, (uint8_t *)wUvPtr, "B");
    Tensor wUvScaleWI(dTypeFp32, wUvScaleWShape, (uint8_t *)wUvScaleWPtr, "B1");
    Tensor wOI(dTypeInt8, wOShape, (uint8_t *)wOPtr, "C", TileOpFormat::TILEOP_NZ);
    Tensor wOScaleWI(dTypeFp32, wOScaleWShape, (uint8_t *)wOScaleWPtr, "C1");
    Tensor outputT(dType, outputShapeT, outPtr, "D1");
    Tensor t1I(dType, t1Shape, (uint8_t *)t1Ptr, "E");

    TileFwkBeginFunction("ATTENTION_POST_T", {inputI, wUvI, wUvScaleWI, wOI, wOScaleWI, outputT});
    {
        Program::GetInstance().GetTileShape().SetVecTileShapes({4, 4, 1, kvLoraRank});
        Tensor attenRes0 = Transpose(inputI, {0, 1});
        // Program::GetInstance().GetTileShape().SetVecTileShapes({1, 1, 1, kvLoraRank});
        Tensor t2Res = Reshape(attenRes0, {N, B * S, kvLoraRank});

        Program::GetInstance().GetTileShape().SetCubeTileShapes({4, 4}, {std::min(128, kvLoraRank), std::min(128, kvLoraRank)}, {std::min(128, vHeadDim), std::min(128, vHeadDim)});  // M 16对齐
        // 所有子图申请的UB空间总和可能大于192K，所以tileShape不能太大（1、ooo pass申请UB空间的方式不合理，在重构；2、RMS里面有repeattimes写死的64，可能会导致tileShape太大）
        // [n,bs,kvLoraRank] * [n, kvLoraRank, vHeadDim] = [n,bs,vHeadDim]
        Tensor bmm4Res = Matrix::BatchMatmul(dType, t2Res, wUvI);

        Program::GetInstance().GetTileShape().SetVecTileShapes(4, 4, vHeadDim); // 必须切，但是尾轴不能切
        Tensor t3Res = Transpose(bmm4Res, {0, 1}); // [bs,n,vHeadDim]

        // Program::GetInstance().GetTileShape().SetVecTileShapes({4, 32, vHeadDim});
        Tensor r2Res = Reshape(t3Res, {B * S, N * vHeadDim});

        Program::GetInstance().GetTileShape().SetVecTileShapes(2, std::min(512, N * vHeadDim));
        auto quantA = Quant(r2Res);
        auto quantizedA = std::get<0>(quantA);
        auto dequantScaleA = std::get<1>(quantA);
        Tensor res;
        Program::GetInstance().GetTileShape().SetCubeTileShapes({4, 4},
            {std::min(128, N * vHeadDim), std::min(128, N * vHeadDim)},
            {std::min(512, H), std::min(512, H)});
        if (r2Res->shape.size() == 2) {
            res = npu::tile_fwk::Matrix::Matmul<false, false>(DataType::DT_INT32, quantizedA, wOI);
        } else if (r2Res->shape.size() == 3) {
            res = npu::tile_fwk::Matrix::BatchMatmul(DataType::DT_INT32, quantizedA, wOI);
        } else {
            assert(r2Res->shape.size() <= 3);
        }
        Program::GetInstance().GetTileShape().SetVecTileShapes(4, std::min(512, N*vHeadDim));
        res = Cast(res, DataType::DT_FP32);
        res = Mul(res, dequantScaleA);
        Tensor weightOScaleW2Dim = Reshape(wOScaleWI, {1, H});
        res = Mul(res, weightOScaleW2Dim);
        Tensor bmm5Res = Cast(res, DataType::DT_BF16, CAST_RINT);

        // Program::GetInstance().GetTileShape().SetVecTileShapes({4, std::min(8192, H)});
        outputT = Reshape(bmm5Res, {B, S, H});
    }
    TileFwkEndFunction();

    /* torch compile */
    void* handle = TileFwkCompile();

    /* torch prepare workspace memory */
    uint8_t* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    TileFwkGetWorkspaceSize(handle, &workspaceSize);
    machine::GetRA()->AllocDevAddr(&workspaceAddr, workspaceSize);

    std::vector<void*> opArgsRun = {inputPtr, wUvPtr, wUvScaleWPtr, wOPtr, wOScaleWPtr, outPtr};

    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::vector<T> golden(outputSize);
    std::vector<T> res(outputSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)outPtr, outputByteSize);
    readInput<T>(GetGoldenDir() + "/attn_output.bin", golden);
    bool ret = resultCmp<T>(golden, res, 0.001f);

    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardCostTest, test_attention_post_bf16_real_quant_n128_onlymm5) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit("");

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];
    int dtypeNum = params[6];

    DataType dTypeInt8 = DT_INT8;
    DataType dTypeFp32 = DT_FP32;
    DataType dType = DT_FP32;
    if (dtypeNum == 0) {
        dType = DT_FP32;
    } else if (dtypeNum == 1) {
        dType = DT_FP16;
    } else if (dtypeNum == 2) {
        dType = DT_BF16;
    }
    int dtypeSize = BytesOf(dType);

    typedef npu::tile_fwk::bfloat16 T;
    typedef int8_t T_INT8;
    typedef float T_FLOAT;

    int inputSize = B * N * S * kvLoraRank;
    int wUvSize = N * kvLoraRank * vHeadDim;
    int wUvScaleWSize = N * 1 * vHeadDim;
    int wOSize = N * vHeadDim * H;
    int wOScaleWSize = H;
    int outputSize = B * S * H;
    int t1Size = B * S * N * kvLoraRank;

    uint64_t outputByteSize = outputSize * dtypeSize;
    uint8_t* outPtr = allocDevAddr(outputByteSize);

    std::vector<int64_t> inputShape = {B, N, S, kvLoraRank};
    std::vector<int64_t> wUvShape = {N, kvLoraRank, vHeadDim};
    std::vector<int64_t> wUvScaleWShape = {N, 1, vHeadDim};
    std::vector<int64_t> wOShape = {N * vHeadDim, H};
    std::vector<int64_t> wOScaleWShape = {H};
    std::vector<int64_t> outputShapeT = {B, S, H};
    std::vector<int64_t> t1Shape = {B, S, N, kvLoraRank};
    void *inputPtr = readToDev<T>(GetGoldenDir() + "/input.bin", inputSize);
    void *wUvPtr = readToDev<T>(GetGoldenDir() + "/w_uv.bin", wUvSize);
    void *wUvScaleWPtr = readToDev<T_FLOAT>(GetGoldenDir() + "/w_uv_scale_w.bin", wUvScaleWSize);
    void *wOPtr = readToDev<T_INT8>(GetGoldenDir() + "/w_o.bin", wOSize);
    void *wOScaleWPtr = readToDev<T_FLOAT>(GetGoldenDir() + "/w_o_scale_w.bin", wOScaleWSize);

    void *t1Ptr = readToDev<T>(GetGoldenDir() + "/t1.bin", t1Size);

    Tensor inputI(dType, inputShape, (uint8_t *)inputPtr, "A");
    Tensor wUvI(dType, wUvShape, (uint8_t *)wUvPtr, "B");
    Tensor wUvScaleWI(dTypeFp32, wUvScaleWShape, (uint8_t *)wUvScaleWPtr, "B1");
    Tensor wOI(dTypeInt8, wOShape, (uint8_t *)wOPtr, "C", TileOpFormat::TILEOP_NZ);
    Tensor wOScaleWI(dTypeFp32, wOScaleWShape, (uint8_t *)wOScaleWPtr, "C1");
    Tensor outputT(dType, outputShapeT, outPtr, "D1");
    Tensor t1I(dType, t1Shape, (uint8_t *)t1Ptr, "E");

    TileFwkBeginFunction("ATTENTION_POST_T", {inputI, wUvI, wUvScaleWI, wOI, wOScaleWI, outputT});
    {
        Program::GetInstance().GetTileShape().SetVecTileShapes({1, 32, 1, kvLoraRank});
        Tensor attenRes0 = Transpose(inputI, {0, 1});
        // Program::GetInstance().GetTileShape().SetVecTileShapes({1, 1, 1, kvLoraRank});
        Tensor t2Res = Reshape(attenRes0, {N, B * S, kvLoraRank});

        // Program::GetInstance().GetConfig().Set<int>(L1_REUSE, 6);
        Program::GetInstance().GetConfig().Set<std::map<int, int>>(CUBE_NBUFFER_MAP, {{0, 2}});
        Program::GetInstance().GetTileShape().SetCubeTileShapes({std::min(32, B*S), std::min(32, B*S)},
            {std::min(256, kvLoraRank), std::min(256, kvLoraRank)},
            {std::min(128, vHeadDim), std::min(128, vHeadDim)});  // M 16对齐
        // 所有子图申请的UB空间总和可能大于192K，所以tileShape不能太大（1、ooo pass申请UB空间的方式不合理，在重构；2、RMS里面有repeattimes写死的64，可能会导致tileShape太大）
        // 原[n,bs,kvLoraRank] * [n, kvLoraRank, vHeadDim] = [n,bs,vHeadDim]
        Tensor bmm4Res = Matrix::BatchMatmul(dType, t2Res, wUvI);

        Program::GetInstance().GetTileShape().SetVecTileShapes(2, 32, vHeadDim); // 必须切，但是尾轴不能切
        Tensor t3Res = Transpose(bmm4Res, {0, 1}); // [bs,n,vHeadDim]

        Tensor r2Res = Reshape(t3Res, {B * S, N * vHeadDim});

        Program::GetInstance().GetTileShape().SetVecTileShapes(16, std::min(32, N * vHeadDim));
        auto quantA = Quant(r2Res);
        auto quantizedA = std::get<0>(quantA);
        auto dequantScaleA = std::get<1>(quantA);
        Tensor res;
        Program::GetInstance().GetTileShape().SetCubeTileShapes({std::min(32, B*S), std::min(32, B*S)}, {std::min(128, N*vHeadDim), std::min(128, N*vHeadDim)}, {std::min(512, H), std::min(512, H)});
        if (r2Res->shape.size() == 2) {
            res = npu::tile_fwk::Matrix::Matmul<false, false>(DataType::DT_INT32, quantizedA, wOI);
        } else if (r2Res->shape.size() == 3) {
            res = npu::tile_fwk::Matrix::BatchMatmul(DataType::DT_INT32, quantizedA, wOI);
        } else {
            assert(r2Res->shape.size() <= 3);
        }
        Program::GetInstance().GetTileShape().SetVecTileShapes(4, std::min(32, N*vHeadDim));
        res = Cast(res, DataType::DT_FP32);
        res = Mul(res, dequantScaleA);
        Tensor weightOScaleW2Dim = Reshape(wOScaleWI, {1, H});
        res = Mul(res, weightOScaleW2Dim);
        Tensor bmm5Res = Cast(res, DataType::DT_BF16, CAST_RINT);

        // Program::GetInstance().GetTileShape().SetVecTileShapes({32, std::min(32, H)});
        outputT = Reshape(bmm5Res, {B, S, H});
    }
    TileFwkEndFunction();

    /* torch compile */
    void* handle = TileFwkCompile();

    /* torch prepare workspace memory */
    uint8_t* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    TileFwkGetWorkspaceSize(handle, &workspaceSize);
    machine::GetRA()->AllocDevAddr(&workspaceAddr, workspaceSize);

    std::vector<void*> opArgsRun = {inputPtr, wUvPtr, wUvScaleWPtr, wOPtr, wOScaleWPtr, outPtr};

    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::vector<T> golden(outputSize);
    std::vector<T> res(outputSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)outPtr, outputByteSize);
    readInput<T>(GetGoldenDir() + "/attn_output.bin", golden);
    bool ret = resultCmp<T>(golden, res, 0.03f, 700, 1000, false);

    EXPECT_EQ(ret, true);

}


std::tuple<Tensor, Tensor> QuantTmp(
    const Tensor &input, bool isSymmetry, bool hasSmoothFactor, const Tensor smoothFactor) {
    auto inputFp32 = Cast(input, DataType::DT_FP32, CAST_NONE);
    if (hasSmoothFactor) {
        inputFp32 = Mul(inputFp32, smoothFactor);
    }
    // perToken
    if (isSymmetry) {
        auto absRes = Abs(inputFp32); // (32, 16384)
        auto maxValue = RowMaxSingle(absRes); // (32, 1)
        auto scaleQuant = ScalarDivS(maxValue, Element(DataType::DT_FP32, 127.0), true);  // (32, 1)
        auto outFp32 = Mul(inputFp32, scaleQuant); // (32, 16384) * (32, 1) = (32, 16384)
        auto outInt32 = Cast(outFp32, DataType::DT_INT32, CAST_RINT); // (32, 16384)
        auto outHalf = Cast(outInt32, DataType::DT_FP16, CAST_ROUND); // (32, 16384)
        auto outInt8 = Cast(outHalf, DataType::DT_INT8, CAST_TRUNC); // (32, 16384)
        auto scaleDeQuant = ScalarDivS(scaleQuant, Element(DataType::DT_FP32, 1.0), true);   // (32, 1)
        return std::tie(outInt8, scaleDeQuant);  // (32, 16384)   (32, 1)
    } else {
        // 优先级低
        auto maxValue = RowMaxSingle(inputFp32);
        auto minValue = RowMinSingle(inputFp32);
        auto scaleDeQuant = ScalarMaxS(ScalarDivS(ScalarSub(maxValue, minValue),
            Element(DataType::DT_FP32, 255.0)), Element(DataType::DT_FP32, 1e-12f));
        auto offset = ScalarSubS(ScalarDiv(maxValue, scaleDeQuant), Element(DataType::DT_FP32, 127.0),
            true);
        auto scaleQuant = ScalarDivS(scaleDeQuant, Element(DataType::DT_FP32, 1.0), true);
        auto outFp32 = Mul(inputFp32, scaleQuant);
        auto outInt32 = Cast(outFp32, DataType::DT_INT32, CAST_RINT);
        auto outHalf = Cast(outInt32, DataType::DT_FP16, CAST_ROUND);
        auto outInt8 = Cast(outHalf, DataType::DT_INT8, CAST_TRUNC);
        return std::tie(outInt8, scaleDeQuant);
    }
}

TEST_F(OnBoardCostTest, test_attention_post_bf16_real_quant_n128_onlymm5He) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit("");

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];
    int dtypeNum = params[6];

    DataType dTypeInt8 = DT_INT8;
    DataType dTypeFp32 = DT_FP32;
    DataType dType = DT_FP32;
    if (dtypeNum == 0) {
        dType = DT_FP32;
    } else if (dtypeNum == 1) {
        dType = DT_FP16;
    } else if (dtypeNum == 2) {
        dType = DT_BF16;
    }
    int dtypeSize = BytesOf(dType);

    typedef npu::tile_fwk::bfloat16 T;
    typedef int8_t T_INT8;
    typedef float T_FLOAT;

    int inputSize = B * N * S * kvLoraRank;
    int wUvSize = N * kvLoraRank * vHeadDim;
    int wUvScaleWSize = N * 1 * vHeadDim;
    int wOSize = N * vHeadDim * H;
    int wOScaleWSize = H;
    int outputSize = B * S * H;
    int t1Size = B * S * N * kvLoraRank;

    uint64_t outputByteSize = outputSize * dtypeSize;
    uint8_t* outPtr = allocDevAddr(outputByteSize);

    std::vector<int64_t> inputShape = {B, N, S, kvLoraRank};
    std::vector<int64_t> wUvShape = {N, kvLoraRank, vHeadDim};
    std::vector<int64_t> wUvScaleWShape = {N, 1, vHeadDim};
    std::vector<int64_t> wOShape = {N * vHeadDim, H};
    std::vector<int64_t> wOScaleWShape = {H};
    std::vector<int64_t> outputShapeT = {B, S, H};
    std::vector<int64_t> t1Shape = {B, S, N, kvLoraRank};
    void *inputPtr = readToDev<T>(GetGoldenDir() + "/input.bin", inputSize);
    void *wUvPtr = readToDev<T>(GetGoldenDir() + "/w_uv.bin", wUvSize);
    void *wUvScaleWPtr = readToDev<T_FLOAT>(GetGoldenDir() + "/w_uv_scale_w.bin", wUvScaleWSize);
    void *wOPtr = readToDev<T_INT8>(GetGoldenDir() + "/w_o.bin", wOSize);
    void *wOScaleWPtr = readToDev<T_FLOAT>(GetGoldenDir() + "/w_o_scale_w.bin", wOScaleWSize);

    void *t1Ptr = readToDev<T>(GetGoldenDir() + "/t1.bin", t1Size);

    Tensor inputI(dType, inputShape, (uint8_t *)inputPtr, "A");
    Tensor wUvI(dType, wUvShape, (uint8_t *)wUvPtr, "B");
    Tensor wUvScaleWI(dTypeFp32, wUvScaleWShape, (uint8_t *)wUvScaleWPtr, "B1");
    Tensor wOI(dTypeInt8, wOShape, (uint8_t *)wOPtr, "C", TileOpFormat::TILEOP_NZ);
    Tensor wOScaleWI(dTypeFp32, wOScaleWShape, (uint8_t *)wOScaleWPtr, "C1");
    Tensor outputT(dType, outputShapeT, outPtr, "D1");
    Tensor t1I(dType, t1Shape, (uint8_t *)t1Ptr, "E");

    TileFwkBeginFunction("ATTENTION_POST_T", {inputI, wUvI, wUvScaleWI, wOI, wOScaleWI, outputT});
    {
        Program::GetInstance().GetTileShape().SetVecTileShapes({1, 16, 1, kvLoraRank});
        Tensor attenRes0 = Transpose(inputI, {0, 1}); // (32, 128, 1, 512)
        // Program::GetInstance().GetTileShape().SetVecTileShapes({16, 1, 1, kvLoraRank});
        Tensor t2Res = Reshape(attenRes0, {N, B * S, kvLoraRank}); // (128, 32, 1, 512)

        // Program::GetInstance().GetConfig().Set<int>(L1_REUSE, 6);
        Program::GetInstance().GetConfig().Set<std::map<int, int>>(CUBE_NBUFFER_MAP, {{0, 2}});
        Program::GetInstance().GetTileShape().SetCubeTileShapes({std::min(32, B*S), std::min(32, B*S)},
            {std::min(256, kvLoraRank), std::min(256, kvLoraRank)},
            {std::min(128, vHeadDim), std::min(128, vHeadDim)});  // 改变M大小，生成256个块
        // 所有子图申请的UB空间总和可能大于192K，所以tileShape不能太大（1、ooo pass申请UB空间的方式不合理，在重构；2、RMS里面有repeattimes写死的64，可能会导致tileShape太大）
        // 原[n,bs,kvLoraRank] * [n, kvLoraRank, vHeadDim] = [n,bs,vHeadDim]
        Tensor bmm4Res = Matrix::BatchMatmul(dType, t2Res, wUvI); // (128, 32, 512) @ (128, 512, 128)

        Program::GetInstance().GetTileShape().SetVecTileShapes(1, 16, vHeadDim); // 必须切，但是尾轴不能切
        Tensor t3Res = Transpose(bmm4Res, {0, 1}); // [bs,n,vHeadDim]   // (128, 32, 128)

        Tensor r2Res = Reshape(t3Res, {B * S, N * vHeadDim});  // (32, 128, 128)

        Program::GetInstance().GetTileShape().SetVecTileShapes(16, std::min(128, N * vHeadDim));

        // auto quantA = Quant(r2Res);   // (32, 16384)
        auto quantA = QuantTmp(r2Res, true, false, Tensor());   // (32, 16384)
        auto quantizedA = std::get<0>(quantA);
        auto dequantScaleA = std::get<1>(quantA);

        Tensor res;
        Program::GetInstance().GetTileShape().SetCubeTileShapes({std::min(32, B*S), std::min(32, B*S)}, {std::min(128, N*vHeadDim), std::min(128, N*vHeadDim)}, {std::min(512, H), std::min(512, H)});
        if (r2Res->shape.size() == 2) {
            res = npu::tile_fwk::Matrix::Matmul(DataType::DT_INT32, quantizedA, wOI); // (32, 16384) @ (16384, 7168)
        } else if (r2Res->shape.size() == 3) {
            res = npu::tile_fwk::Matrix::BatchMatmul(DataType::DT_INT32, quantizedA, wOI);
        } else {
            assert(r2Res->shape.size() <= 3);
        }
        Program::GetInstance().GetTileShape().SetVecTileShapes(2, std::min(32, N*vHeadDim));
        res = Cast(res, DataType::DT_FP32); // (32, 7168)
        res = Mul(res, dequantScaleA);
        Tensor weightOScaleW2Dim = Reshape(wOScaleWI, {1, H});
        res = Mul(res, weightOScaleW2Dim);
        Tensor bmm5Res = Cast(res, DataType::DT_BF16, CAST_RINT);
        outputT = Reshape(bmm5Res, {B, S, H});
    }
    TileFwkEndFunction();

    /* torch compile */
    void* handle = TileFwkCompile();

    /* torch prepare workspace memory */
    uint8_t* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    TileFwkGetWorkspaceSize(handle, &workspaceSize);
    machine::GetRA()->AllocDevAddr(&workspaceAddr, workspaceSize);

    std::vector<void*> opArgsRun = {inputPtr, wUvPtr, wUvScaleWPtr, wOPtr, wOScaleWPtr, outPtr};

    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::vector<T> golden(outputSize);
    std::vector<T> res(outputSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)outPtr, outputByteSize);
    readInput<T>(GetGoldenDir() + "/attn_output.bin", golden);
    bool ret = resultCmp<T>(golden, res, 0.03f);

    EXPECT_EQ(ret, true);

}


TEST_F(OnBoardCostTest, test_attention_post_bf16_real_quant_batch4_onlymm5K) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit("");

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];
    int dtypeNum = params[6];

    DataType dTypeInt8 = DT_INT8;
    DataType dTypeFp32 = DT_FP32;
    DataType dType = DT_FP32;
    if (dtypeNum == 0) {
        dType = DT_FP32;
    } else if (dtypeNum == 1) {
        dType = DT_FP16;
    } else if (dtypeNum == 2) {
        dType = DT_BF16;
    }
    int dtypeSize = BytesOf(dType);

    typedef npu::tile_fwk::bfloat16 T;
    typedef int8_t T_INT8;
    typedef float T_FLOAT;

    int inputSize = B*N*S*kvLoraRank;
    int wUvSize = N*kvLoraRank*vHeadDim;
    int wUvScaleWSize = N*1*vHeadDim;
    int wOSize = N*vHeadDim*H;
    int wOScaleWSize = 1*H;
    int outputSize = B*S*H;
    int t1Size = B*S*N*kvLoraRank;

    uint64_t outputByteSize = outputSize * dtypeSize;
    uint8_t* outPtr = allocDevAddr(outputByteSize);

    std::vector<int64_t> inputShape = {B,N,S,kvLoraRank};
    std::vector<int64_t> wUvShape = {N,kvLoraRank,vHeadDim};
    std::vector<int64_t> wUvScaleWShape = {N,1,vHeadDim};
    std::vector<int64_t> wOShape = {N*vHeadDim,H};
    std::vector<int64_t> wOScaleWShape = {H};
    std::vector<int64_t> outputShapeT = {B, S, H};
    std::vector<int64_t> t1Shape = {B, S, N, kvLoraRank};
    void *inputPtr = readToDev<T>(GetGoldenDir() + "/input.bin", inputSize);
    void *wUvPtr = readToDev<T>(GetGoldenDir() + "/w_uv.bin", wUvSize);
    void *wUvScaleWptr = readToDev<T_FLOAT>(GetGoldenDir() + "/w_uv_scale_w.bin", wUvScaleWSize);
    void *wOptr = readToDev<T_INT8>(GetGoldenDir() + "/w_o.bin", wOSize);
    void *wOscaleWptr = readToDev<T_FLOAT>(GetGoldenDir() + "/w_o_scale_w.bin", wOScaleWSize);

    void *t1Ptr = readToDev<T>(GetGoldenDir() + "/t1.bin", t1Size);

    Tensor inputI(dType, inputShape, (uint8_t *)inputPtr, "A");
    Tensor wUvI(dType, wUvShape, (uint8_t *)wUvPtr, "B");
    Tensor wUvScaleWi(dTypeFp32, wUvScaleWShape, (uint8_t *)wUvScaleWptr, "B1");
    Tensor wOi(dTypeInt8, wOShape, (uint8_t *)wOptr, "C", TileOpFormat::TILEOP_NZ);
    Tensor wOscaleWi(dTypeFp32, wOScaleWShape, (uint8_t *)wOscaleWptr, "C1");
    Tensor outputT(dType, outputShapeT, outPtr, "D1");
    Tensor t1I(dType, t1Shape, (uint8_t *)t1Ptr, "E");

    TileFwkBeginFunction("ATTENTION_POST_T", {inputI, wUvI, wUvScaleWi, wOi, wOscaleWi, outputT});
    {
        Program::GetInstance().GetTileShape().SetVecTileShapes({B, 1, 1, kvLoraRank}); // 32 ge
        Tensor attenRes0 = Transpose(inputI, {0, 1}); // (4,32,1,512) -> (32,4,1,512)
        Tensor t2Res = Reshape(attenRes0, {N, B * S, kvLoraRank}); // (32,4,1,512) -> (32,4,512)

        Program::GetInstance().GetConfig().Set<std::map<int, int>>(CUBE_NBUFFER_MAP, {{0, 2}});
        Program::GetInstance().GetTileShape().SetCubeTileShapes({4, 4},
            {std::min(256, kvLoraRank), std::min(256, kvLoraRank)},
            {std::min(128, vHeadDim), std::min(128, vHeadDim)});  // 32/2 ge
        // [n,bs,kvLoraRank] * [n, kvLoraRank, vHeadDim] = [n,bs,vHeadDim]
        Tensor bmm4Res = Matrix::BatchMatmul(dType, t2Res, wUvI); // (32,4,512) @ (32,512,128) -> (32,4,128)

        Program::GetInstance().GetTileShape().SetVecTileShapes(2, 4, vHeadDim); // 必须切，但是尾轴不能切
        Tensor t3Res = Transpose(bmm4Res, {0, 1}); // [bs,n,vHeadDim]  (32,4,128) _> (4,32,128)
        Tensor r2Res = Reshape(t3Res, {B * S, N*vHeadDim}); // (4,32,128) -> (4,4096)

        Program::GetInstance().GetTileShape().SetVecTileShapes(1, 4096);
        auto quantA = Quant(r2Res); // (4,4096)
        auto quantizedA = std::get<0>(quantA);
        auto dequantScaleA = std::get<1>(quantA);

        // (B*S, N*vHeadDim) @ (N*vHeadDim, H) = (B*S, H)
        // int8 @ int8 = int32
        Tensor tmpC(DT_INT32, {B*S, H}, "tmp_c"); // (4,7168)
        Program::GetInstance().GetTileShape().SetVecTileShapes(4, std::min(1024, H));  // 7个
        tmpC = MulS(tmpC, Element(DataType::DT_FP32, 0.0f));
        std::vector<Tensor> matmulResult;
        auto kSplit = 2;
        auto kSplitSize = N*vHeadDim / kSplit; // 4096 / 2
        // (4, 4096) @ (4096, 7168) = (4, 7168)   M:4,K:4096,N:7168
        Program::GetInstance().GetTileShape().SetCubeTileShapes({4, 4},
            {std::min(128, N*vHeadDim), std::min(128, N*vHeadDim)},
            {std::min(512, H), std::min(512, H)}); // 14ge
        for (int ki = 0; ki < kSplit; ki++) {
            auto inputMk = View(quantizedA, {B*S, kSplitSize}, {0, ki * kSplitSize});
            auto inputKn = View(wOi, {kSplitSize, H}, {ki * kSplitSize, 0});
            auto tmp = npu::tile_fwk::Matrix::Matmul(DT_INT32, inputMk, inputKn, tmpC);    // B * 256
            matmulResult.emplace_back(tmp);
        }
        Program::GetInstance().GetTileShape().SetVecTileShapes(4, 512);
        Tensor res = npu::tile_fwk::Reduce(matmulResult, ReduceMode::ATOMIC_ADD);

        Program::GetInstance().GetTileShape().SetVecTileShapes(4, std::min(512, N*vHeadDim));
        res = Cast(res, DataType::DT_FP32); // (4,7168)
        res = Mul(res, dequantScaleA);
        Tensor weightOScaleW2Dim = Reshape(wOscaleWi, {1, H});
        res = Mul(res, weightOScaleW2Dim);
        Tensor bmm5Res = Cast(res, DataType::DT_BF16, CAST_RINT);

        outputT = Reshape(bmm5Res, {B, S, H});
    }
    TileFwkEndFunction();

    /* torch compile */
    void* handle = TileFwkCompile();

    /* torch prepare workspace memory */
    uint8_t* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    TileFwkGetWorkspaceSize(handle, &workspaceSize);
    machine::GetRA()->AllocDevAddr(&workspaceAddr, workspaceSize);

    std::vector<void*> opArgsRun = {inputPtr, wUvPtr, wUvScaleWptr, wOptr, wOscaleWptr, outPtr};

    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::vector<T> golden(outputSize);
    std::vector<T> res(outputSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)outPtr, outputByteSize);
    readInput<T>(GetGoldenDir() + "/attn_output.bin", golden);
    bool ret = resultCmp<T>(golden, res, 0.001f);

    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardCostTest, test_attention_post_bf16_real_quant_n128_onlymm5K) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit("");

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];
    int dtypeNum = params[6];

    DataType dTypeInt8 = DT_INT8;
    DataType dTypeFp32 = DT_FP32;
    DataType dType = DT_FP32;
    if (dtypeNum == 0) {
        dType = DT_FP32;
    } else if (dtypeNum == 1) {
        dType = DT_FP16;
    } else if (dtypeNum == 2) {
        dType = DT_BF16;
    }
    int dtypeSize = BytesOf(dType);

    typedef npu::tile_fwk::bfloat16 T;
    typedef int8_t T_INT8;
    typedef float T_FLOAT;

    int inputSize = B*N*S*kvLoraRank;
    int wUvSize = N*kvLoraRank*vHeadDim;
    int wUvScaleWSize = N*1*vHeadDim;
    int wOSize = N*vHeadDim*H;
    int wOScaleWSize = 1*H;
    int outputSize = B*S*H;
    int t1Size = B*S*N*kvLoraRank;

    uint64_t outputByteSize = outputSize * dtypeSize;
    uint8_t* outPtr = allocDevAddr(outputByteSize);

    uint64_t mm5Int32ByteSize = outputSize * BytesOf(DT_INT32);
    uint8_t* mm5int32_ptr = allocDevAddr(mm5Int32ByteSize);

    uint64_t mm5fp32ByteSize = outputSize * BytesOf(DT_FP32);
    uint8_t* mm5fp32_ptr = allocDevAddr(mm5fp32ByteSize);

    std::vector<int64_t> inputShape = {B,N,S,kvLoraRank};
    std::vector<int64_t> wUvShape = {N,kvLoraRank,vHeadDim};
    std::vector<int64_t> wUvScaleWShape = {N,1,vHeadDim};
    std::vector<int64_t> wOShape = {N*vHeadDim,H};
    std::vector<int64_t> wOScaleWShape = {H};
    std::vector<int64_t> outputShapeT = {B, S, H};
    std::vector<int64_t> mm5Int32ShapeT = {B * S, H};
    std::vector<int64_t> t1Shape = {B, S, N, kvLoraRank};
    void *inputPtr = readToDev<T>(GetGoldenDir() + "/input.bin", inputSize);
    void *wUvPtr = readToDev<T>(GetGoldenDir() + "/w_uv.bin", wUvSize);
    void *wUvScaleWptr = readToDev<T_FLOAT>(GetGoldenDir() + "/w_uv_scale_w.bin", wUvScaleWSize);
    void *wOptr = readToDev<T_INT8>(GetGoldenDir() + "/w_o.bin", wOSize);
    void *wOscaleWptr = readToDev<T_FLOAT>(GetGoldenDir() + "/w_o_scale_w.bin", wOScaleWSize);

    void *t1Ptr = readToDev<T>(GetGoldenDir() + "/t1.bin", t1Size);

    Tensor inputI(dType, inputShape, (uint8_t *)inputPtr, "A");
    Tensor wUvI(dType, wUvShape, (uint8_t *)wUvPtr, "B");
    Tensor wUvScaleWi(dTypeFp32, wUvScaleWShape, (uint8_t *)wUvScaleWptr, "B1");
    Tensor wOi(dTypeInt8, wOShape, (uint8_t *)wOptr, "C", TileOpFormat::TILEOP_NZ);
    Tensor wOscaleWi(dTypeFp32, wOScaleWShape, (uint8_t *)wOscaleWptr, "C1");
    Tensor outputT(dType, outputShapeT, outPtr, "D1");
    Tensor mm5Int32(DT_INT32, mm5Int32ShapeT, mm5int32_ptr, "mm5Int32");
    Tensor mm5fp32(DT_FP32, mm5Int32ShapeT, mm5fp32_ptr, "mm5fp32");
    Tensor t1I(dType, t1Shape, (uint8_t *)t1Ptr, "E");

    Program::GetInstance().GetConfig().Set<int>(SG_CYCLE_UPPER_BOUND, 300000);  // 300000(167us)
    TileFwkBeginFunction("ATTENTION_POST_T", {inputI, wUvI, wUvScaleWi, wOi, wOscaleWi, outputT, mm5Int32, mm5fp32});
    {
        Program::GetInstance().GetTileShape().SetVecTileShapes({B, 2, 1, kvLoraRank}); // 128个
        Tensor attenRes0 = Transpose(inputI, {0, 1}); // (32, 128, 1, 512)
        Tensor t2Res = Reshape(attenRes0, {N, B * S, kvLoraRank}); // (128, 32, 1, 512)

        // 原AscendProgram::GetInstance().GetConfig().Set<int>(L1_REUSE, 6);
        Program::GetInstance().GetConfig().Set<std::map<int, int>>(CUBE_NBUFFER_MAP, {{0, 4}});
        Program::GetInstance().GetTileShape().SetCubeTileShapes({std::min(32, B*S), std::min(32, B*S)},
            {std::min(256, kvLoraRank), std::min(256, kvLoraRank)},
            {std::min(128, vHeadDim), std::min(128, vHeadDim)});  // 128/4 个
        // 原[n,bs,kvLoraRank] * [n, kvLoraRank, vHeadDim] = [n,bs,vHeadDim]
        Tensor bmm4Res = Matrix::BatchMatmul(dType, t2Res, wUvI); // (128, 32, 512) @ (128, 512, 128) = (128, 32, 128)

        Program::GetInstance().GetTileShape().SetVecTileShapes(1, 32, vHeadDim); // 32个
        Tensor t3Res = Transpose(bmm4Res, {0, 1}); // [bs,n,vHeadDim]    // (128, 32, 128)
        Tensor r2Res = Reshape(t3Res, {B * S, N*vHeadDim});  // (32, 128, 128)

        Program::GetInstance().GetTileShape().SetVecTileShapes(1, 16384); // 32个
        auto quantA = QuantTmp(r2Res, true, false, Tensor());   // (32, 16384)
        // auto quantA = Quant(r2Res);   // (32, 16384)
        auto quantizedA = std::get<0>(quantA);
        auto dequantScaleA = std::get<1>(quantA);

        // (B*S, N*vHeadDim) @ (N*vHeadDim, H) = (B*S, H)
        // int8 @ int8 = int32
        Tensor tmpC(DT_INT32, {B*S, H}, "tmp_c");
        Program::GetInstance().GetTileShape().SetVecTileShapes(32, std::min(1024, H));  // 7个
        tmpC = MulS(tmpC, Element(DataType::DT_FP32, 0.0f));
        std::vector<Tensor> matmulResult;
        auto kSplit = 8;
        auto kSplitSize = N*vHeadDim / kSplit;
        Program::GetInstance().GetTileShape().SetCubeTileShapes({std::min(32, B*S), std::min(32, B*S)},
            {std::min(128, N*vHeadDim), std::min(128, N*vHeadDim)},
            {std::min(512, H), std::min(512, H)});  // 14个
        for (int ki = 0; ki < kSplit; ki++) {
            auto inputMk = View(quantizedA, {B*S, kSplitSize}, {0, ki * kSplitSize});
            auto inputKn = View(wOi, {kSplitSize, H}, {ki * kSplitSize, 0});
            auto tmp = npu::tile_fwk::Matrix::Matmul(DT_INT32, inputMk, inputKn, tmpC);  // (32, 16384) @ (16384, 7168)
            matmulResult.emplace_back(tmp);
        }
        Program::GetInstance().GetTileShape().SetVecTileShapes(std::min(32, B*S), std::min(512, H));  // 14个
        Tensor res = npu::tile_fwk::Reduce(matmulResult, ReduceMode::ATOMIC_ADD);

        // 原mm5Int32 = res;  //检测
        // Program::GetInstance().GetTileShape().SetVecTileShapes(32, std::min(64, N*vHeadDim)); // 112块 fail
        Program::GetInstance().GetTileShape().SetVecTileShapes(32, std::min(32, N*vHeadDim)); // 224块 ok
        res = Cast(res, DataType::DT_FP32); // (32, 7168)
        // mm5fp32 = res;   //检测
        res = Mul(res, dequantScaleA);
        Tensor weightOScaleW2Dim = Reshape(wOscaleWi, {1, H});
        res = Mul(res, weightOScaleW2Dim);
        Tensor bmm5Res = Cast(res, DataType::DT_BF16, CAST_RINT);

        outputT = Reshape(bmm5Res, {B, S, H});
        // 原Reshape(bmm5Res, {B, S, H});
    }
    TileFwkEndFunction();

    /* torch compile */
    void* handle = TileFwkCompile();

    /* torch prepare workspace memory */
    uint8_t* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    TileFwkGetWorkspaceSize(handle, &workspaceSize);
    machine::GetRA()->AllocDevAddr(&workspaceAddr, workspaceSize);

    std::vector<void*> opArgsRun = {inputPtr, wUvPtr, wUvScaleWptr, wOptr, wOscaleWptr, outPtr, mm5int32_ptr, mm5fp32_ptr};

    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::vector<T> golden(outputSize);
    std::vector<T> res(outputSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)outPtr, outputByteSize);
    readInput<T>(GetGoldenDir() + "/attn_output.bin", golden);
    bool ret = resultCmp<T>(golden, res, 0.03f, 700, 1000, false);

    /*
    // debug
    std::vector<int32_t> golden(outputSize);
    std::vector<int32_t> res(outputSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)mm5int32_ptr, mm5Int32ByteSize);
    readInput<int32_t>(GetGoldenDir() + "/mm5_int32.bin", golden);
    bool ret = resultCmp<int32_t>(golden, res, 0.03f);

    std::vector<float> golden(outputSize);
    std::vector<float> res(outputSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)mm5fp32_ptr, mm5fp32ByteSize);
    readInput<float>(GetGoldenDir() + "/mm5_fp32.bin", golden);
    bool ret = resultCmp<float>(golden, res, 0.03f);
    */

    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardCostTest, dynamic_pa_post_static_cast_first) {
    aclInit(nullptr);
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
    TileFwkInit("");

    int paramsSize = 7;
    std::vector<int64_t> params(paramsSize);
    readInput<int64_t>(GetGoldenDir() + "/params.bin", params);
    int B = params[0];
    int S = params[1];
    int N = params[2];
    int H = params[3];
    int kvLoraRank = params[4];
    int vHeadDim = params[5];
    int dtypeNum = params[6];

    DataType dTypeInt8 = DT_INT8;
    DataType dTypeFp32 = DT_FP32;
    DataType dType = DT_FP32;
    if (dtypeNum == 0) {
        dType = DT_FP32;
    } else if (dtypeNum == 1) {
        dType = DT_FP16;
    } else if (dtypeNum == 2) {
        dType = DT_BF16;
    }
    int dtypeSize = BytesOf(dType);

    typedef npu::tile_fwk::bfloat16 T;
    typedef int8_t T_INT8;
    typedef float T_FLOAT;

    int inputSize = B*N*S*kvLoraRank;
    int wUvSize = N*kvLoraRank*vHeadDim;
    // int wUvScaleWSize = N*1*vHeadDim;
    int wOSize = N*vHeadDim*H;
    int wOScaleWSize = 1*H;
    int outputSize = B*S*H;
    // int t1Size = B*S*N*kvLoraRank;

    uint64_t outputByteSize = outputSize * dtypeSize;
    uint8_t* outPtr = allocDevAddr(outputByteSize);

    uint64_t mm5Int32ByteSize = outputSize * BytesOf(DT_INT32);
    uint8_t* mm5int32_ptr = allocDevAddr(mm5Int32ByteSize);

    uint64_t mm5fp32ByteSize = outputSize * BytesOf(DT_FP32);
    uint8_t* mm5fp32_ptr = allocDevAddr(mm5fp32ByteSize);

    std::vector<int64_t> inputShape = {B*N*S,kvLoraRank};
    std::vector<int64_t> wUvShape = {N,kvLoraRank,vHeadDim};
    std::vector<int64_t> wUvScaleWShape = {N,1,vHeadDim};
    std::vector<int64_t> wOShape = {N*vHeadDim,H};
    std::vector<int64_t> wOScaleWShape = {H};
    std::vector<int64_t> outputShapeT = {B, S, H};
    std::vector<int64_t> mm5Int32ShapeT = {B * S, H};
    // std::vector<int64_t> t1Shape = {B, S, N, kvLoraRank};
    void *inputPtr = readToDev<T_FLOAT>(GetGoldenDir() + "/input.bin", inputSize);
    void *wUvPtr = readToDev<T>(GetGoldenDir() + "/w_uv.bin", wUvSize);
    // void *wUvScaleWptr = readToDev<T_FLOAT>(GetGoldenDir() + "/w_uv_scale_w.bin", wUvScaleWSize);
    void *wOptr = readToDev<T_INT8>(GetGoldenDir() + "/w_o.bin", wOSize);
    void *wOscaleWptr = readToDev<T_FLOAT>(GetGoldenDir() + "/w_o_scale_w.bin", wOScaleWSize);

    // void *t1Ptr = readToDev<T>(GetGoldenDir() + "/t1.bin", t1Size);

    Tensor inputI(DT_FP32, inputShape, (uint8_t *)inputPtr, "A");
    Tensor wUvI(dType, wUvShape, (uint8_t *)wUvPtr, "B");
    // Tensor wUvScaleWi(dTypeFp32, wUvScaleWShape, (uint8_t *)wUvScaleWptr, "B1");
    Tensor wOi(dTypeInt8, wOShape, (uint8_t *)wOptr, "C", TileOpFormat::TILEOP_NZ);
    Tensor wOscaleWi(dTypeFp32, wOScaleWShape, (uint8_t *)wOscaleWptr, "C1");
    Tensor outputT(dType, outputShapeT, outPtr, "D1");
    Tensor mm5Int32(DT_INT32, mm5Int32ShapeT, mm5int32_ptr, "mm5Int32");
    Tensor mm5fp32(DT_FP32, mm5Int32ShapeT, mm5fp32_ptr, "mm5fp32");
    // Tensor t1I(dType, t1Shape, (uint8_t *)t1Ptr, "E");

    TileFwkBeginFunction("ATTENTION_POST_T", {inputI, wUvI, wOi, wOscaleWi, outputT, mm5Int32, mm5fp32});
    {
        Program::GetInstance().GetTileShape().SetVecTileShapes({64, kvLoraRank});
        Tensor cast1 = Cast(inputI, DataType::DT_BF16); // (32*128, 512)
        Tensor r1Res = Reshape(cast1, {B * S, N, kvLoraRank}); // (32, 128, 512)
        Program::GetInstance().GetTileShape().SetVecTileShapes({B * 1, 2, kvLoraRank});
        Tensor t1Res = Transpose(r1Res, {0, 1}); // (32, 128, 512)

        // 原AscendProgram::GetInstance().GetConfig().Set<int>(L1_REUSE, 6);
        Program::GetInstance().GetConfig().Set<std::map<int, int>>(CUBE_NBUFFER_MAP, {{0, 4}});
        Program::GetInstance().GetTileShape().SetCubeTileShapes({std::min(32, B*S), std::min(32, B*S)},
            {std::min(256, kvLoraRank), std::min(256, kvLoraRank)},
            {std::min(128, vHeadDim), std::min(128, vHeadDim)});  // 128/4 个
        // 原[n,bs,kvLoraRank] * [n, kvLoraRank, vHeadDim] = [n,bs,vHeadDim]
        Tensor bmm4Res = Matrix::BatchMatmul(dType, t1Res, wUvI); // (128, 32, 512) @ (128, 512, 128) = (128, 32, 128)

        Program::GetInstance().GetTileShape().SetVecTileShapes(1, 32, vHeadDim); // 32个
        Tensor t3Res = Transpose(bmm4Res, {0, 1}); // [bs,n,vHeadDim]    // (128, 32, 128)
        Tensor r2Res = Reshape(t3Res, {B * S, N*vHeadDim});  // (32, 128, 128)

        Program::GetInstance().GetTileShape().SetVecTileShapes(1, 16384); // 32个
        auto quantA = QuantTmp(r2Res, true, false, Tensor());   // (32, 16384)
        // auto quantA = Quant(r2Res);   // (32, 16384)
        auto quantizedA = std::get<0>(quantA);
        auto dequantScaleA = std::get<1>(quantA);

        // (B*S, N*vHeadDim) @ (N*vHeadDim, H) = (B*S, H)
        // int8 @ int8 = int32
        Tensor tmpC(DT_INT32, {B*S, H}, "tmp_c");
        Program::GetInstance().GetTileShape().SetVecTileShapes(32, std::min(1024, H));  // 7个
        tmpC = MulS(tmpC, Element(DataType::DT_FP32, 0.0f));
        std::vector<Tensor> matmulResult;
        auto kSplit = 8;
        auto kSplitSize = N*vHeadDim / kSplit;
        Program::GetInstance().GetTileShape().SetCubeTileShapes({std::min(32, B*S), std::min(32, B*S)},
            {std::min(128, N*vHeadDim), std::min(128, N*vHeadDim)},
            {std::min(512, H), std::min(512, H)});  // 14个
        for (int ki = 0; ki < kSplit; ki++) {
            auto inputMk = View(quantizedA, {B*S, kSplitSize}, {0, ki * kSplitSize});
            auto inputKn = View(wOi, {kSplitSize, H}, {ki * kSplitSize, 0});
            auto tmp = npu::tile_fwk::Matrix::Matmul(DT_INT32, inputMk, inputKn, tmpC);  // (32, 16384) @ (16384, 7168)
            matmulResult.emplace_back(tmp);
        }
        Program::GetInstance().GetTileShape().SetVecTileShapes(std::min(32, B*S), std::min(512, H));  // 14个
        Tensor res = npu::tile_fwk::Reduce(matmulResult, ReduceMode::ATOMIC_ADD);

        // 原mm5Int32 = res;  //检测
        // Program::GetInstance().GetTileShape().SetVecTileShapes(32, std::min(64, N*vHeadDim)); // 112块 fail
        Program::GetInstance().GetTileShape().SetVecTileShapes(32, std::min(32, N*vHeadDim)); // 224块 ok
        res = Cast(res, DataType::DT_FP32); // (32, 7168)
        // mm5fp32 = res;   //检测
        res = Mul(res, dequantScaleA);
        Tensor weightOScaleW2Dim = Reshape(wOscaleWi, {1, H});
        res = Mul(res, weightOScaleW2Dim);
        Tensor bmm5Res = Cast(res, DataType::DT_BF16, CAST_RINT);

        outputT = Reshape(bmm5Res, {B, S, H});
        // 原Reshape(bmm5Res, {B, S, H});
    }
    TileFwkEndFunction();

    /* torch compile */
    void* handle = TileFwkCompile();

    /* torch prepare workspace memory */
    uint8_t* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    TileFwkGetWorkspaceSize(handle, &workspaceSize);
    machine::GetRA()->AllocDevAddr(&workspaceAddr, workspaceSize);

    std::vector<void*> opArgsRun = {inputPtr, wUvPtr, wOptr, wOscaleWptr, outPtr, mm5int32_ptr, mm5fp32_ptr};

    TileFwkRunAsync(handle, workspaceAddr, machine::GetRA()->GetStreamAICPU(),  opArgsRun);
    int rc = rtStreamSynchronize(machine::GetRA()->GetStreamAICPU());
    if (rc < 0) {
        ASSERT(false);
        ALOG_INFO_F("FA function aicpu stream sync failed");
    }

    std::vector<T> golden(outputSize);
    std::vector<T> res(outputSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)outPtr, outputByteSize);
    readInput<T>(GetGoldenDir() + "/attn_output.bin", golden);
    bool ret = resultCmp<T>(golden, res, 0.03f, 1000, 1000, false);

    /*
    // debug
    std::vector<int32_t> golden(outputSize);
    std::vector<int32_t> res(outputSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)mm5int32_ptr, mm5Int32ByteSize);
    readInput<int32_t>(GetGoldenDir() + "/mm5_int32.bin", golden);
    bool ret = resultCmp<int32_t>(golden, res, 0.03f);

    std::vector<float> golden(outputSize);
    std::vector<float> res(outputSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)mm5fp32_ptr, mm5fp32ByteSize);
    readInput<float>(GetGoldenDir() + "/mm5_fp32.bin", golden);
    bool ret = resultCmp<float>(golden, res, 0.03f);
    */

    uint64_t taskTime = npu::tile_fwk::DeviceRunner::Get().GetTasksTime();
    uint64_t threshold = 10500;
    std::cout<<"dynamic_pa_post_static_cast_first threshold:"<<threshold<<" cost:"<<taskTime<<std::endl;

    EXPECT_EQ(ret, true);
}
