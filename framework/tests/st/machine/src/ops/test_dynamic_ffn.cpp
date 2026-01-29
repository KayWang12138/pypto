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
 * \file test_dynamic_ffn.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "test_dev_func_runner.h"
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/deepseek/deepseek_moeinfer.h"
#include "machine/utils/dynamic/dev_encode.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class DynamicFFNTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

namespace {
TEST_F(DynamicFFNTest, TestOnbroadDynamicFFN) {
    SetInterpreterConfig();
    TileShape::Current().SetVecTile(32, 256);
    TileShape::Current().SetCubeTile({32, 32}, {128, 256}, {128, 128});
    constexpr int BATCH_SIZE = 32;
    constexpr int SEQUENCE = 1;
    constexpr int H = 7168;
    constexpr int ExpertDim = 2048;
    constexpr int BS = BATCH_SIZE * SEQUENCE;
    constexpr int BASIC_BATCH = 32;
    std::vector<uint8_t> devProgBinary;
    std::vector<int64_t> hiddenStatesShape{BS, H};
    std::vector<int64_t> weightShape{H, ExpertDim};
    std::vector<int64_t> OutShape{BS, H};

    Tensor hiddenStates(DT_FP32, hiddenStatesShape, "hiddenStates");
    Tensor ffnweight1(DT_FP16, weightShape, "weightShape1", TileOpFormat::TILEOP_NZ);
    Tensor ffnweight2(DT_FP16, weightShape, "weightShape2", TileOpFormat::TILEOP_NZ);
    Tensor ffnweight3(DT_FP16, weightShape, "weightShape3", TileOpFormat::TILEOP_NZ);
    Tensor ffnout(DT_FP32, OutShape, "ffnout");

    std::vector<float>hiddenStatesData(BS * H);
    std::vector<npu::tile_fwk::float16>ffnweight1Data(ExpertDim * H);
    std::vector<npu::tile_fwk::float16>ffnweight2Data(ExpertDim * H);
    std::vector<npu::tile_fwk::float16>ffnweight3Data(ExpertDim * H);
    std::vector<float>golden(BS * H);

    readInput(GetGoldenDir() + "/hidden_states.bin", hiddenStatesData);
    readInput(GetGoldenDir() + "/ffnWeight1.bin", ffnweight1Data);
    readInput(GetGoldenDir() + "/ffnWeight2.bin", ffnweight2Data);
    readInput(GetGoldenDir() + "/ffnWeight3.bin", ffnweight3Data);
    readInput(GetGoldenDir() + "/final_out.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(hiddenStates, hiddenStatesData),
        RawTensorData::CreateTensor<npu::tile_fwk::float16>(ffnweight1, ffnweight1Data),
        RawTensorData::CreateTensor<npu::tile_fwk::float16>(ffnweight2, ffnweight2Data),
        RawTensorData::CreateTensor<npu::tile_fwk::float16>(ffnweight3, ffnweight3Data),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(ffnout, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(ffnout, golden),
    });

    DynamicFFN(hiddenStates, ffnweight1, ffnweight2, ffnweight3, ffnout, BASIC_BATCH);

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
#endif
}

static void SetFFNQuantConfig()
{
    TileShape::Current().SetVecTile(32, 256);
    TileShape::Current().SetCubeTile({32, 32}, {256, 256}, {128, 128}, true);
    config::SetPassOption(VEC_NBUFFER_MODE, 1);
    config::SetPassOption(CUBE_L1_REUSE_SETTING, std::map<int64_t, int64_t>{{-1, 2}});
    config::SetPassOption(MG_COPYIN_UPPER_BOUND, 10*1024*1024);
    config::SetCodeGenOption(SUPPORT_DYNAMIC_ALIGNED, true);
}

static void CreateFFNQuantTensors(int BS, int H, int ExpertDim,
    Tensor& hiddenStates, Tensor& hiddenStatesScale, Tensor& ffnWeight1,
    Tensor& ffnWeight2, Tensor& ffnWeight3, Tensor& ffnScale1,
    Tensor& ffnScale2, Tensor& ffnScale3, Tensor& ffnout)
{
    std::vector<int64_t> hiddenStatesShape{BS, H};
    std::vector<int64_t> weightShape{H, ExpertDim};
    std::vector<int64_t> outShape{BS, H};

    hiddenStates = Tensor(DT_INT8, hiddenStatesShape, "hiddenStates");
    hiddenStatesScale = Tensor(DT_FP32, {BS, 1}, "hiddenStatesScale");
    ffnWeight1 = Tensor(DT_INT8, weightShape, "ffnWeight1", TileOpFormat::TILEOP_NZ);
    ffnWeight2 = Tensor(DT_INT8, weightShape, "ffnWeight2", TileOpFormat::TILEOP_NZ);
    ffnWeight3 = Tensor(DT_INT8, weightShape, "ffnWeight3", TileOpFormat::TILEOP_NZ);
    ffnScale1 = Tensor(DT_FP32, {1, ExpertDim}, "ffnScale1");
    ffnScale2 = Tensor(DT_FP32, {1, ExpertDim}, "ffnScale2");
    ffnScale3 = Tensor(DT_FP32, {1, H}, "ffnScale3");
    ffnout = Tensor(DT_FP32, outShape, "ffnout");
}

static void ReadFFNQuantData(int BS, int H, int ExpertDim,
    std::vector<int8_t>& hiddenStatesData, std::vector<float>& hiddenStatesScaleData,
    std::vector<int8_t>& ffnweight1Data, std::vector<int8_t>& ffnweight2Data,
    std::vector<int8_t>& ffnweight3Data, std::vector<float>& ffnScale1Data,
    std::vector<float>& ffnScale2Data, std::vector<float>& ffnScale3Data,
    std::vector<float>& golden)
{
    hiddenStatesData.resize(BS * H);
    hiddenStatesScaleData.resize(BS);
    ffnweight1Data.resize(ExpertDim * H);
    ffnweight2Data.resize(ExpertDim * H);
    ffnweight3Data.resize(ExpertDim * H);
    ffnScale1Data.resize(ExpertDim);
    ffnScale2Data.resize(ExpertDim);
    ffnScale3Data.resize(H);
    golden.resize(BS * H);

    readInput(GetGoldenDir() + "/hidden_states.bin", hiddenStatesData);
    readInput(GetGoldenDir() + "/hidden_states_scale.bin", hiddenStatesScaleData);
    readInput(GetGoldenDir() + "/ffnWeight1.bin", ffnweight1Data);
    readInput(GetGoldenDir() + "/ffnWeight2.bin", ffnweight2Data);
    readInput(GetGoldenDir() + "/ffnWeight3.bin", ffnweight3Data);
    readInput(GetGoldenDir() + "/ffnScale1.bin", ffnScale1Data);
    readInput(GetGoldenDir() + "/ffnScale2.bin", ffnScale2Data);
    readInput(GetGoldenDir() + "/ffnScale3.bin", ffnScale3Data);
    readInput(GetGoldenDir() + "/final_out.bin", golden);
}

TEST_F(DynamicFFNTest, TestOnbroadDynamicFFNQuant) {
    SetFFNQuantConfig();

    constexpr int BATCH_SIZE = 32;
    constexpr int SEQUENCE = 1;
    constexpr int H = 7168;
    constexpr int ExpertDim = 2048;
    constexpr int BS = BATCH_SIZE * SEQUENCE;
    constexpr int BASIC_BATCH = 32;

    Tensor hiddenStates, hiddenStatesScale, ffnWeight1, ffnWeight2, ffnWeight3;
    Tensor ffnScale1, ffnScale2, ffnScale3, ffnout;
    CreateFFNQuantTensors(BS, H, ExpertDim, hiddenStates, hiddenStatesScale, ffnWeight1,
        ffnWeight2, ffnWeight3, ffnScale1, ffnScale2, ffnScale3, ffnout);

    std::vector<int8_t> hiddenStatesData;
    std::vector<float> hiddenStatesScaleData;
    std::vector<int8_t> ffnweight1Data, ffnweight2Data, ffnweight3Data;
    std::vector<float> ffnScale1Data, ffnScale2Data, ffnScale3Data;
    std::vector<float> golden;
    ReadFFNQuantData(BS, H, ExpertDim, hiddenStatesData, hiddenStatesScaleData, ffnweight1Data,
        ffnweight2Data, ffnweight3Data, ffnScale1Data, ffnScale2Data, ffnScale3Data, golden);

    DynamicFFNQuant(hiddenStates, hiddenStatesScale, ffnWeight1, ffnWeight2, ffnWeight3,
        ffnScale1, ffnScale2, ffnScale3, ffnout, BASIC_BATCH);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int8_t>(hiddenStates, hiddenStatesData),
        RawTensorData::CreateTensor<float>(hiddenStatesScale, hiddenStatesScaleData),
        RawTensorData::CreateTensor<int8_t>(ffnWeight1, ffnweight1Data),
        RawTensorData::CreateTensor<int8_t>(ffnWeight2, ffnweight2Data),
        RawTensorData::CreateTensor<int8_t>(ffnWeight3, ffnweight3Data),
        RawTensorData::CreateTensor<float>(ffnScale1, ffnScale1Data),
        RawTensorData::CreateTensor<float>(ffnScale2, ffnScale2Data),
        RawTensorData::CreateTensor<float>(ffnScale3, ffnScale3Data),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(ffnout, 0),
    });

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
#endif
}
}
