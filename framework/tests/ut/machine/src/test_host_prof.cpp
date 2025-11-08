/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_prof.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <stdint.h>
#include "machine/runtime/host_prof.h"
#include "interface/function/function.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "tilefwk/function.h"
#include "interface/program/program.h"


using namespace npu::tile_fwk;
constexpr uint32_t BlOCK_DIM = 25;

class TestHostPro : public testing::Test {
public:
    static void SetUpTestCase() {
    }

    static void TearDownTestCase() {}

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(TestHostPro, test_ini) {
  TileShape::Current().SetVecTile(64, 64);
  int b = 1;
  int sq = 128;
  int d = 64;
  std::vector<int64_t> inputShape = {b * sq, d};
  std::vector<int64_t> outShape = {b * sq, d};

  Tensor input1(DT_FP32, inputShape, "intput1");
  Tensor input2(DT_FP32, inputShape, "intput2");
  Tensor curSeq(DT_INT32, {b, 1}, "curSeq");
  Tensor out(DT_FP32, outShape, "out");

  std::vector<int> actSeqsData(b, 100);
  std::vector<float> golden(b * sq * d, 0.001f);
  for (int i = 0; i < b; i++) {
      int offset = i * sq * d;
      std::fill(golden.begin() + offset, golden.begin() + offset + actSeqsData[i] * d, 2.0);
  }

  ProgramData::GetInstance().AppendInputs({
      RawTensorData::CreateConstantTensor<float>(input1, 1.0),
      RawTensorData::CreateConstantTensor<float>(input2, 1.0),
      RawTensorData::CreateTensor<int32_t>(curSeq, actSeqsData),
  });

  ProgramData::GetInstance().AppendOutputs({
      RawTensorData::CreateConstantTensor<float>(out, 0.001f),
  });

  ProgramData::GetInstance().AppendGoldens({
      RawTensorData::CreateTensor<float>(out, golden),
  });

  FUNCTION("main", {input1, input2, curSeq}, {out}) {
      LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(b)) {
          auto seq = GetTensorData(curSeq, {batchId, 0});
          Tensor intput11 = View(input1, {sq, d}, {seq, d}, {batchId, 0});
          Tensor intput22 = View(input2, {sq, d}, {seq, d}, {batchId, 0});
          auto tmp = Add(intput11, intput22);
          Assemble(tmp, {batchId * sq, 0}, out);
      }
  }
  HostProf hostProf;
  hostProf.SetProfFunction(Program::GetInstance().GetLastFunction());
  uint64_t startTime = 0;
  uint64_t endTime = 1;
  
  hostProf.HostProfReportApi(startTime, endTime);
  hostProf.HostProfReportNodeInfo(endTime, BlOCK_DIM, 0);
  hostProf.HostProfReportContextInfo(endTime);
}