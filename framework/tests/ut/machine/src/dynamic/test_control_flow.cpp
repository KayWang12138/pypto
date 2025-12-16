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
 * \file test_control_flow.cpp
 * \brief
 */

#include "interface/utils/string_utils.h"

#include "test_machine_common.h"

struct ControlFlowTest : UnitTestBase {};

std::string GetDeclName(const std::string &name) {
    std::vector<std::string> descList = StringUtils::Split(name, "_");
    return descList[1];
}

TEST_F(ControlFlowTest, RunDeviceContext) {
    config::SetRuntimeOption<int64_t>(FIRST_STITCH_TASK_LOOP_NUM, 0x4);
    config::SetRuntimeOption<int64_t>(SUBSEQ_STITCH_TASK_INCR_LOOP_NUM, 0);

    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);

    int n = tiling * 4;
    Tensor inputA(DT_INT32, {n, n}, "A");
    Tensor inputB(DT_INT32, {n, n}, "B");
    Tensor output(DT_INT32, {n, n}, "O");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(inputA, 1),
        RawTensorData::CreateConstantTensor<int32_t>(inputB, 2),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });

    FUNCTION("main", {inputA, inputB}, {output}) {
        LOOP("s0", FunctionType::DYNAMIC_LOOP, _, LoopRange(0x1)) {
            (void)_;
            output = Add(inputA, inputB);
        }
        LOOP("s1", FunctionType::DYNAMIC_LOOP, _, LoopRange(0x40 - 0x1)) {
            (void)_;
            output = Add(output, inputB);
        }
    }

    struct Inspector {
        int count{0};
        std::vector<DevAscendFunction *> rootList;
        static void Entry(void *inspector_, DeviceExecuteContext *execCtx, DynDeviceTask *task) {
            Inspector *inspector = reinterpret_cast<Inspector *>(inspector_);
            (void)execCtx; (void)task;
            inspector->count++;
            DynFuncDataCache *cacheList = task->GetDynFuncDataCacheList();
            for (size_t k = 0; k < task->dynFuncDataCacheListSize; k++) {
                inspector->rootList.push_back(cacheList->At(k).devFunc);
            }
        }
    };
    Inspector inspector;
    PyptoKernelCtrlServerRegisterTaskInspector(Inspector::Entry, &inspector);

    DeviceLauncherConfig config;
    config.blockdim = 24; // 24: max blockdim
    EXPECT_EQ(0, EmulationLauncher::EmulationRunOnce(Program::GetInstance().GetLastFunction(), config));
    EXPECT_EQ(0x10, inspector.count);
    EXPECT_EQ(0x40, inspector.rootList.size());
    EXPECT_EQ("s0", GetDeclName(inspector.rootList[0]->GetRawName()));
    for (size_t k = 1; k < 0x40; k++) {
        EXPECT_EQ("s1", GetDeclName(inspector.rootList[k]->GetRawName()));
    }
}