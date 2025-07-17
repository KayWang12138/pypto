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
 * \file test_merge_view_assemble.cpp
 * \brief Unit test for merge_view_assemble pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"
#include "passes/tile_graph_pass/merge_view_assemble.h"
#include <fstream>
#include <vector>
#include <string>

using namespace npu::tile_fwk;

class MergeViewAssembleTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "ViewAssembleTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

TEST_F(MergeViewAssembleTest, TestMergeViewAssemble) {
    constexpr int32_t tilex = 8;
    constexpr int32_t tiley = 16;
    constexpr int expectedOps = 8;
    constexpr int expectedView1 = 2;
    constexpr int expectedView2 = 2;
    constexpr int expectedAdd = 2;
    constexpr int expectedAssemble = 1;
    std::vector<int> shape{16, 16};
    Tensor a(DT_FP32, shape, "a");
    Tensor in_tensor(DT_FP32, shape, "in_tensor");
    Tensor out_tensor(DT_FP32, shape, "out_tensor");

    Program::GetInstance().GetTileShape().SetVecTileShapes(tilex, tiley);

    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("ViewAssembleTestStrategy", {
        {   "RemoveRedundentReshape",   "RemoveRedundentReshape",  PassType::TYPE_TENSOR_GRAPH},
        {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
        {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
    });

    Function* originFunction = nullptr;
    std::vector<int> originOpmagic;
    FUNCTION("AddFunction") {
        out_tensor = Add(in_tensor, a);
        originFunction = Program::GetInstance().GetCurrentFunction();
        ASSERT_NE(originFunction, nullptr) << "Current function pointer is null";
        auto operations = originFunction->Operations();
        for (const auto &op : operations) {
            originOpmagic.emplace_back(op.opmagic);
        }
    }

    std::string jsonFilePath = "./config/pass/json/merge_view_assemble.json";
    bool dumpJsonFlag = false;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
        Json readData = LoadJsonFile(jsonFilePath);
        Program::GetInstance().LoadJson(readData);
    }

    Function* currentFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_AddFunction");

    Program testProgram(HostMachineMode::SERVER);
    MergeViewAssemble mergeViewAssemble;
    mergeViewAssemble.PreCheck(*currentFunction);
    mergeViewAssemble.RunOnFunction(*currentFunction);
    mergeViewAssemble.PostCheck(*currentFunction);

    // ================== Verify Pass Effect ==================
    auto updated_operations = currentFunction->Operations();
    EXPECT_EQ(updated_operations.size(), expectedOps) << "14 operations should remain";
    int view1_count = 0;
    int view2_count = 0;
    int add_count = 0;
    int assemble1_count = 0;
    int assemble2_count = 0;
    std::vector offset1 = {0, 0};
    std::vector offset2 = {8, 0};
    for (const auto &op : updated_operations) {
        if (op.GetOpcodeStr() == "VIEW") {
            auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute().get());
            ASSERT_NE(viewOpAttribute, nullptr);
            if (viewOpAttribute->GetFrom() == offset1) {
                view1_count++;
            } else if (viewOpAttribute->GetFrom() == offset2) {
                view2_count++;
            }
        } else if (op.GetOpcodeStr() == "ADD") {
            add_count++;
        } else if (op.GetOpcodeStr() == "ASSEMBLE") {
            auto assembleOpAttribute = dynamic_cast<AssembleOpAttribute*>(op.GetOpAttribute().get());
            ASSERT_NE(assembleOpAttribute, nullptr);
            if (assembleOpAttribute->GetToOffset() == offset1) {
                assemble1_count++;
            } else if (assembleOpAttribute->GetToOffset() == offset2) {
                assemble2_count++;
            }
        }
    }

    EXPECT_EQ(view1_count, expectedView1) << "6 VIEW1 operations should remain";
    EXPECT_EQ(view2_count, expectedView2) << "4 VIEW2 operations should remain";
    EXPECT_EQ(add_count, expectedAdd) << "2 ADD operations should remain";
    EXPECT_EQ(assemble1_count, expectedAssemble) << "1 ASSEMBLE1 operation should remain";
    EXPECT_EQ(assemble2_count, expectedAssemble) << "1 ASSEMBLE2 operation should remain";

    // Check the offset of the View operation
}
