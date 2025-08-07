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
 * \file test_split_large_fanout_tensor.cpp
 * \brief Unit test for Split Large Fanout Tensor pass.
 */

#include <vector>
#include <numeric>
#include "gtest/gtest.h"
#include "operation/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"
#include "passes/tile_graph_pass/split_large_fanout_tensor.h"
#include "computational_graph_builder.h"

using namespace npu::tile_fwk;

namespace npu {
namespace tile_fwk {

class SplitLargeFanoutTensorTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "SplitLargeFanoutTensorTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}

    std::vector<int> CountViewAssemble(Function &func) {
        std::vector<int> result = {0, 0};
        for (auto &op : func.Operations()) {
            std::cout << op.GetOpcodeStr() << " " << op.GetOpMagic() << std::endl;
            if (op.GetOpcode() == Opcode::OP_VIEW) {
                result[0]++;
            } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                result[1]++;
            }
        }
        return result;
    }

    // [16, 16] --> View --> [8, 8] --> Sub --> [8, 8] --> Assemble --> [16, 16]
    void TileExpandSub(ComputationalGraphBuilder &G, const int N, const int T) {
        std::vector<int> tileShape{T, T};
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                std::vector<int> offset = {i * T, j * T};

                std::string localA = "a_" + std::to_string(i * N + j);
                G.AddTensor(DataType::DT_FP32, tileShape, localA);
                auto tensorA = G.GetTensor(localA);
                tensorA->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
                G.AddOp(Opcode::OP_VIEW, {"a"}, {localA}, "View_A_" + std::to_string(i * N + j));
                auto View_A = G.GetOp("View_A_" + std::to_string(i * N + j));
                auto attrA = std::make_shared<ViewOpAttribute>(offset, MemoryType::MEM_UB);
                View_A->SetOpAttribute(attrA);

                std::string localB = "b_" + std::to_string(i * N + j);
                G.AddTensor(DataType::DT_FP32, tileShape, localB);
                auto tensorB = G.GetTensor(localB);
                tensorB->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
                G.AddOp(Opcode::OP_VIEW, {"b"}, {localB}, "View_B_" + std::to_string(i * N + j));
                auto View_B = G.GetOp("View_B_" + std::to_string(i * N + j));
                auto attrB = std::make_shared<ViewOpAttribute>(offset, MemoryType::MEM_UB);
                View_B->SetOpAttribute(attrB);

                std::string localSubOut = "sub_out_" + std::to_string(i * N + j);
                G.AddTensor(DataType::DT_FP32, tileShape, localSubOut);
                G.AddOp(Opcode::OP_SUB, {localA, localB}, {localSubOut}, "Sub_" + std::to_string(i * N + j));
                auto tensorSubOut = G.GetTensor(localSubOut);
                tensorSubOut->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

                G.AddOp(Opcode::OP_ASSEMBLE, {localSubOut}, {"sub_out"}, "Assemble_" + std::to_string(i * N + j));
                auto attrAssemble = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset);
                auto assembleOp = G.GetOp("Assemble_" + std::to_string(i * N + j));
                assembleOp->SetOpAttribute(attrAssemble);
            }
        }
    }
};

TEST_F(SplitLargeFanoutTensorTest, BeCovered_Full) {
    int NUM_32 = 32;
    int NUM_64 = 64;
    int NUM_128 = 128;
    int NUM_256 = 256;
    int NUM_512 = 512;
    int NUM_576 = 512 + 64;
    std::vector<int> shape0{NUM_128, NUM_64};
    std::vector<int> tiledShape0{NUM_32, NUM_64};

    std::vector<int> shape1{NUM_128, NUM_512};
    std::vector<int> tiledShape1{NUM_32, NUM_512};

    std::vector<int> tiledShape2{NUM_32, NUM_576};
    std::vector<int> tiledShape3{NUM_32, NUM_256};
    ComputationalGraphBuilder G;

    // [128, 512] --> View(64, 0) --> [32, 512]
    G.AddTensor(DataType::DT_FP32, shape1, "a"); // [128, 512]
    auto a = G.GetTensor("a");
    a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(DataType::DT_FP32, tiledShape1, "tiledA"); // [32, 512]
    auto tiledA = G.GetTensor("tiledA");
    tiledA->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddOp(Opcode::OP_VIEW, {"a"}, {"tiledA"}, "View_A");
    auto View_A =  G.GetOp("View_A");
    std::vector<int> offsetA = {NUM_64, 0};
    auto attrA = std::make_shared<ViewOpAttribute>(offsetA, MemoryType::MEM_DEVICE_DDR);
    View_A->SetOpAttribute(attrA);

    // [128, 64] --> View(64, 0) --> [32, 64]
    G.AddTensor(DataType::DT_FP32, shape0, "b"); // [128, 64]
    auto b = G.GetTensor("b");
    b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(DataType::DT_FP32, tiledShape0, "tiledB"); // [32, 64]
    auto tiledB = G.GetTensor("tiledB");
    tiledB->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddOp(Opcode::OP_VIEW, {"b"}, {"tiledB"}, "View_B");
    auto View_B =  G.GetOp("View_B");
    std::vector<int> offsetB = {NUM_64, 0};
    auto attrB = std::make_shared<ViewOpAttribute>(offsetB, MemoryType::MEM_DEVICE_DDR);
    View_B->SetOpAttribute(attrB);

    // [32, 512] concat [32, 64] -->  [32, 576]
    G.AddTensor(DataType::DT_FP32, tiledShape2, "tiledConcat"); // [32, 512] + [32, 64] --> [32, 576]
    auto tiledConcat = G.GetTensor("tiledConcat");
    tiledConcat->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddOp(Opcode::OP_ASSEMBLE, {"tiledA"}, {"tiledConcat"}, "Assemble_A");
    auto attrAssembleA = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, std::vector<int> {0, 0});
    auto assembleA = G.GetOp("Assemble_A");
    assembleA->SetOpAttribute(attrAssembleA);
    G.AddOp(Opcode::OP_ASSEMBLE, {"tiledB"}, {"tiledConcat"}, "Assemble_B");
    auto attrAssembleB = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, std::vector<int> {0, NUM_512});
    auto assembleB = G.GetOp("Assemble_B");
    assembleB->SetOpAttribute(attrAssembleB);

    // part 1: BeCovered
    // [32, 576] --> View(0, 0) --> [32, 256]
    G.AddTensor(DataType::DT_FP32, tiledShape3, "tiledView_1");
    auto tiledView_1 = G.GetTensor("tiledView_1");
    tiledView_1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_VIEW, {"tiledConcat"}, {"tiledView_1"}, "View_1");
    auto View_1 =  G.GetOp("View_1");
    std::vector<int> offset1 = {0, 0};
    auto attr1 = std::make_shared<ViewOpAttribute>(offset1, MemoryType::MEM_UB);
    View_1->SetOpAttribute(attr1);

    // part 2: BeCovered
    // [32, 576] --> View(0, 256) --> [32, 256]
    G.AddTensor(DataType::DT_FP32, tiledShape3, "tiledView_2");
    auto tiledView_2 = G.GetTensor("tiledView_2");
    tiledView_2->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_VIEW, {"tiledConcat"}, {"tiledView_2"}, "View_2");
    auto View_2 =  G.GetOp("View_2");
    std::vector<int> offset2 = {0, NUM_256};
    auto attr2 = std::make_shared<ViewOpAttribute>(offset2, MemoryType::MEM_UB);
    View_2->SetOpAttribute(attr2);

    // part 3: Perfectly Match
    // [32, 576] --> View(0, 512) --> [32, 64]
    G.AddTensor(DataType::DT_FP32, tiledShape0, "tiledView_3");
    auto tiledView_3 = G.GetTensor("tiledView_3");
    tiledView_3->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_VIEW, {"tiledConcat"}, {"tiledView_3"}, "View_3");
    auto View_3 =  G.GetOp("View_3");
    std::vector<int> offset3 = {0, NUM_512};
    auto attr3 = std::make_shared<ViewOpAttribute>(offset3, MemoryType::MEM_UB);
    View_3->SetOpAttribute(attr3);

    // part1 + part2
    // [32, 256] + [32, 256] --> [32, 256]
    G.AddTensor(DataType::DT_FP32, tiledShape3, "add_out");
    G.AddOp(Opcode::OP_ADD, {"tiledView_1", "tiledView_2"}, {"add_out"}, "Add");
    auto addOut = G.GetTensor("add_out");
    addOut->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

    G.AddTensor(DataType::DT_FP32, tiledShape3, "out1");
    auto out1 = G.GetTensor("out1");
    out1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddOp(Opcode::OP_ASSEMBLE, {"add_out"}, {"out1"}, "Assemble_1");
    auto attrAssemble1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int> {0, 0});
    auto assemble1 = G.GetOp("Assemble_1");
    assemble1->SetOpAttribute(attrAssemble1);

    // part3 exp
    // [32, 64] --> Exp --> [32, 64]
    G.AddTensor(DataType::DT_FP32, tiledShape0, "exp_out");
    G.AddOp(Opcode::OP_EXP, {"tiledView_3"}, {"exp_out"}, "Exp");
    auto expOut = G.GetTensor("exp_out");
    expOut->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    
    G.AddTensor(DataType::DT_FP32, tiledShape0, "out2");
    auto out2 = G.GetTensor("out2");
    out2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddOp(Opcode::OP_ASSEMBLE, {"exp_out"}, {"out2"}, "Assemble_2");
    auto attrAssemble2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int> {0, 0});
    auto assemble2 = G.GetOp("Assemble_2");
    assemble2->SetOpAttribute(attrAssemble2);

    G.SetInCast({"a", "b"});
    G.SetOutCast({"out1", "out2"});
    Function *function = G.GetFunction();

    // 确认构图完毕
    constexpr int opNumBefore = 11;
    constexpr int viewNumBefore = 5;
    constexpr int assembleNumBefore = 4;
    auto countResultBefore = CountViewAssemble(*function);
    int viewNumCount = countResultBefore[0];
    int assembleNumCount = countResultBefore[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW before pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;
    std::unordered_map<int, std::vector<int>> viewOpToUbBfore;
    for (auto &op : function->Operations()) {
        if (op.GetOpcode() != Opcode::OP_VIEW) {
            continue;
        }
        auto viewAttr = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
        MemoryType toType = viewAttr->GetTo();
        if (toType != MemoryType::MEM_UB) {
            continue;
        }
        viewOpToUbBfore.insert({op.GetOpMagic(), viewAttr->GetFromOffset()});
    }
    /*
    dump graph before Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    constexpr int opNumAfter = 7;
    constexpr int viewNumAfter = 3;
    constexpr int assembleNumAfter = 2;
    auto countResultAfter = CountViewAssemble(*function);
    viewNumCount = countResultAfter[0];
    assembleNumCount = countResultAfter[1];
    EXPECT_EQ(function->Operations().size(), opNumAfter) << opNumAfter << " operations after pass";
    EXPECT_EQ(viewNumCount, viewNumAfter) << viewNumAfter << " OP_VIEW after pass";
    EXPECT_EQ(assembleNumCount, assembleNumAfter) << assembleNumAfter << " OP_ASSEMBLE after pass";

    for (auto &op : function->Operations()) {
        if (op.GetOpcode() != Opcode::OP_VIEW) {
            continue;
        }
        auto viewAttr = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
        MemoryType toType = viewAttr->GetTo();
        if (toType != MemoryType::MEM_UB) {
            continue;
        }
        EXPECT_NE(viewOpToUbBfore.find(op.GetOpMagic()), viewOpToUbBfore.end());
        auto oldOffset = viewOpToUbBfore.at(op.GetOpMagic());
        auto newOffset = viewAttr->GetFromOffset();
        EXPECT_EQ(newOffset[0], NUM_64);
    }
}

TEST_F(SplitLargeFanoutTensorTest, Unmatched) {
    int N = 2;
    int T = 8;
    std::vector<int> shape0{N * T, N * T};
    std::vector<int> shape1{T, T};
    std::vector<int> shape2{N * T, T};
    std::vector<int> shape3{T / N, N * T};
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, shape0, "a");
    G.AddTensor(DataType::DT_FP32, shape0, "b");
    G.AddTensor(DataType::DT_FP32, shape0, "sub_out");
    G.AddTensor(DataType::DT_FP32, shape0, "out");
    // [16, 16] --> View --> [8, 8] --> Sub --> [8, 8] --> Assemble --> [16, 16]
    TileExpandSub(G, N, T);
    auto subOut = G.GetTensor("sub_out");
    subOut->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

    //  [16, 16] --> View --> [4, 16] --> Exp --> [4, 16]  --> [16, 16]
    for (int i = 0; i < T / N; i++) {
        // View
        G.AddTensor(DataType::DT_FP32, shape3, "viewOut_" + std::to_string(i));
        auto viewOut = G.GetTensor("viewOut_" + std::to_string(i));
        viewOut->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
        G.AddOp(Opcode::OP_VIEW, {"sub_out"}, {"viewOut_" + std::to_string(i)}, "View_" + std::to_string(i));
        auto viewOp =  G.GetOp("View_" + std::to_string(i));
        std::vector<int> offset = {i * T / N, 0};
        auto attr = std::make_shared<ViewOpAttribute>(offset, MemoryType::MEM_UB);
        viewOp->SetOpAttribute(attr);

        // Exp
        G.AddTensor(DataType::DT_FP32, shape3, "expOut_" + std::to_string(i));
        auto expOut = G.GetTensor("expOut_" + std::to_string(i));
        expOut->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
        G.AddOp(Opcode::OP_EXP, {"viewOut_" + std::to_string(i)}, {"expOut_" + std::to_string(i)}, "Exp_" + std::to_string(i));

        // Assemble
        G.AddOp(Opcode::OP_ASSEMBLE, {"expOut_" + std::to_string(i)}, {"out"}, "AssembleFinal_" + std::to_string(i));
        auto attrAssemble = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int> {i * T / N, 0});
        auto assembleOp = G.GetOp("AssembleFinal_" + std::to_string(i));
        assembleOp->SetOpAttribute(attrAssemble);
    }

    auto a = G.GetTensor("a");
    a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto b = G.GetTensor("b");
    b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto out = G.GetTensor("out");
    out->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    
    G.SetInCast({"a", "b"});
    G.SetOutCast({"out"});
    Function *function = G.GetFunction();
    // 确认构图完毕
    constexpr int opNumBefore = 28;
    constexpr int viewNumBefore = 12;
    constexpr int assembleNumBefore = 8;
    auto countResultBefore = CountViewAssemble(*function);
    int viewNumCount = countResultBefore[0];
    int assembleNumCount = countResultBefore[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW before pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;
    /*
    dump graph before Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    auto countResultAfter = CountViewAssemble(*function);
    viewNumCount = countResultAfter[0];
    assembleNumCount = countResultAfter[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations after pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW after pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE after pass";
}

TEST_F(SplitLargeFanoutTensorTest, PerfectlyMatchWithAll_Full) {
    int N = 2;
    int T = 8;
    std::vector<int> shape0{N * T, N * T};
    std::vector<int> shape1{T, T};
    std::vector<int> shape2{N * T, T};
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, shape0, "a");
    G.AddTensor(DataType::DT_FP32, shape0, "b");
    G.AddTensor(DataType::DT_FP32, shape2, "out");
    G.AddTensor(DataType::DT_FP32, shape0, "sub_out");
    // [16, 16] --> View --> [8, 8] --> Sub --> [8, 8] --> Assemble --> [16, 16]
    TileExpandSub(G, N, T);
    auto subOut = G.GetTensor("sub_out");
    subOut->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

    // View 获取 [16, 16] 左半边 [16, 8]
    G.AddTensor(DataType::DT_FP32, shape2, "sub_out_left");
    auto subOutLeft = G.GetTensor("sub_out_left");
    subOutLeft->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_VIEW, {"sub_out"}, {"sub_out_left"}, "View_Left");
    auto View_Left =  G.GetOp("View_Left");
    std::vector<int> offsetLeft = {0, 0};
    auto attrLeft = std::make_shared<ViewOpAttribute>(offsetLeft, MemoryType::MEM_UB);
    View_Left->SetOpAttribute(attrLeft);

    // View 获取 [16, 16] 右半边 [16, 8]
    G.AddTensor(DataType::DT_FP32, shape2, "sub_out_right");
    auto subOutRight = G.GetTensor("sub_out_right");
    subOutRight->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_VIEW, {"sub_out"}, {"sub_out_right"}, "View_Right");
    auto View_UR =  G.GetOp("View_Right");
    std::vector<int> offsetRight = {0, T};
    auto attrUR = std::make_shared<ViewOpAttribute>(offsetRight, MemoryType::MEM_UB);
    View_UR->SetOpAttribute(attrUR);

    // 左半边 [16, 8] + 右半边 [16, 8]
    G.AddTensor(DataType::DT_FP32, shape2, "add_out");
    G.AddOp(Opcode::OP_ADD, {"sub_out_right", "sub_out_left"}, {"add_out"}, "Add");
    auto addOut = G.GetTensor("add_out");
    addOut->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

    G.AddOp(Opcode::OP_ASSEMBLE, {"add_out"}, {"out"}, "Assemble_final");
    auto attrAssembleFinal = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int> {0, 0});
    auto Op = G.GetOp("Assemble_final");
    Op->SetOpAttribute(attrAssembleFinal);

    auto a = G.GetTensor("a");
    a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto b = G.GetTensor("b");
    b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto out = G.GetTensor("out");
    out->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    
    G.SetInCast({"a", "b"});
    G.SetOutCast({"out"});
    Function *function = G.GetFunction();
    // 确认构图完毕
    constexpr int singleViewOpmagic = 10017;
    constexpr int opNumBefore = 20;
    constexpr int viewNumBefore = 10;
    constexpr int assembleNumBefore = 5;
    auto countResultBefore = CountViewAssemble(*function);
    int viewNumCount = countResultBefore[0];
    int assembleNumCount = countResultBefore[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW before pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;
    for (auto &op : function->Operations()) {
        if (op.GetOpMagic() == singleViewOpmagic) {
            auto viewAttr = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
            auto offset = viewAttr->GetFromOffset();
            EXPECT_NE(accumulate(offset.begin(), offset.end(), 0), 0) << "OP_VIEW offset should be all zero";
        }
    }
    /*
    dump graph before Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    auto countResultAfter = CountViewAssemble(*function);
    viewNumCount = countResultAfter[0];
    assembleNumCount = countResultAfter[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations after pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW after pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE after pass";
    for (auto &op : function->Operations()) {
        if (op.GetOpMagic() == singleViewOpmagic) {
            auto viewAttr = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
            auto offset = viewAttr->GetFromOffset();
            EXPECT_EQ(accumulate(offset.begin(), offset.end(), 0), 0) << "OP_VIEW offset should be all zero";
        }
    }
}

TEST_F(SplitLargeFanoutTensorTest, PerfectlyMatch_Full) {
    int N = 2;
    int T = 8;
    std::vector<int> shape0{N * T, N * T};
    std::vector<int> shape1{T, T};
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, shape0, "a");
    G.AddTensor(DataType::DT_FP32, shape0, "b");
    G.AddTensor(DataType::DT_FP32, shape1, "out");
    G.AddTensor(DataType::DT_FP32, shape0, "sub_out");
    // [16, 16] --> View --> [8, 8] --> Sub --> [8, 8] --> Assemble --> [16, 16]
    TileExpandSub(G, N, T);
    auto subOut = G.GetTensor("sub_out");
    subOut->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

    // View 获取 [16, 16] 左上角 [8, 8]
    G.AddTensor(DataType::DT_FP32, shape1, "sub_out_upper_right");
    auto subOutUR = G.GetTensor("sub_out_upper_right");
    subOutUR->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_VIEW, {"sub_out"}, {"sub_out_upper_right"}, "View_Upper_Right");
    auto View_UR =  G.GetOp("View_Upper_Right");
    std::vector<int> offsetUR = {0, T};
    auto attrUR = std::make_shared<ViewOpAttribute>(offsetUR, MemoryType::MEM_UB);
    View_UR->SetOpAttribute(attrUR);

    // View 获取 [16, 16] 右上角 [8, 8]
    G.AddTensor(DataType::DT_FP32, shape1, "sub_out_lower_left");
    auto subOutLL = G.GetTensor("sub_out_lower_left");
    subOutLL->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_VIEW, {"sub_out"}, {"sub_out_lower_left"}, "View_Lower_Left");
    auto View_LL =  G.GetOp("View_Lower_Left");
    std::vector<int> offsetLL = {T, 0};
    auto attrLL = std::make_shared<ViewOpAttribute>(offsetLL, MemoryType::MEM_UB);
    View_LL->SetOpAttribute(attrLL);

    // 左上角 [8, 8] + 右上角 [8, 8] --> [8, 8]
    G.AddTensor(DataType::DT_FP32, shape1, "add_out");
    G.AddOp(Opcode::OP_ADD, {"sub_out_upper_right", "sub_out_lower_left"}, {"add_out"}, "Add");
    auto addOut = G.GetTensor("add_out");
    addOut->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

    G.AddOp(Opcode::OP_ASSEMBLE, {"add_out"}, {"out"}, "Assemble_final");
    auto attrAssembleFinal = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int> {0, 0});
    auto Op = G.GetOp("Assemble_final");
    Op->SetOpAttribute(attrAssembleFinal);

    auto a = G.GetTensor("a");
    a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto b = G.GetTensor("b");
    b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto out = G.GetTensor("out");
    out->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    
    G.SetInCast({"a", "b"});
    G.SetOutCast({"out"});
    Function *function = G.GetFunction();
    // 确认构图完毕
    constexpr int opNumBefore = 20;
    constexpr int viewNumBefore = 10;
    constexpr int assembleNumBefore = 5;
    auto countResultBefore = CountViewAssemble(*function);
    int viewNumCount = countResultBefore[0];
    int assembleNumCount = countResultBefore[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW before pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;

    constexpr int opNumAfter = 10;
    constexpr int viewNumAfter = 6;
    constexpr int assembleNumAfter = 1;
    auto countResultAfter = CountViewAssemble(*function);
    viewNumCount = countResultAfter[0];
    assembleNumCount = countResultAfter[1];
    EXPECT_EQ(function->Operations().size(), opNumAfter) << opNumAfter << " operations after pass";
    EXPECT_EQ(viewNumCount, viewNumAfter) << viewNumAfter << " OP_VIEW after pass";
    EXPECT_EQ(assembleNumCount, assembleNumAfter) << assembleNumAfter << " OP_ASSEMBLE after pass";
}

TEST_F(SplitLargeFanoutTensorTest, PerfectlyMatch_Full_V2) {
    int N = 2;
    int T = 8;
    std::vector<int> shape0{N * T, N * T};
    std::vector<int> shape1{T, T};
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, shape0, "a");
    G.AddTensor(DataType::DT_FP32, shape0, "b");
    G.AddTensor(DataType::DT_FP32, shape1, "out");
    G.AddTensor(DataType::DT_FP32, shape0, "out2");
    G.AddTensor(DataType::DT_FP32, shape0, "sub_out");
    // [16, 16] --> View --> [8, 8] --> Sub --> [8, 8] --> Assemble --> [16, 16]
    TileExpandSub(G, N, T);
    auto subOut = G.GetTensor("sub_out");
    subOut->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

    // [16, 16] --> exp --> [16, 16] --> Assemble --> OCAST
    G.AddTensor(DataType::DT_FP32, shape0, "expOut");
    auto expOut = G.GetTensor("expOut");
    expOut->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_EXP, {"sub_out"}, {"expOut"}, "Exp");
    G.AddOp(Opcode::OP_ASSEMBLE, {"expOut"}, {"out2"}, "Assemble_exp");
    auto attrAssembleExp = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int> {0, 0});
    auto assembleExp = G.GetOp("Assemble_exp");
    assembleExp->SetOpAttribute(attrAssembleExp);

    // View 获取 [16, 16] 左上角 [8, 8]
    G.AddTensor(DataType::DT_FP32, shape1, "sub_out_upper_right");
    auto subOutUR = G.GetTensor("sub_out_upper_right");
    subOutUR->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_VIEW, {"sub_out"}, {"sub_out_upper_right"}, "View_Upper_Right");
    auto View_UR =  G.GetOp("View_Upper_Right");
    std::vector<int> offsetUR = {0, T};
    auto attrUR = std::make_shared<ViewOpAttribute>(offsetUR, MemoryType::MEM_UB);
    View_UR->SetOpAttribute(attrUR);

    // View 获取 [16, 16] 右上角 [8, 8]
    G.AddTensor(DataType::DT_FP32, shape1, "sub_out_lower_left");
    auto subOutLL = G.GetTensor("sub_out_lower_left");
    subOutLL->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_VIEW, {"sub_out"}, {"sub_out_lower_left"}, "View_Lower_Left");
    auto View_LL =  G.GetOp("View_Lower_Left");
    std::vector<int> offsetLL = {T, 0};
    auto attrLL = std::make_shared<ViewOpAttribute>(offsetLL, MemoryType::MEM_UB);
    View_LL->SetOpAttribute(attrLL);

    // 左上角 [8, 8] + 右上角 [8, 8] --> [8, 8]
    G.AddTensor(DataType::DT_FP32, shape1, "add_out");
    G.AddOp(Opcode::OP_ADD, {"sub_out_upper_right", "sub_out_lower_left"}, {"add_out"}, "Add");
    auto addOut = G.GetTensor("add_out");
    addOut->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

    G.AddOp(Opcode::OP_ASSEMBLE, {"add_out"}, {"out"}, "Assemble_final");
    auto attrAssembleFinal = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int> {0, 0});
    auto Op = G.GetOp("Assemble_final");
    Op->SetOpAttribute(attrAssembleFinal);

    auto a = G.GetTensor("a");
    a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto b = G.GetTensor("b");
    b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto out = G.GetTensor("out");
    out->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto out2 = G.GetTensor("out2");
    out2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    
    G.SetInCast({"a", "b"});
    G.SetOutCast({"out", "out2"});
    Function *function = G.GetFunction();
    // 确认构图完毕
    constexpr int opNumBefore = 22;
    constexpr int viewNumBefore = 10;
    constexpr int assembleNumBefore = 6;
    auto countResultBefore = CountViewAssemble(*function);
    int viewNumCount = countResultBefore[0];
    int assembleNumCount = countResultBefore[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW before pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;

    auto countResultAfter = CountViewAssemble(*function);
    viewNumCount = countResultAfter[0];
    assembleNumCount = countResultAfter[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations after pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW after pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE after pass";
}

TEST_F(SplitLargeFanoutTensorTest, OneViewOneAssemble) {
    int NUM_32 = 32;
    int NUM_64 = 64;
    int NUM_96 = 96;
    int NUM_128 = 128;
    std::vector<int> shape0{NUM_128, NUM_64};
    std::vector<int> shape1{NUM_32, NUM_64};
    std::vector<int> shape2{NUM_64, NUM_64};
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, shape0, "a");
    auto a = G.GetTensor("a");
    a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(DataType::DT_FP32, shape2, "out");
    auto out = G.GetTensor("out");
    out->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    // a[128, 64] --> View(0, 0) --> a_ub[128, 64]
    G.AddTensor(DataType::DT_FP32, shape0, "a_ub");
    auto a_ub = G.GetTensor("a_ub");
    a_ub->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_VIEW, {"a"}, {"a_ub"}, "View_To_Ub");
    auto view2ub =  G.GetOp("View_To_Ub");
    std::vector<int> offset2ub = {0, 0};
    auto attr2ub = std::make_shared<ViewOpAttribute>(offset2ub, MemoryType::MEM_UB);
    view2ub->SetOpAttribute(attr2ub);

    /*
    a_ub[128, 64] --> View1(32, 0) --> tensor1[32, 64] --> Assemble1(0, 0) -->\
                  \--> View2(96, 0) --> tensor2[32, 64] --> Assemble2(32, 0) --> tensor3[64, 64] --> Exp --> tensor4 --> Assemble --> out
    */
    G.AddTensor(DataType::DT_FP32, shape1, "tensor1");
    auto tensor1 = G.GetTensor("tensor1");
    tensor1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_VIEW, {"a_ub"}, {"tensor1"}, "View_1");
    auto viewOp1 =  G.GetOp("View_1");
    std::vector<int> offset1 = {NUM_32, 0};
    auto attrView1 = std::make_shared<ViewOpAttribute>(offset1, MemoryType::MEM_UB);
    viewOp1->SetOpAttribute(attrView1);

    G.AddTensor(DataType::DT_FP32, shape1, "tensor2");
    auto tensor2 = G.GetTensor("tensor2");
    tensor2->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_VIEW, {"a_ub"}, {"tensor2"}, "View_2");
    auto viewOp2 =  G.GetOp("View_2");
    std::vector<int> offset2 = {NUM_96, 0};
    auto attrView2 = std::make_shared<ViewOpAttribute>(offset2, MemoryType::MEM_UB);
    viewOp2->SetOpAttribute(attrView2);

    G.AddTensor(DataType::DT_FP32, shape2, "tensor3");
    auto tensor3 = G.GetTensor("tensor3");
    tensor3->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_ASSEMBLE, {"tensor1"}, {"tensor3"}, "Assemble_1");
    auto attrAssemble1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int> {0, 0});
    auto assembleOp1 = G.GetOp("Assemble_1");
    assembleOp1->SetOpAttribute(attrAssemble1);
    G.AddOp(Opcode::OP_ASSEMBLE, {"tensor2"}, {"tensor3"}, "Assemble_2");
    auto attrAssemble2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int> {NUM_32, 0});
    auto assembleOp2 = G.GetOp("Assemble_2");
    assembleOp2->SetOpAttribute(attrAssemble2);

    G.AddTensor(DataType::DT_FP32, shape2, "tensor4");
    auto tensor4 = G.GetTensor("tensor4");
    tensor4->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_EXP, {"tensor3"}, {"tensor4"}, "Exp");
    G.AddOp(Opcode::OP_ASSEMBLE, {"tensor4"}, {"out"}, "Assemble_Final");
    auto attrAssembleFinal = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int> {0, 0});
    auto assembleFinal = G.GetOp("Assemble_Final");
    assembleFinal->SetOpAttribute(attrAssembleFinal);
    
    G.SetInCast({"a"});
    G.SetOutCast({"out"});
    Function *function = G.GetFunction();
    // 确认构图完毕
    constexpr int opNumBefore = 7;
    constexpr int viewNumBefore = 3;
    constexpr int assembleNumBefore = 3;
    auto countResultBefore = CountViewAssemble(*function);
    int viewNumCount = countResultBefore[0];
    int assembleNumCount = countResultBefore[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW before pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;

    constexpr int opNumAfter = 6;
    constexpr int viewNumAfter = 2;
    constexpr int assembleNumAfter = 3;
    auto countResultAfter = CountViewAssemble(*function);
    viewNumCount = countResultAfter[0];
    assembleNumCount = countResultAfter[1];
    EXPECT_EQ(function->Operations().size(), opNumAfter) << opNumAfter << " operations after pass";
    EXPECT_EQ(viewNumCount, viewNumAfter) << viewNumAfter << " OP_VIEW after pass";
    EXPECT_EQ(assembleNumCount, assembleNumAfter) << assembleNumAfter << " OP_ASSEMBLE after pass";
}

TEST_F(SplitLargeFanoutTensorTest, OneViewMultiAssemble) {
    int NUM_32 = 32;
    int NUM_64 = 64;
    int NUM_128 = 128;
    std::vector<int> shape0{NUM_32, NUM_64};
    std::vector<int> shape1{NUM_64, NUM_64};
    std::vector<int> shape2{NUM_32, NUM_128};
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, shape1, "a");
    auto a = G.GetTensor("a");
    a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(DataType::DT_FP32, shape1, "out1");
    auto out1 = G.GetTensor("out1");
    out1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(DataType::DT_FP32, shape2, "out2");
    auto out2 = G.GetTensor("out2");
    out2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    // a[64, 64] --> View(0, 0) --> a_ub[64, 64] --> Muls --> a_ub_new[64, 64]
    G.AddTensor(DataType::DT_FP32, shape1, "a_ub");
    auto a_ub = G.GetTensor("a_ub");
    a_ub->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_VIEW, {"a"}, {"a_ub"}, "View_To_Ub");
    auto view2ub =  G.GetOp("View_To_Ub");
    std::vector<int> offset2ub = {0, 0};
    auto attr2ub = std::make_shared<ViewOpAttribute>(offset2ub, MemoryType::MEM_UB);
    view2ub->SetOpAttribute(attr2ub);
    G.AddTensor(DataType::DT_FP32, shape1, "a_ub_new");
    auto a_ub_new = G.GetTensor("a_ub_new");
    a_ub_new->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_MULS, {"a_ub"}, {"a_ub_new"}, "Muls");

    /*
    a_ub_new[64, 64] --> View1(0, 0) --> tensor1[32, 64] --> Assemble1(0, 0) -->\
                  \--> View2(32, 0) --> tensor2[32, 64] --> Assemble2(32, 0) --> tensor3[64, 64] --> Exp --> tensor4 --> AssembleOut1 --> out1
    */
    G.AddTensor(DataType::DT_FP32, shape0, "tensor1");
    auto tensor1 = G.GetTensor("tensor1");
    tensor1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_VIEW, {"a_ub_new"}, {"tensor1"}, "View_1");
    auto viewOp1 =  G.GetOp("View_1");
    std::vector<int> offset1 = {0, 0};
    auto attrView1 = std::make_shared<ViewOpAttribute>(offset1, MemoryType::MEM_UB);
    viewOp1->SetOpAttribute(attrView1);

    G.AddTensor(DataType::DT_FP32, shape0, "tensor2");
    auto tensor2 = G.GetTensor("tensor2");
    tensor2->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_VIEW, {"a_ub_new"}, {"tensor2"}, "View_2");
    auto viewOp2 =  G.GetOp("View_2");
    std::vector<int> offset2 = {NUM_32, 0};
    auto attrView2 = std::make_shared<ViewOpAttribute>(offset2, MemoryType::MEM_UB);
    viewOp2->SetOpAttribute(attrView2);

    G.AddTensor(DataType::DT_FP32, shape1, "tensor3");
    auto tensor3 = G.GetTensor("tensor3");
    tensor3->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_ASSEMBLE, {"tensor1"}, {"tensor3"}, "Assemble_1");
    auto attrAssemble1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int> {0, 0});
    auto assembleOp1 = G.GetOp("Assemble_1");
    assembleOp1->SetOpAttribute(attrAssemble1);
    G.AddOp(Opcode::OP_ASSEMBLE, {"tensor2"}, {"tensor3"}, "Assemble_2");
    auto attrAssemble2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int> {NUM_32, 0});
    auto assembleOp2 = G.GetOp("Assemble_2");
    assembleOp2->SetOpAttribute(attrAssemble2);

    G.AddTensor(DataType::DT_FP32, shape1, "tensor4");
    auto tensor4 = G.GetTensor("tensor4");
    tensor4->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_EXP, {"tensor3"}, {"tensor4"}, "Exp");
    G.AddOp(Opcode::OP_ASSEMBLE, {"tensor4"}, {"out1"}, "Assemble_Out1");
    auto attrAssembleOut1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int> {0, 0});
    auto assembleOut1 = G.GetOp("Assemble_Out1");
    assembleOut1->SetOpAttribute(attrAssembleOut1);

    /*
    a_ub_new[64, 64] --> View1(0, 0) --> tensor1[32, 64] --> Assemble3(0, 0) -->\
                  \--> View2(32, 0) --> tensor2[32, 64] --> Assemble4(0, 64) --> tensor5[32, 128] --> Exp --> tensor6 --> AssembleOut2 --> out2
    */
    G.AddTensor(DataType::DT_FP32, shape2, "tensor5");
    auto tensor5 = G.GetTensor("tensor5");
    tensor5->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_ASSEMBLE, {"tensor1"}, {"tensor5"}, "Assemble_3");
    auto attrAssemble3 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int> {0, 0});
    auto assembleOp3 = G.GetOp("Assemble_3");
    assembleOp3->SetOpAttribute(attrAssemble3);
    G.AddOp(Opcode::OP_ASSEMBLE, {"tensor2"}, {"tensor5"}, "Assemble_4");
    auto attrAssemble4 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int> {0, NUM_64});
    auto assembleOp4 = G.GetOp("Assemble_4");
    assembleOp4->SetOpAttribute(attrAssemble4);

    G.AddTensor(DataType::DT_FP32, shape2, "tensor6");
    auto tensor6 = G.GetTensor("tensor6");
    tensor6->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_ABS, {"tensor5"}, {"tensor6"}, "Abs");
    G.AddOp(Opcode::OP_ASSEMBLE, {"tensor6"}, {"out2"}, "Assemble_Out2");
    auto attrAssembleOut2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int> {0, 0});
    auto assembleOut2 = G.GetOp("Assemble_Out2");
    assembleOut2->SetOpAttribute(attrAssembleOut2);
    
    G.SetInCast({"a"});
    G.SetOutCast({"out1", "out2"});
    Function *function = G.GetFunction();
    // 确认构图完毕
    constexpr int opNumBefore = 12;
    constexpr int viewNumBefore = 3;
    constexpr int assembleNumBefore = 6;
    auto countResultBefore = CountViewAssemble(*function);
    int viewNumCount = countResultBefore[0];
    int assembleNumCount = countResultBefore[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW before pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;

    auto countResultAfter = CountViewAssemble(*function);
    viewNumCount = countResultAfter[0];
    assembleNumCount = countResultAfter[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations after pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW after pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE after pass";
}

} // namespace tile_fwk
} // namespace npu