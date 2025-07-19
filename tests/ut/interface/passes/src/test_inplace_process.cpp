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
 * \file test_inplace_process.cpp
 * \brief Unit test for InplaceProcess pass.
 */

#include <vector>
#include "gtest/gtest.h"
#include "operation/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"
#include "passes/tile_graph_pass/inplace_process.h"
#include "computational_graph_builder.h"

using namespace npu::tile_fwk;

namespace npu {
namespace tile_fwk {

class InplaceProcessTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}

    bool IsInplace(const Operation &op) {
        auto input = op.GetIOperands().front();
        auto output = op.GetOOperands().front();
        return input->tensor->GetRawMagic() == output->tensor->GetRawMagic();
    }

    void CheckInplace(Function &function) {
        for (auto &op : function.Operations()) {
            if (op.GetOpcode() == Opcode::OP_VIEW) {
                EXPECT_EQ(IsInplace(op), true) << op.GetOpcodeStr() << " " << op.GetOpMagic() << " should be processed.";
            } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                EXPECT_EQ(IsInplace(op), true) << op.GetOpcodeStr() << " " << op.GetOpMagic() << " should be processed.";
            } else if (op.GetOpcode() == Opcode::OP_RESHAPE) {
                EXPECT_EQ(IsInplace(op), true) << op.GetOpcodeStr() << " " << op.GetOpMagic() << " should be processed.";
            }
        }
    }
};


TEST_F(InplaceProcessTest, CopyInDirectAssemble) {
    int NUM_16 = 16;
    int NUM_32 = 32;
    std::vector<int> shape0{NUM_16, NUM_16};
    std::vector<int> shape1{NUM_32, NUM_32};
    std::vector<int> shape2{NUM_32, NUM_16};
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, shape0, "a");
    auto a = G.GetTensor("a");
    a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(DataType::DT_FP32, shape0, "b");
    auto b = G.GetTensor("b");
    b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(DataType::DT_FP32, shape1, "c");
    auto c = G.GetTensor("c");
    c->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(DataType::DT_FP32, shape2, "out");
    auto out = G.GetTensor("out");
    out->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    // a [16, 16] --> Copy_IN (0, 0) --> [16, 16]
    G.AddTensor(DataType::DT_FP32, shape0, "a_ub");
    auto a_ub = G.GetTensor("a_ub");
    a_ub->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_COPY_IN, {"a"}, {"a_ub"}, "Copy_In_a");
    auto copyInA =  G.GetOp("Copy_In_a");
    std::vector<int> offsetA = {0, 0};
    auto attrCopyInA = std::make_shared<CopyOpAttribute>(
                OpImmediate::Specified(offsetA), MemoryType::MEM_UB,
                OpImmediate::Specified(a_ub->GetShape()), OpImmediate::Specified(a_ub->tensor->GetRawShape()));
    copyInA->SetOpAttribute(attrCopyInA);

    // b [16, 16] --> Copy_IN (0, 0) --> [16, 16]
    G.AddTensor(DataType::DT_FP32, shape0, "b_ub");
    auto b_ub = G.GetTensor("b_ub");
    b_ub->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_COPY_IN, {"b"}, {"b_ub"}, "Copy_In_b");
    auto copyInB =  G.GetOp("Copy_In_b");
    std::vector<int> offsetB = {0, 0};
    auto attrCopyInB = std::make_shared<CopyOpAttribute>(
                OpImmediate::Specified(offsetB), MemoryType::MEM_UB,
                OpImmediate::Specified(b_ub->GetShape()), OpImmediate::Specified(b_ub->tensor->GetRawShape()));
    copyInB->SetOpAttribute(attrCopyInB);

    // a[16, 16] + b[16, 16] --> add_out[16, 16]
    G.AddTensor(DataType::DT_FP32, shape0, "add_out");
    G.AddOp(Opcode::OP_ADD, {"a_ub", "b_ub"}, {"add_out"}, "Add");
    auto addOut = G.GetTensor("add_out");
    addOut->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

    // c[32, 32] --> Copy_IN (16, 0) --> c1[16, 16]
    G.AddTensor(DataType::DT_FP32, shape0, "c1_ub");
    auto c1_ub = G.GetTensor("c1_ub");
    c1_ub->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_COPY_IN, {"c"}, {"c1_ub"}, "Copy_In_C");
    auto copyInC =  G.GetOp("Copy_In_C");
    std::vector<int> offsetC = {0, NUM_16};
    auto attrCopyInC = std::make_shared<CopyOpAttribute>(
                OpImmediate::Specified(offsetC), MemoryType::MEM_UB,
                OpImmediate::Specified(c1_ub->GetShape()), OpImmediate::Specified(c1_ub->tensor->GetRawShape()));
    copyInC->SetOpAttribute(attrCopyInC);

    // c1[16, 16]  --> Assemble(16, 0) --> [32, 16]
    G.AddTensor(DataType::DT_FP32, shape2, "assembleOut");
    auto assembleOut = G.GetTensor("assembleOut");
    assembleOut->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_ASSEMBLE, {"c1_ub"}, {"assembleOut"}, "Assemble_1");
    auto attrAssemble1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int> {16, 0});
    auto assemble1 = G.GetOp("Assemble_1");
    assemble1->SetOpAttribute(attrAssemble1);

    // add_out[16, 16]  --> Assemble(0, 0) --> [32, 16]
    G.AddOp(Opcode::OP_ASSEMBLE, {"add_out"}, {"assembleOut"}, "Assemble_2");
    auto attrAssemble2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int> {0, 0});
    auto assemble2 = G.GetOp("Assemble_2");
    assemble2->SetOpAttribute(attrAssemble2);

    // [32, 16] --> Exp --> [32, 16] --> Copy Out
    G.AddTensor(DataType::DT_FP32, shape2, "exp_out");
    G.AddOp(Opcode::OP_EXP, {"assembleOut"}, {"exp_out"}, "Exp");
    auto expOut = G.GetTensor("exp_out");
    expOut->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    G.AddOp(Opcode::OP_COPY_OUT, {"exp_out"}, {"out"}, "Copy_Out");
    auto copyOut =  G.GetOp("Copy_Out");
    std::vector<int> offsetOut = {0, 0};
    auto attrCopyOut = std::make_shared<CopyOpAttribute>(MemoryType::MEM_UB,
            OpImmediate::Specified(offsetOut), OpImmediate::Specified(expOut->GetShape()),
            OpImmediate::Specified(expOut->tensor->GetRawShape()));
    copyOut->SetOpAttribute(attrCopyOut);

    G.SetInCast({"a", "b", "c"});
    G.SetOutCast({"out"});
    Function *function = G.GetFunction();

    // 确认构图完毕
    constexpr int opNumBefore = 8;
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    std::cout << "Build Graph Done." << std::endl;
    /*
    dump graph before Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::InplaceProcess inplaceProcess;
    inplaceProcess.PreCheck(*function);
    inplaceProcess.RunOnFunction(*function);
    inplaceProcess.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    CheckInplace(*function);
}
} // namespace tile_fwk
} // namespace npu