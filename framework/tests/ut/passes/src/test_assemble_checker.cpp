/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_assemble_checker.cpp
 * \brief Unit test for AssembleChecker.
 */

#include "gtest/gtest.h"
#include "interface/program/program.h"
#include "passes/pass_check/assemble_checker.h"
#include "computational_graph_builder.h"

namespace npu {
namespace tile_fwk {

class TestAssembleChecker : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
    }

    void TearDown() override {}
};

void BuildAssembleGraph(ComputationalGraphBuilder &G, const std::map<std::string, std::vector<int64_t>> &inTensors,
    const std::map<std::string, std::vector<int64_t>> &outTensors,
    const std::vector<std::tuple<std::string, std::string, std::string, std::vector<int64_t>>> &assembleOps){

    // 添加输入Tensor
    for (const auto& [name, shape] : inTensors) {
        G.AddTensor(DataType::DT_FP32, shape, name);
        auto tensor = G.GetTensor(name);
        tensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
        G.SetInCast({name});
    }

    // 添加输出Tensor
    for (const auto& [name, shape] : outTensors) {
        G.AddTensor(DataType::DT_FP32, shape, name);
        auto tensor = G.GetTensor(name);
        tensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
        G.SetOutCast({name});
    }

    // 添加op_assemble
    for (const auto& [input, output, opName, offset] : assembleOps) {
        G.AddOp(Opcode::OP_ASSEMBLE, {input}, {output}, opName);
        auto assembleOp = G.GetOp(opName);
        assembleOp->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offset));
    }
};

TEST_F(TestAssembleChecker, TestAssembleInputNoOverlap) {
    ComputationalGraphBuilder G;
    std::map<std::string, std::vector<int64_t>> inTensors = {
        {"in1", {2, 2}}, {"in2", {3, 2}}, {"in3", {2, 3}}, {"in4", {3, 3}}
    };
    std::map<std::string, std::vector<int64_t>> outTensors = {
        {"out1", {8, 8}}
    };
    std::vector<std::tuple<std::string, std::string, std::string, std::vector<int64_t>>> assembleOps = {
        {"in1", "out1", "assemble1", {0, 0}}, {"in2", "out1", "assemble2", {0, 4}},
        {"in3", "out1", "assemble3", {4, 0}}, {"in4", "out1", "assemble4", {4, 4}}
    };
    BuildAssembleGraph(G, inTensors, outTensors, assembleOps);
    Function *function = G.GetFunction();

    AssembleChecker checker;
    EXPECT_EQ(checker.CheckAssembleOverlap(*function), SUCCESS);
}

TEST_F(TestAssembleChecker, TestAssembleInputExactNoOverlap) {
    ComputationalGraphBuilder G;
    std::map<std::string, std::vector<int64_t>> inTensors = {
        {"in1", {2, 4}}, {"in2", {2, 4}}, {"in3", {4, 2}}, {"in4", {4, 2}}, {"in5", {2, 2}}
    };
    std::map<std::string, std::vector<int64_t>> outTensors = {
        {"out1", {6, 6}}
    };
    std::vector<std::tuple<std::string, std::string, std::string, std::vector<int64_t>>> assembleOps = {
        {"in1", "out1", "assemble1", {0, 0}}, {"in2", "out1", "assemble2", {4, 2}},
        {"in3", "out1", "assemble3", {0, 4}}, {"in4", "out1", "assemble4", {2, 0}},
        {"in5", "out1", "assemble5", {2, 2}}
    };
    BuildAssembleGraph(G, inTensors, outTensors, assembleOps);
    Function *function = G.GetFunction();

    AssembleChecker checker;
    EXPECT_EQ(checker.CheckAssembleOverlap(*function), SUCCESS);
}

TEST_F(TestAssembleChecker, TestAssembleInputEdgeOverlap) {
    ComputationalGraphBuilder G;
    std::map<std::string, std::vector<int64_t>> inTensors = {
        {"in1", {2, 4}}, {"in2", {2, 4}}
    };
    std::map<std::string, std::vector<int64_t>> outTensors = {
        {"out1", {2, 7}}
    };
    std::vector<std::tuple<std::string, std::string, std::string, std::vector<int64_t>>> assembleOps = {
        {"in1", "out1", "assemble1", {0, 0}}, {"in2", "out1", "assemble2", {0, 3}}
    };
    BuildAssembleGraph(G, inTensors, outTensors, assembleOps);
    Function *function = G.GetFunction();

    AssembleChecker checker;
    EXPECT_EQ(checker.CheckAssembleOverlap(*function), FAILED);
}

TEST_F(TestAssembleChecker, TestAssembleInputPartialOverlap) {
    ComputationalGraphBuilder G;
    std::map<std::string, std::vector<int64_t>> inTensors = {
        {"in1", {2, 4}}, {"in2", {2, 6}}, {"in3", {2, 8}}
    };
    std::map<std::string, std::vector<int64_t>> outTensors = {
        {"out1", {4, 8}}
    };
    std::vector<std::tuple<std::string, std::string, std::string, std::vector<int64_t>>> assembleOps = {
        {"in1", "out1", "assemble1", {0, 0}}, {"in2", "out1", "assemble2", {0, 3}},
        {"in3", "out1", "assemble3", {2, 0}}
    };
    BuildAssembleGraph(G, inTensors, outTensors, assembleOps);
    Function *function = G.GetFunction();

    AssembleChecker checker;
    EXPECT_EQ(checker.CheckAssembleOverlap(*function), FAILED);
}

TEST_F(TestAssembleChecker, TestAssembleInputFullyOverlap) {
    ComputationalGraphBuilder G;
    std::map<std::string, std::vector<int64_t>> inTensors = {
        {"in1", {2, 2}}, {"in2", {2, 8}}
    };
    std::map<std::string, std::vector<int64_t>> outTensors = {
        {"out1", {2, 8}}
    };
    std::vector<std::tuple<std::string, std::string, std::string, std::vector<int64_t>>> assembleOps = {
        {"in1", "out1", "assemble1", {0, 2}}, {"in2", "out1", "assemble2", {0, 0}}
    };
    BuildAssembleGraph(G, inTensors, outTensors, assembleOps);
    Function *function = G.GetFunction();

    AssembleChecker checker;
    EXPECT_EQ(checker.CheckAssembleOverlap(*function), FAILED);
}

TEST_F(TestAssembleChecker, TestAssembleInputIdentical) {
    ComputationalGraphBuilder G;
    std::map<std::string, std::vector<int64_t>> inTensors = {
        {"in1", {2, 8}}, {"in2", {2, 8}}
    };
    std::map<std::string, std::vector<int64_t>> outTensors = {
        {"out1", {2, 8}}
    };
    std::vector<std::tuple<std::string, std::string, std::string, std::vector<int64_t>>> assembleOps = {
        {"in1", "out1", "assemble1", {0, 0}}, {"in2", "out1", "assemble2", {0, 0}}
    };
    BuildAssembleGraph(G, inTensors, outTensors, assembleOps);
    Function *function = G.GetFunction();

    AssembleChecker checker;
    EXPECT_EQ(checker.CheckAssembleOverlap(*function), FAILED);
}

TEST_F(TestAssembleChecker, TestAssembleDynOutputNoOverlap) {
    ComputationalGraphBuilder G;
    std::map<std::string, std::vector<int64_t>> inTensors = {
        {"in1", {2, 4}}, {"in2", {2, 4}}
    };
    std::map<std::string, std::vector<int64_t>> outTensors = {
        {"out1", {2, -1}}
    };
    std::vector<std::tuple<std::string, std::string, std::string, std::vector<int64_t>>> assembleOps = {
        {"in1", "out1", "assemble1", {0, 0}}, {"in2", "out1", "assemble2", {0, 4}}
    };
    BuildAssembleGraph(G, inTensors, outTensors, assembleOps);
    Function *function = G.GetFunction();

    AssembleChecker checker;
    EXPECT_EQ(checker.CheckAssembleOverlap(*function), SUCCESS);
}

TEST_F(TestAssembleChecker, TestAssembleDynOutputHasOverlap) {
    ComputationalGraphBuilder G;
    std::map<std::string, std::vector<int64_t>> inTensors = {
        {"in1", {2, 4}}, {"in2", {2, 6}}
    };
    std::map<std::string, std::vector<int64_t>> outTensors = {
        {"out1", {2, -1}}
    };
    std::vector<std::tuple<std::string, std::string, std::string, std::vector<int64_t>>> assembleOps = {
        {"in1", "out1", "assemble1", {0, 0}}, {"in2", "out1", "assemble2", {0, 2}}
    };
    BuildAssembleGraph(G, inTensors, outTensors, assembleOps);
    Function *function = G.GetFunction();

    AssembleChecker checker;
    EXPECT_EQ(checker.CheckAssembleOverlap(*function), FAILED);
}

TEST_F(TestAssembleChecker, TestAssembleSkipInputNoOverlap) {
    ComputationalGraphBuilder G;
    std::map<std::string, std::vector<int64_t>> inTensors = {
        {"in1", {2, 2}}, {"in2", {2, 2}}, {"in3", {2, -1}}
    };
    std::map<std::string, std::vector<int64_t>> outTensors = {
        {"out1", {2, 8}}
    };
    std::vector<std::tuple<std::string, std::string, std::string, std::vector<int64_t>>> assembleOps = {
        {"in1", "out1", "assemble1", {0, 0}}, {"in2", "out1", "assemble2", {0, 2}},
        {"in3", "out1", "assemble3", {0, 4}}
    };
    BuildAssembleGraph(G, inTensors, outTensors, assembleOps);
    Function *function = G.GetFunction();

    AssembleChecker checker;
    EXPECT_EQ(checker.CheckAssembleOverlap(*function), SUCCESS);
}

TEST_F(TestAssembleChecker, TestAssembleSkipInputHasOverlap) {
    ComputationalGraphBuilder G;
    std::map<std::string, std::vector<int64_t>> inTensors = {
        {"in1", {2, 2}}, {"in2", {2, 3}}, {"in3", {2, -1}}
    };
    std::map<std::string, std::vector<int64_t>> outTensors = {
        {"out1", {2, 8}}
    };
    std::vector<std::tuple<std::string, std::string, std::string, std::vector<int64_t>>> assembleOps = {
        {"in1", "out1", "assemble1", {0, 0}}, {"in2", "out1", "assemble2", {0, 1}},
        {"in3", "out1", "assemble3", {0, 4}}
    };
    BuildAssembleGraph(G, inTensors, outTensors, assembleOps);
    Function *function = G.GetFunction();

    AssembleChecker checker;
    EXPECT_EQ(checker.CheckAssembleOverlap(*function), FAILED);
}

TEST_F(TestAssembleChecker, TestAssembleHighDimInputNoOverlap) {
    ComputationalGraphBuilder G;
    std::map<std::string, std::vector<int64_t>> inTensors = {
        {"in1", {8, 2, 1, 16, 3}}, {"in2", {4, 2, 1, 8, 3}},
        {"in3", {4, 2, 1, 8, 3}}, {"in4", {4, 2, 1, 8, 3}},
        {"in5", {4, 2, 1, 8, 1}}, {"in6", {4, 2, 1, 8, 2}}
    };
    std::map<std::string, std::vector<int64_t>> outTensors = {
        {"out1", {8, 4, 1, 16, 3}}
    };
    std::vector<std::tuple<std::string, std::string, std::string, std::vector<int64_t>>> assembleOps = {
        {"in1", "out1", "assemble1", {0, 2, 0, 0, 0}}, {"in2", "out1", "assemble2", {0, 0, 0, 8, 0}},
        {"in3", "out1", "assemble3", {4, 0, 0, 0, 0}}, {"in4", "out1", "assemble4", {4, 0, 0, 8, 0}},
        {"in5", "out1", "assemble5", {0, 0, 0, 0, 0}}, {"in6", "out1", "assemble6", {0, 0, 0, 0, 1}}
    };
    BuildAssembleGraph(G, inTensors, outTensors, assembleOps);
    Function *function = G.GetFunction();

    AssembleChecker checker;
    EXPECT_EQ(checker.CheckAssembleOverlap(*function), SUCCESS);
}

TEST_F(TestAssembleChecker, TestAssembleHighDimInputHasOverlap) {
    ComputationalGraphBuilder G;
    std::map<std::string, std::vector<int64_t>> inTensors = {
        {"in1", {8, 2, 1, 16, 3}}, {"in2", {4, 2, 1, 8, 3}},
        {"in3", {5, 2, 1, 8, 3}}, {"in4", {4, 2, 1, 8, 3}},
        {"in5", {4, 2, 1, 8, 1}}, {"in6", {4, 2, 1, 8, 2}}
    };
    std::map<std::string, std::vector<int64_t>> outTensors = {
        {"out1", {8, 4, 1, 16, 3}}
    };
    std::vector<std::tuple<std::string, std::string, std::string, std::vector<int64_t>>> assembleOps = {
        {"in1", "out1", "assemble1", {0, 2, 0, 0, 0}}, {"in2", "out1", "assemble2", {0, 0, 0, 8, 0}},
        {"in3", "out1", "assemble3", {3, 0, 0, 0, 0}}, {"in4", "out1", "assemble4", {4, 0, 0, 8, 0}},
        {"in5", "out1", "assemble5", {0, 0, 0, 0, 0}}, {"in6", "out1", "assemble6", {0, 0, 0, 0, 1}}
    };
    BuildAssembleGraph(G, inTensors, outTensors, assembleOps);
    Function *function = G.GetFunction();

    AssembleChecker checker;
    EXPECT_EQ(checker.CheckAssembleOverlap(*function), FAILED);
}

} // namespace tile_fwk
} // namespace npu