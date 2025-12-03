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
 * \file test_api.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include "interface/inner/tilefwk/tilefwk_api.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "tilefwk/data_type.h"
#include "operator/models/llama/llama_def.h"
#include "machine/runtime/runtime.h"
#include "interface/utils/file_utils.h"
#include "tilefwk/op_registry.h"

using namespace npu::tile_fwk;
class TestAstOpCompile : public testing::Test {
public:
    static void SetUpTestCase() {
        config::Reset();
    }

    static void TearDownTestCase() {}

    void SetUp() override {
        // config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
        Program::GetInstance().Reset();
    }

    void TearDown() override {}
};

void TestAddOp(uint64_t configKey) {
    std::unordered_set<uint64_t> regKeySet = {0};
    if (regKeySet.count(configKey) == 0) {
        return;
    }
    Tensor input_tensor0(DT_FP32, {64, 64}, "x0");
    Tensor input_tensor1(DT_FP32, {64, 64}, "x1");
    Tensor output_tensor(DT_FP32, {64, 64}, "y0");

    std::vector<Tensor> inputTensors = {input_tensor0, input_tensor1};
    std::vector<Tensor> outputTensors = {output_tensor};

    TileShape::Current().SetVecTile({8, 8});
    std::vector<std::reference_wrapper<Tensor>> opArgs;
    opArgs.insert(opArgs.end(), inputTensors.begin(), inputTensors.end());
    opArgs.insert(opArgs.end(), outputTensors.begin(), outputTensors.end());

    TileFwkBeginFunction("test_add", opArgs);
    {
        outputTensors.at(0) = Add(inputTensors.at(0), inputTensors.at(1));
    }
    TileFwkEndFunction(true);
}
REGISTER_OP(TestAdd).ImplFunc({{0, TestAddOp}});

TEST_F(TestAstOpCompile, test_compile_add_op) {
    bool ret = TileOpCompile("TestAdd", 0, "ast_op_test_add_0", "dump_path");
    EXPECT_EQ(ret, true);
    EXPECT_EQ(RealPath("./dump_path/ast_op_test_add_0.json").empty(), false);
    EXPECT_EQ(RealPath("./dump_path/ast_op_test_add_0.o").empty(), false);
}

void DynamicDD(uint64_t configKey) {
    std::unordered_set<uint64_t> regKeySet = {0, 1, 2};
    if (regKeySet.count(configKey) == 0) {
        return;
    }
    int s = 32;
    int n = 8;
    std::map<std::string, std::string> attrMap;
    attrMap.emplace("s", "32");
    Tensor input_tensor0(DT_FP32, {n * s, s}, "x0");
    Tensor input_tensor1(DT_FP32, {s, s}, "x1");
    Tensor input_tensor2(DT_INT32, {n, 1}, "x2");
    Tensor output_tensor(DT_FP32, {n * s, s}, "y0");
    std::vector<Tensor> inputTensors = {input_tensor0, input_tensor1, input_tensor2};
    std::vector<Tensor> outputTensors = {output_tensor};
    if (inputTensors.size() != 3 || outputTensors.size() != 1) {
        return;
    }
    auto attrIter = attrMap.find("s");
    if (attrIter == attrMap.end()) {
        return;
    }
    s = std::atoi(attrIter->second.c_str());
    if (s == 0) {
        return;
    }
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});

    Tensor &t0 = inputTensors.at(0);
    Tensor &t1 = inputTensors.at(1);
    Tensor &blockTable = inputTensors.at(2);
    Tensor &out = outputTensors.at(0);

    FUNCTION("main", {t0, t1, blockTable}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(t0, 0) / s)) {
            SymbolicScalar idx = GetTensorData(blockTable, {i, 0});
            Tensor t0s = View(t0, {s, s}, {idx * s, 0});

            Tensor qi(DT_FP32, {s, 2*s}, "qi");
            Assemble(t1, {0, 0}, qi);
            Assemble(t0s, {0, s}, qi);

            Tensor ki(DT_FP32, {s, 2*s}, "ki");
            Assemble(t0s, {0, 0}, ki);
            Assemble(t1, {0, s}, ki);

            Tensor t2 = Matrix::Matmul<false, true>(DataType::DT_FP32, qi, ki);
            // conat((t0s + t1, t1)) @ concat (t0s, t1)^T
            Assemble(t2, {idx * s, 0}, out);
        }
    }
}
REGISTER_OP(ViewAssemble).ImplFunc({{0, DynamicDD}});

TEST_F(TestAstOpCompile, test_dynamic_ViewAssemble) {
    bool ret = TileOpCompile("ViewAssemble", 0, "ast_op_dd_0", "dump_path");
    EXPECT_EQ(ret, true);
    EXPECT_EQ(RealPath("./dump_path/ast_op_dd_0.json").empty(), false);
    EXPECT_EQ(RealPath("./dump_path/ast_op_dd_0.o").empty(), false);
}
TEST_F(TestAstOpCompile, test_compile_fatbin) {
    bool ret = TileFwkCompileFatbin("ViewAssemble", "Ascend910B1", "./dump_path", "ast_op_add");
    EXPECT_EQ(ret, true);
}
