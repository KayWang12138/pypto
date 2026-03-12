/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <memory>
#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"
#include "codegen/glsl/codegen_glsl.h"
#include "interface/utils/id_gen.h"
#include "interface/configs/config_manager.h"
#include "interface/program/program.h"

using namespace npu::tile_fwk;

class TestCodegenGLSL : public ::testing::Test {
protected:
    void SetUp() override {
        config::Reset();
        Program::GetInstance().Reset();
        // Set default tile shape for mock testing
        TileShape::Current().SetVecTile({256});
    }
};

TEST_F(TestCodegenGLSL, TestBasicFlow) {
    // 1. Setup Context
    CodeGenCtx ctx("/tmp/include", "/tmp/glsl_out");
    CodeGenGLSL codegen(ctx);
    
    // 2. Create Dummy Function using Program
    Program::GetInstance().BeginFunction("test_glsl_func");
    Function* funcPtr = Program::GetInstance().GetCurrentFunction();
    ASSERT_NE(funcPtr, nullptr);
    Function& func = *funcPtr;
    
    // 3. Create Tensors
    // Input 0: Data (for Gather) - 5D Tensor
    // Shape: {2, 2, 1024, 2, 32}
    auto tensorIn0 = std::make_shared<LogicalTensor>(func, DataType::DT_FP32, Shape{2, 2, 1024, 2, 32}, TileOpFormat::TILEOP_ND, "in0", NodeType::INCAST);
    tensorIn0->magic = 100;
    tensorIn0->tensor->memoryId = static_cast<int>(MemoryType::MEM_DEVICE_DDR); // Set Memory Type
    tensorIn0->memoryrange.memId = static_cast<int>(MemoryType::MEM_DEVICE_DDR);
    tensorIn0->memoryrange.start = 0; tensorIn0->memoryrange.end = 262144;
    func.inCasts_.push_back(tensorIn0);
    
    // Input 1: Indices (for Gather) - 5D Tensor
    // Shape: {2, 2, 2, 2, 32} (Total 256 elements)
    auto tensorIn1 = std::make_shared<LogicalTensor>(func, DataType::DT_INT32, Shape{2, 2, 2, 2, 32}, TileOpFormat::TILEOP_ND, "in1", NodeType::INCAST);
    tensorIn1->magic = 101;
    tensorIn1->tensor->memoryId = static_cast<int>(MemoryType::MEM_DEVICE_DDR);
    tensorIn1->memoryrange.memId = static_cast<int>(MemoryType::MEM_DEVICE_DDR);
    tensorIn1->memoryrange.start = 262144; tensorIn1->memoryrange.end = 262400;
    func.inCasts_.push_back(tensorIn1);
    
    // Input 2: Updates (for Scatter)
    // Shape: {2, 2, 2, 2, 32}
    auto tensorIn2 = std::make_shared<LogicalTensor>(func, DataType::DT_FP32, Shape{2, 2, 2, 2, 32}, TileOpFormat::TILEOP_ND, "in2", NodeType::INCAST);
    tensorIn2->magic = 102;
    tensorIn2->tensor->memoryId = static_cast<int>(MemoryType::MEM_DEVICE_DDR);
    tensorIn2->memoryrange.memId = static_cast<int>(MemoryType::MEM_DEVICE_DDR);
    tensorIn2->memoryrange.start = 262400; tensorIn2->memoryrange.end = 262656;
    func.inCasts_.push_back(tensorIn2);

    // Intermediate: Gather Result
    auto tensorGatherOut = std::make_shared<LogicalTensor>(func, DataType::DT_FP32, Shape{2, 2, 2, 2, 32}, TileOpFormat::TILEOP_ND, "gather_out", NodeType::LOCAL);
    tensorGatherOut->magic = 200;
    tensorGatherOut->tensor->memoryId = static_cast<int>(MemoryType::MEM_UB); // Intermediate in UB
    tensorGatherOut->memoryrange.memId = static_cast<int>(MemoryType::MEM_UB);
    tensorGatherOut->memoryrange.start = 0; tensorGatherOut->memoryrange.end = 256;
    
    // Intermediate: Add Result
    auto tensorAddOut = std::make_shared<LogicalTensor>(func, DataType::DT_FP32, Shape{2, 2, 2, 2, 32}, TileOpFormat::TILEOP_ND, "add_out", NodeType::LOCAL);
    tensorAddOut->magic = 201;
    tensorAddOut->tensor->memoryId = static_cast<int>(MemoryType::MEM_UB);
    tensorAddOut->memoryrange.memId = static_cast<int>(MemoryType::MEM_UB);
    tensorAddOut->memoryrange.start = 256; tensorAddOut->memoryrange.end = 512;

    // Output 0: Result of Scatter
    // Shape: {2, 2, 1024, 2, 32}
    auto tensorOut0 = std::make_shared<LogicalTensor>(func, DataType::DT_FP32, Shape{2, 2, 1024, 2, 32}, TileOpFormat::TILEOP_ND, "out0", NodeType::OUTCAST);
    tensorOut0->magic = 300;
    tensorOut0->tensor->memoryId = static_cast<int>(MemoryType::MEM_DEVICE_DDR);
    tensorOut0->memoryrange.memId = static_cast<int>(MemoryType::MEM_DEVICE_DDR);
    tensorOut0->memoryrange.start = 262656; tensorOut0->memoryrange.end = 524800;
    func.outCasts_.push_back(tensorOut0);

    // 4. Create Operations
    // Op 1: Gather Elements (out = Gather(in0, in1))
    // Note: We use AddOperation from Function, which handles creation and adding to list.
    func.AddOperation(Opcode::OP_GATHER_ELEMENT, 
        std::vector<std::shared_ptr<LogicalTensor>>{tensorIn0, tensorIn1},
        std::vector<std::shared_ptr<LogicalTensor>>{tensorGatherOut}, false);

    // Op 2: Add (out = Add(gather_out, 1.0)) - Simulating Element-wise
    func.AddOperation(Opcode::OP_ADD,
        std::vector<std::shared_ptr<LogicalTensor>>{tensorGatherOut, tensorIn2},
        std::vector<std::shared_ptr<LogicalTensor>>{tensorAddOut}, false);

    // Op 3: Scatter Elements (out0[indices] = add_out)
    // Scatter inputs: [data(out0), indices(in1), updates(add_out)]
    func.AddOperation(Opcode::OP_SCATTER_ELEMENT,
        std::vector<std::shared_ptr<LogicalTensor>>{tensorOut0, tensorIn1, tensorAddOut},
        std::vector<std::shared_ptr<LogicalTensor>>{}, false); // Scatter usually doesn't produce local result in this context

    // 5. Generate Code
    std::cout << "Generating GLSL Shader..." << std::endl;
    codegen.GenCode(func, {});
    
    // 6. Verify Output
    // Check if file exists and contains expected strings
    std::string filename = "/tmp/glsl_out/shader.comp";
    std::ifstream infile(filename);
    ASSERT_TRUE(infile.is_open());
    std::stringstream buffer;
    buffer << infile.rdbuf();
    std::string content = buffer.str();
    
    // Basic checks
    EXPECT_NE(content.find("#version 450"), std::string::npos);
    EXPECT_NE(content.find("layout(std430, binding = 0) buffer In0"), std::string::npos);
    // Check for 5D decomposition
    EXPECT_NE(content.find("// Gather: Decompose global_idx"), std::string::npos);
    EXPECT_NE(content.find("uint d4 = tmp_idx % 32;"), std::string::npos);
    EXPECT_NE(content.find("uint d3 = tmp_idx % 2;"), std::string::npos);
    EXPECT_NE(content.find("uint d2 = tmp_idx % 2;"), std::string::npos);
    
    // Check Gather Logic (Gather Dim is assumed 0, so d0 is replaced by index)
    // d0 is replaced by uint(v_in_1)
    // in_offset = ... + uint(v_in_1) * ...
    EXPECT_NE(content.find("uint(v_in_1) * 131072"), std::string::npos); // Stride for dim 0 is 131072
    // Stride calculation verification:
    // Dim 4 (32) -> Stride 1
    // Dim 3 (2) -> Stride 32
    // Dim 2 (1024) -> Stride 64
    // Dim 1 (2) -> Stride 65536
    // Dim 0 (2) -> Stride 131072
    
    // Let's print content to verify
    std::cout << "Generated Shader Content:\n" << content << std::endl;
}
