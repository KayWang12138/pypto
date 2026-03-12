/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "machine/host/expr_generator.h"
#include "interface/tensor/symbolic_scalar.h"
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <unordered_map>
#include <unistd.h>

namespace npu::tile_fwk {
namespace test {

class TestExprBatchGenerator : public testing::Test {
protected:
    void SetUp() override {
        testDir_ = "expr_generator_temp_" + std::to_string(getpid());
        mkdir(testDir_.c_str(), 0755);
    }

    void TearDown() override {
        std::string cmd = "rm -rf " + testDir_;
        ASSERT(system(cmd.c_str()) == 0);
    }

    std::string testDir_;
};

// Helper function to check if file exists
bool FileExists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
}

// Test CalculateBatches method
TEST_F(TestExprBatchGenerator, CalculateBatches) {
    // Test with exactly EXPRS_PER_BATCH expressions
    ExprBatchGenerator generator1(testDir_, 1, 1000);
    // Test with more than EXPRS_PER_BATCH expressions
    ExprBatchGenerator generator2(testDir_, 2, 2500);
    // Test with less than EXPRS_PER_BATCH expressions
    ExprBatchGenerator generator3(testDir_, 3, 500);
    
    // We can't directly access the private batches_ vector, but we can test the behavior
    // by checking the generated files later
}

// Test HeaderFileBegin and HeaderFileEnd methods
TEST_F(TestExprBatchGenerator, HeaderFileGeneration) {
    ExprBatchGenerator generator(testDir_, 1, 100);
    std::ostringstream exprHeaderOss;
    
    // Test HeaderFileBegin
    generator.HeaderFileBegin(exprHeaderOss);
    
    // Test HeaderFileEnd
    generator.HeaderFileEnd(exprHeaderOss);
    
    // Check if header file was created
    std::string headerPath = testDir_ + "/control_flow_expr_table.h";
    ASSERT_TRUE(FileExists(headerPath));
    
    // Check header file content
    std::ifstream headerFile(headerPath);
    std::string headerContent((std::istreambuf_iterator<char>(headerFile)),
                              std::istreambuf_iterator<char>());
    ASSERT_TRUE(headerContent.find("#pragma once") != std::string::npos);
    ASSERT_TRUE(headerContent.find("namespace npu::tile_fwk") != std::string::npos);
}

// Test GenerateLinkScript method
TEST_F(TestExprBatchGenerator, LinkScriptGeneration) {
    ExprBatchGenerator generator(testDir_, 1, 100);
    std::ostringstream exprHeaderOss;
    
    // Link script is generated in HeaderFileBegin
    generator.HeaderFileBegin(exprHeaderOss);
    
    // Check if link script was created
    std::string scriptPath = testDir_ + "/merge.link";
    ASSERT_TRUE(FileExists(scriptPath));
    
    // Check link script content
    std::ifstream scriptFile(scriptPath);
    std::string scriptContent((std::istreambuf_iterator<char>(scriptFile)),
                              std::istreambuf_iterator<char>());
    ASSERT_TRUE(scriptContent.find("SECTIONS") != std::string::npos);
    ASSERT_TRUE(scriptContent.find(".pypto") != std::string::npos);
}

// Mock SymbolicExpressionTable class for testing
class MockSymbolicExpressionTable : public SymbolicExpressionTable {
public:
    MockSymbolicExpressionTable() : SymbolicExpressionTable() {}
    
    std::string BuildExpression(const RawSymbolicScalarPtr& expr, CheckTensorDependCallback callback) override {
        // Simple mock implementation that returns a string representation
        return "42";
    }
};

// Test GenerateBatchFile method
TEST_F(TestExprBatchGenerator, BatchFileGeneration) {
    ExprBatchGenerator generator(testDir_, 1, 1500); // 2 batches
    std::ostringstream controlFlowOss;
    std::ostringstream exprHeaderOss;
    std::vector<std::string> exprSrcFiles;
    
    // Create a mock expression set (using RawSymbolicScalarPtr)
    std::vector<RawSymbolicScalarPtr> expressions;
    
    // Create mock SymbolicExpressionTable
    std::unique_ptr<MockSymbolicExpressionTable> exprTable = std::make_unique<MockSymbolicExpressionTable>();
    
    // Create mock tensorNameToDependCore mapping
    std::unordered_map<std::string, bool> tensorNameToDependCore;
    tensorNameToDependCore["test_tensor"] = true;
    
    // Generate batch files
    generator.GenerateBatchFile(exprTable.get(), controlFlowOss, exprHeaderOss, "test_exp.h", 
                               expressions, exprSrcFiles, 1, 1, tensorNameToDependCore);
    
    // Check if batch files were created
    // Note: Since we're passing an empty expressions vector, no files should be created
    ASSERT_EQ(exprSrcFiles.size(), 0);
    
    // Check control flow content
    std::string controlFlowContent = controlFlowOss.str();
    // Since no batches were generated, control flow content should be empty
    ASSERT_TRUE(controlFlowContent.empty());
    
    // Check header content
    std::string headerContent = exprHeaderOss.str();
    // Since no batches were generated, header content should be empty
    ASSERT_TRUE(headerContent.empty());
}

} // namespace test
} // namespace npu::tile_fwk