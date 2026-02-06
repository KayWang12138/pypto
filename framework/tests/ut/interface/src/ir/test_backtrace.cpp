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
 * \file test_backtrace.cpp
 * \brief Backtrace test using new IR error mechanism (pypto::ir::Error)
 *
 * This test demonstrates the optimized backtrace implementation in the new IR.
 * Key features:
 * - Python-style traceback header
 * - Stack reversal (most recent call last)
 * - File/Line format in Debug mode
 * - Source code display in Debug mode
 * - Graceful fallback in Release mode
 *
 * Usage:
 *   # Release mode (traditional format with Python-style header and stack reversal)
 *   python3 build_ci.py -u=BacktraceTest.NestedCallTest -d=11 -f cpp
 *
 *   # Debug mode (full optimization with File/Line and source code)
 *   python3 build_ci.py -u=BacktraceTest.NestedCallTest -d=11 -f cpp --build_type=Debug
 */

#include "gtest/gtest.h"
#include "core/error.h"
#include <iostream>

using namespace pypto::ir;

class BacktraceTest : public testing::Test {
protected:
    void SetUp() override {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "BACKTRACE OUTPUT TEST (New IR Error Mechanism)\n";
        std::cout << std::string(80, '=') << "\n";
    }

    void TearDown() override {
        std::cout << std::string(80, '=') << "\n\n";
    }
};

/**
 * Test 1: Simple Error Exception
 *
 * This test throws a simple error and prints the backtrace using new IR error mechanism.
 * Expected output:
 * - Release mode: Python-style header + stack reversal + traditional format
 * - Debug mode: Python-style header + stack reversal + File/Line format + source code
 */
TEST_F(BacktraceTest, SimpleErrorTest) {
    std::cout << "\nTest: Simple Error Exception\n";
    std::cout << std::string(40, '-') << "\n\n";

    try {
        // Throw an error with backtrace (new IR error mechanism)
        throw Error("Test error: Invalid operation");
    } catch (const Error& e) {
        // Print the complete error message with backtrace
        std::cout << "Caught Error Exception:\n";
        std::cout << e.GetFullMessage() << "\n";
    }
}

/**
 * Test 2: Nested Function Calls
 *
 * This test demonstrates backtrace with multiple function call levels.
 * The output will show the full call stack with proper formatting.
 */

// Helper functions to create nested call stack
void Level3Function() {
    throw Error("Error occurred at Level 3");
}

void Level2Function() {
    Level3Function();  // Call next level
}

void Level1Function() {
    Level2Function();  // Call next level
}

TEST_F(BacktraceTest, NestedCallTest) {
    std::cout << "\nTest: Nested Function Calls (3 levels)\n";
    std::cout << std::string(40, '-') << "\n\n";

    try {
        Level1Function();  // Start the call chain
    } catch (const Error& e) {
        std::cout << "Caught Error from Nested Calls:\n";
        std::cout << e.GetFullMessage() << "\n";
    }
}

/**
 * Test 3: Different Error Types
 *
 * This test demonstrates different error types from the new IR.
 */
TEST_F(BacktraceTest, ErrorTypesTest) {
    std::cout << "\nTest: Different Error Types\n";
    std::cout << std::string(40, '-') << "\n\n";

    // ValueError
    try {
        throw ValueError("Invalid value: expected positive number, got -5");
    } catch (const Error& e) {
        std::cout << "ValueError:\n" << e.GetFullMessage() << "\n\n";
    }

    // RuntimeError
    try {
        throw RuntimeError("Failed to allocate GPU memory");
    } catch (const Error& e) {
        std::cout << "RuntimeError:\n" << e.GetFullMessage() << "\n";
    }
}
