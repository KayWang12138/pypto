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
 * \file test_backtrace_compare.cpp
 * \brief Simple backtrace comparison test - runs on both branches to compare output
 *
 * Run this test on both branches to see the difference:
 * - origin_cmp branch: Traditional C++ format
 * - modify_backtrack branch: Python-style format with source code display
 *
 * Usage:
 *   python3 build_ci.py -u=BacktraceCompareTest.SimpleErrorTest -d=11 -f cpp
 *   python3 build_ci.py -u=BacktraceCompareTest.NestedCallTest -d=11 -f cpp
 */

#include "gtest/gtest.h"
#include "tilefwk/error.h"
#include <iostream>

using namespace npu::tile_fwk;

class BacktraceCompareTest : public testing::Test {
protected:
    void SetUp() override {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "BACKTRACE OUTPUT TEST\n";
        std::cout << std::string(80, '=') << "\n";
    }

    void TearDown() override {
        std::cout << std::string(80, '=') << "\n\n";
    }
};

/**
 * Test 1: Simple Error Exception
 *
 * This test throws a simple error and prints the backtrace.
 * Compare the output format between branches:
 *
 * origin_cmp branch output:
 *   - Traditional format: libname(function+offset)
 *   - No file/line information
 *   - No source code display
 *
 * modify_backtrack branch output:
 *   - Python-style format: File "path", line X
 *   - Clear traceback header
 *   - Source code line display
 *   - Stack reversed (most recent last)
 */
TEST_F(BacktraceCompareTest, SimpleErrorTest) {
    std::cout << "\nTest: Simple Error Exception\n";
    std::cout << std::string(40, '-') << "\n\n";

    try {
        // Throw an error with backtrace
        throw Error(__func__, __FILE__, __LINE__,
                    "Test error: Invalid operation",
                    GetBacktrace(0, 32));
    } catch (const Error& e) {
        // Print the complete error message with backtrace
        std::cout << "Caught Error Exception:\n";
        std::cout << e.what() << "\n";
    }
}

/**
 * Test 2: Nested Function Calls
 *
 * This test demonstrates backtrace with multiple function call levels.
 * The output will show the full call stack.
 */

// Helper functions to create nested call stack
void Level3Function() {
    throw Error(__func__, __FILE__, __LINE__,
                "Error occurred at Level 3",
                GetBacktrace(0, 32));
}

void Level2Function() {
    Level3Function();  // Call next level
}

void Level1Function() {
    Level2Function();  // Call next level
}

TEST_F(BacktraceCompareTest, NestedCallTest) {
    std::cout << "\nTest: Nested Function Calls (3 levels)\n";
    std::cout << std::string(40, '-') << "\n\n";

    try {
        Level1Function();  // Start the call chain
    } catch (const Error& e) {
        std::cout << "Caught Error from Nested Calls:\n";
        std::cout << e.what() << "\n";
    }
}

/**
 * Test 3: Multiple Errors
 *
 * This test throws multiple errors to demonstrate consistent formatting.
 */
TEST_F(BacktraceCompareTest, MultipleErrorsTest) {
    std::cout << "\nTest: Multiple Errors\n";
    std::cout << std::string(40, '-') << "\n\n";

    // First error
    try {
        throw Error(__func__, __FILE__, __LINE__,
                    "First error: Division by zero",
                    GetBacktrace(0, 32));
    } catch (const Error& e) {
        std::cout << "Error #1:\n" << e.what() << "\n\n";
    }

    // Second error
    try {
        throw Error(__func__, __FILE__, __LINE__,
                    "Second error: Null pointer access",
                    GetBacktrace(0, 32));
    } catch (const Error& e) {
        std::cout << "Error #2:\n" << e.what() << "\n";
    }
}

/**
 * Test 4: ASSERT Macro Test
 *
 * This test uses the ASSERT macro to trigger backtrace.
 */
TEST_F(BacktraceCompareTest, AssertMacroTest) {
    std::cout << "\nTest: ASSERT Macro\n";
    std::cout << std::string(40, '-') << "\n\n";

    try {
        int x = 5;
        int y = 10;
        // This assertion will fail and throw an error
        ASSERT(x > y) << "Expected x > y, but got x=" << x << ", y=" << y;
    } catch (const Error& e) {
        std::cout << "Caught ASSERT Error:\n";
        std::cout << e.what() << "\n";
    }
}

/**
 * Test 5: Deep Call Stack
 *
 * This test creates a deeper call stack to show more frames.
 */

void DeepFunction5() {
    throw Error(__func__, __FILE__, __LINE__,
                "Error at deepest level (Level 5)",
                GetBacktrace(0, 32));
}

void DeepFunction4() { DeepFunction5(); }
void DeepFunction3() { DeepFunction4(); }
void DeepFunction2() { DeepFunction3(); }
void DeepFunction1() { DeepFunction2(); }

TEST_F(BacktraceCompareTest, DeepCallStackTest) {
    std::cout << "\nTest: Deep Call Stack (5 levels)\n";
    std::cout << std::string(40, '-') << "\n\n";

    try {
        DeepFunction1();
    } catch (const Error& e) {
        std::cout << "Caught Error from Deep Call Stack:\n";
        std::cout << e.what() << "\n";
    }
}
