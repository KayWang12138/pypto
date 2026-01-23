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
 * \file test_error.cpp
 * \brief Unit tests for error handling and exception classes
 */

#include "gtest/gtest.h"

#include <string>

#include "core/error.h"

namespace pypto {

TEST(CoreErrorTest, TestValueError) {
    // Test ValueError basic functionality
    try {
        throw ValueError("Invalid value provided");
        FAIL() << "Expected ValueError to be thrown";
    } catch (const ValueError& e) {
        std::string msg(e.what());
        ASSERT_TRUE(msg.find("Invalid value provided") != std::string::npos);
    } catch (...) {
        FAIL() << "Expected ValueError but caught different exception";
    }
}

TEST(CoreErrorTest, TestTypeError) {
    // Test TypeError basic functionality
    try {
        throw TypeError("Type mismatch detected");
        FAIL() << "Expected TypeError to be thrown";
    } catch (const TypeError& e) {
        std::string msg(e.what());
        ASSERT_TRUE(msg.find("Type mismatch detected") != std::string::npos);
    } catch (...) {
        FAIL() << "Expected TypeError but caught different exception";
    }
}

TEST(CoreErrorTest, TestRuntimeError) {
    // Test RuntimeError basic functionality
    try {
        throw RuntimeError("Runtime error occurred");
        FAIL() << "Expected RuntimeError to be thrown";
    } catch (const RuntimeError& e) {
        std::string msg(e.what());
        ASSERT_TRUE(msg.find("Runtime error occurred") != std::string::npos);
    } catch (...) {
        FAIL() << "Expected RuntimeError but caught different exception";
    }
}

TEST(CoreErrorTest, TestNotImplementedError) {
    // Test NotImplementedError basic functionality
    try {
        throw NotImplementedError("Feature not implemented");
        FAIL() << "Expected NotImplementedError to be thrown";
    } catch (const NotImplementedError& e) {
        std::string msg(e.what());
        ASSERT_TRUE(msg.find("Feature not implemented") != std::string::npos);
    } catch (...) {
        FAIL() << "Expected NotImplementedError but caught different exception";
    }
}

TEST(CoreErrorTest, TestIndexError) {
    // Test IndexError basic functionality
    try {
        throw IndexError("Index out of bounds");
        FAIL() << "Expected IndexError to be thrown";
    } catch (const IndexError& e) {
        std::string msg(e.what());
        ASSERT_TRUE(msg.find("Index out of bounds") != std::string::npos);
    } catch (...) {
        FAIL() << "Expected IndexError but caught different exception";
    }
}

TEST(CoreErrorTest, TestAssertionError) {
    // Test AssertionError basic functionality
    try {
        throw AssertionError("Assertion failed");
        FAIL() << "Expected AssertionError to be thrown";
    } catch (const AssertionError& e) {
        std::string msg(e.what());
        ASSERT_TRUE(msg.find("Assertion failed") != std::string::npos);
    } catch (...) {
        FAIL() << "Expected AssertionError but caught different exception";
    }
}

TEST(CoreErrorTest, TestInternalError) {
    // Test InternalError basic functionality
    try {
        throw InternalError("Internal system error");
        FAIL() << "Expected InternalError to be thrown";
    } catch (const InternalError& e) {
        std::string msg(e.what());
        ASSERT_TRUE(msg.find("Internal system error") != std::string::npos);
    } catch (...) {
        FAIL() << "Expected InternalError but caught different exception";
    }
}

TEST(CoreErrorTest, TestErrorInheritance) {
    // Test that all errors inherit from Error base class
    try {
        throw ValueError("Test inheritance");
        FAIL() << "Expected exception to be thrown";
    } catch (const Error& e) {
        // Should catch ValueError as Error
        std::string msg(e.what());
        ASSERT_TRUE(msg.find("Test inheritance") != std::string::npos);
    } catch (...) {
        FAIL() << "Expected to catch as Error base class";
    }

    try {
        throw TypeError("Test inheritance");
        FAIL() << "Expected exception to be thrown";
    } catch (const Error& e) {
        // Should catch TypeError as Error
        ASSERT_TRUE(std::string(e.what()).find("Test inheritance") != std::string::npos);
    } catch (...) {
        FAIL() << "Expected to catch as Error base class";
    }
}

TEST(CoreErrorTest, TestGetFullMessage) {
    // Test GetFullMessage() includes error message
    try {
        throw ValueError("Test full message");
    } catch (const Error& e) {
        std::string full_msg = e.GetFullMessage();
        // Should contain the error message
        ASSERT_TRUE(full_msg.find("Test full message") != std::string::npos);
        // May contain stack trace if libbacktrace is enabled
        // We don't strictly require it since it depends on build configuration
    }
}

TEST(CoreErrorTest, TestMultipleExceptionTypes) {
    // Test that we can distinguish between different exception types
    bool caught_value_error = false;
    bool caught_type_error = false;
    bool caught_runtime_error = false;

    try {
        throw ValueError("Test");
    } catch (const ValueError&) {
        caught_value_error = true;
    } catch (const TypeError&) {
        FAIL() << "Should not catch as TypeError";
    } catch (const RuntimeError&) {
        FAIL() << "Should not catch as RuntimeError";
    }

    try {
        throw TypeError("Test");
    } catch (const ValueError&) {
        FAIL() << "Should not catch as ValueError";
    } catch (const TypeError&) {
        caught_type_error = true;
    } catch (const RuntimeError&) {
        FAIL() << "Should not catch as RuntimeError";
    }

    try {
        throw RuntimeError("Test");
    } catch (const ValueError&) {
        FAIL() << "Should not catch as ValueError";
    } catch (const TypeError&) {
        FAIL() << "Should not catch as TypeError";
    } catch (const RuntimeError&) {
        caught_runtime_error = true;
    }

    ASSERT_TRUE(caught_value_error);
    ASSERT_TRUE(caught_type_error);
    ASSERT_TRUE(caught_runtime_error);
}

TEST(CoreErrorTest, TestEmptyMessage) {
    // Test that errors can be created with empty messages
    try {
        throw ValueError("");
    } catch (const ValueError& e) {
        // Should not crash, message can be empty
        ASSERT_NE(e.what(), nullptr);
    }
}

}  // namespace pypto
