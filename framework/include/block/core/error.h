/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * @file error.h
 * @brief Core error handling framework
 *
 * This header provides a comprehensive error handling system with a base Error class
 * and several specialized error types that mirror Python's exception hierarchy.
 */

#ifndef PYPTO_CORE_ERROR_H_
#define PYPTO_CORE_ERROR_H_

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/common.h"
#include "ir/span.h"  // For Span in Diagnostic

namespace pypto {

/**
 * @brief Base exception class
 *
 * This is the fundamental exception type in PyPTO's error hierarchy.
 * All PyPTO exceptions should inherit from this class.
 */
class Error : public std::runtime_error {
 public:
  explicit Error(const std::string& message) : std::runtime_error(message) {}
};

/**
 * @brief Exception raised when a function receives an argument of correct type but inappropriate value
 *
 * Use this exception when:
 * - An argument value is outside the valid range
 * - A string argument doesn't match an expected format
 * - A numeric value violates domain constraints
 *
 * Example: ValueError("Dimension size must be positive, got -5")
 */
class ValueError : public Error {
 public:
  PYPTO_ALWAYS_INLINE explicit ValueError(const std::string& message) : Error(message) {}
};

/**
 * @brief Exception raised when an operation is applied to an object of inappropriate type
 *
 * Use this exception when:
 * - An argument has the wrong type
 * - A type conversion is invalid
 * - An operation doesn't support the given type combination
 *
 * Example: TypeError("Expected tensor but got scalar value")
 */
class TypeError : public Error {
 public:
  PYPTO_ALWAYS_INLINE explicit TypeError(const std::string& message) : Error(message) {}
};

/**
 * @brief Exception raised when an error occurs during program execution
 *
 * Use this exception for general runtime failures that don't fit into
 * more specific categories, such as:
 * - Resource allocation failures
 * - Invalid program state
 * - External system errors
 *
 * Example: RuntimeError("Failed to allocate GPU memory")
 */
class RuntimeError : public Error {
 public:
  PYPTO_ALWAYS_INLINE explicit RuntimeError(const std::string& message) : Error(message) {}
};

/**
 * @brief Exception raised when a feature or method is not yet implemented
 *
 * Use this exception for:
 * - Placeholder implementations
 * - Abstract methods that must be overridden
 * - Features planned but not yet developed
 *
 * Example: NotImplementedError("GPU backend not yet supported for this operation")
 */
class NotImplementedError : public Error {
 public:
  PYPTO_ALWAYS_INLINE explicit NotImplementedError(const std::string& message) : Error(message) {}
};

/**
 * @brief Exception raised when a sequence index is out of range
 *
 * Use this exception when:
 * - Array or vector access is out of bounds
 * - Tensor dimension index is invalid
 * - Attempting to access a non-existent element
 *
 * Example: IndexError("Index 10 is out of bounds for dimension of size 5")
 */
class IndexError : public Error {
 public:
  PYPTO_ALWAYS_INLINE explicit IndexError(const std::string& message) : Error(message) {}
};

/**
 * @brief Exception raised when an assertion fails
 *
 * Use this exception when:
 * - An internal consistency check fails
 * - A precondition or postcondition is violated
 * - A debug assertion fails in production code
 *
 * Example: AssertionError("Expected x > 0, but got x = -5")
 */
class AssertionError : public Error {
 public:
  PYPTO_ALWAYS_INLINE explicit AssertionError(const std::string& message) : Error(message) {}
};

/**
 * @brief Exception raised when an internal system error occurs
 *
 * Use this exception when:
 * - An unexpected internal state is encountered
 * - A system invariant is violated
 * - An error occurs that indicates a bug in the system itself
 * - Internal data structures become corrupted
 *
 * This exception type helps distinguish between user errors (ValueError, TypeError, etc.)
 * and internal system failures that should not normally occur in production.
 *
 * Example: InternalError("Corrupted tensor metadata detected")
 */
class InternalError : public Error {
 public:
  PYPTO_ALWAYS_INLINE explicit InternalError(const std::string& message) : Error(message) {}
};

/**
 * @brief Severity level for diagnostics
 *
 * Diagnostics can be either errors (must be fixed) or warnings (should be reviewed).
 */
enum class DiagnosticSeverity {
  Error,    ///< Error that must be fixed
  Warning,  ///< Warning that should be reviewed
};

/**
 * @brief Single diagnostic message from verification
 *
 * Represents a single issue found during IR verification. Contains information
 * about the severity, which rule detected it, the specific error code, a human-readable
 * message, and the source location where the issue was found.
 */
struct Diagnostic {
  DiagnosticSeverity severity;  ///< Severity level (Error or Warning)
  std::string rule_name;        ///< Name of the verification rule (e.g., "SSAVerify", "TypeCheck")
  int error_code;               ///< Specific error code from the rule's error type enum
  std::string message;          ///< Human-readable error message
  ir::Span span;                ///< Source location of the issue

  /**
   * @brief Default constructor
   */
  Diagnostic() : severity(DiagnosticSeverity::Error), error_code(0), span(ir::Span::unknown()) {}

  /**
   * @brief Construct a diagnostic with all fields
   */
  Diagnostic(DiagnosticSeverity sev, std::string rule, int code, std::string msg, ir::Span s)
      : severity(sev),
        rule_name(std::move(rule)),
        error_code(code),
        message(std::move(msg)),
        span(std::move(s)) {}
};

/**
 * @brief Exception raised when IR verification fails
 *
 * This exception is thrown when IR verification detects errors (not warnings).
 * It contains a formatted report of all diagnostics and the raw diagnostic data.
 *
 * Use this exception when:
 * - IR verification finds one or more errors
 * - You want to report all verification issues at once rather than failing on first error
 *
 * Example: VerificationError("IR verification failed with 3 errors", diagnostics)
 */
class VerificationError : public Error {
 public:
  /**
   * @brief Construct a verification error with report and diagnostics
   * @param report Formatted verification report
   * @param diagnostics Vector of all diagnostics (errors and warnings)
   */
  PYPTO_ALWAYS_INLINE explicit VerificationError(const std::string& report,
                                                 std::vector<Diagnostic> diagnostics)
      : Error(report), diagnostics_(std::move(diagnostics)) {}

  /**
   * @brief Get the diagnostics that caused this error
   * @return Const reference to vector of diagnostics
   */
  [[nodiscard]] const std::vector<Diagnostic>& GetDiagnostics() const { return diagnostics_; }

 private:
  std::vector<Diagnostic> diagnostics_;  ///< All diagnostics (errors and warnings)
};

}  // namespace pypto

#endif  // PYPTO_CORE_ERROR_H_
