# IR Verifier

Extensible verification system for validating PyPTO IR correctness through pluggable rules with diagnostic reporting and Pass integration.

## Overview

| Component | Description |
|-----------|-------------|
| **VerifyRule (C++)** | Base class for verification rules - each rule implements a specific IR check |
| **IRVerifier (C++)** | Manages rule collection and executes verification on Programs |
| **Diagnostic** | Structured error/warning report with severity, location, and message |
| **VerificationError** | Exception thrown when verification fails in throw mode |

### Key Features

- **Pluggable Rule System**: Extend with custom verification rules
- **Selective Verification**: Enable/disable rules individually per use case
- **Dual Verification Modes**: Collect diagnostics or throw on first error
- **Pass Integration**: Use as a Pass in optimization pipelines
- **Comprehensive Diagnostics**: Collect all issues with source locations

### Use Cases

- **Development**: Catch IR errors early during construction
- **Testing**: Validate transformations preserve correctness
- **Pipeline Integration**: Insert verification between optimization passes
- **Debugging**: Generate detailed diagnostic reports for malformed IR

## Architecture

### Verification Rule System

The verifier uses a **plugin architecture** where each verification rule is an independent component:

- **VerifyRule base class**: Defines the interface all rules must implement
- **Rule registration**: Rules are added to IRVerifier at construction or runtime
- **Execution order**: Rules run in registration order across all functions
- **Independence**: Each rule operates independently - one rule's failure doesn't affect others

**Design principle**: Rules should be **composable** and **focused** - each checks one aspect of IR correctness (SSA form, types, etc.).

**Enable/Disable mechanism**: Rules can be selectively disabled without removing them. This allows:
- Testing with subsets of checks
- Disabling expensive checks in production
- Gradual migration when adding new rules

### Verification Modes

| Mode | Method | Behavior | Use When |
|------|--------|----------|----------|
| **Diagnostic Collection** | `Verify()` | Collects all errors/warnings, returns vector | Need complete error list, building tools, reporting |
| **Fail-Fast** | `VerifyOrThrow()` | Throws VerificationError on first error | Pipeline validation, testing, development |

**Mode selection guide**:
- Use `Verify()` for IDE/tool integration - users want to see all issues
- Use `VerifyOrThrow()` in pipelines - fail immediately on invalid IR
- Use `VerifyOrThrow()` in tests - clear pass/fail with exception handling

### Diagnostic System

**Diagnostic structure**:

| Field | Type | Purpose |
|-------|------|---------|
| `severity` | `DiagnosticSeverity` | Error or Warning |
| `rule_name` | `string` | Which rule detected the issue |
| `error_code` | `int` | Numeric error identifier |
| `message` | `string` | Human-readable description |
| `span` | `Span` | Source location information |

**Severity levels**:
- `Error`: IR is invalid, must be fixed
- `Warning`: IR is valid but potentially problematic

**Report generation**: `GenerateReport()` formats diagnostics into a human-readable report with counts, grouping, and location details.

### Integration with Pass System

The verifier integrates into Pass pipelines via `run_verifier()`:

- **Returns**: A `Pass` object (Program → Program transformation)
- **Behavior**: Validates program, logs diagnostics, throws on error
- **Configuration**: Accepts `disabled_rules` parameter
- **Pipeline position**: Typically inserted after transformations to validate output

**Design consideration**: The verifier Pass is **transparent** - it returns the input program unchanged if valid, making it safe to insert anywhere in a pipeline.

## C++ API Reference

**Header**: `include/pypto/ir/transforms/verifier.h`

### VerifyRule Interface

Base class for implementing custom verification rules.

| Method | Signature | Description |
|--------|-----------|-------------|
| `GetName()` | `std::string GetName() const` | Return unique rule identifier |
| `Verify()` | `void Verify(const FunctionPtr&, std::vector<Diagnostic>&)` | Check function and append diagnostics |

**Implementation requirements**:
- `GetName()` must return a unique, stable identifier
- `Verify()` should append to diagnostics, not throw exceptions
- Rules should be stateless (or use thread-safe state)

### IRVerifier Class

Manages verification rules and executes verification.

#### Construction and Configuration

| Method | Description |
|--------|-------------|
| `IRVerifier()` | Construct empty verifier with no rules |
| `static IRVerifier CreateDefault()` | Factory method - returns empty verifier (rules can be added via AddRule) |
| `void AddRule(VerifyRulePtr rule)` | Register a verification rule (ignored if duplicate name) |

#### Rule Management

| Method | Description |
|--------|-------------|
| `void EnableRule(const std::string& name)` | Enable previously disabled rule (no-op if not found) |
| `void DisableRule(const std::string& name)` | Disable rule by name - it will be skipped during verification |
| `bool IsRuleEnabled(const std::string& name) const` | Check if rule is currently enabled |

#### Verification Execution

| Method | Return | Throws | Description |
|--------|--------|--------|-------------|
| `Verify(const ProgramPtr&)` | `std::vector<Diagnostic>` | No | Run all enabled rules, collect all diagnostics |
| `VerifyOrThrow(const ProgramPtr&)` | `void` | `VerificationError` | Run verification, throw if any errors found |

#### Reporting

| Method | Description |
|--------|-------------|
| `static std::string GenerateReport(const std::vector<Diagnostic>&)` | Format diagnostics into readable report with counts and details |

**Report format**: Summary line with error/warning counts, followed by detailed listing of each diagnostic with rule name, severity, location, and message.

## Python API Reference

**Module**: `pypto.pypto_core.passes`

### IRVerifier Class

Python binding of C++ IRVerifier with snake_case naming.

#### Factory and Construction

| Method | Description |
|--------|-------------|
| `IRVerifier()` | Create empty verifier (usually not used directly) |
| `IRVerifier.create_default()` | Static method - returns empty verifier (rules can be added via custom C++ implementation) |

#### Rule Management

| Method | Parameter | Description |
|--------|-----------|-------------|
| `enable_rule(name)` | `name: str` | Enable a disabled rule |
| `disable_rule(name)` | `name: str` | Disable a rule by name |
| `is_rule_enabled(name)` | `name: str` | Check if rule is enabled (returns `bool`) |

#### Verification

| Method | Parameter | Returns | Throws | Description |
|--------|-----------|---------|--------|-------------|
| `verify(program)` | `program: Program` | `list[Diagnostic]` | No | Collect all diagnostics |
| `verify_or_throw(program)` | `program: Program` | `None` | Exception | Throw on error |

#### Reporting

| Method | Parameter | Returns | Description |
|--------|-----------|---------|-------------|
| `generate_report(diagnostics)` | `diagnostics: list[Diagnostic]` | `str` | Static method - format diagnostics |

### Diagnostic Type

Read-only structure representing a single verification issue.

| Field | Type | Description |
|-------|------|-------------|
| `severity` | `DiagnosticSeverity` | `Error` or `Warning` |
| `rule_name` | `str` | Name of rule that detected issue |
| `error_code` | `int` | Numeric identifier |
| `message` | `str` | Human-readable description |
| `span` | `Span` | Source code location |

### DiagnosticSeverity Enum

| Value | Meaning |
|-------|---------|
| `DiagnosticSeverity.Error` | IR is invalid |
| `DiagnosticSeverity.Warning` | Potentially problematic but valid |

## Usage Examples

### Basic Verification

```python
from pypto import ir
from pypto.pypto_core import passes

# Build program (assume 'program' is constructed)
verifier = passes.IRVerifier.create_default()

# Note: Default verifier is empty, custom rules must be added via C++
diagnostics = verifier.verify(program)

if diagnostics:
    report = passes.IRVerifier.generate_report(diagnostics)
    print(report)
```

### Error Handling with Exceptions

```python
verifier = passes.IRVerifier.create_default()

try:
    verifier.verify_or_throw(program)
    print("Program is valid")
except Exception as e:
    print(f"Verification failed: {e}")
```

### Inspecting Diagnostics

```python
verifier = passes.IRVerifier.create_default()
diagnostics = verifier.verify(program)

for diag in diagnostics:
    if diag.severity == passes.DiagnosticSeverity.Error:
        print(f"ERROR in {diag.rule_name}: {diag.message}")
        print(f"  Location: {diag.span}")
```

## Adding Custom Rules

Custom verification rules can be added to extend the verifier with domain-specific checks. **Note**: Custom rules can only be registered at the **C++ level**. The Python IRVerifier API does not expose `add_rule()`, so rules must be integrated into the default verifier or instantiated directly in C++.

### Implementation Steps

**1. Create Rule Class** (C++)

Inherit from `VerifyRule` and implement required methods:

```cpp
// my_custom_rule.cpp
#include "pypto/ir/transforms/verifier.h"

namespace {
class MyCustomRule : public VerifyRule {
 public:
  std::string GetName() const override { return "MyCustom"; }

  void Verify(const FunctionPtr& func,
              std::vector<Diagnostic>& diagnostics) override {
    // Implement verification logic
    // Traverse IR, check conditions, append diagnostics
  }
};
}
```

**2. Create Factory Function**

```cpp
VerifyRulePtr CreateMyCustomRule() {
  return std::make_shared<MyCustomRule>();
}
```

**3. Register Rule in C++**

Add to default verifier (recommended for project-wide rules):

```cpp
// In src/ir/transforms/verifier.cpp CreateDefault():
IRVerifier IRVerifier::CreateDefault() {
  IRVerifier verifier;
  // Add your custom rules here
  verifier.AddRule(CreateMyCustomRule());
  return verifier;
}
```

**Alternative**: Use programmatically in C++ code:

```cpp
auto verifier = IRVerifier();
verifier.AddRule(CreateMyCustomRule());
verifier.Verify(program);
```

**Python Usage**: Once added to `CreateDefault()`, the rule is automatically available:

```python
from pypto.pypto_core import passes

# Custom rule included in default verifier
verifier = passes.IRVerifier.create_default()
verifier.disable_rule("MyCustom")  # Can disable if needed
diagnostics = verifier.verify(program)
```

### Implementation Guidelines

**Use IRVisitor**: Leverage the visitor pattern to traverse IR nodes systematically.

**Create descriptive diagnostics**:
```cpp
Diagnostic diag;
diag.severity = DiagnosticSeverity::Error;
diag.rule_name = GetName();
diag.error_code = 1001;  // Unique code for this check
diag.message = "Descriptive error message";
diag.span = problematic_node->span_;
diagnostics.push_back(diag);
```

**Keep rules focused**: Each rule should check one category of issues. Multiple small rules are better than one complex rule.

**Avoid side effects**: Rules should only read IR and write diagnostics, not modify IR or maintain state between functions.

### Rule Integration Points

| Location | Purpose |
|----------|---------|
| `src/ir/transforms/your_rule.cpp` | Rule implementation |
| `src/ir/transforms/verifier.cpp` | Register in `CreateDefault()` for automatic inclusion |
| `tests/ut/ir/transforms/test_verifier.py` | Test cases |

**Note on Python Integration**: The current Python bindings do not expose `VerifyRule` or `IRVerifier.add_rule()`. Custom rules must be added to `CreateDefault()` in C++ to be available from Python. To make rules dynamically addable from Python, the bindings would need to be extended with:
- `VerifyRule` class binding
- `IRVerifier.add_rule(rule)` method binding

This limitation ensures rules are properly tested and validated before deployment.

## Design Rationale

| Design Choice | Rationale |
|--------------|-----------|
| **Plugin Architecture** | Extensibility - projects can add domain-specific checks without modifying core verifier |
| **Rule Enable/Disable** | Flexibility - expensive or experimental rules can be toggled per use case |
| **Dual Verification Modes** | Usability - tools need all errors (IDE), pipelines need fast failure (CI) |
| **Diagnostic Collection** | Completeness - users see all issues at once rather than fix-rerun cycles |
| **Pass System Integration** | Consistency - verification uses same interface as transformations |
| **Separation of Concerns** | Maintainability - each rule is independent, rules don't know about each other |
| **Function-Level Verification** | Scalability - parallelize verification across functions in future |
| **Static Report Generation** | Utility - diagnostics are data, formatting is presentation |

**Key tradeoff**: Collecting all diagnostics (vs. stopping at first error) requires more memory and processing, but provides better user experience. The dual-mode API lets users choose based on context.

**Why not inline verification?**: Separating verification from construction/transformation enables:
- Optional verification (skip in trusted pipelines)
- Selective checks (expensive rules in debug only)
- Post-construction validation (catch bugs in transformations)

## Summary

The IR Verifier provides:

- **Extensible validation framework** for PyPTO IR correctness
- **Built-in rules** for SSA form and type consistency
- **Flexible configuration** through selective rule enable/disable
- **Dual usage modes** for different contexts (tools vs. pipelines)
- **Pass integration** for seamless pipeline validation
- **Comprehensive diagnostics** with structured error reporting

### When to Use

| Scenario | Approach |
|----------|----------|
| **After IR construction** | Use `verify_or_throw()` to catch frontend bugs |
| **Between transformations** | Insert `run_verifier()` pass to validate each step |
| **In tests** | Use `verify_or_throw()` to ensure test inputs are valid |
| **IDE/tool integration** | Use `verify()` to collect all issues for display |
| **Production pipelines** | Optionally disable expensive rules, use in debug builds only |

### Related Components

- **Pass System** (`10-pass_manager.md`): Verifier integrates as a Pass
- **IRBuilder** (`08-ir_builder.md`): Construct IR that verifier validates
- **Type System** (`02-ir_types_examples.md`): TypeCheck rule validates against type system
- **Error Handling** (`include/pypto/core/error.h`): Diagnostic and VerificationError definitions

### Testing

Comprehensive test coverage in `tests/ut/ir/transforms/test_verifier.py`:
- Valid and invalid program verification
- Rule enable/disable behavior
- Exception vs. diagnostic collection modes
- Pass integration
- Diagnostic field access
- Report generation

The verifier ensures PyPTO IR maintains correctness invariants throughout compilation, enabling reliable code generation and optimization.
