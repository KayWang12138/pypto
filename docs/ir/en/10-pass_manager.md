# Pass and PassManager

Framework for organizing and executing IR transformation passes on Programs with strategy-based optimization pipelines (Default/PTOAS).

## Overview

| Component | Description |
|-----------|-------------|
| **Pass (C++)** | Standalone class for Program → Program transformations |
| **PassManager (Python)** | Manages pass sequences and execution strategies |
| **Factory Functions** | Create passes (e.g., `pass::InitMemRef()`, `pass::BasicMemoryReuse()`) |

### Key Features

- **Program-Only Interface**: All passes transform Program → Program
- **Immutable Transformations**: Return new IR nodes, don't modify in place
- **Strategy-based Pipelines**: Pre-configured optimization levels
- **Factory Pattern**: Passes created via factory functions, implementation details hidden
- **Unified Header**: All declarations in `include/pypto/ir/transforms/passes.h`

## C++ Pass Infrastructure

### Pass Base Class

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
class Pass {
 public:
  ProgramPtr operator()(const ProgramPtr& program) const;  // Execute pass
};

// Factory functions for built-in passes
namespace pass {
  Pass ExamplePass();   // Example transformation pass
  // Additional passes available - see implementation
}
```

**Key points**: Pimpl pattern hides implementation; all declarations in single header; Program → Program transformations only.

### Pass Implementation Patterns

### Pass Implementation Example

```cpp
// Example: Complex pass with state
namespace {
class ExamplePassImpl : public PassImpl {
 public:
  ProgramPtr operator()(const ProgramPtr& program) override {
    for (const auto& [name, func] : program->functions_)
      state_ += ComputeSomething(func);
    return program;
  }
  std::string GetName() const override { return "ExamplePass"; }
 private:
  int state_ = 0;
};
}
namespace pass {
Pass ExamplePass() { return Pass(std::make_shared<ExamplePassImpl>()); }
}
```

### Python Bindings

**File**: `python/bindings/modules/passes.cpp`

```cpp
void BindPass(nb::module_& m) {
  nb::module_ passes = m.def_submodule("passes", "IR transformation passes");

  // Opaque pass object
  nb::class_<Pass>(passes, "Pass")
      .def("__call__", &Pass::operator(), nb::arg("program"));

  // Factory functions (snake_case)
  passes.def("example_pass", &pass::ExamplePass);
  // Additional pass bindings available
}
```

Creates `pypto.pypto_core.passes` module with opaque `Pass` class and factory functions.

## Python PassManager

**File**: `python/pypto/ir/pass_manager.py`

### Optimization Strategies

```python
class OptimizationStrategy(Enum):
    Default = "Default"      # Full optimization pipeline
    PTOAS = "PTOAS"         # PTO assembly strategy
```

### PassManager API

| Method | Description |
|--------|-------------|
| `get_strategy(strategy)` | Get PassManager configured for strategy |
| `run_passes(program)` | Execute all passes sequentially on Program |
| `get_pass_names()` | Get names of all passes in manager |

### Strategy Configuration

Strategies are configured in `_register_passes` with pass sequences appropriate for each optimization level.

## Usage Examples

```python
from pypto import ir, DataType

# Create program with multiple functions
span = ir.Span.unknown()
dtype = DataType.INT64
x1, y1 = ir.Var("x", ir.ScalarType(dtype), span), ir.Var("y", ir.ScalarType(dtype), span)
func1 = ir.Function("func1", [x1], [ir.ScalarType(dtype)], ir.AssignStmt(x1, y1, span), span)
x2, y2 = ir.Var("x", ir.ScalarType(dtype), span), ir.Var("y", ir.ScalarType(dtype), span)
func2 = ir.Function("func2", [x2], [ir.ScalarType(dtype)], ir.AssignStmt(x2, y2, span), span)
program = ir.Program([func1, func2], "test_program", span)

# Run passes with PTOAS strategy
pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.PTOAS)
result = pm.run_passes(program)
# Result has same function names; passes apply transformations based on strategy

# One-liner shorthand
result = ir.PassManager.get_strategy(ir.OptimizationStrategy.PTOAS).run_passes(program)
```

## Implementation Details

### Program Transformation Flow

```python
def run_passes(self, program: core_ir.Program) -> core_ir.Program:
    current = program
    for pass_instance in self.passes:
        current = pass_instance(current)  # Program → Program
    return current
```

Pipeline composition: `Pass3(Pass2(Pass1(program)))` - each pass receives and returns a Program.

### Pass Registration Pattern

- Each strategy maps to `(name, factory)` tuples
- Factories are lambdas creating fresh pass instances
- Enables independent PassManager instances

## Testing

**Location**: `tests/ut/ir/transforms/test_pass_manager.py`

**Example**: Strategy execution preserves function names:

```python
def test_run_passes_on_program_with_strategy(self):
    program = ir.Program([func1, func2], "test_program", span)
    pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.PTOAS)
    result = pm.run_passes(program)
    func_names = [func.name for func in result.functions.values()]
    assert "func1" in func_names
    assert "func2" in func_names
```

## Adding New Passes

1. **Declare in `passes.h`**: `Pass YourNewPass();`

2. **Implement** (`src/ir/transforms/your_new_pass.cpp`):
   ```cpp
   // Example implementation
   namespace pass {
   Pass YourNewPass() {
     return CreateFunctionPass([](const FunctionPtr& func) {
       // Transform function
       return func;
     }, "YourNewPass");
   }
   }
   ```

3. **Python binding** (`python/bindings/modules/passes.cpp`):
   ```cpp
   passes.def("your_new_pass", &pass::YourNewPass, "Description");
   ```

4. **Register in PassManager** (`python/pypto/ir/pass_manager.py`):
   ```python
   ("YourNewPass", lambda: passes.your_new_pass()),
   ```

5. **Type stub** (`python/pypto/pypto_core/passes.pyi`):
   ```python
   def your_new_pass() -> Pass: """Description."""
   ```

6. **Test** (`tests/ut/ir/transforms/test_your_new_pass.py`)

## Design Rationale

| Design Choice | Rationale |
|---------------|-----------|
| **Immutable Transformations** | Thread safety, debugging (preserve original IR), easy rollback, functional style |
| **Strategy-Based Config** | Ease of use, consistency, centralized maintenance, extensibility |
| **Program-Only Interface** | Uniform API, enables inter-procedural optimizations, simpler mental model |
| **Single Header (`passes.h`)** | Reduced bloat, clear discovery, opaque implementation via pimpl |

## Summary

The Pass and PassManager system provides:
- **Extensible Framework**: Easy to add passes via factory functions
- **Strategy-Based Optimization**: Pre-configured levels (Default/PTOAS)
- **Unified Interface**: All passes transform Program → Program
- **Clean API**: Opaque pass objects with factory functions
- **Well-Tested**: Comprehensive test coverage
- **Immutable Transformations**: Safe, functional-style IR transformations
- **Organized Structure**: Single header file with all declarations

This infrastructure provides the foundation for building sophisticated optimization pipelines in PyPTO.
