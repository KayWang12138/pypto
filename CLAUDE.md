# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyPTO (Parallel Tensor/Tile Operation) is a high-performance programming framework for Huawei Ascend AI accelerators (CANN). It compiles AI operators written in Python through a multi-layer IR pipeline down to hardware instructions for NPU execution. The project combines a C++ compiler/backend with a Python frontend API.

## Build Commands

The build uses CMake invoked through setuptools. Requires CANN toolkit installed with `ASCEND_HOME_PATH` set.

```bash
# Standard build and install
pip install .

# Editable install (development)
pip install -e .

# CI build with options (see build_ci.py for full flag list)
python build_ci.py -f python3 -b npu -j 8

# Clean build
python build_ci.py -c

# Build with specific CMake options
pip install . --cmake-options="-DENABLE_UTEST=ON -DCMAKE_BUILD_TYPE=Debug"
```

Key CMake options (set via `-D` flags or `--cmake-options`):
- `BUILD_WITH_CANN=ON/OFF` — link against CANN (default ON)
- `ENABLE_UTEST=ON/OFF` — build C++ unit tests
- `ENABLE_STEST=ON/OFF` — build C++ system tests (requires NPU)
- `ENABLE_ASAN=ON/OFF` — AddressSanitizer
- `ENABLE_FEATURE_PYTHON_FRONT_END=ON/OFF` — Python frontend (default ON)

## Test Commands

### Python Tests (pytest)

```bash
# Run all unit tests (CPU, no NPU needed)
pytest python/tests/ut

# Run a specific test file
pytest python/tests/ut/operation/test_matmul.py

# Run system tests on specific NPU device
pytest python/tests/st --device 0

# Run with parallel workers
pytest python/tests/ut -n 4

# Run model examples
pytest models/ -k "test_matmul"

# Run with multi-card support
pytest python/tests/st --device 0 1 2 3 --cards-per-case 2
```

Test markers: `@pytest.mark.soc("950")` for SOC filtering, `@pytest.mark.world_size(2)` for multi-card, `@duration_estimate(N)` for scheduling hints.

### C++ Tests (GTest, via build_ci.py)

```bash
python build_ci.py -u          # Run UTest
python build_ci.py -s          # Run STest (requires NPU)
python build_ci.py -u -s       # Run both
```

## Architecture

### Compilation Pipeline

The core data flow through the compiler:

```
Python API (user code)
  → Parser (AST → PTO IR via @pypto.frontend.jit)
    → Tensor Graph
      → [tensor_graph_passes] → Tile Graph
        → [tile_graph_passes] → Block Graph
          → [block_graph_passes] → Execution Graph
            → CodeGen (CCE instructions)
              → AICore Compiler (binary)
                → MPMD execution on NPU
```

Each IR level has its own pass pipeline managed by `PassManager` (singleton). Passes are organized into strategies (named ordered sequences).

### Key Directories

| Path | Purpose |
|------|---------|
| `python/pypto/` | Python package — Tensor, Element, ops, config, runtime API |
| `python/pypto/frontend/` | JIT compiler frontend (`@pypto.frontend.jit` decorator, Python AST parser) |
| `python/src/` | pybind11 bindings (`pypto_impl` module) — does NOT link C++ libs directly |
| `python/tests/ut/` | Python unit tests (CPU, no NPU) |
| `python/tests/st/` | Python system tests (require NPU) |
| `framework/include/` | C++ public headers (`tilefwk/`, `ir/`, `core/`) |
| `framework/src/interface/` | Core C++ layer — tensor, operation, function, interpreter, cache, configs |
| `framework/src/passes/` | Compiler passes organized by IR level (tensor_graph_pass, tile_graph_pass, block_graph_pass) |
| `framework/src/codegen/` | Code generation (CCE output) |
| `framework/src/machine/` | Device runtime, AICore compiler, host/device backend |
| `framework/src/adapter/` | Hardware abstraction (ACL, HAL, profiling APIs) |
| `framework/tests/ut/` | C++ unit tests (GTest) |
| `framework/tests/st/` | C++ system tests (require NPU) |
| `models/` | Large model operator implementations (DeepSeek, GLM, Qwen) |
| `examples/` | Tutorials from beginner to advanced |
| `cmake/` | Build scripts, third-party dependency management |

### Python-C++ Bridge

`python/pypto/__init__.py` loads 11+ shared libraries via `ctypes.CDLL(mode=RTLD_GLOBAL)` in a specific order before importing the pybind11 module `pypto_impl`. The loading order matters — libraries depend on symbols from previously loaded ones.

Python wrapper classes (`Tensor`, `Element`, `SymbolicScalar`) wrap their C++ counterparts. The `@op_wrapper` decorator in `_op_wrapper.py` handles automatic unwrapping/wrapping between Python and C++ types.

C++ build order (defined in `framework/src/CMakeLists.txt`): `utils → adapter → interface → passes → codegen → machine → cost_model → cann_host_runtime → platform`

### Configuration System

Configuration is scope-based with a stack: `pypto.options()` works as context manager or decorator. `CompStage` enum controls compilation depth (`ALL_COMPLETE`, `TENSOR_GRAPH`, `TILE_GRAPH`, etc.).

## Development Conventions

- C++ standard: C++17, C standard: C11, GCC required for Python frontend
- Python: >=3.9, pybind11 >=3.0.1
- Language: Comments and docs are primarily in Chinese; code identifiers are in English
- Operator development: Keep golden reference, implementation, and test as separate files
- When encountering errors: first check `docs/` API docs, then `examples/`, then fix — do not simplify or rewrite
- Codebase exploration should use subagents (per AGENTS.md guidelines)
- When NPU is available (`npu-smi info`), use real NPU for verification, not simulation mode
- Do not bypass or work around compiler/linter gates — fix root causes

## Key Entry Points

- **User-facing**: `python/pypto/__init__.py` — package bootstrap and API exports
- **JIT compilation**: `python/pypto/frontend/parser/entry.py` — `JitCallableWrapper` (the `@jit` decorator)
- **AST parsing**: `python/pypto/frontend/parser/parser.py` — Python function to PTO IR translation
- **Pass pipeline**: `framework/src/passes/pass_mgr/pass_manager.h` — singleton pass orchestration
- **Code generation**: `framework/src/codegen/codegen.h` — Execution Graph to CCE code
- **Device execution**: `framework/src/machine/runtime/runtime.h` — NPU launcher/runner
