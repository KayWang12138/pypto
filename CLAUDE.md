# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyPTO (Parallel Tensor/Tile Operation) is a high-performance programming framework for Huawei Ascend NPU accelerators. It compiles Python-level Tensor graphs through multiple IR stages down to hardware instructions via a tile-based programming model.

## Build Commands

```bash
# Standard install
python3 -m pip install . --verbose

# Editable/development install
python3 -m pip install -e . --verbose

# Debug build
python3 -m pip install . --verbose --config-setting=--build-option='build_ext --cmake-build-type=Debug --cmake-verbose'

# CI build script (supports -f frontend, -b backend, -t targets, -j jobs, --build_type, -u utest, -s stest, -c clean)
python build_ci.py [options]
```

Set `PYPTO_THIRD_PARTY_PATH` for offline builds. Set `PYPTO_BUILD_EXT_ARGS` for editable mode build options.

## Test Commands

```bash
# Run all tests
pytest

# Unit tests only
pytest python/tests/ut/

# System tests only
pytest python/tests/st/

# Filter by SOC type
pytest -m "soc:950"   # or soc:910

# Specify device
pytest --device 0

# Multi-card tests
pytest --cards-per-case 2
```

C++ tests are under `framework/tests/` and built via CMake.

## Code Style

- Python: follow existing patterns in `python/pypto/`
- C++: formatted with `.clang-format` at repo root (run `clang-format` before committing C++ changes)

## Architecture

The compilation pipeline has multiple IR levels:

```
User API (Tensor Graph)
    → Tile Graph   (tile-aware data layout and tiling decisions)
    → Block Graph  (block-level scheduling)
    → Execution Graph
    → CodeGen (PTO virtual instructions)
    → Target platform executable
```

Key source locations:
- `python/pypto/` — Python API and frontend (Tensor-level abstractions, operator definitions)
- `python/src/` — pybind11 bindings connecting Python to C++ backend
- `framework/src/passes/` — Compiler passes that transform between IR levels
- `framework/src/codegen/` — Code generation from Execution Graph to virtual instructions
- `framework/include/` — Public C++ headers
- `python/tests/ut/` — Python unit tests; `python/tests/st/` — system/integration tests
- `framework/tests/` — C++ unit tests
- `examples/` — Learning examples (01_beginner → 02_intermediate → 03_advanced)
- `models/` — Real LLM operator implementations (DeepSeek, GLM, Qwen, Arctic)

## Runtime

The framework loads several shared libraries at runtime:
`libtile_fwk_utils.so`, `libtile_fwk_cann_host_runtime.so`, `libtile_fwk_platform.so`, `libtile_fwk_interface.so`, `libtile_fwk_codegen.so`, `libtile_fwk_compiler.so`, `libtile_fwk_runtime.so`, `libtile_fwk_simulation.so`

Execution uses MPMD (Multiple Program Multiple Data) scheduling onto NPU processor cores.

## Key Concepts

- **Tile**: hardware-aware data block; all computation is tile-centric
- **Pass**: a compiler transformation step between IR levels (see `framework/src/passes/`)
- **SOC markers**: tests are tagged `soc:950` or `soc:910` to target specific Ascend chip generations
- **Simulation mode**: uses `libtile_fwk_simulation.so` for running without physical hardware
