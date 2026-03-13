# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyPTO (Parallel Tensor/Tile Operation) is a high-performance programming framework for Huawei Ascend AI accelerators. It uses a tile-based programming model with multi-level IR (Intermediate Representation) to compile user-defined AI operators from Tensor-level graphs down to hardware instructions.

## Build Commands

```bash
# Standard build and install
python3 -m pip install . --verbose

# Editable install (for development)
python3 -m pip install -e . --verbose

# CI build script (supports more options)
python3 build_ci.py -f python3 -b npu                    # Basic build
python3 build_ci.py -f python3 --disable_auto_execute   # Build without running tests
python3 build_ci.py -u -s -j 8                          # Build with UTest and STest

# Clean build
python3 build_ci.py -c --build_type Debug
```

## Testing

```bash
# Set device ID before running tests (required for NPU tests)
export TILE_FWK_DEVICE_ID=0

# Python unit tests
pytest python/tests/ut/

# System tests (requires NPU)
pytest python/tests/st/ --device 0

# Run specific test file
pytest python/tests/ut/test_file.py -v

# Parallel test execution
pytest -n 8 python/tests/ut/
```

## Environment Variables

```bash
export TILE_FWK_DEVICE_ID=0              # NPU device ID (required for STest)
export PTO_TILE_LIB_CODE_PATH=/path/     # PTO-ISA library path
export PYPTO_THIRD_PARTY_PATH=/path/     # Third-party source path (if cannot access cann-src-third-party)
```

## Architecture

The project has a multi-layer architecture:

1. **Tensor Layer** (Python: `python/pypto/`): High-level tensor operations with `pypto.Tensor`
2. **Tile Layer** (C++: `framework/src/interface/tileop/`): Hardware-aware tile operations
3. **Block Layer** (C++): Low-level block computations
4. **IR Layer** (C++: `framework/src/interface/ir/`): Intermediate representation
5. **Codegen** (C++: `framework/src/codegen/`): Virtual instruction generation

Key directories:
- `python/pypto/`: Python frontend with JIT compilation (`frontend/`), operations (`op/`), tensor implementation
- `framework/src/interface/tileop/`: Tile operator implementations (`vector/`, `cube/`, etc.)
- `framework/src/interface/operation/`: Operation definitions
- `examples/`: Sample code organized by difficulty (`00_hello_world/`, `01_beginner/`, etc.)
- `models/`: Large model implementations (DeepSeekV3, GLM, etc.)
- `docs/api/`: API documentation

## Code Patterns

### Python Kernel Pattern

```python
import pypto

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def kernel(x: pypto.Tensor, y: pypto.Tensor, out: pypto.Tensor):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)  # Set tile configuration
    out[:] = x + y
```

### Tile Operator Pattern (C++)

Tile operators in `framework/src/interface/tileop/vector/` follow a consistent pattern:
- Use templates for type flexibility
- Use `LoopVar` for iteration
- Use `pto::Tile` for tile definitions
- Use `pto::TASSIGN` for memory assignment

## Development Workflow

1. **Before implementing**: Check `docs/api/` for available APIs and `examples/` for similar implementations
2. **When errors occur**: Locate the specific error point and fix that part directly. Do NOT rewrite entire implementations
3. **Testing progression**: Start with small inputs (8-16 elements), then scale up to verify correctness

## Key Files

- `AGENTS.md`: Detailed guidelines for operator development workflow
- `conftest.py`: Pytest configuration with device management hooks
- `build_ci.py`: CI build script with comprehensive options
- `pyproject.toml`: Python package configuration
