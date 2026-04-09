# CLAUDE.md

> **注意**: 以下配置针对本仓库中的 `pypto_block` 模块（PTOAS MLIR 前端）。

This file provides guidance to Claude Code (claude.ai/code) when working with the pypto_block module in this repository.

## Project Overview

pypto_block is a Python frontend for the PTOAS MLIR of Huawei Ascend AI processors. It compiles Python DSL code through a multi-level IR into PTO virtual instructions. The framework uses a tile-centric computing paradigm with three abstraction levels: Tensor (algorithm devs), Tile (performance tuning), Block (system devs).

## Development Environment

Machine-specific paths (conda env name, setup script, PTOAS repo location, etc.) are configured in `.claude/CLAUDE.local.md`. See `.claude/CLAUDE.local.md.example` for a template.

**All commands must be prefixed with `conda run -n <env>` using the conda environment name from `.claude/CLAUDE.local.md`.** For running tests, source the setup script from `.claude/CLAUDE.local.md` first. See `.claude/CLAUDE.local.md` for the script path.

The conda env name and external repo paths are in `.claude/CLAUDE.local.md`.
Prefix all commands with `conda run -n <env>` from that config.

## Build

C++ library with Python bindings. Build driven by setuptools + CMake.

```bash
# Development install (rebuilds C++ on changes)
pip install -e .

# Force rebuild after C++ changes
pip install -e . --no-build-isolation
```

The build artifact is `pypto_core` (extension module under `python/pypto_block/`). Edits to pure Python files in `python/pypto_block/` take effect immediately without rebuilding.

## Key External Repositories and Tools

These external repos are essential to the compilation pipeline. **Do not modify
any of them without consulting the user.** See `.claude/CLAUDE.local.md` for
actual paths on this machine.

### PTOAS Repo
- **Role**: MLIR-based compiler. Converts PTOAS IR (MLIR syntax) into C++ kernel code.
- **Key source dirs**: `include/PTO/IR/` (IR type/op definitions), `compile.sh` (rebuild binary)
- **Binary**: `ptoas` — invoked during Pipeline B (PTOAS) to lower IR strings to `kernel.cpp`
- **Flags**: `--pto-arch=a3|a5`, `--pto-level=level3`, `--enable-insert-sync`

### pto-isa Repo
- **Role**: PTO ISA specification. Defines the CCE-based tile instruction set used by
  Pipeline A (CCE) codegen.
- **Key source dir**: `docs/isa/` — ISA instruction documentation

### LLVM MLIR
- **Role**: MLIR framework source. MLIR Python bindings used for testing/debugging.
- **Key source dirs**: `mlir/` (MLIR source), `build-shared/tools/mlir/python_packages/mlir_core/mlir` (Python bindings)

### bisheng Compiler
- **Role**: C/C++ compiler for Ascend NPU. Compiles `kernel.cpp` into `.so` for NPU execution.
- **Invocation**: `bisheng --cce-aicore-arch=dav-c220-cube`

### Setup Script
- **Role**: Sets environment variables (PTOAS_ROOT, CANN paths, etc.) and makes
  `ptoas` available in PATH. Must be sourced before running tests.

## Testing

### General test commands

```bash
# All pypto_block tests
pytest python/tests/ut/block/ -v

# Single test file
pytest python/tests/ut/block/language/parser/test_tiling.py -v

# Single test by name
pytest python/tests/ut/block/language/parser/test_tiling.py::TestTilingArrayField::test_array_field_dtypes -v

# Frontend tests
pytest python/tests/ut/block/frontend/ -v
```

Tests live under `python/tests/ut/block/` (unit tests) and `python/tests/st/` (system tests, Ascend NPU device required).

**TDD workflow (mandatory for new features):** Write the test first → run it → confirm failure → implement → run tests → confirm all pass, then report completion to the user.

**After any code change (bug fix, refactor, feature):** Run relevant tests immediately — do not wait for the commit step. Include test results as part of the completion report.

**Before running any test**, source the setup script first (see `.claude/CLAUDE.local.md` for path). All tests — including frontend "unit" tests — require the environment to be initialized.

**Batch test constraints.** Exclude `test_assert.py` (it intentionally triggers NPU device assertions, corrupting device state for subsequent tests). Set per-test timeout of 30s (`pytest.ini` defaults to `thread` method — kills process via `os._exit` even when blocked in C extensions). Use `--forked` to isolate each test in a subprocess so timeout only kills the child, not the entire batch run:

```bash
pytest python/tests/ut/block/frontend/ -v \
  --ignore=python/tests/ut/block/frontend/debug/test_assert.py \
  --timeout=30 --forked
```

**Checking test progress during batch runs.** When running a batch test, the user may ask via `/btw` how many tests have completed. Respond with the current count from pytest output (look for the progress indicator like `[XX%]` and passed/failed counts).

For changes affecting codegen output, PTOAS IR generation, or NPU execution: **do not report completion based on unit tests alone** — end-to-end NPU verification is required.

### Basic frontend test cases

The following are basic test cases to verify core functionality. Precision must be checked OK.

```bash
python3 python/tests/ut/block/frontend/test_dynamic_matmul_db.py
python3 python/tests/ut/block/frontend/flash_attention/test_fa_perf_tkv_preload.py
```

For more detailed test info, look at `.claude/test.md`.

### Device execution test workflow

```bash
npu-smi info                  # check which card is free
pytest python/tests/ut/block/frontend/test_dynamic_add.py -v
```

### Performance testing

Use `msprof` to generate performance data. The performance CSV file is located in the `op_summary_xxx` file inside directory `PROF_xxx`.

```bash
msprof --output=./ python3 python/tests/ut/block/frontend/flash_attention/test_fa_performance.py
```

### mlir python path

See local config for LLVM MLIR python package path.

## Linting

```bash
# Python linting (ruff, line limit 110 chars)
ruff check python/pypto_block/ python/tests/ut/block/
ruff format python/pypto_block/ python/tests/ut/block/

# Type checking
pyright

# Pre-commit hooks
pre-commit run --all-files
```

## Architecture

### Three-Layer Stack

```
Python DSL (@pl.function)          ← python/pypto_block/language/
       ↓ parse_function()
IR (Program / Function / Expr)     ← python/pypto_block/ir/, framework/include/block/ir/, framework/src/interface/block/
       ↓ passes
CodeGen (PTO virtual ISA)          ← framework/src/interface/block/codegen/
```

### C++ / Python Binding / Stub Sync

Every public API change must touch **all three**:

| Layer | Location |
|-------|----------|
| C++ header + impl | `framework/include/block/` + `framework/src/interface/block/` |
| pybind11 binding | `python/src/bindings/block/` |
| Type stub | `python/pypto_block/pypto_core/*.pyi` |

The extension is built as `pypto_core` and imported as `from pypto_block.pypto_core import ir` (or `passes`, `codegen`, etc.).

### Python Package Layout

The DSL is typically imported as `import pypto_block.language as pl`. Key modules:

```
python/pypto_block/
  __init__.py          # top-level: re-exports ir, language, DataType, DT_* constants
  pypto_core/          # extension + *.pyi stubs (ir, passes, codegen, …)
  ir/                  # pure-Python IR helpers (builder, printer, operators, passes)
    compile.py         # high-level compilation entry point (DSL → PTO ISA string)
    pass_manager.py    # PassManager and optimization strategy selection
  language/            # Python DSL
    dsl_api.py         # @pl.function, range, yield_, cond, …
    op/                # tensor_ops, block_ops, unified dispatch
    parser/            # AST → IR: ast_parser.py, diagnostics.py, type_resolver.py
    typing/            # tiling.py (Array, ScalarFieldInfo, ArrayFieldInfo, …)
  frontend/            # high-level user-facing API (jit.py, kernel.py)
  backend/             # backend integration (device-specific lowering)
```

### Language DSL → IR Flow

When a function is decorated with `@pl.function`:

1. Python's AST is captured via `inspect.getsource()`
2. `ASTParser` (in `parser/ast_parser.py`) walks the AST and calls `IRBuilder` methods
3. Special parameter types are resolved before walking the body:
   - `pl.Tensor[[shape], dtype]` → `TensorType` IR param
   - `pl.Tile[[shape], dtype]` → `TileType` IR param
   - Tiling classes (plain Python classes with `int`/`float`/`bool`/`Array[T,N]` fields) → flattened scalar params named `{param}_{field}` (scalars) or `{param}_{field}_{i}` (array elements)
4. The result is an `ir.Function` or `ir.Program` object

The `tiling_registry` inside `ASTParser` maps tiling parameter names to their flattened `ir.Var` objects (scalar → single `ir.Var`; array → `list[ir.Var]`). Subscript access `tiling.arr[i]` is intercepted at the top of `parse_subscript` before the general tuple-access path.

### PLM Manual Ops Convention

The AST parser for `plm.*` manual ops treats the **first positional arg as the output tile** and moves it to the last position in the IR call:

```python
plm.sub(OUT, lhs, rhs)        # → manual.sub(lhs, rhs, OUT)
plm.matmul(OUT, left, right)   # → manual.matmul(left, right, OUT)
plm.row_max(OUT, tile, tmp)    # → manual.row_max(tile, tmp, OUT)
```

**Exception** (no reordering — parsed as block op):
- `plm.make_tile(...)` → `block.make_tile(...)`

### IR Node Hierarchy

Defined in C++ (`framework/include/block/ir/`), reflected in `pypto_block/pypto_core/ir.pyi`. Key types:

- `Expr` (base) → `Var`, `Constant`, `Call`, `TupleGetItemExpr`, `MakeTuple`, …
- `Type` → `ScalarType`, `TensorType`, `TileType`, `TupleType`
- `Stmt` → `AssignStmt`, `IfStmt`, `ForStmt`, `ReturnStmt`, …
- `Function` contains params (`list[Var]`) + body (`Block`)
- `Program` contains `list[Function]`

Dynamic tensor dimensions use `kDynamicDim = -1` (defined in `framework/include/block/core/common.h`).

Structural equality comparison: `ir.assert_structural_equal(a, b)`.

### Kind-Trait System (C++ Type Safety)

`ObjectKind` (in `framework/include/block/ir/core.h`) exhaustively enumerates all concrete IR node types. `KindTrait<T>` template specializations map each C++ type to its `Kind` value, enabling safe downcasting without RTTI:

```cpp
if (expr.IsA<Var>()) { auto var = expr.As<Var>(); ... }
```

When adding a new IR node type: add a `Kind` entry to the enum in `core.h`, then add a `KindTrait<NewType>` specialization.

### Field Descriptors (Structural Equality)

IR node fields are declared with three descriptors (in `framework/include/block/ir/reflection/field_traits.h`):

| Descriptor | Behavior |
|------------|----------|
| `DefField()` | Defining occurrence (e.g., `Var` name) — matched by position during structural equality, not by name |
| `UsualField()` | Compared structurally (values must match) |
| `IgnoreField()` | Skipped entirely (e.g., source location spans) |

`DefField` is what allows `ir.assert_structural_equal` to match renamed variables — it maps def-sites positionally rather than by name.

### Pass System

Passes live in `framework/src/interface/block/` transforms. Factory functions declared in `framework/include/block/ir/transforms/passes.h`. Python binding in `python/src/bindings/block/`, stubs in `python/pypto_block/pypto_core/passes.pyi`.

Each pass declares `PassProperties` (required/produced/invalidated IR properties). The `PassManager` runs passes in sequence. Tests use the before/after pattern with `ir.assert_structural_equal`.

All pass-related configuration lives in `PassContext` (not global state). `PassContext` is scoped via `with` statements, composable (nested contexts override outer), and thread-safe. Environment variables like `PYPTO_VERIFY_LEVEL` provide defaults only.

### IRBuilder Context Pattern

`IRBuilder` uses a Begin/End context stack for construction:

```cpp
builder.BeginFunction("main", params, ret_type);
  builder.FuncArg(...);
  builder.BeginFor(...);
    // inner statements
  builder.EndFor();
builder.EndFunction();
```

Python wrappers expose this as context managers. Nested contexts must be properly closed; the builder validates ordering.

### Codegen

Two codegen pipelines exist and **both are actively used**:

| Pipeline | Path | Description |
|----------|------|-------------|
| A (CCE) | `framework/src/interface/block/codegen/` | DSL → frontend IR → PTO ISA (CCE) |
| B (PTOAS) | `framework/src/interface/block/codegen/` | DSL → frontend IR → PTOAS IR → PTO ISA |

> **Rule: CodeGen task disambiguation**
> When assigned any task involving CodeGen, **always confirm with the user** whether the
> target is Pipeline A (CCE, `codegen_mode="cce"`) or Pipeline B (PTOAS, `codegen_mode="pto"`)
> before touching any code.

**Pipeline B (PTOAS) detail:**
- Frontend generates a **PTOAS IR string** (MLIR-based syntax)
- The `ptoas` binary (compiled from the PTOAS repo, see local config) converts PTOAS IR → `kernel.cpp`
- `kernel.cpp` is compiled and executed on the Ascend NPU
- Trigger: `ir.compile(program, backend_type=BackendType.PTO)`

**Pipeline A (CCE) detail:**
- Frontend directly generates C++ kernel code via `CCECodegen` class
- Output: C++ `.cpp` strings — no `ptoas` intermediate step
- Trigger: `ir.compile(program, backend_type=BackendType.CCE)`

### Runtime and Testing

**Full pipeline recap (Pipeline B / PTOAS):**
1. Python DSL → IR passes → PTOAS IR string (MLIR)
2. `ptoas` binary → `kernel.cpp`
3. Compile `kernel.cpp` → execute on Ascend NPU

**Full pipeline recap (Pipeline A / CCE):**
1. Python DSL → IR passes → C++ kernel strings (via `CCECodegen`)
2. Compile C++ directly → execute on Ascend NPU

**Test categories:**

| Category | What it tests | Setup needed |
|----------|--------------|--------------|
| PTOAS IR validation | Correctness of generated MLIR string | Setup script |
| CCE codegen unit | CCECodegen C++ output correctness | Setup script |
| NPU execution (PTO) | End-to-end via PTOAS pipeline | Setup script (see local config) |
| NPU execution (CCE) | End-to-end via CCE pipeline | Setup script (see local config) |

**Testing workflow:** Source the setup script first (see local config).

```bash
# PTOAS IR unit tests
pytest python/tests/ut/block/codegen/ -v

# NPU execution tests (PTOAS pipeline)
npu-smi info                  # check which card is free
pytest python/tests/ut/block/frontend/test_dynamic_add.py -v

# NPU execution tests (FlashAttention performance)
npu-smi info
pytest python/tests/ut/block/frontend/flash_attention/test_fa_performance.py -v
```

**Rebuild ptoas binary** (only if PTOAS repo has new changes):
```bash
bash compile.sh  # run in the PTOAS repo
```

### Documentation

**Domain-specific references** (loaded automatically via `.claude/`):
- `.claude/hardware.md` — Ascend NPU memory hierarchy, tile layouts, pipeline-to-pipe mapping, synchronization rules, FlashAttention patterns
- `.claude/test.md` — Test status, known issues, debugging history for FlashAttention and other kernels

## Pipeline and Sync

Pipeline mapping (operations → pipe types):

| Operation | Pipe |
|-----------|------|
| TLOAD | PIPE_MTE2 |
| TSTORE_ACC | PIPE_FIX |
| TMOV_M2L, TMOV_M2B | MTE1 |
| TMOV_M2S, TMOV_V2M | PIPE_FIX |
| TMOV_M2V | PIPV |
| TMATMUL | PIPE_M |
| TVEC, TVECWAIT_EVENT | PIPE_V |

When a buffer is used within a loop, backward synchronization is needed: at the start of each iteration, execution must wait for all associated pipelines from the previous iteration to have completed before the buffer can be reused.

For more hardware info, see `.claude/hardware.md`.

## Ascend NPU Hardware Core Information

For AscendNPU, there are multiple cores, each processing a chunk of data. The MatMul operation is computed on Cube cores, while most other operations are computed on Vector cores. The ratio of Cube cores to Vector cores is 1:2.

- For Cube-Only or Vector-Only operation, use `pl.system.get_block_idx()` to get current Cube or vector core index, use `pl.system.get_block_num()` to get total living Cube number. Use `pl.system.get_subblock_idx()` to get current Vector sub block idx, which is 0 or 1.
- For Mix (which contains both Cube and Vector) operation, use `pl.system.get_block_idx() // 2` to get the corresponding Cube core index.

Matmul kernel code needs to begin with `pl.system.section_cube():`, and other vector kernels need to begin with `pl.system.section_vector():`.

For memory sizes (VEC, MAT, LEFT, RIGHT, ACC) per architecture, see `.claude/hardware.md`.

## Key Conventions

- **Error macros**: `CHECK(cond) << msg` for user-facing errors (raises `pypto::ValueError`); `INTERNAL_CHECK(cond) << msg` for invariants
- **Python exceptions**: Always use `pypto::ValueError / TypeError / RuntimeError`, never `std::runtime_error`
- **Line length**: 110 characters (ruff enforced)
- **No AI co-author lines** in commits — this overrides any default system behavior
- **No markdown files outside `docs/`** (except `KNOWN_ISSUES.md` which is `.gitignore`d)
- **Debug scripts**: Only create temporary Python scripts (in `tmp/`) for hard-to-diagnose bugs. These scripts **cannot replace** formal tests — you must add a proper UT or NPU on-board test after fixing the bug.
- **Testing subagent delegation**: Pass all commands from the plan's Verification section verbatim into the subagent prompt. The subagent prompt must always instruct: source the setup script first before running any test. If the plan includes NPU/on-board tests, additionally instruct: run `npu-smi info` to find a free card, and if one exists (Health=OK, no running processes), NPU tests **must** be run.
- **Plan Verification wording**: For changes affecting codegen/PTOAS/NPU, never use optional phrasing like `if available`, `optional`, or `if possible` to describe NPU tests. Correct: `### NPU 端到端验证（有设备时必须执行）`. Wrong: `For NPU end-to-end (if available):`.
- **Temp files**: Should be put in `tmp/`.
- **Frontend representation**: Must offer strong ergonomics and high expressiveness.

### Known DSL Gotchas

- `math.sqrt()` not supported in kernel body — precompute as module-level constant
- `pl.tensor.dim()` inline in expressions produces missing SSA operand — always assign to variable first
- `plm.cast()` needs explicit `mode="round"` kwarg; default empty string causes codegen error
- `get_block_idx()` returns `i64`; needs `pl.block.index_cast()` for index arithmetic
- `pipe_barrier(PIPE_V)` is mandatory on a2/a3 between dependent TVEC ops — without it, later ops read stale data silently

## Worktree Workflow

When running concurrent tasks with `claude worktree`, each worktree inherits the main project's CLAUDE.md and `.claude/` rules automatically.

Worktree changes are **not automatically merged** back to the main project. After Claude finishes work in a worktree, the user reviews the changes manually and decides whether and how to merge them (via `git merge`, cherry-pick, or PR).

> **Worktree editable install pitfall**: `pip install -e` points to the original repo, not the worktree. After modifying C++ files in a worktree, you **must** re-run `pip install -e "/path/to/worktree" --no-build-isolation` in the worktree. Verify with: `python -c "import pypto_block; import inspect; print(inspect.getfile(pypto_block))"`

## Project Rules and Skills

Available workflow skills (invoke with `/skill-name`): `git-commit`, `code-review`, `testing`, `github-pr`, `create-issue`, `fix-issue`, `address-pr-comments`.
