# 5. Program & Function Dialects

This section defines the **top-most structural dialects** of PTO-IR:

* `pto.program` — global/module-level constructs
* `pto.func` — SSA functions, blocks, control flow

These dialects form the "host IR" that contains tensor/tile/statement/blockgraph/pipe/execution IR inside their regions.

---

## 5.1 `pto.program` Dialect

### 5.1.1 Purpose

`pto.program` is the container for compiling and executing PTO applications.
It captures:

* The module
* Global tensors & data objects
* Entrypoints
* Device launch specifications
* Build/runtime attributes
* Version metadata
* Configuration dialect (`pto.config`, see Section 16) for transformation passes
* Platform abstractions dialect (`pto.platform`, see Section 17) for hardware modeling
* Multi-device/distributed launch plans (in coordination with `pto.dist`)

It is the canonical root of every PTO-IR file.

---

### 5.1.2 Syntax

#### Module

```
program.module @main {
    ... contents ...
}
```

Modules contain functions, globals, launches, and metadata.

---

### 5.1.3 Global Objects

#### Global tensors

```
program.global @W : tensor<1024x1024xf16> = {initial_value}
```

Attributes:

* `constant` (optional)
* `mutable` (optional)
* `device = X` (optional placement)

#### Global buffers

```
program.global_buffer @buf0 : memref<4096xf32, ND, DDR>
```

---

### 5.1.4 Entrypoints

The module must define at least one entrypoint:

```
program.entry @main_function
```

Multiple entrypoints allowed—for example, training/inference kernels.

---

### 5.1.5 Launch Descriptors

Describes how many workers/devices to launch:

```
program.launch @kernel on device="gpu0" groups=8 threads=1
```

Or multi-device:

```
program.launch @train_step {
    devices = ["gpu0", "gpu1", "gpu2", "gpu3"]
    exec_mode = "replicated"
}
```

Launch descriptors integrate with `pto.dist` for multi-worker execution.

---

### 5.1.6 Program Attributes

Examples:

```
program.attr arch = "PTOv2"
program.attr tile_default = { M=16, N=16, K=16 }
program.attr enable_debug = true
```

These attributes may influence lowering passes.

---

### 5.1.7 Verification Rules

* A program must have exactly one `program.module`.
* Entrypoints must refer to existing `func.func`.
* All globals must have fully defined types.
* Launch descriptors must reference valid devices.
* No recursive module nesting.

---

## 5.2 `pto.func` Dialect

`pto.func` provides the **SSA function abstraction** used across all dialects.
It defines:

* Function declarations
* Function definitions
* Calls and returns

Within `pto.func`, we distinguish three broad kinds of functions:

1. **Control-flow function**: contain structured control flow (expressed with `pto.statement`) and are typically written at the tensor level. These are the primary entrypoints emitted by frontends and are the source for lowering into data-flow and kernel functions.
2. **Data-flow function**: contain no explicit control flow; they describe pure data-flow graphs at the tensor or tile level (e.g., fused matmul + activation) and are amenable to aggressive optimizations and fusion.
3. **Execution function**: contains no control flow, only call operations, used to build the call topology of kernel functions
4. **Kernel function**: may contain control flow but operate at the instruction level (or very low-level tile/memory ops) and are intended to map closely to hardware execution (per-core or per-thread kernels).

---

### 5.2.1 Function Definition Syntax

```
func.func @matmul(%A: tensor<...>, %B: tensor<...>) -> tensor<...> {
    // function scope
    statement.for{}
    statement.block{}
    statement.return %C
}
```

A function consists of:

* Name
* Argument list
* Return types
* A body region whose lexical scope is directly managed by the function

---

### 5.2.2 SSA Semantics

PTO-IR follows MLIR/LLVM-style SSA:

* Each value is defined exactly once
* Uses must dominate definitions (except φ-like merges)
* Block arguments function as φ-nodes

---

### 5.2.3 Regions

Functions define the top-level lexical region for their body, while control-flow constructs such as `statement.for` and `statement.if` maintain their own nested scopes. Verification ensures every scope nests properly (no uncontrolled nesting) and that yields satisfy the parent’s contract.

---

### 5.2.4 Control Flow Constructs

`pto.func` functions rely exclusively on the `pto.statement` dialect to encode structured control flow, with every loop or branch expressed using statement operations:

- **`statement.block`**: linear sequence of concrete operations (tensor/tile/memory ops) within a scope.
- **`statement.for`**: counted loops with explicit induction variables, `iter_args`, and result values.
- **`statement.if`**: conditional execution that produces SSA values via `statement.yield`.
- **`statement.yield`**: terminator that returns values from the current scope to its parent scope or loop.
- **`statement.return`**: terminator that exits the enclosing function scope, returning values to the caller.

All function-level control flow must be expressed this way before tensor/tile lowering. Frontends targeting PTO-IR are expected to emit statements directly—SCF/CFG compatibility layers are intentionally omitted to keep the canonical control-flow model consistent with the execution pipeline (Sections 2 and 8).

### 5.2.5 Function Calls

Function calls are represented by `statement.call` in the `pto.statement` dialect (see Section 6.2.3). The `pto.func` dialect itself does not define a separate `func.call` operation.

Users may independently define kernel functions (e.g., low-level `func.func` implementations that operate at the instruction or tile/memory level) and invoke them from control-flow functions via `statement.call`. This enables a common calling convention where high-level control-flow functions orchestrate computation, while kernel functions encapsulate hardware-near implementations that can be reused across different call sites.

---

### 5.2.6 Function Attributes

Examples:

```
func.attr private = true
func.attr inline = true
func.attr distributed = false
```

Attributes may affect:

* Distributed placement
* Inlining
* Specialization
* Tiling decisions
* Autotuning hints

---

### 5.2.7 Function-Level Shape Constraints

Functions may declare symbolic shape constraints:

```
func.func @foo(%A: tensor<%B x 128xf32>) 
    requires (%B > 0)
```

Or more complex constraints:

```
requires (%seq <= 1024 && %batch % 8 == 0)
```

These constraints propagate into shape inference and verifier checks.

---

### 5.2.8 Function Verification Rules

A valid function satisfies:

#### Structural

* Exactly one top-level lexical scope maintained by the function
* All values SSA-defined
* Terminators present

#### Types & Shapes

* Argument/result types must be valid PTO types
* Symbolic shape constraints must be satisfiable

#### Distributed

* Functions marked `distributed` must contain only allowed ops (`pto.dist` can only appear at function/module level)

#### Lowering correctness

* No `tensor` ops below `pto.func` level
* No `inst` ops above `pto.pipe` level

---

### 5.2.9 Interaction With Other Dialects

#### With `pto.statement`

* Region, control flow and memory operations inserted through lowering passes

#### With `pto.tensor`

* Tensor ops appear inside function regions
* Shape/type inference uses function arguments
* Tensor lowering triggered from here

#### With `pto.tile`

* Tile ops appear after tensor → tile lowering
* Blocks of tile ops may appear inside loops

#### With `pto.blockgraph`

* Block definitions are emitted after statement lowering and may capture SSA values defined inside functions.

#### With `pto.pipe`

* Pipe-execution programs reference `pto.blockgraph` symbols generated from functions but are emitted outside the original function regions.

#### With `pto.exec`

* Execution graphs call into functions or block symbols and use function attributes (e.g., affinity metadata) when constructing the global DAG.

#### With `pto.dist`

* Distributed ops appear in outer blocks (not inside tile/statement regions)

---

### 5.2.10 Example

```
pto.program.module @main {
  pto.program.global @W : tensor<1024x1024xf16> = { ... }
  pto.program.entry @matmul_kernel

  pto.func.func @matmul_kernel(%A: tensor<1024x1024xf16>,
                               %B: tensor<1024x1024xf16>)
      -> (tensor<1024x1024xf16>)
  {
      statement.block {
        %C = pto.tensor.matmul %A, %B : tensor<1024x1024xf16>, tensor<1024x1024xf16> -> tensor<1024x1024xf16>
      }
      statement.return %C
  }
}
```

After lowering, this function will eventually be rewritten into tile/statement/pipe/instruction form.

---

## 5.3 Summary

`pto.program` establishes the global structure
`pto.func` provides SSA function and control-flow semantics

They anchor the entire IR system and host statement, tensor, tile, blockgraph, pipe execution, execution, and distributed computations in nested regions.

---

