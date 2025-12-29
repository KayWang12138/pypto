# 3. Dialect Overview

PTO-IR is a **multi-dialect, multi-level intermediate representation**.
Each dialect defines:

* A *domain of abstraction*
* A *set of operations*
* A *set of types*
* *Verification rules*
* Canonical *lowering paths* to other dialects
* A textual and in-memory representation
* Optional performance or debug metadata

This section provides an architectural overview of each dialect.
Detailed operation listings and grammars appear in Sections 5–15.

---

## 3.1 List of All Dialects

The PTO-IR dialect set is as follows:

| Dialect           | Purpose                                                                              |
| ----------------- | ------------------------------------------------------------------------------------ |
| **`pto.program`** | Modules, globals, kernel launch descriptors, function containers                     |
| **`pto.func`**    | SSA functions, blocks, regions, CFG/SCF control flow                                 |
| **`pto.statement`**   | Structured statement tree, control flow (for/if), loop nests, memory ops             |
| **`pto.scalar`**  | Scalar arithmetic, logic, comparisons, index calculations                            |
| **`pto.tensor`**  | Tensor computation (matmul, conv, elementwise, reductions, normalize, reshape, etc.) |
| **`pto.tile`**    | Tiled tensor ops, tile formats, tile-level transforms                                |
| **`pto.blockgraph`**  | Canonical tile subgraphs and dependency metadata                                   |
| **`pto.inst`**    | Hardware-level PTO instructions (L0 load, matmul micro-ops, atomic ops)              |
| **`pto.pipe`**   | Per-core pipe scheduling, synchronization, and buffer management                    |
| **`pto.exec`**    | Execution graph construction and runtime scheduling                                   |
| **`pto.dist`**    | Distributed launch, sharding, collective ops, remote memory                          |
| **`pto.mem`**     | Memory allocation, DMA, address spaces, layout transformations                       |
| **`pto.debug`**   | Debug info, source correlation, annotations, traces                                  |
| **`pto.config`**  | Transformation pass configuration, optimization hints, auto-tuning settings         |
| **`pto.platform`**| Platform abstractions: execution units, memory hierarchy, interconnections          |

Lowering between dialects follows a hierarchical pipeline (dialects themselves remain isolated, but they appear in specific stages of the pipeline):

```
pto.program
   ↓
pto.func + pto.dist
   ↓
pto.statement
   ↓
pto.tensor
   ↓
pto.tile
   ↓
pto.blockgraph
  ↙          ↘
pto.pipe  pto.exec
   ↓             ↓
pto.inst  runtime scheduler
```

**Note:** Meta-dialects (`pto.debug`, `pto.config`, `pto.platform`) do not participate in lowering flows but are used by transformation passes for decision-making and metadata.

---

## 3.2 Dialect Architecture

Each dialect is defined by:

* **Type system extensions**
* **Attributes**
* **Operations**
* **Dialect interfaces**
* **Canonical lowering interfaces**
* **Verification constraints**

PTO-IR uses MLIR-style "dialect isolation":
Ops from one dialect cannot implicitly depend on ops from another unless explicitly allowed via interfaces or lowering rules.

---

## 3.3 Dialects in Increasingly Lower Abstraction

### 3.3.1 `pto.program` — Program & Module Dialect

#### Purpose

Top-level representation of a PTO program, similar to MLIR's `module` + XLA's HLO module + TVM's PrimFunc metadata.

#### Key concepts

* Modules containing functions
* Kernel launch metadata
* Device assignment, worker counts
* Global tensors, global buffers
* Versioning, attributes, IR metadata
* Multi-function entrypoints for distributed jobs

#### Key operations

* `program.module` — container
* `program.global` — global tensor/buffer constant
* `program.entry` — sets the main kernel(s)
* `program.launch` — multi-device launch description
* `program.attr` — program attributes (arch, tile config, debug switches)

#### Types

* Global tensor types
* Launch descriptor types

#### Lowering

* Multiplies into `pto.func` + `pto.dist` dialects
* No direct lowering below that level

#### Verification

* Exactly one entrypoint
* All referenced functions exist
* Launch descriptors use valid devices

---

### 3.3.2 `pto.func` — Function & Control Flow Dialect

#### Purpose

* Define SSA functions with blocks and regions
* Provide structured and CFG control-flow ops
* Bridge between high-level program representation and tensor ops

#### Key operations

* `func.func @name(args) -> results`
* `func.fork` / `func.join` (optional parallel forks)

#### Types

* Function types `(T1, T2, ...) -> (R1, R2)`
* Closure-like types are illegal (no capturing)

#### Lowering

* Function inlining passes (optional)
* Eventually consumed by tensor dialect lowers

#### Verification

* SSA dominance
* All blocks terminate
* No unused block arguments

---

### 3.3.3 `pto.scalar` — Scalar Arithmetic Dialect

#### Purpose

Primitive arithmetic & control ops.
Used everywhere (loop bounds, dynamic shapes, tiling bounds).

#### Key operations

* Add/Sub/Mul/Div
* Min/Max
* Comparisons (`lt`, `le`, `eq`, …)
* Type conversions
* Boolean ops
* Index ops (compute strides, offsets)

#### Types

* `i32`, `i64`, `f16`, `bf16`, `f32`, `bool`
* `index` (abstract integer index type)

#### Lowering

* Final lowering goes to instruction dialect (`add_i32`, etc.)
* Used inside all other dialects

#### Verification

* Pure ops (no side effects)
* No implicit broadcasting

---

### 3.3.4 `pto.tensor` — Tensor Computation Dialect

#### Purpose

High-level tensor algebra used by frontend and graph optimizers.

#### Key operations

**Algebraic**

* `tensor.matmul`
* `tensor.conv`
* `tensor.add`, `tensor.mul`, `tensor.sub`
* `tensor.softmax`, `tensor.norm`, `tensor.rms_norm`
* `tensor.broadcast_in_dim`
* `tensor.gather`, `tensor.scatter`

**Shape transform**

* `tensor.transpose`, `tensor.permute`
* `tensor.reshape`
* `tensor.slice`

**Shape ops**

* `tensor.shape_of`
* `tensor.dim`

**Extract/Update**

* `tensor.update_slice`
* `tensor.extract_element`
* `tensor.update_element`

#### Types

* `tensor<shape, elem_type, layout?>`
* Dynamic dims: `?`
* Symbolic dims: `%sym0`

#### Lowering

* Pattern rewriters convert tensor ops into tile ops
  (`tensor.matmul` → `tile.matmul` according to tile config)
* Broadcasting lowers into explicit loops or tile ops

#### Verification

* Shape/symbolic correctness
* Layout consistency
* No aliasing (value semantics)

---

### 3.3.5 `pto.tile` — Tile Dialect

#### Purpose

* Tile-level operations for MMU/Vector units
* Bridge between tensor and blockgraph dialects

#### Key operations

**Core**

* `tile.matmul` (tile matmul micro-kernel)
* `tile.reduce`
* `tile.transpose`
* `tile.load_tile` / `tile.store_tile`
* `tile.set_format` (ND → NZ / fractal)
* `tile.fma` (if needed before lowering to instructions)

#### Types

* `tile<Tx, Ty, elem_type, format>`
* Tile formats (ND, NDHWC, NZ, FRACTAL_Z, custom)

#### Lowering

* Tile loads/stores → DMA or local-copy ops
* Tile matmul → block graph patterns that later map into pipe execution programs

#### Verification

* Tile must match declared sizes
* Format legality
* Inputs and outputs use compatible tile shapes

---

### 3.3.6 `pto.statement` — Statement Dialect

#### Purpose

The statement dialect is implemented as a **structured statement tree per function**. It provides a structured way to express control flow, loop nests, iteration domains, and memory access patterns. Each statement node can see all values defined in its ancestor statements and in the enclosing function.

The statement dialect serves to:
* Express loop nests and iteration domains
* Express conditional control flow (if branches) with value-producing semantics
* Represent memory access patterns with explicit indexing
* Enable optimization-friendly loop transformations

#### Key operations

**Control flow**

* `statement.for` — sequential for loop with explicit induction variable and loop-carried values. Returns results matching the types of `iter_args`.
* `statement.parallel_for` — parallel for loop (iterations must be independent)
* `statement.if` — value-producing conditional construct with phi-like semantics. Both branches must yield values matching the declared result types.
* `statement.yield` — generic region terminator that returns values from the current scope to its parent statement. Used in if branches and for loop bodies.

**Basic blocks**

* `statement.block` — linear basic block of operations with no nested control-flow statements. Always contained inside a `statement.scope`. Variables created in blocks are automatically added to the enclosing scope.

**Memory operations**

* `statement.load`, `statement.store` — load/store operations with multi-dimensional indexing
* `statement.alloc`, `statement.dealloc` — memory buffer allocation and deallocation
* `statement.gep` — get element pointer (compute offset from indices)

#### Types

* Memref-like types:
  `memref<shape, elem, layout, space>`
  where `space` ∈ {DDR, L1, UB, L0A/B/C, REGISTER}

#### Verification

* Well-formed loop bounds
* No illegal aliasing (unless marked)
* Buffer sizes consistent with shape/layout
* `statement.scope` constraints: cannot directly contain another `statement.scope`
* `statement.if` and `statement.for` must have proper `statement.yield` terminators
* Loop-carried values must match declared types

---

### 3.3.7 `pto.blockgraph` — Block Graph Dialect

#### Purpose

* Canonicalize tile subgraphs after tiling/partitioning.
* Serve as reusable units of work for runtime dispatch.
* Capture dependency metadata for later scheduling.

#### Key operations

* `blockgraph.block`
* `blockgraph.dep`
* `blockgraph.bind_tile`
* `blockgraph.meta`

#### Types

* `blockgraph.block_type`
* `blockgraph.dep_type`
* `blockgraph.binding_list`

#### Lowering

* Consumes `pto.tile` + partition metadata.
* Produces canonical blocks for `pto.pipe` and `pto.exec`.

#### Verification

* Unique `(color, key)` per block.
* Dependencies reference valid ops and dependency kinds.
* Bindings cover all parameters/tiles.

---

### 3.3.8 `pto.inst` — Instruction Dialect

#### Purpose

Lowest-level IR for direct hardware mapping.

#### Key operations

* `inst.mma` (matrix multiply accumulate)
* `inst.load_L0A`, `inst.load_L0B`
* `inst.store`
* `inst.add`, `inst.mul`
* `inst.barrier`
* `inst.atomic_add`
* `inst.branch`

#### Types

* Hardware register types
* Micro-tile types
* Instruction immediate types

#### Lowering

* To binary encoding (final codegen)

#### Verification

* All registers allocated
* All addresses resolved
* Schedule constraints satisfied

---

### 3.3.9 `pto.pipe` — Pipe Execution Dialect

#### Purpose

* Map block graph ops onto hardware pipes for a core.
* Insert synchronization (set/wait/barrier) and buffer management.
* Prepare for instruction selection.

#### Key operations

* `pipe.exec.pipe`
* `pipe.exec.assign`
* `pipe.exec.set`, `pipe.exec.wait`, `pipe.exec.barrier`
* `pipe.exec.buffer`, `pipe.exec.spill`

#### Types

* `pipe.exec.event`
* `pipe.exec.buffer_type`
* `pipe.exec.pipe_type`

#### Lowering

* Consumes `pto.blockgraph`, `pto.platform`, `pto.config`.
* Produces per-pipe schedules fed to `pto.inst`.

#### Verification

* Each block op assigned to exactly one pipe.
* Event tokens have single producer, ≥1 consumer.
* Buffers respect lifetimes; spills target valid memory spaces.

---

### 3.3.10 `pto.exec` — Execution Dialect

#### Purpose

* Describe the global execution DAG of block invocations.
* Allocate global buffers and track dependencies.
* Provide hints to runtime schedulers.

#### Key operations

* `exec.graph`
* `exec.call`
* `exec.allocate`, `exec.deallocate`
* `exec.fence`, `exec.schedule_hint`

#### Types

* Graph handles
* Dependency descriptors
* Host-visible buffer handles

#### Lowering

* Consumes `pto.blockgraph` metadata and platform topology.
* Drives runtime/scheduler codegen.

#### Verification

* Graph is acyclic unless marked streaming.
* Dependencies/nodes belong to same graph.
* Allocated buffers freed or marked persistent.

---

### 3.3.11 `pto.dist` — Distributed Execution Dialect

#### Purpose

Support multi-device, multi-host distributed execution.

#### Key operations

* `dist.launch` — spawn N workers on devices
* `dist.shard_tensor` — partition tensor
* `dist.allreduce`, `dist.allgather`, `dist.broadcast`
* `dist.shmem_put`, `dist.shmem_get`
* `dist.barrier`, `dist.event`

#### Types

* `dist.tensor<sharding>`
* Worker group types (`dist.group<world_size>`)

#### Lowering

* To runtime primitives
* No lowering to `pto.tile`; it interacts at program/function boundaries

#### Verification

* Collective op participation matches
* Sharding rules verified (compatible partitions)
* Remote memory addresses valid

---

### 3.3.12 `pto.mem` — Memory Dialect

#### Purpose

Explicit memory management and DMA.

#### Key operations

* `mem.alloc`, `mem.free`
* `mem.copy`, `mem.dma_copy`
* `mem.set_layout`
* `mem.barrier`

#### Types

* Memory reference types (with address spaces)
* Layout types

#### Lowering

* Into pipeline loads/stores and instruction-level DMA ops

#### Verification

* Buffer life tracking
* DMA region bounds checking

---

### 3.3.13 `pto.debug` — Debugging Dialect

#### Purpose

Embed source information, trace points, checkpoints.

#### Key operations

* `debug.source_loc`
* `debug.trace`
* `debug.meta`

#### Attributes

* File name, line:col
* IR-level symbolic annotations

#### Lowering

* Propagated across all dialects until final codegen
* Optionally removed in release mode

#### Verification

* No semantic impact
* Attached to valid ops/regions

---

### 3.3.14 `pto.config` — Configuration Dialect

#### Purpose

Provide configuration and hints for transformation passes. Control optimization strategies, resource allocation, and code generation decisions.

#### Key operations

* `config.tile_strategy` — tile size and format configuration
* `config.partitioning` — data and work partitioning strategies
* `config.scheduling` — loop, pipe-execution, and memory scheduling
* `config.memory_allocation` — memory management policies
* `config.optimization` — optimization level and operation-specific settings
* `config.autotuning` — auto-tuning configuration

#### Types

* `config.tile_config` — tile configuration type
* `config.partitioning_config` — partitioning configuration type
* `config.scheduling_config` — scheduling configuration type

#### Lowering

* Meta-dialect: does not participate in lowering
* Used by transformation passes to guide optimization decisions
* Can be attached to program, function, or operation level

#### Verification

* Configuration values must be valid for target platform
* Resource constraints must be satisfiable
* Conflicting configurations must be resolved

---

### 3.3.15 `pto.platform` — Platform Abstractions Dialect

#### Purpose

Model execution environment: execution units, memory hierarchy, interconnections, and distributed structures. Used by transformation passes for optimization decisions.

#### Key operations

* `platform.execution_unit` — define compute units (MMU, VMAC, etc.)
* `platform.memory` — define memory levels (L0, L1, UB, DDR)
* `platform.interconnect` — define on-chip and off-chip interconnects
* `platform.numa_domain` — define NUMA domains
* `platform.cluster` — define cluster organization
* `platform.platform` — complete platform model

#### Types

* `platform.execution_unit_type` — execution unit type
* `platform.memory_type` — memory type
* `platform.interconnect_type` — interconnect type
* `platform.platform_type` — platform type

#### Lowering

* Meta-dialect: does not participate in lowering
* Used by transformation passes to query hardware characteristics
* Typically attached at program or module level

#### Verification

* Platform model must be consistent
* Execution units must have valid properties
* Memory hierarchy must be well-formed
* Interconnects must connect valid components

---

## 3.4 Dialect Interactions

### Upward interactions (high → low)

* `pto.func` → `pto.statement` via structured statement tree formation
* `pto.statement` → `pto.tensor` (statement blocks can contain tensor operations)
* `pto.tensor` → `pto.tile` via tile selection
* `pto.tile` → `pto.blockgraph` via partitioning / canonicalization
* `pto.blockgraph` → `pto.inst` + `pto.pipe` via instruction selection and pipe placement
* `pto.blockgraph` → `pto.exec` via execution graph construction

### Horizontal interactions

* `pto.mem` interacts with all compute dialects
* `pto.debug` attaches to any op
* `pto.config` attaches to program, function, or operation level
* `pto.platform` attaches to program or module level
* `pto.dist` interacts primarily with `pto.program` and `pto.func`
* `pto.blockgraph` consumes `pto.tile` and feeds both `pto.inst`, `pto.pipe` and `pto.exec`
* `pto.pipe` consumes `pto.blockgraph` plus `pto.platform`
* `pto.exec` consumes `pto.blockgraph` metadata and collaborates with `pto.dist`

### Disallowed interactions

* `pto.tensor` cannot directly reference `pto.inst` ops
* `pto.inst` cannot appear in higher-level dialects
* `pto.dist` cannot appear inside tile/statement/blockgraph/pipe regions

---

## 3.5 Summary of Dialect Roles

| Dialect  | Level      | Purpose                           |
| -------- | ---------- | --------------------------------- |
| program  | Global     | modules, device launches          |
| func     | Global     | CFG, SSA blocks                   |
| statement    | High    | loops, memory references          |
| scalar   | High        | arithmetic primitives             |
| tensor   | High       | graph-level tensor algebra        |
| tile     | Mid        | tile computation, tile transforms |
| inst     | Lower     | hardware instructions             |
| pipe     | Lower      | pipeline scheduling               |
| mem      | Lower      | memory actions                    |
| debug    | Orthogonal | debug metadata                    |
| config   | Orthogonal | transformation pass configuration |
| platform | Orthogonal | hardware platform modeling        |

---

