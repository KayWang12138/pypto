# 2. Execution Model & IR Levels

This section defines the PTO-IR **execution model**, the semantics of each IR level, and the canonical lowering/translation relationships between levels. It explains the runtime view of a PTO program, how work is partitioned and scheduled, and the invariants that must hold as code is lowered from high-level tensor expressions down to PTO instructions.

---

## 2.1 High-level view

A PTO-IR program represents a complete computation that can be executed on one or more devices and/or nodes. Execution proceeds in three broad phases:

1. **Compile-time formation & analysis**

   * Frontend converts user code into an in-memory PTO-IR representation (multi-dialect).
   * Platform abstractions dialect (`pto.platform`, see Section 17) is loaded to model execution units, memory hierarchy, and interconnections.
   * Configuration dialect (`pto.config`, see Section 16) is specified to control transformation passes and runtime behavior.
   * Static analyses (type/shape inference, SSA validation, lifetime analysis, tiling heuristics) run on relevant dialects.
   * Optional auto-tuning / tile selection may produce alternate tile configs based on platform characteristics.

2. **Lowering & scheduling**

   * Dialect-to-dialect lowering transforms higher abstractions into lower ones (canonical path: `pto.func` → `pto.statement` → `pto.tensor` → `pto.tile` → `pto.blockgraph` → `pto.pipe` / `pto.exec` → `pto.inst`).
   * Transformation passes use `pto.config` dialect operations and `pto.platform` dialect queries to make optimization decisions.
   * Scheduling and memory planning allocate buffers to memory spaces (DDR, L1, UB, L0A/B/C, registers) based on platform memory hierarchy.
   * Lowerings must preserve semantics (types, shapes, SSA dominance, control flow).

3. **Runtime execution**

   * A runtime / driver executes generated kernels and coordinates distributed primitives.
   * The runtime uses launch metadata from `pto.program`/`pto.dist` to create processes/threads, map kernels to devices, and manage synchronization (barriers, remote put/get, collective ops).

### 2.1.1 Compile Pass Lowering Stages

PTO-IR lowering is defined in terms of **deterministic compile stages**, not ad‑hoc dialect-to-dialect conversions. Every stage:

* owns a named **allow-list** of dialects that may appear in the IR while the stage is active,
* implicitly **forbids** all other dialects (a stage boundary is only crossed once its allow-list is satisfied), and
* runs a known set of passes/optimizations before emitting IR for the next stage.

The pipeline is organized into six such stages:

1. **Stage 1 — Tensor Graph**

   * **Allowed dialects:** `pto.program`, `pto.func`, `pto.statement`, `pto.tensor`, `pto.dist`, `pto.config`, `pto.debug`.
   * **Forbidden dialects:** `pto.tile`, `pto.blockgraph`, `pto.inst`, `pto.pipe`, `pto.exec`, `pto.mem` (references only via attributes).
   * Purpose: Users express computation as structured statement trees with tensor operations; passes perform control flow optimizations, algebraic simplifications, automatic differentiation, and layout-aware rewrites on the high-level graph.

2. **Stage 2 — Tile Graph**

   * **Allowed dialects:** `pto.tile`, `pto.dist`, `pto.config`, `pto.debug` (buffer planning ops only).
   * **Forbidden dialects:** `pto.blockgraph`, `pto.inst`, `pto.pipe`, `pto.exec`, `pto.mem`.
   * Purpose: Split tensors into tiles based on configuration and platform hints; run tile-level optimizations (layout transforms). The resulting tile graph is partitioned into colored subgraphs that will later become blocks.

3. **Stage 3 — Block Graph**

   * **Allowed dialects:** `pto.tile`, `pto.blockgraph`, `pto.config`, `pto.platform`, `pto.debug`.
   * **Forbidden dialects:** `pto.tensor`, `pto.pipe`, `pto.exec`, `pto.inst` (until references are resolved into block metadata).
   * Purpose: Detect isomorphic subgraphs among the colored tile partitions, emit canonical block definitions (`blockgraph.block`), record dependency metadata, and bind canonical tiles back to original tiles. Each block corresponds to the unit of work dispatched to an execution unit.

4. **Stage 4 — Instruction List**

   * **Allowed dialects:** `pto.inst`, `pto.pipe`, read-only references to `pto.blockgraph`, plus meta-dialects (`pto.config`, `pto.platform`, `pto.debug`).
   * **Forbidden dialects:** `pto.tensor`, `pto.tile`, `pto.exec`.
   * Purpose: Map every tile operation inside a block graph onto specific instructions (DMA, compute, store, reduction) available on the target execution unit. Convert tile operations into hardware instructions stored as linear instruction lists. Map Tile objects to on-core buffers, insert allocation instructions (`alloc`), and perform buffer reuse analysis to optimize memory usage. Perform instruction scheduling and insert synchronization instructions (`set`, `wait`, `barrier`) to maximize utilization of on-core pipes through parallel execution. 

5. **Stage 5 — Execution Graph**

   * **Allowed dialects:** `pto.exec`, `pto.dist`, `pto.config`, `pto.platform`, `pto.debug`.
   * **Forbidden dialects:** `pto.tensor`, `pto.tile`, `pto.statement`, `pto.pipe`, `pto.inst`.
   * Purpose: Build the host-level execution graph consisting of `exec.call` operations. Dependencies are derived from block graph producer/consumer relationships; passes optimize edges, allocate global memory via `exec.allocate`, annotate scheduling hints, and prepare the dispatch plan used by the runtime scheduler.

Each stage feeds metadata to the next: tensor-level analyses influence tiling, tile partitioning supplies `blockgraph` keys, block graphs drive instruction ordering and pipe placement, and feed execution graph decisions.

---

## 2.2 The Layered IR Stack (Execution responsibilities)

The layered stack defines *what* each level is primarily responsible for and *what invariants* it enforces:

### 2.2.1 `pto.program` / `pto.func` (Program & Function Level)

**Responsibility**

* Represent modules, functions, global objects, and entry points.
* Describe the global layout of the computation: which functions run on which device(s), staging of distributed launch, and module-level attributes.

**Execution semantics**

* `program` contains launch descriptors (single process, multi-process, multi-node).
* Functions are pure computational units in SSA form; calls are explicit (`call`, `invoke`).
* Control flow (`if`, `for`, `while`) exists here as SSA blocks and regions.

**Invariants**

* Function signatures are well typed.
* `main_function` exists.
* Function call graph must be resolvable (no dangling references).

### 2.2.2 `pto.dist` (Distributed Layer)（TODO）

**Responsibility**

* Partition work across devices/nodes, specify remote memory semantics and global synchronization primitives (barrier, broadcast, shmem_put/get).
* Map tensors to distributed layouts (sharding, replication).

**Execution semantics**

* Launch units (worker instances) execute kernels with parameters; runtime maps these to processes/threads.
* Distributed memory operations are explicit and synchronous/asynchronous semantics are annotated (e.g., `dist.shmem_put` may be non-blocking with completion events).

**Invariants**

* Remote memory references include owner identifiers and valid memory spaces.
* Collective ops must have matched participants and consistent semantics.
* Lowering must preserve network semantics (ordering, visibility).

### 2.2.3 `pto.statement` (Statement / Control Flow Layer)

**Responsibility**

* Implement structured statement tree per function, providing control flow (for/if), loop nests, iteration domains, and memory operations.
* Express explicit loop-carried state and memory access patterns with explicit indexing.
* Control SSA value visibility: each statement node can see all values defined in its ancestor statements and in the enclosing function.

**Execution semantics**

* The statement dialect is organized as a tree of nested statements within a function. Functions define the top-level lexical region for their body.
* Control-flow statements (`statement.for`, `statement.parallel_for`, `statement.if`) maintain their own nested scopes via their body regions.
* `statement.block` contains linear sequences of concrete operations (tensor ops, memory ops, arithmetic).
* Loop constructs operate on indices; `statement.for` supports explicit induction variables and loop-carried values.
* Memory accesses (`statement.load`, `statement.store`) must reference memory spaces; `statement.alloc`/`statement.dealloc` ops manage buffer lifetimes.

**Invariants**

* Functions maintain the top-level scope; control-flow constructs maintain their own nested scopes.
* Loop bound expressions are scalar integers or symbolic expressions resolvable at runtime.
* `statement.for` and `statement.if` must have proper `statement.yield` terminators.
* Loop-carried values must match declared types.
* Variables created in `statement.block` are visible to subsequent statements within the same enclosing scope.

### 2.2.4 `pto.tensor` (Tensor Layer)

**Responsibility**

* High-level tensor algebra and primitive ops: `matmul`, `conv`, `rms_norm`, `reshape`, `view`, `quant/dequant`, `gather`, `scatter`, etc.
* Fusion-friendly representation for graph transforms and algebraic rewrites.

**Execution semantics**

* Ops are expressed in SSA, result types carry shape/layout information (static/dynamic).
* No explicit memory or buffer allocation semantics required at this level; ops are value-based.

**Invariants**

* Type and shape correctness (broadcast rules, matmul dims match).
* Ops may annotate preferred tile/layout hints, but semantics remain independent of hardware.

### 2.2.5 `pto.tile` (Tile Layer)

**Responsibility**

* Convert tensor-level ops into **tile-aware** ops which explicitly define how high-level operations are partitioned into tiles that fit into on-chip buffers and matrix/vector units.
* Represent tile shapes, local transforms (e.g., tile transpose), and tile-to-buffer mapping.

**Execution semantics**

* Tile ops produce tile-sized values that will be moved into local buffers (L1/L0A/B/C/UB).
* Tile-level ops carry attributes for tile sizes, tile-format (ND/NZ/fractal), and per-op tile overrides.

**Invariants**

* Each tile op includes either an explicit tile config or inherits one from the enclosing `leaf`/tile config.
* Tiling must be consistent with tensor shapes (handling edge/partial tiles via padding or tail handling logic).
* Tile lowering must emit appropriate loads/stores and buffer allocations.

### 2.2.6 `pto.blockgraph` (Block Graph Layer)

**Responsibility**

* Canonically represent isomorphic tile subgraphs that emerge after tile graph partitioning.
* Serve as the unit of work submitted to execution units.
* Preserve intra-block dependencies and tile bindings for later scheduling.

**Execution semantics**

* `blockgraph.block` encapsulates the tile ops, inputs/outputs, and dependency edges for one equivalence class of subgraphs.
* Blocks are hardware-agnostic; no pipe assignments or buffer placements occur here.
* Each block exposes explicit arguments/results so that execution graph calls can provide buffers and tiles.

**Invariants**

* Each block has a unique `(color, key)` pair corresponding to its partition and structure hash.
* Dependencies (`blockgraph.dep`) must fully cover RAW/WAR/WAW relationships among ops.
* All tiles referenced inside the block must originate from the source tile graph partition.
* Blocks have no side effects beyond their declared results.

### 2.2.7 `pto.inst` (Instruction Layer)

**Responsibility**

* Lowest-level ops that map closely to hardware instructions (matrix multiply or vector micro-ops, tile loads and stores, register moves, atomic ops).
* Provide the final representation consumed by device-specific backends or JIT assemblers.

**Execution semantics**

* Can include explicit cycles/latency annotations (optional), and explicit buffer/register operands.
* Generated from `pto.pipe` lowering (per-core schedules) plus resource planning from `pto.exec`.

**Invariants**

* All operands must be bound to real memory/register addresses or virtual registers assigned by register allocation.
* Instruction ordering must preserve correctness of data dependencies (SSA dominance ensures this).


### 2.2.8 `pto.pipe` (Instruction Layer)(TODO)

**Responsibility**

* Map block graph tile operations onto concrete instruction pipes (DMA, compute, store, etc.) per execution unit.
* Insert synchronization primitives (`pipe.exec.set`, `pipe.exec.wait`, `pipe.exec.barrier`) to satisfy dependencies.
* Manage on-core private buffers, reuse policies, and spilling to global memory when needed.

**Execution semantics**

* `pipe.exec.core` contains the per-core schedule for one block instance.
* Each `pipe.exec.pipe` is an ordered sequence of instructions that runs independently but obeys synchronization tokens.
* Event tokens provide precise sequencing between pipes; barriers synchronize multiple pipes simultaneously.

**Invariants**

* Every blockgraph operation must be assigned to exactly one pipe.
* Event tokens must have exactly one producer and at least one consumer; dead events are illegal.
* Buffer lifetimes declared via `pipe.exec.buffer` must enclose all uses.
* Spills must specify legal memory spaces and respect platform bandwidth constraints.

### 2.2.9 `pto.exec` (Execution Graph Layer)

**Responsibility**

* Encode the global DAG of function invocations, capturing dependencies, ordering constraints, and scheduler hints.
* Allocate global memory buffers and associate them with function calls.
* Provide metadata for runtime scheduling, including core-group affinities, priorities, and repetition patterns.

**Execution semantics**

* `exec.graph` behaves like a `func.func`; its body is a `statement.block` containing a sequence of `exec.call` operations that invoke executable functions.
* `exec.call` operations call into function symbols (typically `blockgraph.block` that have been lowered into functions).
* Dependencies among calls are inferred from tensor outputs and inputs: when a call's output tensor is used as another call's input, the dependency is automatically established.
* Graphs cannot span beyond a single device or a centralized scheduling domain; distributed fences and collectives appear here.

**Invariants**

* Execution graphs must be acyclic unless explicitly marked as allowing cycles (e.g., streaming loops with `repeat` semantics).
* Every `exec.call` must correspond to a real `func.func` definition so that every call is resolvable to executable code.
* Allocated resources (`exec.allocate`) must either be freed (`exec.deallocate`) or marked persistent at graph exit.
* Hints (e.g., `target_core`) must match available resources described in `pto.platform`.

---

## 2.3 Lowering semantics & preserved invariants

Lowering is the canonical and verifiable transformation from a higher-level dialect to a lower-level dialect. Each lowering must preserve the program semantics, and several invariants must hold after each stage:

**Primary invariants preserved across lowering**

1. **Type correctness** — every op's operand and result types are valid for the target dialect.
2. **Shape semantics** — numeric relationships and data layout decisions must be preserved (or recorded via attributes).
3. **SSA dominance & correctness** — defs must dominate uses; phi nodes inserted at merges.
4. **Control flow equivalence** — structured control constructs must be mapped into appropriate CFG and blocks while preserving behavior.
5. **Memory safety** — no use-after-free; allocations and releases properly placed.
6. **Data dependency preservation** — producer/consumer ordering preserved; no illegal reordering across synchronization boundaries.

**Typical lowering pipeline**

```
pto.program/pto.dist
     ↓ (module & launch decomposition)
pto.func
     ↓ (structured statement tree formation)
pto.statement
     ↓ (control flow optimization, statement elimination)
pto.tensor
     ↓ (algebraic simplification, tensor-level optimizations)
pto.tile
     ↓ (tiling selection, tile shape assignment, partitioning)
pto.blockgraph
   ↙             ↘
 (pipe placement) (execution DAG)
pto.inst         pto.exec
     ↓               ↓
pto.pipe         runtime scheduler
     ↓               ↓
execution units  launch/runtime
```

Lowerings can be multi-pass and may involve feedback loops (e.g., tile selection informed by estimated cost models). The IR supports carrying both *semantic* data (mandatory. e.g. types, shapes) and *meta* data (optional. e.g. performance hints, tile choices, debug locs) during lowering.

---

## 2.4 Execution & scheduling models

PTO-IR separates concerns between **logical parallelism** (what can be parallel) and **mapping strategies** (how parallelism is assigned to hardware).

### 2.4.1 Logical Parallelism

Logical parallelism is expressed at the `pto.tile` levels:

* **Parallel loop dimensions** — loop annotations indicate which loop dimensions can be executed in parallel.
* **Tile independence** — when tiles do not overlap in memory and have no cross-tile dependencies they can execute concurrently.
* **Vector / cube parallelism** — within a tile, vector lanes or cube units can process lanes/blocks in parallel.

### 2.4.2 Mapping to hardware resources

Mapping strategies include:

* **Single-core mapping** — entire kernel executes on a single core, using L1/L0/UB buffers.
* **Multi-core mapping** — tiles are distributed across cores.
* **Device-level mapping** — tasks span multiple devices (multi-NPU / multi-AI-core), coordinated via `pto.dist`.
* **NUMA-aware mapping** — for multi-socket hosts, memory locality considered.

The runtime or an external scheduler uses mapping metadata (e.g., `num_workers`, `target_device`, `affinity`) from `pto.program` / `pto.dist` to perform resource allocation.

### 2.4.3 Worker / Thread Model

A typical runtime instantiation:

* Each worker (process or thread) executes a **sequence of kernel launches** derived from `pto.program`.
* Workers may be homogeneous or heterogeneous; each worker has local buffers and may access remote host memory or remote device memory.
* Synchronization is handled by:

  * Global collectives in `pto.dist` (barriers, broadcasts).

### 2.4.4 Scheduling constraints

Schedulers must respect:

* **Data parallelism vs. compute resources** — avoid oversubscription.
* **Memory capacity** — onerous allocations to L1/L0/UB must be avoided; tile config must fit buffers.
* **Bandwidth & latency** — schedule DMA loads/stores within `pto.pipe` pipes to hide memory latency.
* **Ordering constraints from distributed primitives** — e.g., a `dist.shmem_put` followed by a `dist.barrier` has global ordering requirements.

---

## 2.5 Synchronization & visibility rules

Correctness across intra-core instruction pipes, intra-device cores and distributed devices/hosts relies on explicit synchronization semantics:

### Core domain (within a core)

* `pipe.exec.set`/`pipe.exec.wait` and `pipe.exec.barrier` define synchronization points between pipes; producers must signal completion before consumers proceed.
* Memory writes to shared buffers must be completed or synchronized before dependent reads (store-release / load-acquire semantics may be expressed as attributes or event ordering).

### Device domain (within a device)

* Runtie scheduler manages the dependencies among block-dispatch operations within a execution graph

### Global (across devices/nodes)

* `dist.barrier` provides global synchronization at specified program points. Collective ops require participation by all specified workers.
* Remote memory ops (`dist.shmem_put/get`) have annotations: `blocking`, `non_blocking` plus optional completion events. Ordering requirements are part of the op's contract and must be enforced in the lowering/runtime.

### Visibility guarantees

* Within a worker, consistency guarantees follow the hardware memory model chosen by backend (e.g., sequentially consistent for high-level correctness; relaxed models possible with explicit memory fences).
* Between workers, visibility is established by explicit `dist` synchronization and remote operation completion semantics.

---

## 2.6 Mapping control-flow & dynamic behavior

PTO-IR explicitly supports dynamic control flow and dynamic shapes. The execution model for these features:

### Dynamic control flow

* `if`, `for` constructs are lowered to SSA blocks plus phi nodes (as described in Section 6). The runtime executes loop bounds and branch conditions using scalar expressions evaluable at runtime.
* Loops with dynamic bounds produce iterators (`range(start, stop, step)`) where `stop` may be a dynamic scalar derived from `get_shape` or computed expressions.
* Lowerings must ensure correct codegen for variable trip counts (e.g., generate conditional branches or runtime loop prologues to handle partial tiles).

### Dynamic shapes

* A tensor shape may contain dynamic dimensions (`?` or symbolic). At runtime these are resolved via `get_shape` ops.
* Memory allocations for dynamic tensors may:

  * Allocate with exact runtime size (if runtime allocator supports it).
  * Allocate with an upper bound and maintain a mask/validity tensor to indicate active elements.
* Tiled loops where a dimension is dynamic compute `num_tiles = ceil(dim / tile_size)` at runtime and handle tail tiles explicitly.

---

## 2.7 Error handling and runtime checks

The execution model provides well-defined behavior for runtime errors:

* **Bounds checks**: optional; by default, lowered code can include runtime bounds checks for dynamic shapes or omit them for performance with the expectation the compiler enforces shape preconditions.
* **Allocation failure**: `alloc` returns an error status or raises an exception depending on runtime calling convention. Compilers may insert fallback paths.
* **Undefined behavior**: use-after-free, invalid remote memory accesses must not occur in well-formed IR; verification should catch these compile-time where possible.
* **Numerical issues**: NaN/Inf checks are optional and annotated as attributes (e.g., `sanitize_nan_inf`).

---

## 2.8 Performance metadata & hints

PTO-IR carries performance metadata through lowering to guide scheduling and code generation:

* **Tile hints**: preferred tile sizes for ops.
* **Pipe execution hints**: pipe latencies, concurrency hints, preferred pipe kinds.
* **Memory hints**: preferred memory space for intermediate values.
* **Cost models**: estimated compute and memory cost per op to guide auto-tiling / scheduling.
* **Affinity**: preference for certain cores or NUMA domains.

These are advisory and may be ignored by backends; however they are part of the IR and should be preserved or updated during lowering passes.

---

## 2.9 Interplay between lowering and autotuning

Because tile choices and schedule decisions strongly affect performance, the canonical flow often integrates autotuning:

1. Initial lowering with default tile hints.
2. Cost estimation or empirical profiling.
3. Auto-tuner explores alternative tile configs and produces variants (multiple versions of lowered code).
4. A final selection is made (offline or at runtime JIT) and compiled to `pto.inst`.

PTO-IR supports storing multiple variants and associated metadata in `pto.program` or module attributes.

---

## 2.10 Summary — Execution Model obligations

Implementations must ensure:

* **Correctness**: each lowering preserves semantics (types/shapes/SSA/dom).
* **Buffer fit**: tile configs chosen must fit physical buffers available on target hardware.
* **Explicit synchronization**: pipeline and distributed synchronizations are explicit in the IR; lowerings must produce matching runtime primitives.
* **Determinism (unless explicitly relaxed)**: lowering must not introduce nondeterminism except where semantics declare it.
* **Observability & Debug**: debug/loc info preserved to allow mapping runtime traces to source IR.

---

