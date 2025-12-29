# 12. Execution Dialect

The `pto.exec` dialect represents Stage 5 of the lowering pipeline: the execution graph that orchestrates block graph invocations, tracks dependencies among calls, and performs global resource planning (buffer allocation, launch ordering, etc.). It bridges high-level scheduling decisions and the runtime that manages execution units.

---

## 12.1 Purpose

* **Execution graph modeling:** Encode the DAG of block invocations (calls) and their dependencies.
* **Global resource planning:** Allocate global memory buffers, shared scratchpads, and runtime handles.
* **Scheduler hints:** Provide information so the runtime dispatcher can pick which execution unit runs which block when dependencies are satisfied.
* **Optimization stage:** Enable passes that prune dependencies, fuse compatible calls, or reorder execution for better utilization.

---

## 12.2 Core Operations

The execution graph is embodied in an `exec.graph` region, which behaves like a `func.func`. Its body is a `statement.block` whose sequence consists of `exec.call` operations. Dependencies are inferred from tensor outputs and inputs among those calls.

### 12.2.1 `exec.call`

An `exec.call` is a call into an executable function symbol (typically a `blockgraph.block` that has been lowered into a function). Each call maps block arguments to concrete tiles and can tag which core should execute the invocation.

```
%call = exec.call @mm_block(%A0: tile<16x16xf16, NZ>,
                                  %B0: tile<16x16xf16, NZ>)
    -> (tile<16x16xf16, NZ>) {
  color = 3,
  priority = 5
}
```

Attributes/Hints:

* `priority`: scheduling priority.
* `core_group`: subset of execution units eligible to run the block.
* `target_core`: specific execution core (or affinity slot) where the call should land.
* `repeat`: iteration count when the same call repeats over ranges of tiles.

### 12.2.2 `exec.dependency`（TODO）

Define a dependency edge between two calls.

```
exec.dependency %call_A -> %call_B {type = "data", distance = 1}
```

Types:

* `data`: producer → consumer tile dependency.
* `order`: required ordering (e.g., due to shared resource).
* `control`: generic sequencing constraint.

### 12.2.3 `exec.fence`

Insert a global synchronization point used for multi-device coordination or to flush outstanding operations.

### 12.2.4 `exec.allocate` / `exec.deallocate`

Manage global buffers owned by the execution graph.

```
%scratch = exec.allocate : memref<32768xf16, ND, DDR>
...
exec.deallocate %scratch
```

### 12.2.5 `exec.schedule_hint`

Attach hints consumed by the runtime scheduler (e.g., batching size, fairness policy, preferred NUMA domain).

---

## 12.3 Interaction with Other Dialects

* References `blockgraph.block` symbols for call targets.
* Consumes distributed annotations from `pto.dist` (e.g., worker IDs) to build multi-device graphs.
* Consults `pto.config` for placement policies (e.g., `config.partitioning`).
* Emits global memory allocation requests executed by the memory dialect (`pto.mem`).
* Provides inputs to the runtime code generator that builds a scheduler around the execution graph.

---

## 12.4 Lowering & Code Generation

1. **Input:** Block graphs produced by Stage 3 and pipe-execution programs from Stage 4. The execution graph determines *when* to invoke each block, while pipe-exec determines *how* a block runs on a core.
2. **Optimization passes:** Remove redundant dependencies, coalesce compatible calls, insert fences for distributed boundaries, allocate global buffers, and propagate cost models.
3. **Runtime generation:** Stage 6 uses `exec.graph` to emit host/runtime code that:
   * Tracks outstanding dependencies.
   * Dispatches `blockgraph` instances to execution units when ready.
   * Performs global memory management according to `exec.allocate`.

---

## 12.5 Verification Rules

1. Execution graph must be acyclic unless explicitly marked as allowing cycles (e.g., streaming loops with `repeat` semantics).
2. Every `exec.call` must correspond to a real `func.func` definition (either directly or via the referenced block) so that every call is resolvable to executable code.
3. All dependencies must reference calls within the same `exec.graph`.
4. Allocated buffers must be deallocated or marked as persistent at graph exit.
5. Hints must be compatible with platform capabilities (e.g., `core_group` must exist on the target platform).

---

## 12.6 Examples

### Simple Execution Graph

```
pto.func @matmul_pipeline scope = "device" {
  statement.block {
    %A_buf = exec.allocate : memref<262144xf16, ND, DDR>
    %B_buf = exec.allocate : memref<262144xf16, ND, DDR>

    %call0 = exec.call @mm_block(inputs = [%A_tiles0, %B_tiles0],
                                 outputs = [%C_tiles0]) {
      core_group = "mmu0"
    }
    %call1 = exec.call @mm_block(inputs = [%C_tiles0, %B_tiles1],
                                 outputs = [%C_tiles1]) {
      core_group = "mmu1"
    }
  }
}
```

### Distributed Graph

```
exec.graph @encoder scope = "cluster" {
  %call_shard0 = exec.call @mm_block(... ) {worker = 0}
  %call_shard1 = exec.call @mm_block(... ) {worker = 1}
  exec.dependency %call_shard0 -> %call_shard1 {type = "control"}
  exec.fence {workers = [0, 1]}
}
```

---

## 12.7 Summary

The execution dialect lifts block-level schedules to a global view, enabling powerful graph-level optimizations and providing the runtime with a precise dependency DAG. Together with `pto.pipe`, it completes the bridge between tile-level computation and executable code.

---

