# 11. Pipe Dialect(TODO)

The `pto.pipe` dialect models Stage 4 of the compile pipeline: mapping block-graph tile operations onto the instruction pipes that exist inside an execution unit (core). Each core exposes multiple pipes (DMA/data-move, compute, reduction, store, etc.) plus synchronization primitives (`set`, `wait`, `barrier`). `pto.pipe` captures the per-core schedule, buffer management, and synchronization required before code generation.

---

## 11.1 Purpose

* **Placement:** Assign every tile op inside a `blockgraph.block` to a concrete pipe kind on a target execution unit.
* **Synchronization:** Insert explicit `set`, `wait`, and `barrier` operations that encode RAW/WAR/WAW dependencies.
* **Buffer management:** Model allocation, reuse, and spilling of on-core private buffers.
* **Spill/fill:** Emit transfers back to global memory when local storage is insufficient.
* **Preparation for codegen:** Provide a linearized, per-pipe schedule that can be directly converted to hardware instructions (Stage 6).

---

## 11.2 Core Operations

### 11.2.1 `pipe.pipe`

Define the instruction sequence for one pipe.

```
pipe.pipe "dma" {
  pipe.assign %tile_A = tile.load_tile %global[%tile_id]
  pipe.set %event0
}
```

Semantics:

* Instructions inside a pipe execute sequentially.
* Different pipes run concurrently subject to synchronization.

### 11.2.2 `pipe.assign`

Associate a tile operation from the block graph with a specific pipe.

```
pipe.assign "compute" %acc = tile.matmul %tile_A, %tile_B
```

Attributes capture the originating `blockgraph` op for traceability.

### 11.2.3 `pipe.wait` / `pipe.set`

Synchronization primitives.

```
pipe.set "load" %event0
pipe.wait "compute" %event0
```

* `set` produces an event token.
* `wait` blocks the issuing pipe until the event is satisfied.

### 11.2.4 `pipe.barrier`

Multi-pipe barrier.

```
pipe.barrier {pipes = ["dma", "compute", "store"]}
```

Ensures all listed pipes reach the barrier before any continue.

### 11.2.5 `pipe.buffer`

Manage per-pipe (or shared) buffers.

```
%buf = pipe.buffer {space = L0A, size = 4096, reuse = true}
```

### 11.2.6 `pipe.spill`

Spill tile data to global memory (or refill from global memory) when local capacity is insufficient.

```
pipe.spill %buf -> %global : memref<16x16xf16, ND, DDR>
```

---

## 11.3 Types

* `pipe.event` — token produced by `pipe.set`, consumed by `pipe.wait`.
* `pipe.buffer_type` — describes buffer size, memory space, and reuse policy.
* `pipe.pipe_type` — identifies pipe kind (`dma`, `compute`, `reduce`, `store`, `custom`).

---

## 11.4 Lowering Inputs & Outputs

* **Input:** `blockgraph.block` plus platform information (`pto.platform`) that lists available pipes per execution unit.
* **Output:** Fully scheduled per-pipe programs ready for instruction selection (`pto.inst`).

Lowering steps:

1. **Dependency expansion:** Convert `blockgraph.dep` edges into synchronization requirements (RAW → wait on producer; WAR/WAW → enforce ordering or insert barriers).
2. **Pipe selection:** Use configuration hints (`config.scheduling`, `config.memory_allocation`) and platform capabilities to choose pipes.
3. **Buffer planning:** Allocate on-core buffers, insert spill/fill if necessary.
4. **Event wiring:** Insert `pipe.set`/`wait` so that dependencies hold across pipes.

---

## 11.5 Verification Rules

1. Every `pipe.core` must reference a valid `blockgraph.block`.
2. Each tile op from the block graph must be assigned to exactly one pipe.
3. Event tokens must have a single producer (`pipe.set`) and can have multiple consumers (`pipe.wait`).
4. Buffers cannot be used outside their declared lifetime.
5. Spill operations must reference valid memory spaces.

---

## 11.6 Examples

### Dual-Pipe Example

```
pipe.core @core0(block = @mm_block, unit = "MMU") {
  pipe.pipe "dma" {
    %event_load = pipe.set
    pipe.assign %tile_A = tile.load_tile %A[%tid] {bind = %arg0}
    pipe.assign %tile_B = tile.load_tile %B[%tid] {bind = %arg1}
    pipe.set %event_dma = %event_load
  }

  pipe.pipe "compute" {
    pipe.wait %event_dma
    pipe.assign %acc = tile.matmul %tile_A, %tile_B
    pipe.set %event_compute
  }

  pipe.pipe "store" {
    pipe.wait %event_compute
    pipe.assign tile.store_tile %acc, %C[%tid]
  }
}
```

### Spill Example

```
pipe.pipe "dma" {
  %buf = pipe.buffer {space = L0B, size = 2048, reuse = false}
  pipe.assign %tile = tile.load_tile %A[%idx]
  pipe.spill %buf -> %global_A
}
```

---

## 11.7 Summary

`pto.pipe` captures the resource-constrained scheduling of tile operations on a core. It is the final IR before instruction selection and ensures that:

* All dependencies are represented by explicit synchronization.
* Buffer usage and spills are visible to later passes.
* Mapping from logical tile ops to hardware pipes is fully described.

---

