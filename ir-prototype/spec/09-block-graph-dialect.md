# 9. Block Graph Dialect

The `pto.blockgraph` dialect canonically represents isomorphic tile subgraphs that arise after partitioning the tile graph. Each block graph captures a reusable pattern of tile operations, data flow, and dependency structure which can be instantiated many times by the execution graph.

---

## 9.1 Position in the Lowering Pipeline

* **Input:** Lowered tile graph (Section 7) annotated with tiling directives from `pto.config`.
* **Process:** Partition tile graph into colored subgraphs → detect isomorphic subgraphs → emit canonical block graph definitions.
* **Output:** `pto.blockgraph` IR consumed by the pipe-execution dialect (`pto.pipe`) and the execution dialect (`pto.exec`).

This is Stage 3 of the compile pass pipeline described in Section 2. Each block graph:

1. Encapsulates exactly one equivalence class of isomorphic tile subgraphs.
2. Contains the full set of tile ops, buffers, and dependencies required to execute the pattern once.
3. Is emitted as a `pto.func` definition that later feeds the execution dialect (`pto.exec`) for scheduling/dispatch.

---

## 9.2 Purpose

The block graph dialect serves to:

* **Deduplicate work:** Capture repeating tile patterns once and reuse via calls.
* **Define dispatch units:** Provide the units that schedulers submit to execution units.
* **Preserve dependencies:** Record intra-block producer/consumer relationships needed for later synchronization.
* **Bridge stages:** Form the boundary between algorithmic tiling and device pipe execution.

---

## 9.3 Core Operations

### 9.3.1 `blockgraph.block`

Define a canonical block graph.

```
blockgraph.block @mmu_block(
    %input_A: tile<16x16xf16, NZ>,
    %input_B: tile<16x16xf16, NZ>)
    -> (tile<16x16xf16, NZ>) {
  %acc = blockgraph.init tile<16x16xf16, NZ>
  %acc1 = tile.matmul %input_A, %input_B : ...
  blockgraph.yield %acc1
}
```

**Attributes:**

* `color`: identifier of the partition color that produced this block.
* `key`: hash of the subgraph structure (used by isomorphism detection).

### 9.3.2 `blockgraph.region`

Encapsulate a connected component inside a block to preserve original topology (optional when multiple independent components exist).

### 9.3.3 `blockgraph.bind_tile`

Map a canonical tile parameter to one or more concrete tiles in the original tile graph.

```
blockgraph.bind_tile @mmu_block %A_tiles = [%tile0, %tile24]
```

### 9.3.4 `blockgraph.dep`

Explicitly record producer/consumer relationships between operations inside the block. These become synchronization edges during pipe execution placement.

```
blockgraph.dep %load_A -> %matmul : {type = "RAW"}
```

Types: `RAW`, `WAR`, `WAW`.

### 9.3.5 `blockgraph.meta`

Attach metadata such as tiling configuration snapshot, layout constraints, or perf counters that were used during pattern detection. The metadata is preserved for debugging and analysis passes.

---

## 9.4 Types

The dialect reuses existing tile/memref types and introduces the following structural types:

* `blockgraph.block_type`: identifies block signatures (inputs, outputs, invariants).
* `blockgraph.dep_type`: enumerates dependency kinds (`RAW`, `WAR`, `WAW`).
* `blockgraph.binding_list`: structure describing which tile IDs are bound to each parameter.

---

## 9.5 Interaction with Other Dialects

* Consumes tile ops; no new computation is introduced.
* References configuration hints (e.g., `config.tile_strategy`) to document how the partition was produced.
* Annotates node-level platform expectations (e.g., required execution unit type) but does not bind to concrete hardware.
* Lowers exclusively to `pto.pipe` (Stage 4) where tile ops are placed onto device pipes.
* Instantiated by the execution dialect (`pto.exec.call`) with runtime tile bindings.

---

## 9.6 Lowering Guidelines

1. **Partitioning:** Tile graphs are colored using a deterministic algorithm (e.g., BFS-based coloring). Each color corresponds to one candidate block.
2. **Isomorphism detection:** Subgraphs with identical topology, op types, and tile configurations are merged into a single `blockgraph.block`.
3. **Canonical ordering:** Operations inside `blockgraph.block` must follow SSA dominance order to make downstream analysis deterministic.
4. **Boundary materialization:** Inputs/outputs become explicit block arguments/results so that `exec.call` can provide buffers.

---

## 9.7 Verification Rules

1. Every `blockgraph.block` must have a unique `(color, key)` pair.
2. All referenced tile ops must originate from the tile graph partition that produced the block.
3. Dependencies must reference existing ops and use valid dependency types.
4. Bindings must cover every tile parameter when the block is instantiated.
5. No side effects that escape the block are allowed (e.g., global memory writes) unless they are modeled as explicit results.

---

## 9.8 Examples

### Simple Block Graph

```
blockgraph.block @mm_block(%A: tile<16x16xf16, NZ>,
                          %B: tile<16x16xf16, NZ>)
    -> (tile<16x16xf16, NZ>) {
  %acc = tile.matmul %A, %B
  blockgraph.yield %acc
}
```

### Block with Multiple Regions and Dependencies

```
blockgraph.block @load_compute_store(
    %A: memref<16x16xf16, ND, L1>,
    %B: memref<16x16xf16, ND, L1>)
    -> (memref<16x16xf16, ND, L1>) {
  %tile_A = tile.load_tile %A
  %tile_B = tile.load_tile %B
  blockgraph.dep %tile_A -> %tile_B : {type = "RAW"}
  %tile_C = tile.matmul %tile_A, %tile_B
  %out = tile.store_tile %tile_C, %B
  blockgraph.yield %out
}
```

### Binding Metadata

```
blockgraph.bind_tile @mm_block %A_tiles = [%tile_0_0, %tile_0_16]
blockgraph.bind_tile @mm_block %B_tiles = [%tile_16_0, %tile_16_16]
```

---

## 9.9 Summary

The block graph dialect captures reusable tile subgraphs immediately after partitioning. It enables:

* Precise accounting of dependencies and tile bindings.
* Efficient reuse through canonical block definitions.
* A clean interface to pipe execution placement (Stage 4) and execution graph scheduling (Stage 5).

By modeling blocks explicitly, PTO-IR cleanly separates tiling decisions from hardware-specific scheduling and runtime orchestration.

---

