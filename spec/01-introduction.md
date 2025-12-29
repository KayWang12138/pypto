# 1. Introduction

## 1.1 Purpose

PTO-IR (PTO Intermediate Representation) is a **multi-level, multi-dialect, SSA-based intermediate representation** for the PyPTO AI programming framework. It is designed to support:

* **Distributed execution** across devices and clusters
* **Tensor-level computation** with dynamic shapes
* **Tile-level computation** optimized for the capabilities of execution units and on-chip memory
* **Statement-level structured control flow** (structured statement tree) for explicit loop-carried state, conditional branches, and memory traffic control
* **Block-graph canonicalization** to capture reusable tile subgraphs
* **Low-level instructions** (PTO instructions) that target the hardware virtual ISA
* **Pipe schedules** that map tile ops to execution-unit instruction pipes
* **Execution graphs** that orchestrate block dispatch across execution units on a device
* **Extensibility**, allowing new operations, dialects, instructions, or analysis passes to be added
* **Multiple formats**: in-memory IR, textual IR, binary bytecode
* **Configuration dialect (`pto.config`)** for controlling transformation passes and runtime behavior
* **Platform abstractions dialect (`pto.platform`)** for modeling hardware topology and guiding optimizations
* **Debuggability** and source-code correlation throughout the lowering pipeline

The specification defines the **semantics**, **syntax**, **type system**, **dialects**, **operations**, and **execution model** of the PTO-IR, and targets high-performance training and inference workloads for modern accelerators.

---

## 1.2 Goals & Design Principles

PTO-IR is constructed around the following principles:

### 1. High-Performance Multi-Level Representation

The IR must express computation across levels:

1. **Distributed Level (dist dialect)**
   Work partitioning across nodes/devices with synchronization and remote memory ops.

2. **Program Level (program dialect)**
   Modules, functions, global objects.

3. **Function Level (func dialect)**
   SSA blocks, structured control flow.

4. **Statement Level (statement dialect)**
   Structured statement tree with control flow (for/if), loop nests, iteration domains, and memory operations. Each statement node can see all values defined in its ancestor statements.

5. **Tensor Level (tensor dialect)**
   High-level ops like matmul, attention, norm, reduction.

6. **Tile Level (tile dialect)**
   Tiled operations that map to execution unit capability.

7. **Block Graph Level (blockgraph dialect)**
   Canonicalized tile subgraphs that form dispatch units for execution unit.

8. **Instruction Level (instruction dialect)**
    Hardware virtual ISA, i.e. PTO instructions

9. **Pipe Execution Level (pipe dialect)**
   Per-core instruction pipe scheduling, buffer management, and synchronization.

10. **Execution Graph Level (exec dialect)**
   Global call graphs that manage dependencies and scheduling.

This hierarchy enables optimization at each layer.

---

### 2. Explicit Semantics, No Implicit Python Behavior

Like XLA/TVM/MLIR, values are **explicit**, **typed**, and **pure SSA**.
There is **no Python capture**, no closure semantics, and no hidden state.

---

### 3. Strong Type & Shape System

The IR supports:

* Static, dynamic, and symbolic shapes
* Memory layouts
* Tiling formats (ND, NZ, fractal, custom)
* Memory spaces (DDR, L1, L0A/B/C, UB, Register)
* Shape and type inference
* Compatibility validation

---

### 4. Multi-dialect system

The IR is subdivided into dialects, each with its own:

* Operations
* Types
* Attributes
* Lowering rules
* Verification rules

This allows independent evolution of distributed, tensor, tile, or pipe semantics.

---

### 5. Explicit Control Flow

The IR uses structured control flow via the `pto.statement` dialect:

* `statement.for` - Sequential for loops with explicit induction variables and loop-carried values via `iter_args`
* `statement.parallel_for` - Parallel for loops (independent iterations)
* `statement.if` - Conditional constructs with then/else regions
* `statement.yield` - Region terminator that returns values from current scope to parent statement
* `statement.block` - Linear basic blocks containing concrete operations
* `statement.return` - Function-level terminator

---

### 6. Hardware Awareness

The IR must express:

* Tile loads into execution unit's private buffers
* DMA operations
* Shared and remote memory semantics
* Core-level pipe execution schedules (`pto.pipe`)
* Runtime-level execution graphs (`pto.exec`)

This allows optimal scheduling on AI accelerators.

---

### 7. Dialect Extensibility and Custom Ops

Users can define:

* New dialects
* New ops
* Custom tile ops
* New PTO instructions

Without modifying core IR semantics.

---

### 8. Multi-format Representation

The IR supports:

* **In-Memory Format:** C++ or Python objects
* **Textual Format:** human-readable format
* **Binary Format:** Compact bytecode for fast loading & execution

Serialization rules are defined for all types, ops, attributes. The textual format is formally specified using **BNF (Backus-Naur Form) grammar**, providing:

* Unambiguous syntax definition for parser implementation
* Complete coverage of Module/Function/Block structures, Operations, Types, and SSA value references
* Support for automatic parser generation using tools like Yacc/Bison or ANTLR
* Reference documentation for IR text format validation

---

### 9. Verification & Well-Formedness Rules

The IR specifies:

* SSA dominance
* Type & shape correctness
* Tile correctness
* Control-flow correctness
* Memory safety (no-use-after-delete)
* Distributed consistency

---

## 1.3 Why Multi-Level IR?

This IR adopts a multi-level design because modern AI accelerators require multiple abstractions:

| Level        | Purpose                                              |
| ------------ | ---------------------------------------------------- |
| **Tensor**   | Graph-level optimization (fusion, layout transforms) |
| **Tile**     | Express compute patterns for MMUs/VMAC units         |
| **Block**    | Loop scheduling, memory tiling, buffer reuse         |
| **Pipe**     | Expose pipeline steps (load/compute/store)           |
| **PTO Inst** | Final hardware instructions                          |

This structure mirrors:

* XLA HLO → Thunks/RT
* TVM TE → TIR → TensorIR → CodeGen
* MLIR Linalg → Affine → Vector → NVVM
* CUDA Graph → PTX → SASS

This ensures both high-level expressiveness and low-level control.

---

## 1.4 Scope of the Specification

This specification defines:

### Included

* IR semantics & structure
* Dialects & operations
* Type and shape system
* Memory model
* SSA rules
* Control flow
* Tile semantics
* Distributed execution model
* Compile-pass lowering pipeline (tensor → tile → blockgraph → pipe → execution → instruction)
* Serialization formats (textual, binary, in-memory)
* **Formal BNF grammar** for textual format syntax
* Verification & invariants

### Excluded

* Python or C++ frontend syntax
* Hardware ISA details (covered in PTO Instruction Set Architecture)
* Tuning algorithms
* Runtime APIs (covered in PTO Runtime Spec)

---

## 1.5 Relation to Other Systems

PTO-IR draws from:

* **MLIR**: multi-dialect, SSA, region/block structure
* **XLA**: HLO fusion, layout propagation
* **TVM**: tensor expression + TIR + schedule separation
* **Halide**: algorithm/schedule separation
* **CUTLASS/Triton**: tile-level computation
* **OpenSHMEM**: distributed memory operations

However, PTO-IR integrates these concepts into **one unified multi-level IR** rather than separate representations.

PTO-IR can be translated from or to other system to provide inter-operationbility, to be used as front-end or back-end of other frameworks.

---

## 1.6 High-Level IR Layer Stack

```
 ┌─────────────────────────────┐
 │     Distributed Layer       │  (pto.dist)
 └───────────────┬─────────────┘
                 │
 ┌───────────────▼─────────────┐
 │     Program / Function      │  (pto.program, pto.func)
 └───────────────┬─────────────┘
                 │
 ┌───────────────▼─────────────┐
 │      Statement Layer        │  (pto.statement)
 └───────────────┬─────────────┘
                 │
 ┌───────────────▼─────────────┐
 │        Tensor Layer         │  (pto.tensor)
 └───────────────┬─────────────┘
                 │
 ┌───────────────▼─────────────┐
 │         Tile Layer          │  (pto.tile)
 └───────────────┬─────────────┘
                 │
 ┌───────────────▼─────────────┐
 │      Block Graph Layer      │  (pto.blockgraph)
 └───────────────┬─────────────┘
                 │
 ┌───────────────▼─────────────┐
 │        PTO Instructions     │  (pto.inst)
 └───────────────┬─────────────┘
                 │
 ┌───────────────▼─────────────┐
 │     Pipe Execution Layer    │  (pto.pipe)
 └───────────────┬─────────────┘
                 │
 ┌───────────────▼─────────────┐
 │     Execution Graph Layer   │  (pto.exec)
 └───────────────┬─────────────┘

```

Each layer provides lowering rules to the next.

---

## 1.7 Complete List of Dialects Defined in PTO-IR

| Dialect         | Purpose                                   |
| --------------- | ----------------------------------------- |
| **pto.program** | Program/module/global structure           |
| **pto.func**    | Functions, blocks, CFG                    |
| **pto.statement**   | Structured statement tree, control flow (for/if), loop nests, memory ops |
| **pto.scalar**  | Scalar arithmetic                         |
| **pto.tensor**  | Tensor ops (matmul, add, reshape…)        |
| **pto.tile**    | Tiled tensor ops                          |
| **pto.blockgraph**  | Canonical tile subgraphs (Stage 3)     |
| **pto.inst**    | Low-level hardware instructions (Stage 4)          |
| **pto.pipe**   | Per-core pipe schedules (Stage 4)      |
| **pto.exec**    | Execution graphs / block dispatch (Stage 5) |
| **pto.dist**    | Distributed execution + shared memory ops |
| **pto.mem**     | Memory spaces, allocations, DMA ops       |
| **pto.debug**   | Debug and source location metadata        |
| **pto.config**  | Transformation pass configuration         |
| **pto.platform**| Platform abstractions (hardware modeling) |

Each dialect is self-contained and layered. `pto.config` and `pto.platform` are meta-dialects that do not participate in lowering but guide transformation passes.

---

## 1.8 Document Overview

This specification is organized as:

1. **Introduction** (`01-introduction.md`)
2. **Execution Model & IR Levels** (`02-execution-model.md`)
3. **Dialect Overview** (`03-dialect-overview.md`)
4. **Type & Shape System** (`04-type-shape-system.md`)
5. **Program & Function Dialects** (`05-program-function-dialects.md`)
6. **Statement Dialect** (`06-statement-dialect.md`)
7. **Tensor Dialect** (`07-tensor-dialect.md`)
8. **Tile Dialect** (`08-tile-dialect.md`)
9. **Block Graph Dialect** (`09-block-graph-dialect.md`)
10. **Instruction Dialect** (`10-instruction-dialect.md`)
11. **Pipe Dialect** (`11-pipe-dialect.md`)
12. **Execution Dialect** (`12-execution-dialect.md`)
13. **Distributed Dialect** (`13-distributed-dialect.md`)
14. **Memory Dialect** (`14-memory-dialect.md`)
15. **Debug Dialect** (`15-debug-dialect.md`)
16. **Configuration Dialect** (`16-configuration.md`)
17. **Platform Abstractions Dialect** (`17-platform-abstractions.md`)
18. **Serialization** (`18-serialization.md`)
19. **Verification** (`19-verification.md`)
20. **Examples** (`20-examples.md`)

---

