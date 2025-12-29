# PTO-IR Specification

**PTO Intermediate Representation (PTO-IR) Specification v2**

A Multi-Level, Multi-Dialect Intermediate Representation for AI Kernel Programming and Distributed Execution

---

## Overview

PTO-IR is a comprehensive intermediate representation designed for AI programming frameworks targeting tile-based tensor computation, on-chip pipelines, and distributed execution. It supports multiple levels of abstraction from high-level tensor operations down to hardware-specific instructions.

---

## Table of Contents

1. [Introduction](01-introduction.md) - Purpose, goals, and design principles
2. [Execution Model & IR Levels](02-execution-model.md) - Execution semantics and IR layering
3. [Dialect Overview](03-dialect-overview.md) - Overview of all dialects
4. [Type & Shape System](04-type-shape-system.md) - Type system and shape inference
5. [Program & Function Dialects](05-program-function-dialects.md) - Program structure and SSA functions
6. [Tensor Dialect](06-tensor-dialect.md) - High-level tensor operations
7. [Tile Dialect](07-tile-dialect.md) - Tile-level operations
8. [Statement Dialect](08-statement-dialect.md) - Loop nests and memory operations
9. [Block Graph Dialect](09-block-graph-dialect.md) - Canonical tile subgraphs
10. [Pipe Execution Dialect](10-pipe-execution-dialect.md) - Per-core pipe scheduling
11. [Execution Dialect](11-execution-dialect.md) - Host-level execution graphs
12. [Distributed Dialect](12-distributed-dialect.md) - Multi-device execution
13. [Memory Dialect](13-memory-dialect.md) - Memory management
14. [Instruction Dialect](14-instruction-dialect.md) - Hardware instructions
15. [Debug Dialect](15-debug-dialect.md) - Debugging and source correlation
16. [Configuration Dialect](16-configuration.md) - Transformation pass configuration and hints
17. [Platform Abstractions Dialect](17-platform-abstractions.md) - Execution units, memory hierarchy, and distributed structure
18. [Serialization](18-serialization.md) - Textual and binary formats
19. [Verification](19-verification.md) - Well-formedness rules
20. [Examples](20-examples.md) - Comprehensive examples

---

## Key Features

### Multi-Level Abstraction

- **Distributed Level**: Multi-device and multi-node execution
- **Program Level**: Modules, functions, global objects
- **Tensor Level**: High-level tensor algebra
- **Tile Level**: Hardware-aware tile operations
- **Statement Level**: Loop nests and memory access
- **Block Graph Level**: Canonicalized tile subgraphs
- **Pipe Execution Level**: Per-core pipe scheduling and synchronization
- **Execution Graph Level**: Host/runtime scheduling and dispatch
- **Instruction Level**: Hardware-specific instructions

### Compile Pass Lowering Pipeline

Lowering is stage-based: each stage has a deterministic allow/deny list of dialects, and a stage boundary is crossed only when all forbidden dialects have been eliminated.

1. **Tensor Graph Stage** – Allowed: `pto.program`, `pto.func`, `pto.tensor`, `pto.statement`, `pto.dist`, `pto.config`, `pto.debug`. All tiling/blockgraph/pipe/exec dialects are forbidden.
2. **Tile Graph Stage** – Allowed: Tensor-stage dialects plus `pto.tile` (and limited `pto.mem`). Block graph, pipe execution, and execution graph dialects remain forbidden.
3. **Block Graph Stage** – Allowed: `pto.blockgraph` with meta-dialects. All tensor/tile/statement/pipe/exec ops must be absent (only referenced as metadata).
4. **Pipe Execution Stage** – Allowed: `pto.pipe` plus blockgraph references and meta-dialects. No tensor/tile/statement/exec ops may appear here.
5. **Execution Graph Stage** – Allowed: `pto.exec`, `pto.dist`, and meta-dialects with symbolic references to block graphs. Device-level dialects are forbidden.
6. **Code Generation Stage** – Allowed: `pto.inst` (for device binaries) and generated runtime/scheduler code derived from `pto.exec`. All higher-level dialects must have been resolved.

### Dialect System

PTO-IR uses a multi-dialect architecture:

- `pto.program` - Program and module structure
- `pto.func` - SSA functions and control flow
- `pto.scalar` - Scalar arithmetic
- `pto.tensor` - Tensor operations
- `pto.tile` - Tile operations
- `pto.statement` - Loop and memory operations
- `pto.blockgraph` - Canonical tile subgraphs
- `pto.pipe` - Per-core pipe scheduling
- `pto.exec` - Execution graph coordination
- `pto.dist` - Distributed execution
- `pto.mem` - Memory management
- `pto.inst` - Hardware instructions
- `pto.debug` - Debug metadata
- `pto.config` - Transformation pass configuration
- `pto.platform` - Platform abstractions

### Type System

- Static, dynamic, and symbolic shapes
- Multiple memory spaces (DDR, L1, UB, L0A/B/C, REG)
- Tile formats (ND, NZ, FRACTAL_Z)
- Layout support (NHWC, NCHW, etc.)

### Formats

- **In-Memory**: C++/Python data structures
- **Textual**: Human-readable MLIR-like format
- **Binary**: Compact bytecode format

### Meta-Dialects

- **Configuration Dialect (`pto.config`)**: Fine-grained control over transformation passes
- **Platform Abstractions Dialect (`pto.platform`)**: Model execution units, memory hierarchy, and interconnections
- **Auto-Tuning Support**: Automatic optimization exploration via configuration dialect

---

## Quick Start

### Example Program

```mlir
pto.program.module @main {
  pto.program.entry @matmul_kernel
  
  pto.func.func @matmul_kernel(%A: tensor<1024x512xf16>, %B: tensor<512x256xf16>)
      -> (tensor<1024x256xf16>) {
    %C = pto.tensor.matmul %A, %B : tensor<1024x512xf16>, tensor<512x256xf16> -> tensor<1024x256xf16>
    pto.func.return %C
  }
}
```

---

## Design Principles

1. **Multi-Level Representation**: Support computation at all abstraction levels
2. **Explicit Semantics**: No implicit behavior, all operations explicit
3. **Strong Type System**: Comprehensive type and shape system
4. **Dialect Isolation**: Dialects are self-contained with clear interfaces
5. **Hardware Awareness**: Express hardware-specific optimizations
6. **Extensibility**: Support custom operations and dialects
7. **Verification**: Strong verification and well-formedness guarantees

---

## Related Systems

PTO-IR draws inspiration from:

- **MLIR**: Multi-dialect architecture and SSA form
- **XLA**: High-level optimizations and layout propagation
- **TVM**: Tensor expressions and scheduling
- **Halide**: Algorithm/schedule separation
- **CUTLASS/Triton**: Tile-level computation
- **OpenSHMEM**: Distributed memory operations

---

## Specification Status

This is version 2 of the PTO-IR specification. The specification is organized into individual markdown files for each major section, making it easy to navigate and maintain.

---

## Contributing

When contributing to the specification:

1. Follow the existing structure and format
2. Ensure examples are complete and correct
3. Maintain consistency across sections
4. Update the table of contents if adding new sections

---

## License

[Specify license here]

---

