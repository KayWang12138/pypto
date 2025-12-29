# PTO-IR Feature Matrix

This matrix captures, at a high level, the responsibilities and relationships of each dialect defined in the PTO-IR specification. It is a living document; update it as the spec evolves.

| Dialect | Representative Operations | Result / Operand Types | Core Invariants | Lowering / Consumption |
| --- | --- | --- | --- | --- |
| `pto.program` | `program.module`, `program.global`, `program.entry`, `program.launch` | `i32`, descriptors | Exactly one root module, valid entrypoints/devices | Lowers to `pto.func` + `pto.dist` |
| `pto.func` | `func.func`, `func.call`, `func.return`, `cf.*`, `scf.*` | SSA values of any dialect | SSA dominance, block terminators, matching signatures | Hosts tensor/tile/statement ops |
| `pto.scalar` | `scalar.add`, `scalar.mul`, comparisons, casts | scalar types (`i*`, `f*`, `bool`, `index`) | Pure ops, no side effects | Used by all other dialects |
| `pto.tensor` | `tensor.matmul`, `tensor.conv`, `tensor.add`, `tensor.reshape`, `tensor.dim` | `tensor<shape, element, layout?>` | Shape/layout correctness, no aliasing | Lowers to `pto.tile` via tiling passes |
| `pto.tile` | `tile.matmul`, `tile.load_tile`, `tile.store_tile`, `tile.set_format` | `tile<dims, elem, format>` | Tile dims consistent, formats supported, carries `tiling_mode` from `pto.config` | Lowers to `pto.statement` loops and feeds `pto.blockgraph` |
| `pto.statement` | `statement.for`, `statement.parallel_for`, `statement.load/store`, `statement.alloc` | `memref<shape, elem, layout, space>` | Loop bounds valid, memory accesses within space, leaves sequential | Partitioned into `pto.blockgraph` |
| `pto.blockgraph` | `blockgraph.block`, `blockgraph.dep`, `blockgraph.bind_tile`, `blockgraph.meta` | `blockgraph.block` | Unique `(color,key)`, deps cover RAW/WAR/WAW, side-effects explicit; canonical block IDs emitted by BlockGraph pass | Consumed by `pto.pipe` and `pto.exec` |
| `pto.pipe` | `pipe.exec.core`, `pipe.exec.pipe`, `pipe.exec.assign`, `pipe.exec.set/wait/barrier`, `pipe.exec.buffer` | `pipe.exec.schedule`, events, buffers | Each op assigned to one core/pipe, events single producer, buffer lifetime obeyed; placement pass emits `assign`/`barrier` scaffolding | Lowers to `pto.inst`; referenced by `pto.exec` |
| `pto.exec` | `exec.graph`, `exec.call`, `exec.dependency`, `exec.allocate`, `exec.fence`, `exec.schedule_hint` | `exec.graph`, handles | Graph acyclic (unless streaming), dependencies intra-graph, resources deallocated; ExecutionGraph pass emits call/dependency/alloc ops | Drives runtime scheduler + codegen |
| `pto.dist` | `dist.launch`, `dist.shard_tensor`, `dist.allreduce`, `dist.shmem_put/get`, `dist.barrier` | `dist.tensor<layout>` | Participant counts match, remote memory valid | Interacts with program/exec stages |
| `pto.mem` | `mem.alloc`, `mem.free`, `mem.copy`, `mem.dma_copy`, `mem.set_layout` | `memory.buffer<space>` | Proper lifetimes, DMA bounds legal | Used across tile/statement/pipe stages |
| `pto.inst` | `inst.mma`, `inst.load_*`, `inst.store`, `inst.barrier`, `inst.branch`, `inst.atomic_*` | hardware registers, `inst.*` | All operands bound, schedule constraints satisfied; Codegen pass generates load/mma/store bundles per block | Final device binary |
| `pto.debug` | `debug.source_loc`, `debug.trace`, `debug.meta` | metadata | No semantic impact, attached to valid ops | Propagates through lowering |
| `pto.config` | `config.tile_strategy`, `config.partitioning`, `config.scheduling`, `config.memory_allocation`, `config.optimization`, `config.autotuning` | config structs | Values valid for platform, conflicts resolved | Consumed by all transformation passes |
| `pto.platform` | `platform.execution_unit`, `platform.memory`, `platform.interconnect`, `platform.numa_domain`, `platform.cluster`, `platform.platform` | platform descriptors | Model consistency, valid capabilities | Queried by tile/blockgraph/pipe/exec |

### Type Requirements Snapshot

| Type | Description | Key Fields / Constraints |
| --- | --- | --- |
| Scalar (`i*`, `f*`, `bool`, `index`) | Primitive numerical types | Standard ranges |
| Tensor | `tensor<dims x ... x elem, layout?>` | Static or `?` dynamic dims, layout hints |
| Tile | `tile<W x H x ... x elem, format>` | Fixed dims, format (ND, NZ, etc.) |
| MemRef | `memref<shape, elem, layout, space>` | Memory space (DDR/L1/UB/L0*/REG) |
| Block | `blockgraph.block` | Canonical subgraph handle |
| Pipe schedule | `pipe.exec.schedule` | Ordered instructions per pipe |
| Exec graph | `exec.graph` | DAG of block invocations |
| Distributed tensor | `dist.tensor<strategy>` | Strategy details (replicated, sharded) |
| Memory buffer | `memory.buffer<space>` | Tracks allocation region |
| Instruction | `inst.*` | ISA-specific operands |

### Serialization & Grammar

| Feature | Description | Specification Location |
| --- | --- | --- |
| **Textual Format** | MLIR-like human-readable syntax | Section 18.2 |
| **BNF Grammar** | Formal syntax definition using Extended BNF (EBNF) | Section 18.2.4 |
| **Binary Format** | Compact bytecode encoding | Section 18.3 |
| **In-Memory Format** | C++/Python data structures | Section 18.1.1 |
| **Grammar Coverage** | Module/Function/Block, Operations, Types, SSA values, Attributes | Section 18.2.4 |

The BNF grammar provides:
- Complete syntax specification for parser implementation
- Support for automatic parser generation (Yacc/Bison, ANTLR)
- Unambiguous definition of IR text format
- Reference for validation and error reporting

Update this matrix whenever a dialect/type gains new capabilities.

