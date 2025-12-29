# 14. Memory Dialect(TODO)

The `pto.mem` dialect provides explicit memory management operations, including allocation, deallocation, copying, and DMA operations across different memory spaces.

---

## 14.1 Purpose

The memory dialect serves to:

* Explicitly manage memory allocations
* Express memory transfers (DMA operations)
* Represent memory layout transformations
* Enable memory-level optimizations

---

## 14.2 Core Operations

### 14.2.1 Allocation Operations

#### `mem.alloc`

Allocate memory buffer.

**Syntax:**
```
%memref = mem.alloc {size = 1024, memory_space = L1, alignment = 16}
    : memref<1024xf16, ND, L1>
```

**Attributes:**
- `size`: allocation size in elements
- `memory_space`: target memory space (DDR, L2, L1, UB, L0A/B/C, REG)
- `alignment`: memory alignment requirement

---

#### `mem.free`

Deallocate memory buffer.

**Syntax:**
```
mem.free %memref : memref<...xf16, ND, L1>
```

**Lifetime:**
- Must be paired with `mem.alloc`
- Verification ensures no use-after-free

---

### 14.2.2 Memory Transfer Operations

#### `mem.copy`

Copy data between memory regions.

**Syntax:**
```
mem.copy %src, %dst : memref<...xf16>, memref<...xf16>
```

**Semantics:**
- Synchronous copy operation
- Source and destination must have compatible types
- Size must match

---

#### `mem.dma_copy`

DMA (Direct Memory Access) copy operation.

**Syntax:**
```
%token = mem.dma_copy %src, %dst {async = true} 
    : memref<...xf16, ND, DDR>, memref<...xf16, ND, L1> -> dma_token
```

**Attributes:**
- `async`: whether operation is asynchronous
- Returns completion token if async

**Use cases:**
- Transfer from DDR to on-chip memory
- Overlap computation with memory transfer

---

### 14.2.3 Layout Operations

#### `mem.set_layout`

Set or transform memory layout.

**Syntax:**
```
%result = mem.set_layout %input {layout = NZ, block_size = 16}
    : memref<...xf16, ND, L1> -> memref<...xf16, NZ, L1>
```

**Note:** May involve data reorganization.

---

### 14.2.4 Synchronization

#### `mem.barrier`

Memory synchronization barrier.

**Syntax:**
```
mem.barrier {memory_space = L1}
```

**Semantics:**
- Ensures all previous memory operations complete
- Optional memory space specification

---

## 14.3 Memory Spaces

Supported memory spaces:

| Space       | Description                          | Typical Use Case           |
| ----------- | ------------------------------------ | -------------------------- |
| **DDR**     | Off-chip DRAM                        | Large data storage         |
| **L2**      | Large on-chip SRAM                   | Shared cache               |
| **L1**      | Per-core cache/SRAM                  | Core-local data            |
| **UB**      | Unified Buffer                       | Intermediate results       |
| **L0A/B/C** | Matrix unit input/output buffers     | Tile operations            |
| **REG**     | Registers                            | Scalar values              |
| **SHMEM**   | Shared memory (distributed)           | Inter-worker communication |

---

## 14.4 Memory Lifetime Management

### Allocation Patterns

**Stack-like allocation:**
```
%buf1 = mem.alloc : memref<1024xf16>
// use buf1
mem.free %buf1

%buf2 = mem.alloc : memref<1024xf16>
// use buf2
mem.free %buf2
```

**Nested allocation:**
```
%outer = mem.alloc : memref<1024xf16>
statement.for %i = 0 to N {
  %inner = mem.alloc : memref<256xf16>
  // use inner
  mem.free %inner
}
mem.free %outer
```

---

## 14.5 Verification Rules

A valid memory operation must satisfy:

1. **Lifetime correctness**: No use-after-free, no double-free
2. **Size compatibility**: Copy operations must have matching sizes
3. **Space validity**: Memory space must be supported by hardware
4. **Alignment**: Allocations must satisfy alignment requirements

---

## 14.6 Examples

### Memory Transfer Pipeline

```
%src = mem.alloc {memory_space = DDR} : memref<1024xf16, ND, DDR>
%dst = mem.alloc {memory_space = L1} : memref<1024xf16, ND, L1>

// DMA transfer
%token = mem.dma_copy %src, %dst {async = true}

// Do other work while transfer happens
// ...

// Wait for transfer
mem.wait %token

mem.free %src
mem.free %dst
```

### Layout Transformation

```
%input = mem.alloc {memory_space = L1} : memref<1024x1024xf16, ND, L1>
%output = mem.alloc {memory_space = L1} : memref<1024x1024xf16, NZ, L1>

// Transform layout
%transformed = mem.set_layout %input {layout = NZ, block_size = 16}
mem.copy %transformed, %output

mem.free %input
mem.free %output
```

---

## 14.7 Summary

The memory dialect provides:

* Explicit memory management
* DMA operations for efficient transfers
* Memory layout transformations
* Clear memory lifetime semantics

It enables efficient memory usage and hardware-aware memory management.

---

