# 21. Examples(TODO)

This section provides comprehensive examples of PTO-IR programs at various levels of abstraction.

---

## 21.1 Simple Matrix Multiplication

### High-Level (Tensor Dialect)

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

## 21.2 Tiled Matrix Multiplication

### Tile-Level

```mlir
pto.func.func @tiled_matmul(%A: memref<1024x512xf16,ND,L1>,
                            %B: memref<512x256xf16,ND,L1>)
    -> (memref<1024x256xf16,ND,L1>) {
  %C = pto.statement.alloc : memref<1024x256xf16,ND,L1>
  
  pto.statement.for %i = 0 to 1024 step 16 {
    pto.statement.for %j = 0 to 256 step 16 {
      pto.statement.for %k = 0 to 512 step 16 {
        %tile_A = pto.tile.load_tile %A[%i, %k] {tile_shape = [16, 16]}
        %tile_B = pto.tile.load_tile %B[%k, %j] {tile_shape = [16, 16]}
        %tile_C = pto.tile.matmul %tile_A, %tile_B
        pto.tile.store_tile %tile_C, %C[%i, %j]
      }
    }
  }
  
  pto.func.return %C
}
```

---

## 21.3 Block Graph, Pipe Execution, and Execution Graph Example

### 21.3.1 Block Graph Definition

```mlir
pto.blockgraph.block @mm_block(
    %A: memref<16x16xf16,ND,L1>,
    %B: memref<16x16xf16,ND,L1>)
    -> (memref<16x16xf16,ND,L1>) {
  %tile_A = pto.tile.load_tile %A
  %tile_B = pto.tile.load_tile %B
  pto.blockgraph.dep %tile_A -> %tile_B {type = "RAW"}
  %tile_C = pto.tile.matmul %tile_A, %tile_B
  pto.blockgraph.yield %tile_C
}
```

### 21.3.2 Pipe Execution Placement

```mlir
pto.pipe_exec.core @core0 {block = @mm_block, unit = "MMU"} {
  pto.pipe_exec.pipe "dma" {
    %event_load = pto.pipe_exec.set
    pto.pipe_exec.assign %A_tile = pto.tile.load_tile %A[%idx] {bind = %arg0}
    pto.pipe_exec.assign %B_tile = pto.tile.load_tile %B[%idx] {bind = %arg1}
    pto.pipe_exec.set %event_dma = %event_load
  }

  pto.pipe_exec.pipe "compute" {
    pto.pipe_exec.wait %event_dma
    pto.pipe_exec.assign %C_tile = pto.tile.matmul %A_tile, %B_tile
    pto.pipe_exec.set %event_compute
  }

  pto.pipe_exec.pipe "store" {
    pto.pipe_exec.wait %event_compute
    pto.pipe_exec.assign pto.tile.store_tile %C_tile, %C[%idx]
  }
}
```

### 21.3.3 Execution Graph

```mlir
pto.exec.graph @matmul_schedule {scope = "device"} {
  %call0 = pto.exec.call_block @mm_block(
      inputs = [%A_tiles0, %B_tiles0],
      outputs = [%C_tiles0]) {core_group = "mmu0"}
  %call1 = pto.exec.call_block @mm_block(
      inputs = [%A_tiles1, %B_tiles1],
      outputs = [%C_tiles1]) {core_group = "mmu1"}
  pto.exec.dependency %call0 -> %call1 {type = "data"}
  pto.exec.allocate : memref<262144xf16,ND,DDR>
}
```

---

## 21.4 Distributed Example

### Multi-Device Execution

```mlir
pto.program.module @distributed {
  pto.program.entry @distributed_matmul
  
  pto.func.func @distributed_matmul(%A: tensor<4096x2048xf16>,
                                     %B: tensor<2048x1024xf16>)
      -> (tensor<4096x1024xf16>) {
    
    pto.dist.launch {num_workers = 4} {
      %wid = pto.dist.worker_id
      
      // Shard input tensors
      %A_shard = pto.dist.shard_tensor %A {axis = 0, num_shards = 4}
      %B_shard = pto.dist.shard_tensor %B {axis = 1, num_shards = 4}
      
      // Local computation
      %C_local = pto.tensor.matmul %A_shard, %B_shard : tensor<4096x2048xf16>, tensor<2048x1024xf16> -> tensor<4096x1024xf16>
      
      // All-reduce to combine results
      %C = pto.dist.allreduce %C_local {op = "sum"}
    }
    
    pto.func.return %C
  }
}
```

---

## 21.5 Control Flow Example

### Conditional Execution

```mlir
pto.func.func @conditional(%input: tensor<1024xf16>, %flag: i1)
    -> (tensor<1024xf16>) {
  pto.scf.if %flag {
    %scaled = pto.tensor.mul %input, %input {alpha = 2.0} : tensor<1024xf16>, tensor<1024xf16> -> tensor<1024xf16>
    pto.scf.yield %scaled
  } else {
    %squared = pto.tensor.mul %input, %input : tensor<1024xf16>, tensor<1024xf16> -> tensor<1024xf16>
    pto.scf.yield %squared
  }
}
```

---

## 21.6 Loop Example

### Reduction Loop

```mlir
pto.func.func @reduce_sum(%input: tensor<1024xf16>) -> (f16) {
  %init = pto.scalar.constant 0.0 : f16
  
  %result = pto.scf.for %i = 0 to 1024 iter_args(%acc = %init) -> (f16) {
    %val = pto.tensor.dim %input, %i
    %new_acc = pto.scalar.add %acc, %val : f16, f16 -> f16
    pto.scf.yield %new_acc
  }
  
  pto.func.return %result
}
```

---

## 21.7 Memory Management Example

### DMA Transfer

```mlir
pto.func.func @dma_example(%src: memref<1024xf16,ND,DDR>,
                           %dst: memref<1024xf16,ND,L1>) {
  // Start async DMA transfer
  %token = pto.mem.dma_copy %src, %dst {async = true}
  
  // Do other work
  // ...
  
  // Wait for DMA completion
  pto.mem.wait %token
}
```

---

## 21.8 Instruction-Level Example

### Hardware Instructions

```mlir
pto.func.func @mma_kernel(%A: memref<1024x512xf16,ND,L1>,
                          %B: memref<512x256xf16,ND,L1>)
    -> (memref<1024x256xf16,ND,L1>) {
  %C = pto.mem.alloc : memref<1024x256xf16,ND,L1>
  
  pto.statement.for %i = 0 to 1024 step 16 {
    pto.statement.for %j = 0 to 256 step 16 {
      pto.statement.for %k = 0 to 512 step 16 {
        // Load tiles
        %tile_A = pto.inst.load_L0A %A[%i, %k]
        %tile_B = pto.inst.load_L0B %B[%k, %j]
        pto.inst.barrier
        
        // Matrix multiply
        %tile_C = pto.inst.mma %tile_A, %tile_B, %zero
        pto.inst.barrier
        
        // Store result
        pto.inst.store %tile_C, %C[%i, %j]
      }
    }
  }
  
  pto.func.return %C
}
```

---

## 21.9 Complete Program Example

### End-to-End Program

```mlir
pto.program.module @complete_example {
  pto.program.global @weights : tensor<1024x512xf16> = {...}
  pto.program.entry @inference
  
  pto.func.func @inference(%input: tensor<32x1024xf16>) -> (tensor<32x512xf16>) {
    // Load weights
    %W = pto.program.get_global @weights
    
    // Matrix multiplication
    %hidden = pto.tensor.matmul %input, %W : tensor<32x1024xf16>, tensor<1024x512xf16> -> tensor<32x512xf16>
    
    // Activation
    %activated = pto.tensor.relu %hidden : tensor<32x512xf16> -> tensor<32x512xf16>
    
    // Normalization
    %normalized = pto.tensor.rms_norm %activated {axis = 1, epsilon = 1e-5} : tensor<32x512xf16> -> tensor<32x512xf16>
    
    pto.func.return %normalized
  }
}
```

---

## 21.10 Configuration and Platform Examples

### Configuration Dialect Usage

```mlir
pto.program.module @main {
  // Program-level configuration
  pto.config.tile_strategy {
    matmul_tile = {M = 16, N = 16, K = 16}
    default_tile = {size = 16}
  }
  
  pto.config.optimization {
    level = "aggressive"
    enable_fusion = true
  }
  
  pto.func.func @matmul(...) {
    // Function-level configuration override
    pto.config.tile_size = {M = 32, N = 32, K = 32}
    // ... operations ...
  }
}
```

### Platform Abstractions Dialect Usage

```mlir
pto.program.module @main {
  // Define platform model
  pto.platform.execution_unit @MMU {
    type = "matrix_multiply"
    throughput = 256
    latency = 4
    input_buffers = ["L0A", "L0B"]
    output_buffer = "L0C"
  }
  
  pto.platform.memory @L0A {
    level = 0
    type = "register_file"
    size = 16 * 1024
    bandwidth = 1024
    latency = 1
  }
  
  pto.platform.platform @PTOv2 {
    execution_units = [@MMU]
    memory_hierarchy = {
      L0A -> L1
      L1 -> UB
      UB -> DDR
    }
  }
  
  // Functions use platform for optimization
  pto.func.func @optimized_kernel(...) {
    // Transformation passes query platform
    // to make optimization decisions
  }
}
```

### Combined Configuration and Platform

```mlir
pto.program.module @main {
  // Platform definition
  pto.platform.platform @target_hw {
    execution_units = [@MMU, @VMAC]
    memory_hierarchy = {...}
  }
  
  // Configuration using platform info
  pto.config.tile_strategy {
    // Tile sizes chosen based on platform L0 buffer size
    matmul_tile = {M = 16, N = 16, K = 16}
  }
  
  pto.config.autotuning {
    enabled = true
    search_space = {
      tile_sizes = [16, 32, 64]
    }
  }
  
  pto.func.func @kernel(...) {
    // Operations optimized using config + platform
  }
}
```

---

## 21.11 Summary

These examples demonstrate:

* **Various abstraction levels**: From tensor to instruction level
* **Different dialects**: Usage of all major dialects including meta-dialects
* **Common patterns**: Typical computation patterns
* **Configuration usage**: How to use `pto.config` dialect
* **Platform modeling**: How to use `pto.platform` dialect
* **Best practices**: Recommended IR construction

They serve as reference implementations for PTO-IR programs.

---

