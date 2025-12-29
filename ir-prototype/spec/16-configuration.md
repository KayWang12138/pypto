# 16. Configuration Dialect（TODO）

The `pto.config` dialect provides configuration and hints for transformation passes when processing PTO-IR. It allows fine-grained control over optimization strategies, resource allocation, and code generation decisions.

Configuration must be bound to specific IR objects, such as program, function, or operation. If both inner and outer IR objects have the same configuration value bound, the inner value takes precedence as the effective value.

---

## 16.1 Purpose

The `pto.config` dialect serves to:

* **Control transformation passes**: Guide how passes transform the IR
* **Provide optimization hints**: Suggest preferred strategies to the compiler
* **Enable auto-tuning**: Support exploration of different configurations
* **Express user preferences**: Allow explicit control over compilation

As a **meta-dialect**, `pto.config` does not participate in lowering but is used by transformation passes to guide optimization decisions.

---

## 16.2 Dialect Characteristics

### 16.2.1 Meta-Dialect Nature

`pto.config` is a meta-dialect similar to `pto.debug`:
* Does not participate in lowering flows
* Attaches to program, function, or operation level
* Used by transformation passes for decision-making
* Can be queried and transformed like other IR operations

### 16.2.2 Configuration Levels

Configuration can be specified at multiple levels:

1. **Program-level**: Global configuration for entire program
2. **Function-level**: Per-function configuration
3. **Operation-level**: Per-operation configuration (via attributes)
4. **Pass-level**: Configuration for specific transformation passes

---

## 16.3 Tiling Strategy Configuration

### 16.3.1 Tile Size Configuration

Specify preferred tile sizes for operations.

**Syntax:**
```
config.tile_strategy {
  matmul_tile = {M = 16, N = 16, K = 16}
  conv_tile = {H = 8, W = 8, C = 32}
  default_tile = {size = 16}
}
```

**Attributes:**
- `matmul_tile`: Tile sizes for matrix multiplication
- `conv_tile`: Tile sizes for convolution
- `reduce_tile`: Tile sizes for reduction operations
- `default_tile`: Default tile size when not specified

---

### 16.3.2 Tiling Heuristics

Control tiling algorithm selection.

**Syntax:**
```
config.tiling_heuristic {
  strategy = "auto"  // "auto", "fixed", "dynamic", "adaptive"
  prefer_large_tiles = true
  prefer_square_tiles = false
  max_tile_size = 128
  min_tile_size = 8
}
```

**Strategies:**
- `auto`: Automatic selection based on hardware
- `fixed`: Use fixed tile sizes from configuration
- `dynamic`: Runtime tile size selection
- `adaptive`: Adaptive tiling based on workload

---

### 16.3.3 Tile Format Configuration

Specify preferred tile formats.

**Syntax:**
```
config.tile_format {
  matmul_input_format = "NZ"
  matmul_output_format = "ND"
  default_format = "ND"
}
```

---

## 16.4 Partitioning Configuration

### 16.4.1 Data Partitioning

Configure how tensors are partitioned across devices.

**Syntax:**
```
config. partitioning {
  strategy = "data_parallel"  // "data_parallel", "tensor_parallel", "pipe_parallel"
  shard_axis = 0
  num_shards = 8
  replication_factor = 1
}
```

**Strategies:**
- `data_parallel`: Partition along batch dimension
- `tensor_parallel`: Partition along feature dimensions
- `pipe_parallel`: Partition along layer/sequence dimension and coordinate via block graphs + pipe execution schedules
- `hybrid`: Combination of strategies

---

### 16.4.2 Work Partitioning

Configure work distribution across workers.

**Syntax:**
```
config. work_partitioning {
  load_balance = "even"  // "even", "weighted", "custom"
  affinity = "local"     // "local", "remote", "balanced"
  overlap_compute_comm = true
}
```

---

## 16.5 Scheduling Configuration

### 16.5.1 Loop Scheduling

Configure loop transformation and scheduling.

**Syntax:**
```
config. loop_scheduling {
  fusion_enabled = true
  tiling_enabled = true
  unroll_factor = 4
  vectorize = true
  parallelize = true
  reorder_strategy = "cache_friendly"
}
```

**Reorder strategies:**
- `cache_friendly`: Optimize for cache locality
- `compute_intensive`: Optimize for compute throughput
- `memory_intensive`: Optimize for memory bandwidth
- `balanced`: Balance compute and memory

---

### 16.5.2 Pipe Execution Scheduling

Configure how block graph operations are mapped onto `pto.pipe` pipes.

**Syntax:**
```
config. pipe_execution {
  assignment = "greedy"  // "greedy", "optimal", "heuristic"
  overlap_pipes = true
  prefetch_enabled = true
  double_buffering = true
}
```

---

### 16.5.3 Memory Scheduling

Configure memory access scheduling.

**Syntax:**
```
config. memory_scheduling {
  coalesce_accesses = true
  prefetch_distance = 2
  memory_bandwidth_aware = true
  cache_aware = true
}
```

---

## 16.6 Memory Management Configuration

### 16.6.1 Allocation Strategy

Configure memory allocation policies.

**Syntax:**
```
config. memory_allocation {
  strategy = "static"  // "static", "dynamic", "pooled"
  alignment = 16
  reuse_buffers = true
  memory_pool_size = 1024 * 1024 * 1024  // 1GB
}
```

---

### 16.6.2 Memory Space Assignment

Configure which memory spaces to use.

**Syntax:**
```
config. memory_spaces {
  input_tensors = "L1"
  intermediate_tensors = "UB"
  output_tensors = "L1"
  temporary_buffers = "L0A"
  prefer_fast_memory = true
}
```

---

### 16.6.3 DMA Configuration

Configure DMA transfer behavior.

**Syntax:**
```
config. dma {
  async_enabled = true
  batch_size = 1024
  prefetch_enabled = true
  double_buffering = true
}
```

---

## 16.7 Optimization Configuration

### 16.7.1 Optimization Level

Set overall optimization level.

**Syntax:**
```
config. optimization {
  level = "aggressive"  // "none", "basic", "standard", "aggressive"
  enable_fusion = true
  enable_simplification = true
  enable_canonicalization = true
}
```

---

### 16.7.2 Operation-Specific Optimizations

Configure optimizations for specific operations.

**Syntax:**
```
config. operation_optimizations {
  matmul_optimization = "tiled"  // "tiled", "direct", "blocked"
  conv_optimization = "winograd"  // "direct", "winograd", "fft"
  reduce_optimization = "tree"     // "sequential", "tree", "parallel"
}
```

---

## 16.8 Hardware-Specific Configuration

### 16.8.1 Target Hardware

Specify target hardware characteristics.

**Syntax:**
```
config. target_hardware {
  architecture = "PTOv2"
  num_cores = 8
  memory_hierarchy = "L0/L1/UB/DDR"
  compute_units = ["MMU", "VMAC", "DMA"]
}
```

---

### 16.8.2 Resource Constraints

Specify resource limits.

**Syntax:**
```
config. resources {
  max_memory_per_core = 64 * 1024 * 1024  // 64MB
  max_tile_size = 128
  max_pipe_exec_stages = 8
  max_parallel_workers = 16
}
```

---

## 16.9 Auto-Tuning Configuration

### 16.9.1 Tuning Strategy

Configure auto-tuning behavior.

**Syntax:**
```
config. autotuning {
  enabled = true
  strategy = "grid_search"  // "grid_search", "random", "bayesian", "genetic"
  search_space = {
    tile_sizes = [8, 16, 32, 64]
    unroll_factors = [1, 2, 4, 8]
    pipe_exec_depths = [2, 4, 6, 8]
  }
  max_iterations = 100
  timeout_seconds = 3600
}
```

---

### 16.9.2 Cost Model

Configure cost model for auto-tuning.

**Syntax:**
```
config. cost_model {
  weights = {
    compute_cost = 1.0
    memory_cost = 0.5
    communication_cost = 2.0
  }
  use_empirical = true
  cache_results = true
}
```

---

## 16.10 Configuration Inheritance and Override

### 16.10.1 Inheritance Rules

Configuration follows inheritance hierarchy:

1. **Program-level**: Base configuration
2. **Function-level**: Overrides program-level
3. **Operation-level**: Overrides function-level
4. **Pass-level**: Overrides all for specific pass

---

### 16.10.2 Configuration Merging

When multiple configurations apply:

* **Override**: More specific configuration overrides general
* **Merge**: Compatible settings are merged
* **Conflict resolution**: Explicit settings take precedence

---

## 16.11 Configuration Syntax

### 16.11.1 Textual Format

```
config.config_name {
  key1 = value1
  key2 = {nested_key = nested_value}
  key3 = [value1, value2, value3]
}
```

---

### 16.11.2 Attribute Format

Configuration can be attached as attributes:

```
func.func @kernel(...) {
  config.tile_size = {M = 32, N = 32, K = 32}
  config.memory_space = "L1"
  // ...
}
```

---

## 16.12 Types

The `pto.config` dialect defines the following types:

### 16.12.1 Configuration Types

**`config.tile_config`**

Type for tile configuration values.

```
config.tile_config {
  M: i32
  N: i32
  K: i32
  format: string
}
```

**`config.partitioning_config`**

Type for partitioning configuration.

```
config.partitioning_config {
  strategy: string  // "data_parallel", "tensor_parallel", etc.
  shard_axis: i32
  num_shards: i32
  replication_factor: i32
}
```

**`config.scheduling_config`**

Type for scheduling configuration.

```
config.scheduling_config {
  fusion_enabled: bool
  tiling_enabled: bool
  unroll_factor: i32
  vectorize: bool
  parallelize: bool
}
```

**`config.memory_config`**

Type for memory allocation configuration.

```
config.memory_config {
  strategy: string  // "static", "dynamic", "pooled"
  alignment: i32
  reuse_buffers: bool
  memory_pool_size: i64
}
```

---

## 16.13 Verification Rules

A valid `pto.config` operation must satisfy:

1. **Type correctness**: Configuration values must match expected types
2. **Value validity**: Values must be within valid ranges (e.g., tile sizes > 0)
3. **Platform compatibility**: Configuration must be compatible with target platform
4. **Resource constraints**: Resource limits must be satisfiable
5. **Conflict resolution**: Conflicting configurations must be resolved according to inheritance rules

### 16.13.1 Tile Configuration Verification

* Tile sizes must be positive integers
* Tile sizes must fit in target memory spaces (queryable from `pto.platform`)
* Tile formats must be supported by hardware

### 16.13.2 Partitioning Configuration Verification

* Number of shards must be positive
* Shard axis must be valid for tensor dimensions
* Replication factor must be non-negative

### 16.13.3 Scheduling Configuration Verification

* Unroll factors must be positive
* Optimization levels must be valid
* Memory pool sizes must be positive

---

## 16.14 Examples

### Basic Configuration

```mlir
pto.program.module @main {
  pto.config.default {
    tile_strategy = {default_tile = {size = 16}}
    optimization = {level = "standard"}
    memory_allocation = {strategy = "static", reuse_buffers = true}
  }
  
  pto.func.func @matmul(...) {
    // Uses default configuration
  }
}
```

### Function-Specific Configuration

```mlir
pto.func.func @custom_matmul(...) {
  pto.config.tile_size = {M = 64, N = 64, K = 64}  // Override default
  pto.config.memory_space = "L0A"                   // Override default
  
  // Operations use function-level configuration
}
```

### Auto-Tuning Configuration

```
  config.autotune {
  enabled = true
  search_space = {
    tile_sizes = [16, 32, 64]
    unroll_factors = [1, 2, 4]
  }
  max_iterations = 50
}
```

---

## 16.15 Summary

The `pto.config` dialect provides:

* **Fine-grained control**: Over transformation passes
* **Optimization hints**: Guide compiler decisions
* **Auto-tuning support**: Enable automatic optimization
* **Flexible specification**: At multiple IR levels
* **First-class IR operations**: Queryable and transformable like other dialects

As a meta-dialect, it enables users and compilers to control and optimize IR transformation effectively without participating in lowering flows.

---

