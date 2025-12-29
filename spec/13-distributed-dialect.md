# 13. Distributed Dialect(TODO)

The `pto.dist` dialect supports distributed execution across multiple devices and nodes, providing OpenSHMEM-style operations for remote memory access and collective operations.

---

## 13.1 Purpose

The distributed dialect serves to:

* Express multi-device and multi-node execution
* Provide remote memory operations
* Support collective communication operations
* Enable distributed tensor parallelism

---

## 13.2 Core Operations

### 13.2.1 Launch Operations

#### `dist.launch`

Launch distributed execution across workers.

**Syntax:**
```
dist.launch {num_workers = 8, devices = ["gpu0", "gpu1", ...]} {
  %wid = dist.worker_id
  // worker code
}
```

**Attributes:**
- `num_workers`: number of worker processes/threads
- `devices`: list of target devices
- `exec_mode`: execution mode (replicated, sharded, etc.)

---

#### `dist.worker_id`

Get current worker ID.

**Syntax:**
```
%wid = dist.worker_id : index
```

**Returns:** Worker ID in range [0, num_workers)

---

### 13.2.2 Remote Memory Operations

#### `dist.shmem_put`

Put data to remote shared memory (OpenSHMEM-style).

**Syntax:**
```
dist.shmem_put %local, %remote {target_worker = 1, blocking = false} 
    : memref<...xf16>, dist.memref<...xf16>
```

**Attributes:**
- `target_worker`: destination worker ID
- `blocking`: whether operation blocks until completion
- `async` (optional): return completion token

---

#### `dist.shmem_get`

Get data from remote shared memory.

**Syntax:**
```
dist.shmem_get %remote, %local {source_worker = 0, blocking = true}
    : dist.memref<...xf16>, memref<...xf16>
```

---

#### `dist.alloc_shared`

Allocate shared memory accessible by all workers.

**Syntax:**
```
%shared = dist.alloc_shared {size = 1024, memory_space = SHMEM}
    : dist.memref<1024xf16, SHMEM>
```

---

### 13.2.3 Collective Operations

#### `dist.allreduce`

All-reduce operation across workers.

**Syntax:**
```
%result = dist.allreduce %input {op = "sum", group = "all"}
    : tensor<...xf16> -> tensor<...xf16>
```

**Reduction operations:**
- `sum`, `max`, `min`, `prod`, `mean`

---

#### `dist.allgather`

All-gather operation.

**Syntax:**
```
%result = dist.allgather %input {group = "all"}
    : tensor<...xf16> -> tensor<...xf16>
```

**Semantics:**
- Each worker contributes input
- All workers receive concatenated result

---

#### `dist.broadcast`

Broadcast value from root to all workers.

**Syntax:**
```
%result = dist.broadcast %input {root = 0, group = "all"}
    : tensor<...xf16> -> tensor<...xf16>
```

---

#### `dist.reduce_scatter`

Reduce-scatter operation.

**Syntax:**
```
%result = dist.reduce_scatter %input {op = "sum", group = "all"}
    : tensor<...xf16> -> tensor<...xf16>
```

---

### 13.2.4 Synchronization

#### `dist.barrier`

Global synchronization barrier.

**Syntax:**
```
dist.barrier {group = "all"}
```

**Semantics:**
- All workers in group must reach barrier
- Execution continues only after all workers arrive

---

#### `dist.event`

Create synchronization event.

**Syntax:**
```
%event = dist.event : dist.event_token
```

**Use with:**
- `dist.wait` to wait for event
- `dist.signal` to signal event completion

---

#### `dist.wait`

Wait for event or operation completion.

**Syntax:**
```
dist.wait %event : dist.event_token
```

---

### 13.2.5 Sharding Operations

#### `dist.shard_tensor`

Partition tensor across workers.

**Syntax:**
```
%sharded = dist.shard_tensor %input {axis = 0, num_shards = 8}
    : tensor<...xf16> -> dist.tensor<shard(axis=0, parts=8)>
```

**Sharding strategies:**
- `shard(axis, parts)`: partition along axis
- `replicate`: replicate on all workers
- `partial`: partial replication

---

## 13.3 Distributed Types

### `dist.tensor<sharding>`

Distributed tensor with sharding information.

**Examples:**
```
dist.tensor<shard(axis=0, parts=8)>
dist.tensor<replicate>
dist.tensor<partial>
```

---

### `dist.group<n>`

Worker group type.

**Syntax:**
```
dist.group<8>  // group of 8 workers
```

---

### `dist.memref<...>`

Remote memory reference.

**Syntax:**
```
dist.memref<1024xf16, SHMEM, worker=1>
```

---

## 13.4 Execution Model

### Worker Model

- Each worker executes independently
- Workers can access local and remote memory
- Synchronization via barriers and events

### Memory Model

- **Local memory**: Private to each worker
- **Shared memory**: Accessible by all workers (with proper synchronization)
- **Remote memory**: Memory on other workers (accessed via put/get)

---

## 13.5 Verification Rules

A valid distributed operation must satisfy:

1. **Worker count**: Collective ops must have matching participant counts
2. **Sharding compatibility**: Sharding strategies must be compatible
3. **Remote memory validity**: Remote memory references must be valid
4. **Synchronization**: Barriers and events properly matched

---

## 13.6 Examples

### Distributed Matrix Multiplication

```
dist.launch {num_workers = 4} {
  %wid = dist.worker_id
  
  // Shard input tensors
  %A_shard = dist.shard_tensor %A {axis = 0, num_shards = 4}
  %B_shard = dist.shard_tensor %B {axis = 1, num_shards = 4}
  
  // Local computation
  %C_local = tensor.matmul %A_shard, %B_shard
  
  // All-reduce to combine results
  %C = dist.allreduce %C_local {op = "sum"}
}
```

### Data Parallel Training Step

```
dist.launch {num_workers = 8} {
  %wid = dist.worker_id
  
  // Get data shard for this worker
  %batch = dist.shard_tensor %dataset {axis = 0, num_shards = 8}
  
  // Forward pass
  %loss = call @forward(%batch, %model)
  
  // All-reduce gradients
  %grads = dist.allreduce %gradients {op = "sum"}
  
  // Update model
  %model = call @update(%model, %grads)
  
  dist.barrier
}
```

### Pipeline Parallelism

```
dist.launch {num_workers = 4} {
  %wid = dist.worker_id
  
  statement.for %layer = 0 to num_layers {
    if (%layer % 4 == %wid) {
      // This worker processes this layer
      %output = call @layer[%layer](%input)
      
      // Send to next worker
      if (%wid < 3) {
        dist.shmem_put %output, %next_worker_input {target_worker = %wid + 1}
      }
    } else {
      // Receive from previous worker
      if (%wid > 0) {
        dist.shmem_get %prev_worker_output, %input {source_worker = %wid - 1}
      }
    }
    
    dist.barrier
  }
}
```

---

## 13.7 Summary

The distributed dialect provides:

* Multi-device and multi-node execution
* Remote memory operations (OpenSHMEM-style)
* Collective communication primitives
* Distributed tensor parallelism support

It enables scaling computations across multiple devices and nodes.

---

