# 6. Statement Dialect

The `pto.statement` dialect represents control flow, indexing, and memory operations. It provides a structured way to express control flow domains and memory access patterns.

---

## 6.1 Introduction

The statement dialect is implemented as a structured statement tree per function.  Each `statement` node can see all `value`s defined in its ancestor statements.

The statement dialect serves to:

* Express loop nests and iteration domains
* Express if branch and yield
* Represent memory access patterns with explicit indexing

It contains following core operations:
* statement.for
* statement.parallel_for
* statement.if
* statement.yield
* statement.block

An example of nested statements:

```
function
  ├── statement.block
  ├── statement.for
  │     ├── statement.if
  │     │     ├── statement.block
  │     │     ├── statement.block
  │     │     └── statement.yield
  │     ├── statement.block
  │     └── statement.yield
  ├── statement.block
  └── statement.return
```

---

## 6.2 Core Operations

### 6.2.1 Control Flow

#### `statement.for`

Sequential for loop, with an explicit induction variable and loop-carried values.

**Syntax:**
```
%res0, %res1 = statement.for %iv = %lb to %ub step %step
                 [iter_args(%acc0 = %init0 : T0, %acc1 = %init1 : T1, ...)]
                 [attributes]
{
  // sequence of nested statements: statement.block, statement.for, statement.if, ...
  ...
  statement.yield %new_acc0, %new_acc1, ...        // or statement.yield
}
```

- `%lb`: lower bound (inclusive)
- `%ub`: upper bound (exclusive)
- `%step`: step size
- `%init`: initial values for iter_args
- `iter_args`: Optional. If present, `statement.for` returns a result list matching the types of `iter_args` (number and types must strictly correspond)

Each `statement.for` introduces its own body scope that contains the nested statements and scopes listed above.

**Results:**

- The result types of `statement.for` are given by `result_types`, and usually match the types of `iter_args`.

Constraints:
- The last statement in the loop body scope must be a `statement.yield`.

---

#### `statement.parallel_for` (TODO)

Parallel for loop (can be executed in parallel).

**Syntax:**
```
statement.parallel_for %i = %lb to %ub step %step {
  // loop body (must be independent)
}
```

**Verification:**
- Loop iterations must be independent
- No loop-carried dependencies allowed

#### `statement.if`

`statement.if` is a value-producing conditional construct.

**Syntax:**

```
%r0, %r1 = statement.if %cond -> (T0, T1, ...)
{
  // then-region statements
  ...
  statement.yield %then_v0, %then_v1, ...
} else {
  // else-region statements
  ...
  statement.yield %else_v0, %else_v1, ...
}
```

Semantics and constraints:

- `%cond` must be a scalar boolean SSA value.
- `-> (T0, T1, ...)` specifies the `result_types` of the `statement.if`.
- Each branch maintains its own lexical scope and terminates with `statement.yield`.
- When the yields differ between the two scopes, the system inserts or coerces values so both sides present results matching `result_types`.

---

#### `statement.yield`

`statement.yield` is a generic region terminator that returns values from the current scope to its parent statement.

**Syntax:**

```
statement.yield %value0, %value1, ...
```

Used in:

- Then / else regions of `statement.if`, to provide per-branch results back to the `if`.
- At the end of a `statement.for` body, to provide updated loop-carried values.

### 6.2.2 Block

`statement.block` represents a linear basic block of operations with no nested control-flow statements inside itself.

`statement.block` contains the concrete sequence of operations (loads, stores, arithmetic, etc.). Control-flow constructs (`statement.for`, `statement.if`) appear as sibling statements at the same hierarchical level.

Variables defined inside a `statement.block` are SSA values visible to subsequent statements within the same region.

**Syntax:**

```
statement.block {
  // linear sequence of operations
  %v0 = ...
  %v1 = ...
  ...
}
```
---

### 6.2.3 Call

`statement.call` invokes a user-defined function from within a statement context.

**Syntax:**

```
%res0, %res1 = statement.call @function_name(%arg0, %arg1, ...) 
    : (T0, T1, ...) -> (R0, R1, ...)
```

**Semantics:**

- Calls a function defined with `func.func` by its symbol name.
- Arguments and return types must match the function signature.
- `statement.call` is a statement-level operation that appears alongside other statement structures (like `statement.for`, `statement.if`, `statement.block`) at the same hierarchical level, not inside `statement.block`.
- The call is synchronous and blocks until the function returns.

**Example:**

```
func.func @main() {
  statement.block {
    %A = ...
    %B = ...
  }
  
  %result = statement.call @matmul(%A, %B) 
      : (tensor<16x16xf16>, tensor<16x16xf16>) -> tensor<16x16xf16>
  
  statement.block {
    // use %result
  }
  
  statement.return %result
}
```

---

### 6.2.4 Return

`statement.return` terminates a function-wide statement sequence by returning the latest loop-carried results or the final SSA values produced inside the top-level region.

**Syntax:**
```
statement.return %value0, %value1, ...
```

**Notes:**
- The values returned must match the function signature.
- Each function should have exactly one `statement.return` at the end of its outermost statement sequence.
- Unlike in general programming languages, `statement.return` cannot appear arbitrarily inside a block; it is only valid at the function-level terminator position to keep the statement tree well-structured.
---

### 6.2.5 Memory Operations (TODO)

#### `statement.load`

Load value from memory.

**Syntax:**

```
%value = statement.load %memref[%i, %j] : memref<...xf16, ND, L1> -> f16
```

**Indexing:**

- Supports multi-dimensional indexing
- Bounds checking optional (can be disabled for performance)

---

#### `statement.store`

Store value to memory.

**Syntax:**

```
statement.store %value, %memref[%i, %j] : f16, memref<...xf16, ND, L1>
```

---

#### `statement.alloc`

Allocate memory buffer.

**Syntax:**

```
%memref = statement.alloc {size = 1024, memory_space = L1} : memref<1024xf16, ND, L1>
```

**Attributes:**

- `size`: allocation size
- `memory_space`: target memory space (DDR, L1, UB, L0A/B/C, REG)

---

#### `statement.dealloc`

Deallocate memory buffer.

**Syntax:**

```
statement.dealloc %memref : memref<...xf16, ND, L1>
```

**Lifetime:**

- Must be paired with `statement.alloc`
- Verification ensures no use-after-free

---

#### `statement.gep`

Get element pointer (compute offset from indices).

**Syntax:**

```
%ptr = statement.gep %memref, %i, %j : memref<...xf16>, index, index -> memref<1xf16>
```

---

## 6.3 Memory Access Patterns (TODO)

Statement dialect explicitly represents memory access patterns:

### Strided Access

```
statement.for %i = 0 to N {
  %val = statement.load %A[%i * stride]  // strided access
}
```

### Tiled Access

```
statement.for %ti = 0 to M step TILE_SIZE {
  statement.for %tj = 0 to N step TILE_SIZE {
    statement.for %i = 0 to TILE_SIZE {
      statement.for %j = 0 to TILE_SIZE {
        %val = statement.load %A[%ti + %i, %tj + %j]  // tiled access
      }
    }
  }
}
```

---

## 6.4 Verification Rules

A valid statement operation must satisfy:

1. **Loop bounds**: Lower bound < upper bound (for non-empty loops)
2. **Memory safety**: No use-after-free, no double-free
3. **Index bounds**: Memory accesses within allocated bounds
4. **SSA correctness**: All values properly defined
5. **Parallel safety**: Parallel loops have no dependencies

---

## 6.5 Optimization Passes（TODO）

Common optimizations on statement dialect:

### Loop Transformations

- **Loop fusion**: Combine adjacent loops
- **Loop fission**: Split loops for better cache behavior
- **Loop tiling**: Break loops into smaller tiles
- **Loop unrolling**: Unroll small loops
- **Loop reordering**: Change loop order for better locality

### Memory Optimizations

- **Buffer reuse**: Reuse allocated buffers
- **Memory coalescing**: Combine memory accesses
- **Prefetching**: Insert prefetch operations

---

## 6.6 Examples

### Control Flow
Pseudo code
```python
def function(A, B, Count):
    for i in range(Count):
        if i < 5:
            A = A + 5
            B = B * 2
        else:
            A = A + 1
    return A, B
```

syntax
```
func.func @Function(%A_in: tensor<?xf32>, %B_in: tensor<?xf32>, %Count: index)
    -> (tensor<?xf32>, tensor<?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c5_idx = constant 5 : index
  %c5_f = constant 5.0 : f32
  %c2_f = constant 2.0 : f32
  %c1_f = constant 1.0 : f32

  %A_final, %B_final = statement.for %iv = %c0 to %Count step %c1
                      iter_args(%A_acc = %A_in : tensor<?xf32>,
                                %B_acc = %B_in : tensor<?xf32>) {
    %cond = cmpi "slt", %iv, %c5_idx : index
    %A_then, %B_then = statement.if %cond -> (tensor<?xf32>, tensor<?xf32>) {
      statement.block { ... }
      statement.yield %A_then, %B_then
    } else {
      statement.block { ... }
      statement.yield %A_else, %B_else
    }
    statement.yield %A_then, %B_then
  }

  statement.yield
  func.return %A_final, %B_final : tensor<?xf32>, tensor<?xf32>
}
```

### Attention
### Attention
pseudo code
```python
def attention(Q, K, V):
    d_k = Q.shape[1]
    
    # Step 1: Compute attention scores
    # scores = Q @ K^T / sqrt(d_k)
    scores = matmul(Q, transpose(K)) / sqrt(d_k)
    
    # Step 2: Apply softmax to get attention weights
    weights = softmax(scores, dim=-1)
    
    # Step 3: Weighted sum of values
    output = matmul(weights, V)
    
    return output
```
syntax
```
func.func @Attention(%Q: tensor<?xf32>, %K: tensor<?xf32>, %V: tensor<?xf32>) -> tensor<?xf32> {
  
  // constants
  %d_k = constant 64 : f32

  // single block for all computations
  statement.block {
    // Step 1: scores = Q @ K^T / sqrt(d_k)
    %K_T = tensor.transpose %K : tensor<?xf32>
    %scores_raw = tensor.matmul %Q, %K_T : tensor<?xf32>
    %sqrt_dk = math.sqrt %d_k : f32
    %scores = tensor.div %scores_raw, %sqrt_dk : tensor<?xf32>

    // Step 2: weights = softmax(scores)
    %weights = tensor.softmax %scores : tensor<?xf32>

    // Step 3: output = weights @ V
    %output = tensor.matmul %weights, %V : tensor<?xf32>
  }

  statement.return %output : tensor<?xf32>
}
```

### FlashAttention
pseudo code
```python
def flash_attention(Q, K, V, block_size_m, block_size_n):
    seq_len_q, d = Q.shape
    seq_len_k = K.shape[0]
    output = zeros([seq_len_q, d])
    l = zeros([seq_len_q])  # row-wise max values
    m = full([seq_len_q], -inf)  # row-wise max values for softmax
    
    # Tile over Q dimension
    for i in range(0, seq_len_q, block_size_m):
        q_block = Q[i:i+block_size_m, :]
        o_block = zeros([block_size_m, d])
        l_block = zeros([block_size_m])
        m_block = full([block_size_m], -inf)
        
        # Tile over K/V dimension
        for j in range(0, seq_len_k, block_size_n):
            k_block = K[j:j+block_size_n, :]
            v_block = V[j:j+block_size_n, :]
            
            # Compute attention scores for this block
            s_block = matmul(q_block, transpose(k_block))  # [block_m, block_n]
            
            # Online softmax: update max and compute exp
            m_new = max(m_block, max(s_block, dim=1))
            p_block = exp(s_block - m_new)
            l_new = exp(m_block - m_new) * l_block + sum(p_block, dim=1)
            
            # Update output block
            o_block = (l_block / l_new) * exp(m_block - m_new) * o_block + \
                      matmul(p_block, v_block) / l_new
            
            # Update running statistics
            l_block = l_new
            m_block = m_new
        
        # Write back to output
        output[i:i+block_size_m, :] = o_block
        l[i:i+block_size_m] = l_block
        m[i:i+block_size_m] = m_block
    
    return output
```
syntax
```
func.func @FlashAttention(%Q: tensor<?x?xf32>, %K: tensor<?x?xf32>, %V: tensor<?x?xf32>,
                          %block_size_m: index, %block_size_n: index) -> tensor<?x?xf32> {
  
  // constants
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %neg_inf = constant -inf : f32
  
  // get tensor dimensions
  statement.block {
    %seq_len_q = tensor.dim %Q, %c0 : tensor<?x?xf32>
    %d = tensor.dim %Q, %c1 : tensor<?x?xf32>
    %seq_len_k = tensor.dim %K, %c0 : tensor<?x?xf32>
    
    // initialize output, l, m
    %output = tensor.empty [%seq_len_q, %d] : tensor<?x?xf32>
    %l = tensor.empty [%seq_len_q] : tensor<?xf32>
    %m = tensor.full [%seq_len_q] %neg_inf : tensor<?xf32>
  }
  
  // outer loop: tile over Q dimension
  %output_final, %l_final, %m_final = statement.for %iv_i = %c0 to %seq_len_q step %block_size_m
      iter_args(%output_acc = %output : tensor<?x?xf32>,
                %l_acc = %l : tensor<?xf32>,
                %m_acc = %m : tensor<?xf32>)
  {
    statement.block {
      // extract Q block: Q[i:i+block_size_m, :]
      %i_end = addi %iv_i, %block_size_m : index
      %q_block = tensor.extract_slice %Q[%iv_i, %c0][%block_size_m, %d][%c1, %c1]
          : tensor<?x?xf32> to tensor<?x?xf32>
      
      // initialize block-local accumulators
      %o_block = tensor.empty [%block_size_m, %d] : tensor<?x?xf32>
      %l_block = tensor.empty [%block_size_m] : tensor<?xf32>
      %m_block = tensor.full [%block_size_m] %neg_inf : tensor<?xf32>
    }
    
    // inner loop: tile over K/V dimension
    %o_block_final, %l_block_final, %m_block_final = 
        statement.for %iv_j = %c0 to %seq_len_k step %block_size_n
        iter_args(%o_acc = %o_block : tensor<?x?xf32>,
                  %l_acc_inner = %l_block : tensor<?xf32>,
                  %m_acc_inner = %m_block : tensor<?xf32>)
    {
      statement.block {
        // extract K and V blocks
        %j_end = addi %iv_j, %block_size_n : index
        %k_block = tensor.extract_slice %K[%iv_j, %c0][%block_size_n, %d][%c1, %c1]
            : tensor<?x?xf32> to tensor<?x?xf32>
        %v_block = tensor.extract_slice %V[%iv_j, %c0][%block_size_n, %d][%c1, %c1]
            : tensor<?x?xf32> to tensor<?x?xf32>
        
        // compute attention scores: s_block = q_block @ k_block^T
        %k_block_T = tensor.transpose %k_block : tensor<?x?xf32>
        %s_block = tensor.matmul %q_block, %k_block_T : tensor<?x?xf32>
        
        // online softmax: compute row-wise max
        %s_max = tensor.reduce_max %s_block, dim=1 : tensor<?x?xf32> -> tensor<?xf32>
        %m_new = tensor.maximum %m_acc_inner, %s_max : tensor<?xf32>
        
        // compute exp(s_block - m_new) for each row
        %m_new_expanded = tensor.expand_shape %m_new : tensor<?xf32> -> tensor<?x1xf32>
        %s_block_shifted = tensor.sub %s_block, %m_new_expanded : tensor<?x?xf32>
        %p_block = tensor.exp %s_block_shifted : tensor<?x?xf32>
        
        // compute l_new: exp(m_old - m_new) * l_old + sum(p_block, dim=1)
        %m_diff = tensor.sub %m_acc_inner, %m_new : tensor<?xf32>
        %exp_m_diff = tensor.exp %m_diff : tensor<?xf32>
        %l_old_scaled = tensor.mul %exp_m_diff, %l_acc_inner : tensor<?xf32>
        %p_sum = tensor.reduce_sum %p_block, dim=1 : tensor<?x?xf32> -> tensor<?xf32>
        %l_new = tensor.add %l_old_scaled, %p_sum : tensor<?xf32>
        
        // update output block: (l_old/l_new) * exp(m_old - m_new) * o_old + (p_block @ v_block) / l_new
        %l_ratio = tensor.div %l_acc_inner, %l_new : tensor<?xf32>
        %l_ratio_expanded = tensor.expand_shape %l_ratio : tensor<?xf32> -> tensor<?x1xf32>
        %exp_m_diff_expanded = tensor.expand_shape %exp_m_diff : tensor<?xf32> -> tensor<?x1xf32>
        %o_old_scaled = tensor.mul %l_ratio_expanded, %exp_m_diff_expanded : tensor<?x1xf32>
        %o_old_scaled2 = tensor.mul %o_old_scaled, %o_acc : tensor<?x?xf32>
        
        %p_v = tensor.matmul %p_block, %v_block : tensor<?x?xf32>
        %l_new_expanded = tensor.expand_shape %l_new : tensor<?xf32> -> tensor<?x1xf32>
        %p_v_scaled = tensor.div %p_v, %l_new_expanded : tensor<?x?xf32>
        
        %o_new = tensor.add %o_old_scaled2, %p_v_scaled : tensor<?x?xf32>
      }
      
      statement.yield %o_new, %l_new, %m_new
    }
    
    statement.block {
      // write back to output, l, m
      %output_updated = tensor.insert_slice %o_block_final into %output_acc[%iv_i, %c0]
          : tensor<?x?xf32> into tensor<?x?xf32>
      %l_updated = tensor.insert_slice %l_block_final into %l_acc[%iv_i]
          : tensor<?xf32> into tensor<?xf32>
      %m_updated = tensor.insert_slice %m_block_final into %m_acc[%iv_i]
          : tensor<?xf32> into tensor<?xf32>
    }
    
    statement.yield %output_updated, %l_updated, %m_updated
  }
  
  statement.return %output_final : tensor<?x?xf32>
}
```
---

## 6.7 Summary

The statement dialect provides:

* Structured loop representation
* Explicit memory operations
* Optimization-friendly structure
* Clear path to block graph canonicalization and downstream pipe execution

It enables loop-level optimizations and efficient code generation.

---