# 7. Tensor Dialect

The `pto.tensor` dialect provides high-level tensor algebra operations used by frontends and graph optimizers. It represents computation at the tensor level, focusing on mathematical operations rather than hardware-specific details.

The `pto.tensor` dialect is an SSA-form, mathematical representation with no memory side effects.

---

## 7.1 Purpose

The tensor dialect serves as:

* A high-level abstraction for tensor computations
* A fusion-friendly representation for graph-level optimizations
* A bridge between frontend code and hardware-specific lowerings
* A platform for algebraic rewrites and simplifications

---

## 7.2 Core Operations

### 7.2.1 Algebraic Operations

#### `tensor.matmul`

Matrix multiplication operation.

**Syntax:**
```mlir
%result = pto.tensor.matmul %A, %B : tensor<MxKxf16>, tensor<KxNxf16> -> tensor<MxNxf16>
```

**Shape inference:**
- Input A: `[M x K]`
- Input B: `[K x N]`
- Output: `[M x N]`

**Attributes:**
- `transpose_a` (optional): transpose first operand
- `transpose_b` (optional): transpose second operand
- `alpha` (optional): scaling factor

---

#### `tensor.conv`

Convolution operation.

**Syntax:**
```mlir
%result = pto.tensor.conv %input, %filter {strides = [1, 1], padding = [0, 0, 0, 0]} 
    : tensor<NxHxWxCxf16>, tensor<FHxFWxCxOCxf16> -> tensor<NxOHxOWxOCxf16>
```

**Attributes:**
- `strides`: stride values for each dimension
- `padding`: padding values
- `dilation`: dilation rates
- `groups`: number of groups for grouped convolution

---

#### Elementwise Operations

**Addition:**
```mlir
%result = pto.tensor.add %A, %B : tensor<...xf16>, tensor<...xf16> -> tensor<...xf16>
```

**Multiplication:**
```mlir
%result = pto.tensor.mul %A, %B : tensor<...xf16>, tensor<...xf16> -> tensor<...xf16>
```

**Subtraction:**
```mlir
%result = pto.tensor.sub %A, %B : tensor<...xf16>, tensor<...xf16> -> tensor<...xf16>
```

**Division:**
```mlir
%result = pto.tensor.div %A, %B : tensor<...xf16>, tensor<...xf16> -> tensor<...xf16>
```

All elementwise ops support broadcasting following NumPy rules.

---

### 7.2.2 Normalization Operations

#### `tensor.softmax`

Softmax normalization.

**Syntax:**
```mlir
%result = pto.tensor.softmax %input {axis = -1} : tensor<...xf16> -> tensor<...xf16>
```

**Attributes:**
- `axis`: dimension along which to apply softmax

---

#### `tensor.norm`

L2 normalization.

**Syntax:**
```mlir
%result = pto.tensor.norm %input {axis = -1, epsilon = 1e-5} : tensor<...xf16> -> tensor<...xf16>
```

---

#### `tensor.rms_norm`

Root Mean Square normalization.

**Syntax:**
```mlir
%result = pto.tensor.rms_norm %input {axis = -1, epsilon = 1e-5} : tensor<...xf16> -> tensor<...xf16>
```

---

### 7.2.3 Shape Transformation Operations

#### `tensor.reshape`

Reshape tensor without data movement.

**Syntax:**
```
%result = tensor.reshape %input {shape = [new_shape]} : tensor<...xf16> -> tensor<new_shape xf16>
```

**Verification:**
- Total number of elements must match
- At most one dimension can be `-1` (inferred)

---

#### `tensor.slice`

Produce a value slice of a tensor (sub-tensor extraction or reshape) without introducing aliasing semantics.

**Syntax:**
```
%result = tensor.slice %input
          {offsets = [...], sizes = [...], strides = [...], new_shape = [...]}
          : tensor<...xf16> -> tensor<new_shape xf16>
```

**Notes:**
- Slices are value-based; the result is a new SSA tensor value.
- Lowerings may implement slices via views plus copy-on-write or by materializing the region directly, but tensor-level semantics remain alias-free.

---

#### `tensor.transpose`

Swap two adjacent axes of a tensor (e.g., axis `i` and `i+1`).

**Syntax:**
```
%result = tensor.transpose %input {permutation = [1, 0, 2]} : tensor<...xf16> -> tensor<...xf16>
```

Use this op for lightweight axis swaps commonly used before matmul/conv operations.

---

#### `tensor.permute`

Perform a generalized axis permutation (arbitrary reordering of multiple axes at once).

**Syntax:**
```
%result = tensor.permute %input {permutation = [2, 0, 1]} : tensor<...xf16> -> tensor<...xf16>
```

Use this op when more than a pair of axes must be rearranged or when the permutation is non-adjacent.

---

### 7.2.4 Broadcasting Operations

#### `tensor.broadcast_in_dim`

Explicitly broadcast tensor to target shape.

**Syntax:**
```
%result = tensor.broadcast_in_dim %input {broadcast_dimensions = [0, 2], target_shape = [M, 1, N]} 
    : tensor<MxNxf16> -> tensor<Mx1xNxf16>
```

---

### 7.2.5 Reduction Operations

#### `tensor.reduce`

Generic reduction operation.

**Syntax:**
```
%result = tensor.reduce %input {dimensions = [1, 2], kind = "sum"} : tensor<...xf16> -> tensor<...xf16>
```

**Reduction kinds:**
- `sum`: sum reduction
- `max`: maximum reduction
- `min`: minimum reduction
- `mean`: mean reduction
- `prod`: product reduction

---

### 7.2.6 Indexing Operations

#### `tensor.gather`

Gather elements from tensor using indices.

**Syntax:**
```
%result = tensor.gather %input, %indices {axis = 0} : tensor<...xf16>, tensor<...xi32> -> tensor<...xf16>
```

---

#### `tensor.scatter`

Scatter elements into tensor using indices.

**Syntax:**
```
%result = tensor.scatter %input, %indices, %updates {axis = 0} 
    : tensor<...xf16>, tensor<...xi32>, tensor<...xf16> -> tensor<...xf16>
```

---

### 7.2.7 Shape Operations

#### `tensor.shape_of`

Get the shape of a tensor as a runtime value.

**Syntax:**
```
%shape = tensor.shape_of %input : tensor<...xf16> -> shape<[...]>
```

---

#### `tensor.dim`

Get a specific dimension size.

**Syntax:**
```
%dim_size = tensor.dim %input, %dim_index : tensor<...xf16>, index -> index
```

---

### 7.2.8 Get/Set Value Operations

#### `tensor.update_slice`

Return a new tensor where a sub-region is replaced with the provided patch tensor while preserving SSA semantics (no in-place aliasing).

**Syntax:**
```
%result = tensor.update_slice %input, %patch
          {offsets = [...], sizes = [...], strides = [...]}
          : tensor<...xf16>, tensor<patch_shape xf16> -> tensor<...xf16>
```

**Notes:**
- `%patch` shape must match the specified slice region (`sizes` and `strides` semantics mirror `tensor.slice`).
- The operation yields a fresh tensor value; `%input` remains unchanged.
- Lowerings may implement this via copy-on-write, tiled stores, or statement-level loops, but tensor-level semantics stay purely functional.

---

#### `tensor.extract_element`

Extract a single scalar element from a tensor at the specified indices.

**Syntax:**
```
%scalar = tensor.extract_element %input[%i, %j, ...]
          : tensor<...xf16> -> f16
```

**Notes:**
- All indices must be scalar SSA values (typically `index` type).
- Semantics are value-based: `%scalar` is an independent SSA value.
- Invalid indices result in verification failure; lowering may insert bounds checks if requested.

---

#### `tensor.update_element`

Return a new tensor where a single element is updated with the provided scalar value (SSA semantics preserved).

**Syntax:**
```
%result = tensor.update_element %value into %input[%i, %j, ...]
          : f16, tensor<...xf16> -> tensor<...xf16>
```

**Notes:**
- Produces a fresh tensor value; `%input` remains unchanged.
- Indices semantics mirror `tensor.extract_element`.
- Lowerings may implement this via copy-on-write, tiled stores, or statement-level loops.

---

## 7.3 Type System

Tensor dialect uses the `tensor<shape, element_type, layout?>` type system defined in Section 4.

**Key points:**
- Supports static, dynamic, and symbolic dimensions
- Layout information is optional at this level
- Element types must be scalar types

---

## 7.4 Broadcasting Rules

Tensor dialect follows NumPy-style broadcasting:

1. Align dimensions from the right
2. Dimensions of size 1 can be broadcast
3. Missing dimensions are treated as size 1
4. All dimensions must be compatible (equal or one of them is 1)

**Example:**
```
tensor<3x1x5xf16> + tensor<1x4x5xf16> -> tensor<3x4x5xf16>
```

---

## 7.5 Lowering to Tile Dialect

Tensor operations lower to tile operations through two canonical expansion styles:

### 1. Fully unrolled expansion

The tensor op is rewritten into an explicit collection of tile ops with all loops fully expanded (conceptually “materialize every tile operation”).

Steps:
1. **Tile selection** based on configuration, platform constraints, hints, or auto-tuning.
2. **Operation decomposition**:
   - `tensor.matmul` → emit every `tile.matmul` invocation directly
   - `tensor.add` → emit `tile.add` for each tile (if tile-aligned) or per-element ops
   - `tensor.reshape` → emit layout/view transforms tile-by-tile
3. **Edge handling** for partial tiles at tensor boundaries.

This approach produces a flat sequence of tile ops (no loops), suitable for cases where the iteration space is small or already explicitly enumerated.

### 2. Loop-based expansion via `statement.for`

The tensor op is rewritten into nested `statement.for` loops that enumerate tiles, with tile ops inside the loops.

Example:

```mlir
// Before (tensor dialect)
%result = pto.tensor.matmul %A, %B :
          tensor<1024x512xf16>, tensor<512x256xf16> -> tensor<1024x256xf16>

// After (loop-based expansion)
statement.scope() {
  statement.for %i = 0 to 1024 step 16 {
    statement.scope(%i) {
      statement.for %j = 0 to 256 step 16 {
        statement.scope(%j) {
          statement.for %k = 0 to 512 step 16 {
            statement.scope(%k) {
              %tile_A = tile.load %A[%i:%i+16, %k:%k+16]
              %tile_B = tile.load %B[%k:%k+16, %j:%j+16]
              %tile_C = tile.matmul %tile_A, %tile_B
              tile.store %tile_C, %result[%i:%i+16, %j:%j+16]
            }
          }
        }
      }
    }
  }
  statement.yield
}
```

This style leverages the statement dialect to represent iteration domains, enabling loop transformations (fusion, tiling, unrolling) before lowering to blockgraph/pipe exec.

Both expansion styles must enforce identical semantics; the lowering pass selects one depending on optimization goals and backend requirements.

---

## 7.6 Verification Rules

A valid tensor operation must satisfy:

1. **Type correctness**: Operand and result types must be valid tensor types
2. **Shape compatibility**: Shapes must be compatible for the operation
3. **Broadcasting validity**: Broadcasting rules must be satisfied
4. **Dimension bounds**: All dimension indices must be within valid range
5. **Layout consistency**: Layouts must be compatible (if specified)

---

## 7.7 Optimization Opportunities

The tensor dialect is designed to enable various optimizations:

### Algebraic Simplification

- Identity operations: `tensor.add %x, 0` → `%x`
- Constant folding: operations with constant operands
- Associativity: `(A + B) + C` → `A + (B + C)`

### Layout Propagation and Optimization

- Layout propagation: choose optimal layouts for operations
- Transpose elimination: `transpose(transpose(A))` → `A`

---

## 7.8 Examples

### Matrix Multiplication

```mlir
pto.func.func @matmul_kernel(%A: tensor<1024x512xf16>, %B: tensor<512x256xf16>) 
    -> (tensor<1024x256xf16>) {
  %C = pto.tensor.matmul %A, %B : tensor<1024x512xf16>, tensor<512x256xf16> -> tensor<1024x256xf16>
  pto.func.return %C
}
```

### Elementwise Operations with Broadcasting

```mlir
pto.func.func @add_broadcast(%A: tensor<1x128xf16>, %B: tensor<32x128xf16>) 
    -> (tensor<32x128xf16>) {
  %result = pto.tensor.add %A, %B : tensor<1x128xf16>, tensor<32x128xf16> -> tensor<32x128xf16>
  func.return %result
}
```

### Softmax

```mlir
pto.func.func @softmax(%input: tensor<32x128xf16>) -> (tensor<32x128xf16>) {
  %result = pto.tensor.softmax %input {axis = 1} : tensor<32x128xf16> -> tensor<32x128xf16>
  pto.func.return %result
}
```

---

## 7.9 Summary

The tensor dialect provides:

* High-level tensor algebra operations
* Shape-aware operations with dynamic shape support
* Broadcasting and reduction semantics
* Fusion-friendly representation
* Clear lowering path to tile dialect

It serves as the primary interface between frontend code and the PTO-IR optimization pipeline.

---

