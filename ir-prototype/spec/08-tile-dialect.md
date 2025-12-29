# 8. Tile Dialect

The `pto.tile` dialect represents tile-level operations that map to cube and vector units. It bridges the gap between high-level tensor operations and block graph canonicalization.

---

## 8.1 Purpose

The tile dialect serves to:

* Express computation at the tile granularity (fixed-size blocks)
* Map operations to hardware cube/vector units
* Represent tile formats and transformations
* Enable tile-level optimizations and scheduling

---

## 8.2 Core Operations

### 8.2.1 Tile Load/Store

#### `tile.load_tile`

Load a tile from memory into a tile buffer.

**Syntax:**
```
%tile = tile.load_tile %memref[%i, %j] {tile_shape = [16, 16], format = ND} 
    : memref<...xf16, ND, L1> -> tile<16x16xf16, ND>
```

**Attributes:**
- `tile_shape`: dimensions of the tile
- `format`: tile format (ND, NZ, FRACTAL_Z, etc.)
- `memory_space`: source memory space

---

#### `tile.store_tile`

Store a tile from buffer to memory.

**Syntax:**
```
tile.store_tile %tile, %memref[%i, %j] {format = ND} 
    : tile<16x16xf16, ND>, memref<...xf16, ND, L1>
```

---

### 8.2.2 Tile Computation

#### `tile.matmul`

Tile-level matrix multiplication.

**Syntax:**
```
%result = tile.matmul %A, %B : tile<16x16xf16, ND>, tile<16x16xf16, ND> -> tile<16x16xf16, ND>
```

**Semantics:**
- Performs matrix multiply-accumulate on tile-sized operands
- Maps to hardware MMU operations
- Supports different tile formats for inputs/outputs

**Attributes:**
- `transpose_a`, `transpose_b`: transpose operands
- `alpha`: scaling factor

---

#### `tile.reduce`

Tile-level reduction.

**Syntax:**
```
%result = tile.reduce %input {axis = 0, kind = "sum"} : tile<16x16xf16, ND> -> tile<16xf16, ND>
```

**Reduction kinds:**
- `sum`, `max`, `min`, `mean`, `prod`

---

#### `tile.matmul_acc`

Fused multiply-add operation.

**Syntax:**
```
%result = tile.fma %A, %B, %C : tile<16x16xf16>, tile<16x16xf16>, tile<16x16xf16> -> tile<16x16xf16>
```

**Semantics:** `result = A * B + C`

---

### 8.2.3 Tile Transformations

#### `tile.transpose`

Transpose a tile.

**Syntax:**
```
%result = tile.transpose %input : tile<16x16xf16, ND> -> tile<16x16xf16, ND>
```

---

#### `tile.set_format`

Change tile format (e.g., ND to NZ).

**Syntax:**
```
%result = tile.set_format %input {from_format = ND, to_format = NZ} 
    : tile<16x16xf16, ND> -> tile<16x16xf16, NZ>
```

**Note:** Format conversion may involve data reorganization.

---

## 8.3 Control Edge（TODO）

## 8.4 Tile Types

Tile types are defined as:

```
tile<tx x ty x elem_type, format>
```

**Constraints:**
- Tile dimensions must be fully static
- Tile size must fit in target memory space (L0A/B/C, UB) as specified by platform abstractions (see Section 17)
- Format must be supported by hardware
- Tile sizes may be constrained by configuration settings (see Section 16)

**Supported formats:**
- `ND`: Normal dense format (row-major)
- `NZ`: Blocked format (16x16 blocks)
- `FRACTAL_Z`: Fractal Z-format for matrix operations
- Custom formats (via extensibility)

---

## 8.5 Tile Formats

### ND Format

Normal dense format, row-major layout.

**Use cases:**
- General-purpose tile operations
- Initial tile loads from memory

---

### NZ Format

Blocked format with 16x16 blocks.

**Use cases:**
- Optimized for matrix multiplication units
- Better cache locality for certain operations

**Constraints:**
- Dimensions must be multiples of 16 (or handle tails)

---

### FRACTAL_Z Format

Fractal Z-format optimized for matrix operations.

**Use cases:**
- High-performance matrix multiplication
- Hardware-optimized layout

---

## 8.6 Lowering to block graph（TODO）

### Example Lowering（TODO）

```

```

---

## 8.7 Verification Rules

A valid tile operation must satisfy:

1. **Tile size constraints**: Tile dimensions must fit in hardware buffers
2. **Format compatibility**: Formats must be compatible for operations
3. **Memory space validity**: Source/destination memory spaces must be valid
4. **Dimension matching**: Tile dimensions must match operation requirements

---

## 8.8 Hardware Mapping

Tile operations map to hardware as follows:

- `tile.matmul` → cube units
- `tile.load_tile` → DMA or load instructions to L0 buffers
- `tile.store_tile` → Store instructions from L0 buffers
- `tile.reduce` → vector units
- `tile.matmul_acc` → cube units

---

## 8.9 Examples

### Tile Matrix Multiplication

```
func.func @tile_matmul(%A: memref<1024x512xf16, ND, L1>, 
                       %B: memref<512x256xf16, ND, L1>) 
    -> memref<1024x256xf16, ND, L1> {
  %C = statement.alloc : memref<1024x256xf16, ND, L1>
  
  statement.for %i = 0 to 1024 step 16 {
    statement.for %j = 0 to 256 step 16 {
      statement.for %k = 0 to 512 step 16 {
        %tile_A = tile.load_tile %A[%i, %k] {tile_shape = [16, 16]}
        %tile_B = tile.load_tile %B[%k, %j] {tile_shape = [16, 16]}
        %tile_C = tile.matmul %tile_A, %tile_B
        tile.store_tile %tile_C, %C[%i, %j]
      }
    }
  }
  
  func.return %C
}
```

### Format Conversion

```
%tile_nd = tile.load_tile %input {tile_shape = [16, 16], format = ND}
%tile_nz = tile.set_format %tile_nd {from_format = ND, to_format = NZ}
tile.store_tile %tile_nz, %output {format = NZ}
```

---

## 8.10 Summary

The tile dialect provides:

* Hardware-aware tile operations
* Support for multiple tile formats
* Direct mapping to matrix/vector units
* Clear interface between tensor and statement levels

It enables efficient code generation for tile-based accelerators.

---

