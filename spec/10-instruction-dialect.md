# 10. Instruction Dialect

The `pto.inst` dialect represents the lowest-level operations that map directly to hardware instructions. It provides a direct interface to hardware capabilities.

---

## 10.1 Purpose

The instruction dialect serves to:

* Represent hardware-level operations
* Provide direct mapping to instruction set architecture
* Enable final code generation
* Express hardware-specific optimizations

---

## 10.2 Core Operations

### 10.2.1 Matrix Operations

#### `inst.mma`

Matrix multiply-accumulate instruction.

**Syntax:**
```
inst.mma %A, %B, %C -> %D 
    : mma.tile<16x16xf16>, mma.tile<16x16xf16>, mma.tile<16x16xf16> -> mma.tile<16x16xf16>
```

**Semantics:**
- Performs `D = A * B + C`
- Maps directly to hardware MMU

---

### 10.2.2 Load/Store Operations

#### `inst.load_L0A`

Load data into L0A buffer.

**Syntax:**
```
inst.load_L0A %memref[%i, %j] -> %L0A_buffer
    : memref<...xf16, ND, L1> -> mma.tile<16x16xf16>
```

---

#### `inst.load_L0B`

Load data into L0B buffer.

**Syntax:**
```
inst.load_L0B %memref[%i, %j] -> %L0B_buffer
    : memref<...xf16, ND, L1> -> mma.tile<16x16xf16>
```

---

#### `inst.store`

Store data from buffer to memory.

**Syntax:**
```
inst.store %buffer, %memref[%i, %j]
    : mma.tile<16x16xf16>, memref<...xf16, ND, L1>
```

---

### 10.2.3 Arithmetic Operations

#### `inst.add`

Add instruction.

**Syntax:**
```
%result = inst.add %a, %b : reg.f16, reg.f16 -> reg.f16
```

---

#### `inst.mul`

Multiply instruction.

**Syntax:**
```
%result = inst.mul %a, %b : reg.f16, reg.f16 -> reg.f16
```

---

### 10.2.4 Synchronization

#### `inst.barrier`

Hardware synchronization barrier.

**Syntax:**
```
inst.barrier
```

**Semantics:**
- Hardware-level synchronization
- All previous instructions must complete

---

### 10.2.5 Atomic Operations

#### `inst.atomic_add`

Atomic add operation.

**Syntax:**
```
inst.atomic_add %memref[%i], %value : memref<...xf16>, reg.f16
```

---

### 10.2.6 Control Flow

#### `inst.branch`

Unconditional branch.

**Syntax:**
```
inst.branch ^target_block
```

---

#### `inst.cond_branch`

Conditional branch.

**Syntax:**
```
inst.cond_branch %cond, ^then_block, ^else_block : i1
```

---

## 10.3 Instruction Types

### Register Types

```
reg.f16
reg.f32
reg.i32
reg.vec<f16, 8>  // vector register
reg.pred         // predicate register
```

### Micro-tile Types

```
mma.tile<16x16xf16>  // matrix fragment in MMU
```

### Address Types

```
inst.addr          // resolved address
inst.immediate     // immediate value
```

---

## 10.4 Hardware Mapping

Instruction operations map directly to hardware:

- `inst.mma` → MMU instruction
- `inst.load_L0A/B` → Load instruction to L0 buffers
- `inst.store` → Store instruction from buffers
- `inst.barrier` → Hardware barrier instruction
- `inst.atomic_add` → Atomic instruction

---

## 10.5 Verification Rules

A valid instruction must satisfy:

1. **Register allocation**: All operands must be allocated registers
2. **Address resolution**: All addresses must be resolved
3. **Type compatibility**: Operand types must match instruction requirements
4. **Schedule constraints**: Instruction ordering must respect dependencies

---

## 10.6 Examples

### Matrix Multiplication Kernel

```
func.func @mma_kernel(%A: memref<1024x512xf16, ND, L1>,
                      %B: memref<512x256xf16, ND, L1>)
    -> memref<1024x256xf16, ND, L1> {
  %C = mem.alloc : memref<1024x256xf16, ND, L1>
  
  statement.for %i = 0 to 1024 step 16 {
    statement.for %j = 0 to 256 step 16 {
      statement.for %k = 0 to 512 step 16 {
        // Load tiles
        %tile_A = inst.load_L0A %A[%i, %k]
        %tile_B = inst.load_L0B %B[%k, %j]
        inst.barrier
        
        // Matrix multiply
        %tile_C = inst.mma %tile_A, %tile_B, %zero
        inst.barrier
        
        // Store result
        inst.store %tile_C, %C[%i, %j]
      }
    }
  }
  
  func.return %C
}
```

---

## 10.7 Summary

The instruction dialect provides:

* Direct hardware instruction mapping
* Low-level register and buffer operations
* Final representation for code generation
* Hardware-specific optimizations

It serves as the final IR before binary code generation.

---

