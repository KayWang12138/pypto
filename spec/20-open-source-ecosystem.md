# 20. Open Source Ecosystem Integration

This section describes how various open-source frontends can be integrated with PTO-IR, enabling developers to leverage existing tools and frameworks while targeting PTO hardware.

---

## 20.1 Overview

PTO-IR supports integration with multiple frontend frameworks, allowing developers to write code in familiar languages and frameworks that are then lowered to PTO-IR. The main frontends supported are:

* **PyTorch**: Python-based deep learning framework
* **Triton**: GPU kernel programming language
* **TileLang**: Domain-specific language for tensor operations

Each frontend has its own lowering path to PTO-IR, preserving semantics while adapting to PTO-IR's structure and dialects.

---

## 20.2 PyTorch Frontend（TODO）

### 20.2.1 Overview

PyTorch frontend integration allows PyTorch models and operations to be compiled to PTO-IR. This enables running PyTorch workloads on PTO hardware with minimal code changes.

### 20.2.2 Integration Architecture

The PyTorch-to-PTO-IR lowering process follows these steps:

1. **TorchScript/TorchDynamo Capture**: Capture PyTorch computation graph
2. **Operator Mapping**: Map PyTorch operators to PTO-IR operations
3. **Type Conversion**: Convert PyTorch tensor types to PTO-IR tensor types
4. **IR Generation**: Generate PTO-IR program structure

### 20.2.3 Operator Mapping

Common PyTorch operators map to PTO-IR operations as follows:

| PyTorch Operator | PTO-IR Operation |
|-----------------|------------------|
| `torch.matmul` | `pto.tensor.matmul` |
| `torch.add` | `pto.tensor.add` |
| `torch.mul` | `pto.tensor.mul` |
| `torch.relu` | `pto.tensor.relu` |
| `torch.softmax` | `pto.tensor.softmax` |
| `torch.layer_norm` | `pto.tensor.layer_norm` |
| `torch.rms_norm` | `pto.tensor.rms_norm` |

### 20.2.4 Example: PyTorch Model to PTO-IR

**PyTorch Code:**
```python
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
```

**Lowered PTO-IR:**
```mlir
pto.program.module @pytorch_model {
  pto.program.entry @forward
  
  pto.program.global @linear1_weight : tensor<1024x512xf16> = {...}
  pto.program.global @linear1_bias : tensor<512xf16> = {...}
  pto.program.global @linear2_weight : tensor<512x256xf16> = {...}
  pto.program.global @linear2_bias : tensor<256xf16> = {...}
  
  pto.func.func @forward(%input: tensor<?x1024xf16>) -> (tensor<?x256xf16>) {
    // Load weights
    %w1 = pto.program.get_global @linear1_weight
    %b1 = pto.program.get_global @linear1_bias
    %w2 = pto.program.get_global @linear2_weight
    %b2 = pto.program.get_global @linear2_bias
    
    // First linear layer
    %hidden = pto.tensor.matmul %input, %w1 : tensor<?x1024xf16>, tensor<1024x512xf16> -> tensor<?x512xf16>
    %hidden_bias = pto.tensor.add %hidden, %b1 : tensor<?x512xf16>, tensor<512xf16> -> tensor<?x512xf16>
    
    // ReLU activation
    %activated = pto.tensor.relu %hidden_bias : tensor<?x512xf16> -> tensor<?x512xf16>
    
    // Second linear layer
    %output = pto.tensor.matmul %activated, %w2 : tensor<?x512xf16>, tensor<512x256xf16> -> tensor<?x256xf16>
    %output_bias = pto.tensor.add %output, %b2 : tensor<?x256xf16>, tensor<256xf16> -> tensor<?x256xf16>
    
    pto.func.return %output_bias
  }
}
```

### 20.2.5 Integration Points

* **TorchDynamo**: Use TorchDynamo to capture FX graphs from PyTorch models
* **FX Graph Lowering**: Lower FX graph nodes to PTO-IR operations
* **Shape Inference**: Preserve and propagate shape information through lowering
* **Gradient Support**: Support `torch.autograd` for training workloads (future work)

---

## 20.3 Triton Frontend

### 20.3.1 Overview

Triton is a GPU kernel programming language that provides high-level abstractions for writing efficient GPU kernels. The Triton frontend allows Triton kernels to be compiled to PTO-IR, enabling portability between GPU and PTO hardware.

### 20.3.2 Integration Architecture

Triton-to-PTO-IR lowering involves:

1. **Triton IR Parsing**: Parse Triton's internal IR representation
2. **Block Structure Mapping**: Map Triton's block-based computation to PTO-IR block graphs
3. **Memory Hierarchy Mapping**: Map Triton's memory hierarchy to PTO-IR memory dialects
4. **Tile Operation Mapping**: Map Triton tile operations to PTO-IR tile dialect

### 20.3.3 Triton Block to PTO-IR Block Graph

Triton's block-based programming model maps naturally to PTO-IR's block graph dialect:

**Triton Code:**
```python
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(A, B, C, M, N, K):
    pid = tl.program_id(0)
    block_start = pid * 16
    offsets = block_start + tl.arange(0, 16)
    
    a = tl.load(A + offsets)
    b = tl.load(B + offsets)
    c = tl.dot(a, b)
    tl.store(C + offsets, c)
```

**Lowered PTO-IR:**
```mlir
pto.blockgraph.block @matmul_block(
    %A: memref<16xf16,ND,L1>,
    %B: memref<16xf16,ND,L1>)
    -> (memref<16xf16,ND,L1>) {
  %tile_A = pto.tile.load_tile %A
  %tile_B = pto.tile.load_tile %B
  pto.blockgraph.dep %tile_A -> %tile_B {type = "RAW"}
  %tile_C = pto.tile.matmul %tile_A, %tile_B
  pto.blockgraph.yield %tile_C
}

pto.func.func @matmul_kernel(%A: memref<?xf16,ND,L1>,
                              %B: memref<?xf16,ND,L1>,
                              %C: memref<?xf16,ND,L1>,
                              %M: i32, %N: i32, %K: i32) {
  pto.statement.for %pid = 0 to %M step 16 {
    %block_start = pto.scalar.mul %pid, 16 : i32, i32 -> i32
    %A_offset = pto.mem.offset %A, %block_start
    %B_offset = pto.mem.offset %B, %block_start
    %C_offset = pto.mem.offset %C, %block_start
    
    %result = pto.exec.call_block @matmul_block(
        inputs = [%A_offset, %B_offset],
        outputs = [%C_offset])
  }
  pto.func.return
}
```

### 20.3.4 Memory Hierarchy Mapping

Triton's memory hierarchy concepts map to PTO-IR:

| Triton Concept | PTO-IR Mapping |
|----------------|----------------|
| Global memory | `memref<...,ND,DDR>` |
| Shared memory | `memref<...,ND,L1>` |
| Register tiles | `memref<...,ND,L0>` |

### 20.3.5 Integration Points

* **Triton IR Parser**: Parse Triton's LLVM-based IR
* **Block Program ID**: Map `tl.program_id` to PTO-IR execution graph nodes
* **Tile Operations**: Direct mapping of Triton tile ops to PTO-IR tile dialect
* **Atomic Operations**: Support Triton atomic operations (future work)

---

## 20.4 TileLang Frontend

### 20.4.1 Overview

TileLang is a domain-specific language designed for expressing tensor computations with explicit tiling and memory hierarchy management. It provides a natural path to PTO-IR due to similar design principles.

### 20.4.2 Integration Architecture

TileLang-to-PTO-IR lowering:

1. **TileLang AST Parsing**: Parse TileLang source code
2. **Tile Operation Extraction**: Extract tile-level operations
3. **Memory Layout Mapping**: Map TileLang memory layouts to PTO-IR memref layouts
4. **Loop Structure Mapping**: Map TileLang loops to PTO-IR statement dialect

### 20.4.3 TileLang Example

**TileLang Code:**
```tilelang
func matmul(A: tensor<1024x512xf16>, B: tensor<512x256xf16>) -> tensor<1024x256xf16> {
  let C = alloc<1024x256xf16>
  
  for i in 0..1024 step 16 {
    for j in 0..256 step 16 {
      for k in 0..512 step 16 {
        let tile_A = load_tile(A[i:i+16, k:k+16])
        let tile_B = load_tile(B[k:k+16, j:j+16])
        let tile_C = matmul_tile(tile_A, tile_B)
        store_tile(tile_C, C[i:i+16, j:j+16])
      }
    }
  }
  
  return C
}
```

**Lowered PTO-IR:**
```mlir
pto.func.func @matmul(%A: memref<1024x512xf16,ND,L1>,
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

### 20.4.4 TileLang Features Mapping

| TileLang Feature | PTO-IR Mapping |
|-----------------|----------------|
| `tensor<T>` | `tensor<T>` or `memref<T,ND,L1>` |
| `load_tile` | `pto.tile.load_tile` |
| `store_tile` | `pto.tile.store_tile` |
| `matmul_tile` | `pto.tile.matmul` |
| `for` loops | `pto.statement.for` |
| Memory annotations | PTO-IR memref memory space attributes |

### 20.4.5 Integration Points

* **Direct Syntax Mapping**: TileLang syntax closely matches PTO-IR concepts
* **Tile Operations**: One-to-one mapping for most tile operations
* **Memory Management**: TileLang's explicit memory management aligns with PTO-IR
* **Optimization Hints**: TileLang annotations can inform PTO-IR optimization passes

---

## 20.5 Lowering Pipeline

### 20.5.1 Common Lowering Steps

All frontends follow a similar lowering pipeline:

1. **Frontend IR Parsing**: Parse frontend-specific IR or source code
2. **Semantic Analysis**: Perform type checking and shape inference
3. **Operator Mapping**: Map frontend operations to PTO-IR operations
4. **IR Construction**: Build PTO-IR program structure
5. **Optimization**: Apply PTO-IR optimization passes
6. **Code Generation**: Generate target code from PTO-IR

### 20.5.2 Frontend-Specific Considerations

**PyTorch:**
* Handle dynamic shapes and control flow
* Preserve gradient computation graphs
* Support JIT compilation paths

**Triton:**
* Preserve block-based parallelism
* Map memory hierarchy correctly
* Handle program ID and grid dimensions

**TileLang:**
* Preserve explicit tiling information
* Maintain memory layout annotations
* Support domain-specific optimizations

---

## 20.6 Best Practices

### 20.6.1 Frontend Selection

Choose the appropriate frontend based on use case:

* **PyTorch**: For existing PyTorch models and Python workflows
* **Triton**: For GPU kernel developers familiar with Triton
* **TileLang**: For new projects requiring explicit tiling control

### 20.6.2 Optimization Opportunities

* Leverage frontend-specific optimizations before lowering
* Use PTO-IR optimization passes after lowering
* Combine frontend and PTO-IR optimizations for best results

### 20.6.3 Debugging

* Preserve source location information during lowering
* Use frontend debugging tools before lowering
* Use PTO-IR verification and debugging tools after lowering

---

## 20.7 Future Work

Potential enhancements to frontend integration:

* **Additional Frontends**: Support for JAX, TensorFlow, ONNX
* **Bidirectional Translation**: Convert PTO-IR back to frontend formats
* **Hybrid Execution**: Support mixed frontend/PTO-IR execution
* **Auto-tuning Integration**: Leverage frontend auto-tuning capabilities
* **Training Support**: Full support for training workloads from all frontends

---

