# IR Examples
This directory contains IR (Intermediate Representation) examples in text format, based on the documentation in `docs/ir/`.
## Overview
These examples demonstrate the IR text format syntax as documented in:
- `docs/ir/ir_builder.md` - IRBuilder API documentation
- `docs/ir/ir_statement.md` - Statement types and IR text format
- `docs/ir/ir_type_and_value.md` - Type system and value types
## Examples
### example00_basic_operations.ir
**Based on:** `docs/ir/ir_statement.md` Example 1
Demonstrates basic tensor operations:
- Program module and entry point
- Function with tensor inputs/outputs
- View operation to extract a tile from tensor
- Binary operations (mul) with tensors and scalars
- Assemble operation to write results back
**Key Features:**
- `program.module` and `program.entry`
- `func.func` with tensor parameters
- `statement.op` block with operations
- `statement.return`
### example01_tile_scalar_operations.ir
**Based on:** `docs/ir/ir_statement.md` Example 2
Demonstrates tile and scalar operations:
- Kernel function with tile inputs/outputs
- Tile operations (OP_MUL, OP_ADD, OP_SUB, OP_DIV)
- Scalar operations (OP_SCALAR_ADD, OP_SCALAR_MUL)
- Returning scalar values
**Key Features:**
- `func.kind = kernel`
- Tile type: `tile<[16, 32], [16, 32], fp32>`
- Scalar constants and operations
- Return statement with scalar value
### example02_for_loop.ir
**Based on:** `docs/ir/ir_statement.md` ForStatement section
Demonstrates for loop structure:
- For statement with iteration variable
- Loop bounds and step
- Iteration arguments (loop-carried variables)
- Yield statement for loop results
**Key Features:**
- `statement.for` with `iter_args`
- Loop body with operations
- `statement.yield` for loop-carried values
### example03_if_statement.ir
**Based on:** `docs/ir/ir_statement.md` IfStatement section
Demonstrates if-else statement:
- If statement with condition
- Then and else branches
- Yield statements in each branch
- Result values from if statement
**Key Features:**
- `statement.if` with condition
- Then and else regions
- `statement.yield` in each branch
### example04_nested_control_flow.ir
Demonstrates nested control flow:
- For loop containing an if statement
- Multiple loop-carried variables
- Conditional operations inside loop
- Complex control flow patterns
**Key Features:**
- Nested `statement.for` and `statement.if`
- Multiple `iter_args` and `yield` values
- Result tuple access (`%r0#0`, `%r0#1`)
### example05_types_and_values.ir
**Based on:** `docs/ir/ir_type_and_value.md`
Demonstrates different types and values:
- Scalar constants and symbolic values
- Tensor values with symbolic dimensions
- Type annotations
**Key Features:**
- Scalar constants: `%const_0 = 0 : int64`
- Symbolic scalars: `%scale1_4 : fp32`
- Tensor types: `tensor<[%b_1, 128], fp32>`
### example06_binary_operations.ir
Demonstrates binary operations:
- Addition, subtraction, multiplication, division
- Chained operations
- Result assembly
**Key Features:**
- `tensor.add`, `tensor.sub`, `tensor.mul`, `tensor.div`
- Operation chaining
### example07_unary_operations.ir
Demonstrates unary operations:
- Negation, absolute value, exponential, square root, logarithm
- Unary operation sequences
**Key Features:**
- `tensor.neg`, `tensor.abs`, `tensor.exp`, `tensor.sqrt`, `tensor.ln`
### example08_scalar_operations.ir
Demonstrates scalar operations:
- Scalar arithmetic operations
- Scalar return values
- Scalar constant definitions
**Key Features:**
- `tensor.OP_SCALAR_ADD`, `tensor.OP_SCALAR_MUL`, etc.
- Return statement with scalar value
### example09_softmax.ir
Demonstrates softmax operation:
- Exponential and normalization operations
- Reduction operations
**Key Features:**
- `tensor.exp`, `tensor.div`
- Reduction patterns
### example10_mla.ir
Demonstrates Multi-Head Latent Attention (MLA):
- Attention computation patterns
- Matrix operations
**Key Features:**
- `tensor.mul` for attention computation
- Complex tensor operations
### example11_flash_attention.ir
Demonstrates Flash Attention:
- Efficient attention computation with block-wise processing
- Block-wise attention computation
**Key Features:**
- `statement.for` with block processing
- `tensor.mul` for attention scores
- Efficient memory patterns
## IR Structure
The IR follows this hierarchical structure (from `docs/ir/ir_statement.md`):
```
program
  └── function
        ├── statement.op
        ├── statement.for
        │     ├── statement.if
        │     |     |   // then
        │     │     ├── statement.op
        │     │     ├── statement.yield
        |     |     |   // else
        │     │     ├── statement.op
        │     │     └── statement.yield
        │     ├── statement.op
        │     └── statement.yield
        ├── statement.op
        └── statement.return
```
## IR Syntax Elements
### Program Module
```ir
program.module @main {
  program.entry @function_name
  attr arch = "PTOv2"
  attr enable_debug = true
  attr tile_default = { M=16, N=16, K=16 }
  ...
}
```
### Function
```ir
func.func @function_name(%arg1: tensor<[%b, 128], fp32>, %arg2: fp32) -> (fp64) {
  ...
}
// func.kind = control_flow
```
### Statement Types
- `statement.op { ... }` - Operation block
- `statement.for %iv = %start to %end step %step iter_args(...) { ... }` - For loop
- `statement.if %cond { ... } else { ... }` - If statement
- `statement.yield %value0, %value1, ...` - Yield values
- `statement.return %value0, %value1, ...` - Return values
### Value Types
- **Scalar**: `%name : fp32` or `%name = 3.14 : fp64`
- **Tensor**: `%name : tensor<[%b, 128], fp32>`
- **Tile**: `%name : tile<[16, 32], [16, 32], fp32>`
### Operations
- Binary: `tensor.mul %lhs, %rhs : (type1, type2) -> result_type`
- Unary: `tensor.neg %input : (type) -> result_type`
- Scalar: `tensor.OP_SCALAR_ADD %lhs, %rhs : (type1, type2) -> result_type`
## References
- `docs/ir/ir_builder.md` - IRBuilder API documentation
- `docs/ir/ir_statement.md` - Statement types and IR text format
- `docs/ir/ir_type_and_value.md` - Type system and value types
