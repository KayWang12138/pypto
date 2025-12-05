# Basic Operations Example

A beginner-friendly example demonstrating fundamental PyPTO operations.

## Overview

This example covers the core operations you'll use in most PyPTO programs:
- Tensor creation and properties
- Element-wise arithmetic operations
- Matrix multiplication
- Activation functions
- View operations for tiling
- Combining multiple operations

## Features

- **Beginner-friendly**: Simple, well-commented code
- **Comprehensive**: Covers all basic operation types
- **Verified**: Each example includes PyTorch verification
- **Educational**: Step-by-step explanations

## Prerequisites

- PyPTO installed (see [Installation Guide](../../../docs/user/getting_started/installation.md))
- CANN environment configured
- NPU device available
- PyTorch and torch_npu installed

## Running the Example

### Prerequisites

```bash
# Source CANN environment
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

# Set device ID (required for NPU examples)
export TILE_FWK_DEVICE_ID=0
```

### Basic Usage

```bash
# Run all examples
python3 examples/beginner/01_basic_operations/basic_operations.py

# Run specific example by ID
python3 examples/beginner/01_basic_operations/basic_operations.py 1  # Tensor Creation
python3 examples/beginner/01_basic_operations/basic_operations.py 2  # Element-wise Operations
python3 examples/beginner/01_basic_operations/basic_operations.py 3  # Matrix Multiplication
python3 examples/beginner/01_basic_operations/basic_operations.py 4  # Activation Functions
python3 examples/beginner/01_basic_operations/basic_operations.py 5  # View Operations
python3 examples/beginner/01_basic_operations/basic_operations.py 6  # Combined Operations

# List all available examples
python3 examples/beginner/01_basic_operations/basic_operations.py --list
```

### Available Examples

1. **Tensor Creation** - Creating tensors with different properties (No NPU required)
2. **Element-wise Operations** - Element-wise arithmetic operations (add, mul)
3. **Matrix Multiplication** - Matrix multiplication operations
4. **Activation Functions** - Activation functions (sigmoid)
5. **View Operations** - View operations for tiling
6. **Combined Operations** - Combining multiple operations

## Examples Included

### Example 1: Tensor Creation

Demonstrates how to create tensors and access their properties:

```python
tensor = pypto.tensor([4, 4], pypto.DT_FP16, "my_tensor")
print(f"Shape: {tensor.shape}, Dtype: {tensor.dtype}")
```

**Key Concepts**:
- Tensor creation with shape, dtype, and name
- Accessing tensor properties
- Understanding symbolic tensors

### Example 2: Element-wise Operations

Shows basic arithmetic operations:

```python
@pypto.jit
def element_wise_ops(inputs, outputs):
    a, b = inputs[0], inputs[1]
    result = outputs[0]
    pypto.set_vec_tile_shapes(8, 8)
    result[:] = pypto.mul(pypto.add(a, b), 2.0)
```

**Key Concepts**:
- Addition, multiplication with scalars
- Chaining operations
- Vector tiling configuration

### Example 3: Matrix Multiplication

Demonstrates matrix operations:

```python
@pypto.jit
def matrix_multiply(inputs, outputs):
    A, B = inputs[0], inputs[1]
    C = outputs[0]
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    C[:] = pypto.matmul(A, B, out_dtype=pypto.DT_BF16)
```

**Key Concepts**:
- Matrix multiplication
- Cube tiling configuration
- Output dtype specification

### Example 4: Activation Functions

Shows how to apply activation functions:

```python
@pypto.jit
def apply_activations(inputs, outputs):
    x = inputs[0]
    result = outputs[0]
    pypto.set_vec_tile_shapes(32, 64)
    result[:] = pypto.sigmoid(x)
```

**Key Concepts**:
- Activation functions (sigmoid, relu, gelu, etc.)
- Unary operations

### Example 5: View Operations

Demonstrates tiling with views:

```python
@pypto.jit
def tiled_operation(inputs, outputs):
    # Create views for tiling
    view = pypto.view(tensor, [tile_h, tile_w], [offset_h, offset_w])
    # Process tile
    result = process(view)
    # Assemble back
    pypto.assemble(result, [offset_h, offset_w], output)
```

**Key Concepts**:
- View creation for tiling
- Loop-based processing
- Assembly operations

### Example 6: Combined Operations

Shows combining multiple operations:

```python
@pypto.jit
def linear_layer_with_activation(inputs, outputs):
    # y = sigmoid(x @ W + b)
    linear = pypto.matmul(x, W, out_dtype=pypto.DT_BF16)
    biased = pypto.add(linear, b)
    y[:] = pypto.sigmoid(biased)
```

**Key Concepts**:
- Operation composition
- Linear layers
- Combining matrix and vector operations

## Expected Output

```
============================================================
PyPTO Basic Operations Examples
============================================================

============================================================
Example 1: Tensor Creation
============================================================
Tensor name: my_tensor
Tensor shape: [4, 4]
Tensor dtype: DataType.DT_FP16
Tensor format: TILEOP_ND
Tensor dimensions: 2

============================================================
Example 2: Element-wise Operations
============================================================
Input A shape: torch.Size([8, 8])
Input B shape: torch.Size([8, 8])
Output shape: torch.Size([8, 8])
Max difference from PyTorch: 0.000000
✓ Element-wise operations completed successfully

[... more examples ...]

============================================================
All examples completed successfully!
============================================================
```

## Learning Path

After running this example, you should understand:

1. ✅ How to create tensors
2. ✅ How to perform basic operations
3. ✅ How to use JIT compilation
4. ✅ How to configure tiling
5. ✅ How to verify results

## Next Steps

- **Learn More**: See [Quick Start Guide](../../../docs/user/getting_started/quick_start.md)
- **Core Concepts**: Read [Tensors](../../../docs/user/core_concepts/tensors.md) and [Operations](../../../docs/user/core_concepts/operations.md)
- **Try Examples**: Check out [Layer Normalization Example](../../intermediate/01_layer_normalization/layer_norm.py)
- **Advanced**: See [Advanced Topics](../../../docs/user/advanced/advanced_topics.md)

## Troubleshooting

### Issue: "Device not found"

**Solution**:
```bash
# Check NPU availability
npu-smi info

# Verify device ID
python3 -c "import torch; torch.npu.set_device(0)"
```

### Issue: "Compilation failed"

**Solution**:
- Check tensor shapes are compatible
- Verify tiling configuration
- Ensure CANN environment is sourced

### Issue: "Result mismatch"

**Solution**:
- Check data types match
- Verify operations are correct
- Consider numerical precision (FP16/BF16 have limited precision)

## Code Structure

```
basic_operations.py
├── example_tensor_creation()          # Example 1
├── example_element_wise_operations()  # Example 2
├── example_matrix_multiplication()    # Example 3
├── example_activation_functions()     # Example 4
├── example_view_operations()          # Example 5
└── example_combined_operations()      # Example 6
```

## Key Patterns Demonstrated

### Pattern 1: Basic JIT Function

```python
@pypto.jit
def my_function(inputs, outputs):
    input_tensor = inputs[0]
    output_tensor = outputs[0]
    pypto.set_vec_tile_shapes(32, 32)
    output_tensor[:] = pypto.operation(input_tensor)
```

### Pattern 2: PyTorch Integration

```python
# Create PyTorch tensors on NPU
input_torch = torch.randn(shape, dtype=torch.float16, device='npu:0')
output_torch = torch.zeros(shape, dtype=torch.float16, device='npu:0')

# Execute PyPTO function
my_function([input_torch], [output_torch])

# Verify with PyTorch
expected = torch.operation(input_torch)
assert_allclose(output_torch.cpu(), expected.cpu(), rtol=1e-3)
```

## See Also

- [PyPTO User Guide](../../../docs/user/README.md)
- [FFN Module Example](../../intermediate/03_ffn_module/README.md) - More complex example
- [Softmax Example](../../intermediate/04_softmax/softmax.py) - Dynamic shapes example

## License

This code is part of the CANN Open Software and is licensed under the CANN Open Software License Agreement Version 2.0.

