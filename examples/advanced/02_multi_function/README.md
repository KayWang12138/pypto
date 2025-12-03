# Multi-Function Module Example

A comprehensive example demonstrating how to use multiple `@pypto.jit` functions together to build complex computation pipelines.

## Overview

This example shows how to:
- **Compose multiple JIT functions**: Chain functions together
- **Reuse functions**: Use the same function with different inputs
- **Build complex modules**: Create transformer blocks from smaller functions
- **Manage data flow**: Pass data between functions

This pattern is essential for building modular neural network architectures.

## Features

- ✅ Multiple independent JIT functions
- ✅ Sequential function composition
- ✅ Residual connections
- ✅ Function reuse patterns
- ✅ Complete transformer block example
- ✅ Well-documented code

## Prerequisites

- PyPTO installed
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
python3 examples/advanced/02_multi_function/multi_function_module.py

# Run specific example by ID
python3 examples/advanced/02_multi_function/multi_function_module.py 1  # Sequential Functions
python3 examples/advanced/02_multi_function/multi_function_module.py 2  # Residual Connection
python3 examples/advanced/02_multi_function/multi_function_module.py 3  # Transformer Block
python3 examples/advanced/02_multi_function/multi_function_module.py 4  # Function Reuse

# List all available examples
python3 examples/advanced/02_multi_function/multi_function_module.py --list
```

### Available Examples

1. **Sequential Functions** - Using multiple functions in sequence
2. **Residual Connection** - Residual connection pattern
3. **Transformer Block** - Complete transformer block with multiple functions
4. **Function Reuse** - Reusing the same function with different inputs

## Function Composition Patterns

### Pattern 1: Sequential Functions

Chain functions one after another:

```python
# Step 1: Layer normalization
layer_norm([x, gamma, beta], [normed], eps=1e-6)

# Step 2: GELU activation
gelu_activation([normed], [activated])
```

**Data Flow**:
```
Input → LayerNorm → GELU → Output
```

### Pattern 2: Residual Connections

Add skip connections:

```python
# Process input
processed = process(x)

# Add residual
residual_add([x, processed], [output])
```

**Data Flow**:
```
Input ──┐
        ├─→ Add → Output
        │
Process ─┘
```

### Pattern 3: Parallel Branches

Process multiple branches:

```python
# Branch 1: Gate projection
linear_projection([x, gate_weight], [gate])

# Branch 2: Up projection
linear_projection([x, up_weight], [up])

# Combine
activated = combine(gate, up)
```

**Data Flow**:
```
Input ──┬─→ Gate Projection ──┐
        │                      ├─→ Combine → Output
        └─→ Up Projection ─────┘
```

### Pattern 4: Function Reuse

Use the same function multiple times:

```python
# Reuse layer_norm with different inputs
layer_norm([x1, gamma, beta], [out1], eps=1e-6)
layer_norm([x2, gamma, beta], [out2], eps=1e-6)
layer_norm([x3, gamma, beta], [out3], eps=1e-6)
```

## Included Functions

### 1. Layer Normalization

```python
@pto.jit
def layer_norm(inputs, outputs, eps: float = 1e-6):
    x, gamma, beta = inputs[0], inputs[1], inputs[2]
    out = outputs[0]
    # ... normalization logic
```

**Usage**:
```python
layer_norm([x, gamma, beta], [out], eps=1e-6)
```

### 2. Linear Projection

```python
@pto.jit
def linear_projection(inputs, outputs):
    x, weight = inputs[0], inputs[1]
    bias = inputs[2] if len(inputs) > 2 else None
    out = outputs[0]
    # ... matrix multiplication
```

**Usage**:
```python
linear_projection([x, weight], [out])
linear_projection([x, weight, bias], [out])  # With bias
```

### 3. GELU Activation

```python
@pto.jit
def gelu_activation(inputs, outputs):
    x = inputs[0]
    out = outputs[0]
    # ... GELU computation
```

**Usage**:
```python
gelu_activation([x], [out])
```

### 4. Residual Connection

```python
@pto.jit
def residual_add(inputs, outputs):
    x, residual = inputs[0], inputs[1]
    out = outputs[0]
    # ... addition
```

**Usage**:
```python
residual_add([x, residual], [out])
```

### 5. Attention (Simplified)

```python
@pto.jit
def attention(inputs, outputs, scale: float):
    q, k, v = inputs[0], inputs[1], inputs[2]
    out = outputs[0]
    # ... attention computation
```

**Usage**:
```python
scale = 1.0 / (head_dim ** 0.5)
attention([q, k, v], [out], scale=scale)
```

## Code Examples

### Example 1: Sequential Processing

```python
# Create tensors
x = torch.randn(32, 128, dtype=torch.bfloat16, device='npu:0')
gamma = torch.ones(128, dtype=torch.bfloat16, device='npu:0')
beta = torch.zeros(128, dtype=torch.bfloat16, device='npu:0')
normed = torch.zeros(32, 128, dtype=torch.bfloat16, device='npu:0')
activated = torch.zeros(32, 128, dtype=torch.bfloat16, device='npu:0')

# Step 1: Normalize
layer_norm([x, gamma, beta], [normed], eps=1e-6)

# Step 2: Activate
gelu_activation([normed], [activated])
```

### Example 2: Transformer Block

```python
# Complete transformer block
def transformer_block(x, gamma, beta, gate_weight, up_weight, down_weight):
    # 1. Layer normalization
    normed = torch.zeros_like(x)
    layer_norm([x, gamma, beta], [normed], eps=1e-6)
    
    # 2. FFN projections
    gate = torch.zeros(x.shape[0], intermediate_size, dtype=x.dtype, device=x.device)
    up = torch.zeros(x.shape[0], intermediate_size, dtype=x.dtype, device=x.device)
    linear_projection([normed, gate_weight], [gate])
    linear_projection([normed, up_weight], [up])
    
    # 3. Activation
    activated = torch.zeros_like(gate)
    gelu_activation([gate], [activated])
    activated = activated * up  # SwiGLU-like
    
    # 4. Down projection
    ffn_out = torch.zeros_like(x)
    linear_projection([activated, down_weight], [ffn_out])
    
    # 5. Residual
    output = torch.zeros_like(x)
    residual_add([x, ffn_out], [output])
    
    return output
```

### Example 3: Function Reuse

```python
# Process multiple inputs with the same function
inputs = [x1, x2, x3, x4]
outputs = [out1, out2, out3, out4]

for i, (x, out) in enumerate(zip(inputs, outputs)):
    layer_norm([x, gamma, beta], [out], eps=1e-6)
```

## Best Practices

### 1. Function Design

- **Single responsibility**: Each function should do one thing well
- **Reusability**: Design functions to be reusable
- **Clear interfaces**: Use consistent input/output patterns

### 2. Data Management

- **Pre-allocate outputs**: Create output tensors before calling functions
- **Manage memory**: Reuse tensors when possible
- **Device placement**: Ensure all tensors are on the same device

### 3. Composition

- **Sequential**: Chain functions for sequential processing
- **Parallel**: Use multiple functions for parallel branches
- **Conditional**: Switch between functions based on conditions

### 4. Performance

- **Compile once**: Functions are compiled on first use
- **Reuse compiled functions**: Same function with different inputs reuses compilation
- **Batch operations**: Process multiple samples together when possible

## Expected Output

```
============================================================
PyPTO Multi-Function Module Examples
============================================================

============================================================
Test: Sequential Functions
============================================================
Input shape: torch.Size([32, 128])
Normalized max diff: 0.012345
Activated max diff: 0.012345
✓ Sequential functions passed

============================================================
Test: Residual Connection
============================================================
Input shape: torch.Size([32, 128])
Residual shape: torch.Size([32, 128])
Output shape: torch.Size([32, 128])
Max difference: 0.000123
✓ Residual connection passed

============================================================
Test: Transformer Block (Multi-Function)
============================================================
Input shape: torch.Size([32, 128])
Output shape: torch.Size([32, 128])
Output range: [-2.1234, 2.5678]
✓ Transformer block (multi-function) completed

============================================================
Test: Function Reuse
============================================================
Function reused 3 times with different inputs
Max diff 1: 0.012345
Max diff 2: 0.012345
Max diff 3: 0.012345
✓ Function reuse passed

============================================================
All multi-function module tests passed!
============================================================
```

## Advanced Patterns

### Pattern 1: Conditional Function Selection

```python
def process_with_activation(x, activation_type):
    if activation_type == "gelu":
        gelu_activation([x], [out])
    elif activation_type == "silu":
        silu_activation([x], [out])
    # ...
```

### Pattern 2: Nested Function Calls

```python
# Function A calls Function B
@pto.jit
def function_a(inputs, outputs):
    # ... some processing
    intermediate = ...
    function_b([intermediate], [output])
```

### Pattern 3: Function Pipelines

```python
def create_pipeline(functions):
    def pipeline(inputs, outputs):
        current = inputs[0]
        for func in functions:
            next_output = torch.zeros_like(current)
            func([current], [next_output])
            current = next_output
        outputs[0][:] = current
    return pipeline
```

## Troubleshooting

### Issue: "Function not found"

**Solution**: Ensure functions are defined before use:
```python
# Define functions first
@pto.jit
def my_function(...):
    ...

# Then use them
my_function([x], [out])
```

### Issue: "Tensor shape mismatch"

**Solution**: Check tensor shapes match function expectations:
```python
# Verify shapes
assert x.shape == expected_shape
```

### Issue: "Device mismatch"

**Solution**: Ensure all tensors are on the same device:
```python
# All on NPU
x = x.to('npu:0')
out = out.to('npu:0')
```

## See Also

- [Basic Operations Example](../../beginner/01_basic_operations/basic_operations.py) - Start here for basics
- [Custom Activation Example](../../intermediate/02_custom_activation/custom_activation.py) - Activation functions
- [Layer Normalization Example](../../intermediate/01_layer_normalization/layer_norm.py) - Normalization details
- [Attention Example](../01_attention/attention.py) - Attention mechanism
- [Functions & JIT Guide](../../../docs/user/core_concepts/functions_jit.md) - JIT compilation

## License

This code is part of the CANN Open Software and is licensed under the CANN Open Software License Agreement Version 2.0.

