# Custom Activation Functions Example

A comprehensive example demonstrating how to implement custom activation functions by composing PyPTO operations.

## Overview

This example shows how to build custom activation functions from basic PyPTO operations:
- **SiLU (Swish)**: `x * sigmoid(x)`
- **GELU**: `x * sigmoid(1.702 * x)` approximation
- **SwiGLU**: `Swish(gate) * up`
- **GeGLU**: `GELU(gate) * up`

These activations are commonly used in modern transformer architectures like GPT, LLaMA, and PaLM.

## Features

- ✅ Multiple activation functions
- ✅ Single-input activations (SiLU, GELU)
- ✅ Gated activations (SwiGLU, GeGLU)
- ✅ Static and dynamic shape support
- ✅ Verified against PyTorch reference
- ✅ Well-documented composition patterns

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
python3 examples/intermediate/02_custom_activation/custom_activation.py

# Run specific example by ID
python3 examples/intermediate/02_custom_activation/custom_activation.py 1  # GELU
python3 examples/intermediate/02_custom_activation/custom_activation.py 2  # SiLU
python3 examples/intermediate/02_custom_activation/custom_activation.py 3  # SwiGLU
python3 examples/intermediate/02_custom_activation/custom_activation.py 4  # GeGLU

# List all available examples
python3 examples/intermediate/02_custom_activation/custom_activation.py --list
```

### Available Examples

1. **GELU Activation** - Gaussian Error Linear Unit
2. **SiLU Activation** - Sigmoid Linear Unit (Swish)
3. **SwiGLU Activation** - Swish-Gated Linear Unit
4. **GeGLU Activation** - GELU-Gated Linear Unit

## Activation Functions

### SiLU (Swish)

**Formula**: `SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))`

**Properties**:
- Smooth, non-monotonic activation
- Self-gating property
- Used in EfficientNet and other architectures

**Implementation**:
```python
def silu_activation(x: pto.tensor) -> pto.tensor:
    # sigmoid(x) = 1 / (1 + exp(-x))
    x_neg = pto.mul(x, pto.element(x.dtype, -1.0))
    exp_neg = pto.exp(x_neg)
    one = pto.element(x.dtype, 1.0)
    sigmoid = pto.div(one, pto.add(exp_neg, one))
    
    # SiLU(x) = x * sigmoid(x)
    return pto.mul(x, sigmoid)
```

### GELU

**Formula**: `GELU(x) ≈ x * sigmoid(1.702 * x)`

**Properties**:
- Smooth approximation of ReLU
- Used in BERT, GPT-2, GPT-3
- Better gradient flow than ReLU

**Implementation**:
```python
def gelu_activation(x: pto.tensor) -> pto.tensor:
    # GELU approximation: x * sigmoid(1.702 * x)
    coeff = pto.element(x.dtype, 1.702)
    x_scaled = pto.mul(x, coeff)
    
    # Compute sigmoid(1.702 * x)
    x_scaled_neg = pto.mul(x_scaled, pto.element(x.dtype, -1.0))
    exp_neg = pto.exp(x_scaled_neg)
    one = pto.element(x.dtype, 1.0)
    sigmoid = pto.div(one, pto.add(exp_neg, one))
    
    # GELU(x) = x * sigmoid(1.702 * x)
    return pto.mul(x, sigmoid)
```

### SwiGLU

**Formula**: `SwiGLU(gate, up) = Swish(gate) * up = (gate * sigmoid(gate)) * up`

**Properties**:
- Gated linear unit with Swish gating
- Used in PaLM, LLaMA models
- Requires two inputs (gate and up projections)

**Implementation**:
```python
def swiglu_activation(gate: pto.tensor, up: pto.tensor) -> pto.tensor:
    # Swish(gate) = gate * sigmoid(gate)
    gate_neg = pto.mul(gate, pto.element(gate.dtype, -1.0))
    exp_neg = pto.exp(gate_neg)
    one = pto.element(gate.dtype, 1.0)
    sigmoid = pto.div(one, pto.add(exp_neg, one))
    swish = pto.mul(gate, sigmoid)
    
    # Multiply with up projection
    return pto.mul(swish, up)
```

### GeGLU

**Formula**: `GeGLU(gate, up) = GELU(gate) * up`

**Properties**:
- Gated linear unit with GELU gating
- Alternative to SwiGLU
- Used in some transformer variants

**Implementation**:
```python
def geglu_activation(gate: pto.tensor, up: pto.tensor) -> pto.tensor:
    # GELU(gate)
    gelu_gate = gelu_activation(gate)
    
    # Multiply with up projection
    return pto.mul(gelu_gate, up)
```

## Code Examples

### Basic Usage

```python
import pto
import torch
import torch_npu

# Create input tensor
shape = (32, 128)
x = torch.randn(shape, dtype=torch.bfloat16, device='npu:0')
out = torch.zeros(shape, dtype=torch.bfloat16, device='npu:0')

# Apply SiLU activation
config = ActivationConfig(activation_type="silu", dtype=pto.DT_BF16)
apply_activation_static([x], [out], config)
```

### Gated Activation

```python
# For SwiGLU or GeGLU
gate = torch.randn(shape, dtype=torch.bfloat16, device='npu:0')
up = torch.randn(shape, dtype=torch.bfloat16, device='npu:0')
out = torch.zeros(shape, dtype=torch.bfloat16, device='npu:0')

# Apply SwiGLU
config = ActivationConfig(activation_type="swiglu", dtype=pto.DT_BF16)
apply_gated_activation_static([gate, up], [out], config)
```

### Dynamic Shapes

```python
# Works with variable batch sizes
config = ActivationConfig(activation_type="silu", use_dynamic_shape=True)
apply_activation_dynamic([x], [out], config)
```

## Composition Patterns

### Pattern 1: Building Sigmoid

```python
# sigmoid(x) = 1 / (1 + exp(-x))
x_neg = pto.mul(x, pto.element(x.dtype, -1.0))
exp_neg = pto.exp(x_neg)
one = pto.element(x.dtype, 1.0)
sigmoid = pto.div(one, pto.add(exp_neg, one))
```

### Pattern 2: Element-wise Operations

```python
# Create scalar elements
coeff = pto.element(x.dtype, 1.702)
zero = pto.element(x.dtype, 0.0)
one = pto.element(x.dtype, 1.0)

# Use in operations
x_scaled = pto.mul(x, coeff)
result = pto.maximum(x, zero)  # ReLU
```

### Pattern 3: Function Composition

```python
# Compose functions
def complex_activation(x):
    # Step 1: Apply first activation
    gelu_x = gelu_activation(x)
    
    # Step 2: Apply second operation
    scaled = pto.mul(gelu_x, pto.element(x.dtype, 0.5))
    
    return scaled
```

## Configuration

### ActivationConfig

```python
@dataclass
class ActivationConfig:
    activation_type: Literal["silu", "gelu", "swiglu", "geglu"] = "silu"
    dtype: pto.DataType = pto.DT_BF16
    use_dynamic_shape: bool = False
```

## Performance Considerations

### Tiling Configuration

```python
# Configure tiling based on input shape
if len(x.shape) >= 2:
    pto.set_vec_tile_shapes(x.shape[0], x.shape[1])
else:
    pto.set_vec_tile_shapes(32, 128)
```

### Memory Optimization

- Activation functions are element-wise, so memory usage is O(n)
- Consider in-place operations when possible
- Use appropriate data types (BF16 vs FP16)

## Integration with Neural Networks

### In FFN Modules

```python
@pto.jit
def ffn_with_activation(inputs, outputs):
    hidden_states = inputs[0]
    gate_weight, up_weight = inputs[1], inputs[2]
    out = outputs[0]
    
    # Projections
    gate = pto.matmul(hidden_states, gate_weight)
    up = pto.matmul(hidden_states, up_weight)
    
    # Apply SwiGLU
    activated = swiglu_activation(gate, up)
    
    # Down projection
    out[:] = pto.matmul(activated, down_weight)
```

## Expected Output

```
============================================================
PyPTO Custom Activation Functions Examples
============================================================

============================================================
Test: SiLU Activation (Static)
============================================================
Input shape: torch.Size([32, 128])
Output shape: torch.Size([32, 128])
Max difference: 0.012345
✓ SiLU (static) passed

============================================================
Test: GELU Activation (Static)
============================================================
Input shape: torch.Size([32, 128])
Output shape: torch.Size([32, 128])
Max difference: 0.012345
✓ GELU (static) passed

============================================================
Test: SwiGLU Activation (Static)
============================================================
Gate shape: torch.Size([32, 128])
Up shape: torch.Size([32, 128])
Output shape: torch.Size([32, 128])
Max difference: 0.012345
✓ SwiGLU (static) passed

============================================================
Test: GeGLU Activation (Static)
============================================================
Gate shape: torch.Size([32, 128])
Up shape: torch.Size([32, 128])
Output shape: torch.Size([32, 128])
Max difference: 0.012345
✓ GeGLU (static) passed

============================================================
Test: Activation (Dynamic)
============================================================
Batch size: 16, Max difference: 0.012345
Batch size: 32, Max difference: 0.012345
Batch size: 64, Max difference: 0.012345
Batch size: 100, Max difference: 0.012345
✓ Activation (dynamic) passed for all batch sizes

============================================================
All custom activation tests passed!
============================================================
```

## Creating Your Own Activation

### Step-by-Step Guide

1. **Define the mathematical formula**
   ```python
   # Example: Custom activation = x * tanh(x)
   ```

2. **Break down into PyPTO operations**
   ```python
   # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
   two_x = pto.mul(x, pto.element(x.dtype, 2.0))
   exp_2x = pto.exp(two_x)
   tanh = pto.div(pto.sub(exp_2x, one), pto.add(exp_2x, one))
   ```

3. **Compose the final function**
   ```python
   def custom_activation(x):
       tanh = compute_tanh(x)
       return pto.mul(x, tanh)
   ```

4. **Test against reference**
   ```python
   expected = x * torch.tanh(x)
   assert_allclose(output, expected, rtol=1e-3)
   ```

## Troubleshooting

### Issue: "Numerical precision errors"

**Solution**: Use BF16 instead of FP16:
```python
config = ActivationConfig(dtype=pto.DT_BF16)
```

### Issue: "Compilation fails"

**Solution**: Check tensor shapes and tiling configuration:
```python
# Ensure tiling matches input shape
pto.set_vec_tile_shapes(*x.shape[:2])
```

## See Also

- [Basic Operations Example](../../beginner/01_basic_operations/basic_operations.py) - Start here for basics
- [FFN Module Example](../03_ffn_module/README.md) - Real-world usage
- [Operations Guide](../../../docs/user/core_concepts/operations.md) - Available operations
- [Advanced Topics Guide](../../../docs/user/advanced/advanced_topics.md) - Dynamic shapes

## License

This code is part of the CANN Open Software and is licensed under the CANN Open Software License Agreement Version 2.0.

