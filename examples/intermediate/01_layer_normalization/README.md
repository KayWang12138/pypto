# Layer Normalization Example

A comprehensive example demonstrating Layer Normalization and RMS Normalization implementations using PyPTO.

## Overview

This example shows how to implement:
- **LayerNorm**: Standard layer normalization with mean and variance
- **RMSNorm**: Root Mean Square normalization (simpler variant)
- **Static shapes**: Fixed batch size
- **Dynamic shapes**: Variable batch size support

Layer normalization is a critical component in transformer architectures, used in models like GPT, BERT, and LLaMA.

## Features

- ✅ Standard LayerNorm implementation
- ✅ RMSNorm implementation
- ✅ Static and dynamic batch size support
- ✅ Verified against PyTorch reference
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
python3 examples/intermediate/01_layer_normalization/layer_norm.py

# Run specific example by ID
python3 examples/intermediate/01_layer_normalization/layer_norm.py 1  # LayerNorm
python3 examples/intermediate/01_layer_normalization/layer_norm.py 2  # RMSNorm

# List all available examples
python3 examples/intermediate/01_layer_normalization/layer_norm.py --list
```

### Available Examples

1. **LayerNorm** - Standard Layer Normalization
2. **RMSNorm** - Root Mean Square Normalization

## Normalization Types

### LayerNorm

Standard layer normalization formula:
```
mean = mean(x, dim=-1)
var = var(x, dim=-1)
normalized = (x - mean) / sqrt(var + eps)
output = gamma * normalized + beta
```

**Features**:
- Uses both mean and variance
- Has learnable scale (gamma) and shift (beta) parameters
- More computationally expensive than RMSNorm

### RMSNorm

Root Mean Square normalization formula:
```
rms = sqrt(mean(x^2) + eps)
output = gamma * (x / rms)
```

**Features**:
- Only uses RMS (no mean subtraction)
- Simpler computation
- Only has scale parameter (gamma)
- Used in models like LLaMA

## Code Examples

### Static LayerNorm

```python
@pto.jit
def layer_norm_static(inputs, outputs, config):
    x, gamma, beta = inputs[0], inputs[1], inputs[2]
    out = outputs[0]
    
    hidden_size = x.shape[-1]
    eps = pto.element(config.dtype, config.eps)
    
    pto.set_vec_tile_shapes(64, 128)
    
    with pto.function("LAYER_NORM", [x, gamma, beta], [out], static=True):
        # Compute mean
        mean = pto.sum(x, dim=-1, keepdim=True) / hidden_size
        centered = pto.sub(x, mean)
        
        # Compute variance
        var = pto.sum(pto.mul(centered, centered), dim=-1, keepdim=True) / hidden_size
        
        # Normalize
        normalized = pto.div(centered, pto.sqrt(pto.add(var, eps)))
        
        # Scale and shift
        out[:] = pto.add(pto.mul(normalized, gamma), beta)
```

### Dynamic LayerNorm

```python
@pto.jit
def layer_norm_dynamic(inputs, outputs, config):
    x, gamma, beta = inputs[0], inputs[1], inputs[2]
    out = outputs[0]
    
    # Mark batch dimension as dynamic
    pto.mark_dynamic(x, 0)
    pto.mark_dynamic(out, 0)
    
    # Enable dynamic unaligned support
    pto.set_codegen_options(support_dynamic_unaligned=True)
    
    # Same computation as static version
    # ...
```

## Usage

### Basic Usage

```python
import pto
import torch
import torch_npu

# Create tensors
batch_size, hidden_size = 32, 128
x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device='npu:0')
gamma = torch.ones(hidden_size, dtype=torch.bfloat16, device='npu:0')
beta = torch.zeros(hidden_size, dtype=torch.bfloat16, device='npu:0')
out = torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, device='npu:0')

# Execute
config = NormConfig(norm_type="layernorm", dtype=pto.DT_BF16)
layer_norm_static([x, gamma, beta], [out], config)
```

### Dynamic Batch Size

```python
# Works with any batch size
for batch_size in [16, 32, 64, 100]:
    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device='npu:0')
    out = torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, device='npu:0')
    
    config = NormConfig(norm_type="layernorm", use_dynamic_shape=True)
    layer_norm_dynamic([x, gamma, beta], [out], config)
```

## Configuration

### NormConfig

```python
@dataclass
class NormConfig:
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    eps: float = 1e-6  # Epsilon for numerical stability
    dtype: pto.DataType = pto.DT_BF16
    use_dynamic_shape: bool = False
```

## Comparison: LayerNorm vs RMSNorm

| Feature | LayerNorm | RMSNorm |
|---------|-----------|---------|
| Mean subtraction | Yes | No |
| Variance calculation | Yes | No (uses RMS) |
| Parameters | gamma, beta | gamma only |
| Computation | More expensive | Simpler |
| Common usage | BERT, GPT-2 | LLaMA |

## Performance Considerations

### Tiling Configuration

```python
# For layer normalization
pto.set_vec_tile_shapes(64, 128)  # Good for hidden_size=128

# Adjust based on hidden size
# For hidden_size=512: pto.set_vec_tile_shapes(64, 512)
# For hidden_size=1024: pto.set_vec_tile_shapes(128, 1024)
```

### Static vs Dynamic

- **Static**: Faster compilation, better optimization
- **Dynamic**: More flexible, handles variable batch sizes

Choose based on your use case:
- Use static when batch size is fixed
- Use dynamic when batch size varies

## Integration with Transformers

Layer normalization is typically used in transformer blocks:

```python
# Typical transformer block pattern
@pto.jit
def transformer_block(inputs, outputs):
    x, norm_gamma, norm_beta = inputs[0], inputs[1], inputs[2]
    out = outputs[0]
    
    # Layer normalization
    normed = layer_norm(x, norm_gamma, norm_beta)
    
    # Attention or FFN
    processed = process(normed)
    
    # Residual connection
    out[:] = pto.add(x, processed)
```

## Expected Output

```
============================================================
PyPTO Layer Normalization Examples
============================================================

============================================================
Test: LayerNorm (Static)
============================================================
Input shape: torch.Size([32, 128])
Output shape: torch.Size([32, 128])
Max difference: 0.012345
✓ LayerNorm (static) passed

============================================================
Test: RMSNorm (Static)
============================================================
Input shape: torch.Size([32, 128])
Output shape: torch.Size([32, 128])
Max difference: 0.012345
✓ RMSNorm (static) passed

============================================================
Test: LayerNorm (Dynamic)
============================================================
Batch size: 16, Max difference: 0.012345
Batch size: 32, Max difference: 0.012345
Batch size: 64, Max difference: 0.012345
Batch size: 100, Max difference: 0.012345
✓ LayerNorm (dynamic) passed for all batch sizes

============================================================
All layer normalization tests passed!
============================================================
```

## Troubleshooting

### Issue: "Numerical precision errors"

**Solution**: Adjust epsilon value:
```python
config = NormConfig(eps=1e-5)  # Larger epsilon for FP16
```

### Issue: "Dynamic shape compilation fails"

**Solution**: Enable dynamic unaligned support:
```python
pto.set_codegen_options(support_dynamic_unaligned=True)
```

## See Also

- [Basic Operations Example](../../beginner/01_basic_operations/basic_operations.py) - Start here for basics
- [FFN Module Example](../03_ffn_module/README.md) - Complete transformer component
- [Dynamic Shapes Guide](../../../docs/user/advanced/advanced_topics.md#dynamic-shapes) - Dynamic shape details
- [Operations Guide](../../../docs/user/core_concepts/operations.md) - Available operations

## License

This code is part of the CANN Open Software and is licensed under the CANN Open Software License Agreement Version 2.0.

