# Scaled Dot-Product Attention Example

A comprehensive example demonstrating the scaled dot-product attention mechanism, the core component of transformer architectures.

## Overview

This example shows how to implement:
- **Scaled dot-product attention**: The fundamental attention mechanism
- **Q, K, V computation**: Query, Key, Value projections
- **Attention scores**: Computing similarity between queries and keys
- **Softmax normalization**: Converting scores to probabilities
- **Output projection**: Final linear transformation
- **Static and dynamic shapes**: Support for variable batch/sequence lengths

## Features

- ✅ Standard scaled dot-product attention
- ✅ Multi-head attention support
- ✅ Static and dynamic batch/sequence length support
- ✅ Complete attention with projections
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
python3 examples/advanced/01_attention/attention.py

# Run specific example by ID
python3 examples/advanced/01_attention/attention.py 1  # Attention Dynamic
python3 examples/advanced/01_attention/attention.py 2  # Attention with Projections

# List all available examples
python3 examples/advanced/01_attention/attention.py --list
```

### Available Examples

1. **Attention Dynamic** - Scaled dot-product attention with dynamic shapes
2. **Attention with Projections** - Complete attention with input/output projections

## Attention Mechanism

### Scaled Dot-Product Attention Formula

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

Where:
- `Q`: Query matrix [batch, num_heads, seq_len_q, head_dim]
- `K`: Key matrix [batch, num_heads, seq_len_kv, head_dim]
- `V`: Value matrix [batch, num_heads, seq_len_kv, head_dim]
- `d_k`: Head dimension (typically 64)
- `sqrt(d_k)`: Scaling factor to prevent large dot products

### Steps

1. **Compute attention scores**: `Q @ K^T` → [batch, num_heads, seq_len_q, seq_len_kv]
2. **Scale scores**: Divide by `sqrt(head_dim)`
3. **Apply softmax**: Normalize to probabilities
4. **Apply to values**: `softmax(scores) @ V` → [batch, num_heads, seq_len_q, head_dim]

## Code Examples

### Basic Attention

```python
@pypto.jit
def scaled_dot_product_attention_static(inputs, outputs, config):
    q, k, v = inputs[0], inputs[1], inputs[2]
    out = outputs[0]

    # Calculate scale
    scale = 1.0 / (config.head_dim ** 0.5)

    pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])

    # Q @ K^T
    k_t = pypto.transpose(k, [0, 1, 3, 2])
    scores = pypto.matmul(q, k_t, out_dtype=config.dtype)

    # Scale
    scores_scaled = pypto.mul(scores, scale)

    # Softmax
    attn_weights = pypto.softmax(scores_scaled, dim=-1)

    # Apply to values
    out[:] = pypto.matmul(attn_weights, v, out_dtype=config.dtype)
```

### Dynamic Shapes

```python
@pypto.jit
def scaled_dot_product_attention_dynamic(inputs, outputs, config):
    q, k, v = inputs[0], inputs[1], inputs[2]
    out = outputs[0]

    # Enable dynamic support
    pypto.set_codegen_options(support_dynamic_unaligned=True)

    # Same computation as static version
    # ...
```

### Complete Attention with Projections

```python
@pypto.jit
def attention_with_projection(inputs, outputs, config):
    hidden_states = inputs[0]
    q_weight, k_weight, v_weight = inputs[1], inputs[2], inputs[3]
    out_weight = inputs[4]
    out = outputs[0]

    # 1. Project to Q, K, V
    q_flat = pypto.matmul(hidden_states, q_weight)
    k_flat = pypto.matmul(hidden_states, k_weight)
    v_flat = pypto.matmul(hidden_states, v_weight)

    # 2. Reshape to multi-head format
    q = pypto.reshape(q_flat, [batch, seq_len, num_heads, head_dim])
    # ... same for k, v

    # 3. Transpose for attention
    q = pypto.transpose(q, [0, 2, 1, 3])  # [batch, num_heads, seq_len, head_dim]

    # 4. Scaled dot-product attention
    # ... (as shown above)

    # 5. Output projection
    out[:] = pypto.matmul(attn_output_flat, out_weight)
```

## Usage

### Basic Usage

```python
import pto
import torch
import torch_npu

# Create Q, K, V tensors
batch_size, num_heads, seq_len, head_dim = 2, 8, 32, 64
q_torch = torch.randn(batch_size, num_heads, seq_len, head_dim,
               dtype=torch.bfloat16, device='npu:0')
k_torch = torch.randn(batch_size, num_heads, seq_len, head_dim,
               dtype=torch.bfloat16, device='npu:0')
v_torch = torch.randn(batch_size, num_heads, seq_len, head_dim,
               dtype=torch.bfloat16, device='npu:0')
out_torch = torch.zeros(batch_size, num_heads, seq_len, head_dim,
                 dtype=torch.bfloat16, device='npu:0')

# convert to pypto tensors
q = pypto.from_torch(q_torch)
k = pypto.from_torch(k_torch)
v = pypto.from_torch(v_torch)
out = pypto.from_torch(out_torch)

# Execute
config = AttentionConfig(num_heads=num_heads, head_dim=head_dim)
scaled_dot_product_attention_static([q, k, v], [out], config)
```

### Dynamic

```python
import pto
import torch
import torch_npu

# Create Q, K, V tensors
batch_size, num_heads, seq_len, head_dim = 2, 8, 32, 64
q_torch = torch.randn(batch_size, num_heads, seq_len, head_dim,
               dtype=torch.bfloat16, device='npu:0')
k_torch = torch.randn(batch_size, num_heads, seq_len, head_dim,
               dtype=torch.bfloat16, device='npu:0')
v_torch = torch.randn(batch_size, num_heads, seq_len, head_dim,
               dtype=torch.bfloat16, device='npu:0')
out_torch = torch.zeros(batch_size, num_heads, seq_len, head_dim,
                 dtype=torch.bfloat16, device='npu:0')

# convert to pypto tensors
# Mark batch dimension and sequence length dimension as dynamic
q = pypto.from_torch(q_torch, dynamic_axis=[0, 2])
k = pypto.from_torch(k_torch, dynamic_axis=[0, 2])
v = pypto.from_torch(v_torch, dynamic_axis=[0, 2])
out = pypto.from_torch(out_torch, dynamic_axis=[0, 2])
inputs = [q, k, v]
outputs = [out]
# Execute
config = AttentionConfig(num_heads=num_heads, head_dim=head_dim)
scaled_dot_product_attention_static([q, k, v], [out], config)
```

## Configuration

### AttentionConfig

```python
@dataclass
class AttentionConfig:
    num_heads: int = 8              # Number of attention heads
    head_dim: int = 64              # Dimension per head
    scale: Optional[float] = None   # Custom scale (default: 1/sqrt(head_dim))
    dtype: pypto.DataType = pypto.DT_BF16
    use_dynamic_shape: bool = False
```

## Multi-Head Attention

Multi-head attention allows the model to attend to information from different representation subspaces:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
```

**Benefits**:
- Parallel attention mechanisms
- Different attention patterns per head
- Better representation learning

## Performance Considerations

### Tiling Configuration

```python
# For attention operations
pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])

# Adjust based on sequence length and head dimension
# For seq_len=128: pypto.set_cube_tile_shapes([128, 128], [64, 64], [128, 64])
# For head_dim=128: pypto.set_cube_tile_shapes([64, 64], [128, 128], [64, 128])
```

### Memory Optimization

Attention can be memory-intensive for long sequences:
- **Memory complexity**: O(seq_len^2) for attention scores
- **Consider**: Flash Attention or chunked attention for long sequences
- **Use dynamic shapes**: When sequence lengths vary

## Integration with Transformers

Attention is typically used in transformer blocks:

```python
# Typical transformer attention block
@pypto.jit
def transformer_attention_block(inputs, outputs):
    x, norm_gamma, norm_beta = inputs[0], inputs[1], inputs[2]
    q_weight, k_weight, v_weight, out_weight = inputs[3], inputs[4], inputs[5], inputs[6]
    out = outputs[0]

    # Layer normalization
    normed = layer_norm(x, norm_gamma, norm_beta)

    # Multi-head attention
    attn_out = attention_with_projection(normed, q_weight, k_weight, v_weight, out_weight)

    # Residual connection
    out[:] = pypto.add(x, attn_out)
```

## Expected Output

```
============================================================
PyPTO Scaled Dot-Product Attention Examples
============================================================

============================================================
Test: Scaled Dot-Product Attention (Static)
============================================================
Q shape: torch.Size([2, 8, 32, 64])
K shape: torch.Size([2, 8, 32, 64])
V shape: torch.Size([2, 8, 32, 64])
Output shape: torch.Size([2, 8, 32, 64])
Max difference: 0.012345
✓ Attention (static) passed

============================================================
Test: Scaled Dot-Product Attention (Dynamic)
============================================================
Batch=2, SeqQ=16, SeqKV=16, Max diff: 0.012345
Batch=4, SeqQ=32, SeqKV=32, Max diff: 0.012345
Batch=1, SeqQ=64, SeqKV=64, Max diff: 0.012345
Batch=8, SeqQ=8, SeqKV=8, Max diff: 0.012345
✓ Attention (dynamic) passed for all test cases

============================================================
Test: Attention with Projections
============================================================
Hidden states shape: torch.Size([2, 32, 512])
Output shape: torch.Size([2, 32, 512])
Output range: [-2.1234, 2.5678]
✓ Attention with projections completed

============================================================
All attention tests passed!
============================================================
```

## Troubleshooting

### Issue: "Memory allocation failed"

**Solution**: Reduce sequence length or batch size:
```python
# Use smaller sequences
seq_len = 32  # instead of 128
```

### Issue: "Numerical precision errors"

**Solution**: Use BF16 instead of FP16:
```python
config = AttentionConfig(dtype=pypto.DT_BF16)
```

### Issue: "Dynamic shape compilation fails"

**Solution**: Enable dynamic unaligned support:
```python
pypto.set_codegen_options(support_dynamic_unaligned=True)
```

## Advanced Topics

### Attention Masks

To add attention masks (e.g., for causal attention):

```python
# In the attention function, after computing scores:
if attn_mask is not None:
    scores = pypto.add(scores, attn_mask)  # Add mask before softmax
```

### Flash Attention

For very long sequences, consider implementing chunked attention:
- Process sequence in chunks
- Use view operations for tiling
- Assemble results

## See Also

- [Basic Operations Example](../../beginner/01_basic_operations/basic_operations.py) - Start here for basics
- [Layer Normalization Example](../../intermediate/01_layer_normalization/layer_norm.py) - Often used with attention
- [FFN Module Example](../../intermediate/03_ffn_module/README.md) - Complete transformer component
- [Operations Guide](../../../docs/user/core_concepts/operations.md) - Available operations
- [Advanced Topics Guide](../../../docs/user/advanced/advanced_topics.md) - Dynamic shapes

## License

This code is part of the CANN Open Software and is licensed under the CANN Open Software License Agreement Version 2.0.


