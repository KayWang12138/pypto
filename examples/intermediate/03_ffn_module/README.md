# FFN Module for PyPTO

A comprehensive Feed-Forward Network (FFN) module implementation using PyPTO, designed for efficient execution on NPU hardware.

## Features

- **Multiple Activation Functions**: Supports GELU, SwiGLU, and ReLU activations
- **Static and Dynamic Shapes**: Handles both fixed and variable batch sizes
- **Configurable Tiling**: Optimized tile shapes for NPU performance
- **Type Safety**: Full type hints and configuration dataclasses
- **Well Documented**: Comprehensive docstrings and examples

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
python3 examples/intermediate/03_ffn_module/ffn_module_example.py

# Run specific example by ID
python3 examples/intermediate/03_ffn_module/ffn_module_example.py 1  # Static FFN with GELU
python3 examples/intermediate/03_ffn_module/ffn_module_example.py 2  # Static FFN with SwiGLU
python3 examples/intermediate/03_ffn_module/ffn_module_example.py 3  # Static FFN with ReLU
python3 examples/intermediate/03_ffn_module/ffn_module_example.py 4  # Dynamic FFN with GELU
python3 examples/intermediate/03_ffn_module/ffn_module_example.py 5  # Example Usage

# List all available examples
python3 examples/intermediate/03_ffn_module/ffn_module_example.py --list
```

### Available Examples

1. **Static FFN with GELU** - Static FFN with GELU activation
2. **Static FFN with SwiGLU** - Static FFN with SwiGLU activation
3. **Static FFN with ReLU** - Static FFN with ReLU activation
4. **Dynamic FFN with GELU** - Dynamic FFN with GELU activation
5. **Example Usage** - Show example usage of FFN module (No NPU required)

## Architecture

The FFN module implements a standard transformer feed-forward network:

```
Input [B, H] 
  → Gate Projection [B, H] @ [H, I] → [B, I]
  → Activation (GELU/SwiGLU/ReLU)
  → Down Projection [B, I] @ [I, H] → [B, H]
  → Output [B, H]
```

For SwiGLU:
```
Input [B, H]
  → Gate Projection [B, H] @ [H, I] → [B, I]
  → Up Projection [B, H] @ [H, I] → [B, I]
  → SwiGLU(Gate, Up) → [B, I]
  → Down Projection [B, I] @ [I, H] → [B, H]
  → Output [B, H]
```

Where:
- `B` = batch size
- `H` = hidden size
- `I` = intermediate size

## Usage

### Basic Usage

```python
import pto
from ffn_module import FFNConfig, create_ffn_module

# Create configuration
config = FFNConfig(
    hidden_size=2048,
    intermediate_size=8192,
    activation="gelu",  # or "swiglu", "relu"
    dtype=pto.DT_BF16,
    use_dynamic_shape=False,
    vec_tile_shape=(64, 128),
    cube_tile_shape=(64, 128, 128)
)

# Create FFN module
ffn = create_ffn_module(config)

# Define tensors
batch_size = 32
hidden_states = pto.tensor([batch_size, 2048], pto.DT_BF16, "hidden_states")
gate_proj_weight = pto.tensor([2048, 8192], pto.DT_BF16, "gate_proj")
up_proj_weight = pto.tensor([2048, 8192], pto.DT_BF16, "up_proj")  # For SwiGLU
down_proj_weight = pto.tensor([8192, 2048], pto.DT_BF16, "down_proj")
output = pto.tensor([batch_size, 2048], pto.DT_BF16, "output")

# Execute
inputs = [hidden_states, gate_proj_weight, up_proj_weight, down_proj_weight]
outputs = [output]
ffn(inputs, outputs)
```

### Dynamic Batch Size

For variable batch sizes:

```python
config = FFNConfig(
    hidden_size=2048,
    intermediate_size=8192,
    activation="gelu",
    dtype=pto.DT_BF16,
    use_dynamic_shape=True,  # Enable dynamic shapes
    basic_batch=32,  # Process 32 samples at a time
    vec_tile_shape=(64, 128),
    cube_tile_shape=(64, 128, 128)
)

ffn = create_ffn_module(config)
# Same usage as above, but batch_size can vary
```

### Different Activations

#### GELU
```python
config = FFNConfig(
    hidden_size=2048,
    intermediate_size=8192,
    activation="gelu",
    # ... other config
)
```

#### SwiGLU
```python
config = FFNConfig(
    hidden_size=2048,
    intermediate_size=8192,
    activation="swiglu",  # Requires up_proj_weight
    # ... other config
)
```

#### ReLU
```python
config = FFNConfig(
    hidden_size=2048,
    intermediate_size=8192,
    activation="relu",
    # ... other config
)
```

## Configuration Options

### FFNConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_size` | `int` | Required | Hidden dimension size |
| `intermediate_size` | `int` | Required | Intermediate dimension size |
| `activation` | `str` | `"gelu"` | Activation function: `"gelu"`, `"swiglu"`, or `"relu"` |
| `dtype` | `pto.DataType` | `pto.DT_FP16` | Data type for computation |
| `use_dynamic_shape` | `bool` | `False` | Enable dynamic batch size support |
| `vec_tile_shape` | `tuple` | `(64, 128)` | Vector operation tile shape |
| `cube_tile_shape` | `tuple` | `(64, 128, 128)` | Matrix multiplication tile shape |
| `basic_batch` | `int` | `32` | Batch size for dynamic processing |

## Activation Functions

### GELU (Gaussian Error Linear Unit)
Approximated as: `x * sigmoid(1.702 * x)`

### SwiGLU (Swish-Gated Linear Unit)
Formula: `Swish(gate) * up` where `Swish(x) = x * sigmoid(x)`

### ReLU (Rectified Linear Unit)
Formula: `max(0, x)`

## Performance Tuning

### Tiling Configuration

The tile shapes significantly impact performance:

```python
# For smaller models
config.vec_tile_shape = (32, 64)
config.cube_tile_shape = (32, 64, 64)

# For larger models
config.vec_tile_shape = (128, 256)
config.cube_tile_shape = (128, 256, 256)
```

### Dynamic Batching

For dynamic batch sizes, adjust `basic_batch` based on your typical batch size:

```python
# For small batches
config.basic_batch = 16

# For large batches
config.basic_batch = 64
```

## Testing

Run the test suite:

```bash
python test_ffn_module.py
```

The test suite includes:
- Static FFN with GELU activation
- Static FFN with SwiGLU activation
- Static FFN with ReLU activation
- Dynamic FFN with GELU activation

## File Structure

```
python/example/
├── ffn_module.py          # Main FFN implementation
├── test_ffn_module.py     # Test suite and examples
└── FFN_MODULE_README.md    # This file
```

## Implementation Details

### Static FFN
- Fixed batch size
- Single-pass computation
- Optimized for known shapes

### Dynamic FFN
- Variable batch size
- Chunked processing with `basic_batch` size
- Handles unaligned batch sizes gracefully

### Memory Layout
- All operations use in-place or view operations where possible
- Results are assembled back to output tensor
- Efficient memory usage for NPU execution

## Integration with PyTorch

The module is designed to work with PyTorch tensors via `torch_npu`:

```python
import torch
import torch_npu
import pto

# Create PyTorch tensors on NPU
hidden_states_torch = torch.randn(32, 2048, dtype=torch.bfloat16, device='npu:0')

# Convert to PyPTO tensors (in real usage, this is handled by runtime)
# Execute FFN
# Convert back to PyTorch (handled by runtime)
```

## Best Practices

1. **Choose appropriate tile shapes** based on your model size and NPU capabilities
2. **Use dynamic shapes** only when batch size varies significantly
3. **Set `basic_batch`** close to your typical batch size for dynamic processing
4. **Use appropriate data types** (FP16/BF16) for your accuracy requirements
5. **Profile performance** with different tile configurations

## Limitations

- Currently supports 2D tensors (batch_size, hidden_size)
- Activation functions are approximations (GELU)
- Quantization support not yet implemented (can be added)

## Future Enhancements

- [ ] Quantization support (INT8)
- [ ] Layer normalization integration
- [ ] Dropout support
- [ ] Multi-dimensional input support
- [ ] More activation functions (SiLU, GeGLU, etc.)

## License

This code is part of the CANN Open Software and is licensed under the CANN Open Software License Agreement Version 1.0.

