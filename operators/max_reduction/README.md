# max_reduction

PyPTO implementation of the max reduction operator.

## Overview

This operator computes the maximum value along a specified dimension of a tensor, similar to PyTorch's `torch.max(input, dim, keepdim)`.

## Mathematical Formula

$$y = \max(x, \text{dim}) = \max_{i \in \text{dim\_axis} x_i$$

## Features

- Supports dynamic axes (batch, seq_len dimensions)
- Supports `keepdim` parameter to preserve reduced dimension
- Supports negative dimension indexing
- Direct API mapping to `pypto.max`

## API Reference

### max_reduction_wrapper

```python
def max_reduction_wrapper(
    x: torch.Tensor,
    dim: int,
    keepdim: bool = False,
) -> torch.Tensor
```

**Parameters:**
- `x` (torch.Tensor): Input tensor of shape [b, s, n, d] or arbitrary shape
- `dim` (int): Dimension to reduce along
- `keepdim` (bool, optional): Whether to keep the reduced dimension. Default: False

**Returns:**
- torch.Tensor: Tensor with maximum values along the reduced dimension

## Usage Examples

```python
import torch
from max_reduction_impl import max_reduction_wrapper

# Basic usage
x = torch.randn(2, 512, 4096, dtype=torch.float32)
result = max_reduction_wrapper(x, dim=1, keepdim=False)
# result.shape: [2, 4096]

# With keepdim=True
result = max_reduction_wrapper(x, dim=1, keepdim=True)
# result.shape: [2, 1, 4096]
```

## Implementation Details

- Uses `pypto.max` for direct computation
- No substitute implementation needed
- TileShape: [8, 8, 8] for 3D input (32B aligned)
- Automatic handling of dynamic axes through PyPTO runtime

## Performance Characteristics
- Memory efficient: single pass reduction
- NPU optimized: uses PyPTO vector operations
- No explicit loops required for standard use cases

## Supported Data Types
- float32 (primary)
- float16 (optional)
- bfloat16 (optional)

## Precision Requirements
- Absolute tolerance (atol): 0.001
- Relative tolerance (rtol): 0.001

## See Also
- `torch.max`: PyTorch reference implementation
- `max_reduction_golden.py`: Golden reference implementation
- `test_max_reduction.py`: Test cases

