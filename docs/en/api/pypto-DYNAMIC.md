# pypto.DYNAMIC

## Supported Products

| AI Processor Type | Supported |
|------------|:--------:|
| Ascend 910C | √ |
| Ascend 910B | √ |
| Ascend 310B | ☓ |
| Ascend 310P | ☓ |
| Ascend 910 | ☓ |

## Description

`pypto.DYNAMIC` is used to define dynamic dimensions, allowing certain dimensions of a tensor to vary at runtime. This is useful for handling scenarios with variable batch sizes, sequence lengths, and similar cases. Dynamic dimensions are typically defined at the module level and then used in type annotations of JIT-compiled kernel functions.

Main use cases:
- **Dynamic Batch Size**: Inference batch size may vary with the number of requests
- **Dynamic Sequence Length**: Text sequence lengths are not fixed in NLP tasks
- **Dynamic Graph Structure**: Variable number of nodes in graph neural networks
- **Conditional Computation**: Determine computation flow based on input shape

## Shape Annotation Methods

| Annotation | Meaning |
| --- | --- |
| `pypto.DYNAMIC` or `pypto.DYN` | Dynamic axis — when this dimension of the input torch tensor changes, **no recompilation is needed** |
| `pypto.STATIC` | Static axis — when this dimension of the input torch tensor changes, **recompilation is triggered** |
| `64` | Fixed axis — only tensors of exactly this size are allowed; passing any other size will raise an error (when runtime_debug_mode is 3, validation is enabled) |
| `...` | Remaining axes are treated as static axes |

## Constraints

1. Dynamic dimensions must be used in the type annotations of JIT functions

## Example

### Example 1: Basic Usage — Dynamic Batch Size

```python
import pypto

# Fixed axis
HIDDEN_SIZE = 128

@pypto.frontend.jit
def add_bias(
    x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    bias: pypto.Tensor([HIDDEN_SIZE], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32)
):
    # Implement add logic
    # [pypto.DYNAMIC, ...] first dim is dynamic, ellipsis means remaining dims are static
    ...

# Can be called with different batch sizes
x1 = torch.randn(2, 128, dtype=torch.float32, device='npu:0')
out1 = torch.randn(2, 128, dtype=torch.float32, device='npu:0')
result1 = add_bias(x1, bias, out1)  # batch=2

x2 = torch.randn(8, 128, dtype=torch.float32, device='npu:0')
out2 = torch.randn(2, 128, dtype=torch.float32, device='npu:0')
result2 = add_bias(x2, bias, out2)  # batch=8
```

### Example 2: Multiple Dynamic Dimensions

```python
HIDDEN = 768

@pypto.frontend.jit
def attention_kernel(
    q: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, HIDDEN], pypto.DT_FP32),
    k: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, HIDDEN], pypto.DT_FP32),
    v: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, HIDDEN], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, HIDDEN], pypto.DT_FP32),
):
    # Implement attention logic
    # The first two dimensions (batch, sequence length) are both dynamic
    ...
    return output

# Can handle different batch sizes and sequence lengths
attention_kernel(q_4_128, k_4_128, v_4_128, out)  # B=4, SEQ=128
attention_kernel(q_2_256, k_2_256, v_2_256, out)  # B=2, SEQ=256, no recompilation needed
```

## Best Practices

1. **Documentation**: Add comments explaining which dimensions are dynamic and what they represent
2. **Test Coverage**: Test different values for dynamic dimensions to ensure correctness
