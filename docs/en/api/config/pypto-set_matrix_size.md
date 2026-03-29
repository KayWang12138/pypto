# pypto.set\_matrix\_size

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series / Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series / Atlas A2 Inference Series |    √     |

## Description

When an NZ-format input tensor is reshaped before being passed to matmul for computation, the original m, k, n values of the tensor (before reshaping) must be provided so that matmul can retrieve the original m, k, n values.

## Function Prototype

```python
set_matrix_size(size: List[int])-> None
```

## Parameters


| Parameter | Input/Output | Description                          |
|-----------|--------------|--------------------------------------|
| size      | Input        | The m, k, n values of the input tensor |

## Return Value

void

## Constraints

1. For NZ-format input tensors that have been reshaped before calling matmul, this parameter must be set.

2. When the matmul input is a 3D/4D NZ-format tensor, this parameter must be set.

## Example

```python
a = pypto.tensor((1, 32, 64), pypto.DT_FP32, "tensor_a")
b = pypto.tensor((3, 64, 16), pypto.DT_FP32, "tensor_b")
pypto.set_matrix_size([32, 64, 16]) #corresponds to the m, k, n values of the input tensor
out = pypto.matmul(a, b, pypto.DT_FP32)
```

