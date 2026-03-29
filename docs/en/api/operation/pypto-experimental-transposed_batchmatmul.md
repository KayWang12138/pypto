# pypto.experimental.transposed\_batchmatmul

## Supported Products

| Product | Supported |
|:--------|:---------:|
| Atlas A3 Training Series / Atlas A3 Inference Series | √ |
| Atlas A2 Training Series / Atlas A2 Inference Series | √ |

## Description

This is a custom interface with many constraints. Stability is not guaranteed.

This operator performs a transposed batch matrix multiplication. The specific steps are:
1. Transpose the input tensor `tensor_a` from shape (M, B, K) to (B, M, K).
2. Perform a batch matrix multiplication of the transposed `tensor_a` (B, M, K) with `tensor_b` (B, K, N) to obtain an intermediate result of shape (B, M, N).
3. Transpose the intermediate result back to shape (M, B, N) as the final output.

## Function Prototype

```python
transposed_batchmatmul(tensor_a: Tensor, tensor_b: Tensor, out_dtype: dtype) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description |
|-----------|--------------|-------------|
| tensor_a | Input | Left-hand input tensor. <br> Supported data types: DT_FP16, DT_BF16. <br> Empty Tensors are not supported; only 3D shapes are supported. <br> Shape must be (M, B, K). |
| tensor_b | Input | Right-hand input tensor. <br> Supported data types: DT_FP16, DT_BF16. <br> Empty Tensors are not supported; only 3D shapes are supported. <br> Shape must be (B, K, N). |
| out_dtype | Input | Data type of the output tensor. <br> Supported data types: DT_FP16, DT_BF16. |

## Return Value

Returns the output Tensor. The data type is specified by `out_dtype`, and the shape is (M, B, N).

## Example

```python
import pypto

# Create input tensors
a = pypto.tensor((16, 2, 32), pypto.DT_FP16, "tensor_a")
b = pypto.tensor((2, 32, 64), pypto.DT_FP16, "tensor_b")

# Call the operator
c = pypto.experimental.transposed_batchmatmul(a, b, pypto.DT_FP16)

# The shape of output tensor c is (16, 2, 64)
```
