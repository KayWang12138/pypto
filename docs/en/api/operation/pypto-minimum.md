# pypto.minimum

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Computes the element-wise minimum of the input and another input. Supports 2D, 3D, or 4D tensors.

## Function Prototype

```python
minimum(
    input: Union[Tensor, Element, int, float], other: Union[Tensor, Element, int, float]
) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | Input        | First source operand. <br> Supported types: int, float, Element, and Tensor. <br> When the type is int or float, it is automatically converted to Element type DT_INT_32/DT_FP32. To use other data types, construct via Element. <br> Supported data types for Tensor and Element: DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32. <br> Empty tensors are not supported; shape supports only 2–4 dimensions; shape size must not exceed 2147483647 (INT32_MAX). |
| other     | Input        | Second source operand. <br> Supported types: int, float, Element, and Tensor. <br> When the type is int or float, it is automatically converted to Element type DT_INT_32/DT_FP32. To use other data types, construct via Element. <br> Supported data types for Tensor and Element: DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32. <br> Empty tensors are not supported; shape supports only 2–4 dimensions; shape size must not exceed 2147483647 (INT32_MAX). <br> The type and data type must be consistent with the first source operand. |

At least one of the two source operands must be a Tensor.

## Return Value

When both source operands are tensors, the two tensors must satisfy the broadcast relationship. This interface returns a tensor with the same shape as the broadcasted result of the two source operands, with the same data type, where each element is the element-wise minimum of the two source operands. When both operands are tensors, only multi-axis broadcasting is supported; when the data type is DT_FP32 or DT_FP16, second-to-last axis broadcasting with automatic inline handling is supported.

When one of the two source operands is a tensor, returns a tensor with the same shape as the input tensor, where each element is the element-wise minimum of the two source operands.

## Example

### TileShape Configuration Example

Before calling this operation interface, set the TileShape via `set_vec_tile_shapes`.

The TileShape dimensions must match the output dimensions.

Non-broadcast scenario: input `input` shape is `[m, n]`, `other` is `[m, n]`, output is `[m, n]`. Set TileShape to `[m1, n1]`, where `m1` and `n1` tile the `m` and `n` axes respectively.

Broadcast scenario: input `input` shape is `[m, n]`, `other` is `[m, 1]`, output is `[m, n]`. Set TileShape to `[m1, n1]`, where `m1` and `n1` tile the `m` and `n` axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
a = pypto.tensor([3], pypto.DT_INT32)
b = pypto.tensor([3], pypto.DT_INT32)
out = pypto.minimum(a, b)
```

Example result:

```python
Input a: [0, 2, 4]
Input b: [3, 1, 3]
Output out: [0, 1, 3]
```
