# pypto.lt

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Element-wise less-than comparison operation.

## Function Prototype

```python
lt(input: Tensor, other: Union[Tensor, float, Element]) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | Input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP16, DT_BF16, DT_FP32; the data types of both source operands must be consistent. <br> Empty tensors are not supported; shape supports only 2–4 dimensions; shape size must not exceed 2147483647 (INT32_MAX). |
| other     | Input        | Source operand. <br> Supported types: Tensor, float, Element. <br> When the type is float, it is automatically converted to Element type (float maps to DT_FP32). To use other data types, construct via Element. <br> Supported data types for Tensor and Element: DT_FP16, DT_BF16, DT_FP32; the data types of both source operands must be consistent. <br> Empty tensors are not supported; shape supports only 2–4 dimensions; shape size must not exceed 2147483647 (INT32_MAX). |

## Return Value

Returns a tensor with the same shape as the input tensor and data type DT\_BOOL. If the element value at a given position in `input` is strictly less than the corresponding element in `other`, the return value at that position is True; otherwise it is False.

## Constraints

1.  The types of `input` and `other` must be consistent.
2.  One-dimensional broadcasting is supported.

## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via `set_vec_tile_shapes`.

The TileShape dimensions must match the output dimensions.

Example 1: Non-broadcast scenario — input `input` shape is `[m, n]`, `other` is `[m, n]`, output is `[m, n]`. Set TileShape to `[m1, n1]`, where `m1` and `n1` tile the `m` and `n` axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

Example 2: Broadcast scenario — input `input` shape is `[m, n]`, `other` is `[m, 1]`, output is `[m, n]`. Set TileShape to `[m1, n1]`, where `m1` and `n1` tile the `m` and `n` axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
a = pypto.tensor([3], pypto.DT_FP32)
b = pypto.tensor([3], pypto.DT_FP32)
out = pypto.lt(a, b)
```

Example result:

```python
Input a: [1.0 2.0 3.0]
Input b: [2.0 2.0 2.0]
Output out: [True, False, False]
```
